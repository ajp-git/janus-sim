/// Janus N-body GPU — 8M Production Run with FULL state snapshots
/// Saves positions + velocities in f64 for perfect resume capability
///
/// Snapshot format v2:
/// - Header: 64 bytes (magic, version, n, step, tau, a, ke_ratio, seg)
/// - Positions: N × 3 × f64
/// - Velocities: N × 3 × f64
/// - Signs: N × i32
/// Total: ~416 MB per snapshot for 8M particles

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Read, Write, BufWriter, BufReader};
use std::time::Instant;

const SNAPSHOT_MAGIC: u64 = 0x4A414E5553534E32;  // "JANUSSN2"
const SNAPSHOT_VERSION: u64 = 2;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Parse arguments
        let args: Vec<String> = std::env::args().collect();
        let resume_path = args.iter().position(|a| a == "--resume")
            .map(|i| args.get(i + 1).cloned())
            .flatten();

        // N particles from --n argument or default 8M
        let n_particles: usize = args.iter().position(|a| a == "--n")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(8_000_000);
        let eta: f64 = 1.045;
        let total_steps: usize = 50000;  // Run until exhaustion or manual stop
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;
        let snapshot_interval: usize = 200;
        let checkpoint_interval: usize = 1000;  // Full state saved here

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

        // Output directory
        let timestamp = chrono_lite();
        let output_dir = if resume_path.is_some() {
            format!("/app/output/resume_8m_{}", timestamp)
        } else {
            format!("/app/output/run_8m_full_{}", timestamp)
        };

        fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();

        // Save PID
        let pid = std::process::id();
        fs::write(format!("{}/pid.txt", output_dir), pid.to_string()).ok();

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Janus N-body GPU — 8M with FULL State Snapshots              ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nPID: {}", pid);
        println!("Output: {}\n", output_dir);

        // Initialize or resume
        let (mut gpu_sim, start_step, tau_start, ke0_restored) = if let Some(ref snap_path) = resume_path {
            println!("RESUMING from: {}", snap_path);
            match load_full_snapshot(snap_path) {
                Ok((header, positions, velocities, signs)) => {
                    println!("  Step: {}", header.step);
                    println!("  τ: {:.6}", header.tau);
                    println!("  a: {:.6}", header.a);
                    println!("  KE/KE₀: {:.4}", header.ke_ratio);
                    println!("  S: {:.6}", header.segregation);

                    let n_pos = signs.iter().filter(|&&s| s > 0).count();
                    let n_neg = header.n_particles as usize - n_pos;

                    match GpuNBodySimulation::new_with_state(
                        n_pos, n_neg, box_size,
                        positions, velocities, signs
                    ) {
                        Ok(mut sim) => {
                            sim.set_theta(theta);
                            // Restore KE₀ from header (ke_ratio = KE/KE₀, so KE₀ = KE/ke_ratio)
                            let ke_current = sim.kinetic_energy().unwrap();
                            let ke0_est = ke_current / header.ke_ratio;
                            (sim, header.step as usize, header.tau, Some(ke0_est))
                        }
                        Err(e) => {
                            eprintln!("Failed to initialize from snapshot: {}", e);
                            return;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to load snapshot: {}", e);
                    return;
                }
            }
        } else {
            println!("NEW RUN");
            println!("Parameters:");
            println!("  N = {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
            println!("  N+ = {} ({:.1}%)", n_positive, n_positive as f64 / n_particles as f64 * 100.0);
            println!("  N- = {} ({:.1}%)", n_negative, n_negative as f64 / n_particles as f64 * 100.0);
            println!("  η = {:.4}", eta);
            println!("  box = {:.2}", box_size);
            println!("  θ = {} (Barnes-Hut)", theta);
            println!("  dt = {}", dt);

            match GpuNBodySimulation::new_bvh_only(n_positive, n_negative, box_size) {
                Ok(mut sim) => {
                    sim.set_theta(theta);
                    (sim, 0, 0.0, None)
                }
                Err(e) => {
                    eprintln!("Failed to initialize: {}", e);
                    return;
                }
            }
        };

        // Save config
        let config = format!(r#"{{
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {:.4},
  "box_size": {:.4},
  "total_steps": {},
  "dt": {},
  "theta": {},
  "snapshot_interval": {},
  "checkpoint_interval": {},
  "snapshot_format": "v2_full",
  "output_dir": "{}",
  "resumed_from": {}
}}"#, n_particles, n_positive, n_negative, eta, box_size, total_steps, dt, theta,
            snapshot_interval, checkpoint_interval, output_dir,
            resume_path.as_ref().map(|p| format!("\"{}\"", p)).unwrap_or("null".to_string()));
        fs::write(format!("{}/config.json", output_dir), &config).ok();

        println!("\nIntervals:");
        println!("  Snapshots: every {} steps (~{:.0} MB each)", snapshot_interval,
                 (64 + n_particles * 3 * 8 * 2 + n_particles * 4) as f64 / 1e6);
        println!("  Checkpoints: every {} steps", checkpoint_interval);
        println!("  Auto-stop: KE/KE₀ > 50");

        // Virialization already done analytically in new_bvh_only
        // Just get the KE₀ for monitoring
        let ke0 = if let Some(ke0_val) = ke0_restored {
            println!("\n--- Using restored KE₀ = {:.4e} ---", ke0_val);
            ke0_val
        } else {
            // Virialization was done analytically during init
            let ke = gpu_sim.kinetic_energy().unwrap();
            println!("  KE₀ (post-virialization) = {:.4e}", ke);
            ke
        };

        let seg0 = gpu_sim.segregation_distance().unwrap();
        println!("  S₀ = {:.6}", seg0);

        // Cosmological interpolator
        println!("\n--- Cosmological Expansion Setup ---");
        let janus_params = JanusParams::from_eta(eta);
        let cosmo = CosmoInterpolator::new(&janus_params, 5.0);
        let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (12000.0);  // Based on z=5→0 in ~12000 steps
        let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);

        let tau_init = if start_step > 0 { tau_start } else { cosmo.tau_start };
        println!("  z: 5 → 0");
        println!("  Starting τ: {:.6}", tau_init);

        // Time series file
        let ts_filename = if start_step > 0 {
            format!("{}/time_series_resume.csv", output_dir)
        } else {
            format!("{}/time_series.csv", output_dir)
        };
        let mut ts_file = BufWriter::new(File::create(&ts_filename).unwrap());
        writeln!(ts_file, "step,tau,a,H,ke,ke_ratio,segregation,step_time_ms").unwrap();

        // Save initial snapshot if new run
        if start_step == 0 {
            save_full_snapshot(&gpu_sim, 0, &output_dir, n_particles, cosmo.tau_start,
                               cosmo.get_params_at_tau(cosmo.tau_start).0, seg0, 1.0);
        }

        println!("\n--- Starting simulation from step {} ---\n", start_step);
        println!("{:>6}  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}",
                 "Step", "τ", "a", "KE/KE₀", "S", "Time");
        println!("{}", "─".repeat(70));

        let sim_start = Instant::now();
        let mut s_max = seg0;
        let mut s_max_step = start_step;
        let mut steps_completed = 0usize;
        let mut stop_reason: Option<String> = None;

        for step in (start_step + 1)..=total_steps {
            let step_start = Instant::now();

            // Cosmological parameters
            let current_tau = cosmo.tau_start + (step as f64) * dtau_cosmo;
            let (a, h) = if current_tau <= cosmo.tau_end {
                cosmo.get_params_at_tau(current_tau)
            } else {
                (1.0, 0.0)  // Post-expansion
            };

            // DKD step
            let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };
            if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, dtau_eff) {
                eprintln!("ERROR at step {}: {}", step, e);
                stop_reason = Some(format!("Error: {}", e));
                break;
            }

            let step_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            let ke = gpu_sim.kinetic_energy().unwrap();
            let ke_ratio = ke / ke0;
            steps_completed = step - start_step;

            // Segregation every 100 steps
            let seg = if step % 100 == 0 || step <= start_step + 10 {
                let s = gpu_sim.segregation_distance().unwrap();
                if s > s_max {
                    s_max = s;
                    s_max_step = step;
                }
                s
            } else {
                -1.0
            };

            // Time series
            let tau_display = current_tau.min(cosmo.tau_end);
            writeln!(ts_file, "{},{:.6},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.1}",
                     step, tau_display, a, h, ke, ke_ratio, seg, step_time_ms).unwrap();

            // Snapshot every 200 steps (lightweight - positions only for visualization)
            // Full checkpoint every 1000 steps (with velocities for resume)
            if step % checkpoint_interval == 0 {
                // FULL snapshot with velocities
                let s = if seg < 0.0 { gpu_sim.segregation_distance().unwrap() } else { seg };
                save_full_snapshot(&gpu_sim, step, &output_dir, n_particles,
                                   tau_display, a, s, ke_ratio);
                println!("{:>6}  {:>8.4}  {:>10.6}  {:>10.4}  {:>10.6}  {:>10.1} ms  [FULL]",
                         step, tau_display.min(0.0), a, ke_ratio, s, step_time_ms);
                ts_file.flush().unwrap();

                // JSON checkpoint
                let checkpoint = format!(r#"{{
  "step": {},
  "tau": {:.6},
  "a": {:.6},
  "ke_ratio": {:.6},
  "s_max": {:.6},
  "s_max_step": {},
  "runtime_seconds": {:.1}
}}"#, step, tau_display, a, ke_ratio, s_max, s_max_step, sim_start.elapsed().as_secs_f64());
                fs::write(format!("{}/checkpoint_{:05}.json", output_dir, step), &checkpoint).ok();
            } else if step % snapshot_interval == 0 {
                // Quick report (no snapshot save to reduce I/O)
                let s = if seg < 0.0 { gpu_sim.segregation_distance().unwrap() } else { seg };
                println!("{:>6}  {:>8.4}  {:>10.6}  {:>10.4}  {:>10.6}  {:>10.1} ms",
                         step, tau_display.min(0.0), a, ke_ratio, s, step_time_ms);
                ts_file.flush().unwrap();
            }

            // Auto-stop: KE/KE₀ > 50
            if ke_ratio > 50.0 {
                println!("\n*** AUTO-STOP: KE/KE₀ = {:.2} > 50 at step {} ***", ke_ratio, step);
                stop_reason = Some(format!("KE ratio {:.2} > 50", ke_ratio));
                // Save final full snapshot
                let s = gpu_sim.segregation_distance().unwrap();
                save_full_snapshot(&gpu_sim, step, &output_dir, n_particles,
                                   tau_display, a, s, ke_ratio);
                break;
            }
        }

        ts_file.flush().unwrap();
        drop(ts_file);

        let total_time = sim_start.elapsed();
        let final_ke = gpu_sim.kinetic_energy().unwrap();
        let final_seg = gpu_sim.segregation_distance().unwrap();
        let final_step = start_step + steps_completed;

        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                       FINAL RESULTS                            ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("Steps: {} → {} ({} completed)", start_step, final_step, steps_completed);
        if let Some(ref reason) = stop_reason {
            println!("Stop reason: {}", reason);
        }
        println!("Runtime: {:.2?}", total_time);
        println!("Avg step: {:.1} ms", total_time.as_secs_f64() * 1000.0 / steps_completed.max(1) as f64);
        println!("\nPhysics:");
        println!("  KE/KE₀: {:.4}", final_ke / ke0);
        println!("  S₀: {:.6}", seg0);
        println!("  S_final: {:.6}", final_seg);
        println!("  S_max: {:.6} @ step {}", s_max, s_max_step);

        // Summary
        let stop_reason_json = stop_reason.as_ref().map(|r| format!("\"{}\"", r)).unwrap_or("null".to_string());
        let summary = format!(r#"{{
  "model": "Janus N-body GPU 8M Full",
  "n_particles": {},
  "eta": {:.4},
  "theta": {},
  "start_step": {},
  "final_step": {},
  "steps_completed": {},
  "stop_reason": {},
  "dt": {},
  "initial_ke": {:.6e},
  "final_ke": {:.6e},
  "ke_ratio": {:.6},
  "initial_segregation": {:.6},
  "final_segregation": {:.6},
  "max_segregation": {:.6},
  "max_segregation_step": {},
  "runtime_seconds": {:.1},
  "avg_step_ms": {:.1}
}}"#,
            n_particles, eta, theta, start_step, final_step, steps_completed, stop_reason_json, dt,
            ke0, final_ke, final_ke / ke0,
            seg0, final_seg, s_max, s_max_step,
            total_time.as_secs_f64(),
            total_time.as_secs_f64() * 1000.0 / steps_completed.max(1) as f64);

        fs::write(format!("{}/summary.json", output_dir), &summary).unwrap();
        println!("\nResults saved to {}/", output_dir);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA not enabled. Use: cargo run --release --features cuda --bin run_8m_full");
    }
}

#[derive(Debug)]
struct SnapshotHeader {
    magic: u64,
    version: u64,
    n_particles: u64,
    step: u64,
    tau: f64,
    a: f64,
    ke_ratio: f64,
    segregation: f64,
}

#[cfg(feature = "cuda")]
fn save_full_snapshot(sim: &GpuNBodySimulation, step: usize, output_dir: &str,
                      n_particles: usize, tau: f64, a: f64, seg: f64, ke_ratio: f64) {
    let filename = format!("{}/snapshots/snap_{:05}.bin", output_dir, step);

    let positions = match sim.get_positions() {
        Ok(p) => p,
        Err(_) => return,
    };
    let velocities = match sim.get_velocities() {
        Ok(v) => v,
        Err(_) => return,
    };
    let signs = sim.signs();

    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Header: 64 bytes
    writer.write_all(&SNAPSHOT_MAGIC.to_le_bytes()).unwrap();
    writer.write_all(&SNAPSHOT_VERSION.to_le_bytes()).unwrap();
    writer.write_all(&(n_particles as u64).to_le_bytes()).unwrap();
    writer.write_all(&(step as u64).to_le_bytes()).unwrap();
    writer.write_all(&tau.to_le_bytes()).unwrap();
    writer.write_all(&a.to_le_bytes()).unwrap();
    writer.write_all(&ke_ratio.to_le_bytes()).unwrap();
    writer.write_all(&seg.to_le_bytes()).unwrap();

    // Positions: N × 3 × f64
    for &p in &positions {
        writer.write_all(&p.to_le_bytes()).unwrap();
    }

    // Velocities: N × 3 × f64
    for &v in &velocities {
        writer.write_all(&v.to_le_bytes()).unwrap();
    }

    // Signs: N × i32
    for &s in &signs {
        writer.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn load_full_snapshot(path: &str) -> Result<(SnapshotHeader, Vec<f64>, Vec<f64>, Vec<i32>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Header: 64 bytes
    let mut header_buf = [0u8; 64];
    reader.read_exact(&mut header_buf)?;

    let magic = u64::from_le_bytes(header_buf[0..8].try_into()?);
    let version = u64::from_le_bytes(header_buf[8..16].try_into()?);

    if magic != SNAPSHOT_MAGIC {
        return Err(format!("Invalid snapshot magic: {:016x} (expected {:016x})", magic, SNAPSHOT_MAGIC).into());
    }
    if version != SNAPSHOT_VERSION {
        return Err(format!("Unsupported snapshot version: {} (expected {})", version, SNAPSHOT_VERSION).into());
    }

    let header = SnapshotHeader {
        magic,
        version,
        n_particles: u64::from_le_bytes(header_buf[16..24].try_into()?),
        step: u64::from_le_bytes(header_buf[24..32].try_into()?),
        tau: f64::from_le_bytes(header_buf[32..40].try_into()?),
        a: f64::from_le_bytes(header_buf[40..48].try_into()?),
        ke_ratio: f64::from_le_bytes(header_buf[48..56].try_into()?),
        segregation: f64::from_le_bytes(header_buf[56..64].try_into()?),
    };

    let n = header.n_particles as usize;

    // Positions: N × 3 × f64
    let mut pos_buf = vec![0u8; n * 3 * 8];
    reader.read_exact(&mut pos_buf)?;
    let positions: Vec<f64> = pos_buf.chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Velocities: N × 3 × f64
    let mut vel_buf = vec![0u8; n * 3 * 8];
    reader.read_exact(&mut vel_buf)?;
    let velocities: Vec<f64> = vel_buf.chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Signs: N × i32
    let mut sign_buf = vec![0u8; n * 4];
    reader.read_exact(&mut sign_buf)?;
    let signs: Vec<i32> = sign_buf.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Ok((header, positions, velocities, signs))
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    format!("{:05}_{:02}{:02}{:02}", days, hours, minutes, seconds)
}
