/// Janus N-body GPU — Resume 8M Run from Snapshot
/// Loads positions/signs from snapshot, re-virializes, continues simulation

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Read, Write, BufWriter, BufReader};
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Configuration
        let snapshot_path = std::env::args().nth(1)
            .unwrap_or_else(|| "/app/output/run_8m_20507_115801/snapshots/snap_03000.bin".to_string());
        let start_step: usize = std::env::args().nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(3000);

        // Fixed parameters (matching original run)
        let eta: f64 = 1.045;
        let total_steps: usize = 50000;  // Run until exhaustion
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;
        let box_size: f64 = 430.8869;  // From original config
        let snapshot_interval: usize = 200;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Janus N-body GPU — Resume 8M Run                             ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nLoading snapshot: {}", snapshot_path);

        // Load snapshot
        let (n_particles, positions, signs) = match load_snapshot(&snapshot_path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Failed to load snapshot: {}", e);
                return;
            }
        };

        println!("Loaded {} particles", n_particles);

        // Count positive/negative
        let n_positive = signs.iter().filter(|&&s| s > 0).count();
        let n_negative = n_particles - n_positive;

        println!("  N+ = {} ({:.1}%)", n_positive, n_positive as f64 / n_particles as f64 * 100.0);
        println!("  N- = {} ({:.1}%)", n_negative, n_negative as f64 / n_particles as f64 * 100.0);

        // Output directory
        let timestamp = chrono_lite();
        let output_dir = format!("/app/output/resume_8m_{}", timestamp);
        fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();

        // Save config
        let config = format!(r#"{{
  "resumed_from": "{}",
  "start_step": {},
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {:.4},
  "box_size": {:.4},
  "dt": {},
  "theta": {},
  "output_dir": "{}"
}}"#, snapshot_path, start_step, n_particles, n_positive, n_negative, eta, box_size, dt, theta, output_dir);
        fs::write(format!("{}/config.json", output_dir), &config).ok();

        // Save PID
        let pid = std::process::id();
        fs::write(format!("{}/pid.txt", output_dir), pid.to_string()).ok();

        println!("\nPID: {}", pid);
        println!("Output: {}", output_dir);
        println!("\nParameters:");
        println!("  η = {:.4}", eta);
        println!("  box = {:.2}", box_size);
        println!("  θ = {} (Barnes-Hut)", theta);
        println!("  dt = {}", dt);
        println!("  Snapshot interval: {} steps", snapshot_interval);

        // Initialize simulation with loaded positions
        println!("\n--- Initializing GPU simulation with loaded positions ---");
        let init_start = Instant::now();

        // Initialize velocities to zero (will virialize)
        let velocities = vec![0.0f64; n_particles * 3];

        match GpuNBodySimulation::new_with_state(
            n_positive, n_negative, box_size,
            positions, velocities, signs
        ) {
            Ok(mut gpu_sim) => {
                gpu_sim.set_theta(theta);
                println!("Init time: {:.2?}", init_start.elapsed());

                // Re-virialize at current positions
                println!("\n--- Re-virialization (velocities not in snapshot) ---");
                if let Err(e) = gpu_sim.virialize() {
                    eprintln!("Virialization failed: {}", e);
                }

                let ke0 = gpu_sim.kinetic_energy().unwrap();
                let seg0 = gpu_sim.segregation_distance().unwrap();

                println!("  KE₀ = {:.4e}", ke0);
                println!("  S₀ = {:.6}", seg0);

                // Cosmological interpolator
                println!("\n--- Cosmological Expansion Setup ---");
                let janus_params = JanusParams::from_eta(eta);
                let cosmo = CosmoInterpolator::new(&janus_params, 5.0);

                // Calculate tau at start_step
                let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (12000.0);  // Based on original 12000 steps
                let tau_start_resume = cosmo.tau_start + (start_step as f64) * dtau_cosmo;
                let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);

                let (a_start, _) = cosmo.get_params_at_tau(tau_start_resume);
                println!("  Resuming from τ = {:.4}", tau_start_resume);
                println!("  a(τ) = {:.6}", a_start);

                // Time series file
                let mut ts_file = BufWriter::new(
                    File::create(format!("{}/time_series_resume.csv", output_dir)).unwrap()
                );
                writeln!(ts_file, "step,tau,a,H,ke,ke_ratio,segregation,step_time_ms").unwrap();

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

                    // Check if we've passed the end of the cosmological model
                    if current_tau > cosmo.tau_end {
                        // Continue without expansion (a=1, H=0 effectively)
                        let (a, h) = (1.0, 0.0);
                        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, 0.0) {
                            eprintln!("ERROR at step {}: {}", step, e);
                            stop_reason = Some(format!("Error: {}", e));
                            break;
                        }
                    } else {
                        let (a, h) = cosmo.get_params_at_tau(current_tau);
                        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, dtau_per_dt) {
                            eprintln!("ERROR at step {}: {}", step, e);
                            stop_reason = Some(format!("Error: {}", e));
                            break;
                        }
                    }

                    let step_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    let ke = gpu_sim.kinetic_energy().unwrap();
                    let ke_ratio = ke / ke0;
                    steps_completed = step - start_step;

                    // Segregation every 100 steps
                    let seg = if step % 100 == 0 {
                        let s = gpu_sim.segregation_distance().unwrap();
                        if s > s_max {
                            s_max = s;
                            s_max_step = step;
                        }
                        s
                    } else {
                        -1.0
                    };

                    // Time series (every step)
                    let tau_display = current_tau.min(cosmo.tau_end);
                    let (a_display, h_display) = if current_tau <= cosmo.tau_end {
                        cosmo.get_params_at_tau(current_tau)
                    } else {
                        (1.0, 0.0)
                    };
                    writeln!(ts_file, "{},{:.6},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.1}",
                             step, tau_display, a_display, h_display, ke, ke_ratio, seg, step_time_ms).unwrap();

                    // Snapshot every 200 steps
                    if step % snapshot_interval == 0 {
                        let s = if seg < 0.0 { gpu_sim.segregation_distance().unwrap() } else { seg };
                        save_snapshot(&gpu_sim, step, &output_dir, n_particles, s, ke_ratio);

                        // Report
                        println!("{:>6}  {:>8.4}  {:>10.6}  {:>10.4}  {:>10.6}  {:>10.1} ms",
                                 step, tau_display.min(0.0), a_display, ke_ratio, s, step_time_ms);
                        ts_file.flush().unwrap();
                    }

                    // Checkpoint every 1000 steps
                    if step % 1000 == 0 {
                        let checkpoint = format!(r#"{{
  "step": {},
  "tau": {:.6},
  "a": {:.6},
  "ke_ratio": {:.6},
  "s_max": {:.6},
  "s_max_step": {},
  "runtime_seconds": {:.1}
}}"#, step, tau_display, a_display, ke_ratio, s_max, s_max_step,
                            sim_start.elapsed().as_secs_f64());
                        fs::write(format!("{}/checkpoint_{:05}.json", output_dir, step), &checkpoint).ok();
                        println!("  [CHECKPOINT: s_max={:.6} @ step {}]", s_max, s_max_step);
                    }

                    // Auto-stop: KE/KE₀ > 50
                    if ke_ratio > 50.0 {
                        println!("\n*** AUTO-STOP: KE/KE₀ = {:.2} > 50 at step {} ***", ke_ratio, step);
                        stop_reason = Some(format!("KE ratio {:.2} > 50", ke_ratio));
                        break;
                    }
                }

                ts_file.flush().unwrap();
                drop(ts_file);

                let total_time = sim_start.elapsed();
                let final_ke = gpu_sim.kinetic_energy().unwrap();
                let final_seg = gpu_sim.segregation_distance().unwrap();

                println!("\n╔════════════════════════════════════════════════════════════════╗");
                println!("║                       FINAL RESULTS                            ║");
                println!("╚════════════════════════════════════════════════════════════════╝\n");
                println!("Steps: {} (from {} to {})", steps_completed, start_step, start_step + steps_completed);
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
                let stop_reason_json = match &stop_reason {
                    Some(r) => format!("\"{}\"", r),
                    None => "null".to_string(),
                };
                let summary = format!(r#"{{
  "model": "Janus N-body GPU 8M Resume",
  "resumed_from": "{}",
  "start_step": {},
  "n_particles": {},
  "eta": {:.4},
  "theta": {},
  "steps_completed": {},
  "final_step": {},
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
                    snapshot_path, start_step, n_particles, eta, theta,
                    steps_completed, start_step + steps_completed, stop_reason_json, dt,
                    ke0, final_ke, final_ke / ke0,
                    seg0, final_seg, s_max, s_max_step,
                    total_time.as_secs_f64(),
                    total_time.as_secs_f64() * 1000.0 / steps_completed.max(1) as f64);

                fs::write(format!("{}/summary.json", output_dir), &summary).unwrap();
                println!("\nResults saved to {}/", output_dir);
            }
            Err(e) => {
                eprintln!("Failed to initialize: {}", e);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA not enabled. Use: cargo run --release --features cuda --bin resume_8m");
    }
}

/// Load snapshot in the format: header (32 bytes) + interleaved particle data
fn load_snapshot(path: &str) -> Result<(usize, Vec<f64>, Vec<i32>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Header: n(u64), step(u64), ke_ratio(f64), seg(f64)
    let mut header = [0u8; 32];
    reader.read_exact(&mut header)?;

    let n_particles = u64::from_le_bytes(header[0..8].try_into()?) as usize;
    let step = u64::from_le_bytes(header[8..16].try_into()?) as usize;
    let ke_ratio = f64::from_le_bytes(header[16..24].try_into()?);
    let seg = f64::from_le_bytes(header[24..32].try_into()?);

    println!("Snapshot header:");
    println!("  N = {}", n_particles);
    println!("  Step = {}", step);
    println!("  KE/KE₀ = {:.4}", ke_ratio);
    println!("  S = {:.6}", seg);

    // Data: x(f32), y(f32), z(f32), sign(i8) per particle = 13 bytes
    let mut positions = Vec::with_capacity(n_particles * 3);
    let mut signs = Vec::with_capacity(n_particles);

    let mut particle_buf = [0u8; 13];
    for _ in 0..n_particles {
        reader.read_exact(&mut particle_buf)?;

        let x = f32::from_le_bytes(particle_buf[0..4].try_into()?) as f64;
        let y = f32::from_le_bytes(particle_buf[4..8].try_into()?) as f64;
        let z = f32::from_le_bytes(particle_buf[8..12].try_into()?) as f64;
        let sign = particle_buf[12] as i8;

        positions.push(x);
        positions.push(y);
        positions.push(z);
        signs.push(sign as i32);
    }

    Ok((n_particles, positions, signs))
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

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodySimulation, step: usize, output_dir: &str,
                 n_particles: usize, seg: f64, ke_ratio: f64) {
    use std::io::BufWriter;

    let filename = format!("{}/snapshots/snap_{:05}.bin", output_dir, step);

    if let Ok(positions) = sim.get_positions() {
        let signs = sim.signs();
        let file = File::create(&filename).unwrap();
        let mut writer = BufWriter::new(file);

        // Header: 32 bytes
        writer.write_all(&(n_particles as u64).to_le_bytes()).unwrap();
        writer.write_all(&(step as u64).to_le_bytes()).unwrap();
        writer.write_all(&ke_ratio.to_le_bytes()).unwrap();
        writer.write_all(&seg.to_le_bytes()).unwrap();

        // Interleaved: x(f32), y(f32), z(f32), sign(i8)
        for i in 0..n_particles {
            let x = positions[i * 3] as f32;
            let y = positions[i * 3 + 1] as f32;
            let z = positions[i * 3 + 2] as f32;
            let sign = signs[i];

            writer.write_all(&x.to_le_bytes()).unwrap();
            writer.write_all(&y.to_le_bytes()).unwrap();
            writer.write_all(&z.to_le_bytes()).unwrap();
            writer.write_all(&[sign as u8]).unwrap();
        }
    }
}
