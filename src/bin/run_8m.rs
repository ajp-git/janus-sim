/// Janus N-body GPU — 8M Production Run
/// θ=1.5, 12000 steps, dt=0.005
/// Snapshots every 200 steps, checkpoints every 1000 steps

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Fixed parameters for 8M run
        let n_particles: usize = 8_000_000;
        let eta: f64 = 1.045;
        let steps: usize = 12000;
        let dt: f64 = 0.005;
        let theta: f64 = 1.5;

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

        // Output directory with timestamp
        let timestamp = chrono_lite();
        let output_dir = format!("/app/output/run_8m_{}", timestamp);

        // Create output directories
        fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();

        // Save PID
        let pid = std::process::id();
        fs::write(format!("{}/pid.txt", output_dir), pid.to_string()).ok();

        // Save config
        let config = format!(r#"{{
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {:.4},
  "box_size": {:.4},
  "steps": {},
  "dt": {},
  "theta": {},
  "output_dir": "{}"
}}"#, n_particles, n_positive, n_negative, eta, box_size, steps, dt, theta, output_dir);
        fs::write(format!("{}/config.json", output_dir), &config).ok();

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Janus N-body GPU — 8M Production Run                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nPID: {}", pid);
        println!("Output: {}\n", output_dir);
        println!("Parameters:");
        println!("  N = {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
        println!("  N+ = {} ({:.1}%)", n_positive, n_positive as f64 / n_particles as f64 * 100.0);
        println!("  N- = {} ({:.1}%)", n_negative, n_negative as f64 / n_particles as f64 * 100.0);
        println!("  η = {:.4}", eta);
        println!("  box = {:.2}", box_size);
        println!("  θ = {} (Barnes-Hut)", theta);
        println!("  dt = {}", dt);
        println!("  steps = {}", steps);
        println!("\nIntervals:");
        println!("  Snapshots: every 200 steps");
        println!("  Checkpoints: every 1000 steps");
        println!("  Report S: every 500 steps");
        println!("  Auto-stop: KE/KE₀ > 50");

        // Initialize
        println!("\n--- Initializing GPU simulation ---");
        let init_start = Instant::now();

        match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
            Ok(mut gpu_sim) => {
                gpu_sim.set_theta(theta);
                println!("Init time: {:.2?}", init_start.elapsed());
                println!("Theta set to: {}", gpu_sim.get_theta());

                // Virialize initial conditions
                println!("\n--- Virialization ---");
                if let Err(e) = gpu_sim.virialize() {
                    eprintln!("Virialization failed: {}", e);
                }

                let ke0 = gpu_sim.kinetic_energy().unwrap();
                let seg0 = gpu_sim.segregation_distance().unwrap();

                println!("  KE₀ = {:.4e}", ke0);
                println!("  S₀ = {:.6}", seg0);

                // Initialize cosmological interpolator
                println!("\n--- Cosmological Expansion Setup ---");
                let janus_params = JanusParams::from_eta(eta);
                let cosmo = CosmoInterpolator::new(&janus_params, 5.0);
                let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (steps as f64);
                let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
                println!("  z: 5 → 0");
                println!("  dtau_per_dt = {:.6}", dtau_per_dt);

                // Time series file
                let mut ts_file = BufWriter::new(
                    File::create(format!("{}/time_series.csv", output_dir)).unwrap()
                );
                writeln!(ts_file, "step,tau,a,H,ke,ke_ratio,segregation,step_time_ms").unwrap();

                // Save initial snapshot (step 0)
                save_snapshot(&gpu_sim, 0, &output_dir, n_particles, seg0, 1.0);

                println!("\n--- Starting simulation ---\n");
                println!("{:>6}  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}",
                         "Step", "τ", "a", "KE/KE₀", "S", "Time");
                println!("{}", "─".repeat(70));

                let sim_start = Instant::now();
                let mut s_max = seg0;
                let mut s_max_step = 0usize;
                let mut steps_completed = 0usize;
                let mut stop_reason: Option<String> = None;

                for step in 1..=steps {
                    let step_start = Instant::now();

                    // Get cosmological parameters
                    let current_tau = cosmo.tau_start + (step as f64) * dtau_cosmo;
                    let (a, h) = cosmo.get_params_at_tau(current_tau);

                    // DKD step with GPU tree
                    if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, dtau_per_dt) {
                        eprintln!("ERROR at step {}: {}", step, e);
                        stop_reason = Some(format!("Error: {}", e));
                        break;
                    }

                    let step_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    let ke = gpu_sim.kinetic_energy().unwrap();
                    let ke_ratio = ke / ke0;
                    steps_completed = step;

                    // Calculate segregation every 100 steps (expensive)
                    let seg = if step % 100 == 0 || step <= 10 {
                        let s = gpu_sim.segregation_distance().unwrap();
                        if s > s_max {
                            s_max = s;
                            s_max_step = step;
                        }
                        s
                    } else {
                        -1.0  // Placeholder
                    };

                    // Write to time series (every step)
                    writeln!(ts_file, "{},{:.6},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.1}",
                             step, current_tau, a, h, ke, ke_ratio, seg, step_time_ms).unwrap();

                    // Save snapshot every 200 steps
                    if step % 200 == 0 {
                        let s = if seg < 0.0 { gpu_sim.segregation_distance().unwrap() } else { seg };
                        save_snapshot(&gpu_sim, step, &output_dir, n_particles, s, ke_ratio);
                    }

                    // Report every 500 steps
                    if step % 500 == 0 {
                        let s = if seg < 0.0 { gpu_sim.segregation_distance().unwrap() } else { seg };
                        println!("{:>6}  {:>8.4}  {:>10.6}  {:>10.4}  {:>10.6}  {:>10.1} ms",
                                 step, current_tau, a, ke_ratio, s, step_time_ms);
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
}}"#, step, current_tau, a, ke_ratio, s_max, s_max_step,
                            sim_start.elapsed().as_secs_f64());
                        fs::write(format!("{}/checkpoint_{:05}.json", output_dir, step), &checkpoint).ok();
                        println!("  [CHECKPOINT saved: s_max={:.6} @ step {}]", s_max, s_max_step);
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
                println!("Steps: {} / {}", steps_completed, steps);
                if let Some(ref reason) = stop_reason {
                    println!("Stop reason: {}", reason);
                }
                println!("Runtime: {:.2?}", total_time);
                println!("Avg step: {:.1} ms", total_time.as_secs_f64() * 1000.0 / steps_completed as f64);
                println!("\nPhysics:");
                println!("  KE/KE₀: {:.4}", final_ke / ke0);
                println!("  S₀: {:.6}", seg0);
                println!("  S_final: {:.6}", final_seg);
                println!("  S_max: {:.6} @ step {}", s_max, s_max_step);

                // Final summary
                let stop_reason_json = match &stop_reason {
                    Some(r) => format!("\"{}\"", r),
                    None => "null".to_string(),
                };
                let summary = format!(r#"{{
  "model": "Janus N-body GPU 8M",
  "n_particles": {},
  "eta": {:.4},
  "theta": {},
  "steps_requested": {},
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
                    n_particles, eta, theta, steps, steps_completed, stop_reason_json, dt,
                    ke0, final_ke, final_ke / ke0,
                    seg0, final_seg, s_max, s_max_step,
                    total_time.as_secs_f64(),
                    total_time.as_secs_f64() * 1000.0 / steps_completed as f64);

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
        println!("CUDA not enabled. Use: cargo run --release --features cuda --bin run_8m");
    }
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    // Simple timestamp: YYYYMMDD_HHMMSS approximation
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    // Approximate date (good enough for unique ID)
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

        // Header: 32 bytes (per KNOWN_FIXES.md format)
        // n(u64), step(u64), scale_factor(f64), segregation(f64)
        writer.write_all(&(n_particles as u64).to_le_bytes()).unwrap();
        writer.write_all(&(step as u64).to_le_bytes()).unwrap();
        writer.write_all(&ke_ratio.to_le_bytes()).unwrap();  // Using ke_ratio as scale proxy
        writer.write_all(&seg.to_le_bytes()).unwrap();

        // Interleaved format: x(f32), y(f32), z(f32), sign(i8) per particle
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
