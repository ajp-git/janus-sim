/// Janus N-body GPU Overnight Production Run
/// Saves NPZ snapshots and generates PNG frames at every step

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use std::process::Command;

fn main() {
    #[cfg(feature = "cuda")]
    {
        // Parse command line arguments
        let args: Vec<String> = std::env::args().collect();

        let n_particles: usize = args.iter()
            .position(|a| a == "--n")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(500_000);

        let eta: f64 = args.iter()
            .position(|a| a == "--eta")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.045);

        let steps: usize = args.iter()
            .position(|a| a == "--steps")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000);

        let dt: f64 = args.iter()
            .position(|a| a == "--dt")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.01);

        let output_dir: String = args.iter()
            .position(|a| a == "--output")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.clone())
            .unwrap_or_else(|| "output/run".to_string());

        let test_only: bool = args.iter().any(|a| a == "--test");

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

        // Create output directories
        fs::create_dir_all(format!("{}/frames", output_dir)).ok();
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
  "output_dir": "{}"
}}"#, n_particles, n_positive, n_negative, eta, box_size, steps, dt, output_dir);
        fs::write(format!("{}/config.json", output_dir), &config).ok();

        println!("======================================================================");
        println!("Janus N-body GPU — Overnight Production Run");
        println!("======================================================================");
        println!("PID: {}", pid);
        println!("\nParameters:");
        println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
        println!("  eta = {:.4}", eta);
        println!("  box = {:.4}", box_size);
        println!("  steps = {}", steps);
        println!("  dt = {}", dt);
        println!("  output = {}", output_dir);

        // Initialize
        println!("\n--- Initializing GPU simulation ---");
        let init_start = Instant::now();

        match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
            Ok(mut gpu_sim) => {
                println!("Init time: {:.2?}", init_start.elapsed());

                // Virialize initial conditions
                println!("\n--- Virialization ---");
                if let Err(e) = gpu_sim.virialize() {
                    eprintln!("Virialization failed: {}", e);
                }

                let ke0 = gpu_sim.kinetic_energy().unwrap();
                let seg0 = gpu_sim.segregation_distance().unwrap();

                println!("\nPost-virialization state:");
                println!("  KE₀ = {:.4e}", ke0);
                println!("  Seg₀ = {:.4}", seg0);

                // Time series file
                let mut ts_file = BufWriter::new(
                    File::create(format!("{}/time_series.csv", output_dir)).unwrap()
                );
                writeln!(ts_file, "step,time,ke,ke_ratio,segregation,step_time_s").unwrap();

                // Save initial state (step 0)
                save_snapshot(&gpu_sim, 0, &output_dir, n_positive, n_particles,
                              eta, box_size, 0.0, seg0, 1.0);

                // Render initial frame
                render_frame(0, &output_dir, eta, n_particles, 0.0, seg0, 1.0);

                if test_only {
                    println!("\n*** TEST MODE: Step 0 saved. Exiting. ***");
                    return;
                }

                // Header for progress
                println!("\n{:>6}  {:>10}  {:>10}  {:>12}  {:>10}",
                         "Step", "KE/KE0", "Seg", "Seg Δ%", "Time");
                println!("{:-<60}", "");

                let sim_start = Instant::now();

                // Auto-stop tracking
                let mut prev_seg = seg0;
                let mut seg_decrease_count = 0usize;
                let mut steps_completed = 0usize;
                let mut stop_reason: Option<String> = None;

                for step in 1..=steps {
                    let step_start = Instant::now();

                    if let Err(e) = gpu_sim.step(dt) {
                        eprintln!("ERROR at step {}: {}", step, e);
                        stop_reason = Some(format!("Error: {}", e));
                        break;
                    }

                    let step_time = step_start.elapsed();
                    let ke = gpu_sim.kinetic_energy().unwrap();
                    let seg = gpu_sim.segregation_distance().unwrap();
                    let ke_ratio = ke / ke0;
                    let seg_pct = (seg - seg0) / seg0 * 100.0;
                    let sim_time = step as f64 * dt;
                    steps_completed = step;

                    // Write to time series
                    writeln!(ts_file, "{},{:.6},{:.6e},{:.6},{:.6},{:.3}",
                             step, sim_time, ke, ke_ratio, seg, step_time.as_secs_f64()).unwrap();

                    // Save snapshot and render frame at every step
                    save_snapshot(&gpu_sim, step, &output_dir, n_positive, n_particles,
                                  eta, box_size, sim_time, seg, ke_ratio);
                    render_frame(step, &output_dir, eta, n_particles, sim_time, seg, ke_ratio);

                    // Progress at key steps
                    if step == 500 || step == 2000 || step == 5000 || step == 10000
                       || step % 1000 == 0 || step <= 100 && step % 10 == 0 {
                        println!("{:>6}  {:>10.4}  {:>10.4}  {:>+11.2}%  {:>10.1?}",
                                 step, ke_ratio, seg, seg_pct, step_time);
                        ts_file.flush().unwrap();
                    }

                    // === AUTO-STOP CONDITIONS ===

                    // Condition 1: KE/KE₀ > 50
                    if ke_ratio > 50.0 {
                        println!("\n*** AUTO-STOP: KE/KE₀ = {:.2} > 50 at step {} ***", ke_ratio, step);
                        stop_reason = Some(format!("KE ratio {} > 50", ke_ratio));
                        break;
                    }

                    // Condition 2: Seg decreases for 500 consecutive steps
                    if seg < prev_seg {
                        seg_decrease_count += 1;
                        if seg_decrease_count >= 500 {
                            println!("\n*** AUTO-STOP: Seg decreased for 500 consecutive steps at step {} ***", step);
                            stop_reason = Some("Seg decreased 500 consecutive steps".to_string());
                            break;
                        }
                    } else {
                        seg_decrease_count = 0;
                    }
                    prev_seg = seg;
                }

                ts_file.flush().unwrap();
                drop(ts_file);

                let total_time = sim_start.elapsed();
                let final_ke = gpu_sim.kinetic_energy().unwrap();
                let final_seg = gpu_sim.segregation_distance().unwrap();

                println!("\n======================================================================");
                println!("FINAL RESULTS");
                println!("======================================================================");
                println!("Steps completed: {} / {}", steps_completed, steps);
                if let Some(ref reason) = stop_reason {
                    println!("Stop reason: {}", reason);
                }
                println!("Runtime: {:.2?}", total_time);
                println!("KE/KE0: {:.4}", final_ke / ke0);
                println!("Segregation: {:.4} -> {:.4} ({:+.2}%)",
                         seg0, final_seg, (final_seg - seg0) / seg0 * 100.0);

                // Summary JSON
                let stop_reason_json = match &stop_reason {
                    Some(r) => format!("\"{}\"", r),
                    None => "null".to_string(),
                };
                let summary = format!(r#"{{
  "model": "Janus N-body GPU Overnight",
  "n_particles": {},
  "eta": {:.4},
  "steps_requested": {},
  "steps_completed": {},
  "stop_reason": {},
  "dt": {},
  "initial_ke": {:.6e},
  "final_ke": {:.6e},
  "ke_ratio": {:.6},
  "initial_segregation": {:.6},
  "final_segregation": {:.6},
  "runtime_seconds": {:.1}
}}"#,
                    n_particles, eta, steps, steps_completed, stop_reason_json, dt,
                    ke0, final_ke, final_ke / ke0,
                    seg0, final_seg, total_time.as_secs_f64());

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
        println!("CUDA not enabled. Use: cargo run --release --features cuda --bin nbody_overnight");
    }
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodySimulation, step: usize, output_dir: &str,
                 n_positive: usize, n_particles: usize, eta: f64, box_size: f64,
                 sim_time: f64, seg: f64, ke_ratio: f64) {
    use std::io::BufWriter;

    // Save as binary format for efficiency (positions only, signs can be reconstructed)
    let filename = format!("{}/snapshots/snap_{:05}.bin", output_dir, step);

    if let Ok(positions) = sim.get_positions() {
        let file = File::create(&filename).unwrap();
        let mut writer = BufWriter::new(file);

        // Header: n_particles, n_positive, eta, box_size, step, time, seg, ke_ratio
        writer.write_all(&(n_particles as u64).to_le_bytes()).unwrap();
        writer.write_all(&(n_positive as u64).to_le_bytes()).unwrap();
        writer.write_all(&eta.to_le_bytes()).unwrap();
        writer.write_all(&box_size.to_le_bytes()).unwrap();
        writer.write_all(&(step as u64).to_le_bytes()).unwrap();
        writer.write_all(&sim_time.to_le_bytes()).unwrap();
        writer.write_all(&seg.to_le_bytes()).unwrap();
        writer.write_all(&ke_ratio.to_le_bytes()).unwrap();

        // Positions as f64 array
        for &p in &positions {
            writer.write_all(&p.to_le_bytes()).unwrap();
        }
    }
}

#[cfg(feature = "cuda")]
fn render_frame(step: usize, output_dir: &str, eta: f64, n_particles: usize,
                sim_time: f64, seg: f64, ke_ratio: f64) {
    // Call Python renderer
    let snapshot_path = format!("{}/snapshots/snap_{:05}.bin", output_dir, step);
    let frame_path = format!("{}/frames/frame_{:05}.png", output_dir, step);

    let _ = Command::new("python3")
        .args(&[
            "/app/scripts/render_overnight.py",
            &snapshot_path,
            &frame_path,
            &format!("{:.4}", eta),
            &format!("{}", n_particles),
            &format!("{}", step),
            &format!("{:.4}", sim_time),
            &format!("{:.4}", seg),
            &format!("{:.2}", ke_ratio),
        ])
        .output();
}
