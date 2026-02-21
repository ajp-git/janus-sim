/// Janus N-body GPU Production Simulation
/// 5M particles, 300 steps with monitoring and export

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 5_000_000;
        let eta = 1.045;
        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let box_size = 368.4;  // From 10-step test
        let steps = 300;
        let dt = 0.001;
        let sample_size = 100_000;  // For visualization

        // Create output directory
        fs::create_dir_all("output/phase1c").ok();

        // Save PID
        let pid = std::process::id();
        fs::write("output/phase1c/pid.txt", pid.to_string()).ok();

        println!("======================================================================");
        println!("Janus N-body GPU — 5M Production Run");
        println!("======================================================================");
        println!("PID: {}", pid);
        println!("\nParameters:");
        println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
        println!("  eta = {:.3}", eta);
        println!("  box = {:.1}", box_size);
        println!("  steps = {}", steps);
        println!("  dt = {}", dt);
        println!("  sample = {} particles for viz", sample_size);

        // Initialize
        println!("\n--- Initializing ---");
        let init_start = Instant::now();

        match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
            Ok(mut gpu_sim) => {
                println!("Init time: {:.2?}", init_start.elapsed());

                let ke0 = gpu_sim.kinetic_energy().unwrap();
                let seg0 = gpu_sim.segregation_distance().unwrap();

                println!("Initial KE: {:.4e}", ke0);
                println!("Initial Seg: {:.4}", seg0);

                // Time series file
                let mut ts_file = BufWriter::new(File::create("output/phase1c/time_series.csv").unwrap());
                writeln!(ts_file, "step,time,ke,ke_ratio,segregation,step_time_s").unwrap();

                // Header for progress
                println!("\n{:>6}  {:>10}  {:>10}  {:>12}  {:>10}",
                         "Step", "KE/KE0", "Seg", "Seg Δ%", "Time");
                println!("{:-<60}", "");

                let sim_start = Instant::now();
                let mut should_stop = false;

                for step in 1..=steps {
                    let step_start = Instant::now();

                    if let Err(e) = gpu_sim.step(dt) {
                        eprintln!("ERROR at step {}: {}", step, e);
                        break;
                    }

                    let step_time = step_start.elapsed();
                    let ke = gpu_sim.kinetic_energy().unwrap();
                    let seg = gpu_sim.segregation_distance().unwrap();
                    let ke_ratio = ke / ke0;
                    let seg_pct = (seg - seg0) / seg0 * 100.0;
                    let sim_time = step as f64 * dt;

                    // Write to time series
                    writeln!(ts_file, "{},{:.6},{:.6e},{:.6},{:.6},{:.3}",
                             step, sim_time, ke, ke_ratio, seg, step_time.as_secs_f64()).unwrap();

                    // Progress at key steps
                    if step == 20 || step == 50 || step == 100 || step % 50 == 0 || step == steps {
                        println!("{:>6}  {:>10.4}  {:>10.4}  {:>+11.2}%  {:>10.1?}",
                                 step, ke_ratio, seg, seg_pct, step_time);

                        // Export snapshot
                        export_snapshot(&gpu_sim, step, sample_size, n_positive);
                    }

                    // Monitoring at step 100
                    if step == 100 {
                        ts_file.flush().unwrap();

                        if seg < 0.025 {
                            println!("\n*** STOP: Segregation {:.4} < 0.025 at step 100 ***", seg);
                            should_stop = true;
                        } else if seg > 0.030 {
                            println!("\n*** CONTINUE: Segregation {:.4} > 0.030 at step 100 ***", seg);
                        } else {
                            println!("\n*** MARGINAL: Segregation {:.4} in [0.025, 0.030] ***", seg);
                        }

                        if should_stop {
                            break;
                        }
                    }
                }

                ts_file.flush().unwrap();
                drop(ts_file);

                let total_time = sim_start.elapsed();
                let final_ke = gpu_sim.kinetic_energy().unwrap();
                let final_seg = gpu_sim.segregation_distance().unwrap();

                println!("\n======================================================================");
                println!("FINAL RESULTS");
                println!("======================================================================");
                println!("Runtime: {:.2?}", total_time);
                println!("KE/KE0: {:.4}", final_ke / ke0);
                println!("Segregation: {:.4} -> {:.4} ({:+.2}%)",
                         seg0, final_seg, (final_seg - seg0) / seg0 * 100.0);

                // Summary JSON
                let summary = format!(r#"{{
  "model": "Janus N-body GPU",
  "n_particles": {},
  "eta": {:.4},
  "steps_completed": {},
  "dt": {},
  "initial_ke": {:.6e},
  "final_ke": {:.6e},
  "ke_ratio": {:.6},
  "initial_segregation": {:.6},
  "final_segregation": {:.6},
  "runtime_seconds": {:.1},
  "stopped_early": {}
}}"#,
                    n_particles, eta,
                    if should_stop { 100 } else { steps },
                    dt, ke0, final_ke, final_ke / ke0,
                    seg0, final_seg, total_time.as_secs_f64(), should_stop);

                fs::write("output/phase1c/summary.json", &summary).unwrap();
                println!("\nResults saved to output/phase1c/");
            }
            Err(e) => {
                eprintln!("Failed to initialize: {}", e);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA not enabled. Use: cargo run --release --features cuda --bin nbody_gpu_prod");
    }
}

#[cfg(feature = "cuda")]
fn export_snapshot(sim: &GpuNBodySimulation, step: usize, sample_size: usize, n_positive: usize) {
    use std::io::BufWriter;

    let filename = format!("output/phase1c/snapshot_{:04}.csv", step);
    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    writeln!(writer, "x,y,z,vx,vy,vz,sign").unwrap();

    // Get particle data from simulation
    if let (Ok(positions), Ok(velocities)) = (sim.get_positions(), sim.get_velocities()) {
        let n_total = positions.len() / 3;
        let sample_rate = n_total / sample_size;

        for i in (0..n_total).step_by(sample_rate.max(1)) {
            let sign = if i < n_positive { 1 } else { -1 };
            writeln!(writer, "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                     positions[i*3], positions[i*3+1], positions[i*3+2],
                     velocities[i*3], velocities[i*3+1], velocities[i*3+2],
                     sign).unwrap();
        }
    }
}
