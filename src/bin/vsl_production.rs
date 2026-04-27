//! VSL Production Run — Full Janus simulation with snapshots
//!
//! Usage:
//!   cargo run --release --features cuda --bin vsl_production -- \
//!     --n 2000000 --box 100 --mu 19 --steps 2000 --output /app/output/janus_vsl_2M

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const N_CELLS: usize = 32;
const MPC_GYR_TO_KMS: f64 = 977.8;  // 1 Mpc/Gyr = 977.8 km/s

#[cfg(feature = "cuda")]
fn main() {
    run_vsl_production();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
}

#[cfg(feature = "cuda")]
fn run_vsl_production() {
    let args: Vec<String> = std::env::args().collect();

    let n_particles: usize = args.iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);

    let box_size: f64 = args.iter()
        .position(|a| a == "--box")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100.0);

    let mu: f64 = args.iter()
        .position(|a| a == "--mu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(19.0);

    let n_steps: usize = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let dt: f64 = args.iter()
        .position(|a| a == "--dt")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);

    let output_dir: PathBuf = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/app/output/janus_vsl_2M"));

    let snap_interval: usize = args.iter()
        .position(|a| a == "--snap-interval")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // VSL: c_ratio = 1/sqrt(mu) from Petit MPLA 2014
    let c_ratio = 1.0 / mu.sqrt();
    let c_ratio_sq = 1.0 / mu;

    // Janus mass fractions
    let eta = 1.045;
    let f_plus = eta / (1.0 + eta);
    let n_positive = (n_particles as f64 * f_plus).round() as usize;
    let n_negative = n_particles.saturating_sub(n_positive);

    // Create output directories
    let snap_dir = output_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         VSL PRODUCTION — Janus Bimetric Simulation           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", n_particles, n_positive, n_negative);
    println!("║  Box = {} Mpc", box_size);
    println!("║  μ = {} → c⁻/c⁺ = {:.4} → (c⁻/c⁺)² = {:.4}", mu, c_ratio, c_ratio_sq);
    println!("║  dt = {} Gyr, steps = {}", dt, n_steps);
    println!("║  Snapshots every {} steps → {}", snap_interval, snap_dir.display());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Initialize GPU simulation
    println!("Initializing GPU...");
    let gpu_start = Instant::now();

    let mut gpu_sim = match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        }
    };

    // Set VSL c_ratio
    gpu_sim.set_c_ratio(c_ratio);
    gpu_sim.set_theta(0.8);  // Accurate mode for production

    println!("GPU init: {:.2}s, c_ratio_sq = {:.4}, theta = {:.1}",
             gpu_start.elapsed().as_secs_f64(),
             gpu_sim.get_c_ratio_sq(),
             gpu_sim.get_theta());

    // Create evolution CSV
    let csv_path = output_dir.join("evolution.csv");
    let mut csv_file = BufWriter::new(File::create(&csv_path).expect("Failed to create CSV"));
    writeln!(csv_file, "step,time_gyr,rho_plus_max,rho_minus_max,delta_max,v_rms_plus,v_rms_minus,v_mean_plus,v_mean_minus,step_time_s").unwrap();

    let half_box = box_size / 2.0;
    let cell_size = box_size / N_CELLS as f64;

    println!("\n{:>6} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>10}",
             "Step", "ρ+_max", "ρ-_max", "v_rms+", "v_rms-", "δ_max", "time");
    println!("{:-<80}", "");

    let sim_start = Instant::now();
    let mut last_step_time = Instant::now();
    let mut time_gyr = 0.0;

    for step in 0..=n_steps {
        // Get positions and velocities
        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to get positions: {}", e);
                break;
            }
        };

        let vel = match gpu_sim.get_velocities() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to get velocities: {}", e);
                break;
            }
        };

        let signs = gpu_sim.signs();

        // Compute density in cells
        let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut counts_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];

        // Velocity statistics
        let mut v2_plus = 0.0f64;
        let mut v2_minus = 0.0f64;
        let mut v_plus = 0.0f64;
        let mut v_minus = 0.0f64;
        let mut n_plus_count = 0usize;
        let mut n_minus_count = 0usize;

        for i in 0..n_particles {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];
            let vx = vel[i * 3];
            let vy = vel[i * 3 + 1];
            let vz = vel[i * 3 + 2];

            let v_mag = (vx * vx + vy * vy + vz * vz).sqrt();

            let ix = ((px + half_box) / cell_size).floor() as usize % N_CELLS;
            let iy = ((py + half_box) / cell_size).floor() as usize % N_CELLS;
            let iz = ((pz + half_box) / cell_size).floor() as usize % N_CELLS;
            let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;

            if signs[i] > 0 {
                counts_plus[idx] += 1;
                v2_plus += vx * vx + vy * vy + vz * vz;
                v_plus += v_mag;
                n_plus_count += 1;
            } else {
                counts_minus[idx] += 1;
                v2_minus += vx * vx + vy * vy + vz * vz;
                v_minus += v_mag;
                n_minus_count += 1;
            }
        }

        let rho_plus_max = *counts_plus.iter().max().unwrap_or(&0) as f64;
        let rho_minus_max = *counts_minus.iter().max().unwrap_or(&0) as f64;
        let mean_count = n_particles as f64 / (N_CELLS * N_CELLS * N_CELLS) as f64;
        let delta_max = (rho_plus_max / mean_count - 1.0).max(rho_minus_max / mean_count - 1.0);

        // Velocity RMS in km/s
        let v_rms_plus = if n_plus_count > 0 {
            (v2_plus / n_plus_count as f64).sqrt() * MPC_GYR_TO_KMS
        } else { 0.0 };
        let v_rms_minus = if n_minus_count > 0 {
            (v2_minus / n_minus_count as f64).sqrt() * MPC_GYR_TO_KMS
        } else { 0.0 };
        let v_mean_plus = if n_plus_count > 0 {
            (v_plus / n_plus_count as f64) * MPC_GYR_TO_KMS
        } else { 0.0 };
        let v_mean_minus = if n_minus_count > 0 {
            (v_minus / n_minus_count as f64) * MPC_GYR_TO_KMS
        } else { 0.0 };

        let step_time = last_step_time.elapsed().as_secs_f64();
        last_step_time = Instant::now();

        // Write to CSV
        writeln!(csv_file, "{},{:.6},{:.0},{:.0},{:.4},{:.2},{:.2},{:.2},{:.2},{:.3}",
                 step, time_gyr, rho_plus_max, rho_minus_max, delta_max,
                 v_rms_plus, v_rms_minus, v_mean_plus, v_mean_minus, step_time).unwrap();

        // Progress output every 10 steps
        if step % 10 == 0 || step == n_steps {
            let elapsed = sim_start.elapsed().as_secs_f64();
            println!("{:>6} | {:>10.0} | {:>10.0} | {:>7.0} | {:>7.0} | {:>8.2} | {:>9.1}s",
                     step, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, delta_max, elapsed);
        }

        // Save snapshot
        if step % snap_interval == 0 {
            let snap_path = snap_dir.join(format!("snap_{:05}.bin", step));
            save_snapshot(&snap_path, &pos, &vel, &signs, n_particles, box_size as f32, step as u32, 0.0);
        }

        if step >= n_steps {
            break;
        }

        // Step using GPU tree (DKD integrator with BVH)
        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0) {
            eprintln!("GPU step failed: {}", e);
            break;
        }

        time_gyr += dt;
    }

    csv_file.flush().unwrap();

    let total_time = sim_start.elapsed().as_secs_f64();
    let avg_step_time = total_time / n_steps as f64;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE: {:.1}s for {} steps ({:.2}s/step)", total_time, n_steps, avg_step_time);
    println!("║  Output: {}", output_dir.display());
    println!("║  Snapshots: {}", snap_dir.display());
    println!("║  Evolution: {}", csv_path.display());
    println!("╚══════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn save_snapshot(path: &std::path::PathBuf, pos: &[f64], vel: &[f64], signs: &[i32],
                 n: usize, box_size: f32, step: u32, z: f32) {
    use std::io::Write;

    let mut file = BufWriter::new(File::create(path).expect("Failed to create snapshot"));

    // Header: n(u32) + box(f32) + step(u32) + z(f32) = 16 bytes
    file.write_all(&(n as u32).to_le_bytes()).unwrap();
    file.write_all(&box_size.to_le_bytes()).unwrap();
    file.write_all(&step.to_le_bytes()).unwrap();
    file.write_all(&z.to_le_bytes()).unwrap();

    // Per-particle: x,y,z,vx,vy,vz (f32×6) + sign (i8) = 25 bytes
    for i in 0..n {
        file.write_all(&(pos[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(pos[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(pos[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        file.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
    }

    file.flush().unwrap();
}
