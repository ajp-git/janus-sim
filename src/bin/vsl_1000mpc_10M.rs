//! VSL Production Run — 1000 Mpc Box, 10M Particles
//!
//! Validated parameters from vsl_test_bigbox.rs:
//! - 1000 Mpc box reduces m- runaway (v_rms- = 437 km/s at step 50)
//! - Thermal ICs: σ_v = 12 km/s
//! - μ = 19 → c⁻/c⁺ = 0.2294

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

const MPC_GYR_TO_KMS: f64 = 977.8;
const N_CELLS: usize = 64;

// Production parameters
const N_PARTICLES: usize = 10_000_000;
const BOX_SIZE: f64 = 1000.0;  // 1000 Mpc box
const MU: f64 = 19.0;
const N_STEPS: usize = 30000;
const DT: f64 = 0.001;
const SNAP_INTERVAL: usize = 5;
const CSV_INTERVAL: usize = 50;

// Janus
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;

// Thermal velocity: σ_v = sqrt(k_B T / (mu_mol m_p))
const K_B_OVER_MP_CODE: f64 = 8.7e-9;
const T_INIT: f64 = 1.0e4;
const MU_MOL: f64 = 0.6;

// Emergency stops
const V_RMS_MINUS_MAX: f64 = 50000.0;  // km/s
const V_RMS_PLUS_MAX: f64 = 10000.0;   // km/s
const RHO_MAX_STOP: f64 = 1e8;

#[cfg(feature = "cuda")]
fn main() {
    run_production();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
}

#[cfg(feature = "cuda")]
fn run_production() {
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    let output_dir = "/app/output/janus_vsl_1000mpc_10M";
    let snap_dir = format!("{}/snapshots", output_dir);

    fs::create_dir_all(&snap_dir).expect("Failed to create output directories");

    let c_ratio = 1.0 / MU.sqrt();
    let c_ratio_sq = 1.0 / MU;

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    // Thermal velocity dispersion
    let sigma_v = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();
    let sigma_v_kms = sigma_v * MPC_GYR_TO_KMS;

    // Density
    let density = N_PARTICLES as f64 / (BOX_SIZE * BOX_SIZE * BOX_SIZE);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL PRODUCTION — 1000 Mpc, 10M Particles                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc → density = {:.2e} /Mpc³", BOX_SIZE, density);
    println!("║  μ = {} → c⁻/c⁺ = {:.4} → (c⁻/c⁺)² = {:.5}", MU, c_ratio, c_ratio_sq);
    println!("║  T_init = {} K → σ_v = {:.1} km/s", T_INIT, sigma_v_kms);
    println!("║  z_init = {}, dt = {} Gyr, steps = {}", Z_INIT, DT, N_STEPS);
    println!("║  Snapshots: every {} steps → {} files", SNAP_INTERVAL, N_STEPS / SNAP_INTERVAL);
    println!("║  Output: {}", output_dir);
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("Initializing GPU with {} particles...", N_PARTICLES);
    let gpu_start = Instant::now();

    let mut gpu_sim = match GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        }
    };

    gpu_sim.set_c_ratio(c_ratio);
    gpu_sim.set_theta(0.7);

    println!("GPU init: {:.2}s, c_ratio_sq = {:.5}, theta = 0.7",
             gpu_start.elapsed().as_secs_f64(), c_ratio_sq);

    // Replace virial velocities with thermal velocities
    println!("Setting thermal velocities (σ = {:.1} km/s)...", sigma_v_kms);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, sigma_v).unwrap();

    let mut thermal_vel = vec![0.0f64; N_PARTICLES * 3];
    for i in 0..N_PARTICLES {
        thermal_vel[i * 3]     = normal.sample(&mut rng);
        thermal_vel[i * 3 + 1] = normal.sample(&mut rng);
        thermal_vel[i * 3 + 2] = normal.sample(&mut rng);
    }

    if let Err(e) = gpu_sim.set_velocities(&thermal_vel) {
        eprintln!("Failed to set thermal velocities: {}", e);
        return;
    }
    drop(thermal_vel);
    println!("Thermal velocities set.\n");

    // CSV file for evolution
    let csv_path = format!("{}/evolution.csv", output_dir);
    let csv_file = File::create(&csv_path).expect("Failed to create CSV");
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,rho_plus_max,rho_minus_max,delta_max,v_rms_plus,v_rms_minus,v_mean_plus,v_mean_minus,segregation,step_time_s").unwrap();

    let half_box = BOX_SIZE / 2.0;
    let cell_size = BOX_SIZE / N_CELLS as f64;

    println!("{:>6} | {:>6} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>12}",
             "Step", "z", "ρ+_max", "ρ-_max", "v_rms+", "v_rms-", "Seg", "time");
    println!("{:-<100}", "");

    let sim_start = Instant::now();
    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;
    let mut step_start = Instant::now();

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;
        let t_gyr = step as f64 * DT;

        // Get data from GPU
        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to get positions: {}", e); break; }
        };
        let vel = match gpu_sim.get_velocities() {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to get velocities: {}", e); break; }
        };
        let signs = gpu_sim.signs();

        // Compute metrics
        let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut counts_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut v2_plus = 0.0f64;
        let mut v2_minus = 0.0f64;
        let mut v_sum_plus = 0.0f64;
        let mut v_sum_minus = 0.0f64;
        let mut n_plus = 0usize;
        let mut n_minus = 0usize;
        let mut com_plus = [0.0f64; 3];
        let mut com_minus = [0.0f64; 3];

        for i in 0..N_PARTICLES {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];
            let vx = vel[i * 3];
            let vy = vel[i * 3 + 1];
            let vz = vel[i * 3 + 2];
            let v2 = vx * vx + vy * vy + vz * vz;
            let v = v2.sqrt();

            let ix = ((px + half_box) / cell_size).floor() as usize % N_CELLS;
            let iy = ((py + half_box) / cell_size).floor() as usize % N_CELLS;
            let iz = ((pz + half_box) / cell_size).floor() as usize % N_CELLS;
            let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;

            if signs[i] > 0 {
                counts_plus[idx] += 1;
                v2_plus += v2;
                v_sum_plus += v;
                com_plus[0] += px;
                com_plus[1] += py;
                com_plus[2] += pz;
                n_plus += 1;
            } else {
                counts_minus[idx] += 1;
                v2_minus += v2;
                v_sum_minus += v;
                com_minus[0] += px;
                com_minus[1] += py;
                com_minus[2] += pz;
                n_minus += 1;
            }
        }

        let rho_plus_max = *counts_plus.iter().max().unwrap_or(&0) as f64;
        let rho_minus_max = *counts_minus.iter().max().unwrap_or(&0) as f64;
        let rho_mean = N_PARTICLES as f64 / (N_CELLS * N_CELLS * N_CELLS) as f64;
        let delta_max = (rho_plus_max.max(rho_minus_max) - rho_mean) / rho_mean;

        let v_rms_plus = if n_plus > 0 { (v2_plus / n_plus as f64).sqrt() * MPC_GYR_TO_KMS } else { 0.0 };
        let v_rms_minus = if n_minus > 0 { (v2_minus / n_minus as f64).sqrt() * MPC_GYR_TO_KMS } else { 0.0 };
        let v_mean_plus = if n_plus > 0 { v_sum_plus / n_plus as f64 * MPC_GYR_TO_KMS } else { 0.0 };
        let v_mean_minus = if n_minus > 0 { v_sum_minus / n_minus as f64 * MPC_GYR_TO_KMS } else { 0.0 };

        // Segregation (COM distance / box)
        if n_plus > 0 {
            com_plus[0] /= n_plus as f64;
            com_plus[1] /= n_plus as f64;
            com_plus[2] /= n_plus as f64;
        }
        if n_minus > 0 {
            com_minus[0] /= n_minus as f64;
            com_minus[1] /= n_minus as f64;
            com_minus[2] /= n_minus as f64;
        }
        let dx = com_plus[0] - com_minus[0];
        let dy = com_plus[1] - com_minus[1];
        let dz = com_plus[2] - com_minus[2];
        let segregation = (dx * dx + dy * dy + dz * dz).sqrt() / BOX_SIZE;

        let step_time = step_start.elapsed().as_secs_f64();
        step_start = Instant::now();

        // Write CSV
        if step % CSV_INTERVAL == 0 {
            writeln!(csv_writer, "{},{:.6},{:.4},{},{},{:.4},{:.2},{:.2},{:.2},{:.2},{:.4},{:.3}",
                     step, t_gyr, z, rho_plus_max as u64, rho_minus_max as u64, delta_max,
                     v_rms_plus, v_rms_minus, v_mean_plus, v_mean_minus, segregation, step_time).unwrap();
            csv_writer.flush().unwrap();
        }

        // Save snapshot
        if step % SNAP_INTERVAL == 0 {
            let snap_path = format!("{}/snap_{:06}.bin", snap_dir, step);
            save_snapshot(&snap_path, &pos, &signs, z, BOX_SIZE);
        }

        // Output
        if step % 100 == 0 || step <= 10 {
            let elapsed = sim_start.elapsed().as_secs_f64();
            let eta_h = if step > 0 { elapsed / step as f64 * (N_STEPS - step) as f64 / 3600.0 } else { 0.0 };

            println!("{:>6} | {:>6.3} | {:>10.0} | {:>10.0} | {:>7.0} | {:>7.0} | {:>7.4} | {:>7.1}s (ETA {:.1}h)",
                     step, z, rho_plus_max, rho_minus_max, v_rms_plus, v_rms_minus, segregation, elapsed, eta_h);
        }

        // Validation at step 50
        if step == 50 {
            if v_rms_minus < 10000.0 {
                println!("✓ Step 50 validation PASSED: v_rms_minus = {:.0} km/s < 10000 km/s", v_rms_minus);
            } else {
                println!("⚠ Step 50: v_rms_minus = {:.0} km/s (elevated but continuing)", v_rms_minus);
            }
        }

        // Emergency stops
        if v_rms_minus > V_RMS_MINUS_MAX {
            println!("❌ EMERGENCY STOP: v_rms_minus = {:.0} km/s > {} km/s", v_rms_minus, V_RMS_MINUS_MAX);
            break;
        }
        if v_rms_plus > V_RMS_PLUS_MAX {
            println!("❌ EMERGENCY STOP: v_rms_plus = {:.0} km/s > {} km/s", v_rms_plus, V_RMS_PLUS_MAX);
            break;
        }
        if rho_plus_max > RHO_MAX_STOP {
            println!("❌ EMERGENCY STOP: rho_plus_max = {:.0} > {:.0}", rho_plus_max, RHO_MAX_STOP);
            break;
        }

        if step >= N_STEPS {
            break;
        }

        // Hubble parameter H(a) ~ 0.07/a^1.5 in Gyr^-1
        let h = 0.07 / a.powf(1.5);

        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0) {
            eprintln!("GPU step failed: {}", e);
            break;
        }

        a += a * h * DT;
    }

    let total_time = sim_start.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE: {:.1}s ({:.2}h) for {} steps", total_time, total_time / 3600.0, N_STEPS);
    println!("║  Average: {:.3}s/step", total_time / N_STEPS as f64);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn save_snapshot(path: &str, pos: &[f64], signs: &[i32], z: f64, box_size: f64) {
    use std::io::Write;

    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create snapshot {}: {}", path, e);
            return;
        }
    };
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

    let n = pos.len() / 3;

    // Header: magic + version + n_particles + z + box_size
    writer.write_all(b"JSNP").unwrap();  // 4 bytes magic
    writer.write_all(&1u32.to_le_bytes()).unwrap();  // 4 bytes version
    writer.write_all(&(n as u64).to_le_bytes()).unwrap();  // 8 bytes n
    writer.write_all(&z.to_le_bytes()).unwrap();  // 8 bytes z
    writer.write_all(&box_size.to_le_bytes()).unwrap();  // 8 bytes box

    // Particles: x, y, z (f64) + sign (i8)
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        let s = if signs[i] > 0 { 1i8 } else { -1i8 };

        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&s.to_le_bytes()).unwrap();
    }

    writer.flush().unwrap();
}
