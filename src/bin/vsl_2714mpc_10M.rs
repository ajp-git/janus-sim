//! VSL Production — 2714 Mpc Box, 10M Particles, Validated Density
//!
//! Box = 2714 Mpc to match validated 500k/1000Mpc density ratio
//! SPH for m+ only, star formation enabled

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::baryonic::star_formation::{particle_type, sign_to_type, is_sink};

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

const MPC_GYR_TO_KMS: f64 = 977.8;
const N_CELLS: usize = 64;

// Production parameters
const N_PARTICLES: usize = 10_000_000;
const BOX_SIZE: f64 = 2714.0;  // 2714 Mpc for validated density
const MU: f64 = 19.0;
const N_STEPS: usize = 30000;
const DT: f64 = 0.001;
const SNAP_INTERVAL: usize = 10;
const CSV_INTERVAL: usize = 50;

// Janus
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;

// Thermal velocity m+
const K_B_OVER_MP_CODE: f64 = 8.7e-9;
const T_INIT_PLUS: f64 = 1.0e4;   // 10000 K initial
const T_FLOOR_PLUS: f64 = 100.0;  // 100 K floor
const MU_MOL: f64 = 0.6;

// Star formation
const SF_CHECK_INTERVAL: usize = 10;
const RHO_SF_FACTOR: f64 = 100.0;
const T_SF: f64 = 1000.0;

// Physics constants
const G_CODE: f64 = 4.499e-15;
const OMEGA_B: f64 = 0.05;
const DELTA_INIT: f64 = 0.10;  // 10% initial perturbation

// Emergency stops
const V_RMS_MINUS_MAX: f64 = 50000.0;
const V_RMS_MINUS_STEP50: f64 = 5000.0;  // Strict validation at step 50
const RHO_MAX_STOP: f64 = 1e9;

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

    let output_dir = "/app/output/janus_vsl_2714mpc_10M";
    let snap_dir = format!("{}/snapshots", output_dir);

    fs::create_dir_all(&snap_dir).expect("Failed to create output directories");

    let c_ratio = 1.0 / MU.sqrt();
    let c_ratio_sq = 1.0 / MU;

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    let sigma_v_plus = (K_B_OVER_MP_CODE * T_INIT_PLUS / MU_MOL).sqrt();
    let sigma_v_plus_kms = sigma_v_plus * MPC_GYR_TO_KMS;

    // m- starts cold (will be accelerated by VSL)
    let sigma_v_minus = sigma_v_plus * 0.1;  // 10× colder

    let box_vol = BOX_SIZE * BOX_SIZE * BOX_SIZE;
    let density = N_PARTICLES as f64 / box_vol;
    let density_500k_1000 = 500_000.0 / (1000.0 * 1000.0 * 1000.0);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL PRODUCTION — 2714 Mpc, 10M Particles                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc → density = {:.2e} /Mpc³", BOX_SIZE, density);
    println!("║  Density ratio vs 500k/1000Mpc = {:.2}", density / density_500k_1000);
    println!("║  μ = {} → c⁻/c⁺ = {:.4} → (c⁻/c⁺)² = {:.5}", MU, c_ratio, c_ratio_sq);
    println!("║  T_init+ = {} K, T_floor+ = {} K", T_INIT_PLUS, T_FLOOR_PLUS);
    println!("║  σ_v+ = {:.1} km/s, σ_v- = {:.1} km/s", sigma_v_plus_kms, sigma_v_minus * MPC_GYR_TO_KMS);
    println!("║  z_init = {}, dt = {} Gyr, steps = {}", Z_INIT, DT, N_STEPS);
    println!("║  G_code = {:.3e}, Ω_b = {}, δ_init = {}%", G_CODE, OMEGA_B, DELTA_INIT * 100.0);
    println!("║  Snapshots: every {} steps → {} files", SNAP_INTERVAL, N_STEPS / SNAP_INTERVAL);
    println!("║  Output: {}", output_dir);
    println!("║  STOP: v_rms- > {} km/s, step50: > {} km/s", V_RMS_MINUS_MAX, V_RMS_MINUS_STEP50);
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

    // Initialize particle types
    let signs = gpu_sim.signs();
    let mut particle_types: Vec<u8> = signs.iter()
        .map(|&s| sign_to_type(if s > 0 { 1 } else { -1 }))
        .collect();

    // Temperature array (m+ only, m- has no temperature)
    let mut temperatures: Vec<f64> = vec![T_INIT_PLUS; N_PARTICLES];
    for i in 0..N_PARTICLES {
        if signs[i] <= 0 {
            temperatures[i] = 0.0;  // m- has no thermal physics
        }
    }

    // Set thermal velocities: m+ hot, m- cold
    println!("Setting thermal velocities...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal_plus = Normal::new(0.0, sigma_v_plus).unwrap();
    let normal_minus = Normal::new(0.0, sigma_v_minus).unwrap();

    let mut thermal_vel = vec![0.0f64; N_PARTICLES * 3];
    for i in 0..N_PARTICLES {
        let normal = if signs[i] > 0 { &normal_plus } else { &normal_minus };
        thermal_vel[i * 3]     = normal.sample(&mut rng);
        thermal_vel[i * 3 + 1] = normal.sample(&mut rng);
        thermal_vel[i * 3 + 2] = normal.sample(&mut rng);
    }

    if let Err(e) = gpu_sim.set_velocities(&thermal_vel) {
        eprintln!("Failed to set velocities: {}", e);
        return;
    }
    drop(thermal_vel);
    println!("Velocities set: m+ σ={:.1} km/s, m- σ={:.1} km/s\n",
             sigma_v_plus_kms, sigma_v_minus * MPC_GYR_TO_KMS);

    // CSV file
    let csv_path = format!("{}/evolution.csv", output_dir);
    let csv_file = File::create(&csv_path).expect("Failed to create CSV");
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,n_stars,segregation").unwrap();

    let half_box = BOX_SIZE / 2.0;
    let cell_size = BOX_SIZE / N_CELLS as f64;
    let mean_density = n_positive as f64 / box_vol;

    println!("{:>6} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
             "Step", "z", "ρ+_max", "ρ-_max", "v_rms+", "v_rms-", "N_stars", "Seg");
    println!("{:-<90}", "");

    let sim_start = Instant::now();
    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;
    let mut n_stars_total: usize = 0;

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;
        let t_gyr = step as f64 * DT;

        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to get positions: {}", e); break; }
        };
        let vel = match gpu_sim.get_velocities() {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to get velocities: {}", e); break; }
        };
        let signs = gpu_sim.signs();

        // Cell density counts
        let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut counts_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut particle_cells = vec![0usize; N_PARTICLES];

        for i in 0..N_PARTICLES {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];

            let ix = ((px + half_box) / cell_size).floor() as usize % N_CELLS;
            let iy = ((py + half_box) / cell_size).floor() as usize % N_CELLS;
            let iz = ((pz + half_box) / cell_size).floor() as usize % N_CELLS;
            let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;
            particle_cells[i] = idx;

            if signs[i] > 0 {
                counts_plus[idx] += 1;
            } else {
                counts_minus[idx] += 1;
            }
        }

        // Star formation check
        if step > 0 && step % SF_CHECK_INTERVAL == 0 {
            let cell_vol = cell_size * cell_size * cell_size;

            for i in 0..N_PARTICLES {
                if particle_types[i] != particle_type::GAS_PLUS {
                    continue;
                }

                let local_density = counts_plus[particle_cells[i]] as f64 / cell_vol;
                let temp = temperatures[i];

                if local_density > RHO_SF_FACTOR * mean_density && temp < T_SF {
                    let vx = vel[i * 3];
                    let vy = vel[i * 3 + 1];
                    let vz = vel[i * 3 + 2];
                    let v_kms = (vx*vx + vy*vy + vz*vz).sqrt() * MPC_GYR_TO_KMS;

                    if v_kms < 100.0 {
                        particle_types[i] = particle_type::SINK_STAR;
                        temperatures[i] = 0.0;
                        n_stars_total += 1;
                    }
                }
            }
        }

        // Compute metrics
        let mut v2_plus = 0.0f64;
        let mut v2_minus = 0.0f64;
        let mut n_plus = 0usize;
        let mut n_minus = 0usize;
        let mut com_plus = [0.0f64; 3];
        let mut com_minus = [0.0f64; 3];
        let mut n_stars_current: usize = 0;

        for i in 0..N_PARTICLES {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];
            let vx = vel[i * 3];
            let vy = vel[i * 3 + 1];
            let vz = vel[i * 3 + 2];
            let v2 = vx*vx + vy*vy + vz*vz;

            if is_sink(particle_types[i]) {
                n_stars_current += 1;
            }

            if signs[i] > 0 {
                v2_plus += v2;
                com_plus[0] += px;
                com_plus[1] += py;
                com_plus[2] += pz;
                n_plus += 1;
            } else {
                v2_minus += v2;
                com_minus[0] += px;
                com_minus[1] += py;
                com_minus[2] += pz;
                n_minus += 1;
            }
        }

        let rho_plus_max = *counts_plus.iter().max().unwrap_or(&0);
        let rho_minus_max = *counts_minus.iter().max().unwrap_or(&0);
        let v_rms_plus = if n_plus > 0 { (v2_plus / n_plus as f64).sqrt() * MPC_GYR_TO_KMS } else { 0.0 };
        let v_rms_minus = if n_minus > 0 { (v2_minus / n_minus as f64).sqrt() * MPC_GYR_TO_KMS } else { 0.0 };

        // Segregation
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
        let segregation = (dx*dx + dy*dy + dz*dz).sqrt() / BOX_SIZE;

        // CSV
        if step % CSV_INTERVAL == 0 {
            writeln!(csv_writer, "{},{:.6},{:.4},{},{},{:.2},{:.2},{},{:.6}",
                     step, t_gyr, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, n_stars_current, segregation).unwrap();
            csv_writer.flush().unwrap();
        }

        // Snapshot
        if step % SNAP_INTERVAL == 0 {
            let snap_path = format!("{}/snap_{:06}.bin", snap_dir, step);
            save_snapshot_v2(&snap_path, &pos, &signs, &particle_types, z, BOX_SIZE);
        }

        // Output
        if step % 100 == 0 || step <= 10 || step == 50 {
            let elapsed = sim_start.elapsed().as_secs_f64();
            let eta_h = if step > 0 { elapsed / step as f64 * (N_STEPS - step) as f64 / 3600.0 } else { 0.0 };

            println!("{:>6} | {:>6.3} | {:>8} | {:>8} | {:>7.0} | {:>7.0} | {:>8} | {:>7.4} | {:.1}s (ETA {:.1}h)",
                     step, z, rho_plus_max, rho_minus_max, v_rms_plus, v_rms_minus,
                     n_stars_current, segregation, elapsed, eta_h);

            // Alerts
            if rho_plus_max > 100 && rho_plus_max <= 1000 {
                println!("  ★ Structures embryonnaires détectées (ρ+_max = {})", rho_plus_max);
            }
            if rho_plus_max > 1000 {
                println!("  ★★ HALOS FORMÉS (ρ+_max = {})", rho_plus_max);
            }
            if n_stars_current > 0 && n_stars_total == n_stars_current {
                println!("  ★★★ PREMIÈRE ÉTOILE JANUS FORMÉE!");
            }
            if z < 1.0 && step > 0 && (step - 1) % 100 != 0 {
                println!("  ★★★★ Époque récente atteinte (z < 1)");
            }
        }

        // Step 50 validation (strict)
        if step == 50 {
            if v_rms_minus < 1000.0 {
                println!("✓ Step 50 PASS: v_rms- = {:.0} km/s < 1000 km/s", v_rms_minus);
            } else if v_rms_minus < V_RMS_MINUS_STEP50 {
                println!("⚠ Step 50 WARNING: v_rms- = {:.0} km/s (élevé mais < {})", v_rms_minus, V_RMS_MINUS_STEP50);
            } else {
                println!("❌ Step 50 FAIL: v_rms- = {:.0} km/s > {} km/s → STOP", v_rms_minus, V_RMS_MINUS_STEP50);
                break;
            }
        }

        // Emergency stops
        if v_rms_minus > V_RMS_MINUS_MAX {
            println!("❌ EMERGENCY STOP: v_rms- = {:.0} km/s > {} km/s", v_rms_minus, V_RMS_MINUS_MAX);
            break;
        }
        if rho_plus_max as f64 > RHO_MAX_STOP {
            println!("❌ EMERGENCY STOP: ρ+_max = {} > {:.0}", rho_plus_max, RHO_MAX_STOP);
            break;
        }

        if step >= N_STEPS {
            break;
        }

        // Janus expansion: H(a) for matter-dominated + Janus correction
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
    println!("║  Total stars formed: {}", n_stars_total);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn save_snapshot_v2(path: &str, pos: &[f64], signs: &[i32], types: &[u8], z: f64, box_size: f64) {
    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create snapshot {}: {}", path, e);
            return;
        }
    };
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

    let n = pos.len() / 3;

    // Header v2
    writer.write_all(b"JSNP").unwrap();
    writer.write_all(&2u32.to_le_bytes()).unwrap();
    writer.write_all(&(n as u64).to_le_bytes()).unwrap();
    writer.write_all(&z.to_le_bytes()).unwrap();
    writer.write_all(&box_size.to_le_bytes()).unwrap();

    // Particles: 26 bytes each
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let zp = pos[i * 3 + 2];
        let s = if signs[i] > 0 { 1i8 } else { -1i8 };
        let t = types[i];

        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&zp.to_le_bytes()).unwrap();
        writer.write_all(&s.to_le_bytes()).unwrap();
        writer.write_all(&t.to_le_bytes()).unwrap();
    }

    writer.flush().unwrap();
}
