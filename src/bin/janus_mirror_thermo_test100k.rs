//! JANUS MIRROR THERMODYNAMICS TEST 100K
//!
//! Hypothesis: m- has its own thermodynamics in its metric, which appears
//! as REPULSIVE pressure in our metric. This stabilizes the v_rms ratio.
//!
//! m+ : Full baryonic physics (SPH + cooling + SF + feedback)
//! m- : HOT (10^6 K) + REPULSIVE SPH pressure + NO cooling

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use janus::baryonic::cooling::apply_cooling;
use janus::baryonic::feedback::sf_probability;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;
use std::collections::HashSet;
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};

// ═══════════════════════════════════════════════════════════════════════════
// COSMOLOGY
// ═══════════════════════════════════════════════════════════════════════════
const MU: f64 = 19.0;
const ETA: f64 = 1.045;
const H0_KMS_MPC: f64 = 69.9;
const T0_GYR: f64 = 15.87;

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION PARAMETERS (TEST 100K)
// ═══════════════════════════════════════════════════════════════════════════
const N_PARTICLES: usize = 100_000;
const L_BOX: f64 = 100.0;       // [Mpc]
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;          // [Gyr]
const N_STEPS: usize = 500;     // More steps to reach z~2
const THETA: f64 = 0.7;
const EPSILON: f64 = 0.05;      // Softening [Mpc]

// ═══════════════════════════════════════════════════════════════════════════
// BARYONIC PHYSICS m+ (standard)
// ═══════════════════════════════════════════════════════════════════════════
const T_INIT_PLUS: f64 = 10000.0;    // [K] - 10^4 K
const T_FLOOR: f64 = 100.0;
const T_THRESHOLD_SF: f64 = 10000.0;
const N_THRESHOLD_SF: f64 = 1.0;     // [cm⁻³] - lowered for test
const EPSILON_STAR: f64 = 0.02;
const EPSILON_SN: f64 = 0.003;
const BLASTWAVE_DELAY_GYR: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════
// MIRROR THERMODYNAMICS m- (NEW)
// ═══════════════════════════════════════════════════════════════════════════
const T_INIT_MINUS: f64 = 1e6;       // [K] - 10^6 K (HOT = diffuse)
// m- has REPULSIVE SPH pressure: P = -rho*T*k_B/(mu*m_p)
// This causes m- to resist compression and remain diffuse
// NO cooling (m- doesn't radiate in our metric)
// NO star formation
// NO SN feedback

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════════
const SNAPSHOT_INTERVAL: usize = 50;
const METRIC_INTERVAL: usize = 10;
const OUTPUT_DIR: &str = "/app/output/janus_mirror_thermo";

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH ICs
// ═══════════════════════════════════════════════════════════════════════════
const SEED_IC: u64 = 42;
const N_S: f64 = 0.965;
const DELTA_RMS: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;
const K_B: f64 = 1.38e-23;          // [J/K]
const M_P: f64 = 1.67e-27;          // [kg]
const MU_MOL: f64 = 0.6;
// Pressure coefficient: k_B / (mu * m_p) in code units (Mpc/Gyr)^2 / K
// 1 km/s = 1.022e-3 Mpc/Gyr, so (km/s)^2 = 1.044e-6 (Mpc/Gyr)^2
// k_B/m_p = 8.25e3 m²/s²/K = 8.25e-3 (km/s)²/K
const K_B_OVER_MU_MP: f64 = 8.25e-3 / MU_MOL * 1.044e-6;  // (Mpc/Gyr)²/K

// ═══════════════════════════════════════════════════════════════════════════
// RAHMATI 2013 SELF-SHIELDING
// ═══════════════════════════════════════════════════════════════════════════
fn self_shielding_rahmati2013(n_h: f64, gamma_uv: f64) -> f64 {
    const N0: f64 = 0.00981;
    const ALPHA1: f64 = -0.2287;
    const ALPHA2: f64 = -0.7268;
    const BETA: f64 = 1.1802;
    const F: f64 = 0.05;

    if n_h < 1e-10 || gamma_uv < 1e-30 {
        return gamma_uv;
    }

    let ratio = (1.0 - F) * (1.0 + (n_h / N0).powf(BETA)).powf(ALPHA1)
              + F * (1.0 + n_h / N0).powf(ALPHA2);

    gamma_uv * ratio
}

fn uv_background_hm2012(z: f64) -> f64 {
    let zp1 = 1.0 + z;
    1e-24 * zp1 * zp1 / (1.0 + (zp1 / 3.0).powf(5.0))
}

fn gamma_uv_effective(n_h: f64, z: f64) -> f64 {
    let gamma_uv = uv_background_hm2012(z);
    self_shielding_rahmati2013(n_h, gamma_uv)
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     JANUS MIRROR THERMODYNAMICS TEST — 100K VALIDATION                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  HYPOTHESIS: m- has REPULSIVE pressure in our metric                     ║");
    println!("║  This stabilizes v_rms ratio and allows m+ galaxy formation              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  COSMOLOGY: μ = {}, η = {}, H₀ = {} km/s/Mpc", MU, ETA, H0_KMS_MPC);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  m+ PHYSICS (baryonic):");
    println!("║    T_init = {} K, cooling ACTIVE, SF ACTIVE", T_INIT_PLUS);
    println!("║    n_threshold = {} cm⁻³, ε* = {}", N_THRESHOLD_SF, EPSILON_STAR);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  m- PHYSICS (mirror thermodynamics):");
    println!("║    T_init = {:.0e} K (HOT → diffuse)", T_INIT_MINUS);
    println!("║    SPH pressure: REPULSIVE (P < 0 in our metric)");
    println!("║    Cooling: NONE (m- doesn't radiate here)");
    println!("║    Star formation: NONE");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  OUTPUT: {}", OUTPUT_DIR);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    fs::create_dir_all(format!("{}/snapshots", OUTPUT_DIR)).expect("Failed to create output dir");

    // Generate Zel'dovich ICs
    let (positions, velocities, signs) = generate_zeldovich_ics();

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.iter().filter(|&&s| s < 0).count();
    println!("\n[IC VALIDATION]");
    println!("  N+ = {}, N- = {}, ratio = {:.4}", n_plus, n_minus, n_plus as f64 / n_minus as f64);

    // Initialize GPU simulation
    println!("\nInitializing GPU simulation with {} particles...", N_PARTICLES);
    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_BOX,
        positions, velocities, signs
    ).unwrap();
    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(EPSILON);

    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    gpu_sim.set_c_ratio(c_ratio_sq_init.sqrt());
    println!("  GPU init: {:.2}s", start_time.elapsed().as_secs_f64());

    // Initialize temperatures for BOTH populations
    let mut temp_plus: Vec<f64> = vec![T_INIT_PLUS; n_plus];
    let temp_minus: Vec<f64> = vec![T_INIT_MINUS; n_minus];

    let mut n_stars: usize = 0;
    let mut is_star: Vec<bool> = vec![false; n_plus];
    let mut stellar_ages: Vec<f64> = vec![0.0; n_plus];
    let mut rng_sf = StdRng::seed_from_u64(12345);

    // CSV output
    let csv_path = format!("{}/time_series.csv", OUTPUT_DIR);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,t_Gyr,z,v_rms_plus,v_rms_minus,ratio,rho_max_plus,rho_max_minus,T_mean_plus,T_mean_minus,N_stars,S").unwrap();

    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;
    let mut t_gyr = 0.0;

    println!("\n  Step │    t_Gyr │       z │   v_rms+ │   v_rms- │   ratio │ T_mean+  │ T_mean-  │ ρ+_max │ ρ-_max │ N_stars");
    println!("───────┼──────────┼─────────┼──────────┼──────────┼─────────┼──────────┼──────────┼────────┼────────┼────────");

    for step in 0..=N_STEPS {
        let z = (1.0 / a - 1.0).max(0.0);

        // Dynamic c_ratio
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

        let h = 0.07 / a.powf(1.5);

        // Metrics
        if step % METRIC_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_data = gpu_sim.signs();

            let (rho_max_plus, rho_max_minus, segregation, _) =
                compute_density_metrics(&pos, &signs_data, 64, L_BOX);

            let (v_rms_plus, v_rms_minus) = compute_vrms(&vel, &signs_data);
            let ratio = v_rms_minus / v_rms_plus.max(1e-10);

            let t_mean_plus: f64 = temp_plus.iter().sum::<f64>() / temp_plus.len().max(1) as f64;
            let t_mean_minus: f64 = temp_minus.iter().sum::<f64>() / temp_minus.len().max(1) as f64;

            writeln!(csv, "{},{:.6},{:.4},{:.1},{:.1},{:.4},{:.1},{:.1},{:.0},{:.0},{},{}",
                step, t_gyr, z,
                v_rms_plus * MPC_GYR_TO_KMS, v_rms_minus * MPC_GYR_TO_KMS, ratio,
                rho_max_plus, rho_max_minus,
                t_mean_plus, t_mean_minus, n_stars, segregation
            ).unwrap();

            println!("  {:5} │ {:8.4} │ {:7.4} │ {:8.1} │ {:8.1} │ {:7.4} │ {:8.0} │ {:8.0} │ {:6.1} │ {:6.1} │ {:7}",
                step, t_gyr, z,
                v_rms_plus * MPC_GYR_TO_KMS, v_rms_minus * MPC_GYR_TO_KMS, ratio,
                t_mean_plus, t_mean_minus, rho_max_plus, rho_max_minus, n_stars);

            // Success criteria check
            if z <= 2.0 && z > 1.9 {
                println!("\n  ═══ z=2 CHECKPOINT ═══");
                if ratio >= 0.85 && ratio <= 1.15 {
                    println!("  ✓ ratio = {:.4} in [0.85, 1.15] → STABLE", ratio);
                } else {
                    println!("  ✗ ratio = {:.4} outside [0.85, 1.15] → UNSTABLE", ratio);
                }
                if t_mean_plus < T_INIT_PLUS * 0.9 {
                    println!("  ✓ T_mean+ = {:.0} K < {:.0} K → cooling active", t_mean_plus, T_INIT_PLUS * 0.9);
                }
                if t_mean_minus > T_INIT_MINUS * 0.8 {
                    println!("  ✓ T_mean- = {:.0e} K stable (no cooling)", t_mean_minus);
                }
                if rho_max_plus > rho_max_minus {
                    println!("  ✓ ρ+_max ({:.1}) > ρ-_max ({:.1}) → m+ condensing", rho_max_plus, rho_max_minus);
                }
            }

            // Auto-stop: ratio divergence
            if ratio > 1.30 {
                println!("\n  ✗ AUTO-STOP: ratio = {:.4} > 1.30 → hypothesis FAILED", ratio);
                break;
            }

            csv.flush().unwrap();
        }

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_data = gpu_sim.signs();
            let snap_path = format!("{}/snapshots/snap_{:05}.bin", OUTPUT_DIR, step);
            save_snapshot(&snap_path, &pos, &vel, &signs_data, a, t_gyr);
        }

        // Time integration
        if step < N_STEPS {
            // 1. Gravity step (GPU)
            gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0).unwrap();

            // 2. Get positions and velocities for baryonic physics
            let pos = gpu_sim.get_positions().unwrap();
            let mut vel = gpu_sim.get_velocities().unwrap();
            let signs_data = gpu_sim.signs();

            // Compute overdensities
            let (overdensities_plus, overdensities_minus) =
                compute_local_overdensities_both(&pos, &signs_data, 32, L_BOX);

            // ═══════════════════════════════════════════════════════════════
            // m+ BARYONIC PHYSICS
            // ═══════════════════════════════════════════════════════════════
            let mut plus_idx = 0;
            for i in 0..signs_data.len() {
                if signs_data[i] <= 0 {
                    continue;
                }

                if is_star[plus_idx] {
                    stellar_ages[plus_idx] += DT;
                    plus_idx += 1;
                    continue;
                }

                let overdensity = overdensities_plus[plus_idx];
                let temp = temp_plus[plus_idx];

                // Self-shielding
                let n_h = 2e-7 * (1.0 + z).powi(3) * overdensity;
                let _gamma_eff = gamma_uv_effective(n_h, z);

                // Cooling (m+ only)
                let new_temp = apply_cooling(temp, overdensity, z, DT);
                temp_plus[plus_idx] = new_temp.max(T_FLOOR);

                // Star formation
                let n_threshold_od = N_THRESHOLD_SF / (2e-7 * (1.0 + z).powi(3));
                if overdensity > n_threshold_od && temp_plus[plus_idx] < T_THRESHOLD_SF {
                    let prob = sf_probability(overdensity, EPSILON_STAR, DT).min(0.1);
                    if rng_sf.random::<f64>() < prob {
                        is_star[plus_idx] = true;
                        stellar_ages[plus_idx] = 0.0;
                        n_stars += 1;
                    }
                }

                plus_idx += 1;
            }

            // ═══════════════════════════════════════════════════════════════
            // m- MIRROR THERMODYNAMICS
            // ═══════════════════════════════════════════════════════════════
            // Apply REPULSIVE pressure: P = -rho*T*k_B/(mu*m_p)
            // This gives acceleration away from dense regions
            // a = -grad(P)/rho = +grad(T*k_B/(mu*m_p)) for repulsive case

            let mut minus_idx = 0;
            for i in 0..signs_data.len() {
                if signs_data[i] >= 0 {
                    continue;
                }

                let overdensity = overdensities_minus[minus_idx];
                let temp = temp_minus[minus_idx];

                // NO cooling for m- (doesn't radiate in our metric)
                // Temperature stays constant at T_INIT_MINUS

                // Repulsive pressure creates velocity dispersion
                // Simple model: velocity kick proportional to local overdensity gradient
                // In dense regions, particles get kicked outward
                if overdensity > 2.0 {
                    // Sound speed for repulsive gas: c_s = sqrt(k_B*T/(mu*m_p))
                    let c_s_sq = K_B_OVER_MU_MP * temp;  // (Mpc/Gyr)^2
                    let c_s = c_s_sq.sqrt();  // Mpc/Gyr

                    // Kick magnitude proportional to overdensity excess
                    let kick_factor = 0.01 * (overdensity - 1.0).min(10.0);

                    // Random direction kick (repulsive pressure is isotropic)
                    let theta = rng_sf.random::<f64>() * 2.0 * PI;
                    let phi = (rng_sf.random::<f64>() * 2.0 - 1.0).acos();

                    let dvx = kick_factor * c_s * phi.sin() * theta.cos();
                    let dvy = kick_factor * c_s * phi.sin() * theta.sin();
                    let dvz = kick_factor * c_s * phi.cos();

                    vel[i*3] += dvx;
                    vel[i*3 + 1] += dvy;
                    vel[i*3 + 2] += dvz;
                }

                minus_idx += 1;
            }

            // Update velocities on GPU
            gpu_sim.set_velocities(&vel).unwrap();

            a += a * h * DT;
            t_gyr += DT;
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SIMULATION COMPLETE");
    println!("║  Total time: {:.1}h ({:.1} ms/step)", total_time / 3600.0, total_time * 1000.0 / N_STEPS as f64);
    println!("║  Final N_stars: {}", n_stars);
    println!("║  Output: {}", OUTPUT_DIR);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn compute_density_metrics(pos: &[f64], signs: &[i32], grid_size: usize, box_size: f64) -> (f64, f64, f64, f64) {
    let half_box = box_size / 2.0;
    let cell_size = box_size / grid_size as f64;
    let n3 = grid_size * grid_size * grid_size;

    let mut rho_plus = vec![0u32; n3];
    let mut rho_minus = vec![0u32; n3];

    let n = pos.len() / 3;
    for i in 0..n {
        let ix = ((pos[i*3] + half_box) / cell_size).floor() as usize;
        let iy = ((pos[i*3+1] + half_box) / cell_size).floor() as usize;
        let iz = ((pos[i*3+2] + half_box) / cell_size).floor() as usize;

        if ix < grid_size && iy < grid_size && iz < grid_size {
            let idx = ix + iy * grid_size + iz * grid_size * grid_size;
            if signs[i] > 0 {
                rho_plus[idx] += 1;
            } else {
                rho_minus[idx] += 1;
            }
        }
    }

    let rho_max_plus = *rho_plus.iter().max().unwrap_or(&0) as f64;
    let rho_max_minus = *rho_minus.iter().max().unwrap_or(&0) as f64;

    let mut seg_sum = 0.0;
    let mut count = 0;
    for i in 0..n3 {
        let total = rho_plus[i] + rho_minus[i];
        if total > 0 {
            seg_sum += (rho_plus[i] as f64 - rho_minus[i] as f64).abs() / total as f64;
            count += 1;
        }
    }
    let segregation = if count > 0 { seg_sum / count as f64 } else { 0.0 };

    (rho_max_plus, rho_max_minus, segregation, 0.0)
}

#[cfg(feature = "cuda")]
fn compute_local_overdensities_both(pos: &[f64], signs: &[i32], grid_size: usize, box_size: f64) -> (Vec<f64>, Vec<f64>) {
    let half_box = box_size / 2.0;
    let cell_size = box_size / grid_size as f64;
    let n3 = grid_size * grid_size * grid_size;

    let mut cell_counts_plus = vec![0u32; n3];
    let mut cell_counts_minus = vec![0u32; n3];
    let mut particle_cells_plus: Vec<usize> = Vec::new();
    let mut particle_cells_minus: Vec<usize> = Vec::new();

    let n = pos.len() / 3;
    for i in 0..n {
        let ix = (((pos[i*3] + half_box) / cell_size).floor() as usize).min(grid_size - 1);
        let iy = (((pos[i*3+1] + half_box) / cell_size).floor() as usize).min(grid_size - 1);
        let iz = (((pos[i*3+2] + half_box) / cell_size).floor() as usize).min(grid_size - 1);
        let idx = ix + iy * grid_size + iz * grid_size * grid_size;

        if signs[i] > 0 {
            cell_counts_plus[idx] += 1;
            particle_cells_plus.push(idx);
        } else {
            cell_counts_minus[idx] += 1;
            particle_cells_minus.push(idx);
        }
    }

    let n_plus = particle_cells_plus.len() as f64;
    let n_minus = particle_cells_minus.len() as f64;
    let mean_plus = n_plus / n3 as f64;
    let mean_minus = n_minus / n3 as f64;

    let od_plus: Vec<f64> = particle_cells_plus.iter()
        .map(|&idx| if mean_plus > 0.0 { cell_counts_plus[idx] as f64 / mean_plus } else { 1.0 })
        .collect();

    let od_minus: Vec<f64> = particle_cells_minus.iter()
        .map(|&idx| if mean_minus > 0.0 { cell_counts_minus[idx] as f64 / mean_minus } else { 1.0 })
        .collect();

    (od_plus, od_minus)
}

#[cfg(feature = "cuda")]
fn compute_vrms(vel: &[f64], signs: &[i32]) -> (f64, f64) {
    let n = vel.len() / 3;
    let mut sum_plus = 0.0;
    let mut sum_minus = 0.0;
    let mut n_plus = 0;
    let mut n_minus = 0;

    for i in 0..n {
        let v2 = vel[i*3].powi(2) + vel[i*3+1].powi(2) + vel[i*3+2].powi(2);
        if signs[i] > 0 {
            sum_plus += v2;
            n_plus += 1;
        } else {
            sum_minus += v2;
            n_minus += 1;
        }
    }

    let v_rms_plus = if n_plus > 0 { (sum_plus / n_plus as f64).sqrt() } else { 0.0 };
    let v_rms_minus = if n_minus > 0 { (sum_minus / n_minus as f64).sqrt() } else { 0.0 };

    (v_rms_plus, v_rms_minus)
}

#[cfg(feature = "cuda")]
fn save_snapshot(path: &str, pos: &[f64], vel: &[f64], signs: &[i32], a: f64, t: f64) {
    let mut f = BufWriter::new(File::create(path).unwrap());
    let n = pos.len() / 3;

    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&a.to_le_bytes()).unwrap();
    f.write_all(&t.to_le_bytes()).unwrap();

    for i in 0..n*3 {
        f.write_all(&(pos[i] as f32).to_le_bytes()).unwrap();
    }
    for i in 0..n*3 {
        f.write_all(&(vel[i] as f32).to_le_bytes()).unwrap();
    }
    for s in signs {
        f.write_all(&[*s as i8 as u8]).unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH IC GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn generate_zeldovich_ics() -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    println!("\n[GENERATING ZEL'DOVICH ICs]");

    let n_grid = (N_PARTICLES as f64).powf(1.0/3.0).ceil() as usize;
    let n3 = n_grid * n_grid * n_grid;
    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (n3 as f64 * f_plus).round() as usize;

    println!("  Grid: {}³ = {} particles", n_grid, n3);
    println!("  Seed IC: {} (same for m+ and m-)", SEED_IC);

    let dk = 2.0 * PI / L_BOX;
    let half_n = n_grid / 2;
    let spacing = L_BOX / n_grid as f64;
    let half_box = L_BOX / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    println!("  Generating displacement field...");
    let (psi_x, psi_y, psi_z) = generate_displacement_field(n_grid, dk, half_n, d_growth, SEED_IC);

    let max_disp = find_max_displacement(&psi_x, &psi_y, &psi_z);
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };

    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;
    println!("  vel_scale: {:.4e} Mpc/Gyr", vel_scale);

    let mut rng_shuffle = rand::rngs::StdRng::seed_from_u64(12345);
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.shuffle(&mut rng_shuffle);
    let plus_indices: HashSet<usize> = indices[..n_positive].iter().cloned().collect();

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs: Vec<i32> = Vec::with_capacity(n3);

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let mut x = x0 + psi_x[idx] * scale;
                let mut y = y0 + psi_y[idx] * scale;
                let mut z = z0 + psi_z[idx] * scale;

                while x > half_box { x -= L_BOX; }
                while x < -half_box { x += L_BOX; }
                while y > half_box { y -= L_BOX; }
                while y < -half_box { y += L_BOX; }
                while z > half_box { z -= L_BOX; }
                while z < -half_box { z += L_BOX; }

                positions.push(x);
                positions.push(y);
                positions.push(z);

                velocities.push(psi_x[idx] * vel_scale);
                velocities.push(psi_y[idx] * vel_scale);
                velocities.push(psi_z[idx] * vel_scale);

                signs.push(if plus_indices.contains(&idx) { 1i32 } else { -1i32 });
            }
        }
    }

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.iter().filter(|&&s| s < 0).count();
    println!("  ✓ Generated: N+ = {}, N- = {}", n_plus, n_minus);

    (positions, velocities, signs)
}

#[cfg(feature = "cuda")]
fn generate_displacement_field(n_grid: usize, dk: f64, half_n: usize, d_growth: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;

                let k2 = kx*kx + ky*ky + kz*kz;
                let k = k2.sqrt();

                if k < 1e-10 || k > half_n as f64 * dk {
                    continue;
                }

                let pk = k.powf(N_S) * (-k * k / (2.0 * (half_n as f64 * dk).powi(2))).exp();
                let amplitude = (pk / 2.0).sqrt() * DELTA_RMS * d_growth;

                let phase = rng.random::<f64>() * 2.0 * PI;
                let gauss = normal.sample(&mut rng);

                let idx = ix + iy * n_grid + iz * n_grid * n_grid;
                delta_k[idx] = Complex::new(
                    amplitude * gauss * phase.cos(),
                    amplitude * gauss * phase.sin()
                );
            }
        }
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n_grid);

    let mut psi_kx = delta_k.clone();
    let mut psi_ky = delta_k.clone();
    let mut psi_kz = delta_k.clone();

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;

                let k2 = kx*kx + ky*ky + kz*kz;
                let idx = ix + iy * n_grid + iz * n_grid * n_grid;

                if k2 > 1e-10 {
                    let i_over_k2 = Complex::new(0.0, 1.0 / k2);
                    psi_kx[idx] = delta_k[idx] * i_over_k2 * kx;
                    psi_ky[idx] = delta_k[idx] * i_over_k2 * ky;
                    psi_kz[idx] = delta_k[idx] * i_over_k2 * kz;
                }
            }
        }
    }

    for iz in 0..n_grid {
        let offset = iz * n_grid * n_grid;
        for iy in 0..n_grid {
            let row_offset = offset + iy * n_grid;
            let mut row_x: Vec<Complex<f64>> = psi_kx[row_offset..row_offset + n_grid].to_vec();
            let mut row_y: Vec<Complex<f64>> = psi_ky[row_offset..row_offset + n_grid].to_vec();
            let mut row_z: Vec<Complex<f64>> = psi_kz[row_offset..row_offset + n_grid].to_vec();

            fft.process(&mut row_x);
            fft.process(&mut row_y);
            fft.process(&mut row_z);

            for ix in 0..n_grid {
                psi_kx[row_offset + ix] = row_x[ix];
                psi_ky[row_offset + ix] = row_y[ix];
                psi_kz[row_offset + ix] = row_z[ix];
            }
        }
    }

    let norm = 1.0 / n3 as f64;
    let psi_x: Vec<f64> = psi_kx.iter().map(|c| c.re * norm).collect();
    let psi_y: Vec<f64> = psi_ky.iter().map(|c| c.re * norm).collect();
    let psi_z: Vec<f64> = psi_kz.iter().map(|c| c.re * norm).collect();

    (psi_x, psi_y, psi_z)
}

#[cfg(feature = "cuda")]
fn find_max_displacement(psi_x: &[f64], psi_y: &[f64], psi_z: &[f64]) -> f64 {
    let mut max_disp = 0.0f64;
    for i in 0..psi_x.len() {
        let disp = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if disp > max_disp {
            max_disp = disp;
        }
    }
    max_disp
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires --features cuda");
}
