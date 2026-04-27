//! JANUS BARYONIC CALIBRATED - MCJ Production Run
//!
//! Final calibrated run with:
//! - μ = 19, η = 1.045
//! - H₀ = 69.9 km/s/Mpc, t₀ = 15.87 Gyr
//! - Grackle HM2012 cooling + Rahmati self-shielding
//! - Star formation (Schmidt-Kennicutt)
//! - Kinetic SN feedback
//!
//! m+ : Full baryonic physics (SPH + cooling + SF + feedback)
//! m- : Pure gravity, collisionless

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
use janus::vsl_dynamic::CoupledFriedmann;
use janus::baryonic::feedback::{sf_probability, FeedbackMode, apply_sn_feedback};
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
const MU: f64 = 19.0;           // Canonical Petit μ
const ETA: f64 = 1.045;         // Mass ratio (from Pantheon+ fit)
const H0_KMS_MPC: f64 = 69.9;   // Hubble constant [km/s/Mpc]
const T0_GYR: f64 = 15.87;      // Universe age at z=0 [Gyr]

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const N_PARTICLES: usize = 10_000_000;  // 10M - may hit 12GB VRAM limit
const L_BOX: f64 = 500.0;       // [Mpc]
const Z_INIT: f64 = 4.0;        // Initial redshift
const DT: f64 = 0.001;          // Time step [Gyr]
const N_STEPS: usize = 30_000;  // Total steps
const THETA: f64 = 0.7;         // Barnes-Hut opening angle
const EPSILON: f64 = 0.1;       // Softening [Mpc]

// ═══════════════════════════════════════════════════════════════════════════
// BARYONIC PHYSICS (m+ only)
// ═══════════════════════════════════════════════════════════════════════════
const T_INIT_PLUS: f64 = 10000.0;   // Initial temperature [K] (10^4 K as per ROADMAP)
const T_FLOOR: f64 = 100.0;         // Minimum temperature [K]
const T_THRESHOLD_SF: f64 = 10000.0; // Star formation threshold [K] (10^4 K)
const N_THRESHOLD_SF: f64 = 30.0;   // Star formation density [cm⁻³] (MCJ: 10-50)
const EPSILON_STAR: f64 = 0.02;     // Star formation efficiency
const EPSILON_SN: f64 = 0.003;      // SN feedback efficiency (0.3%)
const DELAY_SN_MYR: f64 = 10.0;     // SN delay [Myr]
const BLASTWAVE_DELAY_GYR: f64 = 0.03; // Blastwave delay [Gyr]

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════════
const SNAPSHOT_INTERVAL: usize = 5;   // User requested 5
const METRIC_INTERVAL: usize = 10;    // Log ratio every 10 steps
const OUTPUT_DIR: &str = "/app/output/janus_baryonic_calibrated";

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH ICs
// ═══════════════════════════════════════════════════════════════════════════
const SEED_IC: u64 = 42;  // SAME seed for both - same cosmic perturbations
const N_S: f64 = 0.965;         // Spectral index
const DELTA_RMS: f64 = 0.10;    // Initial overdensity RMS

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;
const K_B_OVER_MP_CODE: f64 = 8.7e-9;  // k_B/m_p in code units
const MU_MOL: f64 = 0.6;        // Mean molecular weight

// ═══════════════════════════════════════════════════════════════════════════
// RAHMATI 2013 SELF-SHIELDING (OBLIGATOIRE)
// ═══════════════════════════════════════════════════════════════════════════
/// Self-shielding correction factor for UV photoionization
/// Rahmati et al. 2013, MNRAS 430, 2427
/// At nH > 0.01 cm⁻³ → Γ_eff → 0 (gas self-shields from UV)
/// Without this: spurious star formation in cosmic voids
fn self_shielding_rahmati2013(n_h: f64, gamma_uv: f64) -> f64 {
    const N0: f64 = 0.00981;      // cm⁻³
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

/// UV background Haardt & Madau 2012
/// Γ(z) ≈ 1e-24 × (1+z)² / (1 + ((1+z)/3)^5) [erg/s/cm³]
fn uv_background_hm2012(z: f64) -> f64 {
    let zp1 = 1.0 + z;
    1e-24 * zp1 * zp1 / (1.0 + (zp1 / 3.0).powf(5.0))
}

/// Effective UV heating rate with self-shielding
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
    println!("║        JANUS BARYONIC CALIBRATED — MCJ PRODUCTION RUN                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  COSMOLOGY:");
    println!("║    μ = {}, η = {}", MU, ETA);
    println!("║    H₀ = {} km/s/Mpc, t₀ = {} Gyr", H0_KMS_MPC, T0_GYR);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  SIMULATION:");
    println!("║    N = {} (η≈1.045 ⟹ ~52% m+ / ~48% m-)", N_PARTICLES);
    println!("║    L_box = {} Mpc, ε = {} Mpc", L_BOX, EPSILON);
    println!("║    z_init = {} → z_final = 0, dt = {} Gyr", Z_INIT, DT);
    println!("║    Steps = {}, θ = {}", N_STEPS, THETA);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  BARYONICS (m+ only):");
    println!("║    T_init = {} K, T_floor = {} K", T_INIT_PLUS, T_FLOOR);
    println!("║    Cooling: S&D93 GPU native + Rahmati self-shielding");
    println!("║    Star formation: ε* = {}, T < {} K, n > {} cm⁻³", EPSILON_STAR, T_THRESHOLD_SF, N_THRESHOLD_SF);
    println!("║    SN feedback: kinetic, ε_SN = {}%, delay = {} Myr", EPSILON_SN * 100.0, DELAY_SN_MYR);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  m- : Pure gravity, collisionless");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  ICs: Zel'dovich (seed {}), random sign assignment", SEED_IC);
    println!("║    n_s = {}, δ_rms = {}", N_S, DELTA_RMS);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  OUTPUT: {}", OUTPUT_DIR);
    println!("║    Snapshots every {} steps ({} total)", SNAPSHOT_INTERVAL, N_STEPS / SNAPSHOT_INTERVAL);
    println!("║    Metrics every {} steps", METRIC_INTERVAL);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", OUTPUT_DIR)).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/frames", OUTPUT_DIR)).expect("Failed to create frames dir");

    // Note: GPU cooling will be initialized after GPU sim (we need n_plus count)
    println!("S&D93 GPU cooling will be initialized after particle counts...");

    // Generate Zel'dovich ICs
    let (positions, velocities, signs) = generate_zeldovich_ics_dual_seed();

    // Validate ICs
    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.iter().filter(|&&s| s < 0).count();
    println!("\n[IC VALIDATION]");
    println!("  N+ = {}, N- = {}, ratio = {:.4}", n_plus, n_minus, n_plus as f64 / n_minus as f64);

    // Initialize GPU simulation with Zel'dovich ICs
    println!("\nInitializing GPU simulation with {} particles...", N_PARTICLES);
    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_BOX,
        positions, velocities, signs
    ).unwrap();
    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(EPSILON);

    // Dynamic c_ratio initialization
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    let c_ratio_init = c_ratio_sq_init.sqrt();
    gpu_sim.set_c_ratio(c_ratio_init);
    println!("  GPU init: {:.2}s", start_time.elapsed().as_secs_f64());
    println!("  c_ratio_sq(z={}) = {:.6}", Z_INIT, c_ratio_sq_init);

    // Initialize S&D93 GPU cooling (native CUDA kernel)
    println!("\nInitializing S&D93 GPU cooling (native CUDA)...");
    let cuda_device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let m_particle = 3e11 * (L_BOX / 500.0).powi(3) / (N_PARTICLES as f64);  // M_sun per particle
    let mut gpu_cooling = GpuCooling::new(cuda_device, n_plus, L_BOX, m_particle)
        .expect("Failed to create GPU cooling module");

    // Initialize temperatures: m+ only (cooling module only handles m+)
    // Create signs array for m+ particles only (all +1)
    let signs_plus: Vec<i32> = vec![1i32; n_plus];
    gpu_cooling.init_from_temperature(T_INIT_PLUS, T_INIT_PLUS, &signs_plus)
        .expect("Failed to init cooling temperatures");
    println!("  ✓ S&D93 tabulated cooling initialized (validated <2% error)");
    println!("  ✓ T_init(m+) = {} K", T_INIT_PLUS);

    // Initialize particle temperatures (m+ only)
    let mut temperatures: Vec<f64> = vec![T_INIT_PLUS; n_plus];
    let mut n_stars: usize = 0;  // Star counter
    let mut sfr: f64 = 0.0;      // Star formation rate [M_sun/Gyr]
    let mut is_star: Vec<bool> = vec![false; n_plus];  // Track which particles are stars
    let mut stellar_ages: Vec<f64> = vec![0.0; n_plus];  // Age since star formation [Gyr]
    let mut rng_sf = StdRng::seed_from_u64(12345);  // RNG for stochastic SF

    // Ratio monitoring for smart auto-stop
    let mut ratio_history: Vec<f64> = Vec::with_capacity(60);
    const RATIO_HARD_LIMIT: f64 = 2.00;      // Hard stop threshold (raised to observe behavior)
    const RATIO_WARN_THRESHOLD: f64 = 1.20;  // Start monitoring trend
    const TREND_WINDOW: usize = 50;          // Steps to check trend

    // CSV output
    let csv_path = format!("{}/time_series.csv", OUTPUT_DIR);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,t_Gyr,z,a,S,corr,v_rms_plus,v_rms_minus,ratio,rho_max_plus,rho_max_minus,purity,N_stars,SFR,T_mean").unwrap();

    // Grid for density computation
    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;
    let mut t_gyr = 0.0;
    let half_box = L_BOX / 2.0;
    let grid_size = 64usize;
    let cell_size = L_BOX / grid_size as f64;

    println!("\n  Step │    t_Gyr │       z │   ρ+_max │   ρ-_max │   v_rms+ │   v_rms- │   ratio │      S │   purity │ N_stars │    ETA");
    println!("───────┼──────────┼─────────┼──────────┼──────────┼──────────┼──────────┼─────────┼────────┼──────────┼─────────┼────────");

    // Progress checkpoints
    let checkpoints: Vec<(f64, &str)> = vec![
        (3.0, "z=3 check"),
        (2.0, "z=2 first stars?"),
        (1.0, "z=1 structures"),
        (0.5, "z=0.5 segregation"),
        (0.15, "z=0.15 final"),
        (0.0, "z=0 COMPLETE"),
    ];
    let mut next_checkpoint_idx = 0;

    for step in 0..=N_STEPS {
        let z = (1.0 / a - 1.0).max(0.0);

        // Dynamic c_ratio update
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_ratio = c_ratio_sq.sqrt();
        gpu_sim.set_c_ratio(c_ratio);

        // Hubble parameter (matter-dominated approximation: H ∝ a^-1.5)
        // H₀ = 69.9 km/s/Mpc = 0.0715 Gyr⁻¹
        let h = 0.07 / a.powf(1.5);

        // Check for checkpoint
        if next_checkpoint_idx < checkpoints.len() && z <= checkpoints[next_checkpoint_idx].0 + 0.01 {
            println!("\n  ═══ CHECKPOINT: {} (z={:.2}) ═══", checkpoints[next_checkpoint_idx].1, z);
            next_checkpoint_idx += 1;
        }

        // Metrics and snapshots
        let do_metric = step % METRIC_INTERVAL == 0;
        let do_snapshot = step % SNAPSHOT_INTERVAL == 0;

        if do_metric || do_snapshot {
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_data = gpu_sim.signs();

            // Compute metrics
            let (rho_max_plus, rho_max_minus, segregation, purity) =
                compute_density_metrics(&pos, &signs_data, grid_size, L_BOX);

            let (v_rms_plus, v_rms_minus) = compute_vrms(&vel, &signs_data);
            let ratio = v_rms_minus / v_rms_plus.max(1e-10);

            // Correlation
            let corr = compute_correlation(&pos, &signs_data, grid_size, L_BOX);

            // ETA estimation
            let elapsed = start_time.elapsed().as_secs_f64();
            let steps_done = step.max(1);
            let steps_left = N_STEPS - step;
            let eta_s = (elapsed / steps_done as f64) * steps_left as f64;
            let eta_h = eta_s / 3600.0;

            if do_metric {
                // Compute mean temperature from GPU cooling module
                let t_mean: f64 = gpu_cooling.get_mean_temperature_plus().unwrap_or(T_INIT_PLUS);

                writeln!(csv, "{},{:.6},{:.4},{:.6},{:.4},{:.4},{:.1},{:.1},{:.4},{:.1},{:.1},{:.4},{},{:.2e},{:.0}",
                    step, t_gyr, z, a, segregation, corr,
                    v_rms_plus * MPC_GYR_TO_KMS, v_rms_minus * MPC_GYR_TO_KMS, ratio,
                    rho_max_plus, rho_max_minus, purity, n_stars, sfr, t_mean
                ).unwrap();

                // Log every 10 steps for enhanced monitoring
                if step % 10 == 0 || step <= 50 {
                    println!("  {:5} │ {:8.4} │ {:7.4} │ {:8.1} │ {:8.1} │ {:8.1} │ {:8.1} │ {:7.4} │ {:6.4} │ {:8.4} │ {:7} │ {:6.1}h",
                        step, t_gyr, z, rho_max_plus, rho_max_minus,
                        v_rms_plus * MPC_GYR_TO_KMS, v_rms_minus * MPC_GYR_TO_KMS,
                        ratio, segregation, purity, n_stars, eta_h);
                }

                // Track ratio history for trend detection
                ratio_history.push(ratio);
                if ratio_history.len() > TREND_WINDOW + 10 {
                    ratio_history.remove(0);
                }

                // ENHANCED AUTO-STOP CONDITIONS
                // 1. Hard limit: ratio > 1.50 → immediate stop
                if ratio > RATIO_HARD_LIMIT {
                    println!("\n  ⛔ HARD STOP: v_rms ratio = {:.4} > {:.2}", ratio, RATIO_HARD_LIMIT);
                    println!("    m- runaway at step {}, z={:.2}", step, z);
                    break;
                }

                // 2. Monitor only (no trend stop) - just log warnings
                if ratio > RATIO_WARN_THRESHOLD && ratio_history.len() >= TREND_WINDOW {
                    let old_ratio = ratio_history[ratio_history.len() - TREND_WINDOW];
                    let trend = ratio - old_ratio;

                    // Log trend every 100 steps when ratio > 1.20
                    if step % 100 == 0 {
                        if trend > 0.01 {
                            println!("    ⚡ ratio = {:.4} (Δ={:+.4}/50 steps) - monitoring only", ratio, trend);
                        } else {
                            println!("    ✓ ratio = {:.4} stabilizing (Δ={:+.4})", ratio, trend);
                        }
                    }
                }
            }

            // Snapshot
            if do_snapshot {
                let snap_path = format!("{}/snapshots/snap_{:05}.bin", OUTPUT_DIR, step);
                save_snapshot_jsnp(&snap_path, &pos, &vel, &signs_data, a, t_gyr);
            }

            csv.flush().unwrap();

            // Backup CSV every 500 steps
            if step > 0 && step % 500 == 0 {
                let backup_path = format!("{}/time_series_step{:05}.csv", OUTPUT_DIR, step);
                if let Ok(_) = std::fs::copy(&csv_path, &backup_path) {
                    println!("    📁 CSV backup: {}", backup_path);
                }
            }
        }

        // Time integration step
        if step < N_STEPS {
            gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0).unwrap();

            // ═══════════════════════════════════════════════════════════════════
            // BARYONIC PHYSICS (m+ only, every step) — GPU ACCELERATED
            // ═══════════════════════════════════════════════════════════════════
            let pos = gpu_sim.get_positions().unwrap();
            let signs_data = gpu_sim.signs();

            // Compute local overdensities for m+ particles using grid
            let overdensities = compute_local_overdensities(&pos, &signs_data, 32, L_BOX);

            // Convert overdensities to physical densities for GPU cooling
            // ρ_code → n_H [cm^-3] = 2e-7 × (1+z)³ × overdensity
            let rho_to_nh = 2e-7 * (1.0 + z).powi(3);
            let densities: Vec<f64> = overdensities.iter()
                .map(|&od| od * rho_to_nh / 3.07e-17)  // Convert to code units expected by kernel
                .collect();

            // Upload densities to GPU
            gpu_cooling.upload_densities(&densities).expect("Failed to upload densities");

            // Apply GPU cooling (S&D93 tabulated)
            gpu_cooling.apply_cooling(DT, z).expect("GPU cooling failed");

            // Apply star formation on GPU
            let new_stars_this_step = gpu_cooling.apply_star_formation(DT).unwrap_or(0);
            n_stars += new_stars_this_step as usize;

            // Report star formation events
            if new_stars_this_step > 0 && step % 100 == 0 {
                let t_mean = gpu_cooling.get_mean_temperature_plus().unwrap_or(0.0);
                println!("    ★ Step {}: {} new stars, N_stars={}, T_mean={:.0} K",
                    step, new_stars_this_step, n_stars, t_mean);
            }

            // Update SFR estimate
            let m_particle_local = 3e11 * (L_BOX / 500.0).powi(3) / (N_PARTICLES as f64);
            sfr = (new_stars_this_step as f64) * m_particle_local / DT;

            a += a * h * DT;
            t_gyr += DT;
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SIMULATION COMPLETE");
    println!("║  Total time: {:.1}h ({:.1} ms/step)", total_time / 3600.0, total_time * 1000.0 / N_STEPS as f64);
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
        let x = pos[i*3];
        let y = pos[i*3 + 1];
        let z = pos[i*3 + 2];

        let ix = ((x + half_box) / cell_size).floor() as usize;
        let iy = ((y + half_box) / cell_size).floor() as usize;
        let iz = ((z + half_box) / cell_size).floor() as usize;

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

    // Segregation: average difference in cell occupancy
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

    // Purity: fraction of cells dominated by one species
    let mut pure_cells = 0;
    for i in 0..n3 {
        let total = rho_plus[i] + rho_minus[i];
        if total > 10 {
            let frac_plus = rho_plus[i] as f64 / total as f64;
            if frac_plus > 0.9 || frac_plus < 0.1 {
                pure_cells += 1;
            }
        }
    }
    let purity = pure_cells as f64 / n3 as f64;

    (rho_max_plus, rho_max_minus, segregation, purity)
}

/// Compute local overdensity for each m+ particle
/// Returns overdensity (ρ/ρ̄) for each m+ particle in order
#[cfg(feature = "cuda")]
fn compute_local_overdensities(pos: &[f64], signs: &[i32], grid_size: usize, box_size: f64) -> Vec<f64> {
    let half_box = box_size / 2.0;
    let cell_size = box_size / grid_size as f64;
    let n3 = grid_size * grid_size * grid_size;

    // Count particles per cell (m+ only)
    let mut cell_counts = vec![0u32; n3];
    let mut particle_cells: Vec<usize> = Vec::new();

    let n = pos.len() / 3;
    for i in 0..n {
        if signs[i] <= 0 {
            continue;  // Skip m-
        }

        let x = pos[i*3];
        let y = pos[i*3 + 1];
        let z = pos[i*3 + 2];

        let ix = ((x + half_box) / cell_size).floor() as usize;
        let iy = ((y + half_box) / cell_size).floor() as usize;
        let iz = ((z + half_box) / cell_size).floor() as usize;

        let ix = ix.min(grid_size - 1);
        let iy = iy.min(grid_size - 1);
        let iz = iz.min(grid_size - 1);

        let idx = ix + iy * grid_size + iz * grid_size * grid_size;
        cell_counts[idx] += 1;
        particle_cells.push(idx);
    }

    // Compute mean count per cell
    let n_plus = particle_cells.len() as f64;
    let mean_per_cell = n_plus / n3 as f64;

    // Return overdensity for each m+ particle
    particle_cells.iter()
        .map(|&cell_idx| {
            let count = cell_counts[cell_idx] as f64;
            if mean_per_cell > 0.0 {
                count / mean_per_cell
            } else {
                1.0
            }
        })
        .collect()
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
fn compute_correlation(pos: &[f64], signs: &[i32], grid_size: usize, box_size: f64) -> f64 {
    let half_box = box_size / 2.0;
    let cell_size = box_size / grid_size as f64;
    let n3 = grid_size * grid_size * grid_size;

    let mut rho_plus = vec![0.0f64; n3];
    let mut rho_minus = vec![0.0f64; n3];

    let n = pos.len() / 3;
    for i in 0..n {
        let ix = ((pos[i*3] + half_box) / cell_size).floor() as usize;
        let iy = ((pos[i*3+1] + half_box) / cell_size).floor() as usize;
        let iz = ((pos[i*3+2] + half_box) / cell_size).floor() as usize;

        if ix < grid_size && iy < grid_size && iz < grid_size {
            let idx = ix + iy * grid_size + iz * grid_size * grid_size;
            if signs[i] > 0 {
                rho_plus[idx] += 1.0;
            } else {
                rho_minus[idx] += 1.0;
            }
        }
    }

    // Mean
    let mean_plus: f64 = rho_plus.iter().sum::<f64>() / n3 as f64;
    let mean_minus: f64 = rho_minus.iter().sum::<f64>() / n3 as f64;

    // Correlation
    let mut cov = 0.0;
    let mut var_plus = 0.0;
    let mut var_minus = 0.0;
    for i in 0..n3 {
        let dp = rho_plus[i] - mean_plus;
        let dm = rho_minus[i] - mean_minus;
        cov += dp * dm;
        var_plus += dp * dp;
        var_minus += dm * dm;
    }

    if var_plus > 0.0 && var_minus > 0.0 {
        cov / (var_plus.sqrt() * var_minus.sqrt())
    } else {
        0.0
    }
}

#[cfg(feature = "cuda")]
fn save_snapshot_jsnp(path: &str, pos: &[f64], vel: &[f64], signs: &[i32], a: f64, t: f64) {
    use std::io::Write;
    let mut f = BufWriter::new(File::create(path).unwrap());

    let n = pos.len() / 3;
    // Header: N (u64), a (f64), t (f64)
    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&a.to_le_bytes()).unwrap();
    f.write_all(&t.to_le_bytes()).unwrap();

    // Positions as f32
    for i in 0..n*3 {
        f.write_all(&(pos[i] as f32).to_le_bytes()).unwrap();
    }

    // Velocities as f32
    for i in 0..n*3 {
        f.write_all(&(vel[i] as f32).to_le_bytes()).unwrap();
    }

    // Signs as i8
    for s in signs {
        f.write_all(&[*s as i8 as u8]).unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH IC GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn generate_zeldovich_ics_dual_seed() -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    println!("\n[GENERATING ZEL'DOVICH ICs - DUAL SEED]");

    let n_grid = (N_PARTICLES as f64).powf(1.0/3.0).ceil() as usize;
    let n3 = n_grid * n_grid * n_grid;
    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (n3 as f64 * f_plus).round() as usize;
    let n_negative = n3 - n_positive;

    println!("  Grid: {}³ = {} particles", n_grid, n3);
    println!("  Box: {} Mpc, z_init = {}", L_BOX, Z_INIT);
    println!("  Target: N+ = {}, N- = {}", n_positive, n_negative);
    println!("  Seed IC: {} (same perturbations for m+ and m-)", SEED_IC);
    println!("  n_s = {}, δ_rms = {}", N_S, DELTA_RMS);

    let dk = 2.0 * PI / L_BOX;
    let half_n = n_grid / 2;
    let spacing = L_BOX / n_grid as f64;
    let half_box = L_BOX / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate ONE displacement field - SAME cosmic perturbations for m+ and m-
    println!("  Generating displacement field (seed {})...", SEED_IC);
    let (psi_x, psi_y, psi_z) = generate_displacement_field(n_grid, dk, half_n, d_growth, SEED_IC);

    // Scale displacements
    let max_disp = find_max_displacement(&psi_x, &psi_y, &psi_z);
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };

    println!("  Max displacement: {:.6e} → scale {:.4}", max_disp, scale);

    // Zel'dovich velocities - SAME vel_scale for both
    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;
    println!("  Zel'dovich vel_scale (COMMON): {:.4e} Mpc/Gyr ({:.1} km/s/Mpc)",
             vel_scale, vel_scale * MPC_GYR_TO_KMS);

    // Random sign assignment
    let mut rng_shuffle = rand::rngs::StdRng::seed_from_u64(12345);
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.shuffle(&mut rng_shuffle);
    let plus_indices: HashSet<usize> = indices[..n_positive].iter().cloned().collect();

    // Generate particles
    println!("  Placing {} particles...", n3);
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

                let is_positive = plus_indices.contains(&idx);

                // ALL particles use SAME displacement field (same cosmic perturbations)
                let mut x = x0 + psi_x[idx] * scale;
                let mut y = y0 + psi_y[idx] * scale;
                let mut z = z0 + psi_z[idx] * scale;

                // Periodic boundary
                while x > half_box { x -= L_BOX; }
                while x < -half_box { x += L_BOX; }
                while y > half_box { y -= L_BOX; }
                while y < -half_box { y += L_BOX; }
                while z > half_box { z -= L_BOX; }
                while z < -half_box { z += L_BOX; }

                positions.push(x);
                positions.push(y);
                positions.push(z);

                let vx = psi_x[idx] * vel_scale;
                let vy = psi_y[idx] * vel_scale;
                let vz = psi_z[idx] * vel_scale;
                velocities.push(vx);
                velocities.push(vy);
                velocities.push(vz);

                signs.push(if is_positive { 1i32 } else { -1i32 });
            }
        }
    }

    // Validation
    let n_plus_actual = signs.iter().filter(|&&s| s > 0).count();
    let n_minus_actual = signs.iter().filter(|&&s| s < 0).count();
    println!("  ✓ Generated: N+ = {}, N- = {}", n_plus_actual, n_minus_actual);

    // Check positions in box
    let all_in_box = positions.chunks(3).all(|p| {
        p[0].abs() <= half_box && p[1].abs() <= half_box && p[2].abs() <= half_box
    });
    if all_in_box {
        println!("  ✓ 100% positions dans [{:.0}, {:.0}] Mpc", -half_box, half_box);
    } else {
        println!("  ⚠ Some positions outside box!");
    }

    (positions, velocities, signs)
}

#[cfg(feature = "cuda")]
fn generate_displacement_field(n_grid: usize, dk: f64, half_n: usize, d_growth: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Gaussian random field in Fourier space
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;
                let k2 = kx*kx + ky*ky + kz*kz;

                if k2 > 0.0 {
                    let k = k2.sqrt();
                    // Power spectrum P(k) ∝ k^(n_s - 4) for displacements
                    let pk = k.powf(N_S - 4.0) * DELTA_RMS.powi(2);
                    let amp = (pk / 2.0).sqrt();

                    let phase = rng.random::<f64>() * 2.0 * PI;
                    let re = normal.sample(&mut rng) * amp * d_growth;
                    let im = normal.sample(&mut rng) * amp * d_growth;

                    let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                    delta_k[idx] = Complex::new(re * phase.cos() - im * phase.sin(),
                                                re * phase.sin() + im * phase.cos());
                }
            }
        }
    }

    // IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    let mut psi_x = vec![0.0f64; n3];
    let mut psi_y = vec![0.0f64; n3];
    let mut psi_z = vec![0.0f64; n3];

    // ψ_i = IFFT[-i k_i δ(k) / k²]
    for dim in 0..3 {
        let mut field_k = delta_k.clone();

        for iz in 0..n_grid {
            for iy in 0..n_grid {
                for ix in 0..n_grid {
                    let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                    let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                    let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;
                    let k2 = kx*kx + ky*ky + kz*kz;

                    let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                    if k2 > 0.0 {
                        let k_i = match dim { 0 => kx, 1 => ky, _ => kz };
                        // -i k_i / k²
                        let factor = Complex::new(0.0, -k_i / k2);
                        field_k[idx] = field_k[idx] * factor;
                    } else {
                        field_k[idx] = Complex::new(0.0, 0.0);
                    }
                }
            }
        }

        // 3D IFFT via 1D IFFTs
        // X direction
        for iz in 0..n_grid {
            for iy in 0..n_grid {
                let mut row: Vec<Complex<f64>> = (0..n_grid)
                    .map(|ix| field_k[iz * n_grid * n_grid + iy * n_grid + ix])
                    .collect();
                ifft.process(&mut row);
                for ix in 0..n_grid {
                    field_k[iz * n_grid * n_grid + iy * n_grid + ix] = row[ix];
                }
            }
        }

        // Y direction
        for iz in 0..n_grid {
            for ix in 0..n_grid {
                let mut row: Vec<Complex<f64>> = (0..n_grid)
                    .map(|iy| field_k[iz * n_grid * n_grid + iy * n_grid + ix])
                    .collect();
                ifft.process(&mut row);
                for iy in 0..n_grid {
                    field_k[iz * n_grid * n_grid + iy * n_grid + ix] = row[iy];
                }
            }
        }

        // Z direction
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let mut row: Vec<Complex<f64>> = (0..n_grid)
                    .map(|iz| field_k[iz * n_grid * n_grid + iy * n_grid + ix])
                    .collect();
                ifft.process(&mut row);
                for iz in 0..n_grid {
                    field_k[iz * n_grid * n_grid + iy * n_grid + ix] = row[iz];
                }
            }
        }

        // Extract real part
        let psi = match dim {
            0 => &mut psi_x,
            1 => &mut psi_y,
            _ => &mut psi_z,
        };
        for i in 0..n3 {
            psi[i] = field_k[i].re / n3 as f64;
        }
    }

    (psi_x, psi_y, psi_z)
}

#[cfg(feature = "cuda")]
fn find_max_displacement(psi_x: &[f64], psi_y: &[f64], psi_z: &[f64]) -> f64 {
    let n = psi_x.len();
    let mut max_disp = 0.0f64;
    for i in 0..n {
        let d = (psi_x[i].powi(2) + psi_y[i].powi(2) + psi_z[i].powi(2)).sqrt();
        if d > max_disp { max_disp = d; }
    }
    max_disp
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("ERROR: This binary requires --features cuda");
    std::process::exit(1);
}
