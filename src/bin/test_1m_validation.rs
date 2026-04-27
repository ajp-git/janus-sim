//! Test 1M Validation Run with Zel'dovich ICs
//!
//! Separate seeds for m+ (42) and m- (43) populations
//! L_box = 100 Mpc for sufficient resolution at 1M particles

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use sha2::{Sha256, Digest};

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
#[cfg(feature = "cuda")]
use janus::treepm::pm_grid::PmGrid;

// ============================================================================
// SIMULATION PARAMETERS
// ============================================================================

const N_PARTICLES: usize = 1_000_000;
const L_BOX: f64 = 100.0;           // Mpc (reduced for better resolution)
const ETA: f64 = 1.045;             // Janus mass ratio
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.005;              // Gyr
const STEPS: usize = 1200;          // ~z=0
const THETA: f64 = 0.7;             // Tree opening angle
const SOFTENING: f64 = 0.05;        // Softening [Mpc] (scaled for 100 Mpc box)
const R_CUT: f64 = 10.0;            // TreePM cutoff [Mpc]
const PM_GRID: usize = 128;         // PM grid resolution

const SEED_PLUS: u64 = 42;          // Seed for m+ ICs
const SEED_MINUS: u64 = 43;         // Seed for m- ICs

const SNAPSHOT_INTERVAL: usize = 10;
const CSV_INTERVAL: usize = 1;
const VRMS_RATIO_MAX: f64 = 1.15;   // Auto-stop threshold

// Zel'dovich IC parameters
const N_S: f64 = 0.965;             // Spectral index
const DELTA_RMS: f64 = 0.1;         // Target δ_rms
const K0: f64 = 0.1;                // Turnover scale [Mpc⁻¹]
const MPC_GYR_TO_KMS: f64 = 977.8;  // Mpc/Gyr → km/s conversion

#[cfg(feature = "cuda")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     JANUS TEST 1M VALIDATION - Zel'dovich ICs                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  N = 1,000,000  |  L = 100 Mpc  |  η = 1.045                 ║");
    println!("║  z: 5 → 0       |  θ = 0.7      |  ε = 0.05 Mpc              ║");
    println!("║  Seeds: m+ = 42, m- = 43  |  n_s = 0.965  |  δ_rms = 0.1     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Output directory
    let base_dir = std::path::Path::new("/app/output/test_1m_zeldovich");
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    // Generate Zel'dovich ICs with separate seeds
    let (positions, velocities, signs) = generate_zeldovich_ics_dual_seed();

    let actual_n_plus = signs.iter().filter(|&&s| s > 0).count();
    let actual_n_minus = signs.iter().filter(|&&s| s < 0).count();
    println!("Final particle counts: N+ = {}, N- = {}", actual_n_plus, actual_n_minus);

    // Initialize GPU solver
    println!("\nInitializing GPU solver...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        positions, velocities, signs.clone(), L_BOX
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);

    println!("  θ = {} (Barnes-Hut)", THETA);
    println!("  ε = {} Mpc (softening)", SOFTENING);
    println!("  λ₀ = 0.0 (pure anti-Newton)");

    // Initialize PM grid
    println!("Initializing PM grid ({}³)...", PM_GRID);
    let mut pm_grid = PmGrid::new(PM_GRID, L_BOX);
    println!("  PM memory: {:.2} MB", pm_grid.memory_bytes() as f64 / 1e6);

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    // CSV logging
    let csv_path = base_dir.join("time_series.csv");
    let mut csv_file = BufWriter::new(File::create(&csv_path).expect("Failed to create CSV"));
    writeln!(csv_file, "step,time_gyr,a,z,S,corr,vrms_plus_kms,vrms_minus_kms,vrms_ratio,rho_max_plus,purity").unwrap();

    let start_time = Instant::now();

    println!();
    println!("Starting integration from z={:.1}...", Z_INIT);
    println!("══════════════════════════════════════════════════════════════════");
    println!("{:>6} {:>8} {:>6} {:>8} {:>8} {:>10} {:>8} {:>8}",
             "step", "z", "S", "corr", "v_ratio", "ρ_max+", "P", "status");
    println!("──────────────────────────────────────────────────────────────────");

    let mut final_snapshot_path = String::new();
    let mut final_z = Z_INIT;

    for step in 0..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };
        final_z = z;

        // Integration step
        if step > 0 {
            sim.set_current_z(z);
            sim.step_treepm_hybrid(DT, &mut pm_grid, R_CUT, h, dtau_per_dt)
                .expect("TreePM step failed");
        }

        // Compute metrics
        if step % CSV_INTERVAL == 0 || step % SNAPSHOT_INTERVAL == 0 {
            let (positions, velocities, _) = sim.get_particles()
                .expect("Failed to get particles");

            let purity = sim.local_purity(32).unwrap_or(0.0);
            let t = step as f64 * DT;

            let (seg, corr, vrms_plus, vrms_minus, rho_max_plus) =
                compute_metrics(&positions, &velocities, &signs, L_BOX as f32);

            // Convert to km/s for reporting
            let vrms_plus_kms = vrms_plus * MPC_GYR_TO_KMS;
            let vrms_minus_kms = vrms_minus * MPC_GYR_TO_KMS;
            let vrms_ratio = vrms_plus_kms / vrms_minus_kms.max(1e-10);

            // Log to CSV (velocities in km/s)
            writeln!(csv_file, "{},{:.6},{:.6},{:.4},{:.4},{:.4},{:.1},{:.1},{:.4},{:.3e},{:.4}",
                step, t, a, z, seg, corr, vrms_plus_kms, vrms_minus_kms, vrms_ratio, rho_max_plus, purity).unwrap();

            // Console output
            if step % 10 == 0 {
                let status = if vrms_ratio > VRMS_RATIO_MAX { "⚠ WARN" } else { "✓ OK" };
                println!("{:>6} {:>8.3} {:>6.3} {:>8.4} {:>8.4} {:>10.2e} {:>8.3} {:>8}",
                    step, z, seg, corr, vrms_ratio, rho_max_plus, purity, status);
            }

            // Auto-stop check
            if vrms_ratio > VRMS_RATIO_MAX && step > 100 {
                println!();
                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  ⛔ AUTO-STOP: v_rms ratio {:.3} > {:.2} threshold           ║", vrms_ratio, VRMS_RATIO_MAX);
                println!("╚══════════════════════════════════════════════════════════════╝");
                break;
            }
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = snap_dir.join(format!("snap_{:05}.bin", step));
            save_snapshot(&sim, &snap_path, step, z);
            final_snapshot_path = snap_path.to_string_lossy().to_string();
        }

        // Progress every 100 steps
        if step > 0 && step % 100 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            let remaining = (STEPS - step) as f64 / rate;
            println!("  [Progress: {}/{} steps, {:.1}s elapsed, ~{:.0}s remaining]",
                step, STEPS, elapsed, remaining);
        }
    }

    csv_file.flush().unwrap();

    let elapsed = start_time.elapsed();

    println!("──────────────────────────────────────────────────────────────────");
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    SIMULATION COMPLETE                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Steps: {:>6}  |  z_final: {:>6.3}  |  Time: {:>6.1} min       ║",
        STEPS, final_z, elapsed.as_secs_f64() / 60.0);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Compute SHA-256 of final snapshot
    println!();
    println!("Computing SHA-256 checksum of final snapshot...");
    let checksum = compute_sha256(&final_snapshot_path).unwrap_or_else(|_| "N/A".to_string());
    println!("Final snapshot: {}", final_snapshot_path);
    println!("SHA-256: {}", checksum);

    // Final metrics
    let (positions, velocities, _) = sim.get_particles().expect("Failed to get particles");
    let purity = sim.local_purity(32).unwrap_or(0.0);
    let (seg, corr, vrms_plus, vrms_minus, rho_max_plus) =
        compute_metrics(&positions, &velocities, &signs, L_BOX as f32);

    // Convert to km/s
    let vrms_plus_kms = vrms_plus * MPC_GYR_TO_KMS;
    let vrms_minus_kms = vrms_minus * MPC_GYR_TO_KMS;

    // Save summary
    let summary_path = base_dir.join("summary.json");
    let summary = format!(r#"{{
    "n_particles": {},
    "l_box": {},
    "eta": {},
    "z_init": {},
    "z_final": {:.4},
    "steps": {},
    "elapsed_seconds": {:.1},
    "segregation": {:.4},
    "correlation": {:.4},
    "vrms_plus_kms": {:.1},
    "vrms_minus_kms": {:.1},
    "vrms_ratio": {:.4},
    "rho_max_plus": {:.3e},
    "purity": {:.4},
    "final_snapshot": "{}",
    "sha256": "{}",
    "ics": "zeldovich_dual_seed",
    "seed_plus": {},
    "seed_minus": {},
    "n_s": {},
    "delta_rms": {}
}}"#,
        N_PARTICLES, L_BOX, ETA, Z_INIT, final_z, STEPS,
        elapsed.as_secs_f64(), seg, corr, vrms_plus_kms, vrms_minus_kms,
        vrms_plus_kms / vrms_minus_kms.max(1e-10), rho_max_plus, purity,
        final_snapshot_path, checksum,
        SEED_PLUS, SEED_MINUS, N_S, DELTA_RMS
    );

    fs::write(&summary_path, &summary).expect("Failed to write summary");
    println!();
    println!("Summary saved to: {}", summary_path.display());
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
    std::process::exit(1);
}

/// Generate Zel'dovich ICs with separate seeds for m+ and m- populations
#[cfg(feature = "cuda")]
fn generate_zeldovich_ics_dual_seed() -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let n_grid = (N_PARTICLES as f64).powf(1.0/3.0).ceil() as usize;
    let n3 = n_grid * n_grid * n_grid;
    let n_positive = (n3 as f64 / (1.0 + ETA)) as usize;
    let n_negative = n3 - n_positive;

    println!("Generating Zel'dovich ICs with DUAL SEEDS...");
    println!("  Grid: {}³ = {} particles", n_grid, n3);
    println!("  Box: {:.1} Mpc", L_BOX);
    println!("  n_s = {}, δ_rms = {}", N_S, DELTA_RMS);
    println!("  Seed m+: {}, Seed m-: {}", SEED_PLUS, SEED_MINUS);
    println!("  Target: N+ = {}, N- = {}", n_positive, n_negative);

    let dk = 2.0 * PI / L_BOX;
    let half_n = n_grid / 2;
    let spacing = L_BOX / n_grid as f64;
    let half_box = L_BOX / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate displacement fields for m+ (seed 42)
    println!("  Generating m+ displacement field (seed {})...", SEED_PLUS);
    let (psi_x_plus, psi_y_plus, psi_z_plus) = generate_displacement_field(n_grid, dk, half_n, d_growth, SEED_PLUS);

    // Generate displacement fields for m- (seed 43)
    println!("  Generating m- displacement field (seed {})...", SEED_MINUS);
    let (psi_x_minus, psi_y_minus, psi_z_minus) = generate_displacement_field(n_grid, dk, half_n, d_growth, SEED_MINUS);

    // Scale displacements
    let max_disp_plus = find_max_displacement(&psi_x_plus, &psi_y_plus, &psi_z_plus);
    let max_disp_minus = find_max_displacement(&psi_x_minus, &psi_y_minus, &psi_z_minus);
    let target_disp = spacing * 0.3;
    let scale_plus = if max_disp_plus > 1e-10 { target_disp / max_disp_plus } else { 1.0 };
    let scale_minus = if max_disp_minus > 1e-10 { target_disp / max_disp_minus } else { 1.0 };

    println!("  Max displacement m+: {:.6e} → scale {:.4}", max_disp_plus, scale_plus);
    println!("  Max displacement m-: {:.6e} → scale {:.4}", max_disp_minus, scale_minus);

    // Zel'dovich velocities: v = D_dot × Ψ
    // D_dot = sqrt(1+z) in matter-dominated era
    let d_dot = (1.0 + Z_INIT).sqrt();  // sqrt(6) ≈ 2.45 at z=5
    let vel_scale_plus = d_dot * scale_plus;
    let vel_scale_minus = d_dot * scale_minus;
    println!("  Zel'dovich vel_scale m+: {:.4e} Mpc/Gyr ({:.1} km/s/Mpc)",
             vel_scale_plus, vel_scale_plus * MPC_GYR_TO_KMS);
    println!("  Zel'dovich vel_scale m-: {:.4e} Mpc/Gyr ({:.1} km/s/Mpc)",
             vel_scale_minus, vel_scale_minus * MPC_GYR_TO_KMS);

    // Create shuffled sign assignment
    let mut rng_shuffle = rand::rngs::StdRng::seed_from_u64(12345);
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.shuffle(&mut rng_shuffle);

    let plus_indices: std::collections::HashSet<usize> = indices[..n_positive].iter().cloned().collect();

    // Generate particles with Zel'dovich velocities
    println!("  Placing {} particles with Zel'dovich velocities...", n3);
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n3);

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let is_positive = plus_indices.contains(&idx);
                let (psi_x, psi_y, psi_z, scale, vel_scale) = if is_positive {
                    (&psi_x_plus, &psi_y_plus, &psi_z_plus, scale_plus, vel_scale_plus)
                } else {
                    (&psi_x_minus, &psi_y_minus, &psi_z_minus, scale_minus, vel_scale_minus)
                };

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

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);

                // Zel'dovich velocities: v = D_dot × Ψ (proportional to displacement)
                let vx = psi_x[idx] * vel_scale;
                let vy = psi_y[idx] * vel_scale;
                let vz = psi_z[idx] * vel_scale;
                velocities.push(vx as f32);
                velocities.push(vy as f32);
                velocities.push(vz as f32);

                signs.push(if is_positive { 1i8 } else { -1i8 });
            }
        }
    }

    (positions, velocities, signs)
}

#[cfg(feature = "cuda")]
fn generate_displacement_field(n_grid: usize, dk: f64, half_n: usize, d_growth: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate Gaussian random field in Fourier space
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                if k < 1e-10 {
                    delta_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }

                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = pk.sqrt() * DELTA_RMS * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Enforce Hermitian symmetry
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..=half_n {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let iz_conj = if iz == 0 { 0 } else { n_grid - iz };
                let iy_conj = if iy == 0 { 0 } else { n_grid - iy };
                let ix_conj = if ix == 0 { 0 } else { n_grid - ix };
                let idx_conj = iz_conj * n_grid * n_grid + iy_conj * n_grid + ix_conj;

                if idx < idx_conj {
                    delta_k[idx_conj] = delta_k[idx].conj();
                }
            }
        }
    }

    // Compute displacement fields
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k2 = kx*kx + ky*ky + kz*kz;

                if k2 < 1e-20 { continue; }

                let minus_i = Complex::new(0.0, -1.0);
                psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
                psi_y_k[idx] = minus_i * ky * delta_k[idx] / k2;
                psi_z_k[idx] = minus_i * kz * delta_k[idx] / k2;
            }
        }
    }

    // Inverse FFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    (psi_x, psi_y, psi_z)
}

#[cfg(feature = "cuda")]
fn ifft_3d(data: &mut [Complex<f64>], ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;
    let mut buffer = vec![Complex::new(0.0, 0.0); n];

    // 3D IFFT via 1D transforms
    // X direction
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                buffer[ix] = data[iz * n * n + iy * n + ix];
            }
            ifft.process(&mut buffer);
            for ix in 0..n {
                data[iz * n * n + iy * n + ix] = buffer[ix];
            }
        }
    }

    // Y direction
    for iz in 0..n {
        for ix in 0..n {
            for iy in 0..n {
                buffer[iy] = data[iz * n * n + iy * n + ix];
            }
            ifft.process(&mut buffer);
            for iy in 0..n {
                data[iz * n * n + iy * n + ix] = buffer[iy];
            }
        }
    }

    // Z direction
    for iy in 0..n {
        for ix in 0..n {
            for iz in 0..n {
                buffer[iz] = data[iz * n * n + iy * n + ix];
            }
            ifft.process(&mut buffer);
            for iz in 0..n {
                data[iz * n * n + iy * n + ix] = buffer[iz];
            }
        }
    }

    // Extract real parts and normalize
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

#[cfg(feature = "cuda")]
fn find_max_displacement(psi_x: &[f64], psi_y: &[f64], psi_z: &[f64]) -> f64 {
    let mut max_disp = 0.0f64;
    for i in 0..psi_x.len() {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    max_disp
}

fn compute_metrics(
    pos: &[f32],
    vel: &[f32],
    signs: &[i8],
    box_size: f32,
) -> (f64, f64, f64, f64, f64) {
    let n = signs.len();
    let half_box = box_size / 2.0;

    let mut com_plus = [0.0f64; 3];
    let mut com_minus = [0.0f64; 3];
    let mut v2_plus = 0.0f64;
    let mut v2_minus = 0.0f64;
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;

    for i in 0..n {
        let px = pos[i * 3] as f64;
        let py = pos[i * 3 + 1] as f64;
        let pz = pos[i * 3 + 2] as f64;
        let vx = vel[i * 3] as f64;
        let vy = vel[i * 3 + 1] as f64;
        let vz = vel[i * 3 + 2] as f64;
        let v2 = vx * vx + vy * vy + vz * vz;

        if signs[i] > 0 {
            com_plus[0] += px;
            com_plus[1] += py;
            com_plus[2] += pz;
            v2_plus += v2;
            n_plus += 1;
        } else {
            com_minus[0] += px;
            com_minus[1] += py;
            com_minus[2] += pz;
            v2_minus += v2;
            n_minus += 1;
        }
    }

    com_plus[0] /= n_plus as f64;
    com_plus[1] /= n_plus as f64;
    com_plus[2] /= n_plus as f64;
    com_minus[0] /= n_minus as f64;
    com_minus[1] /= n_minus as f64;
    com_minus[2] /= n_minus as f64;

    let dx = com_plus[0] - com_minus[0];
    let dy = com_plus[1] - com_minus[1];
    let dz = com_plus[2] - com_minus[2];
    let d_com = (dx * dx + dy * dy + dz * dz).sqrt();
    let seg = d_com / (box_size as f64 / 2.0);

    let vrms_plus = (v2_plus / n_plus as f64).sqrt();
    let vrms_minus = (v2_minus / n_minus as f64).sqrt();

    // Density grid
    let n_cells = 32usize;
    let cell_size = box_size as f64 / n_cells as f64;
    let mut density_plus = vec![0.0f64; n_cells * n_cells * n_cells];
    let mut density_minus = vec![0.0f64; n_cells * n_cells * n_cells];

    for i in 0..n {
        let px = (pos[i * 3] as f64 + half_box as f64).max(0.0).min(box_size as f64 - 0.001);
        let py = (pos[i * 3 + 1] as f64 + half_box as f64).max(0.0).min(box_size as f64 - 0.001);
        let pz = (pos[i * 3 + 2] as f64 + half_box as f64).max(0.0).min(box_size as f64 - 0.001);
        let ix = (px / cell_size) as usize;
        let iy = (py / cell_size) as usize;
        let iz = (pz / cell_size) as usize;
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        if idx < density_plus.len() {
            if signs[i] > 0 {
                density_plus[idx] += 1.0;
            } else {
                density_minus[idx] += 1.0;
            }
        }
    }

    let rho_max_plus = density_plus.iter().cloned().fold(0.0, f64::max);

    let n_grid = density_plus.len();
    let mean_plus: f64 = density_plus.iter().sum::<f64>() / n_grid as f64;
    let mean_minus: f64 = density_minus.iter().sum::<f64>() / n_grid as f64;

    let mut cov = 0.0f64;
    let mut var_plus = 0.0f64;
    let mut var_minus = 0.0f64;

    for i in 0..n_grid {
        let dp = density_plus[i] - mean_plus;
        let dm = density_minus[i] - mean_minus;
        cov += dp * dm;
        var_plus += dp * dp;
        var_minus += dm * dm;
    }

    let corr = cov / (var_plus.sqrt() * var_minus.sqrt() + 1e-10);

    (seg, corr, vrms_plus, vrms_minus, rho_max_plus)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::Path, step: usize, z: f64) {
    let (positions, velocities, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let file = match File::create(path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    let _ = writer.write_all(&(n as u64).to_le_bytes());
    let _ = writer.write_all(&(1.0 / (1.0 + z)).to_le_bytes());
    let _ = writer.write_all(&(step as f64 * DT).to_le_bytes());

    for chunk in positions.chunks(3) {
        let _ = writer.write_all(&chunk[0].to_le_bytes());
        let _ = writer.write_all(&chunk[1].to_le_bytes());
        let _ = writer.write_all(&chunk[2].to_le_bytes());
    }

    for chunk in velocities.chunks(3) {
        let _ = writer.write_all(&chunk[0].to_le_bytes());
        let _ = writer.write_all(&chunk[1].to_le_bytes());
        let _ = writer.write_all(&chunk[2].to_le_bytes());
    }

    for &s in &signs {
        let _ = writer.write_all(&[s as u8]);
    }

    let _ = writer.flush();
}

fn compute_sha256(path: &str) -> std::io::Result<String> {
    let data = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}
