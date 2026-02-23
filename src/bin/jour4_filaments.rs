//! Jour 4 — Production run: Filament formation
//!
//! Grid: 128³ = 2M particles
//! Full P(k) spectrum Zel'dovich ICs (NO anti-correlation)
//! Random sign assignment to avoid initial spatial segregation
//! Janus α=1, θ=0.7, dt=0.005, 3000+ steps

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

// Physical parameters
// 128³ = 2M particles (validated performance)
const N_GRID: usize = 128;
const L_BOX: f64 = 400.0;      // Mpc
const Z_INIT: f64 = 10.0;      // Initial redshift

// Simulation parameters
const DT: f64 = 0.005;
const N_STEPS: usize = 3000;
const SNAPSHOT_INTERVAL: usize = 100;
const THETA: f64 = 0.7;  // Validated, accurate tree traversal

// Power spectrum: P(k) ∝ k^0.96 / (1 + (k/k0)^4)
const N_S: f64 = 0.96;         // Spectral index
const K0: f64 = 0.02;          // Turnover scale (Mpc⁻¹)

/// Generate Zel'dovich ICs with full P(k) spectrum (no anti-correlation)
/// Returns (positions, velocities, signs, n_positive)
fn generate_zeldovich_ics_full(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating Zel'dovich ICs with full P(k) spectrum...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  z_init = {}", Z_INIT);
    println!("  P(k) ∝ k^{} / (1 + (k/{})⁴)", N_S, K0);

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;

    // Growth factor at z_init (approximate for Janus)
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;  // Linear approximation

    // Generate Gaussian random field in Fourier space
    println!("  Generating Fourier modes...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Amplitude normalization (arbitrary, will be rescaled)
    let amplitude = 0.01;

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                // Wavenumbers (centered FFT convention)
                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - N_GRID as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - N_GRID as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - N_GRID as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                if k < 1e-10 {
                    delta_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }

                // Power spectrum P(k) ∝ k^n_s / (1 + (k/k0)^4)
                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = (pk).sqrt() * amplitude * d_growth;

                // Gaussian random field
                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Enforce Hermitian symmetry for real field
    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..=half_n {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;
                let iz_conj = if iz == 0 { 0 } else { N_GRID - iz };
                let iy_conj = if iy == 0 { 0 } else { N_GRID - iy };
                let ix_conj = if ix == 0 { 0 } else { N_GRID - ix };
                let idx_conj = iz_conj * N_GRID * N_GRID + iy_conj * N_GRID + ix_conj;

                if idx < idx_conj {
                    delta_k[idx_conj] = delta_k[idx].conj();
                }
            }
        }
    }

    // Compute displacement field ψ_x = -i k_x δ_k / k²
    println!("  Computing displacement fields...");
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - N_GRID as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - N_GRID as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - N_GRID as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k2 = kx*kx + ky*ky + kz*kz;

                if k2 < 1e-20 {
                    continue;
                }

                // ψ = -i k δ_k / k²
                let minus_i = Complex::new(0.0, -1.0);
                psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
                psi_y_k[idx] = minus_i * ky * delta_k[idx] / k2;
                psi_z_k[idx] = minus_i * kz * delta_k[idx] / k2;
            }
        }
    }

    // Inverse FFT to get displacement in real space
    println!("  Performing inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    // Process each dimension
    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Max displacement: {:.4} Mpc", max_disp);

    // Scale to reasonable amplitude (10% of cell size)
    let target_disp = spacing * 0.3;
    let scale = target_disp / max_disp;
    println!("  Scaling factor: {:.4} → target {:.4} Mpc", scale, target_disp);

    // Generate particle positions - SAME displacement for both populations
    // Signs assigned randomly to avoid initial spatial segregation
    println!("  Placing particles (no anti-correlation)...");
    let mut n_positive = 0usize;

    let mut positions = Vec::with_capacity(n3 * 3);
    let velocities = vec![0.0f64; n3 * 3];  // Start at rest
    let mut signs = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                // Grid position centered at [-box/2, box/2] (FIX-008)
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // SAME displacement for all particles (no anti-correlation)
                let x = x0 + psi_x[idx] * scale;
                let y = y0 + psi_y[idx] * scale;
                let z = z0 + psi_z[idx] * scale;

                // Random sign assignment (50/50)
                let sign = if rng.gen::<bool>() { 1 } else { -1 };
                if sign > 0 { n_positive += 1; }

                positions.push(x);
                positions.push(y);
                positions.push(z);
                signs.push(sign);
            }
        }
    }

    let n_negative = n3 - n_positive;
    println!("  Total particles: {} ({}+ and {}-)", n3, n_positive, n_negative);

    (positions, velocities, signs, n_positive)
}

/// 3D inverse FFT
fn ifft_3d(data: &mut Vec<Complex<f64>>, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;

    // Process along z
    for iy in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n)
                .map(|iz| data[iz * n * n + iy * n + ix])
                .collect();
            ifft.process(&mut slice);
            for iz in 0..n {
                data[iz * n * n + iy * n + ix] = slice[iz];
            }
        }
    }

    // Process along y
    for iz in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n)
                .map(|iy| data[iz * n * n + iy * n + ix])
                .collect();
            ifft.process(&mut slice);
            for iy in 0..n {
                data[iz * n * n + iy * n + ix] = slice[iy];
            }
        }
    }

    // Process along x
    for iz in 0..n {
        for iy in 0..n {
            let base = iz * n * n + iy * n;
            let mut slice: Vec<Complex<f64>> = data[base..base+n].to_vec();
            ifft.process(&mut slice);
            for ix in 0..n {
                data[base + ix] = slice[ix];
            }
        }
    }

    // Extract real part and normalize
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

/// Compute σ_x, σ_y, σ_z separately
fn compute_sigma_xyz(positions: &[f64]) -> (f64, f64, f64) {
    let n = positions.len() / 3;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_z = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut sum_z2 = 0.0f64;

    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        sum_x += x;
        sum_y += y;
        sum_z += z;
        sum_x2 += x * x;
        sum_y2 += y * y;
        sum_z2 += z * z;
    }

    let nf = n as f64;
    let var_x = sum_x2 / nf - (sum_x / nf).powi(2);
    let var_y = sum_y2 / nf - (sum_y / nf).powi(2);
    let var_z = sum_z2 / nf - (sum_z / nf).powi(2);

    (var_x.sqrt(), var_y.sqrt(), var_z.sqrt())
}

/// Compute segregation metric
fn compute_segregation(positions: &[f64], signs: &[i32]) -> f64 {
    let n = positions.len() / 3;
    let n_positive = signs.iter().filter(|&&s| s > 0).count();

    let mut sum_pos = [0.0f64; 3];
    let mut sum_neg = [0.0f64; 3];

    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        if signs[i] > 0 {
            sum_pos[0] += x;
            sum_pos[1] += y;
            sum_pos[2] += z;
        } else {
            sum_neg[0] += x;
            sum_neg[1] += y;
            sum_neg[2] += z;
        }
    }

    let n_pos = n_positive as f64;
    let n_neg = (n - n_positive) as f64;

    let com_pos = [sum_pos[0]/n_pos, sum_pos[1]/n_pos, sum_pos[2]/n_pos];
    let com_neg = [sum_neg[0]/n_neg, sum_neg[1]/n_neg, sum_neg[2]/n_neg];

    let dx = com_pos[0] - com_neg[0];
    let dy = com_pos[1] - com_neg[1];
    let dz = com_pos[2] - com_neg[2];

    (dx*dx + dy*dy + dz*dz).sqrt()
}

/// Write lightweight snapshot (positions + signs only)
fn write_snapshot(
    path: &str,
    positions: &[f64],
    signs: &[i32],
    step: usize,
    scale_factor: f64,
    segregation: f64,
) -> std::io::Result<()> {
    let n = positions.len() / 3;
    let mut file = BufWriter::new(File::create(path)?);

    // Header: 32 bytes (FIX-003)
    file.write_all(&(n as u64).to_le_bytes())?;
    file.write_all(&(step as u64).to_le_bytes())?;
    file.write_all(&scale_factor.to_le_bytes())?;
    file.write_all(&segregation.to_le_bytes())?;

    // Interleaved data: x, y, z (f32), sign (i8) per particle (FIX-002)
    for i in 0..n {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let s = signs[i] as i8;

        file.write_all(&x.to_le_bytes())?;
        file.write_all(&y.to_le_bytes())?;
        file.write_all(&z.to_le_bytes())?;
        file.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn main() {
    println!("==============================================");
    println!("Jour 4 — Production Run: Filament Formation");
    println!("              256³ = 16.7M particles          ");
    println!("==============================================\n");

    let seed = 42u64;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Create output directory
    let output_dir = format!("/app/output/jour4_{}", timestamp);
    create_dir_all(&output_dir).expect("Failed to create output dir");
    let snap_dir = format!("{}/snapshots", output_dir);
    create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    println!("Output directory: {}", output_dir);

    // Generate ICs (no anti-correlation - FIX for initial segregation bias)
    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_zeldovich_ics_full(seed);
    println!("IC generation took {:.1}s\n", start_ic.elapsed().as_secs_f64());

    let n3 = N_GRID * N_GRID * N_GRID;
    let n_negative = n3 - n_positive;

    // Initialize simulation
    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        L_BOX,
        positions,
        velocities,
        signs.clone(),
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  N_steps = {}", N_STEPS);
    println!("  Snapshot interval = {}", SNAPSHOT_INTERVAL);

    // Open CSV log
    let csv_path = format!("{}/evolution.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,time,sigma_x,sigma_y,sigma_z,segregation,rate").unwrap();

    // Initial measurement
    let pos = sim.get_positions().expect("get_positions failed");
    let (sx, sy, sz) = compute_sigma_xyz(&pos);
    let seg = compute_segregation(&pos, &signs);
    writeln!(csv, "0,0.0,{:.4},{:.4},{:.4},{:.6},0.0", sx, sy, sz, seg).unwrap();
    println!("\nStep 0: σ = ({:.2}, {:.2}, {:.2}) Mpc, S = {:.6}", sx, sy, sz, seg);

    // Write initial snapshot
    let snap_path = format!("{}/snap_{:05}.bin", snap_dir, 0);
    write_snapshot(&snap_path, &pos, &signs, 0, 1.0 / (1.0 + Z_INIT), seg)
        .expect("Failed to write snapshot");

    // Main loop
    let start = Instant::now();

    for step in 1..=N_STEPS {
        sim.step_with_cross_factor(DT, -1.0).expect("Step failed");

        if step % SNAPSHOT_INTERVAL == 0 {
            let pos = sim.get_positions().expect("get_positions failed");
            let (sx, sy, sz) = compute_sigma_xyz(&pos);
            let seg = compute_segregation(&pos, &signs);

            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;

            writeln!(csv, "{},{:.2},{:.4},{:.4},{:.4},{:.6},{:.1}",
                     step, elapsed, sx, sy, sz, seg, rate).unwrap();

            println!("Step {}: σ = ({:.2}, {:.2}, {:.2}) Mpc, S = {:.6} ({:.1} steps/s)",
                     step, sx, sy, sz, seg, rate);

            // Write snapshot
            let a = 1.0 / (1.0 + Z_INIT) + step as f64 * DT * 0.01;  // Approximate
            let snap_path = format!("{}/snap_{:05}.bin", snap_dir, step);
            write_snapshot(&snap_path, &pos, &signs, step, a, seg)
                .expect("Failed to write snapshot");
        }

        // Progress update every 500 steps
        if step % 500 == 0 && step % SNAPSHOT_INTERVAL != 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            let eta = (N_STEPS - step) as f64 / rate;
            println!("  ... step {} ({:.1} steps/s, ETA {:.0}s)", step, rate, eta);
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    println!("\n=== COMPLETED ===");
    println!("Total time: {:.1}s ({:.1} steps/s)", total_time, N_STEPS as f64 / total_time);
    println!("CSV: {}", csv_path);
    println!("Snapshots: {}/snap_*.bin", snap_dir);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature. Compile with:");
    eprintln!("  cargo build --release --features cuda --bin jour4_filaments");
}
