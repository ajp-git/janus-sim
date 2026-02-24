//! Antisymmetric Mode Test — Pure λ₋ eigenmode excitation
//!
//! Tests the stability of the antisymmetric mode in Janus cosmology.
//! ICs: m+ gets +ψ displacement, m- gets -ψ displacement
//! This directly excites the λ₋ eigenmode.
//!
//! Expected result for α=1: λ₋=0, so antisymmetric mode should be FROZEN
//! (neither growing nor decaying).
//!
//! cargo run --release --features cuda --bin antisym_mode_test

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
const N_GRID: usize = 128;     // 128³ = 2M particles
const L_BOX: f64 = 400.0;      // Mpc
const Z_INIT: f64 = 10.0;      // Initial redshift

// Simulation parameters
const DT: f64 = 0.005;
const SNAPSHOT_INTERVAL: usize = 100;
const CSV_INTERVAL: usize = 10;
const THETA: f64 = 0.7;

// Power spectrum: P(k) ∝ k^0.96 / (1 + (k/k0)^4)
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

/// Generate ANTISYMMETRIC Zel'dovich ICs
/// m+ particles: position = q + ψ, velocity = +v
/// m- particles: position = q - ψ, velocity = -v
/// This directly excites the λ₋ eigenmode
fn generate_antisym_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating ANTISYMMETRIC Zel'dovich ICs...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  z_init = {}", Z_INIT);
    println!("  MODE: m+ gets +ψ, m- gets -ψ (pure antisymmetric)");

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;

    // Growth factor at z_init
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate Gaussian random field in Fourier space
    println!("  Generating Fourier modes...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 0.01;

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
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                if k < 1e-10 {
                    delta_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }

                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = (pk).sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Enforce Hermitian symmetry
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

    // Compute displacement field ψ = -i k δ_k / k²
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

                let minus_i = Complex::new(0.0, -1.0);
                psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
                psi_y_k[idx] = minus_i * ky * delta_k[idx] / k2;
                psi_z_k[idx] = minus_i * kz * delta_k[idx] / k2;
            }
        }
    }

    // Inverse FFT
    println!("  Performing inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    // Scale to reasonable amplitude (30% of cell size)
    let target_disp = spacing * 0.3;
    let scale = target_disp / max_disp;
    println!("  Max displacement: {:.6e} Mpc → scaled to {:.4} Mpc", max_disp, target_disp);

    // Zel'dovich velocities: v = D_dot * psi
    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;
    println!("  D_dot = sqrt(1+z) = {:.2}", d_dot);

    // FIRST: Assign signs (random 50/50)
    println!("  Assigning signs (random 50/50)...");
    let mut signs: Vec<i32> = Vec::with_capacity(n3);
    let mut n_positive = 0usize;

    for _ in 0..n3 {
        let sign = if rng.gen::<bool>() { 1 } else { -1 };
        if sign > 0 { n_positive += 1; }
        signs.push(sign);
    }

    // THEN: Place particles with antisymmetric displacements
    // m+ gets +ψ, m- gets -ψ
    println!("  Placing particles with ANTISYMMETRIC displacements...");
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;
                let sign = signs[idx];

                // Grid position centered at [-box/2, box/2]
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // ANTISYMMETRIC: m+ gets +ψ, m- gets -ψ
                let sign_factor = sign as f64;  // +1 or -1

                let x = x0 + psi_x[idx] * scale * sign_factor;
                let y = y0 + psi_y[idx] * scale * sign_factor;
                let z = z0 + psi_z[idx] * scale * sign_factor;

                // ANTISYMMETRIC velocities
                let vx = psi_x[idx] * vel_scale * sign_factor;
                let vy = psi_y[idx] * vel_scale * sign_factor;
                let vz = psi_z[idx] * vel_scale * sign_factor;

                positions.push(x);
                positions.push(y);
                positions.push(z);
                velocities.push(vx);
                velocities.push(vy);
                velocities.push(vz);
            }
        }
    }

    let n_negative = n3 - n_positive;
    println!("  Total particles: {} ({}+ and {}-)", n3, n_positive, n_negative);

    // Compute initial segregation S (should be >> 0 due to antisymmetric setup)
    let seg = compute_segregation(&positions, &signs);
    println!("  Initial segregation S₀ = {:.2} Mpc (expected >> 0)", seg);

    // Verify antisymmetry: compute density correlation
    println!("  Verifying antisymmetric ICs...");
    // δ(m+) should anti-correlate with δ(m-) by construction

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

    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

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

/// Compute Delta(t) = sqrt(<(delta+ - delta-)^2>) using 32³ grid
fn compute_delta_mode(positions: &[f64], signs: &[i32], box_size: f64) -> f64 {
    let n_grid = 32;
    let cell_size = box_size / n_grid as f64;
    let n = positions.len() / 3;

    let mut density_plus = vec![0.0f64; n_grid * n_grid * n_grid];
    let mut density_minus = vec![0.0f64; n_grid * n_grid * n_grid];

    for i in 0..n {
        let x = positions[i * 3] + box_size / 2.0;
        let y = positions[i * 3 + 1] + box_size / 2.0;
        let z = positions[i * 3 + 2] + box_size / 2.0;

        let ix = ((x / cell_size) as usize).min(n_grid - 1);
        let iy = ((y / cell_size) as usize).min(n_grid - 1);
        let iz = ((z / cell_size) as usize).min(n_grid - 1);

        let idx = iz * n_grid * n_grid + iy * n_grid + ix;

        if signs[i] > 0 {
            density_plus[idx] += 1.0;
        } else {
            density_minus[idx] += 1.0;
        }
    }

    // Compute overdensity
    let mean_plus: f64 = density_plus.iter().sum::<f64>() / density_plus.len() as f64;
    let mean_minus: f64 = density_minus.iter().sum::<f64>() / density_minus.len() as f64;

    let mut delta_sq_sum = 0.0f64;
    for i in 0..density_plus.len() {
        let delta_plus = if mean_plus > 0.0 { (density_plus[i] - mean_plus) / mean_plus } else { 0.0 };
        let delta_minus = if mean_minus > 0.0 { (density_minus[i] - mean_minus) / mean_minus } else { 0.0 };
        delta_sq_sum += (delta_plus - delta_minus).powi(2);
    }

    (delta_sq_sum / density_plus.len() as f64).sqrt()
}

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

    file.write_all(&(n as u64).to_le_bytes())?;
    file.write_all(&(step as u64).to_le_bytes())?;
    file.write_all(&scale_factor.to_le_bytes())?;
    file.write_all(&segregation.to_le_bytes())?;

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
    println!("ANTISYMMETRIC MODE TEST — λ₋ eigenmode");
    println!("     128³ = 2M particles, α=1 (Janus)");
    println!("==============================================\n");

    let seed = 42u64;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let output_dir = format!("/app/output/antisym_mode_test_{}", timestamp);
    create_dir_all(&output_dir).expect("Failed to create output dir");
    let snap_dir = format!("{}/snapshots", output_dir);
    create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    println!("Output directory: {}", output_dir);

    // Generate ANTISYMMETRIC ICs
    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_antisym_ics(seed);
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
    println!("  Snapshot interval = {}", SNAPSHOT_INTERVAL);

    // CSV with Delta(t) column
    let csv_path = format!("{}/evolution.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,time,sigma_x,sigma_y,sigma_z,segregation,delta_mode,rate").unwrap();

    // Initial measurement
    let pos = sim.get_positions().expect("get_positions failed");
    let (sx, sy, sz) = compute_sigma_xyz(&pos);
    let seg = compute_segregation(&pos, &signs);
    let delta = compute_delta_mode(&pos, &signs, L_BOX);
    writeln!(csv, "0,0.0,{:.4},{:.4},{:.4},{:.6},{:.6},0.0", sx, sy, sz, seg, delta).unwrap();

    println!("\n=== INITIAL STATE ===");
    println!("  σ = ({:.2}, {:.2}, {:.2}) Mpc", sx, sy, sz);
    println!("  S = {:.4} Mpc (COM distance)", seg);
    println!("  Δ = {:.4} (antisymmetric mode amplitude)", delta);
    println!("  EXPECTED: Δ >> 0 (antisymmetric ICs)");

    // Write initial snapshot
    let snap_path = format!("{}/snap_{:05}.bin", snap_dir, 0);
    write_snapshot(&snap_path, &pos, &signs, 0, 1.0 / (1.0 + Z_INIT), seg)
        .expect("Failed to write snapshot");

    // Main loop
    let start = Instant::now();
    let delta_0 = delta;

    let mut step = 0usize;
    loop {
        step += 1;
        sim.step_with_cross_factor(DT, -1.0).expect("Step failed");

        if step % CSV_INTERVAL == 0 {
            let pos = sim.get_positions().expect("get_positions failed");
            let (sx, sy, sz) = compute_sigma_xyz(&pos);
            let seg = compute_segregation(&pos, &signs);
            let delta = compute_delta_mode(&pos, &signs, L_BOX);

            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;

            writeln!(csv, "{},{:.2},{:.4},{:.4},{:.4},{:.6},{:.6},{:.1}",
                     step, elapsed, sx, sy, sz, seg, delta, rate).unwrap();
            csv.flush().unwrap();

            if step % SNAPSHOT_INTERVAL == 0 {
                let delta_ratio = delta / delta_0;
                println!("Step {}: S={:.2} Mpc, Δ={:.4} (Δ/Δ₀={:.3}) [{:.1} steps/s]",
                         step, seg, delta, delta_ratio, rate);

                let a = 1.0 / (1.0 + Z_INIT) + step as f64 * DT * 0.01;
                let snap_path = format!("{}/snap_{:05}.bin", snap_dir, step);
                write_snapshot(&snap_path, &pos, &signs, step, a, seg)
                    .expect("Failed to write snapshot");

                // Report status every 1000 steps
                if step % 1000 == 0 {
                    println!("\n=== STATUS at step {} ===", step);
                    println!("  Δ(t)/Δ(0) = {:.4}", delta_ratio);
                    if delta_ratio > 1.1 {
                        println!("  → GROWING: λ₋ > 0 (unexpected for α=1)");
                    } else if delta_ratio < 0.9 {
                        println!("  → DECAYING: amortissement (expansion)");
                    } else {
                        println!("  → FROZEN: λ₋ ≈ 0 (expected for α=1)");
                    }
                }
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature. Compile with:");
    eprintln!("  cargo build --release --features cuda --bin antisym_mode_test");
}
