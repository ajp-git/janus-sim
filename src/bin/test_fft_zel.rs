//! Test FFT Zel'dovich — Debug IC generation only
//! Quick test to identify where displacement becomes zero

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

const N_GRID: usize = 128;  // Same as production
const L_BOX: f64 = 400.0;
const Z_INIT: f64 = 10.0;
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

fn main() {
    println!("=== FFT Zel'dovich Debug Test ===\n");

    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(42);

    println!("Grid: {}³ = {} particles", N_GRID, n3);
    println!("Box: {} Mpc, z_init = {}", L_BOX, Z_INIT);

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;
    println!("a_init = {:.4}, d_growth = {:.4}", a_init, d_growth);

    // Generate Gaussian random field
    println!("\n--- Step 1: Generate delta_k ---");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 0.01;

    let mut count_nonzero = 0usize;
    let mut sum_pk = 0.0f64;

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
                let sigma_k = pk.sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);

                if delta_k[idx].norm() > 1e-20 {
                    count_nonzero += 1;
                    sum_pk += pk;
                }
            }
        }
    }

    let max_delta_k = delta_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("  Non-zero modes: {}/{}", count_nonzero, n3);
    println!("  max(|delta_k|) = {:.6e}", max_delta_k);
    println!("  mean P(k) = {:.6e}", sum_pk / count_nonzero as f64);

    // Hermitian symmetry
    println!("\n--- Step 2: Enforce Hermitian symmetry ---");
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

    let max_delta_k_after = delta_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("  max(|delta_k|) after = {:.6e}", max_delta_k_after);

    // Compute psi_k
    println!("\n--- Step 3: Compute psi_k = -i k delta_k / k² ---");
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

    let max_psi_x_k = psi_x_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    let max_psi_y_k = psi_y_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    let max_psi_z_k = psi_z_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("  max(|psi_x_k|) = {:.6e}", max_psi_x_k);
    println!("  max(|psi_y_k|) = {:.6e}", max_psi_y_k);
    println!("  max(|psi_z_k|) = {:.6e}", max_psi_z_k);

    // IFFT
    println!("\n--- Step 4: Apply IFFT ---");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    let max_psi_x = psi_x.iter().cloned().fold(0.0f64, |a, b| a.abs().max(b.abs()));
    let max_psi_y = psi_y.iter().cloned().fold(0.0f64, |a, b| a.abs().max(b.abs()));
    let max_psi_z = psi_z.iter().cloned().fold(0.0f64, |a, b| a.abs().max(b.abs()));
    println!("  max(|psi_x|) = {:.6e} Mpc", max_psi_x);
    println!("  max(|psi_y|) = {:.6e} Mpc", max_psi_y);
    println!("  max(|psi_z|) = {:.6e} Mpc", max_psi_z);

    // Total displacement
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("\n  Max total displacement: {:.6e} Mpc", max_disp);

    let spacing = L_BOX / N_GRID as f64;
    println!("  Cell spacing: {:.2} Mpc", spacing);
    println!("  Displacement / spacing: {:.2}%", max_disp / spacing * 100.0);

    // Test scaling like jour4_filaments.rs
    println!("\n--- Step 5: Test scaling (like jour4) ---");
    let target_disp = spacing * 0.3;  // 30% of cell
    let scale = if max_disp > 1e-20 { target_disp / max_disp } else { 0.0 };
    println!("  target_disp = {:.4} Mpc", target_disp);
    println!("  scale = {:.6e}", scale);

    // Verify scaled displacement
    let mut max_scaled_disp = 0.0f64;
    let mut sum_scaled_x = 0.0f64;
    let mut sum_scaled_x2 = 0.0f64;

    for i in 0..n3 {
        let dx = psi_x[i] * scale;
        let dy = psi_y[i] * scale;
        let dz = psi_z[i] * scale;
        let d = (dx*dx + dy*dy + dz*dz).sqrt();
        if d > max_scaled_disp { max_scaled_disp = d; }
        sum_scaled_x += dx;
        sum_scaled_x2 += dx * dx;
    }

    let mean_dx = sum_scaled_x / n3 as f64;
    let var_dx = sum_scaled_x2 / n3 as f64 - mean_dx * mean_dx;
    let sigma_dx = var_dx.sqrt();

    println!("  max scaled displacement = {:.4} Mpc", max_scaled_disp);
    println!("  mean(dx) = {:.6e} (should be ~0)", mean_dx);
    println!("  sigma(dx) = {:.4} Mpc", sigma_dx);
    println!("  sigma(dx) / spacing = {:.2}%", sigma_dx / spacing * 100.0);

    // Diagnosis
    println!("\n=== DIAGNOSIS ===");
    if max_disp < 1e-20 {
        println!("BUG: Displacement is essentially zero!");
        if max_delta_k < 1e-10 {
            println!("  -> Problem in delta_k generation");
        } else if max_psi_x_k < 1e-10 {
            println!("  -> Problem in psi_k calculation (gradient)");
        } else {
            println!("  -> Problem in IFFT");
        }
    } else if scale.is_infinite() || scale.is_nan() {
        println!("BUG: Scale factor is inf/nan!");
    } else if max_scaled_disp < spacing * 0.01 {
        println!("WARNING: Scaled displacement still small ({:.4}% of cell)", max_scaled_disp / spacing * 100.0);
    } else {
        println!("OK: Scaled displacement = {:.2}% of cell", max_scaled_disp / spacing * 100.0);
        println!("OK: σ(dx) = {:.2}% of spacing", sigma_dx / spacing * 100.0);
    }
}

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
