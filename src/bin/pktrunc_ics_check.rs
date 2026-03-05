//! Quick check: Generate P(k) truncated ICs and write step 0 snapshot only
//! For verifying multi-scale structures before full 12M run

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

// Physical parameters — 12M
const N_GRID: usize = 229;
const L_BOX: f64 = 492.0;
const Z_INIT: f64 = 5.0;

// P(k) truncation — v2 fenêtre élargie
const K_MIN: f64 = 2.0 * PI / 200.0;   // suppress λ > 200 Mpc
const K_MAX: f64 = 2.0 * PI / 8.0;     // suppress λ < 8 Mpc

const ETA: f64 = 1.045;
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

fn generate_pktrunc_zeldovich_ics(seed: u64) -> (Vec<f64>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating P(k) truncated Zel'dovich ICs (v2)...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  k_min = 2π/{:.0} Mpc⁻¹  (suppress λ > {:.0} Mpc)",
             2.0 * PI / K_MIN, 2.0 * PI / K_MIN);
    println!("  k_max = 2π/{:.0} Mpc⁻¹  (suppress λ < {:.0} Mpc)",
             2.0 * PI / K_MAX, 2.0 * PI / K_MAX);

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    println!("  Generating Fourier modes with truncation...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 0.01;

    let mut n_modes_kept = 0usize;
    let mut n_modes_suppressed = 0usize;

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

                let window = if k < K_MIN || k > K_MAX {
                    n_modes_suppressed += 1;
                    0.0
                } else {
                    n_modes_kept += 1;
                    1.0
                };

                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4)) * window;
                let sigma_k = pk.sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    let pct_kept = 100.0 * n_modes_kept as f64 / (n_modes_kept + n_modes_suppressed) as f64;
    println!("  Modes kept: {} ({:.1}%)", n_modes_kept, pct_kept);
    println!("  Modes suppressed: {} ({:.1}%)", n_modes_suppressed, 100.0 - pct_kept);

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

    // Compute displacement field
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
                if k2 < 1e-20 { continue; }
                let minus_i = Complex::new(0.0, -1.0);
                psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
                psi_y_k[idx] = minus_i * ky * delta_k[idx] / k2;
                psi_z_k[idx] = minus_i * kz * delta_k[idx] / k2;
            }
        }
    }

    println!("  Performing inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    println!("  Computing density field δ(x)...");
    let delta_real = ifft_3d(&mut delta_k, &ifft, N_GRID);

    // Scale displacement
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    let target_disp = spacing * 0.3;
    let scale = target_disp / max_disp;
    println!("  Max displacement: {:.6e} Mpc, scale: {:.2e}", max_disp, scale);

    // Density-based sign assignment
    println!("  Assigning signs based on density...");
    let n_positive_target = (n3 as f64 / (1.0 + ETA)) as usize;
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.sort_by(|&a, &b| delta_real[b].partial_cmp(&delta_real[a]).unwrap());

    let mut signs_ordered = vec![0i32; n3];
    for (rank, &idx) in indices.iter().enumerate() {
        signs_ordered[idx] = if rank < n_positive_target { 1 } else { -1 };
    }

    // Build particles
    struct Particle { x: f64, y: f64, z: f64, sign: i32 }
    let mut particles: Vec<Particle> = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;
                particles.push(Particle {
                    x: x0 + psi_x[idx] * scale,
                    y: y0 + psi_y[idx] * scale,
                    z: z0 + psi_z[idx] * scale,
                    sign: signs_ordered[idx],
                });
            }
        }
    }

    println!("  Shuffling indices...");
    particles.shuffle(&mut rng);

    // Separate positives first
    let pos_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign > 0).collect();
    let neg_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign < 0).collect();

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut n_positive_final = 0usize;

    for p in pos_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        n_positive_final += 1;
    }
    for p in neg_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
    }

    println!("  Final: {} particles ({} + / {} -)", n3, n_positive_final, n3 - n_positive_final);

    // Verify correlation
    let idx_vec: Vec<f64> = (0..n3).map(|i| i as f64).collect();
    let z_vec: Vec<f64> = (0..n3).map(|i| positions[i * 3 + 2]).collect();
    let corr = pearson_correlation(&idx_vec, &z_vec);
    println!("  corr(idx, z) = {:.4}", corr);

    (positions, n_positive_final)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x < 1e-10 || var_y < 1e-10 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn ifft_3d(data: &mut Vec<Complex<f64>>, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;
    for iy in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|iz| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for iz in 0..n { data[iz * n * n + iy * n + ix] = slice[iz]; }
        }
    }
    for iz in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|iy| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for iy in 0..n { data[iz * n * n + iy * n + ix] = slice[iy]; }
        }
    }
    for iz in 0..n {
        for iy in 0..n {
            let base = iz * n * n + iy * n;
            let mut slice: Vec<Complex<f64>> = data[base..base+n].to_vec();
            ifft.process(&mut slice);
            for ix in 0..n { data[base + ix] = slice[ix]; }
        }
    }
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  P(k) Truncated ICs Check — v2 (k_min=2π/200, k_max=2π/8)");
    println!("═══════════════════════════════════════════════════════════\n");

    let output_dir = "/app/output/pktrunc_12m_v2_ics_check";
    fs::create_dir_all(output_dir).expect("Failed to create output dir");

    let start = Instant::now();
    let (positions, n_positive) = generate_pktrunc_zeldovich_ics(42);
    println!("\nIC generation took {:.1}s", start.elapsed().as_secs_f64());

    let n3 = N_GRID * N_GRID * N_GRID;

    // Write step 0 snapshot
    println!("\nWriting step 0 snapshot...");
    let snap_path = format!("{}/snap_000000.bin", output_dir);
    let file = File::create(&snap_path).unwrap();
    let mut writer = BufWriter::new(file);

    writer.write_all(&(n3 as u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();

    for i in 0..n3 {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let sign: f32 = if i < n_positive { 1.0 } else { -1.0 };
        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }

    println!("Saved: {}", snap_path);
    println!("\n→ Generate image to verify multi-scale structures");
}
