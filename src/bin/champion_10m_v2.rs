//! Champion 10M run v2 — Proper FFT Zeldovich ICs
//!
//! Parameters from scan_eta_lambda v4:
//! - eta = 0.87, lambda_0 = 1.0 Mpc
//! - P(z=0) = 0.70 target
//!
//! ICs: FFT-based Zeldovich with multi-mode P(k)
//! Fix for v1 catastrophic central collapse

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Champion parameters from scan v4
const ETA: f64 = 0.87;
const LAMBDA_0: f64 = 1.0;

// Simulation setup
const N_GRID: usize = 215;              // 215³ = 9,938,375 ≈ 10M
const BOX_SIZE: f64 = 1000.0;           // Mpc
const Z_INIT: f64 = 5.0;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;             // Mpc (scaled for 10M in 1000 Mpc)
const SEED: u64 = 42;
const DT: f64 = 0.01;

// P(k) truncation for multi-mode structure
const K_MIN: f64 = 2.0 * PI / 300.0;    // suppress lambda > 300 Mpc
const K_MAX: f64 = 2.0 * PI / 10.0;     // suppress lambda < 10 Mpc

// Power spectrum: P(k) ~ k^n_s / (1 + (k/k0)^4)
const N_S: f64 = 0.965;
const K0: f64 = 0.02;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Requires --features cuda");
        std::process::exit(1);
    }
    #[cfg(feature = "cuda")]
    run_champion();
}

#[cfg(feature = "cuda")]
fn run_champion() {
    println!("========================================================");
    println!("  CHAMPION 10M v2 — FFT Zeldovich ICs");
    println!("========================================================");
    println!("  eta = {:.2}, lambda_0 = {:.1} Mpc", ETA, LAMBDA_0);
    println!("  N = {}^3 = {}", N_GRID, N_GRID * N_GRID * N_GRID);
    println!("  Box = {} Mpc, z = {} -> 0", BOX_SIZE, Z_INIT);
    println!("  k_min = {:.4} (lambda_max = {:.0} Mpc)", K_MIN, 2.0 * PI / K_MIN);
    println!("  k_max = {:.4} (lambda_min = {:.0} Mpc)", K_MAX, 2.0 * PI / K_MAX);
    println!("========================================================");
    println!();

    let base_dir = std::path::Path::new("/app/output/champion_10m_v2");
    fs::create_dir_all(base_dir).ok();
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).ok();

    // Time series CSV
    let ts_path = base_dir.join("time_series.csv");
    let mut ts_file = BufWriter::new(File::create(&ts_path).unwrap());
    writeln!(ts_file, "step,z,a,purity,ke_ratio").unwrap();

    let start = Instant::now();

    // Generate proper FFT Zeldovich ICs
    println!("Generating FFT Zeldovich ICs...");
    let (pos_data, vel_data, signs_data, n_positive) = generate_fft_zeldovich_ics(SEED);
    let n_total = signs_data.len();
    let n_negative = n_total - n_positive;
    println!("  N+ = {}, N- = {}", n_positive, n_negative);

    // Convert to f32 for GPU
    let pos_f32: Vec<f32> = pos_data.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel_data.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs_data.iter().map(|&s| s as i8).collect();

    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(LAMBDA_0);

    let ke0 = sim.kinetic_energy().unwrap_or(1.0).max(1e-20);

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let snapshot_interval = 5;  // Every 5 steps for smooth animation
    let mut last_purity = 0.0;

    println!();
    println!("Starting evolution...");
    for step in 1..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };

        sim.set_current_z(z);
        sim.step_dkd(DT, h, dtau_per_dt).expect("Step failed");

        // Write time series every 10 steps
        if step % 10 == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let ke_ratio = ke / ke0;
            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4e}", step, z, a, purity, ke_ratio).unwrap();
        }

        // Snapshot every 5 steps
        if step % snapshot_interval == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            last_purity = purity;
            println!("  step {:4} | z={:.3} | P={:.4}", step, z, purity);
            save_snapshot(&sim, &snap_dir, step, z);
        } else if step % 100 == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            println!("  step {:4} | z={:.2} | P={:.4}", step, z, purity);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64() / 60.0;
    println!();
    println!("========================================================");
    println!("  CHAMPION 10M v2 COMPLETE");
    println!("========================================================");
    println!("  P(z=0) = {:.4}", last_purity);
    println!("  Time: {:.1} min", elapsed);
    println!("  Snapshots: {} frames", STEPS / snapshot_interval);
    println!("========================================================");
}

#[cfg(feature = "cuda")]
fn generate_fft_zeldovich_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("  Grid: {}^3 = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", BOX_SIZE);
    println!("  z_init = {}", Z_INIT);

    let dk = 2.0 * PI / BOX_SIZE;
    let half_n = N_GRID / 2;
    let spacing = BOX_SIZE / N_GRID as f64;
    let half_box = BOX_SIZE / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate delta(k) with P(k) truncation
    println!("  Generating Fourier modes...");
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

                // P(k) truncation window
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

    println!("  Modes kept: {} ({:.1}%)", n_modes_kept,
             100.0 * n_modes_kept as f64 / (n_modes_kept + n_modes_suppressed) as f64);

    // Enforce Hermitian symmetry for real IFFT
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

    // Compute displacement field psi(k) = -i * k * delta(k) / k^2
    println!("  Computing displacement field...");
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

    // Inverse FFT
    println!("  Performing inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    // Compute delta(x) for density-based sign assignment
    println!("  Computing density field...");
    let delta_real = ifft_3d(&mut delta_k, &ifft, N_GRID);

    let delta_mean: f64 = delta_real.iter().sum::<f64>() / n3 as f64;
    let delta_std: f64 = (delta_real.iter().map(|d| (d - delta_mean).powi(2)).sum::<f64>() / n3 as f64).sqrt();
    println!("  delta field: mean = {:.6e}, std = {:.6e}", delta_mean, delta_std);

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Max displacement: {:.6e} Mpc", max_disp);

    let target_disp = spacing * 0.3;
    let scale = target_disp / max_disp;
    println!("  Scaling: {:.4} -> target {:.4} Mpc ({:.1}% of cell)",
             scale, target_disp, 100.0 * target_disp / spacing);

    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;

    // Density-based sign assignment (m+ in overdense, m- in underdense)
    println!("  Assigning signs based on density...");
    let n_positive_target = (n3 as f64 / (1.0 + ETA)) as usize;

    let mut indices: Vec<usize> = (0..n3).collect();
    indices.sort_by(|&a, &b| delta_real[b].partial_cmp(&delta_real[a]).unwrap());

    let mut signs_ordered = vec![0i32; n3];
    for (rank, &idx) in indices.iter().enumerate() {
        signs_ordered[idx] = if rank < n_positive_target { 1 } else { -1 };
    }

    // Build particles
    println!("  Building particle arrays...");
    struct Particle { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, sign: i32 }
    let mut particles: Vec<Particle> = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // Apply displacement with periodic wrapping
                let x = ((x0 + psi_x[idx] * scale + half_box) % BOX_SIZE + BOX_SIZE) % BOX_SIZE - half_box;
                let y = ((y0 + psi_y[idx] * scale + half_box) % BOX_SIZE + BOX_SIZE) % BOX_SIZE - half_box;
                let z = ((z0 + psi_z[idx] * scale + half_box) % BOX_SIZE + BOX_SIZE) % BOX_SIZE - half_box;

                particles.push(Particle {
                    x, y, z,
                    vx: psi_x[idx] * vel_scale,
                    vy: psi_y[idx] * vel_scale,
                    vz: psi_z[idx] * vel_scale,
                    sign: signs_ordered[idx],
                });
            }
        }
    }

    // Shuffle to avoid memory layout bias
    particles.shuffle(&mut rng);

    // Separate positives first (GPU requirement)
    let pos_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign > 0).collect();
    let neg_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign < 0).collect();

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs = Vec::with_capacity(n3);
    let mut n_positive_final = 0usize;

    for p in pos_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        velocities.extend_from_slice(&[p.vx, p.vy, p.vz]);
        signs.push(1);
        n_positive_final += 1;
    }

    for p in neg_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        velocities.extend_from_slice(&[p.vx, p.vy, p.vz]);
        signs.push(-1);
    }

    println!("  Final: {} particles ({} + / {} -)", n3, n_positive_final, n3 - n_positive_final);

    (positions, velocities, signs, n_positive_final)
}

fn ifft_3d(data: &mut Vec<Complex<f64>>, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;

    // Z direction
    for iy in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|iz| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for iz in 0..n { data[iz * n * n + iy * n + ix] = slice[iz]; }
        }
    }

    // Y direction
    for iz in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|iy| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for iy in 0..n { data[iz * n * n + iy * n + ix] = slice[iy]; }
        }
    }

    // X direction
    for iz in 0..n {
        for iy in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|ix| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for ix in 0..n { data[iz * n * n + iy * n + ix] = slice[ix]; }
        }
    }

    // Extract real parts and normalize
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, z: f64) {
    use std::io::BufWriter;

    let (positions, _, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let snap_path = path.join(format!("snap_{:05}.bin", step));

    let file = match File::create(&snap_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(BOX_SIZE as f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

    for i in 0..n {
        let _ = writer.write_all(&positions[i*3].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+1].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+2].to_le_bytes());
        let _ = writer.write_all(&(signs[i] as i8).to_le_bytes());
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda");
    std::process::exit(1);
}
