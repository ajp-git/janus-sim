//! Grid Exploration for Janus ICs (EXPLORATION_GRID.md)
//!
//! 6 Cases:
//! A - Uniform random positions, random 50/50 signs, NO Zel'dovich
//! B - Grid + Zel'dovich, density-based signs, amplitude 0.3×spacing
//! C - Grid + Zel'dovich, density-based signs, amplitude 1.0×spacing
//! D - Grid + Zel'dovich, density-based signs, amplitude 2.0×spacing
//! E - Grid + Zel'dovich ±ψ by sign, random 50/50 signs, amplitude 0.3×spacing
//! F - Grid + Zel'dovich ±ψ by sign, random 50/50 signs, amplitude 1.0×spacing

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::env;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Common parameters from EXPLORATION_GRID.md
const N_PARTICLES: usize = 100_000;
const BOX_SIZE: f64 = 100.0;           // Mpc
const Z_INIT: f64 = 5.0;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 2000;       // z=5 → z=0
const SOFTENING: f64 = 0.65;           // Mpc
const SEED: u64 = 42;

// P(k) spectrum
const N_S: f64 = 0.96;
const K_PEAK: f64 = 0.02;

// Output intervals
const RENDER_INTERVAL: usize = 100;    // Every 100 steps (20 render files)
const CSV_INTERVAL: usize = 1;

#[derive(Clone, Copy, Debug)]
enum IcCase {
    A,  // Uniform random (control)
    B,  // Density-based, 0.3×spacing
    C,  // Density-based, 1.0×spacing
    D,  // Density-based, 2.0×spacing
    E,  // ±ψ opposed, 0.3×spacing
    F,  // ±ψ opposed, 1.0×spacing
}

impl IcCase {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "A" => Some(IcCase::A),
            "B" => Some(IcCase::B),
            "C" => Some(IcCase::C),
            "D" => Some(IcCase::D),
            "E" => Some(IcCase::E),
            "F" => Some(IcCase::F),
            _ => None,
        }
    }

    fn amplitude_factor(&self) -> f64 {
        match self {
            IcCase::A => 0.0,  // No displacement
            IcCase::B | IcCase::E => 0.3,
            IcCase::C | IcCase::F => 1.0,
            IcCase::D => 2.0,
        }
    }

    fn description(&self) -> &'static str {
        match self {
            IcCase::A => "Uniform random (CONTROL)",
            IcCase::B => "Density-based, 0.3×spacing",
            IcCase::C => "Density-based, 1.0×spacing",
            IcCase::D => "Density-based, 2.0×spacing",
            IcCase::E => "±ψ opposed, 0.3×spacing",
            IcCase::F => "±ψ opposed, 1.0×spacing",
        }
    }

    fn ic_type(&self) -> &'static str {
        match self {
            IcCase::A => "uniform_random",
            IcCase::B | IcCase::C | IcCase::D => "density_based",
            IcCase::E | IcCase::F => "pm_opposed",
        }
    }
}

/// Generate ICs for Case A: uniform random positions, random signs
fn generate_case_a(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let half_box = BOX_SIZE / 2.0;

    println!("Generating Case A ICs: uniform random positions, 50/50 signs");

    let mut positions = Vec::with_capacity(N_PARTICLES * 3);
    let mut velocities = Vec::with_capacity(N_PARTICLES * 3);
    let mut signs = Vec::with_capacity(N_PARTICLES);

    let virial_velocity = ((N_PARTICLES as f64) / BOX_SIZE).sqrt() * 0.3;

    for _ in 0..N_PARTICLES {
        // Random position in [-box/2, +box/2]
        positions.push(rng.gen::<f64>() * BOX_SIZE - half_box);
        positions.push(rng.gen::<f64>() * BOX_SIZE - half_box);
        positions.push(rng.gen::<f64>() * BOX_SIZE - half_box);

        // Random 50/50 sign
        signs.push(if rng.gen::<bool>() { 1 } else { -1 });

        // Random velocities
        velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
        velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
        velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
    }

    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    println!("  N+ = {}, N- = {}", n_positive, N_PARTICLES - n_positive);

    (positions, velocities, signs, n_positive)
}

/// Generate ICs for Cases B, C, D: density-based sign assignment
fn generate_density_based(seed: u64, amplitude_factor: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n_side = (N_PARTICLES as f64).cbrt().round() as usize;
    let n_grid = n_side * n_side * n_side;
    let spacing = BOX_SIZE / n_side as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating density-based ICs: amplitude = {:.1}×spacing", amplitude_factor);
    println!("  Grid: {}³ = {} particles", n_side, n_grid);
    println!("  Spacing: {:.3} Mpc", spacing);

    // Generate density field via FFT
    let dk = 2.0 * PI / BOX_SIZE;
    let half_n = n_side / 2;
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 1.0;

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_side as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_side as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_side as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                if k < 1e-10 {
                    continue;
                }

                let pk = k.powf(N_S) / (1.0 + (k / K_PEAK).powi(4));
                let sigma_k = (pk).sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Hermitian symmetry
    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..=half_n {
                let idx = iz * n_side * n_side + iy * n_side + ix;
                let iz_conj = if iz == 0 { 0 } else { n_side - iz };
                let iy_conj = if iy == 0 { 0 } else { n_side - iy };
                let ix_conj = if ix == 0 { 0 } else { n_side - ix };
                let idx_conj = iz_conj * n_side * n_side + iy_conj * n_side + ix_conj;

                if idx < idx_conj {
                    delta_k[idx_conj] = delta_k[idx].conj();
                }
            }
        }
    }

    // Compute displacement ψ = -i k δ_k / k²
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_side as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_side as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_side as i32 };

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
    let ifft = planner.plan_fft_inverse(n_side);

    let mut delta_k_copy = delta_k.clone();
    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_side);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_side);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_side);
    let delta_r = ifft_3d(&mut delta_k_copy, &ifft, n_side);

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n_grid {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    let target_disp = spacing * amplitude_factor;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Max raw disp: {:.4e}, target: {:.4} Mpc, scale: {:.4}", max_disp, target_disp, scale);

    let delta_min = delta_r.iter().cloned().fold(f64::INFINITY, f64::min);
    let delta_max = delta_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Density field: δ_min={:.4}, δ_max={:.4}", delta_min, delta_max);

    // Generate positions
    let mut positions = Vec::with_capacity(n_grid * 3);
    let mut velocities = Vec::with_capacity(n_grid * 3);
    let mut signs = Vec::with_capacity(n_grid);

    let jitter_amplitude = 0.1 * spacing;
    let virial_velocity = ((n_grid as f64) / BOX_SIZE).sqrt() * 0.3;

    let wrap = |mut x: f64| {
        while x > half_box { x -= BOX_SIZE; }
        while x < -half_box { x += BOX_SIZE; }
        x
    };

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                let jitter_x = (rng.gen::<f64>() - 0.5) * jitter_amplitude;
                let jitter_y = (rng.gen::<f64>() - 0.5) * jitter_amplitude;
                let jitter_z = (rng.gen::<f64>() - 0.5) * jitter_amplitude;

                let x0 = (ix as f64 + 0.5) * spacing - half_box + jitter_x;
                let y0 = (iy as f64 + 0.5) * spacing - half_box + jitter_y;
                let z0 = (iz as f64 + 0.5) * spacing - half_box + jitter_z;

                let dx = psi_x[idx] * scale;
                let dy = psi_y[idx] * scale;
                let dz = psi_z[idx] * scale;

                positions.push(wrap(x0 + dx));
                positions.push(wrap(y0 + dy));
                positions.push(wrap(z0 + dz));

                // Sign from local density
                signs.push(if delta_r[idx] > 0.0 { 1 } else { -1 });

                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
            }
        }
    }

    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    println!("  N+ = {}, N- = {}", n_positive, n_grid - n_positive);

    (positions, velocities, signs, n_positive)
}

/// Generate ICs for Cases E, F: ±ψ opposed displacement, random signs
fn generate_pm_opposed(seed: u64, amplitude_factor: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n_side = (N_PARTICLES as f64).cbrt().round() as usize;
    let n_grid = n_side * n_side * n_side;
    let spacing = BOX_SIZE / n_side as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating ±ψ opposed ICs: amplitude = {:.1}×spacing", amplitude_factor);
    println!("  Grid: {}³ = {} particles", n_side, n_grid);
    println!("  Signs: random 50/50, displacement ±ψ by sign");

    // Generate density field via FFT (same as density-based)
    let dk = 2.0 * PI / BOX_SIZE;
    let half_n = n_side / 2;
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 1.0;

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_side as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_side as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_side as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                if k < 1e-10 { continue; }

                let pk = k.powf(N_S) / (1.0 + (k / K_PEAK).powi(4));
                let sigma_k = (pk).sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Hermitian symmetry
    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..=half_n {
                let idx = iz * n_side * n_side + iy * n_side + ix;
                let iz_conj = if iz == 0 { 0 } else { n_side - iz };
                let iy_conj = if iy == 0 { 0 } else { n_side - iy };
                let ix_conj = if ix == 0 { 0 } else { n_side - ix };
                let idx_conj = iz_conj * n_side * n_side + iy_conj * n_side + ix_conj;

                if idx < idx_conj {
                    delta_k[idx_conj] = delta_k[idx].conj();
                }
            }
        }
    }

    // Compute displacement ψ
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_side as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_side as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_side as i32 };

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
    let ifft = planner.plan_fft_inverse(n_side);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_side);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_side);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_side);

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n_grid {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    let target_disp = spacing * amplitude_factor;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Max raw disp: {:.4e}, target: {:.4} Mpc, scale: {:.4}", max_disp, target_disp, scale);

    // Generate positions with RANDOM signs and ±ψ displacement
    let mut positions = Vec::with_capacity(n_grid * 3);
    let mut velocities = Vec::with_capacity(n_grid * 3);
    let mut signs = Vec::with_capacity(n_grid);

    let jitter_amplitude = 0.1 * spacing;
    let virial_velocity = ((n_grid as f64) / BOX_SIZE).sqrt() * 0.3;

    let wrap = |mut x: f64| {
        while x > half_box { x -= BOX_SIZE; }
        while x < -half_box { x += BOX_SIZE; }
        x
    };

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                let jitter_x = (rng.gen::<f64>() - 0.5) * jitter_amplitude;
                let jitter_y = (rng.gen::<f64>() - 0.5) * jitter_amplitude;
                let jitter_z = (rng.gen::<f64>() - 0.5) * jitter_amplitude;

                let x0 = (ix as f64 + 0.5) * spacing - half_box + jitter_x;
                let y0 = (iy as f64 + 0.5) * spacing - half_box + jitter_y;
                let z0 = (iz as f64 + 0.5) * spacing - half_box + jitter_z;

                // Random sign FIRST (50/50)
                let sign: i32 = if rng.gen::<bool>() { 1 } else { -1 };
                signs.push(sign);

                // Displacement ±ψ according to sign
                // + particles get +ψ, - particles get -ψ
                let sign_factor = sign as f64;
                let dx = psi_x[idx] * scale * sign_factor;
                let dy = psi_y[idx] * scale * sign_factor;
                let dz = psi_z[idx] * scale * sign_factor;

                positions.push(wrap(x0 + dx));
                positions.push(wrap(y0 + dy));
                positions.push(wrap(z0 + dz));

                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
            }
        }
    }

    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    println!("  N+ = {}, N- = {}", n_positive, n_grid - n_positive);

    (positions, velocities, signs, n_positive)
}

/// 3D inverse FFT
fn ifft_3d(data: &mut Vec<Complex<f64>>, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;

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

/// Compute COMs for + and - populations
fn compute_coms(positions: &[f64], signs: &[i32]) -> ([f64; 3], [f64; 3]) {
    let n = positions.len() / 3;
    let mut sum_pos = [0.0f64; 3];
    let mut sum_neg = [0.0f64; 3];
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;

    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        if signs[i] > 0 {
            sum_pos[0] += x;
            sum_pos[1] += y;
            sum_pos[2] += z;
            n_pos += 1;
        } else {
            sum_neg[0] += x;
            sum_neg[1] += y;
            sum_neg[2] += z;
            n_neg += 1;
        }
    }

    let com_pos = [sum_pos[0]/n_pos as f64, sum_pos[1]/n_pos as f64, sum_pos[2]/n_pos as f64];
    let com_neg = [sum_neg[0]/n_neg as f64, sum_neg[1]/n_neg as f64, sum_neg[2]/n_neg as f64];

    (com_pos, com_neg)
}

fn compute_segregation(positions: &[f64], signs: &[i32]) -> f64 {
    let (com_pos, com_neg) = compute_coms(positions, signs);
    let dx = com_pos[0] - com_neg[0];
    let dy = com_pos[1] - com_neg[1];
    let dz = com_pos[2] - com_neg[2];
    (dx*dx + dy*dy + dz*dz).sqrt() / BOX_SIZE
}

fn write_render_data(
    path: &str,
    positions: &[f64],
    signs: &[i32],
    step: usize,
    box_size: f64,
    seg: f64,
    ke_ratio: f64,
    redshift: f64,
) -> std::io::Result<()> {
    let n = positions.len() / 3;
    let mut file = BufWriter::new(File::create(path)?);

    file.write_all(&(step as u32).to_le_bytes())?;
    file.write_all(&box_size.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    file.write_all(&ke_ratio.to_le_bytes())?;
    file.write_all(&redshift.to_le_bytes())?;
    file.write_all(&(n as u32).to_le_bytes())?;

    for i in 0..n {
        file.write_all(&(positions[i * 3] as f32).to_le_bytes())?;
        file.write_all(&(positions[i * 3 + 1] as f32).to_le_bytes())?;
        file.write_all(&(positions[i * 3 + 2] as f32).to_le_bytes())?;
    }

    for i in 0..n {
        file.write_all(&(signs[i] as i8).to_le_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <CASE>", args[0]);
        eprintln!("  CASE: A, B, C, D, E, or F");
        eprintln!("\nCases:");
        eprintln!("  A - Uniform random (control)");
        eprintln!("  B - Density-based, 0.3×spacing");
        eprintln!("  C - Density-based, 1.0×spacing");
        eprintln!("  D - Density-based, 2.0×spacing");
        eprintln!("  E - ±ψ opposed, 0.3×spacing");
        eprintln!("  F - ±ψ opposed, 1.0×spacing");
        std::process::exit(1);
    }

    let case = match IcCase::from_str(&args[1]) {
        Some(c) => c,
        None => {
            eprintln!("Invalid case: {}. Use A, B, C, D, E, or F.", args[1]);
            std::process::exit(1);
        }
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  Grid Exploration — Case {:?}: {}", case, case.description());
    println!("═══════════════════════════════════════════════════════════\n");

    // Create output directory
    let output_dir = format!("/app/output/grid_{:?}_100k", case);
    create_dir_all(&output_dir).expect("Failed to create output dir");
    let render_dir = format!("{}/render_data", output_dir);
    create_dir_all(&render_dir).expect("Failed to create render_data dir");

    println!("Output: {}", output_dir);
    println!("Parameters:");
    println!("  N = {}", N_PARTICLES);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  θ = {}", THETA);
    println!("  softening = {} Mpc", SOFTENING);
    println!("  dt = {}", DT);
    println!("  steps = {}", TOTAL_STEPS);
    println!("  seed = {}\n", SEED);

    // Generate ICs based on case
    let (positions, velocities, signs, n_positive) = match case {
        IcCase::A => generate_case_a(SEED),
        IcCase::B | IcCase::C | IcCase::D => generate_density_based(SEED, case.amplitude_factor()),
        IcCase::E | IcCase::F => generate_pm_opposed(SEED, case.amplitude_factor()),
    };

    let n_total = signs.len();
    let n_negative = n_total - n_positive;

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        BOX_SIZE,
        positions.clone(),
        velocities,
        signs.clone(),
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);

    // Virialization
    println!("\nVirializing with PE_binding method...");
    let n_sample = (n_total / 100).max(1000).min(10000);
    sim.virialize_sampled(n_sample).expect("virialize_sampled failed");
    println!("  ✓ Virialization complete");

    // Setup cosmology
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    // dtau_per_step: for advancing tau each step
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    // dtau_per_dt: FIXED convention from February - 10000 steps cover z=5→0
    // This is what the kick kernel expects for Hubble friction
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * DT);
    println!("  dtau_per_step = {:.6}", dtau_per_step);
    println!("  dtau_per_dt = {:.6} (February convention)", dtau_per_dt);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke_ratio,seg,step_ms").unwrap();

    // Initial state
    let pos = sim.get_positions().expect("get_positions failed");
    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = compute_segregation(&pos, &signs);

    writeln!(csv, "0,{:.4},{:.6},{:.6},0", Z_INIT, 1.0, seg_0).unwrap();

    let render_path = format!("{}/step_{:06}.bin", render_dir, 0);
    write_render_data(&render_path, &pos, &signs, 0, BOX_SIZE, seg_0, 1.0, Z_INIT)
        .expect("Failed to write render_data");

    println!("\n══════════════════════════════════════════════════");
    println!("  Starting simulation — Case {:?}", case);
    println!("══════════════════════════════════════════════════\n");
    println!("Step 0: z={:.2}, KE/KE₀=1.000, Seg={:.4}", Z_INIT, seg_0);

    let mut tau = cosmo.tau_start;
    let start = std::time::Instant::now();
    let mut ke_ratio_max = 1.0f64;
    let mut seg_max = seg_0;
    let mut seg_max_z = Z_INIT;

    for step in 1..=TOTAL_STEPS {
        let step_start = std::time::Instant::now();

        tau += dtau_per_step;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        let ke = sim.kinetic_energy().expect("kinetic_energy failed");
        let ke_ratio = ke / ke_0;
        let pos = sim.get_positions().expect("get_positions failed");
        let seg = compute_segregation(&pos, &signs);

        ke_ratio_max = ke_ratio_max.max(ke_ratio);
        if seg > seg_max {
            seg_max = seg;
            seg_max_z = z;
        }

        writeln!(csv, "{},{:.4},{:.6},{:.6},{:.1}", step, z, ke_ratio, seg, step_ms).unwrap();

        // Step 5 check (FAIL criterion from EXPLORATION_GRID.md)
        if step == 5 {
            println!("\n=== STEP 5 CHECK ===");
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            if ke_ratio > 10.0 {
                println!("  ❌ FAIL: KE/KE₀ > 10 → stopping");
                csv.flush().unwrap();
                write_summary(&output_dir, case, seg_0, seg_max, seg_max_z, ke_ratio_max, ke_ratio, "FAIL");
                return;
            }
            println!("  ✓ PASS: KE/KE₀ < 10");
        }

        // Render at intervals
        if step % RENDER_INTERVAL == 0 {
            let render_path = format!("{}/step_{:06}.bin", render_dir, step);
            write_render_data(&render_path, &pos, &signs, step, BOX_SIZE, seg, ke_ratio, z)
                .expect("Failed to write render_data");
        }

        // Progress every 100 steps
        if step % 100 == 0 {
            let rate = step as f64 / start.elapsed().as_secs_f64();
            println!("Step {}: z={:.2}, KE/KE₀={:.4}, Seg={:.4} ({:.1} steps/s)",
                     step, z, ke_ratio, seg, rate);
        }

        // Explosion check
        if ke_ratio > 50.0 && step > 20 {
            println!("\n❌ EXPLOSION: KE/KE₀ = {:.1} > 50 at step {}", ke_ratio, step);
            csv.flush().unwrap();
            write_summary(&output_dir, case, seg_0, seg_max, seg_max_z, ke_ratio_max, ke_ratio, "FAIL");
            return;
        }
    }

    csv.flush().unwrap();

    // Determine verdict
    let ke_final = sim.kinetic_energy().expect("kinetic_energy failed") / ke_0;
    let verdict = if seg_max > 0.30 && ke_ratio_max < 2.0 {
        "EXCEL"
    } else if seg_max > 0.10 && ke_ratio_max < 3.0 {
        "GOOD"
    } else if seg_max > 0.05 && ke_ratio_max < 5.0 {
        "PASS"
    } else if (seg_max - seg_0).abs() < 0.005 {
        "FROZEN"
    } else {
        "MARGINAL"
    };

    write_summary(&output_dir, case, seg_0, seg_max, seg_max_z, ke_ratio_max, ke_final, verdict);

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n══════════════════════════════════════════════════");
    println!("  Case {:?} Complete — {}", case, verdict);
    println!("══════════════════════════════════════════════════");
    println!("  Total time: {:.1}s ({:.1} ms/step)", elapsed, elapsed * 1000.0 / TOTAL_STEPS as f64);
    println!("  Seg_0: {:.4}, Seg_max: {:.4} @ z={:.2}", seg_0, seg_max, seg_max_z);
    println!("  KE/KE₀ max: {:.4}, final: {:.4}", ke_ratio_max, ke_final);
    println!("  Verdict: {}", verdict);
    println!("  Output: {}", output_dir);
}

fn write_summary(output_dir: &str, case: IcCase, seg_0: f64, seg_max: f64, seg_max_z: f64, ke_max: f64, ke_final: f64, verdict: &str) {
    let summary_path = format!("{}/summary.json", output_dir);
    let mut file = File::create(&summary_path).expect("Failed to create summary");
    writeln!(file, "{{").unwrap();
    writeln!(file, "  \"case\": \"{:?}\",", case).unwrap();
    writeln!(file, "  \"ic_type\": \"{}\",", case.ic_type()).unwrap();
    writeln!(file, "  \"amplitude_mpc\": {:.3},", case.amplitude_factor() * (BOX_SIZE / (N_PARTICLES as f64).cbrt())).unwrap();
    writeln!(file, "  \"seg_0\": {:.6},", seg_0).unwrap();
    writeln!(file, "  \"seg_max\": {:.6},", seg_max).unwrap();
    writeln!(file, "  \"seg_max_z\": {:.2},", seg_max_z).unwrap();
    writeln!(file, "  \"ke_max\": {:.4},", ke_max).unwrap();
    writeln!(file, "  \"ke_final\": {:.4},", ke_final).unwrap();
    writeln!(file, "  \"verdict\": \"{}\"", verdict).unwrap();
    writeln!(file, "}}").unwrap();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature. Compile with:");
    eprintln!("  cargo build --release --features cuda --bin janus_grid_exploration");
}
