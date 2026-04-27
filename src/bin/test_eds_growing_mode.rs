//! Proper EdS validation with pure growing-mode IC.
//!
//! IC generation:
//!   1. Generate random Gaussian δ(k) on FFT grid with flat power spectrum
//!   2. Use Zel'dovich growing-mode: ψ_k = +i·k·δ(k)/k²  →  ψ = ∇⁻¹(δ) (curl-free)
//!   3. Normalize ψ so that σ_8 at z_init matches EdS expectation
//!   4. Velocities: v_pec = a·H·ψ (peculiar in EdS, D=a)
//!
//! Validation:
//!   At each snapshot, FFT δ on grid, integrate P(k) over k < k_NL.
//!   D_meas(a) = sqrt(<δ²>_filtered)(a) / sqrt(<δ²>_filtered)(a_init)
//!   D_th(a)   = a / a_init
//!   Check |R(a) − 1| < 0.10 over a > 3·a_init.
//!
//! σ_8 sanity check at IC: σ_8(z=49, EdS) = σ_8(z=0)·a/a_0 ≈ 0.8·(1/50) = 0.016.

use janus::nbody_gpu::GpuNBodySimulation;
use rustfft::{FftPlanner, num_complex::Complex};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use std::fs::File;
use std::io::Write;

const N_SIDE: usize = 100;
const N_PART: usize = N_SIDE * N_SIDE * N_SIDE;
const L_BOX: f64 = 200.0;
const N_GRID: usize = 64;             // FFT grid for IC and analysis
const Z_INIT_DEFAULT: f64 = 49.0;
const Z_FINAL: f64 = 0.0;
const H0_KMS_MPC: f64 = 70.0;
const MPC_GYR_TO_KMS: f64 = 977.8;
const N_SNAPSHOTS: usize = 50;
const SIGMA_8_TARGET_AT_Z0: f64 = 0.8;  // standard ΛCDM-ish value
const HUBBLE_LITTLE_H: f64 = 0.70;       // for σ_8 unit conversion

fn h_eds(a: f64, h0_gyr: f64) -> f64 { h0_gyr * a.powf(-1.5) }
fn a_eds_step(a: f64, dt: f64, h0_gyr: f64) -> f64 {
    a + h0_gyr * a.powf(-0.5) * dt
}

/// 3-D forward FFT in place (columns-major flattened, n_grid³).
fn fft3d_forward(data: &mut [Complex<f64>], n_grid: usize) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_grid);
    // x
    for j in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid)
                .map(|i| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            fft.process(&mut row);
            for i in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[i]; }
        }
    }
    // y
    for i in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid)
                .map(|j| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            fft.process(&mut row);
            for j in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[j]; }
        }
    }
    // z
    for i in 0..n_grid {
        for j in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid)
                .map(|k| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            fft.process(&mut row);
            for k in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[k]; }
        }
    }
}

fn fft3d_backward(data: &mut [Complex<f64>], n_grid: usize) {
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n_grid);
    for j in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid)
                .map(|i| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            ifft.process(&mut row);
            for i in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[i]; }
        }
    }
    for i in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid)
                .map(|j| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            ifft.process(&mut row);
            for j in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[j]; }
        }
    }
    for i in 0..n_grid {
        for j in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid)
                .map(|k| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            ifft.process(&mut row);
            for k in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[k]; }
        }
    }
    // Normalize for inverse: divide by N
    let n3 = (n_grid * n_grid * n_grid) as f64;
    for c in data.iter_mut() { *c /= n3; }
}

fn k_vec(i: usize, n: usize, kf: f64) -> f64 {
    let half = n / 2;
    if i <= half { i as f64 * kf } else { (i as f64 - n as f64) * kf }
}

/// Build δ_k field with flat power spectrum (white noise) → useful for IC.
/// Returns δ_k array. Real space σ_δ scales with `amplitude` parameter.
fn build_delta_k(n_grid: usize, box_size: f64, amplitude: f64, seed: u64) -> Vec<Complex<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let kf = 2.0 * std::f64::consts::PI / box_size;
    let n3 = n_grid * n_grid * n_grid;
    let mut delta_k = vec![Complex::new(0.0, 0.0); n3];

    // Build a Gaussian random REAL field, then FFT to get δ_k.
    // (Easier than building Hermitian-symmetric field directly.)
    let mut delta_real: Vec<Complex<f64>> = (0..n3)
        .map(|_| Complex::new(normal.sample(&mut rng), 0.0))
        .collect();

    fft3d_forward(&mut delta_real, n_grid);

    // Apply k-dependent power spectrum (here: flat with k cutoff → mimics simple ICs).
    // P(k) ∝ k^n_s with n_s ≈ 1 for scale-invariant. We use n_s=0 (flat) for simplicity.
    // The amplitude is set later via σ_8 normalization.
    for i in 0..n_grid {
        for j in 0..n_grid {
            for k in 0..n_grid {
                let kx = k_vec(i, n_grid, kf);
                let ky = k_vec(j, n_grid, kf);
                let kz = k_vec(k, n_grid, kf);
                let kmag = (kx*kx + ky*ky + kz*kz).sqrt();
                // Cut at k_max = π·N/L (Nyquist), zero at k=0 to enforce <δ>=0
                let weight = if kmag > 0.0 { 1.0 } else { 0.0 };
                delta_k[i*n_grid*n_grid + j*n_grid + k] = delta_real[i*n_grid*n_grid + j*n_grid + k] * weight;
            }
        }
    }

    // Apply amplitude normalization
    for c in delta_k.iter_mut() { *c *= amplitude; }

    delta_k
}

/// Compute Zel'dovich displacement ψ = ∇⁻¹(δ) [scalar gradient].
/// In Fourier: ψ_k = i·k·δ(k)/k²
/// Returns three real fields ψ_x(x), ψ_y(x), ψ_z(x) on the grid.
fn build_psi_fields(delta_k: &[Complex<f64>], n_grid: usize, box_size: f64)
    -> (Vec<f64>, Vec<f64>, Vec<f64>)
{
    let kf = 2.0 * std::f64::consts::PI / box_size;
    let n3 = n_grid * n_grid * n_grid;

    let mut psi_x_k = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_y_k = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_z_k = vec![Complex::new(0.0, 0.0); n3];

    for i in 0..n_grid {
        for j in 0..n_grid {
            for k in 0..n_grid {
                let kx = k_vec(i, n_grid, kf);
                let ky = k_vec(j, n_grid, kf);
                let kz = k_vec(k, n_grid, kf);
                let k2 = kx*kx + ky*ky + kz*kz;
                if k2 == 0.0 { continue; }
                let idx = i*n_grid*n_grid + j*n_grid + k;
                let dk = delta_k[idx];
                // ψ_k = i·k·δ(k)/k² (in component form)
                let i_over_k2 = Complex::new(0.0, 1.0) / k2;
                psi_x_k[idx] = i_over_k2 * dk * kx;
                psi_y_k[idx] = i_over_k2 * dk * ky;
                psi_z_k[idx] = i_over_k2 * dk * kz;
            }
        }
    }

    fft3d_backward(&mut psi_x_k, n_grid);
    fft3d_backward(&mut psi_y_k, n_grid);
    fft3d_backward(&mut psi_z_k, n_grid);

    let psi_x: Vec<f64> = psi_x_k.iter().map(|c| c.re).collect();
    let psi_y: Vec<f64> = psi_y_k.iter().map(|c| c.re).collect();
    let psi_z: Vec<f64> = psi_z_k.iter().map(|c| c.re).collect();
    (psi_x, psi_y, psi_z)
}

/// CIC density on a grid (returns count-per-cell, must normalize externally).
fn cic_density(positions: &[f64], n_grid: usize, box_size: f64, n_part: usize) -> Vec<f64> {
    let cell = box_size / n_grid as f64;
    let mut rho = vec![0.0_f64; n_grid * n_grid * n_grid];
    for p in 0..n_part {
        let x = (positions[3*p]   + box_size / 2.0).rem_euclid(box_size) / cell;
        let y = (positions[3*p+1] + box_size / 2.0).rem_euclid(box_size) / cell;
        let z = (positions[3*p+2] + box_size / 2.0).rem_euclid(box_size) / cell;
        let i0 = (x.floor() as i64).rem_euclid(n_grid as i64) as usize;
        let j0 = (y.floor() as i64).rem_euclid(n_grid as i64) as usize;
        let k0 = (z.floor() as i64).rem_euclid(n_grid as i64) as usize;
        let dx = x - x.floor();
        let dy = y - y.floor();
        let dz = z - z.floor();
        let i1 = (i0 + 1) % n_grid;
        let j1 = (j0 + 1) % n_grid;
        let k1 = (k0 + 1) % n_grid;
        for &(ix, wx) in &[(i0, 1.0 - dx), (i1, dx)] {
            for &(iy, wy) in &[(j0, 1.0 - dy), (j1, dy)] {
                for &(iz, wz) in &[(k0, 1.0 - dz), (k1, dz)] {
                    rho[ix*n_grid*n_grid + iy*n_grid + iz] += wx*wy*wz;
                }
            }
        }
    }
    rho
}

/// Compute σ_R = sqrt(<δ²>) using top-hat smoothing of radius R [Mpc].
/// δ_k_smoothed = δ_k · W_TH(k·R), W_TH(x) = 3(sin x − x cos x)/x³.
fn sigma_R(rho: &[f64], n_grid: usize, box_size: f64, r_smooth: f64) -> f64 {
    let n3 = n_grid * n_grid * n_grid;
    let mean: f64 = rho.iter().sum::<f64>() / n3 as f64;
    let mut delta: Vec<Complex<f64>> = rho.iter()
        .map(|&r| Complex::new(r/mean - 1.0, 0.0)).collect();
    fft3d_forward(&mut delta, n_grid);

    let kf = 2.0 * std::f64::consts::PI / box_size;
    let mut var = 0.0_f64;
    for i in 0..n_grid {
        for j in 0..n_grid {
            for k in 0..n_grid {
                let kx = k_vec(i, n_grid, kf);
                let ky = k_vec(j, n_grid, kf);
                let kz = k_vec(k, n_grid, kf);
                let km = (kx*kx + ky*ky + kz*kz).sqrt();
                if km == 0.0 { continue; }
                let x = km * r_smooth;
                let w = if x.abs() < 1e-6 { 1.0 } else { 3.0 * (x.sin() - x*x.cos()) / x.powi(3) };
                let c = delta[i*n_grid*n_grid + j*n_grid + k];
                var += (c.re*c.re + c.im*c.im) * w*w;
            }
        }
    }
    let nf = n3 as f64;
    var /= nf * nf;
    var.sqrt()
}

/// σ filtered low-pass: include only modes with |k| < k_max.
fn sigma_filtered_kmax(rho: &[f64], n_grid: usize, box_size: f64, k_max: f64) -> f64 {
    let n3 = n_grid * n_grid * n_grid;
    let mean: f64 = rho.iter().sum::<f64>() / n3 as f64;
    let mut delta: Vec<Complex<f64>> = rho.iter()
        .map(|&r| Complex::new(r/mean - 1.0, 0.0)).collect();
    fft3d_forward(&mut delta, n_grid);

    let kf = 2.0 * std::f64::consts::PI / box_size;
    let mut var = 0.0_f64;
    for i in 0..n_grid {
        for j in 0..n_grid {
            for k in 0..n_grid {
                let kx = k_vec(i, n_grid, kf);
                let ky = k_vec(j, n_grid, kf);
                let kz = k_vec(k, n_grid, kf);
                let km = (kx*kx + ky*ky + kz*kz).sqrt();
                if km > 0.0 && km < k_max {
                    let c = delta[i*n_grid*n_grid + j*n_grid + k];
                    var += c.re*c.re + c.im*c.im;
                }
            }
        }
    }
    let nf = n3 as f64;
    var /= nf * nf;
    var.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EdS validation with PROPER growing-mode IC ===");

    let z_init: f64 = std::env::var("Z_INIT_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(Z_INIT_DEFAULT);
    let z_final_val: f64 = std::env::var("Z_FINAL_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(Z_FINAL);
    let snap_save_z: f64 = std::env::var("SNAP_SAVE_Z")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(-1.0);  // negative means no save
    let a_init = 1.0 / (1.0 + z_init);
    let a_final = 1.0 / (1.0 + z_final_val);
    let h0_gyr = H0_KMS_MPC / MPC_GYR_TO_KMS;

    println!("[CONFIG] z_init={}  a_init={:.4}  N={}  L={} Mpc  N_grid={}",
        z_init, a_init, N_PART, L_BOX, N_GRID);

    // === Build δ_k field on N_GRID³ ===
    let amplitude_guess = 1.0;
    let delta_k = build_delta_k(N_GRID, L_BOX, amplitude_guess, 4242);

    // Build δ in real space (for σ_8 normalization)
    let mut delta_grid = delta_k.clone();
    fft3d_backward(&mut delta_grid, N_GRID);
    let delta_real: Vec<f64> = delta_grid.iter().map(|c| c.re).collect();

    // Check σ_R8 of unnormalized δ (treating as +1 background, computing δ directly)
    // For pure δ field: σ² = <δ²> in real space.
    // σ_8 = sqrt(<δ²>) at smoothing scale R8 = 8/h Mpc.
    let r8_mpc = 8.0 / HUBBLE_LITTLE_H;  // ≈ 11.43 Mpc

    // Treat delta_real as ρ = 1 + delta_real (for sigma_R function)
    let rho_for_sigma: Vec<f64> = delta_real.iter().map(|&d| 1.0 + d).collect();
    let sigma_8_raw = sigma_R(&rho_for_sigma, N_GRID, L_BOX, r8_mpc);
    println!("[IC] σ_8(unnormalized) = {:.4e}", sigma_8_raw);

    // Normalize so σ_8 at z_init equals EdS prediction
    let sigma_8_target = SIGMA_8_TARGET_AT_Z0 * a_init;
    let norm_factor = sigma_8_target / sigma_8_raw;
    println!("[IC] σ_8(target z={}, EdS) = {:.4e}", z_init, sigma_8_target);
    println!("[IC] normalization factor = {:.4e}", norm_factor);

    // Renormalize δ_k
    let delta_k_norm: Vec<Complex<f64>> = delta_k.iter().map(|&c| c * norm_factor).collect();

    // === Build ψ field from normalized δ_k ===
    let (psi_x, psi_y, psi_z) = build_psi_fields(&delta_k_norm, N_GRID, L_BOX);
    let psi_rms = {
        let n3 = N_GRID*N_GRID*N_GRID;
        let s2: f64 = (0..n3).map(|i| psi_x[i].powi(2) + psi_y[i].powi(2) + psi_z[i].powi(2))
            .sum::<f64>() / n3 as f64;
        s2.sqrt()
    };
    println!("[IC] ψ_rms = {:.4e} Mpc (3-component)", psi_rms);

    // === Place particles on perturbed grid ===
    let cell_p = L_BOX / N_SIDE as f64;
    let cell_g = L_BOX / N_GRID as f64;
    let half = L_BOX / 2.0;
    let h_init = h_eds(a_init, h0_gyr);
    let vel_factor = a_init * h_init;  // v_pec = a·H·ψ for D=a
    println!("[IC] H(z=49) = {:.4} 1/Gyr  vel_factor = a·H = {:.4e}", h_init, vel_factor);

    let mut positions = Vec::with_capacity(N_PART * 3);
    let mut velocities = Vec::with_capacity(N_PART * 3);
    let signs: Vec<i32> = vec![1; N_PART];

    for i in 0..N_SIDE {
        for j in 0..N_SIDE {
            for k in 0..N_SIDE {
                let x_lag = (i as f64 + 0.5) * cell_p - half;
                let y_lag = (j as f64 + 0.5) * cell_p - half;
                let z_lag = (k as f64 + 0.5) * cell_p - half;

                // Lookup ψ on FFT grid via NGP
                let gx = (((x_lag + half) / cell_g) as usize) % N_GRID;
                let gy = (((y_lag + half) / cell_g) as usize) % N_GRID;
                let gz = (((z_lag + half) / cell_g) as usize) % N_GRID;
                let gidx = gx*N_GRID*N_GRID + gy*N_GRID + gz;

                let px = psi_x[gidx];
                let py = psi_y[gidx];
                let pz = psi_z[gidx];

                let mut x = x_lag + px;
                let mut y = y_lag + py;
                let mut z = z_lag + pz;
                if x >  half { x -= L_BOX; } else if x < -half { x += L_BOX; }
                if y >  half { y -= L_BOX; } else if y < -half { y += L_BOX; }
                if z >  half { z -= L_BOX; } else if z < -half { z += L_BOX; }

                positions.push(x); positions.push(y); positions.push(z);
                velocities.push(px * vel_factor);
                velocities.push(py * vel_factor);
                velocities.push(pz * vel_factor);
            }
        }
    }

    // Check actual σ_8 of placed particles
    let rho0 = cic_density(&positions, N_GRID, L_BOX, N_PART);
    let sigma_8_placed = sigma_R(&rho0, N_GRID, L_BOX, r8_mpc);
    println!("[IC] σ_8(placed particles) = {:.4e}  (target {:.4e})  ratio {:.3}",
        sigma_8_placed, sigma_8_target, sigma_8_placed/sigma_8_target);

    // === Run sim ===
    let mut sim = GpuNBodySimulation::new_with_state(
        N_PART, 0, L_BOX, positions.clone(), velocities, signs.clone(),
    )?;
    sim.set_theta(0.7);
    sim.set_softening(0.05);
    sim.set_phi(1.0, 1.0);
    sim.c_ratio_sq = 1.0;
    sim.repulsion_scale = 0.0;
    sim.set_mass_factor(1.0 / 0.3);
    println!("[SIM] θ=0.7  ε=0.05 Mpc  Ω_m=1");

    let log_path = if std::path::Path::new("/app/output").is_dir() {
        "/app/output/eds_growing_mode.log"
    } else {
        "/mnt/T2/janus-sim/output/eds_growing_mode.log"
    };
    let mut log = File::create(log_path)?;
    writeln!(log, "# step  a         z         t_Gyr     sigma_filt  sigma_8     D_meas    D_th      ratio_R")?;

    let log_ai = a_init.ln();
    let log_af = a_final.ln();
    let snap_a: Vec<f64> = (0..N_SNAPSHOTS)
        .map(|i| (log_ai + (log_af - log_ai) * i as f64 / (N_SNAPSHOTS - 1) as f64).exp())
        .collect();

    let dt: f64 = std::env::var("DT_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    println!("[RUN] dt={} Gyr  k_max=0.1 h/Mpc = {} 1/Mpc",
        dt, 0.1 * HUBBLE_LITTLE_H);

    let k_max = 0.1 * HUBBLE_LITTLE_H;  // h/Mpc → 1/Mpc

    let mut a = a_init;
    let mut t_gyr = 0.0_f64;
    let mut snap_idx = 0;
    let mut sigma_filt_init: Option<f64> = None;
    let mut ratios: Vec<(f64, f64)> = Vec::new();
    let mut snap_saved = false;
    let snap_save_a = if snap_save_z > 0.0 { 1.0/(1.0+snap_save_z) } else { -1.0 };

    println!();
    println!("step    a        z         t_Gyr     σ_filt(k<k_NL)  σ_8       D_meas   D_th    R");

    let max_steps = 1_000_000;
    for step in 0..max_steps {
        if snap_idx < N_SNAPSHOTS && a >= snap_a[snap_idx] {
            let pos = sim.get_positions()?;
            let rho = cic_density(&pos, N_GRID, L_BOX, N_PART);
            let sf = sigma_filtered_kmax(&rho, N_GRID, L_BOX, k_max);
            let s8 = sigma_R(&rho, N_GRID, L_BOX, r8_mpc);
            let z = 1.0 / a - 1.0;
            if sigma_filt_init.is_none() { sigma_filt_init = Some(sf); }
            let sf0 = sigma_filt_init.unwrap();
            let d_meas = sf / sf0;
            let d_th = a / a_init;
            let r = d_meas / d_th;
            ratios.push((a, r));

            let line = format!("{:>5}  {:.5}  {:.4}  {:.4}  {:.4e}    {:.4e}  {:.4}  {:.4}  {:.4}",
                step, a, z, t_gyr, sf, s8, d_meas, d_th, r);
            println!("{}", line);
            writeln!(log, "{}", line)?;
            log.flush()?;
            snap_idx += 1;
            if snap_idx == N_SNAPSHOTS { break; }
        }

        let h = h_eds(a, h0_gyr);
        sim.step_with_expansion_dkd_gpu_cosmo(dt, a, a, h, h)?;
        a = a_eds_step(a, dt, h0_gyr);
        t_gyr += dt;

        // Save snapshot once we cross snap_save_a
        if !snap_saved && snap_save_a > 0.0 && a >= snap_save_a {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            let signs_now = sim.signs();
            let path = if std::path::Path::new("/app/output").is_dir() {
                "/app/output/eds_snapshot_save.bin".to_string()
            } else {
                "/mnt/T2/janus-sim/output/eds_snapshot_save.bin".to_string()
            };
            let mut f = File::create(&path)?;
            // simple header: u64 n, f64 a, f64 t_gyr, f64 box_size
            f.write_all(&(N_PART as u64).to_le_bytes())?;
            f.write_all(&a.to_le_bytes())?;
            f.write_all(&t_gyr.to_le_bytes())?;
            f.write_all(&L_BOX.to_le_bytes())?;
            // particles: pos(3 f64) + vel(3 f64) + sign(i32) per particle
            for i in 0..N_PART {
                f.write_all(&pos[3*i].to_le_bytes())?;
                f.write_all(&pos[3*i+1].to_le_bytes())?;
                f.write_all(&pos[3*i+2].to_le_bytes())?;
                f.write_all(&vel[3*i].to_le_bytes())?;
                f.write_all(&vel[3*i+1].to_le_bytes())?;
                f.write_all(&vel[3*i+2].to_le_bytes())?;
                f.write_all(&signs_now[i].to_le_bytes())?;
            }
            f.flush()?;
            println!("[SAVED] snapshot at a={:.4} (z={:.2}) -> {}", a, 1.0/a-1.0, path);
            snap_saved = true;
        }

        if a >= a_final { break; }
    }

    let ratios_skip: Vec<f64> = ratios.iter()
        .filter(|(av, _)| *av > 3.0 * a_init)
        .map(|(_, r)| *r).collect();
    let n = ratios_skip.len() as f64;
    let mean: f64 = ratios_skip.iter().sum::<f64>() / n.max(1.0);
    let var: f64 = ratios_skip.iter().map(|r| (r-mean).powi(2)).sum::<f64>() / n.max(1.0);
    let std = var.sqrt();
    println!();
    println!("=== VERDICT ===");
    println!("⟨R⟩ over a > {:.4}: {:.4}", 3.0*a_init, mean);
    println!("σ(R)             : {:.4}", std);
    if (mean - 1.0).abs() < 0.10 && std < 0.10 {
        println!("✅ Linear growth recovered within 10%");
    } else if (mean - 1.0).abs() < 0.20 {
        println!("⚠ Linear growth recovered within 20%");
    } else {
        println!("❌ Linear growth NOT recovered (off by >20%)");
    }
    Ok(())
}
