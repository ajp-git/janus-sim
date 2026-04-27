//! Bi-sector trivial test (Task 2 from validation plan):
//!   - 50/50 m+/m- random
//!   - c_ratio_sq = 1, φ = 1 (cross-coupling on but trivial)
//!   - a+ = a- (single-cosmology EdS)
//!   - z=49 → z=5
//!   - IC FFT growing-mode (same as test_eds_growing_mode)
//!   - Softening 0.1 Mpc
//!
//! Validation: Corr(δ+, δ-) ∈ [-0.15, +0.15] at all 5 snapshots
//! No NaN, no runaway in v_rms or ρ_max.

use janus::nbody_gpu::GpuNBodySimulation;
use rustfft::{FftPlanner, num_complex::Complex};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::Write;

const N_SIDE: usize = 100;
const N_PART: usize = N_SIDE * N_SIDE * N_SIDE;
const L_BOX: f64 = 200.0;
const N_GRID: usize = 64;
const Z_INIT: f64 = 49.0;
const Z_FINAL: f64 = 5.0;
const H0_KMS_MPC: f64 = 70.0;
const MPC_GYR_TO_KMS: f64 = 977.8;
const HUBBLE_LITTLE_H: f64 = 0.70;
const SIGMA_8_TARGET_AT_Z0: f64 = 0.8;
const EPS: f64 = 0.1;
const SEED: u64 = 4242;

fn h_eds(a: f64, h0_gyr: f64) -> f64 { h0_gyr * a.powf(-1.5) }
fn a_eds_step(a: f64, dt: f64, h0_gyr: f64) -> f64 { a + h0_gyr * a.powf(-0.5) * dt }

fn fft3d_forward(data: &mut [Complex<f64>], n_grid: usize) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_grid);
    for j in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid).map(|i| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            fft.process(&mut row);
            for i in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[i]; }
        }
    }
    for i in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid).map(|j| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            fft.process(&mut row);
            for j in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[j]; }
        }
    }
    for i in 0..n_grid {
        for j in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid).map(|k| data[i*n_grid*n_grid + j*n_grid + k]).collect();
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
            let mut row: Vec<Complex<f64>> = (0..n_grid).map(|i| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            ifft.process(&mut row);
            for i in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[i]; }
        }
    }
    for i in 0..n_grid {
        for k in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid).map(|j| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            ifft.process(&mut row);
            for j in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[j]; }
        }
    }
    for i in 0..n_grid {
        for j in 0..n_grid {
            let mut row: Vec<Complex<f64>> = (0..n_grid).map(|k| data[i*n_grid*n_grid + j*n_grid + k]).collect();
            ifft.process(&mut row);
            for k in 0..n_grid { data[i*n_grid*n_grid + j*n_grid + k] = row[k]; }
        }
    }
    let n3 = (n_grid * n_grid * n_grid) as f64;
    for c in data.iter_mut() { *c /= n3; }
}

fn k_vec(i: usize, n: usize, kf: f64) -> f64 {
    let half = n / 2;
    if i <= half { i as f64 * kf } else { (i as f64 - n as f64) * kf }
}

fn build_delta_k(n_grid: usize, _box_size: f64, amplitude: f64, seed: u64) -> Vec<Complex<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n3 = n_grid * n_grid * n_grid;
    let mut delta_real: Vec<Complex<f64>> = (0..n3).map(|_| Complex::new(normal.sample(&mut rng), 0.0)).collect();
    fft3d_forward(&mut delta_real, n_grid);
    let mut delta_k = delta_real.clone();
    delta_k[0] = Complex::new(0.0, 0.0); // zero mean
    for c in delta_k.iter_mut() { *c *= amplitude; }
    delta_k
}

fn build_psi_fields(delta_k: &[Complex<f64>], n_grid: usize, box_size: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

fn cic_density_signed(positions: &[f64], signs: &[i32], target_sign: i32,
                      n_grid: usize, box_size: f64, n_part: usize) -> Vec<f64> {
    let cell = box_size / n_grid as f64;
    let mut rho = vec![0.0_f64; n_grid * n_grid * n_grid];
    for i in 0..n_part {
        if signs[i] != target_sign { continue; }
        let x = (positions[3*i]   + box_size / 2.0).rem_euclid(box_size) / cell;
        let y = (positions[3*i+1] + box_size / 2.0).rem_euclid(box_size) / cell;
        let z = (positions[3*i+2] + box_size / 2.0).rem_euclid(box_size) / cell;
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

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma: f64 = a.iter().sum::<f64>() / n;
    let mb: f64 = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    num / (da.sqrt() * db.sqrt())
}

fn sigma_R(rho: &[f64], n_grid: usize, box_size: f64, r_smooth: f64) -> f64 {
    let n3 = n_grid * n_grid * n_grid;
    let mean: f64 = rho.iter().sum::<f64>() / n3 as f64;
    let mut delta: Vec<Complex<f64>> = rho.iter().map(|&r| Complex::new(r/mean - 1.0, 0.0)).collect();
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
                let w = if x.abs() < 1e-6 { 1.0 } else { 3.0*(x.sin() - x*x.cos())/x.powi(3) };
                let c = delta[i*n_grid*n_grid + j*n_grid + k];
                var += (c.re*c.re + c.im*c.im) * w*w;
            }
        }
    }
    let nf = n3 as f64;
    var /= nf * nf;
    var.sqrt()
}

fn vrms(velocities: &[f64], signs: &[i32], target_sign: i32, n: usize) -> f64 {
    let mut s2 = 0.0_f64;
    let mut count = 0;
    for i in 0..n {
        if signs[i] != target_sign { continue; }
        let v2 = velocities[3*i].powi(2) + velocities[3*i+1].powi(2) + velocities[3*i+2].powi(2);
        s2 += v2;
        count += 1;
    }
    if count > 0 { (s2 / count as f64).sqrt() } else { 0.0 }
}

fn rho_max_signed(positions: &[f64], signs: &[i32], target_sign: i32,
                  n_grid: usize, box_size: f64, n_part: usize) -> f64 {
    let rho = cic_density_signed(positions, signs, target_sign, n_grid, box_size, n_part);
    let mean: f64 = rho.iter().sum::<f64>() / rho.len() as f64;
    let max = rho.iter().cloned().fold(0.0_f64, f64::max);
    max / mean
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bi-sector trivial test (Task 2) ===");
    println!("[CONFIG] N={}  L={} Mpc  z {}->{}", N_PART, L_BOX, Z_INIT, Z_FINAL);

    let h0_gyr = H0_KMS_MPC / MPC_GYR_TO_KMS;
    let a_init = 1.0 / (1.0 + Z_INIT);
    let a_final = 1.0 / (1.0 + Z_FINAL);

    // === Build δ_k field, normalize to σ_8 EdS at z=49 ===
    let delta_k_raw = build_delta_k(N_GRID, L_BOX, 1.0, SEED);
    let mut delta_grid = delta_k_raw.clone();
    fft3d_backward(&mut delta_grid, N_GRID);
    let delta_real: Vec<f64> = delta_grid.iter().map(|c| c.re).collect();
    let r8 = 8.0 / HUBBLE_LITTLE_H;
    let rho_for_sigma: Vec<f64> = delta_real.iter().map(|&d| 1.0 + d).collect();
    let sigma_8_raw = sigma_R(&rho_for_sigma, N_GRID, L_BOX, r8);
    let sigma_8_target = SIGMA_8_TARGET_AT_Z0 * a_init;
    let norm = sigma_8_target / sigma_8_raw;
    println!("[IC] σ_8 raw={:.4e}  target z=49 EdS={:.4e}  norm={:.4e}", sigma_8_raw, sigma_8_target, norm);

    let delta_k: Vec<Complex<f64>> = delta_k_raw.iter().map(|&c| c * norm).collect();
    let (psi_x, psi_y, psi_z) = build_psi_fields(&delta_k, N_GRID, L_BOX);

    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let cell_p = L_BOX / N_SIDE as f64;
    let cell_g = L_BOX / N_GRID as f64;
    let half = L_BOX / 2.0;
    let h_init = h_eds(a_init, h0_gyr);
    let vel_factor = a_init * h_init;

    // 50/50 sign assignment
    let n_plus = N_PART / 2;
    let mut signs_arr: Vec<i32> = (0..N_PART).map(|i| if i < n_plus { 1i32 } else { -1i32 }).collect();
    signs_arr.shuffle(&mut rng);

    let mut positions = Vec::with_capacity(N_PART * 3);
    let mut velocities = Vec::with_capacity(N_PART * 3);

    for i in 0..N_SIDE {
        for j in 0..N_SIDE {
            for k in 0..N_SIDE {
                let x_lag = (i as f64 + 0.5) * cell_p - half;
                let y_lag = (j as f64 + 0.5) * cell_p - half;
                let z_lag = (k as f64 + 0.5) * cell_p - half;
                let gx = (((x_lag + half)/cell_g) as usize) % N_GRID;
                let gy = (((y_lag + half)/cell_g) as usize) % N_GRID;
                let gz = (((z_lag + half)/cell_g) as usize) % N_GRID;
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

    let n_minus = N_PART - n_plus;
    let mut sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_BOX, positions, velocities, signs_arr.clone()
    )?;
    sim.set_theta(0.7);
    sim.set_softening(EPS);
    sim.set_phi(1.0, 1.0);     // φ = 1 (a+ = a- → no asymmetry)
    sim.c_ratio_sq = 1.0;       // VSL OFF
    sim.repulsion_scale = 1.0;  // cross-coupling ON (but trivial since φ=1, c̄=1)
    sim.set_mass_factor(1.0 / 0.3);
    println!("[SIM] θ=0.7  ε={} Mpc  Ω_m=1  φ=1  c̄=1  repulsion=1", EPS);

    let log_path = if std::path::Path::new("/app/output").is_dir() {
        "/app/output/bisector_trivial.log"
    } else {
        "/mnt/T2/janus-sim/output/bisector_trivial.log"
    };
    let mut log = File::create(log_path)?;
    writeln!(log, "# Bi-sector trivial test (a+ = a-, c=1, φ=1)")?;
    writeln!(log, "# step  a       z       sigma+      sigma-      Corr(+,-)   v_rms+      v_rms-      rho_max+    rho_max-")?;

    // 5 log-spaced snapshots in a (a=0.02, 0.05, 0.10, 0.15, 0.20 per spec — adjust to log-space)
    let log_ai = a_init.ln();
    let log_af = a_final.ln();
    let snap_a: Vec<f64> = (0..5).map(|i| (log_ai + (log_af - log_ai) * i as f64 / 4.0).exp()).collect();
    println!("[SNAPS] a values: {:?}", snap_a);

    let dt = 0.001_f64;
    let mut a = a_init;
    let mut snap_idx = 0;

    println!();
    println!("step    a       z       sigma+      sigma-      Corr(+,-)   v_rms+      v_rms-      rho_max+    rho_max-");

    let max_steps = 200_000;
    for step in 0..max_steps {
        if snap_idx < 5 && a >= snap_a[snap_idx] {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            let signs_now = sim.signs();
            let rho_p = cic_density_signed(&pos, &signs_now,  1, 64, L_BOX, N_PART);
            let rho_m = cic_density_signed(&pos, &signs_now, -1, 64, L_BOX, N_PART);
            let mp: f64 = rho_p.iter().sum::<f64>() / rho_p.len() as f64;
            let mm: f64 = rho_m.iter().sum::<f64>() / rho_m.len() as f64;
            let dp: Vec<f64> = rho_p.iter().map(|r| r/mp - 1.0).collect();
            let dm: Vec<f64> = rho_m.iter().map(|r| r/mm - 1.0).collect();
            let corr = correlation(&dp, &dm);
            let sig_p = (dp.iter().map(|x| x*x).sum::<f64>() / dp.len() as f64).sqrt();
            let sig_m = (dm.iter().map(|x| x*x).sum::<f64>() / dm.len() as f64).sqrt();
            let v_p = vrms(&vel, &signs_now,  1, N_PART);
            let v_m = vrms(&vel, &signs_now, -1, N_PART);
            let rmax_p = rho_max_signed(&pos, &signs_now,  1, 64, L_BOX, N_PART);
            let rmax_m = rho_max_signed(&pos, &signs_now, -1, 64, L_BOX, N_PART);
            let z = 1.0 / a - 1.0;
            let line = format!("{:>5}  {:.4}  {:.3}  {:.4e}  {:.4e}  {:+.4}      {:.4e}  {:.4e}  {:.3e}  {:.3e}",
                step, a, z, sig_p, sig_m, corr, v_p, v_m, rmax_p, rmax_m);
            println!("{}", line);
            writeln!(log, "{}", line)?;
            log.flush()?;
            snap_idx += 1;
        }
        let h = h_eds(a, h0_gyr);
        sim.step_with_expansion_dkd_gpu_cosmo(dt, a, a, h, h)?;
        a = a_eds_step(a, dt, h0_gyr);
        if a >= a_final || snap_idx == 5 { break; }
    }

    println!();
    println!("=== VERDICT ===");
    println!("(Manual interpretation of Corr values from log)");
    println!("Critère: Corr(δ+,δ-) ∈ [-0.15, +0.15] sur tous snapshots → ✅ infrastructure bi-secteur OK");
    println!("Pas de NaN, pas de runaway → ✅ stabilité numérique");
    Ok(())
}
