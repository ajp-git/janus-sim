//! Softening Scaling Test — τ_relax(ε) measurement
//!
//! Tests if relaxation time depends on softening length ε.
//! If τ varies with ε → numerical artifact
//! If τ = const       → structural phenomenon
//!
//! Usage:
//!   cargo run --release --features cuda --bin softening_test -- --soft-factor 0.5 --name soft_low
//!   cargo run --release --features cuda --bin softening_test -- --soft-factor 1.0 --name soft_nom
//!   cargo run --release --features cuda --bin softening_test -- --soft-factor 2.0 --name soft_high

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::env;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

// Physical parameters
const L_BOX: f64 = 400.0;
const Z_INIT: f64 = 10.0;
const N_GRID: usize = 126;  // 126³ ≈ 2M particles

// Simulation parameters
const DT: f64 = 0.005;
const R_K_INTERVAL: usize = 50;
const MAX_STEPS: usize = 1000;  // Shorter runs for softening test
const THETA: f64 = 0.7;
const AMPLITUDE: f64 = 0.05;  // Fixed at 5%

// Power spectrum
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

// Target k for r(k,t)
const K_TARGET: f64 = 0.13;

fn generate_antisym_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize, f64) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;
    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let target_amplitude = AMPLITUDE * L_BOX;

    println!("Generating ANTISYMMETRIC ICs...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Amplitude: A = {:.1}% × L_box = {:.1} Mpc", AMPLITUDE * 100.0, target_amplitude);

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();

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

                if k < 1e-10 { continue; }

                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = pk.sqrt() * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Hermitian symmetry
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

    // Displacement field
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

    // IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);
    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    let scale = if max_disp > 0.0 { target_amplitude / max_disp } else { 1.0 };
    println!("  Raw max displacement: {:.4e} → scaled to {:.4} Mpc", max_disp, target_amplitude);

    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;

    let mut signs: Vec<i32> = Vec::with_capacity(n3);
    let mut n_positive = 0usize;
    for _ in 0..n3 {
        let sign = if rng.random::<bool>() { 1 } else { -1 };
        if sign > 0 { n_positive += 1; }
        signs.push(sign);
    }

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;
                let sign_factor = signs[idx] as f64;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let mut px = x0 + psi_x[idx] * scale * sign_factor;
                let mut py = y0 + psi_y[idx] * scale * sign_factor;
                let mut pz = z0 + psi_z[idx] * scale * sign_factor;

                while px > half_box { px -= L_BOX; }
                while px < -half_box { px += L_BOX; }
                while py > half_box { py -= L_BOX; }
                while py < -half_box { py += L_BOX; }
                while pz > half_box { pz -= L_BOX; }
                while pz < -half_box { pz += L_BOX; }

                positions.push(px);
                positions.push(py);
                positions.push(pz);

                velocities.push(psi_x[idx] * vel_scale * sign_factor);
                velocities.push(psi_y[idx] * vel_scale * sign_factor);
                velocities.push(psi_z[idx] * vel_scale * sign_factor);
            }
        }
    }

    println!("  N+ = {}, N- = {}", n_positive, n3 - n_positive);

    let mut rms_disp = 0.0f64;
    for i in 0..n3 {
        let d2 = (psi_x[i]*scale).powi(2) + (psi_y[i]*scale).powi(2) + (psi_z[i]*scale).powi(2);
        rms_disp += d2;
    }
    rms_disp = (rms_disp / n3 as f64).sqrt();

    (positions, velocities, signs, n_positive, rms_disp)
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

fn compute_r_k013(positions: &[f64], signs: &[i32], box_size: f64) -> f64 {
    let n_grid = 64;
    let n = positions.len() / 3;
    let cell_size = box_size / n_grid as f64;

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

        if signs[i] > 0 { density_plus[idx] += 1.0; }
        else { density_minus[idx] += 1.0; }
    }

    let mean_plus: f64 = density_plus.iter().sum::<f64>() / density_plus.len() as f64;
    let mean_minus: f64 = density_minus.iter().sum::<f64>() / density_minus.len() as f64;

    let delta_plus: Vec<f64> = density_plus.iter()
        .map(|&d| if mean_plus > 0.0 { (d - mean_plus) / mean_plus } else { 0.0 })
        .collect();
    let delta_minus: Vec<f64> = density_minus.iter()
        .map(|&d| if mean_minus > 0.0 { (d - mean_minus) / mean_minus } else { 0.0 })
        .collect();

    let mut fft_plus: Vec<Complex<f64>> = delta_plus.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut fft_minus: Vec<Complex<f64>> = delta_minus.iter().map(|&x| Complex::new(x, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_grid);

    for iy in 0..n_grid {
        for ix in 0..n_grid {
            let mut sp: Vec<Complex<f64>> = (0..n_grid).map(|iz| fft_plus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            let mut sm: Vec<Complex<f64>> = (0..n_grid).map(|iz| fft_minus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            fft.process(&mut sp); fft.process(&mut sm);
            for iz in 0..n_grid { fft_plus[iz * n_grid * n_grid + iy * n_grid + ix] = sp[iz]; fft_minus[iz * n_grid * n_grid + iy * n_grid + ix] = sm[iz]; }
        }
    }
    for iz in 0..n_grid {
        for ix in 0..n_grid {
            let mut sp: Vec<Complex<f64>> = (0..n_grid).map(|iy| fft_plus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            let mut sm: Vec<Complex<f64>> = (0..n_grid).map(|iy| fft_minus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            fft.process(&mut sp); fft.process(&mut sm);
            for iy in 0..n_grid { fft_plus[iz * n_grid * n_grid + iy * n_grid + ix] = sp[iy]; fft_minus[iz * n_grid * n_grid + iy * n_grid + ix] = sm[iy]; }
        }
    }
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            let base = iz * n_grid * n_grid + iy * n_grid;
            let mut sp: Vec<Complex<f64>> = fft_plus[base..base+n_grid].to_vec();
            let mut sm: Vec<Complex<f64>> = fft_minus[base..base+n_grid].to_vec();
            fft.process(&mut sp); fft.process(&mut sm);
            for ix in 0..n_grid { fft_plus[base + ix] = sp[ix]; fft_minus[base + ix] = sm[ix]; }
        }
    }

    let dk = 2.0 * PI / box_size;
    let half_n = n_grid / 2;
    let k_width = 0.03;

    let mut p_pp = 0.0f64;
    let mut p_mm = 0.0f64;
    let mut p_pm = 0.0f64;
    let mut count = 0usize;

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                if (k - K_TARGET).abs() < k_width {
                    let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                    let fp = fft_plus[idx];
                    let fm = fft_minus[idx];

                    p_pp += fp.norm_sqr();
                    p_mm += fm.norm_sqr();
                    p_pm += (fp * fm.conj()).re;
                    count += 1;
                }
            }
        }
    }

    if count > 0 && p_pp > 0.0 && p_mm > 0.0 {
        p_pm / (p_pp * p_mm).sqrt()
    } else {
        0.0
    }
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    let mut soft_factor = 1.0f64;  // Multiplier on nominal softening
    let mut run_name = "soft_test".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--soft-factor" => { soft_factor = args[i+1].parse().expect("Invalid soft-factor"); i += 2; }
            "--name" => { run_name = args[i+1].clone(); i += 2; }
            _ => { i += 1; }
        }
    }

    // Compute nominal softening
    let n3 = N_GRID * N_GRID * N_GRID;
    let mean_sep = L_BOX / (n3 as f64).powf(1.0/3.0);
    let soft_nominal = 0.5 * mean_sep;
    let softening = soft_factor * soft_nominal;

    println!("==============================================");
    println!("SOFTENING SCALING TEST — τ_relax(ε)");
    println!("==============================================");
    println!("  Softening factor: {:.1}× nominal", soft_factor);
    println!("  ε_nominal = {:.3} Mpc", soft_nominal);
    println!("  ε_actual  = {:.3} Mpc", softening);
    println!("  Amplitude: A = {:.1}% (fixed)", AMPLITUDE * 100.0);
    println!("  N = {}³ = {}", N_GRID, n3);
    println!("  Max steps: {}", MAX_STEPS);
    println!("==============================================\n");

    let seed = 42u64;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let output_dir = format!("/app/output/{}_{}", run_name, timestamp);
    create_dir_all(&output_dir).expect("Failed to create output dir");

    println!("Output: {}", output_dir);

    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive, _rms_disp) = generate_antisym_ics(seed);
    println!("IC generation: {:.1}s\n", start_ic.elapsed().as_secs_f64());

    let n_negative = n3 - n_positive;

    let r_k_0 = compute_r_k013(&positions, &signs, L_BOX);

    println!("=== VALIDATION TESTS ===");
    println!("  r(k=0.13, step=0) = {:.4}", r_k_0);
    println!("  N+ = {}, N- = {} (diff = {})", n_positive, n_negative,
             (n_positive as i64 - n_negative as i64).abs());
    println!();

    // Initialize simulation with custom softening
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive, n_negative, L_BOX,
        positions, velocities, signs.clone(),
    ).expect("Failed to create GPU simulation");
    sim.set_theta(THETA);
    sim.set_softening(softening);

    println!("Simulation initialized with ε = {:.3} Mpc", softening);

    // CSV
    let csv_path = format!("{}/r_k_evolution.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,time,r_k013,softening,soft_factor").unwrap();
    writeln!(csv, "0,0.0,{:.6},{:.4},{:.2}", r_k_0, softening, soft_factor).unwrap();

    println!("Step 0: r(0.13)={:.3}", r_k_0);

    let start = Instant::now();
    let mut relaxed_step = None;

    for step in 1..=MAX_STEPS {
        sim.step_with_cross_factor(DT, -1.0).expect("Step failed");

        if step % R_K_INTERVAL == 0 {
            let pos = sim.get_positions().expect("get_positions failed");
            let r_k = compute_r_k013(&pos, &signs, L_BOX);
            let elapsed = start.elapsed().as_secs_f64();

            writeln!(csv, "{},{:.2},{:.6},{:.4},{:.2}", step, elapsed, r_k, softening, soft_factor).unwrap();
            csv.flush().unwrap();

            let rate = step as f64 / elapsed;
            println!("Step {}: r(0.13)={:+.3} [{:.1} steps/s]", step, r_k, rate);

            if r_k > 0.9 && relaxed_step.is_none() {
                relaxed_step = Some(step);
                println!("  → Mode RELAXED at step {}", step);
            }
        }
    }

    println!("\n=== RUN COMPLETE ===");
    println!("  Softening: {:.3} Mpc ({:.1}× nominal)", softening, soft_factor);
    if let Some(s) = relaxed_step {
        println!("  τ_relax: {} steps", s);
    } else {
        println!("  τ_relax: > {} steps (not relaxed)", MAX_STEPS);
    }
    println!("  Output: {}/r_k_evolution.csv", output_dir);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires 'cuda' feature.");
}
