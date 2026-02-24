//! Scaling Test — τ_relax(N) measurement
//!
//! Tests if relaxation time τ_relax scales with N (numerical artifact)
//! or is constant (physical property).
//!
//! Usage:
//!   cargo run --release --features cuda --bin scaling_test -- --n 500000 --name scale_500K
//!   cargo run --release --features cuda --bin scaling_test -- --n 2000000 --name scale_2M
//!
//! Measures r(k,t) = P₊₋(k) / √(P₊₊·P₋₋) for k = 0.05, 0.13, 0.30, 0.63 Mpc⁻¹

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::env;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

// Physical parameters
const L_BOX: f64 = 400.0;      // Mpc
const Z_INIT: f64 = 10.0;      // Initial redshift

// Simulation parameters
const DT: f64 = 0.005;
const R_K_INTERVAL: usize = 50;      // Measure r(k) every 50 steps
const SNAPSHOT_INTERVAL: usize = 200;
const THETA: f64 = 0.7;

// Power spectrum
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

// Target k values for r(k,t) measurement (Mpc⁻¹)
const K_TARGETS: [f64; 4] = [0.05, 0.13, 0.30, 0.63];

/// Compute N_GRID from total particles (cubic root)
fn compute_n_grid(n_total: usize) -> usize {
    let n_grid = (n_total as f64).powf(1.0/3.0).round() as usize;
    // Ensure it's reasonable
    n_grid.max(32).min(256)
}

/// Generate antisymmetric Zel'dovich ICs
fn generate_antisym_ics(n_grid: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = StdRng::seed_from_u64(seed);
    let spacing = L_BOX / n_grid as f64;
    let half_box = L_BOX / 2.0;
    let dk = 2.0 * PI / L_BOX;
    let half_n = n_grid / 2;

    println!("Generating ANTISYMMETRIC ICs...");
    println!("  Grid: {}³ = {} particles", n_grid, n3);

    // Growth factor
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate Gaussian random field
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 0.01;

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

                if k < 1e-10 { continue; }

                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = pk.sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Hermitian symmetry
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

    // Displacement field
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

    // IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);
    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    // Scale displacement
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 0.0 { target_disp / max_disp } else { 1.0 };

    // Velocities
    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;

    // Assign signs first
    let mut signs: Vec<i32> = Vec::with_capacity(n3);
    let mut n_positive = 0usize;
    for _ in 0..n3 {
        let sign = if rng.random::<bool>() { 1 } else { -1 };
        if sign > 0 { n_positive += 1; }
        signs.push(sign);
    }

    // Place particles with antisymmetric displacements
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let sign_factor = signs[idx] as f64;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // ANTISYMMETRIC: m+ gets +ψ, m- gets -ψ
                positions.push(x0 + psi_x[idx] * scale * sign_factor);
                positions.push(y0 + psi_y[idx] * scale * sign_factor);
                positions.push(z0 + psi_z[idx] * scale * sign_factor);

                velocities.push(psi_x[idx] * vel_scale * sign_factor);
                velocities.push(psi_y[idx] * vel_scale * sign_factor);
                velocities.push(psi_z[idx] * vel_scale * sign_factor);
            }
        }
    }

    println!("  N+ = {}, N- = {}", n_positive, n3 - n_positive);
    (positions, velocities, signs, n_positive)
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

/// Compute r(k) for target k values
fn compute_r_k_targets(positions: &[f64], signs: &[i32], box_size: f64) -> [f64; 4] {
    let n_grid = 64;  // Fixed grid for r(k) measurement
    let n = positions.len() / 3;
    let cell_size = box_size / n_grid as f64;

    // Grid densities
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

    // Overdensity
    let mean_plus: f64 = density_plus.iter().sum::<f64>() / density_plus.len() as f64;
    let mean_minus: f64 = density_minus.iter().sum::<f64>() / density_minus.len() as f64;

    let delta_plus: Vec<f64> = density_plus.iter()
        .map(|&d| if mean_plus > 0.0 { (d - mean_plus) / mean_plus } else { 0.0 })
        .collect();
    let delta_minus: Vec<f64> = density_minus.iter()
        .map(|&d| if mean_minus > 0.0 { (d - mean_minus) / mean_minus } else { 0.0 })
        .collect();

    // FFT
    let mut fft_plus: Vec<Complex<f64>> = delta_plus.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut fft_minus: Vec<Complex<f64>> = delta_minus.iter().map(|&x| Complex::new(x, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_grid);

    // 3D FFT
    for iy in 0..n_grid {
        for ix in 0..n_grid {
            let mut slice: Vec<Complex<f64>> = (0..n_grid).map(|iz| fft_plus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            fft.process(&mut slice);
            for iz in 0..n_grid { fft_plus[iz * n_grid * n_grid + iy * n_grid + ix] = slice[iz]; }

            let mut slice: Vec<Complex<f64>> = (0..n_grid).map(|iz| fft_minus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            fft.process(&mut slice);
            for iz in 0..n_grid { fft_minus[iz * n_grid * n_grid + iy * n_grid + ix] = slice[iz]; }
        }
    }
    for iz in 0..n_grid {
        for ix in 0..n_grid {
            let mut slice: Vec<Complex<f64>> = (0..n_grid).map(|iy| fft_plus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            fft.process(&mut slice);
            for iy in 0..n_grid { fft_plus[iz * n_grid * n_grid + iy * n_grid + ix] = slice[iy]; }

            let mut slice: Vec<Complex<f64>> = (0..n_grid).map(|iy| fft_minus[iz * n_grid * n_grid + iy * n_grid + ix]).collect();
            fft.process(&mut slice);
            for iy in 0..n_grid { fft_minus[iz * n_grid * n_grid + iy * n_grid + ix] = slice[iy]; }
        }
    }
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            let base = iz * n_grid * n_grid + iy * n_grid;
            let mut slice_p: Vec<Complex<f64>> = fft_plus[base..base+n_grid].to_vec();
            let mut slice_m: Vec<Complex<f64>> = fft_minus[base..base+n_grid].to_vec();
            fft.process(&mut slice_p);
            fft.process(&mut slice_m);
            for ix in 0..n_grid {
                fft_plus[base + ix] = slice_p[ix];
                fft_minus[base + ix] = slice_m[ix];
            }
        }
    }

    // Compute k grid
    let dk = 2.0 * PI / box_size;
    let half_n = n_grid / 2;

    // For each target k, find matching cells and compute r(k)
    let mut r_k_results = [0.0f64; 4];

    for (ki, &k_target) in K_TARGETS.iter().enumerate() {
        let mut p_pp_sum = 0.0f64;
        let mut p_mm_sum = 0.0f64;
        let mut p_pm_sum = 0.0f64;
        let mut count = 0usize;

        let k_width = 0.03;  // Bin width

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

                    if (k - k_target).abs() < k_width {
                        let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                        let fp = fft_plus[idx];
                        let fm = fft_minus[idx];

                        p_pp_sum += fp.norm_sqr();
                        p_mm_sum += fm.norm_sqr();
                        p_pm_sum += (fp * fm.conj()).re;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 && p_pp_sum > 0.0 && p_mm_sum > 0.0 {
            r_k_results[ki] = p_pm_sum / (p_pp_sum * p_mm_sum).sqrt();
        }
    }

    r_k_results
}

fn write_snapshot(path: &str, positions: &[f64], signs: &[i32], step: usize, a: f64, seg: f64) -> std::io::Result<()> {
    let n = positions.len() / 3;
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(&(n as u64).to_le_bytes())?;
    file.write_all(&(step as u64).to_le_bytes())?;
    file.write_all(&a.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    for i in 0..n {
        file.write_all(&(positions[i*3] as f32).to_le_bytes())?;
        file.write_all(&(positions[i*3+1] as f32).to_le_bytes())?;
        file.write_all(&(positions[i*3+2] as f32).to_le_bytes())?;
        file.write_all(&(signs[i] as i8).to_le_bytes())?;
    }
    Ok(())
}

fn compute_segregation(positions: &[f64], signs: &[i32]) -> f64 {
    let n = positions.len() / 3;
    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let mut sum_pos = [0.0f64; 3];
    let mut sum_neg = [0.0f64; 3];

    for i in 0..n {
        if signs[i] > 0 {
            sum_pos[0] += positions[i*3];
            sum_pos[1] += positions[i*3+1];
            sum_pos[2] += positions[i*3+2];
        } else {
            sum_neg[0] += positions[i*3];
            sum_neg[1] += positions[i*3+1];
            sum_neg[2] += positions[i*3+2];
        }
    }

    let n_pos = n_positive as f64;
    let n_neg = (n - n_positive) as f64;
    let dx = sum_pos[0]/n_pos - sum_neg[0]/n_neg;
    let dy = sum_pos[1]/n_pos - sum_neg[1]/n_neg;
    let dz = sum_pos[2]/n_pos - sum_neg[2]/n_neg;
    (dx*dx + dy*dy + dz*dz).sqrt()
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut n_total = 500_000usize;
    let mut run_name = "scale_test".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { n_total = args[i+1].parse().expect("Invalid N"); i += 2; }
            "--name" => { run_name = args[i+1].clone(); i += 2; }
            _ => { i += 1; }
        }
    }

    let n_grid = compute_n_grid(n_total);
    let n3 = n_grid * n_grid * n_grid;

    // τ_coll estimate
    let tau_coll = n3 as f64 / (n3 as f64).ln();

    println!("==============================================");
    println!("SCALING TEST — τ_relax(N) measurement");
    println!("==============================================");
    println!("  N = {} (grid {}³ = {})", n_total, n_grid, n3);
    println!("  τ_coll ~ N/ln(N) = {:.0} steps", tau_coll);
    println!("  If τ_relax ~ 2000 << τ_coll → PHYSICAL");
    println!("  If τ_relax ∝ N   → NUMERICAL ARTIFACT");
    println!("==============================================\n");

    let seed = 42u64;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let output_dir = format!("/app/output/{}_{}", run_name, timestamp);
    create_dir_all(&output_dir).expect("Failed to create output dir");
    let snap_dir = format!("{}/snapshots", output_dir);
    create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    println!("Output: {}", output_dir);

    // Generate ICs
    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_antisym_ics(n_grid, seed);
    println!("IC generation: {:.1}s\n", start_ic.elapsed().as_secs_f64());

    // Validation: r(k=0.13, step=0) should be -1
    let r_k_0 = compute_r_k_targets(&positions, &signs, L_BOX);
    println!("=== VALIDATION TEST ===");
    println!("  r(k=0.05, t=0) = {:.4}", r_k_0[0]);
    println!("  r(k=0.13, t=0) = {:.4}", r_k_0[1]);
    println!("  r(k=0.30, t=0) = {:.4}", r_k_0[2]);
    println!("  r(k=0.63, t=0) = {:.4}", r_k_0[3]);

    if r_k_0[1] > -0.95 {
        println!("  WARNING: r(k=0.13) != -1.0 → ICs may not be properly antisymmetric!");
    } else {
        println!("  PASS: r(k=0.13) ≈ -1.0 → antisymmetric ICs confirmed");
    }
    println!();

    // Initialize simulation
    let n_negative = n3 - n_positive;
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive, n_negative, L_BOX,
        positions, velocities, signs.clone(),
    ).expect("Failed to create GPU simulation");
    sim.set_theta(THETA);

    // CSV with r(k) columns
    let csv_path = format!("{}/r_k_evolution.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,time,r_k005,r_k013,r_k030,r_k063,segregation").unwrap();

    // Initial
    let pos = sim.get_positions().expect("get_positions failed");
    let seg = compute_segregation(&pos, &signs);
    writeln!(csv, "0,0.0,{:.6},{:.6},{:.6},{:.6},{:.4}",
             r_k_0[0], r_k_0[1], r_k_0[2], r_k_0[3], seg).unwrap();

    // Write initial snapshot
    write_snapshot(&format!("{}/snap_{:05}.bin", snap_dir, 0), &pos, &signs, 0, 1.0/(1.0+Z_INIT), seg).ok();

    println!("Step 0: r(0.13)={:.3}, S={:.2} Mpc", r_k_0[1], seg);

    // Main loop
    let start = Instant::now();
    let mut step = 0usize;

    loop {
        step += 1;
        sim.step_with_cross_factor(DT, -1.0).expect("Step failed");

        if step % R_K_INTERVAL == 0 {
            let pos = sim.get_positions().expect("get_positions failed");
            let r_k = compute_r_k_targets(&pos, &signs, L_BOX);
            let seg = compute_segregation(&pos, &signs);
            let elapsed = start.elapsed().as_secs_f64();

            writeln!(csv, "{},{:.2},{:.6},{:.6},{:.6},{:.6},{:.4}",
                     step, elapsed, r_k[0], r_k[1], r_k[2], r_k[3], seg).unwrap();
            csv.flush().unwrap();

            let rate = step as f64 / elapsed;
            println!("Step {}: r(0.13)={:+.3}, S={:.2} Mpc [{:.1} steps/s]",
                     step, r_k[1], seg, rate);

            // Write snapshot
            if step % SNAPSHOT_INTERVAL == 0 {
                let a = 1.0/(1.0+Z_INIT) + step as f64 * DT * 0.01;
                write_snapshot(&format!("{}/snap_{:05}.bin", snap_dir, step), &pos, &signs, step, a, seg).ok();
            }

            // Report transition
            if r_k[1] > 0.0 && (step - R_K_INTERVAL..step).contains(&(step - R_K_INTERVAL)) {
                println!("\n*** r(k=0.13) crossed 0 at step {} ***\n", step);
            }

            // Status every 500 steps
            if step % 500 == 0 {
                println!("\n=== STATUS at step {} ===", step);
                println!("  r(k=0.05) = {:+.4}", r_k[0]);
                println!("  r(k=0.13) = {:+.4}", r_k[1]);
                println!("  r(k=0.30) = {:+.4}", r_k[2]);
                println!("  r(k=0.63) = {:+.4}", r_k[3]);
                println!("  S = {:.2} Mpc", seg);
                println!("  τ_coll estimate = {:.0} steps", tau_coll);

                // Check if relaxation complete
                if r_k[1] > 0.9 {
                    println!("  → Mode relaxed (r > 0.9) at step {}", step);
                    println!("  → τ_relax ≈ {} steps", step);
                    println!("  → τ_relax/τ_coll = {:.4}", step as f64 / tau_coll);
                }
                println!();
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires 'cuda' feature. Use: cargo run --release --features cuda --bin scaling_test");
}
