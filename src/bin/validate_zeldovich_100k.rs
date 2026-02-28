//! Validation: 100K particles with Zel'dovich ICs (multi-mode FFT)
//!
//! Test criteria:
//!   - Segregation onset between z=3.0 and z=2.0
//!   - NO premature collapse (KE should stay reasonable)
//!
//! ICs: Full P(k) spectrum Zel'dovich (same as jour4_filaments.rs)
//!      NO artificial virialization - velocities from Zel'dovich approximation

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_GRID: usize = 46;  // 46³ ≈ 97K particles (closest to 100K)
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 2000;

// Power spectrum parameters (same as jour4_filaments.rs)
const N_S: f64 = 0.96;   // Spectral index
const K0: f64 = 0.02;    // Turnover scale (Mpc⁻¹)

/// 3D inverse FFT helper
fn ifft_3d(data: &mut Vec<Complex<f64>>, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;

    // Process along z
    for iz in 0..n {
        for iy in 0..n {
            let start = iz * n * n + iy * n;
            let mut row: Vec<Complex<f64>> = data[start..start+n].to_vec();
            ifft.process(&mut row);
            for ix in 0..n {
                data[start + ix] = row[ix];
            }
        }
    }

    // Process along y
    for iz in 0..n {
        for ix in 0..n {
            let mut col: Vec<Complex<f64>> = (0..n).map(|iy| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut col);
            for iy in 0..n {
                data[iz * n * n + iy * n + ix] = col[iy];
            }
        }
    }

    // Process along x
    for iy in 0..n {
        for ix in 0..n {
            let mut col: Vec<Complex<f64>> = (0..n).map(|iz| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut col);
            for iz in 0..n {
                data[iz * n * n + iy * n + ix] = col[iz];
            }
        }
    }

    // Extract real part and normalize
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

/// Generate Zel'dovich ICs with full P(k) spectrum
fn generate_zeldovich_ics(box_size: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating Zel'dovich ICs with full P(k) spectrum...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {:.1} Mpc", box_size);
    println!("  z_init = {}", Z_INIT);
    println!("  P(k) ∝ k^{} / (1 + (k/{})⁴)", N_S, K0);

    let dk = 2.0 * PI / box_size;
    let half_n = N_GRID / 2;
    let spacing = box_size / N_GRID as f64;
    let half_box = box_size / 2.0;

    // Growth factor at z_init
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate Gaussian random field in Fourier space
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

                // Power spectrum P(k) ∝ k^n_s / (1 + (k/k0)^4)
                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = pk.sqrt() * amplitude * d_growth;

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
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    // Find max displacement and scale
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Raw max displacement: {:.6e} Mpc", max_disp);

    // Scale to 30% of cell size (same as jour4_filaments.rs)
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Scaling: {:.4} → target {:.4} Mpc", scale, target_disp);

    // Zel'dovich velocities: v = D_dot * psi
    // D_dot = sqrt(1+z) in matter-dominated era
    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;
    println!("  Zel'dovich velocities: D_dot = {:.2}, vel_scale = {:.4e}", d_dot, vel_scale);

    // Generate particle data
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n3);

    // Calculate N+ and N- based on eta
    let n_positive = (n3 as f64 / (1.0 + ETA)) as usize;
    let n_negative = n3 - n_positive;

    // Random sign assignment (no spatial correlation)
    let mut sign_vec: Vec<i8> = Vec::with_capacity(n3);
    for _ in 0..n_positive { sign_vec.push(1); }
    for _ in 0..n_negative { sign_vec.push(-1); }
    sign_vec.shuffle(&mut rng);

    let mut idx = 0;
    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                // Grid position (centered box)
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let i = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                // Apply Zel'dovich displacement
                let mut x = x0 + psi_x[i] * scale;
                let mut y = y0 + psi_y[i] * scale;
                let mut z = z0 + psi_z[i] * scale;

                // Periodic wrap
                while x > half_box { x -= box_size; }
                while x < -half_box { x += box_size; }
                while y > half_box { y -= box_size; }
                while y < -half_box { y += box_size; }
                while z > half_box { z -= box_size; }
                while z < -half_box { z += box_size; }

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);

                // Zel'dovich velocities
                velocities.push((psi_x[i] * vel_scale) as f32);
                velocities.push((psi_y[i] * vel_scale) as f32);
                velocities.push((psi_z[i] * vel_scale) as f32);

                signs.push(sign_vec[idx]);
                idx += 1;
            }
        }
    }

    let actual_pos = signs.iter().filter(|&&s| s > 0).count();
    let actual_neg = signs.iter().filter(|&&s| s < 0).count();
    println!("  Generated: N+ = {}, N- = {}", actual_pos, actual_neg);

    (positions, velocities, signs, actual_pos)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Validation: Zel'dovich ICs (100K, FFT multi-mode)            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let n3 = N_GRID * N_GRID * N_GRID;
    let box_size = 100.0 * (n3 as f64 / 100_000.0).powf(1.0/3.0);
    let r_cut = box_size / 16.0;

    println!("Parameters:");
    println!("  N = {} ({}³ grid)", n3, N_GRID);
    println!("  η = {}", ETA);
    println!("  θ = {}", THETA);
    println!("  box = {:.2} Mpc", box_size);
    println!("  r_cut = {:.2} Mpc", r_cut);
    println!("  steps = {}", TOTAL_STEPS);
    println!("  ICs = Zel'dovich FFT (NO virialization)");
    println!();

    // Generate Zel'dovich ICs
    let t0 = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_zeldovich_ics(box_size, 42);
    println!("  IC generation: {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // Cosmological setup
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / 12000.0;
    let dtau_per_dt = dtau_cosmo / DT;

    let (a_init, _) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("Cosmology:");
    println!("  z_init = {:.2}", z_init_actual);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
    let output_dir = format!("/app/output/validate_zeldovich_{}", date);
    fs::create_dir_all(&output_dir)?;

    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv_file = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;

    println!("Output: {}", output_dir);
    println!();

    // Create simulation with custom ICs
    println!("Creating simulation with Zel'dovich ICs...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(positions, velocities, signs, box_size)?;
    sim.set_theta(THETA);

    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", seg0);
    println!();

    // Tracking
    let start_time = Instant::now();
    let mut step = 0usize;
    let mut current_tau = cosmo.tau_start;
    let mut s_max = 0.0f64;
    let mut s_max_step = 0usize;
    let mut s_max_z = Z_INIT;

    // Onset detection: S > 0.05 for first time
    let mut onset_step: Option<usize> = None;
    let mut onset_z: Option<f64> = None;
    const ONSET_THRESHOLD: f64 = 0.05;

    println!("Running {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE₀     Seg      ms/step");
    println!("----------------------------------------------");

    loop {
        let step_start = Instant::now();

        let (a, h) = if current_tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(current_tau)
        } else {
            (1.0, 0.0)
        };
        let z = 1.0 / a - 1.0;

        let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };

        sim.step_treepm_gpu_morton(DT, r_cut, h, dtau_eff)?;
        step += 1;
        current_tau += dtau_cosmo;

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        let ke_ratio = if ke0 > 0.0 { ke / ke0 } else { 1.0 };

        // Track S_max
        if seg > s_max {
            s_max = seg;
            s_max_step = step;
            s_max_z = z.max(0.0);
        }

        // Detect onset
        if onset_step.is_none() && seg > ONSET_THRESHOLD {
            onset_step = Some(step);
            onset_z = Some(z.max(0.0));
            println!(">>> ONSET detected: step {} (z = {:.2})", step, z.max(0.0));
        }

        // Print progress
        if step % 100 == 0 || step <= 5 {
            println!("{:5}  {:.2}  {:8.2}  {:6.4}  {:6.0}",
                step, z.max(0.0), ke_ratio, seg, step_ms);
        }

        // CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.1}",
            step, step as f64 * DT, z.max(0.0), a, h, ke, ke_ratio, seg, step_ms)?;

        if step % 100 == 0 {
            csv_file.flush()?;
        }

        if step >= TOTAL_STEPS {
            break;
        }

        // Early stop on collapse
        if ke_ratio > 500.0 {
            println!("\n⚠️ KE/KE₀ > 500 — stopping");
            break;
        }
    }

    csv_file.flush()?;

    let total_time = start_time.elapsed();

    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   VALIDATION RESULTS                                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Runtime: {:.1} minutes ({:.0} ms/step)",
        total_time.as_secs_f64() / 60.0,
        total_time.as_secs_f64() * 1000.0 / step as f64);
    println!();
    println!("Results:");
    println!("  Steps: {}", step);
    println!("  Final KE/KE₀: {:.2}", sim.kinetic_energy()? / ke0.max(1e-10));
    println!("  S_max = {:.4} at step {} (z = {:.2})", s_max, s_max_step, s_max_z);
    println!();

    // Onset check
    if let Some(os) = onset_step {
        let oz = onset_z.unwrap_or(0.0);
        let onset_ok = oz >= 2.0 && oz <= 3.0;
        println!("  Onset: step {} (z = {:.2}) {} (criterion: z ∈ [2.0, 3.0])",
            os, oz, if onset_ok { "✓" } else { "✗" });

        if onset_ok {
            println!();
            println!("✅ ZEL'DOVICH ICS VALIDATED — Ready for 85M production");
        } else if oz > 3.0 {
            println!();
            println!("⚠️ Onset too early (z > 3.0) — check amplitude scaling");
        } else {
            println!();
            println!("⚠️ Onset too late (z < 2.0) — check perturbation amplitude");
        }
    } else {
        println!("  Onset: NOT DETECTED (S never exceeded {:.2})", ONSET_THRESHOLD);
        println!();
        println!("❌ VALIDATION FAILED — Insufficient segregation");
    }

    println!();
    println!("CSV: {}", csv_path);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
    std::process::exit(1);
}
