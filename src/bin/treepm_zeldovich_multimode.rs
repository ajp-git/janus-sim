//! TreePM + Multi-mode Zel'dovich ICs validation test
//!
//! Uses full P(k) spectrum FFT-based Zel'dovich (like jour4_filaments.rs)
//! with TreePM integrator for 100K particles.
//!
//! Criterion: Segregation onset between z=3.0 and z=2.0

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

const N_GRID: usize = 46;  // 46³ ≈ 97K particles
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 2000;

// Power spectrum parameters (same as jour4_filaments.rs)
const N_S: f64 = 0.96;   // Spectral index
const K0: f64 = 0.02;    // Turnover scale (Mpc⁻¹)
const VIRIAL_FACTOR: f64 = 0.8;  // For virialized velocities

/// 3D inverse FFT (from jour4_filaments.rs)
fn ifft_3d(data: &mut [Complex<f64>], ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
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

/// Generate full multi-mode Zel'dovich ICs with P(k) spectrum
/// Adapted from jour4_filaments.rs for TreePM
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_zeldovich_ics_full(box_size: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating multi-mode Zel'dovich ICs with P(k) spectrum...");
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
    println!("  Generating Fourier modes...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Amplitude normalization
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

    let max_delta_k = delta_k.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("  DEBUG: max(|delta_k|) = {:.6e}", max_delta_k);

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

    // Find max displacement
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  DEBUG: max displacement after IFFT = {:.6e} Mpc", max_disp);

    // Scale to 30% of cell size
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Scaling: {:.4} → target {:.4} Mpc", scale, target_disp);

    // Virialized velocities (random, scaled by virial_factor)
    // This works better than Zel'dovich velocities (avoids KE explosion)
    let virial_velocity = ((n3 as f64) / box_size).sqrt() * VIRIAL_FACTOR;
    println!("  Virialized velocities: virial_velocity = {:.4} (factor = {:.2})",
        virial_velocity, VIRIAL_FACTOR);

    // Generate particles
    let n_positive = (n3 as f64 / (1.0 + ETA)) as usize;

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // Apply Zel'dovich displacement
                let mut x = x0 + psi_x[idx] * scale;
                let mut y = y0 + psi_y[idx] * scale;
                let mut z = z0 + psi_z[idx] * scale;

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

                // Virialized velocities (random, scaled by virial_velocity)
                let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
                let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
                let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
                velocities.push(vx as f32);
                velocities.push(vy as f32);
                velocities.push(vz as f32);

                signs.push(1i8);  // Will be reassigned
            }
        }
    }

    // Assign signs based on eta (shuffle for random spatial distribution)
    for i in 0..n_positive {
        signs[i] = 1;
    }
    for i in n_positive..n3 {
        signs[i] = -1;
    }
    signs.shuffle(&mut rng);

    let actual_pos = signs.iter().filter(|&&s| s > 0).count();
    let actual_neg = signs.iter().filter(|&&s| s < 0).count();

    // Print max velocity
    let max_v = velocities.chunks(3)
        .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
        .fold(0.0f32, |a, b| a.max(b));
    println!("  Max velocity: {:.4}", max_v);
    println!("  Generated: N+ = {}, N- = {}", actual_pos, actual_neg);

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   TreePM + Multi-mode Zel'dovich ICs (P(k) spectrum)           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_total = N_GRID * N_GRID * N_GRID;
    let box_size = 100.0 * (n_total as f64 / 100_000.0).powf(1.0/3.0);
    let r_cut = box_size / 16.0;

    println!("Parameters:");
    println!("  N = {} ({}³ grid)", n_total, N_GRID);
    println!("  η = {}", ETA);
    println!("  θ = {}", THETA);
    println!("  box = {:.2} Mpc", box_size);
    println!("  r_cut = {:.2} Mpc", r_cut);
    println!("  steps = {}", TOTAL_STEPS);
    println!("  ICs = Multi-mode Zel'dovich P(k)");
    println!();

    // Generate ICs
    let t0 = Instant::now();
    let (positions, velocities, signs) = generate_zeldovich_ics_full(box_size, 42);
    println!("  IC generation: {:.2}s\n", t0.elapsed().as_secs_f64());

    // Cosmology setup
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = dtau_cosmo / DT;

    println!("Cosmology:");
    println!("  z_init = {:.2}", Z_INIT);
    println!("  dτ/dt = {:.6}\n", dtau_per_dt);

    // Output directory
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S");
    let output_dir = format!("/app/output/treepm_multimode_{}", timestamp);
    fs::create_dir_all(&output_dir)?;
    println!("Output: {}\n", output_dir);

    // Create simulation
    println!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(positions, velocities, signs, box_size)?;
    sim.set_theta(THETA);

    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation()?;
    println!("  KE₀ = {:.4e}", ke_init);
    println!("  S₀ = {:.6}\n", seg_init);

    // Time series file
    let ts_path = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_path)?);
    writeln!(ts_file, "step,z,ke,ke_ratio,segregation,ms_per_step")?;
    writeln!(ts_file, "0,{:.4},{:.6e},1.0,{:.6},0", Z_INIT, ke_init, seg_init)?;

    println!("Running {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE_ref    Seg      ms/step");
    println!("-------------------------------------------------");

    let mut tau = cosmo.tau_start;
    let mut ke_ref: Option<f64> = None;
    let mut seg_max = 0.0f64;
    let mut step_at_seg_max = 0;
    let mut z_at_seg_max = Z_INIT;
    let mut onset_step: Option<usize> = None;
    let mut onset_z: Option<f64> = None;

    let start = Instant::now();

    for step in 1..=TOTAL_STEPS {
        let t_step = Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_treepm_gpu_morton(DT, r_cut, h, dtau_per_dt)?;
        tau += dtau_per_dt * DT;

        let step_ms = t_step.elapsed().as_millis();

        if step % 10 == 0 || step <= 10 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;

            if ke_ref.is_none() && ke > 1e-10 {
                ke_ref = Some(ke);
                println!(">>> KE_ref set to {:.4e} at step {}", ke, step);
            }

            let ke_ratio = ke_ref.map(|r| ke / r).unwrap_or(1.0);

            if seg > seg_max {
                seg_max = seg;
                step_at_seg_max = step;
                z_at_seg_max = z;
            }

            if onset_step.is_none() && seg > 0.05 {
                onset_step = Some(step);
                onset_z = Some(z);
                println!(">>> ONSET: S > 0.05 at step {} (z = {:.2})", step, z);
            }

            writeln!(ts_file, "{},{:.4},{:.6e},{:.2},{:.6},{}",
                     step, z, ke, ke_ratio, seg, step_ms)?;
            ts_file.flush()?;

            if step <= 10 || step % 100 == 0 {
                println!("{:5}  {:.2}    {:8.2}  {:.4}     {:4}",
                         step, z, ke_ratio, seg, step_ms);
            }

            if ke_ratio > 1000.0 {
                println!("\n⚠️ KE/KE_ref > 1000 — stopping\n");
                break;
            }
        }
    }

    let runtime_min = start.elapsed().as_secs_f64() / 60.0;
    let ms_per_step = start.elapsed().as_millis() as f64 / TOTAL_STEPS as f64;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║   VALIDATION RESULTS                                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Runtime: {:.1} minutes ({:.0} ms/step)", runtime_min, ms_per_step);
    println!();
    println!("Results:");
    println!("  S_max = {:.4} at step {} (z = {:.2})", seg_max, step_at_seg_max, z_at_seg_max);

    if let (Some(step), Some(z)) = (onset_step, onset_z) {
        println!("  Onset: step {} (z = {:.2})", step, z);

        if z >= 2.0 && z <= 3.0 {
            println!("\n✅ VALIDATION PASSED — onset z = {:.2} ∈ [2.0, 3.0]", z);
            println!("   Multi-mode P(k) ICs ready for 85M");
        } else if z > 3.0 {
            println!("\n⚠️ WARNING — onset too early (z = {:.2} > 3.0)", z);
        } else {
            println!("\n⚠️ WARNING — onset too late (z = {:.2} < 2.0)", z);
        }
    } else {
        println!("  Onset: NOT DETECTED (S never exceeded 0.05)");
        println!("\n❌ VALIDATION FAILED — Insufficient segregation");
    }

    println!("\nCSV: {}", ts_path);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires 'cuda' and 'cufft' features:");
    eprintln!("  cargo run --release --features cuda,cufft --bin treepm_zeldovich_multimode");
}
