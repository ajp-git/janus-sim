//! Test: Pure Zel'dovich velocities (α=1, no PE_binding rescaling)
//!
//! Zel'dovich ICs are self-consistent: v ∝ displacement gradient.
//! No virialization needed.
//!
//! Criteria:
//!   - KE/KE₀ < 5 at step 100
//!   - Seg < 0.01 at step 40

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

const N_GRID: usize = 46;  // 46³ ≈ 97K
const BOX_SIZE: f64 = 400.0;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 200;
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

fn ifft_3d(data: &mut [Complex<f64>], ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
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

/// Compute current kinetic energy
fn compute_ke(vel: &[f32]) -> f64 {
    let n = vel.len() / 3;
    let mut ke = 0.0_f64;
    for i in 0..n {
        let vx = vel[i * 3] as f64;
        let vy = vel[i * 3 + 1] as f64;
        let vz = vel[i * 3 + 2] as f64;
        ke += 0.5 * (vx*vx + vy*vy + vz*vz);
    }
    ke
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_zeldovich_ics(alpha: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let n_grid = N_GRID;
    let n3 = n_grid * n_grid * n_grid;
    let box_size = BOX_SIZE;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating Zel'dovich ICs (α={:.2})...", alpha);
    println!("  Grid: {}³ = {} particles", n_grid, n3);
    println!("  Box: {:.1} Mpc", box_size);

    let dk = 2.0 * PI / box_size;
    let half_n = n_grid / 2;
    let spacing = box_size / n_grid as f64;
    let half_box = box_size / 2.0;

    // Generate Fourier modes
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

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

                // P(k) ∝ k^n_s / (1 + (k/k0)^4)
                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let amplitude = pk.sqrt();

                let phase: f64 = rng.random::<f64>() * 2.0 * PI;
                delta_k[idx] = Complex::new(amplitude * phase.cos(), amplitude * phase.sin());
            }
        }
    }

    // Displacement fields
    let mut psi_x_k = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_y_k = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_z_k = vec![Complex::new(0.0, 0.0); n3];

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

    // Inverse FFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    // Find max displacement and scale
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Max displacement: {:.6e} Mpc → scaling to {:.4} Mpc", max_disp, target_disp);

    // Generate particles
    let n_positive = (n3 as f64 / (1.0 + ETA)) as usize;
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n3);

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let dx = psi_x[idx] * scale;
                let dy = psi_y[idx] * scale;
                let dz = psi_z[idx] * scale;

                let mut x = x0 + dx;
                let mut y = y0 + dy;
                let mut z = z0 + dz;

                while x > half_box { x -= box_size; }
                while x < -half_box { x += box_size; }
                while y > half_box { y -= box_size; }
                while y < -half_box { y += box_size; }
                while z > half_box { z -= box_size; }
                while z < -half_box { z += box_size; }

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);

                // Zel'dovich velocities (∝ displacement) × α
                let vx = dx * alpha;
                let vy = dy * alpha;
                let vz = dz * alpha;
                velocities.push(vx as f32);
                velocities.push(vy as f32);
                velocities.push(vz as f32);

                signs.push(1i8);
            }
        }
    }

    // Assign signs
    for i in 0..n_positive { signs[i] = 1; }
    for i in n_positive..n3 { signs[i] = -1; }
    use rand::seq::SliceRandom;
    signs.shuffle(&mut rng);

    let ke = compute_ke(&velocities);
    println!("  KE = {:.4e}", ke);

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Test: Pure Zel'dovich Velocities (α=1, no virialization)     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let alpha = 1.0;  // Pure Zel'dovich
    let (positions, velocities, signs) = generate_zeldovich_ics(alpha, 42);

    let n3 = N_GRID * N_GRID * N_GRID;
    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let n_negative = n3 - n_positive;

    println!("\nCreating simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        positions, velocities, signs, BOX_SIZE
    )?;
    sim.set_theta(THETA);
    // Softening = 0.05 × inter-particle spacing (100K in 400 Mpc → spacing 8.7 Mpc → ε ≈ 0.43)
    // But for consistency with 60M test, use same rule: 0.05 × spacing
    let spacing = BOX_SIZE / (N_GRID as f64);
    let softening = 0.05 * spacing;
    sim.set_softening(softening);
    println!("  Softening = {:.4} Mpc (0.05 × {:.2} Mpc spacing)", softening, spacing);

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (TOTAL_STEPS as f64 * DT);
    let dtau_cosmo = dtau_per_dt * DT;
    let r_cut = BOX_SIZE / 16.0;

    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", seg0);

    // Output CSV
    let mut csv = File::create("/app/output/test_zeldovich_pure.csv")?;
    writeln!(csv, "step,z,ke_ratio,seg")?;

    println!("\nRunning {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE₀     Seg");
    println!("----------------------------------");

    let mut current_tau = cosmo.tau_start;

    for step in 1..=TOTAL_STEPS {
        let (a, h) = cosmo.get_params_at_tau(current_tau);
        let z: f64 = 1.0 / a - 1.0;
        let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };

        sim.step_treepm_gpu_morton(DT, r_cut, h, dtau_eff)?;
        current_tau += dtau_cosmo;

        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        let ke_ratio = ke / ke0;

        writeln!(csv, "{},{:.4},{:.6},{:.6}", step, z.max(0.0), ke_ratio, seg)?;

        if step <= 5 || step % 20 == 0 || step == 40 || step == 100 {
            println!("{:5}   {:.3}   {:.4}   {:.4}", step, z.max(0.0), ke_ratio, seg);
        }

        // Check criteria
        if step == 40 && seg >= 0.01 {
            println!("\n⚠ FAIL: Seg={:.4} >= 0.01 at step 40", seg);
        }
        if step == 100 && ke_ratio >= 5.0 {
            println!("\n⚠ FAIL: KE/KE₀={:.2} >= 5 at step 100", ke_ratio);
        }
    }

    // Final report
    let ke_final = sim.kinetic_energy()?;
    let seg_final = sim.segregation()?;
    let ke_ratio_final = ke_final / ke0;

    println!("\n════════════════════════════════════════════");
    println!("RESULTS (α={:.2}):", alpha);
    println!("  KE/KE₀ @ step 100: check CSV");
    println!("  Seg @ step 40: check CSV");
    println!("  Final (step {}): KE/KE₀={:.4}, Seg={:.4}", TOTAL_STEPS, ke_ratio_final, seg_final);
    println!("════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
}
