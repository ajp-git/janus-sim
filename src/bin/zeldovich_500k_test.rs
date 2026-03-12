//! Zel'dovich 500K Test - Density-based ICs + Ordre Février
//!
//! Combinaison:
//! 1. ICs Zel'dovich density-based (δ > 0 → +, δ < 0 → -)
//! 2. Ordre février: toutes particules + d'abord, puis - (brisure de symétrie)
//!
//! Paramètres: PROMPT_ZELDOVICH_THEN_BH12M.md

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// RUN 1 parameters (from PROMPT)
const N_PARTICLES: usize = 500_000;
const BOX_SIZE: f64 = 172.0;          // Mpc (n_side=79, spacing=2.18 Mpc)
const Z_INIT: f64 = 5.0;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 2000;
const SOFTENING: f64 = 0.65;          // Mpc

// P(k) spectrum
const N_S: f64 = 0.96;
const K_PEAK: f64 = 0.02;             // Mpc⁻¹

/// Generate density-based Zel'dovich ICs with ORDRE FÉVRIER
/// Returns (positions, velocities) with positives first, then negatives
fn generate_zeldovich_ics_ordered(seed: u64) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n_side = (N_PARTICLES as f64).cbrt().round() as usize;
    let n_grid = n_side * n_side * n_side;
    let spacing = BOX_SIZE / n_side as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating Zel'dovich ICs with ORDRE FÉVRIER...");
    println!("  Grid: {}³ = {} particles", n_side, n_grid);
    println!("  Box: {} Mpc, spacing: {:.3} Mpc", BOX_SIZE, spacing);
    println!("  Sign: δ > 0 → +, δ < 0 → -");
    println!("  ORDER: positives first, then negatives (février convention)");

    // Generate Fourier modes
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
                let sigma_k = pk.sqrt() * amplitude * d_growth;

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

    // Displacement field ψ = -i k δ_k / k²
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

    // Max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n_grid {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    // Density stats
    let delta_min = delta_r.iter().cloned().fold(f64::INFINITY, f64::min);
    let delta_max = delta_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_pos_delta = delta_r.iter().filter(|&&d| d > 0.0).count();
    println!("  δ_min={:.4}, δ_max={:.4}", delta_min, delta_max);
    println!("  Grid points δ > 0: {} ({:.1}%)", n_pos_delta, 100.0 * n_pos_delta as f64 / n_grid as f64);

    // Scale displacement to 30% of spacing
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Max displacement: {:.4} Mpc (30% of spacing)", target_disp);

    // Build temporary particle list with positions, velocities, signs
    let jitter_amplitude = 0.1 * spacing;
    let virial_velocity = ((n_grid as f64) / BOX_SIZE).sqrt() * 0.3;

    let wrap = |mut x: f64| {
        while x > half_box { x -= BOX_SIZE; }
        while x < -half_box { x += BOX_SIZE; }
        x
    };

    // Collect all particles with their sign
    let mut particles_pos: Vec<(f64, f64, f64, i32)> = Vec::with_capacity(n_grid);
    let mut particles_vel: Vec<(f64, f64, f64)> = Vec::with_capacity(n_grid);

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

                let x = wrap(x0 + dx);
                let y = wrap(y0 + dy);
                let z = wrap(z0 + dz);

                // Sign from density field
                let sign = if delta_r[idx] > 0.0 { 1 } else { -1 };

                particles_pos.push((x, y, z, sign));
                particles_vel.push((
                    (rng.gen::<f64>() - 0.5) * virial_velocity,
                    (rng.gen::<f64>() - 0.5) * virial_velocity,
                    (rng.gen::<f64>() - 0.5) * virial_velocity,
                ));
            }
        }
    }

    // === ORDRE FÉVRIER: Reorder so positives come first ===
    let n_positive = particles_pos.iter().filter(|(_, _, _, s)| *s > 0).count();
    let n_negative = n_grid - n_positive;

    println!("  Reordering: {} positive first, then {} negative", n_positive, n_negative);

    let mut positions = Vec::with_capacity(n_grid * 3);
    let mut velocities = Vec::with_capacity(n_grid * 3);

    // First pass: add all positive particles
    for i in 0..n_grid {
        if particles_pos[i].3 > 0 {
            positions.push(particles_pos[i].0);
            positions.push(particles_pos[i].1);
            positions.push(particles_pos[i].2);
            velocities.push(particles_vel[i].0);
            velocities.push(particles_vel[i].1);
            velocities.push(particles_vel[i].2);
        }
    }

    // Second pass: add all negative particles
    for i in 0..n_grid {
        if particles_pos[i].3 < 0 {
            positions.push(particles_pos[i].0);
            positions.push(particles_pos[i].1);
            positions.push(particles_pos[i].2);
            velocities.push(particles_vel[i].0);
            velocities.push(particles_vel[i].1);
            velocities.push(particles_vel[i].2);
        }
    }

    println!("  ✓ Ordre février: [0..{}] positive, [{}..{}] negative",
             n_positive, n_positive, n_grid);

    (positions, velocities, n_positive, n_negative)
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

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  RUN 1: Zel'dovich 500K Test (Density + Ordre Février)");
    println!("═══════════════════════════════════════════════════════════\n");

    let seed = 42u64;

    // Create output directory
    let output_dir = "/app/output/zeldovich_test_500k";
    create_dir_all(output_dir).expect("Failed to create output dir");

    println!("Output: {}", output_dir);
    println!("Parameters:");
    println!("  N = {}", N_PARTICLES);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  Steps = {}", TOTAL_STEPS);
    println!("  θ = {}, softening = {} Mpc", THETA, SOFTENING);
    println!("  dt = {}", DT);
    println!();

    // Generate ICs with ordre février
    let start_ic = Instant::now();
    let (positions, velocities, n_positive, n_negative) = generate_zeldovich_ics_ordered(seed);
    println!("\nIC generation: {:.1}s\n", start_ic.elapsed().as_secs_f64());

    // Initialize simulation using new_with_state - positives first convention
    println!("Initializing GPU simulation (new_with_state avec ordre février)...");

    // Build signs array: [+1, +1, ..., +1, -1, -1, ..., -1]
    let mut signs: Vec<i32> = Vec::with_capacity(n_positive + n_negative);
    for _ in 0..n_positive { signs.push(1); }
    for _ in 0..n_negative { signs.push(-1); }

    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        BOX_SIZE,
        positions,
        velocities,
        signs,
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);

    println!("  N = {} ({} + / {} -)", n_positive + n_negative, n_positive, n_negative);

    // Virialization with sampled PE (FIX-007)
    println!("\nVirializing (sampled, n={})...", N_PARTICLES / 200);
    let n_sample = N_PARTICLES / 200;  // 2500 for 500K
    sim.virialize_sampled(n_sample).expect("virialize_sampled failed");

    // Setup cosmology
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    // FIX-016: dtau_per_dt = tau_range / (TOTAL_STEPS × DT)
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);

    println!("\nCosmology:");
    println!("  η = {}", eta);
    println!("  dtau_per_dt = {:.6} (FIX-016)", dtau_per_dt);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke_ratio,seg").unwrap();

    // Initial state
    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = sim.segregation_distance().expect("segregation failed");

    writeln!(csv, "0,{:.4},{:.6},{:.6}", Z_INIT, 1.0, seg_0).unwrap();

    println!("\nInitial: KE₀ = {:.4e}, Seg₀ = {:.4}", ke_0, seg_0);
    println!("\n══════════════════════════════════════════════════");
    println!("  Starting simulation ({} steps)", TOTAL_STEPS);
    println!("══════════════════════════════════════════════════\n");

    let mut tau = cosmo.tau_start;
    let start = Instant::now();
    let mut seg_max = seg_0;
    let mut z_at_seg_max = Z_INIT;

    for step in 1..=TOTAL_STEPS {
        tau += dtau_per_step;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        let ke = sim.kinetic_energy().expect("kinetic_energy failed");
        let ke_ratio = ke / ke_0;
        let seg = sim.segregation_distance().expect("segregation failed");

        if seg > seg_max {
            seg_max = seg;
            z_at_seg_max = z;
        }

        writeln!(csv, "{},{:.4},{:.6},{:.6}", step, z, ke_ratio, seg).unwrap();

        // Progress every 100 steps
        if step % 100 == 0 || step <= 10 {
            println!("Step {:4}: z={:.2}, KE/KE₀={:.4}, Seg={:.4}", step, z, ke_ratio, seg);
        }

        // Early stop on explosion
        if ke_ratio > 50.0 && step > 100 {
            println!("\n❌ AUTO-STOP: KE/KE₀ = {:.1} > 50", ke_ratio);
            break;
        }
    }

    csv.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let final_ke = sim.kinetic_energy().unwrap();
    let final_seg = sim.segregation_distance().unwrap();

    println!("\n══════════════════════════════════════════════════");
    println!("  RUN 1 COMPLETE");
    println!("══════════════════════════════════════════════════");
    println!("  Runtime: {:.1}s ({:.0} ms/step)", elapsed, elapsed * 1000.0 / TOTAL_STEPS as f64);
    println!("  KE/KE₀ final: {:.4}", final_ke / ke_0);
    println!("  Seg₀ = {:.4}", seg_0);
    println!("  Seg_max = {:.4} @ z={:.2}", seg_max, z_at_seg_max);
    println!("  Seg final = {:.4}", final_seg);

    // Evaluation criteria
    println!("\n--- ÉVALUATION ---");
    if seg_max > 0.20 {
        println!("  ★ EXCEL: Seg_max > 0.20");
    } else if seg_max > 0.10 && final_ke / ke_0 < 10.0 {
        println!("  ✓ GOOD: Seg_max > 0.10 AND KE/KE₀ < 10");
    } else if seg_max > 0.05 && final_ke / ke_0 < 20.0 {
        println!("  ✓ PASS: Seg_max > 0.05 AND KE/KE₀ < 20");
    } else if (final_seg - seg_0).abs() < 0.01 {
        println!("  ⚠ FROZEN: Seg stagnant");
    } else {
        println!("  ? UNDETERMINED");
    }

    println!("\nOutput: {}", output_dir);
    println!("\n>>> Prêt pour RUN 2: BH 12M <<<");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
