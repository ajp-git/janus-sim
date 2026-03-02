//! Anti-correlated Zel'dovich ICs for Janus simulation
//!
//! Sign assignment based on local density field δ(x):
//!   - δ > 0 (overdense) → positive mass
//!   - δ < 0 (underdense) → negative mass
//! Creates spatial segregation: + in clusters, - in voids
//!
//! Paramètres roadmap FILAMENTS_ROADMAP.md

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Phase B parameters (from FILAMENTS_ROADMAP.md)
const N_PARTICLES: usize = 500_000;
const BOX_SIZE: f64 = 100.0;          // Mpc
const Z_INIT: f64 = 5.0;              // Same as validated runs
const THETA: f64 = 0.7;               // FIX-012 validated
const DT: f64 = 0.01;                 // Same as 2M reference
const TOTAL_STEPS: usize = 2000;
const SOFTENING: f64 = 0.3;           // 0.3 × spacing (spacing≈1.26 Mpc for 500K)

// P(k) spectrum
const N_S: f64 = 0.96;                // Spectral index
const K_PEAK: f64 = 0.02;             // Mpc⁻¹, coherence scale ~50 Mpc

// Output intervals
const RENDER_INTERVAL: usize = 20;
const CSV_INTERVAL: usize = 1;

/// Generate anti-correlated Zel'dovich ICs
/// Returns (positions, velocities, signs, n_positive)
///
/// KEY APPROACH: Assign particle signs based on local density field δ(x):
///   - Where δ > 0 (overdensity) → positive mass
///   - Where δ < 0 (underdensity) → negative mass
/// This creates genuine spatial segregation: + in clusters, - in voids.
fn generate_anticorr_zeldovich_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    // Grid for ~100K particles: 46³ = 97336 ≈ 100K
    let n_side = (N_PARTICLES as f64).cbrt().round() as usize;
    let n_grid = n_side * n_side * n_side;
    let spacing = BOX_SIZE / n_side as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating DENSITY-BASED anti-correlated Zel'dovich ICs...");
    println!("  Grid: {}³ = {} particles", n_side, n_grid);
    println!("  Sign assignment: + where δ > 0, - where δ < 0");
    println!("  Box: {} Mpc, spacing: {:.3} Mpc", BOX_SIZE, spacing);
    println!("  z_init = {}", Z_INIT);
    println!("  P(k) ∝ k^{} / (1 + (k/{})⁴)", N_S, K_PEAK);

    // 2. Generate density field δ(k) via FFT
    let dk = 2.0 * PI / BOX_SIZE;
    let half_n = n_side / 2;

    // Growth factor at z_init
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    println!("  Generating Fourier modes...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_grid];
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Amplitude increased 100x: original 0.01 gave max displacement 6e-8 Mpc (too small)
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
                    delta_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }

                // P(k) ∝ k^n_s / (1 + (k/k_peak)^4)
                let pk = k.powf(N_S) / (1.0 + (k / K_PEAK).powi(4));
                let sigma_k = (pk).sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Enforce Hermitian symmetry
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
    println!("  Computing displacement fields...");
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

    // Inverse FFT for displacement field AND density field
    println!("  Performing inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_side);

    // Clone delta_k for density field computation
    let mut delta_k_copy = delta_k.clone();

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_side);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_side);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_side);
    let delta_r = ifft_3d(&mut delta_k_copy, &ifft, n_side);  // Real-space density

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n_grid {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Max raw displacement: {:.6e} Mpc", max_disp);

    // Report density field statistics
    let delta_min = delta_r.iter().cloned().fold(f64::INFINITY, f64::min);
    let delta_max = delta_r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_pos_delta = delta_r.iter().filter(|&&d| d > 0.0).count();
    println!("  Density field: δ_min={:.4}, δ_max={:.4}", delta_min, delta_max);
    println!("  Grid points with δ > 0: {} ({:.1}%)", n_pos_delta, 100.0 * n_pos_delta as f64 / n_grid as f64);

    // Scale to 30% of cell size (from roadmap)
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Scaling: target {:.4} Mpc (30% of spacing)", target_disp);

    // 3. Generate positions with DENSITY-BASED sign assignment
    // Sign determined by local δ: + where δ > 0 (overdense), - where δ < 0 (underdense)
    println!("  Placing particles with DENSITY-BASED sign assignment...");
    let mut positions = Vec::with_capacity(n_grid * 3);
    let mut velocities = Vec::with_capacity(n_grid * 3);
    let mut signs = Vec::with_capacity(n_grid);

    let jitter_amplitude = 0.1 * spacing;

    // Initial velocity scale (will be rescaled by virialize_sampled)
    let virial_velocity = ((n_grid as f64) / BOX_SIZE).sqrt() * 0.3;

    // Periodic wrap function
    let wrap = |mut x: f64| {
        while x > half_box { x -= BOX_SIZE; }
        while x < -half_box { x += BOX_SIZE; }
        x
    };

    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                let idx = iz * n_side * n_side + iy * n_side + ix;

                // Grid position centered [-box/2, +box/2] (FIX-008)
                let jitter_x = (rng.gen::<f64>() - 0.5) * jitter_amplitude;
                let jitter_y = (rng.gen::<f64>() - 0.5) * jitter_amplitude;
                let jitter_z = (rng.gen::<f64>() - 0.5) * jitter_amplitude;

                let x0 = (ix as f64 + 0.5) * spacing - half_box + jitter_x;
                let y0 = (iy as f64 + 0.5) * spacing - half_box + jitter_y;
                let z0 = (iz as f64 + 0.5) * spacing - half_box + jitter_z;

                // Displacement at this grid point (Zel'dovich)
                let dx = psi_x[idx] * scale;
                let dy = psi_y[idx] * scale;
                let dz = psi_z[idx] * scale;

                // Final position (displaced by ψ toward overdensities)
                positions.push(wrap(x0 + dx));
                positions.push(wrap(y0 + dy));
                positions.push(wrap(z0 + dz));

                // SIGN determined by local density field δ
                // δ > 0 → overdense region → positive mass
                // δ < 0 → underdense region → negative mass
                let sign = if delta_r[idx] > 0.0 { 1 } else { -1 };
                signs.push(sign);

                // Random initial velocities (will be rescaled by virialize_sampled)
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
                velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
            }
        }
    }

    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let n_negative = n_grid - n_positive;
    println!("  Final count: {} positive, {} negative ({:.1}%/{})",
             n_positive, n_negative,
             100.0 * n_positive as f64 / n_grid as f64,
             100.0 * n_negative as f64 / n_grid as f64);

    // Test 1: Verify positions are centered (from roadmap)
    let pos_max = positions.iter().map(|&p| p.abs()).fold(0.0f64, |a, b| a.max(b));
    assert!(pos_max <= half_box + 0.001,
        "Position out of bounds: {:.3} > {:.3}", pos_max, half_box);
    println!("  ✓ Position bounds check passed: max |pos| = {:.3} Mpc", pos_max);

    // Test 2: Verify spatial separation + and - (from roadmap)
    let (com_plus, com_minus) = compute_coms(&positions, &signs);
    let separation = ((com_plus[0]-com_minus[0]).powi(2) +
                     (com_plus[1]-com_minus[1]).powi(2) +
                     (com_plus[2]-com_minus[2]).powi(2)).sqrt();
    println!("  COM+: ({:.3}, {:.3}, {:.3})", com_plus[0], com_plus[1], com_plus[2]);
    println!("  COM-: ({:.3}, {:.3}, {:.3})", com_minus[0], com_minus[1], com_minus[2]);
    println!("  Initial COM separation: {:.4} Mpc ({:.2}% of box)",
             separation, separation / BOX_SIZE * 100.0);

    // Calculate initial segregation
    let seg_0 = separation / BOX_SIZE;
    println!("  Initial Seg₀ = {:.4}", seg_0);

    if seg_0 < 0.01 {
        println!("  ⚠️  WARNING: Seg₀ < 0.01 - anti-correlation may be too weak");
    } else {
        println!("  ✓ Anti-correlation effective: Seg₀ > 0.01");
    }

    (positions, velocities, signs, n_positive)
}

/// 3D inverse FFT
fn ifft_3d(data: &mut Vec<Complex<f64>>, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
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

/// Compute segregation (COM distance / box_size)
fn compute_segregation(positions: &[f64], signs: &[i32]) -> f64 {
    let (com_pos, com_neg) = compute_coms(positions, signs);
    let dx = com_pos[0] - com_neg[0];
    let dy = com_pos[1] - com_neg[1];
    let dz = com_pos[2] - com_neg[2];
    (dx*dx + dy*dy + dz*dz).sqrt() / BOX_SIZE
}

/// Write render_data binary (for visualization)
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

    // Header
    file.write_all(&(step as u32).to_le_bytes())?;
    file.write_all(&box_size.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    file.write_all(&ke_ratio.to_le_bytes())?;
    file.write_all(&redshift.to_le_bytes())?;
    file.write_all(&(n as u32).to_le_bytes())?;

    // Positions (f32)
    for i in 0..n {
        file.write_all(&(positions[i * 3] as f32).to_le_bytes())?;
        file.write_all(&(positions[i * 3 + 1] as f32).to_le_bytes())?;
        file.write_all(&(positions[i * 3 + 2] as f32).to_le_bytes())?;
    }

    // Signs (i8)
    for i in 0..n {
        file.write_all(&(signs[i] as i8).to_le_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Anti-correlated Zel'dovich ICs ({}K particles)", N_PARTICLES / 1000);
    println!("═══════════════════════════════════════════════════════════\n");

    let seed = 42u64;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Create output directory
    let output_dir = format!("/app/output/anticorr_{}k_{}", N_PARTICLES / 1000, timestamp);
    create_dir_all(&output_dir).expect("Failed to create output dir");
    let render_dir = format!("{}/render_data", output_dir);
    create_dir_all(&render_dir).expect("Failed to create render_data dir");

    println!("Output: {}", output_dir);

    // Generate ICs
    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_anticorr_zeldovich_ics(seed);
    println!("\nIC generation: {:.1}s\n", start_ic.elapsed().as_secs_f64());

    let n_total = signs.len();
    let n_negative = n_total - n_positive;

    // Initialize simulation
    println!("Initializing GPU simulation...");
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

    println!("  N = {} ({} + / {} -)", n_total, n_positive, n_negative);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  θ = {}", THETA);
    println!("  softening = {} Mpc", SOFTENING);
    println!("  dt = {}", DT);
    println!("  steps = {}", TOTAL_STEPS);

    // Virialization using PE_binding (FIX-007)
    println!("\nVirializing with PE_binding method...");
    let n_sample = (n_total / 100).max(1000).min(10000);
    sim.virialize_sampled(n_sample).expect("virialize_sampled failed");
    // Note: virialize_sampled prints α internally
    // α is computed as sqrt(|PE_binding|/2KE) and typically 4-6
    println!("  ✓ Virialization complete");

    // Setup cosmology
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;

    println!("\nCosmology:");
    println!("  η = {}", eta);
    println!("  z_init = {}, z_final = 0", Z_INIT);
    println!("  τ_start = {:.4}, τ_end = {:.4}", cosmo.tau_start, cosmo.tau_end);
    println!("  dτ/step = {:.6}", dtau);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke_ratio,seg,step_ms").unwrap();

    // Initial state
    let pos = sim.get_positions().expect("get_positions failed");
    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = compute_segregation(&pos, &signs);
    let z_0 = Z_INIT;

    writeln!(csv, "0,{:.4},{:.6},{:.6},0", z_0, 1.0, seg_0).unwrap();

    // Write initial render_data
    let render_path = format!("{}/step_{:06}.bin", render_dir, 0);
    write_render_data(&render_path, &pos, &signs, 0, BOX_SIZE, seg_0, 1.0, z_0)
        .expect("Failed to write render_data");

    println!("\n══════════════════════════════════════════════════");
    println!("  Starting simulation (Phase A criteria check)");
    println!("══════════════════════════════════════════════════\n");
    println!("Step 0: z={:.2}, KE/KE₀=1.000, Seg={:.4}", z_0, seg_0);

    let mut tau = cosmo.tau_start;
    let start = Instant::now();
    let mut ke_ratio_max = 1.0f64;
    let mut seg_max = seg_0;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        // Get cosmological parameters
        tau += dtau;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        // Step with Hubble friction
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau)
            .expect("Step failed");

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        // Compute metrics
        let ke = sim.kinetic_energy().expect("kinetic_energy failed");
        let ke_ratio = ke / ke_0;
        let pos = sim.get_positions().expect("get_positions failed");
        let seg = compute_segregation(&pos, &signs);

        ke_ratio_max = ke_ratio_max.max(ke_ratio);
        seg_max = seg_max.max(seg);

        // Write CSV
        writeln!(csv, "{},{:.4},{:.6},{:.6},{:.1}", step, z, ke_ratio, seg, step_ms).unwrap();

        // === PHASE A CRITERIA CHECKS ===

        // Step 5 check (FAIL-A1)
        if step == 5 {
            println!("\n=== STEP 5 CHECK ===");
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            if ke_ratio > 10.0 {
                println!("  ❌ FAIL-A1: KE/KE₀ > 10 → explosion immédiate!");
                println!("  → Doubler softening et relancer");
                csv.flush().unwrap();
                return;
            } else if ke_ratio > 5.0 {
                println!("  ⚠️  WARNING: KE/KE₀ > 5 (marginal)");
            } else {
                println!("  ✓ PASS: KE/KE₀ < 5");
            }
        }

        // Step 20 check (FAIL-A2)
        if step == 20 {
            println!("\n=== STEP 20 CHECK ===");
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            println!("  Seg = {:.4}", seg);
            if ke_ratio > 50.0 {
                println!("  ❌ FAIL-A2: KE/KE₀ > 50 → explosion!");
                csv.flush().unwrap();
                return;
            }
            println!("  ✓ Continuing...");
        }

        // Step 50 check
        if step == 50 {
            println!("\n=== STEP 50 CHECK ===");
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            println!("  Seg = {:.4}", seg);
            if seg < 0.001 {
                println!("  ⚠️  WARNING: Seg < 0.001 (FAIL-A3 risk)");
            }
        }

        // Step 100 check (FAIL-A3, FAIL-A4)
        if step == 100 {
            println!("\n=== STEP 100 CHECK (Phase A validation) ===");
            println!("  z = {:.2}", z);
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            println!("  Seg = {:.4}", seg);

            if ke_ratio > 50.0 {
                println!("  ❌ FAIL: KE/KE₀ > 50");
            } else {
                println!("  ✓ KE/KE₀ < 50");
            }

            if seg < 0.0001 {
                println!("  ❌ FAIL-A3: Seg stagnant < 0.0001");
            } else {
                println!("  ✓ Seg > 0.0001");
            }

            println!("\n  → Générer frame step_100 pour validation morphologique");
        }

        // Render data at intervals
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

        // Early stop on explosion
        if ke_ratio > 200.0 && step > 50 {
            println!("\n❌ EXPLOSION DETECTED: KE/KE₀ = {:.1} > 200", ke_ratio);
            println!("   Stopping at step {}", step);
            break;
        }
    }

    csv.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n══════════════════════════════════════════════════");
    println!("  Phase A Complete");
    println!("══════════════════════════════════════════════════");
    println!("  Total time: {:.1}s ({:.1} ms/step)", elapsed, elapsed * 1000.0 / TOTAL_STEPS as f64);
    println!("  KE/KE₀ max: {:.4}", ke_ratio_max);
    println!("  Seg max: {:.4}", seg_max);
    println!("  Output: {}", output_dir);
    println!("\n  → Render frames with scripts/render_dens.py");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature. Compile with:");
    eprintln!("  cargo build --release --features cuda --bin janus_anticorr_test");
}
