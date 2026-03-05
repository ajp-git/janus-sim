//! Validation 500K — P(k) tronqué pour structures multiples
//!
//! Test avant run 12M production. Vérifie que les ICs P(k) tronqué
//! créent plusieurs structures (pas une seule frontière planaire).

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Physical parameters — 500K validation
const N_GRID: usize = 80;         // 80³ ≈ 512K particles
const L_BOX: f64 = 172.0;         // Mpc (n_side × 2.15)
const Z_INIT: f64 = 5.0;          // Initial redshift

// P(k) truncation — adapted for 172 Mpc box
const K_MIN: f64 = 2.0 * PI / 60.0;   // suppress λ > 60 Mpc
const K_MAX: f64 = 2.0 * PI / 6.0;    // suppress λ < 6 Mpc

// Simulation parameters
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 2000;
const CSV_INTERVAL: usize = 10;
const THETA: f64 = 0.7;

// Power spectrum: P(k) ∝ k^0.96 / (1 + (k/k0)^4)
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

const ETA: f64 = 1.045;

/// Generate P(k) truncated Zel'dovich ICs with density-based signs (shuffled)
/// Returns (positions, velocities, signs, n_positive)
fn generate_pktrunc_zeldovich_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating P(k) truncated Zel'dovich ICs...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  z_init = {}", Z_INIT);
    println!("  k_min = 2π/{:.0} Mpc⁻¹  (suppress λ > {:.0} Mpc)",
             2.0 * PI / K_MIN, 2.0 * PI / K_MIN);
    println!("  k_max = 2π/{:.0} Mpc⁻¹  (suppress λ < {:.0} Mpc)",
             2.0 * PI / K_MAX, 2.0 * PI / K_MAX);

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;

    // Growth factor at z_init
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    // Generate Gaussian random field in Fourier space with P(k) truncation
    println!("  Generating Fourier modes with truncation...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();

    let amplitude = 0.01;
    let mut n_modes_kept = 0usize;
    let mut n_modes_suppressed = 0usize;

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

                // P(k) TRUNCATION — spectral window
                let window = if k < K_MIN || k > K_MAX {
                    n_modes_suppressed += 1;
                    0.0  // mode supprimé
                } else {
                    n_modes_kept += 1;
                    1.0  // mode conservé
                };

                // Power spectrum × window
                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4)) * window;
                let sigma_k = pk.sqrt() * amplitude * d_growth;

                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    println!("  Modes kept: {} ({:.1}%)", n_modes_kept,
             100.0 * n_modes_kept as f64 / (n_modes_kept + n_modes_suppressed) as f64);
    println!("  Modes suppressed: {} ({:.1}%)", n_modes_suppressed,
             100.0 * n_modes_suppressed as f64 / (n_modes_kept + n_modes_suppressed) as f64);

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

    // Compute displacement field
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

                if k2 < 1e-20 {
                    continue;
                }

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

    // ALSO compute delta field in real space for density-based signs
    println!("  Computing density field δ(x)...");
    let delta_real = ifft_3d(&mut delta_k, &ifft, N_GRID);

    // Statistics on delta field
    let delta_mean: f64 = delta_real.iter().sum::<f64>() / n3 as f64;
    let delta_std: f64 = (delta_real.iter().map(|d| (d - delta_mean).powi(2)).sum::<f64>() / n3 as f64).sqrt();
    println!("  δ field: mean = {:.6e}, std = {:.6e}", delta_mean, delta_std);

    // Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Max displacement: {:.6e} Mpc", max_disp);

    // Scale to reasonable amplitude
    let target_disp = spacing * 0.3;
    let scale = target_disp / max_disp;
    println!("  Scaling factor: {:.4} → target {:.4} Mpc", scale, target_disp);

    // Zel'dovich velocities
    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;

    // Generate particles with DENSITY-BASED sign assignment
    println!("  Placing particles with density-based signs...");

    // First pass: assign signs based on δ(x)
    // δ(x) > 0 → positive (surdensité)
    // δ(x) < 0 → negative (sous-densité)
    // This gives physically meaningful assignment

    // Target: n_positive = N / (1 + η)
    let n_positive_target = (n3 as f64 / (1.0 + ETA)) as usize;

    // Sort particles by delta value
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.sort_by(|&a, &b| delta_real[b].partial_cmp(&delta_real[a]).unwrap());

    // Assign signs: top n_positive_target by delta → positive
    let mut signs_ordered = vec![0i32; n3];
    for (rank, &idx) in indices.iter().enumerate() {
        if rank < n_positive_target {
            signs_ordered[idx] = 1;
        } else {
            signs_ordered[idx] = -1;
        }
    }

    let actual_n_positive = signs_ordered.iter().filter(|&&s| s > 0).count();
    println!("  Density-based signs: {} positive, {} negative",
             actual_n_positive, n3 - actual_n_positive);

    // CRITICAL: Shuffle particle indices to avoid Z bias
    // The "ordre février" bug was having positives first in array
    // We shuffle the entire particle array to randomize memory layout
    println!("  Shuffling particle indices to avoid memory layout bias...");

    let mut particle_indices: Vec<usize> = (0..n3).collect();
    particle_indices.shuffle(&mut rng);

    // Build final arrays in shuffled order
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs = Vec::with_capacity(n3);
    let mut n_positive_final = 0usize;

    // We need positives first for GpuNBodySimulation::new_with_state
    // So we do TWO passes: first positives, then negatives
    // BUT the positions themselves are already based on grid (not index)
    // so the spatial distribution is independent of array order

    // Actually, let's just build all particles and track separately
    // First collect all particles with their data
    struct Particle {
        x: f64, y: f64, z: f64,
        vx: f64, vy: f64, vz: f64,
        sign: i32,
    }

    let mut particles: Vec<Particle> = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let x = x0 + psi_x[idx] * scale;
                let y = y0 + psi_y[idx] * scale;
                let z = z0 + psi_z[idx] * scale;

                let vx = psi_x[idx] * vel_scale;
                let vy = psi_y[idx] * vel_scale;
                let vz = psi_z[idx] * vel_scale;

                particles.push(Particle { x, y, z, vx, vy, vz, sign: signs_ordered[idx] });
            }
        }
    }

    // Shuffle particles
    particles.shuffle(&mut rng);

    // Now separate into positives first, then negatives (GPU requirement)
    let pos_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign > 0).collect();
    let neg_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign < 0).collect();

    // Interleave: positives first, then negatives
    for p in pos_particles.iter() {
        positions.push(p.x);
        positions.push(p.y);
        positions.push(p.z);
        velocities.push(p.vx);
        velocities.push(p.vy);
        velocities.push(p.vz);
        signs.push(1);
        n_positive_final += 1;
    }

    for p in neg_particles.iter() {
        positions.push(p.x);
        positions.push(p.y);
        positions.push(p.z);
        velocities.push(p.vx);
        velocities.push(p.vy);
        velocities.push(p.vz);
        signs.push(-1);
    }

    let n_negative = n3 - n_positive_final;
    println!("  Final: {} particles ({} + / {} -)", n3, n_positive_final, n_negative);

    // Verify NO index-position correlation
    let idx_vec: Vec<f64> = (0..n3).map(|i| i as f64).collect();
    let z_vec: Vec<f64> = (0..n3).map(|i| positions[i * 3 + 2]).collect();
    let corr = pearson_correlation(&idx_vec, &z_vec);
    println!("  Verification: corr(idx, z) = {:.4}", corr);
    if corr.abs() > 0.1 {
        println!("  ⚠️  WARNING: Index-position correlation detected!");
    } else {
        println!("  ✓ No index-position correlation");
    }

    (positions, velocities, signs, n_positive_final)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}

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

    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

/// Write snapshot for ICs (before simulation starts)
fn write_snapshot_ics(path: &str, positions: &[f64], n_positive: usize, step: usize) -> std::io::Result<()> {
    let n = positions.len() / 3;
    let mut file = BufWriter::new(File::create(path)?);

    // Header: n_particles, step, reserved (3 × u64)
    file.write_all(&(n as u64).to_le_bytes())?;
    file.write_all(&(step as u64).to_le_bytes())?;
    file.write_all(&0u64.to_le_bytes())?;

    // Data: n × (x, y, z, sign) as f32
    for i in 0..n {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let s: f32 = if i < n_positive { 1.0 } else { -1.0 };

        file.write_all(&x.to_le_bytes())?;
        file.write_all(&y.to_le_bytes())?;
        file.write_all(&z.to_le_bytes())?;
        file.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodySimulation, step: usize, dir: &str, n_positive: usize, n_total: usize) {
    let filename = format!("{}/snap_{:06}.bin", dir, step);
    let positions = sim.get_positions().expect("get_positions failed");

    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Header: [n_particles: u64, step: u64, reserved: u64] = 24 bytes
    writer.write_all(&(n_total as u64).to_le_bytes()).unwrap();
    writer.write_all(&(step as u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();

    // Positions as f32 + sign as f32 (compact format)
    for i in 0..n_total {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let sign: f32 = if i < n_positive { 1.0 } else { -1.0 };

        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }
}

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  P(k) Truncated Zel'dovich — 500K Validation Test");
    println!("═══════════════════════════════════════════════════════════\n");

    let seed = 42u64;

    // Create output directory
    let output_dir = "/app/output/pktrunc_500k_test";
    create_dir_all(output_dir).expect("Failed to create output dir");
    let snap_dir = format!("{}/snapshots", output_dir);
    create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    println!("Output: {}", output_dir);
    println!();

    // Generate ICs
    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_pktrunc_zeldovich_ics(seed);
    println!("\nIC generation took {:.1}s", start_ic.elapsed().as_secs_f64());

    let n3 = N_GRID * N_GRID * N_GRID;
    let n_negative = n3 - n_positive;

    // Write step 0 snapshot BEFORE simulation (for image generation)
    println!("\nWriting step 0 snapshot...");
    let snap0_path = format!("{}/snap_000000.bin", snap_dir);
    write_snapshot_ics(&snap0_path, &positions, n_positive, 0).expect("Failed to write snap 0");

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        L_BOX,
        positions,
        velocities,
        signs,
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);

    // Virialize (optional for 500K test, but good practice)
    println!("Virializing...");
    sim.virialize_sampled(20000).expect("Virialize failed");

    println!("\nSimulation parameters:");
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  steps = {}", TOTAL_STEPS);

    // Setup cosmology (FIX-016)
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);
    println!("  dtau_per_dt = {:.6} (FIX-016)", dtau_per_dt);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,time,seg,ke_ratio,z").unwrap();

    // Initial measurements
    let ke0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg0 = sim.segregation_distance().expect("segregation failed");

    println!("\nStep 0: Seg = {:.6}, KE = {:.6e}", seg0, ke0);
    writeln!(csv, "0,0.0,{:.6},{:.4},{:.2}", seg0, 1.0, Z_INIT).unwrap();

    // Main loop
    let start = Instant::now();
    let mut seg_max = seg0;
    let mut ke_ratio_max = 1.0f64;
    let mut tau = cosmo.tau_start;

    for step in 1..=TOTAL_STEPS {
        tau += dtau_per_step;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        if step % CSV_INTERVAL == 0 {
            let seg = sim.segregation_distance().expect("segregation");
            let ke = sim.kinetic_energy().expect("kinetic_energy");
            let ke_ratio = ke / ke0;

            if seg > seg_max { seg_max = seg; }
            if ke_ratio > ke_ratio_max { ke_ratio_max = ke_ratio; }

            writeln!(csv, "{},{:.2},{:.6},{:.4},{:.2}",
                     step, start.elapsed().as_secs_f64(), seg, ke_ratio, z).unwrap();

            if step % 100 == 0 {
                let rate = step as f64 / start.elapsed().as_secs_f64();
                println!("Step {}: z={:.2}, Seg={:.4}, KE/KE₀={:.2} ({:.1} steps/s)",
                         step, z, seg, ke_ratio, rate);
            }
        }
    }

    // Write final snapshot
    println!("\nWriting step {} snapshot...", TOTAL_STEPS);
    save_snapshot(&sim, TOTAL_STEPS, &snap_dir, n_positive, n3);
    let snap_final_path = format!("{}/snap_{:06}.bin", snap_dir, TOTAL_STEPS);

    // Final results
    let elapsed = start.elapsed().as_secs_f64();
    let seg_final = sim.segregation_distance().expect("segregation");

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  VALIDATION RESULTS");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("Runtime: {:.1}s ({:.1} steps/s)", elapsed, TOTAL_STEPS as f64 / elapsed);
    println!("Seg_max = {:.4}", seg_max);
    println!("Seg_final = {:.4}", seg_final);
    println!("KE/KE₀_max = {:.2}", ke_ratio_max);
    println!();

    // PASS criteria from prompt
    let pass_seg = seg_max > 0.05;
    let pass_ke = ke_ratio_max < 20.0;

    if pass_seg && pass_ke {
        println!("✓ PASS — Seg_max > 0.05 AND KE/KE₀ < 20");
        println!("\n→ Ready for 12M production run!");
    } else {
        println!("✗ FAIL");
        if !pass_seg { println!("  - Seg_max = {:.4} < 0.05", seg_max); }
        if !pass_ke { println!("  - KE/KE₀_max = {:.2} > 20", ke_ratio_max); }
    }

    println!("\nSnapshots saved:");
    println!("  {}", snap0_path);
    println!("  {}", snap_final_path);
    println!("\n→ Generate images to verify multiple structures");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
