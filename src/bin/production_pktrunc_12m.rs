//! Production 12M — P(k) tronqué Zel'dovich ICs
//!
//! ICs P(k) truncated: k_min = 2π/150, k_max = 2π/15
//! Multiple structures instead of single planar boundary
//! Density-based signs, shuffled indices

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Physical parameters — 12M production
const N_GRID: usize = 229;         // 229³ ≈ 12M particles (12,008,989)
const L_BOX: f64 = 492.0;          // Mpc (n_side × 2.15 = 492.35)
const Z_INIT: f64 = 5.0;
const SOFTENING: f64 = 0.65;       // Mpc

// P(k) truncation — v2 fenêtre élargie
const K_MIN: f64 = 2.0 * PI / 200.0;   // suppress λ > 200 Mpc
const K_MAX: f64 = 2.0 * PI / 8.0;     // suppress λ < 8 Mpc

// Simulation parameters
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 20000;
const SNAPSHOT_INTERVAL: usize = 20;   // → 1000 snapshots
const CSV_INTERVAL: usize = 10;
const THETA: f64 = 0.7;

const ETA: f64 = 1.045;
const N_SAMPLE_VIRIALIZE: usize = 80000;

// Power spectrum: P(k) ∝ k^0.96 / (1 + (k/k0)^4)
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

/// Generate P(k) truncated Zel'dovich ICs
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

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

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

                // P(k) TRUNCATION
                let window = if k < K_MIN || k > K_MAX {
                    n_modes_suppressed += 1;
                    0.0
                } else {
                    n_modes_kept += 1;
                    1.0
                };

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

    // Compute density field for sign assignment
    println!("  Computing density field δ(x)...");
    let delta_real = ifft_3d(&mut delta_k, &ifft, N_GRID);

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

    let target_disp = spacing * 0.3;
    let scale = target_disp / max_disp;
    println!("  Scaling factor: {:.4} → target {:.4} Mpc", scale, target_disp);

    let d_dot = (1.0 + Z_INIT).sqrt();
    let vel_scale = d_dot * scale;

    // Density-based sign assignment
    println!("  Placing particles with density-based signs...");
    let n_positive_target = (n3 as f64 / (1.0 + ETA)) as usize;

    let mut indices: Vec<usize> = (0..n3).collect();
    indices.sort_by(|&a, &b| delta_real[b].partial_cmp(&delta_real[a]).unwrap());

    let mut signs_ordered = vec![0i32; n3];
    for (rank, &idx) in indices.iter().enumerate() {
        signs_ordered[idx] = if rank < n_positive_target { 1 } else { -1 };
    }

    let actual_n_positive = signs_ordered.iter().filter(|&&s| s > 0).count();
    println!("  Density-based signs: {} positive, {} negative",
             actual_n_positive, n3 - actual_n_positive);

    // Build particles and shuffle
    println!("  Shuffling particle indices to avoid memory layout bias...");

    struct Particle { x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, sign: i32 }
    let mut particles: Vec<Particle> = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                particles.push(Particle {
                    x: x0 + psi_x[idx] * scale,
                    y: y0 + psi_y[idx] * scale,
                    z: z0 + psi_z[idx] * scale,
                    vx: psi_x[idx] * vel_scale,
                    vy: psi_y[idx] * vel_scale,
                    vz: psi_z[idx] * vel_scale,
                    sign: signs_ordered[idx],
                });
            }
        }
    }

    particles.shuffle(&mut rng);

    // Separate positives first (GPU requirement)
    let pos_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign > 0).collect();
    let neg_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign < 0).collect();

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs = Vec::with_capacity(n3);
    let mut n_positive_final = 0usize;

    for p in pos_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        velocities.extend_from_slice(&[p.vx, p.vy, p.vz]);
        signs.push(1);
        n_positive_final += 1;
    }

    for p in neg_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        velocities.extend_from_slice(&[p.vx, p.vy, p.vz]);
        signs.push(-1);
    }

    println!("  Final: {} particles ({} + / {} -)", n3, n_positive_final, n3 - n_positive_final);

    // Verify no index-position correlation
    let idx_vec: Vec<f64> = (0..n3).map(|i| i as f64).collect();
    let z_vec: Vec<f64> = (0..n3).map(|i| positions[i * 3 + 2]).collect();
    let corr = pearson_correlation(&idx_vec, &z_vec);
    println!("  Verification: corr(idx, z) = {:.4}", corr);

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

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodySimulation, step: usize, dir: &str, n_positive: usize, n_total: usize) {
    let filename = format!("{}/snap_{:06}.bin", dir, step);
    let positions = sim.get_positions().expect("get_positions failed");
    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    writer.write_all(&(n_total as u64).to_le_bytes()).unwrap();
    writer.write_all(&(step as u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();

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
    println!("  Production 12M — P(k) Truncated Zel'dovich ICs");
    println!("═══════════════════════════════════════════════════════════\n");

    let seed = 42u64;
    let output_dir = "/app/output/production_pktrunc_12m_v2";
    let snapshots_dir = format!("{}/snapshots", output_dir);
    fs::create_dir_all(&snapshots_dir).expect("Failed to create snapshots dir");

    let pid = std::process::id();
    fs::write(format!("{}/pid.txt", output_dir), pid.to_string()).ok();

    println!("Output: {}", output_dir);
    println!("PID: {}\n", pid);

    // Generate ICs
    let start_ic = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_pktrunc_zeldovich_ics(seed);
    println!("\nIC generation took {:.1}s", start_ic.elapsed().as_secs_f64());

    let n3 = N_GRID * N_GRID * N_GRID;
    let n_negative = n3 - n_positive;

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let init_start = Instant::now();

    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        L_BOX,
        positions,
        velocities,
        signs,
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    println!("  Init time: {:.1}s", init_start.elapsed().as_secs_f64());

    // Virialization
    println!("\nVirializing (sampled, n={})...", N_SAMPLE_VIRIALIZE);
    let virial_start = Instant::now();
    sim.virialize_sampled(N_SAMPLE_VIRIALIZE).expect("virialize_sampled failed");
    println!("  Virialization time: {:.1}s", virial_start.elapsed().as_secs_f64());

    // Setup cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);

    println!("\nParameters:");
    println!("  N = {} ({:.1}M)", n3, n3 as f64 / 1e6);
    println!("  Box = {} Mpc", L_BOX);
    println!("  Steps = {}", TOTAL_STEPS);
    println!("  θ = {}, softening = {} Mpc", THETA, SOFTENING);
    println!("  dtau_per_dt = {:.6} (FIX-016)", dtau_per_dt);
    println!("  Snapshots = {} (every {} steps)", TOTAL_STEPS / SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL);

    // CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke,ke_ratio,seg,step_ms").unwrap();

    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = sim.segregation_distance().expect("segregation failed");
    writeln!(csv, "0,{:.4},{:.6e},{:.6},{:.6},0", Z_INIT, ke_0, 1.0, seg_0).unwrap();

    // Initial snapshot
    save_snapshot(&sim, 0, &snapshots_dir, n_positive, n3);

    println!("\nInitial state:");
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg₀ = {:.4}", seg_0);

    // Step 5 check before starting
    println!("\n══════════════════════════════════════════════════");
    println!("  Starting simulation ({} steps)", TOTAL_STEPS);
    println!("══════════════════════════════════════════════════\n");

    let mut tau = cosmo.tau_start;
    let start = Instant::now();
    let mut seg_max = seg_0;
    let mut z_at_seg_max = Z_INIT;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();
        tau += dtau_per_step;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        let step_ms = step_start.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy().expect("kinetic_energy failed");
        let ke_ratio = ke / ke_0;
        let seg = sim.segregation_distance().expect("segregation failed");

        if seg > seg_max {
            seg_max = seg;
            z_at_seg_max = z;
        }

        writeln!(csv, "{},{:.4},{:.6e},{:.6},{:.6},{:.0}", step, z, ke, ke_ratio, seg, step_ms).unwrap();

        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &snapshots_dir, n_positive, n3);
        }

        if step == 5 {
            println!(">>> STEP 5 CHECK <<<");
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            if ke_ratio > 1.05 {
                println!("  ⚠️ WARNING: KE/KE₀ > 1.05");
            } else {
                println!("  ✓ PASS: KE/KE₀ < 1.05");
            }
        }

        if step <= 10 || step % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            let eta_h = (TOTAL_STEPS - step) as f64 / rate / 3600.0;
            println!("Step {:5}: z={:.2}, KE/KE₀={:.4}, Seg={:.4} ({:.0} ms, ETA {:.1}h)",
                     step, z, ke_ratio, seg, step_ms, eta_h);
        }

        if step == 100 || step == 3000 || step == 4500 || step == 10000 || step == 20000 {
            println!("\n>>> MILESTONE step {} <<<", step);
            println!("  z = {:.2}", z);
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            println!("  Seg = {:.4} (max = {:.4} @ z={:.2})", seg, seg_max, z_at_seg_max);
            csv.flush().unwrap();
        }

        if ke_ratio > 50.0 && step > 100 {
            println!("\n❌ AUTO-STOP: KE/KE₀ = {:.1} > 50 at step {}", ke_ratio, step);
            break;
        }
    }

    csv.flush().unwrap();

    let elapsed = start.elapsed();
    let final_ke = sim.kinetic_energy().unwrap();
    let final_seg = sim.segregation_distance().unwrap();

    println!("\n══════════════════════════════════════════════════");
    println!("  RUN COMPLETE");
    println!("══════════════════════════════════════════════════");
    println!("  Runtime: {:.1}h", elapsed.as_secs_f64() / 3600.0);
    println!("  KE/KE₀ final: {:.4}", final_ke / ke_0);
    println!("  Seg₀ = {:.4}", seg_0);
    println!("  Seg_max = {:.4} @ z={:.2}", seg_max, z_at_seg_max);
    println!("  Seg final = {:.4}", final_seg);

    // Summary JSON
    let summary = format!(r#"{{
  "model": "Production P(k) Truncated 12M",
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {},
  "box_size": {},
  "k_min": "2pi/150",
  "k_max": "2pi/15",
  "steps_completed": {},
  "ke_0": {:.6e},
  "ke_final": {:.6e},
  "ke_ratio_final": {:.6},
  "seg_0": {:.6},
  "seg_max": {:.6},
  "z_at_seg_max": {:.4},
  "seg_final": {:.6},
  "runtime_hours": {:.2}
}}"#,
        n3, n_positive, n_negative, ETA, L_BOX, TOTAL_STEPS,
        ke_0, final_ke, final_ke / ke_0, seg_0, seg_max, z_at_seg_max, final_seg,
        elapsed.as_secs_f64() / 3600.0
    );
    fs::write(format!("{}/summary.json", output_dir), &summary).unwrap();

    println!("\nOutput: {}", output_dir);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
