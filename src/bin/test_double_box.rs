//! Test L=2L — Vérifier si l'instabilité k=1 est intrinsèque ou imposée par la boîte
//!
//! L_BOX = 984 Mpc (2 × 492)
//! N = 1M particles (100³)
//! 500 steps pour diagnostic rapide

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

// TEST: Double box size
const N_GRID: usize = 100;           // 100³ = 1M particles
const L_BOX: f64 = 984.0;            // 2 × 492 Mpc
const Z_INIT: f64 = 5.0;
const SOFTENING: f64 = 3.0;          // ~0.3 × spacing (984/100 = 9.84 Mpc)

// P(k) truncation — SAME physical scales as original (not scaled!)
const K_MIN: f64 = 2.0 * PI / 200.0;   // same λ_max = 200 Mpc
const K_MAX: f64 = 2.0 * PI / 8.0;     // same λ_min = 8 Mpc

// Simulation parameters
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 500;
const SNAPSHOT_INTERVAL: usize = 20;
const CSV_INTERVAL: usize = 10;
const THETA: f64 = 0.7;

const ETA: f64 = 1.045;
const N_SAMPLE_VIRIALIZE: usize = 50000;

const N_S: f64 = 0.96;
const K0: f64 = 0.02;  // Same as production

fn generate_pktrunc_zeldovich_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("═══════════════════════════════════════════════════════════");
    println!("  TEST DOUBLE BOX: L = {} Mpc (2× original)", L_BOX);
    println!("═══════════════════════════════════════════════════════════");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  k_min = 2π/{:.0} Mpc⁻¹", 2.0 * PI / K_MIN);
    println!("  k_max = 2π/{:.0} Mpc⁻¹", 2.0 * PI / K_MAX);

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;

    println!("  Generating Fourier modes...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let amplitude = 1.0;  // Increased from 0.01 for significant displacements

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

    let pct_kept = 100.0 * n_modes_kept as f64 / (n_modes_kept + n_modes_suppressed) as f64;
    println!("  Modes kept: {} ({:.1}%)", n_modes_kept, pct_kept);

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

    // Displacement field
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

    println!("  Inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, N_GRID);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, N_GRID);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, N_GRID);

    println!("  Computing density field...");
    let delta_real = ifft_3d(&mut delta_k, &ifft, N_GRID);

    // Scale displacement
    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }

    // Use larger amplitude for 1M particles
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Max displacement: {:.6e} Mpc, scale: {:.2e}", max_disp, scale);

    // Zel'dovich velocity scale: ψ̇ = D'(a) × ψ
    let d_dot = (1.0 + Z_INIT).sqrt();  // ≈ √6
    let vel_scale = d_dot * scale;

    // Density-based sign assignment
    println!("  Assigning signs based on density...");
    let n_positive_target = (n3 as f64 / (1.0 + ETA)) as usize;
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.sort_by(|&a, &b| delta_real[b].partial_cmp(&delta_real[a]).unwrap());

    let mut signs = vec![0i32; n3];
    for (rank, &idx) in indices.iter().enumerate() {
        signs[idx] = if rank < n_positive_target { 1 } else { -1 };
    }

    // Build particles with Zel'dovich velocities
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
                    sign: signs[idx],
                });
            }
        }
    }

    println!("  Shuffling...");
    particles.shuffle(&mut rng);

    // Separate positives first
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut final_signs = Vec::with_capacity(n3);
    let mut n_positive = 0usize;

    let pos_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign > 0).collect();
    let neg_particles: Vec<&Particle> = particles.iter().filter(|p| p.sign < 0).collect();

    for p in pos_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        velocities.extend_from_slice(&[p.vx, p.vy, p.vz]);
        final_signs.push(1i32);
        n_positive += 1;
    }
    for p in neg_particles.iter() {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        velocities.extend_from_slice(&[p.vx, p.vy, p.vz]);
        final_signs.push(-1i32);
    }

    println!("  Final: {} particles ({} + / {} -)", n3, n_positive, n3 - n_positive);

    (positions, velocities, final_signs, n_positive)
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
    println!("\n════════════════════════════════════════════════════════════");
    println!("  TEST DOUBLE BOX — L = 984 Mpc");
    println!("════════════════════════════════════════════════════════════\n");

    let output_dir = "/app/output/test_double_box_1m";
    let snapshots_dir = format!("{}/snapshots", output_dir);
    fs::create_dir_all(&snapshots_dir).expect("Failed to create output dir");

    let start = Instant::now();
    let (positions, velocities, signs, n_positive) = generate_pktrunc_zeldovich_ics(42);
    println!("\nIC generation took {:.1}s", start.elapsed().as_secs_f64());

    let n3 = N_GRID * N_GRID * N_GRID;
    let n_negative = n3 - n_positive;

    // Initialize GPU simulation
    println!("\nInitializing GPU simulation...");
    let init_start = Instant::now();

    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        L_BOX,
        positions,
        velocities,
        signs,
    ).expect("GPU init failed");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    println!("  Init time: {:.1}s", init_start.elapsed().as_secs_f64());

    // Virialize
    println!("\nVirializing (sampled, n={})...", N_SAMPLE_VIRIALIZE);
    let virial_start = Instant::now();
    sim.virialize_sampled(N_SAMPLE_VIRIALIZE).expect("virialize_sampled failed");
    println!("  Virialization time: {:.1}s", virial_start.elapsed().as_secs_f64());

    // Setup cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);

    // CSV header
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).unwrap();
    writeln!(csv, "step,z,ke,ke_ratio,seg,step_ms").unwrap();

    // Initial state
    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = sim.segregation_distance().expect("segregation failed");
    writeln!(csv, "0,{:.4},{:.6e},{:.6},{:.6},0", Z_INIT, ke_0, 1.0, seg_0).unwrap();
    save_snapshot(&sim, 0, &snapshots_dir, n_positive, n3);

    println!("\nInitial state:");
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg₀ = {:.4}", seg_0);

    println!("\nStarting simulation: {} steps", TOTAL_STEPS);
    println!("  dt = {}, θ = {}", DT, THETA);
    println!("  dtau_per_dt = {:.6}", dtau_per_dt);

    let mut tau = cosmo.tau_start;
    let run_start = Instant::now();

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

        if step % CSV_INTERVAL == 0 {
            writeln!(csv, "{},{:.4},{:.6e},{:.6},{:.6},{:.0}", step, z, ke, ke_ratio, seg, step_ms).unwrap();
        }

        if step % 50 == 0 {
            let elapsed = run_start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            let eta_min = (TOTAL_STEPS - step) as f64 / rate / 60.0;
            println!("  Step {:4} | z={:.3} | Seg={:.4} | KE/KE₀={:.3} | ETA {:.1}min",
                     step, z, seg, ke_ratio, eta_min);
        }

        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &snapshots_dir, n_positive, n3);
        }

        // Auto-stop si explosion
        if ke_ratio > 50.0 && step > 50 {
            println!("\n❌ AUTO-STOP: KE/KE₀ = {:.1} > 50 at step {}", ke_ratio, step);
            break;
        }
    }

    csv.flush().unwrap();

    let elapsed = run_start.elapsed();
    let final_ke = sim.kinetic_energy().unwrap();
    let final_seg = sim.segregation_distance().unwrap();

    println!("\n══════════════════════════════════════════════════════════");
    println!("  TEST COMPLETE — L = 984 Mpc (2× original)");
    println!("══════════════════════════════════════════════════════════");
    println!("  Runtime: {:.1}min", elapsed.as_secs_f64() / 60.0);
    println!("  KE/KE₀ final: {:.4}", final_ke / ke_0);
    println!("  Seg₀ = {:.4}", seg_0);
    println!("  Seg final = {:.4}", final_seg);
    println!("\n  Output: {}", output_dir);
    println!("  → Analyse P(k) pour voir si domaines multiples apparaissent");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires CUDA. Compile with --features cuda");
}
