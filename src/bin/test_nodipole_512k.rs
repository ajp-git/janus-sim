//! Test NO-DIPOLE 512k — GPU Barnes-Hut with COM recentering
//!
//! Parameters:
//!   - N ≈ 512k (80³ grid)
//!   - L = 492 Mpc
//!   - ε = 0.4 Mpc
//!   - COM+ = COM- enforced at each step (suppresses dipole)
//!
//! Goal: See if suppressing dipole allows filaments to form

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 80;              // 80³ = 512,000 particles
const L_BOX: f64 = 492.0;              // Mpc
const Z_INIT: f64 = 5.0;
const SOFTENING: f64 = 0.4;            // Mpc

// IC power spectrum - suppress k=1,2 in ICs too
const K_MIN_IDX: usize = 3;            // Remove k=0,1,2 in ICs
const K_CUT: f64 = 0.5;                // Mpc⁻¹
const AMPLITUDE: f64 = 0.02;

// Simulation
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 2000;
const SNAPSHOT_INTERVAL: usize = 20;
const THETA: f64 = 0.7;

// Janus
const ETA: f64 = 1.045;

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TEST NO-DIPOLE 512k — GPU BH with COM recentering");
    println!("═══════════════════════════════════════════════════════════════");

    let n3 = N_GRID * N_GRID * N_GRID;
    println!("  Particles: {}³ = {}", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  Softening: {} Mpc", SOFTENING);
    println!("  θ (BH): {}", THETA);
    println!("  COM recentering: ENABLED (suppresses dipole)");
    println!();

    // Output directory
    let output_dir = format!("/app/output/nodipole_512k_{}",
        chrono::Local::now().format("%Y-%m-%d_%H%M%S"));
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("  Output: {}", output_dir);
    println!();

    // Generate ICs
    let (positions, velocities, signs) = generate_ics(42);

    // Create GPU simulation
    let masses: Vec<f64> = vec![1.0; n3];

    let mut sim = GpuNBodySimulation::new(
        positions.clone(),
        velocities.clone(),
        masses,
        signs.iter().map(|&s| s as i8).collect(),
        L_BOX,
        SOFTENING,
        THETA,
    ).expect("Failed to create GPU simulation");

    println!("  GPU simulation initialized");

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,ke,segregation,time_ms").unwrap();

    // Initial snapshot
    save_snapshot(&sim.positions, &signs, 0, &output_dir);

    // Main loop
    println!("\n  Starting simulation...\n");
    let start = Instant::now();

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        // One integration step
        sim.step(DT);

        // === KEY: Recenter COM to suppress dipole ===
        recenter_com(&mut sim.positions, &signs, L_BOX);

        let step_time = step_start.elapsed().as_millis();

        // Diagnostics
        if step % 10 == 0 || step == 1 {
            let ke = compute_ke(&sim.velocities);
            let seg = compute_segregation(&sim.positions, &signs, L_BOX);

            writeln!(csv, "{},{:.6e},{:.6},{}", step, ke, seg, step_time).unwrap();

            if step % 100 == 0 {
                println!("  Step {:5} | KE={:.3e} | Seg={:.2} Mpc | {:.0}ms/step",
                    step, ke, seg, step_time);
            }
        }

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim.positions, &signs, step, &output_dir);
        }
    }

    csv.flush().unwrap();
    let total_time = start.elapsed().as_secs_f64();
    println!("\n  Simulation complete!");
    println!("  Total time: {:.1} s ({:.1} ms/step)", total_time, 1000.0 * total_time / TOTAL_STEPS as f64);
    println!("  Output: {}", output_dir);
}

/// Recenter positions so that COM+ = COM-
/// This suppresses the dipole mode at each timestep
fn recenter_com(positions: &mut Vec<nalgebra::Vector3<f64>>, signs: &[i32], box_size: f64) {
    let n = signs.len();

    // Compute periodic COM for each species using angular method
    let mut cos_p = [0.0f64; 3];
    let mut sin_p = [0.0f64; 3];
    let mut cos_m = [0.0f64; 3];
    let mut sin_m = [0.0f64; 3];
    let mut n_p = 0usize;
    let mut n_m = 0usize;

    for i in 0..n {
        let pos = &positions[i];
        for d in 0..3 {
            let theta = 2.0 * PI * pos[d] / box_size;
            if signs[i] > 0 {
                cos_p[d] += theta.cos();
                sin_p[d] += theta.sin();
            } else {
                cos_m[d] += theta.cos();
                sin_m[d] += theta.sin();
            }
        }
        if signs[i] > 0 { n_p += 1; } else { n_m += 1; }
    }

    // Compute COM positions
    let mut com_p = [0.0f64; 3];
    let mut com_m = [0.0f64; 3];
    for d in 0..3 {
        com_p[d] = box_size * (sin_p[d] / n_p as f64).atan2(cos_p[d] / n_p as f64) / (2.0 * PI);
        com_m[d] = box_size * (sin_m[d] / n_m as f64).atan2(cos_m[d] / n_m as f64) / (2.0 * PI);
    }

    // Compute shift to bring COMs together: shift = (COM+ - COM-) / 2
    // Shift + particles by -shift, - particles by +shift
    let mut shift = [0.0f64; 3];
    for d in 0..3 {
        let mut delta = com_p[d] - com_m[d];
        // Periodic wrap
        if delta > box_size / 2.0 { delta -= box_size; }
        if delta < -box_size / 2.0 { delta += box_size; }
        shift[d] = delta / 2.0;
    }

    // Apply shifts
    let half_box = box_size / 2.0;
    for i in 0..n {
        let sign_factor = if signs[i] > 0 { -1.0 } else { 1.0 };
        for d in 0..3 {
            positions[i][d] += sign_factor * shift[d];
            // Periodic wrap
            if positions[i][d] > half_box { positions[i][d] -= box_size; }
            if positions[i][d] < -half_box { positions[i][d] += box_size; }
        }
    }
}

/// Generate Zel'dovich ICs with random sign assignment
fn generate_ics(seed: u64) -> (Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::Vector3<f64>>, Vec<i32>) {
    use nalgebra::Vector3;

    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("  Generating Zel'dovich ICs (k_min={})...", K_MIN_IDX);

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;

    let a_init = 1.0 / (1.0 + Z_INIT);

    // Generate Fourier modes
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

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

                // Filter: skip k < k_min
                let k_idx = (kx_idx.abs() as usize).max(ky_idx.abs() as usize).max(kz_idx.abs() as usize);
                if k_idx < K_MIN_IDX || k < 1e-10 {
                    continue;
                }

                // P(k) = k^-2 × exp(-(k/k_cut)²)
                let pk = AMPLITUDE * k.powi(-2) * (-(k / K_CUT).powi(2)).exp();
                let amp = pk.sqrt();

                let re = normal.sample(&mut rng) * amp;
                let im = normal.sample(&mut rng) * amp;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    // Compute displacement field via IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    let mut psi_x = delta_k.clone();
    let mut psi_y = delta_k.clone();
    let mut psi_z = delta_k.clone();

    // Multiply by -i k_i / k²
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

                if k2 > 1e-10 {
                    let factor = Complex::new(0.0, -1.0) / k2;
                    psi_x[idx] = delta_k[idx] * factor * kx;
                    psi_y[idx] = delta_k[idx] * factor * ky;
                    psi_z[idx] = delta_k[idx] * factor * kz;
                }
            }
        }
    }

    // 1D IFFT along x for each (y,z) slice
    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            let offset = iz * N_GRID * N_GRID + iy * N_GRID;
            let mut slice_x: Vec<_> = psi_x[offset..offset+N_GRID].to_vec();
            let mut slice_y: Vec<_> = psi_y[offset..offset+N_GRID].to_vec();
            let mut slice_z: Vec<_> = psi_z[offset..offset+N_GRID].to_vec();
            ifft.process(&mut slice_x);
            ifft.process(&mut slice_y);
            ifft.process(&mut slice_z);
            psi_x[offset..offset+N_GRID].copy_from_slice(&slice_x);
            psi_y[offset..offset+N_GRID].copy_from_slice(&slice_y);
            psi_z[offset..offset+N_GRID].copy_from_slice(&slice_z);
        }
    }

    // Create particles
    let mut positions = Vec::with_capacity(n3);
    let mut velocities = Vec::with_capacity(n3);
    let mut signs = vec![0i32; n3];

    let n_minus = (n3 as f64 * ETA / (1.0 + ETA)) as usize;
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.shuffle(&mut rng);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let norm = 1.0 / (n3 as f64);
                let dx = psi_x[idx].re * norm * a_init;
                let dy = psi_y[idx].re * norm * a_init;
                let dz = psi_z[idx].re * norm * a_init;

                let pos = Vector3::new(x0 + dx, y0 + dy, z0 + dz);
                positions.push(pos);

                // Zel'dovich velocity
                let h0 = 0.1;
                let vel = Vector3::new(dx * h0, dy * h0, dz * h0);
                velocities.push(vel);
            }
        }
    }

    // Random sign assignment
    for i in 0..n3 {
        signs[i] = if indices[i] < n_minus { -1 } else { 1 };
    }

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus_actual = n3 - n_plus;
    println!("  N+ = {}, N- = {}, η = {:.4}", n_plus, n_minus_actual,
        n_minus_actual as f64 / n_plus as f64);

    (positions, velocities, signs)
}

fn compute_ke(velocities: &[nalgebra::Vector3<f64>]) -> f64 {
    velocities.iter().map(|v| 0.5 * v.norm_squared()).sum()
}

fn compute_segregation(positions: &[nalgebra::Vector3<f64>], signs: &[i32], box_size: f64) -> f64 {
    let n = signs.len();

    let mut cos_p = [0.0f64; 3];
    let mut sin_p = [0.0f64; 3];
    let mut cos_m = [0.0f64; 3];
    let mut sin_m = [0.0f64; 3];
    let mut n_p = 0usize;
    let mut n_m = 0usize;

    for i in 0..n {
        let pos = &positions[i];
        for d in 0..3 {
            let theta = 2.0 * PI * pos[d] / box_size;
            if signs[i] > 0 {
                cos_p[d] += theta.cos();
                sin_p[d] += theta.sin();
            } else {
                cos_m[d] += theta.cos();
                sin_m[d] += theta.sin();
            }
        }
        if signs[i] > 0 { n_p += 1; } else { n_m += 1; }
    }

    let mut com_p = [0.0f64; 3];
    let mut com_m = [0.0f64; 3];
    for d in 0..3 {
        com_p[d] = box_size * (sin_p[d] / n_p as f64).atan2(cos_p[d] / n_p as f64) / (2.0 * PI);
        com_m[d] = box_size * (sin_m[d] / n_m as f64).atan2(cos_m[d] / n_m as f64) / (2.0 * PI);
    }

    let mut d2 = 0.0;
    for d in 0..3 {
        let mut delta = com_p[d] - com_m[d];
        if delta > box_size / 2.0 { delta -= box_size; }
        if delta < -box_size / 2.0 { delta += box_size; }
        d2 += delta * delta;
    }
    d2.sqrt()
}

fn save_snapshot(positions: &[nalgebra::Vector3<f64>], signs: &[i32], step: usize, output_dir: &str) {
    let n = signs.len();
    let path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
    let mut f = BufWriter::new(File::create(&path).unwrap());

    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&(step as u64).to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();

    for i in 0..n {
        f.write_all(&(positions[i][0] as f32).to_le_bytes()).unwrap();
        f.write_all(&(positions[i][1] as f32).to_le_bytes()).unwrap();
        f.write_all(&(positions[i][2] as f32).to_le_bytes()).unwrap();
        f.write_all(&(signs[i] as f32).to_le_bytes()).unwrap();
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Error: This binary requires CUDA. Compile with --features cuda");
    std::process::exit(1);
}
