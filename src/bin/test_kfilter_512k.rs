//! Test k-filter 512k — PM simulation with k ≤ 2 filter
//!
//! Parameters:
//!   - N ≈ 512k (80³ grid)
//!   - L = 492 Mpc
//!   - ε = 0.4 Mpc
//!   - PM solver with k_min = 3 (removes modes k=0,1,2)
//!
//! Goal: See if suppressing dipole allows filaments to form

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

use janus::treepm::pm_grid::PmGrid;

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 80;              // 80³ = 512,000 particles
const L_BOX: f64 = 492.0;              // Mpc
const Z_INIT: f64 = 5.0;
const SOFTENING: f64 = 0.4;            // Mpc

// k-space filter: remove modes |k| < K_MIN_IDX
const K_MIN_IDX: usize = 3;            // Remove k=0,1,2 → keep k≥3

// IC power spectrum
const K_CUT: f64 = 0.5;                // Mpc⁻¹ (λ_min ≈ 12.5 Mpc)
const AMPLITUDE: f64 = 0.02;

// Simulation
const DT: f64 = 0.02;
const TOTAL_STEPS: usize = 2000;
const SNAPSHOT_INTERVAL: usize = 20;

// PM grid resolution (should be >= N_GRID)
const PM_GRID: usize = 128;

// Janus
const ETA: f64 = 1.045;
const G_CONSTANT: f64 = 1.0;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TEST k-FILTER 512k — PM with k ≤ 2 suppression");
    println!("═══════════════════════════════════════════════════════════════");

    let n3 = N_GRID * N_GRID * N_GRID;
    println!("  Particles: {}³ = {}", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  Softening: {} Mpc", SOFTENING);
    println!("  PM grid: {}³", PM_GRID);
    println!("  k_min filter: {} (removes k=0,1,2)", K_MIN_IDX);
    println!();

    // Output directory
    let output_dir = format!("/app/output/kfilter_512k_{}",
        chrono::Local::now().format("%Y-%m-%d_%H%M%S"));
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("  Output: {}", output_dir);
    println!();

    // Generate ICs
    let (mut pos, mut vel, signs) = generate_ics(42);

    // Create PM grid
    let mut pm = PmGrid::new(PM_GRID, L_BOX);
    println!("  PM grid memory: {:.1} MB", pm.memory_bytes() as f64 / 1e6);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,ke,pe,segregation,time_ms").unwrap();

    // Initial snapshot
    save_snapshot(&pos, &signs, 0, &output_dir);

    // Main loop
    println!("\n  Starting simulation...\n");
    let start = Instant::now();

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        // Leapfrog: kick-drift-kick
        // Half kick
        let forces = compute_pm_forces(&pos, &signs, &mut pm);
        for i in 0..n3 {
            vel[i * 3 + 0] += 0.5 * DT * forces[i * 3 + 0];
            vel[i * 3 + 1] += 0.5 * DT * forces[i * 3 + 1];
            vel[i * 3 + 2] += 0.5 * DT * forces[i * 3 + 2];
        }

        // Drift
        for i in 0..n3 {
            pos[i * 3 + 0] += DT * vel[i * 3 + 0];
            pos[i * 3 + 1] += DT * vel[i * 3 + 1];
            pos[i * 3 + 2] += DT * vel[i * 3 + 2];

            // Periodic wrap
            for d in 0..3 {
                let idx = i * 3 + d;
                while pos[idx] > L_BOX / 2.0 { pos[idx] -= L_BOX; }
                while pos[idx] < -L_BOX / 2.0 { pos[idx] += L_BOX; }
            }
        }

        // Second half kick
        let forces = compute_pm_forces(&pos, &signs, &mut pm);
        for i in 0..n3 {
            vel[i * 3 + 0] += 0.5 * DT * forces[i * 3 + 0];
            vel[i * 3 + 1] += 0.5 * DT * forces[i * 3 + 1];
            vel[i * 3 + 2] += 0.5 * DT * forces[i * 3 + 2];
        }

        let step_time = step_start.elapsed().as_millis();

        // Diagnostics
        if step % 10 == 0 || step == 1 {
            let ke = compute_ke(&vel);
            let seg = compute_segregation(&pos, &signs);

            writeln!(csv, "{},{:.6e},{:.6e},{:.6},{}",
                step, ke, 0.0, seg, step_time).unwrap();

            if step % 100 == 0 {
                println!("  Step {:5} | KE={:.3e} | Seg={:.2} Mpc | {:.0}ms/step",
                    step, ke, seg, step_time);
            }
        }

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&pos, &signs, step, &output_dir);
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    println!("\n  Simulation complete!");
    println!("  Total time: {:.1} s ({:.1} ms/step)", total_time, 1000.0 * total_time / TOTAL_STEPS as f64);
    println!("  Output: {}", output_dir);
}

/// Compute PM forces with k_min filter
fn compute_pm_forces(pos: &[f64], signs: &[i32], pm: &mut PmGrid) -> Vec<f64> {
    let n = pos.len() / 3;
    let mass = 1.0;

    // Clear and assign mass
    pm.clear();
    for i in 0..n {
        let x = pos[i * 3 + 0];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        let sign = signs[i] as i8;
        pm.assign_mass(x, y, z, mass, sign);
    }

    // Solve Poisson with k_min filter (removes k=0,1,2)
    pm.solve_poisson_with_k_filter(G_CONSTANT, K_MIN_IDX);

    // Interpolate forces
    let mut forces = vec![0.0; n * 3];
    for i in 0..n {
        let x = pos[i * 3 + 0];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];
        let sign = signs[i] as i8;

        let (fx, fy, fz) = pm.interpolate_force(x, y, z, sign);
        forces[i * 3 + 0] = fx;
        forces[i * 3 + 1] = fy;
        forces[i * 3 + 2] = fz;
    }

    forces
}

/// Generate Zel'dovich ICs with random sign assignment
fn generate_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("  Generating Zel'dovich ICs...");

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

                // Filter: skip k < k_min (already applied in PM, but also in ICs for cleaner start)
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

    // IFFT to get displacement field
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    // Process each component
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

    // IFFT
    for line in psi_x.chunks_mut(N_GRID) {
        ifft.process(line);
    }
    for line in psi_y.chunks_mut(N_GRID) {
        ifft.process(line);
    }
    for line in psi_z.chunks_mut(N_GRID) {
        ifft.process(line);
    }

    // Create particles
    let mut pos = vec![0.0; n3 * 3];
    let mut vel = vec![0.0; n3 * 3];
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

                pos[idx * 3 + 0] = x0 + dx;
                pos[idx * 3 + 1] = y0 + dy;
                pos[idx * 3 + 2] = z0 + dz;

                // Zel'dovich velocity
                let h0 = 0.1;  // Approximate H(z=5)
                vel[idx * 3 + 0] = dx * h0;
                vel[idx * 3 + 1] = dy * h0;
                vel[idx * 3 + 2] = dz * h0;
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

    (pos, vel, signs)
}

fn compute_ke(vel: &[f64]) -> f64 {
    let n = vel.len() / 3;
    let mut ke = 0.0;
    for i in 0..n {
        let vx = vel[i * 3 + 0];
        let vy = vel[i * 3 + 1];
        let vz = vel[i * 3 + 2];
        ke += 0.5 * (vx*vx + vy*vy + vz*vz);
    }
    ke
}

fn compute_segregation(pos: &[f64], signs: &[i32]) -> f64 {
    let n = signs.len();

    // Periodic COM using angular method
    let mut cos_p = [0.0f64; 3];
    let mut sin_p = [0.0f64; 3];
    let mut cos_m = [0.0f64; 3];
    let mut sin_m = [0.0f64; 3];
    let mut n_p = 0usize;
    let mut n_m = 0usize;

    for i in 0..n {
        for d in 0..3 {
            let theta = 2.0 * PI * pos[i * 3 + d] / L_BOX;
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
        com_p[d] = L_BOX * (sin_p[d] / n_p as f64).atan2(cos_p[d] / n_p as f64) / (2.0 * PI);
        com_m[d] = L_BOX * (sin_m[d] / n_m as f64).atan2(cos_m[d] / n_m as f64) / (2.0 * PI);
    }

    // Periodic distance
    let mut d2 = 0.0;
    for d in 0..3 {
        let mut delta = com_p[d] - com_m[d];
        if delta > L_BOX / 2.0 { delta -= L_BOX; }
        if delta < -L_BOX / 2.0 { delta += L_BOX; }
        d2 += delta * delta;
    }
    d2.sqrt()
}

fn save_snapshot(pos: &[f64], signs: &[i32], step: usize, output_dir: &str) {
    let n = signs.len();
    let path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
    let mut f = BufWriter::new(File::create(&path).unwrap());

    // Header: n, step, reserved
    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&(step as u64).to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();

    // Particles: x, y, z, sign (as f32)
    for i in 0..n {
        f.write_all(&(pos[i * 3 + 0] as f32).to_le_bytes()).unwrap();
        f.write_all(&(pos[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        f.write_all(&(pos[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        f.write_all(&(signs[i] as f32).to_le_bytes()).unwrap();
    }
}
