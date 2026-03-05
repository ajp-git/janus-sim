//! No-Dipole Zel'dovich 4M — GPU TreePM with k-space filter
//!
//! Configuration:
//!   - N = 160³ ≈ 4M particles
//!   - Zel'dovich ICs with P(k) = k^-2 × exp(-(k/k_cut)²)
//!   - k_min = 3 in ICs AND in PM solver (suppresses dipole)
//!   - TreePM GPU with cuFFT
//!
//! Build:
//!   ./cuda/build_cufft.sh
//!   cargo build --release --features cuda,cufft --bin nodipole_zeldovich_4m

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 160;             // 160³ ≈ 4.1M particles
const L_BOX: f64 = 492.0;              // Mpc
const Z_INIT: f64 = 5.0;
const SOFTENING: f64 = 0.3;            // Mpc

// P(k) parameters
const K_MIN_IDX: usize = 3;            // Suppress k=0,1,2 in ICs
const K_CUT: f64 = 1.0;                // Mpc⁻¹ (λ_min ≈ 6 Mpc)
const AMPLITUDE: f64 = 0.015;          // Power spectrum amplitude

// PM k-space filter
const PM_K_MIN: usize = 3;             // Suppress k=0,1,2 in PM solver

// Simulation
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 3000;
const SNAPSHOT_INTERVAL: usize = 50;
const LOG_INTERVAL: usize = 20;
const THETA: f64 = 0.7;
const R_CUT_FACTOR: f64 = 16.0;

const ETA: f64 = 1.045;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  NO-DIPOLE ZEL'DOVICH 4M — GPU TreePM + k-filter");
    println!("═══════════════════════════════════════════════════════════════");

    let n3 = N_GRID * N_GRID * N_GRID;
    let r_cut = L_BOX / R_CUT_FACTOR;

    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);
    println!("  z_init: {}", Z_INIT);
    println!("  Softening: {} Mpc", SOFTENING);
    println!("  θ (BH): {}", THETA);
    println!("  r_cut: {:.2} Mpc", r_cut);
    println!();
    println!("  P(k) = k^-2 × exp(-(k/k_cut)²)");
    println!("  k_min (ICs): {} (suppresses k=0,1,2)", K_MIN_IDX);
    println!("  k_min (PM): {} (suppresses k=0,1,2)", PM_K_MIN);
    println!("  k_cut: {} Mpc⁻¹", K_CUT);
    println!();

    // Output directory
    let output_dir = format!("/app/output/nodipole_zel_4m_{}",
        chrono::Local::now().format("%Y-%m-%d_%H%M%S"));
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("  Output: {}", output_dir);
    println!();

    // Generate Zel'dovich ICs
    let (positions, velocities, signs) = generate_zeldovich_ics(42);
    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = n3 - n_plus;

    println!("  N+ = {}, N- = {}, η = {:.4}", n_plus, n_minus,
        n_minus as f64 / n_plus as f64);
    println!();

    // Initialize GPU simulation
    println!("  Initializing GPU TreePM simulation...");
    let t0 = Instant::now();

    // Convert to f32 for GPU
    let pos_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    let mut sim = match GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, L_BOX
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU: {}", e);
            return;
        }
    };

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_pm_k_min(PM_K_MIN);  // Enable k-space filter

    println!("  Init time: {:.2?}", t0.elapsed());
    println!();

    // Open CSV log
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,ke,segregation,time_ms").unwrap();

    // Initial state
    let ke_0 = sim.kinetic_energy().unwrap_or(1.0);
    let seg_0 = sim.segregation().unwrap_or(0.0);
    println!("  Initial KE: {:.4e}", ke_0);
    println!("  Initial Seg: {:.2} Mpc", seg_0);
    println!();

    // Save initial snapshot
    save_snapshot(&sim, 0, &output_dir);

    // Main loop
    println!("  Starting simulation ({} steps)...\n", TOTAL_STEPS);
    let start = Instant::now();
    let mut seg_max = seg_0;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        if let Err(e) = sim.step_treepm_gpu(DT, r_cut, 0.0, 0.0) {
            eprintln!("ERROR at step {}: {}", step, e);
            return;
        }

        let step_time = step_start.elapsed().as_millis();

        if step % LOG_INTERVAL == 0 || step == 1 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let seg = sim.segregation().unwrap_or(0.0);
            seg_max = seg_max.max(seg);

            writeln!(csv, "{},{:.6e},{:.6},{}", step, ke, seg, step_time).unwrap();

            if step % 100 == 0 || step == 1 {
                println!("  Step {:5} | KE={:.3e} | Seg={:.2} Mpc | {:.0}ms/step",
                    step, ke, seg, step_time);
            }
        }

        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &output_dir);
        }
    }

    csv.flush().unwrap();
    let total_time = start.elapsed().as_secs_f64();
    let avg_ms = total_time * 1000.0 / TOTAL_STEPS as f64;

    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let seg_final = sim.segregation().unwrap_or(0.0);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total time: {:.1} s ({:.1} ms/step)", total_time, avg_ms);
    println!();
    println!("  KE: {:.4e} → {:.4e}", ke_0, ke_final);
    println!("  Seg: {:.2} → {:.2} Mpc (max = {:.2})", seg_0, seg_final, seg_max);
    println!();

    let dipole_suppressed = seg_final < 50.0;
    println!("  Dipole suppressed: {} (Seg = {:.2} Mpc)",
        if dipole_suppressed { "YES" } else { "NO" }, seg_final);
    println!("  Output: {}", output_dir);

    // Summary file
    let summary_path = format!("{}/summary.txt", output_dir);
    let mut f = File::create(&summary_path).unwrap();
    writeln!(f, "No-Dipole Zel'dovich 4M — GPU TreePM").unwrap();
    writeln!(f, "=====================================").unwrap();
    writeln!(f, "N: {} ({}³)", n3, N_GRID).unwrap();
    writeln!(f, "Box: {} Mpc", L_BOX).unwrap();
    writeln!(f, "z_init: {}", Z_INIT).unwrap();
    writeln!(f, "k_min (ICs): {}", K_MIN_IDX).unwrap();
    writeln!(f, "k_min (PM): {}", PM_K_MIN).unwrap();
    writeln!(f, "Steps: {}", TOTAL_STEPS).unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "Results:").unwrap();
    writeln!(f, "  Time: {:.1}s ({:.1}ms/step)", total_time, avg_ms).unwrap();
    writeln!(f, "  Seg: {:.2} → {:.2} (max: {:.2})", seg_0, seg_final, seg_max).unwrap();
    writeln!(f, "  Dipole suppressed: {}", dipole_suppressed).unwrap();
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_zeldovich_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("  Generating Zel'dovich ICs with k^-2 spectrum...");

    let dk = 2.0 * PI / L_BOX;
    let half_n = N_GRID / 2;
    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;
    let a_init = 1.0 / (1.0 + Z_INIT);

    // Generate Fourier modes
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    let mut n_kept = 0usize;
    let mut n_suppressed = 0usize;

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
                let k_max_idx = (kx_idx.abs() as usize)
                    .max(ky_idx.abs() as usize)
                    .max(kz_idx.abs() as usize);

                if k_max_idx < K_MIN_IDX || k < 1e-10 {
                    n_suppressed += 1;
                    continue;
                }

                n_kept += 1;

                // P(k) = k^-2 × exp(-(k/k_cut)²)
                let pk = AMPLITUDE * k.powi(-2) * (-(k / K_CUT).powi(2)).exp();
                let amp = pk.sqrt() * a_init;

                let re = normal.sample(&mut rng) * amp;
                let im = normal.sample(&mut rng) * amp;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    println!("    Modes kept: {} | suppressed: {}", n_kept, n_suppressed);

    // Displacement field: ψ_i = -i k_i δ_k / k²
    let mut psi_x = delta_k.clone();
    let mut psi_y = delta_k.clone();
    let mut psi_z = delta_k.clone();

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

    // 1D IFFTs
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_GRID);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            let offset = iz * N_GRID * N_GRID + iy * N_GRID;
            let mut sx: Vec<_> = psi_x[offset..offset+N_GRID].to_vec();
            let mut sy: Vec<_> = psi_y[offset..offset+N_GRID].to_vec();
            let mut sz: Vec<_> = psi_z[offset..offset+N_GRID].to_vec();
            ifft.process(&mut sx);
            ifft.process(&mut sy);
            ifft.process(&mut sz);
            psi_x[offset..offset+N_GRID].copy_from_slice(&sx);
            psi_y[offset..offset+N_GRID].copy_from_slice(&sy);
            psi_z[offset..offset+N_GRID].copy_from_slice(&sz);
        }
    }

    // Create particles
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs = vec![0i32; n3];

    let norm = 1.0 / (n3 as f64);
    let h0 = 0.1;  // Hubble parameter for velocity

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let dx = psi_x[idx].re * norm;
                let dy = psi_y[idx].re * norm;
                let dz = psi_z[idx].re * norm;

                positions.push(x0 + dx);
                positions.push(y0 + dy);
                positions.push(z0 + dz);

                velocities.push(dx * h0);
                velocities.push(dy * h0);
                velocities.push(dz * h0);
            }
        }
    }

    // Random sign assignment
    let n_minus = (n3 as f64 * ETA / (1.0 + ETA)) as usize;
    let mut indices: Vec<usize> = (0..n3).collect();
    indices.shuffle(&mut rng);

    for i in 0..n3 {
        signs[i] = if indices[i] < n_minus { -1 } else { 1 };
    }

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, step: usize, output_dir: &str) {
    let pos = match sim.get_positions() {
        Ok(p) => p,
        Err(_) => return,
    };
    let signs = match sim.get_signs() {
        Ok(s) => s,
        Err(_) => return,
    };

    let n = signs.len();
    let path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
    let mut f = BufWriter::new(File::create(&path).unwrap());

    use std::io::Write as _;
    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    f.write_all(&(step as u64).to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();

    for i in 0..n {
        f.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        f.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        f.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        f.write_all(&(signs[i] as f32).to_le_bytes()).unwrap();
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Error: Requires cuda,cufft features.");
    eprintln!("Build: ./cuda/build_cufft.sh && cargo build --release --features cuda,cufft --bin nodipole_zeldovich_4m");
}
