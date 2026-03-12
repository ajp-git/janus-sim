//! Validation 500K — Comparaison ε=0.15 vs ε=0.25
//!
//! Usage:
//!   cargo run --release --features cuda,cufft --bin val_500k -- --epsilon 0.15
//!   cargo run --release --features cuda,cufft --bin val_500k -- --epsilon 0.25

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 80;              // 80³ = 512,000 particles
const L_BOX: f64 = 492.0;              // Mpc
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 5000;
const SNAPSHOT_INTERVAL: usize = 50;   // 100 snapshots total
const THETA: f64 = 0.7;
const R_CUT: f64 = 30.0;
const DTAU_PER_DT: f64 = 0.0;

// Fixed validation parameters
const K_MIN: usize = 3;
const HUBBLE: f64 = 0.01;
const ETA: f64 = 1.045;

// P(k) IC parameters
const K_CUT: f64 = 0.25;
const PK_INDEX: f64 = -2.0;
const AMPLITUDE: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let epsilon: f64 = if args.len() > 2 && args[1] == "--epsilon" {
        args[2].parse().expect("Invalid epsilon value")
    } else {
        eprintln!("Usage: val_500k --epsilon <0.15|0.25>");
        std::process::exit(1);
    };

    let run_name = if (epsilon - 0.15).abs() < 0.01 {
        "val_500k_eps015"
    } else if (epsilon - 0.25).abs() < 0.01 {
        "val_500k_eps025"
    } else {
        "val_500k_custom"
    };

    let output_dir = format!("/app/output/{}", run_name);
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  VALIDATION 500K — {}", run_name);
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  ε      = {:.2} Mpc", epsilon);
    println!("  k_min  = {}", K_MIN);
    println!("  H      = {:.2}", HUBBLE);
    println!("  η      = {:.3}", ETA);
    println!("  steps  = {}", TOTAL_STEPS);
    println!("  output = {}", output_dir);
    println!();

    // Generate ICs
    println!("Generating Zel'dovich ICs...");
    let seed = 42u64;
    let (positions, velocities, signs) = generate_zeldovich_ics(seed, ETA, K_MIN);
    let n3 = N_GRID * N_GRID * N_GRID;
    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let n_negative = n3 - n_positive;
    println!("  N+ = {}, N- = {} (η = {:.4})", n_positive, n_negative, n_negative as f64 / n_positive as f64);

    // Convert to GPU format
    let pos_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX)
        .expect("Failed to initialize simulation");

    sim.set_theta(THETA);
    sim.set_softening(epsilon);
    sim.set_pm_k_min(K_MIN);

    // Time series CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,seg,ke_ratio,pe_binding,virial").unwrap();

    // Initial energies
    let ke_init = sim.kinetic_energy().unwrap_or(1.0);
    let pe_init = sim.potential_energy_binding_sampled(500).unwrap_or(-1.0);

    // Save initial snapshot
    save_snapshot(&sim, 0, &output_dir, n_positive, n3);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  RUNNING SIMULATION");
    println!("═══════════════════════════════════════════════════════════════\n");

    let start = Instant::now();
    let mut z = Z_INIT;

    for step in 1..=TOTAL_STEPS {
        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, HUBBLE, DTAU_PER_DT) {
            eprintln!("Step {} error: {}", step, e);
            break;
        }

        // Update z (simple approximation)
        z = Z_INIT - (step as f64 / TOTAL_STEPS as f64) * Z_INIT;

        // Metrics every 50 steps
        if step % 50 == 0 {
            let seg = sim.segregation().unwrap_or(0.0);
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let pe = sim.potential_energy_binding_sampled(500).unwrap_or(-1.0);
            let ke_ratio = ke / ke_init;
            let virial = if pe.abs() > 1e-10 { 2.0 * ke / pe.abs() } else { 0.0 };

            writeln!(csv, "{},{:.4},{:.6},{:.4},{:.6e},{:.4}",
                step, z, seg, ke_ratio, pe, virial).unwrap();

            println!("  Step {:5} | z={:.3} | Seg={:.4} | KE/KE0={:.2} | virial={:.1}",
                step, z, seg, ke_ratio, virial);
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &output_dir, n_positive, n3);
        }
    }

    let runtime = start.elapsed().as_secs_f64();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Runtime:   {:.1}s ({:.1} min)", runtime, runtime / 60.0);
    println!("  Snapshots: {}", TOTAL_STEPS / SNAPSHOT_INTERVAL);
    println!("  Output:    {}", output_dir);
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// SNAPSHOT FORMAT (compatible with analyse_pk.py)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, step: usize, output_dir: &str, n_positive: usize, n_total: usize) {
    let filename = format!("{}/snap_{:06}.bin", output_dir, step);
    let (positions, _, signs) = sim.get_particles().expect("get_particles failed");

    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Header: n (u64), step (u64), padding (u64)
    writer.write_all(&(n_total as u64).to_le_bytes()).unwrap();
    writer.write_all(&(step as u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();

    // Data: n × (x, y, z, sign) as f32
    let n = positions.len() / 3;
    for i in 0..n {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let sign = signs[i] as f32;
        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH IC GENERATION
// ═══════════════════════════════════════════════════════════════════════════

fn generate_zeldovich_ics(seed: u64, eta: f64, k_min: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ng = N_GRID;
    let ng3 = ng * ng * ng;
    let cell = L_BOX / ng as f64;

    // Random phases for P(k) field
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(ng);
    let ifft = planner.plan_fft_inverse(ng);

    // Generate 3D displacement field
    let mut phi_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); ng3];

    for kx in 0..ng {
        for ky in 0..ng {
            for kz in 0..ng {
                let ikx = if kx <= ng/2 { kx as i32 } else { kx as i32 - ng as i32 };
                let iky = if ky <= ng/2 { ky as i32 } else { ky as i32 - ng as i32 };
                let ikz = if kz <= ng/2 { kz as i32 } else { kz as i32 - ng as i32 };

                let k_idx = (ikx.abs() as usize).max(iky.abs() as usize).max(ikz.abs() as usize);

                // Skip DC and filtered modes
                if k_idx < k_min { continue; }

                let k_phys = 2.0 * PI / L_BOX * ((ikx*ikx + iky*iky + ikz*ikz) as f64).sqrt();
                if k_phys > K_CUT || k_phys < 1e-10 { continue; }

                let pk = k_phys.powf(PK_INDEX);
                let amp = (pk * AMPLITUDE).sqrt();

                let phase = rng.gen::<f64>() * 2.0 * PI;
                let idx = kx + ng * (ky + ng * kz);
                phi_k[idx] = Complex::new(amp * phase.cos(), amp * phase.sin());
            }
        }
    }

    // IFFT to get displacement potential
    let mut phi_x = phi_k.clone();
    // 3D IFFT via 1D transforms
    for iz in 0..ng {
        for iy in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|ix| phi_x[ix + ng*(iy + ng*iz)]).collect();
            ifft.process(&mut row);
            for ix in 0..ng {
                phi_x[ix + ng*(iy + ng*iz)] = row[ix];
            }
        }
    }
    for iz in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iy| phi_x[ix + ng*(iy + ng*iz)]).collect();
            ifft.process(&mut row);
            for iy in 0..ng {
                phi_x[ix + ng*(iy + ng*iz)] = row[iy];
            }
        }
    }
    for iy in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iz| phi_x[ix + ng*(iy + ng*iz)]).collect();
            ifft.process(&mut row);
            for iz in 0..ng {
                phi_x[ix + ng*(iy + ng*iz)] = row[iz];
            }
        }
    }

    // Gradient for displacements (central differences)
    let phi_real: Vec<f64> = phi_x.iter().map(|c| c.re / ng3 as f64).collect();

    let mut positions = vec![0.0; ng3 * 3];
    let mut velocities = vec![0.0; ng3 * 3];

    let growth_factor = 1.0 / (1.0 + Z_INIT);

    for ix in 0..ng {
        for iy in 0..ng {
            for iz in 0..ng {
                let idx = ix + ng * (iy + ng * iz);

                // Grid position (centered)
                let x0 = (ix as f64 + 0.5) * cell - L_BOX / 2.0;
                let y0 = (iy as f64 + 0.5) * cell - L_BOX / 2.0;
                let z0 = (iz as f64 + 0.5) * cell - L_BOX / 2.0;

                // Gradient via finite differences
                let ixp = (ix + 1) % ng;
                let ixm = (ix + ng - 1) % ng;
                let iyp = (iy + 1) % ng;
                let iym = (iy + ng - 1) % ng;
                let izp = (iz + 1) % ng;
                let izm = (iz + ng - 1) % ng;

                let dphi_dx = (phi_real[ixp + ng*(iy + ng*iz)] - phi_real[ixm + ng*(iy + ng*iz)]) / (2.0 * cell);
                let dphi_dy = (phi_real[ix + ng*(iyp + ng*iz)] - phi_real[ix + ng*(iym + ng*iz)]) / (2.0 * cell);
                let dphi_dz = (phi_real[ix + ng*(iy + ng*izp)] - phi_real[ix + ng*(iy + ng*izm)]) / (2.0 * cell);

                // Displacement
                let disp_x = -dphi_dx * growth_factor * L_BOX;
                let disp_y = -dphi_dy * growth_factor * L_BOX;
                let disp_z = -dphi_dz * growth_factor * L_BOX;

                // Apply with periodic wrap
                positions[3*idx]     = wrap(x0 + disp_x, L_BOX);
                positions[3*idx + 1] = wrap(y0 + disp_y, L_BOX);
                positions[3*idx + 2] = wrap(z0 + disp_z, L_BOX);

                // Velocity ∝ displacement (Zel'dovich)
                let vel_scale = 0.1;
                velocities[3*idx]     = disp_x * vel_scale;
                velocities[3*idx + 1] = disp_y * vel_scale;
                velocities[3*idx + 2] = disp_z * vel_scale;
            }
        }
    }

    // Random signs with ratio eta = N-/N+
    let n_positive = (ng3 as f64 / (1.0 + eta)) as usize;
    let mut signs: Vec<i32> = vec![1; n_positive];
    signs.extend(vec![-1; ng3 - n_positive]);
    signs.shuffle(&mut rng);

    (positions, velocities, signs)
}

fn wrap(x: f64, l: f64) -> f64 {
    let half = l / 2.0;
    if x > half { x - l }
    else if x < -half { x + l }
    else { x }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires --features cuda,cufft");
}
