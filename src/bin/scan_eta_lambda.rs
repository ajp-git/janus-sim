//! Parameter scan for (η, λ₀) — publishable results
//! 9 runs: η ∈ {0.90, 1.00, 1.10} × λ₀ ∈ {10, 15, 20} Mpc
//! 2M particles, 500 Mpc box, z=5→z=0, filament criterion at z=0

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const N_PARTICLES: usize = 2_000_000;
const BOX_SIZE: f64 = 1000.0;  // 8× volume → multiple halos per species
const Z_INIT: f64 = 5.0;
const STEPS: usize = 1500;  // z=5 → z=0
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;
const SEED: u64 = 42;
const DT: f64 = 0.01;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Requires --features cuda");
        std::process::exit(1);
    }
    #[cfg(feature = "cuda")]
    run_scan();
}

#[cfg(feature = "cuda")]
fn run_scan() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PARAMETER SCAN v4 — Boundary Grid (Purity Metric)           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  N = 2M, box = 1000 Mpc, z = 5 → 0                           ║");
    println!("║  η ∈ {{0.85, 0.87, 0.88}} × λ₀ ∈ {{1, 2, 3}} Mpc               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let etas = [0.85, 0.87, 0.88];
    let lambdas = [1.0, 2.0, 3.0];  // Push to boundary λ=1
    // Checkpoints: z≈3, z≈1.5, z≈0.5, z=0
    let snapshot_steps = [300, 750, 1200, 1500];

    let base_dir = Path::new("/app/output/scan_boundary");
    fs::create_dir_all(base_dir).ok();

    // Summary CSV — P is now the primary metric
    let summary_path = base_dir.join("summary.csv");
    let mut summary = BufWriter::new(File::create(&summary_path).unwrap());
    writeln!(summary, "run,eta,lambda,P_z3,P_z1p5,P_z0p5,P_z0,time_min").unwrap();

    let mut run_id = 0;
    for &eta in &etas {
        for &lambda in &lambdas {
            run_id += 1;
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("RUN {}/9: η={:.2}, λ₀={:.0} Mpc", run_id, eta, lambda);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let run_dir = base_dir.join(format!("eta{:.2}_lam{:.0}", eta, lambda));
            fs::create_dir_all(&run_dir).ok();
            let snap_dir = run_dir.join("snapshots");
            fs::create_dir_all(&snap_dir).ok();

            // Time series CSV for this run
            let ts_path = run_dir.join("time_series.csv");
            let mut ts_file = BufWriter::new(File::create(&ts_path).unwrap());
            writeln!(ts_file, "step,z,a,purity,ke_ratio").unwrap();

            let start = Instant::now();

            // Generate ΛCDM ICs
            let (pos_data, vel_data, signs_data) = generate_lcdm_ics(N_PARTICLES, BOX_SIZE, Z_INIT, SEED);

            // Create simulation
            let n_pos = signs_data.iter().filter(|&&s| s > 0).count();
            let n_neg = signs_data.iter().filter(|&&s| s < 0).count();
            println!("  N+ = {}, N- = {}", n_pos, n_neg);

            let mut sim = GpuNBodyTwoPass::with_custom_ics(
                pos_data, vel_data, signs_data, BOX_SIZE
            ).expect("Failed to create simulation");

            sim.set_theta(THETA);
            sim.set_softening(SOFTENING);
            sim.set_lambda_0(lambda);

            let ke0 = sim.kinetic_energy().unwrap_or(1.0).max(1e-20);

            // Cosmology using JanusParams
            let params = JanusParams::from_eta(eta);
            let cosmo = CosmoInterpolator::new(&params, Z_INIT);
            let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

            let mut purs = vec![0.0; 4];  // P is now the primary metric
            let mut snap_idx = 0;

            for step in 1..=STEPS {
                let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
                let (a, h) = if tau <= cosmo.tau_end {
                    cosmo.get_params_at_tau(tau)
                } else {
                    (1.0, 0.0)
                };
                let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };

                sim.set_current_z(z);
                sim.step_dkd(DT, h, dtau_per_dt).expect("Step failed");

                // Write time series every 10 steps
                if step % 10 == 0 {
                    let purity = sim.local_purity(32).unwrap_or(0.0);
                    let ke = sim.kinetic_energy().unwrap_or(0.0);
                    let ke_ratio = ke / ke0;
                    writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4e}", step, z, a, purity, ke_ratio).unwrap();
                }

                // Log and save at specified steps (z≈3, z≈1.5, z≈0.5, z=0)
                if snapshot_steps.contains(&step) {
                    let purity = sim.local_purity(32).unwrap_or(0.0);
                    purs[snap_idx] = purity;
                    snap_idx += 1;
                    println!("  step {:4} | z={:.3} | P={:.4} <<<", step, z, purity);
                    save_snapshot(&sim, &snap_dir, step, z);
                }

                if step % 100 == 0 && !snapshot_steps.contains(&step) {
                    let purity = sim.local_purity(32).unwrap_or(0.0);
                    println!("  step {:4} | z={:.2} | P={:.4}", step, z, purity);
                }
            }

            ts_file.flush().unwrap();

            let elapsed = start.elapsed().as_secs_f64() / 60.0;
            println!("  Completed in {:.1} min", elapsed);
            println!("  Final P(z=0) = {:.4}", purs[3]);

            writeln!(summary, "{},{:.2},{:.0},{:.4},{:.4},{:.4},{:.4},{:.1}",
                run_id, eta, lambda, purs[0], purs[1], purs[2], purs[3], elapsed).unwrap();
            summary.flush().unwrap();
        }
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SCAN COMPLETE — Results in /app/output/scan_boundary/       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn generate_lcdm_ics(n: usize, box_size: f64, z_init: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    use std::f64::consts::PI;

    println!("  Generating ΛCDM Zel'dovich ICs...");

    let n_per_sign = n / 2;
    let n_grid = (n_per_sign as f64).powf(1.0/3.0).ceil() as usize;
    let cell = box_size / n_grid as f64;

    // Cosmological parameters
    let h = 0.7;
    let omega_m = 0.3;
    let omega_b = 0.05;
    let n_s = 0.965;
    let sigma8 = 0.8;

    // Simple transfer function (Eisenstein & Hu approximation)
    let theta = 2.725 / 2.7;
    let k_eq = 0.0746 * omega_m * h * h / (theta * theta);

    // Growth factor D(z)
    let a = 1.0 / (1.0 + z_init);
    let omega_m_z = omega_m / (omega_m + (1.0 - omega_m) * a.powi(3));
    let d_z = a * (omega_m_z.powf(4.0/7.0) - (1.0 - omega_m_z) +
              (1.0 + omega_m_z/2.0) * (1.0 + (1.0-omega_m_z)/70.0)).powf(-1.0);
    let d_0 = 1.0;  // normalized

    // Generate displacement field via FFT
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let box_half = box_size / 2.0;

    let mut pos_data = Vec::with_capacity(n * 3);
    let mut vel_data = Vec::with_capacity(n * 3);
    let mut signs_data = Vec::with_capacity(n);

    // Velocity scaling: v = H(z) * f(Ω) * ψ
    let f_omega = omega_m.powf(0.55);
    let h0_gyr = 0.0715;
    let e_z = ((omega_m * (1.0 + z_init).powi(3)) + (1.0 - omega_m)).sqrt();
    let h_z = h0_gyr * e_z;
    let vel_factor = h_z * f_omega;

    // Displacement amplitude (empirically tuned for σ8=0.8)
    let disp_scale = 25.0 * d_z / d_0;

    let mut count = 0;
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                if count >= n { break; }

                // Grid position
                let x0 = (ix as f64 + 0.5) * cell - box_half;
                let y0 = (iy as f64 + 0.5) * cell - box_half;
                let z0 = (iz as f64 + 0.5) * cell - box_half;

                // Random displacement (simplified - should use FFT for proper correlations)
                let k = 2.0 * PI / box_size * ((ix*ix + iy*iy + iz*iz) as f64).sqrt().max(1.0);
                let pk = k.powf(n_s) / (1.0 + (k/k_eq).powi(2)).powf(2.0);
                let amp = (pk.sqrt() * disp_scale * cell / box_size).min(cell * 0.5);

                let psi_x: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_y: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_z: f64 = rng.gen::<f64>() * 2.0 - 1.0;

                let dx = psi_x * amp;
                let dy = psi_y * amp;
                let dz = psi_z * amp;

                // Position with displacement
                let x = x0 + dx;
                let y = y0 + dy;
                let z = z0 + dz;

                // Velocity from displacement
                let vx = dx * vel_factor;
                let vy = dy * vel_factor;
                let vz = dz * vel_factor;

                // Random sign
                let sign: i8 = if rng.gen::<bool>() { 1 } else { -1 };

                pos_data.extend_from_slice(&[x as f32, y as f32, z as f32]);
                vel_data.extend_from_slice(&[vx as f32, vy as f32, vz as f32]);
                signs_data.push(sign);

                count += 1;
            }
        }
    }

    // Fill remaining with random if needed
    while count < n {
        let x: f64 = rng.gen::<f64>() * box_size - box_half;
        let y: f64 = rng.gen::<f64>() * box_size - box_half;
        let z: f64 = rng.gen::<f64>() * box_size - box_half;
        let sign: i8 = if rng.gen::<bool>() { 1 } else { -1 };

        pos_data.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel_data.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs_data.push(sign);
        count += 1;
    }

    let n_pos = signs_data.iter().filter(|&&s| s > 0).count();
    let n_neg = signs_data.len() - n_pos;
    println!("    N+ = {}, N- = {} (ratio {:.4})", n_pos, n_neg, n_pos as f64 / n_neg as f64);

    (pos_data, vel_data, signs_data)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &PathBuf, step: usize, z: f64) {
    use std::io::BufWriter;

    let (positions, _, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let snap_path = path.join(format!("snap_{:04}_z{:.2}.bin", step, z));

    let file = match File::create(&snap_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    // Header: N, box_size, step, z
    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(500.0f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

    // Positions and signs interleaved
    for i in 0..n {
        let _ = writer.write_all(&positions[i*3].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+1].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+2].to_le_bytes());
        let _ = writer.write_all(&(signs[i] as i8).to_le_bytes());
    }
}
