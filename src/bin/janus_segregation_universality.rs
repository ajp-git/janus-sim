//! JANUS — Segregation Universality Test
//!
//! Tests whether the segregation plateau is universal across different ICs
//!
//! Three runs with identical physics but different velocity amplitudes:
//!   - Run A: α_IC = 1.6 (standard)
//!   - Run B: α_IC = 3.0 (higher velocity)
//!   - Run C: α_IC = 0.8 (lower velocity)
//!
//! If Seg converges to ~0.35-0.45 regardless of ICs, this indicates
//! a universal Janus segregation equilibrium.

use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use std::env;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// COMMON PARAMETERS
const L_BOX: f64 = 200.0;
const N_GRID: usize = 80;          // 80³ ≈ 512k particles
const ETA: f64 = 1.06;
const EPSILON: f64 = 0.18;
const THETA: f64 = 0.7;
const R_CUT: f64 = 18.0;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 1000;
const SNAPSHOT_INTERVAL: usize = 100;
const LOG_INTERVAL: usize = 10;

// Zel'dovich IC parameters
const K_CUT: f64 = 0.25;
const PK_INDEX: f64 = -2.0;
const AMPLITUDE: f64 = 0.02;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    // Get run variant from command line or environment
    let args: Vec<String> = env::args().collect();
    let run_variant = if args.len() > 1 {
        args[1].clone()
    } else {
        env::var("RUN_VARIANT").unwrap_or_else(|_| "A".to_string())
    };

    let (alpha_ic, run_name) = match run_variant.to_uppercase().as_str() {
        "A" => (1.6, "A_standard"),
        "B" => (3.0, "B_high_vel"),
        "C" => (0.8, "C_low_vel"),
        _ => {
            println!("Usage: {} [A|B|C]", args[0]);
            println!("  A: α_IC = 1.6 (standard)");
            println!("  B: α_IC = 3.0 (higher velocity)");
            println!("  C: α_IC = 0.8 (lower velocity)");
            return;
        }
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  JANUS — SEGREGATION UNIVERSALITY TEST                       ║");
    println!("║  Run {}: α_IC = {}                                       ║", run_name, alpha_ic);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_particles = N_GRID * N_GRID * N_GRID;
    println!("Parameters:");
    println!("  N = {} ({:.1}K)", n_particles, n_particles as f64 / 1e3);
    println!("  L = {} Mpc", L_BOX);
    println!("  η = {}", ETA);
    println!("  α_IC = {} ← TEST VARIABLE", alpha_ic);
    println!("  ε = {} Mpc", EPSILON);
    println!("  θ = {}", THETA);
    println!("  R_cut = {} Mpc", R_CUT);
    println!("  dt = {}", DT);
    println!("  z_init = {}", Z_INIT);
    println!("  steps = {}", TOTAL_STEPS);
    println!();

    // Output directory
    let output_dir = format!("/app/output/seg_universality_{}", run_name);
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();
    println!("Output: {}", output_dir);
    println!();

    // Cosmological expansion setup
    println!("--- Cosmological Expansion Setup ---");
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let tau_range = cosmo.tau_end - cosmo.tau_start;
    let dtau_per_dt = tau_range / (TOTAL_STEPS as f64 * DT);

    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("  τ_start = {:.6}", cosmo.tau_start);
    println!("  τ_end = {:.6}", cosmo.tau_end);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!("  z_init = {:.2}", z_init_actual);
    println!("  H_init = {:.6}", h_init);
    println!();

    // Generate Zel'dovich ICs with specified alpha_IC
    // Use same seed for all runs so only alpha_IC differs
    println!("Generating Zel'dovich ICs (seed=42, α_IC={})...", alpha_ic);
    let t0 = Instant::now();
    let (pos, vel, signs) = generate_zeldovich_ics(42, N_GRID, alpha_ic);
    println!("  Generated in {:.1}s", t0.elapsed().as_secs_f64());

    // Compute velocity statistics
    let vel_mag: Vec<f64> = (0..n_particles)
        .map(|i| {
            let vx = vel[i*3];
            let vy = vel[i*3+1];
            let vz = vel[i*3+2];
            (vx*vx + vy*vy + vz*vz).sqrt()
        })
        .collect();
    let vel_mean: f64 = vel_mag.iter().sum::<f64>() / n_particles as f64;
    let vel_max: f64 = vel_mag.iter().cloned().fold(0.0, f64::max);

    println!("  N+ = {}, N- = {}",
             signs.iter().filter(|&&s| s > 0).count(),
             signs.iter().filter(|&&s| s < 0).count());
    println!("  <|v|> = {:.4}, max|v| = {:.4}", vel_mean, vel_max);

    // Convert to GPU format
    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize GPU
    println!("\nInitializing GPU simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX)
        .expect("Failed to create GPU simulation");
    println!("  GPU initialized in {:.1}s", t0.elapsed().as_secs_f64());

    sim.set_theta(THETA);
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(2);

    // Initial state
    let ke0 = sim.kinetic_energy().unwrap_or(1e-20).max(1e-20);
    let seg0 = sim.segregation().unwrap_or(0.0);

    println!("\nInitial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  Seg₀ = {:.4}", seg0);
    println!();

    // Save parameters
    let params_json = format!(r#"{{
  "test": "segregation_universality",
  "run": "{}",
  "alpha_IC": {},
  "N": {},
  "L_box_mpc": {},
  "eta": {},
  "epsilon_mpc": {},
  "theta": {},
  "R_cut_mpc": {},
  "dt": {},
  "z_init": {},
  "total_steps": {},
  "vel_mean": {},
  "vel_max": {},
  "KE_initial": {}
}}"#, run_name, alpha_ic, n_particles, L_BOX, ETA, EPSILON,
     THETA, R_CUT, DT, Z_INIT, TOTAL_STEPS, vel_mean, vel_max, ke0);
    fs::write(format!("{}/params.json", output_dir), params_json).unwrap();

    // Time series file
    let mut ts_file = File::create(format!("{}/time_series.csv", output_dir)).unwrap();
    writeln!(ts_file, "step,tau,z,a,H,KE,KE_ratio,segregation").unwrap();

    // Save initial snapshot
    save_snapshot(&sim, &format!("{}/snapshots/snap_000000.bin", output_dir), n_particles);
    println!("  → Saved snapshot: snap_000000.bin");

    // Run simulation
    println!("\n--- Running {} steps ---\n", TOTAL_STEPS);
    println!("{:>6} {:>8} {:>8} {:>12} {:>12} {:>8}",
             "Step", "z", "a", "KE/KE₀", "Seg", "Time");
    println!("{}", "─".repeat(60));

    let run_start = Instant::now();
    let mut seg_max = seg0;
    let mut seg_max_step = 0;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        let current_tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if current_tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(current_tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { 1.0 / a - 1.0 } else { 0.0 };

        // TreePM step with Hubble friction
        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, h, dtau_per_dt) {
            println!("  ERROR at step {}: {}", step, e);
            break;
        }

        let step_ms = step_start.elapsed().as_millis();

        // Log every LOG_INTERVAL steps
        if step % LOG_INTERVAL == 0 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let ke_ratio = ke / ke0;
            let seg = sim.segregation().unwrap_or(0.0);

            if seg > seg_max {
                seg_max = seg;
                seg_max_step = step;
            }

            writeln!(ts_file, "{},{:.6},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6}",
                     step, current_tau, z, a, h, ke, ke_ratio, seg).unwrap();

            println!("{:>6} {:>8.3} {:>8.5} {:>12.4e} {:>12.4} {:>6}ms",
                     step, z, a, ke_ratio, seg, step_ms);
        }

        // Save snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
            save_snapshot(&sim, &snap_path, n_particles);
            println!("  → Saved snapshot: snap_{:06}.bin", step);
            ts_file.flush().unwrap();
        }
    }

    let total_time = run_start.elapsed().as_secs_f64();
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let seg_final = sim.segregation().unwrap_or(0.0);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  RUN {} COMPLETE (α_IC = {})", run_name, alpha_ic);
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Results:");
    println!("  Total time: {:.1} min ({:.2}s/step)", total_time / 60.0, total_time / TOTAL_STEPS as f64);
    println!("  KE₀ = {:.4e}", ke0);
    println!("  KE_final = {:.4e}", ke_final);
    println!("  KE_ratio = {:.4e}", ke_final / ke0);
    println!();
    println!("  Seg₀ = {:.4}", seg0);
    println!("  Seg_1000 = {:.4} ← KEY METRIC", seg_final);
    println!("  Seg_max = {:.4} @ step {}", seg_max, seg_max_step);
    println!();

    // Save summary
    let summary = format!(r#"{{
  "run": "{}",
  "alpha_IC": {},
  "total_time_min": {:.2},
  "avg_step_ms": {:.1},
  "KE_initial": {:.6e},
  "KE_final": {:.6e},
  "KE_ratio": {:.4e},
  "segregation_initial": {:.6},
  "segregation_1000": {:.6},
  "segregation_max": {:.6},
  "segregation_max_step": {}
}}"#,
        run_name, alpha_ic,
        total_time / 60.0,
        total_time * 1000.0 / TOTAL_STEPS as f64,
        ke0, ke_final, ke_final / ke0,
        seg0, seg_final, seg_max, seg_max_step
    );
    fs::write(format!("{}/summary.json", output_dir), summary).unwrap();

    println!("Output saved to: {}", output_dir);
    println!();

    // Print key result for easy comparison
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  KEY RESULT: Seg_1000 = {:.4}                                ║", seg_final);
    println!("║  (Universal plateau expected: 0.35–0.45)                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_zeldovich_ics(seed: u64, n_grid: usize, alpha_ic: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_total = n_grid * n_grid * n_grid;
    let cell = L_BOX / n_grid as f64;

    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs = Vec::with_capacity(n_total);

    let normal = Normal::new(0.0, 1.0).unwrap();

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let x0 = (ix as f64 + 0.5) * cell - L_BOX / 2.0;
                let y0 = (iy as f64 + 0.5) * cell - L_BOX / 2.0;
                let z0 = (iz as f64 + 0.5) * cell - L_BOX / 2.0;

                let k = 2.0 * PI / L_BOX;
                let phase_x: f64 = rng.random::<f64>() * 2.0 * PI;
                let phase_y: f64 = rng.random::<f64>() * 2.0 * PI;
                let phase_z: f64 = rng.random::<f64>() * 2.0 * PI;

                let mut dx = 0.0;
                let mut dy = 0.0;
                let mut dz = 0.0;
                let mut vx = 0.0;
                let mut vy = 0.0;
                let mut vz = 0.0;

                for mode in 1..=5 {
                    let km = mode as f64 * k;
                    if km < K_CUT {
                        let amp = AMPLITUDE * (km / k).powf(PK_INDEX / 2.0);
                        dx += amp * (km * x0 + phase_x).sin();
                        dy += amp * (km * y0 + phase_y).sin();
                        dz += amp * (km * z0 + phase_z).sin();
                        // Velocity scaled by alpha_IC
                        vx += amp * km * (km * x0 + phase_x).cos() * alpha_ic;
                        vy += amp * km * (km * y0 + phase_y).cos() * alpha_ic;
                        vz += amp * km * (km * z0 + phase_z).cos() * alpha_ic;
                    }
                }

                dx += normal.sample(&mut rng) * cell * 0.1;
                dy += normal.sample(&mut rng) * cell * 0.1;
                dz += normal.sample(&mut rng) * cell * 0.1;

                let x = ((x0 + dx) % L_BOX + L_BOX) % L_BOX - L_BOX / 2.0;
                let y = ((y0 + dy) % L_BOX + L_BOX) % L_BOX - L_BOX / 2.0;
                let z = ((z0 + dz) % L_BOX + L_BOX) % L_BOX - L_BOX / 2.0;

                positions.push(x);
                positions.push(y);
                positions.push(z);

                velocities.push(vx);
                velocities.push(vy);
                velocities.push(vz);

                let sign = if rng.random::<f64>() < 1.0 / (1.0 + ETA) { 1 } else { -1 };
                signs.push(sign);
            }
        }
    }

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &str, n_particles: usize) {
    let (positions, _, signs) = sim.get_particles().expect("get_particles failed");

    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    writer.write_all(&(n_particles as u64).to_le_bytes()).unwrap();

    let n = positions.len() / 3;
    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        let sign = if signs[i] > 0 { 1.0f32 } else { -1.0f32 };

        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    println!("CUDA + cuFFT required. Use: cargo run --release --features \"cuda cufft\" --bin janus_segregation_universality [A|B|C]");
}
