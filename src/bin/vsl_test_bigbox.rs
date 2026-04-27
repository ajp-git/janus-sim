//! VSL Test Big Box — 1000 Mpc to reduce density and VSL forces
//!
//! N = 500k, Box = 1000 Mpc, μ = 19
//! Thermal ICs, 200 steps
//! Expected: v_rms- < 1000 km/s at step 50

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

const MPC_GYR_TO_KMS: f64 = 977.8;
const N_CELLS: usize = 32;

// Test parameters
const N_PARTICLES: usize = 500_000;
const BOX_SIZE: f64 = 1000.0;  // 10× larger box
const MU: f64 = 19.0;
const N_STEPS: usize = 200;
const DT: f64 = 0.001;

// Janus
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;

// Thermal velocity
const K_B_OVER_MP_CODE: f64 = 8.7e-9;
const T_INIT: f64 = 1.0e4;
const MU_MOL: f64 = 0.6;

#[cfg(feature = "cuda")]
fn main() {
    run_bigbox_test();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
}

#[cfg(feature = "cuda")]
fn run_bigbox_test() {
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    let c_ratio = 1.0 / MU.sqrt();

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    // Thermal velocity
    let sigma_v = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();
    let sigma_v_kms = sigma_v * MPC_GYR_TO_KMS;

    // Density comparison: 500k in 1000³ vs 100k in 100³
    // Old: 100k / 100³ = 0.1 particles/Mpc³
    // New: 500k / 1000³ = 0.0005 particles/Mpc³ → 200× lower density!
    let density = N_PARTICLES as f64 / (BOX_SIZE * BOX_SIZE * BOX_SIZE);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           VSL BIG BOX TEST — 1000 Mpc                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc → density = {:.2e} /Mpc³", BOX_SIZE, density);
    println!("║  μ = {} → c⁻/c⁺ = {:.4}", MU, c_ratio);
    println!("║  T_init = {} K → σ_v = {:.1} km/s", T_INIT, sigma_v_kms);
    println!("║  z_init = {}, dt = {} Gyr, steps = {}", Z_INIT, DT, N_STEPS);
    println!("║  Expected: v_rms- < 1000 km/s at step 50", );
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("Initializing GPU...");
    let mut gpu_sim = match GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        }
    };

    gpu_sim.set_c_ratio(c_ratio);
    gpu_sim.set_theta(0.8);

    // Replace virial velocities with thermal
    println!("Setting thermal velocities...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let normal = Normal::new(0.0, sigma_v).unwrap();

    let mut thermal_vel = vec![0.0f64; N_PARTICLES * 3];
    for i in 0..N_PARTICLES {
        thermal_vel[i * 3]     = normal.sample(&mut rng);
        thermal_vel[i * 3 + 1] = normal.sample(&mut rng);
        thermal_vel[i * 3 + 2] = normal.sample(&mut rng);
    }

    if let Err(e) = gpu_sim.set_velocities(&thermal_vel) {
        eprintln!("Failed to set velocities: {}", e);
        return;
    }

    let half_box = BOX_SIZE / 2.0;
    let cell_size = BOX_SIZE / N_CELLS as f64;

    println!("\n{:>6} | {:>6} | {:>8} | {:>8} | {:>10} | {:>10} | {:>8}",
             "Step", "z", "ρ+_max", "ρ-_max", "v_rms+", "v_rms-", "time");
    println!("{:-<80}", "");

    let sim_start = Instant::now();
    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to get positions: {}", e); break; }
        };
        let vel = match gpu_sim.get_velocities() {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to get velocities: {}", e); break; }
        };
        let signs = gpu_sim.signs();

        // Compute metrics
        let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut counts_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut v2_plus = 0.0f64;
        let mut v2_minus = 0.0f64;
        let mut n_plus = 0usize;
        let mut n_minus = 0usize;

        for i in 0..N_PARTICLES {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];
            let vx = vel[i * 3];
            let vy = vel[i * 3 + 1];
            let vz = vel[i * 3 + 2];

            let ix = ((px + half_box) / cell_size).floor() as usize % N_CELLS;
            let iy = ((py + half_box) / cell_size).floor() as usize % N_CELLS;
            let iz = ((pz + half_box) / cell_size).floor() as usize % N_CELLS;
            let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;

            if signs[i] > 0 {
                counts_plus[idx] += 1;
                v2_plus += vx * vx + vy * vy + vz * vz;
                n_plus += 1;
            } else {
                counts_minus[idx] += 1;
                v2_minus += vx * vx + vy * vy + vz * vz;
                n_minus += 1;
            }
        }

        let rho_plus_max = *counts_plus.iter().max().unwrap_or(&0) as f64;
        let rho_minus_max = *counts_minus.iter().max().unwrap_or(&0) as f64;
        let v_rms_plus = if n_plus > 0 { (v2_plus / n_plus as f64).sqrt() * MPC_GYR_TO_KMS } else { 0.0 };
        let v_rms_minus = if n_minus > 0 { (v2_minus / n_minus as f64).sqrt() * MPC_GYR_TO_KMS } else { 0.0 };

        let elapsed = sim_start.elapsed().as_secs_f64();

        // Output at specific steps
        if step == 0 || step == 50 || step == 100 || step == 150 || step == 200 {
            println!("{:>6} | {:>6.3} | {:>8.0} | {:>8.0} | {:>9.1} | {:>9.1} | {:>7.1}s",
                     step, z, rho_plus_max, rho_minus_max, v_rms_plus, v_rms_minus, elapsed);

            // Validation
            if step == 0 {
                if v_rms_plus < 50.0 && v_rms_minus < 50.0 {
                    println!("  ✓ Thermal ICs correct");
                }
            }
            if step == 50 {
                if v_rms_minus < 1000.0 {
                    println!("  ✓ PASS: v_rms- = {:.0} km/s < 1000 km/s", v_rms_minus);
                } else {
                    println!("  ❌ FAIL: v_rms- = {:.0} km/s > 1000 km/s (still runaway)", v_rms_minus);
                }
            }
        }

        if step >= N_STEPS {
            break;
        }

        // Hubble parameter H(a) ~ 0.07/a^1.5 in Gyr^-1
        let h = 0.07 / a.powf(1.5);

        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0) {
            eprintln!("GPU step failed: {}", e);
            break;
        }

        a += a * h * DT;
    }

    let total_time = sim_start.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE: {:.1}s for {} steps ({:.3}s/step)", total_time, N_STEPS, total_time / N_STEPS as f64);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
