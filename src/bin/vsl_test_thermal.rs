//! VSL Test with Thermal ICs — Validate physics before production
//!
//! Fix: Replace virial velocities with proper thermal velocities
//! sigma_v = sqrt(k_B T / (mu m_p)) = 0.012 Mpc/Gyr = 12 km/s
//!
//! Expected step 0: v_rms ≈ 20 km/s (both populations)
//! Expected step 50: v_rms < 500 km/s (no runaway)

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

const MPC_GYR_TO_KMS: f64 = 977.8;
const N_CELLS: usize = 32;

// Test parameters
const N_PARTICLES: usize = 100_000;
const BOX_SIZE: f64 = 100.0;
const MU: f64 = 19.0;
const N_STEPS: usize = 100;
const DT: f64 = 0.001;

// Janus
const ETA: f64 = 1.045;

// Thermal velocity: sigma_v = sqrt(k_B T / (mu_mol m_p))
// k_B/m_p in code units = 8.7e-9 Mpc²/Gyr²/K
// T = 1e4 K, mu_mol = 0.6
// sigma_v = sqrt(8.7e-9 * 1e4 / 0.6) = 0.01204 Mpc/Gyr ≈ 12 km/s
const K_B_OVER_MP_CODE: f64 = 8.7e-9;
const T_INIT: f64 = 1.0e4;  // K
const MU_MOL: f64 = 0.6;

#[cfg(feature = "cuda")]
fn main() {
    run_thermal_test();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
}

#[cfg(feature = "cuda")]
fn run_thermal_test() {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    let c_ratio = 1.0 / MU.sqrt();

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    // Thermal velocity dispersion
    let sigma_v = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();
    let sigma_v_kms = sigma_v * MPC_GYR_TO_KMS;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           VSL THERMAL TEST — Validate Physics                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc", BOX_SIZE);
    println!("║  μ = {} → c⁻/c⁺ = {:.4}", MU, c_ratio);
    println!("║  T_init = {} K → σ_v = {:.4} Mpc/Gyr = {:.1} km/s", T_INIT, sigma_v, sigma_v_kms);
    println!("║  Expected v_rms(step 0) ≈ {:.0} km/s", sigma_v_kms * 3.0_f64.sqrt());
    println!("║  dt = {} Gyr, steps = {}", DT, N_STEPS);
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Initialize GPU simulation (this creates virial velocities - we'll override)
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

    // CRITICAL FIX: Replace virial velocities with thermal velocities
    println!("Replacing virial velocities with thermal velocities...");

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let normal = Normal::new(0.0, sigma_v).unwrap();

    let mut thermal_vel = vec![0.0f64; N_PARTICLES * 3];
    for i in 0..N_PARTICLES {
        thermal_vel[i * 3]     = normal.sample(&mut rng);
        thermal_vel[i * 3 + 1] = normal.sample(&mut rng);
        thermal_vel[i * 3 + 2] = normal.sample(&mut rng);
    }

    if let Err(e) = gpu_sim.set_velocities(&thermal_vel) {
        eprintln!("Failed to set thermal velocities: {}", e);
        return;
    }

    println!("Thermal velocities set.\n");

    let half_box = BOX_SIZE / 2.0;
    let cell_size = BOX_SIZE / N_CELLS as f64;

    println!("{:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
             "Step", "ρ+_max", "ρ-_max", "v_rms+", "v_rms-", "time");
    println!("{:-<75}", "");

    let sim_start = Instant::now();

    for step in 0..=N_STEPS {
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

        // Output every 10 steps
        if step % 10 == 0 || step == N_STEPS {
            println!("{:>6} | {:>10.0} | {:>10.0} | {:>9.1} | {:>9.1} | {:>9.2}s",
                     step, rho_plus_max, rho_minus_max, v_rms_plus, v_rms_minus, elapsed);
        }

        // VALIDATION at step 0
        if step == 0 {
            let expected_v_rms = sigma_v_kms * 3.0_f64.sqrt();
            if (v_rms_plus - expected_v_rms).abs() > expected_v_rms * 0.5 {
                println!("⚠️  WARNING step 0: v_rms+ = {:.1} km/s, expected ≈ {:.1} km/s", v_rms_plus, expected_v_rms);
            } else {
                println!("✓ Step 0 v_rms+ = {:.1} km/s ≈ expected {:.1} km/s", v_rms_plus, expected_v_rms);
            }
            if (v_rms_minus - expected_v_rms).abs() > expected_v_rms * 0.5 {
                println!("⚠️  WARNING step 0: v_rms- = {:.1} km/s, expected ≈ {:.1} km/s", v_rms_minus, expected_v_rms);
            } else {
                println!("✓ Step 0 v_rms- = {:.1} km/s ≈ expected {:.1} km/s", v_rms_minus, expected_v_rms);
            }
        }

        // VALIDATION at step 50
        if step == 50 {
            println!("\n--- STEP 50 VALIDATION ---");
            if v_rms_minus > 500.0 {
                println!("❌ FAIL: v_rms- = {:.1} km/s > 500 km/s (RUNAWAY)", v_rms_minus);
            } else {
                println!("✓ PASS: v_rms- = {:.1} km/s < 500 km/s (stable)", v_rms_minus);
            }
            if v_rms_plus > 500.0 {
                println!("❌ FAIL: v_rms+ = {:.1} km/s > 500 km/s", v_rms_plus);
            } else {
                println!("✓ PASS: v_rms+ = {:.1} km/s < 500 km/s (stable)", v_rms_plus);
            }
            println!("----------------------------\n");
        }

        if step >= N_STEPS {
            break;
        }

        // Step (no expansion for test)
        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(DT, 1.0, 0.0, 0.0) {
            eprintln!("GPU step failed: {}", e);
            break;
        }
    }

    let total_time = sim_start.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE: {:.1}s for {} steps ({:.3}s/step)", total_time, N_STEPS, total_time / N_STEPS as f64);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
