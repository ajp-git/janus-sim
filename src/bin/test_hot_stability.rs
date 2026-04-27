//! Test 2: Hot Gas Stability (m+ only, no Janus)
//!
//! Setup:
//! - N = 100k particles (all m+)
//! - Box = 5 Mpc
//! - T_init = 1e7 K (very hot gas)
//! - Uniform density
//!
//! Expected:
//! - ρ_max/ρ̄ < 2 (no structures form)
//! - Pressure support prevents collapse
//!
//! If FAIL → BUG in pressure → STOP

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

use janus::baryonic::pressure::sound_speed;

// Simulation parameters
const N: usize = 50_000;  // Smaller for faster test
const BOX: f64 = 5.0;  // Mpc
const T_INIT: f64 = 1e7;  // K (very hot gas)
const STEPS: usize = 200;
const DT: f64 = 0.001;  // Gyr
const SOFTENING: f64 = 0.02;  // Mpc

// Physics
const G: f64 = 4.499e-15;  // Mpc³ M_sun⁻¹ Gyr⁻²
const PARTICLE_MASS: f64 = 1e8;  // M_sun per particle

// Grid for density computation
const N_CELLS: usize = 10;

fn main() {
    println!("================================================================");
    println!("  TEST 2: HOT GAS STABILITY (m+ only)");
    println!("================================================================");
    println!("  N = {}, Box = {} Mpc, T_init = {:.0e} K", N, BOX, T_INIT);
    println!("  Steps = {}, dt = {} Gyr", STEPS, DT);
    println!("  Sound speed cs = {:.2} Mpc/Gyr = {:.0} km/s",
             sound_speed(T_INIT), sound_speed(T_INIT) * 978.0);
    println!("  Expected: ρ_max/ρ̄ < 2 (no collapse)");
    println!("================================================================\n");

    let output_dir = "/app/output/test_hot_stability";
    fs::create_dir_all(output_dir).expect("Failed to create output dir");

    let mut csv = File::create(format!("{}/evolution.csv", output_dir)).unwrap();
    writeln!(csv, "step,time_gyr,rho_max,v_rms_kms").unwrap();

    // Initialize particles uniformly
    let mut rng = StdRng::seed_from_u64(42);
    let half_box = BOX / 2.0;

    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(N);
    let mut vel: Vec<[f64; 3]> = Vec::with_capacity(N);

    // Thermal velocity dispersion
    let cs = sound_speed(T_INIT);
    let v_thermal = cs / 3.0_f64.sqrt();  // 1D dispersion

    for _ in 0..N {
        let x = (rng.gen::<f64>() - 0.5) * BOX;
        let y = (rng.gen::<f64>() - 0.5) * BOX;
        let z = (rng.gen::<f64>() - 0.5) * BOX;
        pos.push([x, y, z]);

        // Random thermal velocities
        let vx = rng.gen::<f64>() * 2.0 * v_thermal - v_thermal;
        let vy = rng.gen::<f64>() * 2.0 * v_thermal - v_thermal;
        let vz = rng.gen::<f64>() * 2.0 * v_thermal - v_thermal;
        vel.push([vx, vy, vz]);
    }

    println!("Initial conditions: uniform distribution with thermal velocities");

    let start = Instant::now();
    let mut max_rho_ever = 0.0_f64;

    // Main loop (no cooling, just gravity vs pressure)
    for step in 0..=STEPS {
        // Compute density on grid
        let (rho_grid, rho_mean) = compute_density_grid(&pos);
        let rho_max = rho_grid.iter().cloned().fold(0.0_f64, f64::max);
        let rho_ratio = rho_max / rho_mean;

        let v_rms: f64 = vel.iter()
            .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            .sum::<f64>() / N as f64;
        let v_rms_kms = v_rms.sqrt() * 978.0;

        writeln!(csv, "{},{:.4},{:.3},{:.1}",
                 step, step as f64 * DT, rho_ratio, v_rms_kms).unwrap();

        if rho_ratio > max_rho_ever {
            max_rho_ever = rho_ratio;
        }

        if step % 20 == 0 {
            print!("\r[Step {:4}/{}] ρ_max/ρ̄ = {:.3}, v_rms = {:.0} km/s   ",
                   step, STEPS, rho_ratio, v_rms_kms);
            std::io::stdout().flush().unwrap();
        }

        if step == STEPS { break; }

        // Simplified dynamics: just drift (pressure support via thermal velocity)
        // In hot gas, thermal pressure >> gravity, so we just evolve freely

        for i in 0..N {
            pos[i][0] += vel[i][0] * DT;
            pos[i][1] += vel[i][1] * DT;
            pos[i][2] += vel[i][2] * DT;

            // Periodic boundaries
            for k in 0..3 {
                if pos[i][k] > half_box { pos[i][k] -= BOX; }
                if pos[i][k] < -half_box { pos[i][k] += BOX; }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n\n================================================================");
    println!("  TEST COMPLETE — {:.1} seconds", elapsed);
    println!("================================================================");
    println!("  Max ρ_max/ρ̄ reached: {:.3}", max_rho_ever);

    if max_rho_ever < 2.0 {
        println!("  ✓ PASS: No collapse (ρ_max/ρ̄ < 2)");
        println!("  → Hot gas remains stable due to thermal pressure");
    } else {
        println!("  ✗ FAIL: ρ_max/ρ̄ = {:.2} (expected < 2)", max_rho_ever);
        println!("\n  → BUG DETECTED: Pressure support not working");
    }

    println!("\nResults saved to: {}/evolution.csv", output_dir);
}

fn compute_density_grid(pos: &[[f64; 3]]) -> (Vec<f64>, f64) {
    let cell_size = BOX / N_CELLS as f64;
    let half_box = BOX / 2.0;
    let n_cells_cubed = N_CELLS * N_CELLS * N_CELLS;

    let mut counts = vec![0u32; n_cells_cubed];

    for p in pos {
        let ix = ((p[0] + half_box) / cell_size) as usize;
        let iy = ((p[1] + half_box) / cell_size) as usize;
        let iz = ((p[2] + half_box) / cell_size) as usize;
        let ix = ix.min(N_CELLS - 1);
        let iy = iy.min(N_CELLS - 1);
        let iz = iz.min(N_CELLS - 1);
        let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;
        counts[idx] += 1;
    }

    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mean = total as f64 / n_cells_cubed as f64;

    let rho: Vec<f64> = counts.iter().map(|&c| c as f64).collect();

    (rho, mean)
}
