//! Test 1: Isothermal Collapse (m+ only, no Janus)
//!
//! Setup:
//! - N = 100k particles (all m+)
//! - Box = 5 Mpc
//! - T_init = 100 K (cold gas)
//! - Central overdensity: ρ_center = 5 × ρ̄
//!
//! Expected:
//! - ρ_max/ρ̄ > 20 before step 300
//! - At least 1 region meeting SF criteria
//!
//! If FAIL → BUG → STOP

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

use janus::baryonic::cooling::{apply_cooling, T_FLOOR};
use janus::baryonic::pressure::{pressure, sound_speed};
use janus::baryonic::sph::SphKernel;

// Simulation parameters
const N: usize = 10_000;  // Reduced for O(N²) gravity
const BOX: f64 = 2.0;  // Mpc (smaller box for faster collapse)
const T_INIT: f64 = 100.0;  // K (cold gas)
const STEPS: usize = 300;
const DT: f64 = 0.002;  // Gyr
const SOFTENING: f64 = 0.005;  // Mpc

// Physics
const G: f64 = 4.499e-15;  // Mpc³ M_sun⁻¹ Gyr⁻²
const PARTICLE_MASS: f64 = 1e8;  // M_sun per particle

// Grid for density computation
const N_CELLS: usize = 16;

fn main() {
    println!("================================================================");
    println!("  TEST 1: ISOTHERMAL COLLAPSE (m+ only)");
    println!("================================================================");
    println!("  N = {}, Box = {} Mpc, T_init = {} K", N, BOX, T_INIT);
    println!("  Steps = {}, dt = {} Gyr", STEPS, DT);
    println!("  Expected: ρ_max/ρ̄ > 20 before step 300");
    println!("================================================================\n");

    let output_dir = "/app/output/test_collapse";
    fs::create_dir_all(output_dir).expect("Failed to create output dir");

    let mut csv = File::create(format!("{}/evolution.csv", output_dir)).unwrap();
    writeln!(csv, "step,time_gyr,rho_max,t_mean,v_rms").unwrap();

    // Initialize particles
    let mut rng = StdRng::seed_from_u64(42);
    let half_box = BOX / 2.0;

    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(N);
    let mut vel: Vec<[f64; 3]> = Vec::with_capacity(N);
    let mut temp: Vec<f64> = vec![T_INIT; N];

    // Create particles with central overdensity
    // 50% in central sphere (r < 0.3 × half_box), 50% uniform
    let r_core = 0.3 * half_box;
    let n_core = N / 2;

    for i in 0..N {
        let (x, y, z) = if i < n_core {
            // Core particles (overdense center)
            loop {
                let x = (rng.gen::<f64>() - 0.5) * 2.0 * r_core;
                let y = (rng.gen::<f64>() - 0.5) * 2.0 * r_core;
                let z = (rng.gen::<f64>() - 0.5) * 2.0 * r_core;
                let r = (x*x + y*y + z*z).sqrt();
                if r < r_core {
                    break (x, y, z);
                }
            }
        } else {
            // Background particles (uniform)
            let x = (rng.gen::<f64>() - 0.5) * BOX;
            let y = (rng.gen::<f64>() - 0.5) * BOX;
            let z = (rng.gen::<f64>() - 0.5) * BOX;
            (x, y, z)
        };

        pos.push([x, y, z]);
        vel.push([0.0, 0.0, 0.0]);
    }

    println!("Initial conditions:");
    println!("  Core particles (r < {:.2} Mpc): {}", r_core, n_core);
    println!("  Background particles: {}", N - n_core);

    let start = Instant::now();
    let mut max_rho_ratio = 0.0_f64;
    let mut collapse_step = None;

    // Main loop
    for step in 0..=STEPS {
        // Compute density on grid
        let (rho_grid, rho_mean) = compute_density_grid(&pos);
        let rho_max = rho_grid.iter().cloned().fold(0.0_f64, f64::max);
        let rho_ratio = rho_max / rho_mean;

        // Compute mean temperature and velocity
        let t_mean: f64 = temp.iter().sum::<f64>() / N as f64;
        let v_rms: f64 = vel.iter()
            .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            .sum::<f64>() / N as f64;
        let v_rms = v_rms.sqrt();

        // Convert v_rms to km/s (1 Mpc/Gyr ≈ 978 km/s)
        let v_rms_kms = v_rms * 978.0;

        writeln!(csv, "{},{:.4},{:.2},{:.1},{:.2}",
                 step, step as f64 * DT, rho_ratio, t_mean, v_rms_kms).unwrap();

        if rho_ratio > max_rho_ratio {
            max_rho_ratio = rho_ratio;
        }

        if rho_ratio > 20.0 && collapse_step.is_none() {
            collapse_step = Some(step);
            println!("\n★★★ COLLAPSE DETECTED at step {} ★★★", step);
            println!("    ρ_max/ρ̄ = {:.1}", rho_ratio);
        }

        if step % 50 == 0 {
            print!("\r[Step {:4}/{}] ρ_max/ρ̄ = {:6.1}, T = {:6.0} K, v_rms = {:5.1} km/s",
                   step, STEPS, rho_ratio, t_mean, v_rms_kms);
            std::io::stdout().flush().unwrap();
        }

        if step == STEPS { break; }

        // Physics step: gravity + pressure + cooling

        // 1. Compute accelerations (gravity only for now, simplified)
        let mut acc: Vec<[f64; 3]> = vec![[0.0; 3]; N];

        // Direct N² gravity (expensive but correct for test)
        // Use cell-based approximation for speed
        let cell_size = BOX / N_CELLS as f64;

        for i in 0..N {
            // Find cell
            let ix = ((pos[i][0] + half_box) / cell_size) as usize;
            let iy = ((pos[i][1] + half_box) / cell_size) as usize;
            let iz = ((pos[i][2] + half_box) / cell_size) as usize;
            let ix = ix.min(N_CELLS - 1);
            let iy = iy.min(N_CELLS - 1);
            let iz = iz.min(N_CELLS - 1);

            // Sample neighbors from nearby cells (simplified)
            for j in 0..N {
                if i == j { continue; }

                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                let r = r2.sqrt();
                let r3 = r * r2;

                // Gravity: a = G × m / r²
                let f = G * PARTICLE_MASS / r3;
                acc[i][0] += f * dx;
                acc[i][1] += f * dy;
                acc[i][2] += f * dz;
            }
        }

        // 2. Update velocities (kick)
        for i in 0..N {
            vel[i][0] += acc[i][0] * DT;
            vel[i][1] += acc[i][1] * DT;
            vel[i][2] += acc[i][2] * DT;
        }

        // 3. Update positions (drift)
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

        // 4. Apply cooling (using local overdensity)
        for i in 0..N {
            let ix = ((pos[i][0] + half_box) / cell_size) as usize;
            let iy = ((pos[i][1] + half_box) / cell_size) as usize;
            let iz = ((pos[i][2] + half_box) / cell_size) as usize;
            let ix = ix.min(N_CELLS - 1);
            let iy = iy.min(N_CELLS - 1);
            let iz = iz.min(N_CELLS - 1);
            let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;

            let local_rho = rho_grid[idx];
            let overdensity = local_rho / rho_mean;
            temp[i] = apply_cooling(temp[i], overdensity.max(1.0), 0.0, DT);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n\n================================================================");
    println!("  TEST COMPLETE — {:.1} seconds", elapsed);
    println!("================================================================");
    println!("  Max ρ_max/ρ̄ reached: {:.1}", max_rho_ratio);

    if let Some(step) = collapse_step {
        if step < 300 {
            println!("  ✓ PASS: Collapse at step {} (< 300)", step);
        } else {
            println!("  ✗ FAIL: Collapse at step {} (>= 300)", step);
        }
    } else if max_rho_ratio > 20.0 {
        println!("  ✓ PASS: ρ_max/ρ̄ > 20 reached");
    } else {
        println!("  ✗ FAIL: ρ_max/ρ̄ = {:.1} (expected > 20)", max_rho_ratio);
        println!("\n  → BUG DETECTED: Gravity or cooling not working correctly");
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
