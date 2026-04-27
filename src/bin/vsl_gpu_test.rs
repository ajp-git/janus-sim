//! VSL GPU Test — Tests BVH GPU tree with VSL physics
//! No SPH to isolate force calculation performance
//!
//! Usage:
//!   cargo run --release --features cuda --bin vsl_gpu_test -- \
//!     --n 2000000 --box 100 --c-ratio 10 --steps 100

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

const N_CELLS: usize = 32;

#[cfg(feature = "cuda")]
fn main() {
    run_vsl_gpu_test();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
}

#[cfg(feature = "cuda")]
fn run_vsl_gpu_test() {
    let args: Vec<String> = std::env::args().collect();

    let n_particles: usize = args.iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);

    let box_size: f64 = args.iter()
        .position(|a| a == "--box")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100.0);

    let c_ratio: f64 = args.iter()
        .position(|a| a == "--c-ratio")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let n_steps: usize = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let dt: f64 = args.iter()
        .position(|a| a == "--dt")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);

    // Janus mass fractions
    let eta = 1.045;
    let f_plus = eta / (1.0 + eta);
    let n_positive = (n_particles as f64 * f_plus).round() as usize;
    let n_negative = n_particles.saturating_sub(n_positive);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           VSL GPU TEST — BVH Tree with VSL                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", n_particles, n_positive, n_negative);
    println!("║  Box = {} Mpc", box_size);
    println!("║  c_ratio = {} → (c⁻/c⁺)² = {}", c_ratio, c_ratio * c_ratio);
    println!("║  dt = {}, steps = {}", dt, n_steps);
    println!("║  NO SPH — pure N-body with VSL                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Initialize GPU simulation
    println!("Initializing GPU...");
    let gpu_start = Instant::now();

    let mut gpu_sim = match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        }
    };

    // Set VSL c_ratio
    gpu_sim.set_c_ratio(c_ratio);

    // Set theta for Barnes-Hut (higher = faster, less accurate)
    gpu_sim.set_theta(1.0);  // Fast mode for testing

    println!("GPU init: {:.2}s, c_ratio_sq = {}, theta = {:.1}",
             gpu_start.elapsed().as_secs_f64(),
             gpu_sim.get_c_ratio_sq(),
             gpu_sim.get_theta());

    let half_box = box_size / 2.0;
    let cell_size = box_size / N_CELLS as f64;

    println!("\n{:>6} | {:>10} | {:>10} | {:>8} | {:>10} | {:>8}",
             "Step", "ρ+_max", "ρ-_max", "δ_max", "step_time", "total");
    println!("{:-<70}", "");

    let sim_start = Instant::now();
    let mut last_step_time = Instant::now();

    for step in 0..=n_steps {
        // Get positions for analysis
        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to get positions: {}", e);
                break;
            }
        };

        let signs = gpu_sim.signs();

        // Compute density in cells
        let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut counts_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];

        for i in 0..n_particles {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];

            let ix = ((px + half_box) / cell_size).floor() as usize % N_CELLS;
            let iy = ((py + half_box) / cell_size).floor() as usize % N_CELLS;
            let iz = ((pz + half_box) / cell_size).floor() as usize % N_CELLS;
            let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;

            if signs[i] > 0 {
                counts_plus[idx] += 1;
            } else {
                counts_minus[idx] += 1;
            }
        }

        let rho_plus_max = *counts_plus.iter().max().unwrap_or(&0) as f64;
        let rho_minus_max = *counts_minus.iter().max().unwrap_or(&0) as f64;
        let mean_count = n_particles as f64 / (N_CELLS * N_CELLS * N_CELLS) as f64;
        let delta_max = (rho_plus_max / mean_count - 1.0).max(rho_minus_max / mean_count - 1.0);

        let step_time = last_step_time.elapsed().as_secs_f64();
        last_step_time = Instant::now();

        // Progress output every 10 steps
        if step % 10 == 0 || step == n_steps {
            let elapsed = sim_start.elapsed().as_secs_f64();
            println!("{:>6} | {:>10.2} | {:>10.2} | {:>8.2} | {:>9.2}s | {:>7.1}s",
                     step, rho_plus_max, rho_minus_max, delta_max, step_time, elapsed);
        }

        // ALERTS
        if rho_minus_max > 1000.0 {
            println!("🚨 ρ-_max = {:.0} > 1000 — m- COLLAPSE at step {}", rho_minus_max, step);
        }
        if rho_plus_max > 100.0 {
            println!("🚨 ρ+_max = {:.0} > 100 — m+ STRUCTURES at step {}", rho_plus_max, step);
        }

        if step >= n_steps {
            break;
        }

        // Step using GPU tree (DKD integrator with BVH)
        // a=1.0, H=0, dtau_per_dt=0 (no cosmological expansion)
        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0) {
            eprintln!("GPU step failed: {}", e);
            break;
        }
    }

    let total_time = sim_start.elapsed().as_secs_f64();
    let avg_step_time = total_time / n_steps as f64;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE: {:.1}s for {} steps ({:.2}s/step)", total_time, n_steps, avg_step_time);
    println!("║  ETA for 2000 steps: {:.1} min", avg_step_time * 2000.0 / 60.0);
    println!("╚══════════════════════════════════════════════════════════════╝");
}
