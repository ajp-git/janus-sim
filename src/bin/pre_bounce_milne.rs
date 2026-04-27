//! Pre-Bounce Milne Expansion Simulation
//!
//! Simulates the Janus universe before the bounce (z=100 → z=4.5)
//! using Milne expansion: a(t) = t/t_bounce, H(t) = 1/t
//!
//! No baryonic physics - this is the pre-baryon era.
//! Pure gravity with Janus interaction rules.

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::janus_expansion::{MilneExpansion, T_BOUNCE_GYR, Z_BOUNCE};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const N_CELLS: usize = 64;

#[cfg(feature = "cuda")]
fn main() {
    run_pre_bounce();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires CUDA. Compile with --features cuda");
}

#[cfg(feature = "cuda")]
fn run_pre_bounce() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
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

    let z_init: f64 = args.iter()
        .position(|a| a == "--z-init")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100.0);

    let delta_init: f64 = args.iter()
        .position(|a| a == "--delta-init")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5);

    let dt: f64 = args.iter()
        .position(|a| a == "--dt")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);

    let mu: f64 = args.iter()
        .position(|a| a == "--mu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(19.0);

    let snapshot_every: usize = args.iter()
        .position(|a| a == "--snapshot-every")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let output_dir: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "/app/output/pre_bounce".to_string());

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", output_dir)).ok();

    // Initialize Milne expansion
    let milne = MilneExpansion::new(z_init, 10000);

    // Calculate number of steps
    let t_start = milne.t_start;
    let t_end = milne.t_end;
    let n_steps = ((t_end - t_start) / dt).ceil() as usize;

    println!("\n================================================================");
    println!("  PRE-BOUNCE MILNE SIMULATION");
    println!("================================================================");
    println!("  N = {}", n_particles);
    println!("  Box = {} Mpc (comoving)", box_size);
    println!("  z_init = {} → z_bounce = {}", z_init, Z_BOUNCE);
    println!("  t: {:.4} → {:.4} Gyr", t_start, t_end);
    println!("  δ_init = {:.2e} (primordial fluctuations)", delta_init);
    println!("  μ = {} (mass ratio)", mu);
    println!("  dt = {} Gyr, steps = {}", dt, n_steps);
    println!("  Output: {}", output_dir);
    println!("================================================================\n");

    // Janus mass fractions: f+ = η/(1+η), f- = 1/(1+η)
    let eta = 1.045;
    let f_plus = eta / (1.0 + eta);  // ~0.511 for η=1.045
    let n_positive = (n_particles as f64 * f_plus).round() as usize;
    let n_negative = n_particles.saturating_sub(n_positive);  // Safe subtraction

    println!("Particle distribution: {} m+ / {} m-", n_positive, n_negative);

    // Initialize positions with tiny primordial perturbations
    let half_box = box_size / 2.0;
    let mut rng = StdRng::seed_from_u64(42);

    // Primordial perturbation wavelength (box/8 for multiple modes)
    let k_pert = 2.0 * std::f64::consts::PI * 8.0 / box_size;
    let delta_amp = delta_init;  // Already a fraction, not percentage

    println!("Initializing with primordial perturbations:");
    println!("  δ = {:.2e}", delta_amp);
    println!("  k = {:.4} Mpc⁻¹ (λ = {:.1} Mpc)", k_pert, 2.0 * std::f64::consts::PI / k_pert);

    let mut positions = Vec::with_capacity(n_particles * 3);
    let mut velocities = Vec::with_capacity(n_particles * 3);
    let mut signs = Vec::with_capacity(n_particles);

    // Initial Hubble parameter
    let h_init = milne.at_time(t_start).h;

    for i in 0..n_particles {
        use rand::Rng;

        // Uniform position
        let mut x = rng.gen::<f64>() * box_size - half_box;
        let mut y = rng.gen::<f64>() * box_size - half_box;
        let mut z = rng.gen::<f64>() * box_size - half_box;

        // Apply tiny sinusoidal perturbation (Zel'dovich-like)
        if delta_amp > 0.0 {
            let disp_scale = delta_amp * box_size / (4.0 * k_pert);
            x += disp_scale * (k_pert * x).sin();
            y += disp_scale * (k_pert * y).sin();
            z += disp_scale * (k_pert * z).sin();

            // Re-wrap to box
            if x > half_box { x -= box_size; }
            if x < -half_box { x += box_size; }
            if y > half_box { y -= box_size; }
            if y < -half_box { y += box_size; }
            if z > half_box { z -= box_size; }
            if z < -half_box { z += box_size; }
        }

        positions.push(x);
        positions.push(y);
        positions.push(z);

        // Hubble flow velocity: v = H × r
        velocities.push(h_init * x);
        velocities.push(h_init * y);
        velocities.push(h_init * z);

        // Random sign assignment
        let sign = if rng.gen::<bool>() { 1 } else { -1 };
        signs.push(sign);
    }

    // Initialize GPU simulation
    println!("\nInitializing GPU...");
    let gpu_start = Instant::now();

    let mut gpu_sim = match GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        box_size,
        positions.clone(),
        velocities.clone(),
        signs.clone(),
    ) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        }
    };

    println!("GPU init: {:.2}s", gpu_start.elapsed().as_secs_f64());

    // Open CSV file
    let csv_file = File::create(format!("{}/evolution.csv", output_dir)).unwrap();
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "step,t_gyr,z_today,a,H_gyr,rho_plus_max,rho_minus_max,delta_max,segregation").unwrap();

    // Cell size for density computation
    let cell_size = box_size / N_CELLS as f64;
    let cell_vol = cell_size.powi(3);
    let mean_count = n_particles as f64 / (N_CELLS * N_CELLS * N_CELLS) as f64;

    println!("\nStarting Milne expansion simulation...\n");
    let sim_start = Instant::now();

    let mut current_time = t_start;

    for step in 0..=n_steps {
        // Get expansion state
        let state = milne.at_time(current_time);

        // Get positions for analysis
        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to get positions: {}", e);
                break;
            }
        };

        // Compute density contrast
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
        let delta_max = (rho_plus_max / mean_count - 1.0).max(rho_minus_max / mean_count - 1.0);

        // Segregation
        let mut seg_num = 0.0f64;
        let mut seg_den = 0.0f64;
        for idx in 0..(N_CELLS * N_CELLS * N_CELLS) {
            let np = counts_plus[idx] as f64;
            let nm = counts_minus[idx] as f64;
            seg_num += (np - nm).abs();
            seg_den += np + nm;
        }
        let segregation = if seg_den > 0.0 { seg_num / seg_den } else { 0.0 };

        // Write CSV
        writeln!(csv, "{},{:.6},{:.4},{:.8},{:.6},{:.2},{:.2},{:.6},{:.4}",
                 step, current_time, state.z_today, state.a, state.h,
                 rho_plus_max, rho_minus_max, delta_max, segregation).unwrap();

        // Progress output
        if step % 100 == 0 || step == n_steps {
            let elapsed = sim_start.elapsed().as_secs_f64();
            let eta_sec = if step > 0 {
                (elapsed / step as f64) * (n_steps - step) as f64
            } else {
                0.0
            };

            println!("[Step {:5}] t={:.4}Gyr z={:.1} a={:.6} δ_max={:.2e} S={:.3} | ETA {:.0}s",
                     step, current_time, state.z_today, state.a, delta_max, segregation, eta_sec);

            // Alert if perturbations grow significantly
            if delta_max > 0.01 && delta_init < 0.001 {
                println!("  ⚠️  δ_max > 1% — perturbations growing!");
            }
        }

        // Save snapshot
        if step % snapshot_every == 0 {
            save_snapshot(&output_dir, step, &pos, &signs, state.z_today, current_time);
        }

        // Break if we've reached the bounce
        if current_time >= t_end - dt/2.0 {
            println!("\n✓ Reached bounce at t = {:.4} Gyr, z = {:.2}", current_time, state.z_today);
            break;
        }

        // Evolve with Hubble friction (comoving coordinates)
        // In Milne: H = 1/t
        let h = state.h;
        let dtau_per_dt = 1.0 / state.a;  // Conformal time factor

        gpu_sim.step_with_expansion_dkd(dt, state.a, h, dtau_per_dt);

        current_time += dt;
    }

    csv.flush().unwrap();

    // Final stats
    let total_time = sim_start.elapsed().as_secs_f64();
    println!("\n================================================================");
    println!("  SIMULATION COMPLETE");
    println!("================================================================");
    println!("  Total time: {:.1} seconds", total_time);
    println!("  Final z: {:.2}", milne.at_time(current_time).z_today);
    println!("  Output: {}", output_dir);
    println!("================================================================");
}

fn save_snapshot(output_dir: &str, step: usize, pos: &[f64], signs: &[i32], z: f64, t_gyr: f64) {
    let path = format!("{}/snapshots/snap_{:05}.bin", output_dir, step);
    let mut file = match File::create(&path) {
        Ok(f) => BufWriter::new(f),
        Err(e) => {
            eprintln!("Failed to create snapshot: {}", e);
            return;
        }
    };

    let n = (pos.len() / 3) as u32;

    // Version marker for pre-bounce format
    let version: i32 = -3;  // Different from baryonic format
    file.write_all(&version.to_le_bytes()).unwrap();
    file.write_all(&n.to_le_bytes()).unwrap();
    file.write_all(&z.to_le_bytes()).unwrap();
    file.write_all(&t_gyr.to_le_bytes()).unwrap();

    // Write positions and signs (no velocities for this format)
    for i in 0..n as usize {
        file.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        file.write_all(&(signs[i] as f64).to_le_bytes()).unwrap();
    }
}
