//! μ=3 Pure Gravity Simulation
//! N+ = 25%, N- = 75% — quasi-symmetric densities
//! No cosmological expansion — pure gravitational dynamics

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// Simulation parameters
const N: usize = 20_000_000;
const MU: f64 = 3.0;  // N-/N+ ratio
const BOX_SIZE: f64 = 500.0;  // Mpc
const DT: f64 = 0.005;
const STEPS: usize = 4000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;
const DELTA_INIT: f64 = 0.10;  // 10% perturbation

// Physics
const G_COSMO: f64 = 4.499e-15;
const RHO_CRIT: f64 = 1.36e11;
const OMEGA_B: f64 = 0.05;

// Output
const SNAPSHOT_INTERVAL: usize = 40;
const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 20.0;
const N_CELLS: usize = 32;

#[cfg(feature = "cuda")]
fn main() {
    println!("================================================================");
    println!("  μ=3 PURE GRAVITY SIMULATION");
    println!("================================================================");
    println!("  N = {} ({:.1}M)", N, N as f64 / 1e6);
    println!("  μ = {} (N+ = 25%, N- = 75%)", MU);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  δ_init = {}%", DELTA_INIT * 100.0);
    println!("  Steps = {}, dt = {}", STEPS, DT);
    println!("  Mode: PURE GRAVITY (no expansion)");
    println!("================================================================\n");

    // Calculate particle counts
    let n_plus = (N as f64 / (1.0 + MU)) as usize;
    let n_minus = N - n_plus;

    println!("Particle distribution:");
    println!("  N+ = {} ({:.1}%)", n_plus, 100.0 * n_plus as f64 / N as f64);
    println!("  N- = {} ({:.1}%)", n_minus, 100.0 * n_minus as f64 / N as f64);
    println!("  Actual μ = {:.3}", n_minus as f64 / n_plus as f64);

    // Mass factor
    let rho_plus = OMEGA_B * RHO_CRIT;
    let rho_total = rho_plus * (1.0 + MU);
    let m_total = rho_total * BOX_SIZE.powi(3);
    let mass_factor = G_COSMO * m_total / N as f64;

    println!("\nPhysics:");
    println!("  Ω_tot = {:.2} (sub-critical)", OMEGA_B * (1.0 + MU));
    println!("  mass_factor = {:.4e}", mass_factor);

    // Generate ICs with perturbed grid
    println!("\nGenerating perturbed grid ICs (δ={}%)...", DELTA_INIT * 100.0);

    let n_per_dim = (N as f64).powf(1.0 / 3.0).ceil() as usize;
    let cell_size = BOX_SIZE / n_per_dim as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, DELTA_INIT * cell_size).unwrap();

    let mut positions = Vec::with_capacity(N * 3);
    let mut velocities = Vec::with_capacity(N * 3);
    let mut signs = Vec::with_capacity(N);

    let mut count = 0;
    'outer: for ix in 0..n_per_dim {
        for iy in 0..n_per_dim {
            for iz in 0..n_per_dim {
                if count >= N { break 'outer; }

                // Grid position + perturbation
                let x = -half_box + (ix as f64 + 0.5) * cell_size + normal.sample(&mut rng);
                let y = -half_box + (iy as f64 + 0.5) * cell_size + normal.sample(&mut rng);
                let z = -half_box + (iz as f64 + 0.5) * cell_size + normal.sample(&mut rng);

                // Wrap periodic
                let x = ((x + half_box) % BOX_SIZE) - half_box;
                let y = ((y + half_box) % BOX_SIZE) - half_box;
                let z = ((z + half_box) % BOX_SIZE) - half_box;

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);

                // Zero initial velocity
                velocities.push(0.0f32);
                velocities.push(0.0f32);
                velocities.push(0.0f32);

                // Random sign with probability for μ=3
                // P(+) = 1/(1+μ) = 0.25, P(-) = μ/(1+μ) = 0.75
                let sign: i8 = if rng.gen::<f64>() < 1.0 / (1.0 + MU) { 1 } else { -1 };
                signs.push(sign);

                count += 1;
            }
        }
    }

    let actual_n_plus: usize = signs.iter().filter(|&&s| s > 0).count();
    let actual_n_minus = N - actual_n_plus;
    println!("  Generated {} particles", count);
    println!("  Actual N+ = {}, N- = {}", actual_n_plus, actual_n_minus);
    println!("  Actual μ = {:.3}", actual_n_minus as f64 / actual_n_plus as f64);

    // Setup output
    let output_dir = "/app/output/mu3_pure_gravity";
    let snap_dir = format!("{}/snapshots", output_dir);
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    // CSV file
    let mut csv_file = BufWriter::new(
        File::create(format!("{}/time_series.csv", output_dir)).unwrap()
    );
    writeln!(csv_file, "step,t_gyr,diff_pois,corr_delta,rho_plus_max_ratio,v_plus_kms,v_minus_kms").unwrap();

    // Initialize GPU simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        positions, velocities, signs, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);  // No screening
    sim.set_mass_factor(mass_factor);

    let start = Instant::now();
    println!("\nStarting evolution (PURE GRAVITY, no expansion)...\n");

    let mut t_gyr = 0.0;

    for step in 0..=STEPS {
        if step > 0 {
            // Pure gravity step - h=0, a=1 (no expansion)
            sim.set_current_z(0.0);
            sim.step_treepm_gpu(DT, R_CUT, 0.0, 1.0)
                .expect("TreePM step failed");
            t_gyr += DT;
        }

        // CSV logging
        if step % CSV_INTERVAL == 0 {
            let (pos, vel, sgn) = sim.get_particles().unwrap();
            let (diff_pois, corr_delta, rho_max, v_plus, v_minus) =
                compute_metrics(&pos, &vel, &sgn, BOX_SIZE, N_CELLS);

            writeln!(csv_file, "{},{:.6},{:.4},{:.4},{:.2},{:.1},{:.1}",
                     step, t_gyr, diff_pois, corr_delta, rho_max, v_plus, v_minus).unwrap();

            // Progress with structure alert
            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta = if rate > 0.0 { (STEPS - step) as f64 / rate / 60.0 } else { 0.0 };

            let alert = if rho_max > 50.0 {
                ">>> COLLAPSE <<<"
            } else if rho_max > 10.0 {
                "*** STRUCTURES ***"
            } else if rho_max > 5.0 {
                "* clustering *"
            } else {
                ""
            };

            println!("  step {:4} | t={:.2} Gyr | ρ+max/ρ̄={:.1} | Corr={:.3} | v+={:.0} km/s | ETA {:.0}min {}",
                     step, t_gyr, rho_max, corr_delta, v_plus, eta, alert);
        }

        // Snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, t_gyr, BOX_SIZE);
        }
    }

    csv_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let (pos, vel, sgn) = sim.get_particles().unwrap();
    let (diff_pois, corr_delta, rho_max, v_plus, v_minus) =
        compute_metrics(&pos, &vel, &sgn, BOX_SIZE, N_CELLS);

    println!("\n================================================================");
    println!("  μ=3 PURE GRAVITY COMPLETE");
    println!("================================================================");
    println!("  Final t: {:.2} Gyr", t_gyr);
    println!("  ρ+_max/ρ̄+: {:.2} {}", rho_max,
             if rho_max > 10.0 { "✓ STRUCTURES" } else { "✗ no structures" });
    println!("  Diff/Pois: {:.4}", diff_pois);
    println!("  Corr(δ+,δ-): {:.4}", corr_delta);
    println!("  <v+> = {:.0} km/s, <v-> = {:.0} km/s", v_plus, v_minus);
    println!("  Runtime: {:.1}s ({:.1} min)", elapsed, elapsed / 60.0);
    println!("  Rate: {:.1} steps/s", STEPS as f64 / elapsed);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn compute_metrics(positions: &[f32], velocities: &[f32], signs: &[i8],
                   box_size: f64, n_cells: usize) -> (f64, f64, f64, f64, f64) {
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n_cells_cubed = n_cells * n_cells * n_cells;
    let n = signs.len();

    let mut n_plus_grid = vec![0u32; n_cells_cubed];
    let mut n_minus_grid = vec![0u32; n_cells_cubed];
    let mut v_plus_sum = 0.0f64;
    let mut v_minus_sum = 0.0f64;
    let mut n_plus_count = 0usize;
    let mut n_minus_count = 0usize;

    for i in 0..n {
        let x = ((positions[i*3] as f64 + half_box) % box_size) / cell_size;
        let y = ((positions[i*3+1] as f64 + half_box) % box_size) / cell_size;
        let z = ((positions[i*3+2] as f64 + half_box) % box_size) / cell_size;

        let ix = (x as usize).min(n_cells - 1);
        let iy = (y as usize).min(n_cells - 1);
        let iz = (z as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        let vmag = (velocities[i*3].powi(2) + velocities[i*3+1].powi(2) +
                   velocities[i*3+2].powi(2)).sqrt() as f64 * 977.8;

        if signs[i] > 0 {
            n_plus_grid[idx] += 1;
            v_plus_sum += vmag;
            n_plus_count += 1;
        } else {
            n_minus_grid[idx] += 1;
            v_minus_sum += vmag;
            n_minus_count += 1;
        }
    }

    let v_plus = if n_plus_count > 0 { v_plus_sum / n_plus_count as f64 } else { 0.0 };
    let v_minus = if n_minus_count > 0 { v_minus_sum / n_minus_count as f64 } else { 0.0 };

    // Statistics
    let total_plus: u64 = n_plus_grid.iter().map(|&x| x as u64).sum();
    let total_minus: u64 = n_minus_grid.iter().map(|&x| x as u64).sum();
    let mean_plus = total_plus as f64 / n_cells_cubed as f64;
    let mean_minus = total_minus as f64 / n_cells_cubed as f64;

    // Diff/Poisson
    let diff: Vec<f64> = n_plus_grid.iter().zip(n_minus_grid.iter())
        .map(|(&p, &m)| p as f64 - m as f64).collect();
    let diff_mean: f64 = diff.iter().sum::<f64>() / n_cells_cubed as f64;
    let diff_var: f64 = diff.iter().map(|d| (d - diff_mean).powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let poisson_var = mean_plus + mean_minus;
    let diff_pois = if poisson_var > 0.0 { diff_var / poisson_var } else { 1.0 };

    // Correlation
    let delta_plus: Vec<f64> = n_plus_grid.iter()
        .map(|&x| if mean_plus > 0.0 { (x as f64 - mean_plus) / mean_plus } else { 0.0 }).collect();
    let delta_minus: Vec<f64> = n_minus_grid.iter()
        .map(|&x| if mean_minus > 0.0 { (x as f64 - mean_minus) / mean_minus } else { 0.0 }).collect();

    let cov: f64 = delta_plus.iter().zip(delta_minus.iter())
        .map(|(dp, dm)| dp * dm).sum::<f64>() / n_cells_cubed as f64;
    let var_plus: f64 = delta_plus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let var_minus: f64 = delta_minus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let corr_delta = if var_plus > 0.0 && var_minus > 0.0 {
        cov / (var_plus.sqrt() * var_minus.sqrt())
    } else { 0.0 };

    // ρ+_max/ρ̄+
    let max_plus = *n_plus_grid.iter().max().unwrap_or(&0) as f64;
    let rho_max = if mean_plus > 0.0 { max_plus / mean_plus } else { 1.0 };

    (diff_pois, corr_delta, rho_max, v_plus, v_minus)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, dir: &str, step: usize, t_gyr: f64, box_size: f64) {
    let (positions, velocities, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let path = format!("{}/snap_{:05}.bin", dir, step);

    let file = match File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(box_size as f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(t_gyr as f32).to_le_bytes());

    for i in 0..n {
        let _ = writer.write_all(&positions[i*3].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+1].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+2].to_le_bytes());
        let _ = writer.write_all(&velocities[i*3].to_le_bytes());
        let _ = writer.write_all(&velocities[i*3+1].to_le_bytes());
        let _ = writer.write_all(&velocities[i*3+2].to_le_bytes());
        let _ = writer.write_all(&(signs[i] as i8).to_le_bytes());
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda cufft");
    std::process::exit(1);
}
