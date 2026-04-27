//! Scan initial condition amplitude for Janus structure formation
//!
//! Usage: cargo run --release --features "cuda cufft" --bin scan_ci_amplitude -- --delta 1.0
//!
//! δ_init controls perturbation amplitude as fraction of cell size

use rand::prelude::*;
use rand_distr::StandardNormal;
use std::env;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::janus_expansion::JanusExpansion;

const MU: f64 = 19.0;
const DEFAULT_N: usize = 5_000_000;
const BOX_SIZE: f64 = 500.0;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.005;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 50;
const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 20.0;
const N_CELLS: usize = 32;

// Physical constants
const G_COSMO: f64 = 4.499e-15;
const RHO_CRIT: f64 = 1.36e11;
const OMEGA_B: f64 = 0.05;

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    let delta_init: f64 = args.iter()
        .position(|x| x == "--delta")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .expect("Usage: --delta <percent> [--n <particles>]");

    let n_total: usize = args.iter()
        .position(|x| x == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);

    let delta_frac = delta_init / 100.0;  // Convert percent to fraction

    let run_name = format!("ci_scan_delta{:.1}", delta_init);

    println!("================================================================");
    println!("  SCAN CI AMPLITUDE: δ_init = {}%", delta_init);
    println!("================================================================");
    println!("  μ = {}, N = {}M, Box = {} Mpc", MU, n_total / 1_000_000, BOX_SIZE);
    println!("  Perturbation = {:.4} × cell_size", delta_frac);
    println!("  z_init = {}, Steps = {}", Z_INIT, STEPS);
    println!("================================================================");

    // Initialize Janus expansion
    println!("\nInitializing Janus cosmology...");
    let expansion = JanusExpansion::new(Z_INIT, 5000);

    // Setup output
    let base_dir = std::path::Path::new("/app/output").join(&run_name);
    fs::create_dir_all(&base_dir).expect("Failed to create output dir");

    // Particle counts based on μ
    let n_positive = (n_total as f64 / (1.0 + MU)) as usize;
    let n_negative = n_total - n_positive;

    println!("\n  N+ = {} ({:.2}%)", n_positive, 100.0 * n_positive as f64 / n_total as f64);
    println!("  N- = {} ({:.2}%)", n_negative, 100.0 * n_negative as f64 / n_total as f64);

    // Janus density
    let rho_plus = OMEGA_B * RHO_CRIT;
    let rho_total = rho_plus * (1.0 + MU);
    let m_total = rho_total * BOX_SIZE.powi(3);
    let mass_factor_janus = G_COSMO * m_total / n_total as f64;

    println!("\n  mass_factor = {:.4e}", mass_factor_janus);

    // Generate perturbed grid ICs
    println!("\nGenerating perturbed grid ICs (δ = {}%)...", delta_init);
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let half_box = BOX_SIZE / 2.0;

    // Grid parameters
    let n_per_dim = (n_total as f64).cbrt().ceil() as usize;
    let cell_size = BOX_SIZE / n_per_dim as f64;
    let perturbation_sigma = delta_frac * cell_size;

    println!("  Grid: {}³ = {} cells", n_per_dim, n_per_dim.pow(3));
    println!("  Cell size: {:.2} Mpc", cell_size);
    println!("  Perturbation σ: {:.4} Mpc", perturbation_sigma);

    let mut pos_f32: Vec<f32> = Vec::with_capacity(n_total * 3);
    let mut vel_f32: Vec<f32> = Vec::with_capacity(n_total * 3);
    let mut signs_i8: Vec<i8> = Vec::with_capacity(n_total);

    // Assign signs: first N+ particles positive, rest negative
    let mut particle_idx = 0;

    for ix in 0..n_per_dim {
        for iy in 0..n_per_dim {
            for iz in 0..n_per_dim {
                if particle_idx >= n_total {
                    break;
                }

                // Base grid position (centered on box)
                let x_grid = (ix as f64 + 0.5) * cell_size - half_box;
                let y_grid = (iy as f64 + 0.5) * cell_size - half_box;
                let z_grid = (iz as f64 + 0.5) * cell_size - half_box;

                // Add Gaussian perturbation
                let x = x_grid + rng.sample::<f64, _>(StandardNormal) * perturbation_sigma;
                let y = y_grid + rng.sample::<f64, _>(StandardNormal) * perturbation_sigma;
                let z = z_grid + rng.sample::<f64, _>(StandardNormal) * perturbation_sigma;

                // Wrap to periodic box
                let x_wrapped = ((x + half_box) % BOX_SIZE + BOX_SIZE) % BOX_SIZE - half_box;
                let y_wrapped = ((y + half_box) % BOX_SIZE + BOX_SIZE) % BOX_SIZE - half_box;
                let z_wrapped = ((z + half_box) % BOX_SIZE + BOX_SIZE) % BOX_SIZE - half_box;

                pos_f32.push(x_wrapped as f32);
                pos_f32.push(y_wrapped as f32);
                pos_f32.push(z_wrapped as f32);
                vel_f32.push(0.0);
                vel_f32.push(0.0);
                vel_f32.push(0.0);

                // Assign sign: random but respecting μ ratio
                let sign = if rng.gen::<f64>() < 1.0 / (1.0 + MU) { 1i8 } else { -1i8 };
                signs_i8.push(sign);

                particle_idx += 1;
            }
        }
    }

    // If we have fewer particles than requested (shouldn't happen with cbrt ceil)
    while particle_idx < n_total {
        let x = rng.gen::<f64>() * BOX_SIZE - half_box;
        let y = rng.gen::<f64>() * BOX_SIZE - half_box;
        let z = rng.gen::<f64>() * BOX_SIZE - half_box;
        pos_f32.push(x as f32);
        pos_f32.push(y as f32);
        pos_f32.push(z as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        let sign = if rng.gen::<f64>() < 1.0 / (1.0 + MU) { 1i8 } else { -1i8 };
        signs_i8.push(sign);
        particle_idx += 1;
    }

    let actual_n_plus: usize = signs_i8.iter().filter(|&&s| s > 0).count();
    let actual_n_minus = n_total - actual_n_plus;
    println!("  Actual N+ = {}, N- = {}", actual_n_plus, actual_n_minus);

    // Setup directories
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    // CSV
    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,t_gyr,z,diff_pois,corr_delta,rho_max_ratio").unwrap();

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);
    sim.set_mass_factor(mass_factor_janus);

    // Time mapping
    let t_start = expansion.t_start;
    let t_end = expansion.t_end;
    let dt_gyr = (t_end - t_start) / STEPS as f64;

    let start = Instant::now();
    println!("\nStarting evolution...\n");

    for step in 0..=STEPS {
        let t_current = t_start + step as f64 * dt_gyr;
        let state = expansion.at_time(t_current);

        if step > 0 {
            sim.set_current_z(state.z);
            sim.step_treepm_gpu(DT, R_CUT, state.h_plus, 1.0)
                .expect("TreePM step failed");
        }

        // Logging
        if step % CSV_INTERVAL == 0 {
            let (positions, _, signs) = sim.get_particles().unwrap();
            let (diff_pois, corr_delta, rho_max_ratio) =
                compute_metrics(&positions, &signs, BOX_SIZE, N_CELLS);

            writeln!(ts_file, "{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                     step, t_current, state.z, diff_pois, corr_delta, rho_max_ratio).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_min = if rate > 0.0 { (STEPS - step) as f64 / rate / 60.0 } else { 0.0 };

            println!("  step {:4} | z={:.2} | Diff/Pois={:.3} | Corr={:.3} | ρmax/ρ̄={:.1} | ETA {:.0}min",
                     step, state.z, diff_pois, corr_delta, rho_max_ratio, eta_min);
        }

        // Snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, state.z, BOX_SIZE);
        }
    }

    ts_file.flush().unwrap();

    // Final summary
    let elapsed = start.elapsed().as_secs_f64();
    let final_state = expansion.at_time(t_end);
    let (positions, _, signs) = sim.get_particles().unwrap();
    let (diff_pois, corr_delta, rho_max_ratio) =
        compute_metrics(&positions, &signs, BOX_SIZE, N_CELLS);

    let summary = format!(r#"{{
  "delta_init_percent": {},
  "mu": {},
  "n_total": {},
  "box_mpc": {},
  "z_final": {:.4},
  "diff_pois": {:.4},
  "corr_delta": {:.4},
  "rho_max_ratio": {:.2},
  "runtime_s": {:.1}
}}"#, delta_init, MU, n_total, BOX_SIZE, final_state.z, diff_pois, corr_delta, rho_max_ratio, elapsed);

    fs::write(base_dir.join("summary.json"), &summary).expect("Failed to write summary");

    println!("\n================================================================");
    println!("  CI SCAN δ={}% COMPLETE", delta_init);
    println!("================================================================");
    println!("  Final z: {:.4}", final_state.z);
    println!("  Diff/Pois: {:.4}", diff_pois);
    println!("  Corr(δ+,δ-): {:.4}", corr_delta);
    println!("  ρmax/ρ̄: {:.2} (>10 = nonlinear)", rho_max_ratio);
    println!("  Runtime: {:.1}s ({:.1} min)", elapsed, elapsed / 60.0);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn compute_metrics(positions: &[f32], signs: &[i8], box_size: f64, n_cells: usize) -> (f64, f64, f64) {
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n_cells_cubed = n_cells * n_cells * n_cells;
    let n = signs.len();

    let mut n_plus = vec![0u32; n_cells_cubed];
    let mut n_minus = vec![0u32; n_cells_cubed];

    for i in 0..n {
        let x = ((positions[i*3] as f64 + half_box) % box_size) / cell_size;
        let y = ((positions[i*3+1] as f64 + half_box) % box_size) / cell_size;
        let z = ((positions[i*3+2] as f64 + half_box) % box_size) / cell_size;

        let ix = (x as usize).min(n_cells - 1);
        let iy = (y as usize).min(n_cells - 1);
        let iz = (z as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        if signs[i] > 0 {
            n_plus[idx] += 1;
        } else {
            n_minus[idx] += 1;
        }
    }

    // Compute means
    let total_plus: u64 = n_plus.iter().map(|&x| x as u64).sum();
    let total_minus: u64 = n_minus.iter().map(|&x| x as u64).sum();
    let mean_plus = total_plus as f64 / n_cells_cubed as f64;
    let mean_minus = total_minus as f64 / n_cells_cubed as f64;

    // 1. Diff/Pois
    let diff: Vec<f64> = n_plus.iter().zip(n_minus.iter())
        .map(|(&p, &m)| p as f64 - m as f64)
        .collect();
    let diff_mean: f64 = diff.iter().sum::<f64>() / n_cells_cubed as f64;
    let diff_var: f64 = diff.iter().map(|d| (d - diff_mean).powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let poisson_var = mean_plus + mean_minus;
    let diff_pois = if poisson_var > 0.0 { diff_var / poisson_var } else { 1.0 };

    // 2. Corr(δ+, δ-)
    let delta_plus: Vec<f64> = n_plus.iter()
        .map(|&x| if mean_plus > 0.0 { (x as f64 - mean_plus) / mean_plus } else { 0.0 })
        .collect();
    let delta_minus: Vec<f64> = n_minus.iter()
        .map(|&x| if mean_minus > 0.0 { (x as f64 - mean_minus) / mean_minus } else { 0.0 })
        .collect();

    let cov: f64 = delta_plus.iter().zip(delta_minus.iter())
        .map(|(dp, dm)| dp * dm)
        .sum::<f64>() / n_cells_cubed as f64;
    let var_plus: f64 = delta_plus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let var_minus: f64 = delta_minus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let corr_delta = if var_plus > 0.0 && var_minus > 0.0 {
        cov / (var_plus.sqrt() * var_minus.sqrt())
    } else {
        0.0
    };

    // 3. ρmax/ρ̄ for total density (structure detection)
    let n_total_cells: Vec<u32> = n_plus.iter().zip(n_minus.iter())
        .map(|(&p, &m)| p + m)
        .collect();
    let mean_total = (total_plus + total_minus) as f64 / n_cells_cubed as f64;
    let max_total = *n_total_cells.iter().max().unwrap_or(&0) as f64;
    let rho_max_ratio = if mean_total > 0.0 { max_total / mean_total } else { 1.0 };

    (diff_pois, corr_delta, rho_max_ratio)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, z: f64, box_size: f64) {
    let (positions, velocities, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let snap_path = path.join(format!("snap_{:05}.bin", step));

    let file = match File::create(&snap_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(box_size as f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

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
