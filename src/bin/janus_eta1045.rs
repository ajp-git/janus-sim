//! Janus simulation with observational η = 1.045
//!
//! μ ≈ 1.045 → near-symmetric populations (N+ ≈ N- ≈ 10M each)
//! Expected: symmetric sponge-like morphology
//!
//! Usage: cargo run --release --features "cuda cufft" --bin janus_eta1045

use rand::prelude::*;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::janus_expansion::JanusExpansion;

const N_TOTAL: usize = 20_000_000;
const MU: f64 = 1.045;  // Observational value from Pantheon+ fit
const BOX_SIZE: f64 = 1000.0;  // 1000 Mpc (Gpc scale)
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.005;
const STEPS: usize = 4000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;  // 10 kpc softening for 1 Gpc box
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 10;  // Every 10 steps (401 total)
const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 200.0;  // 200 Mpc PM/Tree cutoff
const N_CELLS: usize = 64;  // For segregation metrics

const G_COSMO: f64 = 4.499e-15;
const RHO_CRIT: f64 = 1.36e11;
const OMEGA_B: f64 = 0.05;

#[cfg(feature = "cuda")]
fn main() {
    let run_name = "janus_eta1045".to_string();

    println!("================================================================");
    println!("  JANUS η=1.045 OBSERVATIONAL RUN");
    println!("================================================================");
    println!("  N = {}M, Box = {} Mpc (1 Gpc)", N_TOTAL / 1_000_000, BOX_SIZE as u32);
    println!("  μ = {} (near-symmetric: N+ ≈ N-)", MU);
    println!("  z_init = {}, Steps = {}", Z_INIT, STEPS);
    println!("  Snapshots every {} steps ({} total)", SNAPSHOT_INTERVAL, STEPS / SNAPSHOT_INTERVAL + 1);
    println!("  λ₀ = 0 (no screening)");
    println!("================================================================");

    println!("\nInitializing Janus cosmology (t⁺ = α²(μ + ½sinh²μ))...");
    let expansion = JanusExpansion::new(Z_INIT, 5000);

    let base_dir = std::path::Path::new("/app/output").join(&run_name);
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");
    expansion.export_csv(base_dir.join("janus_cosmology.csv").to_str().unwrap())
        .expect("Failed to export cosmology");

    // With μ = 1.045, populations are nearly equal
    // N+ = N_total / (1 + μ) ≈ 9.78M
    // N- = N_total - N+ ≈ 10.22M
    let n_positive = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_negative = N_TOTAL - n_positive;

    println!("\n  N+ = {} ({:.2}%)", n_positive, 100.0 * n_positive as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.2}%)", n_negative, 100.0 * n_negative as f64 / N_TOTAL as f64);
    println!("  μ = N-/N+ = {:.4}", n_negative as f64 / n_positive as f64);

    let rho_plus = OMEGA_B * RHO_CRIT;
    let rho_total = rho_plus * (1.0 + MU);
    let m_total = rho_total * BOX_SIZE.powi(3);
    let mass_factor_janus = G_COSMO * m_total / N_TOTAL as f64;

    let omega_sq = 4.0 * std::f64::consts::PI * G_COSMO * rho_total;
    let t_inst = 1.0 / omega_sq.sqrt();
    println!("\n  t_inst = {:.2} Gyr", t_inst);
    println!("  mass_factor = {:.4e}", mass_factor_janus);

    println!("\nGenerating random ICs for {}M particles...", N_TOTAL / 1_000_000);
    let mut rng = StdRng::seed_from_u64(SEED);
    let half_box = BOX_SIZE / 2.0;

    let mut pos_f32 = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32 = Vec::with_capacity(N_TOTAL * 3);
    let mut signs_i8 = Vec::with_capacity(N_TOTAL);

    for i in 0..N_TOTAL {
        let x = rng.random::<f64>() * BOX_SIZE - half_box;
        let y = rng.random::<f64>() * BOX_SIZE - half_box;
        let z = rng.random::<f64>() * BOX_SIZE - half_box;
        pos_f32.push(x as f32);
        pos_f32.push(y as f32);
        pos_f32.push(z as f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        let sign = if i < n_positive { 1i8 } else { -1i8 };
        signs_i8.push(sign);
    }

    // Shuffle to randomize spatial distribution of signs
    let mut indices: Vec<usize> = (0..N_TOTAL).collect();
    indices.shuffle(&mut rng);
    let mut pos_shuffled = vec![0.0f32; N_TOTAL * 3];
    let mut vel_shuffled = vec![0.0f32; N_TOTAL * 3];
    let mut signs_shuffled = vec![0i8; N_TOTAL];
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        pos_shuffled[new_idx * 3] = pos_f32[old_idx * 3];
        pos_shuffled[new_idx * 3 + 1] = pos_f32[old_idx * 3 + 1];
        pos_shuffled[new_idx * 3 + 2] = pos_f32[old_idx * 3 + 2];
        vel_shuffled[new_idx * 3] = vel_f32[old_idx * 3];
        vel_shuffled[new_idx * 3 + 1] = vel_f32[old_idx * 3 + 1];
        vel_shuffled[new_idx * 3 + 2] = vel_f32[old_idx * 3 + 2];
        signs_shuffled[new_idx] = signs_i8[old_idx];
    }

    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_shuffled, vel_shuffled, signs_shuffled, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_mass_factor(mass_factor_janus);
    sim.set_softening(SOFTENING);
    sim.set_theta(THETA);

    let ts_path = base_dir.join("time_series.csv");
    let mut ts_file = BufWriter::new(File::create(&ts_path).unwrap());
    writeln!(ts_file, "step,t_gyr,z,a,H,diff_pois,corr_delta,exc_var_minus,rho_plus_max,v_plus_mean,v_minus_mean").unwrap();

    let state_init = expansion.at_redshift(Z_INIT);
    let state_final = expansion.at_redshift(0.0);
    let t_start = state_init.t_gyr;
    let t_end = state_final.t_gyr;
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

        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = snap_dir.join(format!("snap_{:05}.bin", step));
            save_snapshot(&sim, &snap_path, step, state.z, BOX_SIZE);
        }

        if step % CSV_INTERVAL == 0 {
            let (positions, velocities, signs) = sim.get_particles().unwrap();
            let (diff_pois, corr_delta, exc_var_minus, rho_plus_max, v_plus_mean, v_minus_mean) =
                compute_metrics(&positions, &velocities, &signs, BOX_SIZE, N_CELLS);

            writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.4},{:.4},{:.4},{:.2},{:.4},{:.4}",
                     step, t_current, state.z, state.a_plus, state.h_plus,
                     diff_pois, corr_delta, exc_var_minus, rho_plus_max, v_plus_mean, v_minus_mean).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_min = if rate > 0.0 { (STEPS - step) as f64 / rate / 60.0 } else { 0.0 };

            println!("  step {:4}/{} | z={:.2} | Diff/Pois={:.3} | Corr={:.3} | <v+>={:.1} | ETA {:.0}min",
                     step, STEPS, state.z, diff_pois, corr_delta, v_plus_mean, eta_min);
        }
    }

    ts_file.flush().unwrap();

    let (positions, velocities, signs) = sim.get_particles().unwrap();
    let (diff_pois, corr_delta, exc_var_minus, rho_plus_max, v_plus_mean, v_minus_mean) =
        compute_metrics(&positions, &velocities, &signs, BOX_SIZE, N_CELLS);

    let summary = format!(r#"{{
  "mu": {},
  "eta": 1.045,
  "n_total": {},
  "n_positive": {},
  "n_negative": {},
  "box_mpc": {},
  "z_final": 0.0,
  "diff_pois": {:.4},
  "corr_delta": {:.4},
  "exc_var_minus": {:.4},
  "rho_plus_max": {:.2},
  "v_plus_mean": {:.4},
  "v_minus_mean": {:.4},
  "t_inst_gyr": {:.2},
  "runtime_s": {:.1},
  "n_snapshots": {}
}}"#,
        MU, N_TOTAL, n_positive, n_negative, BOX_SIZE as u32,
        diff_pois, corr_delta, exc_var_minus, rho_plus_max, v_plus_mean, v_minus_mean,
        t_inst, start.elapsed().as_secs_f64(), STEPS / SNAPSHOT_INTERVAL + 1);

    fs::write(base_dir.join("summary.json"), &summary).unwrap();

    let runtime = start.elapsed().as_secs_f64();
    println!("\n================================================================");
    println!("  η=1.045 JANUS SIMULATION COMPLETE");
    println!("================================================================");
    println!("  Diff/Pois: {:.4}", diff_pois);
    println!("  Corr(δ+,δ-): {:.4}", corr_delta);
    println!("  ρ+_max/ρ̄+: {:.2}", rho_plus_max);
    println!("  <v+>: {:.2} km/s", v_plus_mean);
    println!("  <v->: {:.2} km/s", v_minus_mean);
    println!("  Runtime: {:.1} min", runtime / 60.0);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::Path, step: usize, z: f64, box_size: f64) {
    let (positions, velocities, signs) = sim.get_particles().unwrap();
    let n = signs.len();
    let mut file = BufWriter::new(File::create(path).unwrap());
    file.write_all(&(n as u32).to_le_bytes()).unwrap();
    file.write_all(&(box_size as f32).to_le_bytes()).unwrap();
    file.write_all(&(step as u32).to_le_bytes()).unwrap();
    file.write_all(&(z as f32).to_le_bytes()).unwrap();
    for i in 0..n {
        file.write_all(&(positions[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(positions[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(positions[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        file.write_all(&(velocities[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(velocities[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(velocities[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        file.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
    }
}

#[cfg(feature = "cuda")]
fn compute_metrics(positions: &[f32], velocities: &[f32], signs: &[i8], box_size: f64, n_cells: usize)
    -> (f64, f64, f64, f64, f64, f64)
{
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n_particles = signs.len();
    let mut n_plus = vec![0i32; n_cells * n_cells * n_cells];
    let mut n_minus = vec![0i32; n_cells * n_cells * n_cells];

    let mut v_plus_sum = 0.0f64;
    let mut v_minus_sum = 0.0f64;
    let mut n_plus_count = 0usize;
    let mut n_minus_count = 0usize;

    // Velocity conversion: Mpc/Gyr to km/s
    let vel_conv = 977.8;

    for i in 0..n_particles {
        let x = positions[i * 3] as f64;
        let y = positions[i * 3 + 1] as f64;
        let z = positions[i * 3 + 2] as f64;
        let vx = velocities[i * 3] as f64;
        let vy = velocities[i * 3 + 1] as f64;
        let vz = velocities[i * 3 + 2] as f64;
        let v_mag = (vx*vx + vy*vy + vz*vz).sqrt() * vel_conv;

        let ix = (((x + half_box).rem_euclid(box_size)) / cell_size) as usize % n_cells;
        let iy = (((y + half_box).rem_euclid(box_size)) / cell_size) as usize % n_cells;
        let iz = (((z + half_box).rem_euclid(box_size)) / cell_size) as usize % n_cells;
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        if signs[i] > 0 {
            n_plus[idx] += 1;
            v_plus_sum += v_mag;
            n_plus_count += 1;
        } else {
            n_minus[idx] += 1;
            v_minus_sum += v_mag;
            n_minus_count += 1;
        }
    }

    let v_plus_mean = if n_plus_count > 0 { v_plus_sum / n_plus_count as f64 } else { 0.0 };
    let v_minus_mean = if n_minus_count > 0 { v_minus_sum / n_minus_count as f64 } else { 0.0 };

    let n_cells_total = n_cells * n_cells * n_cells;
    let total_plus: i32 = n_plus.iter().sum();
    let total_minus: i32 = n_minus.iter().sum();
    let mean_plus = total_plus as f64 / n_cells_total as f64;
    let mean_minus = total_minus as f64 / n_cells_total as f64;

    // Compute rho_plus_max / rho_plus_mean
    let max_plus = *n_plus.iter().max().unwrap_or(&0) as f64;
    let rho_plus_max = if mean_plus > 0.0 { max_plus / mean_plus } else { 0.0 };

    let mut var_diff = 0.0;
    for i in 0..n_cells_total {
        let diff = n_plus[i] as f64 - n_minus[i] as f64;
        let expected_diff = mean_plus - mean_minus;
        var_diff += (diff - expected_diff).powi(2);
    }
    var_diff /= n_cells_total as f64;
    let poisson_var = mean_plus + mean_minus;
    let diff_over_pois = if poisson_var > 0.0 { var_diff / poisson_var } else { 0.0 };

    let mut sum_delta_plus_sq = 0.0;
    let mut sum_delta_minus_sq = 0.0;
    let mut sum_cross = 0.0;
    for i in 0..n_cells_total {
        let delta_plus = if mean_plus > 0.0 { (n_plus[i] as f64 - mean_plus) / mean_plus } else { 0.0 };
        let delta_minus = if mean_minus > 0.0 { (n_minus[i] as f64 - mean_minus) / mean_minus } else { 0.0 };
        sum_delta_plus_sq += delta_plus * delta_plus;
        sum_delta_minus_sq += delta_minus * delta_minus;
        sum_cross += delta_plus * delta_minus;
    }
    let n = n_cells_total as f64;
    let var_plus = sum_delta_plus_sq / n;
    let var_minus = sum_delta_minus_sq / n;
    let cov = sum_cross / n;
    let corr = if var_plus > 0.0 && var_minus > 0.0 { cov / (var_plus.sqrt() * var_minus.sqrt()) } else { 0.0 };

    let mut var_minus_count = 0.0;
    for i in 0..n_cells_total { var_minus_count += (n_minus[i] as f64 - mean_minus).powi(2); }
    var_minus_count /= n_cells_total as f64;
    let exc_var_minus = if mean_minus > 0.0 { (var_minus_count - mean_minus) / mean_minus } else { 0.0 };

    (diff_over_pois, corr, exc_var_minus, rho_plus_max, v_plus_mean, v_minus_mean)
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("This binary requires --features cuda cufft"); }
