//! Scan μ — Calibration du modèle Janus
//!
//! Usage: cargo run --release --features "cuda cufft" --bin scan_mu -- --mu 16
//!
//! Paramètres fixes:
//! - N = 2M, Box = 500 Mpc
//! - TreePM: PM 256³, r_cut = 20 Mpc
//! - λ = 0, 2000 steps, snapshots every 20

use rand::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// N_TOTAL set via --n argument, default 2M
const DEFAULT_N: usize = 2_000_000;
const DEFAULT_BOX: f64 = 500.0;
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.005;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const ETA: f64 = 1.0;
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 20;
const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 20.0;
const N_CELLS: usize = 32;

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse --mu argument
    let mu: f64 = args.iter()
        .position(|x| x == "--mu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);

    // Parse --n argument (number of particles)
    let n_total: usize = args.iter()
        .position(|x| x == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);

    // Parse --box argument (box size in Mpc)
    let box_size: f64 = args.iter()
        .position(|x| x == "--box")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BOX);

    let run_name = format!("scan_mu_{}_{}M_{}Mpc", mu as u32, n_total / 1_000_000, box_size as u32);

    println!("================================================================");
    println!("  Scan μ — Calibration Janus");
    println!("================================================================");
    println!("  μ = {}", mu);
    println!("  N = {}M", n_total / 1_000_000);

    let n_positive = (n_total as f64 / (1.0 + mu)) as usize;
    let n_negative = n_total - n_positive;

    println!("  N_total = {}", n_total);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / n_total as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / n_total as f64);
    println!("  Box = {} Mpc", box_size);
    println!("  TreePM: PM 256³, r_cut = {} Mpc", R_CUT);
    println!("  λ = 0 (pure anti-Newton)");
    println!("  Steps = {}", STEPS);
    println!("================================================================");
    println!();

    // Generate uniform random ICs
    println!("Generating uniform random ICs...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let half_box = box_size / 2.0;

    let mut pos_f32: Vec<f32> = Vec::with_capacity(n_total * 3);
    let mut vel_f32: Vec<f32> = Vec::with_capacity(n_total * 3);
    let mut signs_i8: Vec<i8> = Vec::with_capacity(n_total);

    for _ in 0..n_positive {
        pos_f32.push((rng.random::<f64>() * box_size - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * box_size - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * box_size - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(1);
    }

    for _ in 0..n_negative {
        pos_f32.push((rng.random::<f64>() * box_size - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * box_size - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * box_size - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(-1);
    }

    println!("  Generated {} particles", n_total);

    // Setup output
    let base_dir = std::path::Path::new("/app/output").join(&run_name);
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,z,a,P,void_frac,wall_frac,n_blobs,r_eff_mean").unwrap();

    // Write run config
    let mut config_file = File::create(base_dir.join("config.txt")).expect("Failed to create config");
    writeln!(config_file, "mu={}", mu).unwrap();
    writeln!(config_file, "n_total={}", n_total).unwrap();
    writeln!(config_file, "n_positive={}", n_positive).unwrap();
    writeln!(config_file, "n_negative={}", n_negative).unwrap();
    writeln!(config_file, "box_size={}", box_size).unwrap();
    writeln!(config_file, "steps={}", STEPS).unwrap();
    writeln!(config_file, "r_cut={}", R_CUT).unwrap();

    // Initialize simulation
    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, box_size
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let start = Instant::now();

    println!();
    println!("Starting scan μ={} evolution...", mu);
    println!();

    for step in 0..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };

        if step > 0 {
            sim.set_current_z(z);
            sim.step_treepm_gpu(DT, R_CUT, h, dtau_per_dt)
                .expect("TreePM step failed");
        }

        // Logging every CSV_INTERVAL steps
        if step % CSV_INTERVAL == 0 {
            let (positions, _, signs) = sim.get_particles().unwrap();

            // Compute all metrics
            let purity = compute_purity(&positions, &signs, box_size, N_CELLS);
            let (void_frac, wall_frac) = compute_void_wall_fractions(&positions, &signs, box_size, N_CELLS);
            let (n_blobs, r_eff_mean) = compute_blob_stats(&positions, &signs, box_size);

            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4},{:.4},{},{:.2}",
                     step, z, a, purity, void_frac, wall_frac, n_blobs, r_eff_mean).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_min = if rate > 0.0 { (STEPS - step) as f64 / rate / 60.0 } else { 0.0 };

            println!("  step {:4} | z={:.2} | P={:.3} | void={:.1}% | wall={:.1}% | blobs={} | ETA {:.0}min",
                     step, z, purity, void_frac * 100.0, wall_frac * 100.0, n_blobs, eta_min);
        }

        // Snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, z, box_size);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let (positions, _, signs) = sim.get_particles().unwrap();
    let final_purity = compute_purity(&positions, &signs, box_size, N_CELLS);
    let (final_void, final_wall) = compute_void_wall_fractions(&positions, &signs, box_size, N_CELLS);
    let (final_blobs, final_r_eff) = compute_blob_stats(&positions, &signs, box_size);

    println!();
    println!("================================================================");
    println!("  SCAN μ={} COMPLETE", mu);
    println!("================================================================");
    println!("  Final Purity:    {:.4}", final_purity);
    println!("  Final void_frac: {:.4} ({:.1}%)", final_void, final_void * 100.0);
    println!("  Final wall_frac: {:.4} ({:.1}%)", final_wall, final_wall * 100.0);
    println!("  Final n_blobs:   {}", final_blobs);
    println!("  Final r_eff:     {:.2} Mpc", final_r_eff);
    println!("  Runtime: {:.1}s ({:.1} min)", elapsed, elapsed / 60.0);
    println!("  Output: {:?}", base_dir);
    println!("================================================================");

    // Write summary
    let mut summary = File::create(base_dir.join("summary.json")).expect("Failed to create summary");
    writeln!(summary, "{{").unwrap();
    writeln!(summary, "  \"mu\": {},", mu).unwrap();
    writeln!(summary, "  \"box_size\": {},", box_size).unwrap();
    writeln!(summary, "  \"n_total\": {},", n_total).unwrap();
    writeln!(summary, "  \"purity\": {:.4},", final_purity).unwrap();
    writeln!(summary, "  \"void_frac\": {:.4},", final_void).unwrap();
    writeln!(summary, "  \"wall_frac\": {:.4},", final_wall).unwrap();
    writeln!(summary, "  \"n_blobs\": {},", final_blobs).unwrap();
    writeln!(summary, "  \"r_eff_mean\": {:.2},", final_r_eff).unwrap();
    writeln!(summary, "  \"runtime_s\": {:.1}", elapsed).unwrap();
    writeln!(summary, "}}").unwrap();
}

#[cfg(feature = "cuda")]
fn compute_purity(positions: &[f32], signs: &[i8], box_size: f64, n_cells: usize) -> f64 {
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

    let mut weighted_purity = 0.0;
    let mut total_weight = 0.0;

    for idx in 0..n_cells_cubed {
        let np = n_plus[idx] as f64;
        let nm = n_minus[idx] as f64;
        let weight = np + nm;
        if weight > 0.0 {
            let purity = (np - nm).abs() / weight;
            weighted_purity += purity * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 { weighted_purity / total_weight } else { 0.0 }
}

#[cfg(feature = "cuda")]
fn compute_void_wall_fractions(positions: &[f32], signs: &[i8], box_size: f64, n_cells: usize) -> (f64, f64) {
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

    let mut void_cells = 0;
    let mut wall_cells = 0;
    let mut occupied_cells = 0;

    for idx in 0..n_cells_cubed {
        let np = n_plus[idx] as f64;
        let nm = n_minus[idx] as f64;
        let total = np + nm;

        if total > 0.0 {
            occupied_cells += 1;
            // Void: >90% m⁻
            if nm / total > 0.90 {
                void_cells += 1;
            }
            // Wall: >90% m⁺
            if np / total > 0.90 {
                wall_cells += 1;
            }
        }
    }

    let void_frac = if occupied_cells > 0 { void_cells as f64 / occupied_cells as f64 } else { 0.0 };
    let wall_frac = if occupied_cells > 0 { wall_cells as f64 / occupied_cells as f64 } else { 0.0 };

    (void_frac, wall_frac)
}

#[cfg(feature = "cuda")]
fn compute_blob_stats(positions: &[f32], signs: &[i8], box_size: f64) -> (usize, f64) {
    // Simple blob detection: find connected components of m+ particles
    // Use grid-based approach with 16³ cells
    let n_cells: usize = 16;
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n = signs.len();

    // Count m+ in each cell
    let mut cell_counts = vec![0u32; n_cells * n_cells * n_cells];
    let mut cell_com = vec![(0.0f64, 0.0f64, 0.0f64); n_cells * n_cells * n_cells];

    for i in 0..n {
        if signs[i] <= 0 { continue; }

        let px = positions[i*3] as f64;
        let py = positions[i*3+1] as f64;
        let pz = positions[i*3+2] as f64;

        let x = ((px + half_box) % box_size) / cell_size;
        let y = ((py + half_box) % box_size) / cell_size;
        let z = ((pz + half_box) % box_size) / cell_size;

        let ix = (x as usize).min(n_cells - 1);
        let iy = (y as usize).min(n_cells - 1);
        let iz = (z as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        cell_counts[idx] += 1;
        cell_com[idx].0 += px;
        cell_com[idx].1 += py;
        cell_com[idx].2 += pz;
    }

    // Find blob centers (cells with significant m+ density)
    let threshold = 10; // minimum particles to be considered a blob seed
    let mut blob_cells: Vec<usize> = Vec::new();

    for idx in 0..cell_counts.len() {
        if cell_counts[idx] >= threshold {
            blob_cells.push(idx);
        }
    }

    // Simple connected component labeling using union-find
    let mut labels: Vec<i32> = vec![-1; n_cells * n_cells * n_cells];
    let mut n_blobs = 0;

    for &idx in &blob_cells {
        if labels[idx] >= 0 { continue; }

        // BFS to find connected component
        let mut queue = vec![idx];
        labels[idx] = n_blobs as i32;

        while let Some(current) = queue.pop() {
            let ix = current / (n_cells * n_cells);
            let iy = (current / n_cells) % n_cells;
            let iz = current % n_cells;

            // Check 6 neighbors
            for (dx, dy, dz) in [(-1i32,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)] {
                let nx = ((ix as i32 + dx + n_cells as i32) % n_cells as i32) as usize;
                let ny = ((iy as i32 + dy + n_cells as i32) % n_cells as i32) as usize;
                let nz = ((iz as i32 + dz + n_cells as i32) % n_cells as i32) as usize;
                let nidx = nx * n_cells * n_cells + ny * n_cells + nz;

                if cell_counts[nidx] >= threshold && labels[nidx] < 0 {
                    labels[nidx] = n_blobs as i32;
                    queue.push(nidx);
                }
            }
        }
        n_blobs += 1;
    }

    // Compute effective radius for each blob
    let mut blob_radii: Vec<f64> = Vec::new();

    for blob_id in 0..n_blobs {
        let mut total_particles = 0u64;
        for idx in 0..labels.len() {
            if labels[idx] == blob_id as i32 {
                total_particles += cell_counts[idx] as u64;
            }
        }

        // r_eff = (3V/4π)^(1/3) where V ∝ N (assuming uniform density)
        // Approximate using cell volume
        let n_blob_cells = labels.iter().filter(|&&l| l == blob_id as i32).count();
        let volume = n_blob_cells as f64 * cell_size.powi(3);
        let r_eff = (3.0 * volume / (4.0 * std::f64::consts::PI)).powf(1.0/3.0);
        blob_radii.push(r_eff);
    }

    let r_eff_mean = if blob_radii.is_empty() { 0.0 } else {
        blob_radii.iter().sum::<f64>() / blob_radii.len() as f64
    };

    (n_blobs, r_eff_mean)
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
