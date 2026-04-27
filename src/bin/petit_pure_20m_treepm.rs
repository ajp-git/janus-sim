//! Petit Pure 20M TreePM — μ=8, λ=0, Isotropic PM + Short-range BH
//!
//! Fixes axial mode artifact by using FFT-based PM for long-range forces.
//! PM is isotropic by construction, eliminating coordinate-aligned bias.
//!
//! Architecture:
//! - Long-range (r > r_cut): PM with cuFFT (isotropic)
//! - Short-range (r < r_cut): BH tree (local, no axis preference)

use rand::prelude::*;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use std::collections::{VecDeque};

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const MU: f64 = 8.0;
const N_TOTAL: usize = 20_000_000;
const BOX_SIZE: f64 = 500.0;
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.005;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const ETA: f64 = 1.0;
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 5;

// TreePM parameters
const R_CUT: f64 = BOX_SIZE / 16.0;  // ~31 Mpc - BH handles r < r_cut

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("========================================================");
    println!("  PETIT PURE 20M TreePM — μ={}, λ=0, ISOTROPIC PM", MU);
    println!("========================================================");
    println!("  TreePM: PM (FFT) + BH (r < {:.1} Mpc)", R_CUT);
    println!("  PM grid: 128³ (isotropic by construction)");
    println!("========================================================");

    let n_positive = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_negative = N_TOTAL - n_positive;

    println!("  N_total = {} (20M)", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", n_negative as f64 / n_positive as f64);
    println!("  Box = {} Mpc, z = {} -> 0", BOX_SIZE, Z_INIT);
    println!("  λ = 0 (pure anti-Newton 1/r²)");
    println!("  Snapshots every {} steps", SNAPSHOT_INTERVAL);
    println!("========================================================");
    println!();

    // Generate uniform random ICs
    println!("Generating uniform random ICs for 20M particles...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let half_box = BOX_SIZE / 2.0;

    let mut pos_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut signs_i8: Vec<i8> = Vec::with_capacity(N_TOTAL);

    // Positive mass particles
    for _ in 0..n_positive {
        pos_f32.push((rng.gen::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.gen::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.gen::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(1);
    }

    // Negative mass particles
    for _ in 0..n_negative {
        pos_f32.push((rng.gen::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.gen::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.gen::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(-1);
    }

    println!("  Generated {} particles", N_TOTAL);

    // Setup output
    let base_dir = std::path::Path::new("/app/output/petit_pure_20m_treepm");
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,z,a,P,void_frac,wall_frac,n_blobs,r_blob_mean").unwrap();

    // Initialize simulation
    println!("Initializing GPU simulation with TreePM...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);  // Pure 1/r², no Yukawa

    println!("  θ = {} (BH opening angle)", THETA);
    println!("  ε = {} Mpc (softening)", SOFTENING);
    println!("  r_cut = {:.1} Mpc (PM/BH split)", R_CUT);
    println!("  λ₀ = 0.0 (pure anti-Newton)");

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let start = Instant::now();

    println!();
    println!("Starting TreePM evolution...");

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
            // TreePM step: PM (FFT) + BH (r < r_cut)
            sim.step_treepm_gpu(DT, R_CUT, h, dtau_per_dt)
                .expect("TreePM step failed");
        }

        // Snapshot + metrics
        if step % SNAPSHOT_INTERVAL == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            let (void_frac, wall_frac, n_blobs, r_blob_mean) = compute_blob_metrics(&sim);

            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4},{:.4},{},{}",
                     step, z, a, purity, void_frac, wall_frac, n_blobs, r_blob_mean).unwrap();

            let elapsed_min = start.elapsed().as_secs_f64() / 60.0;
            let step_time = if step > 0 { elapsed_min / step as f64 * 60.0 } else { 0.0 };

            println!("  step {:4} | z={:.3} | P={:.3} | blobs={} | r={:.1}Mpc | {:.1}s/step",
                     step, z, purity, n_blobs, r_blob_mean, step_time);

            save_snapshot(&sim, &snap_dir, step, z);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64() / 60.0;
    let final_purity = sim.local_purity(32).unwrap_or(0.0);
    let (void_frac, wall_frac, n_blobs, r_blob_mean) = compute_blob_metrics(&sim);

    println!();
    println!("========================================================");
    println!("  PETIT PURE 20M TreePM COMPLETE");
    println!("========================================================");
    println!("  P(z=0) = {:.4}", final_purity);
    println!("  Void fraction (>95% m⁻) = {:.1}%", void_frac * 100.0);
    println!("  Wall fraction (>95% m⁺) = {:.1}%", wall_frac * 100.0);
    println!("  N_blobs (m⁻ conglomerates) = {}", n_blobs);
    println!("  R_blob mean = {:.1} Mpc", r_blob_mean);
    println!("  Time: {:.1} min ({:.1} hours)", elapsed, elapsed / 60.0);
    println!("========================================================");
}

/// Compute void/wall fractions AND blob count using connected component analysis
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_blob_metrics(sim: &GpuNBodyTwoPass) -> (f64, f64, usize, f64) {
    let (positions, _, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return (0.0, 0.0, 0, 0.0),
    };

    let n_cells = 32;
    let cell_size = BOX_SIZE / n_cells as f64;
    let half_box = BOX_SIZE / 2.0;
    let n_cells_3 = n_cells * n_cells * n_cells;

    let mut n_plus = vec![0usize; n_cells_3];
    let mut n_minus = vec![0usize; n_cells_3];

    for i in 0..signs.len() {
        let x = (positions[i*3] as f64 + half_box) % BOX_SIZE;
        let y = (positions[i*3+1] as f64 + half_box) % BOX_SIZE;
        let z = (positions[i*3+2] as f64 + half_box) % BOX_SIZE;

        let ix = ((x / cell_size) as usize).min(n_cells - 1);
        let iy = ((y / cell_size) as usize).min(n_cells - 1);
        let iz = ((z / cell_size) as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        if signs[i] > 0 { n_plus[idx] += 1; }
        else { n_minus[idx] += 1; }
    }

    // Void/wall fractions
    let mut void_cells = 0;
    let mut wall_cells = 0;
    let mut total_cells = 0;
    let mut void_mask = vec![false; n_cells_3];

    for idx in 0..n_cells_3 {
        let total = n_plus[idx] + n_minus[idx];
        if total > 10 {
            total_cells += 1;
            let minus_frac = n_minus[idx] as f64 / total as f64;
            let plus_frac = n_plus[idx] as f64 / total as f64;
            if minus_frac > 0.95 {
                void_cells += 1;
                void_mask[idx] = true;
            }
            if plus_frac > 0.95 { wall_cells += 1; }
        }
    }

    let void_frac = if total_cells > 0 { void_cells as f64 / total_cells as f64 } else { 0.0 };
    let wall_frac = if total_cells > 0 { wall_cells as f64 / total_cells as f64 } else { 0.0 };

    // Connected component analysis for blob counting (periodic BC)
    let (n_blobs, blob_sizes) = count_connected_components(&void_mask, n_cells);

    // Compute mean blob radius (assuming spherical: V = 4/3 π r³)
    let r_blob_mean = if !blob_sizes.is_empty() {
        let mean_volume = blob_sizes.iter().sum::<usize>() as f64 / blob_sizes.len() as f64;
        let cell_volume = cell_size.powi(3);
        let blob_volume = mean_volume * cell_volume;
        (3.0 * blob_volume / (4.0 * std::f64::consts::PI)).powf(1.0/3.0)
    } else {
        0.0
    };

    (void_frac, wall_frac, n_blobs, r_blob_mean)
}

/// BFS connected components with periodic boundary conditions
fn count_connected_components(mask: &[bool], n_cells: usize) -> (usize, Vec<usize>) {
    let n_cells_3 = n_cells * n_cells * n_cells;
    let mut visited = vec![false; n_cells_3];
    let mut blob_sizes = Vec::new();

    // 6-connectivity neighbors (periodic)
    let neighbors = |ix: usize, iy: usize, iz: usize| -> Vec<(usize, usize, usize)> {
        let mut nbrs = Vec::with_capacity(6);
        nbrs.push(((ix + 1) % n_cells, iy, iz));
        nbrs.push(((ix + n_cells - 1) % n_cells, iy, iz));
        nbrs.push((ix, (iy + 1) % n_cells, iz));
        nbrs.push((ix, (iy + n_cells - 1) % n_cells, iz));
        nbrs.push((ix, iy, (iz + 1) % n_cells));
        nbrs.push((ix, iy, (iz + n_cells - 1) % n_cells));
        nbrs
    };

    let idx_to_xyz = |idx: usize| -> (usize, usize, usize) {
        let ix = idx / (n_cells * n_cells);
        let iy = (idx / n_cells) % n_cells;
        let iz = idx % n_cells;
        (ix, iy, iz)
    };

    let xyz_to_idx = |ix: usize, iy: usize, iz: usize| -> usize {
        ix * n_cells * n_cells + iy * n_cells + iz
    };

    for start_idx in 0..n_cells_3 {
        if !mask[start_idx] || visited[start_idx] {
            continue;
        }

        // BFS from this cell
        let mut queue = VecDeque::new();
        queue.push_back(start_idx);
        visited[start_idx] = true;
        let mut component_size = 0;

        while let Some(idx) = queue.pop_front() {
            component_size += 1;
            let (ix, iy, iz) = idx_to_xyz(idx);

            for (nx, ny, nz) in neighbors(ix, iy, iz) {
                let nidx = xyz_to_idx(nx, ny, nz);
                if mask[nidx] && !visited[nidx] {
                    visited[nidx] = true;
                    queue.push_back(nidx);
                }
            }
        }

        blob_sizes.push(component_size);
    }

    (blob_sizes.len(), blob_sizes)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, z: f64) {
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

    // Header: n, box_size, step, z
    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(BOX_SIZE as f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

    // Per particle: x, y, z, vx, vy, vz, sign (25 bytes each)
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

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires --features cuda,cufft");
    eprintln!("Make sure libcufft_wrapper.so is built: ./cuda/build_cufft.sh");
    std::process::exit(1);
}
