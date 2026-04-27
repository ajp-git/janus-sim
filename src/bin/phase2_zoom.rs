//! Phase 2: High-resolution zoom simulation on segregated region
//! Pure gravity (no cosmological expansion)

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const DT: f64 = 0.001;  // Smaller dt for local dynamics
const STEPS: usize = 4000;
const THETA: f64 = 0.5;  // More accurate
const SOFTENING: f64 = 0.1;  // Softer for low N
const SNAPSHOT_INTERVAL: usize = 20;
const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 10.0;
const N_CELLS: usize = 16;

// Physical constants - using Janus mass factor for consistency
const G_COSMO: f64 = 4.499e-15;
const RHO_CRIT: f64 = 1.36e11;
const OMEGA_B: f64 = 0.05;
const MU: f64 = 19.0;

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    let ic_path = args.iter()
        .position(|x| x == "--ic")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone())
        .unwrap_or("/app/output/phase2_zoom50mpc/extracted_ic.bin".to_string());

    let output_dir = args.iter()
        .position(|x| x == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone())
        .unwrap_or("/app/output/phase2_zoom50mpc".to_string());

    println!("================================================================");
    println!("  PHASE 2: Zoom Simulation — Pure Gravity");
    println!("================================================================");

    // Load ICs
    let mut file = File::open(&ic_path).expect("Failed to open IC file");
    let mut buf = vec![0u8; 16];
    file.read_exact(&mut buf).expect("Failed to read header");

    let n = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let box_size = f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as f64;
    let _step = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let _z = f32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

    println!("  IC file: {}", ic_path);
    println!("  N = {}, Box = {} Mpc", n, box_size);

    // Read particles
    let mut pos_f32 = Vec::with_capacity(n * 3);
    let mut vel_f32 = Vec::with_capacity(n * 3);
    let mut signs_i8 = Vec::with_capacity(n);

    for _ in 0..n {
        let mut pbuf = vec![0u8; 25];
        file.read_exact(&mut pbuf).expect("Failed to read particle");

        let x = f32::from_le_bytes([pbuf[0], pbuf[1], pbuf[2], pbuf[3]]);
        let y = f32::from_le_bytes([pbuf[4], pbuf[5], pbuf[6], pbuf[7]]);
        let z = f32::from_le_bytes([pbuf[8], pbuf[9], pbuf[10], pbuf[11]]);
        let vx = f32::from_le_bytes([pbuf[12], pbuf[13], pbuf[14], pbuf[15]]);
        let vy = f32::from_le_bytes([pbuf[16], pbuf[17], pbuf[18], pbuf[19]]);
        let vz = f32::from_le_bytes([pbuf[20], pbuf[21], pbuf[22], pbuf[23]]);
        let sign = pbuf[24] as i8;

        pos_f32.push(x); pos_f32.push(y); pos_f32.push(z);
        vel_f32.push(vx); vel_f32.push(vy); vel_f32.push(vz);
        signs_i8.push(sign);
    }

    let n_plus: usize = signs_i8.iter().filter(|&&s| s > 0).count();
    let n_minus = n - n_plus;
    let mu_local = n_minus as f64 / n_plus.max(1) as f64;

    println!("  N+ = {}, N- = {}, μ_local = {:.2}", n_plus, n_minus, mu_local);

    // Mass factor for this box
    let rho_plus = OMEGA_B * RHO_CRIT;
    let rho_total = rho_plus * (1.0 + MU);
    let m_total = rho_total * box_size.powi(3);
    let mass_factor = G_COSMO * m_total / n as f64;

    println!("  mass_factor = {:.4e}", mass_factor);
    println!("================================================================");

    // Setup output
    let base_dir = std::path::Path::new(&output_dir);
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    // CSV
    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,t_gyr,diff_pois,corr_delta,rho_plus_max_ratio,v_plus,v_minus").unwrap();

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, box_size
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);
    sim.set_mass_factor(mass_factor);

    let start = Instant::now();
    println!("\nStarting evolution (pure gravity, no expansion)...\n");

    let mut t_gyr = 0.0;

    for step in 0..=STEPS {
        if step > 0 {
            // Pure gravity step - no Hubble friction (h=0)
            sim.set_current_z(0.0);  // z=0 for local dynamics
            sim.step_treepm_gpu(DT, R_CUT, 0.0, 1.0)
                .expect("TreePM step failed");
            t_gyr += DT;
        }

        // Logging
        if step % CSV_INTERVAL == 0 {
            let (positions, velocities, signs) = sim.get_particles().unwrap();
            let (diff_pois, corr_delta, rho_plus_max_ratio, v_plus, v_minus) =
                compute_metrics(&positions, &velocities, &signs, box_size, N_CELLS);

            writeln!(ts_file, "{},{:.6},{:.4},{:.4},{:.2},{:.2},{:.2}",
                     step, t_gyr, diff_pois, corr_delta, rho_plus_max_ratio, v_plus, v_minus).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_min = if rate > 0.0 { (STEPS - step) as f64 / rate / 60.0 } else { 0.0 };

            // Alert conditions
            let alert = if rho_plus_max_ratio > 50.0 {
                ">>> COLLAPSE <<<"
            } else if rho_plus_max_ratio > 10.0 {
                "*** STRUCTURES ***"
            } else {
                ""
            };

            println!("  step {:4} | t={:.3} Gyr | ρ+max/ρ̄={:.1} | Corr={:.3} | {}",
                     step, t_gyr, rho_plus_max_ratio, corr_delta, alert);
        }

        // Snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, t_gyr, box_size);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let (positions, velocities, signs) = sim.get_particles().unwrap();
    let (diff_pois, corr_delta, rho_plus_max_ratio, v_plus, v_minus) =
        compute_metrics(&positions, &velocities, &signs, box_size, N_CELLS);

    println!("\n================================================================");
    println!("  PHASE 2 COMPLETE");
    println!("================================================================");
    println!("  Final t: {:.3} Gyr", t_gyr);
    println!("  ρ+_max/ρ̄+: {:.2}", rho_plus_max_ratio);
    println!("  Diff/Pois: {:.4}", diff_pois);
    println!("  Corr(δ+,δ-): {:.4}", corr_delta);
    println!("  <v+> = {:.1} km/s, <v-> = {:.1} km/s", v_plus, v_minus);
    println!("  Runtime: {:.1}s ({:.1} min)", elapsed, elapsed / 60.0);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn compute_metrics(positions: &[f32], velocities: &[f32], signs: &[i8], box_size: f64, n_cells: usize) -> (f64, f64, f64, f64, f64) {
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

        let vmag = (velocities[i*3].powi(2) + velocities[i*3+1].powi(2) + velocities[i*3+2].powi(2)).sqrt() as f64 * 977.8;

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

    // Means
    let total_plus: u64 = n_plus_grid.iter().map(|&x| x as u64).sum();
    let total_minus: u64 = n_minus_grid.iter().map(|&x| x as u64).sum();
    let mean_plus = total_plus as f64 / n_cells_cubed as f64;
    let mean_minus = total_minus as f64 / n_cells_cubed as f64;

    // Diff/Pois
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

    let cov: f64 = delta_plus.iter().zip(delta_minus.iter()).map(|(dp, dm)| dp * dm).sum::<f64>() / n_cells_cubed as f64;
    let var_plus: f64 = delta_plus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let var_minus: f64 = delta_minus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let corr_delta = if var_plus > 0.0 && var_minus > 0.0 {
        cov / (var_plus.sqrt() * var_minus.sqrt())
    } else { 0.0 };

    // ρ+_max / ρ̄+
    let max_plus = *n_plus_grid.iter().max().unwrap_or(&0) as f64;
    let rho_plus_max_ratio = if mean_plus > 0.0 { max_plus / mean_plus } else { 1.0 };

    (diff_pois, corr_delta, rho_plus_max_ratio, v_plus, v_minus)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, t_gyr: f64, box_size: f64) {
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
    let _ = writer.write_all(&(t_gyr as f32).to_le_bytes());  // Using t_gyr instead of z

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
