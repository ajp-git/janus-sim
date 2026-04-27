//! Test N-body with Janus Parametric Expansion
//!
//! Uses the exact parametric solution a⁺(μ) = α² cosh²(μ)
//! instead of coupled Friedmann equations.

use rand::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::janus_expansion::JanusExpansion;

const DEFAULT_N: usize = 2_000_000;
const DEFAULT_BOX: f64 = 1000.0;
const Z_INIT: f64 = 4.0;  // Must be < z_max ≈ 4.5
const DT: f64 = 0.005;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 100;
const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 20.0;
const N_CELLS: usize = 32;

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mu: f64 = args.iter()
        .position(|x| x == "--mu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(35.0);

    let n_total: usize = args.iter()
        .position(|x| x == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);

    let box_size: f64 = args.iter()
        .position(|x| x == "--box")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BOX);

    let run_name = format!("janus_expansion_mu{}_{}M_{}Mpc",
                           mu as u32, n_total / 1_000_000, box_size as u32);

    println!("================================================================");
    println!("  Test Janus Parametric Expansion");
    println!("================================================================");
    println!("  μ = {}", mu);
    println!("  N = {}M", n_total / 1_000_000);
    println!("  Box = {} Mpc", box_size);
    println!("  z_init = {}", Z_INIT);
    println!("================================================================");

    // Initialize Janus expansion
    println!("\nInitializing Janus cosmology...");
    let expansion = JanusExpansion::new(Z_INIT, 5000);

    // Export cosmology table
    let base_dir = std::path::Path::new("/app/output").join(&run_name);
    fs::create_dir_all(&base_dir).expect("Failed to create output dir");
    expansion.export_csv(base_dir.join("janus_cosmology.csv").to_str().unwrap())
        .expect("Failed to export cosmology");

    // Particle counts
    let n_positive = (n_total as f64 / (1.0 + mu)) as usize;
    let n_negative = n_total - n_positive;

    println!("\n  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / n_total as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / n_total as f64);

    // Generate ICs
    println!("\nGenerating uniform random ICs...");
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

    // Setup output
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create snapshots dir");

    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,t_gyr,z,a,H_gyr,P,void_frac,wall_frac").unwrap();

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, box_size
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);

    // JANUS DENSITY CORRECTION
    // ρ+ = Ω_b × ρ_crit (baryons only, no dark matter)
    // ρ_total = ρ+ × (1 + μ)
    // mass_factor = G × ρ_total × V_box / N
    {
        let g_cosmo = 4.499e-15;  // Mpc³/(M_sun·Gyr²)
        let rho_crit = 1.36e11;   // M_sun/Mpc³ (H₀=70)
        let omega_b = 0.05;       // Baryonic fraction only
        let rho_plus = omega_b * rho_crit;
        let rho_total = rho_plus * (1.0 + mu);
        let m_total = rho_total * box_size.powi(3);
        let mass_factor_janus = g_cosmo * m_total / n_total as f64;
        sim.set_mass_factor(mass_factor_janus);
        println!("  ρ+ = {:.2e} M☉/Mpc³ (Ω_b = {})", rho_plus, omega_b);
        println!("  ρ_total = ρ+(1+μ) = {:.2e} M☉/Mpc³ = {:.2} ρ_crit", rho_total, rho_total/rho_crit);
    }

    // Time mapping: map simulation steps to physical time
    let t_start = expansion.t_start;
    let t_end = expansion.t_end;
    let dt_gyr = (t_end - t_start) / STEPS as f64;

    let start = Instant::now();

    println!("\nStarting evolution with Janus expansion...\n");

    for step in 0..=STEPS {
        // Get cosmology at current time
        let t_current = t_start + step as f64 * dt_gyr;
        let state = expansion.at_time(t_current);

        // Hubble friction: dv/dt = -H*v
        // In the integrator: friction = -H * v * dtau_per_dt
        // For Janus: dtau_per_dt converts code time to conformal time
        // Here we use physical time directly, so dtau_per_dt = 1
        let hubble = state.h_plus;
        let dtau_per_dt = 1.0;

        if step > 0 {
            sim.set_current_z(state.z);
            sim.step_treepm_gpu(DT, R_CUT, hubble, dtau_per_dt)
                .expect("TreePM step failed");
        }

        // Logging
        if step % CSV_INTERVAL == 0 {
            let (positions, _, signs) = sim.get_particles().unwrap();
            let purity = compute_purity(&positions, &signs, box_size, N_CELLS);
            let (void_frac, wall_frac) = compute_void_wall_fractions(&positions, &signs, box_size, N_CELLS);

            writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.4},{:.4},{:.4}",
                     step, t_current, state.z, state.a_plus, hubble, purity, void_frac, wall_frac).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_min = if rate > 0.0 { (STEPS - step) as f64 / rate / 60.0 } else { 0.0 };

            println!("  step {:4} | t={:.2}Gyr | z={:.2} | a={:.3} | H={:.4} | void={:.1}% | ETA {:.0}min",
                     step, t_current, state.z, state.a_plus, hubble, void_frac * 100.0, eta_min);
        }

        // Snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, state.z, box_size);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let final_state = expansion.at_time(t_end);

    println!("\n================================================================");
    println!("  TEST COMPLETE");
    println!("================================================================");
    println!("  Final z: {:.4}", final_state.z);
    println!("  Final a: {:.4}", final_state.a_plus);
    println!("  Final H: {:.4} Gyr⁻¹ = {:.1} km/s/Mpc",
             final_state.h_plus, final_state.h_plus / 1.0227e-3);
    println!("  Runtime: {:.1}s ({:.1} min)", elapsed, elapsed / 60.0);
    println!("  Output: {:?}", base_dir);
    println!("================================================================");
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
            if nm / total > 0.90 { void_cells += 1; }
            if np / total > 0.90 { wall_cells += 1; }
        }
    }

    let void_frac = if occupied_cells > 0 { void_cells as f64 / occupied_cells as f64 } else { 0.0 };
    let wall_frac = if occupied_cells > 0 { wall_cells as f64 / occupied_cells as f64 } else { 0.0 };

    (void_frac, wall_frac)
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
