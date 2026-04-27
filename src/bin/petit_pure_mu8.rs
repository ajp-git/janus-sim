//! Petit Pure Run — μ=8, λ=0, Uniform Random ICs
//!
//! Based on original Petit papers (2014-2024):
//! - μ = ρ⁻/ρ⁺ = 8 (N⁻/N⁺ = 8)
//! - Pure anti-Newton 1/r² (λ=0, no Yukawa)
//! - Uniform random ICs (no Zeldovich)

use rand::prelude::*;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const MU: f64 = 8.0;              // ρ⁻/ρ⁺ ratio
const N_TOTAL: usize = 2_000_000;
const BOX_SIZE: f64 = 500.0;      // Mpc
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.005;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const ETA: f64 = 1.0;             // For cosmology (doesn't affect forces with λ=0)
const SEED: u64 = 42;

#[cfg(feature = "cuda")]
fn main() {
    println!("========================================================");
    println!("  PETIT PURE — μ={}, λ=0, Uniform Random ICs", MU);
    println!("========================================================");

    // N⁻/N⁺ = μ, so N⁺ = N_total/(1+μ)
    let n_positive = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_negative = N_TOTAL - n_positive;

    println!("  N_total = {}", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", n_negative as f64 / n_positive as f64);
    println!("  Box = {} Mpc, z = {} -> 0", BOX_SIZE, Z_INIT);
    println!("  λ = 0 (pure anti-Newton 1/r²)");
    println!("========================================================");
    println!();

    // Generate uniform random ICs
    println!("Generating uniform random ICs...");
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
    let base_dir = std::path::Path::new("/app/output/petit_pure_mu8");
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,z,a,P,void_frac,wall_frac").unwrap();

    // Initialize simulation
    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);  // CRITICAL: Pure 1/r², no Yukawa!

    println!("  λ₀ = 0.0 (pure anti-Newton)");

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let snapshot_interval = 5;
    let start = Instant::now();

    println!();
    println!("Starting evolution...");

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
            sim.step_dkd(DT, h, dtau_per_dt).expect("Step failed");
        }

        // Snapshot + metrics
        if step % snapshot_interval == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            let (void_frac, wall_frac) = compute_void_wall_fractions(&sim);

            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4},{:.4}",
                     step, z, a, purity, void_frac, wall_frac).unwrap();

            println!("  step {:4} | z={:.3} | P={:.3} | void={:.1}% | wall={:.1}%",
                     step, z, purity, void_frac * 100.0, wall_frac * 100.0);

            save_snapshot(&sim, &snap_dir, step, z);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64() / 60.0;
    let final_purity = sim.local_purity(32).unwrap_or(0.0);
    let (void_frac, wall_frac) = compute_void_wall_fractions(&sim);

    println!();
    println!("========================================================");
    println!("  PETIT PURE μ=8 COMPLETE");
    println!("========================================================");
    println!("  P(z=0) = {:.4}", final_purity);
    println!("  Void fraction (>95% m⁻) = {:.1}%", void_frac * 100.0);
    println!("  Wall fraction (>95% m⁺) = {:.1}%", wall_frac * 100.0);
    println!("  Time: {:.1} min", elapsed);
    println!("========================================================");
}

#[cfg(feature = "cuda")]
fn compute_void_wall_fractions(sim: &GpuNBodyTwoPass) -> (f64, f64) {
    let (positions, _, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return (0.0, 0.0),
    };

    let n_cells = 32;
    let cell_size = BOX_SIZE / n_cells as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut n_plus = vec![0usize; n_cells * n_cells * n_cells];
    let mut n_minus = vec![0usize; n_cells * n_cells * n_cells];

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

    let mut void_cells = 0;
    let mut wall_cells = 0;
    let mut total_cells = 0;

    for idx in 0..(n_cells * n_cells * n_cells) {
        let total = n_plus[idx] + n_minus[idx];
        if total > 10 {  // Minimum particles for statistics
            total_cells += 1;
            // Void: >95% m⁻
            if n_minus[idx] as f64 / total as f64 > 0.95 { void_cells += 1; }
            // Wall: >95% m⁺
            if n_plus[idx] as f64 / total as f64 > 0.95 { wall_cells += 1; }
        }
    }

    let void_frac = if total_cells > 0 { void_cells as f64 / total_cells as f64 } else { 0.0 };
    let wall_frac = if total_cells > 0 { wall_cells as f64 / total_cells as f64 } else { 0.0 };

    (void_frac, wall_frac)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, z: f64) {
    let (positions, _, signs) = match sim.get_particles() {
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
    let _ = writer.write_all(&(BOX_SIZE as f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

    for i in 0..n {
        let _ = writer.write_all(&positions[i*3].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+1].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+2].to_le_bytes());
        let _ = writer.write_all(&(signs[i] as i8).to_le_bytes());
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda");
    std::process::exit(1);
}
