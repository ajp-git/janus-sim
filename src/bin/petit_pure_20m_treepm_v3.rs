//! Petit Pure 20M — TreePM Production Run
//!
//! Production simulation with corrected TreePM (erfc splitting).
//! Validated on 2M run: P=0.989, k=8 spike eliminated (1.0×).
//!
//! - N = 20M (μ=8: 2.22M m+, 17.78M m-)
//! - Box = 500 Mpc
//! - TreePM: PM 256³, r_cut = 20 Mpc
//! - λ = 0 (pure anti-Newton 1/r²)
//! - 2000 steps, snapshots every 50

use rand::prelude::*;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
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
const CSV_INTERVAL: usize = 5;
const R_CUT: f64 = 20.0;

#[cfg(feature = "cuda")]
fn main() {
    println!("================================================================");
    println!("  Petit Pure 20M — TreePM Production Run");
    println!("================================================================");
    println!("  Corrected TreePM with erfc splitting (k=8 artifact eliminated)");
    println!("================================================================");

    let n_positive = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_negative = N_TOTAL - n_positive;

    println!("  N_total = {} (20M)", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", n_negative as f64 / n_positive as f64);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  PM Grid = 256³");
    println!("  r_cut = {} Mpc (TreePM)", R_CUT);
    println!("  λ = 0 (pure anti-Newton)");
    println!("  Steps = {}", STEPS);
    println!("================================================================");
    println!();

    // Generate uniform random ICs
    println!("Generating uniform random ICs for 20M particles...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let half_box = BOX_SIZE / 2.0;

    let mut pos_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut signs_i8: Vec<i8> = Vec::with_capacity(N_TOTAL);

    for i in 0..n_positive {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(1);
        if i % 1_000_000 == 0 && i > 0 {
            println!("  Generated {}M particles...", i / 1_000_000);
        }
    }

    for i in 0..n_negative {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(-1);
        if (n_positive + i) % 5_000_000 == 0 {
            println!("  Generated {}M particles...", (n_positive + i) / 1_000_000);
        }
    }

    println!("  Generated {} particles total", N_TOTAL);

    // Setup output
    let base_dir = std::path::Path::new("/app/output/petit_pure_20m_treepm_v3");
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,z,a,P").unwrap();

    // Initialize simulation
    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);

    println!("  θ = {} (Barnes-Hut opening angle)", THETA);
    println!("  λ₀ = 0.0 (pure anti-Newton)");

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let start = Instant::now();

    println!();
    println!("Starting 20M TreePM evolution (r_cut={} Mpc)...", R_CUT);
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

        // CSV logging every 10 steps
        if step % CSV_INTERVAL == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            writeln!(ts_file, "{},{:.4},{:.6},{:.4}", step, z, a, purity).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_sec = if rate > 0.0 { (STEPS - step) as f64 / rate } else { 0.0 };
            let eta_min = eta_sec / 60.0;

            println!("  step {:4} | z={:.2} | a={:.3} | P={:.3} | {:.1}s ({:.2} step/s) ETA {:.0}min",
                     step, z, a, purity, elapsed, rate, eta_min);
        }

        // Snapshots every 50 steps
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, z);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let final_purity = sim.local_purity(32).unwrap_or(0.0);

    println!();
    println!("================================================================");
    println!("  20M TreePM PRODUCTION RUN COMPLETE");
    println!("================================================================");
    println!("  Final Purity: {:.4}", final_purity);
    println!("  Runtime: {:.1}s ({:.1} min)", elapsed, elapsed / 60.0);
    println!("  Rate: {:.2} step/s", STEPS as f64 / elapsed);
    println!("  Output: {:?}", base_dir);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
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

    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(BOX_SIZE as f32).to_le_bytes());
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
