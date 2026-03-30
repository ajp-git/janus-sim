//! TreePM Test: PM 256³, r_cut=20 Mpc
//!
//! Test if TreePM with fine PM grid and small r_cut eliminates k=8 spike.
//! - Box: 500 Mpc
//! - N: 2M particles
//! - PM grid: 256³ (hardcoded in nbody_gpu_twopass.rs)
//! - r_cut: 20 Mpc (Box/25) — well below λ=62.5 Mpc
//! - μ=8, λ=0
//! - 200 steps

use rand::prelude::*;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const MU: f64 = 8.0;
const N_TOTAL: usize = 2_000_000;
const BOX_SIZE: f64 = 500.0;
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.005;
const STEPS: usize = 200;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const ETA: f64 = 1.0;
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 25;

// Key parameter: r_cut = 20 Mpc (Box/25)
// PM handles λ > 20 Mpc, BH handles λ < 20 Mpc
// k=8 mode (λ=62.5 Mpc) should be fully PM-dominated
const R_CUT: f64 = 20.0;

#[cfg(feature = "cuda")]
fn main() {
    println!("========================================================");
    println!("  TreePM TEST: PM 256³, r_cut=20 Mpc");
    println!("========================================================");
    println!("  Testing if small r_cut eliminates k=8 octree resonance");
    println!("========================================================");

    let n_positive = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_negative = N_TOTAL - n_positive;

    println!("  N_total = {} (2M)", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", n_negative as f64 / n_positive as f64);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  PM Grid = 256³");
    println!("  r_cut = {} Mpc (Box/{})", R_CUT, BOX_SIZE / R_CUT);
    println!("  k=8 wavelength = {:.1} Mpc (should be PM-dominated)", BOX_SIZE / 8.0);
    println!("  λ = 0 (pure anti-Newton)");
    println!("========================================================");
    println!();

    // Generate uniform random ICs
    println!("Generating uniform random ICs...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let half_box = BOX_SIZE / 2.0;

    let mut pos_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut signs_i8: Vec<i8> = Vec::with_capacity(N_TOTAL);

    for _ in 0..n_positive {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(1);
    }

    for _ in 0..n_negative {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(-1);
    }

    println!("  Generated {} particles", N_TOTAL);

    // Setup output
    let base_dir = std::path::Path::new("/app/output/test_treepm_rcut20");
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
    println!("Starting TreePM evolution (r_cut={} Mpc)...", R_CUT);

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
            // Use TreePM with r_cut = 20 Mpc
            sim.step_treepm_gpu_morton(DT, R_CUT, h, dtau_per_dt)
                .expect("TreePM step failed");
        }

        if step % SNAPSHOT_INTERVAL == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);

            writeln!(ts_file, "{},{:.4},{:.6},{:.4}", step, z, a, purity).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };

            println!("  step {:4} | z={:.2} | P={:.3} | {:.1}s ({:.2} step/s)",
                     step, z, purity, elapsed, rate);

            save_snapshot(&sim, &snap_dir, step, z);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    println!();
    println!("========================================================");
    println!("  TreePM TEST COMPLETE");
    println!("========================================================");
    println!("  {} steps in {:.1}s ({:.2} step/s)", STEPS, elapsed, STEPS as f64 / elapsed);
    println!("  Snapshots: {:?}", snap_dir);
    println!("========================================================");
    println!();
    println!("Run power spectrum analysis on snap_00200.bin to check k=8!");
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
