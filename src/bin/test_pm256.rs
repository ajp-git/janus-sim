//! Test PM grid 256³ vs 128³ to diagnose k=8 anisotropy artifact
//! If k=8 spike disappears with 256³ → confirms PM grid artifact

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use rand::Rng;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::io::{BufWriter, Write};
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
const N_TOTAL: usize = 2_000_000;
#[cfg(feature = "cuda")]
const MU: f64 = 8.0;
#[cfg(feature = "cuda")]
const BOX_SIZE: f64 = 500.0;
#[cfg(feature = "cuda")]
const DT: f64 = 0.005;
#[cfg(feature = "cuda")]
const STEPS: usize = 130;
#[cfg(feature = "cuda")]
const SNAPSHOT_INTERVAL: usize = 25;
#[cfg(feature = "cuda")]
const R_CUT: f64 = BOX_SIZE / 16.0;

#[cfg(feature = "cuda")]
fn main() {
    println!("========================================================");
    println!("  PM GRID 256³ TEST — Diagnosing k=8 anisotropy");
    println!("========================================================");
    println!("  If k=8 spike persists → not PM grid artifact");
    println!("  If k=8 spike disappears → PM grid 128³ was the cause");
    println!("========================================================");

    let n_plus = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_minus = N_TOTAL - n_plus;

    println!("  N_total = {} (2M)", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_plus, 100.0 * n_plus as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_minus, 100.0 * n_minus as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", MU);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  PM Grid = 256³ (testing)");
    println!("  TreePM: r_cut = {:.1} Mpc", R_CUT);
    println!("========================================================");

    // Create output directory
    let output_dir = PathBuf::from("/app/output/test_pm256");
    let snap_dir = output_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    // Generate uniform random ICs
    println!("\nGenerating uniform random ICs...");
    let mut rng = rand::rng();
    let half_box = BOX_SIZE / 2.0;

    let mut pos_f32 = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32 = Vec::with_capacity(N_TOTAL * 3);
    let mut signs = Vec::with_capacity(N_TOTAL);

    for _ in 0..n_plus {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        signs.push(1i8);
    }

    for _ in 0..n_minus {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        signs.push(-1i8);
    }

    println!("  Generated {} particles", N_TOTAL);

    // Initialize GPU simulation with 256³ PM grid
    println!("Initializing GPU simulation...");

    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32,
        vel_f32,
        signs,
        BOX_SIZE,
    ).expect("Failed to create simulation");

    sim.set_theta(0.7);
    sim.set_lambda_0(0.0);

    // PM grid is now 256³ (hardcoded in nbody_gpu_twopass.rs)
    println!("  PM grid = 256³ (hardcoded)");

    println!("  GPU initialized, starting evolution...\n");

    let z_init = 5.0;
    let a_init = 1.0 / (1.0 + z_init);
    let start = Instant::now();

    for step in 0..=STEPS {
        let t_frac = step as f64 / STEPS as f64;
        let a = a_init + (1.0 - a_init) * t_frac;
        let z = 1.0 / a - 1.0;

        let h0 = 70.0;
        let omega_m = 0.3;
        let h = h0 * (omega_m / (a * a * a) + (1.0 - omega_m)).sqrt();
        let dtau_per_dt = 1.0 / (a * a);

        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, z);

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };

            println!("Step {:4} | z={:.2} | snapshot saved | {:.1}s ({:.2} step/s)",
                     step, z, elapsed, rate);
        }

        if step < STEPS {
            sim.step_treepm_gpu(DT, R_CUT, h, dtau_per_dt)
                .expect("TreePM step failed");
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    println!("\n========================================================");
    println!("  COMPLETE: {} steps in {:.1}s ({:.2} step/s)",
             STEPS, total_time, STEPS as f64 / total_time);
    println!("  Snapshots in: {:?}", snap_dir);
    println!("========================================================");
    println!("\nAnalyze snap_00125.bin to check k=8 anisotropy!");
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, snap_dir: &PathBuf, step: usize, z: f64) {
    use std::fs::File;

    let (positions, velocities, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let snap_path = snap_dir.join(format!("snap_{:05}.bin", step));

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
