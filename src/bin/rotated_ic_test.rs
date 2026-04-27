//! Test with 45° rotated ICs to determine if grid pattern is physical or numerical
//! If pattern rotates with ICs → Janus physics
//! If pattern stays on x,y,z axes → numerical artifact

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
const STEPS: usize = 200;
#[cfg(feature = "cuda")]
const SNAPSHOT_INTERVAL: usize = 20;
#[cfg(feature = "cuda")]
const R_CUT: f64 = BOX_SIZE / 16.0;  // ~31 Mpc, TreePM parameter

#[cfg(feature = "cuda")]
fn main() {
    println!("========================================================");
    println!("  ROTATED IC TEST — 45° rotation to test isotropy");
    println!("========================================================");
    println!("  If grid aligns with rotated axes → PHYSICAL");
    println!("  If grid stays on x,y,z → NUMERICAL ARTIFACT");
    println!("========================================================");

    let n_plus = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_minus = N_TOTAL - n_plus;

    println!("  N_total = {} (2M)", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_plus, 100.0 * n_plus as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_minus, 100.0 * n_minus as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", MU);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  TreePM: r_cut = {:.1} Mpc", R_CUT);
    println!("========================================================");

    // Create output directory
    let output_dir = PathBuf::from("/app/output/rotated_ic_test");
    let snap_dir = output_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    // Generate rotated ICs
    println!("\nGenerating 45° ROTATED uniform random ICs...");
    let mut rng = rand::rng();
    let half_box = BOX_SIZE / 2.0;

    // 45° rotation: rotate around z-axis then y-axis
    fn rotate_45(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let sin45 = std::f64::consts::FRAC_1_SQRT_2;

        // First rotate around z by 45°
        let x1 = x * cos45 - y * sin45;
        let y1 = x * sin45 + y * cos45;
        let z1 = z;

        // Then rotate around y by 45°
        let x2 = x1 * cos45 + z1 * sin45;
        let y2 = y1;
        let z2 = -x1 * sin45 + z1 * cos45;

        (x2, y2, z2)
    }

    let mut pos_f32 = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32 = Vec::with_capacity(N_TOTAL * 3);
    let mut signs = Vec::with_capacity(N_TOTAL);

    // Generate m+ particles
    for _ in 0..n_plus {
        // Generate in rotated frame, then transform back
        let x_rot = rng.random::<f64>() * BOX_SIZE - half_box;
        let y_rot = rng.random::<f64>() * BOX_SIZE - half_box;
        let z_rot = rng.random::<f64>() * BOX_SIZE - half_box;

        // Apply rotation to get actual position
        let (x, y, z) = rotate_45(x_rot, y_rot, z_rot);

        // Wrap periodically
        let wrap = |v: f64| -> f64 {
            let mut w = v;
            while w < -half_box { w += BOX_SIZE; }
            while w >= half_box { w -= BOX_SIZE; }
            w
        };

        pos_f32.push(wrap(x) as f32);
        pos_f32.push(wrap(y) as f32);
        pos_f32.push(wrap(z) as f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        signs.push(1i8);
    }

    // Generate m- particles
    for _ in 0..n_minus {
        let x_rot = rng.random::<f64>() * BOX_SIZE - half_box;
        let y_rot = rng.random::<f64>() * BOX_SIZE - half_box;
        let z_rot = rng.random::<f64>() * BOX_SIZE - half_box;

        let (x, y, z) = rotate_45(x_rot, y_rot, z_rot);

        let wrap = |v: f64| -> f64 {
            let mut w = v;
            while w < -half_box { w += BOX_SIZE; }
            while w >= half_box { w -= BOX_SIZE; }
            w
        };

        pos_f32.push(wrap(x) as f32);
        pos_f32.push(wrap(y) as f32);
        pos_f32.push(wrap(z) as f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        vel_f32.push(0.0f32);
        signs.push(-1i8);
    }

    println!("  Generated {} particles with 45° rotated frame", N_TOTAL);

    // Initialize GPU simulation
    println!("Initializing GPU simulation with TreePM...");

    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32,
        vel_f32,
        signs,
        BOX_SIZE,
    ).expect("Failed to create simulation");

    // Set theta (BH opening angle) and lambda (pure anti-Newton)
    sim.set_theta(0.7);
    sim.set_lambda_0(0.0);

    println!("  GPU initialized, starting evolution...\n");

    // Evolution parameters
    let z_init = 5.0;
    let a_init = 1.0 / (1.0 + z_init);

    // Time tracking
    let start = Instant::now();

    for step in 0..=STEPS {
        // Calculate cosmological parameters
        let t_frac = step as f64 / STEPS as f64;
        let a = a_init + (1.0 - a_init) * t_frac;
        let z = 1.0 / a - 1.0;

        // Hubble parameter (simplified)
        let h0 = 70.0;
        let omega_m = 0.3;
        let h = h0 * (omega_m / (a * a * a) + (1.0 - omega_m)).sqrt();
        let dtau_per_dt = 1.0 / (a * a);

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, &snap_dir, step, z);

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };

            println!("Step {:4} | z={:.2} | snapshot saved | {:.1}s ({:.2} step/s)",
                     step, z, elapsed, rate);
        }

        // Advance simulation (skip on last step)
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
    println!("\nNow analyze with 3D FFT to check if k=8 mode rotated!");
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

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda");
    std::process::exit(1);
}
