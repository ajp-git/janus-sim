//! Champion run at 10M particles
//! Parameters will be set based on scan_boundary results

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::path::Path;
use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Champion parameters from scan_boundary
const ETA: f64 = 0.87;      // Best from scan v4
const LAMBDA_0: f64 = 1.0;  // Best from scan v4 (P=0.699)

const N_PARTICLES: usize = 10_000_000;
const BOX_SIZE: f64 = 1000.0;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 2000;  // More steps for higher resolution
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.3;  // Smaller softening for 10M
const SEED: u64 = 42;
const DT: f64 = 0.01;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Requires --features cuda");
        std::process::exit(1);
    }
    #[cfg(feature = "cuda")]
    run_champion();
}

#[cfg(feature = "cuda")]
fn run_champion() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHAMPION RUN — 10M Particles                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  η = {:.2}, λ₀ = {:.1} Mpc                                    ║", ETA, LAMBDA_0);
    println!("║  N = 10M, box = 1000 Mpc, z = 5 → 0                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let base_dir = Path::new("/app/output/champion_10m");
    fs::create_dir_all(base_dir).ok();
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).ok();

    // Time series CSV
    let ts_path = base_dir.join("time_series.csv");
    let mut ts_file = BufWriter::new(File::create(&ts_path).unwrap());
    writeln!(ts_file, "step,z,a,purity,ke_ratio").unwrap();

    let start = Instant::now();

    // Generate ICs
    println!("Generating 10M particle ICs...");
    let (pos_data, vel_data, signs_data) = generate_lcdm_ics(N_PARTICLES, BOX_SIZE, Z_INIT, SEED);

    let n_pos = signs_data.iter().filter(|&&s| s > 0).count();
    let n_neg = signs_data.iter().filter(|&&s| s < 0).count();
    println!("  N+ = {}, N- = {}", n_pos, n_neg);

    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_data, vel_data, signs_data, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(LAMBDA_0);

    let ke0 = sim.kinetic_energy().unwrap_or(1.0).max(1e-20);

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    // Snapshots every 5 steps for smooth animation (400 frames)
    let snapshot_interval = 5;
    let mut last_snap_purity = 0.0;

    println!("Starting evolution...");
    for step in 1..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };

        sim.set_current_z(z);
        sim.step_dkd(DT, h, dtau_per_dt).expect("Step failed");

        // Write time series every 10 steps
        if step % 10 == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let ke_ratio = ke / ke0;
            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4e}", step, z, a, purity, ke_ratio).unwrap();
        }

        // Log and save at snapshot interval
        if step % snapshot_interval == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            last_snap_purity = purity;
            println!("  step {:4} | z={:.3} | P={:.4} <<<", step, z, purity);
            save_snapshot(&sim, &snap_dir, step, z);
        } else if step % 100 == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            println!("  step {:4} | z={:.2} | P={:.4}", step, z, purity);
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64() / 60.0;
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHAMPION 10M COMPLETE                                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  P(z=0) = {:.4}                                              ║", last_snap_purity);
    println!("║  Time: {:.1} min                                              ║", elapsed);
    println!("║  Snapshots: {} frames                                        ║", STEPS / snapshot_interval);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn generate_lcdm_ics(n: usize, box_size: f64, z_init: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    use std::f64::consts::PI;

    let n_grid = (n as f64 / 2.0).powf(1.0/3.0).ceil() as usize;
    let cell = box_size / n_grid as f64;
    let box_half = box_size / 2.0;

    let h = 0.7;
    let omega_m = 0.3;
    let n_s = 0.965;

    let theta = 2.725 / 2.7;
    let k_eq = 0.0746 * omega_m * h * h / (theta * theta);

    let a = 1.0 / (1.0 + z_init);
    let omega_m_z = omega_m / (omega_m + (1.0 - omega_m) * a.powi(3));
    let d_z = a * (omega_m_z.powf(4.0/7.0) - (1.0 - omega_m_z) +
              (1.0 + omega_m_z/2.0) * (1.0 + (1.0-omega_m_z)/70.0)).powf(-1.0);

    let f_omega = omega_m.powf(0.55);
    let h0_gyr = 0.0715;
    let e_z = ((omega_m * (1.0 + z_init).powi(3)) + (1.0 - omega_m)).sqrt();
    let h_z = h0_gyr * e_z;
    let vel_factor = h_z * f_omega;
    let disp_scale = 25.0 * d_z;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut pos_data = Vec::with_capacity(n * 3);
    let mut vel_data = Vec::with_capacity(n * 3);
    let mut signs_data = Vec::with_capacity(n);

    let mut count = 0;
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                if count >= n { break; }

                let x0 = (ix as f64 + 0.5) * cell - box_half;
                let y0 = (iy as f64 + 0.5) * cell - box_half;
                let z0 = (iz as f64 + 0.5) * cell - box_half;

                let k = 2.0 * PI / box_size * ((ix*ix + iy*iy + iz*iz) as f64).sqrt().max(1.0);
                let pk = k.powf(n_s) / (1.0 + (k/k_eq).powi(2)).powf(2.0);
                let amp = (pk.sqrt() * disp_scale * cell / box_size).min(cell * 0.5);

                let psi_x: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_y: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_z: f64 = rng.gen::<f64>() * 2.0 - 1.0;

                let dx = psi_x * amp;
                let dy = psi_y * amp;
                let dz = psi_z * amp;

                let x = x0 + dx;
                let y = y0 + dy;
                let z = z0 + dz;

                let vx = dx * vel_factor;
                let vy = dy * vel_factor;
                let vz = dz * vel_factor;

                let sign: i8 = if rng.gen::<bool>() { 1 } else { -1 };

                pos_data.extend_from_slice(&[x as f32, y as f32, z as f32]);
                vel_data.extend_from_slice(&[vx as f32, vy as f32, vz as f32]);
                signs_data.push(sign);

                count += 1;
            }
        }
    }

    while count < n {
        let x: f64 = rng.gen::<f64>() * box_size - box_half;
        let y: f64 = rng.gen::<f64>() * box_size - box_half;
        let z: f64 = rng.gen::<f64>() * box_size - box_half;
        let sign: i8 = if rng.gen::<bool>() { 1 } else { -1 };

        pos_data.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel_data.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs_data.push(sign);
        count += 1;
    }

    (pos_data, vel_data, signs_data)
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, z: f64) {
    use std::io::BufWriter;

    let (positions, _, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let snap_path = path.join(format!("snap_{:05}_z{:.2}.bin", step, z));

    let file = match File::create(&snap_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(1000.0f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

    for i in 0..n {
        let _ = writer.write_all(&positions[i*3].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+1].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+2].to_le_bytes());
        let _ = writer.write_all(&(signs[i] as i8).to_le_bytes());
    }
}
