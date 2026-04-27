//! VSL Production 12M with velocity cap — Janus z=4→z=0
//!
//! Fix for m- runaway: cap |v_minus| at 5000 km/s
//!
//! Parameters:
//!   N = 12M, Box = 100 Mpc, μ = 19
//!   c_ratio = 1/√19 = 0.2294
//!   v_max_minus = 5000 km/s

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const MPC_GYR_TO_KMS: f64 = 977.8;
const KMS_TO_MPC_GYR: f64 = 1.0 / 977.8;
const N_CELLS: usize = 64;

// Simulation parameters
const N_PARTICLES: usize = 12_000_000;
const BOX_SIZE: f64 = 100.0;
const MU: f64 = 19.0;
const N_STEPS: usize = 30000;
const DT: f64 = 0.0005;
const SNAP_INTERVAL: usize = 10;
const CSV_INTERVAL: usize = 50;
const CHECKPOINT_INTERVAL: usize = 5000;

// Janus parameters
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;

// Velocity cap for m- (km/s)
const V_MAX_MINUS_KMS: f64 = 5000.0;
const V_MAX_MINUS: f64 = V_MAX_MINUS_KMS * KMS_TO_MPC_GYR;  // ~5.1 Mpc/Gyr

// Emergency thresholds
const RHO_EMERGENCY: f64 = 1e8;
const V_RMS_MINUS_ALERT: f64 = 10000.0;  // km/s - warning threshold
const V_RMS_MINUS_STOP: f64 = 50000.0;   // km/s - stop if exceeded at step 50

#[cfg(feature = "cuda")]
fn main() {
    run_vsl_12m();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
}

#[cfg(feature = "cuda")]
fn run_vsl_12m() {
    let c_ratio = 1.0 / MU.sqrt();
    let c_ratio_sq = 1.0 / MU;

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    let output_dir = PathBuf::from("/app/output/janus_vsl_12M");
    let snap_dir = output_dir.join("snapshots");
    let checkpoint_dir = output_dir.join("checkpoints");

    fs::create_dir_all(&snap_dir).expect("Failed to create snapshots dir");
    fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoints dir");

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL 12M + VELOCITY CAP — Janus Bimetric z=4→z=0                ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc, z_init = {}", BOX_SIZE, Z_INIT);
    println!("║  μ = {} → c⁻/c⁺ = {:.4} → (c⁻/c⁺)² = {:.5}", MU, c_ratio, c_ratio_sq);
    println!("║  v_max_minus = {} km/s (velocity cap)", V_MAX_MINUS_KMS);
    println!("║  dt = {} Gyr × {} steps = {} Gyr", DT, N_STEPS, DT * N_STEPS as f64);
    println!("║  Snapshots: every {} steps → {} files", SNAP_INTERVAL, N_STEPS / SNAP_INTERVAL);
    println!("║  Output: {}", output_dir.display());
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("Initializing GPU with {} particles...", N_PARTICLES);
    let gpu_start = Instant::now();

    let mut gpu_sim = match GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("GPU init failed: {}", e);
            return;
        }
    };

    gpu_sim.set_c_ratio(c_ratio);
    gpu_sim.set_theta(0.7);

    println!("GPU init: {:.2}s, c_ratio_sq = {:.5}, theta = {:.1}",
             gpu_start.elapsed().as_secs_f64(),
             gpu_sim.get_c_ratio_sq(),
             gpu_sim.get_theta());

    let csv_path = output_dir.join("evolution.csv");
    let mut csv_file = BufWriter::new(File::create(&csv_path).expect("Failed to create CSV"));
    writeln!(csv_file, "step,t_Gyr,z,rho_plus_max,rho_minus_max,delta_max,v_rms_plus,v_rms_minus,v_mean_plus,v_mean_minus,segregation,n_capped,step_time_s").unwrap();

    // Save parameters
    let params_json = format!(r#"{{"n_particles":{},"n_positive":{},"n_negative":{},"box_size_mpc":{},"mu":{},"c_ratio":{},"eta":{},"z_init":{},"dt_gyr":{},"n_steps":{},"v_max_minus_kms":{},"theta":0.7}}"#,
        N_PARTICLES, n_positive, n_negative, BOX_SIZE, MU, c_ratio, ETA, Z_INIT, DT, N_STEPS, V_MAX_MINUS_KMS);
    fs::write(output_dir.join("parameters.json"), params_json).ok();

    let half_box = BOX_SIZE / 2.0;
    let cell_size = BOX_SIZE / N_CELLS as f64;

    println!("\n{:>6} | {:>6} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10}",
             "Step", "z", "ρ+_max", "ρ-_max", "v_rms+", "v_rms-", "Seg", "capped", "time");
    println!("{:-<105}", "");

    let sim_start = Instant::now();
    let mut last_step_time = Instant::now();
    let mut time_gyr = 0.0;

    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        // Get data from GPU
        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to get positions: {}", e); break; }
        };
        let mut vel = match gpu_sim.get_velocities() {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to get velocities: {}", e); break; }
        };
        let signs = gpu_sim.signs();
        let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

        // Apply velocity cap to m- particles
        let mut n_capped = 0usize;
        for i in 0..N_PARTICLES {
            if signs[i] < 0 {
                let vx = vel[i * 3];
                let vy = vel[i * 3 + 1];
                let vz = vel[i * 3 + 2];
                let v_mag = (vx * vx + vy * vy + vz * vz).sqrt();

                if v_mag > V_MAX_MINUS {
                    let scale = V_MAX_MINUS / v_mag;
                    vel[i * 3] *= scale;
                    vel[i * 3 + 1] *= scale;
                    vel[i * 3 + 2] *= scale;
                    n_capped += 1;
                }
            }
        }

        // Upload capped velocities back to GPU if any were modified
        if n_capped > 0 {
            if let Err(e) = gpu_sim.set_velocities(&vel) {
                eprintln!("Failed to set velocities: {}", e);
                break;
            }
        }

        // Compute metrics
        let (rho_plus_max, rho_minus_max, delta_max, v_rms_plus, v_rms_minus,
             v_mean_plus, v_mean_minus, segregation) =
            compute_metrics(&pos, &vel, &signs_i8, N_PARTICLES, half_box, cell_size);

        let step_time = last_step_time.elapsed().as_secs_f64();
        last_step_time = Instant::now();

        // CSV output
        if step % CSV_INTERVAL == 0 {
            writeln!(csv_file, "{},{:.6},{:.4},{:.0},{:.0},{:.4},{:.2},{:.2},{:.2},{:.2},{:.4},{},{:.3}",
                     step, time_gyr, z, rho_plus_max, rho_minus_max, delta_max,
                     v_rms_plus, v_rms_minus, v_mean_plus, v_mean_minus, segregation, n_capped, step_time).unwrap();
            csv_file.flush().unwrap();
        }

        // Progress output
        if step % 100 == 0 || step == N_STEPS {
            let elapsed = sim_start.elapsed().as_secs_f64();
            let eta_hours = if step > 0 {
                (N_STEPS - step) as f64 * elapsed / step as f64 / 3600.0
            } else { 0.0 };

            println!("{:>6} | {:>6.3} | {:>10.0} | {:>10.0} | {:>7.0} | {:>7.0} | {:>7.4} | {:>7} | {:>7.1}s (ETA {:.1}h)",
                     step, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, segregation, n_capped, elapsed, eta_hours);
        }

        // Step 50 validation
        if step == 50 {
            if v_rms_minus > V_RMS_MINUS_STOP {
                eprintln!("🚨 STOP at step 50: v_rms_minus = {:.0} km/s > {} km/s", v_rms_minus, V_RMS_MINUS_STOP);
                eprintln!("   Velocity cap not effective enough. Increase damping or reduce μ.");
                break;
            } else if v_rms_minus > V_RMS_MINUS_ALERT {
                println!("⚠️  WARNING step 50: v_rms_minus = {:.0} km/s > {} km/s (monitoring)", v_rms_minus, V_RMS_MINUS_ALERT);
            } else {
                println!("✓ Step 50 validation PASSED: v_rms_minus = {:.0} km/s < {} km/s", v_rms_minus, V_RMS_MINUS_ALERT);
            }
        }

        // Emergency stops
        if rho_plus_max > RHO_EMERGENCY {
            eprintln!("🚨 EMERGENCY STOP: ρ+_max = {:.0} > {}", rho_plus_max, RHO_EMERGENCY);
            break;
        }

        // Save snapshot
        if step % SNAP_INTERVAL == 0 {
            let snap_path = snap_dir.join(format!("snap_{:06}.bin", step));
            save_snapshot(&snap_path, &pos, &vel, &signs_i8, N_PARTICLES, BOX_SIZE as f32, step as u32, z as f32);
        }

        // Save checkpoint
        if step % CHECKPOINT_INTERVAL == 0 && step > 0 {
            let ckpt_path = checkpoint_dir.join(format!("checkpoint_{:06}.bin", step));
            save_snapshot(&ckpt_path, &pos, &vel, &signs_i8, N_PARTICLES, BOX_SIZE as f32, step as u32, z as f32);
            println!("  📁 Checkpoint saved: {}", ckpt_path.display());
        }

        if step >= N_STEPS {
            break;
        }

        // Hubble parameter
        let h = 0.07 / a.powf(1.5);

        // GPU step
        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0) {
            eprintln!("GPU step failed: {}", e);
            break;
        }

        a += a * h * DT;
        time_gyr += DT;
    }

    csv_file.flush().unwrap();

    let total_time = sim_start.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE: {:.1}h for {} steps ({:.2}s/step)", total_time / 3600.0, N_STEPS, total_time / N_STEPS as f64);
    println!("║  Output: {}", output_dir.display());
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn compute_metrics(pos: &[f64], vel: &[f64], signs: &[i8], n: usize,
                   half_box: f64, cell_size: f64) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let n_cells = N_CELLS;
    let mut counts_plus = vec![0u32; n_cells * n_cells * n_cells];
    let mut counts_minus = vec![0u32; n_cells * n_cells * n_cells];

    let mut v2_plus = 0.0f64;
    let mut v2_minus = 0.0f64;
    let mut v_plus = 0.0f64;
    let mut v_minus = 0.0f64;
    let mut n_plus_count = 0usize;
    let mut n_minus_count = 0usize;
    let mut com_plus = [0.0f64; 3];
    let mut com_minus = [0.0f64; 3];

    for i in 0..n {
        let px = pos[i * 3];
        let py = pos[i * 3 + 1];
        let pz = pos[i * 3 + 2];
        let vx = vel[i * 3];
        let vy = vel[i * 3 + 1];
        let vz = vel[i * 3 + 2];
        let v_mag = (vx * vx + vy * vy + vz * vz).sqrt();

        let ix = ((px + half_box) / cell_size).floor() as usize % n_cells;
        let iy = ((py + half_box) / cell_size).floor() as usize % n_cells;
        let iz = ((pz + half_box) / cell_size).floor() as usize % n_cells;
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        if signs[i] > 0 {
            counts_plus[idx] += 1;
            v2_plus += vx * vx + vy * vy + vz * vz;
            v_plus += v_mag;
            com_plus[0] += px;
            com_plus[1] += py;
            com_plus[2] += pz;
            n_plus_count += 1;
        } else {
            counts_minus[idx] += 1;
            v2_minus += vx * vx + vy * vy + vz * vz;
            v_minus += v_mag;
            com_minus[0] += px;
            com_minus[1] += py;
            com_minus[2] += pz;
            n_minus_count += 1;
        }
    }

    let rho_plus_max = *counts_plus.iter().max().unwrap_or(&0) as f64;
    let rho_minus_max = *counts_minus.iter().max().unwrap_or(&0) as f64;
    let mean_count = n as f64 / (n_cells * n_cells * n_cells) as f64;
    let delta_max = (rho_plus_max / mean_count - 1.0).max(rho_minus_max / mean_count - 1.0);

    let v_rms_plus = if n_plus_count > 0 {
        (v2_plus / n_plus_count as f64).sqrt() * MPC_GYR_TO_KMS
    } else { 0.0 };
    let v_rms_minus = if n_minus_count > 0 {
        (v2_minus / n_minus_count as f64).sqrt() * MPC_GYR_TO_KMS
    } else { 0.0 };
    let v_mean_plus = if n_plus_count > 0 {
        (v_plus / n_plus_count as f64) * MPC_GYR_TO_KMS
    } else { 0.0 };
    let v_mean_minus = if n_minus_count > 0 {
        (v_minus / n_minus_count as f64) * MPC_GYR_TO_KMS
    } else { 0.0 };

    let box_size = half_box * 2.0;
    if n_plus_count > 0 && n_minus_count > 0 {
        com_plus[0] /= n_plus_count as f64;
        com_plus[1] /= n_plus_count as f64;
        com_plus[2] /= n_plus_count as f64;
        com_minus[0] /= n_minus_count as f64;
        com_minus[1] /= n_minus_count as f64;
        com_minus[2] /= n_minus_count as f64;
    }

    let mut dx = com_plus[0] - com_minus[0];
    let mut dy = com_plus[1] - com_minus[1];
    let mut dz = com_plus[2] - com_minus[2];
    if dx > half_box { dx -= box_size; }
    if dx < -half_box { dx += box_size; }
    if dy > half_box { dy -= box_size; }
    if dy < -half_box { dy += box_size; }
    if dz > half_box { dz -= box_size; }
    if dz < -half_box { dz += box_size; }

    let com_dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let segregation = com_dist / (box_size / 2.0);

    (rho_plus_max, rho_minus_max, delta_max, v_rms_plus, v_rms_minus, v_mean_plus, v_mean_minus, segregation)
}

#[cfg(feature = "cuda")]
fn save_snapshot(path: &std::path::PathBuf, pos: &[f64], vel: &[f64], signs: &[i8],
                 n: usize, box_size: f32, step: u32, z: f32) {
    use std::io::Write;
    let mut file = BufWriter::with_capacity(64 * 1024 * 1024,
        File::create(path).expect("Failed to create snapshot"));

    file.write_all(&(n as u32).to_le_bytes()).unwrap();
    file.write_all(&box_size.to_le_bytes()).unwrap();
    file.write_all(&step.to_le_bytes()).unwrap();
    file.write_all(&z.to_le_bytes()).unwrap();

    for i in 0..n {
        file.write_all(&(pos[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(pos[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(pos[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3 + 2] as f32).to_le_bytes()).unwrap();
        file.write_all(&signs[i].to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
}
