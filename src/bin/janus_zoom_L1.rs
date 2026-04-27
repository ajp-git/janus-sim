//! JANUS ZOOM-IN L1 — High-resolution cluster simulation
//!
//! Extracts r < 50 Mpc region from snap_04550.bin (z=0.459)
//! Subdivides m+ particles in r < 8 Mpc by factor 10
//! Uses GPU Barnes-Hut for N-body evolution

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use std::fs::{self, File};
use std::io::{BufWriter, Write, Read};
use std::time::Instant;
use rand::prelude::*;

// Cosmology
const ETA: f64 = 1.045;

// Zoom parameters
const CENTER: [f64; 3] = [-5.329, 11.171, -39.571];
const R_HR: f64 = 8.0;
const R_EXTRACT: f64 = 50.0;
const N_SUB: usize = 10;
const L_BOX: f64 = 500.0;
const L_ZOOM: f64 = 120.0;

// Simulation parameters
const DT: f64 = 0.0005;
const N_STEPS: usize = 18500;
const THETA: f64 = 0.7;
const EPSILON: f64 = 0.03;

// Output
const SNAPSHOT_INTERVAL: usize = 10;
const CSV_INTERVAL: usize = 5;
const OUTPUT_DIR: &str = "/app/output/janus_zoom_L1";

fn periodic_dist_orig(dx: f64, dy: f64, dz: f64) -> f64 {
    let dx = dx - L_BOX * (dx / L_BOX).round();
    let dy = dy - L_BOX * (dy / L_BOX).round();
    let dz = dz - L_BOX * (dz / L_BOX).round();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn read_source_snapshot(path: &str) -> (Vec<f64>, Vec<f64>, Vec<i32>, f64, f64) {
    let mut file = File::open(path).expect("Cannot open source snapshot");
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf8).unwrap();
    let n = u64::from_le_bytes(buf8) as usize;

    file.read_exact(&mut buf8).unwrap();
    let a = f64::from_le_bytes(buf8);

    file.read_exact(&mut buf8).unwrap();
    let t = f64::from_le_bytes(buf8);

    let mut pos_buf = vec![0u8; n * 3 * 4];
    file.read_exact(&mut pos_buf).unwrap();

    let mut vel_buf = vec![0u8; n * 3 * 4];
    file.read_exact(&mut vel_buf).unwrap();

    let mut signs_buf = vec![0u8; n];
    file.read_exact(&mut signs_buf).unwrap();

    // Parse as flat vectors
    let mut positions = Vec::with_capacity(n * 3);
    let mut velocities = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for i in 0..n {
        let px = f32::from_le_bytes([pos_buf[i*12], pos_buf[i*12+1], pos_buf[i*12+2], pos_buf[i*12+3]]) as f64;
        let py = f32::from_le_bytes([pos_buf[i*12+4], pos_buf[i*12+5], pos_buf[i*12+6], pos_buf[i*12+7]]) as f64;
        let pz = f32::from_le_bytes([pos_buf[i*12+8], pos_buf[i*12+9], pos_buf[i*12+10], pos_buf[i*12+11]]) as f64;

        let vx = f32::from_le_bytes([vel_buf[i*12], vel_buf[i*12+1], vel_buf[i*12+2], vel_buf[i*12+3]]) as f64;
        let vy = f32::from_le_bytes([vel_buf[i*12+4], vel_buf[i*12+5], vel_buf[i*12+6], vel_buf[i*12+7]]) as f64;
        let vz = f32::from_le_bytes([vel_buf[i*12+8], vel_buf[i*12+9], vel_buf[i*12+10], vel_buf[i*12+11]]) as f64;

        positions.push(px);
        positions.push(py);
        positions.push(pz);
        velocities.push(vx);
        velocities.push(vy);
        velocities.push(vz);
        signs.push(signs_buf[i] as i32);
    }

    (positions, velocities, signs, a, t)
}

fn extract_and_subdivide(
    positions: &[f64],
    velocities: &[f64],
    signs: &[i32],
) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize, usize) {
    let mut rng = rand::thread_rng();
    let mut new_pos = Vec::new();
    let mut new_vel = Vec::new();
    let mut new_signs = Vec::new();

    let sigma_pos = EPSILON / 3.0;
    let sigma_vel = 4.0;

    let n = signs.len();
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;

    for i in 0..n {
        let px = positions[i * 3];
        let py = positions[i * 3 + 1];
        let pz = positions[i * 3 + 2];

        let dx = px - CENTER[0];
        let dy = py - CENTER[1];
        let dz = pz - CENTER[2];
        let r = periodic_dist_orig(dx, dy, dz);

        if r > R_EXTRACT {
            continue;
        }

        let rel_x = dx - L_BOX * (dx / L_BOX).round();
        let rel_y = dy - L_BOX * (dy / L_BOX).round();
        let rel_z = dz - L_BOX * (dz / L_BOX).round();

        let sign = signs[i];
        let is_hr = r < R_HR;

        if is_hr && sign > 0 {
            for _ in 0..N_SUB {
                let theta = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                let phi = (1.0 - 2.0 * rng.gen::<f64>()).acos();
                let r_offset = sigma_pos * rng.gen::<f64>().powf(1.0/3.0);

                let off_x = r_offset * phi.sin() * theta.cos();
                let off_y = r_offset * phi.sin() * theta.sin();
                let off_z = r_offset * phi.cos();

                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let dvx = sigma_vel * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let dvy = sigma_vel * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                let dvz = sigma_vel * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

                new_pos.push(rel_x + off_x);
                new_pos.push(rel_y + off_y);
                new_pos.push(rel_z + off_z);
                new_vel.push(velocities[i * 3] + dvx);
                new_vel.push(velocities[i * 3 + 1] + dvy);
                new_vel.push(velocities[i * 3 + 2] + dvz);
                new_signs.push(sign);
                n_plus += 1;
            }
        } else {
            new_pos.push(rel_x);
            new_pos.push(rel_y);
            new_pos.push(rel_z);
            new_vel.push(velocities[i * 3]);
            new_vel.push(velocities[i * 3 + 1]);
            new_vel.push(velocities[i * 3 + 2]);
            new_signs.push(sign);
            if sign > 0 { n_plus += 1; } else { n_minus += 1; }
        }
    }

    (new_pos, new_vel, new_signs, n_plus, n_minus)
}

fn save_snapshot(pos: &[f64], vel: &[f64], signs: &[i32], a: f64, t: f64, step: usize) {
    let path = format!("{}/snapshots/snap_{:05}.bin", OUTPUT_DIR, step);
    let file = File::create(&path).expect("Cannot create snapshot");
    let mut writer = BufWriter::new(file);

    let n = signs.len() as u64;

    writer.write_all(&n.to_le_bytes()).unwrap();
    writer.write_all(&a.to_le_bytes()).unwrap();
    writer.write_all(&t.to_le_bytes()).unwrap();

    for i in 0..signs.len() {
        writer.write_all(&(pos[i * 3] as f32).to_le_bytes()).unwrap();
        writer.write_all(&(pos[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        writer.write_all(&(pos[i * 3 + 2] as f32).to_le_bytes()).unwrap();
    }

    for i in 0..signs.len() {
        writer.write_all(&(vel[i * 3] as f32).to_le_bytes()).unwrap();
        writer.write_all(&(vel[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        writer.write_all(&(vel[i * 3 + 2] as f32).to_le_bytes()).unwrap();
    }

    for s in signs {
        writer.write_all(&[*s as u8]).unwrap();
    }

    writer.flush().unwrap();
}

#[cfg(feature = "cuda")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           JANUS ZOOM-IN L1 — z=0.459 → z=0                               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  SOURCE: snap_04550.bin");
    println!("║  CENTER: [{:.3}, {:.3}, {:.3}] Mpc", CENTER[0], CENTER[1], CENTER[2]);
    println!("║  R_HR = {} Mpc, R_EXTRACT = {} Mpc", R_HR, R_EXTRACT);
    println!("║  SUBDIVISION: m+ in HR ×{}", N_SUB);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  dt = {} Gyr, steps = {}", DT, N_STEPS);
    println!("║  θ = {}, ε = {} Mpc", THETA, EPSILON);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    fs::create_dir_all(format!("{}/snapshots", OUTPUT_DIR)).expect("Failed to create output dir");

    let source_path = "/app/output/janus_baryonic_calibrated/snapshots/snap_04550.bin";
    println!("Loading source snapshot: {}", source_path);
    let (positions, velocities, signs, a_init, t_init) = read_source_snapshot(source_path);
    let z_init = 1.0 / a_init - 1.0;
    println!("  Loaded {} particles, z = {:.4}, t = {:.4} Gyr", signs.len(), z_init, t_init);

    println!("\nExtracting r < {} Mpc and subdividing HR...", R_EXTRACT);
    let (new_pos, new_vel, new_signs, n_plus, n_minus) = extract_and_subdivide(&positions, &velocities, &signs);
    let n_total = new_signs.len();

    println!("  Extracted particles: {}", n_total);
    println!("  N+ = {}, N- = {}", n_plus, n_minus);

    println!("\nInitializing GPU simulation...");
    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_ZOOM,
        new_pos, new_vel, new_signs
    ).expect("Failed to create GPU simulation");

    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(EPSILON);

    let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z_init, ETA);
    gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

    println!("  GPU initialized in {:.1}s", start_time.elapsed().as_secs_f64());

    let csv_path = format!("{}/time_series.csv", OUTPUT_DIR);
    let mut csv_file = File::create(&csv_path).expect("Cannot create CSV");
    writeln!(csv_file, "step,z,t_Gyr,a,rho_center,v_disp").unwrap();

    let mut a = a_init;
    let mut t = t_init;

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("Starting simulation: {} steps", N_STEPS);
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

        if step % SNAPSHOT_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().unwrap_or_default();
            let vel = gpu_sim.get_velocities().unwrap_or_default();
            let signs = gpu_sim.get_signs().unwrap_or_default();
            save_snapshot(&pos, &vel, &signs, a, t, step);
        }

        if step % CSV_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().unwrap_or_default();
            let vel = gpu_sim.get_velocities().unwrap_or_default();
            let signs = gpu_sim.get_signs().unwrap_or_default();
            let n = signs.len();

            // Count particles in center (r < 1 Mpc)
            let mut n_center = 0;
            let mut vx_sum = 0.0;
            let mut vy_sum = 0.0;
            let mut vz_sum = 0.0;
            let mut n_v = 0;

            for i in 0..n {
                if signs[i] <= 0 { continue; }
                let x = pos[i * 3];
                let y = pos[i * 3 + 1];
                let z = pos[i * 3 + 2];
                let r = (x*x + y*y + z*z).sqrt();

                if r < 1.0 {
                    n_center += 1;
                }
                if r < 5.0 {
                    vx_sum += vel[i * 3];
                    vy_sum += vel[i * 3 + 1];
                    vz_sum += vel[i * 3 + 2];
                    n_v += 1;
                }
            }

            let rho_center = n_center as f64 / (4.0/3.0 * std::f64::consts::PI);

            let v_disp = if n_v > 10 {
                let vx_mean = vx_sum / n_v as f64;
                let vy_mean = vy_sum / n_v as f64;
                let vz_mean = vz_sum / n_v as f64;

                let mut var = 0.0;
                for i in 0..n {
                    if signs[i] <= 0 { continue; }
                    let x = pos[i * 3];
                    let y = pos[i * 3 + 1];
                    let pz = pos[i * 3 + 2];
                    let r = (x*x + y*y + pz*pz).sqrt();
                    if r < 5.0 {
                        var += (vel[i*3] - vx_mean).powi(2) +
                               (vel[i*3+1] - vy_mean).powi(2) +
                               (vel[i*3+2] - vz_mean).powi(2);
                    }
                }
                (var / n_v as f64).sqrt()
            } else { 0.0 };

            writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.1},{:.1}",
                     step, z, t, a, rho_center, v_disp).unwrap();

            if step % 100 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
                let eta_h = if rate > 0.0 { (N_STEPS - step) as f64 / rate / 3600.0 } else { 0.0 };

                println!("[{:5}/{:5}] z={:.3} | ρ={:.0} | v_disp={:.0} | ETA={:.1}h",
                         step, N_STEPS, z, rho_center, v_disp, eta_h);
            }
        }

        if step == N_STEPS { break; }

        let H = 100.0 * ETA * a.powf(-1.5) * 1.022e-3;
        let dtau_per_dt = 1.0;

        gpu_sim.step_with_expansion_dkd(DT, a, H, dtau_per_dt)
            .expect("GPU step failed");

        a += a * H * DT;
        t += DT;
    }

    let total_time = start_time.elapsed();
    let z_final = 1.0 / a - 1.0;

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("SIMULATION TERMINÉE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  z_final = {:.4}", z_final);
    println!("  t_final = {:.2} Gyr", t);
    println!("  Temps total: {:.1} h", total_time.as_secs_f64() / 3600.0);
    println!("  Output: {}", OUTPUT_DIR);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("ERROR: This binary requires CUDA. Compile with --features cuda");
}
