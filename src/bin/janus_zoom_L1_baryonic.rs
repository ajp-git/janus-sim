//! JANUS ZOOM-IN L1 BARYONIC — Cluster simulation with full baryonic physics
//!
//! v3: v_kick=20 km/s + gas retention criterion (v < 0.5 × v_escape)
//! v4: --from-snapshot flag to load pre-split snapshots directly
//!
//! HR particles (r < 8 Mpc): Cooling + SF + SN feedback
//! BG particles (r > 8 Mpc): Pure gravity (collisionless)

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::sph_pressure_gpu::GpuSphPressure;
use janus::vsl_dynamic::CoupledFriedmann;
use std::fs::{self, File};
use std::io::{BufWriter, Write, Read};
use std::time::Instant;
use std::collections::HashSet;
use std::env;
use std::sync::Arc;
use rand::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// COSMOLOGY
// ═══════════════════════════════════════════════════════════════════════════
const ETA: f64 = 1.045;
const G_CODE: f64 = 4.498e-15;  // G in [Mpc³/(M_sun·Gyr²)]

// ═══════════════════════════════════════════════════════════════════════════
// ZOOM PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const CENTER: [f64; 3] = [-5.329, 11.171, -39.571];
const R_HR: f64 = 8.0;
const R_EXTRACT: f64 = 50.0;
const N_SUB: usize = 10;
const L_BOX: f64 = 500.0;
const L_ZOOM: f64 = 120.0;

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const DT: f64 = 0.0002;  // Reduced for stability with split particles
const N_STEPS: usize = 46250;  // Adjusted for smaller dt (same total time)
const THETA: f64 = 0.7;
const EPSILON_HR: f64 = 0.08;  // Increased for stability
const EPSILON_BG: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════
// BARYONIC PHYSICS — Schmidt-Kennicutt SF (HR particles only)
// ═══════════════════════════════════════════════════════════════════════════
const T_INIT_HR: f64 = 10000.0;           // Initial temperature [K]
const T_FLOOR: f64 = 1000.0;              // Minimum temperature [K]
const T_SF_MAX: f64 = 10000.0;            // Max T for SF [K]
const OVERDENSITY_THRESHOLD: f64 = 50.0;  // SF threshold δ > 50
const EPSILON_STAR: f64 = 0.01;           // SF efficiency per t_ff
const M_PART_HR: f64 = 5.1e10;            // HR particle mass [M_sun]
const V_KICK_SN: f64 = 20.0;              // SN kick velocity [km/s] — reduced to retain gas
const MASS_LOADING: f64 = 3.0;            // SN mass loading η
const M_HALO: f64 = 7.7e14;               // Halo mass [M_sun]
const V_ESCAPE_FACTOR: f64 = 0.5;         // Max v_kick / v_escape ratio
const DELAYED_COOLING_GYR: f64 = 0.01;    // Cooling delay after SN [Gyr]
const R_NEIGHBOR: f64 = 0.5;              // Neighbor search radius [Mpc]

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════════
const SNAPSHOT_INTERVAL: usize = 10;
const CSV_INTERVAL: usize = 10;
const OUTPUT_DIR: &str = "/app/output/janus_zoom_L1_baryonic";

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

/// Read pre-split v2 snapshot directly (no extraction/subdivision needed)
/// Computes is_hr from radius: r < R_HR
fn read_v2_snapshot_direct(path: &str) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<bool>, f64, f64, usize, usize) {
    let mut file = File::open(path).expect("Cannot open v2 snapshot");
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

    let mut positions = Vec::with_capacity(n * 3);
    let mut velocities = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);
    let mut is_hr = Vec::with_capacity(n);
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;

    for i in 0..n {
        let px = f32::from_le_bytes([pos_buf[i*12], pos_buf[i*12+1], pos_buf[i*12+2], pos_buf[i*12+3]]) as f64;
        let py = f32::from_le_bytes([pos_buf[i*12+4], pos_buf[i*12+5], pos_buf[i*12+6], pos_buf[i*12+7]]) as f64;
        let pz = f32::from_le_bytes([pos_buf[i*12+8], pos_buf[i*12+9], pos_buf[i*12+10], pos_buf[i*12+11]]) as f64;

        let vx = f32::from_le_bytes([vel_buf[i*12], vel_buf[i*12+1], vel_buf[i*12+2], vel_buf[i*12+3]]) as f64;
        let vy = f32::from_le_bytes([vel_buf[i*12+4], vel_buf[i*12+5], vel_buf[i*12+6], vel_buf[i*12+7]]) as f64;
        let vz = f32::from_le_bytes([vel_buf[i*12+8], vel_buf[i*12+9], vel_buf[i*12+10], vel_buf[i*12+11]]) as f64;

        // Compute radius from origin (snapshot is already centered)
        let r = (px*px + py*py + pz*pz).sqrt();

        positions.push(px);
        positions.push(py);
        positions.push(pz);
        velocities.push(vx);
        velocities.push(vy);
        velocities.push(vz);

        let sign = if signs_buf[i] > 1 { -1i32 } else { signs_buf[i] as i32 };
        signs.push(sign);
        is_hr.push(r < R_HR);

        if sign > 0 { n_plus += 1; } else { n_minus += 1; }
    }

    (positions, velocities, signs, is_hr, a, t, n_plus, n_minus)
}

/// Extract and subdivide, returning HR flags
fn extract_and_subdivide(
    positions: &[f64],
    velocities: &[f64],
    signs: &[i32],
) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<bool>, usize, usize) {
    let mut rng = rand::thread_rng();
    let mut new_pos = Vec::new();
    let mut new_vel = Vec::new();
    let mut new_signs = Vec::new();
    let mut is_hr_flags = Vec::new();

    let sigma_pos = EPSILON_HR / 3.0;
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
            // Subdivide m+ HR particles
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
                is_hr_flags.push(true);
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
            is_hr_flags.push(is_hr);
            if sign > 0 { n_plus += 1; } else { n_minus += 1; }
        }
    }

    (new_pos, new_vel, new_signs, is_hr_flags, n_plus, n_minus)
}

fn save_snapshot(pos: &[f64], vel: &[f64], signs: &[i32], a: f64, t: f64, step: usize, out_dir: &str) {
    let path = format!("{}/snapshots/snap_{:05}.bin", out_dir, step);
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

/// Schmidt-Kennicutt probabilistic star formation
/// p_SF = 1 - exp(-ε_star × dt / t_ff)
fn apply_sf_schmidt_kennicutt(
    pos: &mut [f64],
    vel: &mut [f64],
    signs: &[i32],
    is_hr: &[bool],
    temperatures: &mut [f64],
    t_last_sn: &mut [f64],
    stars: &mut HashSet<usize>,
    overdensity: &[f64],
    rho_local: &[f64],
    divv: &[f64],
    t_current: f64,
    dt: f64,
) -> usize {
    let mut rng = rand::thread_rng();
    let n = signs.len();
    let mut new_stars = 0;

    for i in 0..n {
        // Only HR m+ particles
        if signs[i] <= 0 || !is_hr[i] { continue; }
        if stars.contains(&i) { continue; }

        // Check SF conditions:
        // 1. overdensity > threshold
        // 2. T < T_SF_MAX (cold gas)
        // 3. ∇·v < 0 (converging flow)
        if overdensity[i] < OVERDENSITY_THRESHOLD { continue; }
        if temperatures[i] > T_SF_MAX { continue; }
        if divv[i] >= 0.0 { continue; }  // Must be converging

        // Schmidt-Kennicutt: t_ff = sqrt(3π / 32Gρ)
        let rho = rho_local[i].max(1e-10);  // M_sun/Mpc³
        let t_ff = (3.0 * std::f64::consts::PI / (32.0 * G_CODE * rho)).sqrt();

        // p_SF = 1 - exp(-ε_star × dt / t_ff)
        let p_sf = 1.0 - (-EPSILON_STAR * dt / t_ff.max(1e-6)).exp();

        // Probabilistic SF
        if rng.gen::<f64>() < p_sf {
            stars.insert(i);
            new_stars += 1;

            // Apply SN feedback to neighbors
            let xi = pos[i * 3];
            let yi = pos[i * 3 + 1];
            let zi = pos[i * 3 + 2];

            let r_feedback = 0.3;
            let mut neighbors: Vec<usize> = Vec::new();
            for j in 0..n {
                if j == i || signs[j] <= 0 || !is_hr[j] { continue; }
                let dx = pos[j * 3] - xi;
                let dy = pos[j * 3 + 1] - yi;
                let dz = pos[j * 3 + 2] - zi;
                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                if r < r_feedback && r > 0.01 {
                    neighbors.push(j);
                }
            }

            // Apply kicks to neighbors with gas retention criterion
            let n_kick = ((MASS_LOADING + 1.0) as usize).min(neighbors.len());
            for k in 0..n_kick {
                let j = neighbors[k];
                let pjx = pos[j * 3];
                let pjy = pos[j * 3 + 1];
                let pjz = pos[j * 3 + 2];
                let dx = pjx - xi;
                let dy = pjy - yi;
                let dz = pjz - zi;
                let r_from_sn = (dx*dx + dy*dy + dz*dz).sqrt().max(0.01);

                // Particle distance from halo center
                let r_from_center = (pjx*pjx + pjy*pjy + pjz*pjz).sqrt().max(0.1);

                // Local escape velocity: v_escape = sqrt(2GM/r) [Mpc/Gyr] → [km/s]
                // G_CODE is in Mpc³/(M_sun·Gyr²), 1 Mpc/Gyr = 977.8 km/s
                let v_escape_local = (2.0 * G_CODE * M_HALO / r_from_center).sqrt() * 977.8;

                // Current velocity of particle
                let vj = [vel[j * 3], vel[j * 3 + 1], vel[j * 3 + 2]];
                let v_current = (vj[0]*vj[0] + vj[1]*vj[1] + vj[2]*vj[2]).sqrt();

                // Proposed kick direction
                let kick_dir = [dx / r_from_sn, dy / r_from_sn, dz / r_from_sn];

                // Proposed velocity after kick
                let v_after = [
                    vj[0] + V_KICK_SN * kick_dir[0],
                    vj[1] + V_KICK_SN * kick_dir[1],
                    vj[2] + V_KICK_SN * kick_dir[2],
                ];
                let v_after_mag = (v_after[0]*v_after[0] + v_after[1]*v_after[1] + v_after[2]*v_after[2]).sqrt();

                // Gas retention: if v_after > 0.5 × v_escape, reduce kick proportionally
                let v_max = V_ESCAPE_FACTOR * v_escape_local;
                let actual_kick = if v_after_mag > v_max && v_after_mag > v_current {
                    // Reduce kick so final velocity = v_max
                    let reduction_factor = (v_max - v_current).max(0.0) / V_KICK_SN.max(0.01);
                    V_KICK_SN * reduction_factor.min(1.0)
                } else {
                    V_KICK_SN
                };

                vel[j * 3] += actual_kick * kick_dir[0];
                vel[j * 3 + 1] += actual_kick * kick_dir[1];
                vel[j * 3 + 2] += actual_kick * kick_dir[2];

                t_last_sn[j] = t_current;
            }
        }
    }

    new_stars
}

/// Simple S&D93-like cooling for HR particles (CPU)
fn apply_cooling_hr(
    temperatures: &mut [f64],
    signs: &[i32],
    is_hr: &[bool],
    t_last_sn: &[f64],
    t_current: f64,
    dt: f64,
) {
    let n = signs.len();

    for i in 0..n {
        if signs[i] <= 0 || !is_hr[i] { continue; }

        // Check delayed cooling
        if t_current - t_last_sn[i] < DELAYED_COOLING_GYR {
            continue;
        }

        let t = temperatures[i];
        if t < T_FLOOR { continue; }

        // S&D93 approximation
        let cooling_rate = if t > 10000.0 {
            1e-23 * (t / 10000.0).sqrt()
        } else {
            1e-21 * (t / 10000.0).powf(-0.7)
        };

        let t_cool = t / (cooling_rate * 1e7).max(1e-10);
        let dt_cool_gyr = t_cool * 3.15e-17;

        let new_t = t * (-dt / dt_cool_gyr.max(0.001)).exp();
        temperatures[i] = new_t.max(T_FLOOR);
    }
}

#[cfg(feature = "cuda")]
fn main() {
    // Parse CLI arguments
    let args: Vec<String> = env::args().collect();
    let from_snapshot = args.iter().position(|a| a == "--from-snapshot")
        .map(|i| args.get(i + 1).cloned())
        .flatten();
    let output_dir = args.iter().position(|a| a == "--out-dir")
        .map(|i| args.get(i + 1).cloned())
        .flatten()
        .unwrap_or_else(|| OUTPUT_DIR.to_string());

    let start_time = Instant::now();

    // Create output directory
    fs::create_dir_all(format!("{}/snapshots", output_dir)).expect("Failed to create output dir");

    // Load particles - either from pre-split snapshot or original extraction
    let (mut pos_flat, mut vel_flat, signs_flat, is_hr, a_init, t_init, n_plus, n_minus):
        (Vec<f64>, Vec<f64>, Vec<i32>, Vec<bool>, f64, f64, usize, usize);

    if let Some(snap_path) = from_snapshot {
        // Direct loading from pre-split v2 snapshot
        println!("╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║   JANUS ZOOM-IN L1 BARYONIC v4 — FROM PRE-SPLIT SNAPSHOT                 ║");
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  MODE: --from-snapshot (skip extraction/subdivision)");
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  HR PARTICLES (r < {} Mpc):", R_HR);
        println!("║    Cooling: S&D93, T_floor = {} K", T_FLOOR);
        println!("║    SF: Schmidt-Kennicutt, ε_star = {}, δ > {}", EPSILON_STAR, OVERDENSITY_THRESHOLD);
        println!("║    SN: v_kick = {} km/s, η = {}", V_KICK_SN, MASS_LOADING);
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  BG PARTICLES (r > {} Mpc): Pure gravity", R_HR);
        println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

        println!("[BARYONIC] Loading pre-split snapshot: {}", snap_path);
        let result = read_v2_snapshot_direct(&snap_path);
        pos_flat = result.0;
        vel_flat = result.1;
        signs_flat = result.2;
        is_hr = result.3;
        a_init = result.4;
        t_init = result.5;
        n_plus = result.6;
        n_minus = result.7;

        let n_total = signs_flat.len();
        let n_hr_count = is_hr.iter().filter(|&&x| x).count();
        let z_init = 1.0 / a_init - 1.0;

        println!("[BARYONIC] Loaded {} particles from snapshot", n_total);
        println!("[BARYONIC] HR region: {} particles (r < {} Mpc)", n_hr_count, R_HR);
        println!("[BARYONIC] z = {:.4}, t = {:.4} Gyr", 1.0 / a_init - 1.0, t_init);
        println!("[BARYONIC] N+ = {}, N- = {}", n_plus, n_minus);
        println!("[BARYONIC] Cooling + SF + SN feedback: ACTIVE\n");

    } else {
        // Original behavior: load and extract/subdivide
        println!("╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║   JANUS ZOOM-IN L1 BARYONIC v3 — Gas retention + reduced v_kick          ║");
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  SOURCE: snap_04550.bin (z=0.459)");
        println!("║  CENTER: [{:.3}, {:.3}, {:.3}] Mpc", CENTER[0], CENTER[1], CENTER[2]);
        println!("║  R_HR = {} Mpc, R_EXTRACT = {} Mpc, N_SUB = {}", R_HR, R_EXTRACT, N_SUB);
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  HR PARTICLES (r < {} Mpc):", R_HR);
        println!("║    Cooling: S&D93, T_floor = {} K", T_FLOOR);
        println!("║    SF: Schmidt-Kennicutt, ε_star = {}, δ > {}", EPSILON_STAR, OVERDENSITY_THRESHOLD);
        println!("║        p_SF = 1 - exp(-ε_star × dt / t_ff)");
        println!("║    SN: v_kick = {} km/s, η = {}", V_KICK_SN, MASS_LOADING);
        println!("║        Gas retention: v_after < {:.0}% × v_escape_local", V_ESCAPE_FACTOR * 100.0);
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  BG PARTICLES (r > {} Mpc): Pure gravity", R_HR);
        println!("╠══════════════════════════════════════════════════════════════════════════╣");
        println!("║  dt = {} Gyr, steps = {}", DT, N_STEPS);
        println!("║  θ = {}, ε_HR = {} Mpc", THETA, EPSILON_HR);
        println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

        let source_path = "/app/output/janus_baryonic_calibrated/snapshots/snap_04550.bin";
        println!("Loading source snapshot: {}", source_path);
        let (positions, velocities, signs, a, t) = read_source_snapshot(source_path);
        a_init = a;
        t_init = t;
        let z_init = 1.0 / a_init - 1.0;
        println!("  Loaded {} particles, z = {:.4}, t = {:.4} Gyr", signs.len(), z_init, t_init);

        println!("\nExtracting r < {} Mpc and subdividing HR...", R_EXTRACT);
        let result = extract_and_subdivide(&positions, &velocities, &signs);
        pos_flat = result.0;
        vel_flat = result.1;
        signs_flat = result.2;
        is_hr = result.3;
        n_plus = result.4;
        n_minus = result.5;
    }

    let n_total = signs_flat.len();
    let n_hr_count = is_hr.iter().filter(|&&x| x).count();
    let z_init = 1.0 / a_init - 1.0;

    println!("  Total particles: {}", n_total);
    println!("  N+ = {}, N- = {}", n_plus, n_minus);
    println!("  HR particles: {} ({:.1}%)", n_hr_count, 100.0 * n_hr_count as f64 / n_total as f64);

    // Correct total momentum (prevent COM drift)
    let mut v_mean = [0.0f64; 3];
    for i in 0..n_total {
        v_mean[0] += vel_flat[i * 3];
        v_mean[1] += vel_flat[i * 3 + 1];
        v_mean[2] += vel_flat[i * 3 + 2];
    }
    v_mean[0] /= n_total as f64;
    v_mean[1] /= n_total as f64;
    v_mean[2] /= n_total as f64;
    let v_drift = (v_mean[0].powi(2) + v_mean[1].powi(2) + v_mean[2].powi(2)).sqrt();
    println!("  Momentum correction: v_drift = {:.2} km/s", v_drift);
    for i in 0..n_total {
        vel_flat[i * 3] -= v_mean[0];
        vel_flat[i * 3 + 1] -= v_mean[1];
        vel_flat[i * 3 + 2] -= v_mean[2];
    }

    // Initialize GPU simulation
    println!("\nInitializing GPU simulation...");
    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_ZOOM,
        pos_flat.clone(), vel_flat.clone(), signs_flat.clone()
    ).expect("Failed to create GPU simulation");

    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(EPSILON_HR);

    let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z_init, ETA);
    gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

    println!("  GPU initialized in {:.1}s", start_time.elapsed().as_secs_f64());

    // Initialize SPH GPU for HR m+ particles only
    let hr_plus_indices: Vec<usize> = (0..n_total)
        .filter(|&i| signs_flat[i] > 0 && is_hr[i])
        .collect();
    let n_hr_plus = hr_plus_indices.len();
    println!("  HR m+ particles for SPH: {}", n_hr_plus);

    let sph_device = cudarc::driver::CudaDevice::new(0).expect("Failed to get CUDA device for SPH");
    let mut sph_pressure = GpuSphPressure::new(sph_device, n_hr_plus, M_PART_HR, L_ZOOM)
        .expect("Failed to create SPH pressure calculator");
    println!("  SPH density GPU initialized");

    // Mean density for overdensity calculation
    let v_hr = 4.0 / 3.0 * std::f64::consts::PI * R_HR.powi(3);
    let rho_mean_hr = n_hr_plus as f64 / v_hr;

    // Initialize baryonic state
    let mut temperatures: Vec<f64> = (0..n_total)
        .map(|i| if is_hr[i] && signs_flat[i] > 0 { T_INIT_HR } else { 0.0 })
        .collect();
    let mut t_last_sn: Vec<f64> = vec![-1.0; n_total];
    let mut stars: HashSet<usize> = HashSet::new();

    // CSV output
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv_file = File::create(&csv_path).expect("Cannot create CSV");
    writeln!(csv_file, "step,z,t_Gyr,rho_max_HR,rho_max_BG,N_stars_HR,SFR_HR,v_disp_HR,T_mean_HR,ratio_vrms,S_global").unwrap();

    // Simulation state
    let mut a = a_init;
    let mut t = t_init;
    let mut total_new_stars = 0usize;

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("Starting simulation: {} steps (Schmidt-Kennicutt SF)", N_STEPS);
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

        // Get current state from GPU (every 10 steps for baryonic, or for output)
        let need_baryonic = step % 10 == 0;
        let need_output = step % CSV_INTERVAL == 0 || step % SNAPSHOT_INTERVAL == 0;

        if need_baryonic || need_output {
            pos_flat = gpu_sim.get_positions().unwrap_or_default();
            vel_flat = gpu_sim.get_velocities().unwrap_or_default();
        }

        // Apply baryonic physics every 10 steps (skip step 0 for SF)
        let new_stars = if need_baryonic && step > 0 {
            // Extract HR m+ positions for SPH GPU
            let pos_hr_plus: Vec<f64> = hr_plus_indices.iter()
                .flat_map(|&i| vec![pos_flat[i*3], pos_flat[i*3+1], pos_flat[i*3+2]])
                .collect();

            // Compute density on GPU
            sph_pressure.upload_positions(&pos_hr_plus).expect("SPH upload failed");
            sph_pressure.compute_density().expect("SPH density failed");
            let densities_hr = sph_pressure.download_densities().expect("SPH download failed");

            // Map densities back to full arrays
            let mut overdensity = vec![1.0f64; n_total];
            let mut rho_local = vec![0.0f64; n_total];
            let mut divv = vec![0.0f64; n_total];  // Not computed on GPU, keep zero
            for (k, &i) in hr_plus_indices.iter().enumerate() {
                rho_local[i] = densities_hr[k];
                overdensity[i] = densities_hr[k] / rho_mean_hr.max(1e-10);
            }

            let ns = apply_sf_schmidt_kennicutt(
                &mut pos_flat, &mut vel_flat, &signs_flat, &is_hr,
                &mut temperatures, &mut t_last_sn, &mut stars,
                &overdensity, &rho_local, &divv, t, DT * 10.0
            );

            // Upload modified velocities after SN kicks
            if ns > 0 {
                gpu_sim.set_velocities(&vel_flat).expect("Failed to upload velocities");
            }
            ns
        } else { 0 };
        total_new_stars += new_stars;

        // Cooling every step
        apply_cooling_hr(&mut temperatures, &signs_flat, &is_hr, &t_last_sn, t, DT);

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&pos_flat, &vel_flat, &signs_flat, a, t, step, &output_dir);
        }

        // Metrics
        if step % CSV_INTERVAL == 0 {
            let mut rho_max_hr = 0.0f64;
            let mut rho_max_bg = 0.0f64;
            let mut v_sum_hr = [0.0f64; 3];
            let mut v_count_hr = 0;
            let mut t_sum = 0.0f64;
            let mut t_count = 0;
            let mut v2_plus = 0.0f64;
            let mut v2_minus = 0.0f64;
            let mut n_plus_count = 0;
            let mut n_minus_count = 0;
            let mut com_plus = [0.0f64; 3];
            let mut com_minus = [0.0f64; 3];

            for i in 0..n_total {
                let x = pos_flat[i * 3];
                let y = pos_flat[i * 3 + 1];
                let pz = pos_flat[i * 3 + 2];
                let r = (x*x + y*y + pz*pz).sqrt();
                let vx = vel_flat[i * 3];
                let vy = vel_flat[i * 3 + 1];
                let vz = vel_flat[i * 3 + 2];
                let v2 = vx*vx + vy*vy + vz*vz;

                if signs_flat[i] > 0 {
                    v2_plus += v2;
                    n_plus_count += 1;
                    com_plus[0] += x;
                    com_plus[1] += y;
                    com_plus[2] += pz;

                    if is_hr[i] {
                        if r < 0.5 { rho_max_hr += 1.0; }
                        if r < 5.0 {
                            v_sum_hr[0] += vx;
                            v_sum_hr[1] += vy;
                            v_sum_hr[2] += vz;
                            v_count_hr += 1;
                        }
                        if temperatures[i] > 0.0 {
                            t_sum += temperatures[i];
                            t_count += 1;
                        }
                    } else {
                        if r < 2.0 { rho_max_bg += 1.0; }
                    }
                } else {
                    v2_minus += v2;
                    n_minus_count += 1;
                    com_minus[0] += x;
                    com_minus[1] += y;
                    com_minus[2] += pz;
                }
            }

            let v_vol_hr = 4.0 / 3.0 * std::f64::consts::PI * 0.5f64.powi(3);
            let v_vol_bg = 4.0 / 3.0 * std::f64::consts::PI * 2.0f64.powi(3);
            rho_max_hr /= v_vol_hr;
            rho_max_bg /= v_vol_bg;

            let v_disp_hr = if v_count_hr > 10 {
                let v_mean = [v_sum_hr[0] / v_count_hr as f64, v_sum_hr[1] / v_count_hr as f64, v_sum_hr[2] / v_count_hr as f64];
                let mut var = 0.0;
                for i in 0..n_total {
                    if signs_flat[i] <= 0 || !is_hr[i] { continue; }
                    let x = pos_flat[i * 3];
                    let y = pos_flat[i * 3 + 1];
                    let pz = pos_flat[i * 3 + 2];
                    let r = (x*x + y*y + pz*pz).sqrt();
                    if r < 5.0 {
                        var += (vel_flat[i*3] - v_mean[0]).powi(2) +
                               (vel_flat[i*3+1] - v_mean[1]).powi(2) +
                               (vel_flat[i*3+2] - v_mean[2]).powi(2);
                    }
                }
                (var / v_count_hr as f64).sqrt()
            } else { 0.0 };

            let t_mean_hr = if t_count > 0 { t_sum / t_count as f64 } else { 0.0 };
            let sfr = total_new_stars as f64 * M_PART_HR / t.max(0.001);

            let v_rms_plus = if n_plus_count > 0 { (v2_plus / n_plus_count as f64).sqrt() } else { 0.0 };
            let v_rms_minus = if n_minus_count > 0 { (v2_minus / n_minus_count as f64).sqrt() } else { 1.0 };
            let ratio_vrms = v_rms_plus / v_rms_minus.max(1e-6);

            let s_global = if n_plus_count > 0 && n_minus_count > 0 {
                let cp = [com_plus[0] / n_plus_count as f64, com_plus[1] / n_plus_count as f64, com_plus[2] / n_plus_count as f64];
                let cm = [com_minus[0] / n_minus_count as f64, com_minus[1] / n_minus_count as f64, com_minus[2] / n_minus_count as f64];
                let dx = cp[0] - cm[0];
                let dy = cp[1] - cm[1];
                let dz = cp[2] - cm[2];
                (dx*dx + dy*dy + dz*dz).sqrt() / L_ZOOM
            } else { 0.0 };

            writeln!(csv_file, "{},{:.4},{:.4},{:.1},{:.1},{},{:.2e},{:.1},{:.0},{:.4},{:.4}",
                     step, z, t, rho_max_hr, rho_max_bg, stars.len(), sfr, v_disp_hr, t_mean_hr, ratio_vrms, s_global).unwrap();

            if step % 100 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
                let eta_h = if rate > 0.0 { (N_STEPS - step) as f64 / rate / 3600.0 } else { 0.0 };

                println!("[{:5}/{:5}] z={:.3} | ρ_HR={:.0} | N★={} | T={:.0}K | SFR={:.1e} | ETA={:.1}h",
                         step, N_STEPS, z, rho_max_hr, stars.len(), t_mean_hr, sfr, eta_h);

                // Auto-stop checks
                if rho_max_hr > 500000.0 && stars.is_empty() {
                    println!("\n⚠️  AUTO-STOP: rho_max_HR > 500000 AND N_stars = 0");
                    break;
                }
                if stars.len() > 1_000_000 {
                    println!("\n⚠️  AUTO-STOP: N_stars > 1,000,000 — SF runaway!");
                    break;
                }
            }
        }

        if step == N_STEPS { break; }

        // GPU gravity step
        let h = 100.0 * ETA * a.powf(-1.5) * 1.022e-3;
        gpu_sim.step_with_expansion_dkd(DT, a, h, 1.0)
            .expect("GPU step failed");

        a += a * h * DT;
        t += DT;
    }

    let total_time = start_time.elapsed();
    let z_final = 1.0 / a - 1.0;

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("SIMULATION TERMINÉE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  z_final = {:.4}", z_final);
    println!("  t_final = {:.2} Gyr", t);
    println!("  N_stars_final = {}", stars.len());
    println!("  Temps total: {:.1} h", total_time.as_secs_f64() / 3600.0);
    println!("  Output: {}", output_dir);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("ERROR: This binary requires CUDA. Compile with --features cuda");
}
