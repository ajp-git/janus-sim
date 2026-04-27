//! VSL Bimetric Phase 0 — Validation Run
//!
//! 500K particles, 5000 steps, snapshot at EVERY step for frame rendering.
//! GO/NO-GO criteria checked at completion.

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::sph_pressure_gpu::GpuSphPressure;
use janus::vsl_dynamic::CoupledFriedmann;

// Phase 0 Parameters
const N_PLUS: usize = 250_000;
const N_MINUS: usize = 250_000;
const BOX_SIZE: f64 = 100.0;  // Mpc
const MU: f64 = 19.0;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;  // Gyr
const STEPS: usize = 5000;
const CSV_INTERVAL: usize = 5;

// SPH parameters
const T_INIT: f64 = 10000.0;  // Initial temperature K
const T_FLOOR: f64 = 100.0;   // Temperature floor K
const PARTICLE_MASS: f64 = 1e10;

// GO/NO-GO thresholds
const RATIO_MAX: f64 = 1.10;
const DENSITY_RATIO_MAX: f64 = 10.0;
const SEG_MIN: f64 = 0.001;

// Auto-stop (safety)
const VRMS_CRITICAL: f64 = 100_000.0;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: This binary requires --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_phase0();
}

#[cfg(feature = "cuda")]
fn run_phase0() {
    let output_dir = "/app/output/vsl_bimetric_phase0";
    create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 0 lancée (500K) — pipeline automatique P0→P1→P2 activé   ║");
    println!("║  Frames : chaque step | Vidéos : après chaque phase             ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)                          ║", N_PLUS + N_MINUS, N_PLUS, N_MINUS);
    println!("║  Box = {} Mpc, z_init = {}, steps = {}                      ║", BOX_SIZE, Z_INIT, STEPS);
    println!("║  SPH: T_init = {} K, T_floor = {} K                          ║", T_INIT as i32, T_FLOOR as i32);
    println!("║  VSL dynamique: c_ratio²(z) = (1+z)^δ, δ = (η-1)/η             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Initialize CSV
    let csv_path = format!("{}/evolution_phase0.csv", output_dir);
    let csv_file = File::create(&csv_path).unwrap();
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,ratio_vrms,seg,c_ratio_sq,go_status").unwrap();

    // Initialize log
    let log_path = format!("{}/simulation.log", output_dir);
    let log_file = File::create(&log_path).unwrap();
    let mut log_writer = BufWriter::new(log_file);
    writeln!(log_writer, "VSL Bimetric Phase 0 — μ={} η={}", MU, ETA).unwrap();

    // Track GO/NO-GO status
    let mut all_criteria_met = true;
    let mut failure_step = 0;
    let mut failure_reason = String::new();

    // Initial c_ratio
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    let c_ratio_init = c_ratio_sq_init.sqrt();
    println!("\nc_ratio(z={}) = {:.4}", Z_INIT, c_ratio_init);

    // Initialize GPU simulation
    println!("Initializing GPU simulation...");
    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(0.5);
    gpu_sim.set_c_ratio(c_ratio_init);

    // Initialize SPH modules
    println!("Initializing SPH pressure modules...");
    let device = Arc::new(CudaDevice::new(0).expect("Failed to create CUDA device"));

    let mut sph_plus = GpuSphPressure::new(
        Arc::clone(&device),
        N_PLUS,
        PARTICLE_MASS,
        BOX_SIZE,
    ).expect("Failed to create SPH+ module");

    let mut sph_minus = GpuSphPressure::new(
        Arc::clone(&device),
        N_MINUS,
        PARTICLE_MASS,
        BOX_SIZE,
    ).expect("Failed to create SPH- module");

    // Temperature arrays (start at T_INIT, floor at T_FLOOR)
    let temp_plus = vec![T_INIT.max(T_FLOOR); N_PLUS];
    let temp_minus = vec![T_INIT.max(T_FLOOR); N_MINUS];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SPH actif : m+ ✓ m- ✓                                       ║");
    println!("║  T_init = {} K, T_floor = {} K                              ║", T_INIT as i32, T_FLOOR as i32);
    println!("╚══════════════════════════════════════════════════════════════╝");

    let start_time = Instant::now();
    let mut t_gyr = 0.0;
    let mut z = Z_INIT;
    let mut final_step = 0;
    let mut final_seg = 0.0;

    println!("\nStarting Phase 0 simulation...\n");
    println!(" step |   t    |   z    | ρ+_max | ρ-_max | v_rms+ | v_rms- | ratio | seg    | GO");
    println!("------|--------|--------|--------|--------|--------|--------|-------|--------|----");

    for step in 0..=STEPS {
        // Update c_ratio dynamically
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z.max(0.0), ETA);
        let c_ratio = c_ratio_sq.sqrt();
        gpu_sim.set_c_ratio(c_ratio);

        // Get current state
        let pos = gpu_sim.get_positions().expect("get_positions failed");
        let vel = gpu_sim.get_velocities().expect("get_velocities failed");
        let signs = gpu_sim.get_signs().expect("get_signs failed");

        // Compute metrics
        let (v_rms_plus, v_rms_minus) = compute_vrms_by_sign(&vel, &signs);
        let (rho_plus_max, rho_minus_max) = compute_max_densities(&pos, &signs, BOX_SIZE, 64);
        let seg = compute_segregation(&pos, &signs, BOX_SIZE);
        final_seg = seg;

        // Ratios
        let ratio_vrms = if v_rms_plus > 0.0 { v_rms_minus / v_rms_plus } else { 1.0 };
        let ratio_vrms_inv = if v_rms_minus > 0.0 { v_rms_plus / v_rms_minus } else { 1.0 };
        let rho_ratio = if rho_plus_max > 0.0 { rho_minus_max / rho_plus_max } else { 1.0 };

        // Check GO criteria
        let mut step_go = true;
        if ratio_vrms > RATIO_MAX {
            step_go = false;
            if all_criteria_met {
                failure_step = step;
                failure_reason = format!("ratio v_rms-/v_rms+ = {:.3} > {}", ratio_vrms, RATIO_MAX);
            }
            all_criteria_met = false;
        }
        if ratio_vrms_inv > RATIO_MAX {
            step_go = false;
            if all_criteria_met {
                failure_step = step;
                failure_reason = format!("ratio v_rms+/v_rms- = {:.3} > {}", ratio_vrms_inv, RATIO_MAX);
            }
            all_criteria_met = false;
        }
        if rho_ratio > DENSITY_RATIO_MAX {
            step_go = false;
            if all_criteria_met {
                failure_step = step;
                failure_reason = format!("ρ-_max/ρ+_max = {:.1} > {}", rho_ratio, DENSITY_RATIO_MAX);
            }
            all_criteria_met = false;
        }

        let go_status = if step_go { "GO" } else { "FAIL" };

        // Save snapshot at EVERY step
        let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
        save_snapshot(&snap_path, &pos, &signs, z, BOX_SIZE);

        // Log every CSV_INTERVAL steps
        if step % CSV_INTERVAL == 0 {
            writeln!(csv_writer, "{},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.4},{:.6},{:.6},{}",
                     step, t_gyr, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, ratio_vrms, seg, c_ratio_sq, go_status).unwrap();
            csv_writer.flush().unwrap();

            let elapsed = start_time.elapsed().as_secs_f64() / 3600.0;
            let eta_h = if step > 0 { elapsed / step as f64 * (STEPS - step) as f64 } else { 0.0 };

            println!(" {:5} | {:.3} | {:.4} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:.3} | {:.4} | {} ({:.1}h, ETA {:.1}h)",
                     step, t_gyr, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, ratio_vrms, seg, go_status, elapsed, eta_h);

            writeln!(log_writer, "step={} z={:.4} ratio={:.4} seg={:.6} status={}",
                     step, z, ratio_vrms, seg, go_status).unwrap();
        }

        // Auto-stop: critical velocity
        if v_rms_minus > VRMS_CRITICAL || v_rms_plus > VRMS_CRITICAL {
            println!("\n╔════════════════════════════════════════════════════════════════╗");
            println!("║  AUTO-STOP: VITESSE CRITIQUE ATTEINTE                          ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
            failure_reason = format!("v_rms critical at step {} (v+={:.0}, v-={:.0})", step, v_rms_plus, v_rms_minus);
            all_criteria_met = false;
            failure_step = step;
            final_step = step;
            break;
        }

        // Integration step
        if step < STEPS {
            let scale_factor = 1.0 / (1.0 + z);
            let hubble = compute_hubble(z);
            let hubble_gyr = hubble * 1.022e-3;

            gpu_sim.step_with_expansion_dkd_gpu(DT, scale_factor, hubble_gyr, 1.0)
                .expect("Step failed");

            apply_sph_kick(&mut gpu_sim, &mut sph_plus, &mut sph_minus, &temp_plus, &temp_minus, DT);

            t_gyr += DT;
            z = compute_redshift_from_time(t_gyr, Z_INIT);
        }

        final_step = step;
    }

    // Final segregation check
    if final_seg < SEG_MIN {
        all_criteria_met = false;
        if failure_reason.is_empty() {
            failure_reason = format!("Seg = {:.6} < {} at final step", final_seg, SEG_MIN);
            failure_step = final_step;
        }
    }

    // Final flush
    csv_writer.flush().unwrap();
    log_writer.flush().unwrap();

    let total_time = start_time.elapsed().as_secs_f64() / 3600.0;

    // Report result
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    if all_criteria_met {
        println!("║  PHASE 0 COMPLETE — GO                                          ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Tous les critères satisfaits                                    ║");
        println!("║  → Prêt pour Phase 1 (2M particules)                             ║");
    } else {
        println!("║  PHASE 0 COMPLETE — NO-GO                                        ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Critère échoué: {}                           ║", failure_reason);
        println!("║  Step: {}, z = {:.4}                                            ║", failure_step, compute_redshift_from_time(failure_step as f64 * DT, Z_INIT));
    }
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Total time: {:.2} hours                                          ║", total_time);
    println!("║  Final step: {}, z = {:.4}                                       ║", final_step, z);
    println!("║  Final Seg: {:.6}                                                ║", final_seg);
    println!("║  Snapshots: {}/snapshots/                                        ║", output_dir);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Write GO/NO-GO result file
    let result_path = format!("{}/phase0_result.txt", output_dir);
    let mut result_file = File::create(&result_path).unwrap();
    if all_criteria_met {
        writeln!(result_file, "GO").unwrap();
    } else {
        writeln!(result_file, "NO-GO").unwrap();
        writeln!(result_file, "REASON: {}", failure_reason).unwrap();
        writeln!(result_file, "STEP: {}", failure_step).unwrap();
    }
}

#[cfg(feature = "cuda")]
fn compute_vrms_by_sign(vel: &[f64], signs: &[i32]) -> (f64, f64) {
    let n = signs.len();
    let mut sum_v2_plus = 0.0;
    let mut sum_v2_minus = 0.0;
    let mut n_plus = 0;
    let mut n_minus = 0;

    for i in 0..n {
        let vx = vel[i * 3];
        let vy = vel[i * 3 + 1];
        let vz = vel[i * 3 + 2];
        let v2 = vx * vx + vy * vy + vz * vz;

        if signs[i] > 0 {
            sum_v2_plus += v2;
            n_plus += 1;
        } else {
            sum_v2_minus += v2;
            n_minus += 1;
        }
    }

    let mpc_gyr_to_km_s = 978.0;
    let v_rms_plus = if n_plus > 0 { (sum_v2_plus / n_plus as f64).sqrt() * mpc_gyr_to_km_s } else { 0.0 };
    let v_rms_minus = if n_minus > 0 { (sum_v2_minus / n_minus as f64).sqrt() * mpc_gyr_to_km_s } else { 0.0 };

    (v_rms_plus, v_rms_minus)
}

#[cfg(feature = "cuda")]
fn compute_max_densities(pos: &[f64], signs: &[i32], box_size: f64, n_grid: usize) -> (f64, f64) {
    let cell_size = box_size / n_grid as f64;
    let n = signs.len();

    let mut grid_plus = vec![0u32; n_grid * n_grid * n_grid];
    let mut grid_minus = vec![0u32; n_grid * n_grid * n_grid];

    for i in 0..n {
        let ix = ((pos[i * 3] / cell_size) as usize).min(n_grid - 1);
        let iy = ((pos[i * 3 + 1] / cell_size) as usize).min(n_grid - 1);
        let iz = ((pos[i * 3 + 2] / cell_size) as usize).min(n_grid - 1);
        let idx = ix + iy * n_grid + iz * n_grid * n_grid;

        if signs[i] > 0 {
            grid_plus[idx] += 1;
        } else {
            grid_minus[idx] += 1;
        }
    }

    let rho_plus_max = *grid_plus.iter().max().unwrap_or(&0) as f64;
    let rho_minus_max = *grid_minus.iter().max().unwrap_or(&0) as f64;

    (rho_plus_max, rho_minus_max)
}

#[cfg(feature = "cuda")]
fn compute_segregation(pos: &[f64], signs: &[i32], box_size: f64) -> f64 {
    let n = signs.len();
    let mut sum_plus = [0.0f64; 3];
    let mut sum_minus = [0.0f64; 3];
    let mut n_plus = 0;
    let mut n_minus = 0;
    let mut ref_plus = [0.0f64; 3];
    let mut ref_minus = [0.0f64; 3];

    for i in 0..n {
        if signs[i] > 0 && n_plus == 0 {
            ref_plus = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
        }
        if signs[i] < 0 && n_minus == 0 {
            ref_minus = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
        }

        if signs[i] > 0 {
            for k in 0..3 {
                let mut d = pos[i * 3 + k] - ref_plus[k];
                d -= box_size * (d / box_size).round();
                sum_plus[k] += d;
            }
            n_plus += 1;
        } else {
            for k in 0..3 {
                let mut d = pos[i * 3 + k] - ref_minus[k];
                d -= box_size * (d / box_size).round();
                sum_minus[k] += d;
            }
            n_minus += 1;
        }
    }

    let mut com_plus = [0.0f64; 3];
    let mut com_minus = [0.0f64; 3];
    for k in 0..3 {
        com_plus[k] = (ref_plus[k] + sum_plus[k] / n_plus as f64).rem_euclid(box_size);
        com_minus[k] = (ref_minus[k] + sum_minus[k] / n_minus as f64).rem_euclid(box_size);
    }

    let mut d2 = 0.0;
    for k in 0..3 {
        let mut d = com_plus[k] - com_minus[k];
        d -= box_size * (d / box_size).round();
        d2 += d * d;
    }

    d2.sqrt() / box_size
}

#[cfg(feature = "cuda")]
fn compute_hubble(z: f64) -> f64 {
    let h0 = 70.0;
    let omega_m = 0.3;
    let omega_lambda = 0.7;
    h0 * (omega_m * (1.0 + z).powi(3) + omega_lambda).sqrt()
}

#[cfg(feature = "cuda")]
fn compute_redshift_from_time(t_gyr: f64, z_init: f64) -> f64 {
    let t_to_z0 = 4.5;
    let z = z_init * (1.0 - t_gyr / t_to_z0);
    z.max(0.0)
}

#[cfg(feature = "cuda")]
fn save_snapshot(path: &str, pos: &[f64], signs: &[i32], z: f64, box_size: f64) {
    let file = File::create(path).unwrap();
    let mut w = BufWriter::new(file);
    let n = pos.len() / 3;

    w.write_all(b"JSNP").unwrap();
    w.write_all(&2u32.to_le_bytes()).unwrap();
    w.write_all(&(n as u64).to_le_bytes()).unwrap();
    w.write_all(&z.to_le_bytes()).unwrap();
    w.write_all(&box_size.to_le_bytes()).unwrap();

    for i in 0..n {
        w.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        w.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        w.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        w.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
        w.write_all(&0u8.to_le_bytes()).unwrap();
    }

    w.flush().unwrap();
}

#[cfg(feature = "cuda")]
fn apply_sph_kick(
    gpu_sim: &mut GpuNBodySimulation,
    sph_plus: &mut GpuSphPressure,
    sph_minus: &mut GpuSphPressure,
    temp_plus: &[f64],
    temp_minus: &[f64],
    dt: f64,
) {
    let pos = gpu_sim.get_positions().expect("Failed to get positions");
    let mut vel = gpu_sim.get_velocities().expect("Failed to get velocities");
    let signs = gpu_sim.get_signs().expect("Failed to get signs");
    let n = signs.len();

    let mut idx_plus: Vec<usize> = Vec::with_capacity(N_PLUS);
    let mut idx_minus: Vec<usize> = Vec::with_capacity(N_MINUS);

    for i in 0..n {
        if signs[i] > 0 {
            idx_plus.push(i);
        } else {
            idx_minus.push(i);
        }
    }

    let mut pos_plus = vec![0.0f64; idx_plus.len() * 3];
    for (j, &i) in idx_plus.iter().enumerate() {
        pos_plus[j * 3] = pos[i * 3];
        pos_plus[j * 3 + 1] = pos[i * 3 + 1];
        pos_plus[j * 3 + 2] = pos[i * 3 + 2];
    }

    let mut pos_minus = vec![0.0f64; idx_minus.len() * 3];
    for (j, &i) in idx_minus.iter().enumerate() {
        pos_minus[j * 3] = pos[i * 3];
        pos_minus[j * 3 + 1] = pos[i * 3 + 1];
        pos_minus[j * 3 + 2] = pos[i * 3 + 2];
    }

    let acc_plus = sph_plus.compute_pressure_accelerations(&pos_plus, temp_plus)
        .expect("SPH+ computation failed");

    let acc_minus = sph_minus.compute_pressure_accelerations(&pos_minus, temp_minus)
        .expect("SPH- computation failed");

    for (j, &i) in idx_plus.iter().enumerate() {
        vel[i * 3] += acc_plus[j * 3] * dt;
        vel[i * 3 + 1] += acc_plus[j * 3 + 1] * dt;
        vel[i * 3 + 2] += acc_plus[j * 3 + 2] * dt;
    }

    for (j, &i) in idx_minus.iter().enumerate() {
        vel[i * 3] += acc_minus[j * 3] * dt;
        vel[i * 3 + 1] += acc_minus[j * 3 + 1] * dt;
        vel[i * 3 + 2] += acc_minus[j * 3 + 2] * dt;
    }

    gpu_sim.set_velocities(&vel).expect("Failed to set velocities");
}
