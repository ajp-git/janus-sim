//! VSL SPH Bimetric - Option A v2 (T_floor asymétrique)
//!
//! Modifications v2:
//! - T_floor_plus  = 100 K (permet refroidissement m+)
//! - T_floor_minus = 1e4 K (garde m- chaud, empêche effondrement)
//!
//! Hypothèse: T_floor élevé pour m- freine son effondrement,
//! stabilisant le ratio v_rms-/v_rms+.

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

// Phase 0 Parameters (same as Option B for comparison)
const N_PLUS: usize = 250_000;
const N_MINUS: usize = 250_000;
const BOX_SIZE: f64 = 100.0;
const MU: f64 = 19.0;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;
const STEPS: usize = 5000;
const CSV_INTERVAL: usize = 5;

// SPH parameters (v2: asymmetric T_floor)
const T_INIT: f64 = 10000.0;
const T_FLOOR_PLUS: f64 = 100.0;    // m+ peut refroidir
const T_FLOOR_MINUS: f64 = 10000.0; // m- reste chaud (×100)
const PARTICLE_MASS: f64 = 1e10;

// GO/NO-GO thresholds
const RATIO_MAX: f64 = 1.10;
const DENSITY_RATIO_MAX: f64 = 10.0;
const SEG_MIN: f64 = 0.001;
const VRMS_CRITICAL: f64 = 100_000.0;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: --features cuda required");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_simulation();
}

#[cfg(feature = "cuda")]
fn run_simulation() {
    let output_dir = "/app/output/vsl_bimetric_optionA";
    create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  OPTION A : Forces symétriques (pas d'asymétrie VSL)             ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)                          ║", N_PLUS + N_MINUS, N_PLUS, N_MINUS);
    println!("║  Box = {} Mpc, z_init = {}, steps = {}                      ║", BOX_SIZE, Z_INIT, STEPS);
    println!("║  SPH: T_init={} K, T_floor: m+={} K, m-={} K               ║", T_INIT as i32, T_FLOOR_PLUS as i32, T_FLOOR_MINUS as i32);
    println!("║  Force: interaction = (same sign) ? +1.0 : -1.0 [symétrique]   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // CSV
    let csv_path = format!("{}/evolution_optionA.csv", output_dir);
    let csv_file = File::create(&csv_path).unwrap();
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,ratio_vrms,seg,c_ratio_sq,go_status").unwrap();

    // Log
    let log_path = format!("{}/simulation.log", output_dir);
    let log_file = File::create(&log_path).unwrap();
    let mut log_writer = BufWriter::new(log_file);
    writeln!(log_writer, "VSL Bimetric Option A — Forces symétriques").unwrap();

    let mut all_criteria_met = true;
    let mut failure_step = 0;
    let mut failure_reason = String::new();

    // NOTE: c_ratio still computed for reference but NOT used in forces
    // With symmetric forces, we set c_ratio = 1.0 always
    let c_ratio_init = 1.0;  // OPTION A: symmetric
    println!("\nc_ratio = 1.0 (forces symétriques, pas d'asymétrie VSL)");

    // GPU simulation
    println!("Initializing GPU simulation...");
    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(0.5);
    gpu_sim.set_c_ratio(c_ratio_init);  // c_ratio = 1.0 for symmetric forces

    // SPH
    println!("Initializing SPH pressure modules...");
    let device = Arc::new(CudaDevice::new(0).expect("Failed to create CUDA device"));

    let mut sph_plus = GpuSphPressure::new(
        Arc::clone(&device), N_PLUS, PARTICLE_MASS, BOX_SIZE,
    ).expect("Failed to create SPH+ module");

    let mut sph_minus = GpuSphPressure::new(
        Arc::clone(&device), N_MINUS, PARTICLE_MASS, BOX_SIZE,
    ).expect("Failed to create SPH- module");

    let temp_plus = vec![T_INIT.max(T_FLOOR_PLUS); N_PLUS];
    let temp_minus = vec![T_INIT.max(T_FLOOR_MINUS); N_MINUS];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SPH actif : m+ ✓ m- ✓                                       ║");
    println!("║  c_ratio = 1.0 (symmetric forces)                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nStarting Option A simulation...\n");
    println!(" step |   t    |   z    | ρ+_max | ρ-_max | v_rms+ | v_rms- | ratio | seg    | GO");
    println!("------|--------|--------|--------|--------|--------|--------|-------|--------|----");

    let start = Instant::now();
    let mut current_z = Z_INIT;
    let mut final_seg = 0.0;

    for step in 0..=STEPS {
        // Compute metrics at intervals
        if step % CSV_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().expect("get_positions failed");
            let vel = gpu_sim.get_velocities().expect("get_velocities failed");
            let signs = gpu_sim.get_signs().expect("get_signs failed");

            let (v_rms_plus, v_rms_minus) = compute_vrms_by_sign(&vel, &signs);
            let (rho_plus_max, rho_minus_max) = compute_max_densities(&pos, &signs, BOX_SIZE, 64);
            let seg = compute_segregation(&pos, &signs, BOX_SIZE);
            final_seg = seg;

            let ratio_vrms = if v_rms_plus > 0.0 { v_rms_minus / v_rms_plus } else { 1.0 };
            let ratio_vrms_inv = if v_rms_minus > 0.0 { v_rms_plus / v_rms_minus } else { 1.0 };
            let ratio_display = ratio_vrms.max(ratio_vrms_inv);

            // c_ratio_sq for reference only (not used in Option A forces)
            let c_ratio_sq_ref = CoupledFriedmann::c_ratio_sq_at_z(current_z, ETA);

            // GO/NO-GO
            let mut go_status = "GO";

            if ratio_display > RATIO_MAX {
                go_status = "NO-GO:ratio";
                if all_criteria_met {
                    all_criteria_met = false;
                    failure_step = step;
                    failure_reason = format!("ratio={:.4} > {}", ratio_display, RATIO_MAX);
                }
            }

            if rho_minus_max / rho_plus_max.max(1.0) > DENSITY_RATIO_MAX {
                go_status = "NO-GO:ρ";
                if all_criteria_met {
                    all_criteria_met = false;
                    failure_step = step;
                    failure_reason = format!("ρ-/ρ+={:.2} > {}", rho_minus_max/rho_plus_max, DENSITY_RATIO_MAX);
                }
            }

            // Auto-stop
            if v_rms_minus > VRMS_CRITICAL || v_rms_plus > VRMS_CRITICAL {
                println!("\n⚠️  AUTO-STOP: v_rms critical ({:.0}/{:.0} km/s)", v_rms_plus, v_rms_minus);
                writeln!(log_writer, "AUTO-STOP at step {}: v_rms critical", step).unwrap();
                break;
            }

            let elapsed = start.elapsed().as_secs_f64() / 3600.0;
            let eta = if step > 0 { elapsed * (STEPS - step) as f64 / step as f64 } else { 0.0 };

            println!("{:6} | {:6.3} | {:6.4} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:5.3} | {:6.4} | {} ({:.1}h, ETA {:.1}h)",
                step, step as f64 * DT, current_z,
                rho_plus_max, rho_minus_max,
                v_rms_plus, v_rms_minus,
                ratio_display, seg, go_status, elapsed, eta);

            writeln!(csv_writer, "{},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.4},{:.6},{:.6},{}",
                step, step as f64 * DT, current_z,
                rho_plus_max, rho_minus_max,
                v_rms_plus, v_rms_minus,
                ratio_display, seg, c_ratio_sq_ref, go_status).unwrap();
            csv_writer.flush().unwrap();

            // Save snapshot every 100 steps
            if step % 100 == 0 {
                save_snapshot(&gpu_sim, step, current_z, BOX_SIZE, output_dir);
            }
        }

        if step == STEPS { break; }

        // Apply SPH kick
        apply_sph_kick(&mut gpu_sim, &mut sph_plus, &mut sph_minus, &temp_plus, &temp_minus, DT);

        // Step with c_ratio = 1.0 (symmetric)
        // Note: we DON'T update c_ratio, keeping it at 1.0 for symmetric forces
        let _ = gpu_sim.step(DT);

        // Update z
        let a = 1.0 / (1.0 + current_z);
        let h_z = 70.0 * (0.3 * (1.0 + current_z).powi(3) + 0.7).sqrt();
        let h_gyr = h_z / 977.8;
        let da = h_gyr * DT * a;
        let new_a = a + da;
        current_z = 1.0 / new_a - 1.0;
    }

    // Final result
    let result_path = format!("{}/optionA_result.txt", output_dir);
    let mut result_file = File::create(&result_path).unwrap();

    if all_criteria_met && final_seg > SEG_MIN {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  ✅ OPTION A : GO — All criteria met                             ║");
        println!("║  Segregation = {:.4} > {} ✓                                ║", final_seg, SEG_MIN);
        println!("╚══════════════════════════════════════════════════════════════════╝");
        writeln!(result_file, "RESULT=GO").unwrap();
        writeln!(result_file, "FINAL_SEG={:.6}", final_seg).unwrap();
    } else {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  ❌ OPTION A : NO-GO                                             ║");
        if !all_criteria_met {
            println!("║  Failure at step {}: {}              ║", failure_step, failure_reason);
        }
        if final_seg <= SEG_MIN {
            println!("║  Segregation = {:.4} <= {} (no structure)            ║", final_seg, SEG_MIN);
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");
        writeln!(result_file, "RESULT=NO-GO").unwrap();
        if !all_criteria_met {
            writeln!(result_file, "FAILURE_STEP={}", failure_step).unwrap();
            writeln!(result_file, "FAILURE_REASON={}", failure_reason).unwrap();
        }
    }

    writeln!(log_writer, "\nSimulation complete").unwrap();
}

#[cfg(feature = "cuda")]
fn compute_vrms_by_sign(vel: &[f64], signs: &[i32]) -> (f64, f64) {
    let n = signs.len();
    let mut sum_v2_plus = 0.0;
    let mut sum_v2_minus = 0.0;
    let mut n_plus = 0;
    let mut n_minus = 0;

    for i in 0..n {
        let v2 = vel[i*3].powi(2) + vel[i*3+1].powi(2) + vel[i*3+2].powi(2);
        if signs[i] > 0 {
            sum_v2_plus += v2;
            n_plus += 1;
        } else {
            sum_v2_minus += v2;
            n_minus += 1;
        }
    }

    let v_rms_plus = if n_plus > 0 { (sum_v2_plus / n_plus as f64).sqrt() * 977.8 } else { 0.0 };
    let v_rms_minus = if n_minus > 0 { (sum_v2_minus / n_minus as f64).sqrt() * 977.8 } else { 0.0 };
    (v_rms_plus, v_rms_minus)
}

#[cfg(feature = "cuda")]
fn compute_max_densities(pos: &[f64], signs: &[i32], box_size: f64, n_grid: usize) -> (f64, f64) {
    let n = signs.len();
    let mut grid_plus = vec![0u32; n_grid * n_grid * n_grid];
    let mut grid_minus = vec![0u32; n_grid * n_grid * n_grid];

    for i in 0..n {
        let ix = ((pos[i*3] / box_size + 0.5) * n_grid as f64).floor() as usize % n_grid;
        let iy = ((pos[i*3+1] / box_size + 0.5) * n_grid as f64).floor() as usize % n_grid;
        let iz = ((pos[i*3+2] / box_size + 0.5) * n_grid as f64).floor() as usize % n_grid;
        let idx = ix * n_grid * n_grid + iy * n_grid + iz;

        if signs[i] > 0 {
            grid_plus[idx] += 1;
        } else {
            grid_minus[idx] += 1;
        }
    }

    (*grid_plus.iter().max().unwrap_or(&0) as f64, *grid_minus.iter().max().unwrap_or(&0) as f64)
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
            ref_plus = [pos[i*3], pos[i*3+1], pos[i*3+2]];
        }
        if signs[i] < 0 && n_minus == 0 {
            ref_minus = [pos[i*3], pos[i*3+1], pos[i*3+2]];
        }

        if signs[i] > 0 {
            for k in 0..3 {
                let mut d = pos[i*3+k] - ref_plus[k];
                if d > box_size / 2.0 { d -= box_size; }
                if d < -box_size / 2.0 { d += box_size; }
                sum_plus[k] += d;
            }
            n_plus += 1;
        } else {
            for k in 0..3 {
                let mut d = pos[i*3+k] - ref_minus[k];
                if d > box_size / 2.0 { d -= box_size; }
                if d < -box_size / 2.0 { d += box_size; }
                sum_minus[k] += d;
            }
            n_minus += 1;
        }
    }

    let com_plus = [
        ref_plus[0] + sum_plus[0] / n_plus as f64,
        ref_plus[1] + sum_plus[1] / n_plus as f64,
        ref_plus[2] + sum_plus[2] / n_plus as f64,
    ];
    let com_minus = [
        ref_minus[0] + sum_minus[0] / n_minus as f64,
        ref_minus[1] + sum_minus[1] / n_minus as f64,
        ref_minus[2] + sum_minus[2] / n_minus as f64,
    ];

    let mut d2 = 0.0;
    for k in 0..3 {
        let mut d = com_plus[k] - com_minus[k];
        if d > box_size / 2.0 { d -= box_size; }
        if d < -box_size / 2.0 { d += box_size; }
        d2 += d * d;
    }
    d2.sqrt() / box_size
}

#[cfg(feature = "cuda")]
fn save_snapshot(
    gpu_sim: &GpuNBodySimulation,
    step: usize, z: f64, box_size: f64, output_dir: &str
) {
    let pos = gpu_sim.get_positions().expect("get_positions");
    let signs = gpu_sim.get_signs().expect("get_signs");
    let n = signs.len();

    let path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
    let mut w = std::io::BufWriter::new(File::create(&path).unwrap());

    w.write_all(b"JSNP").unwrap();
    w.write_all(&2u32.to_le_bytes()).unwrap();
    w.write_all(&(n as u64).to_le_bytes()).unwrap();
    w.write_all(&z.to_le_bytes()).unwrap();
    w.write_all(&box_size.to_le_bytes()).unwrap();

    for i in 0..n {
        w.write_all(&pos[i*3].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+1].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+2].to_le_bytes()).unwrap();
        w.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
        w.write_all(&0u8.to_le_bytes()).unwrap();
    }
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
    let pos = gpu_sim.get_positions().expect("get_positions");
    let mut vel = gpu_sim.get_velocities().expect("get_velocities");
    let signs = gpu_sim.get_signs().expect("get_signs");
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

    // Extract m+ positions
    let mut pos_plus = Vec::with_capacity(idx_plus.len() * 3);
    for &i in &idx_plus {
        pos_plus.extend_from_slice(&pos[i*3..i*3+3]);
    }

    // Extract m- positions
    let mut pos_minus = Vec::with_capacity(idx_minus.len() * 3);
    for &i in &idx_minus {
        pos_minus.extend_from_slice(&pos[i*3..i*3+3]);
    }

    // Compute SPH accelerations
    let acc_plus = sph_plus.compute_pressure_accelerations(&pos_plus, temp_plus)
        .expect("SPH+ compute failed");
    let acc_minus = sph_minus.compute_pressure_accelerations(&pos_minus, temp_minus)
        .expect("SPH- compute failed");

    // Apply kicks to m+
    for (j, &i) in idx_plus.iter().enumerate() {
        vel[i*3] += acc_plus[j*3] * dt;
        vel[i*3+1] += acc_plus[j*3+1] * dt;
        vel[i*3+2] += acc_plus[j*3+2] * dt;
    }

    // Apply kicks to m-
    for (j, &i) in idx_minus.iter().enumerate() {
        vel[i*3] += acc_minus[j*3] * dt;
        vel[i*3+1] += acc_minus[j*3+1] * dt;
        vel[i*3+2] += acc_minus[j*3+2] * dt;
    }

    gpu_sim.set_velocities(&vel).expect("set_velocities failed");
}
