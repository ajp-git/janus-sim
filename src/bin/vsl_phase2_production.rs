//! VSL Phase 2 Production — Configuration vsl_petit_production
//!
//! SPH m+ uniquement | m- gravité pure | VSL dynamique
//! 10M particules, 500 Mpc, 30000 steps

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

// Phase 2 Parameters
const N_PLUS: usize = 5_110_024;
const N_MINUS: usize = 4_889_976;
const BOX_SIZE: f64 = 500.0;  // Mpc
const MU: f64 = 19.0;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;  // Gyr
const STEPS: usize = 30_000;
const CSV_INTERVAL: usize = 5;
const SNAPSHOT_INTERVAL: usize = 10;
const FRAME_INTERVAL: usize = 5;

// SPH parameters (m+ only)
const T_INIT: f64 = 10000.0;
const T_FLOOR: f64 = 100.0;
const PARTICLE_MASS: f64 = 1e10;

// Warning thresholds (no auto-stop)
const RATIO_WARNING: f64 = 1.5;
const VRMS_WARNING: f64 = 50_000.0;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: --features cuda required");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_phase2();
}

#[cfg(feature = "cuda")]
fn run_phase2() {
    let output_dir = "/app/output/vsl_phase2";
    create_dir_all(format!("{}/snapshots", output_dir)).unwrap();
    create_dir_all(format!("{}/frames", output_dir)).unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2 lancée — configuration vsl_petit_production             ║");
    println!("║  SPH m+ uniquement | m- gravité pure | 30000 steps               ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)                    ║", N_PLUS + N_MINUS, N_PLUS, N_MINUS);
    println!("║  Box = {} Mpc, z_init = {}, μ = {}, η = {}                 ║", BOX_SIZE, Z_INIT, MU, ETA);
    println!("║  SPH m+ : T_init = {} K, T_floor = {} K                       ║", T_INIT as i32, T_FLOOR as i32);
    println!("║  m- : gravité pure (pas de SPH)                                  ║");
    println!("║  VSL dynamique : c_ratio²(z) = (1+z)^δ, δ = (η-1)/η = 0.0431    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // CSV
    let csv_path = format!("{}/evolution_phase2.csv", output_dir);
    let csv_file = File::create(&csv_path).unwrap();
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,a,c_ratio,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,ratio,seg,warning").unwrap();

    // Log
    let log_path = format!("{}/simulation.log", output_dir);
    let log_file = File::create(&log_path).unwrap();
    let mut log_writer = BufWriter::new(log_file);
    writeln!(log_writer, "VSL Phase 2 Production — vsl_petit_production config").unwrap();
    writeln!(log_writer, "N={} ({}+ / {}-), Box={}Mpc", N_PLUS+N_MINUS, N_PLUS, N_MINUS, BOX_SIZE).unwrap();

    // VSL c_ratio
    let delta = (ETA - 1.0) / ETA;  // 0.0431
    let c_ratio_sq_init = (1.0 + Z_INIT).powf(delta);
    let c_ratio_init = c_ratio_sq_init.sqrt();
    println!("δ = (η-1)/η = {:.4}", delta);
    println!("c_ratio(z={}) = {:.6}\n", Z_INIT, c_ratio_init);

    // GPU simulation
    println!("Initializing GPU simulation (10M particles)...");
    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(1.0);  // Larger softening for 500 Mpc box
    gpu_sim.set_c_ratio(c_ratio_init);

    // SPH for m+ only
    println!("Initializing SPH for m+ only...");
    let device = Arc::new(CudaDevice::new(0).expect("Failed to create CUDA device"));
    let mut sph_plus = GpuSphPressure::new(
        Arc::clone(&device), N_PLUS, PARTICLE_MASS, BOX_SIZE,
    ).expect("Failed to create SPH+ module");

    let temp_plus = vec![T_INIT.max(T_FLOOR); N_PLUS];

    println!("Starting Phase 2 simulation...\n");
    println!(" step  |   t    |   z    |  a    | c_ratio | ρ+_max | ρ-_max | v_rms+ | v_rms- | ratio | warning");
    println!("-------|--------|--------|-------|---------|--------|--------|--------|--------|-------|--------");

    let start = Instant::now();
    let mut current_z = Z_INIT;

    for step in 0..=STEPS {
        // Compute metrics at intervals
        if step % CSV_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().expect("get_positions");
            let vel = gpu_sim.get_velocities().expect("get_velocities");
            let signs = gpu_sim.get_signs().expect("get_signs");

            let (v_rms_plus, v_rms_minus) = compute_vrms(&vel, &signs);
            let (rho_plus_max, rho_minus_max) = compute_max_densities(&pos, &signs, BOX_SIZE, 64);
            let seg = compute_segregation(&pos, &signs, BOX_SIZE);

            let ratio = if v_rms_plus > 0.0 { v_rms_minus / v_rms_plus } else { 1.0 };
            let a = 1.0 / (1.0 + current_z);
            let c_ratio = (1.0 + current_z).powf(delta / 2.0);

            // Warnings (no auto-stop)
            let mut warning = String::new();
            if ratio > RATIO_WARNING {
                warning = format!("drift m- ({:.2})", ratio);
                if step % 100 == 0 {
                    println!("⚠️  drift m- : ratio = {:.3}", ratio);
                }
            }
            if v_rms_minus > VRMS_WARNING {
                if !warning.is_empty() { warning.push_str(" + "); }
                warning.push_str(&format!("v- élevée ({:.0})", v_rms_minus));
                if step % 100 == 0 {
                    println!("⚠️  vitesse élevée m- : {:.0} km/s", v_rms_minus);
                }
            }

            let elapsed = start.elapsed().as_secs_f64() / 3600.0;
            let eta = if step > 0 { elapsed * (STEPS - step) as f64 / step as f64 } else { 0.0 };

            if step % 50 == 0 {
                println!("{:6} | {:6.3} | {:6.4} | {:5.4} | {:7.5} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:5.3} | {} ({:.1}h/{:.1}h)",
                    step, step as f64 * DT, current_z, a, c_ratio,
                    rho_plus_max, rho_minus_max,
                    v_rms_plus, v_rms_minus,
                    ratio, if warning.is_empty() { "OK" } else { "⚠" },
                    elapsed, eta);
            }

            writeln!(csv_writer, "{},{:.4},{:.6},{:.6},{:.6},{:.1},{:.1},{:.1},{:.1},{:.4},{:.6},{}",
                step, step as f64 * DT, current_z, a, c_ratio,
                rho_plus_max, rho_minus_max,
                v_rms_plus, v_rms_minus,
                ratio, seg, warning).unwrap();

            if step % 100 == 0 {
                csv_writer.flush().unwrap();
            }
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&gpu_sim, step, current_z, BOX_SIZE, output_dir);
        }

        if step == STEPS { break; }

        // Apply SPH kick to m+ only
        apply_sph_kick_plus_only(&mut gpu_sim, &mut sph_plus, &temp_plus, DT);

        // Update c_ratio for VSL
        let c_ratio_sq = (1.0 + current_z).powf(delta);
        gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

        // Step simulation
        let _ = gpu_sim.step(DT);

        // Update z
        let a = 1.0 / (1.0 + current_z);
        let h_z = 70.0 * (0.3 * (1.0 + current_z).powi(3) + 0.7).sqrt();
        let h_gyr = h_z / 977.8;
        let da = h_gyr * DT * a;
        let new_a = a + da;
        current_z = 1.0 / new_a - 1.0;
    }

    csv_writer.flush().unwrap();

    let total_time = start.elapsed().as_secs_f64() / 3600.0;
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2 terminée — {:.1} heures                                  ║", total_time);
    println!("║  z final = {:.4}                                                ║", current_z);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    writeln!(log_writer, "\nSimulation complete: {:.1}h, z_final={:.4}", total_time, current_z).unwrap();

    // Write completion marker
    let result_path = format!("{}/phase2_complete.txt", output_dir);
    let mut result_file = File::create(&result_path).unwrap();
    writeln!(result_file, "COMPLETE").unwrap();
    writeln!(result_file, "z_final={:.6}", current_z).unwrap();
    writeln!(result_file, "time_hours={:.2}", total_time).unwrap();
}

#[cfg(feature = "cuda")]
fn compute_vrms(vel: &[f64], signs: &[i32]) -> (f64, f64) {
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
fn apply_sph_kick_plus_only(
    gpu_sim: &mut GpuNBodySimulation,
    sph_plus: &mut GpuSphPressure,
    temp_plus: &[f64],
    dt: f64,
) {
    let pos = gpu_sim.get_positions().expect("get_positions");
    let mut vel = gpu_sim.get_velocities().expect("get_velocities");
    let signs = gpu_sim.get_signs().expect("get_signs");
    let n = signs.len();

    // Extract m+ indices and positions
    let mut idx_plus: Vec<usize> = Vec::with_capacity(N_PLUS);
    for i in 0..n {
        if signs[i] > 0 {
            idx_plus.push(i);
        }
    }

    let mut pos_plus = Vec::with_capacity(idx_plus.len() * 3);
    for &i in &idx_plus {
        pos_plus.extend_from_slice(&pos[i*3..i*3+3]);
    }

    // Compute SPH accelerations for m+ only
    let acc_plus = sph_plus.compute_pressure_accelerations(&pos_plus, temp_plus)
        .expect("SPH+ compute failed");

    // Apply kicks to m+ only (m- gets no SPH kick)
    for (j, &i) in idx_plus.iter().enumerate() {
        vel[i*3] += acc_plus[j*3] * dt;
        vel[i*3+1] += acc_plus[j*3+1] * dt;
        vel[i*3+2] += acc_plus[j*3+2] * dt;
    }

    gpu_sim.set_velocities(&vel).expect("set_velocities failed");
}
