//! VSL Phase 2 Production — NO SPH (asymmetric softening only)
//!
//! Uses asymmetric softening: ε_minus = 5 × ε_plus
//! This provides pressure-like support for m- without O(N²) SPH cost
//! 10M particles, 500 Mpc, 30000 steps

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

// Phase 2 Parameters
const N_PLUS: usize = 5_110_024;
const N_MINUS: usize = 4_889_976;
const BOX_SIZE: f64 = 500.0;  // Mpc
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;  // Gyr
const STEPS: usize = 30_000;
const CSV_INTERVAL: usize = 5;
const SNAPSHOT_INTERVAL: usize = 10;  // 3000 snapshots for 30000 steps

// Softening: asymmetric via kernel (SOFTENING_MINUS_RATIO = 5.0)
const SOFTENING: f64 = 1.0;  // Mpc for m+, 5.0 Mpc for m- (via kernel)

// Warning thresholds
const RATIO_WARNING: f64 = 2.0;
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
    let output_dir = "/app/output/vsl_phase2_nosph";
    create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2 — NO SPH (asymmetric softening only)                    ║");
    println!("║  ε_plus = 1.0 Mpc | ε_minus = 5.0 Mpc (×5)                       ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)                    ║", N_PLUS + N_MINUS, N_PLUS, N_MINUS);
    println!("║  Box = {} Mpc, z_init = {}, η = {}                          ║", BOX_SIZE, Z_INIT, ETA);
    println!("║  VSL dynamique : c_ratio²(z) = (1+z)^δ, δ = 0.0431              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let delta = (ETA - 1.0) / ETA;
    println!("δ = (η-1)/η = {:.4}", delta);
    println!("c_ratio(z=4) = {:.6}", (1.0 + Z_INIT).powf(delta / 2.0));
    println!();

    // CSV
    let csv_path = format!("{}/evolution.csv", output_dir);
    let csv_file = File::create(&csv_path).unwrap();
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,a,c_ratio,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,ratio,ratio_trend,seg,warning").unwrap();

    // Log
    let log_path = format!("{}/simulation.log", output_dir);
    let log_file = File::create(&log_path).unwrap();
    let mut log_writer = BufWriter::new(log_file);
    writeln!(log_writer, "VSL Phase 2 — NO SPH (asymmetric softening)").unwrap();
    writeln!(log_writer, "N={} ({}+ / {}-), Box={}Mpc", N_PLUS+N_MINUS, N_PLUS, N_MINUS, BOX_SIZE).unwrap();
    writeln!(log_writer, "Softening: m+ = {} Mpc, m- = {} Mpc (×5)", SOFTENING, SOFTENING * 5.0).unwrap();

    // GPU simulation
    println!("Initializing GPU simulation (10M particles)...");
    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(SOFTENING);
    // Note: asymmetric softening (×5 for m-) is handled in CUDA kernel
    // via SOFTENING_MINUS_RATIO = 5.0

    println!("Starting Phase 2 simulation (NO SPH)...\n");

    let start = Instant::now();
    let mut current_z = Z_INIT;
    let mut ratio_history: Vec<f64> = Vec::with_capacity(10);

    println!(" step  |   t    |   z    |  a    | c_ratio | ρ+_max | ρ-_max | v_rms+ | v_rms- | ratio | trend | warning");
    println!("-------|--------|--------|-------|---------|--------|--------|--------|--------|-------|-------|--------");

    for step in 0..=STEPS {
        // Metrics every CSV_INTERVAL steps
        if step % CSV_INTERVAL == 0 {
            let pos = gpu_sim.get_positions().expect("get_positions");
            let vel = gpu_sim.get_velocities().expect("get_velocities");
            let signs = gpu_sim.get_signs().expect("get_signs");

            let (rho_plus_max, rho_minus_max) = compute_density_max(&pos, &signs, BOX_SIZE);
            let (v_rms_plus, v_rms_minus) = compute_vrms(&vel, &signs);
            let seg = compute_segregation(&pos, &signs, BOX_SIZE);

            let ratio = if v_rms_plus > 0.0 && v_rms_minus > 0.0 {
                (v_rms_minus / v_rms_plus).max(v_rms_plus / v_rms_minus)
            } else {
                1.0
            };

            // Update ratio history and compute trend (average of last 10)
            ratio_history.push(ratio);
            if ratio_history.len() > 10 {
                ratio_history.remove(0);
            }
            let ratio_trend: f64 = ratio_history.iter().sum::<f64>() / ratio_history.len() as f64;

            let a = 1.0 / (1.0 + current_z);
            let c_ratio = (1.0 + current_z).powf(delta / 2.0);

            let mut warning = String::new();
            if ratio > RATIO_WARNING {
                warning = format!("drift ({:.2})", ratio);
            }
            if v_rms_minus > VRMS_WARNING || v_rms_plus > VRMS_WARNING {
                if !warning.is_empty() { warning.push_str(" + "); }
                warning.push_str(&format!("v élevée"));
            }

            let elapsed = start.elapsed().as_secs_f64() / 3600.0;
            let eta = if step > 0 { elapsed * (STEPS - step) as f64 / step as f64 } else { 0.0 };

            if step % 50 == 0 {
                println!("{:6} | {:6.3} | {:6.4} | {:5.4} | {:7.5} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:5.3} | {:5.3} | {} ({:.1}h/{:.1}h)",
                    step, step as f64 * DT, current_z, a, c_ratio,
                    rho_plus_max, rho_minus_max,
                    v_rms_plus, v_rms_minus,
                    ratio, ratio_trend, if warning.is_empty() { "OK" } else { "⚠" },
                    elapsed, eta);
            }

            writeln!(csv_writer, "{},{:.4},{:.6},{:.6},{:.6},{:.1},{:.1},{:.1},{:.1},{:.4},{:.4},{:.6},{}",
                step, step as f64 * DT, current_z, a, c_ratio,
                rho_plus_max, rho_minus_max,
                v_rms_plus, v_rms_minus,
                ratio, ratio_trend, seg, warning).unwrap();

            if step % 100 == 0 {
                csv_writer.flush().unwrap();
            }
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&gpu_sim, step, current_z, BOX_SIZE, output_dir);
        }

        if step == STEPS { break; }

        // Update c_ratio for VSL
        let c_ratio_sq = (1.0 + current_z).powf(delta);
        gpu_sim.set_c_ratio(c_ratio_sq.sqrt());

        // Update z and compute Hubble
        let a = 1.0 / (1.0 + current_z);
        let h_z = 70.0 * (0.3 * (1.0 + current_z).powi(3) + 0.7).sqrt();
        let h_gyr = h_z / 977.8;

        // Step simulation using GPU-only method (NO SPH - just gravity with asymmetric softening)
        let _ = gpu_sim.step_with_expansion_dkd_gpu(DT, a, h_gyr, 0.013205);

        // Update z
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
fn compute_density_max(pos: &[f64], signs: &[i32], box_size: f64) -> (f64, f64) {
    const N_CELLS: usize = 64;
    let mut grid_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
    let mut grid_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];

    let n = signs.len();
    for i in 0..n {
        let x = pos[i * 3];
        let y = pos[i * 3 + 1];
        let z = pos[i * 3 + 2];

        let ix = ((x / box_size + 0.5) * N_CELLS as f64).floor() as usize % N_CELLS;
        let iy = ((y / box_size + 0.5) * N_CELLS as f64).floor() as usize % N_CELLS;
        let iz = ((z / box_size + 0.5) * N_CELLS as f64).floor() as usize % N_CELLS;
        let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;

        if signs[i] > 0 {
            grid_plus[idx] += 1;
        } else {
            grid_minus[idx] += 1;
        }
    }

    let max_plus = *grid_plus.iter().max().unwrap_or(&0) as f64;
    let max_minus = *grid_minus.iter().max().unwrap_or(&0) as f64;
    (max_plus, max_minus)
}

#[cfg(feature = "cuda")]
fn compute_vrms(vel: &[f64], signs: &[i32]) -> (f64, f64) {
    let mut sum_plus = 0.0;
    let mut sum_minus = 0.0;
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;

    for i in 0..signs.len() {
        let vx = vel[i * 3];
        let vy = vel[i * 3 + 1];
        let vz = vel[i * 3 + 2];
        let v2 = vx * vx + vy * vy + vz * vz;

        if signs[i] > 0 {
            sum_plus += v2;
            n_plus += 1;
        } else {
            sum_minus += v2;
            n_minus += 1;
        }
    }

    let v_rms_plus = if n_plus > 0 { (sum_plus / n_plus as f64).sqrt() * 977.8 } else { 0.0 };
    let v_rms_minus = if n_minus > 0 { (sum_minus / n_minus as f64).sqrt() * 977.8 } else { 0.0 };
    (v_rms_plus, v_rms_minus)
}

#[cfg(feature = "cuda")]
fn compute_segregation(pos: &[f64], signs: &[i32], box_size: f64) -> f64 {
    let n = signs.len();
    let half = box_size / 2.0;

    let (mut sx_p, mut sy_p, mut sz_p) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sx_m, mut sy_m, mut sz_m) = (0.0f64, 0.0f64, 0.0f64);
    let (mut n_p, mut n_m) = (0usize, 0usize);

    let ref_x = pos[0];
    let ref_y = pos[1];
    let ref_z = pos[2];

    for i in 0..n {
        let mut dx = pos[i * 3] - ref_x;
        let mut dy = pos[i * 3 + 1] - ref_y;
        let mut dz = pos[i * 3 + 2] - ref_z;

        if dx > half { dx -= box_size; } else if dx < -half { dx += box_size; }
        if dy > half { dy -= box_size; } else if dy < -half { dy += box_size; }
        if dz > half { dz -= box_size; } else if dz < -half { dz += box_size; }

        if signs[i] > 0 {
            sx_p += dx; sy_p += dy; sz_p += dz;
            n_p += 1;
        } else {
            sx_m += dx; sy_m += dy; sz_m += dz;
            n_m += 1;
        }
    }

    if n_p == 0 || n_m == 0 { return 0.0; }

    let com_p = (sx_p / n_p as f64, sy_p / n_p as f64, sz_p / n_p as f64);
    let com_m = (sx_m / n_m as f64, sy_m / n_m as f64, sz_m / n_m as f64);

    let mut dx = com_p.0 - com_m.0;
    let mut dy = com_p.1 - com_m.1;
    let mut dz = com_p.2 - com_m.2;
    if dx > half { dx -= box_size; } else if dx < -half { dx += box_size; }
    if dy > half { dy -= box_size; } else if dy < -half { dy += box_size; }
    if dz > half { dz -= box_size; } else if dz < -half { dz += box_size; }

    (dx * dx + dy * dy + dz * dz).sqrt() / box_size
}

#[cfg(feature = "cuda")]
fn save_snapshot(gpu_sim: &GpuNBodySimulation, step: usize, z: f64, box_size: f64, output_dir: &str) {
    use std::path::Path;

    let snap_path = Path::new(output_dir).join("snapshots").join(format!("snap_{:06}.bin", step));

    let pos = gpu_sim.get_positions().expect("get_positions");
    let _vel = gpu_sim.get_velocities().expect("get_velocities");
    let signs = gpu_sim.get_signs().expect("get_signs");
    let n = signs.len();

    let mut file = BufWriter::new(File::create(&snap_path).expect("create snapshot"));

    // Header: magic(4) + version(4) + n(8) + z(8) + box(8) = 32 bytes
    file.write_all(b"JSNP").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(n as u64).to_le_bytes()).unwrap();
    file.write_all(&z.to_le_bytes()).unwrap();
    file.write_all(&box_size.to_le_bytes()).unwrap();

    // Per-particle: x,y,z (f64×3) + sign(i8) + type(u8) = 26 bytes
    for i in 0..n {
        file.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        file.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
        file.write_all(&0u8.to_le_bytes()).unwrap();  // type = 0 (dark matter)
    }

    file.flush().unwrap();
}
