//! VSL SPH Bimetric Test
//!
//! Test run validating velocity balance between m+ and m- populations.
//! Auto-stop if velocity runaway detected.

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus_sim::nbody_gpu::GpuNBodySimulation;
use janus_sim::vsl_dynamic::CoupledFriedmann;

// Parameters
const N_PLUS: usize = 250_000;
const N_MINUS: usize = 250_000;
const BOX_SIZE: f64 = 100.0;  // Mpc
const MU: f64 = 19.0;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const DT: f64 = 0.001;  // Gyr
const STEPS: usize = 5000;
const SNAPSHOT_INTERVAL: usize = 500;
const CSV_INTERVAL: usize = 5;

// Auto-stop thresholds
const VRMS_RATIO_MAX: f64 = 1.5;
const VRMS_CRITICAL: f64 = 100_000.0;  // km/s

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: This binary requires --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_simulation();
}

#[cfg(feature = "cuda")]
fn run_simulation() {
    let output_dir = "/app/output/vsl_sph_bimetric_test";
    create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        VSL BIMETRIC TEST — μ={} η={:.3}                  ║", MU, ETA);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)                    ║", N_PLUS + N_MINUS, N_PLUS, N_MINUS);
    println!("║  Box = {} Mpc, z_init = {}                              ║", BOX_SIZE, Z_INIT);
    println!("║  dt = {} Gyr, steps = {}                             ║", DT, STEPS);
    println!("║  VSL dynamique: c_ratio²(z) = (1+z)^δ, δ = (η-1)/η         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Initialize CSV
    let csv_path = format!("{}/evolution_test.csv", output_dir);
    let csv_file = File::create(&csv_path).unwrap();
    let mut csv_writer = BufWriter::new(csv_file);
    writeln!(csv_writer, "step,t_Gyr,z,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,ratio_vrms,seg,c_ratio_sq").unwrap();

    // Initialize log
    let log_path = format!("{}/simulation.log", output_dir);
    let log_file = File::create(&log_path).unwrap();
    let mut log_writer = BufWriter::new(log_file);
    writeln!(log_writer, "VSL Bimetric Test — μ={} η={}", MU, ETA).unwrap();
    writeln!(log_writer, "N={} ({}+ / {}-), Box={} Mpc", N_PLUS + N_MINUS, N_PLUS, N_MINUS, BOX_SIZE).unwrap();
    writeln!(log_writer, "").unwrap();
    writeln!(log_writer, " step |   t    |   z    | ρ+_max | ρ-_max | v_rms+ | v_rms- | ratio | seg    | c²").unwrap();
    writeln!(log_writer, "------|--------|--------|--------|--------|--------|--------|-------|--------|------").unwrap();

    // Initial c_ratio
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    let c_ratio_init = c_ratio_sq_init.sqrt();
    println!("\nc_ratio(z={}) = {:.4} (c_ratio² = {:.6})", Z_INIT, c_ratio_init, c_ratio_sq_init);

    // Initialize GPU simulation
    println!("Initializing GPU simulation...");
    let mut gpu_sim = GpuNBodySimulation::new(N_PLUS, N_MINUS, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    gpu_sim.set_theta(0.7);
    gpu_sim.set_softening(0.5);
    gpu_sim.set_c_ratio(c_ratio_init);

    let start_time = Instant::now();
    let mut t_gyr = 0.0;
    let mut z = Z_INIT;
    let mut final_step = 0;

    println!("\nStarting simulation...\n");
    println!(" step |   t    |   z    | ρ+_max | ρ-_max | v_rms+ | v_rms- | ratio | seg    | c²     | status");
    println!("------|--------|--------|--------|--------|--------|--------|-------|--------|--------|-------");

    for step in 0..=STEPS {
        // Update c_ratio dynamically
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z.max(0.0), ETA);
        let c_ratio = c_ratio_sq.sqrt();
        gpu_sim.set_c_ratio(c_ratio);

        // Get current state
        let pos = gpu_sim.get_positions();
        let vel = gpu_sim.get_velocities();
        let signs = gpu_sim.get_signs();

        // Compute v_rms for each population
        let (v_rms_plus, v_rms_minus) = compute_vrms_by_sign(&vel, &signs);

        // Compute densities
        let (rho_plus_max, rho_minus_max) = compute_max_densities(&pos, &signs, BOX_SIZE, 64);

        // Compute segregation
        let seg = compute_segregation(&pos, &signs, BOX_SIZE);

        // Ratio v_rms-/v_rms+
        let ratio_vrms = if v_rms_plus > 0.0 { v_rms_minus / v_rms_plus } else { 1.0 };

        // Log every CSV_INTERVAL steps
        if step % CSV_INTERVAL == 0 {
            writeln!(csv_writer, "{},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.4},{:.6},{:.6}",
                     step, t_gyr, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, ratio_vrms, seg, c_ratio_sq).unwrap();

            let elapsed = start_time.elapsed().as_secs_f64() / 3600.0;
            let status = if ratio_vrms > 1.2 || ratio_vrms < 0.8 { "⚠" } else { "✓" };

            println!(" {:5} | {:.3} | {:.4} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:.3} | {:.4} | {:.4} | {} ({:.1}h)",
                     step, t_gyr, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, ratio_vrms, seg, c_ratio_sq, status, elapsed);

            writeln!(log_writer, " {:5} | {:.3} | {:.4} | {:6.0} | {:6.0} | {:6.0} | {:6.0} | {:.3} | {:.4} | {:.4}",
                     step, t_gyr, z, rho_plus_max, rho_minus_max,
                     v_rms_plus, v_rms_minus, ratio_vrms, seg, c_ratio_sq).unwrap();
        }

        // Check auto-stop conditions
        if ratio_vrms > VRMS_RATIO_MAX {
            println!("\n╔══════════════════════════════════════════════════╗");
            println!("║  STOP: RUNAWAY m- (v_rms-/v_rms+ = {:.3} > {})  ║", ratio_vrms, VRMS_RATIO_MAX);
            println!("╚══════════════════════════════════════════════════╝");
            writeln!(log_writer, "\nSTOP: RUNAWAY m- at step {} (ratio={:.3})", step, ratio_vrms).unwrap();
            final_step = step;
            break;
        }

        if ratio_vrms > 0.0 && 1.0 / ratio_vrms > VRMS_RATIO_MAX {
            println!("\n╔══════════════════════════════════════════════════╗");
            println!("║  STOP: RUNAWAY m+ (v_rms+/v_rms- = {:.3} > {})  ║", 1.0/ratio_vrms, VRMS_RATIO_MAX);
            println!("╚══════════════════════════════════════════════════╝");
            writeln!(log_writer, "\nSTOP: RUNAWAY m+ at step {} (ratio={:.3})", step, 1.0/ratio_vrms).unwrap();
            final_step = step;
            break;
        }

        if v_rms_minus > VRMS_CRITICAL {
            println!("\n╔══════════════════════════════════════════════════════════╗");
            println!("║  STOP: VITESSE CRITIQUE m- ({:.0} > {} km/s)      ║", v_rms_minus, VRMS_CRITICAL);
            println!("╚══════════════════════════════════════════════════════════╝");
            writeln!(log_writer, "\nSTOP: VITESSE CRITIQUE m- at step {} (v={:.0})", step, v_rms_minus).unwrap();
            final_step = step;
            break;
        }

        if v_rms_plus > VRMS_CRITICAL {
            println!("\n╔══════════════════════════════════════════════════════════╗");
            println!("║  STOP: VITESSE CRITIQUE m+ ({:.0} > {} km/s)      ║", v_rms_plus, VRMS_CRITICAL);
            println!("╚══════════════════════════════════════════════════════════╝");
            writeln!(log_writer, "\nSTOP: VITESSE CRITIQUE m+ at step {} (v={:.0})", step, v_rms_plus).unwrap();
            final_step = step;
            break;
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
            save_snapshot(&snap_path, &pos, &signs, z, BOX_SIZE);
            println!("  → Snapshot saved: snap_{:06}.bin", step);
        }

        // Integration step
        if step < STEPS {
            // Compute Hubble friction parameters
            let scale_factor = 1.0 / (1.0 + z);
            let hubble = compute_hubble(z, ETA);  // H(z) in km/s/Mpc -> convert to Gyr^-1
            let hubble_gyr = hubble * 1.022e-3;   // 1 km/s/Mpc = 1.022e-3 Gyr^-1
            let dtau_per_dt = 1.0;  // Conformal time

            gpu_sim.step_with_expansion_dkd_gpu(DT, scale_factor, hubble_gyr, dtau_per_dt)
                .expect("Step failed");

            t_gyr += DT;
            z = compute_redshift_from_time(t_gyr, Z_INIT, ETA);
        }

        final_step = step;
    }

    // Final snapshot if not already saved
    if final_step % SNAPSHOT_INTERVAL != 0 {
        let pos = gpu_sim.get_positions();
        let signs = gpu_sim.get_signs();
        let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, final_step);
        save_snapshot(&snap_path, &pos, &signs, z, BOX_SIZE);
        println!("  → Final snapshot saved: snap_{:06}.bin", final_step);
    }

    // Final flush
    csv_writer.flush().unwrap();
    log_writer.flush().unwrap();

    let total_time = start_time.elapsed().as_secs_f64() / 3600.0;
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  SIMULATION COMPLETE — Total time: {:.2} hours               ║", total_time);
    println!("║  Final step: {}, z = {:.4}                              ║", final_step, z);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nOutputs:");
    println!("  - {}", csv_path);
    println!("  - {}/snapshots/", output_dir);
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

    // Convert from Mpc/Gyr to km/s (1 Mpc/Gyr ≈ 978 km/s)
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

    // Compute COM for each population using minimum image convention
    let mut sum_plus = [0.0f64; 3];
    let mut sum_minus = [0.0f64; 3];
    let mut n_plus = 0;
    let mut n_minus = 0;

    // Find reference particles
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

    // Distance between COMs with periodic boundary
    let mut d2 = 0.0;
    for k in 0..3 {
        let mut d = com_plus[k] - com_minus[k];
        d -= box_size * (d / box_size).round();
        d2 += d * d;
    }

    d2.sqrt() / box_size
}

#[cfg(feature = "cuda")]
fn compute_hubble(z: f64, eta: f64) -> f64 {
    // Janus Hubble parameter H(z) in km/s/Mpc
    // H(z) = H_0 * sqrt(Ω_m * (1+z)^3 + Ω_Λ) for ΛCDM
    // For Janus, use coupled equations but approximate here
    let h0 = 70.0;  // km/s/Mpc
    let omega_m = 0.3;
    let omega_lambda = 0.7;

    h0 * (omega_m * (1.0 + z).powi(3) + omega_lambda).sqrt()
}

#[cfg(feature = "cuda")]
fn compute_redshift_from_time(t_gyr: f64, z_init: f64, _eta: f64) -> f64 {
    // Approximate z(t) - in reality would integrate Friedmann equations
    // Using simple linear approximation for this test
    let t_to_z0 = 4.5;  // Gyr from z=4 to z=0 (approximate for ΛCDM-like)
    let z = z_init * (1.0 - t_gyr / t_to_z0);
    z.max(0.0)
}

#[cfg(feature = "cuda")]
fn save_snapshot(path: &str, pos: &[f64], signs: &[i32], z: f64, box_size: f64) {
    let file = File::create(path).unwrap();
    let mut w = BufWriter::new(file);
    let n = pos.len() / 3;

    // Header: magic, version, n, z, box_size
    w.write_all(b"JSNP").unwrap();
    w.write_all(&2u32.to_le_bytes()).unwrap();
    w.write_all(&(n as u64).to_le_bytes()).unwrap();
    w.write_all(&z.to_le_bytes()).unwrap();
    w.write_all(&box_size.to_le_bytes()).unwrap();

    // Particles: x, y, z (f64), sign (i8), type (u8)
    for i in 0..n {
        w.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        w.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        w.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        w.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
        w.write_all(&0u8.to_le_bytes()).unwrap();
    }

    w.flush().unwrap();
}
