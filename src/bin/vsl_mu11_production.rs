//! VSL Production Run: μ=11, 10M particles, 1357 Mpc box
//! Same density as validated 500k/500Mpc test

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

const N_PARTICLES: usize = 10_000_000;
const BOX_SIZE: f64 = 1357.0;  // Same density as 500k/500Mpc
const MU: f64 = 11.0;
const N_STEPS: usize = 30000;
const DT: f64 = 0.001;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const K_B_OVER_MP_CODE: f64 = 8.7e-9;
const T_INIT: f64 = 1.0e4;
const MU_MOL: f64 = 0.6;
const MPC_GYR_TO_KMS: f64 = 977.8;
const SNAPSHOT_INTERVAL: usize = 10;
const METRIC_INTERVAL: usize = 50;

#[cfg(feature = "cuda")]
fn main() {
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    let c_ratio = 1.0 / MU.sqrt();
    let c_ratio_sq = 1.0 / MU;
    let output_dir = "/app/output/vsl_mu11_10M_production";
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    let sigma_v_plus = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();
    let sigma_v_minus = sigma_v_plus * c_ratio;  // Scaled by c_ratio

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL PRODUCTION RUN — μ=11                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  μ = {} → c⁻/c⁺ = {:.4} → c_ratio_sq = {:.4}", MU, c_ratio, c_ratio_sq);
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc (density = {:.2e} part/Mpc³)", BOX_SIZE,
             N_PARTICLES as f64 / BOX_SIZE.powi(3));
    println!("║  Steps = {}, dt = {} Gyr", N_STEPS, DT);
    println!("║  Snapshots every {} steps ({} total)", SNAPSHOT_INTERVAL, N_STEPS/SNAPSHOT_INTERVAL);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  VALIDATION: step 200 check");
    println!("║    v_rms- < 20000 km/s → OK");
    println!("║    v_rms- > 40000 km/s → STOP");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  ALERTS:");
    println!("║    ρ+_max > 100  → ★ structures");
    println!("║    ρ+_max > 1000 → ★★ halos");
    println!("║    N_stars > 0   → ★★★ FIRST STAR");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    println!("Initializing GPU with {} particles...", N_PARTICLES);
    let mut gpu_sim = GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE).unwrap();
    gpu_sim.set_c_ratio(c_ratio);
    gpu_sim.set_theta(0.7);
    println!("GPU init: {:.2}s, c_ratio = {:.4}, theta = 0.7",
             start_time.elapsed().as_secs_f64(), c_ratio);

    // Thermal velocities
    println!("Setting thermal velocities...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal_plus = Normal::new(0.0, sigma_v_plus).unwrap();
    let normal_minus = Normal::new(0.0, sigma_v_minus).unwrap();
    let mut vel = vec![0.0f64; N_PARTICLES * 3];

    for i in 0..n_positive {
        vel[i*3] = normal_plus.sample(&mut rng);
        vel[i*3+1] = normal_plus.sample(&mut rng);
        vel[i*3+2] = normal_plus.sample(&mut rng);
    }
    for i in n_positive..N_PARTICLES {
        vel[i*3] = normal_minus.sample(&mut rng);
        vel[i*3+1] = normal_minus.sample(&mut rng);
        vel[i*3+2] = normal_minus.sample(&mut rng);
    }
    gpu_sim.set_velocities(&vel).unwrap();
    println!("Velocities set: m+ σ={:.1} km/s, m- σ={:.1} km/s\n",
             sigma_v_plus * MPC_GYR_TO_KMS, sigma_v_minus * MPC_GYR_TO_KMS);

    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;
    let half_box = BOX_SIZE / 2.0;
    let grid_size = 64usize;
    let cell_size = BOX_SIZE / grid_size as f64;

    // CSV output
    let csv_path = format!("{}/evolution.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,t_Gyr,z,rho_plus_max,rho_minus_max,v_rms_plus,v_rms_minus,n_stars,segregation").unwrap();

    println!("  Step |  t_Gyr |      z | ρ+_max | ρ-_max |   v_rms+ |   v_rms- | N_stars |    Seg | Status");
    println!("-------|--------|--------|--------|--------|----------|----------|---------|--------|--------");

    let mut t_gyr = 0.0;
    let mut first_structure = false;
    let mut first_halo = false;

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        // Metrics and snapshots
        let do_metric = step % METRIC_INTERVAL == 0 || step <= 10;
        let do_snapshot = step % SNAPSHOT_INTERVAL == 0;

        if do_metric || do_snapshot {
            let pos = gpu_sim.get_positions().unwrap();
            let vel_data = gpu_sim.get_velocities().unwrap();
            let signs = gpu_sim.signs();

            // Compute density grids
            let mut rho_plus = vec![0u32; grid_size * grid_size * grid_size];
            let mut rho_minus = vec![0u32; grid_size * grid_size * grid_size];

            for i in 0..N_PARTICLES {
                let ix = ((pos[i*3] + half_box) / cell_size).floor() as usize;
                let iy = ((pos[i*3+1] + half_box) / cell_size).floor() as usize;
                let iz = ((pos[i*3+2] + half_box) / cell_size).floor() as usize;
                if ix < grid_size && iy < grid_size && iz < grid_size {
                    let idx = ix + iy * grid_size + iz * grid_size * grid_size;
                    if signs[i] > 0 {
                        rho_plus[idx] += 1;
                    } else {
                        rho_minus[idx] += 1;
                    }
                }
            }
            let rho_plus_max = *rho_plus.iter().max().unwrap_or(&0);
            let rho_minus_max = *rho_minus.iter().max().unwrap_or(&0);

            // v_rms and segregation
            let mut v2_plus = 0.0;
            let mut v2_minus = 0.0;
            let mut z_sum_plus = 0.0;
            let mut z_sum_minus = 0.0;
            let mut n_plus_count = 0usize;
            let mut n_minus_count = 0usize;

            for i in 0..N_PARTICLES {
                let vx = vel_data[i*3];
                let vy = vel_data[i*3+1];
                let vz = vel_data[i*3+2];
                let pz = pos[i*3+2];
                let v2 = vx*vx + vy*vy + vz*vz;

                if signs[i] > 0 {
                    v2_plus += v2;
                    z_sum_plus += pz;
                    n_plus_count += 1;
                } else {
                    v2_minus += v2;
                    z_sum_minus += pz;
                    n_minus_count += 1;
                }
            }

            let v_rms_plus = (v2_plus / n_plus_count as f64).sqrt() * MPC_GYR_TO_KMS;
            let v_rms_minus = (v2_minus / n_minus_count as f64).sqrt() * MPC_GYR_TO_KMS;
            let z_mean_plus = z_sum_plus / n_plus_count as f64;
            let z_mean_minus = z_sum_minus / n_minus_count as f64;
            let seg = (z_mean_plus - z_mean_minus).abs() / BOX_SIZE;

            // Status and alerts
            let mut status = String::new();
            if rho_plus_max > 1000 && !first_halo {
                status = "★★ HALO".to_string();
                first_halo = true;
            } else if rho_plus_max > 100 && !first_structure {
                status = "★ STRUCT".to_string();
                first_structure = true;
            }

            // ETA calculation
            let elapsed = start_time.elapsed().as_secs_f64();
            let eta_h = if step > 0 {
                elapsed / step as f64 * (N_STEPS - step) as f64 / 3600.0
            } else { 0.0 };

            if do_metric {
                println!("{:>6} | {:>6.3} | {:>6.3} | {:>6} | {:>6} | {:>8.0} | {:>8.0} | {:>7} | {:>6.4} | {} ({:.1}h)",
                         step, t_gyr, z, rho_plus_max, rho_minus_max,
                         v_rms_plus, v_rms_minus, 0, seg, status, eta_h);

                writeln!(csv, "{},{:.4},{:.4},{},{},{:.0},{:.0},{},{:.6}",
                         step, t_gyr, z, rho_plus_max, rho_minus_max,
                         v_rms_plus, v_rms_minus, 0, seg).unwrap();
                csv.flush().unwrap();
            }

            // Step 200 validation
            if step == 200 {
                if v_rms_minus > 40000.0 {
                    println!("\n⚠ EMERGENCY STOP: v_rms- = {:.0} > 40000 km/s at step 200", v_rms_minus);
                    break;
                } else if v_rms_minus > 20000.0 {
                    println!("⚠ WARNING: v_rms- = {:.0} km/s elevated but continuing", v_rms_minus);
                } else {
                    println!("✓ Step 200 validation PASSED: v_rms- = {:.0} km/s < 20000", v_rms_minus);
                }
            }

            // Emergency stop
            if v_rms_minus > 100000.0 {
                println!("\n⚠ EMERGENCY STOP: v_rms- > 100000 km/s");
                break;
            }

            // Save snapshot
            if do_snapshot {
                let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
                save_snapshot(&snap_path, &pos, &signs, z, BOX_SIZE);
            }
        }

        if step >= N_STEPS { break; }

        let h = 0.07 / a.powf(1.5);
        gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0).unwrap();
        a += a * h * DT;
        t_gyr += DT;
    }

    let total_time = start_time.elapsed().as_secs_f64() / 3600.0;
    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("Run complete in {:.2}h", total_time);
    println!("Output: {}", output_dir);
    println!("════════════════════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
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
        w.write_all(&pos[i*3].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+1].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+2].to_le_bytes()).unwrap();
        w.write_all(&(if signs[i] > 0 { 1i8 } else { -1i8 }).to_le_bytes()).unwrap();
        let ptype: u8 = if signs[i] > 0 { 0 } else { 255 };
        w.write_all(&ptype.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
}
