//! VSL μ validation runs: test different c_ratio values
//! Usage: cargo run --bin vsl_mu_validation -- --mu 2|4|8

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};

const N_PARTICLES: usize = 500_000;
const BOX_SIZE: f64 = 500.0;  // Same density as 500k/1000Mpc validated
const N_STEPS: usize = 500;
const DT: f64 = 0.001;
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const K_B_OVER_MP_CODE: f64 = 8.7e-9;
const T_INIT: f64 = 1.0e4;
const MU_MOL: f64 = 0.6;
const MPC_GYR_TO_KMS: f64 = 977.8;

#[cfg(feature = "cuda")]
fn main() {
    use rand::SeedableRng;
    use rand_distr::{Normal, Distribution};

    // Parse μ from command line
    let args: Vec<String> = env::args().collect();
    let mu: f64 = if args.len() > 2 && args[1] == "--mu" {
        args[2].parse().expect("Invalid μ value")
    } else {
        eprintln!("Usage: {} --mu <2|4|8>", args[0]);
        std::process::exit(1);
    };

    let c_ratio = 1.0 / mu.sqrt();
    let run_name = format!("vsl_mu{}_validation", mu as i32);
    let output_dir = format!("/app/output/{}", run_name);
    fs::create_dir_all(&output_dir).unwrap();

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    let sigma_v = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL μ VALIDATION RUN                                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  μ = {} → c⁻/c⁺ = {:.4}", mu, c_ratio);
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc, {} steps", BOX_SIZE, N_STEPS);
    println!("║  Criterion: v_rms- < 10000 km/s at step 500", );
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let mut gpu_sim = GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE).unwrap();
    gpu_sim.set_c_ratio(c_ratio);
    gpu_sim.set_theta(0.7);

    // Thermal velocities
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, sigma_v).unwrap();
    let mut vel = vec![0.0f64; N_PARTICLES * 3];
    for i in 0..N_PARTICLES {
        vel[i*3] = normal.sample(&mut rng);
        vel[i*3+1] = normal.sample(&mut rng);
        vel[i*3+2] = normal.sample(&mut rng);
    }
    gpu_sim.set_velocities(&vel).unwrap();

    let a_init = 1.0 / (1.0 + Z_INIT);
    let mut a = a_init;

    // CSV output
    let csv_path = format!("{}/evolution.csv", output_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,z,rho_plus_max,v_rms_minus,segregation").unwrap();

    println!("  step |      z | ρ+_max |   v_rms- |    Seg");
    println!("-------|--------|--------|----------|--------");

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        // Compute metrics every 50 steps or at start/end
        if step % 50 == 0 || step == N_STEPS {
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs = gpu_sim.signs();

            // Compute density on 64³ grid
            let grid_size = 64usize;
            let cell_size = BOX_SIZE / grid_size as f64;
            let half_box = BOX_SIZE / 2.0;
            let mut rho_plus = vec![0u32; grid_size * grid_size * grid_size];

            for i in 0..N_PARTICLES {
                if signs[i] > 0 {
                    let ix = ((pos[i*3] + half_box) / cell_size).floor() as usize;
                    let iy = ((pos[i*3+1] + half_box) / cell_size).floor() as usize;
                    let iz = ((pos[i*3+2] + half_box) / cell_size).floor() as usize;
                    if ix < grid_size && iy < grid_size && iz < grid_size {
                        rho_plus[ix + iy * grid_size + iz * grid_size * grid_size] += 1;
                    }
                }
            }
            let rho_plus_max = *rho_plus.iter().max().unwrap_or(&0);

            // v_rms for m-
            let mut v2_minus = 0.0;
            let mut n_minus_count = 0usize;
            let mut z_sum_plus = 0.0;
            let mut z_sum_minus = 0.0;
            let mut n_plus_count = 0usize;

            for i in 0..N_PARTICLES {
                let vx = vel[i*3];
                let vy = vel[i*3+1];
                let vz = vel[i*3+2];
                let pz = pos[i*3+2];

                if signs[i] > 0 {
                    z_sum_plus += pz;
                    n_plus_count += 1;
                } else {
                    v2_minus += vx*vx + vy*vy + vz*vz;
                    z_sum_minus += pz;
                    n_minus_count += 1;
                }
            }

            let v_rms_minus = (v2_minus / n_minus_count as f64).sqrt() * MPC_GYR_TO_KMS;
            let z_mean_plus = z_sum_plus / n_plus_count as f64;
            let z_mean_minus = z_sum_minus / n_minus_count as f64;
            let seg = (z_mean_plus - z_mean_minus).abs() / BOX_SIZE;

            println!("{:>6} | {:>6.3} | {:>6} | {:>8.0} | {:>6.4}",
                     step, z, rho_plus_max, v_rms_minus, seg);

            writeln!(csv, "{},{:.4},{},{:.0},{:.6}", step, z, rho_plus_max, v_rms_minus, seg).unwrap();
            csv.flush().unwrap();

            // Early termination check
            if v_rms_minus > 50000.0 {
                println!("\n⚠ EMERGENCY STOP: v_rms- > 50000 km/s");
                break;
            }
        }

        if step >= N_STEPS { break; }

        let h = 0.07 / a.powf(1.5);
        gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0).unwrap();
        a += a * h * DT;
    }

    // Final verdict
    let final_line = std::fs::read_to_string(&csv_path).unwrap();
    let last_line = final_line.lines().last().unwrap();
    let v_rms_final: f64 = last_line.split(',').nth(3).unwrap().parse().unwrap();

    println!("\n════════════════════════════════════════════════════════════════════════");
    if v_rms_final < 10000.0 {
        println!("✓ PASS: μ={} stable (v_rms- = {:.0} km/s < 10000)", mu, v_rms_final);
    } else {
        println!("✗ FAIL: μ={} unstable (v_rms- = {:.0} km/s > 10000)", mu, v_rms_final);
    }
    println!("════════════════════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
}
