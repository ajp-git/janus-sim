//! VSL Dynamic c_ratio test: N=100k, 500 steps
//! Verify that dynamic c_ratio stabilizes v_rms-

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use std::fs::{self, File};
use std::io::{BufWriter, Write};

const N_PARTICLES: usize = 100_000;
const BOX_SIZE: f64 = 200.0;  // Same density as production
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

    let output_dir = "/app/output/vsl_dynamic_test";
    fs::create_dir_all(output_dir).unwrap();

    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    // Initial c_ratio from dynamic model
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    let c_ratio_init = c_ratio_sq_init.sqrt();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL DYNAMIC c_ratio TEST                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc, {} steps, dt = {} Gyr", BOX_SIZE, N_STEPS, DT);
    println!("║  η = {}", ETA);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  DYNAMIC c_ratio from Petit MPLA 2014:");
    println!("║    c_ratio_sq(z) = (1+z)^δ where δ = (η-1)/η = {:.4}", (ETA-1.0)/ETA);
    println!("║    c_ratio_sq(z=4) = {:.6}", c_ratio_sq_init);
    println!("║    c_ratio_sq(z=0) = 1.000000");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  HYPOTHESIS: v_rms- should stabilize with dynamic c_ratio");
    println!("║    (vs constant c_ratio which causes linear v_rms- growth)");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Initialize GPU simulation
    println!("Initializing GPU with {} particles...", N_PARTICLES);
    let mut gpu_sim = GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE).unwrap();
    gpu_sim.set_c_ratio(c_ratio_init);
    gpu_sim.set_theta(0.7);
    println!("GPU init complete, initial c_ratio = {:.6}\n", c_ratio_init);

    // Thermal velocities (same σ for both populations initially)
    let sigma_v = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();
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
    writeln!(csv, "step,z,a,c_ratio_sq,c_ratio,v_rms_plus,v_rms_minus,delta_v").unwrap();

    println!("  step |      z |      a | c_ratio_sq | c_ratio |   v_rms+ |   v_rms- |  Δv_rms-");
    println!("-------|--------|--------|------------|---------|----------|----------|----------");

    let mut prev_v_rms_minus = 0.0;

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        // Dynamic c_ratio from coupled Friedmann equations
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_ratio = c_ratio_sq.sqrt();

        // Update GPU simulation with new c_ratio
        gpu_sim.set_c_ratio(c_ratio);

        // Compute metrics every 50 steps
        if step % 50 == 0 || step == N_STEPS {
            let vel_data = gpu_sim.get_velocities().unwrap();
            let signs = gpu_sim.signs();

            let mut v2_plus = 0.0;
            let mut v2_minus = 0.0;
            let mut n_plus = 0usize;
            let mut n_minus = 0usize;

            for i in 0..N_PARTICLES {
                let vx = vel_data[i*3];
                let vy = vel_data[i*3+1];
                let vz = vel_data[i*3+2];
                let v2 = vx*vx + vy*vy + vz*vz;

                if signs[i] > 0 {
                    v2_plus += v2;
                    n_plus += 1;
                } else {
                    v2_minus += v2;
                    n_minus += 1;
                }
            }

            let v_rms_plus = (v2_plus / n_plus as f64).sqrt() * MPC_GYR_TO_KMS;
            let v_rms_minus = (v2_minus / n_minus as f64).sqrt() * MPC_GYR_TO_KMS;
            let delta_v = v_rms_minus - prev_v_rms_minus;

            println!("{:>6} | {:>6.3} | {:>6.4} | {:>10.6} | {:>7.5} | {:>8.0} | {:>8.0} | {:>+8.0}",
                     step, z, a, c_ratio_sq, c_ratio, v_rms_plus, v_rms_minus, delta_v);

            writeln!(csv, "{},{:.4},{:.6},{:.6},{:.6},{:.0},{:.0},{:.0}",
                     step, z, a, c_ratio_sq, c_ratio, v_rms_plus, v_rms_minus, delta_v).unwrap();
            csv.flush().unwrap();

            prev_v_rms_minus = v_rms_minus;
        }

        if step >= N_STEPS { break; }

        // Hubble expansion
        let h = 0.07 / a.powf(1.5);
        gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0).unwrap();
        a += a * h * DT;
    }

    // Analysis
    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("ANALYSIS:");

    // Read CSV and compute growth rate
    let content = fs::read_to_string(&csv_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    if lines.len() >= 3 {
        let line_100 = lines.iter().find(|l| l.starts_with("100,"));
        let line_500 = lines.iter().find(|l| l.starts_with("500,"));

        if let (Some(l100), Some(l500)) = (line_100, line_500) {
            let v100: f64 = l100.split(',').nth(6).unwrap().parse().unwrap();
            let v500: f64 = l500.split(',').nth(6).unwrap().parse().unwrap();
            let growth_per_step = (v500 - v100) / 400.0;

            println!("  v_rms- @ step 100: {:.0} km/s", v100);
            println!("  v_rms- @ step 500: {:.0} km/s", v500);
            println!("  Growth rate: {:.1} km/s per step", growth_per_step);

            if growth_per_step < 30.0 {
                println!("\n  ✓ STABILIZATION DETECTED: growth rate < 30 km/s/step");
                println!("    (constant μ=11 had ~52 km/s/step)");
            } else {
                println!("\n  ✗ Still unstable: growth rate = {:.1} km/s/step", growth_per_step);
            }
        }
    }
    println!("════════════════════════════════════════════════════════════════════════");
    println!("Output: {}", csv_path);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: Requires --features cuda");
}
