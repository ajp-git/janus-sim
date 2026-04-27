//! Quick test: 100k particles, 10 steps

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::fs::{self, File};
use std::io::{BufWriter, Write};

const N_PARTICLES: usize = 100_000;
const BOX_SIZE: f64 = 271.4;  // Same density as 10M/2714
const MU: f64 = 19.0;
const N_STEPS: usize = 10;
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

    let output_dir = "/app/output/vsl_test_100k";
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    let c_ratio = 1.0 / MU.sqrt();
    let f_plus = ETA / (1.0 + ETA);
    let n_positive = (N_PARTICLES as f64 * f_plus).round() as usize;
    let n_negative = N_PARTICLES.saturating_sub(n_positive);

    let sigma_v = (K_B_OVER_MP_CODE * T_INIT / MU_MOL).sqrt();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       VSL TEST — 100k particles, 10 steps                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {} ({} m+ / {} m-)", N_PARTICLES, n_positive, n_negative);
    println!("║  Box = {} Mpc (same density as 10M/2714)", BOX_SIZE);
    println!("║  μ = {} → c⁻/c⁺ = {:.4}", MU, c_ratio);
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
    let half_box = BOX_SIZE / 2.0;

    for step in 0..=N_STEPS {
        let z = 1.0 / a - 1.0;

        let pos = gpu_sim.get_positions().unwrap();
        let vel = gpu_sim.get_velocities().unwrap();
        let signs = gpu_sim.signs();

        // Compute v_rms and Z mean
        let mut v2_plus = 0.0;
        let mut v2_minus = 0.0;
        let mut z_sum_plus = 0.0;
        let mut z_sum_minus = 0.0;
        let mut n_plus = 0usize;
        let mut n_minus = 0usize;

        for i in 0..N_PARTICLES {
            let vx = vel[i*3];
            let vy = vel[i*3+1];
            let vz = vel[i*3+2];
            let pz = pos[i*3+2];

            if signs[i] > 0 {
                v2_plus += vx*vx + vy*vy + vz*vz;
                z_sum_plus += pz;
                n_plus += 1;
            } else {
                v2_minus += vx*vx + vy*vy + vz*vz;
                z_sum_minus += pz;
                n_minus += 1;
            }
        }

        let v_rms_plus = (v2_plus / n_plus as f64).sqrt() * MPC_GYR_TO_KMS;
        let v_rms_minus = (v2_minus / n_minus as f64).sqrt() * MPC_GYR_TO_KMS;
        let z_mean_plus = z_sum_plus / n_plus as f64;
        let z_mean_minus = z_sum_minus / n_minus as f64;

        println!("Step {:>3} | z={:.3} | v+={:>6.0} v-={:>6.0} km/s | Z+={:>+7.1} Z-={:>+7.1} Mpc",
                 step, z, v_rms_plus, v_rms_minus, z_mean_plus, z_mean_minus);

        // Save snapshot
        let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
        save_snapshot(&snap_path, &pos, &signs, z, BOX_SIZE);

        if step >= N_STEPS { break; }

        let h = 0.07 / a.powf(1.5);
        gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 0.0).unwrap();
        a += a * h * DT;
    }

    println!("\n✓ Test complete. Check output/vsl_test_100k/snapshots/");
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

    w.write_all(b"JSNP").unwrap();
    w.write_all(&2u32.to_le_bytes()).unwrap();
    w.write_all(&(n as u64).to_le_bytes()).unwrap();
    w.write_all(&z.to_le_bytes()).unwrap();
    w.write_all(&box_size.to_le_bytes()).unwrap();

    for i in 0..n {
        w.write_all(&pos[i*3].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+1].to_le_bytes()).unwrap();
        w.write_all(&pos[i*3+2].to_le_bytes()).unwrap();
        w.write_all(&(if signs[i] > 0 { 1i8 } else { -1i8 }).to_le_bytes()).unwrap();
        // type: 0 = gas m+, 255 = m-
        let ptype: u8 = if signs[i] > 0 { 0 } else { 255 };
        w.write_all(&ptype.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
}
