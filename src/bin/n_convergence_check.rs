//! N-convergence test: verify mass_factor gives N-independent results
//! Runs η=1.0, λ₀=15 at N=500k, 1M, 2M for 500 steps
//! Expected: S(z) identical to ±5%

use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const BOX_SIZE: f64 = 500.0;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 500;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;
const SEED: u64 = 42;
const LAMBDA_0: f64 = 15.0;

fn main() {
    #[cfg(not(feature = "cuda"))]
    { eprintln!("Requires --features cuda"); std::process::exit(1); }
    #[cfg(feature = "cuda")]
    run_convergence_test();
}

#[cfg(feature = "cuda")]
fn run_convergence_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  N-CONVERGENCE TEST — Verify mass_factor N-independence      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  η=1.0, λ₀=15, 500 steps, box=500 Mpc                        ║");
    println!("║  N ∈ {{500k, 1M, 2M}}                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_values = [500_000usize, 1_000_000, 2_000_000];
    let checkpoint_steps = [167, 333, 500];  // z≈2, z≈1, z≈0

    let mut results: Vec<(usize, Vec<f64>)> = Vec::new();

    for &n in &n_values {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  N = {} particles", n);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let start = Instant::now();
        let (pos_data, vel_data, signs_data) = generate_lcdm_ics(n);

        let mut sim = GpuNBodyTwoPass::with_custom_ics(
            pos_data, vel_data, signs_data, BOX_SIZE
        ).expect("Failed to create simulation");

        sim.set_theta(THETA);
        sim.set_softening(SOFTENING);
        sim.set_lambda_0(LAMBDA_0);

        let h0 = 0.0715;
        let omega_m = 0.3;
        let tau_init = -1.0 / (h0 * (1.0 + Z_INIT).sqrt());
        let tau_end = 0.0;
        let dtau = (tau_end - tau_init) / STEPS as f64;

        let mut segs = Vec::new();

        for step in 1..=STEPS {
            let tau = tau_init + step as f64 * dtau;
            let a = (1.0 + h0 * tau * (1.0 + Z_INIT).sqrt()).max(0.01);
            let z = (1.0 / a - 1.0).max(0.0);
            let h_tau = h0 * ((omega_m / a.powi(3)) + (1.0 - omega_m)).sqrt();
            let dtau_per_dt = a;

            sim.set_current_z(z);
            sim.step_dkd(dtau, h_tau, dtau_per_dt).expect("Step failed");

            if checkpoint_steps.contains(&step) {
                let seg = sim.segregation().unwrap_or(0.0);
                segs.push(seg);
                let z_label = match step { 167 => "z≈2", 333 => "z≈1", _ => "z≈0" };
                println!("    step {:3} | {} | Seg = {:.4}", step, z_label, seg);
            }
        }

        println!("    Completed in {:.1}s", start.elapsed().as_secs_f64());
        results.push((n, segs));
    }

    // Analysis
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS COMPARISON                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("       N     |  S(z≈2)  |  S(z≈1)  |  S(z≈0)  ");
    println!("  -----------+----------+----------+----------");

    for (n, segs) in &results {
        println!("  {:>9} |  {:.4}  |  {:.4}  |  {:.4}  ", n, segs[0], segs[1], segs[2]);
    }

    // Check convergence
    println!();
    let base = &results[2].1;  // 2M as reference
    let mut all_pass = true;

    for i in 0..2 {
        let (n, segs) = &results[i];
        for j in 0..3 {
            let diff = (segs[j] - base[j]).abs() / base[j].max(0.001) * 100.0;
            let status = if diff <= 5.0 { "✓" } else { all_pass = false; "✗" };
            if diff > 5.0 {
                println!("  {} N={} checkpoint {}: Δ = {:.1}% (>5%)", status, n, j+1, diff);
            }
        }
    }

    if all_pass {
        println!("  ✓ ALL CHECKPOINTS PASS — Δmax < 5%");
        println!("  mass_factor N-independence VALIDATED");
    } else {
        println!("  ✗ CONVERGENCE FAILED — mass_factor bug suspected");
    }
}

#[cfg(feature = "cuda")]
fn generate_lcdm_ics(n: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    use std::f64::consts::PI;

    let n_grid = (n as f64 / 2.0).powf(1.0/3.0).ceil() as usize;
    let cell = BOX_SIZE / n_grid as f64;
    let box_half = BOX_SIZE / 2.0;

    let omega_m = 0.3;
    let n_s = 0.965;
    let h = 0.7;
    let theta = 2.725 / 2.7;
    let k_eq = 0.0746 * omega_m * h * h / (theta * theta);

    let a = 1.0 / (1.0 + Z_INIT);
    let omega_m_z = omega_m / (omega_m + (1.0 - omega_m) * a.powi(3));
    let d_z = a * (omega_m_z.powf(4.0/7.0) - (1.0 - omega_m_z) +
              (1.0 + omega_m_z/2.0) * (1.0 + (1.0-omega_m_z)/70.0)).powf(-1.0);

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let f_omega = omega_m.powf(0.55);
    let h0_gyr = 0.0715;
    let e_z = ((omega_m * (1.0 + Z_INIT).powi(3)) + (1.0 - omega_m)).sqrt();
    let h_z = h0_gyr * e_z;
    let vel_factor = h_z * f_omega;
    let disp_scale = 25.0 * d_z;

    let mut pos_data = Vec::with_capacity(n * 3);
    let mut vel_data = Vec::with_capacity(n * 3);
    let mut signs_data = Vec::with_capacity(n);

    let mut count = 0;
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                if count >= n { break; }

                let x0 = (ix as f64 + 0.5) * cell - box_half;
                let y0 = (iy as f64 + 0.5) * cell - box_half;
                let z0 = (iz as f64 + 0.5) * cell - box_half;

                let k = 2.0 * PI / BOX_SIZE * ((ix*ix + iy*iy + iz*iz) as f64).sqrt().max(1.0);
                let pk = k.powf(n_s) / (1.0 + (k/k_eq).powi(2)).powf(2.0);
                let amp = (pk.sqrt() * disp_scale * cell / BOX_SIZE).min(cell * 0.5);

                let psi_x: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_y: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                let psi_z: f64 = rng.gen::<f64>() * 2.0 - 1.0;

                let dx = psi_x * amp;
                let dy = psi_y * amp;
                let dz = psi_z * amp;

                pos_data.extend_from_slice(&[(x0 + dx) as f32, (y0 + dy) as f32, (z0 + dz) as f32]);
                vel_data.extend_from_slice(&[(dx * vel_factor) as f32, (dy * vel_factor) as f32, (dz * vel_factor) as f32]);
                signs_data.push(if rng.gen::<bool>() { 1 } else { -1 });
                count += 1;
            }
        }
    }

    while count < n {
        let x: f64 = rng.gen::<f64>() * BOX_SIZE - box_half;
        let y: f64 = rng.gen::<f64>() * BOX_SIZE - box_half;
        let z: f64 = rng.gen::<f64>() * BOX_SIZE - box_half;
        pos_data.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel_data.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs_data.push(if rng.gen::<bool>() { 1 } else { -1 });
        count += 1;
    }

    (pos_data, vel_data, signs_data)
}
