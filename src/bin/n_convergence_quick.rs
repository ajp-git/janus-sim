//! Quick N-convergence test after reduce_tp fix
//! Verifies S(z) is N-independent within ±5%

use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const BOX_SIZE: f64 = 500.0;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 100;  // Reduced for speed
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
    println!("║  QUICK N-CONVERGENCE TEST — reduce_tp fix verification       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_values = [100_000usize, 250_000, 500_000];
    let checkpoint_steps = [25, 50, 75, 100];  // 4 checkpoints

    let mut results: Vec<(usize, Vec<f64>)> = Vec::new();

    for &n in &n_values {
        println!("━━━ N = {} ━━━", n);
        let start = Instant::now();

        let (pos_data, vel_data, signs_data) = generate_ics(n);

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
                println!("  step {:3} | z={:.2} | S={:.4}", step, z, seg);
            }
        }

        println!("  Done in {:.1}s", start.elapsed().as_secs_f64());
        println!();
        results.push((n, segs));
    }

    // Analysis: compare to largest N
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CONVERGENCE CHECK (vs N={})                          ║", n_values[2]);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let base = &results[2].1;
    let mut all_pass = true;

    for i in 0..2 {
        let (n, segs) = &results[i];
        print!("N={}: ", n);
        let mut max_diff = 0.0f64;
        for j in 0..checkpoint_steps.len() {
            let diff = (segs[j] - base[j]).abs() / base[j].max(0.001) * 100.0;
            max_diff = max_diff.max(diff);
        }
        let status = if max_diff <= 5.0 { "✓" } else { all_pass = false; "✗" };
        println!("{} max Δ = {:.1}%", status, max_diff);
    }

    println!();
    if all_pass {
        println!("✓ N-CONVERGENCE PASSED — mass_factor fix validated");
    } else {
        println!("✗ N-CONVERGENCE FAILED — further investigation needed");
    }
}

#[cfg(feature = "cuda")]
fn generate_ics(n: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let box_half = BOX_SIZE / 2.0;

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for _ in 0..n {
        let x = rng.random::<f64>() * BOX_SIZE - box_half;
        let y = rng.random::<f64>() * BOX_SIZE - box_half;
        let z = rng.random::<f64>() * BOX_SIZE - box_half;
        pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs.push(if rng.random::<bool>() { 1 } else { -1 });
    }

    (pos, vel, signs)
}
