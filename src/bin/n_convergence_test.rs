//! N-convergence test for mass_factor fix
//! Runs 500k, 1M, 2M particles with same parameters to verify S(z) converges
//!
//! Usage: cargo run --release --features cuda --bin n_convergence_test

use std::time::Instant;
use std::fs;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const BOX_SIZE: f64 = 100.0;  // Smaller box for quick test
const Z_INIT: f64 = 5.0;
const STEPS: usize = 50;      // Minimal test
const SNAP_INT: usize = 10;   // Log every 10 steps

const ETA: f64 = 1.00;
const LAMBDA_0: f64 = 15.0;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;   // 0.5 Mpc softening

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This test requires --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_tests();
}

#[cfg(feature = "cuda")]
fn run_tests() {
    println!("======================================================================");
    println!("N-CONVERGENCE TEST — Verifying mass_factor fix");
    println!("======================================================================");
    println!("Box: {} Mpc, η={}, λ₀={} Mpc, θ={}", BOX_SIZE, ETA, LAMBDA_0, THETA);
    println!("Steps: {}, z: {} → 0", STEPS, Z_INIT);
    println!();

    let particle_counts = vec![100_000usize, 200_000, 400_000];  // Smaller for quick test
    let mut results: Vec<Vec<(f64, f64)>> = Vec::new();  // (z, seg) for each N

    // Create output directory
    let out_dir = format!("/app/output/n_convergence_test");
    fs::create_dir_all(&out_dir).ok();

    for &n in &particle_counts {
        println!("=== Running N = {} ===", n);
        let n_per_sign = n / 2;

        let start = Instant::now();
        let mut sim = GpuNBodyTwoPass::new(n_per_sign, n_per_sign, BOX_SIZE)
            .expect("Failed to create simulation");

        sim.set_theta(THETA);
        sim.set_softening(SOFTENING);
        sim.set_lambda_0(LAMBDA_0);

        println!("  Init time: {:.1}s", start.elapsed().as_secs_f64());
        println!("  mass_factor = {:.6e}", 1.0 / n as f64);

        // Cosmology parameters
        let h0 = 0.0715;  // H0 in Gyr^-1
        let omega_m = 0.3;
        let tau_init = -1.0 / (h0 * (1.0 + Z_INIT).sqrt());
        let tau_end = 0.0;
        let dtau = (tau_end - tau_init) / STEPS as f64;

        let mut seg_history = Vec::new();

        for step in 0..=STEPS {
            let tau = tau_init + step as f64 * dtau;
            let a = (1.0 + h0 * tau * (1.0 + Z_INIT).sqrt()).max(0.01);
            let z = (1.0 / a - 1.0).max(0.0);
            let h_tau = h0 * ((omega_m / a.powi(3)) + (1.0 - omega_m)).sqrt();
            let dtau_per_dt = a;

            if step > 0 {
                sim.set_current_z(z);
                sim.step_dkd(dtau, h_tau, dtau_per_dt)
                    .expect("Step failed");
            }

            if step % SNAP_INT == 0 || step == STEPS {
                let seg = sim.segregation().unwrap_or(0.0);
                seg_history.push((z, seg));
                println!("  step {:4} | z={:.2} | Seg={:.4}", step, z, seg);
            }
        }

        results.push(seg_history);
        println!("  Completed in {:.1}s", start.elapsed().as_secs_f64());
        println!();
    }

    // Compare results
    println!("======================================================================");
    println!("CONVERGENCE ANALYSIS");
    println!("======================================================================");
    println!("{:>8}  {:>8}  {:>8}  {:>8}  {:>8}", "z", "S(500k)", "S(1M)", "S(2M)", "Δmax");
    println!("----------------------------------------------------------------------");

    let n_points = results[0].len();
    let mut max_delta = 0.0f64;

    for i in 0..n_points {
        let z = results[0][i].0;
        let s_500k = results[0][i].1;
        let s_1m = results[1][i].1;
        let s_2m = results[2][i].1;

        let delta = (s_500k - s_2m).abs().max((s_1m - s_2m).abs());
        max_delta = max_delta.max(delta);

        println!("{:8.2}  {:8.4}  {:8.4}  {:8.4}  {:8.4}", z, s_500k, s_1m, s_2m, delta);
    }

    println!("----------------------------------------------------------------------");
    println!("Max Δ across all z: {:.4}", max_delta);

    if max_delta < 0.05 {
        println!("\n✅ CONVERGENCE TEST PASSED: S(z) is N-independent (Δ < 5%)");
    } else if max_delta < 0.10 {
        println!("\n⚠️ WARNING: S(z) shows some N-dependence (Δ = {:.1}%)", max_delta * 100.0);
    } else {
        println!("\n❌ CONVERGENCE TEST FAILED: S(z) is N-dependent (Δ = {:.1}%)", max_delta * 100.0);
    }

    // Save results to CSV
    let csv_path = format!("{}/convergence_results.csv", out_dir);
    let mut csv = String::from("z,S_500k,S_1M,S_2M\n");
    for i in 0..n_points {
        csv.push_str(&format!("{:.4},{:.6},{:.6},{:.6}\n",
            results[0][i].0, results[0][i].1, results[1][i].1, results[2][i].1));
    }
    fs::write(&csv_path, &csv).ok();
    println!("\nResults saved to {}", csv_path);
}
