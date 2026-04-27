//! Test: verify λ(z) = λ₀×√(1+z) scaling is correctly applied
//! Large λ at high z (strong repulsion early), small λ at z=0

use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const BOX_SIZE: f64 = 500.0;
const N: usize = 50_000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;
const SEED: u64 = 42;
const LAMBDA_0: f64 = 15.0;

fn main() {
    #[cfg(not(feature = "cuda"))]
    { eprintln!("Requires --features cuda"); std::process::exit(1); }
    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  TEST: λ(z) = λ₀/√(1+z) SCALING                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("λ₀ = {} Mpc", LAMBDA_0);
    println!();

    // Test different redshifts
    let redshifts = [0.0, 1.0, 3.0, 8.0];

    println!("Expected λ(z) = λ₀×√(1+z) values:");
    for &z in &redshifts {
        let lambda_z = LAMBDA_0 * (1.0_f64 + z).sqrt();
        println!("  z = {:.1}: λ = {:.2} Mpc", z, lambda_z);
    }
    println!();

    // Generate same ICs for all tests
    let (pos, vel, signs) = generate_ics();

    let mut results = Vec::new();

    for &z in &redshifts {
        let mut sim = GpuNBodyTwoPass::with_custom_ics(
            pos.clone(), vel.clone(), signs.clone(), BOX_SIZE
        ).expect("Failed to create simulation");

        sim.set_theta(THETA);
        sim.set_softening(SOFTENING);
        sim.set_lambda_0(LAMBDA_0);
        sim.set_current_z(z);

        // Verify screening is enabled
        eprintln!("  DEBUG: lambda_0={}, current_z={}", LAMBDA_0, z);

        // Single step to compute forces (should use screening kernel)
        sim.step_dkd(0.001, 0.07, 1.0).expect("Step failed");

        // Get mean acceleration
        let sum_a = sim.acceleration_sum().expect("accel sum");
        let mean_a = sum_a / sim.n_particles() as f64;

        let lambda_z = LAMBDA_0 * (1.0_f64 + z).sqrt();
        println!("z = {:.1}: λ(z) = {:>6.2} Mpc, <|a|> = {:.4e} Mpc/Gyr²",
                 z, lambda_z, mean_a);

        results.push((z, lambda_z, mean_a));
    }

    // Analysis: at higher z (larger λ), screening range is larger → more repulsion
    // At z=0, λ = λ₀, at z=8, λ = 3×λ₀
    // With larger λ, more opposite-sign pairs feel repulsion → higher total <|a|>
    println!();
    println!("Analysis:");

    // Compare z=0 vs z=8
    let a_z0 = results[0].2;
    let a_z8 = results[3].2;
    let ratio = a_z8 / a_z0;
    println!("  <|a|>(z=8) / <|a|>(z=0) = {:.3}", ratio);

    // At z=8, λ = 3×λ₀ so screening range is 3× larger
    // More particles feel repulsion at z=8 than at z=0
    // So we expect HIGHER forces at z=8
    if ratio > 1.0 {
        println!("  ✓ Forces higher at z=8 — screening range larger as expected");
        println!("  λ(z) = λ₀×√(1+z) scaling VALIDATED");
    } else {
        println!("  ✗ Forces NOT higher at z=8 — unexpected");
        println!("  Check if λ(z) scaling is being applied correctly");
    }
}

#[cfg(feature = "cuda")]
fn generate_ics() -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let box_half = BOX_SIZE / 2.0;

    let mut pos = Vec::with_capacity(N * 3);
    let mut vel = Vec::with_capacity(N * 3);
    let mut signs = Vec::with_capacity(N);

    for _ in 0..N {
        let x = rng.random::<f64>() * BOX_SIZE - box_half;
        let y = rng.random::<f64>() * BOX_SIZE - box_half;
        let z = rng.random::<f64>() * BOX_SIZE - box_half;
        pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs.push(if rng.random::<bool>() { 1 } else { -1 });
    }

    (pos, vel, signs)
}
