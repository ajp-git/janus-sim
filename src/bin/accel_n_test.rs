//! Test: verify mean acceleration is N-independent after reduce_tp fix
//! With mass_factor = G×M/N, the mean |a| should NOT depend on N

use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const BOX_SIZE: f64 = 500.0;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.5;
const SEED: u64 = 42;

fn main() {
    #[cfg(not(feature = "cuda"))]
    { eprintln!("Requires --features cuda"); std::process::exit(1); }
    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ACCELERATION N-INDEPENDENCE TEST                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  With mass_factor = G×M/N, <|a|> should be N-independent     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_values = [50_000usize, 100_000, 250_000, 500_000];
    let mut results = Vec::new();

    for &n in &n_values {
        let start = Instant::now();

        // Use GRID positions for deterministic test
        let (pos, vel, signs) = generate_grid_ics(n);

        let mut sim = GpuNBodyTwoPass::with_custom_ics(
            pos, vel, signs, BOX_SIZE
        ).expect("Failed to create simulation");

        sim.set_theta(THETA);
        sim.set_softening(SOFTENING);

        let mass_factor = sim.get_mass_factor();

        // Do one step to compute accelerations
        sim.step_dkd(0.001, 0.07, 1.0).expect("Step failed");

        // Get mean |a| using existing method
        let sum_a = sim.acceleration_sum().expect("accel sum");
        let n_real = sim.n_particles();
        let mean_a = sum_a / n_real as f64;

        let elapsed = start.elapsed().as_secs_f64();
        println!("N = {:>7}: <|a|> = {:.6e} Mpc/Gyr², mass_factor = {:.4e}, t = {:.1}s",
                 n, mean_a, mass_factor, elapsed);

        results.push((n, mean_a));
    }

    // Analysis
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  N-INDEPENDENCE CHECK                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Use largest N as reference
    let ref_a = results.last().unwrap().1;
    let mut all_pass = true;

    for (n, mean_a) in &results {
        let ratio = mean_a / ref_a;
        let pct = (ratio - 1.0).abs() * 100.0;
        let status = if pct < 10.0 { "✓" } else { all_pass = false; "✗" };
        println!("  {} N = {:>7}: <|a|>/<|a|>_ref = {:.3} ({:+.1}%)", status, n, ratio, (ratio - 1.0) * 100.0);
    }

    println!();
    if all_pass {
        println!("✓ ALL N VALUES WITHIN ±10% — N-independence VERIFIED");
    } else {
        println!("✗ FAILED — <|a|> varies with N, physics issue suspected");

        // Check scaling
        let n1 = results[0].0 as f64;
        let n2 = results.last().unwrap().0 as f64;
        let a1 = results[0].1;
        let a2 = results.last().unwrap().1;
        let alpha = (a1 / a2).ln() / (n1 / n2).ln();
        println!("  Scaling: <|a|> ∝ N^{:.2}", alpha);
    }
}

#[cfg(feature = "cuda")]
fn generate_grid_ics(n: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // Create grid with ~n particles
    let n_grid = (n as f64).powf(1.0/3.0).ceil() as usize;
    let cell = BOX_SIZE / n_grid as f64;
    let box_half = BOX_SIZE / 2.0;

    let mut pos = Vec::new();
    let mut vel = Vec::new();
    let mut signs = Vec::new();

    let mut count = 0;
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                if count >= n { break; }

                // Grid position with small random jitter
                let x = (ix as f64 + 0.5 + 0.1 * (rng.random::<f64>() - 0.5)) * cell - box_half;
                let y = (iy as f64 + 0.5 + 0.1 * (rng.random::<f64>() - 0.5)) * cell - box_half;
                let z = (iz as f64 + 0.5 + 0.1 * (rng.random::<f64>() - 0.5)) * cell - box_half;

                pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
                vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
                // Deterministic sign pattern
                signs.push(if (ix + iy + iz) % 2 == 0 { 1 } else { -1 });
                count += 1;
            }
        }
    }

    // Fill remaining with last position repeated (for exact n)
    while count < n {
        let x = (rng.random::<f64>() - 0.5) * BOX_SIZE;
        let y = (rng.random::<f64>() - 0.5) * BOX_SIZE;
        let z = (rng.random::<f64>() - 0.5) * BOX_SIZE;
        pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs.push(if count % 2 == 0 { 1 } else { -1 });
        count += 1;
    }

    (pos, vel, signs)
}
