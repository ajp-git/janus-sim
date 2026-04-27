//! Unit test: verify mass_factor gives N-independent FORCES (not dynamics)
//! Measures mean |Δv|/dt for uniform random distribution at N=100k, 500k, 1M
//! Expected: mean |a| ≈ identical to ±5% (single step, no dynamics)

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

use rand::prelude::*;
use rand::SeedableRng;

const BOX_SIZE: f64 = 500.0;
const SEED: u64 = 12345;
const DT: f64 = 0.001;  // Small dt for single force evaluation

fn main() {
    #[cfg(not(feature = "cuda"))]
    { eprintln!("Requires --features cuda"); std::process::exit(1); }
    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MASS_FACTOR UNIT TEST — Force magnitude independence        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Uniform random ICs, measure <|Δv|>/dt after single step     ║");
    println!("║  Expected: <|a|> independent of N (±5%)                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_values = [100_000usize, 500_000, 1_000_000];
    let mut results: Vec<(usize, f64)> = Vec::new();

    for &n in &n_values {
        println!("━━━ N = {} ━━━", n);

        // Calculate expected mass_factor
        let g_cosmo = 4.498e-12_f64;  // Mpc³/(M_sun·Gyr²)
        let rho_crit = 2.775e11_f64;  // M_sun/Mpc³
        let omega_m = 0.3_f64;
        let m_total = omega_m * rho_crit * BOX_SIZE.powi(3);
        let expected_mf = g_cosmo * m_total / n as f64;
        println!("  Expected mass_factor = {:.6e}", expected_mf);

        // Generate uniform random ICs with same seed
        let (pos, vel, signs) = generate_uniform_ics(n);

        let mut sim = GpuNBodyTwoPass::with_custom_ics(
            pos.clone(), vel.clone(), signs.clone(), BOX_SIZE
        ).expect("Failed to create simulation");

        sim.set_theta(0.7);
        sim.set_softening(0.5);

        println!("  Actual mass_factor = {:.6e}", sim.get_mass_factor());

        // Get initial velocities (all zero)
        let vel_before = vel.clone();

        // Single step to compute forces
        sim.step_dkd(DT, 0.07, 1.0).expect("Step failed");

        // Get velocities after step
        let vel_after = sim.get_velocities().expect("Failed to get velocities");

        // Compute mean |Δv|/dt = mean |a|
        let mut sum_a = 0.0;
        for i in 0..n {
            let dvx = (vel_after[i * 3] - vel_before[i * 3]) as f64;
            let dvy = (vel_after[i * 3 + 1] - vel_before[i * 3 + 1]) as f64;
            let dvz = (vel_after[i * 3 + 2] - vel_before[i * 3 + 2]) as f64;
            let dv = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
            sum_a += dv / DT;
        }
        let mean_a = sum_a / n as f64;

        println!("  <|a|> = {:.6e} Mpc/Gyr²", mean_a);
        results.push((n, mean_a));
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Use 1M as reference
    let ref_a = results[2].1;
    let mut all_pass = true;

    for (n, mean_a) in &results {
        let diff = (mean_a - ref_a).abs() / ref_a * 100.0;
        let status = if diff <= 5.0 { "✓" } else { all_pass = false; "✗" };
        println!("  {} N={:>7}: <|a|> = {:.6e}, Δ = {:.1}%", status, n, mean_a, diff);
    }

    println!();
    if all_pass {
        println!("  ✓ PASS — mass_factor gives N-independent forces");
    } else {
        println!("  ✗ FAIL — mass_factor bug detected");
    }
}

#[cfg(feature = "cuda")]
fn generate_uniform_ics(n: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
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
