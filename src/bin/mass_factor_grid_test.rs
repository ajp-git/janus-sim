//! Unit test: verify mass_factor with GRID distribution (no Poisson noise)
//! Particles on regular grid → deterministic density → N-independent <|a|>

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

use rand::prelude::*;
use rand::SeedableRng;

const BOX_SIZE: f64 = 500.0;
const SEED: u64 = 12345;
const DT: f64 = 0.001;

fn main() {
    #[cfg(not(feature = "cuda"))]
    { eprintln!("Requires --features cuda"); std::process::exit(1); }
    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MASS_FACTOR GRID TEST — Deterministic force independence    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Regular grid ICs (no Poisson noise), measure <|a|>          ║");
    println!("║  Expected: <|a|> independent of grid resolution (±5%)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Test with different grid sizes: 32³=32768, 50³=125000, 64³=262144
    let grid_sizes = [32usize, 50, 64];
    let mut results: Vec<(usize, f64, f64)> = Vec::new();  // (n, mean_a, mass_factor)

    for &ng in &grid_sizes {
        let n = ng * ng * ng;
        println!("━━━ Grid {}³ = {} particles ━━━", ng, n);

        // Calculate expected mass_factor
        let g_cosmo = 4.498e-12_f64;
        let rho_crit = 2.775e11_f64;
        let omega_m = 0.3_f64;
        let m_total = omega_m * rho_crit * BOX_SIZE.powi(3);
        let expected_mf = g_cosmo * m_total / n as f64;
        println!("  mass_factor = {:.6e}", expected_mf);

        let (pos, vel, signs) = generate_grid_ics(ng);

        let mut sim = GpuNBodyTwoPass::with_custom_ics(
            pos.clone(), vel.clone(), signs.clone(), BOX_SIZE
        ).expect("Failed to create simulation");

        sim.set_theta(0.7);
        sim.set_softening(0.5);

        // Get initial velocities (all zero)
        let vel_before = vel.clone();

        // Single step
        sim.step_dkd(DT, 0.07, 1.0).expect("Step failed");

        // Get velocities after step
        let vel_after = sim.get_velocities().expect("Failed to get velocities");

        // Compute mean |a|
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
        results.push((n, mean_a, expected_mf));
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Use largest grid as reference
    let ref_a = results[2].1;
    let mut all_pass = true;

    for (n, mean_a, mf) in &results {
        let diff = (mean_a - ref_a).abs() / ref_a * 100.0;
        let status = if diff <= 5.0 { "✓" } else { all_pass = false; "✗" };
        println!("  {} N={:>6}: <|a|> = {:.6e}, mass_factor = {:.2e}, Δ = {:.1}%",
                 status, n, mean_a, mf, diff);
    }

    println!();
    if all_pass {
        println!("  ✓ PASS — mass_factor gives N-independent forces (grid test)");
    } else {
        println!("  ✗ FAIL — forces still N-dependent on grid");
        println!();
        // Additional diagnostic: check scaling
        let a1 = results[0].1;
        let a2 = results[2].1;
        let n1 = results[0].0 as f64;
        let n2 = results[2].0 as f64;
        let slope = (a1.ln() - a2.ln()) / (n1.ln() - n2.ln());
        println!("  Scaling: <|a|> ∝ N^{:.2}", slope);
        if slope.abs() < 0.1 {
            println!("  (slope ≈ 0: N-independent, as expected)");
        } else if (slope + 0.5).abs() < 0.1 {
            println!("  (slope ≈ -0.5: suggests missing √N factor somewhere)");
        } else if (slope + 1.0).abs() < 0.1 {
            println!("  (slope ≈ -1: mass_factor not applied!)");
        }
    }
}

#[cfg(feature = "cuda")]
fn generate_grid_ics(n_grid: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let n = n_grid * n_grid * n_grid;
    let cell = BOX_SIZE / n_grid as f64;
    let box_half = BOX_SIZE / 2.0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                // Particle at cell center
                let x = (ix as f64 + 0.5) * cell - box_half;
                let y = (iy as f64 + 0.5) * cell - box_half;
                let z = (iz as f64 + 0.5) * cell - box_half;
                pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
                vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
                // Random sign (but deterministic from seed)
                signs.push(if rng.random::<bool>() { 1 } else { -1 });
            }
        }
    }

    (pos, vel, signs)
}
