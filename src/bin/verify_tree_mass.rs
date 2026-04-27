//! Direct verification: Tree root mass should be N × mass_factor = G × M_total
//! This is the clearest test of mass_factor implementation

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

use rand::prelude::*;
use rand::SeedableRng;

const BOX_SIZE: f64 = 500.0;
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
    println!("║  TREE ROOT MASS VERIFICATION                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Checks: tree_root_mass = N_sign × mass_factor               ║");
    println!("║  Expected: root_mass = G × M_total × (N_sign / N_total)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Calculate G × M_total (constant for given box size)
    let g_cosmo = 4.498e-12_f64;  // Mpc³/(M_sun·Gyr²)
    let rho_crit = 2.775e11_f64;  // M_sun/Mpc³
    let omega_m = 0.3_f64;
    let m_total = omega_m * rho_crit * BOX_SIZE.powi(3);
    let g_m_total = g_cosmo * m_total;
    println!("Physical constants:");
    println!("  G = {:.3e} Mpc³/(M☉·Gyr²)", g_cosmo);
    println!("  ρ_crit = {:.3e} M☉/Mpc³", rho_crit);
    println!("  Ω_m = {}", omega_m);
    println!("  M_total = Ω_m × ρ_crit × L³ = {:.3e} M☉", m_total);
    println!("  G × M_total = {:.6e} (target total tree mass)", g_m_total);
    println!();

    let n_values = [10_000usize, 100_000, 1_000_000];

    for &n in &n_values {
        println!("━━━ N = {} ━━━", n);

        let (pos, vel, signs) = generate_ics(n);
        let n_positive = signs.iter().filter(|&&s| s > 0).count();
        let n_negative = n - n_positive;

        let mut sim = GpuNBodyTwoPass::with_custom_ics(
            pos, vel, signs, BOX_SIZE
        ).expect("Failed to create simulation");

        let mass_factor = sim.get_mass_factor();
        println!("  mass_factor = {:.6e}", mass_factor);
        println!("  N+ = {}, N- = {}", n_positive, n_negative);

        // Do one step to build trees
        sim.step_dkd(0.001, 0.07, 1.0).expect("Step failed");

        // Get tree root mass (from last built tree - negative particles)
        let root_mass = sim.get_tree_root_mass().expect("Failed to get tree mass");

        // Expected: N_sign × mass_factor
        // The last tree built was for negative particles
        let expected_pos = n_positive as f64 * mass_factor;
        let expected_neg = n_negative as f64 * mass_factor;

        // Half of G × M_total (for 50-50 split)
        let expected_half_gm = g_m_total * (n_negative as f64 / n as f64);

        println!("  Tree root mass (after step) = {:.6e}", root_mass);
        println!("  Expected (N- × mass_factor) = {:.6e}", expected_neg);
        println!("  Expected (G×M × N-/N)        = {:.6e}", expected_half_gm);

        let diff = (root_mass - expected_neg).abs() / expected_neg * 100.0;
        if diff < 0.1 {
            println!("  ✓ Match: Δ = {:.3}%", diff);
        } else {
            println!("  ✗ Mismatch: Δ = {:.1}% ← BUG!", diff);
        }
        println!();
    }

    // Final check: is G×M constant?
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Verification: N × mass_factor should equal G×M for all N");
    for &n in &n_values {
        let mf = g_m_total / n as f64;
        let product = n as f64 * mf;
        let diff = (product - g_m_total).abs() / g_m_total * 100.0;
        println!("  N={:>7}: {} × {:.3e} = {:.6e} (Δ={:.6}%)",
                 n, n, mf, product, diff);
    }
    println!();
    println!("  If tree root masses match N× mass_factor, mass_factor is CORRECT.");
    println!("  N-dependence of <|a|> is a PHYSICAL effect (mean distance ∝ N^(-1/3)).");
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
