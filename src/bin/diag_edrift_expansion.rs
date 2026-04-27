//! DIAGNOSTIC E_drift — Expansion Pure (sans gravité)
//!
//! ÉTAPE 1: Test z=4 → z=2, régime matter, 1000 steps
//! ÉTAPE 2: Test z=4.6 → z=4.4, traversée transition, 500 steps
//!
//! Si E_drift élevé sans gravité → bug dans calcul E lui-même

use janus::vsl_dynamic::CoupledFriedmann;
use janus::janus_expansion::{compute_total_energy, energy_drift_pct, a_minus_from_a_plus};

const ETA: f64 = 1.045;
const MU: f64 = 19.0;
const H0: f64 = 69.9;
const OMEGA_B: f64 = 0.05;
const MPC_GYR_TO_KMS: f64 = 977.8;

// Janus H(z) from janus_jpp_production
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;

fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;
    if a < ALPHA_SQ_JANUS {
        h0_gyr_inv / a.powf(1.5)
    } else {
        let cosh2_mu = a / ALPHA_SQ_JANUS;
        let cosh2_mu_safe = cosh2_mu.max(1.0);
        let cosh_mu = cosh2_mu_safe.sqrt();
        let mu_p = cosh_mu.acosh();
        let s2mu = (2.0 * mu_p).sinh();
        s2mu / (TAU_0_JANUS * ALPHA_SQ_JANUS * cosh2_mu_safe * (1.0 + 0.5 * s2mu))
    }
}

fn run_expansion_test(label: &str, z_init: f64, n_steps: usize, dt: f64) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  {} ", label);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  z_init = {:.2}, dt = {} Gyr, {} steps", z_init, dt, n_steps);
    println!("║  GRAVITÉ: OFF (expansion cosmologique pure)", );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Comoving densities (constant)
    let h = H0 / 100.0;
    let rho_crit = 2.775e11 * h * h;  // M☉/Mpc³
    let rho_plus_comoving = OMEGA_B * rho_crit;
    let rho_minus_comoving = -MU * rho_plus_comoving;

    println!("ρ⁺_com = {:.4e} M☉/Mpc³ (constant)", rho_plus_comoving);
    println!("ρ⁻_com = {:.4e} M☉/Mpc³ (constant)", rho_minus_comoving);

    let mut a = 1.0 / (1.0 + z_init);
    let mut e_total_0: Option<f64> = None;
    let mut max_drift = 0.0f64;
    let mut drift_at_100 = 0.0;
    let mut drift_at_500 = 0.0;
    let mut crossed_transition = false;

    println!("\n{:>6} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
             "Step", "z", "a", "c̄(z)", "E_total", "E_drift%", "Era");
    println!("{}", "-".repeat(80));

    for step in 0..=n_steps {
        let z = 1.0 / a - 1.0;
        let h_z = compute_hubble_janus(a, H0);

        // c̄(z) from VSL
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_bar = c_ratio_sq.sqrt();

        // a_minus for reference
        let a_minus = a_minus_from_a_plus(a, ETA);

        // E_total = ρ⁺_com × c² + ρ⁻_com × c̄²
        let (e_total, e_plus, e_minus) = compute_total_energy(
            rho_plus_comoving, rho_minus_comoving,
            1.0, c_bar,
            a, a_minus
        );

        if e_total_0.is_none() {
            e_total_0 = Some(e_total);
        }
        let e_drift = energy_drift_pct(e_total, e_total_0.unwrap());
        max_drift = max_drift.max(e_drift.abs());

        let era = if a < ALPHA_SQ_JANUS { "gauge" } else { "matter" };

        // Check transition crossing
        if !crossed_transition && a >= ALPHA_SQ_JANUS && step > 0 {
            crossed_transition = true;
            println!("  >>> TRANSITION CROSSED at step {} (z = {:.4}) <<<", step, z);
        }

        // Log at key steps
        if step == 0 || step == 100 || step == 500 || step == n_steps || step % 200 == 0 {
            println!("{:>6} {:>10.4} {:>10.6} {:>12.6} {:>12.4e} {:>10.4} {:>10}",
                     step, z, a, c_bar, e_total, e_drift, era);
        }

        if step == 100 { drift_at_100 = e_drift; }
        if step == 500 { drift_at_500 = e_drift; }

        // Leapfrog expansion (no gravity, just da = a × H × dt)
        let da = a * h_z * dt;
        a += da;
    }

    let z_final = 1.0 / a - 1.0;

    println!("\n{}", "=".repeat(80));
    println!("RESULTS:");
    println!("  z_final = {:.4}", z_final);
    println!("  E_drift at step 100: {:.4}%", drift_at_100);
    println!("  E_drift at step 500: {:.4}%", drift_at_500);
    println!("  Max |E_drift|: {:.4}%", max_drift);
    println!();

    if max_drift < 0.1 {
        println!("  ✓ E_drift < 0.1% → Calcul E cosmologique OK");
        println!("    → Le drift vient du N-body (gravité)");
    } else if max_drift < 1.0 {
        println!("  ⚠ E_drift ~ {}% sans gravité", max_drift);
        println!("    → Problème possible dans calcul E ou c̄(z)");
    } else {
        println!("  ✗ E_drift = {}% sans gravité → BUG dans formule E!", max_drift);
    }
    println!("{}", "=".repeat(80));
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DIAGNOSTIC E_drift — Expansion Pure (Gravité OFF)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Étape 1: Matter era pure (z=4 → z~2)
    run_expansion_test("ÉTAPE 1: Matter Era (z=4 → z~2)", 4.0, 1000, 0.001);

    // Étape 2: Traversée transition (z=4.6 → z~4.4)
    run_expansion_test("ÉTAPE 2: Transition Crossing (z=4.6 → z~4.4)", 4.6, 500, 0.001);

    // Bonus: Full run z=10 → z=0 sans gravité
    println!("\n");
    run_expansion_test("BONUS: Full Range (z=10 → z~0)", 10.0, 16000, 0.001);
}
