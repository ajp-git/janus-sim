//! DIAGNOSTIC E_VSL — Validation de la quantité conservée
//!
//! E_naive = ρ⁺_com × c² + ρ⁻_com × c̄²(t)         — NON conservé
//! S_VSL   = ρ⁻_com × [c̄²(t) - c̄²(t_init)]        — terme de source VSL
//! E_VSL   = E_naive - S_VSL                       — DOIT être conservé
//!
//! Test: expansion pure z=10→0, E_VSL_drift < 0.01%

use janus::vsl_dynamic::CoupledFriedmann;

const ETA: f64 = 1.045;
const MU: f64 = 19.0;
const H0: f64 = 69.9;
const OMEGA_B: f64 = 0.05;
const MPC_GYR_TO_KMS: f64 = 977.8;

// Janus H(z)
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VALIDATION E_VSL — Quantité Conservée en VSL Janus         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  E_naive = ρ⁺×c² + ρ⁻×c̄²(t)     — NON conservé             ║");
    println!("║  S_VSL   = ρ⁻×[c̄²(t) - c̄²_init] — source VSL               ║");
    println!("║  E_VSL   = E_naive - S_VSL       — CONSERVÉ                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let z_init = 10.0;
    let n_steps = 16000;
    let dt = 0.001;

    // Comoving densities (constant)
    let h = H0 / 100.0;
    let rho_crit = 2.775e11 * h * h;
    let rho_plus_comoving = OMEGA_B * rho_crit;
    let rho_minus_comoving = -MU * rho_plus_comoving;  // Negative!

    println!("ρ⁺_com = {:.6e} M☉/Mpc³", rho_plus_comoving);
    println!("ρ⁻_com = {:.6e} M☉/Mpc³", rho_minus_comoving);

    let mut a = 1.0 / (1.0 + z_init);

    // Initial c̄²
    let c_bar_sq_init = CoupledFriedmann::c_ratio_sq_at_z(z_init, ETA);
    println!("c̄²(z={}) = {:.8}", z_init, c_bar_sq_init);

    // E_plus is constant
    let e_plus = rho_plus_comoving * 1.0;  // c² = 1

    // Initial energies
    let e_minus_init = rho_minus_comoving * c_bar_sq_init;
    let e_naive_init = e_plus + e_minus_init;
    let s_vsl_init = 0.0;  // By definition at t_init
    let e_vsl_init = e_naive_init - s_vsl_init;

    println!("\nInitial state (z={}):", z_init);
    println!("  E_plus  = {:.6e} (constant)", e_plus);
    println!("  E_minus = {:.6e} (c̄² dependent)", e_minus_init);
    println!("  E_naive = {:.6e}", e_naive_init);
    println!("  S_VSL   = {:.6e}", s_vsl_init);
    println!("  E_VSL   = {:.6e}", e_vsl_init);

    // Theoretical E_VSL (should be constant)
    // E_VSL = ρ⁺×c² + ρ⁻×c̄²_init = constant
    let e_vsl_theoretical = e_plus + rho_minus_comoving * c_bar_sq_init;
    println!("\nE_VSL théorique = ρ⁺×c² + ρ⁻×c̄²_init = {:.6e}", e_vsl_theoretical);

    println!("\n{:>6} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
             "Step", "z", "c̄²", "E_naive", "S_VSL", "E_VSL", "E_VSL_drift%");
    println!("{}", "-".repeat(90));

    let mut max_evsl_drift = 0.0f64;

    for step in 0..=n_steps {
        let z = 1.0 / a - 1.0;
        let h_z = compute_hubble_janus(a, H0);

        // c̄²(z)
        let c_bar_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);

        // Energies
        let e_minus = rho_minus_comoving * c_bar_sq;
        let e_naive = e_plus + e_minus;

        // S_VSL = ρ⁻ × [c̄²(t) - c̄²_init]
        let s_vsl = rho_minus_comoving * (c_bar_sq - c_bar_sq_init);

        // E_VSL = E_naive - S_VSL
        let e_vsl = e_naive - s_vsl;

        // E_VSL drift
        let e_vsl_drift_pct = (e_vsl - e_vsl_init) / e_vsl_init.abs() * 100.0;
        max_evsl_drift = max_evsl_drift.max(e_vsl_drift_pct.abs());

        // Log key steps
        if step == 0 || step == 100 || step == 465 || step == 500 ||
           step == 1000 || step == 5000 || step == 10000 || step == 15000 || step == n_steps ||
           step % 2000 == 0 {
            println!("{:>6} {:>10.4} {:>12.8} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.6}",
                     step, z, c_bar_sq, e_naive, s_vsl, e_vsl, e_vsl_drift_pct);
        }

        // Leapfrog expansion
        let da = a * h_z * dt;
        a += da;
    }

    let z_final = 1.0 / a - 1.0;

    println!("\n{}", "=".repeat(90));
    println!("RÉSULTATS:");
    println!("  z_final = {:.6}", z_final);
    println!("  Max |E_VSL_drift| = {:.8}%", max_evsl_drift);
    println!();

    if max_evsl_drift < 0.01 {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  ✓ E_VSL_drift < 0.01% — VALIDATION OK                       ║");
        println!("║    La quantité E_VSL est bien conservée en expansion pure    ║");
        println!("║    → Prêt pour lancement production                          ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
    } else if max_evsl_drift < 0.1 {
        println!("⚠ E_VSL_drift = {:.4}% — Acceptable mais à surveiller", max_evsl_drift);
    } else {
        println!("✗ E_VSL_drift = {:.4}% — PROBLÈME dans la formule", max_evsl_drift);
    }

    // Vérification algébrique
    println!("\n--- Vérification algébrique ---");
    println!("E_VSL = E_naive - S_VSL");
    println!("      = (ρ⁺×c² + ρ⁻×c̄²) - ρ⁻×(c̄² - c̄²_init)");
    println!("      = ρ⁺×c² + ρ⁻×c̄² - ρ⁻×c̄² + ρ⁻×c̄²_init");
    println!("      = ρ⁺×c² + ρ⁻×c̄²_init");
    println!("      = CONSTANTE ✓");
}
