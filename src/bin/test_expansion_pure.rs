//! Test: Pure Expansion (No N-body Forces)
//!
//! Diagnostic to verify E_total conservation with VSL dynamics only.
//! All gravitational forces disabled - particles only follow Hubble flow.

use janus::vsl_dynamic::CoupledFriedmann;
use janus::janus_expansion::{compute_total_energy, energy_drift_pct, a_minus_from_a_plus, compute_phi_factors};
use std::io::Write;

// Cosmology
const ETA: f64 = 1.045;
const H0: f64 = 69.9;
const MU: f64 = 19.0;
const OMEGA_B: f64 = 0.05;
const L_BOX: f64 = 200.0;

// Simulation
const Z_INIT: f64 = 5.0;
const Z_FINAL: f64 = 4.5;
const DT: f64 = 0.001;
const MPC_GYR_TO_KMS: f64 = 977.8;

// Janus expansion parameters (same as janus_adaptive_zoom)
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;

fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;
    if a < A_TRANSITION_JANUS {
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
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  TEST: PURE EXPANSION (No N-body Forces)                                 ║");
    println!("║  Verifying E_total conservation with VSL dynamics only                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Parameters: z={} → {}, dt={}, η={}, μ={}                       ║", Z_INIT, Z_FINAL, DT, ETA, MU);
    println!("║  All gravitational forces DISABLED                                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Fixed comoving densities (constant by definition)
    // Using same mass calculation as production
    let n_grid = 50;
    let n_total = n_grid * n_grid * n_grid;
    let n_plus = n_total / (1 + MU as usize);
    let n_minus = n_total - n_plus;

    let rho_crit_0 = 2.775e11 * (H0 / 100.0).powi(2);  // M☉/Mpc³
    let m_plus_total = OMEGA_B * rho_crit_0 * L_BOX.powi(3);
    let m_minus_total = MU * m_plus_total;

    let vol = L_BOX * L_BOX * L_BOX;
    let rho_plus_comoving = m_plus_total / vol;
    let rho_minus_comoving = -m_minus_total / vol;  // Negative!

    println!("Comoving densities (CONSTANT):");
    println!("  ρ⁺_comoving = {:.6e} M☉/Mpc³", rho_plus_comoving);
    println!("  ρ⁻_comoving = {:.6e} M☉/Mpc³", rho_minus_comoving);
    println!("  N+ = {}, N- = {}", n_plus, n_minus);
    println!();

    let mut a = 1.0 / (1.0 + Z_INIT);
    let mut t_gyr = 0.5;
    let mut e_total_0: Option<f64> = None;

    let c_plus = 1.0;  // Constant

    println!("{:>6} {:>8} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10}",
             "step", "z", "a", "c̄", "φ", "E_total", "E_drift%", "a⁻/a⁺");
    println!("{}", "-".repeat(90));

    let mut step = 0;
    loop {
        let z = 1.0 / a - 1.0;

        if z < Z_FINAL {
            println!("\n  Reached z_final = {:.2} at step {}", Z_FINAL, step);
            break;
        }

        // VSL: c̄(z) dynamically computed
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_minus = c_ratio_sq.sqrt();

        // Scale factors for both sectors
        let a_plus = a;
        let a_minus = a_minus_from_a_plus(a, ETA);

        // Phi coupling factor
        let (phi, _phi_inv) = compute_phi_factors(a, ETA);

        // Compute E_total
        let (e_total, e_plus, e_minus) = compute_total_energy(
            rho_plus_comoving, rho_minus_comoving,
            c_plus, c_minus,
            a_plus, a_minus
        );

        if e_total_0.is_none() {
            e_total_0 = Some(e_total);
        }
        let e_drift = energy_drift_pct(e_total, e_total_0.unwrap());

        // Log every 10 steps
        if step % 10 == 0 {
            println!("{:>6} {:>8.4} {:>10.6} {:>12.6} {:>12.6} {:>12.4e} {:>12.4} {:>10.6}",
                     step, z, a, c_minus, phi, e_total, e_drift, a_minus/a_plus);
        }

        // Pure expansion: just update scale factor (no forces!)
        let h = compute_hubble_janus(a, H0);
        let da = a * h * DT;
        a += da;
        t_gyr += DT;

        step += 1;
    }

    // Final summary
    let z_final = 1.0 / a - 1.0;
    let c_ratio_sq_final = CoupledFriedmann::c_ratio_sq_at_z(z_final, ETA);
    let c_minus_final = c_ratio_sq_final.sqrt();
    let a_minus_final = a_minus_from_a_plus(a, ETA);
    let (phi_final, _) = compute_phi_factors(a, ETA);
    let (e_total_final, _, _) = compute_total_energy(
        rho_plus_comoving, rho_minus_comoving,
        c_plus, c_minus_final,
        a, a_minus_final
    );
    let e_drift_final = energy_drift_pct(e_total_final, e_total_0.unwrap());

    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS                                                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  z: {:.4} → {:.4}  (Δz = {:.4})                                        ║", Z_INIT, z_final, Z_INIT - z_final);
    println!("║  a: {:.6} → {:.6}  (ratio = {:.4})                                  ║",
             1.0/(1.0+Z_INIT), a, a * (1.0 + Z_INIT));
    println!("║  c̄: {:.6} → {:.6}  (change = {:.4}%)                               ║",
             CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA).sqrt(),
             c_minus_final,
             (c_minus_final / CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA).sqrt() - 1.0) * 100.0);
    println!("║  φ: {:.6} → {:.6}  (change = {:.4}%)                               ║",
             compute_phi_factors(1.0/(1.0+Z_INIT), ETA).0,
             phi_final,
             (phi_final / compute_phi_factors(1.0/(1.0+Z_INIT), ETA).0 - 1.0) * 100.0);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  E_total: {:.6e} → {:.6e}                                  ║", e_total_0.unwrap(), e_total_final);
    println!("║                                                                          ║");
    println!("║  ████  E_DRIFT = {:.4}%  ████                                          ║", e_drift_final);
    println!("║                                                                          ║");
    if e_drift_final.abs() < 1.0 {
        println!("║  ✓ PASS: E_drift < 1% → VSL dynamics working correctly                 ║");
        println!("║  → Ready for μ=8 production                                            ║");
    } else {
        println!("║  ✗ FAIL: E_drift > 1% → Bug in E calculation                           ║");
        println!("║  → DO NOT launch production, debug required                            ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
