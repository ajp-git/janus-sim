/// Sound Horizon Calculator for Janus Cosmology
///
/// Computes r_d = ∫₀^{a_dec} c_s da / (a² H(a))
/// where:
///   - a_dec = 1/1101 (decoupling at z=1100)
///   - c_s = c/√3 (sound speed in photon-baryon fluid)
///
/// Compares Janus r_d to ΛCDM r_d ≈ 147 Mpc

use janus::{
    friedmann::{JanusParams, CosmoInterpolator, sound_horizon_to_z},
    constants::C,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Sound Horizon Calculator — Janus Cosmology                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Physical constants
    let h0_km_s_mpc = 76.0; // From joint fit
    let mpc_in_m = 3.085677581e22;

    // Convert c/H₀ to Mpc
    // H₀ = 76 km/s/Mpc = 76e3 m/s / Mpc = 76e3 / 3.086e22 s⁻¹
    let h0_si = h0_km_s_mpc * 1000.0 / mpc_in_m; // in s⁻¹
    let c_over_h0_mpc = C / h0_si / mpc_in_m;

    println!("Physical constants:");
    println!("  H₀ = {} km/s/Mpc", h0_km_s_mpc);
    println!("  c/H₀ = {:.1} Mpc", c_over_h0_mpc);
    println!();

    // Radiation parameter
    let omega_r = 9.2e-5;
    println!("Radiation density: Ω_r = {:.2e}", omega_r);
    println!();

    // Test different η values (include both η > 1 and η < 1 regimes)
    let eta_values = [1.045, 1.00, 0.95, 0.90, 0.80, 0.70, 0.50];

    println!("══════════════════════════════════════════════════════════════════");
    println!("                    SOUND HORIZON RESULTS                         ");
    println!("══════════════════════════════════════════════════════════════════\n");

    println!("{:>8} {:>12} {:>12} {:>12} {:>12}",
             "η", "r_d (c/H₀)", "r_d (Mpc)", "ΛCDM (Mpc)", "Δ%");
    println!("{:-<60}", "");

    let r_d_lcdm = 147.0; // Planck 2018 value

    for &eta in &eta_values {
        let params = JanusParams::from_eta_with_radiation(eta, omega_r);

        println!("\n=== η = {:.3} ===", eta);
        println!("  Ω₊ = {:.4}, Ω₋ = {:.4}", params.omega_plus, params.omega_minus);
        println!("  E = Ω₊ - Ω₋ = {:.4}", params.e_conserved);
        println!("  q₀ = (1-η)/(1+η) = {:.4}", (1.0 - eta) / (1.0 + eta));

        // Create interpolator to z=1100
        let cosmo = CosmoInterpolator::new_to_cmb(&params);

        // Check we reached high enough z
        let (a_start, h_start) = cosmo.get_params_at_tau(cosmo.tau_start);
        let z_max = 1.0 / a_start - 1.0;
        println!("  Integration range: z = 0 → {:.0}", z_max);
        println!("  a_min = {:.6}, H_min/H₀ = {:.4}", a_start, h_start);

        // Check if integration reached past properly
        if z_max < 1000.0 {
            println!("  ⚠ WARNING: Integration did not reach z=1100!");
            println!("    For η > 1 (E < 0), Janus cosmology may not have a standard Big Bang.");
            println!("    The sound horizon is undefined in this regime.");
            continue;
        }

        // Compute sound horizon (integrate from a_min to z=1100)
        let r_d_dim = sound_horizon_to_z(&cosmo.history, 1100.0);
        let r_d_mpc = r_d_dim * c_over_h0_mpc;

        let delta_pct = (r_d_mpc - r_d_lcdm) / r_d_lcdm * 100.0;

        println!("  r_d = {:.4} × (c/H₀) = {:.1} Mpc", r_d_dim, r_d_mpc);
        println!("  ΛCDM r_d = {:.1} Mpc → Δ = {:+.1}%", r_d_lcdm, delta_pct);
    }

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("DETAILED ANALYSIS for η = 1.045 (Pantheon+ best fit)");
    println!("══════════════════════════════════════════════════════════════════\n");

    let eta = 1.045;
    let params = JanusParams::from_eta_with_radiation(eta, omega_r);
    let cosmo = CosmoInterpolator::new_to_cmb(&params);

    // Dump some history points
    println!("Evolution history (selected points):");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12}", "z", "a", "ā", "H/H₀", "ā̇");
    println!("{:-<60}", "");

    let z_samples = [0.0, 1.0, 10.0, 100.0, 500.0, 1000.0];
    for &z in &z_samples {
        let a_target = 1.0 / (1.0 + z);
        // Find closest point in history
        if let Some(state) = cosmo.history.iter().min_by(|s1, s2| {
            (s1.a - a_target).abs().partial_cmp(&(s2.a - a_target).abs()).unwrap()
        }) {
            let z_actual = 1.0 / state.a - 1.0;
            println!("{:>8.1} {:>12.6} {:>12.6} {:>12.4} {:>12.4}",
                     z_actual, state.a, state.a_bar, state.hubble(), state.a_bar_dot);
        }
    }

    // Verify integration
    let r_d_dim = sound_horizon_to_z(&cosmo.history, 1100.0);
    let r_d_mpc = r_d_dim * c_over_h0_mpc;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("FINAL RESULT");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("\n  Sound horizon at decoupling (z=1100):");
    println!("    r_d (Janus, η={:.3}) = {:.1} Mpc", eta, r_d_mpc);
    println!("    r_d (ΛCDM, Planck)   = {:.1} Mpc", r_d_lcdm);

    let delta = r_d_mpc - r_d_lcdm;
    let delta_pct = delta / r_d_lcdm * 100.0;
    println!("\n  Difference: {:+.1} Mpc ({:+.1}%)", delta, delta_pct);

    if delta_pct.abs() < 5.0 {
        println!("\n  ✓ COMPATIBLE with ΛCDM (< 5% difference)");
    } else if delta_pct.abs() < 10.0 {
        println!("\n  ~ MARGINALLY COMPATIBLE (5-10% difference)");
    } else {
        println!("\n  ✗ SIGNIFICANT TENSION with ΛCDM (> 10% difference)");
    }

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("Done.");
    println!("══════════════════════════════════════════════════════════════════\n");
}
