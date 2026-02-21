use janus::friedmann::{JanusParams, CosmoInterpolator};

fn main() {
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, 5.0);

    // Get state at z=0 for normalization and verification
    let state_z0 = cosmo.get_state_at_tau(cosmo.tau_end);
    let h0 = state_z0.a_dot / state_z0.a;

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("Janus H(z) export — η = {:.4}", eta);
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("Parameters:");
    eprintln!("  Ω₊ = {:.6}", params.omega_plus);
    eprintln!("  Ω₋ = {:.6}", params.omega_minus);
    eprintln!("  E  = Ω₊ - Ω₋ = {:.6}", params.e_conserved);
    eprintln!();
    eprintln!("Integration range:");
    eprintln!("  tau_start = {:.6} (z=5)", cosmo.tau_start);
    eprintln!("  tau_end   = {:.6} (z=0)", cosmo.tau_end);
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════");
    eprintln!("VERIFICATION at z=0:");
    eprintln!("══════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  a(z=0)    = {:.6}  (expected: 1.0)", state_z0.a);
    eprintln!("  ā(z=0)    = {:.6}  (expected: 1.0)", state_z0.a_bar);
    eprintln!();
    eprintln!("  ȧ(z=0)    = {:.6}  (expected: √Ω₊ = {:.6})",
              state_z0.a_dot, params.omega_plus.sqrt());
    eprintln!("  ā̇(z=0)    = {:.6}  (expected: -√Ω₋ = {:.6})",
              state_z0.a_bar_dot, -params.omega_minus.sqrt());
    eprintln!();
    eprintln!("  H(z=0)    = ȧ/a = {:.6}", h0);
    eprintln!("  H̄(z=0)    = ā̇/ā = {:.6}", state_z0.a_bar_dot / state_z0.a_bar);
    eprintln!();

    // Check errors
    let err_a = (state_z0.a - 1.0).abs();
    let err_a_bar = (state_z0.a_bar - 1.0).abs();
    let err_a_dot = (state_z0.a_dot - params.omega_plus.sqrt()).abs();
    let err_a_bar_dot = (state_z0.a_bar_dot + params.omega_minus.sqrt()).abs();

    eprintln!("Errors:");
    eprintln!("  |a - 1|        = {:.2e}  {}", err_a,
              if err_a < 1e-4 { "✓" } else { "✗ FAIL" });
    eprintln!("  |ā - 1|        = {:.2e}  {}", err_a_bar,
              if err_a_bar < 1e-4 { "✓" } else { "✗ FAIL" });
    eprintln!("  |ȧ - √Ω₊|      = {:.2e}  {}", err_a_dot,
              if err_a_dot < 1e-4 { "✓" } else { "✗ FAIL" });
    eprintln!("  |ā̇ - (-√Ω₋)|   = {:.2e}  {}", err_a_bar_dot,
              if err_a_bar_dot < 1e-4 { "✓" } else { "✗ FAIL" });
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════");
    eprintln!();

    // CSV output
    println!("z,a,a_bar,a_dot,a_bar_dot,H_plus,H_minus,H_combined,H_quadratic,H_over_H0");
    let n = 100;
    for i in 0..=n {
        let z = 5.0 * (i as f64) / (n as f64);
        // tau croît avec a, donc tau croît quand z décroît
        // z=5 → tau_start, z=0 → tau_end
        let tau = cosmo.tau_start +
                  (cosmo.tau_end - cosmo.tau_start) *
                  (1.0 - z / 5.0);
        let state = cosmo.get_state_at_tau(tau);

        // H_plus = ȧ/a (secteur positif, toujours > 0)
        let h_plus = state.a_dot / state.a;

        // H_minus = |ā̇/ā| (secteur négatif, valeur absolue car ā̇ < 0)
        let h_minus = (state.a_bar_dot / state.a_bar).abs();

        // H_combined = moyenne arithmétique
        let h_combined = (h_plus + h_minus) / 2.0;

        // H_quadratic = moyenne quadratique (RMS)
        let h_quadratic = ((h_plus * h_plus + h_minus * h_minus) / 2.0).sqrt();

        println!("{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                 z, state.a, state.a_bar, state.a_dot, state.a_bar_dot,
                 h_plus, h_minus, h_combined, h_quadratic, h_plus / h0);
    }
}
