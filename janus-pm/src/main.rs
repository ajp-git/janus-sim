/// Janus Particle-Mesh FFT Solver
/// Target: 20M particles on RTX 3060
///
/// Reuses CosmoInterpolator and JanusParams from janus-sim

use janus::friedmann::{JanusParams, CosmoInterpolator};
use janus::MassSign;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus PM-FFT Solver — Skeleton                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Verify imports from janus-sim work
    let params = JanusParams::from_eta(1.045);
    println!("η = {:.3}", params.eta);
    println!("Ω₊ = {:.4}, Ω₋ = {:.4}", params.omega_plus, params.omega_minus);
    println!("E = {:.4}", params.e_conserved);

    let cosmo = CosmoInterpolator::new(&params, 5.0);
    println!("\nCosmoInterpolator: τ ∈ [{:.4}, {:.4}]", cosmo.tau_start, cosmo.tau_end);

    let (a, h) = cosmo.get_params_at_tau(0.0);
    println!("At z=0: a = {:.4}, H/H₀ = {:.4}", a, h);

    // Verify MassSign
    let sign_pos = MassSign::Positive;
    let sign_neg = MassSign::Negative;
    println!("\nMassSign: {:?}, {:?}", sign_pos, sign_neg);

    println!("\n✓ All imports from janus-sim working correctly.");
    println!("\nNext: PM-1 (CuFFT test)");
}
