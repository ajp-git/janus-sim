//! Diagnostic: Compare a_minus formulas
//!
//! Run: cargo run --bin diag_a_minus_formula

use janus::janus_expansion::a_minus_from_a_plus;
use janus::vsl_dynamic::CoupledFriedmann;

fn main() {
    let eta = 1.045;
    let z_init = 10.0;
    let delta = (eta - 1.0) / eta;

    let a_plus_init = 1.0 / (1.0 + z_init);

    // Formula from vsl_dynamic.rs (Petit 2014)
    // a_minus = a_plus × (1+z)^(-δ)
    let a_minus_vsl = a_plus_init * (1.0_f64 + z_init).powf(-delta);

    // Alternative interpretation: a_minus = a_plus × (1+z)^(+δ/2)
    let a_minus_alt = a_plus_init * (1.0_f64 + z_init).powf(delta / 2.0);

    // What janus_expansion.rs actually computes
    let a_minus_code = a_minus_from_a_plus(a_plus_init, eta);

    // Also check c_bar_sq from vsl_dynamic
    let c_bar_sq = CoupledFriedmann::c_ratio_sq_at_z(z_init, eta);

    println!("========================================");
    println!("DIAGNOSTIC: a_minus FORMULA COMPARISON");
    println!("========================================");
    println!();
    println!("eta                = {:.6}", eta);
    println!("delta              = (eta-1)/eta = {:.6}", delta);
    println!("z_init             = {:.1}", z_init);
    println!("a_plus_init        = 1/(1+z_init) = {:.6}", a_plus_init);
    println!();
    println!("--- FORMULAS ---");
    println!();
    println!("a_minus_vsl        = a_plus × (1+z)^(-δ)     = {:.6}", a_minus_vsl);
    println!("a_minus_alt        = a_plus × (1+z)^(+δ/2)   = {:.6}", a_minus_alt);
    println!("a_minus_code       = a_minus_from_a_plus()   = {:.6}", a_minus_code);
    println!();
    println!("--- RATIOS ---");
    println!();
    println!("a_minus_vsl / a_plus  = {:.6}  (should be (1+z)^(-δ) = {:.6})",
             a_minus_vsl / a_plus_init, (1.0_f64 + z_init).powf(-delta));
    println!("a_minus_code / a_plus = {:.6}", a_minus_code / a_plus_init);
    println!();
    println!("--- c_bar CHECK ---");
    println!();
    println!("c_bar_sq(z=10)     = (1+z)^δ = {:.6}", c_bar_sq);
    println!("c_bar(z=10)        = {:.6}", c_bar_sq.sqrt());
    println!();

    // Show the actual code formula
    println!("--- CODE IN janus_expansion.rs ---");
    println!();
    let deviation = (eta - 1.0) * (1.0 - a_plus_init);
    println!("deviation = (η-1) × (1 - a_plus) = {:.6}", deviation);
    println!("a_minus   = a_plus × (1 + deviation) = {:.6}", a_plus_init * (1.0 + deviation));
    println!();

    // Verify consistency
    println!("========================================");
    if (a_minus_code - a_minus_vsl).abs() < 1e-6 {
        println!("✅ CODE MATCHES VSL FORMULA");
    } else {
        println!("⚠️  MISMATCH DETECTED!");
        println!("   Difference: {:.6e}", (a_minus_code - a_minus_vsl).abs());
        println!("   Relative:   {:.2}%", 100.0 * (a_minus_code - a_minus_vsl).abs() / a_minus_vsl);
    }
    println!("========================================");
}
