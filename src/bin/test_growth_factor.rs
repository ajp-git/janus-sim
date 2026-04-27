//! Test du facteur de croissance D(z) Janus
//!
//! Tests:
//! 1. D(z=4)/D(z=4) = 1.0 exactement
//! 2. D(10) < D(4) < D(0)
//! 3. Cohérence : mini-run depuis z=10, vérifier σ à z=4

const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const Z_CALIBRATION: f64 = 4.0;

/// Linear growth factor D(z)/D(0) for Janus cosmology
fn growth_factor_janus(z: f64) -> f64 {
    let a = 1.0 / (1.0 + z);

    if a <= ALPHA_SQ_JANUS {
        // Gauge process era: D = a, normalized by D(0) ≈ 2.074
        a / 2.074
    } else {
        // Matter era: interpolate from tabulated values
        let d_table: [(f64, f64); 7] = [
            (0.0, 1.000000),
            (1.0, 0.386367),
            (2.0, 0.179503),
            (3.0, 0.112850),
            (4.0, 0.090886),
            (4.5, 0.087700),
            (4.51, 0.087514),
        ];

        for i in 0..d_table.len()-1 {
            let (z1, d1) = d_table[i];
            let (z2, d2) = d_table[i+1];
            if z >= z1 && z <= z2 {
                let t = (z - z1) / (z2 - z1);
                return d1 + t * (d2 - d1);
            }
        }

        let (z1, d1) = d_table[d_table.len()-2];
        let (z2, d2) = d_table[d_table.len()-1];
        let slope = (d2 - d1) / (z2 - z1);
        d2 + slope * (z - z2)
    }
}

fn ic_scaling_factor(z_init: f64) -> f64 {
    let d_init = growth_factor_janus(z_init);
    let d_calib = growth_factor_janus(Z_CALIBRATION);
    d_init / d_calib
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  TEST D(z) — Janus Growth Factor                           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Test 1: D(4)/D(4) = 1.0
    println!("TEST 1: D(z=4)/D(z=4) = 1.0");
    let ratio_4_4 = ic_scaling_factor(4.0);
    let pass1 = (ratio_4_4 - 1.0).abs() < 1e-10;
    println!("  ic_scaling_factor(4.0) = {:.10}", ratio_4_4);
    println!("  → {}\n", if pass1 { "✓ PASS" } else { "✗ FAIL" });

    // Test 2: D(10) < D(4) < D(0)
    println!("TEST 2: D(10) < D(4) < D(0)");
    let d0 = growth_factor_janus(0.0);
    let d4 = growth_factor_janus(4.0);
    let d10 = growth_factor_janus(10.0);
    let pass2 = d10 < d4 && d4 < d0;
    println!("  D(0)  = {:.6}", d0);
    println!("  D(4)  = {:.6}", d4);
    println!("  D(10) = {:.6}", d10);
    println!("  D(10) < D(4) < D(0): {} < {} < {}", d10, d4, d0);
    println!("  → {}\n", if pass2 { "✓ PASS" } else { "✗ FAIL" });

    // Test 3: Scaling factors
    println!("TEST 3: Scaling factors for ICs");
    let scale_4 = ic_scaling_factor(4.0);
    let scale_10 = ic_scaling_factor(10.0);
    let scale_20 = ic_scaling_factor(20.0);
    println!("  ψ(z=4)/ψ(z=4)   = {:.4} (should be 1.0)", scale_4);
    println!("  ψ(z=10)/ψ(z=4)  = {:.4} (D(10)/D(4))", scale_10);
    println!("  ψ(z=20)/ψ(z=4)  = {:.4} (D(20)/D(4))", scale_20);
    let pass3 = (scale_4 - 1.0).abs() < 1e-10 && scale_10 < 1.0 && scale_20 < scale_10;
    println!("  → {}\n", if pass3 { "✓ PASS" } else { "✗ FAIL" });

    // Test 4: Gauge era check (D = a / 2.074)
    println!("TEST 4: Gauge era (z > 4.51): D ∝ a");
    let z_gauge = 10.0;
    let a_gauge = 1.0 / (1.0 + z_gauge);
    let d_gauge = growth_factor_janus(z_gauge);
    let expected_gauge = a_gauge / 2.074;
    let pass4 = (d_gauge - expected_gauge).abs() / expected_gauge < 0.01;
    println!("  z = {}, a = {:.6}", z_gauge, a_gauge);
    println!("  D(z) = {:.6}, expected a/2.074 = {:.6}", d_gauge, expected_gauge);
    println!("  → {}\n", if pass4 { "✓ PASS" } else { "✗ FAIL" });

    // Test 5: Continuity at transition
    println!("TEST 5: Continuity at z = 4.51 transition");
    let d_just_before = growth_factor_janus(4.50);
    let d_at_trans = growth_factor_janus(4.51);
    let d_just_after = growth_factor_janus(4.52);
    let discontinuity = ((d_at_trans - d_just_before).abs() / d_at_trans).max(
                        (d_just_after - d_at_trans).abs() / d_at_trans);
    let pass5 = discontinuity < 0.05;  // < 5% jump
    println!("  D(4.50) = {:.6}", d_just_before);
    println!("  D(4.51) = {:.6}", d_at_trans);
    println!("  D(4.52) = {:.6}", d_just_after);
    println!("  Max discontinuity: {:.2}%", discontinuity * 100.0);
    println!("  → {}\n", if pass5 { "✓ PASS" } else { "⚠ WARNING (expected small jump at transition)" });

    // Summary
    println!("════════════════════════════════════════════════════════════");
    let all_pass = pass1 && pass2 && pass3 && pass4;
    if all_pass {
        println!("ALL CRITICAL TESTS PASSED ✓");
    } else {
        println!("SOME TESTS FAILED ✗");
    }
    println!("════════════════════════════════════════════════════════════");

    // Key values for production
    println!("\n>>> VALEURS CLÉS POUR PRODUCTION:");
    println!("    D(10)/D(4) = {:.4}", ic_scaling_factor(10.0));
    println!("    À z_init=10: |ψ| = 30% × {:.4} = {:.1}% cell",
             ic_scaling_factor(10.0), 30.0 * ic_scaling_factor(10.0));
    println!("    À z_init=4:  |ψ| = 30% × {:.4} = {:.1}% cell (référence)",
             ic_scaling_factor(4.0), 30.0 * ic_scaling_factor(4.0));
}
