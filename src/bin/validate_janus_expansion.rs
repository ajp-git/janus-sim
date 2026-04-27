//! Validate Janus Parametric Cosmology
//!
//! This binary tests the janus_expansion module and exports
//! the cosmology table for plotting.

use janus::janus_expansion::JanusExpansion;

fn main() {
    println!("=== Janus Parametric Cosmology Validation ===\n");

    // Create expansion table from z=10 to z=0
    let expansion = JanusExpansion::new(10.0, 5000);

    // Export to CSV
    let csv_path = "/app/output/janus_expansion_table.csv";
    expansion.export_csv(csv_path).expect("Failed to export CSV");
    println!("\nExported table to: {}\n", csv_path);

    // Print key values
    println!("=== Key Cosmological Values ===");

    let z_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0];
    println!("{:>6} | {:>8} | {:>10} | {:>10} | {:>8}",
             "z", "a⁺", "H⁺ [Gyr⁻¹]", "H [km/s/Mpc]", "t [Gyr]");
    println!("{:-<6}-+-{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<8}", "", "", "", "", "");

    for z in z_values {
        let state = expansion.at_redshift(z);
        let h_km_s_mpc = state.h_plus / 1.0227e-3;  // Convert Gyr⁻¹ to km/s/Mpc
        println!("{:>6.2} | {:>8.4} | {:>10.4} | {:>10.1} | {:>8.3}",
                 z, state.a_plus, state.h_plus, h_km_s_mpc, state.t_gyr);
    }

    // Verify H₀
    let state_z0 = expansion.at_redshift(0.0);
    let h0_check = state_z0.h_plus / 1.0227e-3;
    println!("\n=== Validation ===");
    println!("H₀ at z=0: {:.2} km/s/Mpc (target: 70.0)", h0_check);

    if (h0_check - 70.0).abs() < 2.0 {
        println!("✓ H₀ calibration PASSED");
    } else {
        println!("✗ H₀ calibration FAILED");
    }

    // Check scale factor normalization
    println!("a⁺(z=0): {:.6} (target: 1.0)", state_z0.a_plus);
    if (state_z0.a_plus - 1.0).abs() < 0.01 {
        println!("✓ Scale factor normalization PASSED");
    } else {
        println!("✗ Scale factor normalization FAILED");
    }

    // Check monotonicity
    let mut prev_z = f64::MAX;
    let mut prev_a = 0.0;
    let mut monotonic = true;
    for state in &expansion.table {
        if state.z >= prev_z || state.a_plus <= prev_a {
            monotonic = false;
            break;
        }
        prev_z = state.z;
        prev_a = state.a_plus;
    }
    if monotonic {
        println!("✓ Monotonicity check PASSED (z decreases, a increases)");
    } else {
        println!("✗ Monotonicity check FAILED");
    }
}
