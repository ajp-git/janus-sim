//! Test Grackle cooling library integration
//!
//! Run with: cargo run --features grackle --bin test_grackle

use janus::grackle_wrapper::{GrackleCooling, cleanup};
use std::path::Path;

fn main() {
    // Find Grackle data file
    let data_paths = [
        "/app/external/grackle-dist/grackle/input/CloudyData_UVB=HM2012.h5",
        "/usr/local/share/grackle/input/CloudyData_UVB=HM2012.h5",
        "external/grackle-dist/grackle/input/CloudyData_UVB=HM2012.h5",
    ];

    let data_file = data_paths.iter()
        .find(|p| Path::new(p).exists())
        .expect("Grackle data file not found! Install HM2012 tables.");

    println!("Using Grackle data: {}", data_file);
    println!();

    // Initialize Grackle
    let grackle = GrackleCooling::new(data_file).expect("Failed to initialize Grackle");

    // Test 1: Λ(10^4.5 K) ≈ 1.6×10^-22 erg·cm³/s
    println!("=== Test 1: Λ(10^4.5 K) ===");
    let t = 10f64.powf(4.5);
    let lambda = grackle.lambda_norm(t, 0.0);
    println!("Temperature: {:.1} K (10^4.5 K)", t);
    println!("Λ_norm = {:.3e} erg·cm³/s", lambda);
    println!("Expected: ≈ 1.6e-22 erg·cm³/s");

    let pass = lambda > 1e-23 && lambda < 1e-21;
    println!("Result: {}", if pass { "PASS ✓" } else { "FAIL ✗" });
    println!();

    // Test 2: Cooling curve
    println!("=== Cooling Curve (z=0) ===");
    println!("log(T/K)    Λ [erg·cm³/s]");
    for log_t in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0] {
        let t = 10f64.powf(log_t);
        let l = grackle.lambda_norm(t, 0.0);
        println!("  {:.1}       {:.3e}", log_t, l);
    }
    println!();

    // Test 3: Cooling time
    println!("=== Cooling Time Test ===");
    let t = 1e6;  // 10^6 K
    let n_h = 1e-3;  // cm^-3 (typical IGM)
    let t_cool_gyr = grackle.cooling_time_gyr(t, n_h, 0.0);
    println!("T = 10^6 K, n_H = 10^-3 cm^-3");
    println!("Cooling time: {:.2} Gyr", t_cool_gyr);
    println!();

    // Test 4: Redshift dependence
    println!("=== Λ(10^5 K) vs Redshift ===");
    let t = 1e5;
    for z in [0.0, 1.0, 2.0, 3.0, 4.0] {
        let l = grackle.lambda_norm(t, z);
        println!("z={:.0}: Λ = {:.3e} erg·cm³/s", z, l);
    }
    println!();

    // Test 5: Specific cooling rate for simulation use
    println!("=== Specific Cooling Rate ===");
    let t = 1e4;
    let n_h = 1.0;  // cm^-3 (dense gas)
    let du_dt = grackle.cooling_rate(t, n_h, 0.0, 0.0);
    println!("T = 10^4 K, n_H = 1 cm^-3");
    println!("dU/dt = {:.3e} km²/s²/Gyr", du_dt);

    cleanup();
    println!();
    println!("Grackle integration test complete.");
}
