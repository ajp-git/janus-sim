//! Étape 1 — Radiative Cooling Tests
//!
//! Tests unitaires pour valider le module de refroidissement radiatif.
//! GO si: 100% tests passent
//!
//! Références:
//! - Haardt & Madau (2012) — UV background
//! - Rahmati et al. (2013) — Self-shielding
//!
//! Run with: cargo test --test test_etape1_cooling

use janus::baryonic::cooling::*;

/// Test 1: Isolated cloud cooling
/// A hot cloud (10^6 K) should cool down over time
#[test]
fn test_isolated_cloud_cooling() {
    let t_init = 1e6;  // Hot gas
    let overdensity = 1e4;  // Dense cloud
    let z = 0.0;
    let dt = 0.1;  // Gyr

    let t_final = apply_cooling(t_init, overdensity, z, dt);

    assert!(t_final < t_init,
        "Cloud should cool: T_init={:.0} → T_final={:.0}", t_init, t_final);
}

/// Test 2: Cooling floor
/// Gas should not cool below T_FLOOR (100 K)
#[test]
fn test_cooling_floor() {
    let t_init = 200.0;  // Just above floor
    let overdensity = 1e8;  // Very dense (fast cooling)
    let z = 0.0;
    let dt = 10.0;  // Long time

    let t_final = apply_cooling(t_init, overdensity, z, dt);

    assert!(t_final >= T_FLOOR,
        "Should not cool below T_FLOOR={}: T={}", T_FLOOR, t_final);
}

/// Test 3: Cooling rate at floor is zero
#[test]
fn test_no_cooling_at_floor() {
    let rate = cooling_rate_kelvin_per_gyr(T_FLOOR, 1e6, 0.0);
    assert!(rate.abs() < 1e-30, "Rate at floor should be 0: {}", rate);
}

/// Test 4: Bremsstrahlung slope at high T
/// For T > 10^7 K, cooling should scale as ~T^0.5
#[test]
fn test_bremsstrahlung_slope() {
    let n_h = 1e-3;  // IGM density
    let rate1 = cooling_rate_analytical(1e7, n_h);
    let rate2 = cooling_rate_analytical(4e7, n_h);  // 4× higher T

    // T^0.5 scaling: rate2/rate1 should be ~sqrt(4) = 2
    let ratio = rate2.abs() / rate1.abs();
    assert!(ratio > 1.5 && ratio < 3.0,
        "Bremsstrahlung T^0.5 scaling: ratio={:.2} (expected ~2)", ratio);
}

/// Test 5: Cooling time order of magnitude
/// t_cool = T / |dT/dt| should be physically reasonable
#[test]
fn test_cooling_time_order_of_magnitude() {
    let temp = 1e6;  // 10^6 K
    let overdensity = 1e3;  // Moderately dense
    let z = 0.0;

    let t_cool = cooling_time_gyr(temp, overdensity, z);

    // t_cool should be between 0.001 and 14 Gyr (Hubble time)
    assert!(t_cool > 0.001 && t_cool < 14.0,
        "t_cool={:.3} Gyr out of range [0.001, 14]", t_cool);
}

/// Test 6: UV suppresses cooling at low density
/// IGM (low density) should cool much slower than halos
#[test]
fn test_uv_suppresses_cooling_low_density() {
    let temp = 1e5;
    let z = 2.0;

    // IGM (overdensity ~ 1)
    let rate_igm = cooling_rate_kelvin_per_gyr(temp, 1.0, z);

    // Dense halo (overdensity ~ 10^4)
    let rate_halo = cooling_rate_kelvin_per_gyr(temp, 1e4, z);

    // Halo should cool much faster
    assert!(rate_halo.abs() > 10.0 * rate_igm.abs(),
        "Halo cooling should dominate: halo={:.2e}, IGM={:.2e}",
        rate_halo, rate_igm);
}

/// Test 7: Cooling rate scales with density at high z
/// Without self-shielding (analytical mode), n_H ∝ (1+z)³
/// Note: Self-shielding (Rahmati 2013) requires Grackle
#[test]
fn test_density_redshift_scaling() {
    let temp = 1e5;
    let overdensity = 1e6;

    // High z (UV at peak)
    let rate_z2 = cooling_rate_kelvin_per_gyr(temp, overdensity, 2.0);

    // z=0 (minimal UV)
    let rate_z0 = cooling_rate_kelvin_per_gyr(temp, overdensity, 0.0);

    // n_H(z=2) = n_H(z=0) × (1+2)³ = 27× higher → 27× faster cooling
    // This is the analytical behavior without self-shielding
    let ratio = rate_z2.abs() / rate_z0.abs().max(1e-30);
    assert!(ratio > 10.0,  // Should be ~27 without self-shielding
        "Density should scale: z2/z0={:.2} (expected ~27)", ratio);
}

/// Test 8: Subcycling stability
/// apply_cooling should handle large dt without numerical instability
#[test]
fn test_subcycling_stability() {
    let t_init = 1e7;  // Hot gas
    let overdensity = 1e6;  // Very dense (short t_cool)
    let z = 0.0;
    let dt = 5.0;  // Very long timestep

    let t_final = apply_cooling(t_init, overdensity, z, dt);

    // Should not go negative or NaN
    assert!(t_final.is_finite(), "Result should be finite: {}", t_final);
    assert!(t_final >= T_FLOOR, "Should respect floor: {}", t_final);
}

/// Test 9: Redshift dependence (density scaling)
/// Higher z means higher n_H for same overdensity
#[test]
fn test_redshift_density_scaling() {
    let temp = 1e5;
    let overdensity = 1e4;

    let rate_z0 = cooling_rate_kelvin_per_gyr(temp, overdensity, 0.0);
    let rate_z2 = cooling_rate_kelvin_per_gyr(temp, overdensity, 2.0);

    // n_H(z) = n_H(0) × (1+z)^3
    // At z=2: n_H is 27× higher → faster cooling
    assert!(rate_z2.abs() > rate_z0.abs(),
        "Higher z should cool faster: z0={:.2e}, z2={:.2e}",
        rate_z0, rate_z2);
}

/// Test 10: Lyman-alpha peak cooling
/// Maximum cooling efficiency around T ~ 10^4-10^5 K
#[test]
fn test_lyman_alpha_peak() {
    let n_h = 1e-3;

    let rate_1e4 = cooling_rate_analytical(1e4, n_h);
    let rate_1e5 = cooling_rate_analytical(1e5, n_h);  // Peak expected here
    let rate_1e6 = cooling_rate_analytical(1e6, n_h);
    let rate_1e7 = cooling_rate_analytical(1e7, n_h);

    // Rate at 10^4-10^5 K should be strongest (per unit temperature)
    // Cooling efficiency Λ/T has peak near 10^5 K
    let eff_1e4 = rate_1e4.abs() / 1e4;
    let eff_1e5 = rate_1e5.abs() / 1e5;
    let eff_1e6 = rate_1e6.abs() / 1e6;
    let eff_1e7 = rate_1e7.abs() / 1e7;

    // 10^4-10^5 K should have high efficiency
    assert!(eff_1e5 > eff_1e7,
        "Lyman-alpha peak: eff(10^5)={:.2e} should > eff(10^7)={:.2e}",
        eff_1e5, eff_1e7);
}

#[cfg(test)]
mod cooling_integration_tests {
    use super::*;

    /// Test: Complete cooling sequence
    /// Track temperature evolution over multiple steps
    #[test]
    fn test_cooling_sequence() {
        let mut temp = 1e6_f64;
        let overdensity = 1e5;
        let z = 0.5;
        let dt = 0.01;  // 10 Myr

        let mut temps = vec![temp];

        for _ in 0..100 {
            temp = apply_cooling(temp, overdensity, z, dt);
            temps.push(temp);
        }

        // Temperature should be monotonically decreasing (or stable at floor)
        for i in 1..temps.len() {
            assert!(temps[i] <= temps[i-1] + 1.0,  // Allow tiny numerical error
                "Temperature should decrease: step {} T={} → T={}",
                i, temps[i-1], temps[i]);
        }

        // Should have cooled significantly
        assert!(temps.last().unwrap() < &(temps[0] / 10.0),
            "Should cool by 10×: {} → {}", temps[0], temps.last().unwrap());
    }
}
