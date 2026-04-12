//! Étape 2 — Star Formation + Feedback Tests
//!
//! Tests unitaires pour la formation stellaire et le feedback SN.
//! GO si: 100% tests passent
//!
//! Références:
//! - Schmidt (1959) — SF law
//! - Springel & Hernquist (2003) — Multiphase ISM
//!
//! Run with: cargo test --test test_etape2_star_formation

use janus::baryonic::star_formation::*;
use janus::baryonic::feedback::*;
use janus::baryonic::pressure::sound_speed;

// ============================================================================
// STAR FORMATION TESTS
// ============================================================================

/// Test 1: No SF in hot gas
#[test]
fn test_no_sf_hot_gas() {
    // T = 10^6 K >> T_SF = 1000 K
    let result = should_form_star(
        1e12,       // high density
        1e8,        // mean density
        1e6,        // HOT
        -1.0,       // converging
        sound_speed(1e6),
        1e15,       // mass above Jeans
    );
    assert!(!result, "Should NOT form stars in hot gas");
}

/// Test 2: No SF in diverging flow
#[test]
fn test_no_sf_diverging_flow() {
    // div_v > 0 means expanding
    let result = should_form_star(
        1e12,
        1e8,
        100.0,      // cold
        1.0,        // DIVERGING
        sound_speed(100.0),
        1e15,
    );
    assert!(!result, "Should NOT form stars in expanding gas");
}

/// Test 3: No SF in underdense regions
#[test]
fn test_no_sf_underdense() {
    // ρ < 100 × ρ̄
    let result = should_form_star(
        1e8,        // same as mean
        1e8,
        100.0,
        -1.0,
        sound_speed(100.0),
        1e15,
    );
    assert!(!result, "Should NOT form stars in underdense region");
}

/// Test 4: SF when all criteria met
#[test]
fn test_sf_all_criteria_met() {
    let result = should_form_star(
        1e12,       // 10^4 × mean
        1e8,
        100.0,      // cold
        -1.0,       // converging
        sound_speed(100.0),
        1e15,       // above Jeans
    );
    assert!(result, "Should form stars when all criteria met");
}

/// Test 5: Jeans mass scaling with temperature (cs³)
#[test]
fn test_jeans_mass_temperature_scaling() {
    let rho = 1e10;

    // cs ∝ T^0.5, so M_J ∝ cs³ ∝ T^1.5
    let mj_100 = jeans_mass(rho, sound_speed(100.0));
    let mj_400 = jeans_mass(rho, sound_speed(400.0));  // 4× T

    // T × 4 → cs × 2 → M_J × 8
    let ratio = mj_400 / mj_100;
    assert!(ratio > 6.0 && ratio < 10.0,
        "M_J scaling: ratio = {:.2} (expected ~8)", ratio);
}

/// Test 6: Jeans mass scaling with density (ρ^-0.5)
#[test]
fn test_jeans_mass_density_scaling() {
    let cs = sound_speed(100.0);

    let mj1 = jeans_mass(1e10, cs);
    let mj2 = jeans_mass(4e10, cs);  // 4× density

    // ρ × 4 → M_J × 0.5
    let ratio = mj1 / mj2;
    assert!((ratio - 2.0).abs() < 0.2,
        "M_J density scaling: ratio = {:.2} (expected ~2)", ratio);
}

// ============================================================================
// SN FEEDBACK TESTS
// ============================================================================

/// Test 7: SN energy in correct units
#[test]
fn test_sn_energy_units() {
    // 10^51 erg / (100 M_sun × 2e33 g) ≈ 5×10^5 km²/s²
    assert!((E_SN_KM2S2 - 5.0e5).abs() < 2e5,
        "E_SN = {:.2e} km²/s² (expected ~5e5)", E_SN_KM2S2);
}

/// Test 8: Thermal heating magnitude
#[test]
fn test_sn_thermal_heating_magnitude() {
    let m_star = 1e6;   // 10^6 M_sun of stars
    let m_gas = 1e8;    // 10^8 M_sun of gas
    let eff = 0.1;

    let dt = sn_thermal_heating(m_star, m_gas, eff);

    // Should be ~10^5-10^7 K increase
    assert!(dt > 1e4 && dt < 1e8,
        "ΔT = {:.2e} K out of range [10^4, 10^8]", dt);
}

/// Test 9: Velocity kick magnitude
#[test]
fn test_sn_velocity_kick_magnitude() {
    let m_star = 1e6;
    let m_gas = 1e7;
    let eff = 0.1;

    let v = sn_velocity_kick(m_star, m_gas, eff);

    // Should be ~50-500 km/s
    assert!(v > 10.0 && v < 1000.0,
        "v_kick = {:.1} km/s out of range [10, 1000]", v);
}

/// Test 10: Pure thermal mode - no velocity change
#[test]
fn test_feedback_thermal_mode() {
    let mut vel = [0.0, 0.0, 0.0];
    let t0 = 1e4;

    let t1 = apply_sn_feedback(
        FeedbackMode::Thermal,
        1e6, 1e7, t0,
        &mut vel,
        [1.0, 0.0, 0.0],
        0.1
    );

    // Temperature should increase
    assert!(t1 > t0, "Thermal mode should heat: {} → {}", t0, t1);

    // Velocity should be unchanged
    let v_mag = (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]).sqrt();
    assert!(v_mag < 1e-10, "Thermal mode: velocity should be 0, got {}", v_mag);
}

/// Test 11: Pure kinetic mode - no temperature change
#[test]
fn test_feedback_kinetic_mode() {
    let mut vel = [0.0, 0.0, 0.0];
    let t0 = 1e4;

    let t1 = apply_sn_feedback(
        FeedbackMode::Kinetic,
        1e6, 1e7, t0,
        &mut vel,
        [1.0, 0.0, 0.0],
        0.1
    );

    // Temperature should be unchanged
    assert!((t1 - t0).abs() < 1.0, "Kinetic mode: T should be constant");

    // Velocity should increase
    assert!(vel[0] > 10.0, "Kinetic mode: should kick, got v = {:?}", vel);
}

/// Test 12: Hybrid mode - both heating and kick
#[test]
fn test_feedback_hybrid_mode() {
    let mut vel = [0.0, 0.0, 0.0];
    let t0 = 1e4;

    let t1 = apply_sn_feedback(
        FeedbackMode::Hybrid { thermal_fraction: 0.5 },
        1e6, 1e7, t0,
        &mut vel,
        [1.0, 0.0, 0.0],
        0.1
    );

    // Temperature should increase (but less than pure thermal)
    assert!(t1 > t0, "Hybrid should heat");

    // Velocity should increase (but less than pure kinetic)
    assert!(vel[0] > 1.0, "Hybrid should kick");
}

/// Test 13: Schmidt-Kennicutt SFR scaling
#[test]
fn test_sfr_schmidt_kennicutt() {
    // SFR ∝ ρ / t_ff ∝ ρ^1.5
    let sfr1 = schmidt_kennicutt_sfr(1e12, 0.01);
    let sfr2 = schmidt_kennicutt_sfr(8e12, 0.01);  // 8× density

    // ratio should be 8^1.5 ≈ 22.6
    let ratio = sfr2 / sfr1;
    assert!(ratio > 15.0 && ratio < 30.0,
        "SFR scaling: ratio = {:.1} (expected ~22)", ratio);
}

/// Test 14: SF probability bounds
#[test]
fn test_sf_probability_bounds() {
    // Very high density, long timestep
    let prob = sf_probability(1e15, 0.1, 1.0);
    assert!(prob >= 0.0 && prob <= 1.0,
        "SF probability = {} out of [0, 1]", prob);

    // Low density, short timestep
    let prob_low = sf_probability(1e8, 0.01, 0.001);
    assert!(prob_low >= 0.0 && prob_low <= 1.0,
        "SF probability low = {} out of [0, 1]", prob_low);
}

/// Test 15: SF probability increases with density
#[test]
fn test_sf_probability_density_dependence() {
    let dt = 0.01;
    let eff = 0.01;

    let prob_low = sf_probability(1e10, eff, dt);
    let prob_high = sf_probability(1e12, eff, dt);

    assert!(prob_high > prob_low,
        "Higher density should have higher SF prob: {} vs {}",
        prob_high, prob_low);
}

// ============================================================================
// PARTICLE TYPE TESTS
// ============================================================================

/// Test 16: Particle type classification
#[test]
fn test_particle_types() {
    use janus::baryonic::star_formation::particle_type::*;

    // Gas m+
    assert_eq!(sign_to_type(1), GAS_PLUS);
    assert!(is_gas(GAS_PLUS));
    assert!(!is_sink(GAS_PLUS));
    assert!(!is_negative_mass(GAS_PLUS));

    // m-
    assert_eq!(sign_to_type(-1), MASS_MINUS);
    assert!(is_negative_mass(MASS_MINUS));
    assert!(!is_sink(MASS_MINUS));
    assert!(!is_gas(MASS_MINUS));

    // Sink (star)
    assert!(is_sink(SINK_STAR));
    assert!(!is_gas(SINK_STAR));
    assert!(!is_negative_mass(SINK_STAR));
}

/// Test 17: Sinks don't feel pressure
#[test]
fn test_sink_no_pressure() {
    use janus::baryonic::star_formation::particle_type::SINK_STAR;

    // Sink particles should not be gas
    assert!(!is_gas(SINK_STAR),
        "Sink particles must not feel thermal pressure");
}

/// Test 18: Sinks are positive mass
#[test]
fn test_sink_positive_mass() {
    use janus::baryonic::star_formation::particle_type::SINK_STAR;

    // Sink particles attract m+ and repel m-
    assert!(!is_negative_mass(SINK_STAR),
        "Sink particles must be positive mass");
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test: SF + feedback cycle
    #[test]
    fn test_sf_feedback_cycle() {
        // Cold, dense, converging gas
        let rho = 1e12;
        let t = 100.0;
        let mean_rho = 1e8;
        let cs = sound_speed(t);
        let m_local = 1e15;

        // Should form stars
        assert!(should_form_star(rho, mean_rho, t, -1.0, cs, m_local));

        // After forming stars, feedback should heat the gas
        let m_star = 1e6;
        let m_gas = 1e8;
        let mut vel = [0.0, 0.0, 0.0];

        let t_new = apply_sn_feedback(
            FeedbackMode::Thermal,
            m_star, m_gas, t,
            &mut vel, [0.0, 0.0, 1.0], 0.1
        );

        // Gas should be heated above T_SF
        assert!(t_new > T_SF,
            "Feedback should heat above T_SF={}: T = {}", T_SF, t_new);

        // No more star formation in heated gas
        assert!(!should_form_star(rho, mean_rho, t_new, -1.0,
            sound_speed(t_new), m_local),
            "Heated gas should not form stars");
    }
}
