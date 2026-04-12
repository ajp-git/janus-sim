//! Étape 6 — Dipole Repeller Tests
//!
//! Tests unitaires pour la détection du Dipole Repeller.
//! GO si: 100% tests passent
//!
//! Prédiction Janus: les vides dominés par m- agissent comme répulseurs
//!
//! Références:
//! - Hoffman et al. (2017) — Dipole Repeller discovery
//! - Courtois et al. (2017) — Cosmic flows
//! - Petit (2024) — Janus predictions
//!
//! Run with: cargo test --test test_etape6_repeller

use janus::peculiar_velocity::*;

// ============================================================================
// HUBBLE FLOW TESTS
// ============================================================================

/// Test 1: Hubble velocity at known distances
#[test]
fn test_hubble_velocity_scale() {
    // H0 = 76 km/s/Mpc (Janus value)
    // At 100 Mpc: v_H = 7600 km/s
    let v100 = hubble_velocity(100.0, 76.0);
    assert!((v100 - 7600.0).abs() < 1.0,
        "v_H(100 Mpc) = {} km/s (expected 7600)", v100);

    // At 1000 Mpc (z ≈ 0.25): v_H = 76000 km/s
    let v1000 = hubble_velocity(1000.0, 76.0);
    assert!((v1000 - 76000.0).abs() < 10.0,
        "v_H(1000 Mpc) = {} km/s (expected 76000)", v1000);
}

/// Test 2: Hubble velocity scaling with H0
#[test]
fn test_hubble_velocity_h0_dependence() {
    let d = 100.0;
    let v_76 = hubble_velocity(d, 76.0);
    let v_70 = hubble_velocity(d, 70.0);

    // v ∝ H0
    let ratio = v_76 / v_70;
    assert!((ratio - 76.0/70.0).abs() < 0.01,
        "v ratio = {} (expected 76/70 = 1.086)", ratio);
}

// ============================================================================
// PECULIAR VELOCITY TESTS
// ============================================================================

/// Test 3: Peculiar velocity for object at rest in CMB frame
#[test]
fn test_peculiar_velocity_zero_uniform() {
    // Object at 100 Mpc moving with Hubble flow
    // v_total = v_Hubble → v_pec = 0
    let d = 100.0;
    let v_total = hubble_velocity(d, 76.0);  // Pure Hubble
    let v_pec = peculiar_velocity(v_total, d, 76.0);

    assert!(v_pec.abs() < 0.1,
        "v_pec should be 0 for Hubble flow: got {} km/s", v_pec);
}

/// Test 4: Peculiar velocity - outflow case
#[test]
fn test_peculiar_velocity_outflow() {
    // Object at 100 Mpc moving faster than Hubble
    // v_total = 8000, v_H = 7600 → v_pec = +400 (outflow)
    let v_pec = peculiar_velocity(8000.0, 100.0, 76.0);

    assert!((v_pec - 400.0).abs() < 1.0,
        "v_pec = {} km/s (expected +400 outflow)", v_pec);
}

/// Test 5: Peculiar velocity - infall case
#[test]
fn test_peculiar_velocity_infall() {
    // Object at 100 Mpc moving slower than Hubble (falling toward us)
    // v_total = 7000, v_H = 7600 → v_pec = -600 (infall)
    let v_pec = peculiar_velocity(7000.0, 100.0, 76.0);

    assert!((v_pec + 600.0).abs() < 1.0,
        "v_pec = {} km/s (expected -600 infall)", v_pec);
}

/// Test 6: Peculiar velocity field computation
#[test]
fn test_peculiar_velocity_field() {
    // Create particles with pure Hubble flow
    let h0 = 76.0;
    let positions: Vec<[f64; 3]> = vec![
        [100.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],
        [0.0, 0.0, 75.0],
    ];

    // Pure Hubble velocities (radial)
    let velocities: Vec<[f64; 3]> = positions.iter()
        .map(|pos| {
            let r = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
            let v_h = h0 * r;
            [
                v_h * pos[0] / r,
                v_h * pos[1] / r,
                v_h * pos[2] / r,
            ]
        })
        .collect();

    let v_pec = compute_peculiar_velocities(&positions, &velocities, h0);

    // All peculiar velocities should be ~0
    for (i, v) in v_pec.iter().enumerate() {
        let v_mag = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
        assert!(v_mag < 1.0,
            "v_pec[{}] = {:.1} km/s (expected ~0 for Hubble flow)", i, v_mag);
    }
}

// ============================================================================
// REPELLER DETECTION TESTS
// ============================================================================

/// Test 7: Detect repeller from outflowing particles
#[test]
fn test_repeller_detection() {
    // Create outflowing m- dominated region (repeller signature)
    let n = 300;
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut signs = Vec::new();

    for i in 0..n {
        let t = i as f64 / n as f64;
        let r = 30.0 + t * 40.0;  // 30-70 Mpc from center
        let theta = t * 15.0;
        let phi = t * 7.0;

        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        positions.push([x, y, z]);

        // Outward velocity (repeller signature)
        let v_out = 180.0 + t * 40.0;  // 180-220 km/s outflow
        velocities.push([
            v_out * x / r,
            v_out * y / r,
            v_out * z / r,
        ]);

        // 60% m- (repeller is m- dominated void)
        signs.push(if t < 0.6 { -1.0 } else { 1.0 });
    }

    let repeller = detect_repeller(
        &positions, &velocities, &signs,
        [0.0, 0.0, 0.0], 80.0, 100.0
    );

    assert!(repeller.is_some(), "Should detect repeller");
    let info = repeller.unwrap();

    assert!(info.v_outflow > 150.0,
        "v_outflow = {:.0} km/s (expected >150)", info.v_outflow);
    assert!(info.f_minus > 0.5,
        "f_minus = {:.2} (expected >0.5)", info.f_minus);
    assert!(info.is_significant, "Should be significant repeller");
}

/// Test 8: Detect attractor from inflowing particles
#[test]
fn test_attractor_vs_repeller() {
    // Create inflowing m+ dominated region (attractor signature)
    let n = 300;
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut signs = Vec::new();

    for i in 0..n {
        let t = i as f64 / n as f64;
        let r = 30.0 + t * 40.0;
        let theta = t * 15.0;
        let phi = t * 7.0;

        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        positions.push([x, y, z]);

        // Inward velocity (attractor signature)
        let v_in = -150.0 - t * 50.0;  // -150 to -200 km/s infall
        velocities.push([
            v_in * x / r,
            v_in * y / r,
            v_in * z / r,
        ]);

        // 80% m+ (attractor is m+ dominated halo)
        signs.push(if t > 0.2 { 1.0 } else { -1.0 });
    }

    let attractor = detect_attractor(
        &positions, &velocities, &signs,
        [0.0, 0.0, 0.0], 80.0, 100.0
    );

    assert!(attractor.is_some(), "Should detect attractor");
    let info = attractor.unwrap();

    assert!(info.v_inflow > 100.0,
        "v_inflow = {:.0} km/s (expected >100)", info.v_inflow);
    assert!(info.f_plus > 0.7,
        "f_plus = {:.2} (expected >0.7)", info.f_plus);
    assert!(info.is_significant, "Should be significant attractor");
}

/// Test 9: No repeller in uniform random field
#[test]
fn test_no_repeller_random() {
    // Random velocities (no coherent flow)
    let n = 200;
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [
                (t * 20.0).sin() * 40.0,
                (t * 30.0).cos() * 40.0,
                (t * 15.0).sin() * 40.0,
            ]
        })
        .collect();

    // Random-ish velocities with zero mean
    let velocities: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 * 0.13;
            [
                t.sin() * 50.0 - 25.0,
                t.cos() * 50.0 - 25.0,
                (t * 1.5).sin() * 50.0 - 25.0,
            ]
        })
        .collect();

    let signs: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

    let repeller = detect_repeller(
        &positions, &velocities, &signs,
        [0.0, 0.0, 0.0], 50.0, 100.0
    );

    // Should find region but not significant
    if let Some(info) = repeller {
        assert!(!info.is_significant || info.v_outflow < 50.0,
            "Random field shouldn't have significant repeller");
    }
}

// ============================================================================
// HOFFMAN COMPATIBILITY TESTS
// ============================================================================

/// Test 10: Hoffman et al. (2017) velocity scale
#[test]
fn test_hoffman_velocity_scale() {
    // Dipole Repeller: ~100 Mpc, ~200 km/s outflow
    let repeller = RepellerInfo {
        center: [0.0, 0.0, 0.0],
        radius: 100.0,
        v_outflow: 200.0,
        f_minus: 0.5,
        is_significant: true,
    };

    assert!(is_hoffman_compatible(&repeller),
        "Hoffman-scale repeller should be compatible");
}

/// Test 11: Sub-Hoffman repeller not compatible
#[test]
fn test_sub_hoffman_not_compatible() {
    // Small, slow repeller
    let weak_repeller = RepellerInfo {
        center: [0.0, 0.0, 0.0],
        radius: 20.0,
        v_outflow: 50.0,  // Too slow
        f_minus: 0.5,
        is_significant: false,
    };

    assert!(!is_hoffman_compatible(&weak_repeller),
        "Weak repeller shouldn't match Hoffman");
}

// ============================================================================
// BULK FLOW TESTS
// ============================================================================

/// Test 12: Bulk flow detection
#[test]
fn test_bulk_flow_detection() {
    // Create region with coherent bulk motion
    let n = 150;
    let v_bulk_true = [120.0, -80.0, 50.0];  // Coherent bulk velocity

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [(t * 10.0).sin() * 30.0, (t * 15.0).cos() * 30.0, t * 50.0 - 25.0]
        })
        .collect();

    // Bulk velocity + small random component
    let velocities: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let noise = (i as f64 * 0.1).sin() * 5.0;
            [
                v_bulk_true[0] + noise,
                v_bulk_true[1] - noise,
                v_bulk_true[2] + noise * 0.5,
            ]
        })
        .collect();

    let (v_bulk, sigma) = bulk_flow(&positions, &velocities, 100.0);

    assert!((v_bulk[0] - v_bulk_true[0]).abs() < 15.0,
        "v_bulk_x = {:.0} (expected ~120)", v_bulk[0]);
    assert!((v_bulk[1] - v_bulk_true[1]).abs() < 15.0,
        "v_bulk_y = {:.0} (expected ~-80)", v_bulk[1]);
    assert!((v_bulk[2] - v_bulk_true[2]).abs() < 15.0,
        "v_bulk_z = {:.0} (expected ~50)", v_bulk[2]);
    assert!(sigma < 30.0,
        "sigma_v = {:.0} km/s (should be small)", sigma);
}

/// Test 13: Bulk flow magnitude
#[test]
fn test_bulk_flow_magnitude() {
    let n = 100;
    let v_bulk_true = [200.0, 0.0, 0.0];  // 200 km/s in x direction

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [t * 50.0, t * 30.0, t * 20.0]
        })
        .collect();

    let velocities: Vec<[f64; 3]> = vec![v_bulk_true; n];

    let (v_bulk, sigma) = bulk_flow(&positions, &velocities, 100.0);

    let v_mag = (v_bulk[0].powi(2) + v_bulk[1].powi(2) + v_bulk[2].powi(2)).sqrt();
    assert!((v_mag - 200.0).abs() < 1.0,
        "|v_bulk| = {:.0} km/s (expected 200)", v_mag);
    assert!(sigma < 1.0,
        "sigma should be ~0 for coherent flow: {:.1}", sigma);
}

// ============================================================================
// JANUS SIGNATURE TESTS
// ============================================================================

/// Test 14: Janus repeller has high m- fraction
#[test]
fn test_janus_repeller_m_minus_fraction() {
    // In Janus, repellers are m- dominated voids
    let n = 200;
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut signs = Vec::new();

    for i in 0..n {
        let t = i as f64 / n as f64;
        let r = 40.0 + t * 20.0;
        let theta = t * 20.0;
        let phi = t * 10.0;

        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        positions.push([x, y, z]);

        // Strong outflow
        let v_out = 250.0;
        velocities.push([v_out * x / r, v_out * y / r, v_out * z / r]);

        // 70% m- (Janus prediction for repeller)
        signs.push(if t < 0.7 { -1.0 } else { 1.0 });
    }

    let repeller = detect_repeller(
        &positions, &velocities, &signs,
        [0.0, 0.0, 0.0], 70.0, 150.0
    );

    assert!(repeller.is_some());
    let info = repeller.unwrap();

    assert!(info.f_minus > 0.6,
        "Janus repeller should have high m- fraction: {:.2}", info.f_minus);
    assert!(info.is_significant,
        "Strong outflow + high f_minus = significant");
}

/// Test 15: Janus attractor has high m+ fraction
#[test]
fn test_janus_attractor_m_plus_fraction() {
    // In Janus, attractors are m+ dominated halos
    let n = 200;
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut signs = Vec::new();

    for i in 0..n {
        let t = i as f64 / n as f64;
        let r = 40.0 + t * 20.0;
        let theta = t * 20.0;
        let phi = t * 10.0;

        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        positions.push([x, y, z]);

        // Strong inflow
        let v_in = -200.0;
        velocities.push([v_in * x / r, v_in * y / r, v_in * z / r]);

        // 85% m+ (Janus prediction for attractor/halo)
        signs.push(if t > 0.15 { 1.0 } else { -1.0 });
    }

    let attractor = detect_attractor(
        &positions, &velocities, &signs,
        [0.0, 0.0, 0.0], 70.0, 100.0
    );

    assert!(attractor.is_some());
    let info = attractor.unwrap();

    assert!(info.f_plus > 0.8,
        "Janus attractor should have high m+ fraction: {:.2}", info.f_plus);
    assert!(info.is_significant,
        "Strong inflow + high f_plus = significant");
}
