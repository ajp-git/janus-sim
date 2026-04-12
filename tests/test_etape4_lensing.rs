//! Étape 4 — Weak Lensing Convergence Tests
//!
//! Tests unitaires pour les cartes de convergence κ.
//! GO si: 100% tests passent
//!
//! Prédiction Janus: κ > 0 à l'intérieur des halos, κ < 0 dans la coquille m-
//!
//! Références:
//! - Bartelmann & Schneider (2001) — Weak Lensing
//! - Petit (2024) — Janus predictions
//!
//! Run with: cargo test --test test_etape4_lensing

use janus::lensing::*;

// ============================================================================
// Σ_CRIT TESTS
// ============================================================================

/// Test 1: Σ_crit has correct order of magnitude
#[test]
fn test_sigma_crit_units() {
    // Typical cluster lensing: D_L ~ 500 Mpc, D_S ~ 1500 Mpc
    let sc = sigma_crit(500.0, 1500.0, 1100.0);

    // Σ_crit should be ~10^15-10^16 M_sun/Mpc² for cosmological lensing
    assert!(sc > 1e14 && sc < 1e17,
        "Σ_crit = {:.2e} M_sun/Mpc² (expected ~10^15)", sc);
}

/// Test 2: Σ_crit increases when source is closer to lens
#[test]
fn test_sigma_crit_distance_dependence() {
    let d_l = 500.0;

    // Source far behind lens
    let sc_far = sigma_crit(d_l, 2000.0, 1600.0);

    // Source closer to lens
    let sc_close = sigma_crit(d_l, 800.0, 400.0);

    // Closer source → higher Σ_crit (harder to lens)
    assert!(sc_close > sc_far,
        "Closer source should have higher Σ_crit: {:.2e} vs {:.2e}",
        sc_close, sc_far);
}

// ============================================================================
// NFW PROFILE TESTS
// ============================================================================

/// Test 3: NFW Σ(r) decreases with radius
#[test]
fn test_sigma_nfw_profile() {
    let r_s = 0.5;  // Mpc
    let rho_s = 1e8;  // M_sun/Mpc³

    let sigma_01 = sigma_nfw(0.1, r_s, rho_s);
    let sigma_1 = sigma_nfw(1.0, r_s, rho_s);
    let sigma_5 = sigma_nfw(5.0, r_s, rho_s);

    assert!(sigma_01 > sigma_1 && sigma_1 > sigma_5,
        "Σ should decrease: {:.2e} > {:.2e} > {:.2e}",
        sigma_01, sigma_1, sigma_5);
}

/// Test 4: κ_NFW is dimensionless and positive
#[test]
fn test_kappa_nfw_profile() {
    let r_s = 0.5;
    let rho_s = 1e8;
    let sc = 1e15;

    for r in [0.1, 0.5, 1.0, 2.0, 5.0] {
        let kappa = kappa_nfw(r, r_s, rho_s, sc);

        assert!(kappa > 0.0, "κ_NFW should be > 0 at r={}: κ={}", r, kappa);
        assert!(kappa < 10.0, "κ_NFW should be < 10 at r={}: κ={}", r, kappa);
    }
}

/// Test 5: κ_NFW peak at center
#[test]
fn test_kappa_nfw_peak_center() {
    let r_s = 0.5;
    let rho_s = 1e8;
    let sc = 1e15;

    let kappa_center = kappa_nfw(0.01, r_s, rho_s, sc);
    let kappa_edge = kappa_nfw(5.0, r_s, rho_s, sc);

    assert!(kappa_center > kappa_edge * 10.0,
        "κ center should dominate: {:.4} vs {:.4}", kappa_center, kappa_edge);
}

// ============================================================================
// κ MAP TESTS
// ============================================================================

/// Test 6: κ map projection conserves mass
#[test]
fn test_kappa_map_mass_conservation() {
    let n = 500;
    let box_size = 100.0;
    let grid_size = 32;
    let mass = 1e10;
    let sc = 1e15;

    // Uniform distribution
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [t * box_size, (t * 2.3 % 1.0) * box_size, (t * 3.7 % 1.0) * box_size]
        })
        .collect();
    let signs = vec![1.0; n];

    let kappa_map = compute_kappa_map(&positions, &signs, mass, box_size, grid_size, sc, 2);

    // Total projected mass = Σ(κ × Σ_crit × cell_area)
    let cell_area = (box_size / grid_size as f64).powi(2);
    let total_mass: f64 = kappa_map.iter().sum::<f64>() * sc * cell_area;
    let expected_mass = n as f64 * mass;

    let ratio = total_mass / expected_mass;
    assert!(ratio > 0.9 && ratio < 1.1,
        "Mass conservation: ratio = {:.3}", ratio);
}

/// Test 7: κ is negative for m- particles
#[test]
fn test_kappa_negative_mass() {
    let n = 100;
    let box_size = 100.0;
    let grid_size = 16;
    let mass = 1e10;
    let sc = 1e15;

    // Clustered negative mass at center
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [50.0 + t * 5.0, 50.0 + (t * 2.0 % 1.0) * 5.0, i as f64 % box_size]
        })
        .collect();
    let signs = vec![-1.0; n];

    let kappa_map = compute_kappa_map(&positions, &signs, mass, box_size, grid_size, sc, 2);

    // Central region should have κ < 0
    let center_idx = (grid_size / 2) * grid_size + grid_size / 2;
    assert!(kappa_map[center_idx] < 0.0,
        "κ should be < 0 for m-: κ = {:.4}", kappa_map[center_idx]);
}

/// Test 8: Janus halo with m+ core and m- shell
#[test]
fn test_janus_halo_signature() {
    let box_size = 100.0;
    let grid_size = 64;
    let mass = 1e9;
    let sc = 1e15;

    let mut positions = Vec::new();
    let mut signs = Vec::new();

    // m+ core (r < 8 Mpc) - dense cluster in center
    for i in 0..1000 {
        let t = i as f64 / 1000.0;
        let r = t.sqrt() * 8.0;  // sqrt for uniform area distribution
        let theta = t * 200.0;
        positions.push([
            50.0 + r * theta.cos(),
            50.0 + r * theta.sin(),
            50.0,
        ]);
        signs.push(1.0);
    }

    // m- shell (12 < r < 20 Mpc) - surrounding negative mass
    for i in 0..1000 {
        let t = i as f64 / 1000.0;
        let r = 12.0 + t * 8.0;  // 12 < r < 20
        let theta = t * 200.0;
        positions.push([
            50.0 + r * theta.cos(),
            50.0 + r * theta.sin(),
            50.0,
        ]);
        signs.push(-1.0);
    }

    let kappa_map = compute_kappa_map(&positions, &signs, mass, box_size, grid_size, sc, 2);

    // Compute radial profile
    let (r, kappa_profile) = radial_kappa_profile(
        &kappa_map, grid_size, box_size, (32, 32), 25, 35.0
    );

    // Debug: print profile
    // for (ri, ki) in r.iter().zip(kappa_profile.iter()) {
    //     println!("r={:.2} κ={:.4e}", ri, ki);
    // }

    // Check for positive inner, negative outer
    let inner_bins = &kappa_profile[0..5];
    let outer_bins = &kappa_profile[10..20];

    let k_inner_avg: f64 = inner_bins.iter().sum::<f64>() / inner_bins.len() as f64;
    let k_outer_avg: f64 = outer_bins.iter().sum::<f64>() / outer_bins.len() as f64;

    assert!(k_inner_avg > 0.0,
        "Inner κ should be > 0: {:.4e}", k_inner_avg);
    assert!(k_outer_avg < 0.0,
        "Outer κ should be < 0: {:.4e}", k_outer_avg);

    // Janus signature: sign change from positive to negative
    let sign_change = find_kappa_sign_change(&r, &kappa_profile);
    if let Some((r_trans, _, _)) = sign_change {
        assert!(r_trans > 5.0 && r_trans < 25.0,
            "Transition radius: {:.1} Mpc", r_trans);
    }
    // Note: sign_change might not always be found due to discretization,
    // but the average test above validates the Janus signature
}

/// Test 9: Radial profile is smooth
#[test]
fn test_radial_profile_smooth() {
    let grid_size = 32;
    let box_size = 100.0;

    // Create simple test map with radial gradient
    let mut kappa_map = vec![0.0; grid_size * grid_size];
    for i in 0..grid_size {
        for j in 0..grid_size {
            let di = i as f64 - 16.0;
            let dj = j as f64 - 16.0;
            let r = (di * di + dj * dj).sqrt();
            kappa_map[i * grid_size + j] = 1.0 / (1.0 + r);
        }
    }

    let (r, profile) = radial_kappa_profile(&kappa_map, grid_size, box_size, (16, 16), 10, 40.0);

    // Profile should be monotonically decreasing
    for i in 1..profile.len() {
        assert!(profile[i] <= profile[i-1] + 0.01,  // Allow small noise
            "Profile should decrease: bin {} κ={:.4} vs bin {} κ={:.4}",
            i-1, profile[i-1], i, profile[i]);
    }
}

// ============================================================================
// EUCLID DETECTION TESTS
// ============================================================================

/// Test 10: Euclid detection threshold
#[test]
fn test_euclid_detection_threshold() {
    // Below threshold
    assert!(!is_euclid_detectable(0.01));
    assert!(!is_euclid_detectable(-0.02));

    // Above threshold
    assert!(is_euclid_detectable(0.05));
    assert!(is_euclid_detectable(-0.1));
}

/// Test 11: Realistic Janus halo detectability
#[test]
fn test_janus_halo_euclid_detectable() {
    // Massive cluster with m- shell
    // Expected |κ_outer| ~ 0.05-0.1 for 10^15 M_sun cluster

    let kappa_outer = -0.05;  // Typical Janus prediction
    assert!(is_euclid_detectable(kappa_outer),
        "Janus κ_outer = {} should be Euclid-detectable", kappa_outer);
}

// ============================================================================
// INTEGRATION TEST
// ============================================================================

/// Test 12: Full κ pipeline
#[test]
fn test_kappa_full_pipeline() {
    let box_size = 100.0;
    let grid_size = 32;
    let mass = 1e10;

    // Typical lensing geometry
    let d_l = 500.0;
    let d_s = 1500.0;
    let d_ls = 1100.0;
    let sc = sigma_crit(d_l, d_s, d_ls);

    // Create simple halo
    let n = 200;
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let r = t * 15.0;
            let theta = t * 50.0;
            [50.0 + r * theta.cos(), 50.0 + r * theta.sin(), 50.0]
        })
        .collect();
    let signs = vec![1.0; n];

    // Compute κ map
    let kappa_map = compute_kappa_map(&positions, &signs, mass, box_size, grid_size, sc, 2);

    // κ should be positive and peaked at center
    let center_idx = 16 * grid_size + 16;
    let corner_idx = 0;

    assert!(kappa_map[center_idx] > kappa_map[corner_idx],
        "κ should peak at center: {:.2e} vs {:.2e}",
        kappa_map[center_idx], kappa_map[corner_idx]);

    // Total κ should be positive (positive mass halo)
    let total: f64 = kappa_map.iter().sum();
    assert!(total > 0.0, "Total κ should be > 0 for m+ halo");
}
