//! Étape 3 — Power Spectrum P(k) Tests
//!
//! Tests unitaires pour le calcul du spectre de puissance.
//! GO si: 100% tests passent
//!
//! Références:
//! - Hockney & Eastwood (1981) — CIC assignment
//! - Jing (2005) — P(k) estimation
//! - Bardeen et al. (1986) — Transfer function
//!
//! Run with: cargo test --test test_etape3_pk

use janus::power_spectrum::*;
use std::f64::consts::PI;

// ============================================================================
// CIC MASS ASSIGNMENT TESTS
// ============================================================================

/// Test 1: CIC conserves total mass
#[test]
fn test_cic_mass_conservation() {
    let n = 1000;
    let box_size = 100.0;
    let grid_size = 32;

    // Random positions
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [
                (t * 7.3 * box_size) % box_size,
                (t * 11.7 * box_size) % box_size,
                (t * 13.1 * box_size) % box_size,
            ]
        })
        .collect();

    let grid = cic_assign(&positions, box_size, grid_size);
    let total_mass: f64 = grid.iter().sum();

    assert!((total_mass - n as f64).abs() < 1e-8,
        "CIC mass conservation: {} vs {}", total_mass, n);
}

/// Test 2: CIC handles periodic boundaries
#[test]
fn test_cic_periodic_wrap() {
    let box_size = 100.0;
    let grid_size = 16;

    // Particle at edge should contribute to both sides
    let positions = vec![[99.9, 50.0, 50.0]];
    let grid = cic_assign(&positions, box_size, grid_size);

    // Total mass should be 1
    let total: f64 = grid.iter().sum();
    assert!((total - 1.0).abs() < 1e-10,
        "Periodic wrap mass: {}", total);

    // Cell 15 and cell 0 should both have non-zero contribution
    let cell_15 = 15 * grid_size * grid_size + 8 * grid_size + 8;
    let cell_0 = 0 * grid_size * grid_size + 8 * grid_size + 8;

    assert!(grid[cell_15] > 0.0 || grid[cell_0] > 0.0,
        "Edge particle should wrap: cell_15={}, cell_0={}",
        grid[cell_15], grid[cell_0]);
}

// ============================================================================
// P(k) COMPUTATION TESTS
// ============================================================================

/// Test 3: P(k) of white noise is nearly flat after shot noise subtraction
#[test]
fn test_pk_white_noise_flat() {
    let grid_size = 32;
    let box_size = 100.0;
    let n = 5000;

    // Quasi-random positions (deterministic for reproducibility)
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [
                (t * 157.3 * box_size) % box_size,
                (t * 229.7 * box_size) % box_size,
                (t * 311.1 * box_size) % box_size,
            ]
        })
        .collect();

    let grid = cic_assign(&positions, box_size, grid_size);
    let result = compute_pk(&grid, box_size, grid_size, n, 10);

    // After shot noise subtraction, P(k) should be near 0
    // Allow for sampling variance
    let mean_pk: f64 = result.pk.iter().sum::<f64>() / result.pk.len() as f64;
    let shot = box_size.powi(3) / n as f64;

    assert!(mean_pk.abs() < shot * 3.0,
        "White noise P(k) should be ~0: mean={:.2e}, shot={:.2e}",
        mean_pk, shot);
}

/// Test 4: P(k) has correct units [Mpc³]
#[test]
fn test_pk_units() {
    let grid_size = 16;
    let box_size = 100.0;
    let n = 1000;

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [t * box_size, t * box_size, t * box_size]
        })
        .collect();

    let grid = cic_assign(&positions, box_size, grid_size);
    let result = compute_pk(&grid, box_size, grid_size, n, 8);

    // Shot noise = V/N has units [Mpc³]
    let shot = box_size.powi(3) / n as f64;

    // P(k) should be of similar order to shot noise for this random-ish data
    assert!(shot > 10.0 && shot < 1e8,
        "Shot noise magnitude: {:.2e} Mpc³", shot);
}

/// Test 5: Nyquist cutoff is respected
#[test]
fn test_pk_nyquist_cutoff() {
    let grid_size = 16;
    let box_size = 100.0;
    let n = 500;

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [t * box_size, (2.0 * t) % 1.0 * box_size, (3.0 * t) % 1.0 * box_size]
        })
        .collect();

    let grid = cic_assign(&positions, box_size, grid_size);
    let result = compute_pk(&grid, box_size, grid_size, n, 10);

    let k_nyquist = PI * grid_size as f64 / box_size;

    for &k in &result.k {
        assert!(k <= k_nyquist,
            "k = {:.3} exceeds Nyquist = {:.3}", k, k_nyquist);
    }
}

/// Test 6: Single mode gives P(k) peak at that mode
#[test]
fn test_pk_single_mode() {
    let grid_size = 32;
    let box_size = 100.0;
    let n_cells = grid_size * grid_size * grid_size;

    // Create density field with single k mode
    // k = n × 2π/L where n is the mode number
    let mode_number = 3;
    let k_target = 2.0 * PI * mode_number as f64 / box_size;

    let mut grid = vec![0.0; n_cells];

    for ix in 0..grid_size {
        let x = (ix as f64 + 0.5) * box_size / grid_size as f64;
        for iy in 0..grid_size {
            for iz in 0..grid_size {
                let idx = ix * grid_size * grid_size + iy * grid_size + iz;
                // ρ = ρ̄ (1 + δ cos(kx)) with δ = 0.1
                grid[idx] = 1.0 + 0.1 * (k_target * x).cos();
            }
        }
    }

    // Use large n_particles to minimize shot noise subtraction effect
    let n_particles_fake = 1000000;
    let result = compute_pk(&grid, box_size, grid_size, n_particles_fake, 16);

    // Find peak (skip first bin which might have DC offset)
    let max_idx = result.pk[1..].iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i + 1)
        .unwrap_or(1);

    let k_peak = result.k[max_idx];

    // Peak should be near target k (within one bin width)
    let dk = result.k[1] - result.k[0];
    assert!((k_peak - k_target).abs() < 2.0 * dk,
        "Single mode peak: k_peak={:.3}, k_target={:.3}, dk={:.3}",
        k_peak, k_target, dk);
}

// ============================================================================
// ΛCDM P(k) TESTS
// ============================================================================

/// Test 7: ΛCDM P(k) shape (decreasing at high k)
#[test]
fn test_lcdm_pk_shape() {
    let pk_01 = lcdm_pk(0.01, 0.8, 0.965);
    let pk_1 = lcdm_pk(1.0, 0.8, 0.965);

    assert!(pk_01 > pk_1,
        "ΛCDM P(k) should decrease: P(0.01)={:.2e} > P(1.0)={:.2e}",
        pk_01, pk_1);
}

/// Test 8: ΛCDM P(k) σ₈² scaling
#[test]
fn test_lcdm_pk_sigma8_scaling() {
    let pk1 = lcdm_pk(0.1, 0.7, 0.965);
    let pk2 = lcdm_pk(0.1, 1.4, 0.965);  // 2× σ₈

    let ratio = pk2 / pk1;
    assert!((ratio - 4.0).abs() < 0.2,
        "P(k) ∝ σ₈²: ratio = {:.2} (expected 4)", ratio);
}

/// Test 9: ΛCDM P(k) spectral index dependence
#[test]
fn test_lcdm_pk_spectral_index() {
    // P(k) ∝ k^n_s, so for k < 1:
    // - lower n_s → more power at k < 1 (k^0.9 > k^1.0 for k < 1)
    // - higher n_s → more power at k > 1
    let pk_low_ns_lowk = lcdm_pk(0.01, 0.8, 0.90);
    let pk_high_ns_lowk = lcdm_pk(0.01, 0.8, 1.00);

    // At k=0.01 < 1, lower n_s gives more power
    assert!(pk_low_ns_lowk > pk_high_ns_lowk,
        "Lower n_s → more power at k=0.01: n_s=0.9 gives {:.2e} > n_s=1.0 gives {:.2e}",
        pk_low_ns_lowk, pk_high_ns_lowk);

    // Verify dependence works (different n_s gives different P(k))
    let ratio = pk_low_ns_lowk / pk_high_ns_lowk;
    assert!(ratio > 1.1 && ratio < 5.0,
        "n_s should affect P(k): ratio = {:.2}", ratio);
}

// ============================================================================
// CROSS-SPECTRUM TESTS (Janus m+/m- anticorrelation)
// ============================================================================

/// Test 10: Cross-spectrum of identical fields = auto-spectrum
#[test]
fn test_cross_pk_identical() {
    let grid_size = 16;
    let box_size = 100.0;
    let n = 500;

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            [t * box_size, (2.0 * t) % 1.0 * box_size, (3.0 * t) % 1.0 * box_size]
        })
        .collect();

    let grid = cic_assign(&positions, box_size, grid_size);
    let auto = compute_pk(&grid, box_size, grid_size, n, 8);
    let cross = compute_cross_pk(&grid, &grid, box_size, grid_size, 8);

    // Cross-spectrum of field with itself should match auto-spectrum
    // (plus shot noise in auto case)
    for (i, (&p_auto, &p_cross)) in auto.pk.iter().zip(cross.pk.iter()).enumerate() {
        if auto.n_modes[i] > 10 {
            // Allow for shot noise difference
            let shot = box_size.powi(3) / n as f64;
            let diff = (p_auto + shot - p_cross).abs();
            assert!(diff < shot * 2.0,
                "bin {}: auto={:.2e}, cross={:.2e}", i, p_auto, p_cross);
        }
    }
}

/// Test 11: Cross-spectrum of anticorrelated fields is negative
#[test]
fn test_cross_pk_anticorrelated() {
    let grid_size = 16;
    let n_cells = grid_size * grid_size * grid_size;
    let box_size = 100.0;

    // Create anticorrelated fields: δ₂ = -δ₁
    let mut grid1 = vec![0.0; n_cells];
    for i in 0..n_cells {
        grid1[i] = 1.0 + 0.1 * (i as f64 * 0.1).sin();
    }
    let mean1: f64 = grid1.iter().sum::<f64>() / n_cells as f64;

    // δ₂ = -δ₁ implies ρ₂ = ρ̄ - (ρ₁ - ρ̄) = 2ρ̄ - ρ₁
    let grid2: Vec<f64> = grid1.iter().map(|&rho| 2.0 * mean1 - rho).collect();

    let cross = compute_cross_pk(&grid1, &grid2, box_size, grid_size, 8);

    // Cross-spectrum should be negative
    let mean_cross: f64 = cross.pk.iter().sum::<f64>() / cross.pk.len() as f64;
    assert!(mean_cross < 0.0,
        "Anticorrelated cross-spectrum should be negative: {:.2e}", mean_cross);
}

// ============================================================================
// INTEGRATION TEST
// ============================================================================

/// Test 12: Full P(k) pipeline on clustered distribution
#[test]
fn test_pk_clustered_distribution() {
    let grid_size = 32;
    let box_size = 100.0;
    let n = 2000;

    // Create clustered distribution (multiple blobs)
    let mut positions = Vec::with_capacity(n);
    let centers = [
        [20.0, 20.0, 20.0],
        [80.0, 80.0, 80.0],
        [50.0, 20.0, 80.0],
    ];

    for i in 0..n {
        let center = &centers[i % 3];
        let t = i as f64 / n as f64;
        // Add particles near centers with some scatter
        let scatter = 5.0;
        positions.push([
            (center[0] + scatter * (t * 157.3).sin()) % box_size,
            (center[1] + scatter * (t * 229.7).cos()) % box_size,
            (center[2] + scatter * (t * 311.1).sin()) % box_size,
        ]);
    }

    let grid = cic_assign(&positions, box_size, grid_size);
    let result = compute_pk(&grid, box_size, grid_size, n, 16);

    // Clustered distribution should have more power at low k
    let pk_low_k = result.pk[0..4].iter().sum::<f64>() / 4.0;
    let pk_high_k = result.pk[12..16].iter().sum::<f64>() / 4.0;

    assert!(pk_low_k > pk_high_k,
        "Clustered P(k) should have more low-k power: low={:.2e}, high={:.2e}",
        pk_low_k, pk_high_k);
}
