//! Continuous-form discrete Laplacian for the periodic Poisson solver.
//!
//! Reference: GrGadget Eq. (22), Quintana-Miranda et al. 2023.
//!
//! Returns k² in physical units, used as Φ̂(k) = -4πG · ρ̂(k) / k².
//! This is the "continuous form" preferred over `(2/h)² · Σ sin²(πk_i/N)`
//! (Gevolution form) which introduces additional anisotropy.

/// k² in physical units for periodic Fourier mode (kx, ky, kz).
///
/// Convention:
///   k² = (k_x² + k_y² + k_z²) · (2π/L)²
/// where (kx, ky, kz) are integer mode indices and L is the box size.
///
/// At k = 0 (zero mode), returns 1.0 as a sentinel value. The caller MUST
/// set Φ̂(0) = 0 explicitly (DC mode is unconstrained by Poisson).
#[inline(always)]
pub fn k_squared_continuous(kx: i32, ky: i32, kz: i32, l: f64) -> f64 {
    if kx == 0 && ky == 0 && kz == 0 {
        return 1.0; // sentinel: caller handles k=0 separately
    }
    let two_pi_over_l = 2.0 * std::f64::consts::PI / l;
    let k2_int = (kx * kx + ky * ky + kz * kz) as f64;
    k2_int * two_pi_over_l * two_pi_over_l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k2_zero_mode_sentinel() {
        // Zero mode returns 1.0 sentinel; caller is expected to short-circuit.
        let k2 = k_squared_continuous(0, 0, 0, 100.0);
        assert!(k2 == 1.0);
    }

    #[test]
    fn test_k2_known_modes() {
        let l = 100.0_f64;
        // Mode (1,0,0): k² = (2π/L)²
        let k2 = k_squared_continuous(1, 0, 0, l);
        let expected = (2.0 * std::f64::consts::PI / l).powi(2);
        // Tolerance: 1e-14 (multiplicative FP error).
        assert!((k2 - expected).abs() / expected < 1e-14);

        // Mode (3,4,12): k² = (2π/L)² × (9+16+144) = (2π/L)² × 169
        let k2 = k_squared_continuous(3, 4, 12, l);
        let expected = (2.0 * std::f64::consts::PI / l).powi(2) * 169.0;
        assert!((k2 - expected).abs() / expected < 1e-14);
    }

    #[test]
    fn test_k2_box_scaling() {
        // For a fixed integer mode, k² ∝ 1/L²
        let mode = (1, 0, 0);
        let k2_at_1 = k_squared_continuous(mode.0, mode.1, mode.2, 1.0);
        let k2_at_10 = k_squared_continuous(mode.0, mode.1, mode.2, 10.0);
        // ratio = (1/10)² / (1/1)² = 0.01
        let ratio = k2_at_10 / k2_at_1;
        assert!(
            (ratio - 0.01).abs() / 0.01 < 1e-14,
            "Expected 0.01, got {}",
            ratio
        );
    }

    #[test]
    fn test_k2_negative_modes_equal_positive() {
        // k² is even in each component: k(-i, j, k) = k(i, j, k)
        let l = 100.0_f64;
        let k2_pos = k_squared_continuous(5, 7, -3, l);
        let k2_neg = k_squared_continuous(-5, -7, 3, l);
        // Tolerance: exact (only sign squared)
        assert!((k2_pos - k2_neg).abs() < 1e-14);
    }
}
