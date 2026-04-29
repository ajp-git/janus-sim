//! CIC (Cloud-In-Cell) deconvolution in Fourier space.
//!
//! References:
//! - Sefusatti et al. 2016, MNRAS 460, 3624
//! - GrGadget §3.3.1 (Quintana-Miranda et al. 2023)
//!
//! The CIC deposit convolves the density with a triangle kernel. In Fourier
//! space, this multiplies by W_CIC(k) = sinc²(πk/N) per dimension. To recover
//! the true spectral density, we deconvolve by this factor — TWICE in the
//! TreePM pipeline:
//!   1. After scatter (ρ̂ /= W_CIC²) before Green's function
//!   2. Before gather (Φ̂ /= W_CIC²) before inverse FFT
//!
//! Note: each CIC operation contributes one factor sinc². With CIC = 2nd
//! order interpolation, the total deconvolution per pass is sinc²(πk/N) per
//! dim, hence inv_sinc⁴(πk/N) per dim total when both passes are applied.
//! This module returns inv_sinc² per dim for a single pass.

/// Inverse of W_CIC(k)² = sinc²(πk/N) for one Fourier mode (kx, ky, kz).
///
/// Returns:
///   1 / [sinc²(πkx/N) · sinc²(πky/N) · sinc²(πkz/N)]
///
/// where `sinc(x) = sin(x)/x` (NOT `sin(πx)/(πx)`, despite the common
/// "normalized sinc" naming).
///
/// At k = 0 (any axis), sinc(0) = 1, so the corresponding factor is 1.
///
/// # Arguments
/// * `kx, ky, kz` - integer Fourier mode indices in [-N/2, N/2)
/// * `n` - grid size per dimension
#[inline(always)]
pub fn cic_window_inv_squared(kx: i32, ky: i32, kz: i32, n: usize) -> f64 {
    let n_f = n as f64;

    let inv_sinc = |k: i32| -> f64 {
        if k == 0 {
            1.0
        } else {
            let arg = std::f64::consts::PI * (k as f64) / n_f;
            arg / arg.sin()
        }
    };

    let sx = inv_sinc(kx);
    let sy = inv_sinc(ky);
    let sz = inv_sinc(kz);
    // Per-dim CIC = sinc² → inv = inv_sinc²; product over 3 dims.
    (sx * sy * sz).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cic_window_at_zero() {
        // k=0 in all dimensions: no correction, factor exactly 1.0.
        // Tolerance: 1e-15 (exact computation, only int ops).
        let n = 64;
        let w = cic_window_inv_squared(0, 0, 0, n);
        assert!((w - 1.0).abs() < 1e-15, "Got {}", w);
    }

    #[test]
    fn test_cic_window_at_nyquist_x_only() {
        // kx = N/2, ky=kz=0: sinc(π/2) = 2/π, inv_sinc = π/2
        // w = (π/2)² (one inv_sinc² in x dim, others contribute 1).
        // Tolerance: 1e-13 (FP arithmetic + sin evaluation).
        let n = 64;
        let w = cic_window_inv_squared((n / 2) as i32, 0, 0, n);
        let expected = (std::f64::consts::PI / 2.0).powi(2);
        assert!(
            (w - expected).abs() < 1e-13,
            "Expected {}, got {}",
            expected,
            w
        );
    }

    #[test]
    fn test_cic_window_at_nyquist_xyz() {
        // kx=ky=kz=N/2 corner: w = (π/2)^6
        // Tolerance: 1e-12 (cumulative FP).
        let n = 64;
        let h = (n / 2) as i32;
        let w = cic_window_inv_squared(h, h, h, n);
        let expected = (std::f64::consts::PI / 2.0).powi(6);
        assert!(
            (w - expected).abs() / expected < 1e-12,
            "Expected {}, got {}",
            expected,
            w
        );
    }

    #[test]
    fn test_cic_window_symmetric_in_k() {
        // W(kx, ky, kz) should equal W(±kx, ±ky, ±kz) since sinc is even.
        let n = 64;
        for kx in [-15, -1, 7, 31].iter() {
            for ky in [-7, 0, 12].iter() {
                for kz in [3, -20].iter() {
                    let w_pos = cic_window_inv_squared(*kx, *ky, *kz, n);
                    let w_neg = cic_window_inv_squared(-*kx, -*ky, -*kz, n);
                    // Tolerance: 1e-15 (exact symmetry under sign flip).
                    assert!(
                        (w_pos - w_neg).abs() < 1e-15 * w_pos.max(1e-12),
                        "Asymmetric at ({},{},{}): {} vs {}",
                        kx,
                        ky,
                        kz,
                        w_pos,
                        w_neg
                    );
                }
            }
        }
    }

    #[test]
    fn test_cic_window_monotonic_along_axis() {
        // |W(k,0,0)| should monotonically grow from k=0 to k=N/2.
        let n = 32;
        let mut prev = 1.0;
        for k in 1..=(n as i32 / 2) {
            let w = cic_window_inv_squared(k, 0, 0, n);
            assert!(
                w >= prev,
                "Non-monotonic at k={}: {} < prev {}",
                k,
                w,
                prev
            );
            prev = w;
        }
    }

    #[test]
    fn test_cic_window_known_intermediate_values() {
        // Spot-check at k=N/4 (mid Nyquist): sinc(π/4) = sin(π/4)/(π/4) = (√2/2)/(π/4) = 2√2/π
        // inv_sinc(π/4) = π/(2√2) = π√2/4
        // w_x_only = (π√2/4)² = π²·2/16 = π²/8
        let n = 64;
        let kx = n as i32 / 4;
        let w = cic_window_inv_squared(kx, 0, 0, n);
        let expected = std::f64::consts::PI.powi(2) / 8.0;
        // Tolerance: 1e-14 sufficient for sin approximation.
        assert!(
            (w - expected).abs() / expected < 1e-14,
            "Expected {}, got {}",
            expected,
            w
        );
    }
}
