//! Truncation function lookup table for TreePM short-range kernel.
//!
//! Reference: PhotoNs-GPU §3.3, Wang & Meng 2021.
//!
//! On GPU, evaluating erfc() directly costs ~5.5× the flops of an add/mul.
//! Pre-compute a table of 512 points and use Taylor expansion for accuracy.
//!
//! Truncation function (PhotoNs Eq. 1):
//!   T(x) = erfc(x) + (2x/√π) × exp(-x²)
//!   E(x) = -4·exp(-x²) / √π   (used in Taylor expansion coefficient)
//!
//! where x = r/(2 r_s) and r_s is the splitting scale.
//!
//! At x=0: T(0) = 1, E(0) = -4/√π
//! At x=3: T(3) ≈ 4.4e-4 (effectively zero, beyond cutoff)
//!
//! Use erfc approximation Abramowitz & Stegun §7.1.26 (5-term, |err| < 1.5e-7).

const TABLE_SIZE: usize = 512;
const X_MAX: f64 = 3.0;

/// Approximation of erfc(x) for x >= 0 using Abramowitz & Stegun §7.1.26.
///
/// Maximum absolute error: 1.5e-7 over x ∈ [0, +∞).
/// Reference: M. Abramowitz, I.A. Stegun, "Handbook of Mathematical Functions",
/// 9th edition, eq. 7.1.26 (page 299).
pub fn erfc_approx(x: f64) -> f64 {
    // Polynomial coefficients (Abramowitz & Stegun)
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (a1 + (a2 + (a3 + (a4 + a5 * t) * t) * t) * t) * t * (-x_abs * x_abs).exp();
    // erf(x) = sign × y, so erfc(x) = 1 - sign × y
    1.0 - sign * y
}

/// Build the truncation function tables T(x), E(x) on x ∈ [0, X_MAX].
///
/// Returns (t_table, e_table), each of length TABLE_SIZE.
/// Values stored as f32 (sufficient for SP GPU kernel, plan §3.0).
///
/// Convention:
/// - Index i corresponds to x_i = i × X_MAX / (TABLE_SIZE - 1)
/// - T(x_i) = erfc(x_i) + (2·x_i/√π) × exp(-x_i²)
/// - E(x_i) = -4·exp(-x_i²) / √π
pub fn build_truncation_table() -> (Vec<f32>, Vec<f32>) {
    let n = TABLE_SIZE;
    let mut t_table = vec![0.0_f32; n];
    let mut e_table = vec![0.0_f32; n];

    let sqrt_pi = std::f64::consts::PI.sqrt();

    for i in 0..n {
        let x = (i as f64) * X_MAX / (n as f64 - 1.0);
        let exp_mx2 = (-x * x).exp();

        let ti = erfc_approx(x) + (2.0 * x / sqrt_pi) * exp_mx2;
        let ei = -4.0 * exp_mx2 / sqrt_pi;

        t_table[i] = ti as f32;
        e_table[i] = ei as f32;
    }

    (t_table, e_table)
}

/// Exact truncation function (for verification/testing only).
pub fn truncation_exact(x: f64) -> f64 {
    let exp_mx2 = (-x * x).exp();
    erfc_approx(x) + (2.0 * x / std::f64::consts::PI.sqrt()) * exp_mx2
}

/// Linear interpolation in the table at x.
///
/// `order` parameter for higher-order Taylor approximation; here we use
/// linear (order=1) for simplicity; PhotoNs-GPU §3.3 uses Taylor up to order 4.
/// For testing purposes, linear interp gives error < 1e-3 with TABLE_SIZE=512;
/// good enough for correctness validation. The GPU version will use Taylor 4.
pub fn interpolate_truncation_linear(x: f64, t_table: &[f32]) -> f64 {
    if x >= X_MAX {
        return 0.0;
    }
    if x <= 0.0 {
        return t_table[0] as f64;
    }
    let n = t_table.len();
    let fidx = x * (n as f64 - 1.0) / X_MAX;
    let idx = fidx.floor() as usize;
    let eps = fidx - idx as f64;
    if idx + 1 >= n {
        return t_table[n - 1] as f64;
    }
    (1.0 - eps) * (t_table[idx] as f64) + eps * (t_table[idx + 1] as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erfc_approx_known_values() {
        // erfc(0) = 1.0
        // Tolerance: 1e-7 (Abramowitz §7.1.26 specs)
        assert!((erfc_approx(0.0) - 1.0).abs() < 1e-7);

        // erfc(1) ≈ 0.157299
        assert!((erfc_approx(1.0) - 0.157299).abs() < 1e-5);

        // erfc(2) ≈ 0.004678
        assert!((erfc_approx(2.0) - 0.004678).abs() < 1e-5);

        // erfc(3) ≈ 2.21e-5
        assert!((erfc_approx(3.0) - 2.21e-5).abs() < 1e-5);
    }

    #[test]
    fn test_truncation_table_at_zero() {
        let (t, e) = build_truncation_table();
        // x=0: T(0) = erfc(0) + 0 = 1.0
        // Tolerance: 1e-5 (SP storage + erfc approx)
        assert!((t[0] - 1.0).abs() < 1e-5, "T(0) = {}, expected 1.0", t[0]);

        let expected_e = -4.0 / std::f64::consts::PI.sqrt();
        // E(0) = -4/√π ≈ -2.2568
        assert!(
            (e[0] as f64 - expected_e).abs() < 1e-4,
            "E(0) = {}, expected {}",
            e[0],
            expected_e
        );
    }

    #[test]
    fn test_truncation_table_at_x1() {
        let (t, _) = build_truncation_table();
        let n = TABLE_SIZE;
        // Index for x=1: i = 1.0 × (n-1) / X_MAX = (n-1)/3
        let idx_1 = ((n - 1) as f64 / X_MAX).round() as usize;
        let x_actual = idx_1 as f64 * X_MAX / (n - 1) as f64;

        let t_exact = truncation_exact(x_actual);
        // Tolerance: 1e-5 (SP storage + interp)
        assert!(
            (t[idx_1] as f64 - t_exact).abs() < 1e-5,
            "T at x≈1 differs by {}",
            (t[idx_1] as f64 - t_exact).abs()
        );

        // Reference: T(1) ≈ 0.5724
        assert!(
            (t[idx_1] as f64 - 0.5724).abs() < 0.01,
            "T(1) = {}, expected ~0.5724",
            t[idx_1]
        );
    }

    #[test]
    fn test_truncation_table_at_x2() {
        let (t, _) = build_truncation_table();
        let n = TABLE_SIZE;
        let idx_2 = (2.0 * (n - 1) as f64 / X_MAX).round() as usize;
        let x_actual = idx_2 as f64 * X_MAX / (n - 1) as f64;
        let t_exact = truncation_exact(x_actual);
        assert!((t[idx_2] as f64 - t_exact).abs() < 1e-5);
        // T(2) ≈ 0.04601
        assert!((t[idx_2] as f64 - 0.04601).abs() < 0.01);
    }

    #[test]
    fn test_truncation_at_cutoff() {
        let (t, _) = build_truncation_table();
        // x=X_MAX=3: T(3) ≈ 4.4e-4 (effectively zero)
        assert!(t[TABLE_SIZE - 1].abs() < 1e-2);
    }

    #[test]
    fn test_interpolation_accuracy() {
        let (t, _) = build_truncation_table();

        // Compare interpolated vs exact at non-grid points
        let test_points = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        for &x in &test_points {
            let interp = interpolate_truncation_linear(x, &t);
            let exact = truncation_exact(x);
            // Tolerance: 1e-3 with linear interp, n=512 (sufficient for sanity)
            // PhotoNs Taylor 4 achieves 1e-6, but we test linear here.
            let abs_err = (interp - exact).abs();
            assert!(
                abs_err < 1e-3,
                "x={}, interp={}, exact={}, err={}",
                x,
                interp,
                exact,
                abs_err
            );
        }
    }

    #[test]
    fn test_table_monotonic_decreasing() {
        let (t, _) = build_truncation_table();
        // T(x) should be monotonically decreasing for x > 0
        for w in t.windows(2) {
            assert!(
                w[0] >= w[1],
                "Non-monotonic T table: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_table_size_constant() {
        let (t, e) = build_truncation_table();
        assert_eq!(t.len(), TABLE_SIZE);
        assert_eq!(e.len(), TABLE_SIZE);
    }
}
