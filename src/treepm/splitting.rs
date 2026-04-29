//! Splitting functions for TreePM.
//!
//! Defines how forces are split between PM (long-range) and Tree (short-range).
//!
//! ## Two splitting conventions
//!
//! ### Convention 1 (Springel 2005, GADGET-2 / PhotoNs-GPU) — RECOMMENDED
//!
//! Compatible with PM Gaussian damping `exp(-k²·r_s²)` (`pm_grid.rs`):
//!   F_tree(r) = -(G·m·r̂/r²) × T(r/(2·r_s))
//!   T(x) = erfc(x) + (2x/√π)·exp(-x²)
//!
//! Use [`splitting_tree_springel`].
//!
//! ### Convention 2 (Bagla 2002, polynomial) — DEPRECATED
//!
//! Polynomial `1 - (r/r_cut)^4` for the Tree, complement for PM. Simple,
//! but INCOMPATIBLE with the Gaussian PM convention currently in use.
//!
//! Functions [`splitting_pm`] and [`splitting_tree`] kept for backward
//! compatibility; use the Springel convention for new code.

/// **DEPRECATED — INCONSISTENT WITH GAUSSIAN PM**
///
/// Polynomial PM (long-range) splitting weight (Bagla 2002).
/// Returns 0 at r=0, approaches 1 as r→r_cut, exactly 1 for r>=r_cut.
pub fn splitting_pm(r: f64, r_cut: f64) -> f64 {
    if r >= r_cut {
        1.0
    } else {
        let x = r / r_cut;
        let x2 = x * x;
        x2 * x2 // x^4 for smooth transition
    }
}

/// **DEPRECATED — INCONSISTENT WITH GAUSSIAN PM**
///
/// Polynomial Tree (short-range) splitting weight (Bagla 2002).
/// Use [`splitting_tree_springel`] for TreePM-Gaussian compatibility.
pub fn splitting_tree(r: f64, r_cut: f64) -> f64 {
    1.0 - splitting_pm(r, r_cut)
}

/// Tree (short-range) splitting weight, Springel 2005 / PhotoNs convention.
///
/// Compatible with PM Gaussian damping `exp(-k²·r_s²)` so that
/// `F_PM_long + F_tree_short ≈ F_full_Newton` exactly (within FFT and
/// approximation precision).
///
/// Formula:
///   T(x) = erfc(x) + (2x/√π)·exp(-x²)
/// where x = r/(2·r_s).
///
/// Returns 0 for r ≥ 6·r_s (table cutoff x ≥ 3, T(3) ≈ 4.4e-4 negligible).
///
/// ### Properties
/// - T(0) = 1 (Tree handles 100% at r=0)
/// - T(2·r_s) ≈ 0.5724 (50/50 split at r = 2·r_s, x=1)
/// - T(6·r_s) ≈ 0 (PM handles 100% at r ≥ cutoff)
/// - Monotonically decreasing
///
/// Reference: Springel 2005 MNRAS 364, 1105, §3 + Appendix.
/// PhotoNs-GPU §2 (Wang & Meng 2021).
#[inline(always)]
pub fn splitting_tree_springel(r: f64, r_s: f64) -> f64 {
    let x = r / (2.0 * r_s);
    if x >= 3.0 {
        return 0.0;
    }

    // Use Abramowitz & Stegun §7.1.26 erfc approximation (|err| < 1.5e-7).
    // Same approximation as in truncation_table.rs (Phase 3) for consistency.
    let exp_mx2 = (-x * x).exp();
    erfc_approx(x) + (2.0 * x / std::f64::consts::PI.sqrt()) * exp_mx2
}

/// Approximation of erfc(x) for x ≥ 0 (Abramowitz & Stegun §7.1.26).
/// Maximum absolute error: 1.5e-7. Identical to `truncation_table::erfc_approx`.
#[inline(always)]
fn erfc_approx(x: f64) -> f64 {
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
    1.0 - sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitting_limits() {
        let r_cut = 46.0;

        let s_pm_0 = splitting_pm(0.0, r_cut);
        assert!(s_pm_0.abs() < 1e-10, "At r=0, S_pm should be 0, got {}", s_pm_0);
        assert!(
            (splitting_tree(0.0, r_cut) - 1.0).abs() < 1e-10,
            "At r=0, S_tree should be 1"
        );

        let s_pm_cut = splitting_pm(r_cut, r_cut);
        assert!(
            (s_pm_cut - 1.0).abs() < 1e-10,
            "At r=r_cut, S_pm should be 1, got {}",
            s_pm_cut
        );
        assert!(
            splitting_tree(r_cut, r_cut).abs() < 1e-10,
            "At r=r_cut, S_tree should be 0"
        );

        let s_pm_far = splitting_pm(5.0 * r_cut, r_cut);
        assert!(
            (s_pm_far - 1.0).abs() < 1e-10,
            "At r>>r_cut, S_pm should be 1, got {}",
            s_pm_far
        );

        let s_pm_mid = splitting_pm(r_cut / 2.0, r_cut);
        assert!(
            (s_pm_mid - 0.0625).abs() < 1e-10,
            "At r=r_cut/2, S_pm should be 0.0625, got {}",
            s_pm_mid
        );
    }

    #[test]
    fn test_splitting_complementary() {
        let r_cut = 46.0;
        for r in [0.0, 10.0, 23.0, 46.0, 100.0, 200.0] {
            let sum = splitting_pm(r, r_cut) + splitting_tree(r, r_cut);
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "S_pm + S_tree should = 1.0 at r={}, got {}",
                r,
                sum
            );
        }
    }

    #[test]
    fn test_splitting_monotonic() {
        let r_cut = 46.0;
        let mut prev_pm = 0.0;
        for i in 0..=100 {
            let r = r_cut * i as f64 / 100.0;
            let pm = splitting_pm(r, r_cut);
            assert!(pm >= prev_pm, "S_pm should be monotonically increasing");
            prev_pm = pm;
        }
    }

    // ============================================================
    // Springel splitting tests (Phase 9.6)
    // ============================================================

    #[test]
    fn test_splitting_springel_at_zero() {
        // T(0) = erfc(0) + 0 = 1
        // Tolerance 1e-7 (Abramowitz approx specs)
        let val = splitting_tree_springel(0.0, 1.0);
        assert!((val - 1.0).abs() < 1e-7, "T(0) = {}", val);
    }

    #[test]
    fn test_splitting_springel_at_split_scale() {
        // À r = 2·r_s, x = 1, T(1) = erfc(1) + 2/(√π·e) ≈ 0.5724
        // Tolerance 1e-3 (well within Abramowitz 1.5e-7 approx + simple binary
        // representation)
        let val = splitting_tree_springel(2.0, 1.0);
        let expected = 0.5724_f64;
        assert!(
            (val - expected).abs() < 1e-3,
            "T at r=2rs: {}, expected {}",
            val,
            expected
        );
    }

    #[test]
    fn test_splitting_springel_monotonic_decrease() {
        // T(x) doit être strictement décroissante (sauf cutoff = 0)
        let r_s = 1.0;
        let mut prev = 1.0001_f64;
        for i in 1..30 {
            let r = (i as f64) * 0.2;
            let val = splitting_tree_springel(r, r_s);
            assert!(
                val <= prev,
                "Not monotonic at r={}: {} > {}",
                r,
                val,
                prev
            );
            prev = val;
        }
    }

    #[test]
    fn test_splitting_springel_cutoff() {
        // À r = 6·r_s (x=3), T ≈ 4.4e-4 mais on retourne 0 pour le cutoff dur
        let val = splitting_tree_springel(6.001, 1.0);
        assert_eq!(val, 0.0, "Above cutoff should be 0");

        let val = splitting_tree_springel(10.0, 1.0);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_splitting_springel_at_x_2() {
        // À x = 2 (r = 4·r_s), T(2) = erfc(2) + 4/(√π·e⁴) ≈ 0.04601
        let val = splitting_tree_springel(4.0, 1.0);
        let expected = 0.04601_f64;
        assert!(
            (val - expected).abs() < 1e-3,
            "T at x=2: {}, expected {}",
            val,
            expected
        );
    }
}
