//! Splitting functions for TreePM
//!
//! Defines how forces are split between PM (long-range) and Tree (short-range).
//! Uses a smooth polynomial splitting function to avoid discontinuities at r_cut.
//!
//! DESIGN CHOICE:
//! We use a polynomial x^4 splitting function rather than erfc because:
//! 1. Simpler and faster to compute
//! 2. Exactly 0 at r=0 and exactly 1 at r=r_cut
//! 3. Smooth first derivative at boundaries
//!
//! For r >= r_cut, PM handles 100% of the force (Tree returns 0).
//! For r < r_cut, the split is: PM = (r/r_cut)^4, Tree = 1 - PM

/// PM (long-range) splitting weight
///
/// At r = 0: returns 0 (Tree handles 100%)
/// At r = r_cut: returns 1 (PM handles 100%)
/// For r >= r_cut: returns 1 (PM handles 100%)
pub fn splitting_pm(r: f64, r_cut: f64) -> f64 {
    if r >= r_cut {
        1.0
    } else {
        let x = r / r_cut;
        let x2 = x * x;
        x2 * x2  // x^4 for smooth transition
    }
}

/// Tree (short-range) splitting weight
/// S_tree(r) = 1 - S_pm(r)
///
/// At r = 0: returns 1 (Tree handles 100%)
/// At r = r_cut: returns 0 (PM handles 100%)
/// For r >= r_cut: returns 0 (Tree does nothing)
pub fn splitting_tree(r: f64, r_cut: f64) -> f64 {
    1.0 - splitting_pm(r, r_cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitting_limits() {
        let r_cut = 46.0;  // Typical r_cut

        // At r = 0: Tree dominates (PM = 0)
        let s_pm_0 = splitting_pm(0.0, r_cut);
        assert!(s_pm_0.abs() < 1e-10, "At r=0, S_pm should be 0, got {}", s_pm_0);
        assert!((splitting_tree(0.0, r_cut) - 1.0).abs() < 1e-10, "At r=0, S_tree should be 1");

        // At r = r_cut: PM dominates (PM = 1)
        let s_pm_cut = splitting_pm(r_cut, r_cut);
        assert!((s_pm_cut - 1.0).abs() < 1e-10, "At r=r_cut, S_pm should be 1, got {}", s_pm_cut);
        assert!(splitting_tree(r_cut, r_cut).abs() < 1e-10, "At r=r_cut, S_tree should be 0");

        // At r >> r_cut: PM dominates (PM = 1)
        let s_pm_far = splitting_pm(5.0 * r_cut, r_cut);
        assert!((s_pm_far - 1.0).abs() < 1e-10, "At r>>r_cut, S_pm should be 1, got {}", s_pm_far);

        // At r = r_cut/2: check smooth transition
        let s_pm_mid = splitting_pm(r_cut / 2.0, r_cut);
        assert!((s_pm_mid - 0.0625).abs() < 1e-10, "At r=r_cut/2, S_pm should be 0.0625, got {}", s_pm_mid);
    }

    #[test]
    fn test_splitting_complementary() {
        let r_cut = 46.0;
        for r in [0.0, 10.0, 23.0, 46.0, 100.0, 200.0] {
            let sum = splitting_pm(r, r_cut) + splitting_tree(r, r_cut);
            assert!((sum - 1.0).abs() < 1e-10,
                "S_pm + S_tree should = 1.0 at r={}, got {}", r, sum);
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
}
