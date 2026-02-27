//! Splitting functions for TreePM
//!
//! Defines how forces are split between PM (long-range) and Tree (short-range).
//! Uses a smooth splitting function to avoid discontinuities at r_cut.

/// Gaussian splitting function for force decomposition
///
/// At r = 0: returns 0 (all force from Tree)
/// At r = r_cut: returns ~0.5
/// At r >> r_cut: returns 1 (all force from PM)
pub fn splitting_pm(r: f64, r_cut: f64) -> f64 {
    // Complementary error function approximation
    // S_pm(r) = erfc(r / (2 * r_s)) where r_s = r_cut / 3
    let r_s = r_cut / 3.0;
    let x = r / (2.0 * r_s);

    // erfc approximation (accurate to ~1e-5)
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
               t * (-1.453152027 + t * 1.061405429)))) * (-x * x).exp();

    1.0 - erfc
}

/// Tree splitting function (complementary to PM)
/// S_tree(r) = 1 - S_pm(r)
pub fn splitting_tree(r: f64, r_cut: f64) -> f64 {
    1.0 - splitting_pm(r, r_cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitting_limits() {
        let r_cut = 46.0;  // Typical r_cut

        // At r = 0: Tree dominates
        let s_pm_0 = splitting_pm(0.0, r_cut);
        assert!(s_pm_0 < 0.01, "At r=0, S_pm should be ~0, got {}", s_pm_0);

        // At r = r_cut: roughly equal
        let s_pm_cut = splitting_pm(r_cut, r_cut);
        assert!(s_pm_cut > 0.3 && s_pm_cut < 0.7,
            "At r=r_cut, S_pm should be ~0.5, got {}", s_pm_cut);

        // At r >> r_cut: PM dominates
        let s_pm_far = splitting_pm(5.0 * r_cut, r_cut);
        assert!(s_pm_far > 0.99, "At r>>r_cut, S_pm should be ~1, got {}", s_pm_far);
    }

    #[test]
    fn test_splitting_complementary() {
        let r_cut = 46.0;
        for r in [0.0, 10.0, 46.0, 100.0, 200.0] {
            let sum = splitting_pm(r, r_cut) + splitting_tree(r, r_cut);
            assert!((sum - 1.0).abs() < 1e-10,
                "S_pm + S_tree should = 1.0 at r={}, got {}", r, sum);
        }
    }
}
