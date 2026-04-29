//! Canonical TreePM parameters from PhotoNs-GPU + GreeM.
//!
//! References:
//! - PhotoNs-GPU §2 (Wang & Meng 2021): r_s = 1.2·Δg, r_cut = 6·Δg
//! - GreeM (Ishiyama 2009): θ = 0.5, N_leaf = 10
//! - Plan §4.1
//!
//! These constants define the canonical operating point. The current
//! `mod.rs` exposes `R_CUT_FACTOR = 16` (i.e. r_cut = box/16 = 16·Δg at
//! N_pm=256) which is more conservative than PhotoNs canonical (6·Δg).
//! The plan §4.2 is a parameter sweep that should validate the choice.

/// Canonical PhotoNs-GPU split scale: r_s = SPLIT_SCALE_FACTOR × Δg
pub const SPLIT_SCALE_FACTOR: f64 = 1.2;

/// Canonical PhotoNs-GPU short-range cutoff: r_cut = CUTOFF_FACTOR × Δg
pub const CUTOFF_FACTOR: f64 = 6.0;

/// Canonical GreeM opening angle for tree (z < 10).
pub const THETA_DEFAULT: f64 = 0.5;

/// Canonical particles per leaf cell.
pub const N_LEAF: usize = 10;

/// Canonical group critical threshold for tree task generation.
pub const N_CRIT: usize = 300;

/// Compute mesh cell size from box and PM grid resolution.
#[inline(always)]
pub fn delta_grid(box_size: f64, n_pm: usize) -> f64 {
    box_size / n_pm as f64
}

/// Compute PhotoNs canonical r_s for given box and N_pm.
#[inline(always)]
pub fn r_s_canonical(box_size: f64, n_pm: usize) -> f64 {
    SPLIT_SCALE_FACTOR * delta_grid(box_size, n_pm)
}

/// Compute PhotoNs canonical r_cut for given box and N_pm.
#[inline(always)]
pub fn r_cut_canonical(box_size: f64, n_pm: usize) -> f64 {
    CUTOFF_FACTOR * delta_grid(box_size, n_pm)
}

/// Recommended PM grid size for a given particle count and box size.
///
/// Heuristic from plan §1.4 / §4.1:
/// - Aim for N_pm³ ≈ N_particles (cell size ~ mean inter-particle distance)
/// - Round to power of 2 for FFT efficiency
///
/// Examples:
/// - N=1M, L=250 Mpc → N_pm=256
/// - N=10M, L=500 Mpc → N_pm=512
/// - N=100K, L=100 Mpc → N_pm=128
pub fn recommended_pm_grid(n_particles: usize) -> usize {
    let target = (n_particles as f64).cbrt();
    // Round up to nearest power of 2
    let mut n_pm = 16_usize;
    while (n_pm as f64) < target {
        n_pm *= 2;
    }
    n_pm
}

/// TreePM configuration bundle for a specific simulation.
#[derive(Debug, Clone)]
pub struct TreePMParams {
    pub box_size: f64,
    pub n_pm: usize,
    pub r_s: f64,
    pub r_cut: f64,
    pub theta: f64,
    pub n_leaf: usize,
}

impl TreePMParams {
    /// Build canonical PhotoNs-GPU configuration for given box and N_pm.
    pub fn canonical(box_size: f64, n_pm: usize) -> Self {
        Self {
            box_size,
            n_pm,
            r_s: r_s_canonical(box_size, n_pm),
            r_cut: r_cut_canonical(box_size, n_pm),
            theta: THETA_DEFAULT,
            n_leaf: N_LEAF,
        }
    }

    /// Build with custom factors (for parameter sweep §4.2).
    pub fn with_factors(
        box_size: f64,
        n_pm: usize,
        rs_factor: f64,
        rcut_factor: f64,
        theta: f64,
    ) -> Self {
        let dg = delta_grid(box_size, n_pm);
        Self {
            box_size,
            n_pm,
            r_s: rs_factor * dg,
            r_cut: rcut_factor * dg,
            theta,
            n_leaf: N_LEAF,
        }
    }

    pub fn delta_grid(&self) -> f64 {
        delta_grid(self.box_size, self.n_pm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_grid_basic() {
        // Δg = box / N_pm
        assert_eq!(delta_grid(500.0, 256), 500.0 / 256.0);
        assert_eq!(delta_grid(250.0, 128), 250.0 / 128.0);
    }

    #[test]
    fn test_r_s_canonical() {
        let r_s = r_s_canonical(500.0, 256);
        let dg = 500.0 / 256.0;
        // Tolerance: 1e-15 (exact mul)
        assert!((r_s - 1.2 * dg).abs() < 1e-15);
    }

    #[test]
    fn test_r_cut_canonical() {
        let r_cut = r_cut_canonical(500.0, 256);
        let dg = 500.0 / 256.0;
        assert!((r_cut - 6.0 * dg).abs() < 1e-15);
    }

    #[test]
    fn test_r_cut_over_r_s_is_5() {
        // PhotoNs canonical: r_cut/r_s = 6/1.2 = 5
        let r_s = r_s_canonical(500.0, 256);
        let r_cut = r_cut_canonical(500.0, 256);
        let ratio = r_cut / r_s;
        assert!((ratio - 5.0).abs() < 1e-12, "r_cut/r_s = {}", ratio);
    }

    #[test]
    fn test_recommended_pm_grid_1m() {
        // N=1M, expected N_pm=128 (cbrt(1M)=100, round up to 128)
        let n_pm = recommended_pm_grid(1_000_000);
        assert_eq!(n_pm, 128);
    }

    #[test]
    fn test_recommended_pm_grid_10m() {
        // N=10M, cbrt=215.4, round up to 256
        let n_pm = recommended_pm_grid(10_000_000);
        assert_eq!(n_pm, 256);
    }

    #[test]
    fn test_recommended_pm_grid_100k() {
        // N=100K, cbrt=46.4, round up to 64
        let n_pm = recommended_pm_grid(100_000);
        assert_eq!(n_pm, 64);
    }

    #[test]
    fn test_canonical_params_relationships() {
        let p = TreePMParams::canonical(500.0, 256);
        assert_eq!(p.box_size, 500.0);
        assert_eq!(p.n_pm, 256);
        // Tolerance: 1e-15 (exact)
        assert!((p.r_s - 1.2 * (500.0 / 256.0)).abs() < 1e-15);
        assert!((p.r_cut - 6.0 * (500.0 / 256.0)).abs() < 1e-15);
        assert_eq!(p.theta, 0.5);
        assert_eq!(p.n_leaf, 10);
    }

    #[test]
    fn test_params_with_factors() {
        let p = TreePMParams::with_factors(500.0, 256, 1.5, 8.0, 0.7);
        let dg = 500.0 / 256.0;
        assert!((p.r_s - 1.5 * dg).abs() < 1e-15);
        assert!((p.r_cut - 8.0 * dg).abs() < 1e-15);
        assert_eq!(p.theta, 0.7);
    }

    #[test]
    fn test_canonical_at_typical_sizes() {
        // 1M, L=250 Mpc, N_pm=256: r_s ≈ 1.17 Mpc, r_cut ≈ 5.86 Mpc
        let p = TreePMParams::canonical(250.0, 256);
        let dg = 250.0 / 256.0; // ≈ 0.977
        assert!((p.r_s - 1.2 * dg).abs() < 1e-15);
        assert!((p.r_cut - 6.0 * dg).abs() < 1e-15);
        // Sanity: r_cut should be a few percent of box
        assert!(p.r_cut < 0.1 * p.box_size);
    }
}
