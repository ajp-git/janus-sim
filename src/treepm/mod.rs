//! TreePM Module — FFT long-range + Barnes-Hut short-range
//!
//! Architecture:
//! - PM long-range: Dual-grid FFT (ρ⁺ and ρ⁻ separately)
//! - Tree short-range: Barnes-Hut with r < r_cut cutoff
//!
//! CRITICAL (FIX-009): Must use TWO separate grids for Janus physics.
//! Single grid ρ = Σ(±m) is INVALID — negative mass ≠ repulsion in PM.
//!
//! IMPLEMENTATION NOTE: Initial version uses rustfft (CPU) for FFT.
//! GPU cuFFT optimization planned after architecture validation.

pub mod pm_grid;
pub mod splitting;
pub mod tree_short;

/// Splitting radius for TreePM
/// Forces at r > R_CUT are computed by PM (long-range)
/// Forces at r < R_CUT are computed by Tree (short-range)
pub const R_CUT_FACTOR: f64 = 16.0;  // r_cut = box_size / R_CUT_FACTOR

/// Grid size for PM FFT
pub const PM_GRID_SIZE: usize = 256;  // 256³ grid

/// Compute r_cut from box size
pub fn compute_r_cut(box_size: f64) -> f64 {
    box_size / R_CUT_FACTOR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_cut_computation() {
        let box_size = 736.8;  // Typical 40M box
        let r_cut = compute_r_cut(box_size);
        assert!((r_cut - 46.05).abs() < 0.1, "Expected r_cut ≈ 46 Mpc, got {}", r_cut);
    }
}
