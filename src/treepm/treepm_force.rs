//! TreePM Force Calculator
//!
//! Combines PM long-range forces with Tree short-range corrections.
//!
//! ARCHITECTURE:
//! 1. PM computes long-range forces using Gaussian-damped Green's function
//! 2. Tree computes short-range correction (direct - PM_reconstructed)
//!
//! The splitting ensures: F_total = F_PM_long + F_Tree_short ≈ F_direct

use crate::MassSign;
use crate::nbody::{Vec3, Particle};
use super::pm_grid::PmGrid;
use super::tree_short::TreePMTree;

/// TreePM force calculator
pub struct TreePMForce {
    pub pm: PmGrid,
    pub tree: TreePMTree,
    pub r_cut: f64,
    pub r_s: f64,  // Gaussian splitting scale (typically r_cut / 3)
    pub softening: f64,
    pub g_constant: f64,
    pub pm_only: bool,  // Skip Tree short-range for fast visual validation
}

impl TreePMForce {
    /// Create new TreePM force calculator
    ///
    /// r_cut: splitting radius (Tree handles r < r_cut)
    /// grid_size: PM grid resolution
    /// box_size: simulation box size
    /// theta: Barnes-Hut opening angle
    /// softening: Plummer softening length
    pub fn new(r_cut: f64, grid_size: usize, box_size: f64, theta: f64, softening: f64) -> Self {
        // Phase 9.6: r_s = r_cut/5 (PhotoNs canonical, Springel-compatible).
        // Previously was r_cut/3 (mismatched with polynomial Tree splitting).
        let r_s = r_cut / 5.0;

        let g_constant = 1.0;

        Self {
            pm: PmGrid::new(grid_size, box_size),
            tree: TreePMTree::build_with_rs_and_g(&[], theta, r_cut, r_s, g_constant),
            r_cut,
            r_s,
            softening,
            g_constant,
            pm_only: false,
        }
    }

    /// Create PM-only force calculator (skip Tree for fast visual validation)
    pub fn new_pm_only(grid_size: usize, box_size: f64) -> Self {
        Self {
            pm: PmGrid::new(grid_size, box_size),
            tree: TreePMTree::build_with_g(&[], 0.5, 10.0, 1.0),  // Dummy tree
            r_cut: 0.0,
            r_s: 0.0,
            softening: 0.5,
            g_constant: 1.0,
            pm_only: true,
        }
    }

    /// Update force calculator with new particle positions
    /// Must be called each timestep before computing forces
    pub fn update(&mut self, particles: &[Particle]) {
        // Clear PM grids
        self.pm.clear();

        // Assign masses to PM grid
        for p in particles {
            let sign = match p.sign {
                MassSign::Positive => 1i8,
                MassSign::Negative => -1i8,
            };
            self.pm.assign_mass(p.pos.x, p.pos.y, p.pos.z, p.mass, sign);
        }

        // Solve Poisson (with splitting if using Tree, without if PM-only)
        if self.pm_only {
            self.pm.solve_poisson(self.g_constant);
        } else {
            self.pm.solve_poisson_with_splitting(self.g_constant, Some(self.r_s));
            // Rebuild Tree for short-range (with same G constant)
            self.tree = TreePMTree::build_with_rs_and_g(
                particles,
                self.tree.theta,
                self.r_cut,
                self.r_s,
                self.g_constant,
            );
        }
    }

    /// Compute total force on particle at position (pos, sign)
    /// Returns (Fx, Fy, Fz)
    pub fn compute_force(&self, pos: Vec3, sign: MassSign, particles: &[Particle]) -> Vec3 {
        self.compute_force_excluding(pos, sign, particles, None)
    }

    /// Compute force excluding a specific particle index (to avoid self-interaction)
    pub fn compute_force_excluding(&self, pos: Vec3, sign: MassSign, particles: &[Particle], exclude_idx: Option<usize>) -> Vec3 {
        let sign_i8 = match sign {
            MassSign::Positive => 1i8,
            MassSign::Negative => -1i8,
        };

        // PM force
        let (fx_pm, fy_pm, fz_pm) = self.pm.interpolate_force(pos.x, pos.y, pos.z, sign_i8);
        let f_pm = Vec3::new(fx_pm, fy_pm, fz_pm);

        // PM-only mode: skip Tree short-range
        if self.pm_only {
            return f_pm;
        }

        // Tree short-range force (with self-exclusion)
        let f_tree = self.tree.compute_short_range_acc_excluding(pos, sign, particles, self.softening, exclude_idx);

        // Total = PM_long + Tree_short
        Vec3::new(
            f_pm.x + f_tree.x,
            f_pm.y + f_tree.y,
            f_pm.z + f_tree.z,
        )
    }

    /// Compute forces on all particles (parallel, with self-exclusion)
    /// Returns vector of (Fx, Fy, Fz) for each particle
    pub fn compute_all_forces(&self, particles: &[Particle]) -> Vec<Vec3> {
        use rayon::prelude::*;

        particles.par_iter()
            .enumerate()
            .map(|(i, p)| self.compute_force_excluding(p.pos, p.sign, particles, Some(i)))
            .collect()
    }

    /// Compute forces on all particles (sequential, with self-exclusion)
    pub fn compute_all_forces_sequential(&self, particles: &[Particle]) -> Vec<Vec3> {
        particles.iter()
            .enumerate()
            .map(|(i, p)| self.compute_force_excluding(p.pos, p.sign, particles, Some(i)))
            .collect()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.pm.memory_bytes()
        // Tree memory is dynamic and harder to estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treepm_basic() {
        let r_cut = 20.0;
        let box_size = 100.0;
        let grid_size = 32;
        let theta = 0.5;
        let softening = 0.5;

        let mut treepm = TreePMForce::new(r_cut, grid_size, box_size, theta, softening);

        // Two positive particles
        let particles = vec![
            Particle::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
            Particle::new(Vec3::new(10.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        ];

        treepm.update(&particles);

        // Force on particle 0 (at -10, should be attracted toward +10)
        let f0 = treepm.compute_force(particles[0].pos, particles[0].sign, &particles);
        println!("\nForce on particle 0 at (-10,0,0): ({:.6}, {:.6}, {:.6})", f0.x, f0.y, f0.z);

        // Force should point toward +x (toward the other particle)
        assert!(f0.x > 0.0, "(+,+) should attract: fx should be > 0, got {}", f0.x);

        // Force on particle 1 (at +10, should be attracted toward -10)
        let f1 = treepm.compute_force(particles[1].pos, particles[1].sign, &particles);
        println!("Force on particle 1 at (+10,0,0): ({:.6}, {:.6}, {:.6})", f1.x, f1.y, f1.z);

        // Force should point toward -x
        assert!(f1.x < 0.0, "(+,+) should attract: fx should be < 0, got {}", f1.x);

        // Forces should be symmetric (Newton's 3rd law)
        assert!((f0.x + f1.x).abs() < 0.01 * f0.x.abs(),
                "Forces should be symmetric: f0.x={}, f1.x={}", f0.x, f1.x);

        println!("✓ TreePM basic test passed");
    }

    #[test]
    fn test_treepm_janus_repulsion() {
        let r_cut = 20.0;
        let box_size = 100.0;
        let grid_size = 32;

        let mut treepm = TreePMForce::new(r_cut, grid_size, box_size, 0.5, 0.5);

        // One positive, one negative
        let particles = vec![
            Particle::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
            Particle::new(Vec3::new(10.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
        ];

        treepm.update(&particles);

        // Force on positive particle (should be repelled by negative)
        let f0 = treepm.compute_force(particles[0].pos, particles[0].sign, &particles);
        println!("\n(+,-) Force on + at (-10,0,0): ({:.6}, {:.6}, {:.6})", f0.x, f0.y, f0.z);

        // Repulsion: force should point away from the negative particle (toward -x)
        assert!(f0.x < 0.0, "(+,-) should repel: fx on + should be < 0, got {}", f0.x);

        // Force on negative particle (should be repelled by positive)
        let f1 = treepm.compute_force(particles[1].pos, particles[1].sign, &particles);
        println!("(+,-) Force on - at (+10,0,0): ({:.6}, {:.6}, {:.6})", f1.x, f1.y, f1.z);

        // Repulsion: force should point away from the positive particle (toward +x)
        assert!(f1.x > 0.0, "(+,-) should repel: fx on - should be > 0, got {}", f1.x);

        println!("✓ TreePM Janus repulsion test passed");
    }

    #[test]
    fn test_treepm_all_four_signs() {
        let r_cut = 30.0;
        let box_size = 100.0;

        let mut treepm = TreePMForce::new(r_cut, 32, box_size, 0.5, 0.5);

        println!("\n=== TreePM All Four Sign Combinations ===");

        let test_cases = [
            (MassSign::Positive, MassSign::Positive, "attract", true),
            (MassSign::Negative, MassSign::Negative, "attract", true),
            (MassSign::Positive, MassSign::Negative, "repel", false),
            (MassSign::Negative, MassSign::Positive, "repel", false),
        ];

        for (sign_i, sign_j, expected, should_attract) in test_cases {
            let particles = vec![
                Particle::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::zero(), 1.0, sign_i),
                Particle::new(Vec3::new(10.0, 0.0, 0.0), Vec3::zero(), 1.0, sign_j),
            ];

            treepm.update(&particles);

            let f = treepm.compute_force(particles[0].pos, particles[0].sign, &particles);

            // If attract: f.x > 0 (toward +x where j is)
            // If repel: f.x < 0 (away from j)
            let correct = if should_attract { f.x > 0.0 } else { f.x < 0.0 };

            let sign_i_str = match sign_i { MassSign::Positive => "+", MassSign::Negative => "-" };
            let sign_j_str = match sign_j { MassSign::Positive => "+", MassSign::Negative => "-" };
            let status = if correct { "✓" } else { "✗" };

            println!("  {} ({},{}) → {}: fx = {:.6} {}",
                     status, sign_i_str, sign_j_str, expected, f.x,
                     if correct { "" } else { "WRONG!" });

            assert!(correct, "({},{}) should {}, got fx={}", sign_i_str, sign_j_str, expected, f.x);
        }

        println!("✓ All four sign combinations correct");
    }
}
