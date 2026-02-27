//! TreePM Short-Range Tree Forces
//!
//! Barnes-Hut tree with r_cut cutoff for TreePM splitting.
//! Only computes forces for r < r_cut, with smooth splitting function.
//!
//! ARCHITECTURE:
//! - For r < r_cut: Tree computes force with splitting weight (1 - W_pm(r))
//! - For r >= r_cut: Skip (handled by PM)
//! - W_pm(r) smoothly transitions from 0 to 1 as r approaches r_cut

use crate::MassSign;
use crate::nbody::{Vec3, Particle, BoundingBox, OctreeNode};

/// Splitting function: weight for PM (long-range) component
/// Uses smooth polynomial transition: W_pm = (r/r_cut)^4 for smooth derivative
/// Returns 0 at r=0, approaches 1 as r→r_cut, exactly 1 for r>=r_cut
pub fn splitting_pm_weight(r: f64, r_cut: f64) -> f64 {
    if r >= r_cut {
        1.0
    } else {
        let x = r / r_cut;
        let x2 = x * x;
        x2 * x2  // x^4 for smooth transition
    }
}

/// Splitting function: weight for Tree (short-range) component
/// Tree weight = 1 - PM weight
pub fn splitting_tree_weight(r: f64, r_cut: f64) -> f64 {
    1.0 - splitting_pm_weight(r, r_cut)
}

/// Barnes-Hut tree with TreePM splitting
pub struct TreePMTree {
    pub theta: f64,
    pub r_cut: f64,
    pub g_constant: f64,  // Gravitational constant
    pub root: OctreeNode,
    pub bounds: BoundingBox,
}

impl TreePMTree {
    /// Build tree from particles
    pub fn build(particles: &[Particle], theta: f64, r_cut: f64) -> Self {
        Self::build_with_g(particles, theta, r_cut, 1.0)
    }

    /// Build tree with custom G constant
    pub fn build_with_g(particles: &[Particle], theta: f64, r_cut: f64, g_constant: f64) -> Self {
        // Compute bounding box
        let mut min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec3::new(f64::MIN, f64::MIN, f64::MIN);

        for p in particles {
            min.x = min.x.min(p.pos.x);
            min.y = min.y.min(p.pos.y);
            min.z = min.z.min(p.pos.z);
            max.x = max.x.max(p.pos.x);
            max.y = max.y.max(p.pos.y);
            max.z = max.z.max(p.pos.z);
        }

        // Add small padding
        let pad = 1.0;
        min.x -= pad; min.y -= pad; min.z -= pad;
        max.x += pad; max.y += pad; max.z += pad;

        // Make box cubic (required for θ criterion)
        let size = (max.x - min.x).max(max.y - min.y).max(max.z - min.z);
        let center = Vec3::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        );
        let bounds = BoundingBox {
            min: Vec3::new(center.x - size/2.0, center.y - size/2.0, center.z - size/2.0),
            max: Vec3::new(center.x + size/2.0, center.y + size/2.0, center.z + size/2.0),
        };

        // Build tree
        let indices: Vec<usize> = (0..particles.len()).collect();
        let root = Self::build_node(&indices, particles, &bounds);

        Self { theta, r_cut, g_constant, root, bounds }
    }

    fn build_node(indices: &[usize], particles: &[Particle], bounds: &BoundingBox) -> OctreeNode {
        if indices.is_empty() {
            return OctreeNode::Empty;
        }

        if indices.len() == 1 {
            return OctreeNode::Leaf { particle_idx: indices[0] };
        }

        // Create 8 children
        let mut child_indices: [Vec<usize>; 8] = Default::default();
        let center = bounds.center();

        for &idx in indices {
            let p = &particles[idx].pos;
            let octant = ((p.x >= center.x) as usize)
                       | (((p.y >= center.y) as usize) << 1)
                       | (((p.z >= center.z) as usize) << 2);
            child_indices[octant].push(idx);
        }

        // Build children recursively
        let mut children_arr: [OctreeNode; 8] = Default::default();
        for i in 0..8 {
            children_arr[i] = Self::build_node(&child_indices[i], particles, &bounds.child_box(i));
        }
        let children = Box::new(children_arr);

        // Compute center of mass for positive and negative masses separately
        let (mut com_plus, mut mass_plus) = (Vec3::zero(), 0.0f64);
        let (mut com_minus, mut mass_minus) = (Vec3::zero(), 0.0f64);

        for &idx in indices {
            let p = &particles[idx];
            match p.sign {
                MassSign::Positive => {
                    com_plus.x += p.pos.x * p.mass;
                    com_plus.y += p.pos.y * p.mass;
                    com_plus.z += p.pos.z * p.mass;
                    mass_plus += p.mass;
                }
                MassSign::Negative => {
                    com_minus.x += p.pos.x * p.mass;
                    com_minus.y += p.pos.y * p.mass;
                    com_minus.z += p.pos.z * p.mass;
                    mass_minus += p.mass;
                }
            }
        }

        if mass_plus > 0.0 {
            com_plus.x /= mass_plus;
            com_plus.y /= mass_plus;
            com_plus.z /= mass_plus;
        }
        if mass_minus > 0.0 {
            com_minus.x /= mass_minus;
            com_minus.y /= mass_minus;
            com_minus.z /= mass_minus;
        }

        OctreeNode::Internal {
            children,
            com_plus,
            mass_plus,
            com_minus,
            mass_minus,
        }
    }

    /// Compute short-range acceleration with TreePM splitting
    /// Only includes forces for r < r_cut, weighted by splitting function
    pub fn compute_short_range_acc(&self, pos: Vec3, sign: MassSign, particles: &[Particle],
        softening: f64) -> Vec3
    {
        self.acc_recursive(&self.root, pos, sign, particles, &self.bounds, softening, None)
    }

    /// Compute short-range acceleration excluding a specific particle (for self-exclusion)
    pub fn compute_short_range_acc_excluding(&self, pos: Vec3, sign: MassSign, particles: &[Particle],
        softening: f64, exclude_idx: Option<usize>) -> Vec3
    {
        self.acc_recursive(&self.root, pos, sign, particles, &self.bounds, softening, exclude_idx)
    }

    fn acc_recursive(&self, node: &OctreeNode, pos: Vec3, sign: MassSign,
        particles: &[Particle], bounds: &BoundingBox, softening: f64, exclude_idx: Option<usize>) -> Vec3
    {
        match node {
            OctreeNode::Empty => Vec3::zero(),
            OctreeNode::Leaf { particle_idx } => {
                // Skip self-interaction
                if Some(*particle_idx) == exclude_idx {
                    return Vec3::zero();
                }
                let p = &particles[*particle_idx];
                Self::pairwise_acc_with_split(pos, sign, p.pos, p.mass, p.sign, softening, self.r_cut, self.g_constant)
            }
            OctreeNode::Internal { children, com_plus, mass_plus, com_minus, mass_minus } => {
                // Distance to cell center
                let r_to_cell = (pos - bounds.center()).length();
                let cell_size = bounds.size();

                // If entire cell is beyond r_cut, skip (PM handles it)
                let min_dist_to_cell = (r_to_cell - cell_size * 0.866).max(0.0);  // 0.866 ≈ sqrt(3)/2
                if min_dist_to_cell >= self.r_cut {
                    return Vec3::zero();
                }

                // θ criterion: use cell approximation if s/r < θ AND r is small enough
                if cell_size / r_to_cell < self.theta && r_to_cell < self.r_cut {
                    // Far enough for approximation, but still within r_cut
                    let acc_from_plus = Self::pairwise_acc_with_split(
                        pos, sign, *com_plus, *mass_plus, MassSign::Positive, softening, self.r_cut, self.g_constant);
                    let acc_from_minus = Self::pairwise_acc_with_split(
                        pos, sign, *com_minus, *mass_minus, MassSign::Negative, softening, self.r_cut, self.g_constant);
                    acc_from_plus + acc_from_minus
                } else {
                    // Too close or crossing r_cut boundary: recurse into children
                    let mut acc = Vec3::zero();
                    for (i, child) in children.iter().enumerate() {
                        acc += self.acc_recursive(child, pos, sign, particles,
                            &bounds.child_box(i), softening, exclude_idx);
                    }
                    acc
                }
            }
        }
    }

    /// Pairwise acceleration with TreePM splitting
    /// F_tree = F_full * (1 - W_pm(r))
    fn pairwise_acc_with_split(pos_i: Vec3, sign_i: MassSign, pos_j: Vec3, mass_j: f64,
        sign_j: MassSign, softening: f64, r_cut: f64, g_constant: f64) -> Vec3
    {
        if mass_j == 0.0 { return Vec3::zero(); }

        let r_vec = pos_j - pos_i;
        let r2 = r_vec.length_sq();
        let r = r2.sqrt();

        // Skip if beyond r_cut (PM handles this)
        if r >= r_cut {
            return Vec3::zero();
        }

        // Plummer softening
        let r2_soft = r2 + softening * softening;
        if r2_soft < 1e-20 { return Vec3::zero(); }

        // Janus interaction
        let interaction = if sign_i == sign_j { 1.0 } else { -1.0 } * g_constant;

        // Full force magnitude
        let inv_r3_soft = 1.0 / (r2_soft * r2_soft.sqrt());
        let acc_full = r_vec * (interaction * mass_j * inv_r3_soft);

        // Apply Tree splitting weight (1 - W_pm)
        let tree_weight = splitting_tree_weight(r, r_cut);
        acc_full * tree_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitting_weights() {
        let r_cut = 10.0;

        // At r=0: Tree=1, PM=0
        assert!((splitting_tree_weight(0.0, r_cut) - 1.0).abs() < 1e-10);
        assert!(splitting_pm_weight(0.0, r_cut).abs() < 1e-10);

        // At r=r_cut: Tree=0, PM=1
        assert!(splitting_tree_weight(r_cut, r_cut).abs() < 1e-10);
        assert!((splitting_pm_weight(r_cut, r_cut) - 1.0).abs() < 1e-10);

        // Beyond r_cut: Tree=0, PM=1
        assert!(splitting_tree_weight(r_cut * 2.0, r_cut).abs() < 1e-10);
        assert!((splitting_pm_weight(r_cut * 2.0, r_cut) - 1.0).abs() < 1e-10);

        // Midpoint: weights sum to 1
        let r_mid = r_cut / 2.0;
        let sum = splitting_tree_weight(r_mid, r_cut) + splitting_pm_weight(r_mid, r_cut);
        assert!((sum - 1.0).abs() < 1e-10, "Weights should sum to 1");

        println!("Splitting weights at r/r_cut:");
        for i in 0..=10 {
            let r = r_cut * i as f64 / 10.0;
            println!("  r/r_cut={:.1}: Tree={:.4}, PM={:.4}",
                     i as f64 / 10.0,
                     splitting_tree_weight(r, r_cut),
                     splitting_pm_weight(r, r_cut));
        }
    }

    #[test]
    fn test_short_range_cutoff() {
        // Single positive particle at origin
        let particles = vec![
            Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        ];

        let r_cut = 10.0;
        let tree = TreePMTree::build(&particles, 0.5, r_cut);

        // Test particle at r=3 (within r_cut)
        let pos_near = Vec3::new(3.0, 0.0, 0.0);
        let acc_near = tree.compute_short_range_acc(pos_near, MassSign::Positive, &particles, 0.1);

        // At r=3, Tree weight = 1 - (0.3)^4 ≈ 0.992
        // Should have significant force pointing toward origin (negative x)
        println!("Force at r=3: ({:.6}, {:.6}, {:.6})", acc_near.x, acc_near.y, acc_near.z);
        assert!(acc_near.x < -0.01, "Should have attractive force at short range: {}", acc_near.x);

        // Test particle at r=8 (within r_cut but large tree weight reduction)
        let pos_mid = Vec3::new(8.0, 0.0, 0.0);
        let acc_mid = tree.compute_short_range_acc(pos_mid, MassSign::Positive, &particles, 0.1);
        // At r=8, Tree weight = 1 - (0.8)^4 ≈ 0.59
        println!("Force at r=8: ({:.6}, {:.6}, {:.6})", acc_mid.x, acc_mid.y, acc_mid.z);
        assert!(acc_mid.x < 0.0, "Should have attractive force: {}", acc_mid.x);

        // Test particle beyond r_cut
        let pos_far = Vec3::new(15.0, 0.0, 0.0);
        let acc_far = tree.compute_short_range_acc(pos_far, MassSign::Positive, &particles, 0.1);

        // Beyond r_cut, Tree should return zero
        println!("Force at r=15 (beyond r_cut=10): ({:.6}, {:.6}, {:.6})", acc_far.x, acc_far.y, acc_far.z);
        assert!(acc_far.x.abs() < 1e-10, "Should have no Tree force beyond r_cut: {}", acc_far.x);
    }

    #[test]
    fn test_janus_signs_short_range() {
        let r_cut = 20.0;

        // (+,+) attraction
        {
            let particles = vec![
                Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
            ];
            let tree = TreePMTree::build(&particles, 0.5, r_cut);
            let acc = tree.compute_short_range_acc(Vec3::new(5.0, 0.0, 0.0), MassSign::Positive, &particles, 0.1);
            assert!(acc.x < 0.0, "(+,+) should attract: acc.x = {}", acc.x);
        }

        // (-,-) attraction
        {
            let particles = vec![
                Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
            ];
            let tree = TreePMTree::build(&particles, 0.5, r_cut);
            let acc = tree.compute_short_range_acc(Vec3::new(5.0, 0.0, 0.0), MassSign::Negative, &particles, 0.1);
            assert!(acc.x < 0.0, "(-,-) should attract: acc.x = {}", acc.x);
        }

        // (+,-) repulsion
        {
            let particles = vec![
                Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
            ];
            let tree = TreePMTree::build(&particles, 0.5, r_cut);
            let acc = tree.compute_short_range_acc(Vec3::new(5.0, 0.0, 0.0), MassSign::Positive, &particles, 0.1);
            assert!(acc.x > 0.0, "(+,-) should repel: acc.x = {}", acc.x);
        }

        println!("✓ All Janus sign combinations correct in short-range tree");
    }
}
