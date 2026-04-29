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
use super::splitting::splitting_tree_springel;

/// **DEPRECATED** Polynomial PM weight (Bagla 2002), inconsistent with
/// Gaussian PM. Kept for backward compat. Use Springel via tree force computation.
pub fn splitting_pm_weight(r: f64, r_cut: f64) -> f64 {
    if r >= r_cut {
        1.0
    } else {
        let x = r / r_cut;
        let x2 = x * x;
        x2 * x2 // x^4 for smooth transition
    }
}

/// **DEPRECATED** Polynomial Tree weight (Bagla 2002).
pub fn splitting_tree_weight(r: f64, r_cut: f64) -> f64 {
    1.0 - splitting_pm_weight(r, r_cut)
}

/// Barnes-Hut tree with TreePM splitting (Phase 9.6: Springel convention).
///
/// Uses `splitting_tree_springel(r, r_s)` to apply T(x) = erfc(x) + (2x/√π)·exp(-x²)
/// for compatibility with PM Gaussian damping `exp(-k²·r_s²)`.
///
/// `r_cut` is the neighbor search radius (cells beyond this are pruned).
/// `r_s` is the split scale used for the Springel function.
/// Convention PhotoNs canonical: `r_cut = 5·r_s` (so cutoff at x = r_cut/(2·r_s) = 2.5,
/// where T(2.5) ≈ 6e-3, near zero). Springel hard cutoff inside the function at x ≥ 3.
pub struct TreePMTree {
    pub theta: f64,
    pub r_cut: f64,
    pub r_s: f64,         // Phase 9.6: split scale for Springel splitting
    pub g_constant: f64,
    pub root: OctreeNode,
    pub bounds: BoundingBox,
    /// Phase 10A1: box size for periodic minimum image convention.
    /// 0.0 means MIC disabled (legacy non-periodic).
    pub box_size: f64,
}

impl TreePMTree {
    /// Build tree from particles. Default r_s = r_cut / 5 (PhotoNs canonical),
    /// box_size = 0 (MIC disabled, legacy).
    pub fn build(particles: &[Particle], theta: f64, r_cut: f64) -> Self {
        Self::build_with_g(particles, theta, r_cut, 1.0)
    }

    /// Build tree with custom G constant. Default r_s = r_cut / 5, box_size=0.
    pub fn build_with_g(particles: &[Particle], theta: f64, r_cut: f64, g_constant: f64) -> Self {
        Self::build_with_rs_and_g(particles, theta, r_cut, r_cut / 5.0, g_constant)
    }

    /// Build tree with explicit r_s and G constant (Phase 9.6 API). MIC disabled.
    pub fn build_with_rs_and_g(
        particles: &[Particle],
        theta: f64,
        r_cut: f64,
        r_s: f64,
        g_constant: f64,
    ) -> Self {
        Self::build_with_rs_g_box(particles, theta, r_cut, r_s, g_constant, 0.0)
    }

    /// Build tree with full Phase 10A1 API including periodic box_size.
    /// box_size > 0 enables minimum image convention; 0 disables it.
    pub fn build_with_rs_g_box(
        particles: &[Particle],
        theta: f64,
        r_cut: f64,
        r_s: f64,
        g_constant: f64,
        box_size: f64,
    ) -> Self {
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

        Self { theta, r_cut, r_s, g_constant, root, bounds, box_size }
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
                Self::pairwise_acc_with_split(
                    pos, sign, p.pos, p.mass, p.sign, softening,
                    self.r_cut, self.r_s, self.g_constant, self.box_size,
                )
            }
            OctreeNode::Internal { children, com_plus, mass_plus, com_minus, mass_minus } => {
                // Distance to cell center
                let r_to_cell = (pos - bounds.center()).length();
                let cell_size = bounds.size();

                // If entire cell is beyond r_cut, skip (PM handles it)
                let min_dist_to_cell = (r_to_cell - cell_size * 0.866).max(0.0); // 0.866 ≈ sqrt(3)/2
                if min_dist_to_cell >= self.r_cut {
                    return Vec3::zero();
                }

                // θ criterion: use cell approximation if s/r < θ AND r is small enough
                if cell_size / r_to_cell < self.theta && r_to_cell < self.r_cut {
                    let acc_from_plus = Self::pairwise_acc_with_split(
                        pos, sign, *com_plus, *mass_plus, MassSign::Positive, softening,
                        self.r_cut, self.r_s, self.g_constant, self.box_size,
                    );
                    let acc_from_minus = Self::pairwise_acc_with_split(
                        pos, sign, *com_minus, *mass_minus, MassSign::Negative, softening,
                        self.r_cut, self.r_s, self.g_constant, self.box_size,
                    );
                    acc_from_plus + acc_from_minus
                } else {
                    // Too close or crossing r_cut boundary: recurse into children
                    let mut acc = Vec3::zero();
                    for (i, child) in children.iter().enumerate() {
                        acc += self.acc_recursive(
                            child, pos, sign, particles, &bounds.child_box(i),
                            softening, exclude_idx,
                        );
                    }
                    acc
                }
            }
        }
    }

    /// Pairwise acceleration with TreePM Springel splitting (Phase 9.6).
    ///
    /// `F_tree(r) = -(G·m·r̂/r²) × T(r/(2·r_s))`
    ///
    /// where `T(x) = erfc(x) + (2x/√π)·exp(-x²)` (Springel 2005).
    /// Compatible with PM Gaussian damping `exp(-k²·r_s²)`.
    ///
    /// Hard cutoff at `r >= r_cut` for neighbor pruning. Soft cutoff via
    /// `splitting_tree_springel` at `r ≥ 6·r_s` (where T ≈ 0).
    fn pairwise_acc_with_split(
        pos_i: Vec3,
        sign_i: MassSign,
        pos_j: Vec3,
        mass_j: f64,
        sign_j: MassSign,
        softening: f64,
        r_cut: f64,
        r_s: f64,
        g_constant: f64,
        box_size: f64,
    ) -> Vec3 {
        if mass_j == 0.0 {
            return Vec3::zero();
        }

        // Phase 10A1: minimum image convention (MIC) for periodic BC.
        // box_size = 0.0 disables MIC (legacy non-periodic).
        let mut dx = pos_j.x - pos_i.x;
        let mut dy = pos_j.y - pos_i.y;
        let mut dz = pos_j.z - pos_i.z;
        if box_size > 0.0 {
            let half = box_size * 0.5;
            if dx > half { dx -= box_size; }
            if dx < -half { dx += box_size; }
            if dy > half { dy -= box_size; }
            if dy < -half { dy += box_size; }
            if dz > half { dz -= box_size; }
            if dz < -half { dz += box_size; }
        }
        let r_vec = Vec3::new(dx, dy, dz);
        let r2 = r_vec.length_sq();
        let r = r2.sqrt();

        // Skip if beyond r_cut (PM handles this; Springel T ≈ 0 there too)
        if r >= r_cut {
            return Vec3::zero();
        }

        // Plummer softening
        let r2_soft = r2 + softening * softening;
        if r2_soft < 1e-20 {
            return Vec3::zero();
        }

        // Janus interaction sign
        let interaction = if sign_i == sign_j { 1.0 } else { -1.0 } * g_constant;

        // Full Newton force magnitude
        let inv_r3_soft = 1.0 / (r2_soft * r2_soft.sqrt());
        let acc_full = r_vec * (interaction * mass_j * inv_r3_soft);

        // Apply Springel Tree splitting (replaces polynomial 1 - (r/r_cut)^4)
        let tree_weight = splitting_tree_springel(r, r_s);
        acc_full * tree_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 10A1: vérifier que le Tree utilise MIC pour pairs cross-boundary.
    /// 2 particules à (40, 0, 0) et (-40, 0, 0) dans L=100. Distance MIC = 20.
    /// Tree avec MIC doit donner force vers +x sur particule 0 (image at +60).
    #[test]
    fn test_tree_pbc_minimum_image() {
        let l = 100.0_f64;
        let r_s = 1.0_f64;
        let r_cut = 25.0_f64; // > MIC dist 20, < L/2 = 50
        let particles = vec![
            Particle::new(Vec3::new(40.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
            Particle::new(Vec3::new(-40.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        ];
        let tree = TreePMTree::build_with_rs_g_box(&particles, 0.5, r_cut, r_s, 1.0, l);
        let acc = tree.compute_short_range_acc_excluding(
            Vec3::new(40.0, 0.0, 0.0),
            MassSign::Positive,
            &particles,
            0.05,
            Some(0),
        );

        // MIC: la particule à -40 vue à +60 via boundary x=±50.
        // Force sur particule 0 en (40, 0, 0) attractive vers +60 → +x.
        // Mais splitting Springel atténue: T(20/(2·1)) = T(10) ≈ 0.
        // Avec r_cut=25 et MIC dist=20 < r_cut, Tree compute la pair.
        // Le splitting T(10) est essentiellement 0 → force tree ≈ 0.
        // Test plus robuste: vérifier que Tree NE retourne PAS la force
        // raw (à direct distance 80 = -x direction).
        assert!(
            acc.x.abs() < 1e-3,
            "Tree should NOT compute force at raw direct dist 80; got acc.x = {}",
            acc.x
        );

        // Test 2: r_s = 5 (Springel cutoff = 6·r_s = 30 > MIC=20)
        let r_s2 = 5.0_f64;
        let tree2 = TreePMTree::build_with_rs_g_box(&particles, 0.5, r_cut, r_s2, 1.0, l);
        let acc2 = tree2.compute_short_range_acc_excluding(
            Vec3::new(40.0, 0.0, 0.0),
            MassSign::Positive,
            &particles,
            0.05,
            Some(0),
        );
        // Now T(20/(2·5)) = T(2) ≈ 0.046, force ≈ G·m/r² · T(2) at MIC r=20.
        // direction: vers +x (toward image at +60)
        assert!(
            acc2.x > 0.0,
            "MIC: Tree should give force toward +x via image (PBC), got acc.x = {}",
            acc2.x
        );
        let r_mic = 20.0;
        let expected_mag = (1.0 / (r_mic * r_mic)) * 0.046; // T(2) ≈ 0.046
        let mag = (acc2.x.powi(2) + acc2.y.powi(2) + acc2.z.powi(2)).sqrt();
        // Tolerance loose because Springel approx + softening
        assert!(
            (mag - expected_mag).abs() / expected_mag < 0.5,
            "Magnitude {} differs from expected {} (T(2)·1/r²)",
            mag,
            expected_mag
        );
    }

    /// Phase 9.6: vérifier que Tree utilise bien Springel splitting (pas polynomial).
    /// 2 particules à r = 2·r_s, mass=1, G=1, softening=0.
    /// F_tree = (1/r²) × T(1) avec T(1) ≈ 0.5724.
    /// → F_tree ≈ (1/4) × 0.5724 = 0.1431.
    #[test]
    fn test_tree_uses_springel_splitting() {
        let r_s = 1.0_f64;
        let r_cut = 5.0 * r_s; // PhotoNs canonical
        let particles = vec![
            Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
            Particle::new(Vec3::new(2.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        ];
        let tree = TreePMTree::build_with_rs_and_g(&particles, 0.5, r_cut, r_s, 1.0);
        let f = tree.compute_short_range_acc_excluding(
            Vec3::new(0.0, 0.0, 0.0),
            MassSign::Positive,
            &particles,
            0.0,
            Some(0),
        );

        // Expected: F = G·m·r̂/r² × T(r/(2·r_s)) = 1·1·(+1)/4 × T(1)
        // T(1) ≈ 0.5724; F_x ≈ +0.1431 (attractive toward +x)
        let expected = 0.1431_f64;
        // Tolerance 1e-3 (Abramowitz erfc approximation)
        assert!(
            (f.x - expected).abs() < 1e-3,
            "F_x = {}, expected {} (Springel T(1)≈0.5724)",
            f.x,
            expected
        );
        // y, z components should be 0
        assert!(f.y.abs() < 1e-12);
        assert!(f.z.abs() < 1e-12);
    }

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
