/// N-body simulation with Janus interaction rules
///
/// Implements gravitational dynamics for positive and negative mass particles
/// following the Janus model rules:
///   - Same sign masses attract (Newton)
///   - Opposite sign masses repel (anti-Newton)
///
/// Uses Barnes-Hut tree algorithm for O(N log N) complexity
/// and Leapfrog integrator for symplectic time evolution.

use crate::MassSign;
use rayon::prelude::*;

/// 3D vector operations
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self { x: self.x / len, y: self.y / len, z: self.z / len }
        } else {
            Self::zero()
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

/// Single particle in the N-body simulation
#[derive(Debug, Clone)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub mass: f64,
    pub sign: MassSign,
}

impl Particle {
    pub fn new(pos: Vec3, vel: Vec3, mass: f64, sign: MassSign) -> Self {
        Self { pos, vel, mass, sign }
    }
}

/// Axis-aligned bounding box for Barnes-Hut tree
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        Vec3 {
            x: (self.min.x + self.max.x) * 0.5,
            y: (self.min.y + self.max.y) * 0.5,
            z: (self.min.z + self.max.z) * 0.5,
        }
    }

    pub fn size(&self) -> f64 {
        let dx = self.max.x - self.min.x;
        let dy = self.max.y - self.min.y;
        let dz = self.max.z - self.min.z;
        dx.max(dy).max(dz)
    }

    pub fn contains(&self, p: &Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
    }

    /// Get octant index for position (0-7)
    pub fn octant(&self, p: &Vec3) -> usize {
        let c = self.center();
        let mut idx = 0;
        if p.x > c.x { idx |= 1; }
        if p.y > c.y { idx |= 2; }
        if p.z > c.z { idx |= 4; }
        idx
    }

    /// Get sub-box for octant
    pub fn child_box(&self, octant: usize) -> Self {
        let c = self.center();
        let min = Vec3 {
            x: if octant & 1 != 0 { c.x } else { self.min.x },
            y: if octant & 2 != 0 { c.y } else { self.min.y },
            z: if octant & 4 != 0 { c.z } else { self.min.z },
        };
        let max = Vec3 {
            x: if octant & 1 != 0 { self.max.x } else { c.x },
            y: if octant & 2 != 0 { self.max.y } else { c.y },
            z: if octant & 4 != 0 { self.max.z } else { c.z },
        };
        BoundingBox { min, max }
    }
}

/// Barnes-Hut octree node
#[derive(Debug)]
pub enum OctreeNode {
    Empty,
    Leaf {
        particle_idx: usize,
    },
    Internal {
        children: Box<[OctreeNode; 8]>,
        // Center of mass for positive masses
        com_plus: Vec3,
        mass_plus: f64,
        // Center of mass for negative masses
        com_minus: Vec3,
        mass_minus: f64,
    },
}

impl Default for OctreeNode {
    fn default() -> Self {
        OctreeNode::Empty
    }
}

/// Barnes-Hut octree for N-body gravity
pub struct Octree {
    pub root: OctreeNode,
    pub bounds: BoundingBox,
    pub theta: f64,  // Opening angle parameter (typically 0.5-1.0)
}

impl Octree {
    /// Build octree from particles
    pub fn build(particles: &[Particle], bounds: BoundingBox, theta: f64) -> Self {
        let mut root = OctreeNode::Empty;

        for (idx, particle) in particles.iter().enumerate() {
            if bounds.contains(&particle.pos) {
                Self::insert(&mut root, idx, particles, &bounds);
            }
        }

        // Compute centers of mass
        Self::compute_com(&mut root, particles);

        Self { root, bounds, theta }
    }

    fn insert(node: &mut OctreeNode, idx: usize, particles: &[Particle], bounds: &BoundingBox) {
        match node {
            OctreeNode::Empty => {
                *node = OctreeNode::Leaf { particle_idx: idx };
            }
            OctreeNode::Leaf { particle_idx: existing_idx } => {
                let existing = *existing_idx;
                let mut children: Box<[OctreeNode; 8]> = Box::new(Default::default());

                // Reinsert existing particle
                let oct_existing = bounds.octant(&particles[existing].pos);
                Self::insert(&mut children[oct_existing], existing, particles,
                    &bounds.child_box(oct_existing));

                // Insert new particle
                let oct_new = bounds.octant(&particles[idx].pos);
                Self::insert(&mut children[oct_new], idx, particles,
                    &bounds.child_box(oct_new));

                *node = OctreeNode::Internal {
                    children,
                    com_plus: Vec3::zero(),
                    mass_plus: 0.0,
                    com_minus: Vec3::zero(),
                    mass_minus: 0.0,
                };
            }
            OctreeNode::Internal { children, .. } => {
                let oct = bounds.octant(&particles[idx].pos);
                Self::insert(&mut children[oct], idx, particles, &bounds.child_box(oct));
            }
        }
    }

    fn compute_com(node: &mut OctreeNode, particles: &[Particle]) -> (Vec3, f64, Vec3, f64) {
        match node {
            OctreeNode::Empty => (Vec3::zero(), 0.0, Vec3::zero(), 0.0),
            OctreeNode::Leaf { particle_idx } => {
                let p = &particles[*particle_idx];
                match p.sign {
                    MassSign::Positive => (p.pos * p.mass, p.mass, Vec3::zero(), 0.0),
                    MassSign::Negative => (Vec3::zero(), 0.0, p.pos * p.mass, p.mass),
                }
            }
            OctreeNode::Internal { children, com_plus, mass_plus, com_minus, mass_minus } => {
                let mut total_com_plus = Vec3::zero();
                let mut total_mass_plus = 0.0;
                let mut total_com_minus = Vec3::zero();
                let mut total_mass_minus = 0.0;

                for child in children.iter_mut() {
                    let (cp, mp, cm, mm) = Self::compute_com(child, particles);
                    total_com_plus += cp;
                    total_mass_plus += mp;
                    total_com_minus += cm;
                    total_mass_minus += mm;
                }

                *com_plus = if total_mass_plus > 0.0 {
                    total_com_plus * (1.0 / total_mass_plus)
                } else { Vec3::zero() };
                *mass_plus = total_mass_plus;

                *com_minus = if total_mass_minus > 0.0 {
                    total_com_minus * (1.0 / total_mass_minus)
                } else { Vec3::zero() };
                *mass_minus = total_mass_minus;

                (total_com_plus, total_mass_plus, total_com_minus, total_mass_minus)
            }
        }
    }

    /// Compute acceleration on particle at position with given sign
    pub fn compute_acceleration(&self, pos: Vec3, sign: MassSign, particles: &[Particle],
        softening: f64) -> Vec3
    {
        self.acc_recursive(&self.root, pos, sign, particles, &self.bounds, softening)
    }

    fn acc_recursive(&self, node: &OctreeNode, pos: Vec3, sign: MassSign,
        particles: &[Particle], bounds: &BoundingBox, softening: f64) -> Vec3
    {
        match node {
            OctreeNode::Empty => Vec3::zero(),
            OctreeNode::Leaf { particle_idx } => {
                let p = &particles[*particle_idx];
                Self::pairwise_acc(pos, sign, p.pos, p.mass, p.sign, softening)
            }
            OctreeNode::Internal { children, com_plus, mass_plus, com_minus, mass_minus } => {
                let r = (pos - bounds.center()).length();
                let s = bounds.size();

                if s / r < self.theta {
                    // Far enough: use center of mass approximation
                    let acc_from_plus = Self::pairwise_acc(pos, sign, *com_plus, *mass_plus,
                        MassSign::Positive, softening);
                    let acc_from_minus = Self::pairwise_acc(pos, sign, *com_minus, *mass_minus,
                        MassSign::Negative, softening);
                    acc_from_plus + acc_from_minus
                } else {
                    // Too close: recurse into children
                    let mut acc = Vec3::zero();
                    for (i, child) in children.iter().enumerate() {
                        acc += self.acc_recursive(child, pos, sign, particles,
                            &bounds.child_box(i), softening);
                    }
                    acc
                }
            }
        }
    }

    /// Pairwise acceleration following Janus rules with Plummer softening
    ///
    /// Uses the correct Plummer softening formula:
    ///   a⃗ = G·m·r⃗ / (r² + ε²)^(3/2)
    ///
    /// This prevents singularities at r→0 and ensures energy conservation.
    fn pairwise_acc(pos_i: Vec3, sign_i: MassSign, pos_j: Vec3, mass_j: f64,
        sign_j: MassSign, softening: f64) -> Vec3
    {
        if mass_j == 0.0 { return Vec3::zero(); }

        let r_vec = pos_j - pos_i;
        let r2 = r_vec.length_sq();

        // Plummer softening: r² → r² + ε²
        let r2_soft = r2 + softening * softening;

        // Minimum distance check (shouldn't happen with proper softening)
        if r2_soft < 1e-20 { return Vec3::zero(); }

        // Janus interaction: same sign attract, opposite repel
        let interaction = if sign_i == sign_j { 1.0 } else { -1.0 };

        // Plummer force: F = G·m₁·m₂·r⃗ / (r² + ε²)^(3/2)
        // Acceleration: a⃗ = G·m_j·r⃗ / (r² + ε²)^(3/2)
        // G = 1 in simulation units
        let inv_r3_soft = 1.0 / (r2_soft * r2_soft.sqrt());  // 1/(r² + ε²)^(3/2)

        r_vec * (interaction * mass_j * inv_r3_soft)
    }
}

/// N-body simulation state
pub struct NBodySimulation {
    pub particles: Vec<Particle>,
    pub time: f64,
    pub softening: f64,
    pub theta: f64,
    pub box_size: f64,
}

impl NBodySimulation {
    /// Create new simulation with given parameters
    pub fn new(n_positive: usize, n_negative: usize, box_size: f64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);

        let mut particles = Vec::with_capacity(n_positive + n_negative);

        // Virial velocity scale: v ~ sqrt(G*M/R) = sqrt(N/L) for G=m=1
        let n_total_f = (n_positive + n_negative) as f64;
        let virial_velocity = (n_total_f / box_size).sqrt() * 0.3; // 30% of virial

        // Generate positive mass particles
        for _ in 0..n_positive {
            let pos = Vec3::new(
                (rng.random::<f64>() - 0.5) * box_size,
                (rng.random::<f64>() - 0.5) * box_size,
                (rng.random::<f64>() - 0.5) * box_size,
            );
            let vel = Vec3::new(
                (rng.random::<f64>() - 0.5) * virial_velocity,
                (rng.random::<f64>() - 0.5) * virial_velocity,
                (rng.random::<f64>() - 0.5) * virial_velocity,
            );
            particles.push(Particle::new(pos, vel, 1.0, MassSign::Positive));
        }

        // Generate negative mass particles
        for _ in 0..n_negative {
            let pos = Vec3::new(
                (rng.random::<f64>() - 0.5) * box_size,
                (rng.random::<f64>() - 0.5) * box_size,
                (rng.random::<f64>() - 0.5) * box_size,
            );
            let vel = Vec3::new(
                (rng.random::<f64>() - 0.5) * virial_velocity,
                (rng.random::<f64>() - 0.5) * virial_velocity,
                (rng.random::<f64>() - 0.5) * virial_velocity,
            );
            particles.push(Particle::new(pos, vel, 1.0, MassSign::Negative));
        }

        // Softening: ε ~ 0.5 * L / N^(1/3)
        // Balance between stability and allowing structure formation
        let n_total = (n_positive + n_negative) as f64;
        let mean_sep = box_size / n_total.powf(1.0/3.0);
        let softening = 0.5 * mean_sep;

        Self {
            particles,
            time: 0.0,
            softening,
            theta: 0.7,
            box_size,
        }
    }

    /// Single Leapfrog integration step
    pub fn step(&mut self, dt: f64) {
        let half_dt = dt * 0.5;
        let half_box = self.box_size / 2.0;
        let bounds = BoundingBox::new(
            Vec3::new(-half_box, -half_box, -half_box),
            Vec3::new(half_box, half_box, half_box),
        );

        // Build tree
        let tree = Octree::build(&self.particles, bounds, self.theta);

        // Compute accelerations and update velocities (half step)
        let accelerations: Vec<Vec3> = self.particles.par_iter()
            .map(|p| tree.compute_acceleration(p.pos, p.sign, &self.particles, self.softening))
            .collect();

        // v(t + dt/2) = v(t) + a(t) * dt/2
        for (p, acc) in self.particles.iter_mut().zip(accelerations.iter()) {
            p.vel += *acc * half_dt;
        }

        // x(t + dt) = x(t) + v(t + dt/2) * dt
        for p in self.particles.iter_mut() {
            p.pos += p.vel * dt;

            // Periodic boundary conditions
            if p.pos.x > half_box { p.pos.x -= self.box_size; }
            if p.pos.x < -half_box { p.pos.x += self.box_size; }
            if p.pos.y > half_box { p.pos.y -= self.box_size; }
            if p.pos.y < -half_box { p.pos.y += self.box_size; }
            if p.pos.z > half_box { p.pos.z -= self.box_size; }
            if p.pos.z < -half_box { p.pos.z += self.box_size; }
        }

        // Rebuild tree with new positions
        let tree = Octree::build(&self.particles, bounds, self.theta);

        // Compute new accelerations
        let accelerations: Vec<Vec3> = self.particles.par_iter()
            .map(|p| tree.compute_acceleration(p.pos, p.sign, &self.particles, self.softening))
            .collect();

        // v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        for (p, acc) in self.particles.iter_mut().zip(accelerations.iter()) {
            p.vel += *acc * half_dt;
        }

        self.time += dt;
    }

    /// Compute kinetic energy
    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter()
            .map(|p| 0.5 * p.mass * p.vel.length_sq())
            .sum()
    }

    /// Compute potential energy (expensive O(N²) - use sparingly)
    /// Uses Plummer softening consistent with force calculation
    pub fn potential_energy(&self) -> f64 {
        let eps2 = self.softening * self.softening;
        let mut pe = 0.0;

        for i in 0..self.particles.len() {
            for j in (i+1)..self.particles.len() {
                let pi = &self.particles[i];
                let pj = &self.particles[j];

                let r_vec = pj.pos - pi.pos;
                let r2 = r_vec.length_sq();
                let r_soft = (r2 + eps2).sqrt();

                // Janus: same sign → attraction (negative PE), opposite → repulsion (positive PE)
                let interaction = if pi.sign == pj.sign { -1.0 } else { 1.0 };

                // Plummer potential: φ = -G·m / sqrt(r² + ε²)
                pe += interaction * pi.mass * pj.mass / r_soft;
            }
        }

        pe
    }

    /// Compute binding potential energy (same-sign pairs only)
    /// This is always negative and represents the gravitationally bound energy.
    /// For Janus virialization, we use this instead of total PE because
    /// the +/- repulsive energy is not part of the bound system.
    pub fn potential_energy_binding(&self) -> f64 {
        let eps2 = self.softening * self.softening;
        let mut pe_bind = 0.0;

        for i in 0..self.particles.len() {
            for j in (i+1)..self.particles.len() {
                let pi = &self.particles[i];
                let pj = &self.particles[j];

                // Only same-sign pairs (attractive, bound)
                if pi.sign != pj.sign {
                    continue;
                }

                let r_vec = pj.pos - pi.pos;
                let r2 = r_vec.length_sq();
                let r_soft = (r2 + eps2).sqrt();

                // Same sign → attraction → negative PE
                pe_bind -= pi.mass * pj.mass / r_soft;
            }
        }

        pe_bind
    }

    /// Total energy (KE + PE)
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Virialize velocities to satisfy 2KE + PE_binding = 0
    ///
    /// For Janus cosmology, we virialize against the BINDING energy only
    /// (same-sign attractive pairs). The +/- repulsive interactions are
    /// not part of the bound system and will drive segregation during evolution.
    ///
    /// This ensures the +/+ and -/- clusters start in virial equilibrium.
    /// Call this once at t=0 after initialization.
    pub fn virialize(&mut self) {
        let ke = self.kinetic_energy();
        let pe_total = self.potential_energy();
        let pe_bind = self.potential_energy_binding();

        println!("Virialization (Janus mode - binding energy only):");
        println!("  KE initial    = {:.4e}", ke);
        println!("  PE total      = {:.4e} (includes +/- repulsion)", pe_total);
        println!("  PE binding    = {:.4e} (same-sign pairs only)", pe_bind);

        if ke < 1e-20 {
            println!("  WARNING: KE too small, skipping virialization");
            return;
        }

        if pe_bind >= 0.0 {
            println!("  WARNING: PE_binding >= 0, no bound system to virialize");
            return;
        }

        // Virial condition for bound part: 2KE + PE_bind = 0
        // → KE_target = |PE_bind|/2
        // → alpha = sqrt(KE_target / KE)
        let ke_target = pe_bind.abs() / 2.0;
        let alpha = (ke_target / ke).sqrt();

        println!("  KE target     = {:.4e}", ke_target);
        println!("  Alpha scale   = {:.6}", alpha);

        // Scale velocities
        for p in self.particles.iter_mut() {
            p.vel.x *= alpha;
            p.vel.y *= alpha;
            p.vel.z *= alpha;
        }

        // Verify
        let ke_after = self.kinetic_energy();
        let virial_ratio = 2.0 * ke_after + pe_bind;
        let virial_error = virial_ratio.abs() / pe_bind.abs();

        println!("  KE after      = {:.4e}", ke_after);
        println!("  2KE + PE_bind = {:.4e} (doit être ≈ 0)", virial_ratio);
        println!("  Virial error  = {:.4}%", virial_error * 100.0);

        if virial_error > 0.01 {
            println!("  WARNING: Virial error > 1%");
        }
    }

    /// Compute center of mass for each sign using minimum image convention
    /// to handle periodic boundary conditions correctly.
    /// Uses a SINGLE reference point for both populations to avoid bias.
    pub fn centers_of_mass(&self) -> (Vec3, Vec3) {
        let l = self.box_size;

        // Use the SAME reference for both populations
        // This is critical: using different references biases the result
        // Use first particle of any sign as common reference
        let ref_pos = if !self.particles.is_empty() {
            self.particles[0].pos
        } else {
            return (Vec3::zero(), Vec3::zero());
        };

        let mut sum_plus = Vec3::zero();
        let mut n_plus = 0usize;
        let mut sum_minus = Vec3::zero();
        let mut n_minus = 0usize;

        for p in &self.particles {
            // Add position relative to common reference using minimum image
            let unwrapped = Vec3::new(
                ref_pos.x + minimum_image(p.pos.x - ref_pos.x, l),
                ref_pos.y + minimum_image(p.pos.y - ref_pos.y, l),
                ref_pos.z + minimum_image(p.pos.z - ref_pos.z, l),
            );

            match p.sign {
                MassSign::Positive => {
                    sum_plus += unwrapped;
                    n_plus += 1;
                }
                MassSign::Negative => {
                    sum_minus += unwrapped;
                    n_minus += 1;
                }
            }
        }

        let com_plus = if n_plus > 0 {
            sum_plus * (1.0 / n_plus as f64)
        } else {
            Vec3::zero()
        };

        let com_minus = if n_minus > 0 {
            sum_minus * (1.0 / n_minus as f64)
        } else {
            Vec3::zero()
        };

        (com_plus, com_minus)
    }

    /// Compute segregation distance using minimum image convention,
    /// normalized by box_size for comparability across scales.
    pub fn segregation_distance(&self) -> f64 {
        let (com_plus, com_minus) = self.centers_of_mass();
        let l = self.box_size;

        // Distance with minimum image convention
        let dx = minimum_image(com_plus.x - com_minus.x, l);
        let dy = minimum_image(com_plus.y - com_minus.y, l);
        let dz = minimum_image(com_plus.z - com_minus.z, l);

        // Normalize by box_size
        (dx * dx + dy * dy + dz * dz).sqrt() / l
    }
}

/// Minimum image convention for periodic boundaries
fn minimum_image(dx: f64, box_size: f64) -> f64 {
    let mut d = dx;
    while d > box_size / 2.0 { d -= box_size; }
    while d < -box_size / 2.0 { d += box_size; }
    d
}

/// Apply periodic boundary conditions to a position
fn apply_periodic_bc(pos: &mut Vec3, box_half: f64, box_size: f64) {
    if pos.x > box_half { pos.x -= box_size; }
    if pos.x < -box_half { pos.x += box_size; }
    if pos.y > box_half { pos.y -= box_size; }
    if pos.y < -box_half { pos.y += box_size; }
    if pos.z > box_half { pos.z -= box_size; }
    if pos.z < -box_half { pos.z += box_size; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_janus_attraction() {
        // Two positive masses should attract
        let acc = Octree::pairwise_acc(
            Vec3::new(0.0, 0.0, 0.0), MassSign::Positive,
            Vec3::new(1.0, 0.0, 0.0), 1.0, MassSign::Positive, 0.01  // Small softening
        );
        assert!(acc.x > 0.0, "Positive masses should attract: acc.x = {}", acc.x);
    }

    #[test]
    fn test_janus_repulsion() {
        // Opposite signs should repel
        let acc = Octree::pairwise_acc(
            Vec3::new(0.0, 0.0, 0.0), MassSign::Positive,
            Vec3::new(1.0, 0.0, 0.0), 1.0, MassSign::Negative, 0.01
        );
        assert!(acc.x < 0.0, "Opposite masses should repel: acc.x = {}", acc.x);
    }

    #[test]
    fn test_plummer_softening() {
        // At distance r=1 with ε=0, force should be F = m/r² = 1
        let acc_no_soft = Octree::pairwise_acc(
            Vec3::new(0.0, 0.0, 0.0), MassSign::Positive,
            Vec3::new(1.0, 0.0, 0.0), 1.0, MassSign::Positive, 0.001
        );

        // With large softening, force should be smaller
        let acc_soft = Octree::pairwise_acc(
            Vec3::new(0.0, 0.0, 0.0), MassSign::Positive,
            Vec3::new(1.0, 0.0, 0.0), 1.0, MassSign::Positive, 1.0
        );

        assert!(acc_soft.x < acc_no_soft.x,
            "Softening should reduce force: {} vs {}", acc_soft.x, acc_no_soft.x);
    }

    #[test]
    fn test_force_at_close_range() {
        // With proper softening, force should not blow up at small r
        let acc = Octree::pairwise_acc(
            Vec3::new(0.0, 0.0, 0.0), MassSign::Positive,
            Vec3::new(0.001, 0.0, 0.0), 1.0, MassSign::Positive, 0.1
        );

        // With ε=0.1, at r=0.001: F ~ m/(r² + ε²)^(3/2) ~ 1/(0.01)^(3/2) ~ 1000
        assert!(acc.x < 10000.0, "Softening should prevent force explosion: {}", acc.x);
        assert!(acc.x > 0.0, "Force should still be attractive: {}", acc.x);
    }

    // =========================================================================
    // VALIDATION_RULES.md Section 1 — SÉGRÉGATION
    // =========================================================================

    #[test]
    fn test_minimum_image() {
        let box_size = 100.0;
        // Two particles "close" across the periodic boundary
        // dx = 49 - (-49) = 98, but minimum image should give -2
        assert!((minimum_image(98.0, box_size) + 2.0).abs() < 1e-10,
            "Periodic distance expected -2.0, got {}", minimum_image(98.0, box_size));
        // Normal distance should not change
        assert!((minimum_image(1.0, box_size) - 1.0).abs() < 1e-10,
            "Normal distance expected 1.0, got {}", minimum_image(1.0, box_size));
        // Negative wrap
        assert!((minimum_image(-98.0, box_size) - 2.0).abs() < 1e-10,
            "Periodic distance expected 2.0, got {}", minimum_image(-98.0, box_size));
    }

    #[test]
    fn test_segregation_trivial() {
        // Create a simulation with known segregation
        // 4 particles + at right, 4 particles - at left
        // COM+ = (10, 0, 0), COM- = (-10, 0, 0)
        // Distance = 20, normalized by box_size=100 → 0.2
        let mut sim = NBodySimulation {
            particles: vec![
                Particle { pos: Vec3::new(10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Positive },
                Particle { pos: Vec3::new(10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Positive },
                Particle { pos: Vec3::new(10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Positive },
                Particle { pos: Vec3::new(10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Positive },
                Particle { pos: Vec3::new(-10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Negative },
                Particle { pos: Vec3::new(-10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Negative },
                Particle { pos: Vec3::new(-10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Negative },
                Particle { pos: Vec3::new(-10.0, 0.0, 0.0), vel: Vec3::zero(), mass: 1.0, sign: MassSign::Negative },
            ],
            time: 0.0,
            softening: 0.1,
            theta: 0.7,
            box_size: 100.0,
        };
        let seg = sim.segregation_distance();
        let expected = 20.0 / 100.0;  // 0.2 normalized
        assert!((seg - expected).abs() < 1e-10,
            "Segregation expected {}, got {}", expected, seg);
    }

    // =========================================================================
    // VALIDATION_RULES.md Section 4 — CONDITIONS AUX LIMITES PÉRIODIQUES
    // =========================================================================

    #[test]
    fn test_periodic_bc() {
        let box_half = 50.0;
        let box_size = 100.0;

        // Particle that exits right
        let mut pos = Vec3::new(51.0, 0.0, 0.0);
        apply_periodic_bc(&mut pos, box_half, box_size);
        assert!((pos.x - (-49.0)).abs() < 1e-10, "Wrap right: got {}", pos.x);

        // Particle that exits left
        let mut pos = Vec3::new(-51.0, 0.0, 0.0);
        apply_periodic_bc(&mut pos, box_half, box_size);
        assert!((pos.x - 49.0).abs() < 1e-10, "Wrap left: got {}", pos.x);

        // Particle inside — should not move
        let mut pos = Vec3::new(10.0, 5.0, -3.0);
        apply_periodic_bc(&mut pos, box_half, box_size);
        assert!((pos.x - 10.0).abs() < 1e-10, "No wrap x: got {}", pos.x);
        assert!((pos.y - 5.0).abs() < 1e-10, "No wrap y: got {}", pos.y);
        assert!((pos.z - (-3.0)).abs() < 1e-10, "No wrap z: got {}", pos.z);
    }

    // =========================================================================
    // VALIDATION_RULES.md — VIRIALIZATION TEST
    // =========================================================================

    #[test]
    fn test_virialization() {
        // Create a small simulation to test virialization
        // For Janus: 2KE + PE_binding ≈ 0 (using same-sign pairs only)
        let mut sim = NBodySimulation::new(100, 100, 20.0);

        // Get initial energies
        let ke_before = sim.kinetic_energy();
        let pe_total = sim.potential_energy();
        let pe_bind = sim.potential_energy_binding();

        println!("Before virialization:");
        println!("  KE = {:.4e}", ke_before);
        println!("  PE total = {:.4e}", pe_total);
        println!("  PE binding = {:.4e}", pe_bind);
        println!("  2KE + PE_bind = {:.4e}", 2.0 * ke_before + pe_bind);

        // Virialize (uses PE_binding internally)
        sim.virialize();

        // Check virial condition against PE_binding
        let ke_after = sim.kinetic_energy();
        let virial = 2.0 * ke_after + pe_bind;
        let virial_error = virial.abs() / pe_bind.abs();

        println!("After virialization:");
        println!("  KE = {:.4e}", ke_after);
        println!("  2KE + PE_bind = {:.4e}", virial);
        println!("  Virial error = {:.4}%", virial_error * 100.0);

        // For Janus: virial theorem applies to binding energy only
        assert!(virial_error < 0.01,
            "Virial condition not satisfied: 2KE + PE_bind = {:.4e}, error = {:.4}%",
            virial, virial_error * 100.0);
    }

    #[test]
    fn test_initial_segregation_random() {
        // For uniform random distribution, Seg₀ should be small
        // Expected: Seg₀ ~ 1/sqrt(N) ≈ 0.01 for N=10000
        // Validation rule: Seg₀ < 0.05
        let sim = NBodySimulation::new(5000, 5000, 50.0);
        let seg = sim.segregation_distance();

        println!("Initial segregation for 10K random particles:");
        println!("  Seg₀ = {:.6}", seg);
        println!("  Expected < 0.05 (validation rule)");

        assert!(seg < 0.05,
            "Initial segregation {} > 0.05 for random distribution", seg);
    }
}
