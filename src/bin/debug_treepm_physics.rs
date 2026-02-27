//! Debug TreePM Physics
//!
//! Minimal test: 4+ on right, 4- on left
//! After 10 steps, segregation should INCREASE (populations move apart)

use janus::treepm::treepm_force::TreePMForce;
use janus::nbody::{Vec3, Particle};
use janus::MassSign;

fn compute_segregation(particles: &[Particle]) -> f64 {
    let (mut com_pos, mut n_pos) = (Vec3::zero(), 0.0);
    let (mut com_neg, mut n_neg) = (Vec3::zero(), 0.0);

    for p in particles {
        match p.sign {
            MassSign::Positive => {
                com_pos.x += p.pos.x;
                com_pos.y += p.pos.y;
                com_pos.z += p.pos.z;
                n_pos += 1.0;
            }
            MassSign::Negative => {
                com_neg.x += p.pos.x;
                com_neg.y += p.pos.y;
                com_neg.z += p.pos.z;
                n_neg += 1.0;
            }
        }
    }

    if n_pos > 0.0 {
        com_pos.x /= n_pos;
        com_pos.y /= n_pos;
        com_pos.z /= n_pos;
    }
    if n_neg > 0.0 {
        com_neg.x /= n_neg;
        com_neg.y /= n_neg;
        com_neg.z /= n_neg;
    }

    let dx = com_pos.x - com_neg.x;
    let dy = com_pos.y - com_neg.y;
    let dz = com_pos.z - com_neg.z;
    (dx*dx + dy*dy + dz*dz).sqrt()
}

fn main() {
    println!("=== TreePM Physics Debug ===\n");

    let box_size = 100.0;
    let grid_size = 32;
    let r_cut = box_size / 8.0;  // Larger r_cut to ensure Tree dominates
    let dt = 0.1;
    let softening = 1.0;

    // Create 8 particles: 4+ on right (x > 0), 4- on left (x < 0)
    let mut particles = vec![
        // Positive particles on the right
        Particle::new(Vec3::new(10.0, 5.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        Particle::new(Vec3::new(10.0, -5.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        Particle::new(Vec3::new(15.0, 5.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        Particle::new(Vec3::new(15.0, -5.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        // Negative particles on the left
        Particle::new(Vec3::new(-10.0, 5.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
        Particle::new(Vec3::new(-10.0, -5.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
        Particle::new(Vec3::new(-15.0, 5.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
        Particle::new(Vec3::new(-15.0, -5.0, 0.0), Vec3::zero(), 1.0, MassSign::Negative),
    ];

    let mut treepm = TreePMForce::new(r_cut, grid_size, box_size, 0.5, softening);
    treepm.g_constant = 1.0;

    let seg_0 = compute_segregation(&particles);
    println!("Initial setup:");
    println!("  4+ particles at x ≈ +12.5");
    println!("  4- particles at x ≈ -12.5");
    println!("  Seg₀ = {:.4}", seg_0);
    println!();

    // Check initial forces
    treepm.update(&particles);
    println!("Initial forces (should show repulsion between + and -):");

    for (i, p) in particles.iter().enumerate() {
        let f = treepm.compute_force_excluding(p.pos, p.sign, &particles, Some(i));
        let sign_str = match p.sign { MassSign::Positive => "+", MassSign::Negative => "-" };
        println!("  P{} ({}) at x={:+.1}: Fx={:+.6}", i, sign_str, p.pos.x, f.x);
    }

    // Check: + particles on right should have Fx > 0 (repelled away from - on left)
    // Check: - particles on left should have Fx < 0 (repelled away from + on right)
    let f0 = treepm.compute_force_excluding(particles[0].pos, particles[0].sign, &particles, Some(0));
    let f4 = treepm.compute_force_excluding(particles[4].pos, particles[4].sign, &particles, Some(4));

    println!();
    if f0.x > 0.0 {
        println!("✓ + particle on right has Fx > 0 (repelled from - on left)");
    } else {
        println!("✗ BUG: + particle on right has Fx < 0 (ATTRACTED to - on left!)");
    }
    if f4.x < 0.0 {
        println!("✓ - particle on left has Fx < 0 (repelled from + on right)");
    } else {
        println!("✗ BUG: - particle on left has Fx > 0 (ATTRACTED to + on right!)");
    }

    println!("\n--- Running 10 steps ---");

    for step in 1..=10 {
        treepm.update(&particles);
        let forces = treepm.compute_all_forces(&particles);

        // Leapfrog integration
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            p.vel.x += dt * f.x;
            p.vel.y += dt * f.y;
            p.vel.z += dt * f.z;
            p.pos.x += dt * p.vel.x;
            p.pos.y += dt * p.vel.y;
            p.pos.z += dt * p.vel.z;
        }

        let seg = compute_segregation(&particles);
        println!("Step {}: Seg = {:.4} (Δ = {:+.4})", step, seg, seg - seg_0);
    }

    let seg_final = compute_segregation(&particles);
    println!("\n=== Result ===");
    println!("Seg₀ = {:.4}", seg_0);
    println!("Seg_final = {:.4}", seg_final);

    if seg_final > seg_0 {
        println!("✓ CORRECT: Segregation increased (populations repel)");
    } else {
        println!("✗ BUG: Segregation decreased (physics inverted!)");
    }
}
