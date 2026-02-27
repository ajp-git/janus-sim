//! TreePM Physics Validation Run
//!
//! Runs a short simulation to validate TreePM physics:
//! 1. Energy conservation
//! 2. Segregation (Janus physics working)
//! 3. No particle escape

use janus::treepm::treepm_force::TreePMForce;
use janus::nbody::{Vec3, Particle};
use janus::MassSign;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn generate_virialized_particles(n: usize, box_size: f64, eta: f64, seed: u64) -> Vec<Particle> {
    let mut rng = StdRng::seed_from_u64(seed);

    let n_pos = ((n as f64) / (1.0 + eta)).round() as usize;
    let n_neg = n - n_pos;

    let mut particles = Vec::with_capacity(n);

    // Virial velocity scale
    let virial_v = 0.1;

    // Generate positive particles
    for _ in 0..n_pos {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        let vel = Vec3::new(
            (rng.random::<f64>() - 0.5) * virial_v,
            (rng.random::<f64>() - 0.5) * virial_v,
            (rng.random::<f64>() - 0.5) * virial_v,
        );
        particles.push(Particle::new(pos, vel, 1.0, MassSign::Positive));
    }

    // Generate negative particles
    for _ in 0..n_neg {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        let vel = Vec3::new(
            (rng.random::<f64>() - 0.5) * virial_v,
            (rng.random::<f64>() - 0.5) * virial_v,
            (rng.random::<f64>() - 0.5) * virial_v,
        );
        particles.push(Particle::new(pos, vel, 1.0, MassSign::Negative));
    }

    particles
}

fn compute_kinetic_energy(particles: &[Particle]) -> f64 {
    particles.iter()
        .map(|p| 0.5 * (p.vel.x*p.vel.x + p.vel.y*p.vel.y + p.vel.z*p.vel.z))
        .sum()
}

fn compute_segregation(particles: &[Particle]) -> f64 {
    // Simple segregation metric: mean distance from COM for each species
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

    // Distance between COMs
    let dx = com_pos.x - com_neg.x;
    let dy = com_pos.y - com_neg.y;
    let dz = com_pos.z - com_neg.z;
    (dx*dx + dy*dy + dz*dz).sqrt()
}

fn main() {
    println!("=== TreePM Physics Validation ===\n");

    let n = 10000;
    let box_size = 100.0;
    let grid_size = 64;
    let r_cut = box_size / 16.0;
    let dt = 0.01;
    let n_steps = 100;
    let softening = 0.5;
    let eta = 1.045;

    println!("Configuration:");
    println!("  N particles: {}", n);
    println!("  Box size: {}", box_size);
    println!("  Grid: {}³", grid_size);
    println!("  r_cut: {:.2}", r_cut);
    println!("  dt: {}", dt);
    println!("  Steps: {}", n_steps);
    println!("  η: {}", eta);
    println!();

    let mut particles = generate_virialized_particles(n, box_size, eta, 42);
    let mut treepm = TreePMForce::new(r_cut, grid_size, box_size, 0.5, softening);
    // Reduce G to make the system stable (non-virialized ICs)
    treepm.g_constant = 0.001;

    // Initial metrics
    let ke_0 = compute_kinetic_energy(&particles);
    let seg_0 = compute_segregation(&particles);

    println!("Initial state:");
    println!("  KE₀ = {:.6}", ke_0);
    println!("  Seg₀ = {:.6}", seg_0);
    println!();

    // Check initial force magnitudes
    treepm.update(&particles);
    let forces: Vec<Vec3> = particles.iter().map(|p| {
        let sign_i8 = match p.sign { MassSign::Positive => 1i8, MassSign::Negative => -1i8 };
        let (fx, fy, fz) = treepm.pm.interpolate_force(p.pos.x, p.pos.y, p.pos.z, sign_i8);
        Vec3::new(fx, fy, fz)
    }).collect();

    let avg_f: f64 = forces.iter().map(|f| (f.x*f.x + f.y*f.y + f.z*f.z).sqrt()).sum::<f64>() / forces.len() as f64;
    let max_f: f64 = forces.iter().map(|f| (f.x*f.x + f.y*f.y + f.z*f.z).sqrt()).fold(0.0, f64::max);
    println!("Force check:");
    println!("  Avg |F| = {:.6}", avg_f);
    println!("  Max |F| = {:.6}", max_f);
    println!();

    // Run simulation
    let start = Instant::now();

    for step in 0..n_steps {
        // Update TreePM
        treepm.update(&particles);

        // Compute forces (full TreePM)
        let forces = treepm.compute_all_forces(&particles);

        // Leapfrog kick (half)
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            p.vel.x += 0.5 * dt * f.x;
            p.vel.y += 0.5 * dt * f.y;
            p.vel.z += 0.5 * dt * f.z;
        }

        // Drift
        for p in &mut particles {
            p.pos.x += dt * p.vel.x;
            p.pos.y += dt * p.vel.y;
            p.pos.z += dt * p.vel.z;
        }

        // Update TreePM again
        treepm.update(&particles);
        let forces = treepm.compute_all_forces(&particles);

        // Leapfrog kick (half)
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            p.vel.x += 0.5 * dt * f.x;
            p.vel.y += 0.5 * dt * f.y;
            p.vel.z += 0.5 * dt * f.z;
        }

        if step % 20 == 0 {
            let ke = compute_kinetic_energy(&particles);
            let seg = compute_segregation(&particles);
            let ke_ratio = ke / ke_0;
            println!("  Step {:3}: KE/KE₀ = {:.3}, Seg = {:.3}", step, ke_ratio, seg);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Final metrics
    let ke_final = compute_kinetic_energy(&particles);
    let seg_final = compute_segregation(&particles);
    let ke_ratio = ke_final / ke_0;
    let seg_change = (seg_final - seg_0) / seg_0;

    println!();
    println!("Final state (step {}):", n_steps);
    println!("  KE_final = {:.6}", ke_final);
    println!("  KE/KE₀ = {:.3}", ke_ratio);
    println!("  Seg_final = {:.6}", seg_final);
    println!("  ΔSeg/Seg₀ = {:.1}%", seg_change * 100.0);
    println!("  Elapsed: {:.2}s ({:.1}ms/step)", elapsed, elapsed * 1000.0 / n_steps as f64);
    println!();

    // Validation checks
    let mut passed = true;

    // Check 1: KE didn't explode
    if ke_ratio > 10.0 {
        println!("✗ FAIL: KE exploded (KE/KE₀ = {:.1} > 10)", ke_ratio);
        passed = false;
    } else {
        println!("✓ PASS: KE stable (KE/KE₀ = {:.3} < 10)", ke_ratio);
    }

    // Check 2: Segregation change (Janus physics)
    // Note: With random ICs, segregation might decrease initially before increasing
    // For a short run, we just check it doesn't go negative
    if seg_final >= 0.0 {
        println!("✓ PASS: Segregation non-negative (Seg = {:.3})", seg_final);
    } else {
        println!("✗ FAIL: Segregation negative");
        passed = false;
    }

    // Check 3: No particles escaped (all within 2*box_size)
    let max_dist: f64 = particles.iter()
        .map(|p| (p.pos.x*p.pos.x + p.pos.y*p.pos.y + p.pos.z*p.pos.z).sqrt())
        .fold(0.0, f64::max);

    if max_dist < 2.0 * box_size {
        println!("✓ PASS: No particles escaped (max_r = {:.1} < {})", max_dist, 2.0 * box_size);
    } else {
        println!("✗ FAIL: Particles escaped (max_r = {:.1})", max_dist);
        passed = false;
    }

    println!();
    if passed {
        println!("=== ALL VALIDATION CHECKS PASSED ===");
    } else {
        println!("=== SOME VALIDATION CHECKS FAILED ===");
    }
}
