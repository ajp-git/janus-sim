//! Cold Start Test - No Virialization
//!
//! Test if TreePM produces correct segregation dynamics with cold start

use janus::treepm::treepm_force::TreePMForce;
use janus::nbody::{Vec3, Particle};
use janus::MassSign;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

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

    if n_pos > 0.0 { com_pos.x /= n_pos; com_pos.y /= n_pos; com_pos.z /= n_pos; }
    if n_neg > 0.0 { com_neg.x /= n_neg; com_neg.y /= n_neg; com_neg.z /= n_neg; }

    let dx = com_pos.x - com_neg.x;
    let dy = com_pos.y - com_neg.y;
    let dz = com_pos.z - com_neg.z;
    (dx*dx + dy*dy + dz*dz).sqrt()
}

fn main() {
    println!("=== Cold Start Test (No Virialization) ===\n");

    let n = 1000;  // Small for fast test
    let box_size = 100.0;
    let grid_size = 64;
    let r_cut = box_size / 8.0;
    let dt = 0.1;
    let n_steps = 50;
    let softening = 1.0;
    let eta = 1.045;
    let g_constant = 0.1;  // Moderate G

    println!("N={}, G={}, dt={}, steps={}", n, g_constant, dt, n_steps);

    // Generate particles with random signs, zero velocity (cold start)
    let mut rng = StdRng::seed_from_u64(42);
    let prob_pos = 1.0 / (1.0 + eta);

    let mut particles: Vec<Particle> = (0..n).map(|_| {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        let sign = if rng.random::<f64>() < prob_pos {
            MassSign::Positive
        } else {
            MassSign::Negative
        };
        Particle::new(pos, Vec3::zero(), 1.0, sign)
    }).collect();

    let mut treepm = TreePMForce::new(r_cut, grid_size, box_size, 0.5, softening);
    treepm.g_constant = g_constant;

    let seg_0 = compute_segregation(&particles);
    println!("\nSeg₀ = {:.4} (expected ~ {:.4} for uniform)", seg_0, box_size / (n as f64).sqrt());

    println!("\nRunning {} steps...", n_steps);
    for step in 1..=n_steps {
        treepm.update(&particles);
        let forces = treepm.compute_all_forces(&particles);

        // Simple Euler integration for speed
        for (p, f) in particles.iter_mut().zip(forces.iter()) {
            p.vel.x += dt * f.x;
            p.vel.y += dt * f.y;
            p.vel.z += dt * f.z;
            p.pos.x += dt * p.vel.x;
            p.pos.y += dt * p.vel.y;
            p.pos.z += dt * p.vel.z;
        }

        if step % 10 == 0 {
            let seg = compute_segregation(&particles);
            let ke: f64 = particles.iter()
                .map(|p| 0.5 * (p.vel.x*p.vel.x + p.vel.y*p.vel.y + p.vel.z*p.vel.z))
                .sum();
            println!("Step {:3}: Seg = {:.4} (Δ = {:+.4}), KE = {:.4e}",
                     step, seg, seg - seg_0, ke);
        }
    }

    let seg_final = compute_segregation(&particles);
    println!("\n=== Result ===");
    println!("Seg₀     = {:.4}", seg_0);
    println!("Seg_final = {:.4}", seg_final);
    println!("ΔSeg     = {:+.4} ({:+.1}%)", seg_final - seg_0, (seg_final - seg_0) / seg_0 * 100.0);

    if seg_final > seg_0 {
        println!("\n✓ CORRECT: Segregation INCREASED (repulsion works)");
    } else {
        println!("\n✗ BUG: Segregation DECREASED");
    }
}
