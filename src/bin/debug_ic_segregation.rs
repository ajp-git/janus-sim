//! Debug Initial Conditions Segregation
//!
//! Test that uniform random distribution gives Seg₀ ≈ 0.03

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

fn generate_particles_sequential(n: usize, box_size: f64, eta: f64, seed: u64) -> Vec<Particle> {
    // BUGGY: Generates all + first, then all -
    let mut rng = StdRng::seed_from_u64(seed);
    let n_pos = ((n as f64) / (1.0 + eta)).round() as usize;
    let n_neg = n - n_pos;
    let mut particles = Vec::with_capacity(n);

    // Generate positive particles FIRST (all in early random sequence)
    for _ in 0..n_pos {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        particles.push(Particle::new(pos, Vec3::zero(), 1.0, MassSign::Positive));
    }

    // Then generate negative particles (different random sequence)
    for _ in 0..n_neg {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        particles.push(Particle::new(pos, Vec3::zero(), 1.0, MassSign::Negative));
    }

    particles
}

fn generate_particles_interleaved(n: usize, box_size: f64, eta: f64, seed: u64) -> Vec<Particle> {
    // FIXED: Interleave + and - to ensure same spatial distribution
    let mut rng = StdRng::seed_from_u64(seed);
    let n_pos = ((n as f64) / (1.0 + eta)).round() as usize;
    let n_neg = n - n_pos;
    let mut particles = Vec::with_capacity(n);

    let mut pos_count = 0;
    let mut neg_count = 0;

    for i in 0..n {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );

        // Assign sign based on ratio
        let sign = if (i as f64 / n as f64) < (n_pos as f64 / n as f64) {
            if pos_count < n_pos {
                pos_count += 1;
                MassSign::Positive
            } else {
                neg_count += 1;
                MassSign::Negative
            }
        } else {
            if neg_count < n_neg {
                neg_count += 1;
                MassSign::Negative
            } else {
                pos_count += 1;
                MassSign::Positive
            }
        };

        particles.push(Particle::new(pos, Vec3::zero(), 1.0, sign));
    }

    particles
}

fn generate_particles_random_sign(n: usize, box_size: f64, eta: f64, seed: u64) -> Vec<Particle> {
    // BEST: Random position, random sign (with correct ratio)
    let mut rng = StdRng::seed_from_u64(seed);
    let prob_pos = 1.0 / (1.0 + eta);  // Probability of being positive
    let mut particles = Vec::with_capacity(n);

    for _ in 0..n {
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

        particles.push(Particle::new(pos, Vec3::zero(), 1.0, sign));
    }

    particles
}

fn main() {
    println!("=== IC Segregation Debug ===\n");

    let box_size = 100.0;
    let eta = 1.045;
    let n = 100_000;

    println!("Testing N={}, η={}, box={}\n", n, eta, box_size);

    // Test 1: Sequential generation (current buggy code)
    let p1 = generate_particles_sequential(n, box_size, eta, 42);
    let seg1 = compute_segregation(&p1);
    println!("Sequential (+ first, then -): Seg₀ = {:.4}", seg1);

    // Test 2: Interleaved generation
    let p2 = generate_particles_interleaved(n, box_size, eta, 42);
    let seg2 = compute_segregation(&p2);
    println!("Interleaved:                  Seg₀ = {:.4}", seg2);

    // Test 3: Random sign assignment
    let p3 = generate_particles_random_sign(n, box_size, eta, 42);
    let seg3 = compute_segregation(&p3);
    println!("Random sign:                  Seg₀ = {:.4}", seg3);

    // Expected theoretical value for uniform distribution
    // Seg ~ L / sqrt(N) for random walk
    let expected = box_size / (n as f64).sqrt();
    println!("\nExpected (L/√N):              Seg₀ ≈ {:.4}", expected);

    println!("\n=== Analysis ===");
    if seg1 > 0.3 {
        println!("✗ Sequential generation produces artificially high Seg₀");
        println!("  This is because + and - use different parts of the RNG sequence");
    }
    if seg3 < 0.1 {
        println!("✓ Random sign produces correct low Seg₀");
    }
}
