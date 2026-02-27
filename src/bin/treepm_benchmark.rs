//! TreePM Benchmark
//!
//! Measures performance of TreePM force calculation at various particle counts.

use janus::treepm::treepm_force::TreePMForce;
use janus::nbody::{Vec3, Particle};
use janus::MassSign;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn generate_particles(n: usize, box_size: f64, eta: f64, seed: u64) -> Vec<Particle> {
    let mut rng = StdRng::seed_from_u64(seed);

    let n_pos = ((n as f64) / (1.0 + eta)).round() as usize;
    let n_neg = n - n_pos;

    let mut particles = Vec::with_capacity(n);

    // Generate positive particles
    for _ in 0..n_pos {
        let pos = Vec3::new(
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
            (rng.random::<f64>() - 0.5) * box_size,
        );
        let vel = Vec3::new(
            (rng.random::<f64>() - 0.5) * 0.1,
            (rng.random::<f64>() - 0.5) * 0.1,
            (rng.random::<f64>() - 0.5) * 0.1,
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
            (rng.random::<f64>() - 0.5) * 0.1,
            (rng.random::<f64>() - 0.5) * 0.1,
            (rng.random::<f64>() - 0.5) * 0.1,
        );
        particles.push(Particle::new(pos, vel, 1.0, MassSign::Negative));
    }

    particles
}

fn benchmark_treepm(n: usize, grid_size: usize, box_size: f64, r_cut: f64) -> (f64, f64, f64) {
    let particles = generate_particles(n, box_size, 1.045, 42);

    let mut treepm = TreePMForce::new(r_cut, grid_size, box_size, 0.5, 0.1);

    // Warmup
    treepm.update(&particles);
    let _ = treepm.compute_all_forces(&particles);

    // Time PM update (mass assignment + FFT)
    let start = Instant::now();
    treepm.update(&particles);
    let pm_time = start.elapsed().as_secs_f64();

    // Time force computation (PM interpolation + Tree short-range)
    let start = Instant::now();
    let _forces = treepm.compute_all_forces(&particles);
    let force_time = start.elapsed().as_secs_f64();

    let total_time = pm_time + force_time;

    (pm_time, force_time, total_time)
}

fn main() {
    println!("=== TreePM Benchmark ===\n");

    let box_size = 100.0;
    let grid_size = 64;  // 64³ grid
    let r_cut = box_size / 16.0;

    println!("Configuration:");
    println!("  Box size: {}", box_size);
    println!("  Grid size: {}³", grid_size);
    println!("  r_cut: {:.2}", r_cut);
    println!("  θ (Barnes-Hut): 0.5");
    println!("  η: 1.045");
    println!();

    // Memory estimate
    let mem_grid = 4 * grid_size * grid_size * grid_size * 8;  // 4 grids × f64
    println!("PM grid memory: {:.1} MB\n", mem_grid as f64 / 1024.0 / 1024.0);

    println!("| N | PM (s) | Force (s) | Total (s) | ms/step |");
    println!("|---|--------|-----------|-----------|---------|");

    let test_sizes = [1000, 5000, 10000, 50000, 100000];

    for &n in &test_sizes {
        let (pm_time, force_time, total_time) = benchmark_treepm(n, grid_size, box_size, r_cut);
        let ms_per_step = total_time * 1000.0;

        println!("| {:>5} | {:.4} | {:.4} | {:.4} | {:.1} |",
                 n, pm_time, force_time, total_time, ms_per_step);
    }

    println!();
    println!("Notes:");
    println!("- This is CPU TreePM (rustfft), not GPU cuFFT");
    println!("- Tree short-range is O(N log N), not O(N²)");
    println!("- PM is O(N + G log G) where G = grid_size³");
    println!("- GPU cuFFT would significantly accelerate PM");
}
