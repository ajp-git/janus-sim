//! Benchmark PM vs Tree separately
//! Identify bottleneck for optimization

use janus::treepm::pm_grid::PmGrid;
use janus::treepm::tree_short::TreePMTree;
use janus::nbody::{Vec3, Particle};
use janus::MassSign;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn generate_particles(n: usize, box_size: f64, eta: f64, seed: u64) -> Vec<Particle> {
    let mut rng = StdRng::seed_from_u64(seed);
    let prob_pos = 1.0 / (1.0 + eta);

    (0..n).map(|_| {
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
    }).collect()
}

fn main() {
    println!("=== PM vs Tree Benchmark ===\n");

    let box_size = 100.0;
    let r_cut = box_size / 16.0;
    let softening = 0.5;
    let g_constant = 1.0;

    for &n in &[1000, 5000, 10000, 50000, 100000] {
        // Adjust grid size based on N
        let grid_size = if n <= 10000 { 64 } else { 128 };

        let particles = generate_particles(n, box_size, 1.045, 42);

        // Benchmark PM
        let mut pm = PmGrid::new(grid_size, box_size);

        let start = Instant::now();
        // Mass assignment
        for p in &particles {
            let sign_i8: i8 = match p.sign { MassSign::Positive => 1, MassSign::Negative => -1 };
            pm.assign_mass(p.pos.x, p.pos.y, p.pos.z, p.mass, sign_i8);
        }
        // FFT + Poisson
        pm.solve_poisson(g_constant);
        let pm_time = start.elapsed().as_secs_f64();

        // Benchmark force interpolation (PM part)
        let start = Instant::now();
        let _forces_pm: Vec<_> = particles.iter().map(|p| {
            let sign_i8: i8 = match p.sign { MassSign::Positive => 1, MassSign::Negative => -1 };
            pm.interpolate_force(p.pos.x, p.pos.y, p.pos.z, sign_i8)
        }).collect();
        let pm_interp_time = start.elapsed().as_secs_f64();

        // Benchmark Tree
        let start = Instant::now();
        let tree = TreePMTree::build_with_g(&particles, 0.5, r_cut, g_constant);
        let tree_build_time = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let _forces_tree: Vec<_> = particles.iter().enumerate().map(|(i, p)| {
            tree.compute_short_range_acc_excluding(p.pos, p.sign, &particles, softening, Some(i))
        }).collect();
        let tree_force_time = start.elapsed().as_secs_f64();

        let total_pm = pm_time + pm_interp_time;
        let total_tree = tree_build_time + tree_force_time;

        println!("N = {:>6}:  PM = {:.3}s ({:.3}+{:.3})  Tree = {:.3}s ({:.3}+{:.3})  Total = {:.3}s",
                 n, total_pm, pm_time, pm_interp_time,
                 total_tree, tree_build_time, tree_force_time,
                 total_pm + total_tree);
    }

    println!("\nNote: Tree is O(N log N), PM is O(G³ log G + N)");
    println!("For 1M particles, Tree would dominate.");
    println!("\nOptimization options:");
    println!("1. GPU Barnes-Hut for Tree (already have code in nbody_gpu.rs)");
    println!("2. Higher θ for Tree (less accurate but faster)");
    println!("3. PM-only mode (skip Tree, good for visual validation)");
}
