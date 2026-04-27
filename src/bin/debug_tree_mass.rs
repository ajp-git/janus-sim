//! Debug: examine tree node masses to find where mass is lost

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

use rand::prelude::*;
use rand::SeedableRng;

const BOX_SIZE: f64 = 500.0;
const SEED: u64 = 42;

fn main() {
    #[cfg(not(feature = "cuda"))]
    { eprintln!("Requires --features cuda"); std::process::exit(1); }
    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DEBUG TREE MASS STRUCTURE                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Larger N to see if issue appears
    let n = 100_000;
    println!("Testing with N = {}", n);

    let (pos, vel, signs) = generate_ics(n);
    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let n_negative = n - n_positive;

    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos, vel, signs, BOX_SIZE
    ).expect("Failed to create simulation");

    let mass_factor = sim.get_mass_factor();
    println!("mass_factor = {:.6e}", mass_factor);
    println!("N+ = {}, N- = {}", n_positive, n_negative);
    println!();

    // Expected totals
    let expected_total = n as f64 * mass_factor;
    let expected_neg = n_negative as f64 * mass_factor;
    println!("Expected total (N × mf) = {:.6e}", expected_total);
    println!("Expected neg (N- × mf)  = {:.6e}", expected_neg);
    println!();

    // Do one step
    sim.step_dkd(0.001, 0.07, 1.0).expect("Step failed");

    // Get all node masses
    let node_masses = sim.get_node_masses().expect("Failed to get node masses");

    // Tree structure for N- particles:
    // - Leaves: indices N--1 to 2*N--2 (N- leaves)
    // - Internal: indices 0 to N--2 (N--1 internal nodes)
    // - Root: index 0

    let n_nodes = 2 * n_negative - 1;
    let first_leaf = n_negative - 1;

    println!("Tree structure (last built = neg tree with {} particles):", n_negative);
    println!("  Total nodes: {}", n_nodes);
    println!("  Internal nodes: 0..{}", first_leaf);
    println!("  Leaves: {}..{}", first_leaf, n_nodes);
    println!();

    // Check leaves
    let mut leaf_sum = 0.0_f64;
    let mut leaf_count = 0;
    let mut nonzero_leaves = 0;
    for i in first_leaf..n_nodes {
        let m = node_masses[i] as f64;
        if m.abs() > 1e-30 {
            leaf_sum += m;
            nonzero_leaves += 1;
        }
        leaf_count += 1;
    }
    println!("Leaf analysis:");
    println!("  Leaf count: {}", leaf_count);
    println!("  Non-zero leaves: {}", nonzero_leaves);
    println!("  Sum of leaf masses: {:.6e}", leaf_sum);
    println!("  Expected (N- × mf): {:.6e}", expected_neg);
    println!("  Ratio: {:.4}", leaf_sum / expected_neg);
    println!();

    // Check first few leaves
    println!("First 5 leaf masses:");
    for i in 0..5.min(n_negative) {
        let idx = first_leaf + i;
        println!("  leaf[{}] = {:.6e}", idx, node_masses[idx]);
    }
    println!("  Expected per leaf: {:.6e}", mass_factor);
    println!();

    // Check internal nodes
    println!("Internal node analysis:");
    println!("  Root (node 0) mass: {:.6e}", node_masses[0]);
    println!("  Node 1 mass: {:.6e}", node_masses[1]);
    println!("  Node 2 mass: {:.6e}", node_masses[2]);
    let root_children_sum = node_masses[1] as f64 + node_masses[2] as f64;
    println!("  Sum of root children: {:.6e}", root_children_sum);
    println!("  Root - children sum: {:.6e}", node_masses[0] as f64 - root_children_sum);
    println!();

    // Verify mass conservation
    let internal_sum: f64 = (0..first_leaf).map(|i| node_masses[i] as f64).sum();
    println!("Summary:");
    println!("  Leaf sum:     {:.6e}", leaf_sum);
    println!("  Root mass:    {:.6e}", node_masses[0]);
    println!("  Ratio root/leaves: {:.4}", node_masses[0] as f64 / leaf_sum);

    if (node_masses[0] as f64 - leaf_sum).abs() / leaf_sum < 0.01 {
        println!("  ✓ Root = sum of leaves (mass conserved in tree)");
    } else {
        println!("  ✗ Root ≠ sum of leaves (mass NOT conserved!)");
    }
}

// Note: get_node_masses() must be added to nbody_gpu_twopass.rs

#[cfg(feature = "cuda")]
fn generate_ics(n: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let box_half = BOX_SIZE / 2.0;

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for _ in 0..n {
        let x = rng.random::<f64>() * BOX_SIZE - box_half;
        let y = rng.random::<f64>() * BOX_SIZE - box_half;
        let z = rng.random::<f64>() * BOX_SIZE - box_half;
        pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        signs.push(if rng.random::<bool>() { 1 } else { -1 });
    }

    (pos, vel, signs)
}
