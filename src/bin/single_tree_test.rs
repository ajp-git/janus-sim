//! Test single tree build (no two-pass) to verify mass conservation

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
    println!("║  SINGLE TREE TEST — All positive particles                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n = 50_000;
    println!("Testing with N = {}", n);

    // Mostly positive particles (N-=100 just to avoid empty tree)
    let (pos, vel, signs) = generate_mostly_positive_ics(n);

    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos, vel, signs, BOX_SIZE
    ).expect("Failed to create simulation");

    let mass_factor = sim.get_mass_factor();
    println!("mass_factor = {:.6e}", mass_factor);
    let n_positive = n - 25000;
    let n_negative = 25000;
    println!("N+ = {}, N- = {}", n_positive, n_negative);
    println!();

    // Do one step
    sim.step_dkd(0.001, 0.07, 1.0).expect("Step failed");

    // Get node masses
    let node_masses = sim.get_node_masses().expect("Failed to get node masses");

    // After step_dkd, tree buffer contains the NEGATIVE tree (last built)
    // So we check the N-=100 particle tree
    let n_tree = n_negative;
    let n_nodes = 2 * n_tree - 1;
    let first_leaf = n_tree - 1;

    // Sum leaves
    let leaf_sum: f64 = (first_leaf..n_nodes).map(|i| node_masses[i] as f64).sum();
    let expected = n_tree as f64 * mass_factor;

    println!("Leaf sum:    {:.6e}", leaf_sum);
    println!("Expected:    {:.6e}", expected);
    println!("Root mass:   {:.6e}", node_masses[0]);
    println!();

    let leaf_ratio = leaf_sum / expected;
    let root_ratio = node_masses[0] as f64 / leaf_sum;

    if (leaf_ratio - 1.0).abs() < 0.001 {
        println!("✓ Leaf sum matches expected (±0.1%)");
    } else {
        println!("✗ Leaf sum mismatch: ratio = {:.4}", leaf_ratio);
    }

    if (root_ratio - 1.0).abs() < 0.001 {
        println!("✓ Root matches leaf sum (±0.1%) - mass conserved in tree");
    } else {
        println!("✗ Root/leaf mismatch: ratio = {:.4}", root_ratio);
    }
}

#[cfg(feature = "cuda")]
fn generate_mostly_positive_ics(n: usize) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let box_half = BOX_SIZE / 2.0;

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for i in 0..n {
        let x = rng.random::<f64>() * BOX_SIZE - box_half;
        let y = rng.random::<f64>() * BOX_SIZE - box_half;
        let z = rng.random::<f64>() * BOX_SIZE - box_half;
        pos.extend_from_slice(&[x as f32, y as f32, z as f32]);
        vel.extend_from_slice(&[0.0f32, 0.0, 0.0]);
        // Last 25000 particles are negative (larger tree to test)
        signs.push(if i >= n - 25000 { -1 } else { 1 });
    }

    (pos, vel, signs)
}
