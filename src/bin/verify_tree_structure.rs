//! Verify tree structure connectivity and mass conservation

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
    println!("║  TREE STRUCTURE VERIFICATION                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n = 50_000;  // Standard test size
    println!("Testing with N = {}", n);

    // 50-50 split
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

    // Do one step
    sim.step_dkd(0.001, 0.07, 1.0).expect("Step failed");

    // Get tree structure (after step, contains negative tree)
    let node_masses = sim.get_node_masses().expect("Failed to get masses");
    let left = sim.get_tree_left().expect("Failed to get left");
    let right = sim.get_tree_right().expect("Failed to get right");
    let types = sim.get_tree_types().expect("Failed to get types");

    // Tree parameters for negative tree
    let n_leaves = n_negative;
    let n_internal = n_leaves - 1;
    let n_nodes = 2 * n_leaves - 1;
    let first_leaf = n_internal;

    println!("Tree structure (N- = {}):", n_leaves);
    println!("  Total nodes: {}", n_nodes);
    println!("  Internal nodes: 0..{}", first_leaf);
    println!("  Leaves: {}..{}", first_leaf, n_nodes);
    println!();

    // Check node types
    let mut internal_count = 0;
    let mut leaf_count = 0;
    let mut unused_count = 0;
    for i in 0..n_nodes {
        match types[i] {
            0 => unused_count += 1,
            1 => leaf_count += 1,
            2 => internal_count += 1,
            _ => println!("  Warning: unexpected type {} at node {}", types[i], i),
        }
    }
    println!("Node types:");
    println!("  Internal (type=2): {}", internal_count);
    println!("  Leaf (type=1): {}", leaf_count);
    println!("  Unused (type=0): {}", unused_count);
    println!();

    // Verify tree connectivity by computing mass at each node via DFS
    fn compute_subtree_mass(node: usize, left: &[i32], right: &[i32], types: &[i32], masses: &[f32], n_internal: usize) -> f64 {
        if types[node] == 1 {
            // Leaf
            return masses[node] as f64;
        } else if types[node] == 2 {
            // Internal
            let lc = left[node] as usize;
            let rc = right[node] as usize;
            let ml = compute_subtree_mass(lc, left, right, types, masses, n_internal);
            let mr = compute_subtree_mass(rc, left, right, types, masses, n_internal);
            return ml + mr;
        } else {
            return 0.0;
        }
    }

    // Compute expected root mass by traversing tree
    let computed_root_mass = compute_subtree_mass(0, &left, &right, &types, &node_masses, n_internal);
    let stored_root_mass = node_masses[0] as f64;
    let leaf_sum: f64 = (first_leaf..n_nodes).map(|i| node_masses[i] as f64).sum();

    println!("Mass verification:");
    println!("  Leaf sum:        {:.6e}", leaf_sum);
    println!("  Computed root:   {:.6e} (via DFS)", computed_root_mass);
    println!("  Stored root:     {:.6e} (from buffer)", stored_root_mass);
    println!();

    let leaf_expected = n_leaves as f64 * mass_factor;
    if (leaf_sum - leaf_expected).abs() / leaf_expected < 0.001 {
        println!("✓ Leaf sum matches expected");
    } else {
        println!("✗ Leaf sum mismatch: {:.6e} vs expected {:.6e}", leaf_sum, leaf_expected);
    }

    if (computed_root_mass - leaf_sum).abs() / leaf_sum < 0.001 {
        println!("✓ Computed root matches leaf sum (tree is connected)");
    } else {
        println!("✗ Computed root != leaf sum: ratio = {:.4}", computed_root_mass / leaf_sum);
        println!("  Some leaves may be orphaned!");
    }

    if (stored_root_mass - computed_root_mass).abs() / computed_root_mass < 0.001 {
        println!("✓ Stored root matches computed (reduce_tp is correct)");
    } else {
        println!("✗ Stored root != computed: stored={:.6e}, computed={:.6e}",
                 stored_root_mass, computed_root_mass);
        println!("  reduce_tp has a bug!");
    }

    // Check ALL internal nodes for mass conservation
    println!();
    println!("Checking ALL internal nodes for mass errors...");
    let mut error_count = 0;
    let mut max_error = 0.0f64;
    let mut max_error_node = 0;
    for i in 0..n_internal {
        if types[i] != 2 { continue; }
        let lc = left[i] as usize;
        let rc = right[i] as usize;
        let ml = node_masses[lc] as f64;
        let mr = node_masses[rc] as f64;
        let stored = node_masses[i] as f64;
        let expected = ml + mr;
        let err = (stored - expected).abs();
        if err > 1e-3 {
            error_count += 1;
            if error_count <= 5 {
                println!("  ✗ node[{}]: stored={:.6e}, expected={:.6e}, Δ={:.2e}",
                         i, stored, expected, err);
            }
        }
        if err > max_error {
            max_error = err;
            max_error_node = i;
        }
    }
    if error_count > 5 {
        println!("  ... and {} more errors", error_count - 5);
    }
    println!();
    if error_count == 0 {
        println!("✓ All {} internal nodes have correct mass (stored = left + right)", n_internal);
    } else {
        println!("✗ {} internal nodes have incorrect mass!", error_count);
        println!("  Max error: {:.2e} at node {}", max_error, max_error_node);
    }

    // Additional check: verify DFS computed mass matches stored for each internal node
    println!();
    println!("Verifying DFS mass vs stored for sample nodes...");
    fn compute_dfs_mass(node: usize, left: &[i32], right: &[i32], types: &[i32], masses: &[f32]) -> f64 {
        if types[node] == 1 { return masses[node] as f64; }
        if types[node] != 2 { return 0.0; }
        compute_dfs_mass(left[node] as usize, left, right, types, masses)
            + compute_dfs_mass(right[node] as usize, left, right, types, masses)
    }

    for i in [0, n_internal/4, n_internal/2, 3*n_internal/4, n_internal-1] {
        let dfs_mass = compute_dfs_mass(i, &left, &right, &types, &node_masses);
        let stored = node_masses[i] as f64;
        let status = if (dfs_mass - stored).abs() / dfs_mass.max(1e-10) < 0.001 { "✓" } else { "✗" };
        println!("  {} node[{}]: DFS={:.6e}, stored={:.6e}", status, i, dfs_mass, stored);
    }

    // Find nodes where stored mass is significantly wrong
    println!();
    println!("Finding nodes with DFS/stored mismatch > 1%...");
    let mut mismatched_nodes = Vec::new();
    for i in 0..n_internal {
        if types[i] != 2 { continue; }
        let dfs_mass = compute_dfs_mass(i, &left, &right, &types, &node_masses);
        let stored = node_masses[i] as f64;
        let rel_err = (dfs_mass - stored).abs() / dfs_mass.max(1e-10);
        if rel_err > 0.01 {
            mismatched_nodes.push((i, dfs_mass, stored, rel_err));
        }
    }
    mismatched_nodes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("  Found {} nodes with >1% mismatch", mismatched_nodes.len());
    for (i, dfs, stored, rel_err) in mismatched_nodes.iter().take(10) {
        println!("  node[{}]: DFS={:.4e}, stored={:.4e}, err={:.1}%", i, dfs, stored, rel_err * 100.0);
        // Print path to root
        let lc = left[*i] as usize;
        let rc = right[*i] as usize;
        let ml = node_masses[lc] as f64;
        let mr = node_masses[rc] as f64;
        println!("    children: left={} (m={:.4e}), right={} (m={:.4e}), sum={:.4e}",
                 lc, ml, rc, mr, ml + mr);
    }

    // === PARENT POINTER VERIFICATION ===
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PARENT POINTER VERIFICATION                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let parent = sim.get_tree_parent().expect("Failed to get parent");

    // For each internal node i, check that parent[left[i]] == i and parent[right[i]] == i
    let mut parent_errors = 0;
    let mut left_parent_wrong = 0;
    let mut right_parent_wrong = 0;

    for i in 0..n_internal {
        if types[i] != 2 { continue; }
        let lc = left[i] as usize;
        let rc = right[i] as usize;

        if parent[lc] != i as i32 {
            parent_errors += 1;
            left_parent_wrong += 1;
            if parent_errors <= 5 {
                println!("  ✗ node[{}]: left child {} has parent={}, expected {}",
                         i, lc, parent[lc], i);
            }
        }
        if parent[rc] != i as i32 {
            parent_errors += 1;
            right_parent_wrong += 1;
            if parent_errors <= 5 {
                println!("  ✗ node[{}]: right child {} has parent={}, expected {}",
                         i, rc, parent[rc], i);
            }
        }
    }

    if parent_errors > 5 {
        println!("  ... and {} more parent errors", parent_errors - 5);
    }
    println!();

    if parent_errors == 0 {
        println!("✓ All parent pointers consistent with left/right pointers");
    } else {
        println!("✗ {} parent pointer inconsistencies found!", parent_errors);
        println!("  Left child wrong: {}, Right child wrong: {}", left_parent_wrong, right_parent_wrong);
    }

    // Check if mismatched mass nodes have parent pointer issues
    println!();
    println!("Checking parent pointers for nodes with mass errors...");
    for (i, dfs, stored, _rel_err) in mismatched_nodes.iter().take(5) {
        let lc = left[*i] as usize;
        let rc = right[*i] as usize;
        let lc_parent = parent[lc];
        let rc_parent = parent[rc];
        let ml = node_masses[lc] as f64;
        let mr = node_masses[rc] as f64;

        let lc_ok = lc_parent == *i as i32;
        let rc_ok = rc_parent == *i as i32;

        println!("  node[{}]: stored={:.4e}, expected={:.4e}", i, stored, dfs);
        println!("    left[{}]:  m={:.4e}, parent={} {}", lc, ml, lc_parent, if lc_ok { "✓" } else { "✗" });
        println!("    right[{}]: m={:.4e}, parent={} {}", rc, mr, rc_parent, if rc_ok { "✓" } else { "✗" });

        // Check if stored == left only (the observed pattern)
        if (stored - ml).abs() < 1e-6 {
            println!("    → PATTERN: stored ≈ left child mass only!");
        } else if (stored - mr).abs() < 1e-6 {
            println!("    → PATTERN: stored ≈ right child mass only!");
        }
    }

    // Check root's parent (should be -1 or special value)
    println!();
    println!("Root (node 0) parent: {}", parent[0]);
}

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
