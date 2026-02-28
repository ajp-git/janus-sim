//! Test PM forces vs BH at different scales
//!
//! PM should match BH for large-scale (r >> r_s) forces.
//! If PM underestimates, the k-space damping or CIC is wrong.

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== PM vs BH Force Test ===\n");

    // Use large box with sparse particles to test long-range forces
    let n = 2_000;  // Sparse
    let box_size: f64 = 200.0;  // Large box
    let r_cut = box_size / 8.0;  // 25 Mpc

    println!("Parameters: N={}, box={}, r_cut={}", n, box_size, r_cut);
    println!("Mean separation: {:.1} Mpc\n", (box_size.powi(3) / n as f64).powf(1.0/3.0));

    // Create simulation
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(0.5);

    let (positions, _velocities, signs) = sim.get_particles().expect("get_particles failed");

    // Test 1: Pure BH (reference)
    println!("Test 1: Pure BH (full force calculation)...");
    let mut sim_bh = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        vec![0.0f32; n * 3],
        signs.clone(),
        box_size
    ).expect("init failed");
    sim_bh.set_theta(0.5);
    sim_bh.step_dkd(0.01, 0.0, 0.0).unwrap();
    let acc_bh = sim_bh.acceleration_sum().unwrap();
    let ke_bh = sim_bh.kinetic_energy().unwrap();
    println!("  |acc| = {:.4e}, KE = {:.4e}", acc_bh, ke_bh);

    // Test 2: TreePM full (short-range + PM)
    println!("\nTest 2: TreePM (short-range + PM)...");
    let mut sim_treepm = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        vec![0.0f32; n * 3],
        signs.clone(),
        box_size
    ).expect("init failed");
    sim_treepm.set_theta(0.5);
    sim_treepm.step_treepm_gpu(0.01, r_cut, 0.0, 0.0).unwrap();
    let acc_treepm = sim_treepm.acceleration_sum().unwrap();
    let ke_treepm = sim_treepm.kinetic_energy().unwrap();
    println!("  |acc| = {:.4e}, KE = {:.4e}", acc_treepm, ke_treepm);

    // Test 3: Very large r_cut (almost all BH, minimal PM)
    println!("\nTest 3: TreePM with large r_cut (r_cut = box/2)...");
    let mut sim_large_rcut = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        vec![0.0f32; n * 3],
        signs.clone(),
        box_size
    ).expect("init failed");
    sim_large_rcut.set_theta(0.5);
    let large_rcut = box_size / 2.0;  // 100 Mpc - most pairs within r_cut
    sim_large_rcut.step_treepm_gpu(0.01, large_rcut, 0.0, 0.0).unwrap();
    let acc_large = sim_large_rcut.acceleration_sum().unwrap();
    let ke_large = sim_large_rcut.kinetic_energy().unwrap();
    println!("  |acc| = {:.4e}, KE = {:.4e}", acc_large, ke_large);

    // Analysis
    println!("\n=== Analysis ===");
    println!("  TreePM/BH ratio:        {:.3}", acc_treepm / acc_bh);
    println!("  Large r_cut/BH ratio:   {:.3}", acc_large / acc_bh);

    // Expected: with larger r_cut, more pairs use erfc-weighted short-range
    // If large_rcut/BH < treepm/BH, then erfc weighting is losing force
    // If large_rcut/BH ≈ 1, then short-range kernel is correct, PM is weak

    if acc_large / acc_bh > 0.95 {
        println!("\n→ Large r_cut matches BH - short-range force kernel is correct");
        if acc_treepm / acc_bh < 0.85 {
            println!("→ PM contribution is too weak (missing {:.0}%)",
                     (1.0 - acc_treepm / acc_bh) * 100.0);
        }
    } else {
        println!("\n→ Even large r_cut doesn't match BH");
        println!("→ BUG IN short-range erfc weighting");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
