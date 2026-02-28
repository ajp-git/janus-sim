//! Quick test: Compare BH vs TreePM forces at same configuration
//!
//! Uses velocity change as a proxy for force magnitude.
//! If forces differ significantly, there's a force calculation bug.

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Force Comparison: BH vs TreePM ===\n");

    let n = 10_000;  // Small for quick test
    let box_size = 100.0;
    let r_cut = box_size / 4.0;  // Match diagnostic

    println!("Parameters: N={}, box={}, r_cut={}", n, box_size, r_cut);

    // Create simulation with zero initial velocities
    let mut sim1 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim1.set_theta(0.5);

    // Get particle data and create with zero velocities
    let (positions, _velocities, signs) = sim1.get_particles().expect("get_particles failed");

    // BH simulation
    let mut sim_bh = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        vec![0.0f32; n * 3],  // Zero initial velocities
        signs.clone(),
        box_size
    ).expect("GPU init failed");
    sim_bh.set_theta(0.5);

    // TreePM simulation
    let mut sim_treepm = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        vec![0.0f32; n * 3],  // Zero initial velocities
        signs.clone(),
        box_size
    ).expect("GPU init failed");
    sim_treepm.set_theta(0.5);

    // Initial KE should be 0
    let ke_bh_0 = sim_bh.kinetic_energy().unwrap();
    let ke_treepm_0 = sim_treepm.kinetic_energy().unwrap();
    println!("Initial KE (should be ~0):");
    println!("  BH:     {:.6e}", ke_bh_0);
    println!("  TreePM: {:.6e}", ke_treepm_0);

    // Run single step with small dt, no Hubble friction
    let dt = 0.01;
    println!("\nRunning single step (dt={})...", dt);

    sim_bh.step_dkd(dt, 0.0, 0.0).unwrap();
    sim_treepm.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();

    // KE after one step reflects force magnitude
    let ke_bh_1 = sim_bh.kinetic_energy().unwrap();
    let ke_treepm_1 = sim_treepm.kinetic_energy().unwrap();

    let acc_bh = sim_bh.acceleration_sum().unwrap();
    let acc_treepm = sim_treepm.acceleration_sum().unwrap();

    println!("\nAfter 1 step:");
    println!("  BH KE:     {:.6e}", ke_bh_1);
    println!("  TreePM KE: {:.6e}", ke_treepm_1);
    println!("  BH |acc|:     {:.6e}", acc_bh);
    println!("  TreePM |acc|: {:.6e}", acc_treepm);

    let ke_ratio = ke_treepm_1 / ke_bh_1;
    let acc_ratio = acc_treepm / acc_bh;

    println!("\n=== Results ===");
    println!("  KE ratio (TreePM/BH):  {:.4}", ke_ratio);
    println!("  Acc ratio (TreePM/BH): {:.4}", acc_ratio);

    if acc_ratio > 0.85 && acc_ratio < 1.15 {
        println!("\n✓ Forces match within 15% - TreePM splitting is correct");
    } else if acc_ratio > 0.5 && acc_ratio < 2.0 {
        println!("\n? Forces differ by {:.0}% - moderate discrepancy", (1.0 - acc_ratio).abs() * 100.0);
        println!("  This may explain lower segregation in TreePM");
    } else {
        println!("\n✗ Large force discrepancy ({:.0}%) - bug in force calculation",
                 (1.0 - acc_ratio).abs() * 100.0);
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
