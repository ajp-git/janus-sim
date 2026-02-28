//! Debug: Test TreePM components separately
//!
//! Compare:
//! 1. BH full range
//! 2. TreePM short-range only (no PM)
//! 3. TreePM full (short-range + PM)

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Force Debug: TreePM Components ===\n");

    let n = 10_000;
    let box_size = 100.0;
    let r_cut = box_size / 8.0;  // 12.5 Mpc
    let r_s = r_cut * 0.4;        // 5 Mpc

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  box = {} Mpc", box_size);
    println!("  r_cut = {} Mpc", r_cut);
    println!("  r_s = {} Mpc (short-range smoothing)", r_s);
    println!();

    // Create simulation with zero velocities
    let sim_init = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    let (positions, _velocities, signs) = sim_init.get_particles().expect("get failed");

    // Test 1: Full BH (reference)
    println!("Test 1: Full BH (no splitting)...");
    let mut sim_bh = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(), vec![0.0f32; n * 3], signs.clone(), box_size
    ).expect("init failed");
    sim_bh.set_theta(0.5);
    sim_bh.step_dkd(0.01, 0.0, 0.0).unwrap();
    let ke_bh = sim_bh.kinetic_energy().unwrap();
    let acc_bh = sim_bh.acceleration_sum().unwrap();
    println!("  KE = {:.6e}, |acc| = {:.6e}", ke_bh, acc_bh);

    // Test 2: TreePM full (short-range + PM)
    println!("\nTest 2: TreePM (short-range + PM)...");
    let mut sim_treepm = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(), vec![0.0f32; n * 3], signs.clone(), box_size
    ).expect("init failed");
    sim_treepm.set_theta(0.5);
    sim_treepm.step_treepm_gpu(0.01, r_cut, 0.0, 0.0).unwrap();
    let ke_treepm = sim_treepm.kinetic_energy().unwrap();
    let acc_treepm = sim_treepm.acceleration_sum().unwrap();
    println!("  KE = {:.6e}, |acc| = {:.6e}", ke_treepm, acc_treepm);

    // Test 3: Short-range only (via compute_short_range_forces - need to expose this)
    // For now, infer PM contribution
    println!("\n=== Analysis ===");
    println!("  BH reference |acc|: {:.6e}", acc_bh);
    println!("  TreePM |acc|:       {:.6e} ({:.1}% of BH)", acc_treepm, 100.0 * acc_treepm / acc_bh);

    // Expected erfc weight at mean separation
    // For uniform N particles in box L, mean separation ~ (L³/N)^(1/3)
    let mean_sep = (box_size.powi(3) / (n as f64)).powf(1.0/3.0);
    println!("\n  Mean particle separation: {:.2} Mpc", mean_sep);

    let x = mean_sep / (2.0 * r_s);
    // Approximate erfc using polynomial (Abramowitz & Stegun)
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let erfc_approx = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * (-x*x).exp();
    println!("  erfc({:.3}) ≈ {:.4} (expected short-range weight at mean sep)", x, erfc_approx);

    // If TreePM is correctly splitting, acc_treepm should ≈ acc_bh
    // If acc_treepm < acc_bh, either PM is missing or erfc is too aggressive

    let missing_force = acc_bh - acc_treepm;
    let missing_pct = 100.0 * missing_force / acc_bh;
    println!("\n  Missing force: {:.6e} ({:.1}% of BH)", missing_force, missing_pct);

    if acc_treepm / acc_bh > 0.95 {
        println!("\n✓ TreePM ≈ BH - splitting is correct");
    } else if acc_treepm / acc_bh > 0.80 {
        println!("\n? TreePM underestimates forces by {:.0}%", missing_pct);
        println!("  → PM contribution may be too weak");
        println!("  → Or erfc weight is too aggressive at short range");
    } else {
        println!("\n✗ Large force deficit - check PM or splitting");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
