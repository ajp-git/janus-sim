//! KE diagnostic - verify virialization and Hubble friction
//!
//! cargo build --release --features cuda,cufft --bin diag_ke
//! LD_LIBRARY_PATH=target/release ./target/release/diag_ke

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== KE Diagnostic ===\n");

    // Small test: 10K particles, 20 steps
    let n = 10_000;
    let box_size = 100.0;
    let dt = 0.01;
    let r_cut = box_size / 16.0;

    println!("Creating sim with {} particles...", n);
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(0.5);

    // Check KE immediately after virialization
    let ke_0 = sim.kinetic_energy().expect("KE failed");
    println!("\nKE after virialization (before any step): {:.6e}", ke_0);

    if ke_0 < 1e-10 {
        println!("ERROR: KE is essentially zero! Virialization not working.");
        return;
    }
    println!("OK: KE > 0\n");

    // Test with different Hubble friction strengths
    println!("Testing Hubble friction damping...\n");

    // Test 1: No friction (H=0)
    println!("Test 1: H=0 (no friction)");
    let mut sim1 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim1.set_theta(0.5);
    let ke_init = sim1.kinetic_energy().unwrap();
    for step in 0..10 {
        sim1.step_treepm_gpu(dt, r_cut, 0.0, 1.0).unwrap();  // H=0
        let ke = sim1.kinetic_energy().unwrap();
        println!("  Step {}: KE/KE₀ = {:.6}", step+1, ke / ke_init);
    }

    // Test 2: Strong friction (H=5.6, dtau_per_dt=1.0) - current settings
    println!("\nTest 2: H=5.6, dtau_per_dt=1.0 (current, too strong?)");
    let mut sim2 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim2.set_theta(0.5);
    let ke_init = sim2.kinetic_energy().unwrap();
    for step in 0..10 {
        sim2.step_treepm_gpu(dt, r_cut, 5.6, 1.0).unwrap();
        let ke = sim2.kinetic_energy().unwrap();
        println!("  Step {}: KE/KE₀ = {:.6}", step+1, ke / ke_init);
    }

    // Test 3: Weak friction (H=5.6, dtau_per_dt=0.01)
    println!("\nTest 3: H=5.6, dtau_per_dt=0.01 (weaker)");
    let mut sim3 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim3.set_theta(0.5);
    let ke_init = sim3.kinetic_energy().unwrap();
    for step in 0..10 {
        sim3.step_treepm_gpu(dt, r_cut, 5.6, 0.01).unwrap();
        let ke = sim3.kinetic_energy().unwrap();
        println!("  Step {}: KE/KE₀ = {:.6}", step+1, ke / ke_init);
    }

    // Test 4: Physical H (H=0.07 in code units?)
    println!("\nTest 4: H=0.07, dtau_per_dt=1.0 (physical H?)");
    let mut sim4 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim4.set_theta(0.5);
    let ke_init = sim4.kinetic_energy().unwrap();
    for step in 0..10 {
        sim4.step_treepm_gpu(dt, r_cut, 0.07, 1.0).unwrap();
        let ke = sim4.kinetic_energy().unwrap();
        println!("  Step {}: KE/KE₀ = {:.6}", step+1, ke / ke_init);
    }

    println!("\n=== Diagnostic Complete ===");
    println!("Expected: KE should decrease gradually with friction, not instantly to 0");
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
