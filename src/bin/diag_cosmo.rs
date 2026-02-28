//! Quick diagnostic: CosmoInterpolator + 100 steps
//! Verify KE/KE₀ at step 100 is between 0.8 and 1.0

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== CosmoInterpolator Diagnostic ===\n");

    let n = 10_000;
    let n_steps = 100;
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;
    let r_cut = box_size / 16.0;

    // Setup CosmoInterpolator
    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);

    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    println!("CosmoInterpolator setup:");
    println!("  tau_start: {:.6}", cosmo.tau_start);
    println!("  tau_end: {:.6}", cosmo.tau_end);
    println!("  dtau_per_dt: {:.6}", dtau_per_dt);
    println!("  dtau_per_step: {:.6}", dtau_per_step);
    println!();

    // Create sim
    println!("Creating sim with {} particles...", n);
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(0.5);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    println!("\nKE_0 = {:.6e}\n", ke_0);

    println!("Running {} steps...\n", n_steps);
    for step in 0..n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        sim.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt).unwrap();

        if (step + 1) % 10 == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();
            println!("  Step {:3}: z={:.2}  a={:.4}  H={:.4}  KE/KE₀={:.4}  Seg={:.4}",
                     step + 1, z, a, hubble, ke / ke_0, seg);
        }
    }

    let ke_final = sim.kinetic_energy().unwrap();
    let ke_ratio = ke_final / ke_0;

    println!("\n=== Result ===");
    println!("  KE/KE₀ at step {}: {:.4}", n_steps, ke_ratio);
    println!("  Expected: 0.8 < KE/KE₀ < 1.0 for gradual cooling");

    if ke_ratio > 0.8 && ke_ratio < 1.0 {
        println!("  ✓ PASS: Gradual Hubble cooling");
    } else if ke_ratio < 0.8 {
        println!("  ✗ FAIL: Cooling too fast");
    } else {
        println!("  ✗ FAIL: No cooling (KE increasing?)");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
