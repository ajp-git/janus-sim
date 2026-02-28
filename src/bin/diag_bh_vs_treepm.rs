//! Diagnostic: Pure BH vs TreePM with identical ICs
//!
//! This test runs the SAME initial conditions through:
//! 1. Pure BH (step_dkd) - should match reference S_max=0.694
//! 2. TreePM (step_treepm_gpu) - if different, problem is in TreePM
//!
//! If both show low segregation, problem is in ICs/virialization

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Diagnostic: Pure BH vs TreePM ===\n");
    println!("Reference BH run: S_max = 0.694");
    println!("Parameters: N=100K, η=1.045, box=100, z_init=5, 2000 steps\n");

    let n = 100_000;
    let n_steps = 2000; // Match reference run
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;
    let r_cut = box_size / 4.0;  // Larger r_cut for more BH coverage

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    println!("Creating simulation...\n");
    let mut sim_bh = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim_bh.set_theta(0.5);

    // Get initial particle data
    let (positions, velocities, signs) = sim_bh.get_particles().expect("get_particles failed");

    // Create second simulation with identical ICs
    let mut sim_treepm = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        velocities.clone(),
        signs.clone(),
        box_size
    ).expect("TreePM init failed");
    sim_treepm.set_theta(0.5);

    let ke_0 = sim_bh.kinetic_energy().expect("KE failed");
    let seg_bh_0 = sim_bh.segregation().expect("Seg failed");
    let seg_pm_0 = sim_treepm.segregation().expect("Seg failed");

    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg_BH₀ = {:.4}", seg_bh_0);
    println!("  Seg_PM₀ = {:.4} (should match)\n", seg_pm_0);

    let mut seg_max_bh = seg_bh_0;
    let mut seg_max_pm = seg_pm_0;

    println!("Running {} steps...\n", n_steps);
    println!("{:>6} {:>8} {:>10} {:>10} {:>10} {:>10}",
             "Step", "z", "KE_BH/KE₀", "KE_PM/KE₀", "Seg_BH", "Seg_PM");
    println!("{}", "-".repeat(60));

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        // Run BH step
        sim_bh.step_dkd(dt, hubble, dtau_per_dt).unwrap();

        // Run TreePM step
        sim_treepm.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt).unwrap();

        if step % 100 == 0 || step <= 20 && step % 5 == 0 {
            let ke_bh = sim_bh.kinetic_energy().unwrap();
            let ke_pm = sim_treepm.kinetic_energy().unwrap();
            let seg_bh = sim_bh.segregation().unwrap();
            let seg_pm = sim_treepm.segregation().unwrap();

            if seg_bh > seg_max_bh { seg_max_bh = seg_bh; }
            if seg_pm > seg_max_pm { seg_max_pm = seg_pm; }

            println!("{:>6} {:>8.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                     step, z, ke_bh / ke_0, ke_pm / ke_0, seg_bh, seg_pm);
        }
    }

    let seg_bh_final = sim_bh.segregation().unwrap();
    let seg_pm_final = sim_treepm.segregation().unwrap();

    println!("\n=== Results ===");
    println!("  Pure BH:  Seg_max = {:.4}  Seg_final = {:.4}", seg_max_bh, seg_bh_final);
    println!("  TreePM:   Seg_max = {:.4}  Seg_final = {:.4}", seg_max_pm, seg_pm_final);
    println!("\n  Reference BH: S_max = 0.694");
    println!();

    if seg_max_bh > 0.3 && seg_max_pm < 0.1 {
        println!("→ BH shows segregation, TreePM doesn't");
        println!("  → BUG IS IN TreePM FORCES");
    } else if seg_max_bh > 0.3 && seg_max_pm > 0.3 {
        println!("→ Both show segregation");
        println!("  → BOTH METHODS WORKING (check parameters match reference)");
    } else if seg_max_bh < 0.1 && seg_max_pm < 0.1 {
        println!("→ Neither shows segregation");
        println!("  → BUG IS IN ICs/VIRIALIZATION (not forces)");
    } else {
        println!("→ Mixed results - more investigation needed");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
