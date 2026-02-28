//! Regression test: Pure BH vs BH reference (S_max=0.694)
//!
//! Reference BH run parameters:
//!   N=100K, η=1.045, box=100 Mpc, z_init=5, 2000 steps
//!   Result: S_max=0.694
//!
//! This test uses pure BH (step_dkd) with same parameters.
//! TreePM has a force splitting bug (double counting) - use pure BH.
//! If Seg_max > 0.3 → physics correct
//! If Seg_max < 0.1 → bug in implementation

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Regression Test: Pure BH vs BH Reference ===\n");
    println!("BH Reference: S_max = 0.694");
    println!("Parameters: N=100K, η=1.045, box=100, z_init=5, 2000 steps\n");

    // Exact BH reference parameters
    let n = 100_000;
    let n_steps = 2000;
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;

    // Setup CosmoInterpolator
    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);

    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    println!("Parameters:");
    println!("  N = {} (50k each sign)", n);
    println!("  box = {} Mpc", box_size);
    println!("  η = {}", eta);
    println!("  z_init = {}", z_init);
    println!("  n_steps = {}", n_steps);
    println!("  dt = {}", dt);
    println!("  Method: Pure BH (step_dkd)");
    println!();

    // Create sim (no alpha virialization - uses virial_velocity directly like reference)
    println!("Creating simulation with virial_velocity (no alpha scaling)...\n");
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(0.5);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("\nInitial: KE={:.4e}, Seg={:.4}", ke_0, seg_0);

    // Track segregation
    let mut seg_max = seg_0;
    let mut seg_at_step = 0;

    println!("\nRunning {} steps with pure BH (step_dkd)...\n", n_steps);

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        // Use pure BH instead of TreePM (TreePM has force splitting bug)
        sim.step_dkd(dt, hubble, dtau_per_dt).unwrap();

        if step % 200 == 0 || step <= 20 && step % 5 == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();
            if seg > seg_max {
                seg_max = seg;
                seg_at_step = step;
            }

            println!("  Step {:4}: z={:.2}  KE/KE₀={:.4}  Seg={:.4}  (max={:.4})",
                     step, z, ke / ke_0, seg, seg_max);
        }
    }

    let seg_final = sim.segregation().unwrap();
    let ke_final = sim.kinetic_energy().unwrap();

    println!("\n=== Results ===");
    println!("  Seg_0 = {:.4}", seg_0);
    println!("  Seg_max = {:.4} (at step {})", seg_max, seg_at_step);
    println!("  Seg_final = {:.4}", seg_final);
    println!("  KE/KE₀ = {:.4}", ke_final / ke_0);
    println!();
    println!("  BH Reference: S_max = 0.694");
    println!();

    if seg_max > 0.3 {
        println!("✓ PASS: Seg_max > 0.3 → Physics correct");
        println!("        Pure BH reproduces reference behavior");
    } else if seg_max > 0.1 {
        println!("? MARGINAL: 0.1 < Seg_max < 0.3");
        println!("        May need more steps or parameter tuning");
    } else {
        println!("✗ FAIL: Seg_max < 0.1 → Bug in implementation");
        println!("        Check ICs, force calculation, or integrator");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
