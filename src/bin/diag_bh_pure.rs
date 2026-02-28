//! Diagnostic: Pure BH segregation evolution
//!
//! Compare with reference: 2M run, theta=0.7, S_max=0.694 @ z=1.8
//! Test: Is Seg still increasing? What's Seg at z=1.8?

use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use janus::friedmann::{JanusParams, CosmoInterpolator};

fn main() {
    println!("=== Pure BH Segregation Diagnostic ===\n");

    // Reference: 2M, theta=0.7, S_max=0.694 @ z=1.8
    println!("Reference 2M: theta=0.7, S_max=0.694 @ z=1.8\n");

    let n = 100_000;  // Quick test
    let n_steps = 5000;  // Slower cosmological evolution
    let dt = 0.01;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
    let eta = 1.045;
    let z_init = 5.0;
    let theta = 0.7;  // Match reference

    println!("Test parameters:");
    println!("  N = {}", n);
    println!("  box = {:.1} Mpc", box_size);
    println!("  theta = {}", theta);
    println!("  steps = {}", n_steps);
    println!();

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    println!("Cosmology:");
    println!("  tau_start = {:.4}", cosmo.tau_start);
    println!("  tau_end = {:.4}", cosmo.tau_end);
    println!("  dtau_per_step = {:.6}", dtau_per_step);
    println!();

    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(theta);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("Initial: KE₀ = {:.4e}, Seg₀ = {:.4}", ke_0, seg_0);
    println!();

    let mut seg_max = seg_0;
    let mut seg_max_step = 0;
    let mut seg_max_z = z_init;
    let mut seg_increasing = true;
    let mut last_seg = seg_0;

    println!("{:>6} {:>8} {:>10} {:>10} {:>10}",
             "Step", "z", "KE/KE₀", "Seg", "Trend");
    println!("{}", "-".repeat(55));

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;
        let dtau_per_dt = dtau_per_step / dt;

        sim.step_dkd(dt, hubble, dtau_per_dt).unwrap();

        // Sample more frequently near z=1.8
        let should_print = step % 500 == 0
            || (z < 2.0 && z > 1.5 && step % 100 == 0)
            || step <= 100 && step % 20 == 0;

        if should_print || step == n_steps {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();

            let trend = if seg > last_seg + 0.001 {
                "↑"
            } else if seg < last_seg - 0.001 {
                "↓"
            } else {
                "→"
            };

            if seg > seg_max {
                seg_max = seg;
                seg_max_step = step;
                seg_max_z = z;
            }

            // Check if still increasing
            if seg < last_seg - 0.01 && step > 100 {
                seg_increasing = false;
            }

            println!("{:>6} {:>8.3} {:>10.4} {:>10.4} {:>10}",
                     step, z, ke / ke_0, seg, trend);

            last_seg = seg;
        }
    }

    println!("\n=== Results ===");
    println!("  S_max = {:.4} at step {} (z = {:.2})", seg_max, seg_max_step, seg_max_z);
    println!("  Reference: S_max = 0.694 at z = 1.8");
    println!();

    if seg_max > 0.5 {
        println!("✓ Segregation matches reference range (>0.5)");
    } else if seg_max > 0.3 {
        println!("? Moderate segregation ({:.2}) - may need more particles or steps", seg_max);
    } else {
        println!("✗ Low segregation ({:.2}) - investigate parameters", seg_max);
    }

    if !seg_increasing {
        println!("  Segregation peaked and is now decreasing");
    } else {
        println!("  Segregation may still be increasing - run more steps");
    }
}
