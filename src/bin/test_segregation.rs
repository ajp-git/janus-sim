//! Test segregation growth over 3000 steps
//!
//! Expected: Seg should grow (slowly) from ~0 due to Janus +/- repulsion

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Segregation Growth Test ===\n");
    println!("Testing if Seg grows from ~0 over 3000 steps");
    println!("With η≈1.045, growth should be slow but measurable\n");

    let n = 100_000;  // 50k each sign
    let n_steps = 3000;
    let dt = 0.01;
    let box_size = 200.0;  // Smaller box for stronger interactions
    let eta = 1.045;
    let z_init = 5.0;
    let r_cut = box_size / 8.0;

    // Setup CosmoInterpolator
    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);

    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    println!("Parameters:");
    println!("  N = {} ({}k each sign)", n, n/2/1000);
    println!("  box = {} Mpc, r_cut = {:.1} Mpc", box_size, r_cut);
    println!("  η = {}, z_init = {}", eta, z_init);
    println!("  n_steps = {}, dt = {}", n_steps, dt);
    println!();

    // Create sim
    println!("Creating simulation...\n");
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(0.5);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("Initial state: KE={:.4e}, Seg={:.4}\n", ke_0, seg_0);

    // Track segregation evolution
    let mut seg_max = seg_0;
    let mut seg_history: Vec<f64> = vec![seg_0];

    println!("Running {} steps...\n", n_steps);

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        sim.step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt).unwrap();

        if step % 500 == 0 || step <= 50 && step % 10 == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();
            seg_history.push(seg);
            if seg > seg_max { seg_max = seg; }

            println!("  Step {:4}: z={:.2}  KE/KE₀={:.4}  Seg={:.4}  (max={:.4})",
                     step, z, ke / ke_0, seg, seg_max);
        }
    }

    let ke_final = sim.kinetic_energy().unwrap();
    let seg_final = sim.segregation().unwrap();

    println!("\n=== Results ===");
    println!("  Seg_0 = {:.4}", seg_0);
    println!("  Seg_max = {:.4}", seg_max);
    println!("  Seg_final = {:.4}", seg_final);
    println!("  KE/KE₀ = {:.4}", ke_final / ke_0);

    if seg_max > seg_0 * 2.0 && seg_max > 0.01 {
        println!("\n✓ PASS: Segregation is growing (Seg_max > 2×Seg_0)");
    } else if seg_max > seg_0 * 1.5 {
        println!("\n? MARGINAL: Segregation grew but slowly");
    } else {
        println!("\n✗ FAIL: No significant segregation growth");
        println!("  This may be expected for η≈1 (nearly symmetric populations)");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
