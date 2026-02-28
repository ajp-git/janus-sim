//! Diagnostic: Compare uniform random vs Zel'dovich ICs
//!
//! The reference BH run (S_max=0.694) used uniform random positions.
//! The current GpuNBodyTwoPass uses Zel'dovich perturbations.
//!
//! Test if this difference affects segregation.

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Uniform Random vs Zel'dovich ICs ===\n");

    let n = 100_000;
    let n_steps = 1000;
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    // Generate UNIFORM RANDOM ICs (like reference GpuNBodySimulation)
    println!("Creating simulation with UNIFORM RANDOM ICs (like reference)...\n");

    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let n_positive = n / 2;
    let n_negative = n - n_positive;
    let half = box_size / 2.0;

    // Match reference: virial_velocity = sqrt(N/box) × 0.3
    let virial_velocity = ((n as f64) / box_size).sqrt() * 0.3;

    let mut pos_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut vel_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut signs_data: Vec<i8> = Vec::with_capacity(n);

    // Generate + particles first (like reference)
    for _ in 0..n_positive {
        let x = (rng.random::<f64>() - 0.5) * box_size;
        let y = (rng.random::<f64>() - 0.5) * box_size;
        let z = (rng.random::<f64>() - 0.5) * box_size;
        let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([vx as f32, vy as f32, vz as f32]);
        signs_data.push(1);
    }

    // Then - particles (like reference)
    for _ in 0..n_negative {
        let x = (rng.random::<f64>() - 0.5) * box_size;
        let y = (rng.random::<f64>() - 0.5) * box_size;
        let z = (rng.random::<f64>() - 0.5) * box_size;
        let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([vx as f32, vy as f32, vz as f32]);
        signs_data.push(-1);
    }

    // Create simulation with custom ICs
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_data, vel_data, signs_data, box_size
    ).expect("GPU init failed");
    sim.set_theta(0.5);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("Initial state (uniform random ICs):");
    println!("  virial_velocity = {:.4}", virial_velocity);
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg₀ = {:.4}\n", seg_0);

    let mut seg_max = seg_0;

    println!("Running {} steps with pure BH...\n", n_steps);
    println!("{:>6} {:>8} {:>12} {:>10}", "Step", "z", "KE/KE₀", "Seg");
    println!("{}", "-".repeat(45));

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        sim.step_dkd(dt, hubble, dtau_per_dt).unwrap();

        if step % 100 == 0 || step == 10 || step == 50 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();
            if seg > seg_max { seg_max = seg; }

            println!("{:>6} {:>8.2} {:>12.4} {:>10.4}", step, z, ke / ke_0, seg);
        }
    }

    println!("\n=== Results ===");
    println!("  Seg_0 = {:.4}", seg_0);
    println!("  Seg_max = {:.4}", seg_max);
    println!();

    if seg_max > 0.1 {
        println!("HIGH segregation with uniform random ICs");
        println!("→ Zel'dovich ICs were the problem");
    } else if seg_max > 0.01 {
        println!("MODERATE segregation");
    } else {
        println!("LOW segregation even with uniform random ICs");
        println!("→ Problem is NOT in ICs");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
