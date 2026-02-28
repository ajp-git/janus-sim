//! Test: Run WITHOUT virialization (α=1)
//!
//! Theory: The reference S_max=0.694 came from gravitational collapse
//! because the analytical virialization was wrong (under-virialized).
//!
//! If this is correct, running with α=1 (no virialization) should show
//! rapid collapse and high segregation.

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Test: No Virialization (α=1) ===\n");
    println!("Theory: Reference S_max=0.694 came from gravitational collapse");
    println!("        because analytical virialization was wrong.\n");

    let n = 50_000;  // Smaller for speed
    let n_steps = 1000;
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    println!("Creating simulation WITHOUT virialization...\n");

    // Generate ICs manually to skip virialization
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let half = box_size / 2.0;
    let mut pos_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut vel_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut signs_data: Vec<i8> = Vec::with_capacity(n);

    // Random positions (no Zel'dovich to keep it simple)
    for i in 0..n {
        let x = rng.random::<f64>() * box_size - half;
        let y = rng.random::<f64>() * box_size - half;
        let z = rng.random::<f64>() * box_size - half;
        pos_data.extend([x as f32, y as f32, z as f32]);

        // Small random velocity (NOT virialized)
        let v_scale = 1.0;  // α=1
        let vx = (rng.random::<f64>() - 0.5) * v_scale;
        let vy = (rng.random::<f64>() - 0.5) * v_scale;
        let vz = (rng.random::<f64>() - 0.5) * v_scale;
        vel_data.extend([vx as f32, vy as f32, vz as f32]);

        // Alternate signs
        signs_data.push(if i < n / 2 { 1 } else { -1 });
    }

    // Shuffle signs randomly
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        signs_data.swap(i, j);
    }

    // Create simulation with custom ICs (no virialization)
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_data, vel_data, signs_data, box_size
    ).expect("GPU init failed");
    sim.set_theta(0.5);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    println!("Initial state (NO virialization, α=1):");
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg₀ = {:.4}\n", seg_0);

    let mut seg_max = seg_0;

    println!("Running {} steps with pure BH (step_dkd)...\n", n_steps);
    println!("{:>6} {:>8} {:>12} {:>10}", "Step", "z", "KE/KE₀", "Seg");
    println!("{}", "-".repeat(45));

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        sim.step_dkd(dt, hubble, dtau_per_dt).unwrap();

        if step % 100 == 0 || step <= 50 && step % 10 == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();
            if seg > seg_max { seg_max = seg; }

            println!("{:>6} {:>8.2} {:>12.4} {:>10.4}",
                     step, z, ke / ke_0, seg);

            // Stop if KE explodes
            if ke / ke_0 > 100.0 {
                println!("\n*** KE explosion - stopping ***");
                break;
            }
        }
    }

    println!("\n=== Results ===");
    println!("  Seg_0 = {:.4}", seg_0);
    println!("  Seg_max = {:.4}", seg_max);
    println!();

    if seg_max > 0.3 {
        println!("HIGH segregation with α=1 (no virialization)");
        println!("→ Confirms: Reference S_max=0.694 was from collapse, not Janus physics");
    } else if seg_max > 0.1 {
        println!("MODERATE segregation with α=1");
    } else {
        println!("LOW segregation even with α=1");
        println!("→ Need more steps or different parameters");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
