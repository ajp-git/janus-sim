//! Test PM repulsion: + left, - right → separation should INCREASE
//!
//! This is the critical test for BUG 2: verify PM correctly repels +/- populations

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== PM Repulsion Test ===\n");
    println!("Initial: + particles on LEFT (x < 0), - particles on RIGHT (x > 0)");
    println!("Expected: After evolution, separation should INCREASE (repulsion)\n");

    let n = 50_000;  // 25k each sign
    let box_size = 100.0;
    let r_cut = box_size / 8.0;  // 12.5 Mpc
    let dt = 0.005;
    let n_steps = 500;

    // Create custom initial conditions: + left, - right
    use rand::Rng;
    let mut rng = rand::rng();
    let half = box_size / 2.0;

    let mut pos_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut vel_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut signs_data: Vec<i8> = Vec::with_capacity(n);

    // First n/2: + particles in LEFT half (x in [-50, 0])
    for _ in 0..n/2 {
        let x = rng.random::<f64>() * half - half;  // x in [-50, 0]
        let y = rng.random::<f64>() * box_size - half;  // y in [-50, 50]
        let z = rng.random::<f64>() * box_size - half;  // z in [-50, 50]
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([0.0f32, 0.0f32, 0.0f32]);
        signs_data.push(1);
    }

    // Second n/2: - particles in RIGHT half (x in [0, 50])
    for _ in 0..n/2 {
        let x = rng.random::<f64>() * half;  // x in [0, 50]
        let y = rng.random::<f64>() * box_size - half;
        let z = rng.random::<f64>() * box_size - half;
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([0.0f32, 0.0f32, 0.0f32]);
        signs_data.push(-1);
    }

    // Compute initial COM for each population
    let com_plus_x_0: f32 = pos_data.chunks(3)
        .zip(signs_data.iter())
        .filter(|(_, &s)| s > 0)
        .map(|(p, _)| p[0])
        .sum::<f32>() / (n/2) as f32;

    let com_minus_x_0: f32 = pos_data.chunks(3)
        .zip(signs_data.iter())
        .filter(|(_, &s)| s < 0)
        .map(|(p, _)| p[0])
        .sum::<f32>() / (n/2) as f32;

    let sep_0 = (com_minus_x_0 - com_plus_x_0).abs();

    println!("Initial state:");
    println!("  COM(+).x = {:.2}", com_plus_x_0);
    println!("  COM(-).x = {:.2}", com_minus_x_0);
    println!("  Separation = {:.2} Mpc\n", sep_0);

    // Create simulation with these ICs
    println!("Creating GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_data.clone(),
        vel_data.clone(),
        signs_data.clone(),
        box_size,
    ).expect("GPU init failed");
    sim.set_theta(0.5);

    // No virialization - start with zero velocities
    // This tests pure PM/BH force dynamics

    println!("Running {} steps with TreePM...\n", n_steps);

    // Track evolution
    let mut separations: Vec<f64> = vec![sep_0 as f64];

    for step in 1..=n_steps {
        // Use H=0, dtau_per_dt=0 to disable Hubble friction (pure N-body)
        sim.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();

        if step % 100 == 0 || step <= 10 {
            // Get positions back from GPU
            let (positions, _, signs) = sim.get_particles().unwrap();

            // Compute current COM
            let com_plus_x: f32 = positions.chunks(3)
                .zip(signs.iter())
                .filter(|(_, &s)| s > 0)
                .map(|(p, _)| p[0])
                .sum::<f32>() / (n/2) as f32;

            let com_minus_x: f32 = positions.chunks(3)
                .zip(signs.iter())
                .filter(|(_, &s)| s < 0)
                .map(|(p, _)| p[0])
                .sum::<f32>() / (n/2) as f32;

            let sep = (com_minus_x - com_plus_x).abs();
            separations.push(sep as f64);

            let delta_sep = sep - sep_0;
            let direction = if delta_sep > 0.0 { "↑ SEPARATING" } else { "↓ approaching" };

            println!("  Step {:4}: COM(+).x={:+.2}  COM(-).x={:+.2}  Sep={:.2}  Δ={:+.3} {}",
                     step, com_plus_x, com_minus_x, sep, delta_sep, direction);
        }
    }

    // Final assessment
    println!("\n=== Results ===");
    let sep_final = separations.last().unwrap();
    let delta = sep_final - sep_0 as f64;

    println!("  Initial separation: {:.2} Mpc", sep_0);
    println!("  Final separation:   {:.2} Mpc", sep_final);
    println!("  Change: {:+.3} Mpc ({:+.1}%)", delta, 100.0 * delta / sep_0 as f64);

    if delta > 0.5 {
        println!("\n✓ PASS: Populations are SEPARATING (PM repulsion works)");
    } else if delta < -0.5 {
        println!("\n✗ FAIL: Populations are APPROACHING (PM attraction - WRONG!)");
        println!("  → Check sign in cic_gather kernel");
    } else {
        println!("\n? UNCLEAR: Separation change too small");
        println!("  → May need more steps or stronger initial separation");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
