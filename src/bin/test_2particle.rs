//! Minimal test: 2 particles (1+, 1-) - verify they repel
//!
//! This is the simplest possible test for Janus +/- repulsion

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== 2-Particle Repulsion Test ===\n");
    println!("Initial: + at x=-20, - at x=+20 (both at rest)");
    println!("Expected: particles should REPEL (separation increases)\n");

    let box_size = 100.0;
    let r_cut = box_size / 8.0;  // 12.5 Mpc
    let dt = 0.001;  // Small dt for accuracy
    let n_steps = 1000;

    // Just 2 particles: + at left, - at right
    let pos_data: Vec<f32> = vec![
        -20.0, 0.0, 0.0,  // + particle
        20.0, 0.0, 0.0,   // - particle
    ];
    let vel_data: Vec<f32> = vec![
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let signs_data: Vec<i8> = vec![1, -1];

    let sep_0 = 40.0_f32;

    println!("Initial state:");
    println!("  + particle: x = -20");
    println!("  - particle: x = +20");
    println!("  Separation = {} Mpc\n", sep_0);

    // Create simulation
    println!("Creating GPU simulation (2 particles)...\n");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_data.clone(),
        vel_data.clone(),
        signs_data.clone(),
        box_size,
    ).expect("GPU init failed");
    sim.set_theta(0.5);

    println!("Running {} steps...\n", n_steps);

    for step in 1..=n_steps {
        // No Hubble friction
        sim.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();

        if step % 100 == 0 || step <= 20 || step == 50 {
            let (positions, velocities, _) = sim.get_particles().unwrap();

            let x_plus = positions[0];
            let x_minus = positions[3];
            let vx_plus = velocities[0];
            let vx_minus = velocities[3];
            let sep = (x_minus - x_plus).abs();

            let direction = if (x_minus - x_plus) > sep_0 { "↑ REPELLING" } else { "↓ attracting" };

            println!("  Step {:4}: x+={:+.3} v+={:+.4}  x-={:+.3} v-={:+.4}  Sep={:.3} {}",
                     step, x_plus, vx_plus, x_minus, vx_minus, sep, direction);
        }
    }

    let (positions, velocities, _) = sim.get_particles().unwrap();
    let x_plus = positions[0];
    let x_minus = positions[3];
    let vx_plus = velocities[0];
    let vx_minus = velocities[3];
    let sep_final = (x_minus - x_plus).abs();

    println!("\n=== Results ===");
    println!("  + particle: x = {:+.3}, vx = {:+.5}", x_plus, vx_plus);
    println!("  - particle: x = {:+.3}, vx = {:+.5}", x_minus, vx_minus);
    println!("  Initial separation: {} Mpc", sep_0);
    println!("  Final separation:   {:.3} Mpc", sep_final);
    println!("  Change: {:+.3} Mpc", sep_final - sep_0);

    if sep_final > sep_0 + 0.1 {
        println!("\n✓ PASS: Particles are REPELLING");
    } else if sep_final < sep_0 - 0.1 {
        println!("\n✗ FAIL: Particles are ATTRACTING (wrong physics!)");
    } else {
        println!("\n? UNCLEAR: Separation change too small (forces may be weak)");
    }

    // Also print velocity directions
    println!("\nVelocity analysis:");
    if vx_plus < 0.0 {
        println!("  + particle moving LEFT (away from -) ✓");
    } else {
        println!("  + particle moving RIGHT (toward -) ✗");
    }
    if vx_minus > 0.0 {
        println!("  - particle moving RIGHT (away from +) ✓");
    } else {
        println!("  - particle moving LEFT (toward +) ✗");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
