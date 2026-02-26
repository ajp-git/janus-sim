//! Minimal test: 3 particles to verify Janus cross-interaction
//!
//! Setup:
//! - Particle A (m+) at (0, 0, 0)
//! - Particle B (m+) at (10, 0, 0)
//! - Particle C (m-) at (0, 10, 0)
//!
//! Expected forces on A:
//! - From B (same sign): ATTRACTION toward +x
//! - From C (opposite sign): REPULSION toward -y
//!
//! Run: cargo run --release --features cuda --bin test_janus_force

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Test Janus Force: 3 particles ===\n");

    // Setup: 2 m+ and 1 m-
    let n_plus = 2;
    let n_minus = 1;
    let box_size = 100.0;

    // Positions: A at origin, B at (10,0,0), C at (0,10,0)
    let positions = vec![
        0.0, 0.0, 0.0,   // A (m+)
        10.0, 0.0, 0.0,  // B (m+)
        0.0, 10.0, 0.0,  // C (m-)
    ];

    let velocities = vec![0.0; 9];

    // Signs: A=+1, B=+1, C=-1
    let signs = vec![1, 1, -1];

    println!("Particle positions:");
    println!("  A (m+): ({}, {}, {})", positions[0], positions[1], positions[2]);
    println!("  B (m+): ({}, {}, {})", positions[3], positions[4], positions[5]);
    println!("  C (m-): ({}, {}, {})", positions[6], positions[7], positions[8]);
    println!("\nSigns: {:?}", signs);

    // Create simulation
    let mut sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, box_size,
        positions, velocities, signs
    )?;

    // Set very small theta for direct summation
    sim.set_theta(0.0);

    println!("\nSimulation params: theta=0.0 (direct sum)");

    // Get initial positions
    let pos0 = sim.get_positions()?;
    println!("\nInitial positions (from GPU):");
    for i in 0..3 {
        println!("  Particle {}: ({:.4}, {:.4}, {:.4})",
                 i, pos0[i*3], pos0[i*3+1], pos0[i*3+2]);
    }

    // Step with cross_factor = -1.0 (Janus repulsion)
    println!("\n--- Step with cross_factor = -1.0 (Janus) ---");
    sim.step_with_cross_factor(0.1, -1.0)?;

    let pos1 = sim.get_positions()?;
    let vel1 = sim.get_velocities()?;

    println!("\nAfter 1 step:");
    for i in 0..3 {
        let dx = pos1[i*3] - pos0[i*3];
        let dy = pos1[i*3+1] - pos0[i*3+1];
        let dz = pos1[i*3+2] - pos0[i*3+2];
        println!("  Particle {}: pos=({:.4}, {:.4}, {:.4}), Δ=({:+.4}, {:+.4}, {:+.4})",
                 i, pos1[i*3], pos1[i*3+1], pos1[i*3+2], dx, dy, dz);
        println!("              vel=({:.4}, {:.4}, {:.4})",
                 vel1[i*3], vel1[i*3+1], vel1[i*3+2]);
    }

    // Analysis
    println!("\n=== Analysis ===");

    // Particle A (m+):
    // - From B (m+ at +x): should be attracted → vel_x > 0
    // - From C (m- at +y): should be repelled → vel_y < 0
    let vel_a_x = vel1[0];
    let vel_a_y = vel1[1];

    println!("\nParticle A (m+) velocity:");
    println!("  vx = {:.6} (expect > 0 from B attraction)", vel_a_x);
    println!("  vy = {:.6} (expect < 0 from C repulsion)", vel_a_y);

    if vel_a_x > 0.0 {
        println!("  ✓ A attracted toward B (m+/m+ attraction works)");
    } else {
        println!("  ✗ A NOT attracted toward B — BUG!");
    }

    if vel_a_y < 0.0 {
        println!("  ✓ A repelled from C (m+/m- repulsion works)");
    } else {
        println!("  ✗ A NOT repelled from C — CROSS INTERACTION BUG!");
    }

    // Particle C (m-):
    // - From A (m+ at origin): should be repelled → vel_y > 0
    // - From B (m+ at +x): should be repelled → vel_y > 0, vel_x < 0
    let vel_c_y = vel1[7];

    println!("\nParticle C (m-) velocity:");
    println!("  vy = {:.6} (expect > 0 from repulsion by A and B)", vel_c_y);

    if vel_c_y > 0.0 {
        println!("  ✓ C repelled from m+ particles");
    } else {
        println!("  ✗ C NOT repelled — BUG!");
    }

    // Now test with cross_factor = +1.0 (no repulsion, all attraction)
    println!("\n--- Comparison: cross_factor = +1.0 (all attraction) ---");

    let positions2 = vec![
        0.0, 0.0, 0.0,
        10.0, 0.0, 0.0,
        0.0, 10.0, 0.0,
    ];
    let velocities2 = vec![0.0; 9];
    let signs2 = vec![1, 1, -1];

    let mut sim2 = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, box_size,
        positions2, velocities2, signs2
    )?;
    sim2.set_theta(0.0);

    sim2.step_with_cross_factor(0.1, 1.0)?;  // All attraction

    let vel2 = sim2.get_velocities()?;
    let vel_a_y_attract = vel2[1];

    println!("  Particle A vy with all attraction: {:.6}", vel_a_y_attract);
    println!("  Particle A vy with Janus:          {:.6}", vel_a_y);

    if vel_a_y_attract > 0.0 && vel_a_y < 0.0 {
        println!("\n✓ Cross-factor correctly switches attraction/repulsion");
    } else if (vel_a_y_attract - vel_a_y).abs() < 1e-10 {
        println!("\n✗ Cross-factor has NO EFFECT — BUG IN KERNEL!");
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA not enabled. Use: cargo run --release --features cuda --bin test_janus_force");
}
