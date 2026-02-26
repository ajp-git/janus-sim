//! Test Janus interactions: 3 particles
//!
//! Config 1: m+ at origin, m+ at (10,0,0) → m+ should move toward origin
//! Config 2: m+ at origin, m- at (10,0,0) → m- should move away from origin
//! Config 3: m- at origin, m- at (10,0,0) → m- should move away (self-repulsion)
//!
//! cargo run --release --features cuda --bin test_janus_3body

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn run_test(
    name: &str,
    positions: Vec<f64>,
    signs: Vec<i32>,
    n_plus: usize,
    n_minus: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", name);

    let box_size = 100.0;
    let velocities = vec![0.0; positions.len()];

    let mut sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, box_size,
        positions.clone(), velocities, signs.clone()
    )?;
    sim.set_theta(0.0);  // Direct summation

    println!("Initial:");
    for (i, sign) in signs.iter().enumerate() {
        let s = if *sign > 0 { "m+" } else { "m-" };
        println!("  P{} ({}): ({:.1}, {:.1}, {:.1})",
                 i, s, positions[i*3], positions[i*3+1], positions[i*3+2]);
    }

    // Step 1
    sim.step_with_cross_factor(0.1, -1.0)?;
    let pos1 = sim.get_positions()?;
    let vel1 = sim.get_velocities()?;

    println!("\nAfter step 1:");
    for (i, sign) in signs.iter().enumerate() {
        let s = if *sign > 0 { "m+" } else { "m-" };
        let dx = pos1[i*3] - positions[i*3];
        println!("  P{} ({}): vel=({:+.6}, {:+.6}, {:+.6}), Δx={:+.6}",
                 i, s, vel1[i*3], vel1[i*3+1], vel1[i*3+2], dx);
    }

    // Step 2
    sim.step_with_cross_factor(0.1, -1.0)?;
    let pos2 = sim.get_positions()?;
    let vel2 = sim.get_velocities()?;

    println!("\nAfter step 2:");
    for (i, sign) in signs.iter().enumerate() {
        let s = if *sign > 0 { "m+" } else { "m-" };
        let dx = pos2[i*3] - positions[i*3];
        println!("  P{} ({}): vel=({:+.6}, {:+.6}, {:+.6}), Δx={:+.6}",
                 i, s, vel2[i*3], vel2[i*3+1], vel2[i*3+2], dx);
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA not enabled. Use: cargo run --release --features cuda --bin test_janus_3body");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==============================================");
    println!("Test Janus 3-body interactions");
    println!("==============================================");

    // Test 1: m+/m+ attraction
    // P0 (m+) at origin (fixed reference)
    // P1 (m+) at (10,0,0) → should move toward origin (vx < 0)
    run_test(
        "Test 1: m+/m+ (expect attraction → vx < 0 for P1)",
        vec![
            0.0, 0.0, 0.0,   // P0 m+
            10.0, 0.0, 0.0,  // P1 m+
        ],
        vec![1, 1],
        2, 0,  // 2 m+, 0 m-
    )?;

    // Test 2: m+/m- repulsion
    // P0 (m+) at origin
    // P1 (m-) at (10,0,0) → should move away from origin (vx > 0)
    run_test(
        "Test 2: m+/m- (expect repulsion → vx > 0 for P1)",
        vec![
            0.0, 0.0, 0.0,   // P0 m+
            10.0, 0.0, 0.0,  // P1 m-
        ],
        vec![1, -1],
        1, 1,  // 1 m+, 1 m-
    )?;

    // Test 3: m-/m- interaction (CRITICAL)
    // P0 (m-) at origin
    // P1 (m-) at (10,0,0) → should move AWAY (vx > 0) due to self-repulsion
    // In Janus: m- has negative inertial mass, so even though gravitational
    // force is "attractive", the particle accelerates in opposite direction
    run_test(
        "Test 3: m-/m- (expect self-repulsion → vx > 0 for P1)",
        vec![
            0.0, 0.0, 0.0,   // P0 m-
            10.0, 0.0, 0.0,  // P1 m-
        ],
        vec![-1, -1],
        0, 2,  // 0 m+, 2 m-
    )?;

    // Summary
    println!("\n==============================================");
    println!("Expected behavior:");
    println!("  Test 1: P1 moves toward origin (vx < 0) ✓ m+/m+ attraction");
    println!("  Test 2: P1 moves away from origin (vx > 0) ✓ m+/m- cross-repulsion");
    println!("  Test 3: P1 moves away from origin (vx > 0) ✓ m-/m- self-repulsion");
    println!("==============================================");

    Ok(())
}
