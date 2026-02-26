//! Debug: verify signs are correctly transferred to GPU and used in force calculation
//!
//! cargo run --release --features cuda --bin debug_gpu_signs

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Debug GPU Signs ===\n");

    // Create a small system with known signs
    let n_plus = 5;
    let n_minus = 5;
    let box_size = 100.0;

    // Positions in a line along x-axis
    let mut positions = Vec::new();
    for i in 0..10 {
        positions.push(i as f64 * 5.0);  // x
        positions.push(0.0);              // y
        positions.push(0.0);              // z
    }

    let velocities = vec![0.0; 30];

    // Alternating signs: +, -, +, -, ...
    let signs: Vec<i32> = (0..10).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

    println!("Input signs (CPU): {:?}", signs);
    println!("Expected: [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]");

    let mut sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, box_size,
        positions.clone(), velocities, signs.clone()
    )?;

    // Get signs back from GPU
    let gpu_signs = sim.get_signs()?;
    println!("\nSigns from GPU: {:?}", gpu_signs);

    // Check if they match
    let mut match_count = 0;
    for i in 0..10 {
        if signs[i] == gpu_signs[i] {
            match_count += 1;
        } else {
            println!("  MISMATCH at index {}: CPU={}, GPU={}", i, signs[i], gpu_signs[i]);
        }
    }
    println!("Match count: {}/10", match_count);

    // Now step and check signs are preserved
    sim.set_theta(0.0);
    sim.step_with_cross_factor(0.01, -1.0)?;

    let gpu_signs_after = sim.get_signs()?;
    println!("\nSigns after step: {:?}", gpu_signs_after);

    // Check preservation
    let mut preserved = true;
    for i in 0..10 {
        if signs[i] != gpu_signs_after[i] {
            println!("  CHANGED at index {}: before={}, after={}", i, signs[i], gpu_signs_after[i]);
            preserved = false;
        }
    }
    if preserved {
        println!("✓ Signs preserved after step");
    } else {
        println!("✗ Signs CHANGED after step - BUG!");
    }

    // Check force differences between m+ and m- particles at same position
    // P0 (m+) and P1 (m-) are neighbors
    let vel = sim.get_velocities()?;
    println!("\nVelocities after 1 step:");
    for i in 0..10 {
        let s = if signs[i] > 0 { "m+" } else { "m-" };
        println!("  P{} ({}): vx={:+.8}", i, s, vel[i*3]);
    }

    // If cross_factor works, m+ and m- should have DIFFERENT velocities
    // P0 (m+) at x=0: attracted to P2,P4,P6,P8 (m+), repelled from P1,P3,P5,P7,P9 (m-)
    // P1 (m-) at x=5: attracted to P3,P5,P7,P9 (m-), repelled from P0,P2,P4,P6,P8 (m+)

    println!("\n=== Analysis ===");
    let v0 = vel[0];  // m+ at x=0
    let v1 = vel[3];  // m- at x=5

    println!("P0 (m+) vx = {:+.8}", v0);
    println!("P1 (m-) vx = {:+.8}", v1);

    if (v0 - v1).abs() > 1e-10 {
        println!("✓ m+ and m- have DIFFERENT velocities → cross_factor works");
    } else {
        println!("✗ m+ and m- have SAME velocity → cross_factor NOT applied!");
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA not enabled. Use: cargo run --release --features cuda --bin debug_gpu_signs");
}
