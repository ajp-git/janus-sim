//! Diagnostic for BUG 1 (α too high) and BUG 2 (PM sign verification)
//!
//! BUG 1: Count pairs within r_cut to verify PE_binding calculation
//! BUG 2: Test +/- separation with controlled initial conditions

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(all(feature = "cuda", feature = "cufft"))]
use std::sync::Arc;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== BUG Diagnostics ===\n");

    // ================================================================
    // BUG 1: Count pairs within r_cut
    // ================================================================
    println!("--- BUG 1: Pair count within r_cut ---\n");

    let n = 10_000;
    let box_size = 300.0;

    // Test different r_cut values
    for r_cut_frac in [16.0, 8.0, 4.0] {
        let r_cut = box_size / r_cut_frac;
        let r_cut_sq = r_cut * r_cut;

        // Generate random positions
        use rand::Rng;
        let mut rng = rand::rng();
        let half = box_size / 2.0;

        let mut positions: Vec<[f64; 3]> = Vec::with_capacity(n);
        for _ in 0..n {
            positions.push([
                rng.random::<f64>() * box_size - half,
                rng.random::<f64>() * box_size - half,
                rng.random::<f64>() * box_size - half,
            ]);
        }

        // Count pairs within r_cut (sample 1000 particles)
        let sample_size = 1000.min(n);
        let mut total_neighbors = 0usize;

        for i in 0..sample_size {
            let mut neighbors = 0;
            for j in 0..n {
                if i == j { continue; }
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dz = positions[i][2] - positions[j][2];
                let r2 = dx*dx + dy*dy + dz*dz;
                if r2 < r_cut_sq {
                    neighbors += 1;
                }
            }
            total_neighbors += neighbors;
        }

        let avg_neighbors = total_neighbors as f64 / sample_size as f64;

        // Expected from uniform distribution:
        // neighbors ≈ n * (4/3 * π * r_cut³) / box³
        let volume_ratio = (4.0/3.0 * std::f64::consts::PI * r_cut.powi(3)) / box_size.powi(3);
        let expected = (n as f64) * volume_ratio;

        println!("  r_cut = box/{:.0} = {:.2} Mpc", r_cut_frac, r_cut);
        println!("    Avg neighbors within r_cut: {:.1}", avg_neighbors);
        println!("    Expected (theory): {:.1}", expected);
        println!("    Volume fraction: {:.4}%", volume_ratio * 100.0);
        println!();
    }

    // Compute what α would be with correct neighbor count
    println!("  Implication for α:");
    println!("    If PE_binding only counts r < r_cut pairs,");
    println!("    and r_cut = box/16 gives ~0.1% of pairs,");
    println!("    then PE_binding is 1000x smaller than full PE,");
    println!("    leading to α ~ sqrt(1000) ≈ 30x too large.");
    println!();
    println!("    FIX: Use r_cut = box/4 or larger for virialization,");
    println!("    or compute PE_binding over ALL same-sign pairs (no r_cut).");
    println!();

    // ================================================================
    // BUG 2: Verify PM +/- repulsion sign
    // ================================================================
    println!("--- BUG 2: PM +/- Repulsion Verification ---\n");

    bug2_pm_repulsion_test();
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn bug2_pm_repulsion_test() {
    // Create sim with controlled IC: + left, - right
    let n = 10_000;
    let box_size = 100.0;
    let r_cut = box_size / 8.0;  // Use larger r_cut
    let dt = 0.01;

    println!("  Setup: {} particles, box={}", n, box_size);
    println!("  Initial: + particles in LEFT half (x < 0)");
    println!("           - particles in RIGHT half (x > 0)");
    println!("  Expected: After 10 steps, + and - should SEPARATE further");
    println!();

    // We need to create a custom simulation with controlled ICs
    // For now, let's compute what SHOULD happen

    // Place + at x = -25, - at x = +25 (center of each half)
    let x_plus = -25.0;
    let x_minus = 25.0;
    let initial_separation = x_minus - x_plus;  // 50

    println!("  Initial separation: {} Mpc", initial_separation);
    println!();

    // In Janus model:
    // + particles are REPELLED by - particles
    // So + should move LEFT (more negative x)
    // And - should move RIGHT (more positive x)
    // → Separation should INCREASE

    println!("  Physics check:");
    println!("    + sees - as REPULSIVE → + accelerates LEFT (a_x < 0)");
    println!("    - sees + as REPULSIVE → - accelerates RIGHT (a_x > 0)");
    println!("    → Separation should INCREASE");
    println!();

    // Now test with actual simulation
    println!("  Running actual test...");

    // Create device and compile minimal test
    let device = CudaDevice::new(0).expect("CUDA device");

    // Generate controlled ICs
    use rand::Rng;
    let mut rng = rand::rng();
    let half = box_size / 2.0;

    let mut pos_data = Vec::with_capacity(n * 3);
    let mut vel_data = Vec::with_capacity(n * 3);
    let mut signs_data: Vec<i8> = Vec::with_capacity(n);

    // First n/2: + particles in LEFT half
    for _ in 0..n/2 {
        let x = rng.random::<f64>() * half - half;  // x in [-50, 0]
        let y = rng.random::<f64>() * box_size - half;
        let z = rng.random::<f64>() * box_size - half;
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([0.0f32, 0.0f32, 0.0f32]);
        signs_data.push(1);
    }

    // Second n/2: - particles in RIGHT half
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

    let initial_sep = com_minus_x_0 - com_plus_x_0;

    println!("    Initial COM(+).x = {:.2}", com_plus_x_0);
    println!("    Initial COM(-).x = {:.2}", com_minus_x_0);
    println!("    Initial separation = {:.2}", initial_sep);
    println!();

    // Create simulation with these ICs
    // For this test, we'll use a simplified approach
    // Just check the sign of acceleration

    // The PM force on a + particle from the - grid should point AWAY from -
    // i.e., in the -x direction (left)

    // For this test, let's manually check the force direction
    // by looking at what step_treepm_gpu does

    println!("    [Manual verification needed]");
    println!("    Check: In cic_gather kernel,");
    println!("      + particle: F = -∇φ_plus + ∇φ_minus");
    println!("      The + ∇φ_minus term should push + AWAY from -");
    println!();
    println!("    If - density is on RIGHT (positive x),");
    println!("    then ∇φ_minus points toward - region (positive x)");
    println!("    So + particle feels force in +x direction...");
    println!();
    println!("    WAIT - this is WRONG!");
    println!("    + should be REPELLED from -, so force should be -x");
    println!();
    println!("    The sign in cic_gather may be incorrect!");
    println!("    Current: F = -∇φ_attract + ∇φ_repel");
    println!("    Should be: F = -∇φ_attract - ∇φ_repel (both attract/repel toward minima)");
    println!();

    // Let's think about this more carefully
    println!("  Detailed physics:");
    println!("    φ = potential, ∇φ points UPHILL (toward higher φ)");
    println!("    F = -∇φ points DOWNHILL (toward lower φ)");
    println!();
    println!("    For same-sign (attractive): F = -∇φ_same");
    println!("    → Particle moves toward density peak (correct)");
    println!();
    println!("    For opposite-sign (REPULSIVE): F = +∇φ_opposite");
    println!("    → Particle moves AWAY from density peak");
    println!("    → This gives REPULSION (correct)");
    println!();
    println!("    Current code: F = -∇φ_attract + ∇φ_repel");
    println!("    This looks CORRECT!");
    println!();

    println!("  Need actual simulation test to verify...");
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
