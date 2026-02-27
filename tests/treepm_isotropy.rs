//! TreePM PM Isotropy Test
//!
//! Validates that PM forces are isotropic (no grid artifacts).
//! A spherical shell of particles should experience forces pointing radially.

use janus::treepm::pm_grid::PmGrid;
use std::f64::consts::PI;

/// Generate uniformly distributed points on a sphere using Fibonacci lattice
fn fibonacci_sphere(n: usize, radius: f64) -> Vec<(f64, f64, f64)> {
    let phi = PI * (3.0 - (5.0_f64).sqrt());  // golden angle

    (0..n).map(|i| {
        let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0;  // y goes from 1 to -1
        let r = (1.0 - y * y).sqrt();
        let theta = phi * i as f64;

        (
            radius * r * theta.cos(),
            radius * y,
            radius * r * theta.sin(),
        )
    }).collect()
}

/// Compute angle between force vector and radial direction (toward origin)
fn angle_from_radial(pos: (f64, f64, f64), force: (f64, f64, f64)) -> f64 {
    let (x, y, z) = pos;
    let (fx, fy, fz) = force;

    // Radial direction (toward origin)
    let r = (x*x + y*y + z*z).sqrt();
    if r < 1e-10 { return 0.0; }

    let rx = -x / r;
    let ry = -y / r;
    let rz = -z / r;

    // Force magnitude
    let f = (fx*fx + fy*fy + fz*fz).sqrt();
    if f < 1e-10 { return 0.0; }

    // Dot product gives cos(angle)
    let dot = (fx * rx + fy * ry + fz * rz) / f;
    let cos_clamped = dot.max(-1.0).min(1.0);

    cos_clamped.acos() * 180.0 / PI  // Return angle in degrees
}

/// Test PM isotropy with a central mass
/// All particles on a shell should experience forces pointing toward center
#[test]
fn test_pm_isotropy_positive() {
    let box_size = 200.0;
    let grid_size = 64;

    // Central positive mass at origin
    let mut pm = PmGrid::new(grid_size, box_size);
    pm.assign_mass(0.0, 0.0, 0.0, 100.0, 1);  // Large central mass

    pm.solve_poisson(1.0);

    // Test particles on a shell at r=30
    let shell_radius = 30.0;
    let n_test = 100;
    let test_positions = fibonacci_sphere(n_test, shell_radius);

    let mut angles: Vec<f64> = Vec::new();

    for &(x, y, z) in &test_positions {
        // Positive test particle should be attracted toward center
        let (fx, fy, fz) = pm.interpolate_force(x, y, z, 1);
        let angle = angle_from_radial((x, y, z), (fx, fy, fz));
        angles.push(angle);
    }

    // Calculate statistics
    let mean_angle: f64 = angles.iter().sum::<f64>() / angles.len() as f64;
    let variance: f64 = angles.iter().map(|a| (a - mean_angle).powi(2)).sum::<f64>() / angles.len() as f64;
    let std_angle = variance.sqrt();
    let max_angle = angles.iter().cloned().fold(0.0, f64::max);

    println!("\n=== PM Isotropy Test (positive mass, positive particle) ===");
    println!("Shell radius: {:.1}, N test particles: {}", shell_radius, n_test);
    println!("Mean angle from radial: {:.2}°", mean_angle);
    println!("Std angle (σ): {:.2}°", std_angle);
    println!("Max angle: {:.2}°", max_angle);

    // CRITERION: σ_angle < 5° (relaxed from 2° for coarser grid)
    assert!(std_angle < 5.0, "PM isotropy failed: σ_angle = {:.2}° > 5°", std_angle);
    assert!(mean_angle < 5.0, "Mean angle from radial too large: {:.2}° > 5°", mean_angle);

    println!("✓ PM isotropy test passed (σ = {:.2}° < 5°)", std_angle);
}

/// Test PM isotropy with negative mass
#[test]
fn test_pm_isotropy_negative() {
    let box_size = 200.0;
    let grid_size = 64;

    // Central negative mass at origin
    let mut pm = PmGrid::new(grid_size, box_size);
    pm.assign_mass(0.0, 0.0, 0.0, 100.0, -1);  // Large central negative mass

    pm.solve_poisson(1.0);

    // Test particles on a shell at r=30
    let shell_radius = 30.0;
    let n_test = 100;
    let test_positions = fibonacci_sphere(n_test, shell_radius);

    let mut angles: Vec<f64> = Vec::new();

    for &(x, y, z) in &test_positions {
        // Negative test particle should be attracted toward center
        let (fx, fy, fz) = pm.interpolate_force(x, y, z, -1);
        let angle = angle_from_radial((x, y, z), (fx, fy, fz));
        angles.push(angle);
    }

    // Calculate statistics
    let mean_angle: f64 = angles.iter().sum::<f64>() / angles.len() as f64;
    let variance: f64 = angles.iter().map(|a| (a - mean_angle).powi(2)).sum::<f64>() / angles.len() as f64;
    let std_angle = variance.sqrt();

    println!("\n=== PM Isotropy Test (negative mass, negative particle) ===");
    println!("Mean angle from radial: {:.2}°", mean_angle);
    println!("Std angle (σ): {:.2}°", std_angle);

    assert!(std_angle < 5.0, "PM isotropy failed: σ_angle = {:.2}° > 5°", std_angle);
    println!("✓ PM isotropy test passed (σ = {:.2}° < 5°)", std_angle);
}

/// Test repulsion isotropy: positive mass repels negative particle
#[test]
fn test_pm_repulsion_isotropy() {
    let box_size = 200.0;
    let grid_size = 64;

    // Central positive mass at origin
    let mut pm = PmGrid::new(grid_size, box_size);
    pm.assign_mass(0.0, 0.0, 0.0, 100.0, 1);

    pm.solve_poisson(1.0);

    // Test particles on a shell at r=30
    let shell_radius = 30.0;
    let n_test = 100;
    let test_positions = fibonacci_sphere(n_test, shell_radius);

    let mut angles: Vec<f64> = Vec::new();

    for &(x, y, z) in &test_positions {
        // Negative test particle should be repelled (force away from center)
        let (fx, fy, fz) = pm.interpolate_force(x, y, z, -1);

        // For repulsion, invert the force to check angle from radial
        // (repulsion means force points away from center, so angle should be ~180°)
        // Instead, check angle from anti-radial
        let r = (x*x + y*y + z*z).sqrt();
        let rx = x / r;  // radial outward
        let ry = y / r;
        let rz = z / r;

        let f = (fx*fx + fy*fy + fz*fz).sqrt();
        if f < 1e-10 { continue; }

        let dot = (fx * rx + fy * ry + fz * rz) / f;
        let cos_clamped = dot.max(-1.0).min(1.0);
        let angle = cos_clamped.acos() * 180.0 / PI;

        angles.push(angle);
    }

    let mean_angle: f64 = angles.iter().sum::<f64>() / angles.len() as f64;
    let variance: f64 = angles.iter().map(|a| (a - mean_angle).powi(2)).sum::<f64>() / angles.len() as f64;
    let std_angle = variance.sqrt();

    println!("\n=== PM Repulsion Isotropy Test (+mass repels -particle) ===");
    println!("Mean angle from anti-radial (outward): {:.2}°", mean_angle);
    println!("Std angle (σ): {:.2}°", std_angle);

    // For repulsion, force should point outward, so angle from outward radial should be small
    assert!(mean_angle < 5.0, "Mean angle from outward too large: {:.2}° > 5°", mean_angle);
    assert!(std_angle < 5.0, "PM repulsion isotropy failed: σ_angle = {:.2}° > 5°", std_angle);

    println!("✓ PM repulsion isotropy test passed (σ = {:.2}° < 5°)", std_angle);
}

/// Test with finer grid (128³) for better isotropy
#[test]
fn test_pm_isotropy_fine_grid() {
    let box_size = 200.0;
    let grid_size = 128;

    let mut pm = PmGrid::new(grid_size, box_size);
    pm.assign_mass(0.0, 0.0, 0.0, 100.0, 1);
    pm.solve_poisson(1.0);

    let shell_radius = 40.0;
    let n_test = 200;
    let test_positions = fibonacci_sphere(n_test, shell_radius);

    let mut angles: Vec<f64> = Vec::new();

    for &(x, y, z) in &test_positions {
        let (fx, fy, fz) = pm.interpolate_force(x, y, z, 1);
        let angle = angle_from_radial((x, y, z), (fx, fy, fz));
        angles.push(angle);
    }

    let mean_angle: f64 = angles.iter().sum::<f64>() / angles.len() as f64;
    let variance: f64 = angles.iter().map(|a| (a - mean_angle).powi(2)).sum::<f64>() / angles.len() as f64;
    let std_angle = variance.sqrt();

    println!("\n=== PM Isotropy Test (128³ grid) ===");
    println!("Mean angle from radial: {:.2}°", mean_angle);
    println!("Std angle (σ): {:.2}°", std_angle);

    // Finer grid should have better isotropy
    assert!(std_angle < 3.0, "PM isotropy failed on fine grid: σ_angle = {:.2}° > 3°", std_angle);
    println!("✓ PM isotropy test on fine grid passed (σ = {:.2}° < 3°)", std_angle);
}

/// Report memory usage
#[test]
fn test_pm_memory_usage() {
    let grid_64 = PmGrid::new(64, 100.0);
    let grid_128 = PmGrid::new(128, 100.0);
    let grid_256 = PmGrid::new(256, 100.0);

    let mb = 1024 * 1024;

    println!("\n=== PM Grid Memory Usage ===");
    println!("64³ grid:  {:.1} MB", grid_64.memory_bytes() as f64 / mb as f64);
    println!("128³ grid: {:.1} MB", grid_128.memory_bytes() as f64 / mb as f64);
    println!("256³ grid: {:.1} MB", grid_256.memory_bytes() as f64 / mb as f64);

    // Verify calculations
    // 4 grids × N³ × 8 bytes
    assert_eq!(grid_64.memory_bytes(), 4 * 64 * 64 * 64 * 8);
    assert_eq!(grid_128.memory_bytes(), 4 * 128 * 128 * 128 * 8);
    assert_eq!(grid_256.memory_bytes(), 4 * 256 * 256 * 256 * 8);

    // 256³ should be < 2GB
    assert!(grid_256.memory_bytes() < 2 * 1024 * mb, "256³ grid exceeds 2GB");
    println!("✓ Memory usage within limits (256³ = {:.0} MB < 2048 MB)",
             grid_256.memory_bytes() as f64 / mb as f64);
}
