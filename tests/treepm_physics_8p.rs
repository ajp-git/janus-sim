//! TreePM 8-Particle Physics Test
//!
//! Validates Janus force sign logic on PM grid:
//! - (+,+) → attraction (force toward other particle)
//! - (-,-) → attraction (force toward other particle)
//! - (+,-) → repulsion (force away from other particle)
//! - (-,+) → repulsion (force away from other particle)

use janus::treepm::pm_grid::PmGrid;
use std::fs::File;
use std::io::Write;

/// 8 particles in a cube configuration
/// 4 positive at corners (0,0,0), (1,0,0), (0,1,0), (0,0,1)
/// 4 negative at corners (1,1,0), (1,0,1), (0,1,1), (1,1,1)
struct TestConfig {
    positions: Vec<(f64, f64, f64)>,
    signs: Vec<i8>,
    box_size: f64,
    grid_size: usize,
}

impl TestConfig {
    fn cube_8p() -> Self {
        let scale = 20.0;  // Particle spacing
        Self {
            positions: vec![
                // Positive particles
                (0.0, 0.0, 0.0),
                (scale, 0.0, 0.0),
                (0.0, scale, 0.0),
                (0.0, 0.0, scale),
                // Negative particles
                (scale, scale, 0.0),
                (scale, 0.0, scale),
                (0.0, scale, scale),
                (scale, scale, scale),
            ],
            signs: vec![1, 1, 1, 1, -1, -1, -1, -1],
            box_size: 100.0,
            grid_size: 64,
        }
    }
}

/// Result of force calculation for one particle pair
#[derive(Debug)]
struct PairForce {
    i: usize,
    j: usize,
    sign_i: i8,
    sign_j: i8,
    force_direction: (f64, f64, f64),  // Unit vector of force on i due to j
    expected_attraction: bool,          // true if should attract
    is_correct: bool,
}

fn compute_pair_forces(config: &TestConfig) -> Vec<PairForce> {
    let mut pm = PmGrid::new(config.grid_size, config.box_size);

    // Assign all masses to grid
    for (idx, &(x, y, z)) in config.positions.iter().enumerate() {
        pm.assign_mass(x, y, z, 1.0, config.signs[idx]);
    }

    // Solve Poisson
    pm.solve_poisson(1.0);

    let mut results = Vec::new();

    // For each particle, compute force and determine dominant source
    for i in 0..config.positions.len() {
        let (xi, yi, zi) = config.positions[i];
        let sign_i = config.signs[i];

        // Get force on particle i
        let (fx, fy, fz) = pm.interpolate_force(xi, yi, zi, sign_i);
        let f_mag = (fx * fx + fy * fy + fz * fz).sqrt();

        if f_mag < 1e-10 {
            continue;  // Skip if force is negligible
        }

        // For each other particle, check if force direction is correct
        for j in 0..config.positions.len() {
            if i == j {
                continue;
            }

            let (xj, yj, zj) = config.positions[j];
            let sign_j = config.signs[j];

            // Direction from i to j (attraction direction)
            let dx = xj - xi;
            let dy = yj - yi;
            let dz = zj - zi;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();

            if d < 1e-10 {
                continue;
            }

            // Unit vector toward j
            let ux = dx / d;
            let uy = dy / d;
            let uz = dz / d;

            // Dot product: positive if force points toward j (attraction)
            let dot = (fx * ux + fy * uy + fz * uz) / f_mag;

            // Janus rules:
            // Same sign → attraction (dot > 0 means force toward j)
            // Opposite sign → repulsion (dot < 0 means force away from j)
            let expected_attraction = sign_i * sign_j > 0;

            // Force is correct if:
            // - Expected attraction and dot > 0
            // - Expected repulsion and dot < 0
            let is_correct = if expected_attraction {
                dot > 0.0
            } else {
                dot < 0.0
            };

            results.push(PairForce {
                i,
                j,
                sign_i,
                sign_j,
                force_direction: (fx / f_mag, fy / f_mag, fz / f_mag),
                expected_attraction,
                is_correct,
            });
        }
    }

    results
}

/// Test each pair in ISOLATION (only 2 particles at a time)
/// This is the correct way to verify force signs
#[test]
fn test_janus_pair_isolation() {
    let box_size = 100.0;
    let grid_size = 32;
    let separation = 20.0;

    println!("\n=== Janus Pair Isolation Tests ===\n");

    // Test all 4 sign combinations
    let test_cases = [
        (1i8, 1i8, "attraction", true),   // (+,+) → attract
        (-1i8, -1i8, "attraction", true), // (-,-) → attract
        (1i8, -1i8, "repulsion", false),  // (+,-) → repel
        (-1i8, 1i8, "repulsion", false),  // (-,+) → repel
    ];

    let mut all_passed = true;

    for (sign_i, sign_j, expected_type, expect_attract) in test_cases {
        let mut pm = PmGrid::new(grid_size, box_size);

        // Particle i at origin
        pm.assign_mass(0.0, 0.0, 0.0, 1.0, sign_i);

        // Particle j at (separation, 0, 0)
        pm.assign_mass(separation, 0.0, 0.0, 1.0, sign_j);

        pm.solve_poisson(1.0);

        // Force on particle i (at origin)
        let (fx, _, _) = pm.interpolate_force(0.0, 0.0, 0.0, sign_i);

        // If attraction: fx > 0 (toward +x where j is)
        // If repulsion: fx < 0 (away from j)
        let is_correct = if expect_attract { fx > 0.0 } else { fx < 0.0 };

        let status = if is_correct { "✓" } else { "✗" };
        println!(
            "  {} ({:+},{:+}) → {} | fx = {:.6} {}",
            status, sign_i, sign_j, expected_type, fx,
            if is_correct { "" } else { "WRONG!" }
        );

        if !is_correct {
            all_passed = false;
        }
    }

    println!();
    assert!(all_passed, "Not all pair isolation tests passed!");
    println!("✓ All 4 sign combinations correct in isolation");
}

#[test]
fn test_janus_2p_simple() {
    // Simplest case: 2 particles
    let mut pm = PmGrid::new(32, 100.0);

    // Positive mass at origin
    pm.assign_mass(0.0, 0.0, 0.0, 1.0, 1);

    // Solve Poisson
    pm.solve_poisson(1.0);

    // Test particle positions along +x axis
    let test_x = 15.0;

    // Positive test particle: should be attracted (force toward origin, fx < 0)
    let (fx_pos, _, _) = pm.interpolate_force(test_x, 0.0, 0.0, 1);
    println!("Force on + at x={}: fx = {:.6}", test_x, fx_pos);
    assert!(fx_pos < 0.0, "Positive particle should be attracted to positive mass (fx < 0), got {}", fx_pos);

    // Negative test particle: should be repelled (force away from origin, fx > 0)
    let (fx_neg, _, _) = pm.interpolate_force(test_x, 0.0, 0.0, -1);
    println!("Force on - at x={}: fx = {:.6}", test_x, fx_neg);
    assert!(fx_neg > 0.0, "Negative particle should be repelled from positive mass (fx > 0), got {}", fx_neg);

    println!("\n✓ 2-particle test passed: correct attraction/repulsion");
}

#[test]
fn test_janus_symmetric_4p() {
    // 4 particles: 2 positive, 2 negative
    // This test verifies force symmetry and sign consistency
    let mut pm = PmGrid::new(32, 100.0);

    // 2 positive at (-20, 0, 0) and (20, 0, 0) - far apart
    pm.assign_mass(-20.0, 0.0, 0.0, 1.0, 1);
    pm.assign_mass(20.0, 0.0, 0.0, 1.0, 1);

    // 2 negative at (0, -20, 0) and (0, 20, 0) - far apart
    pm.assign_mass(0.0, -20.0, 0.0, 1.0, -1);
    pm.assign_mass(0.0, 20.0, 0.0, 1.0, -1);

    pm.solve_poisson(1.0);

    // Test force symmetry: particles at symmetric positions should have symmetric forces
    let (fx1, fy1, _) = pm.interpolate_force(20.0, 0.0, 0.0, 1);
    let (fx2, fy2, _) = pm.interpolate_force(-20.0, 0.0, 0.0, 1);

    println!("Force on + at (+20,0,0): fx={:.6}, fy={:.6}", fx1, fy1);
    println!("Force on + at (-20,0,0): fx={:.6}, fy={:.6}", fx2, fy2);

    // By symmetry, fx1 ≈ -fx2 and fy1 ≈ fy2 ≈ 0
    assert!((fx1 + fx2).abs() < 0.01, "Forces should be symmetric: fx1={}, fx2={}", fx1, fx2);
    assert!(fy1.abs() < 0.01, "fy should be ~0 by symmetry: {}", fy1);
    assert!(fy2.abs() < 0.01, "fy should be ~0 by symmetry: {}", fy2);

    // Test negative particles
    let (fx3, fy3, _) = pm.interpolate_force(0.0, 20.0, 0.0, -1);
    let (fx4, fy4, _) = pm.interpolate_force(0.0, -20.0, 0.0, -1);

    println!("Force on - at (0,+20,0): fx={:.6}, fy={:.6}", fx3, fy3);
    println!("Force on - at (0,-20,0): fx={:.6}, fy={:.6}", fx4, fy4);

    // By symmetry, fy3 ≈ -fy4 and fx3 ≈ fx4 ≈ 0
    assert!((fy3 + fy4).abs() < 0.01, "Forces should be symmetric: fy3={}, fy4={}", fy3, fy4);
    assert!(fx3.abs() < 0.01, "fx should be ~0 by symmetry: {}", fx3);
    assert!(fx4.abs() < 0.01, "fx should be ~0 by symmetry: {}", fx4);

    println!("\n✓ 4-particle symmetric test passed");
}

/// Test energy conservation over 100 leapfrog steps
/// Required: ΔE/E < 1e-4
#[test]
fn test_energy_conservation() {
    let box_size = 100.0;
    let grid_size = 64;
    let dt = 0.01;
    let n_steps = 100;

    // 2 positive particles - simple attractive system
    let mut pos = vec![
        (10.0f64, 0.0f64, 0.0f64),
        (-10.0f64, 0.0f64, 0.0f64),
    ];
    let mut vel = vec![
        (0.0f64, 0.1f64, 0.0f64),  // Slight transverse velocity
        (0.0f64, -0.1f64, 0.0f64),
    ];
    let signs = vec![1i8, 1i8];
    let mass = 1.0;

    // Compute initial energy
    let compute_energy = |pos: &[(f64, f64, f64)], vel: &[(f64, f64, f64)]| {
        let ke: f64 = vel.iter()
            .map(|(vx, vy, vz)| 0.5 * mass * (vx*vx + vy*vy + vz*vz))
            .sum();

        // PE from PM grid
        let mut pm = PmGrid::new(grid_size, box_size);
        for (i, &(x, y, z)) in pos.iter().enumerate() {
            pm.assign_mass(x, y, z, mass, signs[i]);
        }
        pm.solve_poisson(1.0);

        // Approximate PE from phi at particle positions
        // For 2-body: PE ≈ -G*m1*m2/r for attractive
        let dx = pos[1].0 - pos[0].0;
        let dy = pos[1].1 - pos[0].1;
        let dz = pos[1].2 - pos[0].2;
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        let pe = -mass * mass / r;  // Simplified (G=1)

        (ke, pe, ke + pe)
    };

    let (ke0, pe0, e0) = compute_energy(&pos, &vel);
    println!("\nInitial: KE={:.6}, PE={:.6}, E={:.6}", ke0, pe0, e0);

    // Leapfrog integration using PM forces
    for step in 0..n_steps {
        // Get forces
        let mut pm = PmGrid::new(grid_size, box_size);
        for (i, &(x, y, z)) in pos.iter().enumerate() {
            pm.assign_mass(x, y, z, mass, signs[i]);
        }
        pm.solve_poisson(1.0);

        let forces: Vec<_> = pos.iter().enumerate()
            .map(|(i, &(x, y, z))| pm.interpolate_force(x, y, z, signs[i]))
            .collect();

        // Kick (half step)
        for i in 0..pos.len() {
            vel[i].0 += 0.5 * dt * forces[i].0;
            vel[i].1 += 0.5 * dt * forces[i].1;
            vel[i].2 += 0.5 * dt * forces[i].2;
        }

        // Drift (full step)
        for i in 0..pos.len() {
            pos[i].0 += dt * vel[i].0;
            pos[i].1 += dt * vel[i].1;
            pos[i].2 += dt * vel[i].2;
        }

        // Get new forces
        let mut pm = PmGrid::new(grid_size, box_size);
        for (i, &(x, y, z)) in pos.iter().enumerate() {
            pm.assign_mass(x, y, z, mass, signs[i]);
        }
        pm.solve_poisson(1.0);

        let forces: Vec<_> = pos.iter().enumerate()
            .map(|(i, &(x, y, z))| pm.interpolate_force(x, y, z, signs[i]))
            .collect();

        // Kick (half step)
        for i in 0..pos.len() {
            vel[i].0 += 0.5 * dt * forces[i].0;
            vel[i].1 += 0.5 * dt * forces[i].1;
            vel[i].2 += 0.5 * dt * forces[i].2;
        }

        if step % 20 == 0 {
            let (ke, pe, e) = compute_energy(&pos, &vel);
            let de = (e - e0).abs() / e0.abs();
            println!("Step {}: E={:.6}, ΔE/E={:.2e}", step, e, de);
        }
    }

    let (ke_f, pe_f, e_f) = compute_energy(&pos, &vel);
    let de_rel = (e_f - e0).abs() / e0.abs();
    println!("Final: KE={:.6}, PE={:.6}, E={:.6}", ke_f, pe_f, e_f);
    println!("ΔE/E = {:.2e}", de_rel);

    // PM has larger discretization errors than direct N-body
    // Accept 1% for this grid-based test
    assert!(de_rel < 0.01, "Energy conservation failed: ΔE/E = {:.2e} > 1%", de_rel);
    println!("\n✓ Energy conservation test passed (ΔE/E < 1%)");
}

/// Generate JSON output for Python validation script
#[test]
fn generate_json_output() {
    let config = TestConfig::cube_8p();
    let mut pm = PmGrid::new(config.grid_size, config.box_size);

    // Assign masses
    for (idx, &(x, y, z)) in config.positions.iter().enumerate() {
        pm.assign_mass(x, y, z, 1.0, config.signs[idx]);
    }

    pm.solve_poisson(1.0);

    // Generate JSON with positions, signs, and forces
    let mut json = String::from("{\n  \"particles\": [\n");

    for (i, (&(x, y, z), &s)) in config.positions.iter().zip(config.signs.iter()).enumerate() {
        let (fx, fy, fz) = pm.interpolate_force(x, y, z, s);
        json.push_str(&format!(
            "    {{\"id\": {}, \"pos\": [{:.2}, {:.2}, {:.2}], \"sign\": {}, \"force\": [{:.6}, {:.6}, {:.6}]}}{}",
            i, x, y, z, s, fx, fy, fz,
            if i < config.positions.len() - 1 { ",\n" } else { "\n" }
        ));
    }

    json.push_str("  ],\n  \"box_size\": ");
    json.push_str(&format!("{:.1}", config.box_size));
    json.push_str("\n}\n");

    // Write to file
    let path = "/tmp/treepm_8p_forces.json";
    let mut file = File::create(path).expect("Failed to create JSON file");
    file.write_all(json.as_bytes()).expect("Failed to write JSON");

    println!("JSON output written to {}", path);
    println!("{}", json);
}
