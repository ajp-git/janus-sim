//! SPH Pressure Tests
//!
//! Test A: Newton III symmetry - F_ij + F_ji = 0
//! Test B: v_rms stability over 100 steps
//! Test C: Cold collapse with pressure support

#[cfg(feature = "cuda")]
use janus::sph_pressure_gpu::GpuSphPressure;
#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  SPH PRESSURE GPU TESTS");
        println!("═══════════════════════════════════════════════════════════════\n");

        let mut all_passed = true;

        // Test A: Newton III symmetry
        println!("TEST A: Newton III Symmetry (F_ij + F_ji = 0)");
        println!("─────────────────────────────────────────────");
        match test_newton_iii() {
            Ok(max_asymmetry) => {
                if max_asymmetry < 1e-6 {
                    println!("  ✓ PASS: max asymmetry = {:.2e} < 1e-6\n", max_asymmetry);
                } else {
                    println!("  ✗ FAIL: max asymmetry = {:.2e} >= 1e-6\n", max_asymmetry);
                    all_passed = false;
                }
            }
            Err(e) => {
                println!("  ✗ ERROR: {}\n", e);
                all_passed = false;
            }
        }

        // Test B: v_rms stability
        println!("TEST B: v_rms Stability (100 steps, N=10k)");
        println!("──────────────────────────────────────────");
        match test_vrms_stability() {
            Ok((v0, v100, ratio)) => {
                if ratio < 3.0 {
                    println!("  v_rms(0) = {:.2} km/s", v0);
                    println!("  v_rms(100) = {:.2} km/s", v100);
                    println!("  ✓ PASS: ratio = {:.2} < 3\n", ratio);
                } else {
                    println!("  v_rms(0) = {:.2} km/s", v0);
                    println!("  v_rms(100) = {:.2} km/s", v100);
                    println!("  ✗ FAIL: ratio = {:.2} >= 3 (instability!)\n", ratio);
                    all_passed = false;
                }
            }
            Err(e) => {
                println!("  ✗ ERROR: {}\n", e);
                all_passed = false;
            }
        }

        // Test C: Force direction check
        println!("TEST C: Pressure Force Direction");
        println!("─────────────────────────────────");
        match test_force_direction() {
            Ok(correct) => {
                if correct {
                    println!("  ✓ PASS: Pressure pushes particles apart in dense regions\n");
                } else {
                    println!("  ✗ FAIL: Wrong force direction!\n");
                    all_passed = false;
                }
            }
            Err(e) => {
                println!("  ✗ ERROR: {}\n", e);
                all_passed = false;
            }
        }

        println!("═══════════════════════════════════════════════════════════════");
        if all_passed {
            println!("  ALL TESTS PASSED ✓");
        } else {
            println!("  SOME TESTS FAILED ✗");
            std::process::exit(1);
        }
        println!("═══════════════════════════════════════════════════════════════");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: This test requires --features cuda");
        std::process::exit(1);
    }
}

/// Test A: Newton III symmetry
/// Place two particles and verify F_ij + F_ji = 0
#[cfg(feature = "cuda")]
fn test_newton_iii() -> Result<f64, Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;  // Already returns Arc<CudaDevice>

    // Two particles at distance 1.0
    let n = 2;
    let box_size = 100.0;
    let mass = 1e10;

    let mut sph = GpuSphPressure::new(device, n, mass, box_size)?;

    // Positions: particle 0 at origin, particle 1 at (1, 0, 0)
    let positions = vec![
        0.0, 0.0, 0.0,  // particle 0
        1.0, 0.0, 0.0,  // particle 1
    ];

    // Same temperature for both
    let temperatures = vec![1e4, 1e4];

    // Compute accelerations
    let acc = sph.compute_pressure_accelerations(&positions, &temperatures)?;

    // Extract accelerations
    let ax0 = acc[0];
    let ay0 = acc[1];
    let az0 = acc[2];
    let ax1 = acc[3];
    let ay1 = acc[4];
    let az1 = acc[5];

    println!("  Particle 0: a = ({:.6e}, {:.6e}, {:.6e})", ax0, ay0, az0);
    println!("  Particle 1: a = ({:.6e}, {:.6e}, {:.6e})", ax1, ay1, az1);

    // F_ij + F_ji should be zero (Newton III)
    let asymmetry_x = (ax0 + ax1).abs();
    let asymmetry_y = (ay0 + ay1).abs();
    let asymmetry_z = (az0 + az1).abs();
    let max_asymmetry = asymmetry_x.max(asymmetry_y).max(asymmetry_z);

    println!("  Sum of forces: ({:.2e}, {:.2e}, {:.2e})", ax0 + ax1, ay0 + ay1, az0 + az1);

    Ok(max_asymmetry)
}

/// Test B: v_rms stability over 100 steps
#[cfg(feature = "cuda")]
fn test_vrms_stability() -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;  // Already returns Arc<CudaDevice>

    let n = 10_000;
    let box_size = 100.0;
    let mass = 1e10;
    let dt = 0.001;  // Small timestep
    let steps = 100;

    // Initialize random positions
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(42);
    let half_box = box_size / 2.0;

    // Initial thermal velocity (T = 10,000 K)
    let sigma_v = 0.012;  // Mpc/Gyr for T=10,000 K
    let normal = Normal::new(0.0, sigma_v).unwrap();

    let mut positions = Vec::with_capacity(n * 3);
    let mut velocities = Vec::with_capacity(n * 3);
    let temperatures = vec![1e4; n];

    for _ in 0..n {
        positions.push(rng.gen::<f64>() * box_size - half_box);
        positions.push(rng.gen::<f64>() * box_size - half_box);
        positions.push(rng.gen::<f64>() * box_size - half_box);

        velocities.push(normal.sample(&mut rng));
        velocities.push(normal.sample(&mut rng));
        velocities.push(normal.sample(&mut rng));
    }

    // Initial v_rms
    let v2_sum: f64 = velocities.chunks(3)
        .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        .sum();
    let v_rms_0 = (v2_sum / n as f64).sqrt() * 978.0;  // km/s

    // Create SPH calculator
    let mut sph = GpuSphPressure::new(Arc::clone(&device), n, mass, box_size)?;

    // Evolution loop (pressure only, no gravity)
    for step in 0..steps {
        // Compute pressure acceleration
        let acc = sph.compute_pressure_accelerations(&positions, &temperatures)?;

        // Kick velocities
        for i in 0..n {
            velocities[i * 3] += acc[i * 3] * dt;
            velocities[i * 3 + 1] += acc[i * 3 + 1] * dt;
            velocities[i * 3 + 2] += acc[i * 3 + 2] * dt;
        }

        // Drift positions
        for i in 0..n {
            positions[i * 3] += velocities[i * 3] * dt;
            positions[i * 3 + 1] += velocities[i * 3 + 1] * dt;
            positions[i * 3 + 2] += velocities[i * 3 + 2] * dt;

            // Periodic boundary
            for d in 0..3 {
                let idx = i * 3 + d;
                if positions[idx] > half_box { positions[idx] -= box_size; }
                if positions[idx] < -half_box { positions[idx] += box_size; }
            }
        }

        if step % 20 == 0 {
            let v2: f64 = velocities.chunks(3)
                .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
                .sum();
            let v_rms = (v2 / n as f64).sqrt() * 978.0;
            println!("  Step {:3}: v_rms = {:.2} km/s", step, v_rms);
        }
    }

    // Final v_rms
    let v2_sum: f64 = velocities.chunks(3)
        .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        .sum();
    let v_rms_100 = (v2_sum / n as f64).sqrt() * 978.0;

    let ratio = v_rms_100 / v_rms_0;

    Ok((v_rms_0, v_rms_100, ratio))
}

/// Test C: Pressure force direction
/// In a dense region, particles should be pushed apart
#[cfg(feature = "cuda")]
fn test_force_direction() -> Result<bool, Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;  // Already returns Arc<CudaDevice>

    // Create a cluster of particles at center
    let n = 100;
    let box_size = 100.0;
    let mass = 1e10;
    let cluster_radius = 5.0;

    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    let mut rng = StdRng::seed_from_u64(123);

    let mut positions = Vec::with_capacity(n * 3);
    let temperatures = vec![1e4; n];

    // All particles in a small cluster at center
    for _ in 0..n {
        let r = rng.gen::<f64>() * cluster_radius;
        let theta = rng.gen::<f64>() * std::f64::consts::PI;
        let phi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;

        positions.push(r * theta.sin() * phi.cos());
        positions.push(r * theta.sin() * phi.sin());
        positions.push(r * theta.cos());
    }

    let mut sph = GpuSphPressure::new(device, n, mass, box_size)?;
    let acc = sph.compute_pressure_accelerations(&positions, &temperatures)?;

    // Check that accelerations point outward from center (radial)
    let mut outward_count = 0;
    for i in 0..n {
        let px = positions[i * 3];
        let py = positions[i * 3 + 1];
        let pz = positions[i * 3 + 2];
        let ax = acc[i * 3];
        let ay = acc[i * 3 + 1];
        let az = acc[i * 3 + 2];

        // Dot product of position and acceleration
        // Should be positive if force is outward (expanding)
        let dot = px * ax + py * ay + pz * az;
        if dot > 0.0 {
            outward_count += 1;
        }
    }

    let outward_fraction = outward_count as f64 / n as f64;
    println!("  Outward fraction: {:.1}%", outward_fraction * 100.0);

    // At least 70% should have outward acceleration
    Ok(outward_fraction > 0.7)
}
