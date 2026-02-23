/// Test incremental tree updates (opt5)
/// Compare S(t) between full rebuild every step vs incremental updates

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 500_000;  // 500K for faster validation
        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let n_steps = 200;
        let dt = 0.003;
        let rebuild_intervals = [1, 5, 10, 20];  // Test different intervals

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Incremental Tree Updates Validation (opt5)                   ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");
        println!("N = {} particles, {} steps, dt = {}\n", n_particles, n_steps, dt);

        // Run reference (full rebuild every step)
        println!("Running REFERENCE (full rebuild every step)...");
        let mut sim_ref = GpuNBodySimulation::new(n_positive, n_negative, box_size)
            .expect("Failed to create reference simulation");

        // Get initial positions for later comparison
        let initial_pos = sim_ref.positions();
        let signs = sim_ref.signs();

        let t_ref_start = Instant::now();
        for step in 0..n_steps {
            sim_ref.step_with_expansion_dkd_gpu_incremental(
                dt, 1.0, 0.0, 0.0,
                step, 1,  // rebuild_interval=1 = full rebuild every step
            ).expect("Reference step failed");
        }
        let t_ref = t_ref_start.elapsed().as_secs_f64();
        let s_ref = compute_segregation(&sim_ref);
        println!("  S(200) = {:.6}", s_ref);
        println!("  Time: {:.2}s ({:.1} ms/step)\n", t_ref, t_ref / n_steps as f64 * 1000.0);

        // Save reference final positions for physics comparison
        let _ref_final_pos = sim_ref.positions();

        // Test each rebuild interval
        for &interval in &rebuild_intervals {
            if interval == 1 { continue; }  // Skip reference case

            println!("Testing INCREMENTAL (rebuild every {} steps)...", interval);

            // Create new simulation with same initial conditions
            let mut sim_test = GpuNBodySimulation::new_with_state(
                n_positive, n_negative, box_size,
                initial_pos.clone(),
                vec![0.0; n_particles * 3],  // Same zero initial velocities
                signs.clone(),
            ).expect("Failed to create test simulation");

            let t_test_start = Instant::now();
            for step in 0..n_steps {
                sim_test.step_with_expansion_dkd_gpu_incremental(
                    dt, 1.0, 0.0, 0.0,
                    step, interval,
                ).expect("Test step failed");
            }
            let t_test = t_test_start.elapsed().as_secs_f64();
            let s_test = compute_segregation(&sim_test);

            // Compare with reference
            let s_diff = ((s_test - s_ref) / s_ref * 100.0).abs();
            let speedup = t_ref / t_test;

            let status = if s_diff < 5.0 { "✓" } else { "✗" };

            println!("  S(200) = {:.6} (diff: {:.2}%) {}", s_test, s_diff, status);
            println!("  Time: {:.2}s ({:.1} ms/step)", t_test, t_test / n_steps as f64 * 1000.0);
            println!("  Speedup: {:.2}×\n", speedup);
        }

        // Also test at 2M for final performance target
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("Performance test @ 2M particles (20 steps)\n");

        let n_2m = 2_000_000;
        let box_2m = 100.0 * (n_2m as f64 / 100_000.0).powf(1.0/3.0);
        let n_pos_2m = (n_2m as f64 / (1.0 + eta)) as usize;
        let n_neg_2m = n_2m - n_pos_2m;
        let n_steps_2m = 20;

        let mut sim_2m = GpuNBodySimulation::new(n_pos_2m, n_neg_2m, box_2m)
            .expect("Failed to create 2M simulation");

        // Full rebuild every step
        let t_full_start = Instant::now();
        for step in 0..n_steps_2m {
            sim_2m.step_with_expansion_dkd_gpu_incremental(
                dt, 1.0, 0.0, 0.0,
                step, 1,
            ).expect("Full step failed");
        }
        let t_full = t_full_start.elapsed().as_secs_f64();
        println!("Full rebuild every step:  {:.1} ms/step", t_full / n_steps_2m as f64 * 1000.0);

        // Reset
        let mut sim_2m = GpuNBodySimulation::new(n_pos_2m, n_neg_2m, box_2m)
            .expect("Failed to create 2M simulation");

        // Incremental (rebuild every 10 steps)
        let t_inc_start = Instant::now();
        for step in 0..n_steps_2m {
            sim_2m.step_with_expansion_dkd_gpu_incremental(
                dt, 1.0, 0.0, 0.0,
                step, 10,
            ).expect("Incremental step failed");
        }
        let t_inc = t_inc_start.elapsed().as_secs_f64();
        let avg_ms = t_inc / n_steps_2m as f64 * 1000.0;
        let target_status = if avg_ms < 200.0 { "✓ TARGET MET" } else { "✗ Target missed" };
        println!("Incremental (every 10):   {:.1} ms/step {}", avg_ms, target_status);
        println!("Speedup:                  {:.2}×", t_full / t_inc);
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}

#[cfg(feature = "cuda")]
fn compute_segregation(sim: &GpuNBodySimulation) -> f64 {
    let pos = sim.positions();
    let signs = sim.signs();
    let n = signs.len();

    // Sample ~1000 random pairs for speed
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(12345);

    let n_samples = 1000;
    let mut same_sign_sum = 0.0;
    let mut diff_sign_sum = 0.0;

    for _ in 0..n_samples {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        if i == j { continue; }

        let dx = pos[i*3] - pos[j*3];
        let dy = pos[i*3+1] - pos[j*3+1];
        let dz = pos[i*3+2] - pos[j*3+2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();

        if signs[i] == signs[j] {
            same_sign_sum += 1.0 / r;
        } else {
            diff_sign_sum += 1.0 / r;
        }
    }

    // S = ratio of same-sign to diff-sign density
    if diff_sign_sum > 0.0 {
        same_sign_sum / diff_sign_sum
    } else {
        1.0
    }
}
