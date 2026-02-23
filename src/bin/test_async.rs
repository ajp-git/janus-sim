/// Test async multi-stream pipelining (opt7)
/// Compare S(t) and performance between sync and async methods

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Async Multi-Stream Pipeline Test (opt7)                      ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        // Test at 500K first for validation
        test_at_scale(500_000, 100, 0.003);

        // Then test at 2M for performance
        test_at_scale(2_000_000, 20, 0.003);
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}

#[cfg(feature = "cuda")]
fn test_at_scale(n_particles: usize, n_steps: usize, dt: f64) {
    let eta = 1.045;
    let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

    let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
    let n_negative = n_particles - n_positive;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("Testing @ {} particles, {} steps\n", n_particles, n_steps);

    // ═══════════════════════════════════════════════════════════════════════
    // Reference: step_with_expansion_dkd_gpu (sync)
    // ═══════════════════════════════════════════════════════════════════════
    println!("Running REFERENCE (step_with_expansion_dkd_gpu)...");
    let mut sim_ref = GpuNBodySimulation::new(n_positive, n_negative, box_size)
        .expect("Failed to create reference simulation");

    // Warm up
    sim_ref.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0)
        .expect("Warmup failed");

    let t_ref_start = Instant::now();
    for _ in 0..n_steps {
        sim_ref.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0)
            .expect("Reference step failed");
    }
    let t_ref = t_ref_start.elapsed().as_secs_f64();
    let s_ref = compute_segregation(&sim_ref);
    let ms_per_step_ref = t_ref / n_steps as f64 * 1000.0;
    println!("  S({}) = {:.6}", n_steps, s_ref);
    println!("  Time: {:.2}s ({:.1} ms/step)\n", t_ref, ms_per_step_ref);

    // ═══════════════════════════════════════════════════════════════════════
    // Test: step_with_expansion_dkd_gpu_async
    // ═══════════════════════════════════════════════════════════════════════
    println!("Running TEST (step_with_expansion_dkd_gpu_async)...");
    let mut sim_async = GpuNBodySimulation::new(n_positive, n_negative, box_size)
        .expect("Failed to create async simulation");

    // Warm up (this initializes buffers)
    sim_async.step_with_expansion_dkd_gpu_async(dt, 1.0, 0.0, 0.0, 0)
        .expect("Async warmup failed");

    let t_async_start = Instant::now();
    for step in 1..=n_steps {
        sim_async.step_with_expansion_dkd_gpu_async(dt, 1.0, 0.0, 0.0, step)
            .expect("Async step failed");
    }
    let t_async = t_async_start.elapsed().as_secs_f64();
    let s_async = compute_segregation(&sim_async);
    let ms_per_step_async = t_async / n_steps as f64 * 1000.0;

    // Compare
    let s_diff = ((s_async - s_ref) / s_ref * 100.0).abs();
    let speedup = t_ref / t_async;
    let physics_ok = s_diff < 5.0;
    let target_ok = n_particles == 2_000_000 && ms_per_step_async < 250.0;

    let status = if physics_ok { "✓" } else { "✗" };
    println!("  S({}) = {:.6} (diff: {:.2}%) {}", n_steps, s_async, s_diff, status);
    println!("  Time: {:.2}s ({:.1} ms/step)", t_async, ms_per_step_async);
    println!("  Speedup: {:.2}×", speedup);

    if n_particles == 2_000_000 {
        let target_status = if target_ok { "✓ TARGET MET" } else { "✗ Target missed" };
        println!("  Target <250 ms/step: {}", target_status);
    }
    println!();
}

#[cfg(feature = "cuda")]
fn compute_segregation(sim: &GpuNBodySimulation) -> f64 {
    let pos = sim.positions();
    let signs = sim.signs();
    let n = signs.len();

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

    if diff_sign_sum > 0.0 {
        same_sign_sum / diff_sign_sum
    } else {
        1.0
    }
}
