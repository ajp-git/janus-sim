/// Final benchmark: optimized GPU BVH at 2M particles

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 2_000_000;
        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let dt = 0.003;
        let n_steps = 100;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Final Benchmark: GPU BVH @ 2M particles                      ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("Configuration:");
        println!("  N particles:   {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
        println!("  N positive:    {} ({:.1}%)", n_positive, n_positive as f64 / n_particles as f64 * 100.0);
        println!("  N negative:    {} ({:.1}%)", n_negative, n_negative as f64 / n_particles as f64 * 100.0);
        println!("  Box size:      {:.1}", box_size);
        println!("  dt:            {}", dt);
        println!("  Steps:         {}\n", n_steps);

        // Create simulation with default theta=2.0
        let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
            .expect("Failed to create simulation");

        println!("Default theta: {:.1}", sim.get_theta());

        // Warmup
        println!("\nWarming up...");
        sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

        // Record initial segregation
        let s_0 = compute_segregation(&sim);
        println!("Initial S(0) = {:.6}\n", s_0);

        // Run simulation
        println!("Running {} steps...\n", n_steps);
        let mut step_times = Vec::with_capacity(n_steps);

        for step in 1..=n_steps {
            let t = Instant::now();
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step failed");
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            step_times.push(elapsed);

            if step % 20 == 0 {
                let s = compute_segregation(&sim);
                let avg = step_times.iter().sum::<f64>() / step_times.len() as f64;
                println!("Step {:3}: S = {:.6}, avg = {:.1} ms/step", step, s, avg);
            }
        }

        let s_final = compute_segregation(&sim);
        let avg_time = step_times.iter().sum::<f64>() / step_times.len() as f64;
        let min_time = step_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = step_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                       FINAL RESULTS                            ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("Physics:");
        println!("  S(0)   = {:.6}", s_0);
        println!("  S({:3}) = {:.6}", n_steps, s_final);
        println!("  Change = {:.2}%\n", (s_final - s_0) / s_0 * 100.0);

        println!("Performance @ 2M particles:");
        println!("  Average:  {:.1} ms/step", avg_time);
        println!("  Min:      {:.1} ms/step", min_time);
        println!("  Max:      {:.1} ms/step", max_time);
        println!();

        // Check target
        if avg_time < 250.0 {
            println!("  ✓ TARGET MET: {:.1} ms < 250 ms", avg_time);
        } else {
            println!("  ✗ Target missed: {:.1} ms >= 250 ms", avg_time);
        }

        // Throughput
        let particles_per_sec = n_particles as f64 / (avg_time / 1000.0);
        println!("  Throughput: {:.1}M particles/sec", particles_per_sec / 1e6);
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

    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(12345);

    let n_samples = 2000;
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
