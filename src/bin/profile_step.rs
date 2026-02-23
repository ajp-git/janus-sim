/// Profile full step breakdown at 2M particles

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

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Step Profiling @ 2M particles                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
            .expect("Failed to create simulation");

        let dt = 0.003;

        // Warm up (first step has CUDA JIT overhead)
        println!("Warming up (step 0)...");
        let t_warmup = Instant::now();
        sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0)
            .expect("Warmup step failed");
        println!("Warmup time: {:.1} ms\n", t_warmup.elapsed().as_secs_f64() * 1000.0);

        // Profile 5 steps
        println!("Profiling 5 steps with step_with_expansion_dkd_gpu...\n");
        let mut times = Vec::new();
        for i in 1..=5 {
            let t = Instant::now();
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0)
                .expect("Step failed");
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            times.push(elapsed);
            println!("Step {}: {:.1} ms", i, elapsed);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        println!("Average: {:.1} ms/step\n", avg);

        // Now test incremental method
        let mut sim2 = GpuNBodySimulation::new(n_positive, n_negative, box_size)
            .expect("Failed to create simulation");

        // Warm up
        println!("Testing incremental method...");
        sim2.step_with_expansion_dkd_gpu_incremental(dt, 1.0, 0.0, 0.0, 0, 10)
            .expect("Warmup failed");

        // Profile
        let mut times_inc = Vec::new();
        for i in 1..=10 {
            let t = Instant::now();
            sim2.step_with_expansion_dkd_gpu_incremental(dt, 1.0, 0.0, 0.0, i, 10)
                .expect("Step failed");
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            times_inc.push(elapsed);
            let rebuild = if i % 10 == 0 { " (rebuild)" } else { "" };
            println!("Step {}: {:.1} ms{}", i, elapsed, rebuild);
        }
        let avg_inc = times_inc.iter().sum::<f64>() / times_inc.len() as f64;
        let avg_inc_only = times_inc[1..9].iter().sum::<f64>() / 8.0;  // Skip rebuilds
        println!("Average: {:.1} ms/step (incremental only: {:.1} ms)", avg_inc, avg_inc_only);
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
