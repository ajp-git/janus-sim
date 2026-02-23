/// Profile full DKD step breakdown at 2M particles

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
        println!("║   Full DKD Step Profiling @ 2M particles                       ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
            .expect("Failed to create simulation");

        let dt = 0.003;

        // Warm up (includes CUDA JIT compilation)
        println!("Warming up (CUDA JIT compilation)...");
        let t_warmup = Instant::now();
        sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0)
            .expect("Warmup failed");
        println!("Warmup time: {:.1} ms (includes JIT)\n", t_warmup.elapsed().as_secs_f64() * 1000.0);

        // Profile individual components
        println!("Profiling step components...\n");

        // Profile tree build alone
        let t_tree = Instant::now();
        sim.build_gpu_tree_profiled().expect("Tree build failed");
        let tree_time = t_tree.elapsed().as_secs_f64() * 1000.0;
        println!("\nTree build total: {:.1} ms\n", tree_time);

        // Profile full steps
        println!("Profiling 5 full steps...\n");
        let mut step_times = Vec::new();
        for i in 1..=5 {
            let t = Instant::now();
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0)
                .expect("Step failed");
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            step_times.push(elapsed);
            println!("Step {}: {:.1} ms", i, elapsed);
        }
        let avg = step_times.iter().sum::<f64>() / step_times.len() as f64;
        println!("\nAverage step time: {:.1} ms", avg);
        println!("Implied force+integrator time: {:.1} ms (step - tree)", avg - tree_time);
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
