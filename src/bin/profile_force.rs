/// Profile force computation kernel at 2M particles

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;
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
        println!("║   Force Computation Profiling @ 2M particles                   ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
            .expect("Failed to create simulation");

        println!("Building tree...");
        let t_tree = Instant::now();
        sim.build_gpu_tree().expect("Tree build failed");
        println!("Tree built in {:.1} ms\n", t_tree.elapsed().as_secs_f64() * 1000.0);

        // Profile force computation alone
        println!("Profiling force computation (5 runs)...\n");

        // The force computation is done via step - let me just run compute_forces_bvh
        // We need direct access to the kernel, but for now let's profile via step

        let mut force_times = Vec::new();
        for i in 1..=5 {
            // Build tree first
            sim.build_gpu_tree().expect("Tree build failed");

            // Time the step (includes drift/kick but not tree rebuild)
            // Actually, step_with_expansion_dkd_gpu rebuilds tree every time

            // Let's just measure multiple tree builds + force to get an idea
            let t = Instant::now();

            // Do what the step does after tree build:
            // - compute_forces_bvh
            // - kick/drift
            // But we can't easily separate these without modifying code

            // Let's just run a full step and subtract tree time
            sim.step_with_expansion_dkd_gpu(0.003, 1.0, 0.0, 0.0)
                .expect("Step failed");
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            force_times.push(elapsed);
            println!("Step {} (incl tree): {:.1} ms", i, elapsed);
        }

        let avg = force_times.iter().sum::<f64>() / force_times.len() as f64;
        println!("\nAverage step: {:.1} ms", avg);
        println!("Tree build: ~280 ms");
        println!("Force+integrator: ~{:.1} ms", avg - 280.0);

        // Check tree quality by examining root node
        println!("\nChecking tree quality...");
        sim.debug_bvh_structure().expect("BVH debug failed");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
