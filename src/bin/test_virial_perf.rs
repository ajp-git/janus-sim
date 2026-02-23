/// Test performance with virialized (clustered) vs uniform initial conditions

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Performance: Uniform vs Virialized Initial Conditions        ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let dt = 0.01;

        for n_particles in [100_000, 500_000, 2_000_000] {
            let eta = 1.045;
            let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
            let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
            let n_negative = n_particles - n_positive;

            println!("═══════════════════════════════════════════════════════════════════");
            println!("@ {} particles\n", n_particles);

            // Test 1: Uniform distribution (no virialization)
            println!("Uniform distribution (random):");
            {
                let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                    .expect("Failed to create simulation");

                // Warmup
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

                // Time 5 steps
                let t = Instant::now();
                for _ in 0..5 {
                    sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step failed");
                }
                let avg = t.elapsed().as_secs_f64() / 5.0 * 1000.0;
                println!("  GPU BVH: {:.1} ms/step", avg);
            }

            // Test 2: Virialized distribution (clustered)
            println!("Virialized distribution (clustered):");
            {
                let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                    .expect("Failed to create simulation");

                // Virialize first
                sim.virialize().expect("Virialization failed");

                // Warmup
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

                // Time 5 steps
                let t = Instant::now();
                for _ in 0..5 {
                    sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step failed");
                }
                let avg = t.elapsed().as_secs_f64() / 5.0 * 1000.0;
                println!("  GPU BVH: {:.1} ms/step", avg);
            }
            println!();
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
