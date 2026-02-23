/// Compare force computation: GPU BVH vs CPU tree

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 100_000;  // Start small
        let eta = 1.045;
        let box_size = 100.0;

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Force Method Comparison @ {} particles                  ║", n_particles);
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let dt = 0.003;

        // Test 1: GPU BVH (step_with_expansion_dkd_gpu)
        println!("Method 1: GPU BVH (step_with_expansion_dkd_gpu)");
        {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed to create simulation");

            // Warmup
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

            let mut times = Vec::new();
            for i in 0..5 {
                let t = Instant::now();
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step failed");
                let elapsed = t.elapsed().as_secs_f64() * 1000.0;
                times.push(elapsed);
                println!("  Step {}: {:.1} ms", i+1, elapsed);
            }
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            println!("  Average: {:.1} ms/step\n", avg);
        }

        // Test 2: Morton sorted (step_with_expansion_dkd_morton)
        println!("Method 2: Morton sorted (step_with_expansion_dkd_morton)");
        {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed to create simulation");

            // Warmup
            sim.step_with_expansion_dkd_morton(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

            let mut times = Vec::new();
            for i in 0..5 {
                let t = Instant::now();
                sim.step_with_expansion_dkd_morton(dt, 1.0, 0.0, 0.0).expect("Step failed");
                let elapsed = t.elapsed().as_secs_f64() * 1000.0;
                times.push(elapsed);
                println!("  Step {}: {:.1} ms", i+1, elapsed);
            }
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            println!("  Average: {:.1} ms/step\n", avg);
        }

        // Test 3: Basic DKD (step_with_expansion_dkd)
        println!("Method 3: Basic DKD (step_with_expansion_dkd)");
        {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed to create simulation");

            // Warmup
            sim.step_with_expansion_dkd(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

            let mut times = Vec::new();
            for i in 0..5 {
                let t = Instant::now();
                sim.step_with_expansion_dkd(dt, 1.0, 0.0, 0.0).expect("Step failed");
                let elapsed = t.elapsed().as_secs_f64() * 1000.0;
                times.push(elapsed);
                println!("  Step {}: {:.1} ms", i+1, elapsed);
            }
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            println!("  Average: {:.1} ms/step\n", avg);
        }

        // Now test at larger scale
        println!("═══════════════════════════════════════════════════════════════════");
        println!("Scaling test at 500K and 2M particles\n");

        for n in [500_000, 2_000_000] {
            let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
            let n_pos = (n as f64 / (1.0 + eta)) as usize;
            let n_neg = n - n_pos;

            println!("@ {} particles:", n);

            // GPU BVH
            {
                let mut sim = GpuNBodySimulation::new(n_pos, n_neg, box_size)
                    .expect("Failed");
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup");
                let t = Instant::now();
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step");
                println!("  GPU BVH: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            }

            // Morton (CPU tree)
            {
                let mut sim = GpuNBodySimulation::new(n_pos, n_neg, box_size)
                    .expect("Failed");
                sim.step_with_expansion_dkd_morton(dt, 1.0, 0.0, 0.0).expect("Warmup");
                let t = Instant::now();
                sim.step_with_expansion_dkd_morton(dt, 1.0, 0.0, 0.0).expect("Step");
                println!("  Morton (CPU tree): {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            }
            println!();
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
