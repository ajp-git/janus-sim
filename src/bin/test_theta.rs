/// Test Barnes-Hut theta parameter effect on performance

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 500_000;  // Use 500K for faster testing
        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let dt = 0.003;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Barnes-Hut Theta Parameter Test @ 500K particles             ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        // Test different theta values
        let theta_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0];

        for &theta in &theta_values {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed to create simulation");
            sim.set_theta(theta);

            // Warmup
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup failed");

            // Time 3 steps
            let t = Instant::now();
            for _ in 0..3 {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step failed");
            }
            let avg = t.elapsed().as_secs_f64() / 3.0 * 1000.0;
            println!("theta = {:.1}: {:.1} ms/step", theta, avg);
        }

        println!("\nHigher theta = more nodes approximated = faster but less accurate");
        println!("If time doesn't decrease with theta, something is wrong with tree traversal");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
