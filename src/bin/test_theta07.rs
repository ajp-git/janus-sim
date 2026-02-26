/// Test θ values at 2M particles - measure time/step

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
        let n_steps = 10;

        println!("θ comparison @ 2M particles\n");

        for &theta in &[0.7, 1.0, 1.5, 2.0] {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size).unwrap();
            sim.set_theta(theta);

            // Warmup
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).unwrap();

            let t = Instant::now();
            for _ in 0..n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).unwrap();
            }
            let avg = t.elapsed().as_secs_f64() / n_steps as f64 * 1000.0;

            let status = if avg < 250.0 { "✓" } else { "✗" };
            println!("θ={:.1}: {:>7.1} ms/step {}", theta, avg, status);
        }

        println!("\nTarget: < 250 ms/step");
    }

    #[cfg(not(feature = "cuda"))]
    eprintln!("Requires CUDA");
}
