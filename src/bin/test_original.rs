// Test original nbody_gpu.rs performance
#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        let n = 85_000_000;
        let eta = 1.045;
        let n_positive = (n as f64 / (1.0 + eta)) as usize;
        let n_negative = n - n_positive;
        let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

        println!("╔═══════════════════════════════════════════════════╗");
        println!("║   Test Original nbody_gpu.rs — 8M particles       ║");
        println!("╚═══════════════════════════════════════════════════╝\n");

        println!("Parameters:");
        println!("  N = {} ({:.1}M)", n, n as f64 / 1e6);
        println!("  N+ = {}, N- = {}", n_positive, n_negative);
        println!("  box = {:.2}", box_size);
        // Test with θ=0.5 (from janus_85m.rs) and θ=2.0 (fast)
        for theta in [0.5, 2.0] {
            println!("\n═══ Testing θ = {} ═══", theta);

            println!("Creating simulation (new_bvh_only)...");
            let t0 = Instant::now();
            let mut sim = GpuNBodySimulation::new_bvh_only(n_positive, n_negative, box_size)?;
            println!("  Created in {:.2}s", t0.elapsed().as_secs_f64());

            sim.set_theta(theta);

            println!("\nRunning 3 steps...");
            for step in 1..=3 {
                let t0 = Instant::now();
                sim.step_with_expansion_dkd_gpu(0.005, 1.0, 0.0, 0.0)?;
                let elapsed = t0.elapsed().as_millis();
                let seg = sim.segregation_distance()?;
                println!("  Step {}: {} ms | S = {:.4}", step, elapsed, seg);
            }
        }

        println!("\nTarget: ~823 ms/step @ 85M θ=0.5");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled!");
    }

    Ok(())
}
