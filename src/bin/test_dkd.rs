/// Quick DKD/Morton benchmark
/// Compare integrator performance on 2M particles for 10 steps

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_000_000);

        let n_steps: usize = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let dt = 0.01;

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   DKD + Morton Sorting Benchmark                               ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Particles: {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
        println!("Box size: {:.1}", box_size);
        println!("Steps: {}", n_steps);
        println!();

        // Fake cosmological params (constant for benchmark)
        let a = 1.0;
        let h = 0.0;
        let dtau_per_dt = 0.0;

        // Test Morton+DKD
        println!("Creating simulation for Morton+DKD...");
        let mut gpu_sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");
        gpu_sim.virialize().expect("Virialization failed");

        println!("\n═══ Testing Morton+DKD integrator ═══");
        let morton_start = Instant::now();
        let mut morton_times = Vec::with_capacity(n_steps);

        for step in 1..=n_steps {
            let step_start = Instant::now();
            gpu_sim.step_with_expansion_dkd_morton(dt, a, h, dtau_per_dt)
                .expect("Morton+DKD step failed");
            let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
            morton_times.push(step_time);
            println!("  Step {}: {:.0} ms", step, step_time);
        }
        let morton_total = morton_start.elapsed().as_secs_f64();
        let morton_avg = morton_times.iter().sum::<f64>() / n_steps as f64;

        // Reset simulation for DKD test
        drop(gpu_sim);
        println!("\nCreating simulation for DKD...");
        let mut gpu_sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");
        gpu_sim.virialize().expect("Virialization failed");

        // Test DKD (no Morton)
        println!("\n═══ Testing DKD integrator (no Morton) ═══");
        let dkd_start = Instant::now();
        let mut dkd_times = Vec::with_capacity(n_steps);

        for step in 1..=n_steps {
            let step_start = Instant::now();
            gpu_sim.step_with_expansion_dkd(dt, a, h, dtau_per_dt)
                .expect("DKD step failed");
            let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
            dkd_times.push(step_time);
            println!("  Step {}: {:.0} ms", step, step_time);
        }
        let dkd_total = dkd_start.elapsed().as_secs_f64();
        let dkd_avg = dkd_times.iter().sum::<f64>() / n_steps as f64;

        // Results
        println!("\n══════════════════════════════════════════════════════════════════");
        println!("                      RESULTS                                      ");
        println!("══════════════════════════════════════════════════════════════════");
        println!();
        println!("  Morton+DKD: {:.1} ms/step average ({:.1}s total)", morton_avg, morton_total);
        println!("  DKD only:   {:.1} ms/step average ({:.1}s total)", dkd_avg, dkd_total);
        println!();

        if morton_avg < dkd_avg {
            println!("  Morton speedup: {:.2}x", dkd_avg / morton_avg);
        } else {
            println!("  Morton overhead: {:.2}x slower", morton_avg / dkd_avg);
        }
        println!();

        // Estimate for 8M particles
        let scale_factor = 4.4;
        let morton_8m = morton_avg * scale_factor;
        let dkd_8m = dkd_avg * scale_factor;

        println!("  Estimated for 8M particles:");
        println!("    Morton+DKD: {:.0} ms/step", morton_8m);
        println!("    DKD only:   {:.0} ms/step", dkd_8m);
        println!();

        // Time for 6000 steps at 8M
        let morton_12h = morton_8m * 6000.0 / 1000.0 / 3600.0;
        let dkd_12h = dkd_8m * 6000.0 / 1000.0 / 3600.0;
        println!("  Time for 6000 steps at 8M:");
        println!("    Morton+DKD: {:.1}h", morton_12h);
        println!("    DKD only:   {:.1}h", dkd_12h);
        println!("══════════════════════════════════════════════════════════════════");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support. Build with --features cuda");
    }
}
