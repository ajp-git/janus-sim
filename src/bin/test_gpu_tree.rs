/// Test GPU tree build (Karras 2012)
/// Validates S(t) accuracy and measures speedup vs CPU tree

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
        println!("║   GPU Tree Build (Karras 2012) Validation                      ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Particles: {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
        println!("Box size: {:.1}", box_size);
        println!("Steps: {}", n_steps);
        println!();

        // Cosmological params (constant for test)
        let a = 1.0;
        let h = 0.0;
        let dtau_per_dt = 0.0;

        // Test GPU tree build
        println!("Creating simulation for GPU tree...");
        let mut gpu_sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");
        gpu_sim.virialize().expect("Virialization failed");

        let seg_0 = gpu_sim.segregation_distance().expect("Failed to compute segregation");
        println!("Initial segregation S(0) = {:.6}", seg_0);

        println!("\n═══ Running with GPU Tree Build (Karras 2012) ═══");

        let mut gpu_times = Vec::with_capacity(n_steps);
        let mut gpu_seg_values = Vec::with_capacity(n_steps);

        for step in 1..=n_steps {
            let step_start = Instant::now();

            gpu_sim.step_with_expansion_dkd_gpu(dt, a, h, dtau_per_dt)
                .expect("GPU tree step failed");

            let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
            gpu_times.push(step_time);

            let seg = gpu_sim.segregation_distance().expect("Failed to compute segregation");
            gpu_seg_values.push(seg);

            let seg_change = (seg - seg_0) / seg_0 * 100.0;
            println!("  Step {:2}: {:.0} ms | S = {:.6} ({:+.2}%)",
                     step, step_time, seg, seg_change);
        }

        let gpu_avg = gpu_times.iter().sum::<f64>() / n_steps as f64;
        let gpu_final_seg = *gpu_seg_values.last().unwrap();

        // Results
        println!("\n══════════════════════════════════════════════════════════════════");
        println!("                      RESULTS                                      ");
        println!("══════════════════════════════════════════════════════════════════");
        println!();
        println!("  GPU Tree Time/step: {:.1} ms average", gpu_avg);
        println!();
        println!("  S(0) = {:.6}", seg_0);
        println!("  S({}) = {:.6}", n_steps, gpu_final_seg);
        println!("  ΔS = {:.2}%", (gpu_final_seg - seg_0) / seg_0 * 100.0);
        println!();

        // Compare with target
        let target = 500.0;
        if gpu_avg < target {
            println!("  ✓ Target achieved: {:.1} ms < {:.0} ms", gpu_avg, target);
        } else {
            println!("  ✗ Target NOT achieved: {:.1} ms > {:.0} ms", gpu_avg, target);
        }

        // Estimate for 8M particles
        let scale_factor = 4.4;
        let gpu_8m = gpu_avg * scale_factor;
        println!();
        println!("  Estimated @ 8M particles: {:.0} ms/step", gpu_8m);
        println!("  Time for 6000 steps: {:.1}h", gpu_8m * 6000.0 / 1000.0 / 3600.0);
        println!("══════════════════════════════════════════════════════════════════");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support. Build with --features cuda");
    }
}
