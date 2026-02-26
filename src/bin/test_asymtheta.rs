/// Test asymmetric theta optimization
/// Validates S(t) accuracy and measures speedup

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
        println!("║   Asymmetric Theta Validation                                  ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Particles: {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
        println!("Box size: {:.1}", box_size);
        println!("Steps: {}", n_steps);
        println!();

        // Create simulation
        println!("Creating simulation...");
        let mut gpu_sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");

        gpu_sim.virialize().expect("Virialization failed");

        // Cosmological params (constant for test)
        let a = 1.0;
        let h = 0.0;
        let dtau_per_dt = 0.0;

        // Get initial segregation
        let seg_0 = gpu_sim.segregation_distance().expect("Failed to compute segregation");
        println!("Initial segregation S(0) = {:.6}", seg_0);

        // Run simulation with asymmetric theta (now built-in)
        println!("\n═══ Running with Asymmetric θ ═══");
        println!("  θ_same = 0.5 (attraction, tight)");
        println!("  θ_opp  = 1.0 (repulsion, loose)");
        println!();

        let mut times = Vec::with_capacity(n_steps);
        let mut seg_values = Vec::with_capacity(n_steps);

        for step in 1..=n_steps {
            let step_start = Instant::now();

            // Using Morton+DKD with asymmetric theta
            gpu_sim.step_with_expansion_dkd_morton(dt, a, h, dtau_per_dt)
                .expect("Step failed");

            let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
            times.push(step_time);

            let seg = gpu_sim.segregation_distance().expect("Failed to compute segregation");
            seg_values.push(seg);

            let seg_change = (seg - seg_0) / seg_0 * 100.0;
            println!("  Step {:2}: {:.0} ms | S = {:.6} ({:+.2}%)",
                     step, step_time, seg, seg_change);
        }

        let avg_time = times.iter().sum::<f64>() / n_steps as f64;
        let final_seg = *seg_values.last().unwrap();

        // Results
        println!("\n══════════════════════════════════════════════════════════════════");
        println!("                      RESULTS                                      ");
        println!("══════════════════════════════════════════════════════════════════");
        println!();
        println!("  Time/step: {:.1} ms average", avg_time);
        println!();
        println!("  S(0) = {:.6}", seg_0);
        println!("  S({}) = {:.6}", n_steps, final_seg);
        println!("  ΔS = {:.2}%", (final_seg - seg_0) / seg_0 * 100.0);
        println!();

        // Estimate for 8M particles
        let scale_factor = 4.4;
        let time_8m = avg_time * scale_factor;
        println!("  Estimated @ 8M particles: {:.0} ms/step", time_8m);
        println!("  Time for 6000 steps: {:.1}h", time_8m * 6000.0 / 1000.0 / 3600.0);
        println!("══════════════════════════════════════════════════════════════════");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support. Build with --features cuda");
    }
}
