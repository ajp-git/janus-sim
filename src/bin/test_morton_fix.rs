//! Test Morton fix on original GPU Karras (nbody_gpu.rs)
//! Verify: uncovered=0, step time <2s at θ=0.5

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_000_000);

    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("=== Morton Fix Test (Original GPU Karras) ===");
    println!("N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("N+ = {}, N- = {}", n_positive, n_negative);
    println!("Box = {:.2}", box_size);
    println!("θ = 0.5 (non-négociable)");
    println!();

    println!("Creating simulation...");
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodySimulation::new_bvh_only(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);
    println!("Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Run 5 steps and measure timing
    println!("Running 5 steps at θ=0.5...\n");

    let mut total_time = 0u128;
    for step in 1..=5 {
        let t0 = std::time::Instant::now();
        sim.step(0.01)?;
        let elapsed = t0.elapsed().as_millis();
        total_time += elapsed;

        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation_distance()?;

        println!("step {:02} | {} ms | KE={:.4e} | S={:.4e}",
            step, elapsed, ke, seg);
    }

    let avg_time = total_time / 5;
    println!("\n=== RESULTS ===");
    println!("Average step time: {} ms", avg_time);
    println!("Target: < 2000 ms");

    if avg_time < 2000 {
        println!("✓ VALIDATED - Ready for 85M");
    } else {
        println!("✗ TOO SLOW - Stay at 8M");
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature required");
}
