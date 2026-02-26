//! Test nbody_gpu.rs (single-tree) at different theta

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("=== nbody_gpu.rs Theta Test (1M) ===\n");

    let mut sim = GpuNBodySimulation::new_bvh_only(n_positive, n_negative, box_size)?;

    for &theta in &[1.0, 0.7, 0.5] {
        sim.set_theta(theta);

        // Warm up
        sim.step(0.01)?;

        // Measure
        let t0 = std::time::Instant::now();
        sim.step(0.01)?;
        let ms = t0.elapsed().as_millis();

        println!("θ = {:.1} → {} ms", theta, ms);
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
