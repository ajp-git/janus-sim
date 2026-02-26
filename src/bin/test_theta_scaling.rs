//! Test force kernel scaling with different theta values

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1_000_000;  // 1M for quick test
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("=== Theta Scaling Test (1M particles) ===\n");

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;

    // Test different theta values
    let thetas = [5.0, 3.0, 2.0, 1.5, 1.0, 0.7];

    for &theta in &thetas {
        sim.set_theta(theta);

        // Warm up
        sim.step_dkd(0.01, 0.0, 0.0)?;

        // Measure
        let t0 = std::time::Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        let ms = t0.elapsed().as_millis();

        println!("θ = {:.1} → {} ms", theta, ms);
    }

    println!("\nExpected scaling: time ∝ θ^(-3) for low θ");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
