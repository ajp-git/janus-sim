//! Profile twopass force kernel at θ=0.5

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("=== Twopass Force Profile (stack[32], θ=0.5) ===");
    println!("N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("N+ = {}, N- = {}", n_positive, n_negative);

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);

    // Warm up
    println!("\nWarm-up step...");
    sim.step_dkd(0.01, 0.0, 0.0)?;

    // Measure
    println!("\nMeasuring 3 steps:");
    for i in 1..=3 {
        let t0 = std::time::Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i, ms);
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
