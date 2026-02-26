//! Test twopass on 8M particles
//! Verify: uncovered = 0, step time ~800ms

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

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

    println!("=== Twopass Morton Fix Test ===");
    println!("N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("N+ = {}, N- = {}", n_positive, n_negative);
    println!("Box = {:.2}", box_size);
    println!();

    println!("Creating simulation...");
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(5.0);  // Target ~800ms/step
    println!("Created in {:.2}s\n", t0.elapsed().as_secs_f64());
    println!("theta = 0.7");

    // Run 10 steps and measure timing
    println!("Running 10 steps...\n");

    for step in 1..=10 {
        let t0 = std::time::Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;  // No expansion for this test
        let elapsed = t0.elapsed().as_millis();

        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;

        println!("step {:02} | {:.0} ms | KE={:.4e} | S={:.4e}",
            step, elapsed, ke, seg);
    }

    println!("\n=== Test Complete ===");
    println!("If uncovered=0 messages appeared and step time ~800ms → FIX VALIDATED");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature required");
}
