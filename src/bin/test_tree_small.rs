//! Test tree building with small N

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing tree with small N...");

    // Test with 1M particles
    let n_positive = 500_000;
    let n_negative = 500_000;
    let box_size = 200.0;

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);

    println!("\nComputing forces...");
    sim.compute_forces()?;

    let acc_sum = sim.acceleration_sum()?;
    println!("\nΣ|acc| = {:.4e}", acc_sum);

    if acc_sum > 0.0 {
        println!("SUCCESS: Forces computed");
    } else {
        println!("FAILURE: Forces are zero");
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("CUDA required");
}
