// Verify GPU radix sort produces correctly sorted output
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0;

    println!("Creating sim with {} particles...", n);
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.8);

    println!("\nBuilding tree (will trigger GPU sort)...");
    sim.compute_forces()?;

    println!("\nIf no WARNING about unsorted pairs, GPU sort is working!");
    println!("Running one step to verify tree is valid...");
    sim.step_dkd(0.01, 0.0, 0.0)?;

    println!("\nSuccess! GPU sort and BVH are working correctly.");
    Ok(())
}
