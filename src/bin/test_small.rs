// Quick test with 1M particles
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("Creating sim: N={} box={:.2}", n, box_size);
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);

    println!("Computing initial forces...");
    sim.compute_forces()?;

    println!("Running 3 steps...");
    for step in 1..=3 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        println!("step {} done in {:.0}ms", step, t0.elapsed().as_millis());
    }

    Ok(())
}
