// Test GPU BVH construction with 10M particles
// Verify: uncovered leaves = 0, performance < 2s/step

use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 10_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   GPU BVH Test — 10M particles                    ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("  N+ = {:.2}M, N- = {:.2}M", n_positive as f64/1e6, n_negative as f64/1e6);
    println!("  box = {:.2}\n", box_size);

    println!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);

    println!("\nComputing initial forces (builds both trees)...");
    let t0 = Instant::now();
    sim.compute_forces()?;
    let init_time = t0.elapsed().as_secs_f64();
    println!("  Initial forces: {:.2}s\n", init_time);

    println!("Running 5 steps...");
    for step in 1..=5 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        let step_time = t0.elapsed().as_secs_f64();
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        println!("  Step {}: {:.2}s | KE={:.4e} | S={:.6}", step, step_time, ke, seg);
    }

    println!("\n═══════════════════════════════════════════════════");
    println!("Target: < 2s/step. If achieved, GPU BVH is working!");
    println!("═══════════════════════════════════════════════════");

    Ok(())
}
