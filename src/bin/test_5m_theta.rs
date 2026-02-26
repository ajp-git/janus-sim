// Test force computation with 5M particles at different θ values
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 5_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   θ Performance Test — 5M particles               ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("  N+ = {:.2}M, N- = {:.2}M", n_positive as f64/1e6, n_negative as f64/1e6);
    println!("  box = {:.2}\n", box_size);

    // Test θ = 0.8 first (faster)
    println!("═══ Test: θ = 0.8 ═══");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.8);

    let t0 = Instant::now();
    sim.compute_forces()?;
    println!("  Initial forces: {:.1}s", t0.elapsed().as_secs_f64());

    for step in 1..=2 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        println!("  Step {}: {:.1}s", step, t0.elapsed().as_secs_f64());
    }

    println!("\n═══ Test: θ = 0.7 ═══");
    drop(sim);
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.7);

    let t0 = Instant::now();
    sim.compute_forces()?;
    println!("  Initial forces: {:.1}s", t0.elapsed().as_secs_f64());

    for step in 1..=2 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        println!("  Step {}: {:.1}s", step, t0.elapsed().as_secs_f64());
    }

    println!("\nDone.");
    Ok(())
}
