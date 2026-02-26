// Test force computation with θ=0.8 vs θ=0.5
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   θ Performance Test — 1M particles               ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("  N+ = {}, N- = {}", n_positive, n_negative);
    println!("  box = {:.2}\n", box_size);

    // Test θ = 0.8
    println!("═══ Test 1: θ = 0.8 ═══");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.8);

    let t0 = Instant::now();
    sim.compute_forces()?;
    println!("  Initial forces: {:.0}ms", t0.elapsed().as_millis());

    for step in 1..=3 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        println!("  Step {}: {:.0}ms", step, t0.elapsed().as_millis());
    }

    drop(sim);

    // Test θ = 0.5
    println!("\n═══ Test 2: θ = 0.5 ═══");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);

    let t0 = Instant::now();
    sim.compute_forces()?;
    println!("  Initial forces: {:.0}ms", t0.elapsed().as_millis());

    for step in 1..=3 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        println!("  Step {}: {:.0}ms", step, t0.elapsed().as_millis());
    }

    println!("\n═══ Test 3: θ = 1.0 (very aggressive) ═══");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(1.0);

    let t0 = Instant::now();
    sim.compute_forces()?;
    println!("  Initial forces: {:.0}ms", t0.elapsed().as_millis());

    for step in 1..=3 {
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        println!("  Step {}: {:.0}ms", step, t0.elapsed().as_millis());
    }

    println!("\nDone.");
    Ok(())
}
