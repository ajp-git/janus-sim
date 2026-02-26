// Test twopass performance with different theta values
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Twopass θ Performance Test — 1M particles       ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    for theta in [0.5, 1.0, 2.0, 4.0] {
        println!("═══ θ = {} ═══", theta);
        let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
        sim.set_theta(theta);

        // Initial force (builds tree)
        sim.compute_forces()?;

        // Time one step
        let t0 = Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        let elapsed = t0.elapsed().as_millis();
        println!("  Step time: {} ms\n", elapsed);
    }

    Ok(())
}
