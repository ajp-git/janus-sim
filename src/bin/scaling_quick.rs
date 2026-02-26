// Quick scaling test: 500K, 1M, 2M with θ=0.8
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn test_n(n: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.8);

    // Initial force computation
    sim.compute_forces()?;

    // Time one step
    let t0 = Instant::now();
    sim.step_dkd(0.01, 0.0, 0.0)?;
    Ok(t0.elapsed().as_secs_f64())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Scaling Test (θ=0.8)                            ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    let sizes = [500_000, 1_000_000, 2_000_000];

    println!("{:>10} | {:>10} | {:>10}", "N", "Step (s)", "Ratio");
    println!("{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "");

    let mut prev_time: Option<f64> = None;
    for &n in &sizes {
        let step_time = test_n(n)?;
        let ratio = prev_time.map(|p| step_time / p).unwrap_or(1.0);
        println!("{:>10} | {:>10.2} | {:>10.1}x",
            n, step_time, if prev_time.is_some() { ratio } else { 0.0 }
        );
        prev_time = Some(step_time);
    }

    // Extrapolate to 10M
    if let Some(t2m) = prev_time {
        let t10m_est = t2m * 5.0 * (10_000_000f64.log2() / 2_000_000f64.log2());
        println!("\nExtrapolated 10M (O(N log N)): {:.1}s", t10m_est);
    }

    Ok(())
}
