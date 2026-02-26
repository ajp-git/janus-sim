// Test force computation scaling with different particle counts

use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

fn test_n(n: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);

    // Initial force computation
    sim.compute_forces()?;

    // Time one step
    let t0 = Instant::now();
    sim.step_dkd(0.01, 0.0, 0.0)?;
    let step_time = t0.elapsed().as_secs_f64();

    Ok(step_time)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Force Computation Scaling Test                  ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    let sizes = [100_000, 200_000, 500_000, 1_000_000, 2_000_000];

    println!("{:>10} | {:>10} | {:>12}", "N", "Step (s)", "per M part/s");
    println!("{:-<10}-+-{:-<10}-+-{:-<12}", "", "", "");

    let mut prev_time: Option<f64> = None;
    for &n in &sizes {
        eprintln!("\n=== Testing N={} ===", n);
        let step_time = test_n(n)?;
        let per_m = step_time / (n as f64 / 1e6);

        let ratio = prev_time.map(|p| step_time / p).unwrap_or(1.0);
        println!("{:>10} | {:>10.2} | {:>12.2}   ({}x)",
            n, step_time, per_m,
            if prev_time.is_some() { format!("{:.1}", ratio) } else { "-".to_string() }
        );
        prev_time = Some(step_time);
    }

    println!("\nExpected scaling: O(N log N) → ratio should be ~2.1x for doubling N");
    println!("If ratio >> 2.1x, tree traversal or memory access is the bottleneck");

    Ok(())
}
