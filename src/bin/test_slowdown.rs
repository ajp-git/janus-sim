//! Test twopass at θ=0.5 for slowdown behavior
//! Measure step time at steps 1, 100, 500, 1000

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

    println!("=== Slowdown Test (Twopass GPU Karras + Morton Fix) ===");
    println!("N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("θ = 0.5 (non-négociable)");
    println!("Measuring at steps 1, 100, 500, 1000");
    println!();

    println!("Creating simulation...");
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);  // θ = 0.5 non-négociable
    println!("Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    let checkpoints = [1, 10, 50, 100, 200, 300, 400, 500, 750, 1000];
    let mut step = 0;
    let mut checkpoint_idx = 0;

    println!("{:>6} | {:>10} | {:>12} | {:>12}", "step", "time (ms)", "KE", "S");
    println!("{}", "-".repeat(50));

    while checkpoint_idx < checkpoints.len() {
        step += 1;
        let t0 = std::time::Instant::now();
        sim.step_dkd(0.01, 0.0, 0.0)?;
        let elapsed = t0.elapsed().as_millis();

        if step == checkpoints[checkpoint_idx] {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;
            println!("{:>6} | {:>10} | {:>12.4e} | {:>12.4e}", step, elapsed, ke, seg);
            checkpoint_idx += 1;
        }
    }

    println!("\n=== ANALYSIS ===");
    println!("If step time stays <10s throughout → FIX VALIDATED");
    println!("If step time grows >10× from step 1 → SLOWDOWN PROBLEM");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature required");
}
