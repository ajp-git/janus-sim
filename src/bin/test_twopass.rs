/// Test two-pass GPU simulation for maximum particle count
/// cargo run --release --features cuda --bin test_twopass -- 60000000

#[cfg(feature = "cuda")]
fn main() {
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
    use std::time::Instant;

    let n: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000_000);

    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Two-Pass GPU Test (separate + and - trees)                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / n as f64);
    println!("N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / n as f64);
    println!("box = {:.2}", box_size);
    println!();

    // Create simulation
    println!("Creating simulation...");
    let t0 = Instant::now();
    let mut sim = match GpuNBodyTwoPass::new(n_positive, n_negative, box_size) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create simulation: {}", e);
            return;
        }
    };
    println!("  Created in {:.2}s", t0.elapsed().as_secs_f64());

    sim.set_theta(1.5);

    // Initial metrics
    let ke0 = sim.kinetic_energy().unwrap_or(0.0);
    let s0 = sim.segregation().unwrap_or(0.0);
    println!();
    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", s0);
    println!();

    // Run 10 steps to measure performance
    println!("Running 10 DKD steps...");
    let dt = 0.005;
    let hubble = 0.0;  // No expansion for this test
    let dtau_per_dt = 0.0;

    let t1 = Instant::now();
    for step in 1..=10 {
        if let Err(e) = sim.step_dkd(dt, hubble, dtau_per_dt) {
            eprintln!("Step {} failed: {}", step, e);
            return;
        }
    }
    let elapsed = t1.elapsed().as_secs_f64();
    println!("  Total: {:.2}s ({:.1} ms/step)", elapsed, elapsed * 100.0);

    // Final metrics
    let ke = sim.kinetic_energy().unwrap_or(0.0);
    let seg = sim.segregation().unwrap_or(0.0);
    println!();
    println!("After 10 steps:");
    println!("  KE/KE₀ = {:.4}", ke / ke0);
    println!("  S = {:.6}", seg);
    println!();
    println!("✓ Two-pass test completed successfully!");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA not enabled");
}
