//! Benchmark: Compare Morton ordering speedup
//! Usage: --n <particles> --steps <count>
//!
//! Compares step_dkd (no Morton) vs step_dkd_morton_reorder (with Morton)
//! Morton ordering should improve cache coherence in tree traversal

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut n = 500_000usize;
    let mut steps = 10usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { n = args[i+1].parse().unwrap(); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
    let dt = 0.01;
    let theta = 0.7;

    println!("=== Morton Ordering Benchmark ===");
    println!("N = {}", n);
    println!("steps = {}", steps);
    println!("box = {:.1} Mpc", box_size);
    println!("theta = {}", theta);
    println!();

    // === BENCHMARK 1: step_dkd (no Morton) ===
    println!("=== Initializing for step_dkd (no Morton) ===");
    let t0 = std::time::Instant::now();
    let mut sim1 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim1.set_theta(theta);
    println!("Init: {:.2}s\n", t0.elapsed().as_secs_f64());

    // Warmup
    println!("Warmup (3 steps)...");
    for _ in 0..3 {
        sim1.step_dkd(dt, 0.0, 0.0).unwrap();
    }

    println!("\n--- step_dkd (no Morton) ---");
    let mut times_no_morton = Vec::new();
    for i in 0..steps {
        let t0 = std::time::Instant::now();
        sim1.step_dkd(dt, 0.0, 0.0).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        times_no_morton.push(ms);
    }
    let avg_no_morton = times_no_morton.iter().sum::<u128>() / times_no_morton.len() as u128;
    println!("  AVG: {} ms\n", avg_no_morton);

    // === BENCHMARK 2: step_dkd_morton_reorder (with Morton) ===
    println!("=== Initializing for step_dkd_morton_reorder ===");
    let t0 = std::time::Instant::now();
    let mut sim2 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim2.set_theta(theta);
    println!("Init: {:.2}s\n", t0.elapsed().as_secs_f64());

    // Warmup
    println!("Warmup (3 steps)...");
    for _ in 0..3 {
        sim2.step_dkd_morton_reorder(dt, 0.0, 0.0).unwrap();
    }

    println!("\n--- step_dkd_morton_reorder (with Morton) ---");
    let mut times_morton = Vec::new();
    for i in 0..steps {
        let t0 = std::time::Instant::now();
        sim2.step_dkd_morton_reorder(dt, 0.0, 0.0).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        times_morton.push(ms);
    }
    let avg_morton = times_morton.iter().sum::<u128>() / times_morton.len() as u128;
    println!("  AVG: {} ms\n", avg_morton);

    // === SUMMARY ===
    println!("=== Summary @ {} particles ===", n);
    println!("  step_dkd (no Morton):          {} ms", avg_no_morton);
    println!("  step_dkd_morton_reorder:       {} ms", avg_morton);

    let speedup = avg_no_morton as f64 / avg_morton as f64;
    if speedup > 1.0 {
        println!("  Speedup: {:.2}x FASTER with Morton", speedup);
    } else {
        println!("  Slowdown: {:.2}x SLOWER with Morton", 1.0/speedup);
    }
    println!();

    // Validate physics: check segregation
    let seg1 = sim1.segregation().unwrap();
    let seg2 = sim2.segregation().unwrap();
    println!("Physics validation:");
    println!("  Seg (no Morton): {:.4}", seg1);
    println!("  Seg (Morton):    {:.4}", seg2);
    let seg_diff = (seg1 - seg2).abs();
    if seg_diff < 0.01 {
        println!("  ✓ Physics match (diff = {:.6})", seg_diff);
    } else {
        println!("  ✗ Physics MISMATCH (diff = {:.4}) - bug in Morton reorder!", seg_diff);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
