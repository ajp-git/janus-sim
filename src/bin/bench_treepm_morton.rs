//! Benchmark: TreePM with and without Morton ordering
//! Usage: --n <particles> --steps <count>

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut n = 500_000usize;
    let mut steps = 5usize;

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
    let r_cut = box_size / 16.0;

    println!("=== TreePM Morton Benchmark ===");
    println!("N = {}", n);
    println!("steps = {}", steps);
    println!("box = {:.1} Mpc", box_size);
    println!("theta = {}", theta);
    println!("r_cut = {:.2} Mpc", r_cut);
    println!();

    // === BENCHMARK 1: step_treepm_gpu (no Morton) ===
    println!("=== step_treepm_gpu (no Morton) ===");
    let t0 = std::time::Instant::now();
    let mut sim1 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim1.set_theta(theta);
    println!("Init: {:.2}s\n", t0.elapsed().as_secs_f64());

    println!("Warmup (2 steps)...");
    for _ in 0..2 {
        sim1.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();
    }

    println!("\nBenchmark:");
    let mut times_no_morton = Vec::new();
    for i in 0..steps {
        let t0 = std::time::Instant::now();
        sim1.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        times_no_morton.push(ms);
    }
    let avg_no_morton = times_no_morton.iter().sum::<u128>() / times_no_morton.len() as u128;
    println!("  AVG: {} ms\n", avg_no_morton);

    // === BENCHMARK 2: step_treepm_gpu (with Morton) ===
    println!("=== step_treepm_gpu (with Morton) ===");
    let t0 = std::time::Instant::now();
    let mut sim2 = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim2.set_theta(theta);
    println!("Init: {:.2}s\n", t0.elapsed().as_secs_f64());

    println!("Warmup (2 steps)...");
    for _ in 0..2 {
        sim2.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();
    }

    println!("\nBenchmark:");
    let mut times_morton = Vec::new();
    for i in 0..steps {
        let t0 = std::time::Instant::now();
        sim2.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        times_morton.push(ms);
    }
    let avg_morton = times_morton.iter().sum::<u128>() / times_morton.len() as u128;
    println!("  AVG: {} ms\n", avg_morton);

    // === SUMMARY ===
    println!("=== Summary @ {} particles ===", n);
    println!("  step_treepm_gpu (no Morton):   {} ms", avg_no_morton);
    println!("  step_treepm_gpu:        {} ms", avg_morton);

    let speedup = avg_no_morton as f64 / avg_morton as f64;
    if speedup > 1.0 {
        println!("  Speedup: {:.2}x FASTER with Morton", speedup);
    } else {
        println!("  Slowdown: {:.2}x SLOWER with Morton", 1.0/speedup);
    }
    println!();

    // Physics validation
    let seg1 = sim1.segregation().unwrap();
    let seg2 = sim2.segregation().unwrap();
    println!("Physics validation:");
    println!("  Seg (no Morton): {:.4}", seg1);
    println!("  Seg (Morton):    {:.4}", seg2);
    let seg_diff = (seg1 - seg2).abs();
    if seg_diff < 0.01 {
        println!("  ✓ Physics match (diff = {:.6})", seg_diff);
    } else {
        println!("  ✗ Physics MISMATCH (diff = {:.4})", seg_diff);
    }

    // Extrapolations
    println!("\nExtrapolations with Morton:");
    let step_85m = avg_morton as f64 * (85.0 / (n as f64 / 1_000_000.0)).powf(1.3);
    let hours_85m = step_85m * 12000.0 / 3_600_000.0;
    println!("  85M: ~{:.0} ms/step", step_85m);
    println!("  85M × 12000 steps: ~{:.1} hours ({:.1} days)", hours_85m, hours_85m / 24.0);
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
