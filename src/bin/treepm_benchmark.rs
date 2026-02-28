//! TreePM Benchmark: Compare step_dkd vs step_treepm_gpu
//! Usage: --n <particles> --steps <count> --mode <bh_only|treepm>

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut n = 100_000usize;
    let mut steps = 5usize;
    let mut mode = "bh_only".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { n = args[i+1].parse().unwrap(); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--mode" => { mode = args[i+1].clone(); i += 2; }
            _ => { i += 1; }
        }
    }

    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
    let dt = 0.01;
    let theta = 0.7;
    let r_cut = box_size / 16.0;

    println!("=== TreePM Benchmark ===");
    println!("N = {}", n);
    println!("steps = {}", steps);
    println!("mode = {}", mode);
    println!("box = {:.1} Mpc", box_size);
    println!("theta = {}", theta);
    println!("r_cut = {:.2} Mpc", r_cut);
    println!();

    println!("Initializing...");
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(theta);
    println!("Init: {:.2}s\n", t0.elapsed().as_secs_f64());

    // Warmup
    println!("Warmup (2 steps)...");
    for _ in 0..2 {
        match mode.as_str() {
            "bh_only" => sim.step_dkd(dt, 0.0, 0.0).unwrap(),
            "treepm" => sim.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap(),
            _ => panic!("Unknown mode: {}", mode),
        }
    }

    // Benchmark
    println!("\nBenchmark ({} steps):", steps);
    let mut times = Vec::new();
    for i in 0..steps {
        let t0 = std::time::Instant::now();
        match mode.as_str() {
            "bh_only" => sim.step_dkd(dt, 0.0, 0.0).unwrap(),
            "treepm" => sim.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap(),
            _ => {}
        }
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        times.push(ms);
    }

    let avg = times.iter().sum::<u128>() / times.len() as u128;
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();

    println!("\n=== Results ({}) ===", mode);
    println!("  AVG: {} ms", avg);
    println!("  MIN: {} ms", min);
    println!("  MAX: {} ms", max);
    println!();

    // Extrapolations
    println!("Extrapolations:");
    println!("  85M particles: ~{:.0} ms/step", avg as f64 * (85.0 / (n as f64 / 1_000_000.0)).powf(1.3));
    println!("  85M x 12000 steps: ~{:.1} hours",
             avg as f64 * (85.0 / (n as f64 / 1_000_000.0)).powf(1.3) * 12000.0 / 3_600_000.0);
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
