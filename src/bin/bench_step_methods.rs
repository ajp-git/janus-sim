//! Benchmark: step_dkd vs step_treepm_gpu @ 2M
//! Compare timing to identify TreePM BH slowdown

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Step Method Benchmark @ 2M ===\n");

    let n = 2_000_000;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);
    let dt = 0.01;
    let theta = 0.7;
    let r_cut = box_size / 16.0;

    println!("N = {}, box = {:.1}, theta = {}, r_cut = {:.1}\n", n, box_size, theta, r_cut);

    // Create simulation
    let mut sim = GpuNBodyTwoPass::new(n/2, n/2, box_size).expect("GPU init failed");
    sim.set_theta(theta);

    // Warmup
    println!("Warmup (3 steps each)...");
    for _ in 0..3 {
        sim.step_dkd(dt, 0.0, 0.0).unwrap();
    }

    // Benchmark step_dkd
    println!("\n--- step_dkd (pure BH) ---");
    let mut dkd_times = Vec::new();
    for i in 0..5 {
        let t0 = std::time::Instant::now();
        sim.step_dkd(dt, 0.0, 0.0).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        dkd_times.push(ms);
    }
    let dkd_avg = dkd_times.iter().sum::<u128>() / dkd_times.len() as u128;
    println!("  AVG: {} ms\n", dkd_avg);

    // Benchmark step_treepm_gpu
    println!("--- step_treepm_gpu (BH + PM) ---");
    let mut treepm_times = Vec::new();
    for i in 0..5 {
        let t0 = std::time::Instant::now();
        sim.step_treepm_gpu(dt, r_cut, 0.0, 0.0).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("  step {}: {} ms", i+1, ms);
        treepm_times.push(ms);
    }
    let treepm_avg = treepm_times.iter().sum::<u128>() / treepm_times.len() as u128;
    println!("  AVG: {} ms\n", treepm_avg);

    println!("=== Summary ===");
    println!("  step_dkd:       {} ms", dkd_avg);
    println!("  step_treepm_gpu: {} ms", treepm_avg);
    println!("  Ratio: {:.2}x", treepm_avg as f64 / dkd_avg as f64);
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
