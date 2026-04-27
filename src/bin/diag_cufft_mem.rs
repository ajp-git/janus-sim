//! Diagnostic: Measure cuFFT memory overhead
//! Tests 256³ vs 128³ grid sizes

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::process::Command;

fn get_vram_mb() -> f64 {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .expect("nvidia-smi failed");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   cuFFT Memory Overhead Diagnostic                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let baseline = get_vram_mb();
    println!("Baseline VRAM: {:.0} MB\n", baseline);

    // Test with N=1M particles
    let n = 1_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("Creating GpuNBodyTwoPass with N=1M...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;

    let after_init = get_vram_mb();
    println!("After particle init: {:.0} MB (+{:.0} MB)", after_init, after_init - baseline);

    // Now run one TreePM step to trigger cuFFT initialization
    println!("\nRunning TreePM step (triggers cuFFT plan creation)...");
    sim.set_theta(0.7);
    let r_cut = box_size / 16.0;
    sim.step_treepm_gpu(0.01, r_cut, 0.0, 0.0)?;

    let after_cufft = get_vram_mb();
    println!("After cuFFT step: {:.0} MB (+{:.0} MB from init)", after_cufft, after_cufft - after_init);

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║   MEMORY BREAKDOWN                                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let particle_mem = after_init - baseline;
    let cufft_mem = after_cufft - after_init;
    let total = after_cufft - baseline;

    println!("Particles (1M):     {:6.0} MB ({:.1} bytes/particle)", particle_mem, particle_mem * 1e6 / n as f64);
    println!("cuFFT overhead:     {:6.0} MB (plans + temp buffers)", cufft_mem);
    println!("Total:              {:6.0} MB\n", total);

    // Extrapolate for different N
    let bytes_per_particle = particle_mem * 1e6 / n as f64;
    println!("Extrapolation (assuming cuFFT overhead = {:.0} MB fixed):", cufft_mem);
    for n_test in [10, 20, 30, 40, 50, 60].iter() {
        let n_particles = n_test * 1_000_000;
        let estimated = baseline + (n_particles as f64 * bytes_per_particle / 1e6) + cufft_mem;
        let fits = if estimated < 12000.0 { "✅" } else { "❌" };
        println!("  N={:2}M: {:.0} MB {}", n_test, estimated, fits);
    }

    // Calculate N_max
    let available = 12000.0 - baseline - cufft_mem;
    let n_max = (available * 1e6 / bytes_per_particle) as usize;
    println!("\nN_max (12 GB RTX 3060): {} particles ({:.1}M)", n_max, n_max as f64 / 1e6);

    println!("\n--- Current PM grid analysis ---");
    println!("Current: 256³ complex f64 = 256³ × 16 bytes = {:.0} MB (theoretical)",
             256.0_f64.powi(3) * 16.0 / 1e6);
    println!("128³ would be: 128³ × 16 bytes = {:.0} MB (theoretical)",
             128.0_f64.powi(3) * 16.0 / 1e6);
    println!("Savings if switching to 128³: ~{:.0} MB (estimated)",
             (256.0_f64.powi(3) - 128.0_f64.powi(3)) * 16.0 / 1e6);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires --features cuda,cufft");
}
