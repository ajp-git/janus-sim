//! Quick VRAM measurement for TreePM
//! Usage: cargo run --release --features cuda,cufft --bin test_vram_limit -- N_MILLIONS

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::env;
use std::process::Command;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let n_millions: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let n = n_millions * 1_000_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    println!("Testing VRAM with N = {}M particles", n_millions);
    println!("Box size: {:.1} Mpc", box_size);

    // Create simulation
    println!("Creating GpuNBodyTwoPass...");
    let sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;

    // Measure VRAM
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()?;
    let vram_mb: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0);

    println!("VRAM used: {:.0} MB ({:.2} GB)", vram_mb, vram_mb / 1024.0);
    println!("Bytes per particle: {:.1}", (vram_mb - 108.0) * 1e6 / n as f64);

    // Try one TreePM step
    println!("Running 1 TreePM step...");
    let mut sim = sim;
    sim.set_theta(0.7);
    let r_cut = box_size / 16.0;
    sim.step_treepm_gpu(0.01, r_cut, 0.0, 0.0)?;

    // Measure VRAM after step
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()?;
    let vram_after: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0);

    println!("VRAM after step: {:.0} MB ({:.2} GB)", vram_after, vram_after / 1024.0);
    println!("\n✅ N = {}M fits in VRAM", n_millions);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires --features cuda,cufft");
}
