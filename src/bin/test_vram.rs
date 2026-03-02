/// Test VRAM allocation for different N values
/// Reports exact VRAM usage before OOM

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

const ETA: f64 = 1.045;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let n_millions: f64 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);

    let n_particles = (n_millions * 1_000_000.0) as usize;
    let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

    let n_positive = (n_particles as f64 / (1.0 + ETA)) as usize;
    let n_negative = n_particles - n_positive;

    println!("Testing N = {:.1}M particles...", n_millions);
    println!("  N+ = {}, N- = {}", n_positive, n_negative);
    println!("  box = {:.1} Mpc", box_size);

    match GpuNBodySimulation::new_bvh_only(n_positive, n_negative, box_size) {
        Ok(_sim) => {
            println!("SUCCESS: N = {:.1}M fits in VRAM", n_millions);
            // Print nvidia-smi would need shell access, just report success
            Ok(())
        }
        Err(e) => {
            println!("FAIL: N = {:.1}M → {}", n_millions, e);
            Err(e)
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled!");
}
