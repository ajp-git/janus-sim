//! Quick test of warp-coherent kernel with 100K particles

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Quick warp-coherent test: 100K particles");
    
    let n = 100_000;
    let eta = 1.045;
    let n_positive = (n as f64 / (1.0 + eta)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0;

    eprintln!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.7);
    
    eprintln!("Running 1 step with warp-coherent...");
    sim.step_dkd_warpcoherent(0.01, 0.0, 0.0)?;
    
    eprintln!("Running 1 step with Morton+warp-coherent...");
    sim.step_dkd_morton_warpcoherent(0.01, 0.0, 0.0)?;
    
    eprintln!("SUCCESS: Warp-coherent kernels work!");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
