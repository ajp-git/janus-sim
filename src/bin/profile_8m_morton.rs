//! Profile 8M particles with Morton-reorder optimization
//! Reduces warp divergence by sorting particles spatially
//! Target: <2s per force kernel

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const STEPS: usize = 3;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║   MORTON REORDER: 8M particles, θ={}, {} steps               ║", THETA, STEPS);
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    eprintln!("Parameters:");
    eprintln!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    eprintln!("  θ = {} (Barnes-Hut)", THETA);
    eprintln!("  Optimization: Morton space-filling curve reorder");
    eprintln!("  → Spatially nearby particles → consecutive threads → coherent tree walk");
    eprintln!();

    eprintln!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    eprintln!();

    eprintln!("Running {} steps with Morton reorder...", STEPS);
    eprintln!("Target: force kernels <2s each");
    eprintln!();

    for _ in 0..STEPS {
        sim.step_dkd_morton_reorder(DT, 0.0, 0.0)?;
    }

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("ANALYSIS:");
    eprintln!("  Compare force+ and force- times to baseline (without reorder).");
    eprintln!("  Baseline at θ=0.7: ~19s per force kernel");
    eprintln!("  Target with Morton: <2s per force kernel (10× improvement)");
    eprintln!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
