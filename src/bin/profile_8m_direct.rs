//! Profile 8M particles with DIRECT N² force computation
//! O(N²) exact - no tree, no approximation
//! Target: <8s/step

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const DT: f64 = 0.01;
const STEPS: usize = 3;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║   N² DIRECT: 8M particles, O(N²) exact, {} steps              ║", STEPS);
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    eprintln!("Parameters:");
    eprintln!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    eprintln!("  N² = {:.2e} interactions", (N_PARTICLES as f64).powi(2));
    eprintln!("  Algorithm: Direct N² with shared memory tiling");
    eprintln!("  Tiles: {} × 256 = {} tile-pairs", N_PARTICLES / 256, (N_PARTICLES / 256) * (N_PARTICLES / 256));
    eprintln!();

    eprintln!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    eprintln!();

    eprintln!("Running {} steps with N² direct force...", STEPS);
    eprintln!("Target: <8s per step");
    eprintln!();

    for _ in 0..STEPS {
        sim.step_dkd_direct(DT, 0.0, 0.0)?;
    }

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("THEORETICAL ANALYSIS:");
    eprintln!("  N² = 64 trillion interactions");
    eprintln!("  ~20 FLOPs/interaction → ~1.3 PFLOPS total");
    eprintln!("  RTX 3060: ~12.7 TFLOPS → minimum ~100s");
    eprintln!("  With shared mem tiling: ~50-80% efficiency");
    eprintln!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
