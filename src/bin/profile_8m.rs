//! Profile 8M particles at θ=0.7 — 3 steps with detailed timing breakdown

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
    eprintln!("║   PROFILING: 8M particles, θ=0.7, {} steps                     ║", STEPS);
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    eprintln!("Parameters:");
    eprintln!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    eprintln!("  N+ = {}, N- = {}", n_positive, n_negative);
    eprintln!("  θ = {}", THETA);
    eprintln!("  box = {:.2}", box_size);
    eprintln!();

    eprintln!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    eprintln!();

    eprintln!("Running {} steps with detailed profiling...", STEPS);
    eprintln!("(Timing breakdown printed to stderr after each step)");
    eprintln!();

    for _ in 0..STEPS {
        sim.step_dkd(DT, 0.0, 0.0)?;
    }

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("ANALYSIS:");
    eprintln!("  Look for the largest time consumers above.");
    eprintln!("  force+ and force- are typically the bottlenecks at low θ.");
    eprintln!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
