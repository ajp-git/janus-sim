//! Quick test: 40M particles, 3 steps, Morton+WarpCoherent
//! Measure step time to decide overnight run strategy

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

const N: usize = 40_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const STEPS: usize = 3;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║   40M QUICK TEST — Morton + WarpCoherent, {} steps             ║", STEPS);
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let n_positive = (N as f64 / (1.0 + ETA)) as usize;
    let n_negative = N - n_positive;
    let box_size = 100.0 * (N as f64 / 100_000.0).powf(1.0/3.0);

    eprintln!("Parameters:");
    eprintln!("  N = {} ({:.0}M)", N, N as f64 / 1e6);
    eprintln!("  N+ = {}, N- = {}", n_positive, n_negative);
    eprintln!("  θ = {}", THETA);
    eprintln!("  box_size = {:.1}", box_size);
    eprintln!();

    eprintln!("Creating simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    eprintln!("  Init time: {:.1}s", t0.elapsed().as_secs_f64());
    eprintln!();

    eprintln!("Running {} steps with Morton+WarpCoherent...", STEPS);
    let mut step_times = Vec::new();

    for i in 0..STEPS {
        let t0 = Instant::now();
        sim.step_dkd_morton_warpcoherent(DT, 0.0, 0.0)?;
        let elapsed = t0.elapsed().as_secs_f64();
        step_times.push(elapsed);
        eprintln!("  Step {}: {:.2}s", i + 1, elapsed);
    }

    let avg_time = step_times.iter().sum::<f64>() / step_times.len() as f64;
    let min_time = step_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = step_times.iter().cloned().fold(0.0, f64::max);

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("                         RESULTS                                ");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  Step times: {:.2}s / {:.2}s / {:.2}s", step_times[0], step_times[1], step_times[2]);
    eprintln!("  Average:    {:.2}s/step", avg_time);
    eprintln!("  Min/Max:    {:.2}s / {:.2}s", min_time, max_time);
    eprintln!();

    let hours_12k = (avg_time * 12000.0) / 3600.0;
    eprintln!("  Estimated 12000 steps: {:.1} hours ({:.1} days)", hours_12k, hours_12k / 24.0);
    eprintln!();

    if avg_time < 40.0 {
        eprintln!("  ✓ DECISION: Run 80M overnight ({:.1}h)", hours_12k);
    } else {
        eprintln!("  ✗ DECISION: Run 8M overnight, 80M tomorrow with PM+PP");
    }
    eprintln!();

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
