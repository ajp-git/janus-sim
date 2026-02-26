//! Force Kernel Benchmark Framework
//!
//! Uses production GpuNBodyTwoPass to measure optimizations
//! Runs: 3 warmup + 7 measured, reports median ± stddev

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const WARMUP: usize = 3;
const MEASURED: usize = 7;

#[derive(Clone, Copy)]
enum StepVariant {
    Baseline,
    Morton,
    MortonShmem,
    WarpCoherent,
    MortonWarpCoherent,
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║   FORCE KERNEL BENCHMARK — 8M particles, θ={}               ║", THETA);
    eprintln!("║   {} warmup + {} measured runs per test                        ║", WARMUP, MEASURED);
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    eprintln!("Creating simulation ({:.1}M particles)...", N_PARTICLES as f64 / 1e6);
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    eprintln!();

    // =========================================================================
    // BASELINE: step_dkd (no Morton reorder)
    // =========================================================================
    eprintln!(">>> BASELINE: step_dkd (no Morton reorder)");
    let baseline_times = benchmark_step(&mut sim, WARMUP, MEASURED, StepVariant::Baseline)?;
    let baseline_median = median(&baseline_times);
    let baseline_stddev = stddev(&baseline_times);
    eprintln!("    Times: {:?}", baseline_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>());
    eprintln!("    Median: {:.0} ms ± {:.0} ms", baseline_median, baseline_stddev);
    eprintln!();

    // =========================================================================
    // OPT-1: Morton reorder
    // =========================================================================
    eprintln!(">>> OPT-1: step_dkd_morton_reorder");
    let morton_times = benchmark_step(&mut sim, WARMUP, MEASURED, StepVariant::Morton)?;
    let morton_median = median(&morton_times);
    let morton_stddev = stddev(&morton_times);
    let morton_speedup = baseline_median / morton_median;
    eprintln!("    Times: {:?}", morton_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>());
    eprintln!("    Median: {:.0} ms ± {:.0} ms", morton_median, morton_stddev);
    eprintln!("    Speedup vs baseline: {:.2}×", morton_speedup);
    eprintln!();

    // =========================================================================
    // OPT-4a: Warp-coherent traversal (without Morton)
    // =========================================================================
    eprintln!(">>> OPT-4a: step_dkd_warpcoherent (warp-coherent, no Morton)");
    let wc_times = benchmark_step(&mut sim, WARMUP, MEASURED, StepVariant::WarpCoherent)?;
    let wc_median = median(&wc_times);
    let wc_stddev = stddev(&wc_times);
    let wc_speedup = baseline_median / wc_median;
    eprintln!("    Times: {:?}", wc_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>());
    eprintln!("    Median: {:.0} ms ± {:.0} ms", wc_median, wc_stddev);
    eprintln!("    Speedup vs baseline: {:.2}×", wc_speedup);
    eprintln!();

    // =========================================================================
    // OPT-4b: Morton + Warp-coherent traversal
    // =========================================================================
    eprintln!(">>> OPT-4b: step_dkd_morton_warpcoherent (Morton + warp-coherent)");
    let mwc_times = benchmark_step(&mut sim, WARMUP, MEASURED, StepVariant::MortonWarpCoherent)?;
    let mwc_median = median(&mwc_times);
    let mwc_stddev = stddev(&mwc_times);
    let mwc_speedup = baseline_median / mwc_median;
    let mwc_vs_morton = morton_median / mwc_median;
    let mwc_vs_wc = wc_median / mwc_median;
    eprintln!("    Times: {:?}", mwc_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>());
    eprintln!("    Median: {:.0} ms ± {:.0} ms", mwc_median, mwc_stddev);
    eprintln!("    Speedup vs baseline: {:.2}×", mwc_speedup);
    eprintln!("    Speedup vs OPT-1:    {:.2}×", mwc_vs_morton);
    eprintln!("    Speedup vs OPT-4a:   {:.2}×", mwc_vs_wc);
    eprintln!();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    eprintln!("═══════════════════════════════════════════════════════════════════");
    eprintln!("                           SUMMARY                                  ");
    eprintln!("═══════════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  {:35} {:>10} {:>10}", "Optimization", "Time (ms)", "Speedup");
    eprintln!("  {:35} {:>10} {:>10}", "─".repeat(35), "─".repeat(10), "─".repeat(10));
    eprintln!("  {:35} {:>10.0} {:>10}", "Baseline (no optim)", baseline_median, "1.00×");
    eprintln!("  {:35} {:>10.0} {:>10.2}×", "+ Morton reorder [OPT-1]", morton_median, morton_speedup);
    eprintln!("  {:35} {:>10.0} {:>10.2}×", "+ Warp-coherent [OPT-4a]", wc_median, wc_speedup);
    eprintln!("  {:35} {:>10.0} {:>10.2}×", "+ Morton+WarpCoh [OPT-4b]", mwc_median, mwc_speedup);
    eprintln!();

    // Find best time among all optimizations
    let best_median = mwc_median.min(wc_median).min(morton_median);
    let best_name = if best_median == mwc_median { "Morton+WarpCoh" }
                    else if best_median == wc_median { "WarpCoherent" }
                    else { "Morton" };

    // Estimate full run time
    let hours_baseline = (baseline_median * 12000.0) / 3600000.0;
    let hours_best = (best_median * 12000.0) / 3600000.0;
    eprintln!("  Full run (12000 steps) estimate:");
    eprintln!("    Baseline:     {:.1} hours ({:.1} days)", hours_baseline, hours_baseline / 24.0);
    eprintln!("    Best ({}): {:.1} hours ({:.1} days)", best_name, hours_best, hours_best / 24.0);
    eprintln!();

    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_step(sim: &mut GpuNBodyTwoPass, warmup: usize, measured: usize, variant: StepVariant) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Warmup
    for i in 0..warmup {
        eprint!("    Warmup {}/{}...\r", i + 1, warmup);
        match variant {
            StepVariant::Baseline => sim.step_dkd(DT, 0.0, 0.0)?,
            StepVariant::Morton => sim.step_dkd_morton_reorder(DT, 0.0, 0.0)?,
            StepVariant::MortonShmem => sim.step_dkd_morton_shmem(DT, 0.0, 0.0)?,
            StepVariant::WarpCoherent => sim.step_dkd_warpcoherent(DT, 0.0, 0.0)?,
            StepVariant::MortonWarpCoherent => sim.step_dkd_morton_warpcoherent(DT, 0.0, 0.0)?,
        }
    }

    // Measured runs
    let mut times = Vec::with_capacity(measured);
    for i in 0..measured {
        eprint!("    Measure {}/{}...   \r", i + 1, measured);
        let t0 = Instant::now();
        match variant {
            StepVariant::Baseline => sim.step_dkd(DT, 0.0, 0.0)?,
            StepVariant::Morton => sim.step_dkd_morton_reorder(DT, 0.0, 0.0)?,
            StepVariant::MortonShmem => sim.step_dkd_morton_shmem(DT, 0.0, 0.0)?,
            StepVariant::WarpCoherent => sim.step_dkd_warpcoherent(DT, 0.0, 0.0)?,
            StepVariant::MortonWarpCoherent => sim.step_dkd_morton_warpcoherent(DT, 0.0, 0.0)?,
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("                         ");

    Ok(times)
}

fn median(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn stddev(times: &[f64]) -> f64 {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    var.sqrt()
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
