//! Test warp-coherent kernel at increasing particle counts
//! Tests: 1M, 2M, 4M, 8M

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;

#[cfg(feature = "cuda")]
fn test_at_scale(n: usize) -> Result<bool, Box<dyn std::error::Error>> {
    let n_positive = (n as f64 / (1.0 + ETA)) as usize;
    let n_negative = n - n_positive;
    let box_size = 100.0 * (n as f64 / 100_000.0).powf(1.0/3.0);

    eprintln!("\n============================================================");
    eprintln!("Testing warp-coherent at N = {} ({:.1}M)", n, n as f64 / 1e6);
    eprintln!("============================================================");

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);

    // Test 1: warp-coherent without Morton
    eprintln!("\n[1] Warp-coherent (no Morton)...");
    let t0 = Instant::now();
    sim.step_dkd_warpcoherent(DT, 0.0, 0.0)?;
    let t_wc = t0.elapsed().as_millis();
    eprintln!("    OK: {} ms", t_wc);

    // Test 2: warp-coherent with Morton
    eprintln!("[2] Warp-coherent + Morton...");
    let t0 = Instant::now();
    sim.step_dkd_morton_warpcoherent(DT, 0.0, 0.0)?;
    let t_mwc = t0.elapsed().as_millis();
    eprintln!("    OK: {} ms", t_mwc);

    // Comparison with Morton-only
    eprintln!("[3] Morton-only (baseline for comparison)...");
    let t0 = Instant::now();
    sim.step_dkd_morton_reorder(DT, 0.0, 0.0)?;
    let t_morton = t0.elapsed().as_millis();
    eprintln!("    OK: {} ms", t_morton);

    eprintln!("\nSummary at N={}:", n);
    eprintln!("  Warp-coherent:         {} ms", t_wc);
    eprintln!("  Morton+Warp-coherent:  {} ms", t_mwc);
    eprintln!("  Morton-only:           {} ms", t_morton);

    if t_mwc < t_morton {
        eprintln!("  → Morton+WC is {:.2}× FASTER than Morton-only", t_morton as f64 / t_mwc as f64);
    } else {
        eprintln!("  → Morton+WC is {:.2}× SLOWER than Morton-only", t_mwc as f64 / t_morton as f64);
    }

    Ok(true)
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║   WARP-COHERENT SCALING TEST (128-level stack)                 ║");
    eprintln!("╚════════════════════════════════════════════════════════════════╝");

    let scales = [1_000_000, 2_000_000, 4_000_000, 8_000_000];

    for &n in &scales {
        match test_at_scale(n) {
            Ok(_) => eprintln!("\n✓ N={} PASSED", n),
            Err(e) => {
                eprintln!("\n✗ N={} FAILED: {}", n, e);
                eprintln!("Stopping tests.");
                return Err(e);
            }
        }
    }

    eprintln!("\n============================================================");
    eprintln!("ALL TESTS PASSED!");
    eprintln!("============================================================");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
