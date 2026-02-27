//! Quick test: 40M, 5 steps to validate setup

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
use std::time::Instant;

const N: usize = 40_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const BOX_SIZE: f64 = 736.8;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("40M Validation Test (5 steps)");
    eprintln!();

    let n_positive = (N as f64 / (1.0 + ETA)) as usize;
    let n_negative = N - n_positive;

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);

    // Get initial segregation
    let pos = sim.get_positions()?;
    let signs = sim.get_signs()?;
    let seg_0 = compute_segregation(&pos, &signs, n_positive, n_negative);
    eprintln!("Seg_0 = {:.6}", seg_0);
    eprintln!();

    let mut step_times = Vec::new();

    for step in 1..=5 {
        let t0 = Instant::now();
        sim.step_dkd_morton_warpcoherent(DT, 0.0, 0.0)?;
        let elapsed = t0.elapsed().as_secs_f64();
        step_times.push(elapsed);

        let pos = sim.get_positions()?;
        let signs = sim.get_signs()?;
        let seg = compute_segregation(&pos, &signs, n_positive, n_negative);

        eprintln!("Step {}: {:.2}s, Seg={:.6}", step, elapsed, seg);
    }

    let avg = step_times.iter().sum::<f64>() / step_times.len() as f64;
    eprintln!();
    eprintln!("Average step time: {:.2}s", avg);
    eprintln!("Expected for 6000 steps: {:.1} hours", avg * 6000.0 / 3600.0);

    if seg_0 < 0.01 && avg < 20.0 {
        eprintln!();
        eprintln!("✓ VALIDATION PASSED - Ready for overnight run");
    } else {
        eprintln!();
        eprintln!("✗ Check values before launching");
    }

    Ok(())
}

fn compute_segregation(pos: &[f32], signs: &[i8], n_pos: usize, n_neg: usize) -> f64 {
    let n = pos.len() / 3;
    let (mut com_pos, mut com_neg) = ([0.0f64; 3], [0.0f64; 3]);

    for i in 0..n {
        let x = pos[i * 3] as f64;
        let y = pos[i * 3 + 1] as f64;
        let z = pos[i * 3 + 2] as f64;

        if signs[i] > 0 {
            com_pos[0] += x; com_pos[1] += y; com_pos[2] += z;
        } else {
            com_neg[0] += x; com_neg[1] += y; com_neg[2] += z;
        }
    }

    com_pos[0] /= n_pos as f64; com_pos[1] /= n_pos as f64; com_pos[2] /= n_pos as f64;
    com_neg[0] /= n_neg as f64; com_neg[1] /= n_neg as f64; com_neg[2] /= n_neg as f64;

    let dx = com_pos[0] - com_neg[0];
    let dy = com_pos[1] - com_neg[1];
    let dz = com_pos[2] - com_neg[2];

    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
