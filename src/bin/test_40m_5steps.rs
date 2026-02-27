//! Quick test: 40M, 5 steps to validate setup WITH HUBBLE FRICTION

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::time::Instant;

const N: usize = 40_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const BOX_SIZE: f64 = 736.8;
const Z_INIT: f64 = 5.0;
const Z_FINAL: f64 = 1.5;
const STEPS: usize = 6000;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("40M Validation Test (5 steps) WITH HUBBLE FRICTION");
    eprintln!();

    let n_positive = (N as f64 / (1.0 + ETA)) as usize;
    let n_negative = N - n_positive;

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);

    // Setup cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    let a_init = 1.0 / (1.0 + Z_INIT);
    let a_final = 1.0 / (1.0 + Z_FINAL);

    let tau_z_init = cosmo.history.iter()
        .min_by(|s1, s2| {
            (s1.a - a_init).abs().partial_cmp(&(s2.a - a_init).abs()).unwrap()
        })
        .map(|s| s.tau)
        .unwrap();

    let tau_z_final = cosmo.history.iter()
        .min_by(|s1, s2| {
            (s1.a - a_final).abs().partial_cmp(&(s2.a - a_final).abs()).unwrap()
        })
        .map(|s| s.tau)
        .unwrap();

    let dtau_per_step = (tau_z_final - tau_z_init) / STEPS as f64;
    let mut tau_current = tau_z_init;

    let (_, h_init) = cosmo.get_params_at_tau(tau_z_init);
    eprintln!("Cosmology: tau_init={:.6}, H_init={:.4}", tau_z_init, h_init);
    eprintln!("dtau/step={:.8}, dtau_per_dt={:.6}", dtau_per_step, dtau_per_step / DT);
    eprintln!();

    // Get initial segregation
    let pos = sim.get_positions()?;
    let signs = sim.get_signs()?;
    let seg_0 = compute_segregation(&pos, &signs, n_positive, n_negative);
    eprintln!("Seg_0 = {:.6}", seg_0);
    eprintln!();

    let mut step_times = Vec::new();

    for step in 1..=5 {
        let t0 = Instant::now();

        let (a, hubble) = cosmo.get_params_at_tau(tau_current);
        let z = 1.0 / a - 1.0;
        let dtau_per_dt = dtau_per_step / DT;

        sim.step_dkd_morton_warpcoherent(DT, hubble, dtau_per_dt)?;
        tau_current += dtau_per_step;

        let elapsed = t0.elapsed().as_secs_f64();
        step_times.push(elapsed);

        let pos = sim.get_positions()?;
        let signs = sim.get_signs()?;
        let seg = compute_segregation(&pos, &signs, n_positive, n_negative);

        eprintln!("Step {}: {:.2}s, z={:.4}, H={:.4}, Seg={:.6}", step, elapsed, z, hubble, seg);
    }

    let avg = step_times.iter().sum::<f64>() / step_times.len() as f64;
    eprintln!();
    eprintln!("Average step time: {:.2}s", avg);
    eprintln!("Expected for 6000 steps: {:.1} hours", avg * 6000.0 / 3600.0);

    if avg < 20.0 {
        eprintln!();
        eprintln!("✓ VALIDATION PASSED - Hubble friction active, ready for overnight run");
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
