//! Test (f) — Zel'dovich IC energy transfer rate (aggregate).
//!
//! Measure ⟨v²⟩ before and after one tiny step (H=0 → no friction).
//! ΔT_per_mass = (mean(v²)_after - mean(v²)_before) / 2
//! Energy transfer rate = ΔT_per_mass / dt = ⟨v·acc⟩ + O(dt)
//! Compare to linear-theory expectation: T·H (growing mode rate in EdS).

use janus::nbody_gpu::GpuNBodySimulation;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};

const N_SIDE: usize = 100;
const N_PART: usize = N_SIDE * N_SIDE * N_SIDE;
const L_BOX: f64 = 200.0;
const PSI_RMS: f64 = 0.10;
const SEED: u64 = 4242;
const H0_KMS_MPC: f64 = 70.0;
const MPC_GYR_TO_KMS: f64 = 977.8;

fn mean_v2(v: &[f64], n: usize) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..n {
        s += v[3*i].powi(2) + v[3*i+1].powi(2) + v[3*i+2].powi(2);
    }
    s / n as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Test (f) — Zel'dovich IC energy transfer (aggregate stats) ===");

    let z_init: f64 = std::env::var("Z_INIT_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(49.0);
    let a = 1.0 / (1.0 + z_init);
    let h0_gyr = H0_KMS_MPC / MPC_GYR_TO_KMS;
    let h = h0_gyr * a.powf(-1.5);

    println!("[CONFIG] z={}  a={:.4}  H={:.4} 1/Gyr  N={}  L={} Mpc  ψ_rms={} Mpc",
        z_init, a, h, N_PART, L_BOX, PSI_RMS);
    println!("[CONFIG] vel_factor = a·H = {:.4e}", a * h);

    let mut rng = StdRng::seed_from_u64(SEED);
    let normal = Normal::new(0.0, PSI_RMS).unwrap();
    let cell = L_BOX / N_SIDE as f64;
    let half = L_BOX / 2.0;
    let vel_factor = a * h;

    let mut positions = Vec::with_capacity(N_PART * 3);
    let mut velocities = Vec::with_capacity(N_PART * 3);
    let signs: Vec<i32> = vec![1; N_PART];

    for i in 0..N_SIDE {
        for j in 0..N_SIDE {
            for k in 0..N_SIDE {
                let x_lag = (i as f64 + 0.5) * cell - half;
                let y_lag = (j as f64 + 0.5) * cell - half;
                let z_lag = (k as f64 + 0.5) * cell - half;
                let psi_x = normal.sample(&mut rng);
                let psi_y = normal.sample(&mut rng);
                let psi_z = normal.sample(&mut rng);
                let mut x = x_lag + psi_x;
                let mut y = y_lag + psi_y;
                let mut z = z_lag + psi_z;
                if x >  half { x -= L_BOX; } else if x < -half { x += L_BOX; }
                if y >  half { y -= L_BOX; } else if y < -half { y += L_BOX; }
                if z >  half { z -= L_BOX; } else if z < -half { z += L_BOX; }
                positions.push(x); positions.push(y); positions.push(z);
                velocities.push(psi_x * vel_factor);
                velocities.push(psi_y * vel_factor);
                velocities.push(psi_z * vel_factor);
            }
        }
    }

    let mut sim = GpuNBodySimulation::new_with_state(
        N_PART, 0, L_BOX, positions, velocities, signs.clone(),
    )?;
    let theta_val: f64 = std::env::var("THETA_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.7);
    sim.set_theta(theta_val);
    println!("[SIM] θ = {}", theta_val);
    let eps_val: f64 = std::env::var("EPS_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.05);
    sim.set_softening(eps_val);
    println!("[SIM] ε = {} Mpc", eps_val);
    sim.set_phi(1.0, 1.0);
    sim.c_ratio_sq = 1.0;
    sim.repulsion_scale = 0.0;
    let gravity_off = std::env::var("GRAVITY_OFF").is_ok();
    if gravity_off {
        sim.set_mass_factor(0.0);
        println!("[SIM] GRAVITY_OFF — pure inertial test (acc=0)");
    } else {
        sim.set_mass_factor(1.0 / 0.3);
    }

    let v_before = sim.get_velocities()?;
    let mean_v2_before = mean_v2(&v_before, N_PART);
    let t_before = 0.5 * mean_v2_before;
    println!("[T] before step: ⟨½v²⟩ = {:.6e} Mpc²/Gyr²", t_before);

    // Single step with H=0
    let dt: f64 = std::env::var("DT_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1e-7);
    sim.step_with_expansion_dkd_gpu_cosmo(dt, a, a, 0.0, 0.0)?;
    let v_after = sim.get_velocities()?;
    let mean_v2_after = mean_v2(&v_after, N_PART);
    let t_after = 0.5 * mean_v2_after;
    let dt_t = (t_after - t_before) / dt;

    // For symmetric DKD with H=0:
    //   v_after = v + acc/a²·dt
    //   ⟨v²⟩_after = ⟨v²⟩ + 2·⟨v·acc⟩/a²·dt + ⟨(acc/a²)²⟩·dt²
    // For tiny dt: ΔT = (½)·2·⟨v·acc⟩/a²·dt = ⟨v·acc⟩/a²·dt
    // So dT/dt = ⟨v·acc⟩/a²
    //
    // Linear-theory growing-mode rate: dT/dt = T·H

    let expected_growing = t_before * h;
    let ratio = dt_t / expected_growing;

    println!("[T] after step:  ⟨½v²⟩ = {:.6e} Mpc²/Gyr²", t_after);
    println!("[ΔT/dt] measured     = {:+.4e} Mpc²/Gyr³", dt_t);
    println!("[ΔT/dt] expected T·H = {:+.4e} Mpc²/Gyr³ (linear EdS growing mode)", expected_growing);
    println!("[ratio] measured/expected = {:.4}", ratio);

    // Also report the magnitude estimate: |Δv_typical|
    let delta_t_per_pp = t_after - t_before;
    let delta_v_typ = (2.0 * delta_t_per_pp).abs().sqrt();
    println!();
    println!("[EQUIV] |Δv_typical| in step = {:.4e} Mpc/Gyr  (over dt={:.0e})", delta_v_typ, dt);
    println!("[EQUIV] equivalent typical |acc/a²| = {:.4e} Mpc/Gyr²", delta_v_typ / dt);

    println!();
    println!("=== INTERPRETATION ===");
    if (ratio - 1.0).abs() < 0.5 {
        println!("✅ Energy transfer matches linear theory within 50% — IC normalisation OK");
    } else if ratio > 5.0 {
        println!("❌ Energy transfer >5× linear → ACC OVER-COUPLED");
    } else if ratio > 1.5 {
        println!("⚠ Energy transfer 1.5-5× linear");
    } else if ratio < -0.5 {
        println!("⚠ Energy DECREASING (decaying mode? friction?)");
    } else if ratio.abs() < 0.5 {
        println!("⚠ Energy transfer << expected — IC may be in shear/decaying mode mix");
    } else {
        println!("ratio = {:.3}", ratio);
    }
    Ok(())
}
