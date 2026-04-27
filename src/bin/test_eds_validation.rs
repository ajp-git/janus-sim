//! EdS validation of the cosmological coupling fix (peculiar convention).
//!
//! Setup:
//!   - 1M m+ particles, all signs +1
//!   - Box L = 200 Mpc comoving, fixed
//!   - EdS-like cosmology: Ω_m = 1, Ω_Λ = 0  →  a(t) ∝ t^(2/3), H(a) = H₀·a^(-3/2)
//!   - z_init = 49 (a=0.02), z_final = 0
//!   - 50 snapshots log-spaced in a
//!   - Cross-coupling OFF (φ=1, c̄²=1, repulsion_scale=0)
//!
//! Validation metrics (3 independent, robust):
//!   1. Mean Lagrangian → Eulerian displacement   d_meas(a) = <|x(a) - x_lag|>_periodic
//!      Linear EdS:  d_meas(a) / d_meas(a_init) = a / a_init
//!   2. RMS peculiar velocity   v_rms(a) = sqrt(<|v|²>)
//!      Linear EdS:  v_rms(a) ∝ sqrt(a)
//!   3. RMS density contrast on coarse 32³ grid (>>1 particle/cell — robust)
//!      Linear EdS:  σ(δ)(a) ∝ a    (early times, before non-linear)
//!
//! Verdict: validate fix if mean of (R_d ± R_σ) is in [0.90, 1.10] over a > 3·a_init.
//! v_rms scaling reported separately (sanity check).

use janus::nbody_gpu::GpuNBodySimulation;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::Write;

const N_SIDE: usize = 100;
const N_PART: usize = N_SIDE * N_SIDE * N_SIDE; // 1_000_000
const L_BOX: f64 = 200.0;
// z_init can be overridden via env var Z_INIT_VAL=9.0 etc.
const Z_INIT_DEFAULT: f64 = 49.0;
const Z_FINAL: f64 = 0.0;
const H0_KMS_MPC: f64 = 70.0;
const MPC_GYR_TO_KMS: f64 = 977.8;
const N_SNAPSHOTS: usize = 50;
const N_GRID_DENSITY: usize = 32; // coarse: ~30 particles/cell — minimal shot noise

const EPS_DEFAULT: f64 = 0.05;
const THETA_BH: f64 = 0.7;
const SEED: u64 = 4242;

const PSI_RMS_AT_ZINIT: f64 = 0.10;

fn h_eds(a: f64, h0_gyr: f64) -> f64 {
    h0_gyr * a.powf(-1.5)
}

fn a_eds_step(a: f64, dt_gyr: f64, h0_gyr: f64) -> f64 {
    a + h0_gyr * a.powf(-0.5) * dt_gyr
}

fn periodic_dx(d: f64, l: f64) -> f64 {
    let half = l / 2.0;
    let mut r = d;
    if r >  half { r -= l; }
    if r < -half { r += l; }
    r
}

fn cic_density(positions: &[f64], n_grid: usize, box_size: f64, n_part: usize) -> Vec<f64> {
    let cell = box_size / n_grid as f64;
    let mut rho = vec![0.0_f64; n_grid * n_grid * n_grid];
    for i in 0..n_part {
        let x = (positions[3*i]   + box_size / 2.0).rem_euclid(box_size) / cell;
        let y = (positions[3*i+1] + box_size / 2.0).rem_euclid(box_size) / cell;
        let z = (positions[3*i+2] + box_size / 2.0).rem_euclid(box_size) / cell;
        let i0 = (x.floor() as i64).rem_euclid(n_grid as i64) as usize;
        let j0 = (y.floor() as i64).rem_euclid(n_grid as i64) as usize;
        let k0 = (z.floor() as i64).rem_euclid(n_grid as i64) as usize;
        let dx = x - x.floor();
        let dy = y - y.floor();
        let dz = z - z.floor();
        let i1 = (i0 + 1) % n_grid;
        let j1 = (j0 + 1) % n_grid;
        let k1 = (k0 + 1) % n_grid;
        for &(ix, wx) in &[(i0, 1.0 - dx), (i1, dx)] {
            for &(iy, wy) in &[(j0, 1.0 - dy), (j1, dy)] {
                for &(iz, wz) in &[(k0, 1.0 - dz), (k1, dz)] {
                    rho[ix * n_grid * n_grid + iy * n_grid + iz] += wx * wy * wz;
                }
            }
        }
    }
    rho
}

fn sigma_delta(rho: &[f64]) -> f64 {
    let n = rho.len() as f64;
    let mean: f64 = rho.iter().sum::<f64>() / n;
    let var: f64 = rho.iter().map(|r| (r/mean - 1.0).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

fn mean_displacement(positions: &[f64], lag: &[f64], n_part: usize, box_size: f64) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..n_part {
        let dx = periodic_dx(positions[3*i]   - lag[3*i],   box_size);
        let dy = periodic_dx(positions[3*i+1] - lag[3*i+1], box_size);
        let dz = periodic_dx(positions[3*i+2] - lag[3*i+2], box_size);
        s += (dx*dx + dy*dy + dz*dz).sqrt();
    }
    s / n_part as f64
}

fn vrms(velocities: &[f64], n_part: usize) -> f64 {
    let mut s2 = 0.0_f64;
    for i in 0..n_part {
        let v2 = velocities[3*i].powi(2) + velocities[3*i+1].powi(2) + velocities[3*i+2].powi(2);
        s2 += v2;
    }
    (s2 / n_part as f64).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EdS Validation (peculiar convention) ===");
    let z_init: f64 = std::env::var("Z_INIT_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(Z_INIT_DEFAULT);
    println!("N={}  box={} Mpc  z_init={}  z_final={}  snapshots={}",
        N_PART, L_BOX, z_init, Z_FINAL, N_SNAPSHOTS);
    println!("Convention: peculiar — drift /a, kick (acc/a² - H·v)");
    println!("Metrics: mean displacement d, σ(δ) on 32³, v_rms");
    println!("");

    let h0_gyr = H0_KMS_MPC / MPC_GYR_TO_KMS;
    let a_init = 1.0 / (1.0 + z_init);
    let a_final = 1.0 / (1.0 + Z_FINAL);

    let mut rng = StdRng::seed_from_u64(SEED);
    let cell = L_BOX / N_SIDE as f64;
    let half = L_BOX / 2.0;

    let mut positions = Vec::with_capacity(N_PART * 3);
    let mut velocities = Vec::with_capacity(N_PART * 3);
    let mut lagrangian = Vec::with_capacity(N_PART * 3); // initial positions BEFORE displacement
    let signs: Vec<i32> = vec![1; N_PART];

    let h_init = h_eds(a_init, h0_gyr);
    // For Zel'dovich with current displacement ψ and D=a (EdS):
    //   ẋ_co = H·ψ   ⇒   v_pec = a·ẋ_co = a·H·ψ
    let vel_factor = a_init * h_init;
    println!("[IC] a_init = {:.4}  H(a_init) = {:.4} 1/Gyr  vel_factor = a·H = {:.4e}",
             a_init, h_init, vel_factor);

    use rand_distr::{Normal, Distribution};
    let normal = Normal::new(0.0, PSI_RMS_AT_ZINIT).unwrap();

    for i in 0..N_SIDE {
        for j in 0..N_SIDE {
            for k in 0..N_SIDE {
                let x_lag = (i as f64 + 0.5) * cell - half;
                let y_lag = (j as f64 + 0.5) * cell - half;
                let z_lag = (k as f64 + 0.5) * cell - half;
                lagrangian.push(x_lag);
                lagrangian.push(y_lag);
                lagrangian.push(z_lag);

                let psi_x = normal.sample(&mut rng);
                let psi_y = normal.sample(&mut rng);
                let psi_z = normal.sample(&mut rng);

                let mut x = x_lag + psi_x;
                let mut y = y_lag + psi_y;
                let mut z = z_lag + psi_z;
                if x >  half { x -= L_BOX; } else if x < -half { x += L_BOX; }
                if y >  half { y -= L_BOX; } else if y < -half { y += L_BOX; }
                if z >  half { z -= L_BOX; } else if z < -half { z += L_BOX; }

                positions.push(x);
                positions.push(y);
                positions.push(z);

                velocities.push(psi_x * vel_factor);
                velocities.push(psi_y * vel_factor);
                velocities.push(psi_z * vel_factor);
            }
        }
    }

    println!("[IC] {} particles ready (psi_rms={:.3} Mpc)", N_PART, PSI_RMS_AT_ZINIT);

    let mut sim = GpuNBodySimulation::new_with_state(
        N_PART, 0, L_BOX, positions, velocities, signs.clone(),
    )?;
    sim.set_theta(THETA_BH);
    let eps_val: f64 = std::env::var("EPS_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(EPS_DEFAULT);
    sim.set_softening(eps_val);
    sim.set_phi(1.0, 1.0);
    sim.c_ratio_sq = 1.0;
    sim.repulsion_scale = 0.0;
    let gravity_off = std::env::var("GRAVITY_OFF").is_ok();
    if gravity_off {
        sim.set_mass_factor(0.0);
        println!("[SIM] GRAVITY_OFF — pure expansion + Hubble drag test");
        println!("[SIM] Expected v_pec(a) = v_init * a_init / a");
    } else {
        sim.set_mass_factor(1.0 / 0.3);
    }
    println!("[SIM] θ={}  ε={} Mpc  Ω_m=1  cross-coupling OFF", THETA_BH, eps_val);

    let log_ai = a_init.ln();
    let log_af = a_final.ln();
    let snap_a: Vec<f64> = (0..N_SNAPSHOTS)
        .map(|i| (log_ai + (log_af - log_ai) * i as f64 / (N_SNAPSHOTS - 1) as f64).exp())
        .collect();

    let log_path = if std::path::Path::new("/app/output").is_dir() {
        "/app/output/eds_validation.log"
    } else {
        "/mnt/T2/janus-sim/output/eds_validation.log"
    };
    let mut log = File::create(&log_path)?;
    writeln!(log, "# EdS Validation log (peculiar convention)")?;
    writeln!(log, "# step  a         z         t_Gyr     d_meas      sigma       v_rms")?;

    let dt: f64 = std::env::var("DT_VAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let mut a = a_init;
    let mut t_gyr = 0.0_f64;
    let mut snap_idx = 0;

    let mut snap_records: Vec<(f64, f64, f64, f64)> = Vec::new(); // (a, d, σ, v_rms)

    println!("[RUN] dt={} Gyr", dt);
    println!("");
    println!("step      a         z         t_Gyr     d_meas       sigma        v_rms");

    let max_steps = 1_000_000;
    for step in 0..max_steps {
        if snap_idx < N_SNAPSHOTS && a >= snap_a[snap_idx] {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;

            let d = mean_displacement(&pos, &lagrangian, N_PART, L_BOX);
            let rho = cic_density(&pos, N_GRID_DENSITY, L_BOX, N_PART);
            let sig = sigma_delta(&rho);
            let v = vrms(&vel, N_PART);
            let z = 1.0 / a - 1.0;

            snap_records.push((a, d, sig, v));

            let line = format!("{:>5}  {:.5}  {:.4}  {:.4}  {:.4e}  {:.4e}  {:.4e}",
                step, a, z, t_gyr, d, sig, v);
            println!("{}", line);
            writeln!(log, "{}", line)?;
            log.flush()?;

            snap_idx += 1;
            if snap_idx == N_SNAPSHOTS { break; }
        }

        let h = h_eds(a, h0_gyr);
        sim.step_with_expansion_dkd_gpu_cosmo(dt, a, a, h, h)?;
        a = a_eds_step(a, dt, h0_gyr);
        t_gyr += dt;

        if a >= a_final { break; }
    }

    // Verdict — three metrics
    let (a0, d0, sig0, v0) = snap_records[0];
    println!("\n--- METRICS ---");
    println!("baseline (a={:.4}): d={:.4e}  σ={:.4e}  v_rms={:.4e}", a0, d0, sig0, v0);

    println!("\nstep   a         R_d=d/D_th    R_σ=σ/D_th    R_v=v/√D_th");
    let mut ratios_d: Vec<f64> = Vec::new();
    let mut ratios_s: Vec<f64> = Vec::new();
    let mut ratios_v: Vec<f64> = Vec::new();
    for (idx, &(a_i, d_i, sig_i, v_i)) in snap_records.iter().enumerate() {
        let d_th = a_i / a0; // EdS linear: D ∝ a
        let r_d = (d_i / d0) / d_th;
        let r_s = (sig_i / sig0) / d_th;
        let r_v = (v_i / v0) / d_th.sqrt(); // v ∝ √a in EdS peculiar
        let line = format!("{:>4}  {:.4}    {:.4}        {:.4}        {:.4}",
                           idx, a_i, r_d, r_s, r_v);
        println!("{}", line);
        writeln!(log, "{}", line)?;
        if a_i > 3.0 * a_init {
            ratios_d.push(r_d);
            ratios_s.push(r_s);
            ratios_v.push(r_v);
        }
    }

    fn stats(v: &[f64]) -> (f64, f64) {
        let n = v.len() as f64;
        let m: f64 = v.iter().sum::<f64>() / n.max(1.0);
        let s: f64 = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0)).sqrt();
        (m, s)
    }
    let (md, sd_) = stats(&ratios_d);
    let (ms, ss_) = stats(&ratios_s);
    let (mv, sv_) = stats(&ratios_v);

    println!("\n--- VERDICT ---");
    println!("Skipping snapshots with a ≤ {:.4} (transient)", 3.0 * a_init);
    println!("R_d (displacement)  : ⟨{:.4}⟩  σ={:.4}", md, sd_);
    println!("R_σ (density)       : ⟨{:.4}⟩  σ={:.4}", ms, ss_);
    println!("R_v (velocity)      : ⟨{:.4}⟩  σ={:.4}", mv, sv_);

    writeln!(log, "\n# VERDICT")?;
    writeln!(log, "# R_d_mean = {:.4}, R_d_std = {:.4}", md, sd_)?;
    writeln!(log, "# R_σ_mean = {:.4}, R_σ_std = {:.4}", ms, ss_)?;
    writeln!(log, "# R_v_mean = {:.4}, R_v_std = {:.4}", mv, sv_)?;

    let validated_d = md >= 0.90 && md <= 1.10 && sd_ < 0.10;
    let rejected_d  = md < 0.80 || md > 1.20;
    let validated_s = ms >= 0.85 && ms <= 1.15 && ss_ < 0.15;
    let rejected_s  = ms < 0.75 || ms > 1.25;

    if validated_d && validated_s {
        println!("✅ FIX VALIDATED — peculiar coupling reproduces EdS linear growth");
        writeln!(log, "# VERDICT: VALIDATED")?;
    } else if rejected_d || rejected_s {
        println!("❌ FIX REJECTED — metrics outside acceptance bands");
        writeln!(log, "# VERDICT: REJECTED")?;
    } else {
        println!("⚠ MARGINAL — investigate before relaunching production");
        writeln!(log, "# VERDICT: MARGINAL")?;
    }
    Ok(())
}
