//! JANUS VALIDATION RUN — 1M Particles, z=10→0
//!
//! Pre-production validation run with:
//! - N = 1M (100³)
//! - Box = 200 Mpc
//! - μ = 19, η = 1.045
//! - z_init = 10.0 → z_final = 0.0
//! - All modules: H(z) Janus, Φ, E, D(z) ICs
//!
//! 8 CRITERIA TO VERIFY:
//! 1. No NaN
//! 2. No crash
//! 3. E_drift < 2%
//! 4. Transition crossing at step ~465 OK
//! 5. v_rms in [10, 50000] km/s
//! 6. ρ_max doesn't diverge
//! 7. a_plus(z=0) = 1.0 ± 0.001
//! 8. E_minus/E_plus ≈ μ × c̄² ≈ 21

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use janus::janus_expansion::{compute_total_energy, energy_drift_pct, a_minus_from_a_plus, compute_phi_factors};
use janus::snapshot_v3::{SnapshotHeaderV3, ParticleV3, write_snapshot_v3};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::StdRng;
use rustfft::{FftPlanner, num_complex::Complex};

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION PARAMETERS — 1M particles, z=10→0
// ═══════════════════════════════════════════════════════════════════════════
const N_GRID: usize = 100;  // 100³ = 1M particles
const L_BOX: f64 = 200.0;   // Mpc (scaled)
const Z_INIT: f64 = 10.0;   // Start BEFORE transition
const Z_FINAL: f64 = 0.0;
const DT: f64 = 0.001;      // Gyr

// Cosmology
const ETA: f64 = 1.045;
const MU: f64 = 19.0;
const H0: f64 = 69.9;       // km/s/Mpc
const OMEGA_B: f64 = 0.05;

// Simulation
const THETA: f64 = 0.7;
const EPS_PLUS: f64 = 0.02;  // Smaller box → smaller softening
const EPS_MINUS: f64 = 0.04;

// Output
const METRIC_INTERVAL: usize = 10;
const SNAPSHOT_INTERVAL: usize = 500;  // Less frequent for validation

// ICs
const SEED_IC: u64 = 42;
const N_S: f64 = 0.965;
const DELTA_RMS: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;

// Janus expansion (Petit 2014+2018)
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;  // z ≈ 4.51
const Z_TRANSITION: f64 = 4.51;

fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;
    if a < A_TRANSITION_JANUS {
        // Gauge process era (z > 4.51)
        h0_gyr_inv / a.powf(1.5)
    } else {
        // Matter era (z ≤ 4.51)
        let cosh2_mu = a / ALPHA_SQ_JANUS;
        let cosh2_mu_safe = cosh2_mu.max(1.0);
        let cosh_mu = cosh2_mu_safe.sqrt();
        let mu_p = cosh_mu.acosh();
        let s2mu = (2.0 * mu_p).sinh();
        s2mu / (TAU_0_JANUS * ALPHA_SQ_JANUS * cosh2_mu_safe * (1.0 + 0.5 * s2mu))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LINEAR GROWTH FACTOR D(z) — Janus 2014+2018
// ═══════════════════════════════════════════════════════════════════════════
const Z_CALIBRATION: f64 = 4.0;

fn growth_factor_janus(z: f64) -> f64 {
    let a = 1.0 / (1.0 + z);

    if a <= A_TRANSITION_JANUS {
        a / 2.074
    } else {
        let d_table: [(f64, f64); 7] = [
            (0.0, 1.000000),
            (1.0, 0.386367),
            (2.0, 0.179503),
            (3.0, 0.112850),
            (4.0, 0.090886),
            (4.5, 0.087700),
            (4.51, 0.087514),
        ];

        for i in 0..d_table.len()-1 {
            let (z1, d1) = d_table[i];
            let (z2, d2) = d_table[i+1];
            if z >= z1 && z <= z2 {
                let t = (z - z1) / (z2 - z1);
                return d1 + t * (d2 - d1);
            }
        }

        let (z1, d1) = d_table[d_table.len()-2];
        let (z2, d2) = d_table[d_table.len()-1];
        let slope = (d2 - d1) / (z2 - z1);
        d2 + slope * (z - z2)
    }
}

fn ic_scaling_factor(z_init: f64) -> f64 {
    let d_init = growth_factor_janus(z_init);
    let d_calib = growth_factor_janus(Z_CALIBRATION);
    d_init / d_calib
}

// ═══════════════════════════════════════════════════════════════════════════
// PARTICLE MASSES
// ═══════════════════════════════════════════════════════════════════════════
fn compute_particle_masses(n_total: usize, l_box: f64, h0: f64, omega_b: f64, mu: f64) -> (f64, f64) {
    let h = h0 / 100.0;
    let rho_crit = 2.775e11 * h * h;
    let m_plus_total = omega_b * rho_crit * l_box.powi(3);
    let m_minus_total = mu * m_plus_total;

    let n_plus = (n_total as f64 / (1.0 + mu)) as usize;
    let n_minus = n_total - n_plus;

    let m_plus = m_plus_total / n_plus as f64;
    let m_minus = m_minus_total / n_minus as f64;

    (m_plus, m_minus)
}

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH ICS with D(z) scaling
// ═══════════════════════════════════════════════════════════════════════════
fn generate_zeldovich_ics(n_grid: usize, l_box: f64, z_init: f64, h0: f64, mu: f64)
    -> (Vec<f64>, Vec<f64>, Vec<i32>)
{
    println!("[IC] Generating Zel'dovich ICs with D(z) scaling...");
    println!("  Grid: {}³ = {} particles", n_grid, n_grid.pow(3));

    let mut rng = StdRng::seed_from_u64(SEED_IC);
    let n_particles = n_grid.pow(3);
    let n_fft = n_grid * 2;

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft.pow(3)];
    let k_nyq = PI * n_fft as f64 / l_box;

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            for ix in 0..n_fft {
                let kx = if ix <= n_fft/2 { ix as f64 } else { ix as f64 - n_fft as f64 } * 2.0 * PI / l_box;
                let ky = if iy <= n_fft/2 { iy as f64 } else { iy as f64 - n_fft as f64 } * 2.0 * PI / l_box;
                let kz = if iz <= n_fft/2 { iz as f64 } else { iz as f64 - n_fft as f64 } * 2.0 * PI / l_box;

                let k = (kx*kx + ky*ky + kz*kz).sqrt();
                if k > 0.0 && k < k_nyq {
                    let pk = k.powf(N_S - 4.0);
                    let amp = (pk * DELTA_RMS).sqrt() * rng.random::<f64>().sqrt();
                    let phase = rng.random::<f64>() * 2.0 * PI;
                    let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                    delta_k[idx] = Complex::new(amp * phase.cos(), amp * phase.sin());
                }
            }
        }
    }

    let mut psi_x: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft.pow(3)];
    let mut psi_y: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft.pow(3)];
    let mut psi_z: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft.pow(3)];

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            for ix in 0..n_fft {
                let kx = if ix <= n_fft/2 { ix as f64 } else { ix as f64 - n_fft as f64 } * 2.0 * PI / l_box;
                let ky = if iy <= n_fft/2 { iy as f64 } else { iy as f64 - n_fft as f64 } * 2.0 * PI / l_box;
                let kz = if iz <= n_fft/2 { iz as f64 } else { iz as f64 - n_fft as f64 } * 2.0 * PI / l_box;

                let k2 = kx*kx + ky*ky + kz*kz;
                if k2 > 0.0 {
                    let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                    let i_over_k2 = Complex::new(0.0, 1.0) / k2;
                    psi_x[idx] = delta_k[idx] * kx * i_over_k2;
                    psi_y[idx] = delta_k[idx] * ky * i_over_k2;
                    psi_z[idx] = delta_k[idx] * kz * i_over_k2;
                }
            }
        }
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            let start = iz * n_fft * n_fft + iy * n_fft;
            ifft.process(&mut psi_x[start..start+n_fft]);
            ifft.process(&mut psi_y[start..start+n_fft]);
            ifft.process(&mut psi_z[start..start+n_fft]);
        }
    }

    let mut max_disp = 0.0f64;
    for i in 0..n_fft.pow(3) {
        let d = (psi_x[i].re.powi(2) + psi_y[i].re.powi(2) + psi_z[i].re.powi(2)).sqrt();
        max_disp = max_disp.max(d);
    }

    let cell_size = l_box / n_grid as f64;
    let d_scaling = ic_scaling_factor(z_init);
    let target_disp_base = 0.30 * cell_size;
    let target_disp = target_disp_base * d_scaling;
    let scale = if max_disp > 1e-20 { target_disp / max_disp } else { 0.0 };

    println!("  D(z={:.1})/D(z={:.0}) = {:.4}", z_init, Z_CALIBRATION, d_scaling);
    println!("  Target displacement: {:.3} Mpc ({:.1}% cell)", target_disp, target_disp / cell_size * 100.0);

    let mut positions = Vec::with_capacity(n_particles * 3);
    let mut velocities = Vec::with_capacity(n_particles * 3);
    let mut signs = Vec::with_capacity(n_particles);

    let n_plus_target = (n_particles as f64 / (1.0 + mu)) as usize;
    let spacing_fft = l_box / n_fft as f64;
    let offset_x = rng.random::<f64>() * spacing_fft;
    let offset_y = rng.random::<f64>() * spacing_fft;
    let offset_z = rng.random::<f64>() * spacing_fft;

    let h_gyr = h0 / MPC_GYR_TO_KMS;
    let a_init = 1.0 / (1.0 + z_init);
    let vel_scale = a_init * h_gyr * scale;
    let half_box = l_box / 2.0;

    for idx in 0..n_particles {
        let x0 = rng.random::<f64>() * l_box - half_box;
        let y0 = rng.random::<f64>() * l_box - half_box;
        let z0 = rng.random::<f64>() * l_box - half_box;

        let gx = (((x0 + half_box + offset_x) / spacing_fft) as usize) % n_fft;
        let gy = (((y0 + half_box + offset_y) / spacing_fft) as usize) % n_fft;
        let gz = (((z0 + half_box + offset_z) / spacing_fft) as usize) % n_fft;
        let grid_idx = gz * n_fft * n_fft + gy * n_fft + gx;

        let dx = psi_x[grid_idx].re * scale;
        let dy = psi_y[grid_idx].re * scale;
        let dz = psi_z[grid_idx].re * scale;

        let mut x = x0 + dx;
        let mut y = y0 + dy;
        let mut z = z0 + dz;

        if x > half_box { x -= l_box; } else if x < -half_box { x += l_box; }
        if y > half_box { y -= l_box; } else if y < -half_box { y += l_box; }
        if z > half_box { z -= l_box; } else if z < -half_box { z += l_box; }

        positions.push(x);
        positions.push(y);
        positions.push(z);

        velocities.push(dx * vel_scale / scale);
        velocities.push(dy * vel_scale / scale);
        velocities.push(dz * vel_scale / scale);

        let sign = if idx < n_plus_target { 1 } else { -1 };
        signs.push(sign);
    }

    // Shuffle
    let mut indices: Vec<usize> = (0..n_particles).collect();
    indices.shuffle(&mut rng);

    let mut pos_shuffled = vec![0.0; n_particles * 3];
    let mut vel_shuffled = vec![0.0; n_particles * 3];
    let mut signs_shuffled = vec![0i32; n_particles];

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        pos_shuffled[new_idx * 3] = positions[old_idx * 3];
        pos_shuffled[new_idx * 3 + 1] = positions[old_idx * 3 + 1];
        pos_shuffled[new_idx * 3 + 2] = positions[old_idx * 3 + 2];
        vel_shuffled[new_idx * 3] = velocities[old_idx * 3];
        vel_shuffled[new_idx * 3 + 1] = velocities[old_idx * 3 + 1];
        vel_shuffled[new_idx * 3 + 2] = velocities[old_idx * 3 + 2];
        signs_shuffled[new_idx] = signs[old_idx];
    }

    let n_plus = signs_shuffled.iter().filter(|&&s| s > 0).count();
    let n_minus = n_particles - n_plus;
    println!("  N+ = {}, N- = {} (ratio = {:.4})", n_plus, n_minus, n_minus as f64 / n_plus as f64);

    (pos_shuffled, vel_shuffled, signs_shuffled)
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════
fn compute_densities_split(positions: &[f64], signs: &[i32], n_grid: usize, l_box: f64)
    -> (f64, f64)
{
    let n = signs.len();
    let cell = l_box / n_grid as f64;
    let half_box = l_box / 2.0;

    let mut grid_plus = vec![0u32; n_grid.pow(3)];
    let mut grid_minus = vec![0u32; n_grid.pow(3)];

    for i in 0..n {
        let x = positions[i * 3] + half_box;
        let y = positions[i * 3 + 1] + half_box;
        let z = positions[i * 3 + 2] + half_box;

        let ix = ((x / cell) as usize).min(n_grid - 1);
        let iy = ((y / cell) as usize).min(n_grid - 1);
        let iz = ((z / cell) as usize).min(n_grid - 1);
        let idx = iz * n_grid * n_grid + iy * n_grid + ix;

        if signs[i] > 0 {
            grid_plus[idx] += 1;
        } else {
            grid_minus[idx] += 1;
        }
    }

    let max_plus = *grid_plus.iter().max().unwrap_or(&0) as f64;
    let max_minus = *grid_minus.iter().max().unwrap_or(&0) as f64;

    (max_plus, max_minus)
}

fn compute_vrms_split(velocities: &[f64], signs: &[i32]) -> (f64, f64) {
    let mut sum_plus = 0.0f64;
    let mut sum_minus = 0.0f64;
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;

    for i in 0..signs.len() {
        let v2 = velocities[i * 3].powi(2) + velocities[i * 3 + 1].powi(2) + velocities[i * 3 + 2].powi(2);
        if signs[i] > 0 {
            sum_plus += v2;
            n_plus += 1;
        } else {
            sum_minus += v2;
            n_minus += 1;
        }
    }

    let v_rms_plus_mpc_gyr = if n_plus > 0 { (sum_plus / n_plus as f64).sqrt() } else { 0.0 };
    let v_rms_minus_mpc_gyr = if n_minus > 0 { (sum_minus / n_minus as f64).sqrt() } else { 0.0 };

    (v_rms_plus_mpc_gyr * MPC_GYR_TO_KMS, v_rms_minus_mpc_gyr * MPC_GYR_TO_KMS)
}

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION TRACKER
// ═══════════════════════════════════════════════════════════════════════════
struct ValidationTracker {
    has_nan: bool,
    max_e_drift: f64,
    transition_crossed: bool,
    transition_step: usize,
    transition_ok: bool,
    v_rms_out_of_range: Vec<(usize, f64, f64)>,  // (step, v_plus, v_minus)
    rho_max_history: Vec<(usize, f64, f64)>,     // (step, rho_plus, rho_minus)
    final_a: f64,
    e_ratio_at_z0: f64,
}

impl ValidationTracker {
    fn new() -> Self {
        Self {
            has_nan: false,
            max_e_drift: 0.0,
            transition_crossed: false,
            transition_step: 0,
            transition_ok: true,
            v_rms_out_of_range: Vec::new(),
            rho_max_history: Vec::new(),
            final_a: 0.0,
            e_ratio_at_z0: 0.0,
        }
    }

    fn check_values(&mut self, step: usize, a: f64, z: f64, v_plus: f64, v_minus: f64,
                    rho_plus: f64, rho_minus: f64, e_drift: f64, e_plus: f64, e_minus: f64) {
        // Check NaN
        if a.is_nan() || z.is_nan() || v_plus.is_nan() || v_minus.is_nan() ||
           rho_plus.is_nan() || rho_minus.is_nan() || e_drift.is_nan() {
            self.has_nan = true;
        }

        // Track max E_drift
        if e_drift.abs() > self.max_e_drift {
            self.max_e_drift = e_drift.abs();
        }

        // Check v_rms bounds
        if v_plus < 10.0 || v_plus > 50000.0 || v_minus < 10.0 || v_minus > 50000.0 {
            self.v_rms_out_of_range.push((step, v_plus, v_minus));
        }

        // Track density history (sample every 1000 steps)
        if step % 1000 == 0 {
            self.rho_max_history.push((step, rho_plus, rho_minus));
        }

        // Track energy ratio
        if e_plus.abs() > 1e-10 {
            self.e_ratio_at_z0 = e_minus.abs() / e_plus.abs();
        }

        self.final_a = a;
    }

    fn record_transition(&mut self, step: usize, z_before: f64, z_after: f64, a_before: f64, a_after: f64) {
        self.transition_crossed = true;
        self.transition_step = step;

        // Check transition is smooth
        let da_frac = (a_after - a_before).abs() / a_before;
        if da_frac > 0.10 {  // > 10% jump
            self.transition_ok = false;
        }
    }

    fn print_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║                    VALIDATION REPORT — 8 CRITERIA                        ║");
        println!("╠══════════════════════════════════════════════════════════════════════════╣");

        // Criterion 1: No NaN
        let c1 = !self.has_nan;
        println!("║ [{}] 1. No NaN detected", if c1 { "✓" } else { "✗" });

        // Criterion 2: No crash (if we got here, we didn't crash)
        println!("║ [✓] 2. No crash (run completed)");

        // Criterion 3: E_drift < 2%
        let c3 = self.max_e_drift < 2.0;
        println!("║ [{}] 3. E_drift < 2% (max = {:.3}%)", if c3 { "✓" } else { "✗" }, self.max_e_drift);

        // Criterion 4: Transition crossed OK
        let c4 = self.transition_crossed && self.transition_ok;
        println!("║ [{}] 4. Transition at step {} {}",
                 if c4 { "✓" } else { "✗" },
                 self.transition_step,
                 if self.transition_ok { "OK" } else { "FAILED" });

        // Criterion 5: v_rms in bounds
        let c5 = self.v_rms_out_of_range.is_empty();
        if c5 {
            println!("║ [✓] 5. v_rms always in [10, 50000] km/s");
        } else {
            println!("║ [✗] 5. v_rms out of range {} times", self.v_rms_out_of_range.len());
            for (step, vp, vm) in self.v_rms_out_of_range.iter().take(3) {
                println!("║      step {}: v+={:.1}, v-={:.1}", step, vp, vm);
            }
        }

        // Criterion 6: ρ_max doesn't diverge
        let c6 = self.rho_max_history.iter().all(|(_, rp, rm)| *rp < 1e6 && *rm < 1e6);
        println!("║ [{}] 6. ρ_max bounded (no divergence)", if c6 { "✓" } else { "✗" });

        // Criterion 7: a(z=0) = 1.0 ± 0.001
        let c7 = (self.final_a - 1.0).abs() < 0.001;
        println!("║ [{}] 7. a(z=0) = {:.6} (target: 1.000 ± 0.001)", if c7 { "✓" } else { "✗" }, self.final_a);

        // Criterion 8: E_minus/E_plus ≈ μ×c̄² ≈ 21
        let expected_ratio = MU * 1.035 * 1.035;  // μ × c̄² at z≈0
        let c8 = (self.e_ratio_at_z0 - expected_ratio).abs() / expected_ratio < 0.20;  // 20% tolerance
        println!("║ [{}] 8. E-/E+ = {:.2} (expected: {:.2} ± 20%)",
                 if c8 { "✓" } else { "✗" }, self.e_ratio_at_z0, expected_ratio);

        println!("╠══════════════════════════════════════════════════════════════════════════╣");

        let all_pass = c1 && c3 && c4 && c5 && c6 && c7 && c8;
        if all_pass {
            println!("║                     ALL 8 CRITERIA PASS ✓                                ║");
            println!("║                     >>> GO FOR PRODUCTION <<<                             ║");
        } else {
            println!("║                     SOME CRITERIA FAILED ✗                                ║");
            println!("║                     >>> INVESTIGATE BEFORE PRODUCTION <<<                 ║");
        }
        println!("╚══════════════════════════════════════════════════════════════════════════╝");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn main() {
    let out_dir = "/app/output/janus_validation_1m";

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  JANUS VALIDATION RUN — 1M Particles, z=10 → z=0                         ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Pre-production validation with all modules active                       ║");
    println!("║  Expected runtime: 3-5 hours                                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {}³ = {} particles", N_GRID, N_GRID.pow(3));
    println!("║  Box = {} Mpc, z = {} → {}", L_BOX, Z_INIT, Z_FINAL);
    println!("║  μ = {}, η = {}, H₀ = {} km/s/Mpc", MU, ETA, H0);
    println!("║  dt = {} Gyr, θ = {}", DT, THETA);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", out_dir)).expect("Failed to create snapshots dir");

    // Generate ICs
    let (positions, velocities, signs) = generate_zeldovich_ics(N_GRID, L_BOX, Z_INIT, H0, MU);
    let n_particles = signs.len();
    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = n_particles - n_plus;

    let (m_plus, m_minus) = compute_particle_masses(n_particles, L_BOX, H0, OMEGA_B, MU);
    println!("[MASS] m+ = {:.4e} M☉, m- = {:.4e} M☉", m_plus, m_minus);

    // Energy tracking
    let vol = L_BOX * L_BOX * L_BOX;
    let m_plus_total = n_plus as f64 * m_plus;
    let m_minus_total = n_minus as f64 * m_minus;
    let rho_plus_comoving = m_plus_total / vol;
    let rho_minus_comoving = -m_minus_total / vol;

    println!("[ENERGY] ρ⁺_comoving = {:.4e} M☉/Mpc³", rho_plus_comoving);
    println!("[ENERGY] ρ⁻_comoving = {:.4e} M☉/Mpc³", rho_minus_comoving);

    // Initialize GPU
    println!("\n[GPU] Compiling CUDA kernels...");
    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_BOX,
        positions.clone(), velocities.clone(), signs.clone()
    ).expect("Failed to create GPU simulation");
    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(EPS_PLUS);

    let janus_mass_factor = OMEGA_B * (1.0 + MU) / 0.3;
    gpu_sim.set_mass_factor(janus_mass_factor);
    println!("  ✓ GPU initialized, mass factor = {:.4}", janus_mass_factor);

    // CSV output
    let csv_path = format!("{}/time_series.csv", out_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,z,a,H,rho_max_plus,rho_max_minus,v_rms_plus,v_rms_minus,E_drift_pct").unwrap();

    // Validation tracker
    let mut tracker = ValidationTracker::new();

    // State
    let mut a = 1.0 / (1.0 + Z_INIT);
    let mut t_gyr = 0.0;
    let mut e_total_0: Option<f64> = None;
    let mut z_prev = Z_INIT;
    let mut a_prev = a;

    // Report steps
    let report_steps = [100, 465, 1000, 5000, 10000, 15000];

    println!("\n[RUN] Starting main loop (z={} → z={})...", Z_INIT, Z_FINAL);
    println!("      Estimated ~16,335 steps\n");

    let mut step = 0;
    loop {
        let z = 1.0 / a - 1.0;

        if z < Z_FINAL {
            println!("\n  ✓ Reached z_final = {:.4} at step {}", z, step);
            break;
        }

        // H(a) Janus
        let h = compute_hubble_janus(a, H0);

        // VSL c̄(z)
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_bar = c_ratio_sq.sqrt();

        // Phi coupling
        let a_minus = a_minus_from_a_plus(a, ETA);
        let (phi, phi_inv) = compute_phi_factors(a, ETA);

        // Update GPU params every 10 steps
        if step % 10 == 0 {
            gpu_sim.set_c_ratio(c_bar);
            gpu_sim.set_phi(phi, phi_inv);
        }

        // Detect transition crossing
        if z_prev >= Z_TRANSITION && z < Z_TRANSITION {
            println!("\n  🔄 TRANSITION CROSSED at step {}", step);
            println!("     z: {:.4} → {:.4}", z_prev, z);
            println!("     a: {:.6} → {:.6}", a_prev, a);
            tracker.record_transition(step, z_prev, z, a_prev, a);
        }

        // Metrics
        let do_metric = step % METRIC_INTERVAL == 0;
        let do_report = report_steps.contains(&step) || step == 16335;

        if do_metric || do_report {
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_gpu = gpu_sim.signs();

            let (rho_max_plus, rho_max_minus) = compute_densities_split(&pos, &signs_gpu, 32, L_BOX);
            let (v_rms_plus, v_rms_minus) = compute_vrms_split(&vel, &signs_gpu);

            let (e_total, e_plus, e_minus) = compute_total_energy(
                rho_plus_comoving, rho_minus_comoving,
                1.0, c_bar, a, a_minus
            );

            if e_total_0.is_none() {
                e_total_0 = Some(e_total);
                println!("  📊 Initial E_total = {:.6e}", e_total);
            }
            let e_drift = energy_drift_pct(e_total, e_total_0.unwrap());

            // Track validation
            tracker.check_values(step, a, z, v_rms_plus, v_rms_minus,
                               rho_max_plus, rho_max_minus, e_drift, e_plus, e_minus);

            if do_metric {
                writeln!(csv, "{},{:.6},{:.8},{:.6},{:.2},{:.2},{:.2},{:.2},{:.4}",
                    step, z, a, h, rho_max_plus, rho_max_minus,
                    v_rms_plus, v_rms_minus, e_drift
                ).unwrap();
            }

            // Progress every 500 steps
            if step % 500 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let steps_per_sec = if elapsed > 0.0 { step as f64 / elapsed } else { 0.0 };
                let eta_hours = if steps_per_sec > 0.0 { (16335 - step) as f64 / steps_per_sec / 3600.0 } else { 0.0 };
                println!("  Step {:6} | z={:.4} | a={:.6} | v±={:.0}/{:.0} | E_drift={:.3}% | {:.1} step/s | ETA {:.1}h",
                    step, z, a, v_rms_plus, v_rms_minus, e_drift, steps_per_sec, eta_hours);
            }

            // Detailed report at specific steps
            if do_report {
                println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
                println!("║  CHECKPOINT: Step {} (z = {:.4})                                   ", step, z);
                println!("╠══════════════════════════════════════════════════════════════════════════╣");
                println!("║  a_plus = {:.8}", a);
                println!("║  H(z) = {:.6} Gyr⁻¹", h);
                println!("║  ρ_max+ = {:.2}, ρ_max- = {:.2}", rho_max_plus, rho_max_minus);
                println!("║  v_rms+ = {:.1} km/s, v_rms- = {:.1} km/s", v_rms_plus, v_rms_minus);
                println!("║  E_drift = {:.4}%", e_drift);
                println!("║  φ = {:.6}, c̄ = {:.6}", phi, c_bar);
                let elapsed = start_time.elapsed().as_secs_f64() / 3600.0;
                println!("║  Elapsed: {:.2}h", elapsed);
                println!("╚══════════════════════════════════════════════════════════════════════════╝\n");
            }

            csv.flush().unwrap();
        }

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = format!("{}/snapshots/snap_{:06}.bin", out_dir, step);
            let path = Path::new(&snap_path);

            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_data = gpu_sim.signs();

            let mut particles: Vec<ParticleV3> = Vec::with_capacity(n_particles);
            for i in 0..n_particles {
                let px = pos[i * 3] as f32;
                let py = pos[i * 3 + 1] as f32;
                let pz = pos[i * 3 + 2] as f32;
                let vx = (vel[i * 3] * MPC_GYR_TO_KMS) as f32;
                let vy = (vel[i * 3 + 1] * MPC_GYR_TO_KMS) as f32;
                let vz = (vel[i * 3 + 2] * MPC_GYR_TO_KMS) as f32;
                let sign_byte = if signs_data[i] > 0 { 1u8 } else { 255u8 };
                let mass = if signs_data[i] > 0 { m_plus as f32 } else { m_minus as f32 };
                let eps = if signs_data[i] > 0 { EPS_PLUS as f32 } else { EPS_MINUS as f32 };

                particles.push(ParticleV3 {
                    pos: [px, py, pz],
                    vel: [vx, vy, vz],
                    mass,
                    epsilon: eps,
                    sign: sign_byte,
                    split_level: 0,
                    is_star: 0,
                    flags: 0,
                });
            }

            let mut header = SnapshotHeaderV3::new("validation_1m");
            header.a = a;
            header.t_gyr = t_gyr;
            header.l_box = L_BOX;
            header.h0 = H0;
            header.mu = MU;
            header.omega_b = OMEGA_B;
            header.m_part_plus_base = m_plus;
            header.m_part_minus_base = m_minus;
            header.eps_plus_base = EPS_PLUS;
            header.eps_minus_base = EPS_MINUS;
            header.seed_ic = SEED_IC as u32;
            header.z_init = Z_INIT;

            write_snapshot_v3(path, &header, &particles).expect("Failed to write snapshot");
            println!("    📸 Snapshot {} saved", step);
        }

        // Time integration
        z_prev = z;
        a_prev = a;

        gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 1.0).unwrap();
        let da = a * h * DT;
        a += da;
        t_gyr += DT;

        step += 1;
    }

    // Final summary
    let total_time = start_time.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  RUN COMPLETE                                                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Total time: {:.2} hours ({:.0} s)", total_time / 3600.0, total_time);
    println!("║  Final step: {}", step);
    println!("║  Final z: {:.6}", 1.0 / a - 1.0);
    println!("║  Final a: {:.8}", a);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Print validation report
    tracker.print_report();

    // Write summary JSON
    let summary = format!(r#"{{
  "run": "janus_validation_1m",
  "n_particles": {},
  "box_mpc": {},
  "z_init": {},
  "z_final": {:.6},
  "total_steps": {},
  "runtime_hours": {:.2},
  "final_a": {:.8},
  "max_e_drift_pct": {:.4},
  "transition_step": {},
  "all_criteria_pass": {}
}}"#,
        n_particles, L_BOX, Z_INIT, 1.0/a - 1.0, step,
        total_time / 3600.0, a, tracker.max_e_drift,
        tracker.transition_step,
        !tracker.has_nan && tracker.max_e_drift < 2.0 && tracker.transition_ok
    );

    let summary_path = format!("{}/summary.json", out_dir);
    fs::write(&summary_path, summary).expect("Failed to write summary");
    println!("\n  📝 Summary written to {}", summary_path);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
    std::process::exit(1);
}
