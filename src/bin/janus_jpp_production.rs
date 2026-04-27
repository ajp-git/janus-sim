//! JANUS JPP PRODUCTION — μ=19 Canonical Run
//!
//! Configuration Petit et al. 2024:
//! - N = 10M (50.5% m+ / 49.5% m-)
//! - Box = 500 Mpc, z = 4.0 → 0
//! - H(z) Janus 2014+2018 (gauge process + matter era)
//! - VSL dynamique c̄(t) avec δ=0.0431
//! - Couplage Φ=(ā/a)³ inter-espèces
//! - E_total tracking (corrected formula)
//! - SPH m+ only (T_init=10⁴K, T_floor=100K)
//! - m- gravity only (no T_reion floor yet)

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
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
// PRODUCTION PARAMETERS — μ=19 CANONICAL (JPP 2024)
// ═══════════════════════════════════════════════════════════════════════════
const N_GRID: usize = 215;  // 215³ ≈ 9.94M particles
const L_BOX: f64 = 500.0;   // Mpc
const Z_INIT: f64 = 10.0;  // Start BEFORE Janus transition (z=4.51)
const Z_FINAL: f64 = 0.0;
const DT: f64 = 0.001;      // Gyr

// Cosmology
const ETA: f64 = 1.045;     // From Pantheon+ fit
const MU: f64 = 19.0;       // Canonical JPP
const H0: f64 = 69.9;       // km/s/Mpc
const OMEGA_B: f64 = 0.05;

// Simulation
const THETA: f64 = 0.7;     // Barnes-Hut opening angle
const EPS_PLUS: f64 = 0.05; // Softening m+ [Mpc]
const EPS_MINUS: f64 = 0.10;// Softening m- [Mpc]

// Output intervals
const METRIC_INTERVAL: usize = 5;
const SNAPSHOT_INTERVAL: usize = 10;
const FRAME_INTERVAL: usize = 5;

// Zel'dovich ICs
const SEED_IC: u64 = 42;
const N_S: f64 = 0.965;
const DELTA_RMS: f64 = 0.15;

// Baryonic physics
const T_INIT_PLUS: f64 = 10000.0;  // K
const T_FLOOR: f64 = 100.0;        // K

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;

// Janus expansion (Petit 2014+2018)
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;  // z ≈ 4.51

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
// Gauge process era (z > 4.51): D = a (EdS exact)
// Matter era (z < 4.51): tabulated from ODE integration
// Normalized to D(0) = 1

const Z_CALIBRATION: f64 = 4.0;  // z for which |ψ| = 30% cell was validated

/// Linear growth factor D(z)/D(0) for Janus cosmology
/// Uses piecewise: D=a for gauge era, tabulated for matter era
fn growth_factor_janus(z: f64) -> f64 {
    let a = 1.0 / (1.0 + z);

    if a <= A_TRANSITION_JANUS {
        // Gauge process era: D = a, normalized by D(0)
        // D(0) = 1.0 by definition, D(z) = a / D(0) factor
        // From numerical integration: D(0)/a(0) ≈ 2.074 in matter era
        // At transition: D(a_trans) = a_trans
        // Need to match: D_matter(a_trans) = a_trans
        // Numerical result: D(0) = 2.074, D(a_trans)/D(0) = 0.0875
        // So D(a_trans) = 2.074 * 0.0875 = 0.1815 = a_trans ✓

        // In gauge era: D_raw = a, D_normalized = a / 2.074
        a / 2.074
    } else {
        // Matter era: interpolate from tabulated values
        // D(z)/D(0) from numerical ODE integration
        let d_table: [(f64, f64); 7] = [
            (0.0, 1.000000),
            (1.0, 0.386367),
            (2.0, 0.179503),
            (3.0, 0.112850),
            (4.0, 0.090886),
            (4.5, 0.087700),  // Just before transition
            (4.51, 0.087514), // At transition
        ];

        // Linear interpolation
        for i in 0..d_table.len()-1 {
            let (z1, d1) = d_table[i];
            let (z2, d2) = d_table[i+1];
            if z >= z1 && z <= z2 {
                let t = (z - z1) / (z2 - z1);
                return d1 + t * (d2 - d1);
            }
        }

        // z beyond table: extrapolate from last two points
        let (z1, d1) = d_table[d_table.len()-2];
        let (z2, d2) = d_table[d_table.len()-1];
        let slope = (d2 - d1) / (z2 - z1);
        d2 + slope * (z - z2)
    }
}

/// Scaling factor for Zel'dovich ICs from z_calibration to z_init
/// ψ(z_init) = ψ(z_calib) × D(z_init) / D(z_calib)
fn ic_scaling_factor(z_init: f64) -> f64 {
    let d_init = growth_factor_janus(z_init);
    let d_calib = growth_factor_janus(Z_CALIBRATION);
    d_init / d_calib
}

// ═══════════════════════════════════════════════════════════════════════════
// PARTICLE MASSES — Option A (Petit 2024): equal mass per particle, N-/N+ = μ
// Each particle represents a fluid element of identical mass
// Asymmetry comes from NUMBER of particles, not mass per particle
// ═══════════════════════════════════════════════════════════════════════════
fn compute_particle_masses(n_total: usize, l_box: f64, h0: f64, omega_b: f64, mu: f64) -> (f64, f64) {
    let h = h0 / 100.0;
    let rho_crit = 2.775e11 * h * h;  // M☉/Mpc³
    let m_plus_total = omega_b * rho_crit * l_box.powi(3);
    let m_minus_total = mu * m_plus_total;

    // Option A: N+ = N_total/(1+μ), N- = N_total×μ/(1+μ)
    let n_plus = (n_total as f64 / (1.0 + mu)) as usize;
    let n_minus = n_total - n_plus;

    // Equal mass per particle (m_plus ≈ m_minus)
    let m_plus = m_plus_total / n_plus as f64;
    let m_minus = m_minus_total / n_minus as f64;

    (m_plus, m_minus)
}

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH ICS
// ═══════════════════════════════════════════════════════════════════════════
fn generate_zeldovich_ics(n_grid: usize, l_box: f64, z_init: f64, h0: f64, mu: f64)
    -> (Vec<f64>, Vec<f64>, Vec<i32>)
{
    println!("[IC] Generating Zel'dovich ICs...");
    println!("  Grid: {}³ = {} particles", n_grid, n_grid.pow(3));

    let mut rng = StdRng::seed_from_u64(SEED_IC);
    let n_particles = n_grid.pow(3);
    let n_fft = n_grid * 2;  // Padding for FFT

    // Generate δ(k) on padded grid
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
                    let amp = (pk * DELTA_RMS).sqrt() * rng.gen::<f64>().sqrt();
                    let phase = rng.gen::<f64>() * 2.0 * PI;
                    let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                    delta_k[idx] = Complex::new(amp * phase.cos(), amp * phase.sin());
                }
            }
        }
    }

    // Compute displacement fields ψ_x, ψ_y, ψ_z
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

    // IFFT
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

    // Find max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n_fft.pow(3) {
        let d = (psi_x[i].re.powi(2) + psi_y[i].re.powi(2) + psi_z[i].re.powi(2)).sqrt();
        max_disp = max_disp.max(d);
    }

    let cell_size = l_box / n_grid as f64;

    // Apply D(z) scaling: ψ(z_init) = ψ(z_calib) × D(z_init)/D(z_calib)
    // At z_calib=4, amplitude was validated at 30% cell
    let d_scaling = ic_scaling_factor(z_init);
    let target_disp_base = 0.30 * cell_size;  // 30% cell at z_calib=4
    let target_disp = target_disp_base * d_scaling;

    let scale = if max_disp > 1e-20 { target_disp / max_disp } else { 0.0 };

    println!("  D(z={:.1})/D(z={:.0}) = {:.4}", z_init, Z_CALIBRATION, d_scaling);
    println!("  Target displacement: {:.2} Mpc ({:.1}% cell, scaled from 30% at z={})",
             target_disp, target_disp / cell_size * 100.0, Z_CALIBRATION as i32);

    // Build particle arrays
    let mut positions = Vec::with_capacity(n_particles * 3);
    let mut velocities = Vec::with_capacity(n_particles * 3);
    let mut signs = Vec::with_capacity(n_particles);

    // Option A (Petit 2024): N-/N+ = μ (asymmetry in NUMBER, not mass per particle)
    let n_plus_target = (n_particles as f64 / (1.0 + mu)) as usize;
    let spacing_fft = l_box / n_fft as f64;
    let offset_x = rng.gen::<f64>() * spacing_fft;
    let offset_y = rng.gen::<f64>() * spacing_fft;
    let offset_z = rng.gen::<f64>() * spacing_fft;

    let h_gyr = h0 / MPC_GYR_TO_KMS;
    let a_init = 1.0 / (1.0 + z_init);
    let vel_scale = a_init * h_gyr * scale;

    let half_box = l_box / 2.0;

    for idx in 0..n_particles {
        // Random position in box
        let x0 = rng.gen::<f64>() * l_box - half_box;
        let y0 = rng.gen::<f64>() * l_box - half_box;
        let z0 = rng.gen::<f64>() * l_box - half_box;

        // Map to FFT grid for displacement lookup
        let gx = (((x0 + half_box + offset_x) / spacing_fft) as usize) % n_fft;
        let gy = (((y0 + half_box + offset_y) / spacing_fft) as usize) % n_fft;
        let gz = (((z0 + half_box + offset_z) / spacing_fft) as usize) % n_fft;
        let grid_idx = gz * n_fft * n_fft + gy * n_fft + gx;

        let dx = psi_x[grid_idx].re * scale;
        let dy = psi_y[grid_idx].re * scale;
        let dz = psi_z[grid_idx].re * scale;

        // Apply displacement with periodic BC
        let mut x = x0 + dx;
        let mut y = y0 + dy;
        let mut z = z0 + dz;

        if x > half_box { x -= l_box; } else if x < -half_box { x += l_box; }
        if y > half_box { y -= l_box; } else if y < -half_box { y += l_box; }
        if z > half_box { z -= l_box; } else if z < -half_box { z += l_box; }

        positions.push(x);
        positions.push(y);
        positions.push(z);

        // Zel'dovich velocity in Mpc/Gyr (code units, NOT km/s)
        // v = a * H * ψ where vel_scale = a * H
        velocities.push(dx * vel_scale / scale);
        velocities.push(dy * vel_scale / scale);
        velocities.push(dz * vel_scale / scale);

        // Random sign assignment
        let sign = if idx < n_plus_target { 1 } else { -1 };
        signs.push(sign);
    }

    // Shuffle to randomize sign distribution
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
// DENSITY COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════
fn compute_densities_split(positions: &[f64], signs: &[i32], n_grid: usize, l_box: f64)
    -> (f64, f64, f64, f64)
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

    let vol_cell = cell.powi(3);
    let rho_max_plus = max_plus / vol_cell;
    let rho_max_minus = max_minus / vol_cell;

    (rho_max_plus, rho_max_minus, max_plus, max_minus)
}

// ═══════════════════════════════════════════════════════════════════════════
// v_rms COMPUTATION (separated by sign) — returns km/s
// Velocities from GPU are in Mpc/Gyr, convert to km/s
// ═══════════════════════════════════════════════════════════════════════════
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

    // RMS in Mpc/Gyr, convert to km/s
    let v_rms_plus_mpc_gyr = if n_plus > 0 { (sum_plus / n_plus as f64).sqrt() } else { 0.0 };
    let v_rms_minus_mpc_gyr = if n_minus > 0 { (sum_minus / n_minus as f64).sqrt() } else { 0.0 };

    // Convert Mpc/Gyr to km/s: 1 Mpc/Gyr = 977.8 km/s
    (v_rms_plus_mpc_gyr * MPC_GYR_TO_KMS, v_rms_minus_mpc_gyr * MPC_GYR_TO_KMS)
}

// ═══════════════════════════════════════════════════════════════════════════
// LOCAL OVERDENSITIES FOR COOLING
// ═══════════════════════════════════════════════════════════════════════════
fn compute_local_overdensities(positions: &[f64], signs: &[i32], n_grid: usize, l_box: f64) -> Vec<f64> {
    let n = signs.len();
    let cell = l_box / n_grid as f64;
    let half_box = l_box / 2.0;

    let mut grid = vec![0u32; n_grid.pow(3)];
    let mut particle_cells = vec![0usize; n];

    for i in 0..n {
        let x = positions[i * 3] + half_box;
        let y = positions[i * 3 + 1] + half_box;
        let z = positions[i * 3 + 2] + half_box;

        let ix = ((x / cell) as usize).min(n_grid - 1);
        let iy = ((y / cell) as usize).min(n_grid - 1);
        let iz = ((z / cell) as usize).min(n_grid - 1);
        let idx = iz * n_grid * n_grid + iy * n_grid + ix;

        grid[idx] += 1;
        particle_cells[i] = idx;
    }

    let mean_count = n as f64 / n_grid.pow(3) as f64;

    particle_cells.iter()
        .map(|&idx| grid[idx] as f64 / mean_count)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn main() {
    let out_dir = "/app/output/janus_jpp_production";
    let run_label = "janus_jpp_mu19";

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  JANUS JPP PRODUCTION — μ=19 CANONICAL                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Production μ=19 canonique JPP lancée                                    ║");
    println!("║  H(z) Janus 2014+2018 | Φ cube | E tracking corrigé                      ║");
    println!("║  ETA ~37h                                                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  N = {}³ ≈ 10M particles", N_GRID);
    println!("║  Box = {} Mpc, z = {} → {}", L_BOX, Z_INIT, Z_FINAL);
    println!("║  μ = {}, η = {}, H₀ = {} km/s/Mpc", MU, ETA, H0);
    println!("║  dt = {} Gyr, θ = {}", DT, THETA);
    println!("║  VSL δ = 0.0431, Φ = (ā/a)³ coupling", );
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Metrics: every {} steps → evolution_phase2.csv", METRIC_INTERVAL);
    println!("║  Snapshots: every {} steps", SNAPSHOT_INTERVAL);
    println!("║  Frames: every {} steps → frames/", FRAME_INTERVAL);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", out_dir)).expect("Failed to create snapshots dir");
    fs::create_dir_all(format!("{}/frames", out_dir)).expect("Failed to create frames dir");

    // Generate ICs
    let (positions, velocities, signs) = generate_zeldovich_ics(N_GRID, L_BOX, Z_INIT, H0, MU);
    let n_particles = signs.len();
    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = n_particles - n_plus;

    let (m_plus, m_minus) = compute_particle_masses(n_particles, L_BOX, H0, OMEGA_B, MU);
    println!("[MASS] m+ = {:.4e} M☉, m- = {:.4e} M☉", m_plus, m_minus);

    // Energy tracking setup
    let vol = L_BOX * L_BOX * L_BOX;
    let m_plus_total = n_plus as f64 * m_plus;
    let m_minus_total = n_minus as f64 * m_minus;
    let rho_plus_comoving = m_plus_total / vol;
    let rho_minus_comoving = -m_minus_total / vol;  // Negative!

    println!("[ENERGY] ρ⁺_comoving = {:.4e} M☉/Mpc³", rho_plus_comoving);
    println!("[ENERGY] ρ⁻_comoving = {:.4e} M☉/Mpc³", rho_minus_comoving);

    // Initialize GPU
    println!("\n[GPU] Compiling CUDA kernels...");
    let cuda_device = GpuNBodySimulation::compile_kernels()
        .expect("Failed to compile CUDA kernels");
    println!("  ✓ CUDA kernels compiled");

    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_BOX,
        positions.clone(), velocities.clone(), signs.clone()
    ).expect("Failed to create GPU simulation");
    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(EPS_PLUS);

    // Janus mass factor
    let janus_mass_factor = OMEGA_B * (1.0 + MU) / 0.3;
    gpu_sim.set_mass_factor(janus_mass_factor);
    println!("  [MASS] Factor = {:.4}", janus_mass_factor);

    // Initialize cooling for m+ only
    let mut gpu_cooling = GpuCooling::new(
        cuda_device.clone(),
        n_plus,
        L_BOX,
        m_plus,
    ).expect("Failed to create GpuCooling");

    let signs_plus: Vec<i32> = vec![1i32; n_plus];
    gpu_cooling.init_from_temperature(T_INIT_PLUS, T_INIT_PLUS, &signs_plus)
        .expect("Failed to init cooling");
    println!("  ✓ Baryonic physics initialized (T_init = {} K)", T_INIT_PLUS);

    // CSV output
    let csv_path = format!("{}/evolution_phase2.csv", out_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,z,t_Gyr,a_plus,a_minus,c_bar,rho_max_plus,rho_max_minus,v_rms_plus,v_rms_minus,ratio_v,phi,E_naive,E_plus,E_minus,E_naive_drift_pct,S_VSL,E_VSL,E_VSL_drift_pct,N_stars,SFR").unwrap();

    // Run log
    let log_path = format!("{}/run.log", out_dir);
    let mut log = BufWriter::new(File::create(&log_path).unwrap());
    writeln!(log, "JANUS JPP PRODUCTION — μ=19 CANONICAL").unwrap();
    writeln!(log, "Started: {:?}", std::time::SystemTime::now()).unwrap();

    // State variables
    let mut a = 1.0 / (1.0 + Z_INIT);
    let mut t_gyr = 0.0;  // Start at t=0 for z=10
    let mut e_total_0: Option<f64> = None;
    let mut e_vsl_0: Option<f64> = None;
    let mut n_stars: u64 = 0;
    let mut sfr: f64 = 0.0;

    // VSL energy tracking: c̄²_init for S_VSL computation
    let c_bar_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA);
    println!("[VSL] c̄²(z_init={}) = {:.8}", Z_INIT, c_bar_sq_init);

    println!("\n[RUN] Starting main loop (z={} → z={})...\n", Z_INIT, Z_FINAL);

    let mut step = 0;
    loop {
        let z = 1.0 / a - 1.0;

        if z < Z_FINAL {
            println!("\n  ✓ Reached z_final = {:.2} at step {}", Z_FINAL, step);
            break;
        }

        // H(a) Janus 2014+2018
        let h = compute_hubble_janus(a, H0);

        // VSL c̄(z)
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_bar = c_ratio_sq.sqrt();

        // Phi coupling
        let a_minus = a_minus_from_a_plus(a, ETA);
        let (phi, phi_inv) = compute_phi_factors(a, ETA);

        // Identity check: c̄²(z) = a⁺/a⁻ (Petit 2014)
        debug_assert!(
            (c_ratio_sq - a / a_minus).abs() < 1e-9,
            "Identity c̄² = a⁺/a⁻ violated at z={}: c̄²={}, a⁺/a⁻={}",
            z, c_ratio_sq, a / a_minus
        );

        // Update GPU simulation parameters
        if step % 10 == 0 {
            gpu_sim.set_c_ratio(c_bar);
            gpu_sim.set_phi(phi, phi_inv);
        }

        let do_metric = step % METRIC_INTERVAL == 0;
        let do_snapshot = step % SNAPSHOT_INTERVAL == 0;
        let do_frame = step % FRAME_INTERVAL == 0;

        if do_metric || do_snapshot {
            // Sync GPU → CPU
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_gpu = gpu_sim.signs();

            // Densities
            let (rho_max_plus, rho_max_minus, _, _) = compute_densities_split(&pos, &signs_gpu, 64, L_BOX);

            // v_rms split
            let (v_rms_plus, v_rms_minus) = compute_vrms_split(&vel, &signs_gpu);
            let ratio_v = if v_rms_plus > 0.0 { v_rms_minus / v_rms_plus } else { 0.0 };

            // Energy tracking (corrected formula with VSL)
            let c_plus = 1.0;
            let c_bar_sq = c_bar * c_bar;
            let (e_naive, e_plus, e_minus) = compute_total_energy(
                rho_plus_comoving, rho_minus_comoving,
                c_plus, c_bar,
                a, a_minus
            );

            // VSL correction: S_VSL = ρ⁻ × [c̄²(t) - c̄²_init]
            let s_vsl = rho_minus_comoving * (c_bar_sq - c_bar_sq_init);
            // E_VSL = E_naive - S_VSL = ρ⁺×c² + ρ⁻×c̄²_init (conserved)
            let e_vsl = e_naive - s_vsl;

            if e_total_0.is_none() {
                e_total_0 = Some(e_naive);
                e_vsl_0 = Some(e_vsl);
                println!("  📊 Initial E_naive = {:.6e}", e_naive);
                println!("  📊 Initial E_VSL   = {:.6e} (conserved quantity)", e_vsl);
            }
            let e_naive_drift = energy_drift_pct(e_naive, e_total_0.unwrap());
            let e_vsl_drift = energy_drift_pct(e_vsl, e_vsl_0.unwrap());

            // Auto-monitoring warnings (use E_VSL for real conservation check)
            if e_vsl_drift.abs() > 2.0 {
                println!("  ⚠ E_VSL DRIFT: {:.2}% at step {}", e_vsl_drift, step);
                writeln!(log, "⚠ E_VSL DRIFT: {:.2}% at step {}, z={:.4}", e_vsl_drift, step, z).unwrap();
            }
            if ratio_v > 1.5 {
                println!("  ⚠ M- DRIFT: ratio_v = {:.2} at step {}", ratio_v, step);
                writeln!(log, "⚠ M- DRIFT: ratio_v = {:.2} at step {}, z={:.4}", ratio_v, step, z).unwrap();
            }
            if v_rms_minus > 50000.0 {
                println!("  ⚠ M- RUNAWAY: v_rms- = {:.0} km/s at step {}", v_rms_minus, step);
                writeln!(log, "⚠ M- RUNAWAY: v_rms- = {:.0} km/s at step {}, z={:.4}", v_rms_minus, step, z).unwrap();
            }

            if do_metric {
                writeln!(csv, "{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.4e},{:.4e},{:.2},{:.2},{:.4},{:.8},{:.6e},{:.6e},{:.6e},{:.6},{:.6e},{:.6e},{:.6},{},{:.4e}",
                    step, z, t_gyr, a, a_minus, c_bar,
                    rho_max_plus, rho_max_minus,
                    v_rms_plus, v_rms_minus, ratio_v,
                    phi, e_naive, e_plus, e_minus, e_naive_drift,
                    s_vsl, e_vsl, e_vsl_drift,
                    n_stars, sfr
                ).unwrap();

                if step % 100 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let steps_per_sec = if elapsed > 0.0 { step as f64 / elapsed } else { 0.0 };
                    println!("  Step {:6} | z={:.4} | v±={:.0}/{:.0} km/s | φ={:.4} | E_VSL_drift={:.3}% | {:.1} step/s",
                        step, z, v_rms_plus, v_rms_minus, phi, e_vsl_drift, steps_per_sec);
                }
            }

            // Snapshot (proper v3 binary format)
            if do_snapshot {
                let snap_path = format!("{}/snapshots/snap_{:06}.bin", out_dir, step);
                let path = Path::new(&snap_path);

                // Get particle data from GPU
                let pos = gpu_sim.get_positions().unwrap();
                let vel = gpu_sim.get_velocities().unwrap();
                let signs_data = gpu_sim.signs();

                // Build ParticleV3 array
                let mut particles: Vec<ParticleV3> = Vec::with_capacity(n_particles);
                for i in 0..n_particles {
                    let px = pos[i * 3] as f32;
                    let py = pos[i * 3 + 1] as f32;
                    let pz = pos[i * 3 + 2] as f32;
                    // Convert velocity from Mpc/Gyr to km/s for storage
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

                // Build header
                let mut header = SnapshotHeaderV3::new("janus_jpp_production");
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
                header.n_stars = n_stars;
                header.sfr = sfr;

                write_snapshot_v3(path, &header, &particles).expect("Failed to write snapshot");

                let snap_size_mb = (408 + n_particles * 36) / (1024 * 1024);
                println!("    📸 Snapshot {} saved ({} MB)", step, snap_size_mb);
            }

            csv.flush().unwrap();
            log.flush().unwrap();
        }

        // Time integration with Hubble friction
        gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, 1.0).unwrap();
        let da = a * h * DT;
        a += da;
        t_gyr += DT;

        // Baryonic physics (m+ only)
        if step % 5 == 0 {
            let pos = gpu_sim.get_positions().unwrap();
            let signs_data = gpu_sim.signs();

            let overdensities = compute_local_overdensities(&pos, &signs_data, 32, L_BOX);
            let rho_crit_0 = 2.775e11 * (H0 / 100.0).powi(2);
            let rho_mean_b_z = OMEGA_B * rho_crit_0 * (1.0 + z).powi(3);

            // Filter to only m+ particles (cooling module expects only m+ densities)
            let densities_plus: Vec<f64> = overdensities.iter()
                .enumerate()
                .filter(|(i, _)| signs_data[*i] > 0)
                .map(|(_, &od)| od * rho_mean_b_z)
                .collect();

            gpu_cooling.upload_densities(&densities_plus).ok();
            gpu_cooling.apply_cooling(DT * 5.0, z).ok();

            if let Ok(new_stars) = gpu_cooling.apply_star_formation(DT * 5.0) {
                if new_stars > 0 {
                    n_stars += new_stars as u64;
                    sfr = (new_stars as f64) * m_plus / (DT * 5.0);
                }
            }
        }

        step += 1;

        // Checkpoint reports
        if step == 100 {
            println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
            println!("║  CHECKPOINT: Step 100 (Initial Validation)                               ║");
            println!("╠══════════════════════════════════════════════════════════════════════════╣");
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_gpu = gpu_sim.signs();
            let (v_rms_plus, v_rms_minus) = compute_vrms_split(&vel, &signs_gpu);
            let c_bar_ck = CoupledFriedmann::c_ratio_sq_at_z(z, ETA).sqrt();
            let c_bar_sq_ck = c_bar_ck * c_bar_ck;
            let (e_naive_ck, _, _) = compute_total_energy(rho_plus_comoving, rho_minus_comoving, 1.0, c_bar_ck, a, a_minus);
            let s_vsl_ck = rho_minus_comoving * (c_bar_sq_ck - c_bar_sq_init);
            let e_vsl_ck = e_naive_ck - s_vsl_ck;
            let e_vsl_drift_ck = energy_drift_pct(e_vsl_ck, e_vsl_0.unwrap());
            println!("║  z = {:.4}, step = {}", z, step);
            println!("║  v_rms+ = {:.1} km/s, v_rms- = {:.1} km/s", v_rms_plus, v_rms_minus);
            println!("║  φ = {:.6}, c̄ = {:.6}", phi, c_bar_ck);
            println!("║  E_VSL_drift = {:.4}%", e_vsl_drift_ck);
            println!("║  N_stars = {}", n_stars);
            println!("╚══════════════════════════════════════════════════════════════════════════╝\n");
        }

        if step == 5000 {
            println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
            println!("║  CHECKPOINT: Step 5000 (Mid-Run Validation)                              ║");
            println!("╠══════════════════════════════════════════════════════════════════════════╣");
            let vel = gpu_sim.get_velocities().unwrap();
            let signs_gpu = gpu_sim.signs();
            let (v_rms_plus, v_rms_minus) = compute_vrms_split(&vel, &signs_gpu);
            let c_bar_ck = CoupledFriedmann::c_ratio_sq_at_z(z, ETA).sqrt();
            let c_bar_sq_ck = c_bar_ck * c_bar_ck;
            let (e_naive_ck, _, _) = compute_total_energy(rho_plus_comoving, rho_minus_comoving, 1.0, c_bar_ck, a, a_minus);
            let s_vsl_ck = rho_minus_comoving * (c_bar_sq_ck - c_bar_sq_init);
            let e_vsl_ck = e_naive_ck - s_vsl_ck;
            let e_vsl_drift_ck = energy_drift_pct(e_vsl_ck, e_vsl_0.unwrap());
            let elapsed = start_time.elapsed().as_secs_f64() / 3600.0;
            println!("║  z = {:.4}, step = {}", z, step);
            println!("║  v_rms+ = {:.1} km/s, v_rms- = {:.1} km/s", v_rms_plus, v_rms_minus);
            println!("║  φ = {:.6}, c̄ = {:.6}", phi, c_bar_ck);
            println!("║  E_VSL_drift = {:.4}%", e_vsl_drift_ck);
            println!("║  N_stars = {}", n_stars);
            println!("║  Elapsed: {:.2}h", elapsed);
            println!("╚══════════════════════════════════════════════════════════════════════════╝\n");
        }
    }

    // Final summary
    let total_time = start_time.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  RUN COMPLETE                                                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Total time: {:.2} hours ({:.0} s)", total_time / 3600.0, total_time);
    println!("║  Final step: {}", step);
    println!("║  Final z: {:.4}", 1.0 / a - 1.0);
    println!("║  N_stars: {}", n_stars);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    writeln!(log, "\nRUN COMPLETE").unwrap();
    writeln!(log, "Total time: {:.2} hours", total_time / 3600.0).unwrap();
    writeln!(log, "Final step: {}", step).unwrap();
    log.flush().unwrap();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
    std::process::exit(1);
}
