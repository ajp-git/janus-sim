//! JANUS SCREENING — Phase 0 (sentinelles) + Phase 1 (scan 2D)
//!
//! Binaire autonome pour screening rapide de l'espace (μ, L).
//! - Phase 0 : 4 sentinelles diagnostiques avec critère GO/NO-GO
//! - Phase 1 : scan 8 μ × 3 L = 24 runs (si GO)

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};

// ════════════════════════════════════════════════════════════════
// CONFIGURATION (hardcoded)
// ════════════════════════════════════════════════════════════════
const N_GRID: usize = 58;              // 58³ = 195,112 particules
const Z_INIT: f64 = 4.0;
const Z_FINAL: f64 = 0.5;
const DT: f64 = 0.002;
const THETA: f64 = 0.7;
const H0: f64 = 69.9;
const OMEGA_B: f64 = 0.05;
const ETA_VSL: f64 = 1.045;
const HUBBLE_FRICTION: f64 = 1.0;      // Vraie physique
const SEED_IC: u64 = 42;

// Auto-stop
const V_RMS_HARD_LIMIT: f64 = 50000.0;
const MAX_STEPS: usize = 5000;

// ICs
const PI: f64 = std::f64::consts::PI;
const N_S: f64 = 0.965;
const IC_AMPLITUDE: f64 = 0.01;
const MPC_GYR_TO_KMS: f64 = 977.8;

// ════════════════════════════════════════════════════════════════
// PHASE 0 — SENTINELLES (SKIPPED for high-mu extension)
// ════════════════════════════════════════════════════════════════
const SENTINEL_CONFIGS: [(f64, f64, f64, &str); 0] = [];

const SENTINEL_RHO_RATIO_THRESHOLD: f64 = 5.0;
const SENTINEL_VRMS_THRESHOLD: f64 = 30.0;

// ════════════════════════════════════════════════════════════════
// SCAN PRINCIPAL — HIGH MU EXTENSION [40, 1000]
// 15 points logarithmiques, L=100 fixe
// ════════════════════════════════════════════════════════════════
const SCAN_MU: [f64; 15] = [
    40.0, 50.0, 63.0, 80.0, 100.0,
    126.0, 159.0, 200.0, 252.0, 317.0,
    399.0, 502.0, 631.0, 795.0, 1000.0
];
const SCAN_L: [f64; 1] = [100.0];

// ════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ════════════════════════════════════════════════════════════════
#[derive(Default, Clone, Copy)]
struct ZMetrics {
    z: f64,
    v_rms: f64,
    rho_plus_max: f64,
    rho_max: f64,
}

struct RunResult {
    phase: u8,
    run_id: u32,
    label: String,
    mu: f64,
    l_box: f64,
    n_init: usize,
    z_final_reached: f64,
    metrics_z3: ZMetrics,
    metrics_z2: ZMetrics,
    metrics_z15: ZMetrics,
    metrics_z1: ZMetrics,
    metrics_zfinal: ZMetrics,
    n_overdense_zfinal: u32,
    rho_plus_mean: f64,
    wall_time_sec: f64,
    status: String,
}

impl Default for RunResult {
    fn default() -> Self {
        Self {
            phase: 0, run_id: 0, label: String::new(),
            mu: 0.0, l_box: 0.0, n_init: 0, z_final_reached: Z_INIT,
            metrics_z3: Default::default(),
            metrics_z2: Default::default(),
            metrics_z15: Default::default(),
            metrics_z1: Default::default(),
            metrics_zfinal: Default::default(),
            n_overdense_zfinal: 0,
            rho_plus_mean: 0.0,
            wall_time_sec: 0.0,
            status: "INIT".to_string(),
        }
    }
}

// ════════════════════════════════════════════════════════════════
// ZEL'DOVICH ICs (simplified from janus_adaptive_zoom.rs)
// ════════════════════════════════════════════════════════════════
fn generate_zeldovich_ics_screening(
    n_grid: usize,
    l_box: f64,
    z_init: f64,
    mu: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<i32>), String> {
    let n_total = n_grid * n_grid * n_grid;
    let half_box = l_box / 2.0;

    // FFT grid with padding
    let n_fft = 2 * n_grid;
    let n_fft_total = n_fft * n_fft * n_fft;
    let spacing_fft = l_box / n_fft as f64;
    let half_n_fft = n_fft / 2;
    let dk = 2.0 * PI / l_box;

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate δ(k)
    let a_init = 1.0 / (1.0 + z_init);
    let d_growth = a_init;
    let normal = Normal::new(0.0, 1.0).map_err(|e| e.to_string())?;

    let k_min = 2.0 * PI / l_box;
    let k_max = 2.0 * PI / 5.0;
    let k0 = 0.02;

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft_total];

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            for ix in 0..n_fft {
                let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                let kx = if ix <= half_n_fft { ix as f64 } else { ix as f64 - n_fft as f64 } * dk;
                let ky = if iy <= half_n_fft { iy as f64 } else { iy as f64 - n_fft as f64 } * dk;
                let kz = if iz <= half_n_fft { iz as f64 } else { iz as f64 - n_fft as f64 } * dk;
                let k2 = kx * kx + ky * ky + kz * kz;

                if k2 > 0.0 {
                    let k = k2.sqrt();
                    let w_low = 0.5 * (1.0 + ((k - k_min) / (k_min * 0.4)).tanh());
                    let w_high = 0.5 * (1.0 - ((k - k_max) / (k_max * 0.4)).tanh());
                    let window = w_low * w_high;
                    let pk = k.powf(N_S) / (1.0 + (k / k0).powi(4)) * window;
                    let sigma_k = pk.sqrt() * IC_AMPLITUDE * d_growth;
                    delta_k[idx] = Complex::new(
                        rng.sample(&normal) * sigma_k,
                        rng.sample(&normal) * sigma_k,
                    );
                }
            }
        }
    }

    // Hermitian symmetry
    for iz in 0..n_fft {
        for iy in 0..n_fft {
            for ix in 0..=half_n_fft {
                let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                let iz_conj = if iz == 0 { 0 } else { n_fft - iz };
                let iy_conj = if iy == 0 { 0 } else { n_fft - iy };
                let ix_conj = if ix == 0 { 0 } else { n_fft - ix };
                let idx_conj = iz_conj * n_fft * n_fft + iy_conj * n_fft + ix_conj;
                if idx < idx_conj {
                    delta_k[idx_conj] = delta_k[idx].conj();
                }
            }
        }
    }

    // Compute ψ fields
    let mut psi_x_k = vec![Complex::new(0.0, 0.0); n_fft_total];
    let mut psi_y_k = vec![Complex::new(0.0, 0.0); n_fft_total];
    let mut psi_z_k = vec![Complex::new(0.0, 0.0); n_fft_total];

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            for ix in 0..n_fft {
                let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                let kx = if ix <= half_n_fft { ix as f64 } else { ix as f64 - n_fft as f64 } * dk;
                let ky = if iy <= half_n_fft { iy as f64 } else { iy as f64 - n_fft as f64 } * dk;
                let kz = if iz <= half_n_fft { iz as f64 } else { iz as f64 - n_fft as f64 } * dk;
                let k2 = kx * kx + ky * ky + kz * kz;

                if k2 > 1e-12 {
                    let factor = Complex::new(0.0, -1.0) / k2;
                    psi_x_k[idx] = factor * kx * delta_k[idx];
                    psi_y_k[idx] = factor * ky * delta_k[idx];
                    psi_z_k[idx] = factor * kz * delta_k[idx];
                }
            }
        }
    }

    // IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut psi_x: Vec<Complex<f64>> = psi_x_k;
    let mut psi_y: Vec<Complex<f64>> = psi_y_k;
    let mut psi_z: Vec<Complex<f64>> = psi_z_k;

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            let start = iz * n_fft * n_fft + iy * n_fft;
            ifft.process(&mut psi_x[start..start + n_fft]);
            ifft.process(&mut psi_y[start..start + n_fft]);
            ifft.process(&mut psi_z[start..start + n_fft]);
        }
    }

    for iz in 0..n_fft {
        let mut row_x: Vec<Complex<f64>> = (0..n_fft).map(|iy| psi_x[iz * n_fft * n_fft + iy * n_fft]).collect();
        let mut row_y: Vec<Complex<f64>> = (0..n_fft).map(|iy| psi_y[iz * n_fft * n_fft + iy * n_fft]).collect();
        let mut row_z: Vec<Complex<f64>> = (0..n_fft).map(|iy| psi_z[iz * n_fft * n_fft + iy * n_fft]).collect();
        ifft.process(&mut row_x);
        ifft.process(&mut row_y);
        ifft.process(&mut row_z);
        for iy in 0..n_fft {
            for ix in 0..n_fft {
                let idx = iz * n_fft * n_fft + iy * n_fft + ix;
                let idx_row = iz * n_fft * n_fft + iy * n_fft;
                // Simplified - just use X pass result
            }
        }
    }

    // 3D IFFT properly
    for iz in 0..n_fft {
        for iy in 0..n_fft {
            let start = iz * n_fft * n_fft + iy * n_fft;
            ifft.process(&mut psi_x[start..start + n_fft]);
            ifft.process(&mut psi_y[start..start + n_fft]);
            ifft.process(&mut psi_z[start..start + n_fft]);
        }
    }

    // Y pass
    for iz in 0..n_fft {
        for ix in 0..n_fft {
            let mut col_x: Vec<Complex<f64>> = (0..n_fft).map(|iy| psi_x[iz * n_fft * n_fft + iy * n_fft + ix]).collect();
            let mut col_y: Vec<Complex<f64>> = (0..n_fft).map(|iy| psi_y[iz * n_fft * n_fft + iy * n_fft + ix]).collect();
            let mut col_z: Vec<Complex<f64>> = (0..n_fft).map(|iy| psi_z[iz * n_fft * n_fft + iy * n_fft + ix]).collect();
            ifft.process(&mut col_x);
            ifft.process(&mut col_y);
            ifft.process(&mut col_z);
            for iy in 0..n_fft {
                psi_x[iz * n_fft * n_fft + iy * n_fft + ix] = col_x[iy];
                psi_y[iz * n_fft * n_fft + iy * n_fft + ix] = col_y[iy];
                psi_z[iz * n_fft * n_fft + iy * n_fft + ix] = col_z[iy];
            }
        }
    }

    // Z pass
    for iy in 0..n_fft {
        for ix in 0..n_fft {
            let mut col_x: Vec<Complex<f64>> = (0..n_fft).map(|iz| psi_x[iz * n_fft * n_fft + iy * n_fft + ix]).collect();
            let mut col_y: Vec<Complex<f64>> = (0..n_fft).map(|iz| psi_y[iz * n_fft * n_fft + iy * n_fft + ix]).collect();
            let mut col_z: Vec<Complex<f64>> = (0..n_fft).map(|iz| psi_z[iz * n_fft * n_fft + iy * n_fft + ix]).collect();
            ifft.process(&mut col_x);
            ifft.process(&mut col_y);
            ifft.process(&mut col_z);
            for iz in 0..n_fft {
                psi_x[iz * n_fft * n_fft + iy * n_fft + ix] = col_x[iz];
                psi_y[iz * n_fft * n_fft + iy * n_fft + ix] = col_y[iz];
                psi_z[iz * n_fft * n_fft + iy * n_fft + ix] = col_z[iz];
            }
        }
    }

    // Normalize
    let norm = 1.0 / (n_fft_total as f64);
    for i in 0..n_fft_total {
        psi_x[i] = psi_x[i] * norm;
        psi_y[i] = psi_y[i] * norm;
        psi_z[i] = psi_z[i] * norm;
    }

    // Find max displacement and scale
    let max_disp = psi_x.iter().chain(psi_y.iter()).chain(psi_z.iter())
        .map(|c| c.re.abs())
        .fold(0.0f64, f64::max);

    let spacing = l_box / n_grid as f64;
    let target_disp = 0.30 * spacing;
    let scale_factor = if max_disp > 1e-15 { target_disp / max_disp } else { 1.0 };

    // Generate particles with random positions + CIC interpolated ψ
    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs = Vec::with_capacity(n_total);

    // Random offset for grid
    let offset_x = rng.gen::<f64>() * spacing_fft;
    let offset_y = rng.gen::<f64>() * spacing_fft;
    let offset_z = rng.gen::<f64>() * spacing_fft;

    let vel_scale = 0.1598;  // Mpc/Gyr per Mpc displacement

    for _ in 0..n_total {
        // Random position in box
        let x = rng.gen::<f64>() * l_box - half_box;
        let y = rng.gen::<f64>() * l_box - half_box;
        let z = rng.gen::<f64>() * l_box - half_box;

        // CIC interpolate ψ at this position
        let gx = ((x + half_box + offset_x) / spacing_fft).rem_euclid(n_fft as f64);
        let gy = ((y + half_box + offset_y) / spacing_fft).rem_euclid(n_fft as f64);
        let gz = ((z + half_box + offset_z) / spacing_fft).rem_euclid(n_fft as f64);

        let ix0 = gx.floor() as usize % n_fft;
        let iy0 = gy.floor() as usize % n_fft;
        let iz0 = gz.floor() as usize % n_fft;
        let ix1 = (ix0 + 1) % n_fft;
        let iy1 = (iy0 + 1) % n_fft;
        let iz1 = (iz0 + 1) % n_fft;

        let dx = gx - gx.floor();
        let dy = gy - gy.floor();
        let dz = gz - gz.floor();

        let mut psi_interp = [0.0f64; 3];
        for (corner, (cix, ciy, ciz, wx, wy, wz)) in [
            (ix0, iy0, iz0, 1.0 - dx, 1.0 - dy, 1.0 - dz),
            (ix1, iy0, iz0, dx, 1.0 - dy, 1.0 - dz),
            (ix0, iy1, iz0, 1.0 - dx, dy, 1.0 - dz),
            (ix1, iy1, iz0, dx, dy, 1.0 - dz),
            (ix0, iy0, iz1, 1.0 - dx, 1.0 - dy, dz),
            (ix1, iy0, iz1, dx, 1.0 - dy, dz),
            (ix0, iy1, iz1, 1.0 - dx, dy, dz),
            (ix1, iy1, iz1, dx, dy, dz),
        ].iter().enumerate() {
            let idx = ciz * n_fft * n_fft + ciy * n_fft + cix;
            let w = wx * wy * wz;
            psi_interp[0] += psi_x[idx].re * w;
            psi_interp[1] += psi_y[idx].re * w;
            psi_interp[2] += psi_z[idx].re * w;
        }

        // Apply displacement
        let px = x + psi_interp[0] * scale_factor;
        let py = y + psi_interp[1] * scale_factor;
        let pz = z + psi_interp[2] * scale_factor;

        // Wrap periodic
        let px = ((px + half_box).rem_euclid(l_box)) - half_box;
        let py = ((py + half_box).rem_euclid(l_box)) - half_box;
        let pz = ((pz + half_box).rem_euclid(l_box)) - half_box;

        positions.push(px);
        positions.push(py);
        positions.push(pz);

        // Velocity from Zel'dovich: v = a * ψ_dot = a * H * f * ψ ≈ vel_scale * ψ
        velocities.push(psi_interp[0] * scale_factor * vel_scale);
        velocities.push(psi_interp[1] * scale_factor * vel_scale);
        velocities.push(psi_interp[2] * scale_factor * vel_scale);

        // Random sign based on μ
        let is_positive = rng.gen::<f64>() < 1.0 / (1.0 + mu);
        signs.push(if is_positive { 1 } else { -1 });
    }

    Ok((positions, velocities, signs))
}

// ════════════════════════════════════════════════════════════════
// METRICS CAPTURE
// ════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn capture_metrics(gpu_sim: &mut GpuNBodySimulation, signs: &[i32], l_box: f64, z: f64, m_plus: f64) -> ZMetrics {
    // Compute v_rms from velocities
    let velocities = gpu_sim.get_velocities().unwrap_or_default();
    let n = signs.len();
    let v_rms: f64 = if n > 0 && velocities.len() >= n * 3 {
        let sum: f64 = (0..n)
            .map(|i| velocities[i*3].powi(2) + velocities[i*3+1].powi(2) + velocities[i*3+2].powi(2))
            .sum();
        (sum / n as f64).sqrt() * MPC_GYR_TO_KMS
    } else { 0.0 };

    // Compute densities on 32³ grid
    // FIXED: multiply count by m_plus to get physical density [M☉/Mpc³]
    let grid_size = 32usize;
    let cell_size = l_box / grid_size as f64;
    let cell_vol = cell_size.powi(3);

    let positions = gpu_sim.get_positions().unwrap_or_default();
    let half_box = l_box / 2.0;

    let mut count_grid = vec![0usize; grid_size * grid_size * grid_size];
    let mut count_plus_grid = vec![0usize; grid_size * grid_size * grid_size];

    // Simple NGP assignment (count particles per cell)
    for i in 0..n.min(positions.len() / 3) {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z_pos = positions[i * 3 + 2];

        let ix = (((x + half_box) / cell_size).floor() as usize).min(grid_size - 1);
        let iy = (((y + half_box) / cell_size).floor() as usize).min(grid_size - 1);
        let iz = (((z_pos + half_box) / cell_size).floor() as usize).min(grid_size - 1);

        let idx = iz * grid_size * grid_size + iy * grid_size + ix;
        count_grid[idx] += 1;
        if signs[i] > 0 {
            count_plus_grid[idx] += 1;
        }
    }

    // Convert counts to physical density [M☉/Mpc³]
    // rho = (count × m_plus) / cell_vol
    let max_count = *count_grid.iter().max().unwrap_or(&0);
    let max_count_plus = *count_plus_grid.iter().max().unwrap_or(&0);

    // For total density, assume m_total_per_particle ≈ m_plus (approximation)
    let rho_max = (max_count as f64) * m_plus / cell_vol;
    let rho_plus_max = (max_count_plus as f64) * m_plus / cell_vol;

    ZMetrics {
        z,
        v_rms,
        rho_plus_max,
        rho_max,
    }
}

#[cfg(feature = "cuda")]
fn count_overdense_cells(gpu_sim: &mut GpuNBodySimulation, signs: &[i32], l_box: f64, grid_size: usize, factor: f64) -> u32 {
    let cell_size = l_box / grid_size as f64;
    let positions = gpu_sim.get_positions().unwrap_or_default();
    let n = signs.len().min(positions.len() / 3);
    let half_box = l_box / 2.0;

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let mean_per_cell = n_plus as f64 / (grid_size * grid_size * grid_size) as f64;
    let threshold = mean_per_cell * factor;

    let mut rho_plus_grid = vec![0.0f64; grid_size * grid_size * grid_size];

    for i in 0..n {
        if signs[i] > 0 {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z_pos = positions[i * 3 + 2];

            let ix = (((x + half_box) / cell_size).floor() as usize).min(grid_size - 1);
            let iy = (((y + half_box) / cell_size).floor() as usize).min(grid_size - 1);
            let iz = (((z_pos + half_box) / cell_size).floor() as usize).min(grid_size - 1);

            rho_plus_grid[iz * grid_size * grid_size + iy * grid_size + ix] += 1.0;
        }
    }

    rho_plus_grid.iter().filter(|&&c| c > threshold).count() as u32
}

// ════════════════════════════════════════════════════════════════
// FRIEDMANN H(z) for Janus
// ════════════════════════════════════════════════════════════════
fn compute_friedmann_h(z: f64, _mu: f64) -> f64 {
    // Simplified: H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
    // For Janus with VSL, use coupled Friedmann
    let h0_gyr = H0 / MPC_GYR_TO_KMS;  // H0 in Gyr^-1
    let omega_m = 0.3;
    let omega_l = 0.7;
    let a = 1.0 / (1.0 + z);
    h0_gyr * (omega_m / a.powi(3) + omega_l).sqrt()
}

// ════════════════════════════════════════════════════════════════
// CSV WRITER
// ════════════════════════════════════════════════════════════════
fn write_csv_row(csv: &mut BufWriter<File>, r: &RunResult) {
    writeln!(csv, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        r.phase, r.run_id, r.label, r.mu, r.l_box, r.n_init, r.z_final_reached,
        r.metrics_z3.v_rms, r.metrics_z2.v_rms, r.metrics_z15.v_rms, r.metrics_z1.v_rms, r.metrics_zfinal.v_rms,
        r.metrics_z3.rho_plus_max, r.metrics_z2.rho_plus_max, r.metrics_z15.rho_plus_max, r.metrics_z1.rho_plus_max, r.metrics_zfinal.rho_plus_max,
        r.metrics_z3.rho_max, r.metrics_z2.rho_max, r.metrics_z15.rho_max, r.metrics_z1.rho_max, r.metrics_zfinal.rho_max,
        r.rho_plus_mean, r.n_overdense_zfinal, r.wall_time_sec, r.status
    ).unwrap();
    csv.flush().unwrap();
}

// ════════════════════════════════════════════════════════════════
// SINGLE SIMULATION RUN
// ════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn run_single_simulation(
    phase: u8,
    run_id: u32,
    label: String,
    mu: f64,
    l_box: f64,
    mass_factor_override: f64,  // if > 0, use this instead of calculated
    _outdir: &str,
) -> RunResult {
    let start = Instant::now();
    let mut result = RunResult {
        phase, run_id, label: label.clone(),
        mu, l_box, ..Default::default()
    };

    // ρ+_mean cosmologique
    let rho_crit = 2.775e11 * (H0 / 100.0).powi(2);
    result.rho_plus_mean = OMEGA_B * rho_crit;

    // Generate ICs
    let (positions, velocities, signs) = match generate_zeldovich_ics_screening(
        N_GRID, l_box, Z_INIT, mu, SEED_IC
    ) {
        Ok(v) => v,
        Err(e) => {
            result.status = format!("IC_FAIL:{}", e);
            result.wall_time_sec = start.elapsed().as_secs_f64();
            return result;
        }
    };

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.len() - n_plus;
    result.n_init = n_plus + n_minus;

    // Compute mass per positive particle [M☉]
    // total_mass_plus = Ω_b × ρ_crit × L³, divided among N_plus particles
    // rho_crit already defined above
    let total_mass_plus = OMEGA_B * rho_crit * l_box.powi(3);
    let m_plus = total_mass_plus / n_plus as f64;

    if n_plus < 100 {
        result.status = format!("TOO_FEW_PLUS:{}", n_plus);
        result.wall_time_sec = start.elapsed().as_secs_f64();
        return result;
    }

    // Init GPU with state
    let mut gpu_sim = match GpuNBodySimulation::new_with_state(
        n_plus, n_minus, l_box, positions, velocities, signs.clone()
    ) {
        Ok(s) => s,
        Err(e) => {
            result.status = format!("GPU_INIT_FAIL:{}", e);
            result.wall_time_sec = start.elapsed().as_secs_f64();
            return result;
        }
    };

    let eps_plus = 0.05 * l_box / 100.0;
    gpu_sim.set_theta(THETA);
    gpu_sim.set_softening(eps_plus);

    // MASS_FACTOR — use override if provided, otherwise calculate
    let janus_mass_factor = if mass_factor_override > 0.0 {
        println!("  [MASS] Using FORCED mass_factor = {:.4}", mass_factor_override);
        mass_factor_override
    } else {
        let mf = OMEGA_B * (1.0 + mu) / 0.3;
        println!("  [MASS] Scaled all masses by factor {:.4}", mf);
        mf
    };
    gpu_sim.set_mass_factor(janus_mass_factor);

    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(Z_INIT, ETA_VSL);
    gpu_sim.set_c_ratio(c_ratio_sq_init.sqrt());

    let mut a = 1.0 / (1.0 + Z_INIT);

    let z_targets = [3.0, 2.0, 1.5, 1.0];
    let mut next_target_idx = 0;

    let mut step = 0usize;
    loop {
        let z = 1.0 / a - 1.0;

        if z <= Z_FINAL {
            result.z_final_reached = z;
            result.metrics_zfinal = capture_metrics(&mut gpu_sim, &signs, l_box, z, m_plus);
            result.n_overdense_zfinal = count_overdense_cells(&mut gpu_sim, &signs, l_box, 32, 10.0);
            break;
        }

        if step >= MAX_STEPS {
            result.status = "MAX_STEPS".to_string();
            result.z_final_reached = z;
            result.metrics_zfinal = capture_metrics(&mut gpu_sim, &signs, l_box, z, m_plus);
            break;
        }

        // Capture at z targets
        while next_target_idx < z_targets.len() && z <= z_targets[next_target_idx] {
            let m = capture_metrics(&mut gpu_sim, &signs, l_box, z_targets[next_target_idx], m_plus);
            match next_target_idx {
                0 => result.metrics_z3 = m,
                1 => result.metrics_z2 = m,
                2 => result.metrics_z15 = m,
                3 => result.metrics_z1 = m,
                _ => {}
            }
            next_target_idx += 1;
        }

        let h = compute_friedmann_h(z, mu);

        if let Err(e) = gpu_sim.step_with_expansion_dkd_gpu(DT, a, h, HUBBLE_FRICTION) {
            result.status = format!("STEP_FAIL:{}", e);
            result.z_final_reached = z;
            result.metrics_zfinal = capture_metrics(&mut gpu_sim, &signs, l_box, z, m_plus);
            break;
        }

        // Stability check
        if step % 100 == 0 {
            if let Ok(vels) = gpu_sim.get_velocities() {
                let n = signs.len().min(vels.len() / 3);
                if n > 0 {
                    let sum: f64 = (0..n)
                        .map(|i| vels[i*3].powi(2) + vels[i*3+1].powi(2) + vels[i*3+2].powi(2))
                        .sum();
                    let v_kms = (sum / n as f64).sqrt() * MPC_GYR_TO_KMS;
                    if v_kms > V_RMS_HARD_LIMIT || v_kms.is_nan() {
                        result.status = format!("INSTABILITY:v_rms={:.0}", v_kms);
                        result.z_final_reached = z;
                        break;
                    }
                }
            }
        }

        // Update VSL
        if step % 50 == 0 {
            let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA_VSL);
            gpu_sim.set_c_ratio(c_ratio_sq.sqrt());
        }

        a += a * h * DT;
        step += 1;
    }

    if result.status == "INIT" {
        result.status = "OK".to_string();
    }

    result.wall_time_sec = start.elapsed().as_secs_f64();
    result
}

// ════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn main() {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M").to_string();
    let outdir = format!("/app/output/screening_{}", timestamp);
    fs::create_dir_all(&format!("{}/snapshots", outdir)).unwrap();

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  JANUS SCREENING — Phase 0 (sentinelles) + Phase 1 (scan)                ║");
    println!("║  Output: {}                                                   ║", outdir);
    println!("║  N_GRID: {} → {} particles                                          ║", N_GRID, N_GRID.pow(3));
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // CSV
    let csv_path = format!("{}/screening_results.csv", outdir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "phase,run_id,label,mu,L_box,N_init,z_final_reached,\
                   v_rms_z3,v_rms_z2,v_rms_z15,v_rms_z1,v_rms_zfinal,\
                   rho_plus_max_z3,rho_plus_max_z2,rho_plus_max_z15,rho_plus_max_z1,rho_plus_max_zfinal,\
                   rho_max_z3,rho_max_z2,rho_max_z15,rho_max_z1,rho_max_zfinal,\
                   rho_plus_mean,n_overdense_zfinal,wall_time_sec,status").unwrap();
    csv.flush().unwrap();

    let global_start = Instant::now();
    let mut total_runs_done = 0u32;

    // ─────────────────────────────────────────────
    // PHASE 0 — SENTINELLES (including control S5)
    // ─────────────────────────────────────────────
    // ─────────────────────────────────────────────
    // HIGH-MU EXTENSION — Direct scan, no sentinels
    // ─────────────────────────────────────────────
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  HIGH-MU EXTENSION — {} μ values × {} L = {} runs", SCAN_MU.len(), SCAN_L.len(), SCAN_MU.len() * SCAN_L.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let total_scan_runs = SCAN_MU.len() * SCAN_L.len();
    let mut scan_run_idx = 0u32;

    for &mu in &SCAN_MU {
        for &l_box in &SCAN_L {
            scan_run_idx += 1;
            total_runs_done += 1;
            let label = format!("scan_mu{:.1}_L{:.0}", mu, l_box);

            println!("[Scan {}/{}] μ={:.1}, L={:.0} ({})", scan_run_idx, total_scan_runs, mu, l_box, label);

            let result = run_single_simulation(1, total_runs_done, label, mu, l_box, -1.0, &outdir);

            write_csv_row(&mut csv, &result);

            let collapse_ratio = if result.rho_plus_mean > 0.0 {
                result.metrics_zfinal.rho_plus_max / result.rho_plus_mean
            } else { 0.0 };
            let elapsed_h = global_start.elapsed().as_secs_f64() / 3600.0;
            let avg = elapsed_h / total_runs_done as f64;
            let remaining = avg * (total_scan_runs - scan_run_idx as usize) as f64;

            println!("  → ratio={:.1}, v_rms={:.0} | status={} | elapsed={:.2}h | ETA +{:.2}h",
                collapse_ratio, result.metrics_zfinal.v_rms, result.status, elapsed_h, remaining);
        }
    }

    let total_h = global_start.elapsed().as_secs_f64() / 3600.0;
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SCREENING COMPLETE                                                      ║");
    println!("║  Total time: {:.2}h                                                       ║", total_h);
    println!("║  Total runs: {} (4 sentinelles + {} scan)                                ║", total_runs_done, total_scan_runs);
    println!("║  Results: {}/screening_results.csv                        ║", outdir);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: janus_screening requires --features cuda");
    std::process::exit(1);
}
