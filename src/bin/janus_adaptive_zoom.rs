//! JANUS ADAPTIVE ZOOM — Production Run with Adaptive Splitting
//!
//! Based on janus_baryonic_calibrated.rs with:
//! - z_init = 10.0 (ICs generated at z=10)
//! - Snapshot format v3 (auto-descriptive)
//! - Adaptive particle splitting every 50 steps
//! - Dynamic zoom rendering on high-resolution regions
//!
//! Split logic:
//!   For each m+ with ρ_sph > DELTA_SPLIT[split_level]:
//!     Create 8 daughters (Blue Noise placement, radius = h_sph/3)
//!     mass_daughter = mass/8, epsilon_daughter = epsilon/2
//!     split_level_daughter = split_level + 1

use clap::Parser;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use janus::snapshot_v3::{SnapshotHeaderV3, ParticleV3, write_snapshot_v3, snapshot_info};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};

// ═══════════════════════════════════════════════════════════════════════════
// CLI ARGUMENTS
// ═══════════════════════════════════════════════════════════════════════════
#[derive(Parser, Debug)]
#[command(name = "janus_adaptive_zoom")]
#[command(about = "Janus adaptive zoom simulation with particle splitting")]
struct Args {
    /// Grid size for ICs (n_grid³ particles)
    #[arg(long, default_value = "126")]
    n_grid: usize,

    /// Initial redshift
    #[arg(long, default_value = "10.0")]
    z_init: f64,

    /// Final redshift (stop when z < z_final)
    #[arg(long, default_value = "0.0")]
    z_final: f64,

    /// Split check interval (steps)
    #[arg(long, default_value = "50")]
    steps_check: usize,

    /// Snapshot interval (steps)
    #[arg(long, default_value = "100")]
    snap_interval: usize,

    /// Output directory
    #[arg(long, default_value = "/app/output/janus_adaptive_zoom")]
    out_dir: String,

    /// First split threshold (M_sun/Mpc³)
    #[arg(long, default_value = "1000.0")]
    delta_split_0: f64,

    /// Box size (Mpc)
    #[arg(long, default_value = "100.0")]
    l_box: f64,

    /// Hubble constant (km/s/Mpc)
    #[arg(long, default_value = "69.9")]
    h0: f64,

    /// Mass ratio m-/m+ (μ)
    #[arg(long, default_value = "19.0")]
    mu: f64,

    /// Baryon density parameter
    #[arg(long, default_value = "0.05")]
    omega_b: f64,

    /// Softening length for m+ (Mpc)
    #[arg(long, default_value = "0.05")]
    eps_plus: f64,

    /// Softening length for m- (Mpc)
    #[arg(long, default_value = "0.10")]
    eps_minus: f64,

    /// Maximum time step (Gyr)
    #[arg(long, default_value = "0.001")]
    dt_max: f64,

    /// Minimum time step (Gyr) - for adaptive timestep
    #[arg(long, default_value = "0.0002")]
    dt_min: f64,

    /// Timestep accuracy parameter (η) for adaptive dt
    #[arg(long, default_value = "0.025")]
    eta: f64,

    /// Run label for snapshots
    #[arg(long, default_value = "janus_adaptive_zoom")]
    run_label: String,

    /// Barnes-Hut opening angle θ
    #[arg(long, default_value = "0.7")]
    theta: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// COSMOLOGY (from CLI, except η which is fixed from Pantheon+ fit)
// ═══════════════════════════════════════════════════════════════════════════
const ETA: f64 = 1.045;         // Mass ratio (from Pantheon+ fit)

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const N_MAX_TOTAL: usize = 12_000_000;      // VRAM limit: ~10-12M on RTX 3060 12GB
const METRIC_INTERVAL: usize = 10;

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH ICs
// ═══════════════════════════════════════════════════════════════════════════
const SEED_IC: u64 = 42;
const N_S: f64 = 0.965;
const DELTA_RMS: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;
const G_COSMO: f64 = 4.499e-15;  // Mpc³ M☉⁻¹ Gyr⁻²

// ═══════════════════════════════════════════════════════════════════════════
// PARTICLE MASSES (M_sun) — computed from cosmology
// ═══════════════════════════════════════════════════════════════════════════
fn compute_particle_masses(n_total: usize, l_box: f64, h0: f64, omega_b: f64, mu: f64) -> (f64, f64) {
    // Cosmological mass calculation
    let rho_crit = 2.775e11 * (h0 / 100.0).powi(2);  // M☉/Mpc³
    let rho_mean_plus = omega_b * rho_crit;

    // n+ = N_total / (1 + μ) for 50/50 split with mass ratio μ
    let n_plus = (n_total as f64) / (1.0 + mu);
    let m_plus = rho_mean_plus * l_box.powi(3) / n_plus;
    let m_minus = mu * m_plus;

    (m_plus, m_minus)
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE STATE — CPU mirror of GPU state with split metadata
// ═══════════════════════════════════════════════════════════════════════════
struct AdaptiveState {
    particles: Vec<ParticleV3>,
    header: SnapshotHeaderV3,
    m_plus_base: f64,
    m_minus_base: f64,
    eps_plus_base: f64,
    eps_minus_base: f64,
}

impl AdaptiveState {
    fn new(
        n_plus: usize,
        n_minus: usize,
        l_box: f64,
        h0: f64,
        mu: f64,
        omega_b: f64,
        eps_plus: f64,
        eps_minus: f64,
        run_label: &str,
    ) -> Self {
        let n_total = n_plus + n_minus;
        let (m_plus, m_minus) = compute_particle_masses(n_total, l_box, h0, omega_b, mu);

        let mut header = SnapshotHeaderV3::new(run_label);
        header.l_box = l_box;
        header.h0 = h0;
        header.mu = mu;
        header.omega_b = omega_b;
        header.m_part_plus_base = m_plus;
        header.m_part_minus_base = m_minus;
        header.eps_plus_base = eps_plus;
        header.eps_minus_base = eps_minus;
        header.seed_ic = SEED_IC as u32;
        header.z_init = 10.0;       // Default, will be overridden
        header.z_start_run = 10.0;  // Default, will be overridden

        Self {
            particles: Vec::with_capacity(N_MAX_TOTAL),
            header,
            m_plus_base: m_plus,
            m_minus_base: m_minus,
            eps_plus_base: eps_plus,
            eps_minus_base: eps_minus,
        }
    }

    /// Initialize particles from GPU arrays
    fn init_from_arrays(&mut self, positions: &[f64], velocities: &[f64], signs: &[i32]) {
        let n = signs.len();
        self.particles.clear();
        self.particles.reserve(n);

        for i in 0..n {
            let is_positive = signs[i] > 0;
            let (mass, epsilon, sign) = if is_positive {
                (self.m_plus_base as f32, self.eps_plus_base as f32, 1u8)
            } else {
                (self.m_minus_base as f32, self.eps_minus_base as f32, 255u8)
            };

            self.particles.push(ParticleV3 {
                pos: [
                    positions[i * 3] as f32,
                    positions[i * 3 + 1] as f32,
                    positions[i * 3 + 2] as f32,
                ],
                vel: [
                    velocities[i * 3] as f32,
                    velocities[i * 3 + 1] as f32,
                    velocities[i * 3 + 2] as f32,
                ],
                mass,
                epsilon,
                sign,
                split_level: 0,
                is_star: 0,
                flags: 0,
            });
        }

        self.header.n_total = n as u64;
    }

    /// Sync positions/velocities from GPU to CPU particles
    fn sync_from_gpu(&mut self, positions: &[f64], velocities: &[f64]) {
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.pos = [
                positions[i * 3] as f32,
                positions[i * 3 + 1] as f32,
                positions[i * 3 + 2] as f32,
            ];
            p.vel = [
                velocities[i * 3] as f32,
                velocities[i * 3 + 1] as f32,
                velocities[i * 3 + 2] as f32,
            ];
        }
    }

    /// Extract flat arrays for GPU recreation (including per-particle masses)
    /// Masses are normalized: base level = 1.0, split level n = 1/8^n
    fn to_gpu_arrays(&self) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>) {
        let n = self.particles.len();
        let mut positions = Vec::with_capacity(n * 3);
        let mut velocities = Vec::with_capacity(n * 3);
        let mut signs = Vec::with_capacity(n);
        let mut masses = Vec::with_capacity(n);

        for p in &self.particles {
            positions.push(p.pos[0] as f64);
            positions.push(p.pos[1] as f64);
            positions.push(p.pos[2] as f64);
            velocities.push(p.vel[0] as f64);
            velocities.push(p.vel[1] as f64);
            velocities.push(p.vel[2] as f64);
            signs.push(if p.sign == 1 { 1 } else { -1 });

            // Mass in force units: G × m_physical (M☉)
            // This gives acceleration directly: a = mass × r⁻³ × r_vec
            let mass_force = G_COSMO * p.mass as f64;
            masses.push(mass_force);
        }

        (positions, velocities, signs, masses)
    }

    /// Count m+ and m- particles
    fn counts(&self) -> (usize, usize) {
        let n_plus = self.particles.iter().filter(|p| p.sign == 1).count();
        let n_minus = self.particles.len() - n_plus;
        (n_plus, n_minus)
    }

    /// Get maximum split level
    fn max_split_level(&self) -> u8 {
        self.particles.iter().map(|p| p.split_level).max().unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BLUE NOISE DAUGHTER PLACEMENT
// ═══════════════════════════════════════════════════════════════════════════
/// Generate 8 daughter positions using Blue Noise (Poisson disk) in a sphere
fn blue_noise_daughters(center: [f32; 3], radius: f32, rng: &mut impl Rng) -> [[f32; 3]; 8] {
    // Cube vertices scaled to fit in sphere (radius/sqrt(3))
    let r = radius / 1.732;  // sqrt(3)
    let offsets: [[f32; 3]; 8] = [
        [-r, -r, -r], [r, -r, -r], [-r, r, -r], [r, r, -r],
        [-r, -r, r], [r, -r, r], [-r, r, r], [r, r, r],
    ];

    // Add small random jitter for Blue Noise effect
    let jitter = radius * 0.1;
    let mut daughters = [[0f32; 3]; 8];
    let jitter_dist = rand::distr::Uniform::new(-jitter, jitter).unwrap();

    for (i, off) in offsets.iter().enumerate() {
        daughters[i] = [
            center[0] + off[0] + rng.sample(jitter_dist),
            center[1] + off[1] + rng.sample(jitter_dist),
            center[2] + off[2] + rng.sample(jitter_dist),
        ];
    }

    daughters
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE SPLIT CHECK
// ═══════════════════════════════════════════════════════════════════════════
/// Check and perform adaptive splits for particles exceeding density threshold
/// Returns number of new particles created
fn adaptive_split_check_with_thresholds(
    state: &mut AdaptiveState,
    densities: &[f64],
    delta_split: &[f64; 10],
    rng: &mut impl Rng,
) -> usize {
    if state.particles.len() >= N_MAX_TOTAL {
        return 0;  // At capacity
    }

    let mut to_split: Vec<usize> = Vec::new();

    // Find m+ particles that need splitting
    for (i, p) in state.particles.iter().enumerate() {
        if p.sign != 1 { continue; }  // Only split m+
        if p.split_level >= 9 { continue; }  // Max level reached

        let threshold = delta_split[p.split_level as usize];
        if densities[i] > threshold {
            to_split.push(i);
        }
    }

    // Limit splits to not exceed N_MAX_TOTAL
    let max_new = (N_MAX_TOTAL - state.particles.len()) / 7;  // Each split adds 7 (8-1)
    if to_split.len() > max_new {
        to_split.truncate(max_new);
    }

    if to_split.is_empty() {
        return 0;
    }

    // Perform splits (process in reverse to avoid index invalidation)
    let mut new_particles: Vec<ParticleV3> = Vec::new();

    for &idx in to_split.iter().rev() {
        let parent = &state.particles[idx];
        let new_level = parent.split_level + 1;
        let new_mass = parent.mass / 8.0;
        let new_eps = parent.epsilon / 2.0;

        // SPH smoothing length estimate: h ≈ 2 × epsilon
        let h_sph = parent.epsilon * 2.0;
        let daughter_radius = h_sph / 3.0;

        let daughter_positions = blue_noise_daughters(parent.pos, daughter_radius, rng);

        // Create 8 daughters
        for pos in daughter_positions.iter() {
            new_particles.push(ParticleV3 {
                pos: *pos,
                vel: parent.vel,  // Inherit velocity
                mass: new_mass,
                epsilon: new_eps,
                sign: parent.sign,
                split_level: new_level,
                is_star: parent.is_star,
                flags: parent.flags | 0x01,  // Mark as HR (high-resolution)
            });
        }
    }

    // Remove parents (in reverse order)
    for &idx in to_split.iter().rev() {
        state.particles.swap_remove(idx);
    }

    // Add daughters
    let n_new = new_particles.len();
    state.particles.extend(new_particles);
    state.header.n_total = state.particles.len() as u64;
    state.header.n_split_max = state.max_split_level() as u32;

    n_new
}

// ═══════════════════════════════════════════════════════════════════════════
// DENSITY COMPUTATION (Grid-based for now, SPH later)
// ═══════════════════════════════════════════════════════════════════════════
fn compute_densities(particles: &[ParticleV3], box_size: f64) -> Vec<f64> {
    let grid_size = 64;
    let cell_size = box_size / grid_size as f64;
    let cell_vol = cell_size.powi(3);

    // Count particles per cell
    let mut grid = vec![0.0f64; grid_size * grid_size * grid_size];
    let box_half = box_size / 2.0;

    for p in particles {
        let x = ((p.pos[0] as f64 + box_half) / cell_size) as usize;
        let y = ((p.pos[1] as f64 + box_half) / cell_size) as usize;
        let z = ((p.pos[2] as f64 + box_half) / cell_size) as usize;

        let x = x.min(grid_size - 1);
        let y = y.min(grid_size - 1);
        let z = z.min(grid_size - 1);

        let idx = x + y * grid_size + z * grid_size * grid_size;
        grid[idx] += p.mass as f64;
    }

    // Convert to density
    for v in &mut grid {
        *v /= cell_vol;
    }

    // Assign density to each particle
    let mut densities = Vec::with_capacity(particles.len());
    for p in particles {
        let x = ((p.pos[0] as f64 + box_half) / cell_size) as usize;
        let y = ((p.pos[1] as f64 + box_half) / cell_size) as usize;
        let z = ((p.pos[2] as f64 + box_half) / cell_size) as usize;

        let x = x.min(grid_size - 1);
        let y = y.min(grid_size - 1);
        let z = z.min(grid_size - 1);

        let idx = x + y * grid_size + z * grid_size * grid_size;
        densities.push(grid[idx]);
    }

    densities
}

// ═══════════════════════════════════════════════════════════════════════════
// SNAPSHOT V3 SAVE
// ═══════════════════════════════════════════════════════════════════════════
fn save_snapshot(
    path: &Path,
    state: &AdaptiveState,
    a: f64,
    t_gyr: f64,
    n_stars: u64,
    sfr: f64,
    rho_max: f64,
) {
    let mut header = state.header.clone();
    header.a = a;
    header.t_gyr = t_gyr;
    header.n_total = state.particles.len() as u64;
    header.n_split_max = state.max_split_level() as u32;
    header.n_stars = n_stars;
    header.sfr = sfr;
    header.rho_max = rho_max;

    if let Err(e) = write_snapshot_v3(path, &header, &state.particles) {
        eprintln!("WARNING: Failed to write snapshot: {}", e);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HR REGION CENTROID (for zoom rendering)
// ═══════════════════════════════════════════════════════════════════════════
fn compute_hr_centroid(particles: &[ParticleV3]) -> Option<[f64; 3]> {
    let hr_particles: Vec<_> = particles.iter()
        .filter(|p| p.split_level > 0)
        .collect();

    if hr_particles.is_empty() {
        return None;
    }

    let mut sum = [0.0f64; 3];
    for p in &hr_particles {
        sum[0] += p.pos[0] as f64;
        sum[1] += p.pos[1] as f64;
        sum[2] += p.pos[2] as f64;
    }

    let n = hr_particles.len() as f64;
    Some([sum[0] / n, sum[1] / n, sum[2] / n])
}

// ═══════════════════════════════════════════════════════════════════════════
// ZEL'DOVICH IC GENERATOR (with correct 3D displacement field)
// ═══════════════════════════════════════════════════════════════════════════

/// Perform 3D inverse FFT
fn ifft_3d(field: &mut [Complex<f64>], ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;
    let mut scratch = vec![Complex::new(0.0, 0.0); n];

    // FFT along x
    for iz in 0..n {
        for iy in 0..n {
            let start = iz * n * n + iy * n;
            let mut row: Vec<Complex<f64>> = field[start..start + n].to_vec();
            ifft.process_with_scratch(&mut row, &mut scratch);
            for ix in 0..n {
                field[start + ix] = row[ix];
            }
        }
    }

    // FFT along y
    for iz in 0..n {
        for ix in 0..n {
            let mut col: Vec<Complex<f64>> = (0..n).map(|iy| field[iz * n * n + iy * n + ix]).collect();
            ifft.process_with_scratch(&mut col, &mut scratch);
            for iy in 0..n {
                field[iz * n * n + iy * n + ix] = col[iy];
            }
        }
    }

    // FFT along z
    for iy in 0..n {
        for ix in 0..n {
            let mut tube: Vec<Complex<f64>> = (0..n).map(|iz| field[iz * n * n + iy * n + ix]).collect();
            ifft.process_with_scratch(&mut tube, &mut scratch);
            for iz in 0..n {
                field[iz * n * n + iy * n + ix] = tube[iz];
            }
        }
    }

    // Return real part, normalized
    let norm = 1.0 / (n3 as f64);
    field.iter().map(|c| c.re * norm).collect()
}

fn generate_zeldovich_ics(n_grid: usize, l_box: f64, z_init: f64, h0: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    println!("\n[1/5] Generating Zel'dovich ICs (correct 3D displacement)...");

    let n_total = n_grid * n_grid * n_grid;
    let spacing = l_box / n_grid as f64;
    let half_box = l_box / 2.0;
    let half_n = n_grid / 2;
    let dk = 2.0 * PI / l_box;

    println!("  Grid: {}³ = {} particles", n_grid, n_total);
    println!("  Box: {} Mpc, z_init = {}", l_box, z_init);
    println!("  Seed: {}, n_s = {}, δ_rms = {}", SEED_IC, N_S, DELTA_RMS);

    let mut rng = StdRng::seed_from_u64(SEED_IC);

    // Step 1: Generate Gaussian random field δ(k) with correct displacement spectrum
    // P_disp(k) ∝ k^(n_s - 4) × δ_rms² — this gives ~30% cell displacement
    println!("  Generating density field δ(k)...");
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_total];

    // Cosmological parameters for IC amplitude
    let a_init = 1.0 / (1.0 + z_init);
    let d_growth = a_init;  // Linear growth factor D(a) ≈ a in matter-dominated era

    let normal = Normal::new(0.0, 1.0).unwrap();

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let kx = if ix <= half_n { ix as f64 } else { ix as f64 - n_grid as f64 } * dk;
                let ky = if iy <= half_n { iy as f64 } else { iy as f64 - n_grid as f64 } * dk;
                let kz = if iz <= half_n { iz as f64 } else { iz as f64 - n_grid as f64 } * dk;
                let k2 = kx * kx + ky * ky + kz * kz;

                if k2 > 0.0 {
                    let k = k2.sqrt();

                    // Power spectrum P(k) ∝ k^(n_s - 4) for displacement field
                    // This is the correct Zel'dovich spectrum (Harrison-Zeldovich × 1/k²)
                    let pk = k.powf(N_S - 4.0) * DELTA_RMS.powi(2);
                    let amp = (pk / 2.0).sqrt();

                    let phase = rng.random::<f64>() * 2.0 * PI;
                    let re = rng.sample(&normal) * amp * d_growth;
                    let im = rng.sample(&normal) * amp * d_growth;

                    delta_k[idx] = Complex::new(
                        re * phase.cos() - im * phase.sin(),
                        re * phase.sin() + im * phase.cos()
                    );
                }
            }
        }
    }

    // Step 2: Compute displacement fields ψ(k) = -i k δ(k) / k²
    println!("  Computing displacement fields ψ_x, ψ_y, ψ_z...");
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_total];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_total];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_total];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };

                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k2 = kx * kx + ky * ky + kz * kz;

                if k2 > 1e-20 {
                    // ψ(k) = -i k δ(k) / k²
                    let minus_i = Complex::new(0.0, -1.0);
                    psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
                    psi_y_k[idx] = minus_i * ky * delta_k[idx] / k2;
                    psi_z_k[idx] = minus_i * kz * delta_k[idx] / k2;
                }
            }
        }
    }

    // Step 3: Inverse FFT to get real-space displacement fields
    println!("  Performing inverse FFT...");
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    // Step 4: Compute max displacement for scaling
    let mut max_disp = 0.0f64;
    for i in 0..n_total {
        let d = (psi_x[i] * psi_x[i] + psi_y[i] * psi_y[i] + psi_z[i] * psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Max displacement (raw): {:.6e} Mpc", max_disp);

    // Scale to target: ~30% of cell size for proper structure
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Scaling: max_disp → {:.4} Mpc ({:.1}% of cell)",
             target_disp, 100.0 * target_disp / spacing);

    // Velocity scaling: v = a * H(a) * D'(a)/D(a) * ψ ≈ a * H * f * ψ
    // At high z, f ≈ 1 and D'(a)/D(a) ≈ sqrt(1+z)
    let a_init = 1.0 / (1.0 + z_init);
    let h_init = h0 / MPC_GYR_TO_KMS;  // H in Gyr^-1 units
    let d_dot_factor = (1.0 + z_init).sqrt();
    let vel_scale = a_init * h_init * d_dot_factor * scale;

    // Step 5: Build particle arrays
    println!("  Building particle arrays...");
    let mut positions = vec![0.0f64; n_total * 3];
    let mut velocities = vec![0.0f64; n_total * 3];
    let mut signs = vec![0i32; n_total];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;

                // Grid position (centered at cell)
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // Apply 3D displacement with periodic wrapping
                let dx = psi_x[idx] * scale;
                let dy = psi_y[idx] * scale;
                let dz = psi_z[idx] * scale;

                positions[idx * 3]     = ((x0 + dx + half_box) % l_box + l_box) % l_box - half_box;
                positions[idx * 3 + 1] = ((y0 + dy + half_box) % l_box + l_box) % l_box - half_box;
                positions[idx * 3 + 2] = ((z0 + dz + half_box) % l_box + l_box) % l_box - half_box;

                // Zel'dovich velocity: v = a * H * D'/D * ψ
                velocities[idx * 3]     = psi_x[idx] * vel_scale;
                velocities[idx * 3 + 1] = psi_y[idx] * vel_scale;
                velocities[idx * 3 + 2] = psi_z[idx] * vel_scale;

                // Random sign assignment (η=1.045 → ~52% m+)
                signs[idx] = if rng.random::<f64>() < 0.52 { 1 } else { -1 };
            }
        }
    }

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = n_total - n_plus;
    println!("  Generated: N+ = {}, N- = {}", n_plus, n_minus);

    (positions, velocities, signs)
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(feature = "cuda")]
fn main() {
    let args = Args::parse();

    // Compute ρ_mean_plus for density-independent thresholds
    let rho_crit = 2.775e11 * (args.h0 / 100.0).powi(2);  // M☉/Mpc³
    let rho_mean_plus = args.omega_b * rho_crit;  // ≈ 6.78e9 M☉/Mpc³

    // Build delta_split array: multiples of ρ_mean_plus
    // First split at 10,000× ρ_mean = 6.78e13 M☉/Mpc³
    // ICs have ρ_max ≈ 3e12 → factor 22× below threshold → no split at z=10
    let delta_split: [f64; 10] = [
        rho_mean_plus * 1.0e4,   // level 0→1 : ×10,000 — real collapse start
        rho_mean_plus * 3.0e4,   // level 1→2 : ×30,000
        rho_mean_plus * 1.0e5,   // level 2→3 : ×100,000
        rho_mean_plus * 3.0e5,   // level 3→4 : ×300,000
        rho_mean_plus * 1.0e6,   // level 4→5 : ×1,000,000
        rho_mean_plus * 3.0e6,   // level 5→6 : ×3,000,000
        rho_mean_plus * 1.0e7,   // level 6→7 : ×10,000,000
        rho_mean_plus * 3.0e7,   // level 7→8 : ×30,000,000
        rho_mean_plus * 1.0e8,   // level 8→9 : ×100,000,000
        rho_mean_plus * 3.0e8,   // level 9→10 : ×300,000,000
    ];

    let n_particles_init = args.n_grid * args.n_grid * args.n_grid;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           JANUS ADAPTIVE ZOOM — Production Run                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  COSMOLOGY: μ = {}, η = {}", args.mu, ETA);
    println!("║    H₀ = {} km/s/Mpc, Ω_b = {}", args.h0, args.omega_b);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  SIMULATION:");
    println!("║    n_grid = {} → N_init = {}, N_max = {}", args.n_grid, n_particles_init, N_MAX_TOTAL);
    println!("║    L_box = {} Mpc, ε+ = {} Mpc, ε- = {} Mpc", args.l_box, args.eps_plus, args.eps_minus);
    println!("║    z_init = {} → z_final = {}, dt = {} Gyr", args.z_init, args.z_final, args.dt_max);
    println!("║    θ = {}, η = {}", args.theta, args.eta);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  ADAPTIVE SPLITTING:");
    println!("║    Check every {} steps", args.steps_check);
    println!("║    ρ_mean_plus = {:.2e} M☉/Mpc³", rho_mean_plus);
    println!("║    δ_split[0] = {:.2e} M☉/Mpc³ (×10⁴ ρ_mean)", delta_split[0]);
    println!("║    δ_split[5] = {:.2e} M☉/Mpc³ (×3×10⁶ ρ_mean)", delta_split[5]);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  OUTPUT: {} (v3 format)", args.out_dir);
    println!("║    Snapshots every {} steps, Label: {}", args.snap_interval, args.run_label);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", args.out_dir)).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/frames", args.out_dir)).expect("Failed to create frames dir");

    // Generate Zel'dovich ICs
    let (positions, velocities, signs) = generate_zeldovich_ics(args.n_grid, args.l_box, args.z_init, args.h0);

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.len() - n_plus;

    // Initialize adaptive state
    let mut state = AdaptiveState::new(
        n_plus, n_minus, args.l_box,
        args.h0, args.mu, args.omega_b,
        args.eps_plus, args.eps_minus,
        &args.run_label,
    );
    state.init_from_arrays(&positions, &velocities, &signs);
    state.header.z_init = args.z_init;
    state.header.z_start_run = args.z_init;

    println!("\n[2/5] Initializing GPU simulation...");
    // CRITICAL: Use to_gpu_arrays() to get masses with G_COSMO factor
    let (gpu_pos, gpu_vel, gpu_signs, gpu_masses) = state.to_gpu_arrays();
    let mut gpu_sim = GpuNBodySimulation::new_with_state_and_masses(
        n_plus, n_minus, args.l_box,
        gpu_pos, gpu_vel, gpu_signs, gpu_masses
    ).expect("Failed to create GPU simulation");
    gpu_sim.set_theta(args.theta);
    gpu_sim.set_softening(args.eps_plus);

    // Dynamic c_ratio
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(args.z_init, ETA);
    gpu_sim.set_c_ratio(c_ratio_sq_init.sqrt());
    println!("  GPU ready: {} particles, θ={}", state.particles.len(), args.theta);

    // RNG for splits
    let mut rng_split = StdRng::seed_from_u64(SEED_IC + 1000);

    // Cosmological state
    let mut a = 1.0 / (1.0 + args.z_init);
    let mut t_gyr = 0.5;  // Approximate cosmic time at z=10

    // CSV output
    let csv_path = format!("{}/time_series.csv", args.out_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,t_Gyr,z,a,N_total,N_hr,split_max,rho_max,v_rms").unwrap();

    println!("\n[3/5] Starting main loop (z={} → z={})...\n", args.z_init, args.z_final);

    let mut step = 0;
    loop {
        let z = 1.0 / a - 1.0;

        // Stop condition
        if z < args.z_final {
            println!("\n  Reached z_final = {:.2} at step {}", args.z_final, step);
            break;
        }

        let h = args.h0 / MPC_GYR_TO_KMS * (1.0 + z).sqrt();  // Simplified H(z)

        // Metrics
        let do_metric = step % METRIC_INTERVAL == 0;
        let do_snapshot = step % args.snap_interval == 0;
        let do_split = step % args.steps_check == 0 && step > 0;

        if do_metric || do_snapshot || do_split {
            // Sync GPU → CPU
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            state.sync_from_gpu(&pos, &vel);

            // Compute densities
            let densities = compute_densities(&state.particles, args.l_box);
            let rho_max = densities.iter().cloned().fold(0.0f64, f64::max);

            // v_rms
            let v_rms: f64 = {
                let sum: f64 = state.particles.iter()
                    .map(|p| (p.vel[0]*p.vel[0] + p.vel[1]*p.vel[1] + p.vel[2]*p.vel[2]) as f64)
                    .sum();
                (sum / state.particles.len() as f64).sqrt() * MPC_GYR_TO_KMS
            };

            // Adaptive split check
            if do_split {
                let n_before = state.particles.len();
                let n_new = adaptive_split_check_with_thresholds(&mut state, &densities, &delta_split, &mut rng_split);

                if n_new > 0 {
                    println!("  🔬 Step {}: Split +{} particles, N={} → {}",
                        step, n_new, n_before, state.particles.len());

                    // Recreate GPU simulation with new particle count and per-particle masses
                    let (new_pos, new_vel, new_signs, new_masses) = state.to_gpu_arrays();
                    let (np, nm) = state.counts();

                    // CRITICAL: Drop old GPU sim BEFORE creating new one to free VRAM
                    // Without this, both sims exist simultaneously during creation = OOM
                    drop(gpu_sim);

                    gpu_sim = GpuNBodySimulation::new_with_state_and_masses(
                        np, nm, args.l_box, new_pos, new_vel, new_signs, new_masses
                    ).expect("Failed to recreate GPU simulation after split");
                    gpu_sim.set_theta(args.theta);
                    gpu_sim.set_softening(args.eps_plus);
                    gpu_sim.set_c_ratio(CoupledFriedmann::c_ratio_sq_at_z(z, ETA).sqrt());
                }
            }

            let n_hr = state.particles.iter().filter(|p| p.split_level > 0).count();
            let split_max = state.max_split_level();

            if do_metric {
                writeln!(csv, "{},{:.6},{:.4},{:.6},{},{},{},{:.3e},{:.2}",
                    step, t_gyr, z, a, state.particles.len(), n_hr, split_max, rho_max, v_rms).unwrap();

                if step % 100 == 0 || do_split {
                    println!("  Step {:5} | z={:.3} | N={:>8} | N_hr={:>6} | ρ_max={:.2e} | v_rms={:.1} km/s",
                        step, z, state.particles.len(), n_hr, rho_max, v_rms);
                }
            }

            // Snapshot
            if do_snapshot {
                let snap_path = Path::new(&args.out_dir).join("snapshots").join(format!("snap_{:05}.bin", step));
                save_snapshot(&snap_path, &state, a, t_gyr, 0, 0.0, rho_max);

                // Use snapshot_info to verify
                if let Ok(info) = snapshot_info(&snap_path) {
                    if split_max > 0 {
                        println!("    📸 Snapshot {} with split_max={}", step, split_max);
                        println!("{}", info);
                    } else {
                        println!("    📸 Snapshot {}", step);
                    }
                }

                if let Some(centroid) = compute_hr_centroid(&state.particles) {
                    println!("    HR centroid: ({:.1}, {:.1}, {:.1}) Mpc",
                        centroid[0], centroid[1], centroid[2]);
                }
            }

            csv.flush().unwrap();
        }

        // Time integration
        gpu_sim.step_with_expansion_dkd_gpu(args.dt_max, a, h, 0.0).unwrap();
        a += a * h * args.dt_max;
        t_gyr += args.dt_max;

        // Update c_ratio dynamically
        if step % 100 == 0 {
            let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
            gpu_sim.set_c_ratio(c_ratio_sq.sqrt());
        }

        step += 1;
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  RUN COMPLETE                                                            ║");
    println!("║  Total time: {:.1} hours ({:.0} s)", total_time / 3600.0, total_time);
    println!("║  Final N: {} particles", state.particles.len());
    println!("║  Max split level: {}", state.max_split_level());
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
    eprintln!("Usage: cargo run --release --features cuda --bin janus_adaptive_zoom");
    std::process::exit(1);
}
