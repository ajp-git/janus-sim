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
#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
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

    /// Zoom cube size in Mpc (0 = disabled, splits anywhere based on density)
    /// If > 0, only particles inside [-size/2, +size/2]³ centered at origin can split
    #[arg(long, default_value = "0.0")]
    zoom_cube_size: f64,

    /// Maximum split level (strict limit to prevent runaway splitting)
    #[arg(long, default_value = "2")]
    max_split_level: u8,

    /// Split threshold for level 0→1 (M_sun/Mpc³). Default ~10× ρ_plus_mean
    #[arg(long, default_value = "6.78e10")]
    delta_split_l1: f64,

    /// Split threshold for level 1→2 (M_sun/Mpc³). Default ~100× ρ_plus_mean
    #[arg(long, default_value = "6.78e11")]
    delta_split_l2: f64,
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
const DELTA_RMS: f64 = 0.15;  // v9: increased from 0.10 for stronger IC perturbations

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;
const G_COSMO: f64 = 4.499e-15;  // Mpc³ M☉⁻¹ Gyr⁻²

// Physique baryonique
const T_INIT_PLUS: f64 = 10000.0;   // Température initiale m+ [K]
const T_FLOOR: f64 = 100.0;         // Température plancher [K]

// ═══════════════════════════════════════════════════════════════════════════
// JANUS EXPANSION (Petit & D'Agostini 2014 + Petit 2018)
// Two-phase model: radiative era (z > 4.51) + matter era (z < 4.51)
// ═══════════════════════════════════════════════════════════════════════════
/// Janus parametric solution constants (calibrated for H₀=69.9, t₀=15.87 Gyr)
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;  // Gyr
/// Transition scale factor: a = α² corresponds to z = 4.5083
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;

/// Compute Hubble parameter H(a) for the Janus cosmological model.
///
/// Implements two-phase expansion:
/// - For a < α²: gauge process era (Petit 2018), EdS-like H = H₀ × a^(-3/2)
/// - For a ≥ α²: matter era (Petit & D'Agostini 2014), parametric cosh²(μ_p)
///
/// Discontinuity at a = α²: physical "Janus point" where the universe
/// transitions between the two regimes.
///
/// # Arguments
/// * `a` - Scale factor (a=1 today, a→0 at Big Bang)
/// * `h0_kms_mpc` - Hubble constant in km/s/Mpc (typically 69.9)
///
/// # Returns
/// H(a) in Gyr⁻¹
///
/// # Reference values
/// - H(a=1)    = 0.071487 Gyr⁻¹  = 69.90 km/s/Mpc
/// - H(z=1)    = 0.117986 Gyr⁻¹  = 115.37 km/s/Mpc
/// - H(z=4.5)  = 0.017622 Gyr⁻¹  = 17.23 km/s/Mpc (matter era, near transition)
/// - H(z=10)   = 2.608052 Gyr⁻¹  = 2550 km/s/Mpc (radiative era)
fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;

    if a < A_TRANSITION_JANUS {
        // Phase radiative / gauge process (Petit 2018)
        // t ∝ a^(3/2) → H = H₀ × a^(-3/2)
        h0_gyr_inv / a.powf(1.5)
    } else {
        // Phase matière (Petit & D'Agostini 2014)
        // a(μ_p) = α² cosh²(μ_p)
        // H(μ_p) = sinh(2μ_p) / [τ₀ α² cosh²(μ_p) (1 + ½ sinh(2μ_p))]
        let cosh2_mu = a / ALPHA_SQ_JANUS;
        // Numerical safety: cosh²(μ) ≥ 1 by definition
        let cosh2_mu_safe = cosh2_mu.max(1.0);
        let cosh_mu = cosh2_mu_safe.sqrt();
        let mu_p = cosh_mu.acosh();
        let s2mu = (2.0 * mu_p).sinh();
        s2mu / (TAU_0_JANUS * ALPHA_SQ_JANUS * cosh2_mu_safe * (1.0 + 0.5 * s2mu))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PARTICLE MASSES (M_sun) — computed from cosmology
// ═══════════════════════════════════════════════════════════════════════════
fn compute_particle_masses(n_total: usize, l_box: f64, h0: f64, omega_b: f64, mu: f64) -> (f64, f64) {
    // Cosmological mass calculation — Janus convention: ONE grid, signs shuffled
    //   N+ = N_total / (1+mu)   (5% pour mu=19)
    //   N- = N_total * mu/(1+mu) (95%)
    //   masse INDIVIDUELLE identique pour m+ et m- ; μ est le rapport de DENSITÉ, pas de masse par particule.
    //   M+ = Omega_b * rho_crit * L^3
    //   M- = mu * Omega_b * rho_crit * L^3
    //   m_plus  = M+ / N+ = Omega_b rho_crit L^3 * (1+mu) / N_total
    //   m_minus = M- / N- = mu Omega_b rho_crit L^3 * (1+mu) / (N_total*mu) = m_plus  (identique)
    let rho_crit = 2.775e11 * (h0 / 100.0).powi(2);  // M☉/Mpc³
    let m_plus = omega_b * rho_crit * l_box.powi(3) * (1.0 + mu) / (n_total as f64);
    let m_minus = m_plus;

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

            // Système d'unités implicite : masse normalisée = 1.0
            // pour une particule non-splittée, comme janus_baryonic_calibrated
            let mass_force = if p.split_level == 0 {
                1.0
            } else {
                1.0 / 8.0_f64.powi(p.split_level as i32)
                // split_level=1 → 0.125 (8 filles × 0.125 = 1.0 conservé)
                // split_level=2 → 0.015625
            };
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
/// Adaptive split check with thermal history preservation.
///
/// Arguments:
///   state: Adaptive state with particles
///   densities_plus: ρ_plus at each particle location (m+ density field)
///   delta_split_l1: threshold for level 0→1 splits
///   delta_split_l2: threshold for level 1→2 splits
///   max_split_level: maximum allowed split level (strict limit)
///   zoom_cube_size: if > 0, only split particles inside [-size/2, +size/2]³
///   rng: random number generator for daughter placement
///   u_plus_old: internal energy array for m+ particles BEFORE split
///
/// Returns:
///   n_new: number of daughter particles created (= 8 * n_split_events)
///   u_new: internal energy array for m+ particles, aligned with the NEW order
///          of m+ particles in state.particles after the split.
///          Each daughter inherits its parent's u.
fn adaptive_split_check_with_thresholds(
    state: &mut AdaptiveState,
    densities_plus: &[f64],
    delta_split_l1: f64,
    delta_split_l2: f64,
    max_split_level: u8,
    zoom_cube_size: f64,
    rng: &mut impl Rng,
    u_plus_old: &[f64],
) -> (usize, Vec<f64>) {
    // Build initial mapping: m+ particle index (in state.particles) -> u index (in u_plus_old)
    // u_plus_old[k] corresponds to the k-th m+ particle in state.particles (in iteration order).
    let mut particle_to_u: Vec<i32> = vec![-1; state.particles.len()];
    {
        let mut k = 0usize;
        for (i, p) in state.particles.iter().enumerate() {
            if p.sign == 1 {
                if k < u_plus_old.len() {
                    particle_to_u[i] = k as i32;
                }
                k += 1;
            }
        }
    }

    if state.particles.len() >= N_MAX_TOTAL {
        // No split -> u unchanged
        return (0, u_plus_old.to_vec());
    }

    let mut to_split: Vec<usize> = Vec::new();
    let zoom_half = zoom_cube_size / 2.0;

    // Find m+ particles that need splitting
    for (i, p) in state.particles.iter().enumerate() {
        if p.sign != 1 { continue; }  // Only split m+
        if p.split_level >= max_split_level { continue; }  // Max level reached (strict limit)

        // Spatial condition: if zoom is enabled, particle must be inside zoom cube
        if zoom_cube_size > 0.0 {
            let px = p.pos[0] as f64;
            let py = p.pos[1] as f64;
            let pz = p.pos[2] as f64;
            if px.abs() > zoom_half || py.abs() > zoom_half || pz.abs() > zoom_half {
                continue;  // Outside zoom zone → no split
            }
        }

        // Density threshold based on current split level (uses ρ_plus, not ρ_total)
        let threshold = if p.split_level == 0 { delta_split_l1 } else { delta_split_l2 };
        if densities_plus[i] > threshold {
            to_split.push(i);
        }
    }

    // Limit splits to not exceed N_MAX_TOTAL AND to avoid GPU reconstruction bottleneck
    const MAX_SPLITS_PER_STEP: usize = 10_000;  // Gradual splitting to avoid GPU stall
    let max_capacity = (N_MAX_TOTAL - state.particles.len()) / 7;  // Each split adds 7 (8-1)
    let max_new = max_capacity.min(MAX_SPLITS_PER_STEP);
    if to_split.len() > max_new {
        to_split.truncate(max_new);
    }

    if to_split.is_empty() {
        return (0, u_plus_old.to_vec());
    }

    // Perform splits (process in reverse to avoid index invalidation)
    // For each daughter we also track which u slot the parent had.
    let mut new_particles: Vec<ParticleV3> = Vec::new();
    let mut new_particle_u: Vec<f64> = Vec::new();  // u for daughters, in extend-order

    for &idx in to_split.iter().rev() {
        let parent = &state.particles[idx];
        let parent_u_idx = particle_to_u[idx];
        let parent_u: f64 = if parent_u_idx >= 0 && (parent_u_idx as usize) < u_plus_old.len() {
            u_plus_old[parent_u_idx as usize]
        } else {
            0.0  // should not happen; safety fallback
        };
        let new_level = parent.split_level + 1;
        let new_mass = parent.mass / 8.0;
        let new_eps = parent.epsilon / 2.0;

        // SPH smoothing length estimate: h ≈ 2 × epsilon
        let h_sph = parent.epsilon * 2.0;
        let daughter_radius = h_sph / 3.0;

        let daughter_positions = blue_noise_daughters(parent.pos, daughter_radius, rng);

        // Create 8 daughters, each inheriting parent's u
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
            new_particle_u.push(parent_u);
        }
    }

    // Remove parents (in reverse order). swap_remove brings the tail particle
    // into the removed slot — we must update particle_to_u accordingly.
    for &idx in to_split.iter().rev() {
        let last = state.particles.len() - 1;
        state.particles.swap_remove(idx);
        // particle_to_u tracks the sign->u mapping; do the same swap
        if idx != last {
            particle_to_u[idx] = particle_to_u[last];
        }
        particle_to_u.pop();  // last element removed
    }

    // Add daughters (their u goes into new_particle_u, aligned with new_particles)
    let n_new = new_particles.len();
    state.particles.extend(new_particles);
    state.header.n_total = state.particles.len() as u64;
    state.header.n_split_max = state.max_split_level() as u32;

    // Build final u array in the NEW m+ order.
    // Preserved m+ particles: occupy the first N_preserved slots of state.particles (in altered order),
    //   their u comes from u_plus_old via particle_to_u.
    // Daughters (appended): their u is in new_particle_u, in extend-order.
    let n_preserved_particles = state.particles.len() - n_new;
    let mut u_new: Vec<f64> = Vec::new();
    // First pass: preserved m+ particles
    for i in 0..n_preserved_particles {
        let p = &state.particles[i];
        if p.sign == 1 {
            let u_idx = particle_to_u[i];
            if u_idx >= 0 && (u_idx as usize) < u_plus_old.len() {
                u_new.push(u_plus_old[u_idx as usize]);
            } else {
                u_new.push(0.0);
            }
        }
    }
    // Second pass: daughters (all m+ since only m+ split)
    // Daughters appear at the end of state.particles in the same order as new_particle_u.
    for &u in new_particle_u.iter() {
        u_new.push(u);
    }

    (n_new, u_new)
}

// ═══════════════════════════════════════════════════════════════════════════
// DENSITY COMPUTATION (Grid-based for now, SPH later)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute separate density fields for m+ and m- particles.
/// Returns (densities_plus, densities_minus, rho_plus_max, rho_minus_max)
/// where densities_X[i] is the density at particle i's location for population X.
fn compute_densities_split(particles: &[ParticleV3], box_size: f64) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let grid_size = 64;
    let cell_size = box_size / grid_size as f64;
    let cell_vol = cell_size.powi(3);
    let n_cells = grid_size * grid_size * grid_size;

    // Separate grids for m+ and m-
    let mut grid_plus = vec![0.0f64; n_cells];
    let mut grid_minus = vec![0.0f64; n_cells];
    let box_half = box_size / 2.0;

    // Accumulate mass per cell, separated by sign
    for p in particles {
        let x = ((p.pos[0] as f64 + box_half) / cell_size) as usize;
        let y = ((p.pos[1] as f64 + box_half) / cell_size) as usize;
        let z = ((p.pos[2] as f64 + box_half) / cell_size) as usize;

        let x = x.min(grid_size - 1);
        let y = y.min(grid_size - 1);
        let z = z.min(grid_size - 1);

        let idx = x + y * grid_size + z * grid_size * grid_size;
        let m = p.mass as f64;

        if p.sign == 1 {
            grid_plus[idx] += m;
        } else {
            grid_minus[idx] += m;
        }
    }

    // Convert to density
    for v in &mut grid_plus {
        *v /= cell_vol;
    }
    for v in &mut grid_minus {
        *v /= cell_vol;
    }

    // Find max densities
    let rho_plus_max = grid_plus.iter().cloned().fold(0.0f64, f64::max);
    let rho_minus_max = grid_minus.iter().cloned().fold(0.0f64, f64::max);

    // Assign density to each particle (both populations)
    let mut densities_plus = Vec::with_capacity(particles.len());
    let mut densities_minus = Vec::with_capacity(particles.len());

    for p in particles {
        let x = ((p.pos[0] as f64 + box_half) / cell_size) as usize;
        let y = ((p.pos[1] as f64 + box_half) / cell_size) as usize;
        let z = ((p.pos[2] as f64 + box_half) / cell_size) as usize;

        let x = x.min(grid_size - 1);
        let y = y.min(grid_size - 1);
        let z = z.min(grid_size - 1);

        let idx = x + y * grid_size + z * grid_size * grid_size;
        densities_plus.push(grid_plus[idx]);
        densities_minus.push(grid_minus[idx]);
    }

    (densities_plus, densities_minus, rho_plus_max, rho_minus_max)
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

/// Compute local overdensities for m+ particles (for cooling physics)
fn compute_local_overdensities(pos: &[f64], signs: &[i32], grid_size: usize, box_size: f64) -> Vec<f64> {
    let half_box = box_size / 2.0;
    let cell_size = box_size / grid_size as f64;
    let n3 = grid_size * grid_size * grid_size;

    // Count particles per cell (m+ only)
    let mut cell_counts = vec![0u32; n3];
    let mut particle_cells: Vec<usize> = Vec::new();

    let n = pos.len() / 3;
    for i in 0..n {
        if signs[i] <= 0 {
            continue;  // Skip m-
        }

        let x = pos[i*3];
        let y = pos[i*3 + 1];
        let z = pos[i*3 + 2];

        let ix = ((x + half_box) / cell_size).floor() as usize;
        let iy = ((y + half_box) / cell_size).floor() as usize;
        let iz = ((z + half_box) / cell_size).floor() as usize;

        let ix = ix.min(grid_size - 1);
        let iy = iy.min(grid_size - 1);
        let iz = iz.min(grid_size - 1);

        let idx = ix + iy * grid_size + iz * grid_size * grid_size;
        cell_counts[idx] += 1;
        particle_cells.push(idx);
    }

    // Compute mean count per cell
    let n_plus = particle_cells.len() as f64;
    let mean_per_cell = n_plus / n3 as f64;

    // Return overdensity for each m+ particle
    particle_cells.iter()
        .map(|&cell_idx| {
            let count = cell_counts[cell_idx] as f64;
            if mean_per_cell > 0.0 {
                count / mean_per_cell
            } else {
                1.0
            }
        })
        .collect()
}

fn generate_zeldovich_ics(n_grid: usize, l_box: f64, z_init: f64, h0: f64, mu: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    println!("\n[1/5] Generating Zel'dovich ICs (correct 3D displacement)...");

    let n_total = n_grid * n_grid * n_grid;
    let spacing = l_box / n_grid as f64;
    let half_box = l_box / 2.0;

    // ═══════════════════════════════════════════════════════════════════════
    // FIX Phase 10: Padding ×2 pour éliminer la contamination grille
    // Générer ψ sur grille 2×n_grid, puis interpoler CIC aux positions random
    // ═══════════════════════════════════════════════════════════════════════
    let n_fft = 2 * n_grid;  // 430 au lieu de 215
    let n_fft_total = n_fft * n_fft * n_fft;
    let spacing_fft = l_box / n_fft as f64;
    let half_n_fft = n_fft / 2;
    let dk = 2.0 * PI / l_box;

    println!("  Particle grid: {}³ = {} particles", n_grid, n_total);
    println!("  FFT grid: {}³ = {} modes (×2 padding)", n_fft, n_fft_total);
    println!("  Box: {} Mpc, z_init = {}", l_box, z_init);
    println!("  Seed: {}, n_s = {}, δ_rms = {}", SEED_IC, N_S, DELTA_RMS);

    let mut rng = StdRng::seed_from_u64(SEED_IC);

    // Step 1: Generate Gaussian random field δ(k) with P_δ(k) ∝ k^n_s
    // Same method as champion_10m_v2.rs - WORKING formula
    // FIX Phase 10: Generate on n_fft³ grid (×2 padding)
    println!("  Generating density field δ(k) on {}³ grid...", n_fft);
    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft_total];

    // IC amplitude from champion_10m_v2.rs (empirically validated)
    const IC_AMPLITUDE: f64 = 0.01;
    let a_init = 1.0 / (1.0 + z_init);
    let d_growth = a_init;  // Linear growth factor D(a) ≈ a in matter-dominated era

    let normal = Normal::new(0.0, 1.0).unwrap();

    // Power spectrum cutoffs
    const K_MIN_IC: f64 = 2.0 * PI / 500.0;  // mode fondamental boîte
    const K_MAX_IC: f64 = 2.0 * PI / 5.0;    // cutoff ~1.26 rad/Mpc (original)
    const K0_IC: f64 = 0.02;
    // FIX Phase 10: Tukey window width (40% transitions for smoother cutoff)
    const TUKEY_WIDTH: f64 = 0.4;

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

                    // FIX Phase 10: Tukey/tanh window instead of step
                    // Smooth transitions at K_MIN and K_MAX to eliminate Gibbs ringing
                    let low_width = K_MIN_IC * TUKEY_WIDTH;
                    let high_width = K_MAX_IC * TUKEY_WIDTH;
                    let w_low = 0.5 * (1.0 + ((k - K_MIN_IC) / low_width).tanh());
                    let w_high = 0.5 * (1.0 - ((k - K_MAX_IC) / high_width).tanh());
                    let window = w_low * w_high;

                    let pk = k.powf(N_S) / (1.0 + (k / K0_IC).powi(4)) * window;
                    let sigma_k = pk.sqrt() * IC_AMPLITUDE * d_growth;

                    let re = rng.sample(&normal) * sigma_k;
                    let im = rng.sample(&normal) * sigma_k;

                    delta_k[idx] = Complex::new(re, im);
                }
            }
        }
    }

    // Enforce Hermitian symmetry for real IFFT (on n_fft³ grid)
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

    // DIAGNOSTIC: delta_k amplitude
    let delta_max = delta_k.iter()
        .map(|c| c.norm())
        .fold(0.0f64, f64::max);
    println!("  delta_k max amplitude = {:.6e}", delta_max);

    // Step 2: Compute displacement fields ψ(k) = -i k δ(k) / k² (on n_fft³)
    println!("  Computing displacement fields ψ_x, ψ_y, ψ_z on {}³...", n_fft);
    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft_total];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft_total];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_fft_total];

    for iz in 0..n_fft {
        for iy in 0..n_fft {
            for ix in 0..n_fft {
                let idx = iz * n_fft * n_fft + iy * n_fft + ix;

                let kx_idx = if ix <= half_n_fft { ix as i32 } else { ix as i32 - n_fft as i32 };
                let ky_idx = if iy <= half_n_fft { iy as i32 } else { iy as i32 - n_fft as i32 };
                let kz_idx = if iz <= half_n_fft { iz as i32 } else { iz as i32 - n_fft as i32 };

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

    // Step 3: Inverse FFT to get real-space displacement fields (on n_fft³)
    println!("  Performing inverse FFT on {}³...", n_fft);
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_fft);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_fft);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_fft);

    // DIAGNOSTIC: psi_x range after IFFT
    let psi_real_max = psi_x.iter().cloned().fold(0.0f64, f64::max);
    let psi_real_min = psi_x.iter().cloned().fold(0.0f64, f64::min);
    println!("  psi_x range = [{:.3e}, {:.3e}]", psi_real_min, psi_real_max);

    // Step 4: Compute max displacement for scaling (on n_fft³ grid)
    let mut max_disp = 0.0f64;
    for i in 0..n_fft_total {
        let d = (psi_x[i] * psi_x[i] + psi_y[i] * psi_y[i] + psi_z[i] * psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    println!("  Max displacement (raw): {:.6e} Mpc", max_disp);

    // Scale to target: 30% of cell size (Zel'dovich standard practice)
    // 0.7 × spacing causait un artefact de « croix » aligné sur axes car les
    // particules traversaient leur cellule d'origine et s'alignaient sur les
    // modes Fourier dominants de la grille. 0.3 × spacing est la valeur
    // utilisée dans janus_baryonic_calibrated (run 10M validé, preprint JPP).
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };
    println!("  Target displacement: {:.4} Mpc ({:.1}% of cell = {:.3} Mpc)",
             target_disp, 30.0, spacing);
    println!("  Scale factor: {:.6e} (max_disp_raw={:.6e} → {:.4} Mpc)",
             scale, max_disp, target_disp);

    // Velocity scaling: v_pec = a × H(z) × ψ_phys
    // = a × (H₀/a^1.5) × ψ = H₀ × sqrt(1+z) × ψ
    let h0_gyr = h0 / MPC_GYR_TO_KMS;  // H₀ in Gyr⁻¹ units
    let vel_scale = h0_gyr * (1.0 + z_init).sqrt();  // = a × H(z_init)
    println!("  Velocity scale: {:.4} Mpc/Gyr ({:.1} km/s per Mpc displacement)",
             vel_scale, vel_scale * MPC_GYR_TO_KMS);

    // Step 5: Build particle arrays
    println!("  Building particle arrays...");
    let mut positions = vec![0.0f64; n_total * 3];
    let mut velocities = vec![0.0f64; n_total * 3];
    let mut signs = vec![0i32; n_total];

    // Sign assignment — Janus convention: N+ = N_total/(1+μ), rest is m-
    let n_positive: usize = (n_total as f64 / (1.0 + mu)).round() as usize;
    let n_negative: usize = n_total - n_positive;
    println!("  Sign assignment (Janus, μ={}):", mu);
    println!("    N+ = N/(1+μ) = {}", n_positive);
    println!("    N- = Nμ/(1+μ) = {}", n_negative);
    let mut sign_indices: Vec<usize> = (0..n_total).collect();
    let mut rng_sign = StdRng::seed_from_u64(SEED_IC + 12345);
    sign_indices.shuffle(&mut rng_sign);
    let mut sign_by_idx = vec![-1i32; n_total];
    for &idx in &sign_indices[..n_positive] {
        sign_by_idx[idx] = 1;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // IC generation: PURE RANDOM positions + CIC-interpolated Zel'dovich displacement
    //
    // Historiquement validé dans le run 40M_v3 (février 2026, preprint).
    // Une grille régulière + jitter produit encore un motif visible à step 100.
    // La seule méthode qui a fonctionné sans artefact de grille est :
    //   (1) positions tirées uniformément dans [-L/2, L/2]³
    //   (2) ψ(x, y, z) interpolé depuis le champ ψ grille par CIC trilinéaire
    //
    // Le spectre de puissance des perturbations est préservé (les modes FFT de
    // ψ ne sont pas dégradés par l'interpolation CIC sur N particules, qui lisse
    // juste à l'échelle sous-cellule).
    // ═══════════════════════════════════════════════════════════════════════
    println!("  Using PURE RANDOM positions + CIC-interpolated Zel'dovich ψ");
    println!("    (grid+jitter produit encore des artefacts — approche validée run 40M_v3)");

    let mut rng_pos = StdRng::seed_from_u64(SEED_IC + 67890);

    // Helper : interpolation CIC trilinéaire du champ ψ aux coordonnées (x, y, z)
    // avec conditions périodiques.
    // FIX Phase 10: ψ est maintenant sur grille n_fft³ (×2 padding)
    // idx = iz * n_fft * n_fft + iy * n_fft + ix, coordonnées centrées [-L/2, L/2].
    let cic_interp = |field: &[f64], x: f64, y: f64, z: f64| -> f64 {
        // Coordonnées en unités de cellule FFT, repère [0, n_fft)
        let fx = ((x + half_box) / spacing_fft) % (n_fft as f64);
        let fy = ((y + half_box) / spacing_fft) % (n_fft as f64);
        let fz = ((z + half_box) / spacing_fft) % (n_fft as f64);
        let fx = if fx < 0.0 { fx + n_fft as f64 } else { fx };
        let fy = if fy < 0.0 { fy + n_fft as f64 } else { fy };
        let fz = if fz < 0.0 { fz + n_fft as f64 } else { fz };

        let ix0 = fx.floor() as usize % n_fft;
        let iy0 = fy.floor() as usize % n_fft;
        let iz0 = fz.floor() as usize % n_fft;
        let ix1 = (ix0 + 1) % n_fft;
        let iy1 = (iy0 + 1) % n_fft;
        let iz1 = (iz0 + 1) % n_fft;

        let dx = fx - fx.floor();
        let dy = fy - fy.floor();
        let dz = fz - fz.floor();

        let idx = |i: usize, j: usize, k: usize| k * n_fft * n_fft + j * n_fft + i;

        // 8 coins
        let c000 = field[idx(ix0, iy0, iz0)];
        let c100 = field[idx(ix1, iy0, iz0)];
        let c010 = field[idx(ix0, iy1, iz0)];
        let c110 = field[idx(ix1, iy1, iz0)];
        let c001 = field[idx(ix0, iy0, iz1)];
        let c101 = field[idx(ix1, iy0, iz1)];
        let c011 = field[idx(ix0, iy1, iz1)];
        let c111 = field[idx(ix1, iy1, iz1)];

        // Interpolation trilinéaire
        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;
        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;
        c0 * (1.0 - dz) + c1 * dz
    };

    // ═══════════════════════════════════════════════════════════════════════
    // FIX Phase 11: offset aléatoire de la grille ψ
    // Les positions particules restent inchangées, mais on décale où on
    // lit ψ dans la grille FFT. Casse la corrélation entre la grille FFT
    // et le spacing moyen des particules m-.
    // ═══════════════════════════════════════════════════════════════════════
    let offset_x = rng_pos.random::<f64>() * spacing_fft;
    let offset_y = rng_pos.random::<f64>() * spacing_fft;
    let offset_z = rng_pos.random::<f64>() * spacing_fft;
    println!("  FIX Phase 11 offset aléatoire grille ψ :");
    println!("    spacing_fft = {:.4} Mpc", spacing_fft);
    println!("    offset      = ({:.4}, {:.4}, {:.4}) Mpc", offset_x, offset_y, offset_z);

    for i in 0..n_total {
        // Position aléatoire uniforme dans [-L/2, L/2]³
        let x0 = (rng_pos.random::<f64>() - 0.5) * l_box;
        let y0 = (rng_pos.random::<f64>() - 0.5) * l_box;
        let z0 = (rng_pos.random::<f64>() - 0.5) * l_box;

        // Interpoler ψ à cette position décalée (CIC) — Phase 11 fix
        // cic_interp gère les conditions périodiques automatiquement
        let psi_xi = cic_interp(&psi_x, x0 + offset_x, y0 + offset_y, z0 + offset_z) * scale;
        let psi_yi = cic_interp(&psi_y, x0 + offset_x, y0 + offset_y, z0 + offset_z) * scale;
        let psi_zi = cic_interp(&psi_z, x0 + offset_x, y0 + offset_y, z0 + offset_z) * scale;

        // Position finale : random + ψ, avec conditions périodiques
        positions[i * 3]     = ((x0 + psi_xi + half_box) % l_box + l_box) % l_box - half_box;
        positions[i * 3 + 1] = ((y0 + psi_yi + half_box) % l_box + l_box) % l_box - half_box;
        positions[i * 3 + 2] = ((z0 + psi_zi + half_box) % l_box + l_box) % l_box - half_box;

        // Vitesse Zel'dovich : v = H(z) × ψ_phys (même ψ interpolé)
        velocities[i * 3]     = psi_xi * vel_scale;
        velocities[i * 3 + 1] = psi_yi * vel_scale;
        velocities[i * 3 + 2] = psi_zi * vel_scale;

        // Signe Janus (pré-calculé plus haut)
        signs[i] = sign_by_idx[i];
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

    // Compute ρ_mean_plus for reference (informational logging)
    let rho_crit = 2.775e11 * (args.h0 / 100.0).powi(2);  // M☉/Mpc³
    let rho_mean_plus = args.omega_b * rho_crit;  // ≈ 6.78e9 M☉/Mpc³

    let n_particles_init = args.n_grid * args.n_grid * args.n_grid;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           JANUS ADAPTIVE ZOOM — Production Run (v8)                      ║");
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
    println!("║  ADAPTIVE SPLITTING (v8: ρ_plus threshold + spatial zoom):");
    println!("║    Check every {} steps, max_split_level = {}", args.steps_check, args.max_split_level);
    println!("║    ρ_mean_plus = {:.2e} M☉/Mpc³", rho_mean_plus);
    println!("║    δ_split_L1 = {:.2e} M☉/Mpc³ (level 0→1)", args.delta_split_l1);
    println!("║    δ_split_L2 = {:.2e} M☉/Mpc³ (level 1→2)", args.delta_split_l2);
    if args.zoom_cube_size > 0.0 {
        println!("║    ZOOM CUBE: [{:.0}, +{:.0}]³ Mpc (size={} Mpc)",
            -args.zoom_cube_size / 2.0, args.zoom_cube_size / 2.0, args.zoom_cube_size);
    } else {
        println!("║    ZOOM: disabled (splits anywhere based on ρ_plus)");
    }
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  OUTPUT: {} (v3 format)", args.out_dir);
    println!("║    Snapshots every {} steps, Label: {}", args.snap_interval, args.run_label);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", args.out_dir)).expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/frames", args.out_dir)).expect("Failed to create frames dir");

    // Generate Zel'dovich ICs
    let (positions, velocities, signs) = generate_zeldovich_ics(args.n_grid, args.l_box, args.z_init, args.h0, args.mu);

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
    // COMPILE KERNELS ONCE at startup — reuse for all splits
    let cuda_device = GpuNBodySimulation::compile_kernels()
        .expect("Failed to compile CUDA kernels");
    println!("  ✓ CUDA kernels compiled (one-time)");

    // Use new_with_state() like janus_baryonic_calibrated (masses = 1.0)
    // new_with_state_and_masses() only needed after splits (masses < 1.0)
    let (gpu_pos, gpu_vel, gpu_signs, _gpu_masses) = state.to_gpu_arrays();
    let mut gpu_sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, args.l_box,
        gpu_pos, gpu_vel, gpu_signs
    ).expect("Failed to create GPU simulation");
    gpu_sim.set_theta(args.theta);
    gpu_sim.set_softening(args.eps_plus);

    // Correct mass factor for Janus physics: omega_b * (1+mu) instead of default omega_m = 0.3
    // For mu=19, omega_b=0.05: factor = 0.05 * 20 / 0.3 = 3.33
    let janus_mass_factor = args.omega_b * (1.0 + args.mu) / 0.3;
    gpu_sim.set_mass_factor(janus_mass_factor);

    // Dynamic c_ratio
    let c_ratio_sq_init = CoupledFriedmann::c_ratio_sq_at_z(args.z_init, ETA);
    gpu_sim.set_c_ratio(c_ratio_sq_init.sqrt());
    println!("  GPU ready: {} particles, θ={}", state.particles.len(), args.theta);
    // Initialiser la physique baryonique (cooling + SF) - réutilise le device compilé
    let n_plus_init = state.particles.iter().filter(|p| p.sign == 1).count();
    let mut gpu_cooling = GpuCooling::new(
        cuda_device.clone(),  // Clone pour garder le device pour les splits
        n_plus_init,
        args.l_box,
        state.m_plus_base,
    ).expect("Failed to create GpuCooling");

    let signs_plus: Vec<i32> = vec![1i32; n_plus_init];
    gpu_cooling.init_from_temperature(T_INIT_PLUS, T_INIT_PLUS, &signs_plus)
        .expect("Failed to init cooling temperatures");
    println!("  ✓ Physique baryonique initialisée (T_init = {} K)", T_INIT_PLUS);

    let mut n_stars: usize = 0;
    let mut sfr: f64 = 0.0;

    // RNG for splits
    let mut rng_split = StdRng::seed_from_u64(SEED_IC + 1000);

    // Cosmological state
    let mut a = 1.0 / (1.0 + args.z_init);
    let mut t_gyr = 0.5;  // Approximate cosmic time at z=10

    // CSV output
    let csv_path = format!("{}/time_series.csv", args.out_dir);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,t_Gyr,z,a,N_total,N_hr,split_max,rho_max,v_rms,rho_plus_max").unwrap();

    println!("\n[3/5] Starting main loop (z={} → z={})...\n", args.z_init, args.z_final);

    let mut step = 0;
    loop {
        let z = 1.0 / a - 1.0;

        // Stop condition
        if z < args.z_final {
            println!("\n  Reached z_final = {:.2} at step {}", args.z_final, step);
            break;
        }

        // H(a) via Janus two-phase expansion (Petit 2014/2018)
        let h = compute_hubble_janus(a, args.h0);

        // Metrics
        let do_metric = step % METRIC_INTERVAL == 0;
        let do_snapshot = step % args.snap_interval == 0;
        let do_split = step % args.steps_check == 0 && step > 0;

        if do_metric || do_snapshot || do_split {
            // Sync GPU → CPU
            let pos = gpu_sim.get_positions().unwrap();
            let vel = gpu_sim.get_velocities().unwrap();
            state.sync_from_gpu(&pos, &vel);

            // Compute densities (separated by sign for proper Janus splitting)
            let (densities_plus, densities_minus, rho_plus_max, _rho_minus_max) =
                compute_densities_split(&state.particles, args.l_box);
            let rho_max = densities_plus.iter().zip(densities_minus.iter())
                .map(|(a, b)| a + b)
                .fold(0.0f64, f64::max);

            // v_rms
            let v_rms: f64 = {
                let sum: f64 = state.particles.iter()
                    .map(|p| (p.vel[0]*p.vel[0] + p.vel[1]*p.vel[1] + p.vel[2]*p.vel[2]) as f64)
                    .sum();
                (sum / state.particles.len() as f64).sqrt() * MPC_GYR_TO_KMS
            };

            // Adaptive split check (uses ρ_plus for threshold, not ρ_total)
            if do_split {
                let n_before = state.particles.len();

                // SAVE thermal state BEFORE split and drop, so it survives the reallocation.
                let u_old = gpu_cooling.get_internal_energy()
                    .expect("Failed to read internal energy before split");

                let (n_new, u_new) = adaptive_split_check_with_thresholds(
                    &mut state,
                    &densities_plus,          // Use ρ_plus for m+ splitting threshold
                    args.delta_split_l1,
                    args.delta_split_l2,
                    args.max_split_level,
                    args.zoom_cube_size,
                    &mut rng_split,
                    &u_old,
                );

                if n_new > 0 {
                    println!("  🔬 Step {}: Split +{} particles, N={} → {}",
                        step, n_new, n_before, state.particles.len());

                    // Recreate GPU simulation with new particle count and per-particle masses
                    let (new_pos, new_vel, new_signs, new_masses) = state.to_gpu_arrays();
                    let (np, nm) = state.counts();

                    // CRITICAL: Drop old GPU sim BEFORE creating new one to free VRAM
                    // Without this, both sims exist simultaneously during creation = OOM
                    drop(gpu_sim);

                    // Use pre-compiled device — NO PTX recompilation!
                    gpu_sim = GpuNBodySimulation::new_with_state_and_masses_with_device(
                        cuda_device.clone(), np, nm, args.l_box, new_pos, new_vel, new_signs, new_masses
                    ).expect("Failed to recreate GPU simulation after split");
                    gpu_sim.set_theta(args.theta);
                    gpu_sim.set_softening(args.eps_plus);
                    gpu_sim.set_c_ratio(CoupledFriedmann::c_ratio_sq_at_z(z, ETA).sqrt());
                    // Re-apply Janus mass factor (to_gpu_arrays returns normalized masses)
                    gpu_sim.set_mass_factor(janus_mass_factor);

                    // Recreate GpuCooling with new n_plus, THEN RESTORE thermal state.
                    // Daughters inherit parent u (already done inside adaptive_split_check).
                    let n_plus_new = state.particles.iter().filter(|p| p.sign == 1).count();
                    assert_eq!(n_plus_new, u_new.len(),
                        "u_new length mismatch: got {} for {} m+ particles", u_new.len(), n_plus_new);
                    gpu_cooling = GpuCooling::new(
                        cuda_device.clone(), n_plus_new, args.l_box, state.m_plus_base
                    ).expect("Failed to recreate GpuCooling after split");
                    let signs_plus_new: Vec<i32> = vec![1i32; n_plus_new];
                    gpu_cooling.upload_signs(&signs_plus_new)
                        .expect("Failed to upload signs after split");
                    gpu_cooling.set_internal_energy(&u_new)
                        .expect("Failed to restore internal energy after split");
                }
            }

            let n_hr = state.particles.iter().filter(|p| p.split_level > 0).count();
            let split_max = state.max_split_level();

            if do_metric {
                writeln!(csv, "{},{:.6},{:.4},{:.6},{},{},{},{:.3e},{:.2},{:.3e}",
                    step, t_gyr, z, a, state.particles.len(), n_hr, split_max, rho_max, v_rms, rho_plus_max).unwrap();

                if step % 100 == 0 || do_split {
                    println!("  Step {:5} | z={:.3} | N={:>8} | N_hr={:>6} | ρ_max={:.2e} | ρ+_max={:.2e} | v_rms={:.1} km/s",
                        step, z, state.particles.len(), n_hr, rho_max, rho_plus_max, v_rms);
                }
            }

            // Snapshot
            if do_snapshot {
                let snap_path = Path::new(&args.out_dir).join("snapshots").join(format!("snap_{:05}.bin", step));
                save_snapshot(&snap_path, &state, a, t_gyr, n_stars as u64, sfr, rho_max);

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

        // Time integration (dtau_per_dt=1.0 enables Hubble friction)
        gpu_sim.step_with_expansion_dkd_gpu(args.dt_max, a, h, 1.0).unwrap();
        let da = a * h * args.dt_max;
        a += da;
        t_gyr += args.dt_max;

        // ═══════════════════════════════════════════════════════
        // PHYSIQUE BARYONIQUE (m+ uniquement, chaque step)
        // ═══════════════════════════════════════════════════════
        {
            let pos = gpu_sim.get_positions().unwrap();
            let signs_data = gpu_sim.signs();

            // Densités locales pour le refroidissement
            let overdensities = compute_local_overdensities(
                &pos, &signs_data, 32, args.l_box
            );

            // Le kernel GpuCooling attend des densités en M☉/Mpc³ et applique
            // rho_to_nh (defini dans cooling_gpu.rs) pour convertir en nH [cm⁻³].
            // Ne PAS pré-convertir ici, sinon double conversion → nH ~1e-17× trop petit
            // et la SF ne se déclenche jamais.
            //
            // ρ_baryon(z) en M☉/Mpc³ :
            //   ρ_crit_0 = 2.775e11 * h² M☉/Mpc³
            //   ρ_mean_b(z) = Ω_b * ρ_crit_0 * (1+z)³   (en comobile, constante ; en propre, ×(1+z)³)
            let rho_crit_0 = 2.775e11_f64 * (args.h0 / 100.0).powi(2);  // M☉/Mpc³
            let rho_mean_b_z = args.omega_b * rho_crit_0 * (1.0 + z).powi(3);  // M☉/Mpc³
            let densities: Vec<f64> = overdensities.iter()
                .map(|&od| od * rho_mean_b_z)
                .collect();

            // Refroidissement GPU
            gpu_cooling.upload_densities(&densities)
                .expect("Failed to upload densities");
            gpu_cooling.apply_cooling(args.dt_max, z)
                .expect("GPU cooling failed");

            // Formation stellaire
            let new_stars = gpu_cooling.apply_star_formation(args.dt_max)
                .unwrap_or(0);
            n_stars += new_stars as usize;
            sfr = (new_stars as f64) * state.m_plus_base / args.dt_max;

            if new_stars > 0 {
                println!("    ★ Step {}: {} nouvelles étoiles, N★={}", step, new_stars, n_stars);
            }
        }

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

// ═══════════════════════════════════════════════════════════════════════════
// UNIT TESTS — Janus expansion (Petit 2014/2018)
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hubble_today() {
        let h = compute_hubble_janus(1.0, 69.9);
        let h_kms_mpc = h * MPC_GYR_TO_KMS;

        // Must give 69.9 km/s/Mpc within 0.01%
        assert!(
            (h_kms_mpc - 69.9).abs() < 0.01,
            "H(a=1) = {} km/s/Mpc, expected 69.9", h_kms_mpc
        );
    }

    #[test]
    fn test_hubble_matter_era() {
        let h0 = 69.9;

        // Reference values computed in Python (precision 0.01%)
        let test_cases = vec![
            // (z,    expected_H_Gyr_inv,   tol)
            (0.0,    0.071487,             1e-4),
            (0.1,    0.077186,             1e-4),
            (0.5,    0.097594,             1e-4),
            (1.0,    0.117986,             1e-4),
            (2.0,    0.142492,             1e-4),
            (3.0,    0.143787,             1e-4),
            (4.0,    0.107606,             1e-4),
        ];

        for (z, expected, tol) in test_cases {
            let a = 1.0 / (1.0 + z);
            let h = compute_hubble_janus(a, h0);
            assert!(
                (h - expected).abs() < tol,
                "H(z={}): got {}, expected {}, diff = {}",
                z, h, expected, (h - expected).abs()
            );
        }
    }

    #[test]
    fn test_hubble_radiation_era() {
        let h0 = 69.9;
        let h0_gyr = h0 / MPC_GYR_TO_KMS;

        // Radiative phase: H = H₀ / a^(3/2)
        let test_zs = vec![5.0, 6.0, 8.0, 10.0];

        for z in test_zs {
            let a = 1.0 / (1.0 + z);
            let h = compute_hubble_janus(a, h0);
            let expected = h0_gyr / a.powf(1.5);
            assert!(
                (h - expected).abs() < 1e-7,
                "H(z={}) radiative: got {}, expected {} (EdS)",
                z, h, expected
            );
        }
    }

    #[test]
    fn test_phase_transition() {
        let h0 = 69.9;

        // Just below α²: radiative phase, H large
        let a_below = ALPHA_SQ_JANUS - 1e-6;
        let h_below = compute_hubble_janus(a_below, h0);
        assert!(
            h_below > 0.9,
            "Just below transition (a={}): H = {} should be ~0.92 (radiation)",
            a_below, h_below
        );

        // At α² exactly: matter phase, μ_p=0, H ≈ 0
        let h_at = compute_hubble_janus(ALPHA_SQ_JANUS, h0);
        assert!(
            h_at.abs() < 1e-5,
            "At transition (a=α²): H = {} should be ~0 (μ_p=0)", h_at
        );

        // Just above α²: matter phase, H small positive
        let a_above = ALPHA_SQ_JANUS + 1e-6;
        let h_above = compute_hubble_janus(a_above, h0);
        assert!(
            h_above > 0.0 && h_above < 0.01,
            "Just above transition (a={}): H = {} should be small positive",
            a_above, h_above
        );
    }

    #[test]
    fn test_cosmic_age() {
        let h0 = 69.9;

        // Integrate ∫_{α²}^{1} da/(aH) should give 15.87 Gyr
        let n_steps = 10000;
        let log_a_start = ALPHA_SQ_JANUS.ln() + 1e-6;  // avoid exact point
        let log_a_end = 0.0;  // ln(1) = 0
        let dlog = (log_a_end - log_a_start) / n_steps as f64;

        let mut t_integral = 0.0;
        for i in 0..n_steps {
            let log_a_mid = log_a_start + (i as f64 + 0.5) * dlog;
            let a = log_a_mid.exp();
            let h = compute_hubble_janus(a, h0);
            // dt = da/(aH) = d(ln a)/H
            t_integral += dlog / h;
        }

        println!("Cosmic age (matter era only): {:.4} Gyr", t_integral);
        assert!(
            (t_integral - 15.87).abs() < 0.05,
            "Cosmic age: got {} Gyr, expected 15.87 Gyr", t_integral
        );
    }

    #[test]
    fn test_a_progression_through_transition() {
        let h0 = 69.9;

        // Simulate leapfrog integration around the junction
        let dt = 0.001;  // Gyr
        let mut a = 1.0 / 11.0;  // z = 10
        let mut steps = 0;
        let max_steps = 50_000;  // safety limit

        while a < 1.0 && steps < max_steps {
            let h = compute_hubble_janus(a, h0);
            // da = a * H * dt
            a += a * h * dt;
            steps += 1;

            // Sanity: no NaN or Inf
            assert!(a.is_finite(), "a became {} at step {}", a, steps);
            assert!(h.is_finite(), "H became {} at step {} (a={})", h, steps, a);
        }

        println!("Reached a={} in {} steps", a, steps);
        assert!(steps < max_steps, "Did not reach a=1 in {} steps", max_steps);
    }

    #[test]
    fn test_reference_table() {
        let h0 = 69.9;

        // Table of precisely computed values in Python
        // Format: (a, H_in_Gyr_inv)
        let reference = vec![
            // Radiative phase (a < α² = 0.1815)
            (0.09091, 2.608052),  // z=10
            (0.11111, 1.930149),  // z=8
            (0.14286, 1.323958),  // z=6
            (0.16667, 1.050640),  // z=5
            // Matter phase (a > α²)
            (0.20000, 0.107606),  // z=4
            (0.25000, 0.143787),  // z=3
            (0.33333, 0.142492),  // z=2
            (0.50000, 0.117986),  // z=1
            (0.66667, 0.097594),  // z=0.5
            (0.90909, 0.077186),  // z=0.1
            (1.00000, 0.071487),  // z=0
        ];

        for (a, expected) in reference {
            let h = compute_hubble_janus(a, h0);
            let rel_err = (h - expected).abs() / expected;
            assert!(
                rel_err < 1e-3,
                "a={}: H={}, expected={}, rel_err={}", a, h, expected, rel_err
            );
        }
    }
}
