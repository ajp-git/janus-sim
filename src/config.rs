//! Configuration module for Janus optimization runs
//!
//! Supports YAML config files for trichotomy parameter search.
//! See `janus_optimisation_plan.md` for usage.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Main configuration struct for Janus simulation runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JanusConfig {
    pub simulation: SimulationConfig,
    pub physics: PhysicsConfig,
    pub pm_grid: PmGridConfig,
    pub output: OutputConfig,
}

/// Simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Box size in comoving Mpc
    pub box_size_mpc: f64,
    /// Total number of particles (will be split by eta)
    pub n_particles: usize,
    /// Number of simulation steps
    pub n_steps: usize,
    /// Starting redshift
    pub z_start: f64,
    /// Ending redshift
    pub z_end: f64,
    /// Random seed for reproducible ICs
    pub seed: u64,
    /// Barnes-Hut opening angle (default 0.7)
    #[serde(default = "default_theta")]
    pub theta: f64,
    /// Softening length in Mpc (default: box/N^(1/3)/30)
    #[serde(default)]
    pub softening_mpc: Option<f64>,
}

fn default_theta() -> f64 { 0.7 }

/// Physics parameters for Janus model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Density ratio η = |ρ̄|/ρ = n_negative/n_positive
    /// Exploration range: 0.5 - 1.5
    pub eta: f64,

    /// Base screening length in Mpc (for variable λ screening)
    /// λ_eff(x) = lambda_base / sqrt(rho_local / rho_mean)
    /// Set to 0.0 to disable screening (pure Janus)
    #[serde(default = "default_lambda_base")]
    pub lambda_base_mpc: f64,

    /// Smoothing radius for local density calculation (Mpc)
    /// Used to compute rho_local for variable screening
    #[serde(default = "default_r_smooth")]
    pub r_smooth_mpc: f64,

    /// Floor for rho_local/rho_mean to avoid divergence in voids
    /// λ_eff is capped at lambda_base / sqrt(lambda_floor)
    #[serde(default = "default_lambda_floor")]
    pub lambda_floor: f64,

    /// Enable Hubble friction (cosmological expansion)
    #[serde(default = "default_true")]
    pub hubble_friction: bool,

    /// Cross-force asymmetry factor (A7): m- receives this × force
    /// Default 1.0 = symmetric. Use 2.0 for faster m- expulsion
    #[serde(default = "default_one")]
    pub cross_force_asymmetry: f64,

    /// Sigmoid activation: redshift at which cross-force starts (Z1)
    /// None = always active. Some(2.0) = activate at z=2
    #[serde(default)]
    pub cross_force_z_start: Option<f64>,

    /// Sigmoid activation width (Z1)
    /// Transition happens over z_start ± z_width
    #[serde(default = "default_z_width")]
    pub cross_force_z_width: f64,
}

fn default_lambda_base() -> f64 { 0.0 }  // Disabled by default
fn default_r_smooth() -> f64 { 5.0 }
fn default_lambda_floor() -> f64 { 0.01 }
fn default_true() -> bool { true }
fn default_one() -> f64 { 1.0 }
fn default_z_width() -> f64 { 0.5 }

/// PM grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmGridConfig {
    /// Number of cells per dimension (typical: 128 or 256)
    pub n_cells: usize,
    /// Minimum k mode to keep (0=all, 2=no dipole)
    #[serde(default)]
    pub k_min: usize,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory (relative to project root)
    pub dir: String,
    /// Redshifts at which to save full snapshots
    #[serde(default = "default_snapshot_redshifts")]
    pub snapshot_redshifts: Vec<f64>,
    /// Save snapshot every N steps (for video generation)
    /// If set, overrides snapshot_redshifts
    #[serde(default)]
    pub snapshot_every_steps: Option<usize>,
    /// Compute metrics every N steps
    #[serde(default = "default_metrics_interval")]
    pub metrics_every_steps: usize,
    /// Save binary snapshots for rendering
    #[serde(default = "default_true")]
    pub save_snapshots: bool,
}

fn default_snapshot_redshifts() -> Vec<f64> { vec![5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0] }
fn default_metrics_interval() -> usize { 25 }

impl JanusConfig {
    /// Load configuration from YAML file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_yaml::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Save configuration to YAML file
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let contents = serde_yaml::to_string(self)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        fs::write(path.as_ref(), contents)
            .map_err(|e| ConfigError::IoError(e.to_string()))
    }

    /// Create default config for Tour 1 with given eta
    pub fn tour1_default(eta: f64, run_label: &str) -> Self {
        Self {
            simulation: SimulationConfig {
                box_size_mpc: 150.0,
                n_particles: 200_000,
                n_steps: 500,
                z_start: 5.0,
                z_end: 1.5,
                seed: 42,
                theta: 0.7,
                softening_mpc: None,
            },
            physics: PhysicsConfig {
                eta,
                lambda_base_mpc: 30.0,
                r_smooth_mpc: 5.0,
                lambda_floor: 0.01,
                hubble_friction: true,
                cross_force_asymmetry: 1.0,
                cross_force_z_start: None,
                cross_force_z_width: 0.5,
            },
            pm_grid: PmGridConfig {
                n_cells: 128,
                k_min: 2,
            },
            output: OutputConfig {
                dir: format!("output/{}", run_label),
                snapshot_redshifts: vec![5.0, 3.0, 2.0, 1.5],
                snapshot_every_steps: None,
                metrics_every_steps: 25,
                save_snapshots: true,
            },
        }
    }

    /// Compute n_positive and n_negative from eta
    pub fn particle_counts(&self) -> (usize, usize) {
        let n = self.simulation.n_particles;
        let eta = self.physics.eta;
        let n_positive = (n as f64 / (1.0 + eta)).round() as usize;
        let n_negative = n - n_positive;
        (n_positive, n_negative)
    }

    /// Compute softening length (default: box/N^(1/3)/30)
    pub fn softening(&self) -> f64 {
        self.simulation.softening_mpc.unwrap_or_else(|| {
            let n = self.simulation.n_particles as f64;
            self.simulation.box_size_mpc / n.powf(1.0/3.0) / 30.0
        })
    }

    /// Compute mean inter-particle separation
    pub fn mean_separation(&self) -> f64 {
        let n = self.simulation.n_particles as f64;
        self.simulation.box_size_mpc / n.powf(1.0/3.0)
    }

    /// Compute cross-force activation factor at given redshift (Z1 sigmoid)
    /// Returns 0.0 for z >> z_start, 1.0 for z << z_start
    pub fn cross_force_factor(&self, z: f64) -> f64 {
        match self.physics.cross_force_z_start {
            None => 1.0,  // Always active
            Some(z0) => {
                let dz = self.physics.cross_force_z_width;
                1.0 / (1.0 + ((z - z0) / dz).exp())
            }
        }
    }
}

/// Configuration error types
#[derive(Debug)]
pub enum ConfigError {
    IoError(String),
    ParseError(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(e) => write!(f, "IO error: {}", e),
            ConfigError::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tour1_defaults() {
        let cfg = JanusConfig::tour1_default(1.0, "test");
        assert_eq!(cfg.physics.eta, 1.0);
        assert_eq!(cfg.simulation.n_particles, 200_000);
        assert_eq!(cfg.physics.lambda_base_mpc, 30.0);

        let (np, nm) = cfg.particle_counts();
        assert_eq!(np, 100_000);
        assert_eq!(nm, 100_000);
    }

    #[test]
    fn test_particle_counts_eta_05() {
        let cfg = JanusConfig::tour1_default(0.5, "test");
        let (np, nm) = cfg.particle_counts();
        // eta=0.5 → n+/(1+0.5) = n+/1.5, so n+ = 2/3 * N
        assert_eq!(np, 133_333);
        assert_eq!(nm, 66_667);
    }
}
