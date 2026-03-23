//! In-simulation metrics for Janus optimization
//!
//! Computes segregation, filament, and void metrics during simulation.
//! Results are saved to metrics.jsonl for post-processing.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Metrics computed at each measurement step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub step: u32,
    pub redshift: f64,
    pub scale_factor: f64,

    // Segregation metrics
    pub s_segregation: f64,       // COM separation / box_size
    pub n_halos_plus: u32,        // FOF halo count (m+)
    pub n_halos_minus: u32,       // FOF halo count (m-)

    // Filament metrics
    pub filament_mean_mpc: f64,   // Mean filament length
    pub filament_max_mpc: f64,    // Max filament length
    pub filament_count: u32,      // Number of detected filaments
    pub fil_matter_fraction: f64, // Fraction of mass in filaments

    // Void metrics
    pub void_fraction: f64,       // Fraction of cells with rho < 0.1 * rho_mean
    pub void_mode_mpc: f64,       // Typical void size

    // Power spectrum
    pub pk_slope: f64,            // Log-log slope at k=0.05-0.5 Mpc^-1
    pub pk_excess_lcdm: f64,      // P_janus / P_lcdm at k=0.1 Mpc^-1

    // Stability metrics
    pub v_rms: f64,               // RMS velocity (km/s)
    pub v_max: f64,               // Max velocity (km/s)
    pub e_kinetic: f64,           // Total kinetic energy (normalized)
    pub ke_ratio: f64,            // KE / KE_initial
}

impl StepMetrics {
    /// Create metrics with basic values (for quick computation)
    pub fn from_basic(
        step: u32,
        redshift: f64,
        s_segregation: f64,
        v_rms: f64,
        v_max: f64,
        ke_ratio: f64,
    ) -> Self {
        Self {
            step,
            redshift,
            scale_factor: 1.0 / (1.0 + redshift),
            s_segregation,
            n_halos_plus: 0,
            n_halos_minus: 0,
            filament_mean_mpc: 0.0,
            filament_max_mpc: 0.0,
            filament_count: 0,
            fil_matter_fraction: 0.0,
            void_fraction: 0.0,
            void_mode_mpc: 0.0,
            pk_slope: 0.0,
            pk_excess_lcdm: 0.0,
            v_rms,
            v_max,
            e_kinetic: 0.0,
            ke_ratio,
        }
    }

    /// Compute composite score for optimization
    ///
    /// Formula (FROZEN after Tour 1 calibration):
    ///   score = 0.35 × min(S/0.5, 1)               - segregation
    ///         + 0.30 × min(filament_mean/10, 1)    - filament length
    ///         + 0.20 × min(fil_matter/0.15, 1)     - matter in filaments
    ///         + 0.15 × void_penalty               - not too empty
    pub fn composite_score(&self) -> f64 {
        // S1: Segregation - target S > 0.5
        let s1 = (self.s_segregation / 0.5).min(1.0);

        // S2: Filaments - target mean length > 10 Mpc
        let s2 = (self.filament_mean_mpc / 10.0).min(1.0);

        // S3: Matter in filaments - target > 15% (DESI/Euclid ~ 18-25%)
        let s3 = (self.fil_matter_fraction / 0.15).min(1.0);

        // S4: Voids - target void_fraction < 0.70 (not a ghost universe)
        let s4 = if self.void_fraction < 0.70 {
            1.0
        } else {
            (1.0 - (self.void_fraction - 0.70) / 0.25).max(0.0)
        };

        0.35 * s1 + 0.30 * s2 + 0.20 * s3 + 0.15 * s4
    }
}

/// Metrics writer for JSONL output
pub struct MetricsWriter {
    writer: BufWriter<File>,
}

impl MetricsWriter {
    /// Create new metrics writer
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Write a single metrics entry
    pub fn write(&mut self, metrics: &StepMetrics) -> std::io::Result<()> {
        let json = serde_json::to_string(metrics)?;
        writeln!(self.writer, "{}", json)?;
        self.writer.flush()
    }
}

/// Compute segregation from particle positions
/// Returns S = |COM+ - COM-| / box_size with periodic boundary handling
pub fn compute_segregation(
    pos_plus: &[[f64; 3]],
    pos_minus: &[[f64; 3]],
    box_size: f64,
) -> f64 {
    if pos_plus.is_empty() || pos_minus.is_empty() {
        return 0.0;
    }

    // Compute COM with periodic boundary (minimum image convention)
    let com_plus = periodic_com(pos_plus, box_size);
    let com_minus = periodic_com(pos_minus, box_size);

    // Distance between COMs with periodic wrapping
    let mut d2 = 0.0;
    for i in 0..3 {
        let mut d = com_plus[i] - com_minus[i];
        // Minimum image
        if d > box_size / 2.0 { d -= box_size; }
        if d < -box_size / 2.0 { d += box_size; }
        d2 += d * d;
    }

    d2.sqrt() / box_size
}

/// Compute center of mass with periodic boundaries
fn periodic_com(positions: &[[f64; 3]], box_size: f64) -> [f64; 3] {
    // Use angle-based method for periodic COM
    let mut sum_cos = [0.0; 3];
    let mut sum_sin = [0.0; 3];
    let two_pi_over_l = 2.0 * std::f64::consts::PI / box_size;

    for pos in positions {
        for i in 0..3 {
            let theta = pos[i] * two_pi_over_l;
            sum_cos[i] += theta.cos();
            sum_sin[i] += theta.sin();
        }
    }

    let n = positions.len() as f64;
    let mut com = [0.0; 3];
    for i in 0..3 {
        let avg_cos = sum_cos[i] / n;
        let avg_sin = sum_sin[i] / n;
        let theta = avg_sin.atan2(avg_cos);
        com[i] = theta / two_pi_over_l;
        // Wrap to [0, box_size)
        if com[i] < 0.0 { com[i] += box_size; }
    }
    com
}

/// Compute void fraction from density grid
/// void_fraction = fraction of cells with rho < threshold * rho_mean
pub fn compute_void_fraction(density_grid: &[f64], threshold: f64) -> f64 {
    if density_grid.is_empty() {
        return 0.0;
    }

    let n = density_grid.len() as f64;
    let rho_mean = density_grid.iter().sum::<f64>() / n;
    let void_threshold = threshold * rho_mean;

    let n_void = density_grid.iter()
        .filter(|&&rho| rho < void_threshold)
        .count();

    n_void as f64 / n
}

/// Simple velocity statistics
pub fn compute_velocity_stats(velocities: &[[f64; 3]]) -> (f64, f64) {
    if velocities.is_empty() {
        return (0.0, 0.0);
    }

    let mut sum_v2 = 0.0;
    let mut max_v = 0.0;

    for v in velocities {
        let v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        sum_v2 += v2;
        if v2 > max_v { max_v = v2; }
    }

    let v_rms = (sum_v2 / velocities.len() as f64).sqrt();
    let v_max = max_v.sqrt();

    (v_rms, v_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segregation_same_position() {
        let pos = [[50.0, 50.0, 50.0]];
        let s = compute_segregation(&pos, &pos, 100.0);
        assert!(s < 0.01);
    }

    #[test]
    fn test_segregation_opposite_corners() {
        let plus = [[0.0, 0.0, 0.0]];
        let minus = [[50.0, 50.0, 50.0]];
        let s = compute_segregation(&plus, &minus, 100.0);
        // Distance = sqrt(3)*50 ≈ 86.6, S = 0.866
        assert!(s > 0.8 && s < 0.9);
    }

    #[test]
    fn test_void_fraction() {
        let grid = vec![0.0, 0.0, 1.0, 2.0, 3.0];
        // mean = 1.2, threshold 0.1 → 0.12
        // 2 cells < 0.12 → 40%
        let vf = compute_void_fraction(&grid, 0.1);
        assert!((vf - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_composite_score() {
        let m = StepMetrics {
            step: 100,
            redshift: 1.5,
            scale_factor: 0.4,
            s_segregation: 0.5,
            n_halos_plus: 10,
            n_halos_minus: 10,
            filament_mean_mpc: 10.0,
            filament_max_mpc: 20.0,
            filament_count: 5,
            fil_matter_fraction: 0.15,
            void_fraction: 0.5,
            void_mode_mpc: 25.0,
            pk_slope: -2.8,
            pk_excess_lcdm: 1.0,
            v_rms: 200.0,
            v_max: 500.0,
            e_kinetic: 1e10,
            ke_ratio: 1.0,
        };
        // All components at max: 0.35 + 0.30 + 0.20 + 0.15 = 1.0
        let score = m.composite_score();
        assert!((score - 1.0).abs() < 0.01);
    }
}
