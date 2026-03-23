//! Early stopping conditions for Janus optimization runs
//!
//! Checks for numerical instability, structural collapse, and convergence.

use crate::metrics::StepMetrics;

/// Decision from early stopping check
#[derive(Debug, Clone)]
pub enum StopDecision {
    /// Continue simulation
    Continue,
    /// Abort simulation with reason
    Abort { reason: String },
    /// Continue but flag as potential winner
    FlagWinner { score: f64 },
}

/// Check early stopping conditions
///
/// Called every `metrics_every_steps` steps.
pub fn check_early_stop(
    metrics: &StepMetrics,
    history: &[StepMetrics],
    n_steps_total: u32,
) -> StopDecision {

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 1 — Numerical divergence (HARD STOP, priority)
    // ══════════════════════════════════════════════════════════════════

    // 1a. Runaway Janus: v_max absolute - 5000 km/s = physical limit
    //     (cosmological escape velocity ~ 1500-2000 km/s at z=2)
    if metrics.v_max > 5_000.0 {
        return StopDecision::Abort {
            reason: format!(
                "RUNAWAY JANUS: v_max={:.0} km/s > 5000 km/s (step {})",
                metrics.v_max, metrics.step
            ),
        };
    }

    // 1b. NaN or Inf - corrupts GPU snapshot
    if metrics.v_max.is_nan() || metrics.v_max.is_infinite()
        || metrics.e_kinetic.is_nan()
    {
        return StopDecision::Abort {
            reason: format!(
                "NaN/Inf detected: v_max={}, KE={} (step {})",
                metrics.v_max, metrics.e_kinetic, metrics.step
            ),
        };
    }

    // 1c. Kinetic energy explosion - KE_ratio = KE_now / KE_initial
    //     > 1e8 = numerical instability (timestep too large or force divergence)
    if metrics.ke_ratio > 1.0e8 {
        return StopDecision::Abort {
            reason: format!(
                "KE explosion: ke_ratio={:.2e} > 1e8 (step {}) - reduce dt or softening",
                metrics.ke_ratio, metrics.step
            ),
        };
    }

    // 1d. v_max/v_rms ratio - single particle runaway
    //     (ratio > 50 = one particle escaping, not global divergence)
    if metrics.v_rms > 1.0 && metrics.v_max > 50.0 * metrics.v_rms {
        return StopDecision::Abort {
            reason: format!(
                "Particle runaway: v_max/v_rms={:.0} > 50 (step {})",
                metrics.v_max / metrics.v_rms, metrics.step
            ),
        };
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 2 — Structural collapse (step >= 200, z ≈ 3)
    // ══════════════════════════════════════════════════════════════════
    if metrics.step >= 200 {
        if metrics.n_halos_plus < 5 && metrics.n_halos_plus > 0 {
            return StopDecision::Abort {
                reason: format!(
                    "Collapse: only {} halos m+ at z={:.1}",
                    metrics.n_halos_plus, metrics.redshift
                ),
            };
        }
        if metrics.s_segregation < 0.05 && metrics.filament_mean_mpc < 2.0 {
            return StopDecision::Abort {
                reason: format!(
                    "No emerging structure at z={:.1}: S={:.3}, filaments={:.1} Mpc",
                    metrics.redshift, metrics.s_segregation, metrics.filament_mean_mpc
                ),
            };
        }
        if metrics.void_fraction > 0.95 {
            return StopDecision::Abort {
                reason: format!(
                    "Universe too empty at z={:.1}: void_fraction={:.2}",
                    metrics.redshift, metrics.void_fraction
                ),
            };
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 3 — Midpoint decision (step ≈ N/2, z ≈ 2.2)
    // ══════════════════════════════════════════════════════════════════
    let midpoint = n_steps_total / 2;
    if metrics.step >= midpoint && metrics.step < midpoint + 25 {
        let score = metrics.composite_score();
        if score < 0.08 {
            return StopDecision::Abort {
                reason: format!(
                    "Score too low at midpoint: {:.3} < 0.08 (z={:.1})",
                    score, metrics.redshift
                ),
            };
        }
        if score > 0.65 {
            return StopDecision::FlagWinner { score };
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 4 — Score convergence (after step 300)
    // If score is not changing → continuing is useless
    // ══════════════════════════════════════════════════════════════════
    if metrics.step >= 300 && history.len() >= 4 {
        let recent: Vec<f64> = history.iter().rev().take(4)
            .map(|m| m.composite_score())
            .collect();
        let score_now = recent[0];
        let all_stable = recent.windows(2).all(|w| {
            score_now > 1e-6 && (w[0] - w[1]).abs() / score_now < 0.005
        });
        if all_stable {
            return StopDecision::Abort {
                reason: format!(
                    "Score converged over 4 consecutive checks: {:.4} (step {})",
                    score_now, metrics.step
                ),
            };
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // CONDITION 5 — Excellent score near end of run
    // ══════════════════════════════════════════════════════════════════
    if metrics.step >= (n_steps_total as f64 * 0.85) as u32 {
        let score = metrics.composite_score();
        if score > 0.80 {
            return StopDecision::FlagWinner { score };
        }
    }

    StopDecision::Continue
}

/// Simple early stop check for basic metrics only
/// (when full metrics are not computed)
pub fn check_basic_early_stop(
    step: u32,
    ke_ratio: f64,
    v_max: f64,
    v_rms: f64,
) -> Option<String> {
    // Numerical checks
    if v_max > 5_000.0 {
        return Some(format!("RUNAWAY: v_max={:.0} km/s", v_max));
    }
    if v_max.is_nan() || v_max.is_infinite() {
        return Some(format!("NaN/Inf detected at step {}", step));
    }
    if ke_ratio > 1.0e8 {
        return Some(format!("KE explosion: ke_ratio={:.2e}", ke_ratio));
    }
    if v_rms > 1.0 && v_max > 50.0 * v_rms {
        return Some(format!("Particle runaway: v_max/v_rms={:.0}", v_max / v_rms));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(step: u32, v_max: f64, ke_ratio: f64) -> StepMetrics {
        StepMetrics {
            step,
            redshift: 3.0,
            scale_factor: 0.25,
            s_segregation: 0.3,
            n_halos_plus: 10,
            n_halos_minus: 10,
            filament_mean_mpc: 5.0,
            filament_max_mpc: 10.0,
            filament_count: 3,
            fil_matter_fraction: 0.1,
            void_fraction: 0.5,
            void_mode_mpc: 20.0,
            pk_slope: -2.8,
            pk_excess_lcdm: 1.0,
            v_rms: 200.0,
            v_max,
            e_kinetic: 1e10,
            ke_ratio,
        }
    }

    #[test]
    fn test_normal_continue() {
        let m = make_metrics(100, 500.0, 1.0);
        match check_early_stop(&m, &[], 500) {
            StopDecision::Continue => (),
            _ => panic!("Should continue"),
        }
    }

    #[test]
    fn test_runaway_abort() {
        let m = make_metrics(100, 6000.0, 1.0);
        match check_early_stop(&m, &[], 500) {
            StopDecision::Abort { reason } => {
                assert!(reason.contains("RUNAWAY"));
            }
            _ => panic!("Should abort"),
        }
    }

    #[test]
    fn test_ke_explosion() {
        let m = make_metrics(100, 500.0, 1e10);
        match check_early_stop(&m, &[], 500) {
            StopDecision::Abort { reason } => {
                assert!(reason.contains("KE explosion"));
            }
            _ => panic!("Should abort"),
        }
    }
}
