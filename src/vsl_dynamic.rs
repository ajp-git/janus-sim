//! VSL Dynamic c_ratio implementation
//!
//! Based on Petit MPLA 2014, equation (25):
//!   c⁺ ∝ 1/√a⁺
//!   c⁻ ∝ 1/√a⁻
//!   c_ratio = c⁻/c⁺ = √(a⁺/a⁻)
//!   c_ratio_sq = a⁺/a⁻
//!
//! The coupled Friedmann equations determine a⁺(t) and a⁻(t).

/// Janus cosmological parameters for VSL
#[derive(Debug, Clone)]
pub struct JanusVSLParams {
    /// Mass ratio η = ρ⁺/ρ⁻ (typically 1.045)
    pub eta: f64,
    /// Initial redshift
    pub z_init: f64,
    /// Hubble parameter h (H₀ = 100h km/s/Mpc)
    pub h: f64,
}

impl Default for JanusVSLParams {
    fn default() -> Self {
        Self {
            eta: 1.045,
            z_init: 4.0,
            h: 0.70,
        }
    }
}

/// State of the two-metric system
#[derive(Debug, Clone)]
pub struct TwoMetricState {
    /// Scale factor of positive sector a⁺
    pub a_plus: f64,
    /// Scale factor of negative sector a⁻
    pub a_minus: f64,
    /// Conformal time τ
    pub tau: f64,
}

impl TwoMetricState {
    /// Create initial state at given redshift
    pub fn at_redshift(z: f64, params: &JanusVSLParams) -> Self {
        let a_plus = 1.0 / (1.0 + z);

        // From coupled Friedmann equations (Petit & D'Agostini 2014):
        // At early times, the negative sector expands more slowly
        // due to the repulsive coupling with positive mass.
        //
        // Approximation: a⁻ ≈ a⁺ · (1 + z)^(-δ) where δ depends on η
        // At z=0: a⁺ = a⁻ = 1
        // At high z: a⁻ < a⁺
        //
        // This gives c_ratio_sq = a⁺/a⁻ > 1 at high z, decreasing to 1 at z=0

        let delta = (params.eta - 1.0) / params.eta;  // ~0.043 for η=1.045
        let a_minus = a_plus * (1.0 + z).powf(-delta);

        Self {
            a_plus,
            a_minus,
            tau: 0.0,
        }
    }

    /// c_ratio² = a⁺/a⁻
    pub fn c_ratio_sq(&self) -> f64 {
        self.a_plus / self.a_minus
    }

    /// c_ratio = √(a⁺/a⁻)
    pub fn c_ratio(&self) -> f64 {
        (self.a_plus / self.a_minus).sqrt()
    }

    /// Current redshift from a⁺
    pub fn redshift(&self) -> f64 {
        1.0 / self.a_plus - 1.0
    }
}

/// Coupled Friedmann solver for Janus VSL
pub struct CoupledFriedmann {
    params: JanusVSLParams,
    state: TwoMetricState,
}

impl CoupledFriedmann {
    pub fn new(params: JanusVSLParams) -> Self {
        let state = TwoMetricState::at_redshift(params.z_init, &params);
        Self { params, state }
    }

    /// Get current c_ratio_sq
    pub fn c_ratio_sq(&self) -> f64 {
        self.state.c_ratio_sq()
    }

    /// Get current c_ratio
    pub fn c_ratio(&self) -> f64 {
        self.state.c_ratio()
    }

    /// Get current redshift
    pub fn redshift(&self) -> f64 {
        self.state.redshift()
    }

    /// Compute c_ratio_sq at given redshift (static method)
    ///
    /// Based on the coupled Friedmann equations:
    /// - a⁺(z) = 1/(1+z)
    /// - a⁻(z) = a⁺(z) · (1+z)^(-δ) where δ = (η-1)/η
    ///
    /// This gives: c_ratio_sq(z) = (1+z)^δ
    pub fn c_ratio_sq_at_z(z: f64, eta: f64) -> f64 {
        let delta = (eta - 1.0) / eta;
        (1.0 + z).powf(delta)
    }

    /// Compute c_ratio_sq from scale factor a⁺
    pub fn c_ratio_sq_at_a(a_plus: f64, eta: f64) -> f64 {
        let z = 1.0 / a_plus - 1.0;
        Self::c_ratio_sq_at_z(z, eta)
    }

    /// Compute a⁻ from a⁺ — CANONICAL SOURCE OF TRUTH (Petit 2014)
    ///
    /// a⁻(z) = a⁺(z) × (1+z)^(-δ)  where δ = (η-1)/η
    ///
    /// Properties:
    /// - At z=0: a⁻ = a⁺ = 1
    /// - At z>0 with η>1: a⁻ < a⁺ (negative sector more contracted)
    /// - Identity: c̄²(z) = a⁺/a⁻
    #[inline]
    pub fn a_minus_from_a_plus(a_plus: f64, eta: f64) -> f64 {
        let z = 1.0 / a_plus - 1.0;
        let delta = (eta - 1.0) / eta;
        a_plus * (1.0_f64 + z).powf(-delta)
    }

    /// Update state for new scale factor a⁺
    pub fn update_a_plus(&mut self, a_plus: f64) {
        self.state.a_plus = a_plus;
        let z = 1.0 / a_plus - 1.0;
        let delta = (self.params.eta - 1.0) / self.params.eta;
        self.state.a_minus = a_plus * (1.0 + z).powf(-delta);
    }

    /// Hubble parameter for positive sector at redshift z
    /// H⁺(z) = H₀ · √(Ω_m(1+z)³ + Ω_Λ) for ΛCDM comparison
    /// For Janus: H⁺(z) ≈ H₀ · (1+z)^(3/2) · √(η/(1+η)) at high z
    pub fn hubble_plus(&self, z: f64) -> f64 {
        let h0 = self.params.h * 100.0;  // km/s/Mpc
        let omega_m = self.params.eta / (1.0 + self.params.eta);
        h0 * ((omega_m * (1.0 + z).powi(3) + (1.0 - omega_m)).sqrt())
    }
}

// ============================================================================
// ALTERNATIVE MODEL: Full coupled integration
// ============================================================================

/// Full coupled Friedmann integration
///
/// Equations from Petit & D'Agostini (2014):
/// (ȧ⁺/a⁺)² = (8πG/3)[ρ⁺ + ρ⁻(a⁺/a⁻)³]
/// (ȧ⁻/a⁻)² = (8πG/3)[ρ⁻ + ρ⁺(a⁻/a⁺)³]
///
/// With matter conservation: ρ⁺ ∝ a⁺⁻³, ρ⁻ ∝ a⁻⁻³
pub struct FullCoupledIntegrator {
    /// Current a⁺
    pub a_plus: f64,
    /// Current a⁻
    pub a_minus: f64,
    /// Initial comoving densities ratio η = ρ⁺₀/ρ⁻₀
    pub eta: f64,
    /// 8πG/3 · ρ_crit (normalized to 1)
    pub h0_sq: f64,
}

impl FullCoupledIntegrator {
    pub fn new(eta: f64, z_init: f64) -> Self {
        let a_init = 1.0 / (1.0 + z_init);
        Self {
            a_plus: a_init,
            a_minus: a_init,  // Start equal at high z
            eta,
            h0_sq: 1.0,  // Normalized units
        }
    }

    /// Hubble rate squared for positive sector
    fn h_plus_sq(&self) -> f64 {
        let rho_plus = self.eta / self.a_plus.powi(3);
        let rho_minus_eff = (1.0 / self.a_minus.powi(3)) * (self.a_plus / self.a_minus).powi(3);
        self.h0_sq * (rho_plus + rho_minus_eff)
    }

    /// Hubble rate squared for negative sector
    fn h_minus_sq(&self) -> f64 {
        let rho_minus = 1.0 / self.a_minus.powi(3);
        let rho_plus_eff = (self.eta / self.a_plus.powi(3)) * (self.a_minus / self.a_plus).powi(3);
        self.h0_sq * (rho_minus + rho_plus_eff)
    }

    /// Step forward in cosmic time by dt
    pub fn step(&mut self, dt: f64) {
        let h_plus = self.h_plus_sq().sqrt();
        let h_minus = self.h_minus_sq().sqrt();

        self.a_plus += self.a_plus * h_plus * dt;
        self.a_minus += self.a_minus * h_minus * dt;
    }

    /// c_ratio_sq = a⁺/a⁻
    pub fn c_ratio_sq(&self) -> f64 {
        self.a_plus / self.a_minus
    }

    pub fn redshift(&self) -> f64 {
        1.0 / self.a_plus - 1.0
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_ratio_sq_at_bounce_large() {
        // At high z (near bounce), c_ratio_sq should be > 1
        let eta = 1.045;
        let z_bounce = 1000.0;  // Very early universe
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z_bounce, eta);

        println!("c_ratio_sq(z={}) = {:.4}", z_bounce, c_ratio_sq);
        assert!(c_ratio_sq > 1.0, "c_ratio_sq at bounce should be > 1");

        // For η=1.045, δ≈0.043, so (1+1000)^0.043 ≈ 1.35
        assert!(c_ratio_sq > 1.2, "c_ratio_sq at z=1000 should be > 1.2");
    }

    #[test]
    fn test_c_ratio_sq_at_z0_finite() {
        // At z=0, c_ratio_sq should approach 1
        let eta = 1.045;
        let z0 = 0.0;
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z0, eta);

        println!("c_ratio_sq(z=0) = {:.6}", c_ratio_sq);
        assert!((c_ratio_sq - 1.0).abs() < 0.001, "c_ratio_sq at z=0 should be ~1");
    }

    #[test]
    fn test_c_ratio_sq_decreasing_with_time() {
        // c_ratio_sq should decrease as z decreases (time increases)
        let eta = 1.045;

        let z_values = [4.0, 3.0, 2.0, 1.0, 0.5, 0.0];
        let mut prev_c_ratio_sq = f64::INFINITY;

        println!("z\t\tc_ratio_sq");
        println!("---\t\t----------");
        for z in z_values {
            let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, eta);
            println!("{:.1}\t\t{:.6}", z, c_ratio_sq);

            assert!(c_ratio_sq < prev_c_ratio_sq,
                "c_ratio_sq should decrease: {} < {}", c_ratio_sq, prev_c_ratio_sq);
            prev_c_ratio_sq = c_ratio_sq;
        }
    }

    #[test]
    fn test_c_ratio_sq_evolution_range() {
        // Test the full range from z=4 to z=0
        let eta = 1.045;

        let c_ratio_sq_z4 = CoupledFriedmann::c_ratio_sq_at_z(4.0, eta);
        let c_ratio_sq_z0 = CoupledFriedmann::c_ratio_sq_at_z(0.0, eta);

        println!("c_ratio_sq evolution:");
        println!("  z=4: {:.6}", c_ratio_sq_z4);
        println!("  z=0: {:.6}", c_ratio_sq_z0);
        println!("  ratio: {:.3}x", c_ratio_sq_z4 / c_ratio_sq_z0);

        // c_ratio_sq should decrease by a factor of ~(1+4)^δ ≈ 1.07 from z=4 to z=0
        assert!(c_ratio_sq_z4 > c_ratio_sq_z0);
        assert!(c_ratio_sq_z4 / c_ratio_sq_z0 > 1.05);
        assert!(c_ratio_sq_z4 / c_ratio_sq_z0 < 1.15);
    }

    #[test]
    fn test_full_coupled_integrator() {
        // Test the full coupled integration
        let mut integrator = FullCoupledIntegrator::new(1.045, 4.0);

        println!("\nFull coupled integration:");
        println!("step\tz\t\ta+\t\ta-\t\tc_ratio_sq");
        println!("----\t---\t\t---\t\t---\t\t----------");

        let dt = 0.001;
        let mut prev_c_ratio_sq = integrator.c_ratio_sq();

        for step in 0..=500 {
            if step % 100 == 0 {
                println!("{}\t{:.3}\t\t{:.4}\t\t{:.4}\t\t{:.6}",
                    step, integrator.redshift(),
                    integrator.a_plus, integrator.a_minus,
                    integrator.c_ratio_sq());
            }

            // c_ratio_sq should be decreasing (or at least not increasing significantly)
            let current_c_ratio_sq = integrator.c_ratio_sq();
            if step > 0 {
                // Allow small numerical fluctuations
                assert!(current_c_ratio_sq <= prev_c_ratio_sq * 1.001,
                    "c_ratio_sq should not increase: {} > {}",
                    current_c_ratio_sq, prev_c_ratio_sq);
            }
            prev_c_ratio_sq = current_c_ratio_sq;

            integrator.step(dt);
        }
    }

    #[test]
    fn test_c_ratio_at_simulation_redshifts() {
        // Test c_ratio values at typical simulation redshifts
        let eta = 1.045;

        println!("\nc_ratio values for η={}:", eta);
        println!("z\t\tc_ratio\t\tc_ratio_sq");
        println!("---\t\t-------\t\t----------");

        for z in [4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.0] {
            let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, eta);
            let c_ratio = c_ratio_sq.sqrt();
            println!("{:.1}\t\t{:.6}\t{:.6}", z, c_ratio, c_ratio_sq);
        }
    }

    #[test]
    fn test_a_minus_petit_2014() {
        // Pin the exact Petit 2014 formula for a_minus
        let eta = 1.045;
        let z_test = 10.0;
        let a_plus = 1.0 / (1.0 + z_test);
        let a_minus = CoupledFriedmann::a_minus_from_a_plus(a_plus, eta);

        // Expected from Petit 2014: a⁻ = a⁺ × (1+z)^(-δ) where δ = (η-1)/η
        let delta = (eta - 1.0) / eta;
        let expected = a_plus * (1.0_f64 + z_test).powf(-delta);
        assert!((a_minus - expected).abs() / expected < 1e-10,
            "a_minus formula mismatch: got {}, expected {}", a_minus, expected);

        // Sanity: a_minus < a_plus at z>0 when η>1
        assert!(a_minus < a_plus,
            "a_minus should be < a_plus at z>0 for η>1: a_minus={}, a_plus={}", a_minus, a_plus);

        // Sanity: a_minus = a_plus at z=0
        let a_minus_today = CoupledFriedmann::a_minus_from_a_plus(1.0, eta);
        assert!((a_minus_today - 1.0).abs() < 1e-10,
            "a_minus should equal 1.0 at z=0: got {}", a_minus_today);

        // Identity: c̄²(z) = a⁺/a⁻
        let c_bar_sq = CoupledFriedmann::c_ratio_sq_at_z(z_test, eta);
        let ratio = a_plus / a_minus;
        assert!((c_bar_sq - ratio).abs() < 1e-10,
            "Identity c̄² = a⁺/a⁻ violated: c̄²={}, a⁺/a⁻={}", c_bar_sq, ratio);
    }
}
