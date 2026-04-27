/// Janus Parametric Cosmology — Exact Bimetric Solution
///
/// Implements the exact parametric solution for the Janus positive sector:
///
///     a⁺(μ) = α² cosh²(μ)
///     t⁺(μ) = α² (μ + ½ sinh²(μ))
///     H⁺(μ) = ȧ⁺/a⁺ = sinh(μ)cosh(μ) / (α² (1 + sinh(μ)cosh(μ)))
///
/// where α is calibrated so that H₀ = 70 km/s/Mpc at z=0 (μ=μ₀).
///
/// This is the CORRECT expansion history for Janus N-body simulations,
/// NOT the standard ΛCDM Friedmann equations.
///
/// Reference: Jean-Pierre Petit, Janus Cosmological Model

/// Physical constants
const H0_KM_S_MPC: f64 = 70.0;  // Hubble constant today [km/s/Mpc]
const H0_GYR_INV: f64 = 0.0715; // H₀ in Gyr⁻¹ (70 km/s/Mpc ≈ 0.0715 Gyr⁻¹)
const KM_S_MPC_TO_GYR_INV: f64 = 1.0227e-3; // Conversion factor

/// Janus expansion state at a given parameter μ
#[derive(Debug, Clone, Copy)]
pub struct JanusExpansionState {
    /// Parameter μ (dimensionless)
    pub mu: f64,
    /// Scale factor a⁺ (normalized so a⁺(μ₀) = 1 at z=0)
    pub a_plus: f64,
    /// Cosmic time t [Gyr]
    pub t_gyr: f64,
    /// Hubble parameter H⁺ [Gyr⁻¹]
    pub h_plus: f64,
    /// Redshift z = a₀/a⁺ - 1
    pub z: f64,
}

/// Janus parametric cosmology with precalculated lookup table
pub struct JanusExpansion {
    /// Calibration constant α (set so H(z=0) = H₀)
    pub alpha: f64,
    /// Parameter μ₀ at z=0 (today)
    pub mu_0: f64,
    /// Scale factor at z=0 (for normalization)
    pub a_0: f64,
    /// Lookup table indexed by time
    pub table: Vec<JanusExpansionState>,
    /// Time at z_init (start of simulation)
    pub t_start: f64,
    /// Time at z=0 (end of simulation)
    pub t_end: f64,
}

impl JanusExpansion {
    /// Create new Janus expansion table from z_init to z=0
    ///
    /// # Arguments
    /// * `z_init` - Initial redshift (will be clamped to z_max ≈ cosh²(μ₀) - 1)
    /// * `n_points` - Number of points in lookup table
    ///
    /// # Note
    /// The parametric solution a⁺(μ) = α² cosh²(μ) has a minimum at μ=0.
    /// This limits the maximum redshift to z_max = a₀/α² - 1 ≈ cosh²(μ₀) - 1.
    pub fn new(z_init: f64, n_points: usize) -> Self {
        // Step 1: Find α and μ₀ such that H(μ₀) = H₀ at z=0
        let (alpha, mu_0) = calibrate_alpha_mu0();

        // Scale factor at z=0
        let a_0 = alpha * alpha * mu_0.cosh().powi(2);

        // Maximum redshift (when μ=0, a=α²)
        let a_min = alpha * alpha;
        let z_max = (a_0 / a_min) - 1.0;

        // Clamp z_init to valid range
        let z_init_clamped = z_init.min(z_max * 0.99);  // Stay slightly below z_max

        if z_init > z_max {
            println!("WARNING: z_init={:.1} exceeds z_max={:.2}, clamped to z={:.2}",
                     z_init, z_max, z_init_clamped);
        }

        // Step 2: Find μ_init corresponding to z_init
        // z = a_0/a - 1  =>  a = a_0/(1+z)
        // a = α² cosh²(μ)  =>  cosh(μ) = sqrt(a/α²)
        let a_init = a_0 / (1.0 + z_init_clamped);
        let cosh_mu_init = (a_init / (alpha * alpha)).sqrt().max(1.0);  // Ensure >= 1
        let mu_init = cosh_mu_init.acosh();

        // Step 3: Build table from μ_init to μ_0
        let mut table = Vec::with_capacity(n_points);
        let d_mu = (mu_0 - mu_init) / (n_points as f64 - 1.0);

        for i in 0..n_points {
            let mu = mu_init + i as f64 * d_mu;
            let state = compute_state(mu, alpha, a_0);
            table.push(state);
        }

        let t_start = table.first().unwrap().t_gyr;
        let t_end = table.last().unwrap().t_gyr;

        let z_start = table.first().unwrap().z;

        println!("=== Janus Expansion Table ===");
        println!("  α = {:.6}", alpha);
        println!("  μ₀ = {:.6} (z=0)", mu_0);
        println!("  μ_init = {:.6} (z={:.2})", mu_init, z_start);
        println!("  a₀ = {:.6}", a_0);
        println!("  z_max = {:.2} (parametric limit)", (a_0 / (alpha * alpha)) - 1.0);
        println!("  t_start = {:.4} Gyr (z={:.2})", t_start, z_start);
        println!("  t_end = {:.4} Gyr (z=0)", t_end);
        println!("  H₀ = {:.4} Gyr⁻¹ = {:.1} km/s/Mpc",
                 table.last().unwrap().h_plus,
                 table.last().unwrap().h_plus / KM_S_MPC_TO_GYR_INV);
        println!("=============================");

        Self {
            alpha,
            mu_0,
            a_0,
            table,
            t_start,
            t_end,
        }
    }

    /// Get expansion parameters at cosmic time t [Gyr]
    /// Uses linear interpolation in the table
    pub fn at_time(&self, t: f64) -> JanusExpansionState {
        // Clamp t to valid range
        let t_clamped = t.clamp(self.t_start, self.t_end);

        // Binary search for bracketing indices
        let idx = self.table.partition_point(|s| s.t_gyr < t_clamped);

        if idx == 0 {
            return self.table[0];
        }
        if idx >= self.table.len() {
            return *self.table.last().unwrap();
        }

        // Linear interpolation
        let s0 = &self.table[idx - 1];
        let s1 = &self.table[idx];
        let frac = (t_clamped - s0.t_gyr) / (s1.t_gyr - s0.t_gyr);

        JanusExpansionState {
            mu: s0.mu + frac * (s1.mu - s0.mu),
            a_plus: s0.a_plus + frac * (s1.a_plus - s0.a_plus),
            t_gyr: t_clamped,
            h_plus: s0.h_plus + frac * (s1.h_plus - s0.h_plus),
            z: s0.z + frac * (s1.z - s0.z),
        }
    }

    /// Get expansion parameters at redshift z
    pub fn at_redshift(&self, z: f64) -> JanusExpansionState {
        // Find by binary search on z (decreasing)
        let idx = self.table.partition_point(|s| s.z > z);

        if idx == 0 {
            return self.table[0];
        }
        if idx >= self.table.len() {
            return *self.table.last().unwrap();
        }

        // Linear interpolation
        let s0 = &self.table[idx - 1];
        let s1 = &self.table[idx];
        let frac = (s0.z - z) / (s0.z - s1.z);

        JanusExpansionState {
            mu: s0.mu + frac * (s1.mu - s0.mu),
            a_plus: s0.a_plus + frac * (s1.a_plus - s0.a_plus),
            t_gyr: s0.t_gyr + frac * (s1.t_gyr - s0.t_gyr),
            h_plus: s0.h_plus + frac * (s1.h_plus - s0.h_plus),
            z,
        }
    }

    /// Export table to CSV for validation
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "mu,t_gyr,a_plus,h_plus_gyr,h_plus_km_s_mpc,z")?;
        for s in &self.table {
            writeln!(file, "{:.6},{:.6},{:.6},{:.6},{:.2},{:.4}",
                     s.mu, s.t_gyr, s.a_plus, s.h_plus,
                     s.h_plus / KM_S_MPC_TO_GYR_INV, s.z)?;
        }
        Ok(())
    }
}

/// Compute state at parameter μ
fn compute_state(mu: f64, alpha: f64, a_0: f64) -> JanusExpansionState {
    let cosh_mu = mu.cosh();
    let sinh_mu = mu.sinh();
    let sinh_2mu = (2.0_f64 * mu).sinh();  // Still needed for H calculation
    let sinh_sq = sinh_mu * sinh_mu;       // sinh²(μ) for t formula

    // a⁺(μ) = α² cosh²(μ)
    let a_plus_raw = alpha * alpha * cosh_mu * cosh_mu;

    // Normalize so a⁺ = 1 at z=0
    let a_plus = a_plus_raw / a_0;

    // t⁺(μ) = α² (μ + ½ sinh²(μ))  [CORRECTED FORMULA]
    // This is in units where 1/H₀ = 1, so multiply by 1/H₀ to get Gyr
    let t_param = alpha * alpha * (mu + 0.5 * sinh_sq);
    let t_gyr = t_param / H0_GYR_INV;

    // H⁺(μ) = sinh(2μ) / (α² cosh²(μ) (1 + ½sinh(2μ)))
    // Derived from H = ȧ/a = (da/dμ)/(dt/dμ)/a with corrected t formula
    // dt/dμ = α² (1 + sinh(μ)cosh(μ)) = α² (1 + ½sinh(2μ))
    let h_param = sinh_2mu / (alpha * alpha * cosh_mu * cosh_mu * (1.0 + 0.5 * sinh_2mu));
    let h_plus = h_param * H0_GYR_INV;

    // Redshift
    let z = (a_0 / a_plus_raw) - 1.0;

    JanusExpansionState {
        mu,
        a_plus,
        t_gyr,
        h_plus,
        z,
    }
}

/// Calibrate α and μ₀ so that H(μ₀) = H₀ at z=0
///
/// We need:
/// 1. At z=0 (today): a⁺ = a₀ (arbitrary normalization)
/// 2. H(μ₀) = H₀ = 70 km/s/Mpc
///
/// From the corrected parametric solution with t⁺(μ) = α² (μ + ½ sinh²(μ)):
///   dt/dμ = α² (1 + sinh(μ)cosh(μ)) = α² (1 + ½sinh(2μ))
///   H(μ) = (da/dμ)/(dt/dμ) / a
///        = (2α²cosh(μ)sinh(μ)) / (α²(1 + ½sinh(2μ))) / (α²cosh²(μ))
///        = sinh(2μ) / ((1 + ½sinh(2μ)) α²cosh²(μ))
///
/// At μ = μ₀: H(μ₀) = H₀ in natural units (H₀ = 1)
/// Solving for α²:
///   α² = sinh(2μ₀) / (cosh²(μ₀) × (1 + ½sinh(2μ₀)))
fn calibrate_alpha_mu0() -> (f64, f64) {
    // Choose μ₀ such that the universe is in late-time matter-dominated phase
    // For Janus, μ₀ ≈ 1.5 gives reasonable behavior
    let mu_0: f64 = 1.5;

    let cosh_mu = mu_0.cosh();
    let sinh_2mu = (2.0_f64 * mu_0).sinh();

    // From the corrected Janus parametric solution:
    // H(μ) = sinh(2μ) / (α² cosh²(μ) (1 + ½sinh(2μ)))
    //
    // At z=0 (μ = μ₀), we want H = 1 in natural units:
    // 1 = sinh(2μ₀) / (α² cosh²(μ₀) (1 + ½sinh(2μ₀)))
    //
    // Solving for α²:
    let alpha_sq = sinh_2mu / (cosh_mu * cosh_mu * (1.0 + 0.5 * sinh_2mu));
    let alpha = alpha_sq.sqrt();

    (alpha, mu_0)
}

/// ============================================================================
/// MILNE EXPANSION REGIME (PRE-BOUNCE)
/// ============================================================================
///
/// Before the bounce at z=4.5, the universe follows a Milne expansion:
///   a(t) = t / t_bounce
///   H(t) = 1/t
///   t_bounce = 1.578 Gyr
///
/// This is a coasting universe with no deceleration (q=0).

/// Time of the bounce in Gyr (corresponds to z=4.5 in post-bounce coordinates)
pub const T_BOUNCE_GYR: f64 = 1.578;

/// Redshift at the bounce (as seen from today)
pub const Z_BOUNCE: f64 = 4.5;

/// Milne expansion state
#[derive(Debug, Clone, Copy)]
pub struct MilneExpansionState {
    /// Cosmic time t [Gyr]
    pub t_gyr: f64,
    /// Scale factor a (normalized so a=1 at bounce)
    pub a: f64,
    /// Hubble parameter H [Gyr⁻¹]
    pub h: f64,
    /// Redshift relative to bounce (z_bounce = 0, earlier = positive)
    pub z_milne: f64,
    /// Redshift as seen from today (z=0 today)
    pub z_today: f64,
}

/// Milne expansion for pre-bounce phase
pub struct MilneExpansion {
    /// Time at bounce [Gyr]
    pub t_bounce: f64,
    /// Scale factor at bounce (relative to today's a=1)
    pub a_bounce: f64,
    /// Lookup table indexed by time
    pub table: Vec<MilneExpansionState>,
    /// Time at z_init
    pub t_start: f64,
    /// Time at bounce
    pub t_end: f64,
}

impl MilneExpansion {
    /// Create Milne expansion table from z_init to z_bounce
    ///
    /// # Arguments
    /// * `z_init_today` - Initial redshift as seen from today (e.g., z=100)
    /// * `n_points` - Number of points in lookup table
    pub fn new(z_init_today: f64, n_points: usize) -> Self {
        let t_bounce = T_BOUNCE_GYR;

        // Scale factor at bounce relative to today (a_today = 1)
        let a_bounce = 1.0 / (1.0 + Z_BOUNCE);

        // In Milne: a(t) = a_bounce × (t / t_bounce)
        // So: a = 1/(1+z_today) = a_bounce × (t / t_bounce)
        // => t = t_bounce × (1/(1+z_today)) / a_bounce
        //      = t_bounce / (a_bounce × (1+z_today))
        //      = t_bounce × (1+Z_BOUNCE) / (1+z_today)

        let a_init = 1.0 / (1.0 + z_init_today);
        let t_init = t_bounce * a_init / a_bounce;

        println!("=== Milne Expansion (Pre-Bounce) ===");
        println!("  t_bounce = {:.4} Gyr (z_today = {:.1})", t_bounce, Z_BOUNCE);
        println!("  a_bounce = {:.6} (relative to today)", a_bounce);
        println!("  z_init = {:.1} → t_init = {:.6} Gyr", z_init_today, t_init);
        println!("  a_init = {:.6}", a_init);

        // Build table from t_init to t_bounce
        let mut table = Vec::with_capacity(n_points);
        let dt = (t_bounce - t_init) / (n_points as f64 - 1.0);

        for i in 0..n_points {
            let t = t_init + i as f64 * dt;
            let state = Self::compute_state(t, t_bounce, a_bounce);
            table.push(state);
        }

        println!("  H_init = {:.4} Gyr⁻¹ = {:.1} km/s/Mpc",
                 1.0/t_init, (1.0/t_init) / KM_S_MPC_TO_GYR_INV);
        println!("  H_bounce = {:.4} Gyr⁻¹ = {:.1} km/s/Mpc",
                 1.0/t_bounce, (1.0/t_bounce) / KM_S_MPC_TO_GYR_INV);
        println!("=====================================");

        Self {
            t_bounce,
            a_bounce,
            table,
            t_start: t_init,
            t_end: t_bounce,
        }
    }

    fn compute_state(t: f64, t_bounce: f64, a_bounce: f64) -> MilneExpansionState {
        // Milne expansion: a(t) = a_bounce × (t / t_bounce)
        let a = a_bounce * t / t_bounce;

        // H(t) = ȧ/a = (a_bounce/t_bounce) / (a_bounce × t/t_bounce) = 1/t
        let h = 1.0 / t;

        // Redshift relative to bounce (z_milne = 0 at bounce)
        let z_milne = (a_bounce / a) - 1.0;

        // Redshift as seen from today (z_today = 0 today, a_today = 1)
        let z_today = (1.0 / a) - 1.0;

        MilneExpansionState {
            t_gyr: t,
            a,
            h,
            z_milne,
            z_today,
        }
    }

    /// Get expansion parameters at cosmic time t [Gyr]
    pub fn at_time(&self, t: f64) -> MilneExpansionState {
        let t_clamped = t.clamp(self.t_start, self.t_end);

        // Binary search for bracketing indices
        let idx = self.table.partition_point(|s| s.t_gyr < t_clamped);

        if idx == 0 {
            return self.table[0];
        }
        if idx >= self.table.len() {
            return *self.table.last().unwrap();
        }

        // Linear interpolation
        let s0 = &self.table[idx - 1];
        let s1 = &self.table[idx];
        let frac = (t_clamped - s0.t_gyr) / (s1.t_gyr - s0.t_gyr);

        MilneExpansionState {
            t_gyr: t_clamped,
            a: s0.a + frac * (s1.a - s0.a),
            h: s0.h + frac * (s1.h - s0.h),
            z_milne: s0.z_milne + frac * (s1.z_milne - s0.z_milne),
            z_today: s0.z_today + frac * (s1.z_today - s0.z_today),
        }
    }

    /// Get expansion parameters at redshift (as seen from today)
    pub fn at_redshift_today(&self, z: f64) -> MilneExpansionState {
        // a = 1/(1+z), t = t_bounce × a / a_bounce
        let a = 1.0 / (1.0 + z);
        let t = self.t_bounce * a / self.a_bounce;
        self.at_time(t)
    }

    /// Export table to CSV
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "t_gyr,a,h_gyr,h_km_s_mpc,z_milne,z_today")?;
        for s in &self.table {
            writeln!(file, "{:.6},{:.8},{:.6},{:.2},{:.4},{:.2}",
                     s.t_gyr, s.a, s.h, s.h / KM_S_MPC_TO_GYR_INV,
                     s.z_milne, s.z_today)?;
        }
        Ok(())
    }
}

// ============================================================================
// Cross-species coupling factors (Petit et al. 2024)
// ============================================================================

/// Coupling factor Φ(t) = (ā/a)³ for forces exerted by m⁻ on m⁺
///
/// From Eq. 95-96 of arXiv:2412.04644v3:
///   √(|ḡ|/|g|) = (ā/a)³
///
/// This modulates the gravitational interaction of negative matter
/// on positive matter. At z=0 with ā≈a, Φ→1.
/// At high z with η=1.045, ā/a≈1.04 so Φ≈1.125 (13% enhancement).
#[inline]
pub fn phi_coupling(a_plus: f64, a_minus: f64) -> f64 {
    let ratio = a_minus / a_plus;
    ratio * ratio * ratio
}

/// Inverse coupling factor 1/Φ(t) = (a/ā)³ for forces exerted by m⁺ on m⁻
///
/// This is the symmetric factor for forces of positive matter on negative.
#[inline]
pub fn phi_coupling_inv(a_plus: f64, a_minus: f64) -> f64 {
    let ratio = a_plus / a_minus;
    ratio * ratio * ratio
}

/// Compute a_minus from a_plus using η parameter
///
/// For η close to 1, the relationship is:
///   ā/a ≈ η^(1/(5-2η)) at high z, converging to 1 at z=0
///
/// Compute a⁻ from a⁺ — delegates to canonical source in vsl_dynamic
///
/// Petit 2014 exact formula:
///   a⁻(z) = a⁺(z) × (1+z)^(-δ)  where δ = (η-1)/η
///
/// Properties:
/// - At z=0: a⁻ = a⁺ = 1
/// - At z>0 with η>1: a⁻ < a⁺ (negative sector more contracted)
/// - Identity: c̄²(z) = a⁺/a⁻
#[inline]
pub fn a_minus_from_a_plus(a_plus: f64, eta: f64) -> f64 {
    crate::vsl_dynamic::CoupledFriedmann::a_minus_from_a_plus(a_plus, eta)
}

/// Compute both coupling factors at once
#[inline]
pub fn compute_phi_factors(a_plus: f64, eta: f64) -> (f64, f64) {
    let a_minus = a_minus_from_a_plus(a_plus, eta);
    let phi = phi_coupling(a_plus, a_minus);
    let phi_inv = phi_coupling_inv(a_plus, a_minus);
    (phi, phi_inv)
}

// ============================================================================
// Total energy conservation (Petit et al. 2024)
// ============================================================================

/// Compute the Janus total energy E = ρ⁺c² + ρ⁻c̄²
///
/// From Eq. 96 of arXiv:2412.04644v3 / Eq. 2.8 of the short paper:
///   E = ρ_proper⁺ · c² · a⁺³ + ρ_proper⁻ · c̄² · a⁻³ = constant
///
/// With ρ_proper = ρ_comoving / a³, this simplifies to:
///   E = ρ_comoving⁺ · c² + ρ_comoving⁻ · c̄²
///
/// Where ρ_comoving is constant (mass conservation), so E varies only with c̄(t).
/// E should be NEGATIVE for accelerated expansion with μ > 1.
///
/// IMPORTANT: Pass COMOVING densities (M/V_comoving), not proper densities!
///
/// Returns (E_total, E_plus, E_minus) for detailed tracking.
#[inline]
pub fn compute_total_energy(
    rho_plus_comoving: f64,    // comoving density m+ (positive)
    rho_minus_comoving: f64,   // comoving density m- (negative, should be < 0)
    c_plus: f64,               // c (constant, = 1 in code units)
    c_minus: f64,              // c̄(t), VSL dynamic speed of light
    _a_plus: f64,              // Not used (kept for API compatibility)
    _a_minus: f64,             // Not used (kept for API compatibility)
) -> (f64, f64, f64) {
    // E = ρ_comoving × c² (since ρ_proper × a³ = ρ_comoving)
    let e_plus = rho_plus_comoving * c_plus * c_plus;
    let e_minus = rho_minus_comoving * c_minus * c_minus;
    let e_total = e_plus + e_minus;
    (e_total, e_plus, e_minus)
}

/// Compute energy drift as percentage from initial value
#[inline]
pub fn energy_drift_pct(e_now: f64, e_initial: f64) -> f64 {
    if e_initial.abs() < 1e-30 {
        return 0.0;  // Avoid division by zero
    }
    (e_now - e_initial) / e_initial.abs() * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_coupling_is_cube() {
        // At z=0 with ā = a → Φ = 1
        assert!((phi_coupling(1.0, 1.0) - 1.0).abs() < 1e-12);
        // ā/a = 2 → Φ = 8
        assert!((phi_coupling(1.0, 2.0) - 8.0).abs() < 1e-12);
        // Symmetry: φ × 1/φ = 1
        let a_p = 0.5;
        let a_m = 0.6;
        assert!((phi_coupling(a_p, a_m) * phi_coupling_inv(a_p, a_m) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_phi_typical_values() {
        // At η=1.045, z=4 : ā/a ≈ 1.04 → Φ ≈ 1.125
        let eta = 1.045;
        let a_p_z4 = 0.2;  // z=4
        let a_m_z4 = a_minus_from_a_plus(a_p_z4, eta);
        let phi = phi_coupling(a_p_z4, a_m_z4);
        println!("At z=4 (a=0.2): ā/a = {:.4}, Φ = {:.4}", a_m_z4/a_p_z4, phi);
        // ā/a should be close to 1.04
        assert!((a_m_z4/a_p_z4 - 1.036).abs() < 0.01);
        // Φ should be around 1.11
        assert!(phi > 1.05 && phi < 1.20);
    }

    #[test]
    fn test_phi_convergence_at_z0() {
        // At z=0, ā = a → Φ = 1 exactly
        let eta = 1.045;
        let a_z0 = 1.0;
        let a_m = a_minus_from_a_plus(a_z0, eta);
        assert!((a_m - a_z0).abs() < 1e-10, "ā should equal a at z=0");
        let phi = phi_coupling(a_z0, a_m);
        assert!((phi - 1.0).abs() < 1e-10, "Φ should be 1 at z=0");
    }

    #[test]
    fn test_milne_expansion() {
        let milne = MilneExpansion::new(100.0, 1000);

        // At bounce, z_today should be Z_BOUNCE
        let state_bounce = milne.at_time(T_BOUNCE_GYR);
        println!("At bounce: z_today = {:.2}, a = {:.4}", state_bounce.z_today, state_bounce.a);
        assert!((state_bounce.z_today - Z_BOUNCE).abs() < 0.1);

        // H should be 1/t
        let state_mid = milne.at_time(0.5);
        assert!((state_mid.h - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_calibration() {
        let (alpha, mu_0) = calibrate_alpha_mu0();
        println!("α = {}, μ₀ = {}", alpha, mu_0);

        // Check H at z=0
        let expansion = JanusExpansion::new(5.0, 1000);
        let state_z0 = expansion.at_redshift(0.0);

        let h_km_s_mpc = state_z0.h_plus / KM_S_MPC_TO_GYR_INV;
        println!("H(z=0) = {} km/s/Mpc", h_km_s_mpc);

        assert!((h_km_s_mpc - 70.0).abs() < 1.0, "H₀ should be ~70 km/s/Mpc");
    }

    #[test]
    fn test_redshift_evolution() {
        let expansion = JanusExpansion::new(10.0, 2000);

        // Check a few redshifts
        for z in [0.0, 1.0, 2.0, 5.0, 10.0] {
            let state = expansion.at_redshift(z);
            println!("z={:.1}: a={:.4}, H={:.4} Gyr⁻¹, t={:.4} Gyr",
                     z, state.a_plus, state.h_plus, state.t_gyr);

            // a should decrease with z
            assert!(state.a_plus > 0.0);
            assert!(state.a_plus <= 1.0 || z < 0.01);
        }
    }

    #[test]
    fn test_e_negative_with_mu_canonical() {
        // μ=8 canonical: ρ⁻ >> ρ⁺ in module
        let rho_p = 1.0;
        let rho_m = -8.0;       // μ=8 (negative mass density)
        let c_p = 1.0;
        let c_m = 1.05;         // VSL c̄ > c
        let a_p = 0.5;          // Not used in calculation
        let a_m = 0.5;          // Not used in calculation
        let (e_total, e_plus, e_minus) = compute_total_energy(rho_p, rho_m, c_p, c_m, a_p, a_m);

        // E = ρ_comoving × c²: E_plus = 1×1² = 1, E_minus = -8×1.05² = -8.82
        println!("E_plus = {:.4}, E_minus = {:.4}, E_total = {:.4}", e_plus, e_minus, e_total);
        assert!(e_total < 0.0, "E must be negative for accelerated expansion");
        assert!(e_plus > 0.0, "E_plus should be positive");
        assert!(e_minus < 0.0, "E_minus should be negative (ρ⁻ < 0)");
    }

    #[test]
    fn test_energy_independent_of_a() {
        // E = ρ_comoving × c² — should NOT depend on a
        let rho_p = 1.0;
        let rho_m = -8.0;
        let c_p = 1.0;
        let c_m = 1.0;

        let (e1, _, _) = compute_total_energy(rho_p, rho_m, c_p, c_m, 0.5, 0.5);
        let (e2, _, _) = compute_total_energy(rho_p, rho_m, c_p, c_m, 1.0, 1.0);

        // e2 / e1 should be 1.0 (no a dependence)
        let ratio = e2 / e1;
        println!("E(a=1)/E(a=0.5) = {:.4} (expected 1.0)", ratio);
        assert!((ratio - 1.0).abs() < 1e-10, "Energy should NOT depend on a");
    }

    #[test]
    fn test_energy_drift() {
        let e_initial = -10.0;
        let e_now = -10.05;  // 0.5% drift
        let drift = energy_drift_pct(e_now, e_initial);
        println!("Drift = {:.4}%", drift);
        // drift = (-10.05 - (-10.0)) / |-10.0| * 100 = -0.05/10 * 100 = -0.5%
        assert!((drift - (-0.5)).abs() < 0.01);
    }
}
