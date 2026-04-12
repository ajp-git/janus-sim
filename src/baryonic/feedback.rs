//! Supernova Feedback Module
//!
//! Energy injection from Type II supernovae after star formation.
//!
//! Physics:
//! - E_SN = 10^51 erg per SN
//! - 1 SN per ~100 M_sun of stars (Salpeter IMF)
//! - Energy deposited as thermal + kinetic
//!
//! Reference: Springel & Hernquist (2003)

/// Boltzmann constant [erg/K]
pub const K_B: f64 = 1.381e-16;

/// SN energy per event [erg]
pub const E_SN_ERG: f64 = 1e51;

/// Solar mass in grams
const M_SUN_G: f64 = 1.989e33;

/// Stellar mass per SN [M_sun]
/// Salpeter IMF: ~1 SN per 100 M_sun of stars formed
pub const M_STAR_PER_SN: f64 = 100.0;

/// SN energy per unit stellar mass [erg/M_sun]
pub const E_SN_PER_MSUN: f64 = E_SN_ERG / M_STAR_PER_SN;

/// SN specific energy [erg/g] = E_SN / (M_star_per_SN × M_sun)
pub const E_SN_SPECIFIC: f64 = E_SN_ERG / (M_STAR_PER_SN * M_SUN_G);

/// SN energy in km²/s² (more practical for simulation)
/// 1 erg/g = 1 cm²/s² = 1e-10 km²/s²
pub const E_SN_KM2S2: f64 = E_SN_SPECIFIC * 1e-10;

/// Feedback efficiency (fraction of SN energy coupled to gas)
pub const EPSILON_FB: f64 = 0.1;  // 10% efficiency

/// Delay time for SN after star formation [Gyr]
/// Type II SNe: ~10-30 Myr delay (massive stars)
pub const T_DELAY_SN: f64 = 0.01;  // 10 Myr

/// SN injection mode
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeedbackMode {
    /// Pure thermal: all energy goes to temperature
    Thermal,
    /// Pure kinetic: all energy goes to velocity kick
    Kinetic,
    /// Hybrid: split between thermal and kinetic
    Hybrid { thermal_fraction: f64 },
}

impl Default for FeedbackMode {
    fn default() -> Self {
        FeedbackMode::Hybrid { thermal_fraction: 0.5 }
    }
}

/// Compute temperature increase from SN feedback
///
/// # Arguments
/// * `stellar_mass` - Mass of stars formed [M_sun]
/// * `gas_mass` - Mass of gas receiving feedback [M_sun]
/// * `efficiency` - Feedback efficiency (0-1)
///
/// # Returns
/// Temperature increase [K]
pub fn sn_thermal_heating(stellar_mass: f64, gas_mass: f64, efficiency: f64) -> f64 {
    if gas_mass < 1e-30 {
        return 0.0;
    }

    // Energy deposited [erg]
    let e_deposited = stellar_mass * E_SN_PER_MSUN * efficiency;

    // Convert to specific energy [erg/g]
    let e_specific = e_deposited / (gas_mass * M_SUN_G);

    // Temperature increase: ΔT = (2/3) × (μ m_H / k_B) × e_specific
    // For fully ionized: μ ≈ 0.6, m_H = 1.67e-24 g
    let mu = 0.6;
    let m_h = 1.67e-24;

    (2.0 / 3.0) * (mu * m_h / K_B) * e_specific
}

/// Compute velocity kick from SN feedback
///
/// # Arguments
/// * `stellar_mass` - Mass of stars formed [M_sun]
/// * `gas_mass` - Mass of gas receiving feedback [M_sun]
/// * `efficiency` - Feedback efficiency (0-1)
///
/// # Returns
/// Velocity kick magnitude [km/s]
pub fn sn_velocity_kick(stellar_mass: f64, gas_mass: f64, efficiency: f64) -> f64 {
    if gas_mass < 1e-30 {
        return 0.0;
    }

    // E_SN_KM2S2 is specific energy [km²/s²] per gram of stars
    // Total energy = stellar_mass × M_SUN × E_SN_KM2S2 × efficiency [km² g/s²]
    // v² = 2 × E_total / (gas_mass × M_SUN) = 2 × stellar_mass × E_SN_KM2S2 × efficiency / gas_mass
    let v_squared = 2.0 * stellar_mass * E_SN_KM2S2 * efficiency / gas_mass;

    if v_squared > 0.0 {
        v_squared.sqrt()
    } else {
        0.0
    }
}

/// Apply SN feedback to gas particle
///
/// # Arguments
/// * `mode` - Feedback mode (thermal, kinetic, or hybrid)
/// * `stellar_mass` - Mass of stars formed [M_sun]
/// * `gas_mass` - Mass of gas receiving feedback [M_sun]
/// * `gas_temp` - Current gas temperature [K]
/// * `gas_vel` - Current gas velocity [km/s, mutable]
/// * `direction` - Unit vector for velocity kick
/// * `efficiency` - Feedback efficiency
///
/// # Returns
/// New gas temperature [K]
pub fn apply_sn_feedback(
    mode: FeedbackMode,
    stellar_mass: f64,
    gas_mass: f64,
    gas_temp: f64,
    gas_vel: &mut [f64; 3],
    direction: [f64; 3],
    efficiency: f64,
) -> f64 {
    let (thermal_frac, kinetic_frac) = match mode {
        FeedbackMode::Thermal => (1.0, 0.0),
        FeedbackMode::Kinetic => (0.0, 1.0),
        FeedbackMode::Hybrid { thermal_fraction } => {
            (thermal_fraction, 1.0 - thermal_fraction)
        }
    };

    // Thermal heating
    let dt = sn_thermal_heating(stellar_mass, gas_mass, efficiency * thermal_frac);
    let new_temp = gas_temp + dt;

    // Kinetic kick
    if kinetic_frac > 0.0 {
        let v_kick = sn_velocity_kick(stellar_mass, gas_mass, efficiency * kinetic_frac);
        gas_vel[0] += v_kick * direction[0];
        gas_vel[1] += v_kick * direction[1];
        gas_vel[2] += v_kick * direction[2];
    }

    new_temp
}

/// Star formation rate using Schmidt-Kennicutt law
/// SFR ∝ ρ^1.4 / t_ff
///
/// # Arguments
/// * `density` - Gas density [M_sun/Mpc³]
/// * `efficiency` - Star formation efficiency per freefall time
///
/// # Returns
/// SFR density [M_sun/Mpc³/Gyr]
pub fn schmidt_kennicutt_sfr(density: f64, efficiency: f64) -> f64 {
    // Freefall time: t_ff = sqrt(3π / 32Gρ)
    // G in code units: 4.499e-15 Mpc³ M_sun⁻¹ Gyr⁻²
    let g_code = 4.499e-15;
    let t_ff = (3.0 * std::f64::consts::PI / (32.0 * g_code * density)).sqrt();

    // SFR = ε × ρ / t_ff
    efficiency * density / t_ff.max(1e-10)
}

/// Probability of star formation in timestep dt
///
/// # Arguments
/// * `density` - Local gas density [M_sun/Mpc³]
/// * `efficiency` - Star formation efficiency
/// * `dt` - Timestep [Gyr]
///
/// # Returns
/// Probability of forming a star (0-1)
pub fn sf_probability(density: f64, efficiency: f64, dt: f64) -> f64 {
    let sfr = schmidt_kennicutt_sfr(density, efficiency);

    // P = SFR × dt / ρ = ε × dt / t_ff
    let prob = sfr * dt / density.max(1e-30);

    prob.min(1.0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sn_energy_units() {
        // E_SN should be ~5×10^5 km²/s² per M_sun of stars
        // Reference: 10^51 erg / (100 M_sun × 2e33 g) = 5e15 erg/g = 5e5 km²/s²
        assert!((E_SN_KM2S2 - 5.0e5).abs() < 1e5,
            "E_SN = {:.2e} km²/s² (expected ~5e5)", E_SN_KM2S2);
    }

    #[test]
    fn test_thermal_heating_scaling() {
        // More stellar mass → more heating
        let dt1 = sn_thermal_heating(1e6, 1e8, 0.1);
        let dt2 = sn_thermal_heating(2e6, 1e8, 0.1);

        assert!((dt2 / dt1 - 2.0).abs() < 0.01,
            "ΔT should scale with M_star: ratio = {}", dt2/dt1);
    }

    #[test]
    fn test_velocity_kick_magnitude() {
        // v_kick should be in reasonable range (10-1000 km/s)
        let v = sn_velocity_kick(1e6, 1e7, 0.1);

        assert!(v > 10.0 && v < 1000.0,
            "v_kick = {:.1} km/s out of range [10, 1000]", v);
    }

    #[test]
    fn test_feedback_mode_thermal() {
        let mut vel = [0.0, 0.0, 0.0];
        let t = apply_sn_feedback(
            FeedbackMode::Thermal,
            1e6, 1e7, 1e4,
            &mut vel, [1.0, 0.0, 0.0], 0.1
        );

        // Temperature should increase
        assert!(t > 1e4, "Thermal feedback should heat: T = {}", t);
        // Velocity should be unchanged
        assert!(vel[0].abs() < 1e-10, "Pure thermal should not kick: v = {:?}", vel);
    }

    #[test]
    fn test_feedback_mode_kinetic() {
        let mut vel = [0.0, 0.0, 0.0];
        let t = apply_sn_feedback(
            FeedbackMode::Kinetic,
            1e6, 1e7, 1e4,
            &mut vel, [1.0, 0.0, 0.0], 0.1
        );

        // Temperature should be unchanged
        assert!((t - 1e4).abs() < 1.0, "Pure kinetic should not heat: T = {}", t);
        // Velocity should increase
        assert!(vel[0] > 1.0, "Kinetic should kick: v = {:?}", vel);
    }

    #[test]
    fn test_schmidt_kennicutt_sfr() {
        // SFR ∝ ρ^1 / t_ff ∝ ρ^1.5
        let sfr1 = schmidt_kennicutt_sfr(1e12, 0.01);
        let sfr2 = schmidt_kennicutt_sfr(4e12, 0.01);  // 4× density

        // ratio should be ~8 (4^1.5)
        let ratio = sfr2 / sfr1;
        assert!(ratio > 6.0 && ratio < 10.0,
            "SFR scaling: ratio = {:.2} (expected ~8)", ratio);
    }

    #[test]
    fn test_sf_probability_bounds() {
        let prob = sf_probability(1e15, 0.1, 0.01);

        assert!(prob >= 0.0 && prob <= 1.0,
            "SF probability = {} out of [0,1]", prob);
    }
}
