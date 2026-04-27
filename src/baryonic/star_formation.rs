//! Star Formation Module
//!
//! Criteria (ALL required):
//! 1. ρ+_local > 100 × ρ̄+
//! 2. T < 1000 K
//! 3. ∇·v < 0 (converging flow)
//! 4. M_local > M_Jeans

use super::pressure::sound_speed;

/// Particle type constants
/// type = 0: gas m+ (positive mass gas)
/// type = 1: sink particle (star)
/// type = 255: m- (negative mass)
pub mod particle_type {
    pub const GAS_PLUS: u8 = 0;
    pub const SINK_STAR: u8 = 1;
    pub const MASS_MINUS: u8 = 255;
}

/// Convert sign to particle type (for non-sink particles)
#[inline]
pub fn sign_to_type(sign: i8) -> u8 {
    if sign > 0 {
        particle_type::GAS_PLUS
    } else {
        particle_type::MASS_MINUS
    }
}

/// Check if particle is a sink (star)
#[inline]
pub fn is_sink(ptype: u8) -> bool {
    ptype == particle_type::SINK_STAR
}

/// Check if particle is gas (should feel pressure)
#[inline]
pub fn is_gas(ptype: u8) -> bool {
    ptype == particle_type::GAS_PLUS
}

/// Check if particle is negative mass
#[inline]
pub fn is_negative_mass(ptype: u8) -> bool {
    ptype == particle_type::MASS_MINUS
}

/// Density threshold factor for star formation
pub const RHO_SF_FACTOR: f64 = 100.0;

/// Temperature threshold for star formation [K]
pub const T_SF: f64 = 1000.0;

/// Gravitational constant in code units [Mpc³ M_sun⁻¹ Gyr⁻²]
const G_CODE: f64 = 4.499e-15;

/// Jeans mass
/// M_Jeans = π^(5/2) cs³ / (6 G^(3/2) ρ^(1/2))
pub fn jeans_mass(density: f64, cs: f64) -> f64 {
    std::f64::consts::PI.powf(2.5) * cs.powi(3)
        / (6.0 * G_CODE.powf(1.5) * density.sqrt())
}

/// Check if star formation criteria are met
pub fn should_form_star(
    density: f64,
    mean_density: f64,
    temp: f64,
    div_v: f64,
    cs: f64,
    local_mass: f64,
) -> bool {
    // All criteria must be satisfied
    density > RHO_SF_FACTOR * mean_density
        && temp < T_SF
        && div_v < 0.0
        && local_mass > jeans_mass(density, cs)
}

/// Compute divergence of velocity field from neighbors
pub fn velocity_divergence(
    pos_i: [f64; 3],
    vel_i: [f64; 3],
    neighbors: &[([f64; 3], [f64; 3], f64)], // (pos, vel, mass)
    h: f64,
    rho_i: f64,
) -> f64 {
    use super::sph::SphKernel;

    let mut div_v = 0.0;

    for &(pos_j, vel_j, mass_j) in neighbors {
        let r_vec = [
            pos_i[0] - pos_j[0],
            pos_i[1] - pos_j[1],
            pos_i[2] - pos_j[2],
        ];
        let r = (r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]).sqrt();

        if r < 1e-10 { continue; }

        let v_rel = [
            vel_j[0] - vel_i[0],
            vel_j[1] - vel_i[1],
            vel_j[2] - vel_i[2],
        ];

        let dw = SphKernel::dw_dr(r, h);

        // ∇·v ≈ Σⱼ (mⱼ/ρᵢ) × (vⱼ - vᵢ) · ∇W
        let v_dot_r = v_rel[0]*r_vec[0] + v_rel[1]*r_vec[1] + v_rel[2]*r_vec[2];
        div_v += (mass_j / rho_i) * (v_dot_r / r) * dw;
    }

    div_v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_sf_hot_gas() {
        // Hot gas → never forms stars
        assert!(!should_form_star(1e12, 1e8, 1e6, -1.0,
            sound_speed(1e6), 1e15),
            "SF dans gaz chaud!");
    }

    #[test]
    fn test_no_sf_diverging() {
        // Expanding gas → no star formation
        assert!(!should_form_star(1e12, 1e8, 100.0, 1.0,
            sound_speed(100.0), 1e15),
            "SF dans gaz divergent!");
    }

    #[test]
    fn test_no_sf_underdense() {
        // Underdense region
        assert!(!should_form_star(1e8, 1e8, 100.0, -1.0,
            sound_speed(100.0), 1e15),
            "SF dans région sous-dense!");
    }

    #[test]
    fn test_sf_all_criteria() {
        // All criteria satisfied → star formation
        assert!(should_form_star(1e12, 1e8, 100.0, -1.0,
            sound_speed(100.0), 1e15),
            "Pas de SF malgré tous critères!");
    }

    #[test]
    fn test_jeans_mass_scaling_cs() {
        // M_Jeans ∝ cs³
        let rho = 1e8;
        let mj1 = jeans_mass(rho, 0.01);
        let mj2 = jeans_mass(rho, 0.02); // cs × 2
        assert!((mj2 / mj1 - 8.0).abs() < 0.1,
            "Scaling Jeans cs incorrect: ratio = {}", mj2/mj1);
    }

    #[test]
    fn test_jeans_mass_density_scaling() {
        // M_Jeans ∝ ρ^(-1/2)
        let cs = 0.01;
        let mj1 = jeans_mass(1e8, cs);
        let mj2 = jeans_mass(4e8, cs); // ρ × 4
        assert!((mj1 / mj2 - 2.0).abs() < 0.1,
            "Scaling densité Jeans incorrect: ratio = {}", mj1/mj2);
    }

    #[test]
    fn test_sink_particle_type_classification() {
        use super::particle_type::*;
        use super::{sign_to_type, is_sink, is_gas, is_negative_mass};

        // Gas m+ → type 0
        assert_eq!(sign_to_type(1), GAS_PLUS);
        assert!(is_gas(GAS_PLUS));
        assert!(!is_sink(GAS_PLUS));
        assert!(!is_negative_mass(GAS_PLUS));

        // m- → type 255
        assert_eq!(sign_to_type(-1), MASS_MINUS);
        assert!(is_negative_mass(MASS_MINUS));
        assert!(!is_sink(MASS_MINUS));
        assert!(!is_gas(MASS_MINUS));

        // Sink (star) → type 1
        assert!(is_sink(SINK_STAR));
        assert!(!is_gas(SINK_STAR));
        assert!(!is_negative_mass(SINK_STAR));
    }

    #[test]
    fn test_sink_particle_no_pressure() {
        // Sink particles should not feel thermal pressure
        // This is enforced by is_gas() returning false for sinks
        use super::particle_type::SINK_STAR;
        use super::is_gas;

        assert!(!is_gas(SINK_STAR),
            "Sink particles should not be gas (no pressure)!");
    }

    #[test]
    fn test_sink_particle_gravity_only() {
        // Sink particles feel gravity but no thermal/pressure forces
        // Sign > 0 means they attract other m+ and repel m-
        use super::particle_type::SINK_STAR;
        use super::{is_sink, is_negative_mass};

        // Sink is positive mass (attracts m+, repels m-)
        assert!(is_sink(SINK_STAR));
        assert!(!is_negative_mass(SINK_STAR),
            "Sink should be positive mass for gravity!");
    }
}
