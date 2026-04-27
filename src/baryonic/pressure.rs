//! Thermal Pressure Module
//!
//! P = ρ × k_B × T / (μ_mol × m_p)
//! cs² = γ × k_B × T / (μ_mol × m_p)

use super::sph::SphKernel;

/// Mean molecular weight (ionized hydrogen + helium)
pub const MU_MOL: f64 = 0.6;

/// Adiabatic index (monoatomic gas)
pub const GAMMA: f64 = 5.0 / 3.0;

/// k_B/m_p in code units [(Mpc/Gyr)²/K]
/// k_B/m_p = 8.314e7 erg/(g·K) in CGS
/// Convert: × (Mpc/cm)² × (s/Gyr)² = × (3.24e-25)² × (3.156e16)²
/// = 8.314e7 × 1.05e-16 = 8.7e-9
pub const K_B_OVER_MP_CODE: f64 = 8.7e-9;

/// Thermal pressure P = ρ (k_B/m_p) T / μ
/// Returns pressure in code units [M_sun/(Mpc·Gyr²)]
pub fn pressure(density: f64, temp: f64) -> f64 {
    density * K_B_OVER_MP_CODE * temp / MU_MOL
}

/// Sound speed cs = sqrt(γ (k_B/m_p) T / μ)
/// Returns cs in code units [Mpc/Gyr]
/// At T=1e4 K: cs ≈ 10 km/s ≈ 0.01 Mpc/Gyr
pub fn sound_speed(temp: f64) -> f64 {
    (GAMMA * K_B_OVER_MP_CODE * temp / MU_MOL).sqrt()
}

/// SPH pressure acceleration
/// a_i = -Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) ∇W(rᵢⱼ, h)
pub fn pressure_acceleration(
    pos_i: [f64; 3], rho_i: f64, p_i: f64, h_i: f64,
    neighbors: &[(f64, f64, [f64; 3], f64, f64)], // (mass, rho, pos, p, h)
) -> [f64; 3] {
    let mut accel = [0.0f64; 3];
    let coeff_i = p_i / (rho_i * rho_i);

    for &(mass_j, rho_j, pos_j, p_j, h_j) in neighbors {
        let coeff_j = p_j / (rho_j * rho_j);

        let r_vec = [
            pos_i[0] - pos_j[0],
            pos_i[1] - pos_j[1],
            pos_i[2] - pos_j[2],
        ];
        let r = (r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]).sqrt();

        if r < 1e-10 { continue; }

        let h_avg = 0.5 * (h_i + h_j);
        let dw = SphKernel::dw_dr(r, h_avg);

        for k in 0..3 {
            accel[k] -= mass_j * (coeff_i + coeff_j) * dw * r_vec[k] / r;
        }
    }
    accel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_positive() {
        assert!(pressure(1e8, 1e4) > 0.0, "Pression négative!");
    }

    #[test]
    fn test_sound_speed_10km_s() {
        // cs(T=1e4 K) ≈ 10 km/s = 0.01022 Mpc/Gyr
        let cs = sound_speed(1e4);
        // Allow 50% tolerance due to unit conversion
        assert!(cs > 0.005 && cs < 0.02,
            "cs incorrect: {} Mpc/Gyr (attendu ~0.01)", cs);
    }

    #[test]
    fn test_sound_speed_sqrt_t() {
        // cs ∝ T^(1/2)
        let cs1 = sound_speed(1e4);
        let cs2 = sound_speed(4e4);
        assert!((cs2 / cs1 - 2.0).abs() < 0.01,
            "Scaling cs incorrect: ratio = {}", cs2/cs1);
    }

    #[test]
    fn test_pressure_antisymmetric() {
        // Forces de pression antisymétriques (Newton III)
        let rho = 1e8;
        let temp = 1e4;
        let h = 2.0;
        let mass = 1e10;
        let p = pressure(rho, temp);

        let pos_i = [0.0, 0.0, 0.0];
        let pos_j = [1.0, 0.0, 0.0];

        let a_i = pressure_acceleration(pos_i, rho, p, h,
            &[(mass, rho, pos_j, p, h)]);
        let a_j = pressure_acceleration(pos_j, rho, p, h,
            &[(mass, rho, pos_i, p, h)]);

        assert!((a_i[0] + a_j[0]).abs() < 1e-10,
            "Forces non antisymétriques: {} vs {}", a_i[0], a_j[0]);
    }

    #[test]
    fn test_pressure_zero_in_symmetric_field() {
        // Champ symétrique → force nulle sur particule centrale
        let rho = 1e8;
        let temp = 1e4;
        let h = 2.0;
        let mass = 1e10;
        let p = pressure(rho, temp);

        let neighbors = vec![
            (mass, rho, [1.0, 0.0, 0.0], p, h),
            (mass, rho, [-1.0, 0.0, 0.0], p, h),
            (mass, rho, [0.0, 1.0, 0.0], p, h),
            (mass, rho, [0.0, -1.0, 0.0], p, h),
            (mass, rho, [0.0, 0.0, 1.0], p, h),
            (mass, rho, [0.0, 0.0, -1.0], p, h),
        ];

        let a = pressure_acceleration([0.0, 0.0, 0.0], rho, p, h, &neighbors);
        assert!(a[0].abs() < 1e-8 && a[1].abs() < 1e-8 && a[2].abs() < 1e-8,
            "Force non nulle en champ symétrique: {:?}", a);
    }
}
