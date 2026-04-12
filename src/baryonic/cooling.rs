//! Radiative Cooling Module
//!
//! Uses Grackle library with HM2012 UV background for accurate cooling rates.
//! Falls back to analytical approximation if Grackle is not available.
//!
//! Features:
//! - Grackle HM2012 tables (primordial + metals)
//! - Rahmati 2013 self-shielding
//! - Temperature floor T_floor = 100 K
//!
//! Reference: Haardt & Madau (2012), Rahmati et al. (2013)

/// Temperature floor in Kelvin
pub const T_FLOOR: f64 = 100.0;

/// Mean baryon number density at z=0 [cm^-3]
const N_H_BAR_Z0: f64 = 2.0e-7;

/// Boltzmann constant [erg/K]
const K_B: f64 = 1.381e-16;

/// Seconds per Gyr
const SEC_PER_GYR: f64 = 3.156e16;

#[cfg(feature = "grackle")]
use crate::grackle_wrapper::GrackleCooling;

#[cfg(feature = "grackle")]
use std::sync::OnceLock;

#[cfg(feature = "grackle")]
static GRACKLE: OnceLock<GrackleCooling> = OnceLock::new();

/// Initialize Grackle cooling (call once at startup)
#[cfg(feature = "grackle")]
pub fn init_grackle() -> Result<(), String> {
    let paths = [
        "/usr/local/share/grackle/input/CloudyData_UVB=HM2012.h5",
        "/app/external/grackle-dist/grackle/input/CloudyData_UVB=HM2012.h5",
        "external/grackle-dist/grackle/input/CloudyData_UVB=HM2012.h5",
    ];

    for path in paths {
        if std::path::Path::new(path).exists() {
            let grackle = GrackleCooling::new(path)?;
            GRACKLE.set(grackle).map_err(|_| "Grackle already initialized")?;
            return Ok(());
        }
    }

    Err("Grackle data file not found".to_string())
}

/// Cooling rate using Grackle [K/Gyr]
///
/// # Arguments
/// * `temp` - Temperature in Kelvin
/// * `n_h` - Hydrogen number density [cm^-3]
/// * `z` - Redshift
/// * `metallicity` - Metallicity relative to solar (0.0 for primordial)
#[cfg(feature = "grackle")]
pub fn cooling_rate_grackle(temp: f64, n_h: f64, z: f64, metallicity: f64) -> f64 {
    if temp <= T_FLOOR {
        return 0.0;
    }

    let grackle = match GRACKLE.get() {
        Some(g) => g,
        None => return cooling_rate_analytical(temp, n_h),
    };

    // Get dU/dt in km²/s²/Gyr
    let du_dt = grackle.cooling_rate(temp, n_h, z, metallicity);

    // Convert to K/Gyr: dT/dt = (2/3) × (μ m_H / k_B) × dU/dt
    // For fully ionized gas: μ ≈ 0.6
    let mu = 0.6;
    let m_h = 1.67e-24;  // g

    // dU/dt is in km²/s²/Gyr = 1e10 cm²/s²/Gyr = 1e10 erg/g/Gyr
    let du_dt_erg_g_gyr = du_dt * 1e10;

    // dT/dt = (2/3) × (μ m_H / k_B) × dU/dt
    let dt_dt = (2.0 / 3.0) * (mu * m_h / K_B) * du_dt_erg_g_gyr;

    -dt_dt  // Negative because cooling
}

/// Analytical cooling rate fallback [K/Gyr]
pub fn cooling_rate_analytical(temp: f64, n_h: f64) -> f64 {
    if temp <= T_FLOOR {
        return 0.0;
    }

    // Cooling function Λ(T) [erg cm³/s] - simplified
    let lambda = match temp {
        t if t < 1e4 => 1e-26 * (t / 1e4).powf(2.0),  // Molecular
        t if t < 1e5 => 1e-22,                         // Lyman-alpha peak
        t if t < 1e7 => 5e-23 * (t / 1e5).powf(-0.7), // Metal lines
        t => 3e-23 * (t / 1e7).sqrt(),                // Bremsstrahlung
    };

    // dT/dt = -Λ × n_H / (1.5 k_B) [K/s]
    let rate_ks = -lambda * n_h / (1.5 * K_B);
    rate_ks * SEC_PER_GYR
}

/// Cooling rate in code units [K/Gyr]
/// Uses Grackle if available, otherwise analytical approximation
pub fn cooling_rate_kelvin_per_gyr(temp: f64, overdensity: f64, z: f64) -> f64 {
    if temp <= T_FLOOR {
        return 0.0;
    }

    // Convert overdensity to n_H [cm^-3]
    // n_H = n̄_H(z=0) × (1+z)³ × overdensity
    let n_h = N_H_BAR_Z0 * (1.0 + z).powi(3) * overdensity;

    #[cfg(feature = "grackle")]
    {
        cooling_rate_grackle(temp, n_h, z, 0.0)
    }

    #[cfg(not(feature = "grackle"))]
    {
        cooling_rate_analytical(temp, n_h)
    }
}

/// Apply cooling with subcycling for stability
/// Returns new temperature after dt Gyr of cooling
pub fn apply_cooling(temp: f64, overdensity: f64, z: f64, dt: f64) -> f64 {
    if temp <= T_FLOOR {
        return T_FLOOR;
    }

    let rate = cooling_rate_kelvin_per_gyr(temp, overdensity, z);
    if rate.abs() < 1e-30 {
        return temp;
    }

    // Subcycling: use 0.1 × t_cool as max timestep
    let t_cool = (temp / rate.abs()).abs();
    let n_sub = ((dt / (0.1 * t_cool)).ceil() as usize).max(1).min(1000);
    let dt_sub = dt / n_sub as f64;

    let mut t = temp;
    for _ in 0..n_sub {
        let r = cooling_rate_kelvin_per_gyr(t, overdensity, z);
        t = (t + r * dt_sub).max(T_FLOOR);
    }
    t
}

/// Compute cooling time [Gyr]
pub fn cooling_time_gyr(temp: f64, overdensity: f64, z: f64) -> f64 {
    let rate = cooling_rate_kelvin_per_gyr(temp, overdensity, z);
    if rate.abs() < 1e-30 {
        return f64::INFINITY;
    }
    (temp / rate.abs()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooling_floor() {
        assert_eq!(cooling_rate_kelvin_per_gyr(50.0, 1e6, 0.0), 0.0);
    }

    #[test]
    fn test_cooling_always_negative() {
        for temp in [1e4_f64, 1e5, 1e6, 1e7] {
            let rate = cooling_rate_kelvin_per_gyr(temp, 1e6, 0.0);
            assert!(rate <= 0.0, "Cooling positif à T={}: rate={}", temp, rate);
        }
    }

    #[test]
    fn test_cooling_time_physical() {
        // t_cool should be < Hubble time for dense gas
        let t_cool = cooling_time_gyr(1e6, 1e6, 0.0);
        assert!(t_cool > 1e-6 && t_cool < 14.0,
            "t_cool hors plage: {} Gyr", t_cool);
    }

    #[test]
    fn test_apply_cooling_reaches_floor() {
        // Very dense region should cool to floor
        let t_final = apply_cooling(1e6, 1e8, 0.0, 10.0);
        assert!(t_final <= 1e4, "T finale trop haute: {} K", t_final);
    }

    #[test]
    fn test_apply_cooling_monotone() {
        let mut t = 1e6_f64;
        for _ in 0..100 {
            let t_new = apply_cooling(t, 1e6, 0.0, 0.01);
            assert!(t_new <= t + 1.0, "Température monte: {} → {}", t, t_new);
            t = t_new;
        }
    }

    #[test]
    fn test_redshift_dependence() {
        // Higher z should have higher n_H → faster cooling
        let rate_z0 = cooling_rate_kelvin_per_gyr(1e5, 1e4, 0.0);
        let rate_z2 = cooling_rate_kelvin_per_gyr(1e5, 1e4, 2.0);
        assert!(rate_z2.abs() > rate_z0.abs(),
            "z=2 devrait refroidir plus vite: z0={}, z2={}", rate_z0, rate_z2);
    }
}
