//! Grackle Cooling Library Wrapper
//!
//! Provides Rust bindings to the Grackle astrophysical cooling library.
//! Uses the HM2012 (Haardt & Madau 2012) UV background.
//!
//! Features:
//! - HM2012 UV background tables
//! - Rahmati 2013 self-shielding correction
//! - T_floor = 100 K applied after cooling calculation
//! - Output in simulation units [km²/s²/Gyr]

use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Once;

// FFI declarations for grackle_bridge
extern "C" {
    fn grackle_bridge_init(data_file_path: *const c_char) -> i32;
    fn grackle_bridge_cooling_rate(temperature_K: f64, density_cgs: f64, redshift: f64) -> f64;
    fn grackle_bridge_lambda_norm(temperature_K: f64, redshift: f64) -> f64;
    fn grackle_bridge_cleanup();
}

static INIT: Once = Once::new();
static mut INITIALIZED: bool = false;

/// Physical constants
const M_H: f64 = 1.6726e-24;    // Hydrogen mass [g]
const X_H: f64 = 0.76;          // Hydrogen mass fraction

/// Temperature floor [K]
pub const T_FLOOR: f64 = 100.0;

/// Unit conversions
/// 1 erg/g = 1e-10 km²/s²
const ERG_PER_G_TO_KM2_S2: f64 = 1e-10;
/// 1 Gyr = 3.156e16 s
const GYR_TO_S: f64 = 3.156e16;

/// Grackle cooling library wrapper
pub struct GrackleCooling {
    _private: (),
}

impl GrackleCooling {
    /// Initialize Grackle with the specified data file
    pub fn new<P: AsRef<Path>>(data_file: P) -> Result<Self, String> {
        let path = data_file.as_ref();
        if !path.exists() {
            return Err(format!("Grackle data file not found: {:?}", path));
        }

        let c_path = CString::new(path.to_str().ok_or("Invalid path encoding")?)
            .map_err(|e| format!("Invalid path: {}", e))?;

        let mut init_result = Ok(());

        INIT.call_once(|| {
            let result = unsafe { grackle_bridge_init(c_path.as_ptr()) };
            if result != 1 {
                init_result = Err("Grackle initialization failed".to_string());
            } else {
                unsafe { INITIALIZED = true };
            }
        });

        init_result?;

        if unsafe { !INITIALIZED } {
            return Err("Grackle was not initialized".to_string());
        }

        Ok(GrackleCooling { _private: () })
    }

    /// Compute cooling rate dU/dt in simulation units [km²/s²/Gyr]
    ///
    /// # Arguments
    /// * `temperature` - Gas temperature in Kelvin (T_floor applied internally)
    /// * `n_h` - Hydrogen number density [cm⁻³]
    /// * `redshift` - Cosmological redshift
    /// * `metallicity` - Metallicity relative to solar (0.0 for primordial)
    ///
    /// # Returns
    /// Specific cooling rate dU/dt in km²/s²/Gyr (positive = cooling)
    pub fn cooling_rate(&self, temperature: f64, n_h: f64, redshift: f64, _metallicity: f64) -> f64 {
        // Apply temperature floor
        let t_eff = temperature.max(T_FLOOR);

        // Convert n_H to density [g/cm³]
        let rho = n_h * M_H / X_H;

        // Get cooling rate from Grackle [erg/s/cm³]
        let cooling_erg_s_cm3 = unsafe { grackle_bridge_cooling_rate(t_eff, rho, redshift) };

        // Apply self-shielding correction (Rahmati 2013)
        let f_shield = self.self_shielding_factor(n_h, redshift);
        let cooling_shielded = cooling_erg_s_cm3 * f_shield;

        // Convert to specific rate [erg/s/g]
        let du_dt_erg_s_g = cooling_shielded / rho;

        // Convert to simulation units [km²/s²/Gyr]
        du_dt_erg_s_g * ERG_PER_G_TO_KM2_S2 * GYR_TO_S
    }

    /// Rahmati et al. 2013 self-shielding correction
    ///
    /// Reduces photo-heating in dense gas where UV is attenuated.
    /// Reference: Rahmati et al. 2013, MNRAS 430, 2427
    ///
    /// # Arguments
    /// * `n_h` - Hydrogen number density [cm⁻³]
    /// * `redshift` - Cosmological redshift
    ///
    /// # Returns
    /// Self-shielding factor f ∈ [0, 1] (1 = no shielding, 0 = fully shielded)
    fn self_shielding_factor(&self, n_h: f64, redshift: f64) -> f64 {
        // Characteristic density for self-shielding [cm⁻³]
        // n_0 ≈ 6.73e-3 * (1+z)³ from Rahmati 2013 eq. 13
        let n_0 = 6.73e-3 * (1.0 + redshift).powi(3);

        // Shielding exponent
        let alpha = 1.0;  // Rahmati 2013: α = 1 for UVB-dominated

        // Self-shielding factor (eq. 14)
        let x = n_h / n_0;
        let f = (1.0 + x.powf(2.0 * alpha)).powf(-0.5 / alpha);

        f.clamp(0.0, 1.0)
    }

    /// Compute normalized cooling function Λ/n_H² [erg·cm³/s]
    pub fn lambda_norm(&self, temperature: f64, redshift: f64) -> f64 {
        let t_eff = temperature.max(T_FLOOR);
        unsafe { grackle_bridge_lambda_norm(t_eff, redshift) }
    }

    /// Compute cooling time [Gyr]
    pub fn cooling_time_gyr(&self, temperature: f64, n_h: f64, redshift: f64) -> f64 {
        let k_b = 1.3807e-16;  // erg/K
        let mu = 0.6;          // mean molecular weight (ionized)

        let t_eff = temperature.max(T_FLOOR);
        let rho = n_h * M_H / X_H;
        let energy_density = (3.0 / 2.0) * (rho / (mu * M_H)) * k_b * t_eff;
        let lambda = self.lambda_norm(t_eff, redshift);
        let cooling_rate = n_h * n_h * lambda;

        if cooling_rate > 0.0 {
            (energy_density / cooling_rate) / GYR_TO_S
        } else {
            f64::INFINITY
        }
    }
}

impl Drop for GrackleCooling {
    fn drop(&mut self) {
        // Don't cleanup - Grackle uses global state
    }
}

/// Cleanup Grackle global state (call at program exit)
pub fn cleanup() {
    unsafe {
        if INITIALIZED {
            grackle_bridge_cleanup();
            INITIALIZED = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_at_10_4_5() {
        let data_path = std::env::var("GRACKLE_DATA_FILE")
            .unwrap_or_else(|_| "/usr/local/share/grackle/input/CloudyData_UVB=HM2012.h5".to_string());

        if !Path::new(&data_path).exists() {
            eprintln!("Skipping: Grackle data not found at {}", data_path);
            return;
        }

        let grackle = GrackleCooling::new(&data_path).expect("Failed to init Grackle");

        // Test: Λ(10^4.5 K, n_H=1e-3, z=0) ≈ 1.6e-22 erg·cm³/s (±30%)
        let t = 10f64.powf(4.5);  // 31623 K
        let lambda = grackle.lambda_norm(t, 0.0);

        println!("Λ(10^4.5 K) = {:.3e} erg·cm³/s", lambda);
        println!("Expected:   ≈ 1.6e-22 erg·cm³/s (±30%)");

        let expected = 1.6e-22;
        let tolerance = 0.30;  // 30%
        let ratio = lambda / expected;

        assert!(ratio > (1.0 - tolerance), "Too low: ratio = {:.2}", ratio);
        assert!(ratio < (1.0 + tolerance), "Too high: ratio = {:.2}", ratio);

        cleanup();
    }

    #[test]
    fn test_self_shielding() {
        let data_path = std::env::var("GRACKLE_DATA_FILE")
            .unwrap_or_else(|_| "/usr/local/share/grackle/input/CloudyData_UVB=HM2012.h5".to_string());

        if !Path::new(&data_path).exists() {
            return;
        }

        let grackle = GrackleCooling::new(&data_path).expect("Failed to init Grackle");

        // Low density: no shielding
        let f_low = grackle.self_shielding_factor(1e-5, 0.0);
        assert!(f_low > 0.9, "Low density should have f ≈ 1: {}", f_low);

        // High density: strong shielding
        let f_high = grackle.self_shielding_factor(1.0, 0.0);
        assert!(f_high < 0.1, "High density should have f ≈ 0: {}", f_high);

        println!("Self-shielding: f(n_H=1e-5) = {:.3}, f(n_H=1) = {:.3}", f_low, f_high);
    }
}
