/// Janus Cosmological Model — Core Library
/// 
/// Implements the bimetric cosmological model of Petit & D'Agostini
/// References:
///   - Petit, Margnat & Zejli (2024), EPJC 84:1226
///   - D'Agostini & Petit (2018), Astrophys. Space Sci. 363:139

pub mod friedmann;
pub mod nbody;
pub mod analysis;

#[cfg(feature = "cuda")]
pub mod nbody_gpu;

/// Physical constants (SI units)
pub mod constants {
    /// Speed of light (m/s)
    pub const C: f64 = 2.997_924_58e8;
    
    /// Gravitational constant (m³ kg⁻¹ s⁻²)
    pub const G: f64 = 6.674_30e-11;
    
    /// Hubble constant — Janus value (km/s/Mpc)
    /// Note: Janus is consistent with H0=70, not 67 (ΛCDM)
    pub const H0_KM_S_MPC: f64 = 70.0;
    
    /// H0 in SI (s⁻¹)
    pub const H0: f64 = H0_KM_S_MPC * 1e3 / MPC_M;
    
    /// Megaparsec in meters
    pub const MPC_M: f64 = 3.085_677_581_5e22;
    
    /// Megaparsec in km
    pub const MPC_KM: f64 = 3.085_677_581_5e19;
    
    /// Hubble time (s)
    pub const T_HUBBLE: f64 = 1.0 / H0;
    
    /// Critical density today (kg/m³)
    pub const RHO_CRIT: f64 = 3.0 * H0 * H0 / (8.0 * std::f64::consts::PI * G);
}

/// Janus interaction rules between masses
/// 
/// This is the core of the model that eliminates the runaway paradox.
/// Derived from the coupled field equations in the Newtonian limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MassSign {
    Positive,
    Negative,
}

/// Compute gravitational acceleration on particle i due to particle j
/// following Janus interaction rules:
///   - same sign  → attraction  (Newton)
///   - opposite sign → repulsion (anti-Newton)
///
/// Returns acceleration vector (m/s²)
pub fn janus_acceleration(
    ri: [f64; 3],  // position of particle i (m)
    rj: [f64; 3],  // position of particle j (m)
    mj: f64,       // |mass| of particle j (kg)
    sign_i: MassSign,
    sign_j: MassSign,
) -> [f64; 3] {
    let dx = rj[0] - ri[0];
    let dy = rj[1] - ri[1];
    let dz = rj[2] - ri[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();
    
    if r < 1e-10 { return [0.0; 3]; }
    
    // Interaction sign: +1 for attraction, -1 for repulsion
    let interaction = if sign_i == sign_j { 1.0 } else { -1.0 };
    
    // |a| = G * |mj| / r²
    // direction: toward j if attractive, away from j if repulsive
    let a_mag = interaction * constants::G * mj / r2;
    
    [
        a_mag * dx / r,
        a_mag * dy / r,
        a_mag * dz / r,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_positive_masses_attract() {
        let ri = [0.0, 0.0, 0.0];
        let rj = [1.0, 0.0, 0.0];
        let a = janus_acceleration(ri, rj, 1e30, MassSign::Positive, MassSign::Positive);
        // Should point toward +x (toward rj)
        assert!(a[0] > 0.0);
    }
    
    #[test]
    fn test_opposite_masses_repel() {
        let ri = [0.0, 0.0, 0.0];
        let rj = [1.0, 0.0, 0.0];
        let a = janus_acceleration(ri, rj, 1e30, MassSign::Positive, MassSign::Negative);
        // Should point away from rj (toward -x)
        assert!(a[0] < 0.0);
    }
    
    #[test]
    fn test_negative_masses_attract() {
        let ri = [0.0, 0.0, 0.0];
        let rj = [1.0, 0.0, 0.0];
        let a = janus_acceleration(ri, rj, 1e30, MassSign::Negative, MassSign::Negative);
        // Should point toward +x (toward rj)
        assert!(a[0] > 0.0);
    }
}
