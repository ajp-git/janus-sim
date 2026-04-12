//! Peculiar Velocity and Repeller Detection Module
//!
//! Computes peculiar velocities and detects large-scale repellers.
//! Key Janus prediction: m- dominated voids act as cosmic repellers.
//!
//! Physics:
//! - v_pec = v_total - H(z) × r (departure from Hubble flow)
//! - Repeller: divergent velocity field pointing away from void center
//! - Attractor: convergent velocity field toward overdensity
//!
//! Reference: Hoffman et al. (2017), Courtois et al. (2017), Petit (2024)

/// H0 in km/s/Mpc (Janus value from Petit 2014)
pub const H0_KMS_MPC: f64 = 76.0;

/// Compute Hubble velocity at given distance
///
/// v_H = H0 × d
///
/// # Arguments
/// * `distance` - Distance [Mpc]
/// * `h0` - Hubble constant [km/s/Mpc]
///
/// # Returns
/// Hubble velocity [km/s]
pub fn hubble_velocity(distance: f64, h0: f64) -> f64 {
    h0 * distance
}

/// Compute peculiar velocity from total velocity
///
/// v_pec = v_total - v_Hubble
///
/// # Arguments
/// * `v_total` - Total observed velocity [km/s]
/// * `distance` - Distance to object [Mpc]
/// * `h0` - Hubble constant [km/s/Mpc]
///
/// # Returns
/// Peculiar velocity [km/s]
pub fn peculiar_velocity(v_total: f64, distance: f64, h0: f64) -> f64 {
    v_total - hubble_velocity(distance, h0)
}

/// Compute peculiar velocity field from particle velocities
///
/// # Arguments
/// * `positions` - Particle positions relative to observer [Mpc]
/// * `velocities` - Particle velocities [km/s]
/// * `h0` - Hubble constant [km/s/Mpc]
///
/// # Returns
/// Peculiar velocity vectors [km/s]
pub fn compute_peculiar_velocities(
    positions: &[[f64; 3]],
    velocities: &[[f64; 3]],
    h0: f64,
) -> Vec<[f64; 3]> {
    positions.iter()
        .zip(velocities.iter())
        .map(|(pos, vel)| {
            let r = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
            if r < 1e-10 {
                return *vel;  // At observer, v_pec = v
            }

            // Unit vector from observer to particle
            let r_hat = [pos[0]/r, pos[1]/r, pos[2]/r];

            // Radial velocity component
            let v_radial = vel[0]*r_hat[0] + vel[1]*r_hat[1] + vel[2]*r_hat[2];

            // Peculiar radial velocity
            let v_pec_radial = v_radial - h0 * r;

            // Full peculiar velocity (keep tangential component)
            let v_hubble = [h0*r*r_hat[0], h0*r*r_hat[1], h0*r*r_hat[2]];
            [
                vel[0] - v_hubble[0],
                vel[1] - v_hubble[1],
                vel[2] - v_hubble[2],
            ]
        })
        .collect()
}

/// Velocity divergence field on a grid
///
/// div(v) > 0 → expansion (repeller)
/// div(v) < 0 → contraction (attractor)
///
/// # Arguments
/// * `velocities` - Velocity grid [grid_size³][3] in km/s
/// * `grid_size` - Grid resolution
/// * `cell_size` - Physical cell size [Mpc]
///
/// # Returns
/// Divergence field [km/s/Mpc]
pub fn velocity_divergence(
    velocities: &[[f64; 3]],
    grid_size: usize,
    cell_size: f64,
) -> Vec<f64> {
    let n_cells = grid_size.pow(3);
    let mut div = vec![0.0; n_cells];

    for ix in 1..grid_size-1 {
        for iy in 1..grid_size-1 {
            for iz in 1..grid_size-1 {
                let idx = ix * grid_size * grid_size + iy * grid_size + iz;

                // Central differences
                let idx_xp = (ix+1) * grid_size * grid_size + iy * grid_size + iz;
                let idx_xm = (ix-1) * grid_size * grid_size + iy * grid_size + iz;
                let idx_yp = ix * grid_size * grid_size + (iy+1) * grid_size + iz;
                let idx_ym = ix * grid_size * grid_size + (iy-1) * grid_size + iz;
                let idx_zp = ix * grid_size * grid_size + iy * grid_size + (iz+1);
                let idx_zm = ix * grid_size * grid_size + iy * grid_size + (iz-1);

                let dvx_dx = (velocities[idx_xp][0] - velocities[idx_xm][0]) / (2.0 * cell_size);
                let dvy_dy = (velocities[idx_yp][1] - velocities[idx_ym][1]) / (2.0 * cell_size);
                let dvz_dz = (velocities[idx_zp][2] - velocities[idx_zm][2]) / (2.0 * cell_size);

                div[idx] = dvx_dx + dvy_dy + dvz_dz;
            }
        }
    }

    div
}

/// Detect repeller from density and velocity fields
///
/// A repeller is a region where:
/// 1. Density is below average (void or m- dominated)
/// 2. Velocity divergence is positive (outflow)
/// 3. Velocities point away from center
///
/// # Arguments
/// * `positions` - Particle positions [Mpc]
/// * `velocities` - Particle velocities [km/s]
/// * `signs` - Particle mass signs
/// * `center` - Search center [Mpc]
/// * `r_max` - Maximum search radius [Mpc]
///
/// # Returns
/// Option<RepellerInfo> if found
#[derive(Debug, Clone)]
pub struct RepellerInfo {
    /// Center position [Mpc]
    pub center: [f64; 3],
    /// Effective radius [Mpc]
    pub radius: f64,
    /// Mean outflow velocity [km/s]
    pub v_outflow: f64,
    /// m- fraction in region
    pub f_minus: f64,
    /// Is this a significant repeller?
    pub is_significant: bool,
}

pub fn detect_repeller(
    positions: &[[f64; 3]],
    velocities: &[[f64; 3]],
    signs: &[f64],
    center: [f64; 3],
    r_max: f64,
    v_threshold: f64,
) -> Option<RepellerInfo> {
    // Collect particles within r_max of center
    let mut n_plus = 0;
    let mut n_minus = 0;
    let mut v_radial_sum = 0.0;
    let mut count = 0;

    for i in 0..positions.len() {
        let dx = positions[i][0] - center[0];
        let dy = positions[i][1] - center[1];
        let dz = positions[i][2] - center[2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();

        if r < r_max && r > 1e-10 {
            // Count by sign
            if signs[i] > 0.0 {
                n_plus += 1;
            } else {
                n_minus += 1;
            }

            // Radial velocity (positive = outflow)
            let r_hat = [dx/r, dy/r, dz/r];
            let v_radial = velocities[i][0]*r_hat[0]
                         + velocities[i][1]*r_hat[1]
                         + velocities[i][2]*r_hat[2];
            v_radial_sum += v_radial;
            count += 1;
        }
    }

    if count < 10 {
        return None;
    }

    let n_total = n_plus + n_minus;
    let f_minus = n_minus as f64 / n_total as f64;
    let v_outflow = v_radial_sum / count as f64;

    // A repeller has:
    // - Positive mean outflow velocity (matter moving away)
    // - Significant m- fraction (Janus prediction)
    let is_significant = v_outflow > v_threshold && f_minus > 0.3;

    Some(RepellerInfo {
        center,
        radius: r_max,
        v_outflow,
        f_minus,
        is_significant,
    })
}

/// Detect attractor from density and velocity fields
///
/// An attractor is a region where:
/// 1. Density is above average (m+ dominated)
/// 2. Velocity divergence is negative (inflow)
/// 3. Velocities point toward center
#[derive(Debug, Clone)]
pub struct AttractorInfo {
    /// Center position [Mpc]
    pub center: [f64; 3],
    /// Effective radius [Mpc]
    pub radius: f64,
    /// Mean inflow velocity [km/s]
    pub v_inflow: f64,
    /// m+ fraction in region
    pub f_plus: f64,
    /// Is this a significant attractor?
    pub is_significant: bool,
}

pub fn detect_attractor(
    positions: &[[f64; 3]],
    velocities: &[[f64; 3]],
    signs: &[f64],
    center: [f64; 3],
    r_max: f64,
    v_threshold: f64,
) -> Option<AttractorInfo> {
    let mut n_plus = 0;
    let mut n_minus = 0;
    let mut v_radial_sum = 0.0;
    let mut count = 0;

    for i in 0..positions.len() {
        let dx = positions[i][0] - center[0];
        let dy = positions[i][1] - center[1];
        let dz = positions[i][2] - center[2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();

        if r < r_max && r > 1e-10 {
            if signs[i] > 0.0 {
                n_plus += 1;
            } else {
                n_minus += 1;
            }

            // Radial velocity (negative = inflow)
            let r_hat = [dx/r, dy/r, dz/r];
            let v_radial = velocities[i][0]*r_hat[0]
                         + velocities[i][1]*r_hat[1]
                         + velocities[i][2]*r_hat[2];
            v_radial_sum += v_radial;
            count += 1;
        }
    }

    if count < 10 {
        return None;
    }

    let n_total = n_plus + n_minus;
    let f_plus = n_plus as f64 / n_total as f64;
    let v_inflow = -v_radial_sum / count as f64;  // Sign flip: negative radial = positive inflow

    // An attractor has:
    // - Positive mean inflow velocity (matter moving toward center)
    // - High m+ fraction
    let is_significant = v_inflow > v_threshold && f_plus > 0.6;

    Some(AttractorInfo {
        center,
        radius: r_max,
        v_inflow,
        f_plus,
        is_significant,
    })
}

/// Compute bulk flow velocity in a spherical region
///
/// Bulk flow is the coherent velocity of a region after subtracting Hubble flow.
///
/// # Arguments
/// * `positions` - Particle positions relative to center [Mpc]
/// * `velocities` - Particle velocities [km/s]
/// * `r_max` - Sphere radius [Mpc]
///
/// # Returns
/// (v_bulk, sigma_v) - Bulk velocity vector and velocity dispersion [km/s]
pub fn bulk_flow(
    positions: &[[f64; 3]],
    velocities: &[[f64; 3]],
    r_max: f64,
) -> ([f64; 3], f64) {
    let mut sum_v = [0.0; 3];
    let mut count = 0;

    for (pos, vel) in positions.iter().zip(velocities.iter()) {
        let r = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
        if r < r_max {
            sum_v[0] += vel[0];
            sum_v[1] += vel[1];
            sum_v[2] += vel[2];
            count += 1;
        }
    }

    if count == 0 {
        return ([0.0; 3], 0.0);
    }

    let v_bulk = [
        sum_v[0] / count as f64,
        sum_v[1] / count as f64,
        sum_v[2] / count as f64,
    ];

    // Compute velocity dispersion
    let mut sum_dv2 = 0.0;
    for (pos, vel) in positions.iter().zip(velocities.iter()) {
        let r = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
        if r < r_max {
            let dv = [vel[0] - v_bulk[0], vel[1] - v_bulk[1], vel[2] - v_bulk[2]];
            sum_dv2 += dv[0].powi(2) + dv[1].powi(2) + dv[2].powi(2);
        }
    }

    let sigma_v = (sum_dv2 / count as f64).sqrt();

    (v_bulk, sigma_v)
}

/// Hoffman et al. (2017) dipole repeller velocity scale
/// The Dipole Repeller is ~100 Mpc away with v ~ 200 km/s
pub const HOFFMAN_REPELLER_DISTANCE: f64 = 100.0;  // Mpc
pub const HOFFMAN_REPELLER_VELOCITY: f64 = 200.0;  // km/s

/// Check if repeller matches Hoffman et al. (2017) characteristics
pub fn is_hoffman_compatible(repeller: &RepellerInfo) -> bool {
    // Hoffman repeller: d ~ 100 Mpc, v ~ 200 km/s
    // Allow factor of 2 variation
    repeller.v_outflow > HOFFMAN_REPELLER_VELOCITY * 0.5 &&
    repeller.radius > HOFFMAN_REPELLER_DISTANCE * 0.3 &&
    repeller.is_significant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hubble_velocity() {
        // At 100 Mpc with H0 = 76 km/s/Mpc: v_H = 7600 km/s
        let v = hubble_velocity(100.0, 76.0);
        assert!((v - 7600.0).abs() < 1.0, "v_H(100 Mpc) = {} km/s", v);
    }

    #[test]
    fn test_peculiar_velocity() {
        // Total v = 8000 km/s at 100 Mpc → v_pec = 8000 - 7600 = 400 km/s
        let v_pec = peculiar_velocity(8000.0, 100.0, 76.0);
        assert!((v_pec - 400.0).abs() < 1.0, "v_pec = {} km/s", v_pec);
    }

    #[test]
    fn test_peculiar_velocity_infall() {
        // Total v = 7000 km/s at 100 Mpc → v_pec = 7000 - 7600 = -600 km/s (infall)
        let v_pec = peculiar_velocity(7000.0, 100.0, 76.0);
        assert!((v_pec + 600.0).abs() < 1.0, "v_pec = {} km/s (infall)", v_pec);
    }

    #[test]
    fn test_detect_repeller_outflow() {
        // Create particles flowing outward from center
        let n = 200;
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        let mut signs = Vec::new();

        for i in 0..n {
            let t = i as f64 / n as f64;
            let r = 20.0 + t * 30.0;  // 20-50 Mpc
            let theta = t * 10.0;
            let phi = t * 5.0;

            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            positions.push([x, y, z]);

            // Outward velocity ~200 km/s
            let r_norm = r;
            let v_out = 200.0;
            velocities.push([
                v_out * x / r_norm,
                v_out * y / r_norm,
                v_out * z / r_norm,
            ]);

            // Mostly m- in center (repeller signature)
            signs.push(if t < 0.5 { -1.0 } else { 1.0 });
        }

        let repeller = detect_repeller(
            &positions, &velocities, &signs,
            [0.0, 0.0, 0.0], 60.0, 100.0
        );

        assert!(repeller.is_some(), "Should detect repeller");
        let info = repeller.unwrap();
        assert!(info.v_outflow > 100.0, "v_outflow = {} km/s", info.v_outflow);
    }

    #[test]
    fn test_detect_attractor_inflow() {
        // Create particles flowing inward toward center
        let n = 200;
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        let mut signs = Vec::new();

        for i in 0..n {
            let t = i as f64 / n as f64;
            let r = 20.0 + t * 30.0;
            let theta = t * 10.0;
            let phi = t * 5.0;

            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            positions.push([x, y, z]);

            // Inward velocity ~-200 km/s
            let r_norm = r;
            let v_in = -200.0;
            velocities.push([
                v_in * x / r_norm,
                v_in * y / r_norm,
                v_in * z / r_norm,
            ]);

            // Mostly m+ (attractor signature)
            signs.push(if t > 0.2 { 1.0 } else { -1.0 });
        }

        let attractor = detect_attractor(
            &positions, &velocities, &signs,
            [0.0, 0.0, 0.0], 60.0, 100.0
        );

        assert!(attractor.is_some(), "Should detect attractor");
        let info = attractor.unwrap();
        assert!(info.v_inflow > 100.0, "v_inflow = {} km/s", info.v_inflow);
    }

    #[test]
    fn test_bulk_flow() {
        // Create particles with coherent bulk velocity
        let n = 100;
        let v_bulk_true = [150.0, -50.0, 30.0];

        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                [t * 50.0, t * 30.0, t * 20.0]
            })
            .collect();

        // Add bulk velocity + small dispersion
        let velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let noise = (i as f64 * 0.1).sin() * 10.0;
                [
                    v_bulk_true[0] + noise,
                    v_bulk_true[1] - noise,
                    v_bulk_true[2] + noise * 0.5,
                ]
            })
            .collect();

        let (v_bulk, sigma) = bulk_flow(&positions, &velocities, 100.0);

        assert!((v_bulk[0] - v_bulk_true[0]).abs() < 20.0,
            "v_bulk_x = {} (expected ~150)", v_bulk[0]);
        assert!((v_bulk[1] - v_bulk_true[1]).abs() < 20.0,
            "v_bulk_y = {} (expected ~-50)", v_bulk[1]);
        assert!(sigma < 100.0, "sigma_v = {} (should be small)", sigma);
    }

    #[test]
    fn test_no_repeller_uniform() {
        // Uniform distribution with no coherent outflow
        let n = 100;
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                [
                    (t * 20.0).sin() * 30.0,
                    (t * 30.0).cos() * 30.0,
                    (t * 15.0).sin() * 30.0,
                ]
            })
            .collect();

        // Random velocities
        let velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                [t.sin() * 50.0, t.cos() * 50.0, -t.sin() * 50.0]
            })
            .collect();

        let signs: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let repeller = detect_repeller(
            &positions, &velocities, &signs,
            [0.0, 0.0, 0.0], 50.0, 100.0
        );

        // Should find something but not significant (v_outflow < threshold)
        if let Some(info) = repeller {
            assert!(!info.is_significant || info.v_outflow.abs() < 100.0,
                "Should not find significant repeller in random field");
        }
    }

    #[test]
    fn test_hoffman_compatible() {
        let repeller = RepellerInfo {
            center: [0.0, 0.0, 0.0],
            radius: 100.0,
            v_outflow: 200.0,
            f_minus: 0.5,
            is_significant: true,
        };

        assert!(is_hoffman_compatible(&repeller),
            "Repeller with v=200, r=100 should be Hoffman-compatible");
    }
}
