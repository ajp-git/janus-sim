//! Weak Lensing Convergence Module
//!
//! Computes convergence maps κ(θ) for weak lensing analysis.
//! Key Janus prediction: κ > 0 inside halos, κ < 0 in surrounding m- shell.
//!
//! Physics:
//! - κ = Σ / Σ_crit (surface mass density / critical surface density)
//! - For Janus: m- contributes negative surface density
//!
//! Reference: Bartelmann & Schneider (2001), Petit (2024)

use std::f64::consts::PI;

/// Critical surface density for lensing
/// Σ_crit = c² / (4πG) × D_s / (D_d × D_ds)
///
/// # Arguments
/// * `d_lens` - Angular diameter distance to lens [Mpc]
/// * `d_source` - Angular diameter distance to source [Mpc]
/// * `d_lens_source` - Angular diameter distance from lens to source [Mpc]
///
/// # Returns
/// Σ_crit in [M_sun / Mpc²]
pub fn sigma_crit(d_lens: f64, d_source: f64, d_lens_source: f64) -> f64 {
    // c² / (4πG) in M_sun / Mpc
    // c = 9.716e-15 Mpc/s, G = 4.499e-48 Mpc³/(M_sun s²)
    // c²/(4πG) = (9.716e-15)² / (4π × 4.499e-48) = 1.67e18 M_sun/Mpc
    let c2_4pi_g = 1.67e18;  // M_sun / Mpc

    c2_4pi_g * d_source / (d_lens * d_lens_source + 1e-30)
}

/// Compute projected surface mass density Σ(r) for NFW profile
///
/// # Arguments
/// * `r` - Projected radius [Mpc]
/// * `r_s` - Scale radius [Mpc]
/// * `rho_s` - Characteristic density [M_sun/Mpc³]
///
/// # Returns
/// Σ(r) in [M_sun/Mpc²]
pub fn sigma_nfw(r: f64, r_s: f64, rho_s: f64) -> f64 {
    let x = r / r_s;
    if x < 1e-10 {
        // Central value: Σ(0) = 2 ρ_s r_s
        return 2.0 * rho_s * r_s;
    }

    let x2 = x * x;

    let f_x = if x < 1.0 {
        let sqrt_term = ((1.0 - x) / (1.0 + x)).abs().sqrt();
        1.0 / (x2 - 1.0) * (1.0 - 2.0 / (1.0 - x2).abs().sqrt() * sqrt_term.atanh())
    } else if x > 1.0 {
        let sqrt_term = ((x - 1.0) / (1.0 + x)).sqrt();
        1.0 / (x2 - 1.0) * (1.0 - 2.0 / (x2 - 1.0).sqrt() * sqrt_term.atan())
    } else {
        // x = 1
        1.0 / 3.0
    };

    2.0 * rho_s * r_s * f_x
}

/// Compute convergence κ for NFW profile
///
/// # Arguments
/// * `r` - Projected radius [Mpc]
/// * `r_s` - Scale radius [Mpc]
/// * `rho_s` - Characteristic density [M_sun/Mpc³]
/// * `sigma_crit` - Critical surface density [M_sun/Mpc²]
///
/// # Returns
/// κ (dimensionless)
pub fn kappa_nfw(r: f64, r_s: f64, rho_s: f64, sigma_crit: f64) -> f64 {
    sigma_nfw(r, r_s, rho_s) / sigma_crit
}

/// Compute convergence from particle data
///
/// Projects particles along one axis and computes 2D convergence map.
///
/// # Arguments
/// * `positions` - Particle positions [Mpc]
/// * `signs` - Particle mass signs (+1 or -1)
/// * `mass_per_particle` - Mass per particle [M_sun]
/// * `box_size` - Simulation box size [Mpc]
/// * `grid_size` - Output grid resolution
/// * `sigma_crit` - Critical surface density [M_sun/Mpc²]
/// * `projection_axis` - Axis to project along (0=x, 1=y, 2=z)
///
/// # Returns
/// 2D convergence map (grid_size × grid_size)
pub fn compute_kappa_map(
    positions: &[[f64; 3]],
    signs: &[f64],
    mass_per_particle: f64,
    box_size: f64,
    grid_size: usize,
    sigma_crit: f64,
    projection_axis: usize,
) -> Vec<f64> {
    let n_cells = grid_size * grid_size;
    let mut sigma_map = vec![0.0; n_cells];
    let cell_size = box_size / grid_size as f64;
    let cell_area = cell_size * cell_size;

    // Project particles onto 2D grid
    let axes = match projection_axis {
        0 => (1, 2),  // project x, use y-z
        1 => (0, 2),  // project y, use x-z
        _ => (0, 1),  // project z, use x-y
    };

    for (pos, &sign) in positions.iter().zip(signs.iter()) {
        let u = (pos[axes.0] % box_size + box_size) % box_size;
        let v = (pos[axes.1] % box_size + box_size) % box_size;

        let iu = (u / cell_size) as usize;
        let iv = (v / cell_size) as usize;

        if iu < grid_size && iv < grid_size {
            let idx = iu * grid_size + iv;
            // m+ adds positive mass, m- adds negative mass
            sigma_map[idx] += sign * mass_per_particle / cell_area;
        }
    }

    // Convert to convergence
    sigma_map.iter().map(|&sigma| sigma / sigma_crit).collect()
}

/// Compute radial κ profile around a center
///
/// # Arguments
/// * `kappa_map` - 2D convergence map
/// * `grid_size` - Map resolution
/// * `box_size` - Box size [Mpc]
/// * `center` - Center position (i, j) in grid units
/// * `n_bins` - Number of radial bins
/// * `r_max` - Maximum radius [Mpc]
///
/// # Returns
/// (r_centers, kappa_profile) in Mpc and dimensionless
pub fn radial_kappa_profile(
    kappa_map: &[f64],
    grid_size: usize,
    box_size: f64,
    center: (usize, usize),
    n_bins: usize,
    r_max: f64,
) -> (Vec<f64>, Vec<f64>) {
    let cell_size = box_size / grid_size as f64;
    let dr = r_max / n_bins as f64;

    let mut sum = vec![0.0; n_bins];
    let mut count = vec![0usize; n_bins];

    for i in 0..grid_size {
        for j in 0..grid_size {
            let di = i as f64 - center.0 as f64;
            let dj = j as f64 - center.1 as f64;
            let r = (di * di + dj * dj).sqrt() * cell_size;

            if r < r_max {
                let bin = (r / dr) as usize;
                if bin < n_bins {
                    let idx = i * grid_size + j;
                    sum[bin] += kappa_map[idx];
                    count[bin] += 1;
                }
            }
        }
    }

    let r_centers: Vec<f64> = (0..n_bins)
        .map(|i| (i as f64 + 0.5) * dr)
        .collect();

    let profile: Vec<f64> = (0..n_bins)
        .map(|i| if count[i] > 0 { sum[i] / count[i] as f64 } else { 0.0 })
        .collect();

    (r_centers, profile)
}

/// Find sign change in κ profile (Janus signature)
///
/// For Janus halos: κ > 0 in center (m+ dominated), κ < 0 in outer shell (m-)
///
/// # Arguments
/// * `r` - Radial positions [Mpc]
/// * `kappa` - κ values
///
/// # Returns
/// Option<(r_transition, kappa_inner, kappa_outer)>
pub fn find_kappa_sign_change(r: &[f64], kappa: &[f64]) -> Option<(f64, f64, f64)> {
    if r.len() < 3 || kappa.len() < 3 {
        return None;
    }

    // Find first sign change from positive to negative
    for i in 1..kappa.len() {
        if kappa[i-1] > 0.0 && kappa[i] < 0.0 {
            // Linear interpolation for transition radius
            let r_trans = r[i-1] + (r[i] - r[i-1]) * kappa[i-1] / (kappa[i-1] - kappa[i]);

            // Average κ in inner and outer regions
            let kappa_inner = kappa[..i].iter().sum::<f64>() / i as f64;
            let kappa_outer = kappa[i..].iter().sum::<f64>() / (kappa.len() - i) as f64;

            return Some((r_trans, kappa_inner, kappa_outer));
        }
    }

    None
}

/// Check if κ profile is detectable by Euclid
///
/// Euclid sensitivity: σ_κ ≈ 0.01 per 10 arcmin² pixel
/// For detection: |κ| > 3σ_κ ≈ 0.03
///
/// # Arguments
/// * `kappa_outer` - κ value in outer region
///
/// # Returns
/// true if detectable
pub fn is_euclid_detectable(kappa_outer: f64) -> bool {
    const KAPPA_THRESHOLD: f64 = 0.03;  // 3σ detection
    kappa_outer.abs() > KAPPA_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_crit_order_of_magnitude() {
        // Typical lensing geometry: D_L ~ 500 Mpc, D_S ~ 1000 Mpc
        let sc = sigma_crit(500.0, 1000.0, 600.0);

        // Σ_crit should be ~10^15 M_sun/Mpc² for cosmological lensing
        assert!(sc > 1e14 && sc < 1e18,
            "Σ_crit = {:.2e} M_sun/Mpc² out of range", sc);
    }

    #[test]
    fn test_sigma_nfw_central() {
        let r_s = 1.0;  // Mpc
        let rho_s = 1e7;  // M_sun/Mpc³

        // Central Σ should be ~2 ρ_s r_s
        let sigma_0 = sigma_nfw(0.0, r_s, rho_s);
        let expected = 2.0 * rho_s * r_s;

        assert!((sigma_0 - expected).abs() / expected < 0.1,
            "Σ(0) = {:.2e}, expected {:.2e}", sigma_0, expected);
    }

    #[test]
    fn test_sigma_nfw_decreases() {
        let r_s = 1.0;
        let rho_s = 1e7;

        let sigma_01 = sigma_nfw(0.1, r_s, rho_s);
        let sigma_1 = sigma_nfw(1.0, r_s, rho_s);
        let sigma_10 = sigma_nfw(10.0, r_s, rho_s);

        assert!(sigma_01 > sigma_1,
            "Σ should decrease: Σ(0.1)={:.2e} > Σ(1)={:.2e}", sigma_01, sigma_1);
        assert!(sigma_1 > sigma_10,
            "Σ should decrease: Σ(1)={:.2e} > Σ(10)={:.2e}", sigma_1, sigma_10);
    }

    #[test]
    fn test_kappa_positive_for_positive_mass() {
        let r = 0.5;
        let r_s = 1.0;
        let rho_s = 1e8;
        let sc = 1e15;

        let kappa = kappa_nfw(r, r_s, rho_s, sc);

        assert!(kappa > 0.0, "κ should be positive for NFW: κ = {}", kappa);
    }

    #[test]
    fn test_kappa_map_mass_conservation() {
        let n = 1000;
        let box_size = 100.0;
        let grid_size = 32;
        let mass = 1e10;
        let sc = 1e15;

        // Uniform distribution
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                [t * box_size, (t * 2.0 % 1.0) * box_size, (t * 3.0 % 1.0) * box_size]
            })
            .collect();
        let signs = vec![1.0; n];

        let kappa_map = compute_kappa_map(
            &positions, &signs, mass, box_size, grid_size, sc, 2
        );

        // Total κ × Σ_crit × area = total mass
        let total_sigma: f64 = kappa_map.iter().sum::<f64>() * sc;
        let cell_area = (box_size / grid_size as f64).powi(2);
        let total_mass = total_sigma * cell_area * grid_size as f64 * grid_size as f64;
        let expected_mass = n as f64 * mass;

        let ratio = total_mass / expected_mass;
        assert!(ratio > 0.9 && ratio < 1.1,
            "Mass conservation: ratio = {:.3}", ratio);
    }

    #[test]
    fn test_kappa_map_negative_mass() {
        let n = 100;
        let box_size = 100.0;
        let grid_size = 16;
        let mass = 1e10;
        let sc = 1e15;

        // All negative mass
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| [50.0, 50.0, i as f64 % box_size])
            .collect();
        let signs = vec![-1.0; n];

        let kappa_map = compute_kappa_map(
            &positions, &signs, mass, box_size, grid_size, sc, 2
        );

        // Central pixel should have negative κ
        let center_idx = (grid_size / 2) * grid_size + grid_size / 2;
        assert!(kappa_map[center_idx] < 0.0,
            "κ should be negative for m-: κ = {}", kappa_map[center_idx]);
    }

    #[test]
    fn test_find_sign_change() {
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kappa = vec![0.1, 0.05, -0.02, -0.05, -0.03];

        let result = find_kappa_sign_change(&r, &kappa);
        assert!(result.is_some(), "Should find sign change");

        let (r_trans, k_in, k_out) = result.unwrap();
        assert!(r_trans > 2.0 && r_trans < 3.0,
            "Transition at r = {:.2}", r_trans);
        assert!(k_in > 0.0, "κ_inner should be positive");
        assert!(k_out < 0.0, "κ_outer should be negative");
    }

    #[test]
    fn test_euclid_detection_threshold() {
        assert!(!is_euclid_detectable(0.01), "0.01 should not be detectable");
        assert!(!is_euclid_detectable(-0.02), "-0.02 should not be detectable");
        assert!(is_euclid_detectable(-0.05), "-0.05 should be detectable");
        assert!(is_euclid_detectable(0.1), "0.1 should be detectable");
    }
}
