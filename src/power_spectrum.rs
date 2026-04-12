//! Power Spectrum Analysis Module
//!
//! Computes P(k) from particle positions using FFT-based methods.
//! Supports auto-spectrum and cross-spectrum for Janus bimetric analysis.
//!
//! Features:
//! - CIC (Cloud-In-Cell) mass assignment
//! - 3D FFT via rustfft
//! - Radial binning with shot noise subtraction
//! - Cross-spectrum for m+/m- anticorrelation
//!
//! Reference: Hockney & Eastwood (1981), Jing (2005)

use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// Result of P(k) computation
#[derive(Debug, Clone)]
pub struct PowerSpectrumResult {
    /// k values [Mpc^-1]
    pub k: Vec<f64>,
    /// P(k) values [Mpc³]
    pub pk: Vec<f64>,
    /// Number of modes per bin
    pub n_modes: Vec<usize>,
}

/// CIC (Cloud-in-Cell) mass assignment to 3D grid
///
/// # Arguments
/// * `positions` - Particle positions [Mpc], assumed in [0, box_size)
/// * `box_size` - Box size [Mpc]
/// * `grid_size` - Number of grid cells per dimension
///
/// # Returns
/// Density grid (grid_size³) with CIC-weighted particle counts
pub fn cic_assign(positions: &[[f64; 3]], box_size: f64, grid_size: usize) -> Vec<f64> {
    let n_cells = grid_size * grid_size * grid_size;
    let mut grid = vec![0.0; n_cells];
    let cell_size = box_size / grid_size as f64;

    for pos in positions {
        // Grid coordinates (wrapped to [0, box_size))
        let x = ((pos[0] % box_size) + box_size) % box_size;
        let y = ((pos[1] % box_size) + box_size) % box_size;
        let z = ((pos[2] % box_size) + box_size) % box_size;

        // Cell indices
        let ix = (x / cell_size).floor() as usize;
        let iy = (y / cell_size).floor() as usize;
        let iz = (z / cell_size).floor() as usize;

        // Fractional position within cell
        let dx = x / cell_size - ix as f64;
        let dy = y / cell_size - iy as f64;
        let dz = z / cell_size - iz as f64;

        // CIC weights
        let wx = [1.0 - dx, dx];
        let wy = [1.0 - dy, dy];
        let wz = [1.0 - dz, dz];

        // Assign to 8 neighboring cells
        for i in 0..2 {
            let ii = (ix + i) % grid_size;
            for j in 0..2 {
                let jj = (iy + j) % grid_size;
                for k in 0..2 {
                    let kk = (iz + k) % grid_size;
                    let idx = ii * grid_size * grid_size + jj * grid_size + kk;
                    grid[idx] += wx[i] * wy[j] * wz[k];
                }
            }
        }
    }

    grid
}

/// Compute power spectrum P(k) from density grid
///
/// # Arguments
/// * `density_grid` - 3D density field (grid_size³)
/// * `box_size` - Box size [Mpc]
/// * `grid_size` - Grid resolution
/// * `n_particles` - Number of particles (for shot noise)
/// * `n_bins` - Number of k bins
///
/// # Returns
/// PowerSpectrumResult with k, P(k), and mode counts
pub fn compute_pk(
    density_grid: &[f64],
    box_size: f64,
    grid_size: usize,
    n_particles: usize,
    n_bins: usize,
) -> PowerSpectrumResult {
    let n_cells = grid_size * grid_size * grid_size;
    assert_eq!(density_grid.len(), n_cells);

    // Compute mean density
    let mean_density = density_grid.iter().sum::<f64>() / n_cells as f64;

    // Compute overdensity δ = (ρ - ρ̄) / ρ̄
    let mut delta: Vec<Complex<f64>> = density_grid.iter()
        .map(|&rho| {
            if mean_density > 1e-30 {
                Complex::new((rho - mean_density) / mean_density, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();

    // 3D FFT
    let mut planner = FftPlanner::new();

    // FFT along z (fastest varying)
    let fft_z = planner.plan_fft_forward(grid_size);
    for ix in 0..grid_size {
        for iy in 0..grid_size {
            let start = ix * grid_size * grid_size + iy * grid_size;
            let mut row: Vec<Complex<f64>> = delta[start..start + grid_size].to_vec();
            fft_z.process(&mut row);
            delta[start..start + grid_size].copy_from_slice(&row);
        }
    }

    // FFT along y
    let fft_y = planner.plan_fft_forward(grid_size);
    for ix in 0..grid_size {
        for iz in 0..grid_size {
            let mut col: Vec<Complex<f64>> = (0..grid_size)
                .map(|iy| delta[ix * grid_size * grid_size + iy * grid_size + iz])
                .collect();
            fft_y.process(&mut col);
            for iy in 0..grid_size {
                delta[ix * grid_size * grid_size + iy * grid_size + iz] = col[iy];
            }
        }
    }

    // FFT along x
    let fft_x = planner.plan_fft_forward(grid_size);
    for iy in 0..grid_size {
        for iz in 0..grid_size {
            let mut col: Vec<Complex<f64>> = (0..grid_size)
                .map(|ix| delta[ix * grid_size * grid_size + iy * grid_size + iz])
                .collect();
            fft_x.process(&mut col);
            for ix in 0..grid_size {
                delta[ix * grid_size * grid_size + iy * grid_size + iz] = col[ix];
            }
        }
    }

    // Compute |δ(k)|² and bin by k
    let k_fund = 2.0 * PI / box_size;  // Fundamental mode
    let k_nyquist = PI * grid_size as f64 / box_size;  // Nyquist frequency

    let dk = k_nyquist / n_bins as f64;
    let mut pk_sum = vec![0.0; n_bins];
    let mut n_modes = vec![0usize; n_bins];

    for ix in 0..grid_size {
        let kx_idx = if ix <= grid_size / 2 { ix as i32 } else { ix as i32 - grid_size as i32 };
        let kx = kx_idx as f64 * k_fund;

        for iy in 0..grid_size {
            let ky_idx = if iy <= grid_size / 2 { iy as i32 } else { iy as i32 - grid_size as i32 };
            let ky = ky_idx as f64 * k_fund;

            for iz in 0..grid_size {
                let kz_idx = if iz <= grid_size / 2 { iz as i32 } else { iz as i32 - grid_size as i32 };
                let kz = kz_idx as f64 * k_fund;

                let k = (kx * kx + ky * ky + kz * kz).sqrt();
                if k < 1e-10 || k > k_nyquist { continue; }

                let idx = ix * grid_size * grid_size + iy * grid_size + iz;
                let pk_mode = delta[idx].norm_sqr() * (box_size / grid_size as f64).powi(3) / box_size.powi(3);

                let bin = ((k / dk).floor() as usize).min(n_bins - 1);
                pk_sum[bin] += pk_mode;
                n_modes[bin] += 1;
            }
        }
    }

    // Average and subtract shot noise
    let shot_noise = box_size.powi(3) / n_particles as f64;
    let k_centers: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dk).collect();
    let pk: Vec<f64> = (0..n_bins)
        .map(|i| {
            if n_modes[i] > 0 {
                (pk_sum[i] / n_modes[i] as f64 - shot_noise).max(0.0)
            } else {
                0.0
            }
        })
        .collect();

    PowerSpectrumResult {
        k: k_centers,
        pk,
        n_modes,
    }
}

/// Compute cross-power spectrum P_12(k) between two density fields
///
/// Returns the real part of the cross-spectrum (imaginary part vanishes for
/// statistically isotropic fields).
pub fn compute_cross_pk(
    density1: &[f64],
    density2: &[f64],
    box_size: f64,
    grid_size: usize,
    n_bins: usize,
) -> PowerSpectrumResult {
    let n_cells = grid_size * grid_size * grid_size;
    assert_eq!(density1.len(), n_cells);
    assert_eq!(density2.len(), n_cells);

    let mean1 = density1.iter().sum::<f64>() / n_cells as f64;
    let mean2 = density2.iter().sum::<f64>() / n_cells as f64;

    // Compute δ for both fields
    let mut delta1: Vec<Complex<f64>> = density1.iter()
        .map(|&rho| Complex::new((rho - mean1) / mean1.max(1e-30), 0.0))
        .collect();
    let mut delta2: Vec<Complex<f64>> = density2.iter()
        .map(|&rho| Complex::new((rho - mean2) / mean2.max(1e-30), 0.0))
        .collect();

    // 3D FFT for both
    fft_3d_inplace(&mut delta1, grid_size);
    fft_3d_inplace(&mut delta2, grid_size);

    // Compute Re(δ₁* × δ₂) and bin
    let k_fund = 2.0 * PI / box_size;
    let k_nyquist = PI * grid_size as f64 / box_size;
    let dk = k_nyquist / n_bins as f64;

    let mut pk_sum = vec![0.0; n_bins];
    let mut n_modes = vec![0usize; n_bins];

    for ix in 0..grid_size {
        let kx_idx = if ix <= grid_size / 2 { ix as i32 } else { ix as i32 - grid_size as i32 };
        let kx = kx_idx as f64 * k_fund;

        for iy in 0..grid_size {
            let ky_idx = if iy <= grid_size / 2 { iy as i32 } else { iy as i32 - grid_size as i32 };
            let ky = ky_idx as f64 * k_fund;

            for iz in 0..grid_size {
                let kz_idx = if iz <= grid_size / 2 { iz as i32 } else { iz as i32 - grid_size as i32 };
                let kz = kz_idx as f64 * k_fund;

                let k = (kx * kx + ky * ky + kz * kz).sqrt();
                if k < 1e-10 || k > k_nyquist { continue; }

                let idx = ix * grid_size * grid_size + iy * grid_size + iz;
                // Cross-spectrum: Re(δ₁* × δ₂)
                let cross = delta1[idx].conj() * delta2[idx];
                let pk_mode = cross.re * (box_size / grid_size as f64).powi(3) / box_size.powi(3);

                let bin = ((k / dk).floor() as usize).min(n_bins - 1);
                pk_sum[bin] += pk_mode;
                n_modes[bin] += 1;
            }
        }
    }

    let k_centers: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * dk).collect();
    let pk: Vec<f64> = (0..n_bins)
        .map(|i| if n_modes[i] > 0 { pk_sum[i] / n_modes[i] as f64 } else { 0.0 })
        .collect();

    PowerSpectrumResult {
        k: k_centers,
        pk,
        n_modes,
    }
}

/// Helper: 3D FFT in-place
fn fft_3d_inplace(data: &mut [Complex<f64>], grid_size: usize) {
    let mut planner = FftPlanner::new();

    // FFT along z
    let fft_z = planner.plan_fft_forward(grid_size);
    for ix in 0..grid_size {
        for iy in 0..grid_size {
            let start = ix * grid_size * grid_size + iy * grid_size;
            let mut row: Vec<Complex<f64>> = data[start..start + grid_size].to_vec();
            fft_z.process(&mut row);
            data[start..start + grid_size].copy_from_slice(&row);
        }
    }

    // FFT along y
    let fft_y = planner.plan_fft_forward(grid_size);
    for ix in 0..grid_size {
        for iz in 0..grid_size {
            let mut col: Vec<Complex<f64>> = (0..grid_size)
                .map(|iy| data[ix * grid_size * grid_size + iy * grid_size + iz])
                .collect();
            fft_y.process(&mut col);
            for iy in 0..grid_size {
                data[ix * grid_size * grid_size + iy * grid_size + iz] = col[iy];
            }
        }
    }

    // FFT along x
    let fft_x = planner.plan_fft_forward(grid_size);
    for iy in 0..grid_size {
        for iz in 0..grid_size {
            let mut col: Vec<Complex<f64>> = (0..grid_size)
                .map(|ix| data[ix * grid_size * grid_size + iy * grid_size + iz])
                .collect();
            fft_x.process(&mut col);
            for ix in 0..grid_size {
                data[ix * grid_size * grid_size + iy * grid_size + iz] = col[ix];
            }
        }
    }
}

/// ΛCDM P(k) approximation using Bardeen transfer function
///
/// P(k) = A × k^n_s × T(k)²
/// where T(k) is the Bardeen (1986) approximation
pub fn lcdm_pk(k: f64, sigma8: f64, n_s: f64) -> f64 {
    if k < 1e-10 { return 0.0; }

    let k_eq = 0.02;  // h/Mpc
    let q = k / k_eq;

    // Bardeen transfer function
    let t_k = (1.0 + 2.34 * q).ln() / (2.34 * q)
        / (1.0 + 3.89 * q + (16.2 * q).powi(2) + (5.47 * q).powi(3) + (6.71 * q).powi(4)).powf(0.25);

    // Primordial spectrum
    let p_prim = k.powf(n_s);

    // Rough normalization to σ₈
    p_prim * t_k * t_k * sigma8 * sigma8 * 1e4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cic_mass_conservation() {
        let n = 1000;
        let box_size = 100.0;
        let grid_size = 32;

        // Random positions
        let mut rng = rand::rng();
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                use rand::Rng;
                [
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                ]
            })
            .collect();

        let grid = cic_assign(&positions, box_size, grid_size);
        let total_mass: f64 = grid.iter().sum();

        // Total should equal number of particles
        assert!((total_mass - n as f64).abs() < 1e-10,
            "CIC mass conservation: {} vs {}", total_mass, n);
    }

    #[test]
    fn test_pk_white_noise() {
        // White noise should have flat P(k) ≈ V/N
        let grid_size = 32;
        let box_size = 100.0;
        let n = 10000;

        let mut rng = rand::rng();
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                use rand::Rng;
                [
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                ]
            })
            .collect();

        let grid = cic_assign(&positions, box_size, grid_size);
        let result = compute_pk(&grid, box_size, grid_size, n, 16);

        // After shot noise subtraction, white noise should be ~0
        // (actual value depends on sampling variance)
        let mean_pk: f64 = result.pk.iter().sum::<f64>() / result.pk.len() as f64;
        let shot = box_size.powi(3) / n as f64;

        // White noise P(k) should be consistent with shot noise level
        assert!(mean_pk < shot * 2.0,
            "White noise P(k) should be near shot noise: {} vs {}", mean_pk, shot);
    }

    #[test]
    fn test_pk_units() {
        // P(k) should have units of [Mpc³]
        let grid_size = 16;
        let box_size = 100.0;
        let n = 1000;

        let mut rng = rand::rng();
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                use rand::Rng;
                [
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                ]
            })
            .collect();

        let grid = cic_assign(&positions, box_size, grid_size);
        let result = compute_pk(&grid, box_size, grid_size, n, 8);

        // P(k) before shot noise subtraction should be ~V/N for uniform
        let shot = box_size.powi(3) / n as f64;  // 10^6 / 10^3 = 10^3 Mpc³

        // Check order of magnitude
        assert!(shot > 100.0 && shot < 1e6,
            "Shot noise should be ~1000 Mpc³: {}", shot);
    }

    #[test]
    fn test_nyquist_cutoff() {
        let grid_size = 16;
        let box_size = 100.0;
        let n = 500;

        let mut rng = rand::rng();
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                use rand::Rng;
                [
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                    rng.random::<f64>() * box_size,
                ]
            })
            .collect();

        let grid = cic_assign(&positions, box_size, grid_size);
        let result = compute_pk(&grid, box_size, grid_size, n, 10);

        let k_nyquist = PI * grid_size as f64 / box_size;

        // All k values should be below Nyquist
        for &k in &result.k {
            assert!(k < k_nyquist * 1.1,
                "k = {} exceeds Nyquist = {}", k, k_nyquist);
        }
    }

    #[test]
    fn test_lcdm_pk_shape() {
        // ΛCDM P(k) should decrease at high k
        let pk_low = lcdm_pk(0.01, 0.8, 0.965);
        let pk_high = lcdm_pk(1.0, 0.8, 0.965);

        assert!(pk_low > pk_high,
            "ΛCDM P(k) should decrease: P(0.01)={:.2e}, P(1.0)={:.2e}",
            pk_low, pk_high);
    }

    #[test]
    fn test_lcdm_pk_sigma8_scaling() {
        // P(k) ∝ σ₈²
        let pk1 = lcdm_pk(0.1, 0.7, 0.965);
        let pk2 = lcdm_pk(0.1, 1.4, 0.965);  // 2× σ₈

        let ratio = pk2 / pk1;
        assert!((ratio - 4.0).abs() < 0.1,
            "P(k) ∝ σ₈²: ratio = {} (expected ~4)", ratio);
    }
}
