//! Cloud-In-Cell (CIC) mass assignment scheme
//!
//! Deposits particle masses onto a 3D grid using trilinear interpolation.
//! Each particle contributes to its 8 nearest grid cells proportionally.

/// CIC deposit for particles onto a 3D grid
///
/// # Arguments
/// * `positions` - Particle positions (x, y, z) in box units [0, box_size)
/// * `masses` - Particle masses (can be negative for Janus)
/// * `grid` - Output grid (will be zeroed then filled)
/// * `nx`, `ny`, `nz` - Grid dimensions
/// * `box_size` - Physical box size
///
/// # Returns
/// Total deposited mass (for conservation check)
pub fn cic_deposit(
    positions: &[(f64, f64, f64)],
    masses: &[f32],
    grid: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f64,
) -> f64 {
    assert_eq!(positions.len(), masses.len());
    assert_eq!(grid.len(), nx * ny * nz);

    // Zero the grid
    grid.fill(0.0);

    let dx = box_size / nx as f64;
    let dy = box_size / ny as f64;
    let dz = box_size / nz as f64;

    let mut total_mass = 0.0_f64;

    for (pos, &mass) in positions.iter().zip(masses.iter()) {
        total_mass += mass as f64;

        // Position in grid units
        let gx = pos.0 / dx;
        let gy = pos.1 / dy;
        let gz = pos.2 / dz;

        // Integer cell indices (lower-left corner of the cell containing the particle)
        let ix0 = gx.floor() as i32;
        let iy0 = gy.floor() as i32;
        let iz0 = gz.floor() as i32;

        // Fractional position within cell [0, 1)
        let fx = (gx - ix0 as f64) as f32;
        let fy = (gy - iy0 as f64) as f32;
        let fz = (gz - iz0 as f64) as f32;

        // CIC weights for the 8 corners
        let wx0 = 1.0 - fx;
        let wx1 = fx;
        let wy0 = 1.0 - fy;
        let wy1 = fy;
        let wz0 = 1.0 - fz;
        let wz1 = fz;

        // Deposit to 8 neighboring cells with periodic boundary conditions
        for (dix, wx) in [(0, wx0), (1, wx1)] {
            for (diy, wy) in [(0, wy0), (1, wy1)] {
                for (diz, wz) in [(0, wz0), (1, wz1)] {
                    let ix = ((ix0 + dix).rem_euclid(nx as i32)) as usize;
                    let iy = ((iy0 + diy).rem_euclid(ny as i32)) as usize;
                    let iz = ((iz0 + diz).rem_euclid(nz as i32)) as usize;

                    let idx = ix * ny * nz + iy * nz + iz;
                    let weight = wx * wy * wz;
                    grid[idx] += mass * weight;
                }
            }
        }
    }

    total_mass
}

/// CIC deposit for Janus simulation: separate grids for positive and negative masses
///
/// # Arguments
/// * `positions` - Particle positions
/// * `mass_signs` - +1 for positive mass, -1 for negative mass
/// * `grid_plus` - Output grid for positive masses
/// * `grid_minus` - Output grid for negative masses (stored as positive values)
/// * `nx`, `ny`, `nz` - Grid dimensions
/// * `box_size` - Physical box size
///
/// # Returns
/// (n_positive, n_negative, total_mass_plus, total_mass_minus)
pub fn cic_deposit_janus(
    positions: &[(f64, f64, f64)],
    mass_signs: &[i8],
    grid_plus: &mut [f32],
    grid_minus: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f64,
) -> (usize, usize, f64, f64) {
    assert_eq!(positions.len(), mass_signs.len());
    assert_eq!(grid_plus.len(), nx * ny * nz);
    assert_eq!(grid_minus.len(), nx * ny * nz);

    // Zero the grids
    grid_plus.fill(0.0);
    grid_minus.fill(0.0);

    let dx = box_size / nx as f64;
    let dy = box_size / ny as f64;
    let dz = box_size / nz as f64;

    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    let mut total_plus = 0.0_f64;
    let mut total_minus = 0.0_f64;

    for (pos, &sign) in positions.iter().zip(mass_signs.iter()) {
        let is_positive = sign > 0;
        if is_positive {
            n_pos += 1;
            total_plus += 1.0;
        } else {
            n_neg += 1;
            total_minus += 1.0;
        }

        // Position in grid units
        let gx = pos.0 / dx;
        let gy = pos.1 / dy;
        let gz = pos.2 / dz;

        // Integer cell indices
        let ix0 = gx.floor() as i32;
        let iy0 = gy.floor() as i32;
        let iz0 = gz.floor() as i32;

        // Fractional position within cell
        let fx = (gx - ix0 as f64) as f32;
        let fy = (gy - iy0 as f64) as f32;
        let fz = (gz - iz0 as f64) as f32;

        // CIC weights
        let wx0 = 1.0 - fx;
        let wx1 = fx;
        let wy0 = 1.0 - fy;
        let wy1 = fy;
        let wz0 = 1.0 - fz;
        let wz1 = fz;

        // Deposit to 8 neighboring cells
        for (dix, wx) in [(0, wx0), (1, wx1)] {
            for (diy, wy) in [(0, wy0), (1, wy1)] {
                for (diz, wz) in [(0, wz0), (1, wz1)] {
                    let ix = ((ix0 + dix).rem_euclid(nx as i32)) as usize;
                    let iy = ((iy0 + diy).rem_euclid(ny as i32)) as usize;
                    let iz = ((iz0 + diz).rem_euclid(nz as i32)) as usize;

                    let idx = ix * ny * nz + iy * nz + iz;
                    let weight = wx * wy * wz;
                    if is_positive {
                        grid_plus[idx] += weight;
                    } else {
                        grid_minus[idx] += weight;
                    }
                }
            }
        }
    }

    (n_pos, n_neg, total_plus, total_minus)
}

/// Compute grid statistics for validation
pub struct GridStats {
    pub sum: f64,
    pub mean: f64,
    pub variance: f64,
    pub min: f32,
    pub max: f32,
}

impl GridStats {
    pub fn compute(grid: &[f32]) -> Self {
        let n = grid.len() as f64;
        let sum: f64 = grid.iter().map(|&x| x as f64).sum();
        let mean = sum / n;

        let variance: f64 = grid.iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>() / n;

        let min = grid.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = grid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Self { sum, mean, variance, min, max }
    }

    /// Normalized variance (variance / mean²) - should be small for uniform distribution
    pub fn normalized_variance(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            0.0
        } else {
            self.variance / (self.mean * self.mean)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cic_single_particle_center() {
        // Single particle at cell center should deposit equally to 8 corners
        let positions = vec![(0.5, 0.5, 0.5)];
        let masses = vec![8.0_f32];
        let mut grid = vec![0.0_f32; 8]; // 2³ grid

        let total = cic_deposit(&positions, &masses, &mut grid, 2, 2, 2, 1.0);

        assert!((total - 8.0).abs() < 1e-6);
        // Each of 8 cells should have 1.0 (8.0 * 0.5 * 0.5 * 0.5 = 1.0)
        for &val in &grid {
            assert!((val - 1.0).abs() < 1e-6, "Expected 1.0, got {}", val);
        }
    }

    #[test]
    fn test_cic_mass_conservation() {
        // Random particles should conserve total mass
        let n = 1000;
        let positions: Vec<_> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (t * 10.0 % 1.0, (t * 17.0) % 1.0, (t * 31.0) % 1.0)
            })
            .collect();
        let masses: Vec<_> = (0..n).map(|i| (i % 10 + 1) as f32).collect();

        let expected_total: f64 = masses.iter().map(|&m| m as f64).sum();

        let mut grid = vec![0.0_f32; 64 * 64 * 64];
        let total = cic_deposit(&positions, &masses, &mut grid, 64, 64, 64, 1.0);

        let grid_sum: f64 = grid.iter().map(|&x| x as f64).sum();

        assert!((total - expected_total).abs() < 1e-6);
        assert!((grid_sum - expected_total).abs() / expected_total < 1e-6);
    }
}
