//! PM Grid — Dual-grid FFT for Janus long-range forces
//!
//! ARCHITECTURE (FIX-009):
//! - rho_plus: density grid for positive masses
//! - rho_minus: density grid for negative masses (absolute value)
//! - FFT each separately → phi_plus, phi_minus
//! - Force on +: F = -∇φ_plus + ∇φ_minus
//! - Force on -: F = -∇φ_minus + ∇φ_plus
//!
//! IMPLEMENTATION NOTE:
//! Initial version uses rustfft (CPU) for FFT operations.
//! GPU cuFFT optimization planned after architecture validation.

use rustfft::{FftPlanner, num_complex::Complex64};
use std::f64::consts::PI;

/// PM Grid state for dual-grid Janus FFT
pub struct PmGrid {
    pub grid_size: usize,
    pub box_size: f64,
    pub cell_size: f64,

    // Density grids (real, grid_size³)
    pub rho_plus: Vec<f64>,
    pub rho_minus: Vec<f64>,

    // Potential grids (real, grid_size³)
    pub phi_plus: Vec<f64>,
    pub phi_minus: Vec<f64>,

    // FFT planner (reusable)
    fft_planner: FftPlanner<f64>,
}

impl PmGrid {
    /// Create new PM grid with given size and box dimensions
    pub fn new(grid_size: usize, box_size: f64) -> Self {
        let n_cells = grid_size * grid_size * grid_size;
        let cell_size = box_size / grid_size as f64;

        Self {
            grid_size,
            box_size,
            cell_size,
            rho_plus: vec![0.0; n_cells],
            rho_minus: vec![0.0; n_cells],
            phi_plus: vec![0.0; n_cells],
            phi_minus: vec![0.0; n_cells],
            fft_planner: FftPlanner::new(),
        }
    }

    /// Clear all grids to zero
    pub fn clear(&mut self) {
        self.rho_plus.fill(0.0);
        self.rho_minus.fill(0.0);
        self.phi_plus.fill(0.0);
        self.phi_minus.fill(0.0);
    }

    /// CIC (Cloud-in-Cell) mass assignment for a single particle
    /// Distributes mass to 8 neighboring cells
    pub fn assign_mass(&mut self, x: f64, y: f64, z: f64, mass: f64, sign: i8) {
        let half = self.box_size / 2.0;
        let gs = self.grid_size as f64;

        // Convert position to grid coordinates [0, grid_size)
        let gx = ((x + half) / self.box_size * gs).rem_euclid(gs);
        let gy = ((y + half) / self.box_size * gs).rem_euclid(gs);
        let gz = ((z + half) / self.box_size * gs).rem_euclid(gs);

        // Integer cell indices
        let ix = gx.floor() as usize;
        let iy = gy.floor() as usize;
        let iz = gz.floor() as usize;

        // Fractional position within cell [0, 1)
        let fx = gx - ix as f64;
        let fy = gy - iy as f64;
        let fz = gz - iz as f64;

        // CIC weights for 8 neighboring cells
        let wx = [1.0 - fx, fx];
        let wy = [1.0 - fy, fy];
        let wz = [1.0 - fz, fz];

        // Select target grid based on sign
        let grid = if sign > 0 { &mut self.rho_plus } else { &mut self.rho_minus };

        // Distribute mass to 8 cells
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let ci = (ix + di) % self.grid_size;
                    let cj = (iy + dj) % self.grid_size;
                    let ck = (iz + dk) % self.grid_size;
                    let idx = ci + self.grid_size * (cj + self.grid_size * ck);
                    let weight = wx[di] * wy[dj] * wz[dk];
                    grid[idx] += mass.abs() * weight;  // Always positive density
                }
            }
        }
    }

    /// Solve Poisson equation: ∇²φ = 4πGρ using FFT
    /// Applies to both rho_plus → phi_plus and rho_minus → phi_minus
    ///
    /// For standard gravity, use r_s = None.
    /// For TreePM long-range only, use r_s = Some(r_cut / 3.0) for Gaussian splitting.
    pub fn solve_poisson(&mut self, g_constant: f64) {
        self.solve_poisson_with_splitting(g_constant, None);
    }

    /// Solve Poisson with optional k-space splitting for TreePM
    ///
    /// With r_s = Some(value), applies Gaussian damping:
    ///   G_pm(k) = -4πG/k² * exp(-k²r_s²)
    /// This suppresses short-range forces (high k), leaving only long-range.
    pub fn solve_poisson_with_splitting(&mut self, g_constant: f64, r_s: Option<f64>) {
        self.solve_poisson_filtered(g_constant, r_s, None);
    }

    /// Solve Poisson with k_min filter to suppress large-scale modes (dipole, etc.)
    ///
    /// k_min: minimum k mode to keep (modes with |k| < k_min are zeroed)
    /// Use k_min = 2 to remove dipole (k=1) and monopole (k=0)
    pub fn solve_poisson_with_k_filter(&mut self, g_constant: f64, k_min: usize) {
        self.solve_poisson_filtered(g_constant, None, Some(k_min));
    }

    /// Core Poisson solver with optional r_s splitting and k_min filter
    fn solve_poisson_filtered(&mut self, g_constant: f64, r_s: Option<f64>, k_min: Option<usize>) {
        let n = self.grid_size;
        let n3 = n * n * n;

        // Get FFT plans
        let fft_forward = self.fft_planner.plan_fft_forward(n);
        let fft_inverse = self.fft_planner.plan_fft_inverse(n);

        // Process both grids
        for (rho, phi) in [
            (&self.rho_plus, &mut self.phi_plus),
            (&self.rho_minus, &mut self.phi_minus),
        ] {
            // Convert to complex for FFT
            let mut data: Vec<Complex64> = rho.iter().map(|&r| Complex64::new(r, 0.0)).collect();

            // 3D FFT (apply 1D FFT along each dimension)
            // X dimension
            for jk in 0..(n * n) {
                let offset = jk * n;
                let mut slice: Vec<Complex64> = data[offset..offset + n].to_vec();
                fft_forward.process(&mut slice);
                data[offset..offset + n].copy_from_slice(&slice);
            }

            // Y dimension (strided)
            for i in 0..n {
                for k in 0..n {
                    let mut slice: Vec<Complex64> = (0..n)
                        .map(|j| data[i + n * (j + n * k)])
                        .collect();
                    fft_forward.process(&mut slice);
                    for (j, val) in slice.into_iter().enumerate() {
                        data[i + n * (j + n * k)] = val;
                    }
                }
            }

            // Z dimension (strided)
            for i in 0..n {
                for j in 0..n {
                    let mut slice: Vec<Complex64> = (0..n)
                        .map(|k| data[i + n * (j + n * k)])
                        .collect();
                    fft_forward.process(&mut slice);
                    for (k, val) in slice.into_iter().enumerate() {
                        data[i + n * (j + n * k)] = val;
                    }
                }
            }

            // Apply Green's function: G(k) = -4πG / k²
            // With optional Gaussian splitting: G_pm(k) = G(k) * exp(-k²r_s²)
            let dk = 2.0 * PI / self.box_size;
            let r_s_sq = r_s.map(|r| r * r);

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let idx = i + n * (j + n * k);

                        // Wave vector components
                        let ki = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
                        let kj = if j <= n / 2 { j as f64 } else { j as f64 - n as f64 };
                        let kk = if k <= n / 2 { k as f64 } else { k as f64 - n as f64 };

                        let k2 = (ki * ki + kj * kj + kk * kk) * dk * dk;

                        // Compute |k| in grid units (integer distance from origin)
                        let k_int = (ki.abs() as usize).max(kj.abs() as usize).max(kk.abs() as usize);

                        // Filter: zero out modes with |k| < k_min
                        let filtered = if let Some(kmin) = k_min {
                            k_int < kmin
                        } else {
                            k2 < 1e-10  // Only filter k=0 if no k_min specified
                        };

                        if filtered {
                            // Zero out this mode (monopole, dipole, etc.)
                            data[idx] = Complex64::new(0.0, 0.0);
                        } else if k2 > 1e-10 {
                            // G(k) = -4πG / k²
                            let mut green = -4.0 * PI * g_constant / k2;

                            // Optional Gaussian damping for TreePM
                            if let Some(rs2) = r_s_sq {
                                green *= (-k2 * rs2).exp();
                            }

                            data[idx] *= green;
                        } else {
                            data[idx] = Complex64::new(0.0, 0.0);
                        }
                    }
                }
            }

            // Inverse 3D FFT
            // Z dimension
            for i in 0..n {
                for j in 0..n {
                    let mut slice: Vec<Complex64> = (0..n)
                        .map(|k| data[i + n * (j + n * k)])
                        .collect();
                    fft_inverse.process(&mut slice);
                    for (k, val) in slice.into_iter().enumerate() {
                        data[i + n * (j + n * k)] = val;
                    }
                }
            }

            // Y dimension
            for i in 0..n {
                for k in 0..n {
                    let mut slice: Vec<Complex64> = (0..n)
                        .map(|j| data[i + n * (j + n * k)])
                        .collect();
                    fft_inverse.process(&mut slice);
                    for (j, val) in slice.into_iter().enumerate() {
                        data[i + n * (j + n * k)] = val;
                    }
                }
            }

            // X dimension
            for jk in 0..(n * n) {
                let offset = jk * n;
                let mut slice: Vec<Complex64> = data[offset..offset + n].to_vec();
                fft_inverse.process(&mut slice);
                data[offset..offset + n].copy_from_slice(&slice);
            }

            // Normalize and extract real part
            let norm = 1.0 / (n3 as f64);
            for (idx, val) in data.iter().enumerate() {
                phi[idx] = val.re * norm;
            }
        }
    }

    /// Interpolate potential gradient (force) at particle position
    /// Returns (Fx, Fy, Fz) using CIC interpolation
    pub fn interpolate_force(&self, x: f64, y: f64, z: f64, sign: i8) -> (f64, f64, f64) {
        let half = self.box_size / 2.0;
        let gs = self.grid_size as f64;
        let n = self.grid_size;

        // Grid coordinates
        let gx = ((x + half) / self.box_size * gs).rem_euclid(gs);
        let gy = ((y + half) / self.box_size * gs).rem_euclid(gs);
        let gz = ((z + half) / self.box_size * gs).rem_euclid(gs);

        let ix = gx.floor() as usize;
        let iy = gy.floor() as usize;
        let iz = gz.floor() as usize;

        let fx = gx - ix as f64;
        let fy = gy - iy as f64;
        let fz = gz - iz as f64;

        // Janus force rule:
        // Particle +: F = -∇φ_plus + ∇φ_minus (attracted by +, repelled by -)
        // Particle -: F = -∇φ_minus + ∇φ_plus (attracted by -, repelled by +)
        let (phi_attract, phi_repel) = if sign > 0 {
            (&self.phi_plus, &self.phi_minus)
        } else {
            (&self.phi_minus, &self.phi_plus)
        };

        // Compute gradient using finite differences
        let mut force = (0.0f64, 0.0f64, 0.0f64);
        let h = self.cell_size;

        // CIC weights
        let wx = [1.0 - fx, fx];
        let wy = [1.0 - fy, fy];
        let wz = [1.0 - fz, fz];

        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let ci = (ix + di) % n;
                    let cj = (iy + dj) % n;
                    let ck = (iz + dk) % n;

                    let weight = wx[di] * wy[dj] * wz[dk];

                    // Get neighboring cell indices for gradient
                    let ci_p = (ci + 1) % n;
                    let ci_m = (ci + n - 1) % n;
                    let cj_p = (cj + 1) % n;
                    let cj_m = (cj + n - 1) % n;
                    let ck_p = (ck + 1) % n;
                    let ck_m = (ck + n - 1) % n;

                    // Central difference gradient for attractive potential
                    let dphi_attract_dx = (phi_attract[ci_p + n * (cj + n * ck)]
                        - phi_attract[ci_m + n * (cj + n * ck)])
                        / (2.0 * h);
                    let dphi_attract_dy = (phi_attract[ci + n * (cj_p + n * ck)]
                        - phi_attract[ci + n * (cj_m + n * ck)])
                        / (2.0 * h);
                    let dphi_attract_dz = (phi_attract[ci + n * (cj + n * ck_p)]
                        - phi_attract[ci + n * (cj + n * ck_m)])
                        / (2.0 * h);

                    // Central difference gradient for repulsive potential
                    let dphi_repel_dx = (phi_repel[ci_p + n * (cj + n * ck)]
                        - phi_repel[ci_m + n * (cj + n * ck)])
                        / (2.0 * h);
                    let dphi_repel_dy = (phi_repel[ci + n * (cj_p + n * ck)]
                        - phi_repel[ci + n * (cj_m + n * ck)])
                        / (2.0 * h);
                    let dphi_repel_dz = (phi_repel[ci + n * (cj + n * ck_p)]
                        - phi_repel[ci + n * (cj + n * ck_m)])
                        / (2.0 * h);

                    // F = -∇φ_attract + ∇φ_repel
                    force.0 += weight * (-dphi_attract_dx + dphi_repel_dx);
                    force.1 += weight * (-dphi_attract_dy + dphi_repel_dy);
                    force.2 += weight * (-dphi_attract_dz + dphi_repel_dz);
                }
            }
        }

        force
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let n3 = self.grid_size * self.grid_size * self.grid_size;
        // 4 grids (rho_plus, rho_minus, phi_plus, phi_minus) × f64
        4 * n3 * std::mem::size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_splitting() {
        // Test that Gaussian splitting reduces short-range forces
        let mut pm_full = PmGrid::new(64, 100.0);
        let mut pm_split = PmGrid::new(64, 100.0);

        // Single mass at origin
        pm_full.assign_mass(0.0, 0.0, 0.0, 1.0, 1);
        pm_split.assign_mass(0.0, 0.0, 0.0, 1.0, 1);

        // Solve without splitting
        pm_full.solve_poisson(1.0);

        // Solve with Gaussian splitting (r_s = 10 corresponds to r_cut ~ 30)
        let r_s = 10.0;
        pm_split.solve_poisson_with_splitting(1.0, Some(r_s));

        // Compare forces at different distances
        println!("\n=== Gaussian Splitting Test ===");
        println!("r_s = {}", r_s);

        for r in [5.0, 10.0, 20.0, 30.0, 40.0] {
            let (fx_full, _, _) = pm_full.interpolate_force(r, 0.0, 0.0, 1);
            let (fx_split, _, _) = pm_split.interpolate_force(r, 0.0, 0.0, 1);

            let ratio = if fx_full.abs() > 1e-10 { fx_split / fx_full } else { 0.0 };
            println!("  r={:.0}: F_full={:.6}, F_split={:.6}, ratio={:.3}",
                     r, fx_full, fx_split, ratio);
        }

        // At short range, split force should be reduced
        let (fx_full_short, _, _) = pm_full.interpolate_force(5.0, 0.0, 0.0, 1);
        let (fx_split_short, _, _) = pm_split.interpolate_force(5.0, 0.0, 0.0, 1);
        assert!(fx_split_short.abs() < fx_full_short.abs() * 0.5,
                "Split force should be significantly reduced at short range");

        // At long range, forces should be similar
        let (fx_full_long, _, _) = pm_full.interpolate_force(40.0, 0.0, 0.0, 1);
        let (fx_split_long, _, _) = pm_split.interpolate_force(40.0, 0.0, 0.0, 1);
        let ratio_long = fx_split_long / fx_full_long;
        assert!(ratio_long > 0.8, "Split force should be close to full at long range: ratio={}", ratio_long);

        println!("✓ Gaussian splitting correctly reduces short-range forces");
    }

    #[test]
    fn test_pm_grid_creation() {
        let pm = PmGrid::new(64, 100.0);
        assert_eq!(pm.grid_size, 64);
        assert!((pm.cell_size - 100.0 / 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_mass_assignment() {
        let mut pm = PmGrid::new(32, 100.0);

        // Place a positive particle at center
        pm.assign_mass(0.0, 0.0, 0.0, 1.0, 1);

        // Check that mass is distributed
        let total_mass: f64 = pm.rho_plus.iter().sum();
        assert!((total_mass - 1.0).abs() < 1e-10, "Total mass should be 1.0");

        // Negative grid should be empty
        let neg_mass: f64 = pm.rho_minus.iter().sum();
        assert!(neg_mass < 1e-10, "Negative grid should be empty");
    }

    #[test]
    fn test_janus_force_signs() {
        let mut pm = PmGrid::new(32, 100.0);

        // Place positive mass at center
        pm.assign_mass(0.0, 0.0, 0.0, 1.0, 1);

        // Solve Poisson
        pm.solve_poisson(1.0);

        // Test particle at (10, 0, 0)
        let (fx_pos, _, _) = pm.interpolate_force(10.0, 0.0, 0.0, 1);  // + particle
        let (fx_neg, _, _) = pm.interpolate_force(10.0, 0.0, 0.0, -1); // - particle

        // + particle should be attracted toward center (fx < 0)
        assert!(fx_pos < 0.0, "Positive particle should be attracted to positive mass");

        // - particle should be repelled from center (fx > 0)
        assert!(fx_neg > 0.0, "Negative particle should be repelled from positive mass");
    }
}
