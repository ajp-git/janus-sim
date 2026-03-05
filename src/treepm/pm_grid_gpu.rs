//! GPU-accelerated PM Grid using cuFFT
//!
//! Complete implementation with:
//! - Forward FFT (D2Z): rho -> rho_k
//! - Green's function with k_min filter
//! - Inverse FFT (Z2D): phi_k -> phi
//! - CIC force interpolation

#[cfg(feature = "cufft")]
use super::cufft_ffi::CuFFTPoisson;

use std::f64::consts::PI;

/// GPU-accelerated PM Grid with k_min filter support
#[cfg(feature = "cufft")]
pub struct PmGridGpu {
    pub grid_size: usize,
    pub box_size: f64,
    pub cell_size: f64,

    // cuFFT Poisson solver (separate for + and -)
    solver_plus: CuFFTPoisson,
    solver_minus: CuFFTPoisson,

    // Host buffers for density and potential
    pub rho_plus: Vec<f64>,
    pub rho_minus: Vec<f64>,
    pub phi_plus: Vec<f64>,
    pub phi_minus: Vec<f64>,

    // k-space filter: modes with |k_idx| < k_min are zeroed
    k_min: usize,
}

#[cfg(feature = "cufft")]
impl PmGridGpu {
    /// Create new GPU PM grid
    /// k_min: minimum k index to keep (use 3 to filter k=0,1,2)
    pub fn new(grid_size: usize, box_size: f64, k_min: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let n3 = grid_size * grid_size * grid_size;

        // Create two cuFFT solvers (one for each density field)
        let solver_plus = CuFFTPoisson::new(grid_size, box_size)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>)?;
        let solver_minus = CuFFTPoisson::new(grid_size, box_size)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>)?;

        Ok(Self {
            grid_size,
            box_size,
            cell_size: box_size / grid_size as f64,
            solver_plus,
            solver_minus,
            rho_plus: vec![0.0; n3],
            rho_minus: vec![0.0; n3],
            phi_plus: vec![0.0; n3],
            phi_minus: vec![0.0; n3],
            k_min,
        })
    }

    /// Clear density grids
    pub fn clear(&mut self) {
        self.rho_plus.fill(0.0);
        self.rho_minus.fill(0.0);
    }

    /// CIC mass assignment for a batch of particles
    pub fn assign_mass(&mut self, positions: &[f64], signs: &[i8], mass: f64) {
        let n = self.grid_size;
        let half = self.box_size / 2.0;
        let gs = n as f64;
        let n_particles = signs.len();

        self.clear();

        for i in 0..n_particles {
            let x = positions[i * 3 + 0];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let sign = signs[i];

            let gx = ((x + half) / self.box_size * gs).rem_euclid(gs);
            let gy = ((y + half) / self.box_size * gs).rem_euclid(gs);
            let gz = ((z + half) / self.box_size * gs).rem_euclid(gs);

            let ix = gx.floor() as usize;
            let iy = gy.floor() as usize;
            let iz = gz.floor() as usize;

            let fx = gx - ix as f64;
            let fy = gy - iy as f64;
            let fz = gz - iz as f64;

            let wx = [1.0 - fx, fx];
            let wy = [1.0 - fy, fy];
            let wz = [1.0 - fz, fz];

            let grid = if sign > 0 { &mut self.rho_plus } else { &mut self.rho_minus };

            for di in 0..2 {
                for dj in 0..2 {
                    for dk in 0..2 {
                        let ci = (ix + di) % n;
                        let cj = (iy + dj) % n;
                        let ck = (iz + dk) % n;
                        let idx = ci + n * (cj + n * ck);
                        grid[idx] += mass * wx[di] * wy[dj] * wz[dk];
                    }
                }
            }
        }
    }

    /// Solve Poisson equation using GPU FFT with k_min filter
    pub fn solve_poisson(&mut self, g_constant: f64) -> Result<(), Box<dyn std::error::Error>> {
        // Solve for positive density
        self.phi_plus = self.solver_plus.solve_filtered(&self.rho_plus, g_constant, 0.0, self.k_min)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>)?;

        // Solve for negative density
        self.phi_minus = self.solver_minus.solve_filtered(&self.rho_minus, g_constant, 0.0, self.k_min)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>)?;

        Ok(())
    }

    /// Interpolate force at particle position using CIC
    /// Returns (Fx, Fy, Fz) for Janus interaction rules
    pub fn interpolate_force(&self, x: f64, y: f64, z: f64, sign: i8) -> (f64, f64, f64) {
        let n = self.grid_size;
        let half = self.box_size / 2.0;
        let gs = n as f64;
        let h = self.cell_size;

        let gx = ((x + half) / self.box_size * gs).rem_euclid(gs);
        let gy = ((y + half) / self.box_size * gs).rem_euclid(gs);
        let gz = ((z + half) / self.box_size * gs).rem_euclid(gs);

        let ix = gx.floor() as usize;
        let iy = gy.floor() as usize;
        let iz = gz.floor() as usize;

        let fx = gx - ix as f64;
        let fy = gy - iy as f64;
        let fz = gz - iz as f64;

        let wx = [1.0 - fx, fx];
        let wy = [1.0 - fy, fy];
        let wz = [1.0 - fz, fz];

        // Janus force rule:
        // Particle +: F = -∇φ_plus + ∇φ_minus (attracted by +, repelled by -)
        // Particle -: F = -∇φ_minus + ∇φ_plus (attracted by -, repelled by +)
        let (phi_attract, phi_repel) = if sign > 0 {
            (&self.phi_plus, &self.phi_minus)
        } else {
            (&self.phi_minus, &self.phi_plus)
        };

        let mut force = (0.0f64, 0.0f64, 0.0f64);

        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let ci = (ix + di) % n;
                    let cj = (iy + dj) % n;
                    let ck = (iz + dk) % n;

                    let weight = wx[di] * wy[dj] * wz[dk];

                    // Neighboring cells for gradient
                    let ci_p = (ci + 1) % n;
                    let ci_m = (ci + n - 1) % n;
                    let cj_p = (cj + 1) % n;
                    let cj_m = (cj + n - 1) % n;
                    let ck_p = (ck + 1) % n;
                    let ck_m = (ck + n - 1) % n;

                    // Central difference gradient for attractive potential
                    let dphi_attract_dx = (phi_attract[ci_p + n * (cj + n * ck)]
                        - phi_attract[ci_m + n * (cj + n * ck)]) / (2.0 * h);
                    let dphi_attract_dy = (phi_attract[ci + n * (cj_p + n * ck)]
                        - phi_attract[ci + n * (cj_m + n * ck)]) / (2.0 * h);
                    let dphi_attract_dz = (phi_attract[ci + n * (cj + n * ck_p)]
                        - phi_attract[ci + n * (cj + n * ck_m)]) / (2.0 * h);

                    // Central difference gradient for repulsive potential
                    let dphi_repel_dx = (phi_repel[ci_p + n * (cj + n * ck)]
                        - phi_repel[ci_m + n * (cj + n * ck)]) / (2.0 * h);
                    let dphi_repel_dy = (phi_repel[ci + n * (cj_p + n * ck)]
                        - phi_repel[ci + n * (cj_m + n * ck)]) / (2.0 * h);
                    let dphi_repel_dz = (phi_repel[ci + n * (cj + n * ck_p)]
                        - phi_repel[ci + n * (cj + n * ck_m)]) / (2.0 * h);

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
        4 * n3 * std::mem::size_of::<f64>() + 2 * self.solver_plus.memory_bytes()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fallback CPU implementation
// ═══════════════════════════════════════════════════════════════════════════

/// CPU PM Grid (uses pm_grid.rs implementation)
pub struct PmGridCpu {
    pub inner: super::pm_grid::PmGrid,
    k_min: usize,
}

impl PmGridCpu {
    pub fn new(grid_size: usize, box_size: f64, k_min: usize) -> Self {
        Self {
            inner: super::pm_grid::PmGrid::new(grid_size, box_size),
            k_min,
        }
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn assign_mass(&mut self, positions: &[f64], signs: &[i8], mass: f64) {
        let n = signs.len();
        for i in 0..n {
            self.inner.assign_mass(
                positions[i * 3 + 0],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
                mass,
                signs[i],
            );
        }
    }

    pub fn solve_poisson(&mut self, g_constant: f64) {
        self.inner.solve_poisson_with_k_filter(g_constant, self.k_min);
    }

    pub fn interpolate_force(&self, x: f64, y: f64, z: f64, sign: i8) -> (f64, f64, f64) {
        self.inner.interpolate_force(x, y, z, sign)
    }

    pub fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
}
