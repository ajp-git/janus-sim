//! Poisson solver using FFT
//!
//! Pipeline: ρ(x) → FFT → ρ̂(k) → ×G(k) → φ̂(k) → IFFT → φ(x) → -∇φ = g(x)
//!
//! Uses spectral Green's function with periodic boundary conditions.

use crate::cufft::Cufft3dC2C;
use crate::cufft_ffi::CufftComplex;
use std::f32::consts::PI;

/// Precomputed Green's function in Fourier space
pub struct GreenFunction {
    /// G(k) values for each grid point (complex, but imaginary part is 0)
    pub g_k: Vec<CufftComplex>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f32,
}

impl GreenFunction {
    /// Create Green's function for 3D Poisson equation with PBC
    ///
    /// Poisson equation: ∇²φ = 4πGρ
    /// In Fourier space: -k²φ̂ = 4πGρ̂
    /// So: φ̂ = -4πG/k² × ρ̂
    ///
    /// With G=1 (code units) and spectral softening:
    /// G(k) = -4π / (k² + k_s²)
    /// where k_s = 2π / (2*dx) = π/dx (softening at 2-cell scale)
    ///
    /// Uses discrete wavenumbers: k_i = 2π * (i < N/2 ? i : i-N) / (N * dx)
    pub fn new(nx: usize, ny: usize, nz: usize, box_size: f32) -> Self {
        let dx = box_size / nx as f32;
        let mut g_k = vec![CufftComplex::default(); nx * ny * nz];

        let twopi_over_lx = 2.0 * PI / box_size;
        let twopi_over_ly = 2.0 * PI / box_size;
        let twopi_over_lz = 2.0 * PI / box_size;

        // 4πG factor (G=1 in code units)
        let four_pi_g = 4.0 * PI;

        // Spectral softening: k_s = π / (8*dx)
        // Softening at 8-cell scale - prevents divergence
        // without killing large-scale Janus segregation forces
        let k_s = PI / (8.0 * dx);
        let k_s2 = k_s * k_s;

        for ix in 0..nx {
            // Discrete wavenumber (handles negative frequencies)
            let kx = if ix < nx / 2 { ix as f32 } else { (ix as i32 - nx as i32) as f32 };
            let kx = kx * twopi_over_lx;

            for iy in 0..ny {
                let ky = if iy < ny / 2 { iy as f32 } else { (iy as i32 - ny as i32) as f32 };
                let ky = ky * twopi_over_ly;

                for iz in 0..nz {
                    let kz = if iz < nz / 2 { iz as f32 } else { (iz as i32 - nz as i32) as f32 };
                    let kz = kz * twopi_over_lz;

                    let k2 = kx * kx + ky * ky + kz * kz;

                    let idx = ix * ny * nz + iy * nz + iz;

                    // G(k) = -4π / (k² + k_s²) with spectral softening
                    // k=0 still gives 0 (mean density doesn't contribute)
                    if k2 > 1e-10 {
                        g_k[idx] = CufftComplex::new(-four_pi_g / (k2 + k_s2), 0.0);
                    } else {
                        g_k[idx] = CufftComplex::new(0.0, 0.0);
                    }
                }
            }
        }

        Self { g_k, nx, ny, nz, dx }
    }

    /// Apply Green's function: φ̂(k) = G(k) × ρ̂(k)
    pub fn apply(&self, rho_k: &mut [CufftComplex]) {
        assert_eq!(rho_k.len(), self.g_k.len());

        for (rho, g) in rho_k.iter_mut().zip(self.g_k.iter()) {
            // Complex multiplication: (a + bi) × (c + 0i) = ac + bci
            rho.x *= g.x;
            rho.y *= g.x;
        }
    }
}

/// Compute gradient of potential using spectral differentiation
///
/// ∂φ/∂x in Fourier space: i * kx * φ̂(k)
///
/// Returns (gx, gy, gz) force components
pub fn spectral_gradient(
    phi_k: &[CufftComplex],
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f32,
) -> (Vec<CufftComplex>, Vec<CufftComplex>, Vec<CufftComplex>) {
    let n = nx * ny * nz;
    let mut gx_k = vec![CufftComplex::default(); n];
    let mut gy_k = vec![CufftComplex::default(); n];
    let mut gz_k = vec![CufftComplex::default(); n];

    let twopi_over_lx = 2.0 * PI / box_size;
    let twopi_over_ly = 2.0 * PI / box_size;
    let twopi_over_lz = 2.0 * PI / box_size;

    for ix in 0..nx {
        let kx = if ix < nx / 2 { ix as f32 } else { (ix as i32 - nx as i32) as f32 };
        let kx = kx * twopi_over_lx;

        for iy in 0..ny {
            let ky = if iy < ny / 2 { iy as f32 } else { (iy as i32 - ny as i32) as f32 };
            let ky = ky * twopi_over_ly;

            for iz in 0..nz {
                let kz = if iz < nz / 2 { iz as f32 } else { (iz as i32 - nz as i32) as f32 };
                let kz = kz * twopi_over_lz;

                let idx = ix * ny * nz + iy * nz + iz;
                let phi = phi_k[idx];

                // Gradient in Fourier space: i * k * φ̂
                // (a + bi) * i * k = -b*k + a*k*i
                // Force = -∇φ, so we use -i * k * φ̂ = b*k - a*k*i
                gx_k[idx] = CufftComplex::new(phi.y * kx, -phi.x * kx);
                gy_k[idx] = CufftComplex::new(phi.y * ky, -phi.x * ky);
                gz_k[idx] = CufftComplex::new(phi.y * kz, -phi.x * kz);
            }
        }
    }

    (gx_k, gy_k, gz_k)
}

/// Full PM solver: density → forces
pub struct PoissonSolver {
    pub fft: Cufft3dC2C,
    pub green: GreenFunction,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub box_size: f32,
}

impl PoissonSolver {
    pub fn new(nx: usize, ny: usize, nz: usize, box_size: f32) -> Result<Self, String> {
        let fft = Cufft3dC2C::new(nx, ny, nz)?;
        let green = GreenFunction::new(nx, ny, nz, box_size);

        Ok(Self { fft, green, nx, ny, nz, box_size })
    }

    /// Solve Poisson equation and compute forces
    ///
    /// Input: density grid ρ(x) as f32
    /// Output: force components (gx, gy, gz) as f32 grids
    pub fn solve(&self, rho: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n = self.nx * self.ny * self.nz;
        assert_eq!(rho.len(), n);

        // Convert to complex for FFT
        let mut rho_k: Vec<CufftComplex> = rho.iter()
            .map(|&r| CufftComplex::new(r, 0.0))
            .collect();

        // Forward FFT: ρ(x) → ρ̂(k)
        self.fft.forward(&mut rho_k)?;

        // Apply Green's function: φ̂(k) = G(k) × ρ̂(k)
        self.green.apply(&mut rho_k);

        // Compute gradient in Fourier space: g = -∇φ
        let (mut gx_k, mut gy_k, mut gz_k) = spectral_gradient(
            &rho_k,
            self.nx, self.ny, self.nz,
            self.box_size,
        );

        // Inverse FFT for each component
        self.fft.inverse(&mut gx_k)?;
        self.fft.inverse(&mut gy_k)?;
        self.fft.inverse(&mut gz_k)?;

        // Normalize and extract real parts
        let norm = 1.0 / n as f32;
        let gx: Vec<f32> = gx_k.iter().map(|c| c.x * norm).collect();
        let gy: Vec<f32> = gy_k.iter().map(|c| c.x * norm).collect();
        let gz: Vec<f32> = gz_k.iter().map(|c| c.x * norm).collect();

        Ok((gx, gy, gz))
    }
}

/// Interpolate scalar field at particle position using CIC
pub fn interpolate_scalar(
    pos: (f64, f64, f64),
    field: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f64,
) -> f32 {
    let dx = box_size / nx as f64;
    let dy = box_size / ny as f64;
    let dz = box_size / nz as f64;

    // Position in grid units
    let gx = pos.0 / dx;
    let gy = pos.1 / dy;
    let gz = pos.2 / dz;

    // Integer cell indices
    let ix0 = gx.floor() as i32;
    let iy0 = gy.floor() as i32;
    let iz0 = gz.floor() as i32;

    // Fractional position
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

    let mut value = 0.0_f32;

    // Interpolate from 8 neighboring cells
    for (dix, wx) in [(0, wx0), (1, wx1)] {
        for (diy, wy) in [(0, wy0), (1, wy1)] {
            for (diz, wz) in [(0, wz0), (1, wz1)] {
                let ix = ((ix0 + dix).rem_euclid(nx as i32)) as usize;
                let iy = ((iy0 + diy).rem_euclid(ny as i32)) as usize;
                let iz = ((iz0 + diz).rem_euclid(nz as i32)) as usize;

                let idx = ix * ny * nz + iy * nz + iz;
                let weight = wx * wy * wz;

                value += field[idx] * weight;
            }
        }
    }

    value
}

/// Interpolate force at particle position using CIC (reverse of deposit)
pub fn interpolate_force(
    pos: (f64, f64, f64),
    gx: &[f32],
    gy: &[f32],
    gz: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f64,
) -> (f32, f32, f32) {
    let dx = box_size / nx as f64;
    let dy = box_size / ny as f64;
    let dz = box_size / nz as f64;

    // Position in grid units
    let gx_pos = pos.0 / dx;
    let gy_pos = pos.1 / dy;
    let gz_pos = pos.2 / dz;

    // Integer cell indices
    let ix0 = gx_pos.floor() as i32;
    let iy0 = gy_pos.floor() as i32;
    let iz0 = gz_pos.floor() as i32;

    // Fractional position
    let fx = (gx_pos - ix0 as f64) as f32;
    let fy = (gy_pos - iy0 as f64) as f32;
    let fz = (gz_pos - iz0 as f64) as f32;

    // CIC weights
    let wx0 = 1.0 - fx;
    let wx1 = fx;
    let wy0 = 1.0 - fy;
    let wy1 = fy;
    let wz0 = 1.0 - fz;
    let wz1 = fz;

    let mut force_x = 0.0_f32;
    let mut force_y = 0.0_f32;
    let mut force_z = 0.0_f32;

    // Interpolate from 8 neighboring cells
    for (dix, wx) in [(0, wx0), (1, wx1)] {
        for (diy, wy) in [(0, wy0), (1, wy1)] {
            for (diz, wz) in [(0, wz0), (1, wz1)] {
                let ix = ((ix0 + dix).rem_euclid(nx as i32)) as usize;
                let iy = ((iy0 + diy).rem_euclid(ny as i32)) as usize;
                let iz = ((iz0 + diz).rem_euclid(nz as i32)) as usize;

                let idx = ix * ny * nz + iy * nz + iz;
                let weight = wx * wy * wz;

                force_x += gx[idx] * weight;
                force_y += gy[idx] * weight;
                force_z += gz[idx] * weight;
            }
        }
    }

    (force_x, force_y, force_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_function_dc_zero() {
        let green = GreenFunction::new(8, 8, 8, 1.0);
        // DC component (k=0) should be zero
        assert!(green.g_k[0].x.abs() < 1e-10);
        assert!(green.g_k[0].y.abs() < 1e-10);
    }

    #[test]
    fn test_green_function_symmetry() {
        let green = GreenFunction::new(8, 8, 8, 1.0);
        // G(k) should be real and negative for k ≠ 0
        for (i, g) in green.g_k.iter().enumerate() {
            if i > 0 {
                assert!(g.y.abs() < 1e-10, "Imaginary part should be zero");
                assert!(g.x <= 0.0, "G(k) should be <= 0");
            }
        }
    }
}
