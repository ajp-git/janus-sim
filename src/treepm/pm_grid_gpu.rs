//! GPU-accelerated PM Grid using cuFFT
//!
//! Uses cudarc + cuFFT for GPU-accelerated Poisson solver.
//! Falls back to CPU if CUDA not available.

#[cfg(feature = "cuda")]
use cudarc::cufft::{CudaComplex64, CudaFFTPlan, CudaFFTDirection};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchConfig, LaunchAsync};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

/// GPU-accelerated PM Grid
#[cfg(feature = "cuda")]
pub struct PmGridGpu {
    pub grid_size: usize,
    pub box_size: f64,
    pub cell_size: f64,

    // GPU device
    device: Arc<CudaDevice>,

    // GPU buffers for density (f64)
    rho_plus_d: CudaSlice<f64>,
    rho_minus_d: CudaSlice<f64>,

    // GPU buffers for complex FFT
    rho_plus_k: CudaSlice<CudaComplex64>,
    rho_minus_k: CudaSlice<CudaComplex64>,

    // GPU buffers for potential
    phi_plus_d: CudaSlice<f64>,
    phi_minus_d: CudaSlice<f64>,

    // FFT plans
    fft_plan: CudaFFTPlan,
}

#[cfg(feature = "cuda")]
impl PmGridGpu {
    /// Create new GPU PM grid
    pub fn new(grid_size: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;
        let n3 = grid_size * grid_size * grid_size;
        let n3_complex = grid_size * grid_size * (grid_size / 2 + 1);  // R2C output size

        // Allocate GPU buffers
        let rho_plus_d = device.alloc_zeros::<f64>(n3)?;
        let rho_minus_d = device.alloc_zeros::<f64>(n3)?;
        let rho_plus_k = device.alloc_zeros::<CudaComplex64>(n3_complex)?;
        let rho_minus_k = device.alloc_zeros::<CudaComplex64>(n3_complex)?;
        let phi_plus_d = device.alloc_zeros::<f64>(n3)?;
        let phi_minus_d = device.alloc_zeros::<f64>(n3)?;

        // Create 3D R2C FFT plan
        let fft_plan = CudaFFTPlan::new_3d(
            &device,
            grid_size as i32,
            grid_size as i32,
            grid_size as i32,
            cudarc::cufft::CudaFFTType::D2Z,  // Double precision Real to Complex
        )?;

        Ok(Self {
            grid_size,
            box_size,
            cell_size: box_size / grid_size as f64,
            device,
            rho_plus_d,
            rho_minus_d,
            rho_plus_k,
            rho_minus_k,
            phi_plus_d,
            phi_minus_d,
            fft_plan,
        })
    }

    /// Clear density grids
    pub fn clear(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n3 = self.grid_size * self.grid_size * self.grid_size;
        let zeros = vec![0.0f64; n3];
        self.device.htod_sync_copy_into(&zeros, &mut self.rho_plus_d)?;
        self.device.htod_sync_copy_into(&zeros, &mut self.rho_minus_d)?;
        Ok(())
    }

    /// Assign mass to grid (CPU side, then upload)
    pub fn assign_mass_batch(&mut self, particles: &[(f64, f64, f64, f64, i8)]) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.grid_size;
        let n3 = n * n * n;
        let half = self.box_size / 2.0;
        let gs = n as f64;

        // CPU-side CIC assignment
        let mut rho_plus = vec![0.0f64; n3];
        let mut rho_minus = vec![0.0f64; n3];

        for &(x, y, z, mass, sign) in particles {
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

            let grid = if sign > 0 { &mut rho_plus } else { &mut rho_minus };

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

        // Upload to GPU
        self.device.htod_sync_copy_into(&rho_plus, &mut self.rho_plus_d)?;
        self.device.htod_sync_copy_into(&rho_minus, &mut self.rho_minus_d)?;

        Ok(())
    }

    /// Solve Poisson equation on GPU with cuFFT
    pub fn solve_poisson_gpu(&mut self, g_constant: f64, r_s: Option<f64>) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.grid_size;

        // Forward FFT: rho -> rho_k
        self.fft_plan.exec_d2z(&self.rho_plus_d, &mut self.rho_plus_k)?;
        self.fft_plan.exec_d2z(&self.rho_minus_d, &mut self.rho_minus_k)?;

        // Apply Green's function in k-space (on CPU for now, GPU kernel TODO)
        let n3_complex = n * n * (n / 2 + 1);
        let mut rho_plus_k_host = vec![CudaComplex64 { x: 0.0, y: 0.0 }; n3_complex];
        let mut rho_minus_k_host = vec![CudaComplex64 { x: 0.0, y: 0.0 }; n3_complex];

        self.device.dtoh_sync_copy_into(&self.rho_plus_k, &mut rho_plus_k_host)?;
        self.device.dtoh_sync_copy_into(&self.rho_minus_k, &mut rho_minus_k_host)?;

        // Apply Green's function
        let dk = 2.0 * PI / self.box_size;
        let r_s_sq = r_s.map(|r| r * r);

        for kz in 0..n {
            for ky in 0..n {
                for kx in 0..(n / 2 + 1) {
                    let idx = kx + (n / 2 + 1) * (ky + n * kz);

                    let kx_val = if kx <= n / 2 { kx as f64 } else { (kx as i64 - n as i64) as f64 } * dk;
                    let ky_val = if ky <= n / 2 { ky as f64 } else { (ky as i64 - n as i64) as f64 } * dk;
                    let kz_val = if kz <= n / 2 { kz as f64 } else { (kz as i64 - n as i64) as f64 } * dk;

                    let k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                    if k2 < 1e-20 {
                        rho_plus_k_host[idx] = CudaComplex64 { x: 0.0, y: 0.0 };
                        rho_minus_k_host[idx] = CudaComplex64 { x: 0.0, y: 0.0 };
                        continue;
                    }

                    let mut green = -4.0 * PI * g_constant / k2;

                    // Gaussian splitting
                    if let Some(rs2) = r_s_sq {
                        green *= (-k2 * rs2).exp();
                    }

                    rho_plus_k_host[idx].x *= green;
                    rho_plus_k_host[idx].y *= green;
                    rho_minus_k_host[idx].x *= green;
                    rho_minus_k_host[idx].y *= green;
                }
            }
        }

        // Upload back to GPU
        self.device.htod_sync_copy_into(&rho_plus_k_host, &mut self.rho_plus_k)?;
        self.device.htod_sync_copy_into(&rho_minus_k_host, &mut self.rho_minus_k)?;

        // Inverse FFT: rho_k -> phi
        // Note: cuFFT Z2D requires separate plan, for now use full complex
        // TODO: proper inverse FFT

        Ok(())
    }
}

/// Fallback CPU implementation (same as pm_grid.rs)
pub struct PmGridCpu {
    pub inner: super::pm_grid::PmGrid,
}

impl PmGridCpu {
    pub fn new(grid_size: usize, box_size: f64) -> Self {
        Self {
            inner: super::pm_grid::PmGrid::new(grid_size, box_size),
        }
    }
}
