//! GPU SPH Pressure Module
//!
//! Computes SPH density and pressure forces on GPU using CUDA kernels.
//! Newton III symmetry guaranteed by symmetric pressure formulation.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Physical constants in code units
pub const K_B_OVER_MP_CODE: f32 = 8.7e-9;  // (Mpc/Gyr)²/K
pub const MU_MOL: f32 = 0.6;               // Mean molecular weight

/// GPU SPH Pressure Calculator
pub struct GpuSphPressure {
    device: Arc<CudaDevice>,

    // GPU buffers (f32 for performance)
    pos_x: CudaSlice<f32>,
    pos_y: CudaSlice<f32>,
    pos_z: CudaSlice<f32>,
    density: CudaSlice<f32>,
    pressure: CudaSlice<f32>,
    smooth_h: CudaSlice<f32>,
    temperature: CudaSlice<f32>,
    acc_x: CudaSlice<f32>,
    acc_y: CudaSlice<f32>,
    acc_z: CudaSlice<f32>,

    // Parameters
    n: usize,
    mass: f32,
    box_size: f32,
    h_min: f32,
    h_max: f32,
    eta: f32,  // Smoothing length factor (1.2)
}

impl GpuSphPressure {
    /// Create new GPU SPH calculator
    pub fn new(
        device: Arc<CudaDevice>,
        n: usize,
        mass: f64,
        box_size: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load PTX module
        let ptx = include_str!("../cuda/sph_pressure.ptx");
        device.load_ptx(ptx.into(), "sph_pressure", &[
            "sph_density_kernel",
            "sph_pressure_force_kernel",
            "compute_pressure_kernel",
            "update_smoothing_length_kernel",
            "apply_pressure_kick_kernel",
        ])?;

        // Allocate GPU buffers
        let pos_x = device.alloc_zeros::<f32>(n)?;
        let pos_y = device.alloc_zeros::<f32>(n)?;
        let pos_z = device.alloc_zeros::<f32>(n)?;
        let density = device.alloc_zeros::<f32>(n)?;
        let pressure = device.alloc_zeros::<f32>(n)?;
        let temperature = device.alloc_zeros::<f32>(n)?;
        let acc_x = device.alloc_zeros::<f32>(n)?;
        let acc_y = device.alloc_zeros::<f32>(n)?;
        let acc_z = device.alloc_zeros::<f32>(n)?;

        // Initial smoothing length estimate: h = box / n^(1/3) * 2
        let h_init = (box_size as f32) / (n as f32).powf(1.0/3.0) * 2.0;
        let h_min = h_init * 0.1;
        let h_max = (box_size as f32) / 10.0;

        // Initialize smooth_h to h_init
        let h_init_vec = vec![h_init; n];
        let smooth_h = device.htod_sync_copy(&h_init_vec)?;

        Ok(Self {
            device,
            pos_x, pos_y, pos_z,
            density, pressure, smooth_h, temperature,
            acc_x, acc_y, acc_z,
            n,
            mass: mass as f32,
            box_size: box_size as f32,
            h_min, h_max,
            eta: 1.2,
        })
    }

    /// Upload positions from f64 arrays (converting to f32)
    pub fn upload_positions(&mut self, pos: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n;
        let mut px = vec![0.0f32; n];
        let mut py = vec![0.0f32; n];
        let mut pz = vec![0.0f32; n];

        for i in 0..n {
            px[i] = pos[i * 3] as f32;
            py[i] = pos[i * 3 + 1] as f32;
            pz[i] = pos[i * 3 + 2] as f32;
        }

        self.device.htod_sync_copy_into(&px, &mut self.pos_x)?;
        self.device.htod_sync_copy_into(&py, &mut self.pos_y)?;
        self.device.htod_sync_copy_into(&pz, &mut self.pos_z)?;
        Ok(())
    }

    /// Upload temperatures
    pub fn upload_temperatures(&mut self, temp: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        let temp_f32: Vec<f32> = temp.iter().map(|&t| t as f32).collect();
        self.device.htod_sync_copy_into(&temp_f32, &mut self.temperature)?;
        Ok(())
    }

    /// Compute SPH density for all particles
    pub fn compute_density(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.device.get_func("sph_pressure", "sph_density_kernel")
            .ok_or("sph_density_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &self.pos_x,
                &self.pos_y,
                &self.pos_z,
                &self.smooth_h,
                &mut self.density,
                self.mass,
                self.box_size,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Compute pressure from density and temperature
    /// P = ρ × (k_B/m_p) × T / μ
    pub fn compute_pressure(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.device.get_func("sph_pressure", "compute_pressure_kernel")
            .ok_or("compute_pressure_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &self.density,
                &self.temperature,
                &mut self.pressure,
                K_B_OVER_MP_CODE,
                MU_MOL,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Update adaptive smoothing lengths
    /// h = η × (m/ρ)^(1/3)
    pub fn update_smoothing_lengths(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.device.get_func("sph_pressure", "update_smoothing_length_kernel")
            .ok_or("update_smoothing_length_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &self.density,
                &mut self.smooth_h,
                self.mass,
                self.eta,
                self.h_min,
                self.h_max,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Compute SPH pressure forces
    pub fn compute_pressure_force(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.device.get_func("sph_pressure", "sph_pressure_force_kernel")
            .ok_or("sph_pressure_force_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &self.pos_x,
                &self.pos_y,
                &self.pos_z,
                &self.density,
                &self.pressure,
                &self.smooth_h,
                &mut self.acc_x,
                &mut self.acc_y,
                &mut self.acc_z,
                self.mass,
                self.box_size,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Download pressure accelerations to CPU (f64)
    pub fn download_accelerations(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let ax = self.device.dtoh_sync_copy(&self.acc_x)?;
        let ay = self.device.dtoh_sync_copy(&self.acc_y)?;
        let az = self.device.dtoh_sync_copy(&self.acc_z)?;

        let mut acc = vec![0.0f64; self.n * 3];
        for i in 0..self.n {
            acc[i * 3] = ax[i] as f64;
            acc[i * 3 + 1] = ay[i] as f64;
            acc[i * 3 + 2] = az[i] as f64;
        }
        Ok(acc)
    }

    /// Download densities to CPU
    pub fn download_densities(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let rho = self.device.dtoh_sync_copy(&self.density)?;
        Ok(rho.iter().map(|&r| r as f64).collect())
    }

    /// Full SPH pressure step:
    /// 1. Upload positions and temperatures
    /// 2. Compute density
    /// 3. Update smoothing lengths
    /// 4. Compute pressure
    /// 5. Compute pressure forces
    /// 6. Return accelerations
    pub fn compute_pressure_accelerations(
        &mut self,
        positions: &[f64],
        temperatures: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        self.upload_positions(positions)?;
        self.upload_temperatures(temperatures)?;
        self.compute_density()?;
        self.update_smoothing_lengths()?;
        self.compute_pressure()?;
        self.compute_pressure_force()?;
        self.download_accelerations()
    }

    /// Get mean density (for diagnostics)
    pub fn get_mean_density(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let rho = self.device.dtoh_sync_copy(&self.density)?;
        let sum: f32 = rho.iter().sum();
        Ok((sum / self.n as f32) as f64)
    }

    /// Get max density (for diagnostics)
    pub fn get_max_density(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let rho = self.device.dtoh_sync_copy(&self.density)?;
        let max = rho.iter().cloned().fold(0.0f32, f32::max);
        Ok(max as f64)
    }

    /// Get mean pressure acceleration magnitude (for diagnostics)
    pub fn get_mean_acc_magnitude(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let ax = self.device.dtoh_sync_copy(&self.acc_x)?;
        let ay = self.device.dtoh_sync_copy(&self.acc_y)?;
        let az = self.device.dtoh_sync_copy(&self.acc_z)?;

        let sum: f32 = (0..self.n)
            .map(|i| (ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]).sqrt())
            .sum();
        Ok((sum / self.n as f32) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sph_gpu_creation() {
        // CudaDevice::new returns Arc<CudaDevice> in recent cudarc versions
        let device = CudaDevice::new(0).expect("No CUDA device");

        let sph = GpuSphPressure::new(device, 1000, 1e10, 100.0);
        assert!(sph.is_ok(), "Failed to create GpuSphPressure");
    }
}
