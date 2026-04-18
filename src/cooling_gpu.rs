//! GPU Cooling Module for Janus Baryonic Physics
//!
//! Native CUDA implementation of:
//! - Radiative cooling (H, He, metals, Bremsstrahlung)
//! - UV photoheating (Haardt-Madau 2012 fit)
//! - Rahmati self-shielding
//! - Star formation
//! - SN feedback
//!
//! Replaces Grackle CPU library for performance.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use rand::Rng;

// Physical constants
pub const K_B_OVER_MP: f64 = 8.254e9;       // k_B/m_p in (km/s)^2/K
pub const MU_IONIZED: f64 = 0.6;            // Mean molecular weight
pub const T_FLOOR: f64 = 100.0;             // Minimum temperature [K]

/// GPU Cooling Calculator
pub struct GpuCooling {
    device: Arc<CudaDevice>,

    // GPU buffers (f64 for accuracy)
    internal_energy: CudaSlice<f64>,
    sph_density: CudaSlice<f64>,
    temperature: CudaSlice<f64>,
    signs: CudaSlice<i32>,
    star_flag: CudaSlice<i32>,
    sn_flag: CudaSlice<i32>,
    random_vals: CudaSlice<f64>,
    random_theta: CudaSlice<f64>,
    random_phi: CudaSlice<f64>,

    // Parameters
    n: usize,
    rho_to_nh: f64,   // Density conversion to nH [cm^-3]
    g_code: f64,      // G in code units

    // Stats
    pub total_stars_formed: u64,
    pub total_sn_events: u64,
}

impl GpuCooling {
    /// Create new GPU cooling calculator
    pub fn new(
        device: Arc<CudaDevice>,
        n: usize,
        l_box: f64,       // Box size [Mpc]
        m_particle: f64,  // Particle mass [M_sun]
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load PTX module
        let ptx = include_str!("../cuda/cooling_kernel.ptx");
        device.load_ptx(ptx.into(), "cooling", &[
            "apply_cooling_kernel",
            "apply_sf_kernel",
            "apply_feedback_kernel",
            "compute_temperature_kernel",
            "init_internal_energy_kernel",
        ])?;

        // Allocate GPU buffers
        let internal_energy = device.alloc_zeros::<f64>(n)?;
        let sph_density = device.alloc_zeros::<f64>(n)?;
        let temperature = device.alloc_zeros::<f64>(n)?;
        let signs = device.alloc_zeros::<i32>(n)?;
        let star_flag = device.alloc_zeros::<i32>(n)?;
        let sn_flag = device.alloc_zeros::<i32>(n)?;
        let random_vals = device.alloc_zeros::<f64>(n)?;
        let random_theta = device.alloc_zeros::<f64>(n)?;
        let random_phi = device.alloc_zeros::<f64>(n)?;

        // Compute density conversion factor
        // rho_code [M_sun/Mpc^3] -> nH [cm^-3]
        // 1 M_sun = 1.989e33 g
        // 1 Mpc = 3.086e24 cm
        // nH = rho * X_H / m_p where X_H = 0.76
        let msun_g = 1.989e33;
        let mpc_cm = 3.086e24;
        let mp_g = 1.6726e-24;
        let x_h = 0.76;
        let rho_to_nh = msun_g / (mpc_cm * mpc_cm * mpc_cm) * x_h / mp_g;

        // G in code units: [Mpc^3 / (M_sun * Gyr^2)]
        let g_code = 4.498e-15;  // G in code units

        Ok(Self {
            device,
            internal_energy,
            sph_density,
            temperature,
            signs,
            star_flag,
            sn_flag,
            random_vals,
            random_theta,
            random_phi,
            n,
            rho_to_nh,
            g_code,
            total_stars_formed: 0,
            total_sn_events: 0,
        })
    }

    /// Initialize internal energy from temperature
    pub fn init_from_temperature(
        &mut self,
        t_init_plus: f64,
        t_init_minus: f64,
        signs_host: &[i32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Upload signs
        self.device.htod_sync_copy_into(signs_host, &mut self.signs)?;

        let func = self.device.get_func("cooling", "init_internal_energy_kernel")
            .ok_or("init_internal_energy_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &mut self.internal_energy,
                &self.signs,
                t_init_plus,
                t_init_minus,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Upload SPH densities
    pub fn upload_densities(&mut self, densities: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        self.device.htod_sync_copy_into(densities, &mut self.sph_density)?;
        Ok(())
    }

    /// Upload signs (if changed)
    pub fn upload_signs(&mut self, signs: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
        self.device.htod_sync_copy_into(signs, &mut self.signs)?;
        Ok(())
    }

    /// Apply cooling for one timestep
    pub fn apply_cooling(&mut self, dt_gyr: f64, z: f64) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.device.get_func("cooling", "apply_cooling_kernel")
            .ok_or("apply_cooling_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &mut self.internal_energy,
                &self.sph_density,
                &self.signs,
                dt_gyr,
                z,
                self.rho_to_nh,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Apply star formation (returns number of new stars)
    pub fn apply_star_formation(&mut self, dt_gyr: f64) -> Result<u64, Box<dyn std::error::Error>> {
        // Generate random values
        let mut rng = rand::thread_rng();
        let random_vals: Vec<f64> = (0..self.n).map(|_| rng.gen::<f64>()).collect();
        self.device.htod_sync_copy_into(&random_vals, &mut self.random_vals)?;

        let func = self.device.get_func("cooling", "apply_sf_kernel")
            .ok_or("apply_sf_kernel not found")?;

        let block_size = 256;
        let grid_size = (self.n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &self.internal_energy,
                &self.sph_density,
                &self.signs,
                &mut self.star_flag,
                &self.random_vals,
                dt_gyr,
                self.rho_to_nh,
                self.g_code,
                self.n as i32,
            ))?;
        }

        self.device.synchronize()?;

        // Count new stars
        let star_flags = self.device.dtoh_sync_copy(&self.star_flag)?;
        let n_new_stars = star_flags.iter().filter(|&&x| x > 0).count() as u64;
        self.total_stars_formed += n_new_stars;

        Ok(n_new_stars)
    }

    /// Get mean temperature of m+ particles
    pub fn get_mean_temperature_plus(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let u = self.device.dtoh_sync_copy(&self.internal_energy)?;
        let signs = self.device.dtoh_sync_copy(&self.signs)?;

        let mut sum_t = 0.0;
        let mut count = 0;

        for i in 0..self.n {
            if signs[i] > 0 {
                let t = (2.0 / 3.0) * MU_IONIZED * u[i] / K_B_OVER_MP;
                if !t.is_nan() && t > 0.0 {
                    sum_t += t;
                    count += 1;
                }
            }
        }

        Ok(if count > 0 { sum_t / count as f64 } else { 0.0 })
    }

    /// Get internal energy array (for diagnostics)
    pub fn get_internal_energy(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.internal_energy)?)
    }

    /// Set internal energy (for reloading from snapshot)
    pub fn set_internal_energy(&mut self, u: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        self.device.htod_sync_copy_into(u, &mut self.internal_energy)?;
        Ok(())
    }

    /// Get number of star flags set
    pub fn get_star_count(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let flags = self.device.dtoh_sync_copy(&self.star_flag)?;
        Ok(flags.iter().filter(|&&x| x > 0).count() as u64)
    }

    /// Check for NaN in internal energy
    pub fn has_nan(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let u = self.device.dtoh_sync_copy(&self.internal_energy)?;
        Ok(u.iter().any(|&x| x.is_nan() || x.is_infinite()))
    }
}

/// Simplified CPU cooling rate calculation (for comparison/validation)
pub fn cpu_cooling_rate(t: f64, n_h: f64, z: f64) -> f64 {
    let sqrt_t = t.sqrt();

    // Cooling
    let lambda_h = 7.5e-19 * (-118348.0 / t).exp() / (1.0 + (t / 1e5).sqrt());
    let lambda_he = 9.1e-27 * sqrt_t * (-13179.0 / t).exp();
    let lambda_ff = 1.42e-27 * sqrt_t;

    // Heating (UV)
    let zp1 = 1.0 + z;
    let gamma_uv = 1e-24 * zp1 * zp1 / (1.0 + (zp1 / 3.0).powi(5));

    // Self-shielding
    let x = n_h / 0.01;
    let shield = 0.98 * (1.0 + x.powf(1.64)).powf(-2.28) + 0.02 * (1.0 + x).powf(-0.84);

    lambda_h + lambda_he + lambda_ff - gamma_uv * shield / n_h
}
