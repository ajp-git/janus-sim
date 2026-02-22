//! GPU-only Janus PM Simulation
//!
//! All particle data lives on GPU. Zero CPU transfers in the main loop.
//! Only initialization and periodic snapshots involve CPU memory.

use crate::cuda_kernels::{
    GpuBuffer, Stream, device_sync,
    cic_deposit, green_gradient, force_interpolation,
    kick, drift, zero_float, scale_velocities,
    kinetic_energy_partial, segregation_partial,
    real_to_complex, complex_to_real,
};
use crate::cufft_ffi::{
    CufftComplex, CufftHandle, CufftType,
    cufftPlan3d, cufftExecC2C, cufftDestroy,
    CUFFT_FORWARD, CUFFT_INVERSE,
};
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::f64::consts::PI;

// CUDA device-to-device memcpy
extern "C" {
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: u64, kind: i32) -> i32;
}
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

unsafe fn cuda_d2d_copy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize) -> i32 {
    cudaMemcpy(dst, src, size as u64, CUDA_MEMCPY_DEVICE_TO_DEVICE)
}

/// Particle data for initialization (CPU side)
#[derive(Clone)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub sign: i8,
}

/// GPU-only Janus PM Simulation
pub struct JanusPMGpu {
    // Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub n_grid: usize,

    // Physical parameters
    pub n_particles: usize,
    pub box_size: f64,
    pub dt: f32,
    pub k_softening: f32,

    // Cosmology
    pub cosmo: CosmoInterpolator,
    pub tau: f64,
    pub dtau_per_dt: f64,

    // Initial values for validation
    pub ke_initial: f64,
    pub seg_initial: f64,

    // Particle counts
    pub n_positive: usize,
    pub n_negative: usize,

    // GPU particle buffers
    d_pos_x: GpuBuffer<f64>,
    d_pos_y: GpuBuffer<f64>,
    d_pos_z: GpuBuffer<f64>,
    d_vel_x: GpuBuffer<f32>,
    d_vel_y: GpuBuffer<f32>,
    d_vel_z: GpuBuffer<f32>,
    d_signs: GpuBuffer<i8>,

    // GPU force buffers
    d_fx: GpuBuffer<f32>,
    d_fy: GpuBuffer<f32>,
    d_fz: GpuBuffer<f32>,

    // GPU density grids
    d_rho_plus: GpuBuffer<f32>,
    d_rho_minus: GpuBuffer<f32>,

    // GPU acceleration grids (real space, after IFFT)
    d_gx_plus: GpuBuffer<f32>,
    d_gy_plus: GpuBuffer<f32>,
    d_gz_plus: GpuBuffer<f32>,
    d_gx_minus: GpuBuffer<f32>,
    d_gy_minus: GpuBuffer<f32>,
    d_gz_minus: GpuBuffer<f32>,

    // GPU FFT work buffer (complex, reused)
    d_fft_work: GpuBuffer<CufftComplex>,

    // GPU gradient work buffers (k-space, complex)
    d_gx_k: GpuBuffer<CufftComplex>,
    d_gy_k: GpuBuffer<CufftComplex>,
    d_gz_k: GpuBuffer<CufftComplex>,

    // GPU reduction buffers
    d_ke_partial: GpuBuffer<f32>,
    d_seg_pos: GpuBuffer<f64>,
    d_seg_neg: GpuBuffer<f64>,
    n_reduction_blocks: usize,

    // CuFFT plans
    fft_plan: CufftHandle,

    // CUDA stream
    stream: Stream,

    // Step counter
    pub step: usize,
}

impl JanusPMGpu {
    /// Create new GPU simulation from particle list
    pub fn new(
        particles: Vec<Particle>,
        nx: usize,
        ny: usize,
        nz: usize,
        box_size: f64,
        dt: f32,
        eta: f64,
        z_init: f64,
    ) -> Result<Self, String> {
        let n_particles = particles.len();
        let n_grid = nx * ny * nz;

        // Cosmology interpolator
        let params = JanusParams::from_eta(eta);
        let cosmo = CosmoInterpolator::new(&params, z_init);
        let tau = cosmo.tau_start;
        let dtau_per_dt = 0.013205;  // Validated from BH production

        // Grid spacing and softening
        let dx = box_size as f32 / nx as f32;
        let k_softening = (PI as f32) / (8.0 * dx);

        // Count positive/negative
        let n_positive = particles.iter().filter(|p| p.sign > 0).count();
        let n_negative = n_particles - n_positive;

        // Create CUDA stream
        let stream = Stream::new()?;

        // Allocate GPU particle buffers
        println!("  Allocating GPU particle buffers ({:.2} GB)...",
                 (n_particles as f64 * (3.0 * 8.0 + 3.0 * 4.0 + 1.0)) / 1e9);

        let d_pos_x = GpuBuffer::<f64>::new(n_particles)?;
        let d_pos_y = GpuBuffer::<f64>::new(n_particles)?;
        let d_pos_z = GpuBuffer::<f64>::new(n_particles)?;
        let d_vel_x = GpuBuffer::<f32>::new(n_particles)?;
        let d_vel_y = GpuBuffer::<f32>::new(n_particles)?;
        let d_vel_z = GpuBuffer::<f32>::new(n_particles)?;
        let d_signs = GpuBuffer::<i8>::new(n_particles)?;

        // Force buffers
        let d_fx = GpuBuffer::<f32>::new(n_particles)?;
        let d_fy = GpuBuffer::<f32>::new(n_particles)?;
        let d_fz = GpuBuffer::<f32>::new(n_particles)?;

        // Grid buffers
        println!("  Allocating GPU grid buffers ({:.2} GB)...",
                 (n_grid as f64 * (2.0 * 4.0 + 6.0 * 4.0 + 4.0 * 8.0)) / 1e9);

        let d_rho_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_rho_minus = GpuBuffer::<f32>::new(n_grid)?;

        let d_gx_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gy_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gz_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gx_minus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gy_minus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gz_minus = GpuBuffer::<f32>::new(n_grid)?;

        // FFT work buffers
        let d_fft_work = GpuBuffer::<CufftComplex>::new(n_grid)?;
        let d_gx_k = GpuBuffer::<CufftComplex>::new(n_grid)?;
        let d_gy_k = GpuBuffer::<CufftComplex>::new(n_grid)?;
        let d_gz_k = GpuBuffer::<CufftComplex>::new(n_grid)?;

        // Reduction buffers
        let block_size = 256;
        let n_reduction_blocks = (n_particles + block_size - 1) / block_size;
        let d_ke_partial = GpuBuffer::<f32>::new(n_reduction_blocks)?;
        let d_seg_pos = GpuBuffer::<f64>::new(n_reduction_blocks * 4)?;
        let d_seg_neg = GpuBuffer::<f64>::new(n_reduction_blocks * 4)?;

        // Create CuFFT plan
        println!("  Creating CuFFT plan ({}³)...", nx);
        let mut fft_plan: CufftHandle = 0;
        let result = unsafe {
            cufftPlan3d(
                &mut fft_plan,
                nx as i32,
                ny as i32,
                nz as i32,
                CufftType::C2C,
            )
        };
        if !result.is_ok() {
            return Err(format!("cufftPlan3d failed: {:?}", result));
        }

        // Copy particle data to GPU
        println!("  Copying {} particles to GPU...", n_particles);
        let pos_x: Vec<f64> = particles.iter().map(|p| p.x).collect();
        let pos_y: Vec<f64> = particles.iter().map(|p| p.y).collect();
        let pos_z: Vec<f64> = particles.iter().map(|p| p.z).collect();
        let vel_x: Vec<f32> = particles.iter().map(|p| p.vx).collect();
        let vel_y: Vec<f32> = particles.iter().map(|p| p.vy).collect();
        let vel_z: Vec<f32> = particles.iter().map(|p| p.vz).collect();
        let signs: Vec<i8> = particles.iter().map(|p| p.sign).collect();

        d_pos_x.copy_from_host(&pos_x)?;
        d_pos_y.copy_from_host(&pos_y)?;
        d_pos_z.copy_from_host(&pos_z)?;
        d_vel_x.copy_from_host(&vel_x)?;
        d_vel_y.copy_from_host(&vel_y)?;
        d_vel_z.copy_from_host(&vel_z)?;
        d_signs.copy_from_host(&signs)?;

        device_sync()?;

        let mut sim = Self {
            nx, ny, nz, n_grid,
            n_particles,
            box_size,
            dt,
            k_softening,
            cosmo,
            tau,
            dtau_per_dt,
            ke_initial: 0.0,
            seg_initial: 0.0,
            n_positive,
            n_negative,
            d_pos_x, d_pos_y, d_pos_z,
            d_vel_x, d_vel_y, d_vel_z,
            d_signs,
            d_fx, d_fy, d_fz,
            d_rho_plus, d_rho_minus,
            d_gx_plus, d_gy_plus, d_gz_plus,
            d_gx_minus, d_gy_minus, d_gz_minus,
            d_fft_work,
            d_gx_k, d_gy_k, d_gz_k,
            d_ke_partial,
            d_seg_pos, d_seg_neg,
            n_reduction_blocks,
            fft_plan,
            stream,
            step: 0,
        };

        // Compute initial KE and segregation
        sim.ke_initial = sim.kinetic_energy()?;
        sim.seg_initial = sim.segregation()?;

        Ok(sim)
    }

    /// Virialize using hardcoded α = 4.57 from BH reference
    pub fn virialize(&mut self) -> Result<f64, String> {
        let alpha = 4.57_f32;
        println!("  Using hardcoded α = {:.2} (BH reference)", alpha);

        scale_velocities(
            &self.d_vel_x,
            &self.d_vel_y,
            &self.d_vel_z,
            alpha,
            &self.stream,
        );
        self.stream.sync()?;

        self.ke_initial = self.kinetic_energy()?;
        self.seg_initial = self.segregation()?;

        Ok(alpha as f64)
    }

    /// Get current scale factor
    pub fn scale_factor(&self) -> f64 {
        let (a, _h) = self.cosmo.get_params_at_tau(self.tau);
        a
    }

    /// Perform one integration step (KDK leapfrog)
    pub fn step(&mut self) -> Result<(), String> {
        let dx = self.box_size as f32 / self.nx as f32;

        // Get cosmological parameters
        let (_a, h) = self.cosmo.get_params_at_tau(self.tau);
        let hubble_friction = (h * self.dtau_per_dt) as f32;

        // Half-kick
        self.compute_forces()?;
        kick(
            &self.d_vel_x, &self.d_vel_y, &self.d_vel_z,
            &self.d_fx, &self.d_fy, &self.d_fz,
            self.dt * 0.5,
            hubble_friction,
            &self.stream,
        );

        // Full drift
        drift(
            &self.d_pos_x, &self.d_pos_y, &self.d_pos_z,
            &self.d_vel_x, &self.d_vel_y, &self.d_vel_z,
            self.dt,
            self.box_size,
            &self.stream,
        );

        // Advance cosmological time
        self.tau += self.dt as f64 * self.dtau_per_dt;

        // Second half-kick with updated cosmology
        let (_a, h) = self.cosmo.get_params_at_tau(self.tau);
        let hubble_friction = (h * self.dtau_per_dt) as f32;

        self.compute_forces()?;
        kick(
            &self.d_vel_x, &self.d_vel_y, &self.d_vel_z,
            &self.d_fx, &self.d_fy, &self.d_fz,
            self.dt * 0.5,
            hubble_friction,
            &self.stream,
        );

        self.stream.sync()?;
        self.step += 1;

        Ok(())
    }

    /// Compute forces using PM method
    fn compute_forces(&mut self) -> Result<(), String> {
        // Zero density grids
        zero_float(&self.d_rho_plus, &self.stream);
        zero_float(&self.d_rho_minus, &self.stream);

        // CIC deposit
        cic_deposit(
            &self.d_pos_x, &self.d_pos_y, &self.d_pos_z,
            &self.d_signs,
            &self.d_rho_plus, &self.d_rho_minus,
            self.nx, self.ny, self.nz,
            self.box_size as f32,
            &self.stream,
        );

        // Process positive density → g+
        self.solve_poisson_plus()?;

        // Process negative density → g-
        self.solve_poisson_minus()?;

        // Interpolate forces to particles
        force_interpolation(
            &self.d_pos_x, &self.d_pos_y, &self.d_pos_z,
            &self.d_signs,
            &self.d_gx_plus, &self.d_gy_plus, &self.d_gz_plus,
            &self.d_gx_minus, &self.d_gy_minus, &self.d_gz_minus,
            &self.d_fx, &self.d_fy, &self.d_fz,
            self.nx, self.ny, self.nz,
            self.box_size as f32,
            &self.stream,
        );

        Ok(())
    }

    /// Solve Poisson for positive density grid
    fn solve_poisson_plus(&mut self) -> Result<(), String> {
        let dx = self.box_size as f32 / self.nx as f32;
        let norm = 1.0 / self.n_grid as f32;

        // Copy ρ+ to complex FFT buffer
        self.copy_real_to_complex_from_rho_plus()?;

        // Forward FFT
        let result = unsafe {
            cufftExecC2C(self.fft_plan, self.d_fft_work.ptr(), self.d_fft_work.ptr(), CUFFT_FORWARD)
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C forward failed: {:?}", result));
        }

        // Green's function + gradient
        green_gradient(
            &self.d_fft_work,
            &self.d_gx_k, &self.d_gy_k, &self.d_gz_k,
            self.nx, self.ny, self.nz,
            dx, self.k_softening, &self.stream,
        );

        // IFFT for each gradient component → g+
        self.ifft_to_gx_plus(norm)?;
        self.ifft_to_gy_plus(norm)?;
        self.ifft_to_gz_plus(norm)?;

        Ok(())
    }

    /// Solve Poisson for negative density grid
    fn solve_poisson_minus(&mut self) -> Result<(), String> {
        let dx = self.box_size as f32 / self.nx as f32;
        let norm = 1.0 / self.n_grid as f32;

        // Copy ρ- to complex FFT buffer
        self.copy_real_to_complex_from_rho_minus()?;

        // Forward FFT
        let result = unsafe {
            cufftExecC2C(self.fft_plan, self.d_fft_work.ptr(), self.d_fft_work.ptr(), CUFFT_FORWARD)
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C forward failed: {:?}", result));
        }

        // Green's function + gradient
        green_gradient(
            &self.d_fft_work,
            &self.d_gx_k, &self.d_gy_k, &self.d_gz_k,
            self.nx, self.ny, self.nz,
            dx, self.k_softening, &self.stream,
        );

        // IFFT for each gradient component → g-
        self.ifft_to_gx_minus(norm)?;
        self.ifft_to_gy_minus(norm)?;
        self.ifft_to_gz_minus(norm)?;

        Ok(())
    }

    /// Copy ρ+ to complex FFT buffer (GPU-only)
    fn copy_real_to_complex_from_rho_plus(&self) -> Result<(), String> {
        real_to_complex(&self.d_rho_plus, &self.d_fft_work, &self.stream);
        Ok(())
    }

    /// Copy ρ- to complex FFT buffer (GPU-only)
    fn copy_real_to_complex_from_rho_minus(&self) -> Result<(), String> {
        real_to_complex(&self.d_rho_minus, &self.d_fft_work, &self.stream);
        Ok(())
    }

    /// IFFT gx_k → gx_plus (GPU-only)
    fn ifft_to_gx_plus(&self, norm: f32) -> Result<(), String> {
        self.ifft_gradient_to_real_gpu(&self.d_gx_k, &self.d_gx_plus, norm)
    }

    /// IFFT gy_k → gy_plus (GPU-only)
    fn ifft_to_gy_plus(&self, norm: f32) -> Result<(), String> {
        self.ifft_gradient_to_real_gpu(&self.d_gy_k, &self.d_gy_plus, norm)
    }

    /// IFFT gz_k → gz_plus (GPU-only)
    fn ifft_to_gz_plus(&self, norm: f32) -> Result<(), String> {
        self.ifft_gradient_to_real_gpu(&self.d_gz_k, &self.d_gz_plus, norm)
    }

    /// IFFT gx_k → gx_minus (GPU-only)
    fn ifft_to_gx_minus(&self, norm: f32) -> Result<(), String> {
        self.ifft_gradient_to_real_gpu(&self.d_gx_k, &self.d_gx_minus, norm)
    }

    /// IFFT gy_k → gy_minus (GPU-only)
    fn ifft_to_gy_minus(&self, norm: f32) -> Result<(), String> {
        self.ifft_gradient_to_real_gpu(&self.d_gy_k, &self.d_gy_minus, norm)
    }

    /// IFFT gz_k → gz_minus (GPU-only)
    fn ifft_to_gz_minus(&self, norm: f32) -> Result<(), String> {
        self.ifft_gradient_to_real_gpu(&self.d_gz_k, &self.d_gz_minus, norm)
    }

    /// IFFT a gradient component to real grid (GPU-only, uses d_fft_work as temp)
    fn ifft_gradient_to_real_gpu(&self, d_complex: &GpuBuffer<CufftComplex>, d_real: &GpuBuffer<f32>, norm: f32) -> Result<(), String> {
        // Copy gradient k-space to work buffer (device-to-device)
        // We need a D2D copy kernel, but for now use cuMemcpy
        unsafe {
            let size = self.n_grid * std::mem::size_of::<CufftComplex>();
            let err = cuda_d2d_copy(
                self.d_fft_work.ptr() as *mut std::ffi::c_void,
                d_complex.ptr() as *const std::ffi::c_void,
                size,
            );
            if err != 0 {
                return Err(format!("cudaMemcpy D2D failed: {}", err));
            }
        }

        // Inverse FFT in-place
        let result = unsafe {
            cufftExecC2C(self.fft_plan, self.d_fft_work.ptr(), self.d_fft_work.ptr(), CUFFT_INVERSE)
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C inverse failed: {:?}", result));
        }

        // Extract real parts with normalization (GPU kernel)
        complex_to_real(&self.d_fft_work, d_real, norm, &self.stream);
        Ok(())
    }

    /// Copy real f32 grid to complex buffer (imaginary = 0) - UNUSED
    #[allow(dead_code)]
    fn copy_real_to_complex(&self, d_real: &GpuBuffer<f32>) -> Result<(), String> {
        // This needs a kernel, but for now we can use cudaMemcpy2D or a simple approach
        // Actually, we need to interleave. Let's use the d_fft_work buffer differently.
        // For now, copy to host, convert, copy back. (Will optimize later)

        // TODO: Add a CUDA kernel for this
        let mut host_real = vec![0.0_f32; self.n_grid];
        d_real.copy_to_host(&mut host_real)?;

        let host_complex: Vec<CufftComplex> = host_real.iter()
            .map(|&r| CufftComplex::new(r, 0.0))
            .collect();

        self.d_fft_work.copy_from_host(&host_complex)?;
        Ok(())
    }

    /// IFFT complex to real (extracts real part, normalizes) - UNUSED
    #[allow(dead_code)]
    fn ifft_complex_to_real(
        &self,
        d_complex: &GpuBuffer<CufftComplex>,
        d_real: &GpuBuffer<f32>,
        norm: f32,
    ) -> Result<(), String> {
        // Copy to work buffer
        let mut host_complex = vec![CufftComplex::new(0.0, 0.0); self.n_grid];
        d_complex.copy_to_host(&mut host_complex)?;
        self.d_fft_work.copy_from_host(&host_complex)?;

        // Inverse FFT
        let result = unsafe {
            cufftExecC2C(
                self.fft_plan,
                self.d_fft_work.ptr(),
                self.d_fft_work.ptr(),
                CUFFT_INVERSE,
            )
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C inverse failed: {:?}", result));
        }

        // Copy back and extract real parts
        self.d_fft_work.copy_to_host(&mut host_complex)?;

        let host_real: Vec<f32> = host_complex.iter()
            .map(|c| c.x * norm)
            .collect();

        d_real.copy_from_host(&host_real)?;
        Ok(())
    }

    /// Compute kinetic energy
    pub fn kinetic_energy(&self) -> Result<f64, String> {
        kinetic_energy_partial(
            &self.d_vel_x, &self.d_vel_y, &self.d_vel_z,
            &self.d_ke_partial,
            &self.stream,
        );
        self.stream.sync()?;

        // Sum on CPU
        let mut partial = vec![0.0_f32; self.n_reduction_blocks];
        self.d_ke_partial.copy_to_host(&mut partial)?;

        let ke: f64 = partial.iter().map(|&x| x as f64).sum();
        Ok(ke)
    }

    /// Compute segregation metric
    pub fn segregation(&self) -> Result<f64, String> {
        segregation_partial(
            &self.d_pos_x, &self.d_pos_y, &self.d_pos_z,
            &self.d_signs,
            &self.d_seg_pos, &self.d_seg_neg,
            self.box_size,
            &self.stream,
        );
        self.stream.sync()?;

        // Sum on CPU
        let mut sum_pos = vec![0.0_f64; self.n_reduction_blocks * 4];
        let mut sum_neg = vec![0.0_f64; self.n_reduction_blocks * 4];
        self.d_seg_pos.copy_to_host(&mut sum_pos)?;
        self.d_seg_neg.copy_to_host(&mut sum_neg)?;

        // Aggregate
        let mut pos_x = 0.0_f64;
        let mut pos_y = 0.0_f64;
        let mut pos_z = 0.0_f64;
        let mut n_pos = 0.0_f64;
        let mut neg_x = 0.0_f64;
        let mut neg_y = 0.0_f64;
        let mut neg_z = 0.0_f64;
        let mut n_neg = 0.0_f64;

        for i in 0..self.n_reduction_blocks {
            pos_x += sum_pos[i * 4 + 0];
            pos_y += sum_pos[i * 4 + 1];
            pos_z += sum_pos[i * 4 + 2];
            n_pos += sum_pos[i * 4 + 3];

            neg_x += sum_neg[i * 4 + 0];
            neg_y += sum_neg[i * 4 + 1];
            neg_z += sum_neg[i * 4 + 2];
            n_neg += sum_neg[i * 4 + 3];
        }

        if n_pos < 1.0 || n_neg < 1.0 {
            return Ok(0.0);
        }

        // COM positions
        let com_pos = (pos_x / n_pos, pos_y / n_pos, pos_z / n_pos);
        let com_neg = (neg_x / n_neg, neg_y / n_neg, neg_z / n_neg);

        // Distance with periodic boundary
        let mut dx = (com_pos.0 - com_neg.0).abs();
        let mut dy = (com_pos.1 - com_neg.1).abs();
        let mut dz = (com_pos.2 - com_neg.2).abs();

        if dx > self.box_size / 2.0 { dx = self.box_size - dx; }
        if dy > self.box_size / 2.0 { dy = self.box_size - dy; }
        if dz > self.box_size / 2.0 { dz = self.box_size - dz; }

        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let max_sep = self.box_size * (3.0_f64).sqrt() / 2.0;

        Ok(distance / max_sep)
    }

    /// Download positions for snapshot (subsampled to reduce I/O)
    pub fn download_positions_subsampled(&self, subsample: usize) -> Result<Vec<(f32, f32, f32, i8)>, String> {
        let n_out = (self.n_particles + subsample - 1) / subsample;

        // Download all positions (we could optimize with a strided copy kernel)
        let mut pos_x = vec![0.0_f64; self.n_particles];
        let mut pos_y = vec![0.0_f64; self.n_particles];
        let mut pos_z = vec![0.0_f64; self.n_particles];
        let mut signs = vec![0_i8; self.n_particles];

        self.d_pos_x.copy_to_host(&mut pos_x)?;
        self.d_pos_y.copy_to_host(&mut pos_y)?;
        self.d_pos_z.copy_to_host(&mut pos_z)?;
        self.d_signs.copy_to_host(&mut signs)?;

        // Subsample
        let result: Vec<(f32, f32, f32, i8)> = (0..self.n_particles)
            .step_by(subsample)
            .map(|i| (pos_x[i] as f32, pos_y[i] as f32, pos_z[i] as f32, signs[i]))
            .collect();

        Ok(result)
    }

    /// Download all positions for full snapshot
    pub fn download_all_positions(&self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<i8>), String> {
        let mut pos_x = vec![0.0_f64; self.n_particles];
        let mut pos_y = vec![0.0_f64; self.n_particles];
        let mut pos_z = vec![0.0_f64; self.n_particles];
        let mut signs = vec![0_i8; self.n_particles];

        self.d_pos_x.copy_to_host(&mut pos_x)?;
        self.d_pos_y.copy_to_host(&mut pos_y)?;
        self.d_pos_z.copy_to_host(&mut pos_z)?;
        self.d_signs.copy_to_host(&mut signs)?;

        Ok((pos_x, pos_y, pos_z, signs))
    }

    /// Download all velocities for full snapshot
    pub fn download_all_velocities(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let mut vel_x = vec![0.0_f32; self.n_particles];
        let mut vel_y = vec![0.0_f32; self.n_particles];
        let mut vel_z = vec![0.0_f32; self.n_particles];

        self.d_vel_x.copy_to_host(&mut vel_x)?;
        self.d_vel_y.copy_to_host(&mut vel_y)?;
        self.d_vel_z.copy_to_host(&mut vel_z)?;

        Ok((vel_x, vel_y, vel_z))
    }

    /// Create simulation from checkpoint data (for resume)
    pub fn new_from_checkpoint(
        pos_x: Vec<f64>,
        pos_y: Vec<f64>,
        pos_z: Vec<f64>,
        vel_x: Vec<f32>,
        vel_y: Vec<f32>,
        vel_z: Vec<f32>,
        signs: Vec<i8>,
        nx: usize,
        box_size: f64,
        dt: f32,
        eta: f64,
        tau: f64,
        step: usize,
        ke_initial: f64,
        n_positive: usize,
        n_negative: usize,
    ) -> Result<Self, String> {
        let n_particles = pos_x.len();
        let ny = nx;
        let nz = nx;
        let n_grid = nx * ny * nz;

        // Cosmology interpolator
        let z_init = 5.0;  // Standard for our runs
        let params = JanusParams::from_eta(eta);
        let cosmo = CosmoInterpolator::new(&params, z_init);
        let dtau_per_dt = 0.013205;  // Validated from BH production

        // Grid spacing and softening
        let dx = box_size as f32 / nx as f32;
        let k_softening = (PI as f32) / (8.0 * dx);

        // Create CUDA stream
        let stream = Stream::new()?;

        // Allocate GPU particle buffers
        println!("  Allocating GPU particle buffers ({:.2} GB)...",
                 (n_particles as f64 * (3.0 * 8.0 + 3.0 * 4.0 + 1.0)) / 1e9);

        let d_pos_x = GpuBuffer::<f64>::new(n_particles)?;
        let d_pos_y = GpuBuffer::<f64>::new(n_particles)?;
        let d_pos_z = GpuBuffer::<f64>::new(n_particles)?;
        let d_vel_x = GpuBuffer::<f32>::new(n_particles)?;
        let d_vel_y = GpuBuffer::<f32>::new(n_particles)?;
        let d_vel_z = GpuBuffer::<f32>::new(n_particles)?;
        let d_signs = GpuBuffer::<i8>::new(n_particles)?;

        // Force buffers
        let d_fx = GpuBuffer::<f32>::new(n_particles)?;
        let d_fy = GpuBuffer::<f32>::new(n_particles)?;
        let d_fz = GpuBuffer::<f32>::new(n_particles)?;

        // Grid buffers
        println!("  Allocating GPU grid buffers ({:.2} GB)...",
                 (n_grid as f64 * (2.0 * 4.0 + 6.0 * 4.0 + 4.0 * 8.0)) / 1e9);

        let d_rho_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_rho_minus = GpuBuffer::<f32>::new(n_grid)?;

        let d_gx_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gy_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gz_plus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gx_minus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gy_minus = GpuBuffer::<f32>::new(n_grid)?;
        let d_gz_minus = GpuBuffer::<f32>::new(n_grid)?;

        // FFT work buffers
        let d_fft_work = GpuBuffer::<CufftComplex>::new(n_grid)?;
        let d_gx_k = GpuBuffer::<CufftComplex>::new(n_grid)?;
        let d_gy_k = GpuBuffer::<CufftComplex>::new(n_grid)?;
        let d_gz_k = GpuBuffer::<CufftComplex>::new(n_grid)?;

        // Reduction buffers
        let block_size = 256;
        let n_reduction_blocks = (n_particles + block_size - 1) / block_size;
        let d_ke_partial = GpuBuffer::<f32>::new(n_reduction_blocks)?;
        let d_seg_pos = GpuBuffer::<f64>::new(n_reduction_blocks * 4)?;
        let d_seg_neg = GpuBuffer::<f64>::new(n_reduction_blocks * 4)?;

        // Create CuFFT plan
        println!("  Creating CuFFT plan ({}³)...", nx);
        let mut fft_plan: CufftHandle = 0;
        let result = unsafe {
            cufftPlan3d(
                &mut fft_plan,
                nx as i32,
                ny as i32,
                nz as i32,
                CufftType::C2C,
            )
        };
        if !result.is_ok() {
            return Err(format!("cufftPlan3d failed: {:?}", result));
        }

        // Copy checkpoint data to GPU
        println!("  Copying {} particles to GPU...", n_particles);
        d_pos_x.copy_from_host(&pos_x)?;
        d_pos_y.copy_from_host(&pos_y)?;
        d_pos_z.copy_from_host(&pos_z)?;
        d_vel_x.copy_from_host(&vel_x)?;
        d_vel_y.copy_from_host(&vel_y)?;
        d_vel_z.copy_from_host(&vel_z)?;
        d_signs.copy_from_host(&signs)?;

        device_sync()?;

        // Compute current segregation for seg_initial (will differ from original)
        let sim = Self {
            nx, ny, nz, n_grid,
            n_particles,
            box_size,
            dt,
            k_softening,
            cosmo,
            tau,
            dtau_per_dt,
            ke_initial,
            seg_initial: 0.0,  // Will be updated
            n_positive,
            n_negative,
            d_pos_x, d_pos_y, d_pos_z,
            d_vel_x, d_vel_y, d_vel_z,
            d_signs,
            d_fx, d_fy, d_fz,
            d_rho_plus, d_rho_minus,
            d_gx_plus, d_gy_plus, d_gz_plus,
            d_gx_minus, d_gy_minus, d_gz_minus,
            d_fft_work,
            d_gx_k, d_gy_k, d_gz_k,
            d_ke_partial,
            d_seg_pos, d_seg_neg,
            n_reduction_blocks,
            fft_plan,
            stream,
            step,
        };

        Ok(sim)
    }

    /// Get simulation time
    pub fn time(&self) -> f64 {
        self.step as f64 * self.dt as f64
    }
}

impl Drop for JanusPMGpu {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.fft_plan) };
    }
}

/// Generate Janus initial conditions
pub fn generate_janus_ic(
    n_particles: usize,
    box_size: f64,
    velocity_dispersion: f32,
    eta: f64,
    seed: u64,
) -> Vec<Particle> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    // Fraction of negative mass from η
    let f_neg = eta / (1.0 + eta);
    let n_neg = (n_particles as f64 * f_neg).round() as usize;
    let n_pos = n_particles - n_neg;

    let mut particles = Vec::with_capacity(n_particles);

    for i in 0..n_particles {
        let x = rng.gen::<f64>() * box_size;
        let y = rng.gen::<f64>() * box_size;
        let z = rng.gen::<f64>() * box_size;

        // Gaussian velocities
        let vx = rng.gen::<f32>() * velocity_dispersion;
        let vy = rng.gen::<f32>() * velocity_dispersion;
        let vz = rng.gen::<f32>() * velocity_dispersion;

        let sign = if i < n_pos { 1_i8 } else { -1_i8 };

        particles.push(Particle { x, y, z, vx, vy, vz, sign });
    }

    particles
}
