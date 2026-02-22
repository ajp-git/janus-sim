//! Safe wrapper for CuFFT 3D Complex-to-Complex transforms
//!
//! Uses cudarc for memory management and our cufft_ffi for FFT execution.

use crate::cufft_ffi::{
    CufftComplex, CufftHandle, CufftResult, CufftType,
    CUFFT_FORWARD, CUFFT_INVERSE,
    cufftPlan3d, cufftExecC2C, cufftDestroy,
};
use std::ffi::c_int;

/// CUDA runtime FFI for memory operations
mod cuda_rt {
    use std::ffi::{c_int, c_void};
    use std::os::raw::c_ulong;

    pub type CudaError = c_int;
    pub const CUDA_SUCCESS: CudaError = 0;

    #[link(name = "cudart")]
    extern "C" {
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: c_ulong) -> CudaError;
        pub fn cudaFree(devPtr: *mut c_void) -> CudaError;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: c_ulong, kind: c_int) -> CudaError;
        pub fn cudaDeviceSynchronize() -> CudaError;
    }

    pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
}

/// Safe 3D C2C FFT plan wrapper with RAII
pub struct Cufft3dC2C {
    plan: CufftHandle,
    nx: usize,
    ny: usize,
    nz: usize,
    /// GPU buffer for in-place FFT
    d_data: *mut CufftComplex,
}

impl Cufft3dC2C {
    /// Create a new 3D FFT plan
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self, String> {
        let mut plan: CufftHandle = 0;

        // Create FFT plan
        let result = unsafe {
            cufftPlan3d(
                &mut plan as *mut CufftHandle,
                nx as c_int,
                ny as c_int,
                nz as c_int,
                CufftType::C2C,
            )
        };

        if !result.is_ok() {
            return Err(format!("cufftPlan3d failed: {:?}", result));
        }

        // Allocate GPU memory
        let n_elements = nx * ny * nz;
        let size_bytes = n_elements * std::mem::size_of::<CufftComplex>();
        let mut d_data: *mut CufftComplex = std::ptr::null_mut();

        let cuda_result = unsafe {
            cuda_rt::cudaMalloc(
                &mut d_data as *mut *mut CufftComplex as *mut *mut std::ffi::c_void,
                size_bytes as u64,
            )
        };

        if cuda_result != cuda_rt::CUDA_SUCCESS {
            unsafe { cufftDestroy(plan) };
            return Err(format!("cudaMalloc failed: {}", cuda_result));
        }

        Ok(Self { plan, nx, ny, nz, d_data })
    }

    /// Total number of elements
    pub fn n_elements(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Execute forward FFT (in-place on provided data)
    /// Input: host data, Output: transformed data written back to same slice
    pub fn forward(&self, data: &mut [CufftComplex]) -> Result<(), String> {
        self.execute(data, CUFFT_FORWARD)
    }

    /// Execute inverse FFT (in-place on provided data)
    /// Note: Does NOT normalize - caller must divide by N
    pub fn inverse(&self, data: &mut [CufftComplex]) -> Result<(), String> {
        self.execute(data, CUFFT_INVERSE)
    }

    /// Copy host data to internal GPU buffer
    pub fn copy_to_gpu(&self, data: &[CufftComplex]) -> Result<(), String> {
        let n = self.n_elements();
        if data.len() != n {
            return Err(format!("Data length {} != plan size {}", data.len(), n));
        }
        let size_bytes = n * std::mem::size_of::<CufftComplex>();
        let result = unsafe {
            cuda_rt::cudaMemcpy(
                self.d_data as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes as u64,
                cuda_rt::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if result != cuda_rt::CUDA_SUCCESS {
            return Err(format!("cudaMemcpy H2D failed: {}", result));
        }
        Ok(())
    }

    /// Copy GPU buffer to host data
    pub fn copy_from_gpu(&self, data: &mut [CufftComplex]) -> Result<(), String> {
        let n = self.n_elements();
        if data.len() != n {
            return Err(format!("Data length {} != plan size {}", data.len(), n));
        }
        let size_bytes = n * std::mem::size_of::<CufftComplex>();
        let result = unsafe {
            cuda_rt::cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.d_data as *const std::ffi::c_void,
                size_bytes as u64,
                cuda_rt::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if result != cuda_rt::CUDA_SUCCESS {
            return Err(format!("cudaMemcpy D2H failed: {}", result));
        }
        Ok(())
    }

    /// Execute forward FFT on internal GPU buffer (no memory transfer)
    pub fn forward_gpu(&self) -> Result<(), String> {
        let result = unsafe {
            cufftExecC2C(self.plan, self.d_data, self.d_data, CUFFT_FORWARD)
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C forward failed: {:?}", result));
        }
        Ok(())
    }

    /// Execute inverse FFT on internal GPU buffer (no memory transfer)
    pub fn inverse_gpu(&self) -> Result<(), String> {
        let result = unsafe {
            cufftExecC2C(self.plan, self.d_data, self.d_data, CUFFT_INVERSE)
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C inverse failed: {:?}", result));
        }
        Ok(())
    }

    /// Get raw GPU pointer for advanced operations
    pub fn gpu_ptr(&self) -> *mut CufftComplex {
        self.d_data
    }

    /// Synchronize GPU (wait for all operations to complete)
    pub fn sync(&self) {
        unsafe { cuda_rt::cudaDeviceSynchronize() };
    }

    fn execute(&self, data: &mut [CufftComplex], direction: c_int) -> Result<(), String> {
        let n = self.n_elements();
        if data.len() != n {
            return Err(format!("Data length {} != plan size {}", data.len(), n));
        }

        let size_bytes = n * std::mem::size_of::<CufftComplex>();

        // Copy data to GPU
        let cuda_result = unsafe {
            cuda_rt::cudaMemcpy(
                self.d_data as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes as u64,
                cuda_rt::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if cuda_result != cuda_rt::CUDA_SUCCESS {
            return Err(format!("cudaMemcpy H2D failed: {}", cuda_result));
        }

        // Execute FFT in-place
        let result = unsafe {
            cufftExecC2C(self.plan, self.d_data, self.d_data, direction)
        };
        if !result.is_ok() {
            return Err(format!("cufftExecC2C failed: {:?}", result));
        }

        // Synchronize
        unsafe { cuda_rt::cudaDeviceSynchronize() };

        // Copy data back to host
        let cuda_result = unsafe {
            cuda_rt::cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.d_data as *const std::ffi::c_void,
                size_bytes as u64,
                cuda_rt::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if cuda_result != cuda_rt::CUDA_SUCCESS {
            return Err(format!("cudaMemcpy D2H failed: {}", cuda_result));
        }

        Ok(())
    }

    /// Execute forward FFT, then inverse FFT (round-trip test)
    /// Normalizes by dividing by N after inverse
    pub fn roundtrip(&self, data: &mut [CufftComplex]) -> Result<(), String> {
        self.forward(data)?;
        self.inverse(data)?;

        // Normalize
        let norm = 1.0 / self.n_elements() as f32;
        for c in data.iter_mut() {
            c.x *= norm;
            c.y *= norm;
        }

        Ok(())
    }
}

impl Drop for Cufft3dC2C {
    fn drop(&mut self) {
        unsafe {
            if !self.d_data.is_null() {
                cuda_rt::cudaFree(self.d_data as *mut std::ffi::c_void);
            }
            cufftDestroy(self.plan);
        }
    }
}

// Safe to send between threads (CUDA handles are thread-safe for execution)
unsafe impl Send for Cufft3dC2C {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_fft() {
        // Small test - 8³ = 512 elements
        let n = 8;
        let plan = Cufft3dC2C::new(n, n, n).expect("Failed to create plan");

        let mut data: Vec<CufftComplex> = (0..n*n*n)
            .map(|i| CufftComplex::new(i as f32, 0.0))
            .collect();

        let original: Vec<CufftComplex> = data.clone();

        plan.roundtrip(&mut data).expect("Roundtrip failed");

        // Check reconstruction
        let max_err: f32 = data.iter().zip(original.iter())
            .map(|(a, b)| ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt())
            .fold(0.0, f32::max);

        assert!(max_err < 1e-4, "Reconstruction error {} >= 1e-4", max_err);
    }
}
