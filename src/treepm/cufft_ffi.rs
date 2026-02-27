//! cuFFT FFI bindings for TreePM GPU Poisson solver
//!
//! Links to libcufft_wrapper.so built from cuda/cufft_wrapper.cu

use std::ffi::c_void;

// FFI declarations
#[link(name = "cufft_wrapper")]
extern "C" {
    /// Initialize cuFFT plans for 3D grid
    fn cufft_init_3d(nx: i32, ny: i32, nz: i32) -> i32;

    /// Execute forward FFT (R2C)
    fn cufft_exec_r2c(d_input: *mut f64, d_output: *mut c_void) -> i32;

    /// Execute inverse FFT (C2R)
    fn cufft_exec_c2r(d_input: *mut c_void, d_output: *mut f64) -> i32;

    /// Cleanup cuFFT plans
    fn cufft_cleanup();

    /// Apply Green's function in k-space
    fn cufft_apply_green(
        d_data: *mut c_void,
        nx: i32, ny: i32, nz: i32,
        box_size: f64, g_constant: f64, r_s: f64,
    ) -> i32;

    /// Allocate GPU memory
    fn cufft_alloc(size_bytes: usize) -> *mut c_void;

    /// Free GPU memory
    fn cufft_free(ptr: *mut c_void);

    /// Copy host to device
    fn cufft_copy_h2d(d_dst: *mut c_void, h_src: *const c_void, size_bytes: usize) -> i32;

    /// Copy device to host
    fn cufft_copy_d2h(h_dst: *mut c_void, d_src: *const c_void, size_bytes: usize) -> i32;

    /// Normalize after inverse FFT
    fn cufft_normalize(d_data: *mut f64, nx: i32, ny: i32, nz: i32) -> i32;

    /// Device-to-device copy
    fn cufft_copy_d2d(d_dst: *mut c_void, d_src: *const c_void, size_bytes: usize) -> i32;

    /// Solve Poisson equation directly on device pointers (no host transfers)
    fn cufft_solve_device(
        d_rho: *mut f64,
        d_phi: *mut f64,
        nx: i32, ny: i32, nz: i32,
        box_size: f64, g_constant: f64, r_s: f64,
    ) -> i32;
}

/// Safe wrapper for cuFFT 3D Poisson solver
pub struct CuFFTPoisson {
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f64,
    /// Device pointer for density (real, nx*ny*nz)
    d_rho: *mut f64,
    /// Device pointer for k-space (complex, nx*ny*(nz/2+1)*2)
    d_rho_k: *mut c_void,
    /// Device pointer for potential (real, nx*ny*nz)
    d_phi: *mut f64,
}

impl CuFFTPoisson {
    /// Create new cuFFT Poisson solver
    ///
    /// # Arguments
    /// * `grid_size` - Grid dimension (assumed cubic)
    /// * `box_size` - Physical box size
    pub fn new(grid_size: usize, box_size: f64) -> Result<Self, String> {
        let nx = grid_size;
        let ny = grid_size;
        let nz = grid_size;

        // Initialize cuFFT plans
        let ret = unsafe { cufft_init_3d(nx as i32, ny as i32, nz as i32) };
        if ret != 0 {
            return Err("cuFFT initialization failed".to_string());
        }

        // Allocate GPU buffers
        let n_real = nx * ny * nz;
        let n_complex = nx * ny * (nz / 2 + 1);

        let d_rho = unsafe { cufft_alloc(n_real * 8) } as *mut f64;  // 8 bytes per f64
        let d_rho_k = unsafe { cufft_alloc(n_complex * 16) };  // 16 bytes per complex f64
        let d_phi = unsafe { cufft_alloc(n_real * 8) } as *mut f64;

        if d_rho.is_null() || d_rho_k.is_null() || d_phi.is_null() {
            return Err("GPU memory allocation failed".to_string());
        }

        Ok(Self {
            nx, ny, nz,
            box_size,
            d_rho, d_rho_k, d_phi,
        })
    }

    /// Solve Poisson equation: ∇²φ = 4πGρ
    ///
    /// Uses Gaussian splitting for TreePM:
    /// - Long-range: k-space multiplication by exp(-k²r_s²)
    /// - Short-range: handled by Tree separately
    ///
    /// # Arguments
    /// * `rho` - Density grid (host memory, nx*ny*nz)
    /// * `g_constant` - Gravitational constant
    /// * `r_s` - Gaussian splitting scale (0 for no splitting)
    ///
    /// # Returns
    /// Potential grid (host memory, nx*ny*nz)
    pub fn solve(&mut self, rho: &[f64], g_constant: f64, r_s: f64) -> Result<Vec<f64>, String> {
        let n_real = self.nx * self.ny * self.nz;

        if rho.len() != n_real {
            return Err(format!("Input size mismatch: {} vs {}", rho.len(), n_real));
        }

        // Copy density to GPU
        let ret = unsafe {
            cufft_copy_h2d(
                self.d_rho as *mut c_void,
                rho.as_ptr() as *const c_void,
                n_real * 8,
            )
        };
        if ret != 0 {
            return Err("H2D copy failed".to_string());
        }

        // Forward FFT: rho -> rho_k
        let ret = unsafe { cufft_exec_r2c(self.d_rho, self.d_rho_k) };
        if ret != 0 {
            return Err("Forward FFT failed".to_string());
        }

        // Apply Green's function in k-space
        let ret = unsafe {
            cufft_apply_green(
                self.d_rho_k,
                self.nx as i32, self.ny as i32, self.nz as i32,
                self.box_size, g_constant, r_s,
            )
        };
        if ret != 0 {
            return Err("Green's function failed".to_string());
        }

        // Inverse FFT: phi_k -> phi
        let ret = unsafe { cufft_exec_c2r(self.d_rho_k, self.d_phi) };
        if ret != 0 {
            return Err("Inverse FFT failed".to_string());
        }

        // Normalize (cuFFT doesn't normalize)
        let ret = unsafe {
            cufft_normalize(self.d_phi, self.nx as i32, self.ny as i32, self.nz as i32)
        };
        if ret != 0 {
            return Err("Normalization failed".to_string());
        }

        // Copy result back to host
        let mut phi = vec![0.0f64; n_real];
        let ret = unsafe {
            cufft_copy_d2h(
                phi.as_mut_ptr() as *mut c_void,
                self.d_phi as *const c_void,
                n_real * 8,
            )
        };
        if ret != 0 {
            return Err("D2H copy failed".to_string());
        }

        Ok(phi)
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let n_real = self.nx * self.ny * self.nz;
        let n_complex = self.nx * self.ny * (self.nz / 2 + 1);
        n_real * 8 * 2 + n_complex * 16  // 2 real buffers + 1 complex buffer
    }
}

/// Solve Poisson equation directly on device pointers (no host transfers)
///
/// This is the fast path for full GPU TreePM - avoids all CPU involvement.
///
/// # Safety
/// Caller must ensure d_rho and d_phi are valid device pointers to
/// grid_size³ f64 arrays.
#[cfg(feature = "cufft")]
pub unsafe fn solve_device(
    d_rho: *mut f64,
    d_phi: *mut f64,
    grid_size: usize,
    box_size: f64,
    g_constant: f64,
    r_s: f64,
) -> Result<(), String> {
    let ret = cufft_solve_device(
        d_rho, d_phi,
        grid_size as i32, grid_size as i32, grid_size as i32,
        box_size, g_constant, r_s,
    );
    if ret != 0 {
        return Err("cufft_solve_device failed".to_string());
    }
    Ok(())
}

impl Drop for CuFFTPoisson {
    fn drop(&mut self) {
        unsafe {
            if !self.d_rho.is_null() {
                cufft_free(self.d_rho as *mut c_void);
            }
            if !self.d_rho_k.is_null() {
                cufft_free(self.d_rho_k);
            }
            if !self.d_phi.is_null() {
                cufft_free(self.d_phi as *mut c_void);
            }
            cufft_cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]  // Requires GPU
    fn test_cufft_init() {
        let solver = CuFFTPoisson::new(64, 100.0);
        assert!(solver.is_ok());
    }

    #[test]
    #[ignore]  // Requires GPU
    fn test_cufft_solve() {
        let mut solver = CuFFTPoisson::new(32, 100.0).unwrap();

        // Simple test: single point mass at center
        let n = 32 * 32 * 32;
        let mut rho = vec![0.0f64; n];
        let center = 16 + 32 * (16 + 32 * 16);
        rho[center] = 1.0;

        let phi = solver.solve(&rho, 1.0, 0.0).unwrap();

        // Potential at center should be negative (self-potential)
        assert!(phi[center] < 0.0);
        println!("phi[center] = {}", phi[center]);
    }
}
