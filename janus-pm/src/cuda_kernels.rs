//! Safe Rust wrappers for CUDA PM kernels
//!
//! All kernel launches are asynchronous on the provided stream.
//! Call sync() before accessing results.

use crate::cufft_ffi::CufftComplex;
use std::ffi::c_void;
use std::os::raw::{c_double, c_float, c_int};

/// CUDA stream handle (opaque pointer)
pub type CudaStream = *mut c_void;

// FFI declarations for our custom kernels
#[link(name = "pm_kernels")]
extern "C" {
    fn launch_cic_deposit(
        pos_x: *const c_double,
        pos_y: *const c_double,
        pos_z: *const c_double,
        signs: *const i8,
        rho_plus: *mut c_float,
        rho_minus: *mut c_float,
        n_particles: c_int,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        box_size: c_float,
        stream: CudaStream,
    );

    fn launch_green_gradient(
        rho_k: *const CufftComplex,
        gx_k: *mut CufftComplex,
        gy_k: *mut CufftComplex,
        gz_k: *mut CufftComplex,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        dx: c_float,
        k_softening: c_float,
        stream: CudaStream,
    );

    fn launch_force_interpolation(
        pos_x: *const c_double,
        pos_y: *const c_double,
        pos_z: *const c_double,
        signs: *const i8,
        gx_plus: *const c_float,
        gy_plus: *const c_float,
        gz_plus: *const c_float,
        gx_minus: *const c_float,
        gy_minus: *const c_float,
        gz_minus: *const c_float,
        fx: *mut c_float,
        fy: *mut c_float,
        fz: *mut c_float,
        n_particles: c_int,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        box_size: c_float,
        stream: CudaStream,
    );

    fn launch_kick(
        vel_x: *mut c_float,
        vel_y: *mut c_float,
        vel_z: *mut c_float,
        fx: *const c_float,
        fy: *const c_float,
        fz: *const c_float,
        n_particles: c_int,
        dt: c_float,
        hubble_friction: c_float,
        stream: CudaStream,
    );

    fn launch_drift(
        pos_x: *mut c_double,
        pos_y: *mut c_double,
        pos_z: *mut c_double,
        vel_x: *const c_float,
        vel_y: *const c_float,
        vel_z: *const c_float,
        n_particles: c_int,
        dt: c_float,
        box_size: c_double,
        stream: CudaStream,
    );

    fn launch_zero_float(arr: *mut c_float, n: c_int, stream: CudaStream);

    fn launch_scale_velocities(
        vel_x: *mut c_float,
        vel_y: *mut c_float,
        vel_z: *mut c_float,
        n_particles: c_int,
        factor: c_float,
        stream: CudaStream,
    );

    fn launch_kinetic_energy(
        vel_x: *const c_float,
        vel_y: *const c_float,
        vel_z: *const c_float,
        partial_sums: *mut c_float,
        n_particles: c_int,
        n_blocks: c_int,
        stream: CudaStream,
    );

    fn launch_real_to_complex(
        real_in: *const c_float,
        complex_out: *mut CufftComplex,
        n: c_int,
        stream: CudaStream,
    );

    fn launch_complex_to_real(
        complex_in: *const CufftComplex,
        real_out: *mut c_float,
        n: c_int,
        norm: c_float,
        stream: CudaStream,
    );

    fn launch_segregation(
        pos_x: *const c_double,
        pos_y: *const c_double,
        pos_z: *const c_double,
        signs: *const i8,
        sum_pos: *mut c_double,
        sum_neg: *mut c_double,
        n_particles: c_int,
        n_blocks: c_int,
        box_size: c_double,
        stream: CudaStream,
    );
}

// CUDA runtime FFI
mod cuda_rt {
    use super::*;

    pub type CudaError = c_int;
    pub const CUDA_SUCCESS: CudaError = 0;

    #[link(name = "cudart")]
    extern "C" {
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: u64) -> CudaError;
        pub fn cudaFree(devPtr: *mut c_void) -> CudaError;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: u64, kind: c_int) -> CudaError;
        pub fn cudaDeviceSynchronize() -> CudaError;
        pub fn cudaStreamCreate(pStream: *mut CudaStream) -> CudaError;
        pub fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
        pub fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
        pub fn cudaGetLastError() -> CudaError;
        pub fn cudaGetErrorString(error: CudaError) -> *const i8;
    }

    pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
}

/// Check CUDA error and return Result
fn check_cuda(err: cuda_rt::CudaError, context: &str) -> Result<(), String> {
    if err != cuda_rt::CUDA_SUCCESS {
        let msg = unsafe {
            let ptr = cuda_rt::cudaGetErrorString(err);
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        Err(format!("{}: {} ({})", context, msg, err))
    } else {
        Ok(())
    }
}

/// GPU memory buffer with automatic cleanup
pub struct GpuBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> GpuBuffer<T> {
    /// Allocate GPU memory for n elements
    pub fn new(n: usize) -> Result<Self, String> {
        let size = n * std::mem::size_of::<T>();
        let mut ptr: *mut T = std::ptr::null_mut();

        let err = unsafe {
            cuda_rt::cudaMalloc(
                &mut ptr as *mut *mut T as *mut *mut c_void,
                size as u64,
            )
        };
        check_cuda(err, "cudaMalloc")?;

        Ok(Self { ptr, len: n })
    }

    /// Copy data from host to device
    pub fn copy_from_host(&self, data: &[T]) -> Result<(), String> {
        if data.len() != self.len {
            return Err(format!("Size mismatch: {} vs {}", data.len(), self.len));
        }
        let size = self.len * std::mem::size_of::<T>();
        let err = unsafe {
            cuda_rt::cudaMemcpy(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size as u64,
                cuda_rt::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        check_cuda(err, "cudaMemcpy H2D")
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), String> {
        if data.len() != self.len {
            return Err(format!("Size mismatch: {} vs {}", data.len(), self.len));
        }
        let size = self.len * std::mem::size_of::<T>();
        let err = unsafe {
            cuda_rt::cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size as u64,
                cuda_rt::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        check_cuda(err, "cudaMemcpy D2H")
    }

    /// Get raw pointer
    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_rt::cudaFree(self.ptr as *mut c_void) };
        }
    }
}

unsafe impl<T> Send for GpuBuffer<T> {}
unsafe impl<T> Sync for GpuBuffer<T> {}

/// CUDA stream wrapper with RAII
pub struct Stream {
    stream: CudaStream,
}

impl Stream {
    pub fn new() -> Result<Self, String> {
        let mut stream: CudaStream = std::ptr::null_mut();
        let err = unsafe { cuda_rt::cudaStreamCreate(&mut stream) };
        check_cuda(err, "cudaStreamCreate")?;
        Ok(Self { stream })
    }

    pub fn sync(&self) -> Result<(), String> {
        let err = unsafe { cuda_rt::cudaStreamSynchronize(self.stream) };
        check_cuda(err, "cudaStreamSynchronize")
    }

    pub fn handle(&self) -> CudaStream {
        self.stream
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self { stream: std::ptr::null_mut() }  // Default stream
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cuda_rt::cudaStreamDestroy(self.stream) };
        }
    }
}

/// Synchronize all GPU operations
pub fn device_sync() -> Result<(), String> {
    let err = unsafe { cuda_rt::cudaDeviceSynchronize() };
    check_cuda(err, "cudaDeviceSynchronize")
}

// ============================================================================
// Safe kernel wrappers
// ============================================================================

/// CIC deposit particles to density grids
pub fn cic_deposit(
    pos_x: &GpuBuffer<f64>,
    pos_y: &GpuBuffer<f64>,
    pos_z: &GpuBuffer<f64>,
    signs: &GpuBuffer<i8>,
    rho_plus: &GpuBuffer<f32>,
    rho_minus: &GpuBuffer<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f32,
    stream: &Stream,
) {
    let n = pos_x.len() as c_int;
    unsafe {
        launch_cic_deposit(
            pos_x.ptr(),
            pos_y.ptr(),
            pos_z.ptr(),
            signs.ptr(),
            rho_plus.ptr(),
            rho_minus.ptr(),
            n,
            nx as c_int,
            ny as c_int,
            nz as c_int,
            box_size,
            stream.handle(),
        );
    }
}

/// Apply Green's function and compute gradient in k-space
pub fn green_gradient(
    rho_k: &GpuBuffer<CufftComplex>,
    gx_k: &GpuBuffer<CufftComplex>,
    gy_k: &GpuBuffer<CufftComplex>,
    gz_k: &GpuBuffer<CufftComplex>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f32,
    k_softening: f32,
    stream: &Stream,
) {
    unsafe {
        launch_green_gradient(
            rho_k.ptr(),
            gx_k.ptr(),
            gy_k.ptr(),
            gz_k.ptr(),
            nx as c_int,
            ny as c_int,
            nz as c_int,
            dx,
            k_softening,
            stream.handle(),
        );
    }
}

/// Interpolate forces from grids to particles
pub fn force_interpolation(
    pos_x: &GpuBuffer<f64>,
    pos_y: &GpuBuffer<f64>,
    pos_z: &GpuBuffer<f64>,
    signs: &GpuBuffer<i8>,
    gx_plus: &GpuBuffer<f32>,
    gy_plus: &GpuBuffer<f32>,
    gz_plus: &GpuBuffer<f32>,
    gx_minus: &GpuBuffer<f32>,
    gy_minus: &GpuBuffer<f32>,
    gz_minus: &GpuBuffer<f32>,
    fx: &GpuBuffer<f32>,
    fy: &GpuBuffer<f32>,
    fz: &GpuBuffer<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: f32,
    stream: &Stream,
) {
    let n = pos_x.len() as c_int;
    unsafe {
        launch_force_interpolation(
            pos_x.ptr(),
            pos_y.ptr(),
            pos_z.ptr(),
            signs.ptr(),
            gx_plus.ptr(),
            gy_plus.ptr(),
            gz_plus.ptr(),
            gx_minus.ptr(),
            gy_minus.ptr(),
            gz_minus.ptr(),
            fx.ptr(),
            fy.ptr(),
            fz.ptr(),
            n,
            nx as c_int,
            ny as c_int,
            nz as c_int,
            box_size,
            stream.handle(),
        );
    }
}

/// Kick velocities: v += (F - H*v) * dt
pub fn kick(
    vel_x: &GpuBuffer<f32>,
    vel_y: &GpuBuffer<f32>,
    vel_z: &GpuBuffer<f32>,
    fx: &GpuBuffer<f32>,
    fy: &GpuBuffer<f32>,
    fz: &GpuBuffer<f32>,
    dt: f32,
    hubble_friction: f32,
    stream: &Stream,
) {
    let n = vel_x.len() as c_int;
    unsafe {
        launch_kick(
            vel_x.ptr(),
            vel_y.ptr(),
            vel_z.ptr(),
            fx.ptr(),
            fy.ptr(),
            fz.ptr(),
            n,
            dt,
            hubble_friction,
            stream.handle(),
        );
    }
}

/// Drift positions: x += v * dt (with periodic wrap)
pub fn drift(
    pos_x: &GpuBuffer<f64>,
    pos_y: &GpuBuffer<f64>,
    pos_z: &GpuBuffer<f64>,
    vel_x: &GpuBuffer<f32>,
    vel_y: &GpuBuffer<f32>,
    vel_z: &GpuBuffer<f32>,
    dt: f32,
    box_size: f64,
    stream: &Stream,
) {
    let n = pos_x.len() as c_int;
    unsafe {
        launch_drift(
            pos_x.ptr(),
            pos_y.ptr(),
            pos_z.ptr(),
            vel_x.ptr(),
            vel_y.ptr(),
            vel_z.ptr(),
            n,
            dt,
            box_size,
            stream.handle(),
        );
    }
}

/// Zero out a float array
pub fn zero_float(arr: &GpuBuffer<f32>, stream: &Stream) {
    let n = arr.len() as c_int;
    unsafe {
        launch_zero_float(arr.ptr(), n, stream.handle());
    }
}

/// Scale velocities by factor
pub fn scale_velocities(
    vel_x: &GpuBuffer<f32>,
    vel_y: &GpuBuffer<f32>,
    vel_z: &GpuBuffer<f32>,
    factor: f32,
    stream: &Stream,
) {
    let n = vel_x.len() as c_int;
    unsafe {
        launch_scale_velocities(
            vel_x.ptr(),
            vel_y.ptr(),
            vel_z.ptr(),
            n,
            factor,
            stream.handle(),
        );
    }
}

/// Compute kinetic energy (returns partial sums on GPU)
pub fn kinetic_energy_partial(
    vel_x: &GpuBuffer<f32>,
    vel_y: &GpuBuffer<f32>,
    vel_z: &GpuBuffer<f32>,
    partial_sums: &GpuBuffer<f32>,
    stream: &Stream,
) {
    let n = vel_x.len() as c_int;
    let n_blocks = partial_sums.len() as c_int;
    unsafe {
        launch_kinetic_energy(
            vel_x.ptr(),
            vel_y.ptr(),
            vel_z.ptr(),
            partial_sums.ptr(),
            n,
            n_blocks,
            stream.handle(),
        );
    }
}

/// Convert real f32 grid to complex (imaginary = 0)
pub fn real_to_complex(
    real_in: &GpuBuffer<f32>,
    complex_out: &GpuBuffer<CufftComplex>,
    stream: &Stream,
) {
    let n = real_in.len() as c_int;
    unsafe {
        launch_real_to_complex(
            real_in.ptr(),
            complex_out.ptr(),
            n,
            stream.handle(),
        );
    }
}

/// Convert complex to real f32 grid (extracts real part with normalization)
pub fn complex_to_real(
    complex_in: &GpuBuffer<CufftComplex>,
    real_out: &GpuBuffer<f32>,
    norm: f32,
    stream: &Stream,
) {
    let n = real_out.len() as c_int;
    unsafe {
        launch_complex_to_real(
            complex_in.ptr(),
            real_out.ptr(),
            n,
            norm,
            stream.handle(),
        );
    }
}

/// Compute segregation metric partial sums
pub fn segregation_partial(
    pos_x: &GpuBuffer<f64>,
    pos_y: &GpuBuffer<f64>,
    pos_z: &GpuBuffer<f64>,
    signs: &GpuBuffer<i8>,
    sum_pos: &GpuBuffer<f64>,
    sum_neg: &GpuBuffer<f64>,
    box_size: f64,
    stream: &Stream,
) {
    let n = pos_x.len() as c_int;
    let n_blocks = (sum_pos.len() / 4) as c_int;  // 4 values per block
    unsafe {
        launch_segregation(
            pos_x.ptr(),
            pos_y.ptr(),
            pos_z.ptr(),
            signs.ptr(),
            sum_pos.ptr(),
            sum_neg.ptr(),
            n,
            n_blocks,
            box_size,
            stream.handle(),
        );
    }
}
