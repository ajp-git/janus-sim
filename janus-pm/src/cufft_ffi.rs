//! FFI bindings for cuFFT C2C (Complex-to-Complex) 3D transforms
//!
//! Links directly to libcufft.so from CUDA toolkit.
//! Only exposes the minimal API needed for PM solver.

use std::ffi::c_int;

/// cuFFT plan handle
pub type CufftHandle = c_int;

/// cuFFT result codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CufftResult {
    Success = 0x0,
    InvalidPlan = 0x1,
    AllocFailed = 0x2,
    InvalidType = 0x3,
    InvalidValue = 0x4,
    InternalError = 0x5,
    ExecFailed = 0x6,
    SetupFailed = 0x7,
    InvalidSize = 0x8,
    UnalignedData = 0x9,
}

/// cuFFT transform types
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CufftType {
    C2C = 0x29,  // Complex to Complex (single precision)
    R2C = 0x2a,  // Real to Complex
    C2R = 0x2c,  // Complex to Real
    Z2Z = 0x69,  // Complex to Complex (double precision)
}

/// Transform direction
pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

/// Complex number (single precision) - matches cuComplex / cufftComplex
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CufftComplex {
    pub x: f32,  // real part
    pub y: f32,  // imaginary part
}

impl CufftComplex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { x: re, y: im }
    }

    pub fn norm_sqr(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn norm(&self) -> f32 {
        self.norm_sqr().sqrt()
    }
}

impl std::ops::Sub for CufftComplex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl std::ops::Mul<f32> for CufftComplex {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs }
    }
}

impl std::ops::MulAssign<f32> for CufftComplex {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

// Link to libcufft.so
#[link(name = "cufft")]
extern "C" {
    /// Create a 3D FFT plan
    pub fn cufftPlan3d(
        plan: *mut CufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        fft_type: CufftType,
    ) -> CufftResult;

    /// Execute Complex-to-Complex transform (in-place or out-of-place)
    pub fn cufftExecC2C(
        plan: CufftHandle,
        idata: *mut CufftComplex,
        odata: *mut CufftComplex,
        direction: c_int,
    ) -> CufftResult;

    /// Destroy FFT plan and free resources
    pub fn cufftDestroy(plan: CufftHandle) -> CufftResult;
}

impl CufftResult {
    pub fn is_ok(&self) -> bool {
        *self == CufftResult::Success
    }

    pub fn check(self, msg: &str) -> Result<(), String> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(format!("{}: {:?}", msg, self))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_ops() {
        let a = CufftComplex::new(3.0, 4.0);
        assert!((a.norm() - 5.0).abs() < 1e-6);

        let b = CufftComplex::new(1.0, 2.0);
        let c = a - b;
        assert!((c.x - 2.0).abs() < 1e-6);
        assert!((c.y - 2.0).abs() < 1e-6);
    }
}
