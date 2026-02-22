//! Janus Particle-Mesh FFT Solver Library
//!
//! Target: 150M particles on RTX 3060 (GPU-only)

pub mod cufft_ffi;
pub mod cufft;
pub mod cic;
pub mod poisson;
pub mod integrator;
pub mod janus_pm;
pub mod snapshot;

// GPU-only modules for PM-5
pub mod cuda_kernels;
pub mod gpu_simulation;
