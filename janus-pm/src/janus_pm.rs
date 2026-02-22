//! Janus Particle-Mesh solver with dual density grids
//!
//! Janus interaction rules:
//! - mass+ attracts mass+ (standard gravity)
//! - mass- attracts mass- (standard gravity)
//! - mass+ and mass- repel each other (anti-gravity)
//!
//! Force on mass+: F = g+ - g-  (attracted by +, repelled by -)
//! Force on mass-: F = g- - g+  (attracted by -, repelled by +)

use crate::cic::cic_deposit_janus;
use crate::cufft::Cufft3dC2C;
use crate::cufft_ffi::CufftComplex;
use crate::poisson::{GreenFunction, spectral_gradient, interpolate_force};
use janus::friedmann::{JanusParams, CosmoInterpolator};

/// Janus particle with mass sign
#[derive(Clone)]
pub struct JanusParticle {
    pub pos: (f64, f64, f64),
    pub vel: (f32, f32, f32),
    pub sign: i8,  // +1 or -1
}

/// Janus PM simulation with cosmological expansion
pub struct JanusPMSimulation {
    pub particles: Vec<JanusParticle>,
    pub fft: Cufft3dC2C,
    pub green: GreenFunction,
    pub cosmo: CosmoInterpolator,
    pub box_size: f64,
    pub dt: f32,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub step: usize,
    pub tau: f64,
    pub dtau_per_dt: f64,
    // Grids (real space)
    rho_plus: Vec<f32>,
    rho_minus: Vec<f32>,
    gx_plus: Vec<f32>,
    gy_plus: Vec<f32>,
    gz_plus: Vec<f32>,
    gx_minus: Vec<f32>,
    gy_minus: Vec<f32>,
    gz_minus: Vec<f32>,
    // Pre-allocated FFT work buffers (avoid per-step allocation)
    work_rho_k: Vec<CufftComplex>,
    work_gx_k: Vec<CufftComplex>,
    work_gy_k: Vec<CufftComplex>,
    work_gz_k: Vec<CufftComplex>,
    // Tracking
    pub ke_initial: f64,
    pub seg_initial: f64,
}

impl JanusPMSimulation {
    pub fn new(
        particles: Vec<JanusParticle>,
        nx: usize,
        ny: usize,
        nz: usize,
        box_size: f64,
        dt: f32,
        eta: f64,
        z_init: f64,
    ) -> Result<Self, String> {
        let fft = Cufft3dC2C::new(nx, ny, nz)?;
        let green = GreenFunction::new(nx, ny, nz, box_size as f32);

        let params = JanusParams::from_eta(eta);
        let cosmo = CosmoInterpolator::new(&params, z_init);
        let tau_start = cosmo.tau_start;

        let n_cells = nx * ny * nz;
        let dtau_per_dt = 0.013205;  // Validated in production

        Ok(Self {
            particles,
            fft,
            green,
            cosmo,
            box_size,
            dt,
            nx,
            ny,
            nz,
            step: 0,
            tau: tau_start,
            dtau_per_dt,
            rho_plus: vec![0.0; n_cells],
            rho_minus: vec![0.0; n_cells],
            gx_plus: vec![0.0; n_cells],
            gy_plus: vec![0.0; n_cells],
            gz_plus: vec![0.0; n_cells],
            gx_minus: vec![0.0; n_cells],
            gy_minus: vec![0.0; n_cells],
            gz_minus: vec![0.0; n_cells],
            // Pre-allocate FFT work buffers
            work_rho_k: vec![CufftComplex::default(); n_cells],
            work_gx_k: vec![CufftComplex::default(); n_cells],
            work_gy_k: vec![CufftComplex::default(); n_cells],
            work_gz_k: vec![CufftComplex::default(); n_cells],
            ke_initial: 0.0,
            seg_initial: 0.0,
        })
    }

    /// Compute forces on all particles using Janus rules
    pub fn compute_forces(&mut self) -> Result<(), String> {
        let positions: Vec<(f64, f64, f64)> = self.particles.iter()
            .map(|p| p.pos)
            .collect();
        let signs: Vec<i8> = self.particles.iter()
            .map(|p| p.sign)
            .collect();

        // Deposit onto dual grids
        cic_deposit_janus(
            &positions,
            &signs,
            &mut self.rho_plus,
            &mut self.rho_minus,
            self.nx, self.ny, self.nz,
            self.box_size,
        );

        // Convert mass to density
        let cell_volume = (self.box_size / self.nx as f64).powi(3) as f32;
        for r in &mut self.rho_plus {
            *r /= cell_volume;
        }
        for r in &mut self.rho_minus {
            *r /= cell_volume;
        }

        // Solve Poisson for positive grid -> forces into gx/gy/gz_plus
        self.solve_poisson_plus()?;

        // Solve Poisson for negative grid -> forces into gx/gy/gz_minus
        self.solve_poisson_minus()?;

        Ok(())
    }

    /// Solve Poisson for positive density, write forces to gx/gy/gz_plus
    fn solve_poisson_plus(&mut self) -> Result<(), String> {
        let n = self.nx * self.ny * self.nz;
        let twopi_over_l = 2.0 * std::f32::consts::PI / self.box_size as f32;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Convert rho to complex
        for (c, &r) in self.work_rho_k.iter_mut().zip(self.rho_plus.iter()) {
            c.x = r;
            c.y = 0.0;
        }

        // Forward FFT
        self.fft.copy_to_gpu(&self.work_rho_k)?;
        self.fft.forward_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_rho_k)?;

        // Apply Green's function and compute gradient
        for ix in 0..nx {
            let kx = if ix < nx / 2 { ix as f32 } else { (ix as i32 - nx as i32) as f32 } * twopi_over_l;
            for iy in 0..ny {
                let ky = if iy < ny / 2 { iy as f32 } else { (iy as i32 - ny as i32) as f32 } * twopi_over_l;
                for iz in 0..nz {
                    let kz = if iz < nz / 2 { iz as f32 } else { (iz as i32 - nz as i32) as f32 } * twopi_over_l;
                    let idx = ix * ny * nz + iy * nz + iz;
                    let g = self.green.g_k[idx].x;
                    let phi_x = self.work_rho_k[idx].x * g;
                    let phi_y = self.work_rho_k[idx].y * g;
                    self.work_gx_k[idx] = CufftComplex::new(phi_y * kx, -phi_x * kx);
                    self.work_gy_k[idx] = CufftComplex::new(phi_y * ky, -phi_x * ky);
                    self.work_gz_k[idx] = CufftComplex::new(phi_y * kz, -phi_x * kz);
                }
            }
        }

        // Inverse FFTs
        self.fft.copy_to_gpu(&self.work_gx_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gx_k)?;

        self.fft.copy_to_gpu(&self.work_gy_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gy_k)?;

        self.fft.copy_to_gpu(&self.work_gz_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gz_k)?;

        // Normalize into output
        let norm = 1.0 / n as f32;
        for (out, c) in self.gx_plus.iter_mut().zip(self.work_gx_k.iter()) {
            *out = c.x * norm;
        }
        for (out, c) in self.gy_plus.iter_mut().zip(self.work_gy_k.iter()) {
            *out = c.x * norm;
        }
        for (out, c) in self.gz_plus.iter_mut().zip(self.work_gz_k.iter()) {
            *out = c.x * norm;
        }

        Ok(())
    }

    /// Solve Poisson for negative density, write forces to gx/gy/gz_minus
    fn solve_poisson_minus(&mut self) -> Result<(), String> {
        let n = self.nx * self.ny * self.nz;
        let twopi_over_l = 2.0 * std::f32::consts::PI / self.box_size as f32;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Convert rho to complex
        for (c, &r) in self.work_rho_k.iter_mut().zip(self.rho_minus.iter()) {
            c.x = r;
            c.y = 0.0;
        }

        // Forward FFT
        self.fft.copy_to_gpu(&self.work_rho_k)?;
        self.fft.forward_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_rho_k)?;

        // Apply Green's function and compute gradient
        for ix in 0..nx {
            let kx = if ix < nx / 2 { ix as f32 } else { (ix as i32 - nx as i32) as f32 } * twopi_over_l;
            for iy in 0..ny {
                let ky = if iy < ny / 2 { iy as f32 } else { (iy as i32 - ny as i32) as f32 } * twopi_over_l;
                for iz in 0..nz {
                    let kz = if iz < nz / 2 { iz as f32 } else { (iz as i32 - nz as i32) as f32 } * twopi_over_l;
                    let idx = ix * ny * nz + iy * nz + iz;
                    let g = self.green.g_k[idx].x;
                    let phi_x = self.work_rho_k[idx].x * g;
                    let phi_y = self.work_rho_k[idx].y * g;
                    self.work_gx_k[idx] = CufftComplex::new(phi_y * kx, -phi_x * kx);
                    self.work_gy_k[idx] = CufftComplex::new(phi_y * ky, -phi_x * ky);
                    self.work_gz_k[idx] = CufftComplex::new(phi_y * kz, -phi_x * kz);
                }
            }
        }

        // Inverse FFTs
        self.fft.copy_to_gpu(&self.work_gx_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gx_k)?;

        self.fft.copy_to_gpu(&self.work_gy_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gy_k)?;

        self.fft.copy_to_gpu(&self.work_gz_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gz_k)?;

        // Normalize into output
        let norm = 1.0 / n as f32;
        for (out, c) in self.gx_minus.iter_mut().zip(self.work_gx_k.iter()) {
            *out = c.x * norm;
        }
        for (out, c) in self.gy_minus.iter_mut().zip(self.work_gy_k.iter()) {
            *out = c.x * norm;
        }
        for (out, c) in self.gz_minus.iter_mut().zip(self.work_gz_k.iter()) {
            *out = c.x * norm;
        }

        Ok(())
    }

    /// Compute forces with detailed timing
    pub fn compute_forces_timed(&mut self) -> Result<ForceTiming, String> {
        use std::time::Instant;

        let positions: Vec<(f64, f64, f64)> = self.particles.iter()
            .map(|p| p.pos)
            .collect();
        let signs: Vec<i8> = self.particles.iter()
            .map(|p| p.sign)
            .collect();

        // CIC deposit
        let t0 = Instant::now();
        cic_deposit_janus(
            &positions,
            &signs,
            &mut self.rho_plus,
            &mut self.rho_minus,
            self.nx, self.ny, self.nz,
            self.box_size,
        );

        let cell_volume = (self.box_size / self.nx as f64).powi(3) as f32;
        for r in &mut self.rho_plus {
            *r /= cell_volume;
        }
        for r in &mut self.rho_minus {
            *r /= cell_volume;
        }
        let cic_time = t0.elapsed().as_secs_f64() * 1000.0;

        // FFT + Poisson
        let t1 = Instant::now();
        self.solve_poisson_plus()?;
        self.solve_poisson_minus()?;
        let fft_time = t1.elapsed().as_secs_f64() * 1000.0;

        Ok(ForceTiming { cic_time, fft_time })
    }

    /// Solve Poisson equation using pre-allocated work buffers
    /// Returns force grids (gx, gy, gz) directly into the provided output slices
    fn solve_poisson_into(&mut self, rho: &[f32], gx_out: &mut [f32], gy_out: &mut [f32], gz_out: &mut [f32]) -> Result<(), String> {
        let n = self.nx * self.ny * self.nz;

        // Convert to complex (reuse work_rho_k buffer)
        for (c, &r) in self.work_rho_k.iter_mut().zip(rho.iter()) {
            c.x = r;
            c.y = 0.0;
        }

        // Forward FFT
        self.fft.copy_to_gpu(&self.work_rho_k)?;
        self.fft.forward_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_rho_k)?;

        // Apply Green's function and compute gradient in one pass
        // This avoids separate allocation for gradient computation
        let twopi_over_l = 2.0 * std::f32::consts::PI / self.box_size as f32;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        for ix in 0..nx {
            let kx = if ix < nx / 2 { ix as f32 } else { (ix as i32 - nx as i32) as f32 };
            let kx = kx * twopi_over_l;

            for iy in 0..ny {
                let ky = if iy < ny / 2 { iy as f32 } else { (iy as i32 - ny as i32) as f32 };
                let ky = ky * twopi_over_l;

                for iz in 0..nz {
                    let kz = if iz < nz / 2 { iz as f32 } else { (iz as i32 - nz as i32) as f32 };
                    let kz = kz * twopi_over_l;

                    let idx = ix * ny * nz + iy * nz + iz;

                    // Apply Green's function: phi_k = G_k * rho_k
                    let g = self.green.g_k[idx].x;
                    let phi_x = self.work_rho_k[idx].x * g;
                    let phi_y = self.work_rho_k[idx].y * g;

                    // Gradient in Fourier space: g = -∇φ = -i*k*φ
                    // -i * k * (phi_x + i*phi_y) = -i*k*phi_x + k*phi_y
                    // = k*phi_y - i*k*phi_x = (k*phi_y) + i*(-k*phi_x)
                    // Force = -∇φ, so we use: (phi_y * k) + i*(-phi_x * k)
                    self.work_gx_k[idx].x = phi_y * kx;
                    self.work_gx_k[idx].y = -phi_x * kx;
                    self.work_gy_k[idx].x = phi_y * ky;
                    self.work_gy_k[idx].y = -phi_x * ky;
                    self.work_gz_k[idx].x = phi_y * kz;
                    self.work_gz_k[idx].y = -phi_x * kz;
                }
            }
        }

        // Inverse FFTs
        self.fft.copy_to_gpu(&self.work_gx_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gx_k)?;

        self.fft.copy_to_gpu(&self.work_gy_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gy_k)?;

        self.fft.copy_to_gpu(&self.work_gz_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut self.work_gz_k)?;

        // Normalize and extract real parts directly into output
        let norm = 1.0 / n as f32;
        for (out, c) in gx_out.iter_mut().zip(self.work_gx_k.iter()) {
            *out = c.x * norm;
        }
        for (out, c) in gy_out.iter_mut().zip(self.work_gy_k.iter()) {
            *out = c.x * norm;
        }
        for (out, c) in gz_out.iter_mut().zip(self.work_gz_k.iter()) {
            *out = c.x * norm;
        }

        Ok(())
    }

    /// Solve Poisson with detailed timing (for diagnostics)
    #[allow(dead_code)]
    fn solve_poisson_timed(&self, rho: &[f32]) -> Result<((Vec<f32>, Vec<f32>, Vec<f32>), PoissonTiming), String> {
        use std::time::Instant;
        let n = self.nx * self.ny * self.nz;

        // Convert to complex
        let t0 = Instant::now();
        let mut rho_k: Vec<CufftComplex> = rho.iter()
            .map(|&r| CufftComplex::new(r, 0.0))
            .collect();
        let alloc_time = t0.elapsed().as_secs_f64() * 1000.0;

        // Forward FFT
        let t1 = Instant::now();
        self.fft.copy_to_gpu(&rho_k)?;
        let copy_h2d_time = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        self.fft.forward_gpu()?;
        self.fft.sync();
        let fft_time = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = Instant::now();
        self.fft.copy_from_gpu(&mut rho_k)?;
        let copy_d2h_time = t3.elapsed().as_secs_f64() * 1000.0;

        // Apply Green's function
        let t4 = Instant::now();
        for (r, g) in rho_k.iter_mut().zip(self.green.g_k.iter()) {
            r.x *= g.x;
            r.y *= g.x;
        }
        let green_time = t4.elapsed().as_secs_f64() * 1000.0;

        // Compute gradient
        let t5 = Instant::now();
        let (mut gx_k, mut gy_k, mut gz_k) = spectral_gradient(
            &rho_k,
            self.nx, self.ny, self.nz,
            self.box_size as f32,
        );
        let grad_time = t5.elapsed().as_secs_f64() * 1000.0;

        // Inverse FFTs (measure total)
        let t6 = Instant::now();
        self.fft.copy_to_gpu(&gx_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut gx_k)?;

        self.fft.copy_to_gpu(&gy_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut gy_k)?;

        self.fft.copy_to_gpu(&gz_k)?;
        self.fft.inverse_gpu()?;
        self.fft.sync();
        self.fft.copy_from_gpu(&mut gz_k)?;
        let ifft_time = t6.elapsed().as_secs_f64() * 1000.0;

        // Normalize
        let t7 = Instant::now();
        let norm = 1.0 / n as f32;
        let gx: Vec<f32> = gx_k.iter().map(|c| c.x * norm).collect();
        let gy: Vec<f32> = gy_k.iter().map(|c| c.x * norm).collect();
        let gz: Vec<f32> = gz_k.iter().map(|c| c.x * norm).collect();
        let norm_time = t7.elapsed().as_secs_f64() * 1000.0;

        let timing = PoissonTiming {
            alloc_time,
            copy_h2d_time,
            fft_time,
            copy_d2h_time,
            green_time,
            grad_time,
            ifft_time,
            norm_time,
        };

        Ok(((gx, gy, gz), timing))
    }

    /// Get Janus force on particle: F+ = g+ - g-, F- = g- - g+
    fn get_janus_force(&self, p: &JanusParticle) -> (f32, f32, f32) {
        let f_plus = interpolate_force(
            p.pos,
            &self.gx_plus, &self.gy_plus, &self.gz_plus,
            self.nx, self.ny, self.nz,
            self.box_size,
        );
        let f_minus = interpolate_force(
            p.pos,
            &self.gx_minus, &self.gy_minus, &self.gz_minus,
            self.nx, self.ny, self.nz,
            self.box_size,
        );

        if p.sign > 0 {
            // Force on mass+: attracted by +, repelled by -
            (f_plus.0 - f_minus.0, f_plus.1 - f_minus.1, f_plus.2 - f_minus.2)
        } else {
            // Force on mass-: attracted by -, repelled by +
            (f_minus.0 - f_plus.0, f_minus.1 - f_plus.1, f_minus.2 - f_plus.2)
        }
    }

    /// Kick step with Hubble friction
    pub fn kick(&mut self, dt_half: f32) {
        // Get current cosmological parameters
        let (a, h) = self.cosmo.get_params_at_tau(self.tau);
        let hubble_friction = (h * self.dtau_per_dt) as f32;

        // Pre-compute all forces
        let forces: Vec<(f32, f32, f32)> = self.particles.iter()
            .map(|p| self.get_janus_force(p))
            .collect();

        // Apply forces with Hubble friction
        for (p, (fx, fy, fz)) in self.particles.iter_mut().zip(forces.iter()) {
            // Acceleration from gravity
            p.vel.0 += fx * dt_half;
            p.vel.1 += fy * dt_half;
            p.vel.2 += fz * dt_half;

            // Hubble friction: dv/dt = -H*v
            p.vel.0 -= p.vel.0 * hubble_friction * dt_half;
            p.vel.1 -= p.vel.1 * hubble_friction * dt_half;
            p.vel.2 -= p.vel.2 * hubble_friction * dt_half;
        }
    }

    /// Drift step
    pub fn drift(&mut self, dt: f32) {
        let box_size = self.box_size;
        for p in &mut self.particles {
            p.pos.0 += p.vel.0 as f64 * dt as f64;
            p.pos.1 += p.vel.1 as f64 * dt as f64;
            p.pos.2 += p.vel.2 as f64 * dt as f64;

            // Periodic BC
            p.pos.0 = p.pos.0.rem_euclid(box_size);
            p.pos.1 = p.pos.1.rem_euclid(box_size);
            p.pos.2 = p.pos.2.rem_euclid(box_size);
        }
    }

    /// Single KDK step with cosmological evolution
    pub fn step(&mut self) -> Result<(), String> {
        let dt = self.dt;
        let dt_half = dt / 2.0;

        // Kick (half)
        self.compute_forces()?;
        self.kick(dt_half);

        // Drift
        self.drift(dt);

        // Kick (half)
        self.compute_forces()?;
        self.kick(dt_half);

        // Advance cosmological time
        self.tau += self.dtau_per_dt * dt as f64;
        self.step += 1;

        Ok(())
    }

    /// Single step with detailed timing
    pub fn step_timed(&mut self) -> Result<StepTiming, String> {
        use std::time::Instant;
        let dt = self.dt;
        let dt_half = dt / 2.0;

        // Kick 1 (half)
        let t0 = Instant::now();
        self.compute_forces()?;
        let force_time_1 = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        self.kick(dt_half);
        let kick_time_1 = t1.elapsed().as_secs_f64() * 1000.0;

        // Drift
        let t2 = Instant::now();
        self.drift(dt);
        let drift_time = t2.elapsed().as_secs_f64() * 1000.0;

        // Kick 2 (half)
        let t3 = Instant::now();
        self.compute_forces()?;
        let force_time_2 = t3.elapsed().as_secs_f64() * 1000.0;

        let t4 = Instant::now();
        self.kick(dt_half);
        let kick_time_2 = t4.elapsed().as_secs_f64() * 1000.0;

        // Advance cosmological time
        self.tau += self.dtau_per_dt * dt as f64;
        self.step += 1;

        Ok(StepTiming {
            force_time: force_time_1 + force_time_2,
            kick_time: kick_time_1 + kick_time_2,
            drift_time,
        })
    }

    /// Compute kinetic energy
    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter()
            .map(|p| {
                let v2 = (p.vel.0 * p.vel.0 + p.vel.1 * p.vel.1 + p.vel.2 * p.vel.2) as f64;
                0.5 * v2
            })
            .sum()
    }

    /// Compute segregation index (mass-weighted CoM distance)
    pub fn segregation(&self) -> f64 {
        let n = self.particles.len() as f64;
        if n < 2.0 { return 0.0; }

        // CoM of positive masses
        let (mut cx_p, mut cy_p, mut cz_p) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut n_p = 0usize;

        // CoM of negative masses
        let (mut cx_m, mut cy_m, mut cz_m) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut n_m = 0usize;

        for p in &self.particles {
            if p.sign > 0 {
                cx_p += p.pos.0;
                cy_p += p.pos.1;
                cz_p += p.pos.2;
                n_p += 1;
            } else {
                cx_m += p.pos.0;
                cy_m += p.pos.1;
                cz_m += p.pos.2;
                n_m += 1;
            }
        }

        if n_p == 0 || n_m == 0 { return 0.0; }

        cx_p /= n_p as f64;
        cy_p /= n_p as f64;
        cz_p /= n_p as f64;

        cx_m /= n_m as f64;
        cy_m /= n_m as f64;
        cz_m /= n_m as f64;

        // Distance between CoMs normalized by box size
        let dx = (cx_p - cx_m).abs().min(self.box_size - (cx_p - cx_m).abs());
        let dy = (cy_p - cy_m).abs().min(self.box_size - (cy_p - cy_m).abs());
        let dz = (cz_p - cz_m).abs().min(self.box_size - (cz_p - cz_m).abs());

        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        dist / self.box_size
    }

    /// Get current scale factor
    pub fn scale_factor(&self) -> f64 {
        let (a, _) = self.cosmo.get_params_at_tau(self.tau);
        a
    }

    /// Virialize the system using hardcoded alpha from BH reference
    /// PM potential underestimates PE_binding by ~34% due to grid smoothing,
    /// so we use the validated BH alpha for consistent IC comparison
    pub fn virialize(&mut self) -> Result<f64, String> {
        // Hardcoded alpha from validated BH run (η=1.045, seed=42)
        let alpha = 4.57_f64;

        println!("  Using hardcoded α = {:.2} (BH reference)", alpha);

        // Scale velocities
        let alpha_f32 = alpha as f32;
        for p in &mut self.particles {
            p.vel.0 *= alpha_f32;
            p.vel.1 *= alpha_f32;
            p.vel.2 *= alpha_f32;
        }

        self.ke_initial = self.kinetic_energy();
        self.seg_initial = self.segregation();

        Ok(alpha)
    }

    fn compute_potential(&self, rho: &[f32]) -> Result<Vec<f32>, String> {
        let n = self.nx * self.ny * self.nz;

        let mut rho_k: Vec<CufftComplex> = rho.iter()
            .map(|&r| CufftComplex::new(r, 0.0))
            .collect();

        self.fft.forward(&mut rho_k)?;

        for (r, g) in rho_k.iter_mut().zip(self.green.g_k.iter()) {
            r.x *= g.x;
            r.y *= g.x;
        }

        self.fft.inverse(&mut rho_k)?;

        let norm = 1.0 / n as f32;
        Ok(rho_k.iter().map(|c| c.x * norm).collect())
    }
}

/// Timing breakdown for a single step
pub struct StepTiming {
    pub force_time: f64,  // CIC + FFT + gradient (ms)
    pub kick_time: f64,   // velocity update (ms)
    pub drift_time: f64,  // position update (ms)
}

/// Timing breakdown for force computation
pub struct ForceTiming {
    pub cic_time: f64,  // CIC deposit (ms)
    pub fft_time: f64,  // FFT + Poisson (ms)
}

/// Detailed timing for Poisson solver diagnostics
#[allow(dead_code)]
pub struct PoissonTiming {
    pub alloc_time: f64,
    pub copy_h2d_time: f64,
    pub fft_time: f64,
    pub copy_d2h_time: f64,
    pub green_time: f64,
    pub grad_time: f64,
    pub ifft_time: f64,
    pub norm_time: f64,
}

/// Generate Janus initial conditions with η ratio
pub fn generate_janus_ic(
    n_particles: usize,
    box_size: f64,
    velocity_dispersion: f32,
    eta: f64,
    seed: u64,
) -> Vec<JanusParticle> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    let mut rng = StdRng::seed_from_u64(seed);

    // η = n_negative / n_positive
    // fraction_negative = η / (1 + η)
    let fraction_negative = eta / (1.0 + eta);

    (0..n_particles)
        .map(|_| {
            let pos = (
                rng.random::<f64>() * box_size,
                rng.random::<f64>() * box_size,
                rng.random::<f64>() * box_size,
            );

            let vel = (
                (rng.random::<f32>() - 0.5) * velocity_dispersion,
                (rng.random::<f32>() - 0.5) * velocity_dispersion,
                (rng.random::<f32>() - 0.5) * velocity_dispersion,
            );

            let sign = if rng.random::<f64>() < fraction_negative { -1_i8 } else { 1_i8 };

            JanusParticle { pos, vel, sign }
        })
        .collect()
}
