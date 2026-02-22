//! Leapfrog integrator for N-body simulation
//!
//! Kick-Drift-Kick (KDK) scheme for symplectic integration.

use crate::cic::{cic_deposit, GridStats};
use crate::poisson::{PoissonSolver, interpolate_force, interpolate_scalar};

/// Particle state
#[derive(Clone)]
pub struct Particle {
    pub pos: (f64, f64, f64),
    pub vel: (f32, f32, f32),
    pub mass: f32,
}

/// PM simulation state
pub struct PMSimulation {
    pub particles: Vec<Particle>,
    pub solver: PoissonSolver,
    pub box_size: f64,
    pub dt: f32,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    // Scratch buffers
    rho: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
    gz: Vec<f32>,
}

impl PMSimulation {
    pub fn new(
        particles: Vec<Particle>,
        nx: usize,
        ny: usize,
        nz: usize,
        box_size: f64,
        dt: f32,
    ) -> Result<Self, String> {
        let solver = PoissonSolver::new(nx, ny, nz, box_size as f32)?;
        let n_cells = nx * ny * nz;

        Ok(Self {
            particles,
            solver,
            box_size,
            dt,
            nx,
            ny,
            nz,
            rho: vec![0.0; n_cells],
            gx: vec![0.0; n_cells],
            gy: vec![0.0; n_cells],
            gz: vec![0.0; n_cells],
        })
    }

    /// Compute forces on all particles
    pub fn compute_forces(&mut self) -> Result<(), String> {
        // Deposit masses onto grid
        let positions: Vec<(f64, f64, f64)> = self.particles.iter()
            .map(|p| p.pos)
            .collect();
        let masses: Vec<f32> = self.particles.iter()
            .map(|p| p.mass)
            .collect();

        cic_deposit(
            &positions,
            &masses,
            &mut self.rho,
            self.nx, self.ny, self.nz,
            self.box_size,
        );

        // Convert mass to density: ρ = mass / cell_volume
        let cell_volume = (self.box_size / self.nx as f64).powi(3) as f32;
        for r in &mut self.rho {
            *r /= cell_volume;
        }

        // Solve Poisson equation
        let (gx, gy, gz) = self.solver.solve(&self.rho)?;
        self.gx = gx;
        self.gy = gy;
        self.gz = gz;

        Ok(())
    }

    /// Get force on a single particle (CIC interpolation)
    pub fn get_force(&self, p: &Particle) -> (f32, f32, f32) {
        interpolate_force(
            p.pos,
            &self.gx, &self.gy, &self.gz,
            self.nx, self.ny, self.nz,
            self.box_size,
        )
    }

    /// Kick step: update velocities by half timestep
    pub fn kick(&mut self, dt_half: f32) {
        // Pre-compute all forces to avoid borrow issues
        let forces: Vec<(f32, f32, f32)> = self.particles.iter()
            .map(|p| interpolate_force(
                p.pos,
                &self.gx, &self.gy, &self.gz,
                self.nx, self.ny, self.nz,
                self.box_size,
            ))
            .collect();

        // Apply forces
        for (p, (fx, fy, fz)) in self.particles.iter_mut().zip(forces.iter()) {
            // v += a * dt/2, where a = F/m = F (unit mass)
            p.vel.0 += fx * dt_half;
            p.vel.1 += fy * dt_half;
            p.vel.2 += fz * dt_half;
        }
    }

    /// Drift step: update positions by full timestep
    pub fn drift(&mut self, dt: f32) {
        let box_size = self.box_size;
        for p in &mut self.particles {
            p.pos.0 += p.vel.0 as f64 * dt as f64;
            p.pos.1 += p.vel.1 as f64 * dt as f64;
            p.pos.2 += p.vel.2 as f64 * dt as f64;

            // Periodic boundary conditions
            p.pos.0 = p.pos.0.rem_euclid(box_size);
            p.pos.1 = p.pos.1.rem_euclid(box_size);
            p.pos.2 = p.pos.2.rem_euclid(box_size);
        }
    }

    /// Single KDK step
    pub fn step(&mut self) -> Result<(), String> {
        let dt = self.dt;
        let dt_half = dt / 2.0;

        // Kick (half)
        self.compute_forces()?;
        self.kick(dt_half);

        // Drift (full)
        self.drift(dt);

        // Kick (half)
        self.compute_forces()?;
        self.kick(dt_half);

        Ok(())
    }

    /// Compute total kinetic energy
    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter()
            .map(|p| {
                let v2 = (p.vel.0 * p.vel.0 + p.vel.1 * p.vel.1 + p.vel.2 * p.vel.2) as f64;
                0.5 * p.mass as f64 * v2
            })
            .sum()
    }

    /// Compute total potential energy
    /// PE = 0.5 * Σ_i m_i * φ(x_i) summed over particles
    pub fn potential_energy(&self) -> f64 {
        // Recompute potential grid from current density
        let n = self.nx * self.ny * self.nz;

        // Convert density to complex
        let mut rho_k: Vec<crate::cufft_ffi::CufftComplex> = self.rho.iter()
            .map(|&r| crate::cufft_ffi::CufftComplex::new(r, 0.0))
            .collect();

        // Forward FFT
        if self.solver.fft.forward(&mut rho_k).is_err() {
            return 0.0;
        }

        // Apply Green's function to get φ̂
        self.solver.green.apply(&mut rho_k);

        // Inverse FFT to get φ
        if self.solver.fft.inverse(&mut rho_k).is_err() {
            return 0.0;
        }

        // Normalize
        let norm = 1.0 / n as f32;
        let phi: Vec<f32> = rho_k.iter().map(|c| c.x * norm).collect();

        // PE = 0.5 * Σ_i m_i * φ(x_i) using CIC interpolation
        let pe: f64 = self.particles.iter()
            .map(|p| {
                let phi_at_p = interpolate_scalar(
                    p.pos,
                    &phi,
                    self.nx, self.ny, self.nz,
                    self.box_size,
                );
                0.5 * p.mass as f64 * phi_at_p as f64
            })
            .sum();

        pe
    }

    /// Total energy (KE + PE)
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }
}

/// Generate uniform random initial conditions
pub fn generate_uniform_ic(
    n_particles: usize,
    box_size: f64,
    velocity_dispersion: f32,
    seed: u64,
) -> Vec<Particle> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    let mut rng = StdRng::seed_from_u64(seed);

    (0..n_particles)
        .map(|_| {
            let pos = (
                rng.random::<f64>() * box_size,
                rng.random::<f64>() * box_size,
                rng.random::<f64>() * box_size,
            );

            // Gaussian velocity distribution
            let vel = (
                (rng.random::<f32>() - 0.5) * velocity_dispersion,
                (rng.random::<f32>() - 0.5) * velocity_dispersion,
                (rng.random::<f32>() - 0.5) * velocity_dispersion,
            );

            Particle { pos, vel, mass: 1.0 }
        })
        .collect()
}
