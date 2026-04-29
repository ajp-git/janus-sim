//! Janus extension for TreePM: dual scale factors, cross-coupling, peculiar
//! drift/kick.
//!
//! References:
//! - Petit & d'Agostini 2014, 2018
//! - Petit, Margnat & Zejli 2024 EPJC (cross-coupling Φ = (ā/a)³)
//! - Plan §5
//!
//! Convention de stockage (issue de l'audit 05) :
//! - `pos[i]` : comoving Mpc, fixed box [-L/2, L/2]
//! - `vel[i]` : peculiar proper velocity v_pec = a · dx_co/dt   [Mpc/Gyr]
//! - `sign[i]` : i8, +1 (m+) or -1 (m-)
//! - `mass[i]` : f32 ABSOLUTE value (always > 0)
//!
//! Conversion BH→TreePM : le code BH stocke `signs: CudaSlice<i32>` avec ±1
//! et `masses` en f64 toujours positif. Le snapshot V3 utilise u8 (1 ou 255).
//! Au chargement, on convertit u8 (1, 255) → i8 (+1, -1).
//!
//! Equations of motion (Peebles peculiar, src/nbody_gpu.rs:3131-3138) :
//!   dx_co/dt   = v_pec / a_eff
//!   dv_pec/dt  = -H_eff · v_pec  +  acc_bare / a_eff²
//! où acc_bare est la sortie du kernel forces (sans facteur a).
//!
//! Cross-coupling Janus (src/nbody_gpu.rs:3162-3164) :
//!   cross_minus_plus = c̄² · φ⁻¹ · repulsion_scale  (force m- ← m+)
//!   cross_plus_minus = φ · repulsion_scale         (force m+ ← m-)
//! où φ = (ā/a)³ et c̄² = a/ā (Petit 2024).
//!
//! Convention force (de compute_forces_bvh, src/nbody_gpu.rs:982,1000) :
//!   F sur m+ depuis m+ contrib : factor = +1.0
//!   F sur m- depuis m+ contrib : factor = -cross_minus_plus
//!   F sur m+ depuis m- contrib : factor = -cross_plus_minus
//!   F sur m- depuis m- contrib : factor = +1.0

use super::gpu_layout::ParticleArrays;

/// Janus cross-coupling factors derived from φ and c̄².
#[derive(Debug, Clone, Copy)]
pub struct JanusCoupling {
    /// φ = (ā/a)³ — Petit 2024 cross-coupling for m+ force from m-
    pub phi: f64,
    /// c̄² = a/ā — VSL ratio squared
    pub c_ratio_sq: f64,
    /// Repulsion scale (1.0 for full Janus, 0.0 for no cross)
    pub repulsion_scale: f64,
}

impl JanusCoupling {
    /// Force factor: m- ← m+ = c̄²·φ⁻¹·repulsion_scale.
    /// Signed convention: actual force factor on m- from m+ contrib is
    /// -cross_minus_plus (repulsive).
    #[inline(always)]
    pub fn cross_minus_plus(&self) -> f64 {
        self.c_ratio_sq * (1.0 / self.phi) * self.repulsion_scale
    }

    /// Force factor: m+ ← m- = φ·repulsion_scale.
    /// Signed convention: actual force factor on m+ from m- contrib is
    /// -cross_plus_minus (repulsive).
    #[inline(always)]
    pub fn cross_plus_minus(&self) -> f64 {
        self.phi * self.repulsion_scale
    }
}

/// Janus dual scale-factor cosmology state.
#[derive(Debug, Clone, Copy)]
pub struct JanusState {
    pub a_plus: f64,
    pub a_minus: f64,
    pub h_plus: f64,
    pub h_minus: f64,
    pub coupling: JanusCoupling,
}

/// Cosmological drift step (peculiar convention) for Janus.
///
/// `pos[i] += vel[i] * dt / a_eff[i]` where `a_eff = a_plus` for m+, `a_minus` for m-.
///
/// Periodic boundary wrap to [-L/2, L/2]. Robust round-based wrap for large dt
/// (ChatGPT recommendation in plan §5.4).
pub fn drift_cosmo(particles: &mut ParticleArrays, state: &JanusState, dt: f64, box_size: f64) {
    let half = box_size / 2.0;
    for i in 0..particles.n {
        let a = if particles.sign[i] > 0 {
            state.a_plus
        } else {
            state.a_minus
        };
        let inv_a = 1.0 / a;
        let dt_inv_a = dt * inv_a;

        let new_x = particles.pos_x[i] + particles.vel_x[i] as f64 * dt_inv_a;
        let new_y = particles.pos_y[i] + particles.vel_y[i] as f64 * dt_inv_a;
        let new_z = particles.pos_z[i] + particles.vel_z[i] as f64 * dt_inv_a;

        particles.pos_x[i] = wrap_pbc(new_x, half, box_size);
        particles.pos_y[i] = wrap_pbc(new_y, half, box_size);
        particles.pos_z[i] = wrap_pbc(new_z, half, box_size);
    }
}

/// Cosmological kick step (peculiar convention) for Janus.
///
/// `vel[i] += (acc[i] / a_eff² - h_eff · vel[i]) * dt`
///
/// Where acc is the BARE comoving acceleration (output of force kernel,
/// no a-factor applied yet).
pub fn kick_cosmo(particles: &mut ParticleArrays, state: &JanusState, dt: f64) {
    for i in 0..particles.n {
        let (a, h) = if particles.sign[i] > 0 {
            (state.a_plus, state.h_plus)
        } else {
            (state.a_minus, state.h_minus)
        };
        let inv_a2 = 1.0 / (a * a);

        let acc_x_co = particles.acc_x[i] as f64 * inv_a2;
        let acc_y_co = particles.acc_y[i] as f64 * inv_a2;
        let acc_z_co = particles.acc_z[i] as f64 * inv_a2;

        let v_x = particles.vel_x[i] as f64;
        let v_y = particles.vel_y[i] as f64;
        let v_z = particles.vel_z[i] as f64;

        particles.vel_x[i] = (v_x + (acc_x_co - h * v_x) * dt) as f32;
        particles.vel_y[i] = (v_y + (acc_y_co - h * v_y) * dt) as f32;
        particles.vel_z[i] = (v_z + (acc_z_co - h * v_z) * dt) as f32;
    }
}

/// Periodic boundary wrap to [-half, +half] using round-based formula.
/// Robust for arbitrary displacement (|x| < several × box_size).
#[inline(always)]
fn wrap_pbc(x: f64, half: f64, box_size: f64) -> f64 {
    x - box_size * (x / box_size).round()
        + if (x - box_size * (x / box_size).round()) > half {
            -box_size
        } else if (x - box_size * (x / box_size).round()) < -half {
            box_size
        } else {
            0.0
        }
}

/// Compute the effective density source for the PM Poisson solver in Janus mode.
///
/// In Janus cosmology, m+ and m- contribute to the SAME PM grid (signed mass):
///   ρ_eff(x) = ρ⁺(x) - ρ⁻(x)
///
/// However, our pipeline stores ρ_+ and ρ_- on SEPARATE grids (PmGrid in
/// pm_grid.rs) which preserves the Janus structure for asymmetric coupling.
/// This function computes a SINGLE-GRID effective density for compatibility
/// with traditional PM solvers if needed.
///
/// Returns flat row-major Vec<f64> of length n_pm³.
pub fn compute_effective_density_single_grid(
    particles: &ParticleArrays,
    n_pm: usize,
    box_size: f64,
) -> Vec<f64> {
    let mut rho = vec![0.0_f64; n_pm * n_pm * n_pm];
    let cell = box_size / n_pm as f64;
    let inv_cell = 1.0 / cell;
    let half = box_size / 2.0;

    for i in 0..particles.n {
        let sign = particles.sign[i] as f64;
        let m_signed = particles.mass[i] as f64 * sign;

        // CIC scatter
        let gx = ((particles.pos_x[i] + half) * inv_cell).rem_euclid(n_pm as f64);
        let gy = ((particles.pos_y[i] + half) * inv_cell).rem_euclid(n_pm as f64);
        let gz = ((particles.pos_z[i] + half) * inv_cell).rem_euclid(n_pm as f64);

        let ix = gx.floor() as usize % n_pm;
        let iy = gy.floor() as usize % n_pm;
        let iz = gz.floor() as usize % n_pm;
        let fx = gx - gx.floor();
        let fy = gy - gy.floor();
        let fz = gz - gz.floor();

        let wx = [1.0 - fx, fx];
        let wy = [1.0 - fy, fy];
        let wz = [1.0 - fz, fz];

        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let ci = (ix + di) % n_pm;
                    let cj = (iy + dj) % n_pm;
                    let ck = (iz + dk) % n_pm;
                    let w = wx[di] * wy[dj] * wz[dk];
                    rho[ci + n_pm * (cj + n_pm * ck)] += m_signed * w;
                }
            }
        }
    }
    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_state() -> JanusState {
        JanusState {
            a_plus: 1.0,
            a_minus: 1.0,
            h_plus: 0.0,
            h_minus: 0.0,
            coupling: JanusCoupling {
                phi: 1.0,
                c_ratio_sq: 1.0,
                repulsion_scale: 1.0,
            },
        }
    }

    #[test]
    fn test_coupling_factors_neutral() {
        let c = JanusCoupling {
            phi: 1.0,
            c_ratio_sq: 1.0,
            repulsion_scale: 1.0,
        };
        // Neutral case: cross factors = 1.0
        assert!((c.cross_minus_plus() - 1.0).abs() < 1e-15);
        assert!((c.cross_plus_minus() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_coupling_factors_with_phi() {
        // φ = 0.5, c̄² = 2.0 → cross_mp = 2.0/0.5 = 4.0, cross_pm = 0.5
        let c = JanusCoupling {
            phi: 0.5,
            c_ratio_sq: 2.0,
            repulsion_scale: 1.0,
        };
        assert!((c.cross_minus_plus() - 4.0).abs() < 1e-15);
        assert!((c.cross_plus_minus() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_drift_unit_velocity() {
        // Unit velocity, dt=1, a=1: pos += vel
        let mut p = ParticleArrays::new(1);
        p.pos_x[0] = 0.0;
        p.pos_y[0] = 0.0;
        p.pos_z[0] = 0.0;
        p.vel_x[0] = 1.0;
        p.vel_y[0] = 2.0;
        p.vel_z[0] = -3.0;
        p.sign[0] = 1;
        let state = unit_state();
        drift_cosmo(&mut p, &state, 1.0, 1000.0);
        // Tolerance: 1e-6 (SP→DP roundtrip on velocity)
        assert!((p.pos_x[0] - 1.0).abs() < 1e-6);
        assert!((p.pos_y[0] - 2.0).abs() < 1e-6);
        assert!((p.pos_z[0] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_drift_with_a_plus() {
        // a_plus=2, vel=1, dt=1 : pos += vel·dt/a = 0.5
        let mut p = ParticleArrays::new(1);
        p.pos_x[0] = 0.0;
        p.vel_x[0] = 1.0;
        p.sign[0] = 1;
        let state = JanusState {
            a_plus: 2.0,
            a_minus: 4.0,
            h_plus: 0.0,
            h_minus: 0.0,
            coupling: JanusCoupling {
                phi: 1.0,
                c_ratio_sq: 1.0,
                repulsion_scale: 1.0,
            },
        };
        drift_cosmo(&mut p, &state, 1.0, 1000.0);
        // m+ uses a_plus=2 → pos += 0.5
        assert!((p.pos_x[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_drift_with_a_minus() {
        // m- uses a_minus, expect pos += vel·dt/a_minus
        let mut p = ParticleArrays::new(1);
        p.pos_x[0] = 0.0;
        p.vel_x[0] = 1.0;
        p.sign[0] = -1;
        let state = JanusState {
            a_plus: 2.0,
            a_minus: 4.0,
            h_plus: 0.0,
            h_minus: 0.0,
            coupling: JanusCoupling {
                phi: 1.0,
                c_ratio_sq: 1.0,
                repulsion_scale: 1.0,
            },
        };
        drift_cosmo(&mut p, &state, 1.0, 1000.0);
        // m- uses a_minus=4 → pos += 0.25
        assert!((p.pos_x[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_kick_no_friction_no_scale() {
        // a=1, H=0, dt=1, acc=2: vel += acc·dt/a² = 2
        let mut p = ParticleArrays::new(1);
        p.vel_x[0] = 1.0;
        p.acc_x[0] = 2.0;
        p.sign[0] = 1;
        let state = unit_state();
        kick_cosmo(&mut p, &state, 1.0);
        // Tolerance: 1e-5 (SP)
        assert!((p.vel_x[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_kick_with_hubble_friction() {
        // H=0.1, vel=1, acc=0, dt=1: vel += -H·vel·dt = -0.1, so vel = 0.9
        let mut p = ParticleArrays::new(1);
        p.vel_x[0] = 1.0;
        p.acc_x[0] = 0.0;
        p.sign[0] = 1;
        let state = JanusState {
            a_plus: 1.0,
            a_minus: 1.0,
            h_plus: 0.1,
            h_minus: 0.0,
            coupling: JanusCoupling {
                phi: 1.0,
                c_ratio_sq: 1.0,
                repulsion_scale: 1.0,
            },
        };
        kick_cosmo(&mut p, &state, 1.0);
        assert!((p.vel_x[0] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_kick_a_squared_factor() {
        // a=2, acc=8: acc_co = acc/a² = 2. vel += 2 → from 0 to 2
        let mut p = ParticleArrays::new(1);
        p.vel_x[0] = 0.0;
        p.acc_x[0] = 8.0;
        p.sign[0] = 1;
        let state = JanusState {
            a_plus: 2.0,
            a_minus: 1.0,
            h_plus: 0.0,
            h_minus: 0.0,
            coupling: JanusCoupling {
                phi: 1.0,
                c_ratio_sq: 1.0,
                repulsion_scale: 1.0,
            },
        };
        kick_cosmo(&mut p, &state, 1.0);
        assert!((p.vel_x[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_pbc_wrap() {
        // Particle at x = box/2 + epsilon should wrap to -box/2 + epsilon
        let half = 50.0;
        let box_size = 100.0;
        let x_in = 50.0001;
        let x_out = wrap_pbc(x_in, half, box_size);
        assert!(x_out < 0.0 && x_out > -50.0, "Wrap failed: {}", x_out);
    }

    #[test]
    fn test_drift_with_pbc_wrap() {
        // Particle at x = box/2 - eps, drift across boundary
        let mut p = ParticleArrays::new(1);
        p.pos_x[0] = 49.9;
        p.vel_x[0] = 1.0;
        p.sign[0] = 1;
        let state = unit_state();
        drift_cosmo(&mut p, &state, 1.0, 100.0); // pos += 1.0 → 50.9 → wraps to -49.1
        assert!(
            p.pos_x[0] < 0.0 && p.pos_x[0] > -50.0,
            "PBC wrap failed: pos = {}",
            p.pos_x[0]
        );
    }

    #[test]
    fn test_effective_density_two_plus_two_minus_zero_total() {
        // 2 m+ at (0,0,0), 2 m- at same position → total ρ_eff sums to 0
        let mut p = ParticleArrays::new(4);
        for i in 0..4 {
            p.pos_x[i] = 0.0;
            p.pos_y[i] = 0.0;
            p.pos_z[i] = 0.0;
            p.mass[i] = 1.0;
        }
        p.sign[0] = 1;
        p.sign[1] = 1;
        p.sign[2] = -1;
        p.sign[3] = -1;
        let rho = compute_effective_density_single_grid(&p, 8, 100.0);
        let total: f64 = rho.iter().sum();
        // 2 m+ × +1 + 2 m- × -1 = 0
        assert!(total.abs() < 1e-10, "Total ρ_eff = {}", total);
    }

    #[test]
    fn test_effective_density_single_plus() {
        // 1 m+ at center: rho_eff has total = +1
        let mut p = ParticleArrays::new(1);
        p.pos_x[0] = 0.0;
        p.pos_y[0] = 0.0;
        p.pos_z[0] = 0.0;
        p.sign[0] = 1;
        p.mass[0] = 1.5;
        let rho = compute_effective_density_single_grid(&p, 16, 50.0);
        let total: f64 = rho.iter().sum();
        // CIC distribute mass=1.5 → total should be 1.5
        assert!((total - 1.5).abs() < 1e-10, "Total = {}", total);
    }
}
