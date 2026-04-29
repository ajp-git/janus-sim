//! Integrated TreePM-Janus step (CPU, for small-N validation).
//!
//! Combines:
//! - janus.rs drift_cosmo + kick_cosmo (peculiar Peebles convention)
//! - pm_grid.rs solve_poisson + interpolate_force_grad4 (PM long-range)
//!
//! This is a CPU-only validation pipeline. Production runs use the GPU
//! step_treepm_gpu_cosmo (deferred to Phase 8/9 GPU integration).
//!
//! Convention de précision (Plan §3.0):
//! - Positions DP, vélocités/accélérations SP within ParticleArrays SoA
//! - Force computation in PmGrid uses DP throughout

use super::gpu_layout::ParticleArrays;
use super::janus::{drift_cosmo, kick_cosmo, JanusState};
use super::pm_grid::PmGrid;

/// Single TreePM-Janus DKD step (PM-only short-range, no BH yet).
///
/// Sequence:
/// 1. Drift dt/2 (peculiar /a)
/// 2. CIC scatter ρ+, ρ- onto PM grid
/// 3. Poisson solve (Janus separate grids)
/// 4. CIC gather force per particle (with sign-dependent attract/repel)
/// 5. Kick dt (acc/a² - H·v)
/// 6. Drift dt/2
pub fn step_treepm_janus_pm_only(
    particles: &mut ParticleArrays,
    pm: &mut PmGrid,
    state: &JanusState,
    dt: f64,
    box_size: f64,
    g_phys: f64,
) {
    let half_dt = dt * 0.5;
    let v_cell = pm.cell_size.powi(3);
    let g_solver = g_phys / v_cell;

    // D1
    drift_cosmo(particles, state, half_dt, box_size);

    // CIC scatter
    pm.clear();
    for i in 0..particles.n {
        pm.assign_mass(
            particles.pos_x[i],
            particles.pos_y[i],
            particles.pos_z[i],
            particles.mass[i] as f64,
            particles.sign[i],
        );
    }

    // Poisson solve (with CIC deconvolution + Laplacien continu, Phase 2)
    pm.solve_poisson(g_solver);

    // CIC gather force (with grad4)
    particles.reset_acc();
    for i in 0..particles.n {
        let (fx, fy, fz) = pm.interpolate_force_grad4(
            particles.pos_x[i],
            particles.pos_y[i],
            particles.pos_z[i],
            particles.sign[i],
        );
        // Mass scaling: F = m × acc. acc = F/m.
        let inv_m = 1.0 / particles.mass[i] as f64;
        // Note: pm_grid returns "force per mass-charge", needs * mass for acc
        // Actually pm_grid returns -∇φ which is acc directly (since φ is from
        // Poisson with mass density on RHS). So no division by m.
        let _ = inv_m; // unused in this convention
        particles.acc_x[i] = fx as f32;
        particles.acc_y[i] = fy as f32;
        particles.acc_z[i] = fz as f32;
    }

    // K (full dt) — peculiar kick
    kick_cosmo(particles, state, dt);

    // D2
    drift_cosmo(particles, state, half_dt, box_size);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treepm::janus::JanusCoupling;

    fn newton_state() -> JanusState {
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
    fn test_step_two_plus_attract() {
        // Two m+ particles separated along x; they should ATTRACT after one step.
        let box_size = 1000.0;
        let n_pm = 64;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = ParticleArrays::new(2);

        // Place at ±100 Mpc in x (= ±10 cells from center)
        particles.pos_x[0] = -100.0;
        particles.pos_y[0] = 0.0;
        particles.pos_z[0] = 0.0;
        particles.pos_x[1] = 100.0;
        particles.pos_y[1] = 0.0;
        particles.pos_z[1] = 0.0;
        // Both at rest, both m+
        particles.sign[0] = 1;
        particles.sign[1] = 1;
        particles.mass[0] = 1.0;
        particles.mass[1] = 1.0;

        let state = newton_state();
        let dt = 0.1;
        let g = 1.0;
        step_treepm_janus_pm_only(&mut particles, &mut pm, &state, dt, box_size, g);

        // After step, particle 0 (left, m+) should have moved to the right
        // (toward particle 1, attractive).
        // Particle 1 should have moved left.
        // Tolerance: not strict, just direction.
        assert!(
            particles.vel_x[0] > 0.0,
            "Particle 0 (left m+) should accelerate toward +x (toward particle 1), got vel_x = {}",
            particles.vel_x[0]
        );
        assert!(
            particles.vel_x[1] < 0.0,
            "Particle 1 (right m+) should accelerate toward -x, got vel_x = {}",
            particles.vel_x[1]
        );
    }

    #[test]
    fn test_step_plus_minus_repel() {
        // m+ at -100, m- at +100. Janus: should REPEL.
        let box_size = 1000.0;
        let n_pm = 64;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = ParticleArrays::new(2);
        particles.pos_x[0] = -100.0;
        particles.pos_x[1] = 100.0;
        particles.sign[0] = 1; // m+
        particles.sign[1] = -1; // m-
        particles.mass[0] = 1.0;
        particles.mass[1] = 1.0;

        let state = newton_state();
        let dt = 0.1;
        let g = 1.0;
        step_treepm_janus_pm_only(&mut particles, &mut pm, &state, dt, box_size, g);

        // m+ at -100: feels REPULSION from m- at +100 → should move further left (vel_x < 0)
        assert!(
            particles.vel_x[0] < 0.0,
            "m+ should be repelled toward -x, got vel_x = {}",
            particles.vel_x[0]
        );
        // m- at +100: feels REPULSION from m+ at -100 → should move further right (vel_x > 0)
        assert!(
            particles.vel_x[1] > 0.0,
            "m- should be repelled toward +x, got vel_x = {}",
            particles.vel_x[1]
        );
    }

    #[test]
    fn test_step_two_minus_attract() {
        // Two m- particles: should attract (Petit p.36, attractive Newton between m-).
        let box_size = 1000.0;
        let n_pm = 64;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = ParticleArrays::new(2);
        particles.pos_x[0] = -100.0;
        particles.pos_x[1] = 100.0;
        particles.sign[0] = -1;
        particles.sign[1] = -1;
        particles.mass[0] = 1.0;
        particles.mass[1] = 1.0;

        let state = newton_state();
        let dt = 0.1;
        let g = 1.0;
        step_treepm_janus_pm_only(&mut particles, &mut pm, &state, dt, box_size, g);

        // Two m- should attract: left m- moves right, right m- moves left
        assert!(
            particles.vel_x[0] > 0.0,
            "Two m- should attract, particle 0 vel_x = {}",
            particles.vel_x[0]
        );
        assert!(
            particles.vel_x[1] < 0.0,
            "Two m- should attract, particle 1 vel_x = {}",
            particles.vel_x[1]
        );
    }

    #[test]
    fn test_multi_step_two_plus_orbit_decay() {
        // Two m+ particles with small initial angular velocity.
        // Without dissipation, they should oscillate. With Hubble friction off,
        // we just check kinetic energy doesn't blow up.
        let box_size = 1000.0;
        let n_pm = 32;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = ParticleArrays::new(2);
        particles.pos_x[0] = -50.0;
        particles.pos_x[1] = 50.0;
        particles.sign[0] = 1;
        particles.sign[1] = 1;
        particles.mass[0] = 1.0;
        particles.mass[1] = 1.0;

        let state = newton_state();
        let dt = 0.01;
        let g = 0.1;

        let mut max_v: f32 = 0.0;
        for _ in 0..20 {
            step_treepm_janus_pm_only(&mut particles, &mut pm, &state, dt, box_size, g);
            let v = particles.vel_x[0].abs().max(particles.vel_x[1].abs());
            max_v = max_v.max(v);
        }

        // Velocity should be finite (no NaN, no runaway)
        assert!(particles.vel_x[0].is_finite());
        assert!(particles.vel_x[1].is_finite());
        // Reasonable magnitude (not >> c or anything insane)
        assert!(max_v < 1e6, "v_max = {} (runaway?)", max_v);
    }

    #[test]
    fn test_step_no_force_no_velocity() {
        // Single particle, no other source → no force → no velocity change.
        let box_size = 1000.0;
        let n_pm = 32;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = ParticleArrays::new(1);
        particles.pos_x[0] = 0.0;
        particles.pos_y[0] = 0.0;
        particles.pos_z[0] = 0.0;
        particles.vel_x[0] = 0.0;
        particles.sign[0] = 1;
        particles.mass[0] = 0.0; // zero-mass test particle
        let state = newton_state();
        step_treepm_janus_pm_only(&mut particles, &mut pm, &state, 0.1, box_size, 1.0);

        // No source, no force on test particle.
        // (The test particle itself is the source, but its self-force is 0.)
        assert_eq!(particles.vel_x[0], 0.0);
        assert_eq!(particles.vel_y[0], 0.0);
        assert_eq!(particles.vel_z[0], 0.0);
    }
}
