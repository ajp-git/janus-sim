//! Test (e) — 2-body force calibration.
//!
//! Setup: 2 particles, sign=+1, separated by r=1 Mpc comoving at z=49.
//! Take one DKD step with v_init=0 and Hubble=0. Then v_after = (acc/a²)·dt.
//! Expected acc/a² = G·m/(a²·r²)  with G·m = mass_per_particle (in code units).
//!
//! Tolerance: 1% on |acc/a²_measured| / |acc/a²_expected|.

use janus::nbody_gpu::GpuNBodySimulation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Test (e) — 2-body force calibration at z=49 ===");

    // Code-unit constants (matches new_with_state defaults)
    let g_cosmo: f64 = 4.499e-15;        // Mpc³/(M⊙·Gyr²)
    let rho_crit: f64 = 2.775e11;        // M⊙/Mpc³
    let omega_m: f64 = 0.3;              // matter fraction baked in new_with_state
    let l_box: f64 = 200.0;              // Mpc

    // Test scenario
    let z = 49.0;
    let a = 1.0 / (1.0 + z);             // 0.02
    let r = 1.0;                          // Mpc comoving separation

    // Use 2 + filler particles to ensure BVH builds correctly. The 2 close pair
    // dominates the force; the rest provides BVH context far away.
    let n_close = 2;
    let n_filler = 100;
    let n_total = n_close + n_filler;

    let mass_per_part_pre = g_cosmo * omega_m * rho_crit * l_box.powi(3) / n_total as f64;
    println!("[CONFIG] z={}  a={}  N={}  L_box={} Mpc  r_pair={} Mpc",
        z, a, n_total, l_box, r);
    println!("[CONFIG] mass_pp (pre-factor) = {:.4e}", mass_per_part_pre);

    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs = Vec::with_capacity(n_total);

    // Pair: at (-r/2, 0, 0) and (+r/2, 0, 0)
    positions.extend_from_slice(&[-r/2.0, 0.0, 0.0]);
    velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
    signs.push(1i32);

    positions.extend_from_slice(&[ r/2.0, 0.0, 0.0]);
    velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
    signs.push(1i32);

    // Filler: random, far from origin
    use rand::prelude::*;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(7);
    let half = l_box / 2.0;
    for _ in 0..n_filler {
        // Place at random pos, but ≥ 50 Mpc from origin to not disturb pair
        let mut x = 0.0; let mut y = 0.0; let mut z = 0.0;
        for _ in 0..50 {
            x = rng.gen::<f64>() * l_box - half;
            y = rng.gen::<f64>() * l_box - half;
            z = rng.gen::<f64>() * l_box - half;
            if (x*x + y*y + z*z).sqrt() > 50.0 { break; }
        }
        positions.extend_from_slice(&[x, y, z]);
        velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
        signs.push(1i32);
    }

    let mut sim = GpuNBodySimulation::new_with_state(
        n_total, 0, l_box, positions, velocities, signs.clone(),
    )?;
    sim.set_theta(0.7);
    sim.set_softening(1e-3);  // Tiny softening — we're testing point-mass force
    sim.set_phi(1.0, 1.0);
    sim.c_ratio_sq = 1.0;
    sim.repulsion_scale = 0.0;
    sim.set_mass_factor(1.0 / omega_m);   // restore to mass = g_cosmo × ρ_crit × L³ / N
    let mass_eff = g_cosmo * rho_crit * l_box.powi(3) / n_total as f64;
    println!("[CONFIG] mass_eff (G·m_phys, after set_mass_factor) = {:.4e} Mpc³/Gyr²", mass_eff);

    // One DKD step with H=0 → v_after = (acc/a²)·dt
    let dt = 1e-7;  // tiny dt: positions barely move during D1, force ≈ at IC
    sim.step_with_expansion_dkd_gpu_cosmo(dt, a, a, 0.0, 0.0)?;

    let vel = sim.get_velocities()?;
    let pos_after = sim.get_positions()?;

    // Internal Morton sort scrambles index ordering. Find the particle with
    // the LARGEST |v|: those will be the close-pair (their force >> filler).
    let mut best_idx = 0;
    let mut best_v2 = 0.0_f64;
    for i in 0..n_total {
        let v2 = vel[3*i].powi(2) + vel[3*i+1].powi(2) + vel[3*i+2].powi(2);
        if v2 > best_v2 { best_v2 = v2; best_idx = i; }
    }
    let v0x = vel[3*best_idx];
    let v0y = vel[3*best_idx + 1];
    let v0z = vel[3*best_idx + 2];
    let acc_a2_mag = best_v2.sqrt() / dt;
    println!("[FOUND] max-|v| particle is at sorted index {} (Morton order)", best_idx);
    println!("[FOUND] its current pos = ({:.4}, {:.4}, {:.4})",
        pos_after[3*best_idx], pos_after[3*best_idx+1], pos_after[3*best_idx+2]);

    // Expected: G·m / (a² · r²) for the close pair, dominant.
    // In code units, G·m = mass_eff. r = 1 Mpc (comoving). a = 0.02.
    let acc_a2_expected = mass_eff / (a * a * r * r);

    // Direction check: particle 0 at -0.5, partner at +0.5. So force is toward +x.
    // Hence v0x should be POSITIVE.
    let direction_ok = v0x > 0.0;

    println!();
    println!("=== RESULT ===");
    println!("particle 0 v_after = ({:.4e}, {:.4e}, {:.4e}) Mpc/Gyr", v0x, v0y, v0z);
    println!("|v_after| / dt   = {:.6e}    (this is acc/a² in our convention)", acc_a2_mag);
    println!("expected G·m/(a²·r²) = {:.6e}", acc_a2_expected);
    println!("ratio measured/expected = {:.6}", acc_a2_mag / acc_a2_expected);
    println!("direction toward partner (v0x>0): {}", direction_ok);
    println!();

    let ratio = acc_a2_mag / acc_a2_expected;
    // Position drift sanity: |D2 drift| = |v_after| · (dt/2) / a (peculiar drift scaling)
    let drift_d2 = (best_v2.sqrt()) * (dt / 2.0) / a;
    println!("[DRIFT] expected D2 drift = {:.4e} Mpc", drift_d2);

    if (ratio - 1.0).abs() < 0.01 {
        println!("✅ ratio within 1% of unity — force normalization OK");
    } else if (ratio - 1.0).abs() < 0.10 {
        println!("⚠ ratio within 10% — borderline (likely BVH approximation θ=0.7)");
    } else {
        println!("❌ ratio off by >10% — NORMALIZATION BUG SUSPECTED");
        println!("   Check: G·m units, position-units convention (comoving vs proper),");
        println!("          a² factor application, mass_factor, soft eps");
    }

    Ok(())
}
