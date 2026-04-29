//! Phase 10A.5 — Test GPU TreePM Janus pipeline.
//!
//! Smoke test: 100 particules, 10 steps DKD, vérifier pas de NaN.
//! Si ce test passe : pipeline GPU compilé + lance + ne crashe pas.
//! Si NaN ou crash : STOP, BLOCKER.

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    println!("=== Phase 10A.5 GPU smoke test ===");

    let n_total: usize = 100;
    let n_plus: usize = 5;
    let n_minus: usize = 95;
    assert_eq!(n_plus + n_minus, n_total);
    let box_size = 100.0_f64;
    let half = box_size as f32 * 0.5;

    let mut rng = StdRng::seed_from_u64(42);
    let mut pos = Vec::with_capacity(n_total * 3);
    let mut vel = Vec::with_capacity(n_total * 3);
    let mut signs = Vec::with_capacity(n_total);

    for i in 0..n_total {
        pos.push(rng.random::<f32>() * box_size as f32 - half);
        pos.push(rng.random::<f32>() * box_size as f32 - half);
        pos.push(rng.random::<f32>() * box_size as f32 - half);
        vel.push(0.0_f32);
        vel.push(0.0_f32);
        vel.push(0.0_f32);
        signs.push(if i < n_plus { 1_i8 } else { -1_i8 });
    }

    println!("Initializing GPU sim (N={}, box={} Mpc)...", n_total, box_size);
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, box_size)?;

    sim.set_mass_factor(1.0);
    sim.set_softening(0.05);
    sim.set_theta(0.5);

    let r_cut = 9.375_f64; // = 6·Δg with n_pm=64, L=100
    let r_s = r_cut / 5.0; // PhotoNs canonical

    // Janus cosmology at z=10 (typical IC)
    let a_plus = 1.0 / (1.0 + 10.0_f64); // = 0.0909
    let a_minus = a_plus * 0.95; // slight asymm
    let h_plus = 0.7_f64; // arbitrary
    let h_minus = 0.7_f64;
    let phi = 0.85_f64; // typical Petit value at z=10
    let c_ratio_sq = 1.05_f64;
    let repulsion_scale = 1.0_f64;

    let dt = 0.001_f64; // small step

    println!("Running 10 steps DKD with TreePM Janus GPU...");
    for step in 1..=10 {
        sim.step_treepm_gpu_cosmo(
            dt, r_cut, r_s,
            a_plus, a_minus, h_plus, h_minus,
            phi, c_ratio_sq, repulsion_scale,
        )?;

        // Check no NaN
        let (positions, velocities, _signs) = sim.get_particles()?;
        let mut nan_count = 0;
        let mut max_abs_pos: f32 = 0.0;
        let mut max_abs_vel: f32 = 0.0;
        for v in positions.iter().chain(velocities.iter()) {
            if v.is_nan() || v.is_infinite() {
                nan_count += 1;
            }
            max_abs_pos = max_abs_pos.max(v.abs());
        }
        for v in velocities.iter() {
            max_abs_vel = max_abs_vel.max(v.abs());
        }
        if nan_count > 0 {
            eprintln!("ERROR step {}: {} NaN/Inf detected", step, nan_count);
            std::process::exit(1);
        }
        println!(
            "  step {}: max|pos|={:.3e}, max|vel|={:.3e}, NaN={}",
            step, max_abs_pos, max_abs_vel, nan_count
        );
    }

    println!();
    println!("✅ Phase 10A.5d PASS — 10 steps GPU TreePM Janus OK, no NaN");
    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires features: cuda cufft");
    std::process::exit(1);
}
