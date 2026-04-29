//! Phase 10A.5c — 3 cas Janus sur GPU.
//!
//! Setup pratique: les 2 particules de test sont entourées de "filler"
//! particles placées loin (masse négligeable via softening) pour donner au
//! BVH par sign assez de leaves. On teste les SIGNES (direction) des
//! accélérations sur les 2 particules d'intérêt.

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

    println!("=== Phase 10A.5c — 3 cas Janus GPU ===");

    let box_size = 100.0_f64;
    let r_cut = 9.375_f64;
    let r_s = r_cut / 5.0;

    // Cosmologie neutre
    let a_plus = 1.0_f64;
    let a_minus = 1.0_f64;
    let h_plus = 0.0_f64;
    let h_minus = 0.0_f64;
    let phi = 1.0_f64;
    let c_ratio_sq = 1.0_f64;
    let repulsion_scale = 1.0_f64;
    let dt = 0.01_f64;

    // Helper: créer un setup avec 2 particules de test sur axe x à ±5,
    // entourées de filler particles à très grande distance (>r_cut donc
    // n'influencent pas via Tree, et leur PM contribution est moyennée).
    let make_setup = |sign_a: i8, sign_b: i8| -> (Vec<f32>, Vec<f32>, Vec<i8>) {
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let mut signs = Vec::new();
        // Test particles
        pos.extend_from_slice(&[-5.0, 0.0, 0.0]);
        vel.extend_from_slice(&[0.0, 0.0, 0.0]);
        signs.push(sign_a);
        pos.extend_from_slice(&[5.0, 0.0, 0.0]);
        vel.extend_from_slice(&[0.0, 0.0, 0.0]);
        signs.push(sign_b);
        // Filler m+ (10) and m- (10) far from the test region (z > 30)
        for k in 0..10 {
            let off = 30.0 + k as f32 * 5.0;
            pos.extend_from_slice(&[0.0, 0.0, off]);
            vel.extend_from_slice(&[0.0, 0.0, 0.0]);
            signs.push(1);
        }
        for k in 0..10 {
            let off = -30.0 - k as f32 * 5.0;
            pos.extend_from_slice(&[0.0, 0.0, off]);
            vel.extend_from_slice(&[0.0, 0.0, 0.0]);
            signs.push(-1);
        }
        (pos, vel, signs)
    };

    let mut all_pass = true;

    // ========== CAS 1: 2 m+ → attraction ==========
    println!();
    println!("--- Cas 1: 2 m+ at x=±5 → expect attraction ---");
    {
        let (pos, vel, signs) = make_setup(1, 1);
        let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, box_size)?;
        sim.set_mass_factor(1.0);
        sim.set_softening(0.05);
        sim.set_theta(0.5);
        sim.step_treepm_gpu_cosmo(
            dt, r_cut, r_s,
            a_plus, a_minus, h_plus, h_minus,
            phi, c_ratio_sq, repulsion_scale,
        )?;
        let (_, vel_out, _) = sim.get_particles()?;
        let v0_x = vel_out[0];
        let v1_x = vel_out[3];
        println!("  v0_x={:.5}, v1_x={:.5}", v0_x, v1_x);
        if v0_x > 0.0 && v1_x < 0.0 {
            println!("  ✅ PASS: attractive");
        } else {
            println!("  ❌ FAIL");
            all_pass = false;
        }
    }

    // ========== CAS 2: m+ et m- → répulsion ==========
    println!();
    println!("--- Cas 2: m+ at -5, m- at +5 → expect repulsion ---");
    {
        let (pos, vel, signs) = make_setup(1, -1);
        let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, box_size)?;
        sim.set_mass_factor(1.0);
        sim.set_softening(0.05);
        sim.set_theta(0.5);
        sim.step_treepm_gpu_cosmo(
            dt, r_cut, r_s,
            a_plus, a_minus, h_plus, h_minus,
            phi, c_ratio_sq, repulsion_scale,
        )?;
        let (_, vel_out, _) = sim.get_particles()?;
        let v0_x = vel_out[0];
        let v1_x = vel_out[3];
        println!("  v0_x={:.5}, v1_x={:.5}", v0_x, v1_x);
        if v0_x < 0.0 && v1_x > 0.0 {
            println!("  ✅ PASS: repulsive (Janus)");
        } else {
            println!("  ❌ FAIL");
            all_pass = false;
        }
    }

    // ========== CAS 3: 2 m- → attraction ==========
    println!();
    println!("--- Cas 3: 2 m- at x=±5 → expect attraction (Petit) ---");
    {
        let (pos, vel, signs) = make_setup(-1, -1);
        let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, box_size)?;
        sim.set_mass_factor(1.0);
        sim.set_softening(0.05);
        sim.set_theta(0.5);
        sim.step_treepm_gpu_cosmo(
            dt, r_cut, r_s,
            a_plus, a_minus, h_plus, h_minus,
            phi, c_ratio_sq, repulsion_scale,
        )?;
        let (_, vel_out, _) = sim.get_particles()?;
        let v0_x = vel_out[0];
        let v1_x = vel_out[3];
        println!("  v0_x={:.5}, v1_x={:.5}", v0_x, v1_x);
        if v0_x > 0.0 && v1_x < 0.0 {
            println!("  ✅ PASS: attractive (Petit)");
        } else {
            println!("  ❌ FAIL");
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("✅ Phase 10A.5c PASS — 3/3 cas Janus GPU");
        Ok(())
    } else {
        eprintln!("❌ Phase 10A.5c FAIL");
        std::process::exit(1)
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires features: cuda cufft");
    std::process::exit(1);
}
