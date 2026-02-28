//! Test: Run with standard ICs but NO virialization
//!
//! This test generates ICs exactly like GpuNBodyTwoPass::new() but
//! skips the virialization step (alpha scaling).
//!
//! If Seg_max > 0.3 → virialization is suppressing segregation
//! If Seg_max < 0.1 → problem is elsewhere

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("=== Test: Standard ICs WITHOUT Virialization ===\n");

    // Same parameters as regression test
    let n = 100_000;
    let n_steps = 2000;
    let dt = 0.01;
    let box_size = 100.0;
    let eta = 1.045;
    let z_init = 5.0;
    let r_cut = box_size / 8.0;  // 12.5 Mpc

    let janus_params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&janus_params, z_init);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * dt);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64);

    // Generate ICs EXACTLY like GpuNBodyTwoPass::new() but skip virialization
    println!("Generating ICs (uniform random, virial_velocity, NO alpha scaling)...\n");

    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let n_positive = n / 2;
    let n_negative = n - n_positive;

    let mut pos_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut vel_data: Vec<f32> = Vec::with_capacity(n * 3);
    let mut signs_data: Vec<i8> = Vec::with_capacity(n);

    // Match GpuNBodyTwoPass::new(): virial_velocity = sqrt(N/box) × 0.3
    let virial_velocity = ((n as f64) / box_size).sqrt() * 0.3;
    println!("  virial_velocity = {:.4}", virial_velocity);

    // Generate + particles first (like GpuNBodyTwoPass::new())
    for _ in 0..n_positive {
        let x = (rng.random::<f64>() - 0.5) * box_size;
        let y = (rng.random::<f64>() - 0.5) * box_size;
        let z = (rng.random::<f64>() - 0.5) * box_size;
        let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([vx as f32, vy as f32, vz as f32]);
        signs_data.push(1);
    }

    // Then - particles
    for _ in 0..n_negative {
        let x = (rng.random::<f64>() - 0.5) * box_size;
        let y = (rng.random::<f64>() - 0.5) * box_size;
        let z = (rng.random::<f64>() - 0.5) * box_size;
        let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
        let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
        pos_data.extend([x as f32, y as f32, z as f32]);
        vel_data.extend([vx as f32, vy as f32, vz as f32]);
        signs_data.push(-1);
    }

    // NO virialization step - use ICs as-is via with_custom_ics()
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_data.clone(), vel_data.clone(), signs_data.clone(), box_size
    ).expect("GPU init failed");
    sim.set_theta(0.5);

    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation().expect("Seg failed");

    // Also compute what alpha WOULD be if we virialized
    let mass = 1.0_f64;
    let g_code = 1.0_f64;
    let softening = 0.1_f64;
    let half_box = box_size / 2.0;

    let mut pe_binding = 0.0_f64;
    // Quick sampling for PE estimate (full N² takes too long)
    let sample_size = 5000;
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let sample_indices: Vec<usize> = indices.into_iter().take(sample_size).collect();

    for &i in &sample_indices {
        let xi = pos_data[i * 3] as f64;
        let yi = pos_data[i * 3 + 1] as f64;
        let zi = pos_data[i * 3 + 2] as f64;
        let si = signs_data[i];

        for &j in &sample_indices {
            if j <= i { continue; }
            if signs_data[j] != si { continue; }

            let xj = pos_data[j * 3] as f64;
            let yj = pos_data[j * 3 + 1] as f64;
            let zj = pos_data[j * 3 + 2] as f64;

            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
            if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
            if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

            let r_sq = dx*dx + dy*dy + dz*dz;
            let r_soft = (r_sq + softening*softening).sqrt();
            pe_binding -= g_code * mass * mass / r_soft;
        }
    }

    // Scale up from sample to full population
    let scale_factor = (n as f64 / sample_size as f64).powi(2);
    pe_binding *= scale_factor;

    let ke_target = pe_binding.abs() / 2.0;
    let alpha_would_be = if ke_0 > 1e-20 { (ke_target / ke_0).sqrt() } else { 1.0 };

    println!("  KE_0 (no virialization) = {:.4e}", ke_0);
    println!("  PE_binding (sampled) = {:.4e}", pe_binding);
    println!("  Alpha (would be if virialized) = {:.4}", alpha_would_be);
    println!("  Seg_0 = {:.4}", seg_0);
    println!();

    let mut seg_max = seg_0;
    let mut seg_at_step = 0;

    println!("Running {} steps with pure BH (step_dkd)...\n", n_steps);
    println!("{:>6} {:>8} {:>12} {:>10} {:>10}", "Step", "z", "KE/KE₀", "Seg", "Seg_max");
    println!("{}", "-".repeat(55));

    for step in 1..=n_steps {
        let current_tau = cosmo.tau_start + (step as f64) * dtau_per_step;
        let (a, hubble) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        // Use pure BH (step_dkd) to match reference
        sim.step_dkd(dt, hubble, dtau_per_dt).unwrap();

        if step % 200 == 0 || step <= 20 && step % 5 == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let seg = sim.segregation().unwrap();
            if seg > seg_max {
                seg_max = seg;
                seg_at_step = step;
            }

            println!("{:>6} {:>8.2} {:>12.4} {:>10.4} {:>10.4}",
                     step, z, ke / ke_0, seg, seg_max);

            // Stop if KE explodes
            if ke / ke_0 > 100.0 {
                println!("\n*** KE explosion - stopping ***");
                break;
            }
        }
    }

    let seg_final = sim.segregation().unwrap();
    let ke_final = sim.kinetic_energy().unwrap();

    println!("\n=== Results ===");
    println!("  Seg_0 = {:.4}", seg_0);
    println!("  Seg_max = {:.4} (at step {})", seg_max, seg_at_step);
    println!("  Seg_final = {:.4}", seg_final);
    println!("  KE/KE₀ = {:.4}", ke_final / ke_0);
    println!();
    println!("  Alpha (would be) = {:.4}", alpha_would_be);
    println!();

    if seg_max > 0.3 {
        println!("✓ HIGH segregation without virialization");
        println!("→ Virialization (alpha={:.2}) is suppressing collapse", alpha_would_be);
        println!("→ Need to use analytical virialization like reference, or skip it");
    } else if seg_max > 0.1 {
        println!("? MODERATE segregation without virialization");
    } else {
        println!("✗ LOW segregation even without virialization");
        println!("→ Problem is in force calculation, not virialization");
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda,cufft features");
}
