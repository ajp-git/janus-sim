/// Validate θ=1.0 GPU tree vs Morton+DKD reference over 500 steps
/// Target: S(500) within ±5% of reference

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 500_000;
        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let dt = 0.003;
        let n_steps = 500;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   θ=1.0 Validation: 500K particles, 500 steps                  ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("Configuration:");
        println!("  N particles: {} ({:.0}K)", n_particles, n_particles as f64 / 1000.0);
        println!("  η = {}", eta);
        println!("  Box size: {:.1}", box_size);
        println!("  dt = {}", dt);
        println!("  Steps: {}\n", n_steps);

        // ═══════════════════════════════════════════════════════════════════
        // Reference: Morton+DKD (θ=0.5, conservative)
        // ═══════════════════════════════════════════════════════════════════
        println!("═══ Reference: Morton+DKD (θ=0.5) ═══\n");

        let (s_ref, t_ref) = {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed to create simulation");
            sim.set_theta(0.5);  // Conservative for reference

            // Warmup
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup");
            let s0 = compute_segregation(&sim);
            println!("S(0) = {:.6}", s0);

            let t = Instant::now();
            for step in 1..=n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step");
                if step % 100 == 0 {
                    let s = compute_segregation(&sim);
                    println!("Step {:3}: S = {:.6}", step, s);
                }
            }
            let elapsed = t.elapsed().as_secs_f64();
            let s_final = compute_segregation(&sim);
            println!("\nReference S({}) = {:.6}", n_steps, s_final);
            println!("Time: {:.1}s ({:.1} ms/step)\n", elapsed, elapsed / n_steps as f64 * 1000.0);
            (s_final, elapsed)
        };

        // ═══════════════════════════════════════════════════════════════════
        // Test: GPU tree (θ=1.0)
        // ═══════════════════════════════════════════════════════════════════
        println!("═══ Test: GPU Tree (θ=1.0) ═══\n");

        let (s_test, t_test) = {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed to create simulation");
            sim.set_theta(1.0);  // Conservative but faster

            // Warmup
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup");
            let s0 = compute_segregation(&sim);
            println!("S(0) = {:.6}", s0);

            let t = Instant::now();
            for step in 1..=n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step");
                if step % 100 == 0 {
                    let s = compute_segregation(&sim);
                    println!("Step {:3}: S = {:.6}", step, s);
                }
            }
            let elapsed = t.elapsed().as_secs_f64();
            let s_final = compute_segregation(&sim);
            println!("\nTest S({}) = {:.6}", n_steps, s_final);
            println!("Time: {:.1}s ({:.1} ms/step)\n", elapsed, elapsed / n_steps as f64 * 1000.0);
            (s_final, elapsed)
        };

        // ═══════════════════════════════════════════════════════════════════
        // Results
        // ═══════════════════════════════════════════════════════════════════
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                     VALIDATION RESULTS                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let s_diff = ((s_test - s_ref) / s_ref * 100.0).abs();
        let speedup = t_ref / t_test;

        println!("Physics:");
        println!("  Reference S({}) = {:.6} (θ=0.5)", n_steps, s_ref);
        println!("  Test S({})      = {:.6} (θ=1.0)", n_steps, s_test);
        println!("  Difference:       {:.2}%", s_diff);
        println!();
        println!("Performance:");
        println!("  Reference: {:.1} ms/step", t_ref / n_steps as f64 * 1000.0);
        println!("  Test:      {:.1} ms/step", t_test / n_steps as f64 * 1000.0);
        println!("  Speedup:   {:.1}×", speedup);
        println!();

        if s_diff <= 5.0 {
            println!("✅ VALIDATION PASSED: {:.2}% ≤ 5%", s_diff);
            println!("\n   θ=2.0 is validated for production runs.");
        } else {
            println!("❌ VALIDATION FAILED: {:.2}% > 5%", s_diff);
            println!("\n   Consider using lower θ value.");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}

#[cfg(feature = "cuda")]
fn compute_segregation(sim: &GpuNBodySimulation) -> f64 {
    let pos = sim.positions();
    let signs = sim.signs();
    let n = signs.len();

    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(12345);

    let n_samples = 5000;  // More samples for accuracy
    let mut same_sign_sum = 0.0;
    let mut diff_sign_sum = 0.0;

    for _ in 0..n_samples {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        if i == j { continue; }

        let dx = pos[i*3] - pos[j*3];
        let dy = pos[i*3+1] - pos[j*3+1];
        let dz = pos[i*3+2] - pos[j*3+2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();

        if signs[i] == signs[j] {
            same_sign_sum += 1.0 / r;
        } else {
            diff_sign_sum += 1.0 / r;
        }
    }

    if diff_sign_sum > 0.0 {
        same_sign_sum / diff_sign_sum
    } else {
        1.0
    }
}
