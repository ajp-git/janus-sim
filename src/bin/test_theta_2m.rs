/// Test Barnes-Hut theta at 2M particles with physics validation

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 2_000_000;
        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        let dt = 0.003;
        let n_steps = 20;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Barnes-Hut Theta Test @ 2M particles                         ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        // First, get reference S(t) at theta=0.7 (accurate)
        println!("Reference run at theta=0.7 (accurate)...");
        let s_ref = {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed");
            sim.set_theta(0.7);
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup");

            let t = Instant::now();
            for _ in 0..n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step");
            }
            let avg = t.elapsed().as_secs_f64() / n_steps as f64 * 1000.0;
            let s = compute_segregation(&sim);
            println!("  S({}) = {:.6}, {:.1} ms/step\n", n_steps, s, avg);
            s
        };

        // Test higher theta values
        let theta_values = [0.9, 1.0, 1.2, 1.5, 2.0];

        println!("Testing higher theta values:\n");
        println!("{:<8} {:>12} {:>12} {:>12} {:>10}", "Theta", "Time (ms)", "S(t)", "S diff %", "Target?");
        println!("{}", "─".repeat(60));

        for &theta in &theta_values {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)
                .expect("Failed");
            sim.set_theta(theta);
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Warmup");

            let t = Instant::now();
            for _ in 0..n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).expect("Step");
            }
            let avg = t.elapsed().as_secs_f64() / n_steps as f64 * 1000.0;
            let s = compute_segregation(&sim);
            let s_diff = ((s - s_ref) / s_ref * 100.0).abs();
            let target = if avg < 250.0 { "✓" } else { "" };

            println!("{:<8.1} {:>12.1} {:>12.6} {:>12.2} {:>10}",
                     theta, avg, s, s_diff, target);
        }

        println!("\nTarget: <250 ms/step with <5% physics error");
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

    let n_samples = 1000;
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
