/// Quick scan of theta values to find best balance (100 steps each)

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
        let n_steps = 100;

        println!("Quick θ scan: 500K particles, 100 steps each\n");

        // Reference θ=0.5
        let s_ref = {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size).unwrap();
            sim.set_theta(0.5);
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).unwrap();
            for _ in 0..n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).unwrap();
            }
            compute_segregation(&sim)
        };
        println!("Reference (θ=0.5): S(100) = {:.6}\n", s_ref);

        // Test theta values
        let thetas = [0.7, 1.0, 1.2, 1.5, 1.7, 2.0];

        println!("{:<8} {:>12} {:>12} {:>10}", "θ", "S(100)", "Error %", "Target?");
        println!("{}", "─".repeat(50));

        for &theta in &thetas {
            let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size).unwrap();
            sim.set_theta(theta);
            sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).unwrap();

            let t = Instant::now();
            for _ in 0..n_steps {
                sim.step_with_expansion_dkd_gpu(dt, 1.0, 0.0, 0.0).unwrap();
            }
            let time_ms = t.elapsed().as_secs_f64() / n_steps as f64 * 1000.0;

            let s = compute_segregation(&sim);
            let err = ((s - s_ref) / s_ref * 100.0).abs();
            let ok = if err <= 5.0 { "✓" } else { "" };

            println!("{:<8.1} {:>12.6} {:>12.2} {:>10}", theta, s, err, ok);
        }

        println!("\nChoose θ with error ≤5% for full 500-step validation.");
    }

    #[cfg(not(feature = "cuda"))]
    eprintln!("Requires CUDA");
}

#[cfg(feature = "cuda")]
fn compute_segregation(sim: &GpuNBodySimulation) -> f64 {
    let pos = sim.positions();
    let signs = sim.signs();
    let n = signs.len();
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let n_samples = 3000;
    let (mut same, mut diff) = (0.0, 0.0);
    for _ in 0..n_samples {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        if i == j { continue; }
        let dx = pos[i*3] - pos[j*3];
        let dy = pos[i*3+1] - pos[j*3+1];
        let dz = pos[i*3+2] - pos[j*3+2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        if signs[i] == signs[j] { same += 1.0/r; } else { diff += 1.0/r; }
    }
    if diff > 0.0 { same / diff } else { 1.0 }
}
