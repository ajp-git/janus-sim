/// GPU Barnes-Hut 10M particle simulation
/// Phase 1c validation for Janus cosmological model

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("{}", "=".repeat(70));
        println!("Janus N-body GPU Simulation — 10M Particles");
        println!("{}", "=".repeat(70));

        // Parse command line for steps override
        let args: Vec<String> = std::env::args().collect();
        let steps: usize = args.iter()
            .position(|a| a == "--steps")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        // Parse particle count from command line
        let n_particles: usize = args.iter()
            .position(|a| a == "--n")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000_000);  // Default to 1M

        // Parse eta from command line (default 1.045 from Friedmann fit)
        let eta: f64 = args.iter()
            .position(|a| a == "--eta")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.045);
        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;
        // Scale box size with particle count (100 for 100K, 1000 for 10M)
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

        // Parse dt from command line (default 0.001)
        let dt: f64 = args.iter()
            .position(|a| a == "--dt")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.001);

        println!("\nParameters:");
        println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
        println!("  eta = {:.3}", eta);
        println!("  box = {}", box_size);
        println!("  steps = {}", steps);
        println!("  dt = {}", dt);

        println!("\n--- Initializing GPU simulation ---");
        let init_start = Instant::now();

        match GpuNBodySimulation::new(n_positive, n_negative, box_size) {
            Ok(mut gpu_sim) => {
                let init_time = init_start.elapsed();
                println!("Initialization time: {:.2?}", init_time);

                let ke0 = gpu_sim.kinetic_energy().unwrap();
                let seg0 = gpu_sim.segregation_distance().unwrap();

                println!("\nInitial state:");
                println!("  KE = {:.4e}", ke0);
                println!("  Segregation = {:.4}", seg0);

                println!("\n{:>6}  {:>12}  {:>12}  {:>12}", "Step", "KE/KE0", "Segregation", "Step time");
                println!("{:-<55}", "");

                let sim_start = Instant::now();

                for step in 1..=steps {
                    let step_start = Instant::now();

                    if let Err(e) = gpu_sim.step(dt) {
                        eprintln!("GPU error at step {}: {}", step, e);
                        break;
                    }

                    let step_time = step_start.elapsed();

                    if step % 10 == 0 || step == steps {
                        let ke = gpu_sim.kinetic_energy().unwrap();
                        let seg = gpu_sim.segregation_distance().unwrap();
                        let ke_ratio = ke / ke0;
                        println!("{:>6}  {:>12.2}  {:>12.4}  {:>12.2?}", step, ke_ratio, seg, step_time);
                    }
                }

                let total_time = sim_start.elapsed();
                let ke_final = gpu_sim.kinetic_energy().unwrap();
                let seg_final = gpu_sim.segregation_distance().unwrap();
                let ke_ratio = ke_final / ke0;
                let seg_change = (seg_final - seg0) / seg0 * 100.0;

                println!("\n{}", "=".repeat(70));
                println!("RESULTS");
                println!("{}", "=".repeat(70));
                println!("\nPerformance:");
                println!("  Total time: {:.2?}", total_time);
                println!("  Time/step: {:.2?}", total_time / steps as u32);
                println!("  Steps/sec: {:.2}", steps as f64 / total_time.as_secs_f64());

                println!("\nPhysics:");
                println!("  Initial KE: {:.4e}", ke0);
                println!("  Final KE: {:.4e}", ke_final);
                println!("  KE/KE0: {:.2}", ke_ratio);

                println!("\nSegregation:");
                println!("  Initial: {:.4}", seg0);
                println!("  Final: {:.4}", seg_final);
                println!("  Change: {:+.2}%", seg_change);

                println!("\n{}", "=".repeat(70));
                if ke_ratio < 50.0 {
                    println!("ENERGY: STABLE (KE/KE0 = {:.2} < 50)", ke_ratio);
                } else {
                    println!("ENERGY: UNSTABLE (KE/KE0 = {:.2} >= 50)", ke_ratio);
                }
                println!("{}", "=".repeat(70));
            }
            Err(e) => {
                eprintln!("Failed to initialize GPU simulation: {}", e);
                eprintln!("Make sure CUDA is available and GPU has enough memory (~12GB)");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with:");
        println!("  cargo run --release --features cuda --bin nbody_gpu_10m");
    }
}
