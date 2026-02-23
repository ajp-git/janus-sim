/// Force comparison test: GPU tree vs CPU tree
/// Validates force computation accuracy

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);

        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);
        let dt = 0.01;

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   Force Comparison: GPU Tree vs CPU Tree                       ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Particles: {}", n_particles);

        // Create simulation
        let mut sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");

        // Compare forces using debug method
        sim.compare_forces_debug().expect("Force comparison failed");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support. Build with --features cuda");
    }
}
