/// Profile GPU tree build stages

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

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║   GPU Tree Build Profiling @ 2M particles                      ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        let mut sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");

        // Warm up
        sim.build_gpu_tree_profiled().expect("Warm-up failed");

        // Profile 5 iterations
        println!("Running 5 profiled builds...\n");
        for i in 1..=5 {
            println!("--- Build {} ---", i);
            sim.build_gpu_tree_profiled().expect("Build failed");
            println!();
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
