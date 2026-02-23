/// Debug BVH structure

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n_particles = 1000;
        let eta = 1.045;
        let box_size = 100.0 * (n_particles as f64 / 100_000.0).powf(1.0/3.0);

        let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
        let n_negative = n_particles - n_positive;

        println!("Creating simulation with {} particles", n_particles);

        let mut sim = GpuNBodySimulation::new(
            n_positive, n_negative, box_size
        ).expect("Failed to create GPU simulation");

        // Debug BVH structure
        sim.debug_bvh_structure().expect("BVH debug failed");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This binary requires CUDA support");
    }
}
