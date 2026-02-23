#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let n = 50000;
        let box_size = 400.0;

        println!("Generating {} positions (centered)...", n);
        let mut rng = StdRng::seed_from_u64(42);
        // Centered around 0 like new() does
        let positions: Vec<f64> = (0..n*3).map(|_| (rng.random::<f64>() - 0.5) * box_size).collect();
        let velocities: Vec<f64> = vec![0.0; n*3];
        let signs: Vec<i32> = (0..n).map(|i| if i < n/2 { 1 } else { -1 }).collect();

        println!("Testing new_with_state({}, {}, {})...", n/2, n/2, box_size);
        match GpuNBodySimulation::new_with_state(n/2, n/2, box_size, positions, velocities, signs) {
            Ok(_sim) => println!("SUCCESS!"),
            Err(e) => println!("FAILED: {}", e),
        }
    }
}
