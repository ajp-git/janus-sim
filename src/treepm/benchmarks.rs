//! Benchmarks for TreePM-Janus pipeline (CPU-side).
//!
//! Run with: `cargo test --release --lib treepm::benchmarks -- --ignored --nocapture`
//!
//! GPU benchmarks (full §8.1 scaling, nvprof profiling) require Phase 5
//! GPU integration of the corrected pipeline (gradient ord 4, CIC W^-2,
//! Janus cross-coupling) into nbody_gpu_twopass.rs. These benchmarks here
//! validate CPU baseline performance and memory predictions.

#[cfg(test)]
mod tests {
    use super::super::gpu_layout::ParticleArrays;
    use super::super::integrated_step::step_treepm_janus_pm_only;
    use super::super::janus::{JanusCoupling, JanusState};
    use super::super::pm_grid::PmGrid;
    use std::time::Instant;

    fn newton_state() -> JanusState {
        JanusState {
            a_plus: 1.0,
            a_minus: 1.0,
            h_plus: 0.0,
            h_minus: 0.0,
            coupling: JanusCoupling {
                phi: 1.0,
                c_ratio_sq: 1.0,
                repulsion_scale: 1.0,
            },
        }
    }

    fn random_particles(n: usize, box_size: f64, seed: u64) -> ParticleArrays {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p = ParticleArrays::new(n);
        let half = box_size / 2.0;
        for i in 0..n {
            p.pos_x[i] = rng.random::<f64>() * box_size - half;
            p.pos_y[i] = rng.random::<f64>() * box_size - half;
            p.pos_z[i] = rng.random::<f64>() * box_size - half;
            p.sign[i] = if i < n / 2 { 1 } else { -1 };
            p.mass[i] = 1.0;
        }
        p
    }

    /// CPU benchmark: integrated step at N=100 (smoke test).
    /// Should complete in seconds.
    #[ignore]
    #[test]
    fn benchmark_integrated_step_100() {
        let n = 100;
        let box_size = 100.0;
        let n_pm = 32;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = random_particles(n, box_size, 42);
        let state = newton_state();
        let dt = 0.001;

        let n_steps = 10;
        let t0 = Instant::now();
        for _ in 0..n_steps {
            step_treepm_janus_pm_only(&mut particles, &mut pm, &state, dt, box_size, 1.0);
        }
        let total_s = t0.elapsed().as_secs_f64();
        let per_step = total_s / n_steps as f64;
        println!("N={}, n_pm={}, {} steps in {:.3}s ({:.3}s/step)",
                 n, n_pm, n_steps, total_s, per_step);
    }

    /// CPU benchmark: integrated step at N=1K. Stresses CPU CIC + FFT.
    #[ignore]
    #[test]
    fn benchmark_integrated_step_1k() {
        let n = 1000;
        let box_size = 250.0;
        let n_pm = 64;
        let mut pm = PmGrid::new(n_pm, box_size);
        let mut particles = random_particles(n, box_size, 42);
        let state = newton_state();
        let dt = 0.001;

        let n_steps = 5;
        let t0 = Instant::now();
        for _ in 0..n_steps {
            step_treepm_janus_pm_only(&mut particles, &mut pm, &state, dt, box_size, 1.0);
        }
        let total_s = t0.elapsed().as_secs_f64();
        let per_step = total_s / n_steps as f64;
        println!("N={}, n_pm={}, {} steps in {:.3}s ({:.3}s/step)",
                 n, n_pm, n_steps, total_s, per_step);
    }

    /// Memory footprint check: ParticleArrays at N=1M.
    /// Plan §3.0: SoA = 53 B/particle → 53 MB total.
    #[test]
    fn measure_memory_1m_particles() {
        let n = 1_000_000;
        let p = ParticleArrays::new(n);
        let bytes = p.memory_bytes();
        let mb = bytes as f64 / (1024.0 * 1024.0);
        println!("ParticleArrays N=1M: {} bytes ({:.2} MB)", bytes, mb);
        // Plan §1.4 predicts ~20-50 MB for 1M particles SoA layout.
        // Our 53 B/particle = 53 MB.
        assert!(mb < 60.0, "Memory exceeds 60 MB: {}", mb);
        assert!(mb > 30.0, "Memory suspiciously low: {} (expected ~53 MB)", mb);
    }

    /// PM grid memory at N_pm=256, 512.
    #[test]
    fn measure_pm_grid_memory() {
        // PmGrid: 4 grids × n³ × f64 = 32 × n³ bytes
        // N_pm=256: 4 × 16,777,216 × 8 = 537 MB
        let pm_256 = PmGrid::new(256, 1000.0);
        let mb_256 = pm_256.memory_bytes() as f64 / (1024.0 * 1024.0);
        println!("PmGrid N_pm=256: {:.1} MB", mb_256);
        assert!(mb_256 > 400.0 && mb_256 < 700.0);

        // N_pm=512: 4 × 134M × 8 = 4.3 GB (tight on RTX 3060 12GB)
        // Don't actually allocate to keep test fast; just verify formula.
        let predicted_512 = 4 * 512_usize.pow(3) * 8;
        let mb_512 = predicted_512 as f64 / (1024.0 * 1024.0);
        println!("PmGrid N_pm=512 predicted: {:.1} MB", mb_512);
        assert!(mb_512 > 4000.0); // > 4 GB
    }

    /// Sanity: ParticleArrays scales linearly with N.
    #[test]
    fn memory_scales_linear_with_n() {
        let p_100k = ParticleArrays::new(100_000);
        let p_1m = ParticleArrays::new(1_000_000);
        let ratio = p_1m.memory_bytes() as f64 / p_100k.memory_bytes() as f64;
        // Tolerance: within 1% of 10×
        assert!(
            (ratio - 10.0).abs() < 0.1,
            "Memory ratio = {} (expected 10)",
            ratio
        );
    }
}
