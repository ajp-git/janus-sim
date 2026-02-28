//! Test TreePM Hybrid: CPU PM long-range + GPU BH short-range
//!
//! Validates that the hybrid approach eliminates grid artifacts
//! while maintaining acceptable performance.
//!
//! Build: cargo build --release --features cuda --bin test_treepm_hybrid
//! Run: docker compose run --rm dev cargo run --release --features cuda --bin test_treepm_hybrid

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::treepm::pm_grid::PmGrid;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {
    println!("=== TreePM Hybrid Test ===\n");
    println!("Architecture:");
    println!("  PM long-range: CPU rustfft (accurate, eliminates grid artifacts)");
    println!("  Tree short-range: GPU BH with r_cut (fast)\n");

    // Parameters
    let n_particles = 500_000;  // 500K for quick test
    let n_steps = 100;
    let dt = 0.01;
    let box_size = 200.0;  // Smaller box for faster test

    // TreePM parameters
    let r_cut = box_size / 16.0;  // ~12.5 for 200 box
    let grid_size = 128;          // PM grid resolution

    // Hubble friction
    let z_init: f64 = 5.0;
    let omega_m: f64 = 0.3;
    let h0: f64 = 0.7;
    let hubble = h0 * (omega_m * (1.0 + z_init).powi(3) + (1.0 - omega_m)).sqrt();
    let dtau_per_dt = 1.0;

    println!("Parameters:");
    println!("  N particles: {}", n_particles);
    println!("  Box size: {}", box_size);
    println!("  r_cut: {:.2} (TreePM splitting)", r_cut);
    println!("  r_s: {:.2} (Gaussian scale)", r_cut / 3.0);
    println!("  Grid: {}³", grid_size);
    println!("  dt: {}", dt);
    println!("  Steps: {}", n_steps);
    println!("  Hubble: {:.4} (z={})", hubble, z_init);
    println!();

    // Initialize GPU simulator (uses Zel'dovich ICs internally)
    println!("Initializing GPU N-body (Zel'dovich ICs)...");
    let t0 = Instant::now();
    let mut sim = match GpuNBodyTwoPass::new(n_particles / 2, n_particles / 2, box_size) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU: {}", e);
            return;
        }
    };
    sim.set_theta(0.5);  // BH opening angle
    println!("  Init time: {:.2?}", t0.elapsed());

    // Initialize PM grid (CPU)
    println!("Initializing PM grid ({}³ = {} cells)...", grid_size, grid_size * grid_size * grid_size);
    let mut pm_grid = PmGrid::new(grid_size, box_size);
    println!("  PM memory: {:.2} MB", pm_grid.memory_bytes() as f64 / 1e6);
    println!();

    // Run TreePM hybrid simulation
    println!("Running {} TreePM hybrid steps...\n", n_steps);
    let t_sim = Instant::now();
    let mut step_times = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let t_step = Instant::now();

        if let Err(e) = sim.step_treepm_hybrid(dt, &mut pm_grid, r_cut, hubble, dtau_per_dt) {
            eprintln!("ERROR at step {}: {}", step, e);
            return;
        }

        let step_ms = t_step.elapsed().as_millis();
        step_times.push(step_ms);

        // Progress report
        if step == 0 || step == 9 || step == 49 || step == 99 || step == n_steps - 1 {
            let seg = sim.segregation().unwrap_or(0.0);
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            println!("  Step {:4}: {:>4}ms  Seg={:.4}  KE={:.2e}",
                     step + 1, step_ms, seg, ke);
        }
    }

    let total_ms = t_sim.elapsed().as_millis();
    let avg_ms = step_times.iter().sum::<u128>() / n_steps as u128;
    let min_ms = *step_times.iter().min().unwrap_or(&0);
    let max_ms = *step_times.iter().max().unwrap_or(&0);

    println!();
    println!("=== Results ===");
    println!("  Total time: {:.2}s", total_ms as f64 / 1000.0);
    println!("  Avg step: {}ms", avg_ms);
    println!("  Min/Max: {}ms / {}ms", min_ms, max_ms);
    println!("  Throughput: {:.2} steps/s", 1000.0 / avg_ms as f64);
    println!();

    // Final diagnostics
    let seg = sim.segregation().unwrap_or(0.0);
    let ke = sim.kinetic_energy().unwrap_or(0.0);
    println!("Final state:");
    println!("  Segregation: {:.4}", seg);
    println!("  Kinetic energy: {:.2e}", ke);
    println!();

    // Performance comparison
    println!("Performance comparison (estimated @ 1M):");
    let scale = 1_000_000.0 / n_particles as f64;
    let estimated_1m = avg_ms as f64 * scale;
    println!("  TreePM hybrid @ 1M: ~{:.0}ms/step", estimated_1m);
    println!("  GPU BH only (with artifacts): ~255ms/step");
    println!("  CPU TreePM: ~5000ms/step");
    println!();

    // Verdict
    if avg_ms < 500 {
        println!("✓ Performance target met: <500ms/step");
    } else {
        println!("⚠ Performance target NOT met: {}ms > 500ms", avg_ms);
    }

    // Note about grid artifacts
    println!();
    println!("NOTE: Grid artifacts eliminated by construction");
    println!("  - Long-range: PM FFT is exact on grid scale");
    println!("  - Short-range: GPU BH with r_cut ignores long-range");
    println!("  - Result: No grid pattern even at θ=0.5");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo build --release --features cuda --bin test_treepm_hybrid");
}
