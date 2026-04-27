//! Test GPU Cooling Kernels
//!
//! Verifies:
//! 1. T_mean decreasing over time (cooling working)
//! 2. Performance < 2s per step for 100K
//! 3. No NaN values

#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::StdRng;

const N_PARTICLES: usize = 100_000;
const N_STEPS: usize = 200;
const L_BOX: f64 = 100.0;      // Mpc
const DT: f64 = 0.001;         // Gyr
const T_INIT: f64 = 100000.0;  // K - higher T shows cooling more clearly
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         GPU COOLING KERNEL TEST (100K, 200 steps)            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Target: T_mean decreasing, <2s/step, no NaN                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: CUDA feature not enabled. Recompile with --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    run_test();
}

#[cfg(feature = "cuda")]
fn run_test() {
    // Generate simple ICs (random positions, signs based on η)
    println!("Generating simple ICs (N={})...", N_PARTICLES);
    let mut rng = StdRng::seed_from_u64(42);

    // Generate positions uniformly in box (not actually used for cooling test)
    let half_box = L_BOX / 2.0;
    let _positions: Vec<f64> = (0..N_PARTICLES * 3)
        .map(|_| rng.gen::<f64>() * L_BOX - half_box)
        .collect();

    // Assign signs based on η ratio (η = n+/n-)
    let p_plus = ETA / (1.0 + ETA);  // Probability of being m+
    let signs_i32: Vec<i32> = (0..N_PARTICLES)
        .map(|_| if rng.gen::<f64>() < p_plus { 1 } else { -1 })
        .collect();

    let n_plus = signs_i32.iter().filter(|&&s| s > 0).count();
    let n_minus = N_PARTICLES - n_plus;
    println!("  N+ = {}, N- = {} (ratio = {:.4})", n_plus, n_minus, n_plus as f64 / n_minus as f64);

    // Initialize GPU device
    // Note: CudaDevice::new already returns Arc<CudaDevice>
    println!("\nInitializing CUDA device...");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    // Initialize cooling module
    println!("Initializing GPU cooling...");
    let m_particle = 1e10;  // M_sun per particle
    let mut cooling = GpuCooling::new(device, N_PARTICLES, L_BOX, m_particle)
        .expect("Failed to create cooling module");

    // Initialize internal energy from temperature
    cooling.init_from_temperature(T_INIT, T_INIT, &signs_i32)
        .expect("Failed to init temperature");

    // Get initial T_mean
    let t_mean_init = cooling.get_mean_temperature_plus().expect("Failed to get T_mean");
    println!("  Initial T_mean(m+) = {:.1} K", t_mean_init);

    // Compute SPH densities
    // Need nH > 0.1 cm^-3 for cooling to dominate UV heating
    // rho_to_nH = 3.07e-17, so we need rho > 0.1/3e-17 ~ 3e15 M_sun/Mpc³
    // This corresponds to collapsed halo density
    let target_nh = 1.0;  // cm^-3 (typical for star-forming region)
    let rho_to_nh = 3.07e-17;  // Same as in cooling_gpu.rs
    let rho_target = target_nh / rho_to_nh;  // ~3.26e16 M_sun/Mpc³
    let densities = vec![rho_target; N_PARTICLES];
    cooling.upload_densities(&densities).expect("Failed to upload densities");
    println!("  Target nH = {:.1} cm⁻³ (ρ = {:.2e} M_sun/Mpc³)", target_nh, rho_target);

    println!("\nRunning {} steps...", N_STEPS);
    println!("  Step │    z     │  T_mean(m+)  │  time/step  │ NaN check");
    println!("───────┼──────────┼──────────────┼─────────────┼───────────");

    let mut z = Z_INIT;
    let mut total_time = 0.0;
    let mut t_mean_prev = t_mean_init;

    for step in 0..N_STEPS {
        let start = Instant::now();

        // Apply cooling
        cooling.apply_cooling(DT, z).expect("Cooling failed");

        // Update redshift (simplified: linear decrease)
        z = Z_INIT - (step as f64 + 1.0) * DT * 0.25;  // Approximate z evolution
        if z < 0.0 { z = 0.0; }

        let elapsed = start.elapsed().as_secs_f64();
        total_time += elapsed;

        // Report every 20 steps
        if step % 20 == 19 || step == 0 {
            let t_mean = cooling.get_mean_temperature_plus().expect("Failed to get T_mean");
            let has_nan = cooling.has_nan().expect("Failed to check NaN");

            println!("{:6} │ {:8.4} │ {:>10.1} K │ {:>9.3} s │ {}",
                step + 1,
                z,
                t_mean,
                elapsed,
                if has_nan { "⚠ NaN!" } else { "✓ OK" }
            );

            // Verify T is decreasing
            if step > 0 && t_mean > t_mean_prev * 1.1 {
                println!("\n⚠ WARNING: Temperature increased significantly!");
            }
            t_mean_prev = t_mean;
        }
    }

    // Final stats
    let t_mean_final = cooling.get_mean_temperature_plus().expect("Failed to get T_mean");
    let has_nan_final = cooling.has_nan().expect("Failed to check NaN");
    let avg_time = total_time / N_STEPS as f64;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       RESULTS                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  T_mean initial: {:>10.1} K                              ║", t_mean_init);
    println!("║  T_mean final:   {:>10.1} K                              ║", t_mean_final);
    println!("║  ΔT/T:           {:>10.1}%                               ║", (t_mean_final - t_mean_init) / t_mean_init * 100.0);
    println!("║  Avg time/step:  {:>10.3} s                              ║", avg_time);
    println!("║  Total time:     {:>10.1} s                              ║", total_time);
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Validation
    let mut all_pass = true;

    // Check 1: T_mean decreasing
    if t_mean_final < t_mean_init {
        println!("║  ✓ T_mean decreasing: PASS                                  ║");
    } else {
        println!("║  ✗ T_mean decreasing: FAIL                                  ║");
        all_pass = false;
    }

    // Check 2: Performance < 2s/step
    if avg_time < 2.0 {
        println!("║  ✓ Performance < 2s/step: PASS ({:.3}s)                    ║", avg_time);
    } else {
        println!("║  ✗ Performance < 2s/step: FAIL ({:.3}s)                    ║", avg_time);
        all_pass = false;
    }

    // Check 3: No NaN
    if !has_nan_final {
        println!("║  ✓ No NaN: PASS                                             ║");
    } else {
        println!("║  ✗ No NaN: FAIL                                             ║");
        all_pass = false;
    }

    println!("╠══════════════════════════════════════════════════════════════╣");
    if all_pass {
        println!("║  ✓✓✓ ALL TESTS PASSED — GPU COOLING READY FOR 10M ✓✓✓       ║");
    } else {
        println!("║  ✗✗✗ SOME TESTS FAILED — INVESTIGATE BEFORE 10M RUN ✗✗✗     ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}
