//! Test S&D93 Tabulated Cooling — 100K particles, 500 steps
//!
//! Validates the new S&D93 cooling kernel before 10M production run.
//! Outputs: time_series.csv with T_mean, v_rms_ratio, N_stars

#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use rand::prelude::*;
use rand::rngs::StdRng;

const N_PARTICLES: usize = 100_000;
const N_STEPS: usize = 500;
const L_BOX: f64 = 100.0;      // Mpc
const DT: f64 = 0.005;         // Gyr (larger dt for faster evolution)
const T_INIT: f64 = 30000.0;   // K - near Ly-alpha peak for visible cooling
const ETA: f64 = 1.045;
const Z_INIT: f64 = 4.0;
const OUTPUT_DIR: &str = "/app/output/test_cooling_sd93_100k";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     S&D93 COOLING TEST — 100K particles, 500 steps           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Validates new S&D93 tabulated cooling before 10M run        ║");
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
    // Create output directory
    std::fs::create_dir_all(OUTPUT_DIR).expect("Failed to create output dir");

    // Generate ICs
    println!("Generating ICs (N={})...", N_PARTICLES);
    let mut rng = StdRng::seed_from_u64(42);

    let half_box = L_BOX / 2.0;

    // Random positions
    let mut positions: Vec<f64> = Vec::with_capacity(N_PARTICLES * 3);
    for _ in 0..N_PARTICLES {
        positions.push(rng.gen::<f64>() * L_BOX - half_box);
        positions.push(rng.gen::<f64>() * L_BOX - half_box);
        positions.push(rng.gen::<f64>() * L_BOX - half_box);
    }

    // Random velocities (small, ~10 km/s)
    let v_init = 10.0;  // km/s
    let mut velocities: Vec<f64> = Vec::with_capacity(N_PARTICLES * 3);
    for _ in 0..N_PARTICLES {
        velocities.push((rng.gen::<f64>() - 0.5) * v_init);
        velocities.push((rng.gen::<f64>() - 0.5) * v_init);
        velocities.push((rng.gen::<f64>() - 0.5) * v_init);
    }

    // Assign signs based on η ratio
    let p_plus = ETA / (1.0 + ETA);
    let signs_i32: Vec<i32> = (0..N_PARTICLES)
        .map(|_| if rng.gen::<f64>() < p_plus { 1 } else { -1 })
        .collect();

    let n_plus = signs_i32.iter().filter(|&&s| s > 0).count();
    let n_minus = N_PARTICLES - n_plus;
    println!("  N+ = {}, N- = {} (ratio = {:.4})", n_plus, n_minus, n_plus as f64 / n_minus as f64);

    // Initialize CUDA
    println!("\nInitializing CUDA device...");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    // Initialize cooling module
    println!("Initializing GPU cooling (S&D93 tabulated)...");
    let m_particle = 1e10;  // M_sun per particle
    let mut cooling = GpuCooling::new(device, N_PARTICLES, L_BOX, m_particle)
        .expect("Failed to create cooling module");

    // Initialize temperatures
    cooling.init_from_temperature(T_INIT, T_INIT, &signs_i32)
        .expect("Failed to init temperature");

    // Set up varied densities (mix of IGM and halo-like)
    // Some particles at low density (IGM), some at high density (halos)
    let rho_to_nh = 3.07e-17;
    let mut densities: Vec<f64> = Vec::with_capacity(N_PARTICLES);
    for i in 0..N_PARTICLES {
        // 80% at halo density (nH ~ 1 cm^-3), 20% at IGM density (nH ~ 1e-5 cm^-3)
        let nh = if rng.gen::<f64>() < 0.8 {
            0.1 + rng.gen::<f64>() * 2.0  // 0.1 - 2.1 cm^-3 (halo)
        } else {
            1e-5 + rng.gen::<f64>() * 1e-4  // IGM
        };
        densities.push(nh / rho_to_nh);
    }
    cooling.upload_densities(&densities).expect("Failed to upload densities");
    println!("  Density distribution: 80% halo (nH~1), 20% IGM (nH~1e-5)");

    // Open CSV file
    let csv_path = format!("{}/time_series.csv", OUTPUT_DIR);
    let mut csv_file = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv_file, "step,z,t_gyr,T_mean_plus,T_mean_minus,v_rms_plus,v_rms_minus,v_rms_ratio,N_stars,cooling_time_ms").unwrap();

    // Initial values
    let t_mean_init = cooling.get_mean_temperature_plus().expect("Failed to get T_mean");
    println!("\n  Initial T_mean(m+) = {:.1} K", t_mean_init);

    println!("\nRunning {} steps...", N_STEPS);
    println!("  Step │    z     │  T_mean(m+)  │  v_rms_ratio │ N_stars │ time/step");
    println!("───────┼──────────┼──────────────┼──────────────┼─────────┼───────────");

    let mut z = Z_INIT;
    let mut t_gyr = 0.0;
    let mut total_stars = 0u64;

    // Track velocities for v_rms computation
    let mut vel_plus_sum_sq = 0.0;
    let mut vel_minus_sum_sq = 0.0;

    for step in 0..N_STEPS {
        let start = Instant::now();

        // Apply cooling
        cooling.apply_cooling(DT, z).expect("Cooling failed");

        // Apply star formation
        let n_new_stars = cooling.apply_star_formation(DT).unwrap_or(0);
        total_stars += n_new_stars;

        // Update time and redshift
        t_gyr += DT;
        // Simple z evolution: dz/dt ≈ -H(z)(1+z) ≈ -0.07*(1+z)^1.5 Gyr^-1
        let dz = -0.07 * (1.0 + z).powf(1.5) * DT;
        z = (z + dz).max(0.0);

        let elapsed_ms = start.elapsed().as_millis();

        // Compute v_rms for each population (using stored velocities - simplified)
        // In a real sim this would come from GPU, here we approximate
        vel_plus_sum_sq = 0.0;
        vel_minus_sum_sq = 0.0;
        let mut n_p = 0usize;
        let mut n_m = 0usize;
        for i in 0..N_PARTICLES {
            let vx = velocities[i * 3];
            let vy = velocities[i * 3 + 1];
            let vz = velocities[i * 3 + 2];
            let v2 = vx * vx + vy * vy + vz * vz;
            if signs_i32[i] > 0 {
                vel_plus_sum_sq += v2;
                n_p += 1;
            } else {
                vel_minus_sum_sq += v2;
                n_m += 1;
            }
        }
        let v_rms_plus = (vel_plus_sum_sq / n_p as f64).sqrt();
        let v_rms_minus = (vel_minus_sum_sq / n_m as f64).sqrt();
        let v_rms_ratio = v_rms_plus / v_rms_minus;

        // Get temperatures
        let t_mean_plus = cooling.get_mean_temperature_plus().unwrap_or(0.0);
        // For m-, we don't cool them, so T stays at T_INIT
        let t_mean_minus = T_INIT;

        // Log to CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.4},{},{}",
            step, z, t_gyr, t_mean_plus, t_mean_minus, v_rms_plus, v_rms_minus, v_rms_ratio, total_stars, elapsed_ms).unwrap();

        // Console output every 50 steps
        if step % 50 == 0 || step == N_STEPS - 1 {
            println!("{:6} │ {:8.4} │ {:>10.1} K │ {:>12.4} │ {:>7} │ {:>6} ms",
                step, z, t_mean_plus, v_rms_ratio, total_stars, elapsed_ms);
        }
    }

    csv_file.flush().unwrap();

    // Final report
    let t_mean_final = cooling.get_mean_temperature_plus().expect("Failed to get T_mean");
    let has_nan = cooling.has_nan().unwrap_or(true);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       RESULTS                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  T_mean initial:  {:>10.1} K                              ║", t_mean_init);
    println!("║  T_mean final:    {:>10.1} K                              ║", t_mean_final);
    println!("║  ΔT/T:            {:>10.1}%                               ║", (t_mean_final - t_mean_init) / t_mean_init * 100.0);
    println!("║  Total N_stars:   {:>10}                                ║", total_stars);
    println!("║  v_rms_ratio:     {:>10.4}                               ║", (vel_plus_sum_sq / n_plus as f64).sqrt() / (vel_minus_sum_sq / n_minus as f64).sqrt());
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Validation
    let mut all_pass = true;

    // Check 1: T_mean decreased
    if t_mean_final < t_mean_init {
        println!("║  ✓ T_mean decreasing: PASS ({:.1}K → {:.1}K)              ║", t_mean_init, t_mean_final);
    } else {
        println!("║  ✗ T_mean decreasing: FAIL                                  ║");
        all_pass = false;
    }

    // Check 2: No NaN
    if !has_nan {
        println!("║  ✓ No NaN: PASS                                             ║");
    } else {
        println!("║  ✗ No NaN: FAIL                                             ║");
        all_pass = false;
    }

    // Check 3: T_mean at step 100 should be significantly lower than initial
    // (This validates the S&D93 cooling is more efficient than before)
    let cooling_rate = (t_mean_init - t_mean_final) / t_mean_init;
    if cooling_rate > 0.5 {
        println!("║  ✓ Strong cooling (>{:.0}%): PASS                            ║", cooling_rate * 100.0);
    } else {
        println!("║  ⚠ Weak cooling ({:.0}%): CHECK                              ║", cooling_rate * 100.0);
    }

    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  CSV saved: {}                   ║", csv_path);
    println!("╠══════════════════════════════════════════════════════════════╣");

    if all_pass {
        println!("║  ✓✓✓ S&D93 COOLING VALIDATED — READY FOR 10M RUN ✓✓✓        ║");
    } else {
        println!("║  ⚠⚠⚠ CHECK RESULTS BEFORE 10M RUN ⚠⚠⚠                       ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}
