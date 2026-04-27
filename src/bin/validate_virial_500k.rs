//! Validation run: 500K particles with virial_factor=0.5
//!
//! Criteria:
//!   - KE/KE₀ < 10 at step 100
//!   - Segregation onset between step 800-1500 (z≈2.4-3.0)
//!   - Seg increasing, no brutal collapse

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_PARTICLES: usize = 500_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 2000;
const VIRIAL_FACTOR: f64 = 0.5;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Validation Run: 500K particles, virial_factor=0.5            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);
    let r_cut = box_size / 16.0;

    println!("Parameters:");
    println!("  N = {} ({:.1}K)", N_PARTICLES, N_PARTICLES as f64 / 1e3);
    println!("  η = {}", ETA);
    println!("  θ = {}", THETA);
    println!("  virial_factor = {}", VIRIAL_FACTOR);
    println!("  box = {:.2} Mpc", box_size);
    println!("  r_cut = {:.2} Mpc", r_cut);
    println!("  steps = {}", TOTAL_STEPS);
    println!();

    // Cosmological setup
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / 12000.0;  // Same rate as full run
    let dtau_per_dt = dtau_cosmo / DT;

    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("Cosmology:");
    println!("  z_init = {:.2}", z_init_actual);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
    let output_dir = format!("/app/output/validate_virial_{}", date);
    fs::create_dir_all(&output_dir)?;

    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv_file = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;

    println!("Output: {}", output_dir);
    println!();

    // Create simulation
    println!("Creating simulation with virial_factor={}...", VIRIAL_FACTOR);
    let mut sim = GpuNBodyTwoPass::new_with_virial_factor(n_positive, n_negative, box_size, VIRIAL_FACTOR)?;
    sim.set_theta(THETA);

    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", seg0);
    println!();

    // Tracking
    let start_time = Instant::now();
    let mut step = 0usize;
    let mut current_tau = cosmo.tau_start;
    let mut s_max = 0.0f64;
    let mut s_max_step = 0usize;
    let mut onset_step: Option<usize> = None;
    let mut onset_z: Option<f64> = None;

    // Validation flags
    let mut ke_check_passed = false;
    let mut onset_in_range = false;

    println!("Running {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE₀     Seg      ms/step");
    println!("----------------------------------------------");

    loop {
        let step_start = Instant::now();

        let (a, h) = if current_tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(current_tau)
        } else {
            (1.0, 0.0)
        };
        let z = 1.0 / a - 1.0;

        let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };

        sim.step_treepm_gpu(DT, r_cut, h, dtau_eff)?;
        step += 1;
        current_tau += dtau_cosmo;

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        let ke_ratio = ke / ke0;

        // Track S_max and onset
        if seg > s_max {
            if onset_step.is_none() && seg > 0.01 && step > 100 {
                onset_step = Some(step);
                onset_z = Some(z.max(0.0));
            }
            s_max = seg;
            s_max_step = step;
        }

        // Check KE at step 100
        if step == 100 {
            ke_check_passed = ke_ratio < 10.0;
            println!(">>> Step 100 check: KE/KE₀ = {:.2} (< 10? {})",
                ke_ratio, if ke_check_passed { "✓" } else { "✗" });
        }

        // Print progress
        if step % 100 == 0 || step <= 10 {
            println!("{:5}  {:.2}  {:8.2}  {:6.4}  {:6.0}",
                step, z.max(0.0), ke_ratio, seg, step_ms);
        }

        // CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.1}",
            step, step as f64 * DT, z.max(0.0), a, h, ke, ke_ratio, seg, step_ms)?;

        if step % 100 == 0 {
            csv_file.flush()?;
        }

        // Stop conditions
        if step >= TOTAL_STEPS {
            break;
        }

        // Early stop if KE explodes
        if ke_ratio > 1000.0 {
            println!("\n⚠️ KE/KE₀ > 1000 — COLLAPSE DETECTED");
            break;
        }
    }

    csv_file.flush()?;

    // Check onset range
    if let Some(os) = onset_step {
        onset_in_range = os >= 800 && os <= 1500;
    }

    let total_time = start_time.elapsed();

    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   VALIDATION RESULTS                                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Runtime: {:.1} minutes", total_time.as_secs_f64() / 60.0);
    println!("Average: {:.0} ms/step", total_time.as_secs_f64() * 1000.0 / step as f64);
    println!();
    println!("Results:");
    println!("  Final step: {}", step);
    println!("  Final KE/KE₀: {:.2}", sim.kinetic_energy()? / ke0);
    println!("  S_max = {:.4} at step {}", s_max, s_max_step);
    if let Some(os) = onset_step {
        println!("  Onset: step {} (z = {:.2})", os, onset_z.unwrap_or(0.0));
    } else {
        println!("  Onset: not detected");
    }
    println!();
    println!("VALIDATION CRITERIA:");
    println!("  [{}] KE/KE₀ < 10 at step 100", if ke_check_passed { "✓" } else { "✗" });
    println!("  [{}] Onset between step 800-1500",
        if onset_in_range { "✓" } else { "✗" });
    println!("  [{}] Seg increasing (S_max = {:.4})",
        if s_max > 0.1 { "✓" } else { "?" }, s_max);
    println!();

    let all_pass = ke_check_passed && onset_in_range && s_max > 0.1;
    if all_pass {
        println!("✅ ALL CRITERIA PASSED — Ready for 85M production");
    } else {
        println!("❌ VALIDATION FAILED — Check parameters");
    }
    println!();
    println!("CSV: {}", csv_path);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
    std::process::exit(1);
}
