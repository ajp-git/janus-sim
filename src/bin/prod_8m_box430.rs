//! 8M Production Run - Box 430 Mpc
//!
//! Parameters validated from 2M reference run (Seg_max=0.452 @ z≈1.7)
//!
//! CRITICAL FIX-016: dtau_per_dt = τ_range / (10000.0 * DT)
//! NOT: dtau = τ_range / TOTAL_STEPS
//!
//! ICs: GpuNBodySimulation::new() (+ first, then -)
//! Virialization: virialize() with full PE calculation

use std::fs;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// =============================================================================
// PRODUCTION PARAMETERS - 8M box=430 Mpc
// =============================================================================

#[cfg(feature = "cuda")]
const N_POSITIVE: usize = 4_000_000;
#[cfg(feature = "cuda")]
const N_NEGATIVE: usize = 4_000_000;
#[cfg(feature = "cuda")]
const BOX_SIZE: f64 = 430.0;  // Mpc - FIX-015: n_side=200, spacing=2.15 Mpc
#[cfg(feature = "cuda")]
const SOFTENING: f64 = 0.65;  // Mpc (0.3 × spacing)
#[cfg(feature = "cuda")]
const THETA: f64 = 0.7;
#[cfg(feature = "cuda")]
const DT: f64 = 0.01;
#[cfg(feature = "cuda")]
const TOTAL_STEPS: usize = 10000;
#[cfg(feature = "cuda")]
const Z_INIT: f64 = 5.0;

// Cosmology
#[cfg(feature = "cuda")]
const ETA: f64 = 1.045;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  8M Production Run - Box 430 Mpc");
    println!("  FIX-016 applied: dtau_per_dt = τ_range / (10000.0 × DT)");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Create output directory
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S");
    let output_dir = format!("/app/output/prod_8M_{}", timestamp);
    fs::create_dir_all(&output_dir)?;

    println!("Output: {}", output_dir);
    println!();

    println!("Parameters:");
    println!("  N+ = {}, N- = {}", N_POSITIVE, N_NEGATIVE);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  Softening = {} Mpc", SOFTENING);
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  steps = {}", TOTAL_STEPS);
    println!("  z_init = {}", Z_INIT);
    println!();

    // =========================================================================
    // Create simulation with GpuNBodySimulation::new() - EXACTLY like February
    // This places ALL positive particles first, THEN all negative particles
    // Creating initial spatial asymmetry for segregation
    // =========================================================================
    println!("Creating simulation with GpuNBodySimulation::new()...");
    println!("  (positives first, then negatives - February convention)");

    let mut sim = GpuNBodySimulation::new(N_POSITIVE, N_NEGATIVE, BOX_SIZE)
        .expect("Failed to create simulation");

    sim.set_softening(SOFTENING);
    sim.set_theta(THETA);

    println!();

    // =========================================================================
    // Virialization with virialize() - EXACTLY like February
    // Uses full PE calculation (not sampled)
    // =========================================================================
    println!("Virializing with virialize() (full PE calculation)...");
    sim.virialize().expect("Virialization failed");
    println!();

    // =========================================================================
    // Cosmology setup
    // =========================================================================
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    // CRITICAL FIX-016: dtau_per_dt calculation
    // This is the conformal time evolution per unit coordinate time
    // February used: dtau_per_dt = τ_range / (10000 × 0.01) = τ_range / 100
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * DT);

    println!("Cosmology:");
    println!("  η = {}", ETA);
    println!("  τ_start = {:.4}, τ_end = {:.4}", cosmo.tau_start, cosmo.tau_end);
    println!("  dtau_per_step = {:.6} (for τ advancement)", dtau_per_step);
    println!("  dtau_per_dt = {:.6} (for Hubble friction - FIX-016)", dtau_per_dt);
    println!();

    // =========================================================================
    // Initial state
    // =========================================================================
    let ke0 = sim.kinetic_energy().expect("KE failed");
    let seg0 = sim.segregation_distance().expect("Seg failed");

    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  Seg₀ = {:.4}", seg0);
    println!();

    // =========================================================================
    // Open log file
    // =========================================================================
    let log_path = format!("{}/run.log", output_dir);
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut log_file = fs::File::create(&log_path)?;
    let mut csv_file = fs::File::create(&csv_path)?;

    writeln!(csv_file, "step,z,a,H,tau,KE,KE_ratio,Seg,Seg_max")?;

    // Write parameters to log
    writeln!(log_file, "8M Production Run - Box 430 Mpc")?;
    writeln!(log_file, "FIX-016 applied: dtau_per_dt = τ_range / (10000.0 × DT)")?;
    writeln!(log_file, "")?;
    writeln!(log_file, "N+ = {}, N- = {}", N_POSITIVE, N_NEGATIVE)?;
    writeln!(log_file, "Box = {} Mpc", BOX_SIZE)?;
    writeln!(log_file, "Softening = {} Mpc", SOFTENING)?;
    writeln!(log_file, "θ = {}", THETA)?;
    writeln!(log_file, "dt = {}", DT)?;
    writeln!(log_file, "steps = {}", TOTAL_STEPS)?;
    writeln!(log_file, "η = {}", ETA)?;
    writeln!(log_file, "dtau_per_dt = {:.6}", dtau_per_dt)?;
    writeln!(log_file, "")?;
    writeln!(log_file, "KE₀ = {:.4e}", ke0)?;
    writeln!(log_file, "Seg₀ = {:.4}", seg0)?;
    writeln!(log_file, "")?;

    // =========================================================================
    // Main simulation loop
    // =========================================================================
    println!("══════════════════════════════════════════════════");
    println!("  Starting 8M Production Run");
    println!("══════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();
    let mut tau = cosmo.tau_start;
    let mut seg_max = seg0;
    let mut step_at_seg_max = 0;
    let mut last_report = Instant::now();

    for step in 0..=TOTAL_STEPS {
        // Get cosmological parameters at current tau
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        // Compute diagnostics
        let ke = sim.kinetic_energy().expect("KE failed");
        let ke_ratio = ke / ke0;
        let seg = sim.segregation_distance().expect("Seg failed");

        if seg > seg_max {
            seg_max = seg;
            step_at_seg_max = step;
        }

        // Write to CSV
        writeln!(csv_file, "{},{:.4},{:.6},{:.6},{:.6},{:.4e},{:.4},{:.4},{:.4}",
            step, z, a, h, tau, ke, ke_ratio, seg, seg_max)?;

        // Report every 500 steps or every 60 seconds
        let should_report = step % 500 == 0 || last_report.elapsed().as_secs() >= 60;

        if should_report {
            let elapsed = start_time.elapsed().as_secs_f64();
            let steps_per_sec = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let remaining_steps = TOTAL_STEPS - step;
            let eta_min = if steps_per_sec > 0.0 {
                remaining_steps as f64 / steps_per_sec / 60.0
            } else {
                0.0
            };

            let msg = format!(
                "Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}, Seg_max={:.4} ({:.2} steps/s, ETA {:.0}min)",
                step, z, ke_ratio, seg, seg_max, steps_per_sec, eta_min
            );
            println!("{}", msg);
            writeln!(log_file, "{}", msg)?;

            // Warning if Seg_max too low at step 1000
            if step == 1000 && seg_max < 0.01 {
                let warn = format!("\n⚠️  WARNING: Seg_max = {:.4} < 0.01 at step 1000", seg_max);
                println!("{}", warn);
                writeln!(log_file, "{}", warn)?;
            }

            last_report = Instant::now();
        }

        // Auto-stop: KE explosion
        if ke_ratio > 50.0 {
            let msg = format!("\n🛑 AUTO-STOP: KE/KE₀ = {:.2} > 50 at step {}", ke_ratio, step);
            println!("{}", msg);
            writeln!(log_file, "{}", msg)?;
            break;
        }

        // Step simulation (except on last iteration)
        if step < TOTAL_STEPS {
            // CRITICAL: Pass dtau_per_dt (FIX-016), NOT dtau_per_step
            sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)?;

            // Advance conformal time
            tau += dtau_per_step;
        }
    }

    // =========================================================================
    // Final summary
    // =========================================================================
    let total_time = start_time.elapsed();
    let final_ke = sim.kinetic_energy().expect("KE failed");
    let final_seg = sim.segregation_distance().expect("Seg failed");
    let (final_a, _) = cosmo.get_params_at_tau(tau);
    let final_z = 1.0 / final_a - 1.0;

    println!();
    println!("══════════════════════════════════════════════════");
    println!("  8M Production Run Complete");
    println!("══════════════════════════════════════════════════");
    println!();
    println!("Final state:");
    println!("  z_final = {:.4}", final_z);
    println!("  KE/KE₀ = {:.4}", final_ke / ke0);
    println!("  Seg_final = {:.4}", final_seg);
    let (a_at_seg_max, _) = cosmo.get_params_at_tau(cosmo.tau_start + step_at_seg_max as f64 * dtau_per_step);
    println!("  Seg_max = {:.4} @ step {} (z≈{:.2})", seg_max, step_at_seg_max, 1.0 / a_at_seg_max - 1.0);
    println!();
    println!("Runtime: {:.1} hours", total_time.as_secs_f64() / 3600.0);
    println!("Output: {}", output_dir);

    // Write final summary to log
    writeln!(log_file, "")?;
    writeln!(log_file, "══════════════════════════════════════════════════")?;
    writeln!(log_file, "  8M Production Run Complete")?;
    writeln!(log_file, "══════════════════════════════════════════════════")?;
    writeln!(log_file, "")?;
    writeln!(log_file, "z_final = {:.4}", final_z)?;
    writeln!(log_file, "KE/KE₀ = {:.4}", final_ke / ke0)?;
    writeln!(log_file, "Seg_final = {:.4}", final_seg)?;
    writeln!(log_file, "Seg_max = {:.4} @ step {}", seg_max, step_at_seg_max)?;
    writeln!(log_file, "")?;
    writeln!(log_file, "Runtime: {:.1} hours", total_time.as_secs_f64() / 3600.0)?;

    // Verdict
    let verdict = if seg_max > 0.3 {
        "✅ EXCELLENT - Segregation dynamics reproduced"
    } else if seg_max > 0.1 {
        "✅ GOOD - Segregation detected"
    } else if seg_max > 0.01 {
        "⚠️  MARGINAL - Weak segregation"
    } else {
        "❌ FAILED - No segregation"
    };

    println!();
    println!("Verdict: {}", verdict);
    writeln!(log_file, "")?;
    writeln!(log_file, "Verdict: {}", verdict)?;

    // Write summary JSON
    let summary_path = format!("{}/summary.json", output_dir);
    let summary = format!(r#"{{
  "run": "prod_8M",
  "N": {},
  "box_size": {},
  "softening": {},
  "theta": {},
  "dt": {},
  "steps": {},
  "eta": {},
  "dtau_per_dt": {},
  "z_init": {},
  "z_final": {},
  "KE0": {},
  "KE_final": {},
  "KE_ratio_final": {},
  "Seg0": {},
  "Seg_final": {},
  "Seg_max": {},
  "step_at_seg_max": {},
  "runtime_hours": {},
  "verdict": "{}"
}}"#,
        N_POSITIVE + N_NEGATIVE,
        BOX_SIZE,
        SOFTENING,
        THETA,
        DT,
        TOTAL_STEPS,
        ETA,
        dtau_per_dt,
        Z_INIT,
        final_z,
        ke0,
        final_ke,
        final_ke / ke0,
        seg0,
        final_seg,
        seg_max,
        step_at_seg_max,
        total_time.as_secs_f64() / 3600.0,
        verdict
    );
    fs::write(&summary_path, summary)?;

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature: cargo run --release --features cuda --bin prod_8m_box430");
}
