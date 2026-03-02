/// Janus 20M Production Run — Pure Barnes-Hut (same as validated 8M)
///
/// TreePM introduces force discontinuity at r_cut → destabilizes large N
/// Solution: Use pure BH like validated 8M run
///
/// Parameters:
///   N = 20_000_000
///   box = 430.0 Mpc (same density as 8M validated)
///   θ = 0.7 (FIX-012 validated)
///   dt = 0.01
///   η = 1.045
///   integrator = Pure BH (step_with_expansion_dkd_gpu)
///
/// Expected performance:
///   8M → ~500ms/step (measured)
///   20M → ~1300ms/step (extrapolated)
///   12000 steps → ~4.3 hours

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

const N_PARTICLES: usize = 20_000_000;
const BOX_SIZE: f64 = 430.0;  // Same density as 8M validated run

const ETA: f64 = 1.045;
const THETA: f64 = 0.7;  // FIX-012 validated
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 15000;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 20M — Pure Barnes-Hut (same as 8M validated)           ║");
    println!("║   virialize_sampled(10000) + step_with_expansion_dkd_gpu       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_PARTICLES as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_PARTICLES as f64);
    println!("  η = {}", ETA);
    println!("  θ = {} (FIX-012 validated)", THETA);
    println!("  dt = {}", DT);
    println!("  box = {:.2} Mpc (same density as 8M validated)", BOX_SIZE);
    println!("  integrator = Pure BH + DKD + Hubble");
    println!("  ICs = virialize_sampled(10000)");
    println!();

    // Cosmological expansion
    println!("--- Cosmological Expansion Setup ---");
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let n_steps_to_z0 = 12000.0;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("  z_init = {:.2}", z_init_actual);
    println!("  a_init = {:.6}", a_init);
    println!("  H_init = {:.6}", h_init);
    println!("  τ range = [{:.4}, {:.4}]", cosmo.tau_start, cosmo.tau_end);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!("  Expected steps to z=0: {}", n_steps_to_z0 as usize);
    println!();

    // Estimate runtime
    let estimated_step_ms = 1300.0;  // Extrapolated from 8M
    let estimated_hours = (n_steps_to_z0 * estimated_step_ms) / (1000.0 * 3600.0);
    println!("Estimated runtime: {:.1} hours ({:.1}s/step)", estimated_hours, estimated_step_ms / 1000.0);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = format!("/app/output/20M_bh_{}", date);
    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(format!("{}/snapshots", output_dir))?;
    println!("Output directory: {}\n", output_dir);

    // Create simulation with BH-only (same as validated 8M run)
    println!("Creating BH simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodySimulation::new_bvh_only(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);
    println!("  Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    // virialize_sampled(10000) — exact same as 8M validated run
    println!("Applying virialize_sampled(10000)...");
    let t0 = Instant::now();
    sim.virialize_sampled(10000)?;
    println!("  Virialized in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Initial state
    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation_distance()?;
    println!("Initial state (after virialization):");
    println!("  KE₀ = {:.4e}", ke_init);
    println!("  S₀ = {:.6}", seg_init);
    println!();

    // Time series file
    let ts_filename = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_filename)?);
    writeln!(ts_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,seg_max,step_time_ms")?;

    let ke_ref = ke_init;
    let mut tau = cosmo.tau_start;
    let mut seg_max = seg_init;

    println!("Starting simulation loop...\n");
    println!("  Step        z     KE/KE₀      Seg     S_max    ms/step");
    println!("---------------------------------------------------------------");

    for step in 1..=MAX_STEPS {
        let t_step = Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        // Pure BH step with Hubble friction (same as 8M validated)
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;

        let step_time = t_step.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation_distance()?;

        let ke_ratio = ke / ke_ref;
        seg_max = seg_max.max(seg);

        // Write CSV every step
        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke_ratio, seg, seg_max, step_time)?;

        // Flush every 10 steps
        if step % 10 == 0 {
            ts_file.flush()?;
        }

        // Progress every 100 steps or first 10
        if step <= 10 || step % 100 == 0 {
            println!("  {:5}   {:.3}   {:7.4}   {:.4}   {:.4}   {:6.0}",
                step, z, ke_ratio, seg, seg_max, step_time);
        }

        // Validation at step 100
        if step == 100 {
            println!("\n=== VALIDATION @ step 100 ===");
            println!("  KE/KE₀ = {:.4} (expected < 5)", ke_ratio);
            if ke_ratio > 5.0 {
                println!("  FAIL: KE/KE₀ > 5 — physics invalid");
                println!("  Stopping run.");
                break;
            } else {
                println!("  PASS: KE/KE₀ < 5 — continuing");
            }
            println!();
        }

        // Stop conditions
        if ke.is_nan() || ke.is_infinite() || ke_ratio > 100.0 {
            println!("\n=== STOPPING: KE explosion (KE/KE₀ = {:.1}) ===", ke_ratio);
            break;
        }

        if z < 0.01 {
            println!("\n=== Reached z ≈ 0, simulation complete ===");
            break;
        }
    }

    ts_file.flush()?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Results:");
    println!("  S_max = {:.6}", seg_max);
    println!("  Output: {}", ts_filename);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled!");
}
