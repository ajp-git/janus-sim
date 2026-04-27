/// Janus 60M TreePM Production Run — Virialized ICs (exact 8M method)
///
/// Based on validated run_8m_virialized.rs:
///   - virialize_sampled(10000) → α = √(|PE_bind|/2KE) ≈ 4-6
///   - Same density as 8M: box scales with N^(1/3)
///   - TreePM for 60M performance
///
/// Parameters:
///   N = 60_000_000
///   box = 843.0 Mpc (density = 0.100 part/Mpc³)
///   softening = 1.0 Mpc (spacing ≈ 2.15 Mpc)
///   θ = 0.7 (FIX-012 validated)
///   dt = 0.01
///   η = 1.045

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

const N_PARTICLES: usize = 60_000_000;
const BOX_SIZE: f64 = 843.0;  // Same density as 8M validated run
const SOFTENING: f64 = 1.0;   // Adapted to spacing ~2.15 Mpc

const ETA: f64 = 1.045;
const THETA: f64 = 0.7;  // FIX-012 validated
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 15000;
const R_CUT: f64 = BOX_SIZE / 16.0;  // PM/Tree splitting scale

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 60M TreePM — Virialized ICs (exact 8M method)          ║");
    println!("║   virialize_sampled(10000) → α ≈ 4-6                           ║");
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
    println!("  box = {:.2} Mpc", BOX_SIZE);
    println!("  softening = {:.1} Mpc", SOFTENING);
    println!("  r_cut = {:.1} Mpc (box/16)", R_CUT);
    println!("  integrator = TreePM Morton + warp-coherent + Hubble");
    println!("  ICs = virialize_sampled(10000) — exact 8M method");
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

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = format!("/app/output/60M_virialized_{}", date);
    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(format!("{}/snapshots", output_dir))?;
    fs::create_dir_all(format!("{}/render_data", output_dir))?;
    println!("Output directory: {}\n", output_dir);

    // Create simulation with TreePM (uses default virial_factor=0.3 for initial velocities)
    println!("Creating TreePM simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    println!("  Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    // CRITICAL: Apply virialize_sampled(10000) — exact same as 8M validated run
    // This computes α = √(|PE_bind|/2KE) using sampled same-sign pairs
    println!("Applying virialize_sampled(10000) — exact 8M method...");
    let t0 = Instant::now();
    sim.virialize_sampled(10000)?;
    println!("  Virialized in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Initial state
    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation()?;
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

        // TreePM step with Hubble friction (Morton + warp-coherent)
        sim.step_treepm_gpu(DT, R_CUT, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;

        let step_time = t_step.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;

        let ke_ratio = ke / ke_ref;
        seg_max = seg_max.max(seg);

        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke_ratio, seg, seg_max, step_time)?;

        // Progress
        if step <= 10 || step % 100 == 0 {
            println!("  {:5}   {:.3}   {:7.4}   {:.4}   {:.4}   {:6.0}",
                step, z, ke_ratio, seg, seg_max, step_time);
            ts_file.flush()?;
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

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
    std::process::exit(1);
}
