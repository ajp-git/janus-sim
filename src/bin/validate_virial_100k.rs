//! Validation run: 100K particles with configurable virial_factor
//!
//! Quick test to calibrate virial_factor for production runs

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_PARTICLES: usize = 100_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 1000;

// TEST PARAMETER - adjust this
const VIRIAL_FACTOR: f64 = 0.8;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Validation Run: 100K particles, virial_factor={}           ║", VIRIAL_FACTOR);
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

    let (a_init, _h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("Cosmology:");
    println!("  z_init = {:.2}", z_init_actual);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
    let output_dir = format!("/app/output/validate_100k_vf{:.1}_{}", VIRIAL_FACTOR, date);
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

    // Track key metrics
    let mut ke_at_100 = 0.0f64;
    let mut ke_at_500 = 0.0f64;

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

        // Track S_max
        if seg > s_max {
            s_max = seg;
            s_max_step = step;
        }

        // Record key checkpoints
        if step == 100 {
            ke_at_100 = ke_ratio;
            println!(">>> Step 100: KE/KE₀ = {:.2}", ke_ratio);
        }
        if step == 500 {
            ke_at_500 = ke_ratio;
            println!(">>> Step 500: KE/KE₀ = {:.2}", ke_ratio);
        }

        // Print progress
        if step % 100 == 0 || step <= 5 {
            println!("{:5}  {:.2}  {:8.2}  {:6.4}  {:6.0}",
                step, z.max(0.0), ke_ratio, seg, step_ms);
        }

        // CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.1}",
            step, step as f64 * DT, z.max(0.0), a, h, ke, ke_ratio, seg, step_ms)?;

        if step % 100 == 0 {
            csv_file.flush()?;
        }

        if step >= TOTAL_STEPS {
            break;
        }

        // Early stop if KE explodes
        if ke_ratio > 500.0 {
            println!("\n⚠️ KE/KE₀ > 500 — COLLAPSE DETECTED");
            break;
        }
    }

    csv_file.flush()?;

    let total_time = start_time.elapsed();
    let final_ke = sim.kinetic_energy()? / ke0;

    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   RESULTS: virial_factor = {}                                 ║", VIRIAL_FACTOR);
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Runtime: {:.1} minutes ({:.0} ms/step)",
        total_time.as_secs_f64() / 60.0,
        total_time.as_secs_f64() * 1000.0 / step as f64);
    println!();
    println!("KE/KE₀ progression:");
    println!("  Step 100:  {:.2}", ke_at_100);
    println!("  Step 500:  {:.2}", ke_at_500);
    println!("  Step {}:  {:.2}", step, final_ke);
    println!();
    println!("Segregation:");
    println!("  S_max = {:.4} at step {}", s_max, s_max_step);
    println!();

    // Summary assessment
    let ke_ok = ke_at_100 < 5.0 && final_ke < 50.0;
    let seg_ok = s_max > 0.1;

    if ke_ok && seg_ok {
        println!("✅ virial_factor={} looks GOOD for production", VIRIAL_FACTOR);
    } else if !ke_ok {
        println!("⚠️ KE still too high — try virial_factor={:.1}", VIRIAL_FACTOR + 0.2);
    } else {
        println!("⚠️ Check results manually");
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
