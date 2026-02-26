/// Janus 8M Run — Virialized ICs (α=4.57)
/// Compare with Zel'dovich v=0 run on same codebase
///
/// Parameters:
///   - Virialization: α ≈ 4.57 (from PE_binding)
///   - η=1.045, θ=0.5, dt=0.01, z_init=5

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.5;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 15000;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 8M — Virialized ICs (α≈4.57)                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_PARTICLES as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_PARTICLES as f64);
    println!("  η = {}", ETA);
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  box = {:.2}", box_size);
    println!("  integrator = DKD + Hubble friction");
    println!("  ICs = Virialized (analytical, α from PE_binding)");
    println!();

    // Setup cosmological expansion
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
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = format!("/app/output/8M_virialized_{}", date);
    fs::create_dir_all(&output_dir)?;
    println!("Output directory: {}\n", output_dir);

    // Create simulation with virialization (new_bvh_only does analytical virialization)
    println!("Creating simulation with analytical virialization...");
    let t0 = Instant::now();
    let mut sim = GpuNBodySimulation::new_bvh_only(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    println!("  Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Initial state
    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation_distance()?;
    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke_init);
    println!("  S₀ = {:.6}", seg_init);
    println!();

    // Time series file
    let ts_filename = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_filename)?);
    writeln!(ts_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;

    // Reference KE
    let ke_ref = ke_init;
    let mut tau = cosmo.tau_start;
    let mut seg_max = seg_init;

    println!("Starting simulation loop...\n");

    for step in 1..=MAX_STEPS {
        let t_step = Instant::now();

        // Get cosmological parameters
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;
        let dtau_eff = dtau_per_dt;

        // DKD step with Hubble friction
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_eff)?;
        tau += DT * dtau_eff;

        let step_time = t_step.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation_distance()?;

        let ke_ratio = ke / ke_ref;
        seg_max = seg_max.max(seg);

        // Log to file
        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke_ratio, seg, step_time)?;
        ts_file.flush()?;  // Flush every step to avoid hang on mounted volume

        // Progress every 100 steps
        if step % 100 == 0 || step <= 10 {
            println!("step {:06} | z={:.2} | a={:.4} | H={:.4} | KE/KE_ref={:.4} | S={:.3e} | S_max={:.3e} | {} ms",
                step, z, a, h, ke_ratio, seg, seg_max, step_time as u64);
        }

        // Stop if numerical explosion
        if ke.is_nan() || ke.is_infinite() {
            println!("\n=== KE is NaN/Inf, stopping (numerical explosion) ===");
            break;
        }

        // Stop at z ≈ 0
        if z < 0.01 {
            println!("\n=== Reached z ≈ 0, simulation complete ===");
            break;
        }
    }

    ts_file.flush()?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Final results:");
    println!("  S_max = {:.6}", seg_max);
    println!("  Output: {}", ts_filename);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled!");
}
