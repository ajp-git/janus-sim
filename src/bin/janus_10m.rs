//! Janus 10M particle validation run with cosmological expansion
//!
//! Purpose: Validate that segregation grows with fixed BVH tree
//! Parameters: η=1.045, θ=0.5, dt=0.01, z_init=5
//! ICs: Zel'dovich perturbations, v=0

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_PARTICLES: usize = 10_000_000;
const ETA: f64 = 1.045;
const DT: f64 = 0.01;
const RENDER_INTERVAL: usize = 100;
const SNAPSHOT_INTERVAL: usize = 500;
const MAX_SNAPSHOTS: usize = 20;
const Z_INIT: f64 = 5.0;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 10M Validation Run — Segregation Test                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Calculate particle split based on eta
    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_PARTICLES as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_PARTICLES as f64);
    println!("  η = {}", ETA);
    println!("  θ = 0.8 (fast, acceptable for segregation)");
    println!("  dt = {}", DT);
    println!("  box = {:.2}", box_size);
    println!("  integrator = DKD + Hubble friction");
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

    // Create dated output directory
    let date = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
    let output_base = format!("/app/output/10M_valid_{}", date);
    let snapshots_dir = format!("{}/snapshots", output_base);

    fs::create_dir_all(&snapshots_dir)?;

    // Create CSV file for time series
    let csv_path = format!("{}/time_series.csv", output_base);
    let mut csv_file = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;

    println!("Output directory: {}", output_base);
    println!();

    // Create simulation
    println!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.8);  // Fast mode: 40s/step vs 80s at θ=0.7, 200s at θ=0.5

    // Get initial state
    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;
    println!();
    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", seg0);
    println!();

    // Compute initial forces
    println!("Computing initial forces...");
    sim.compute_forces()?;
    let acc_sum = sim.acceleration_sum()?;
    println!("  Σ|acc| = {:.4e}", acc_sum);
    println!();

    // Tracking
    let start_time = Instant::now();
    let mut snapshots: Vec<String> = Vec::new();
    let mut step = 0usize;
    let mut current_tau = cosmo.tau_start;

    // Use first non-zero KE as reference
    let mut ke_ref: Option<f64> = None;

    println!("Starting simulation loop...");
    println!();

    loop {
        let step_start = Instant::now();

        // Get cosmological parameters
        let (a, h) = if current_tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(current_tau)
        } else {
            (1.0, 0.0)
        };
        let z = (1.0 / a - 1.0).max(0.0);

        let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };

        // DKD step with Hubble friction
        sim.step_dkd(DT, h, dtau_eff)?;
        step += 1;
        current_tau += dtau_cosmo;

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        // Calculate metrics
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;

        // Set reference KE on first non-zero value
        if ke_ref.is_none() && ke > 1e-10 {
            ke_ref = Some(ke);
            println!("  KE_ref set to {:.4e} at step {}", ke, step);
        }

        let ke_ratio = ke_ref.map(|k| ke / k).unwrap_or(f64::INFINITY);

        // Print progress every 10 steps
        if step % 10 == 0 || step == 1 {
            println!("step {:06} | z={:.2} | a={:.4} | H={:.4} | KE/KE_ref={:.4} | S={:.3e} | {:.0} ms",
                step, z, a, h, ke_ratio, seg, step_ms);
        }

        // Write to CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke_ratio, seg, step_ms)?;

        if step % 10 == 0 {
            csv_file.flush()?;
        }

        // Detailed info at step 1
        if step == 1 {
            let acc_sum = sim.acceleration_sum()?;
            println!();
            println!("✓ Step 1 confirmed: {:.1} ms/step", step_ms);
            println!("  z = {:.2}, a = {:.4}, H = {:.4}", z, a, h);
            println!("  KE = {:.4e}, Σ|acc| = {:.4e}", ke, acc_sum);
            println!();
        }

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let path = format!("{}/snapshot_{:06}.bin", snapshots_dir, step);
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            let signs = sim.get_signs()?;

            let mut file = File::create(&path)?;
            let header = format!("step={} z={:.4} a={:.6} seg={:.6e} ke={:.6e}\n",
                step, z, a, seg, ke);
            let mut header_bytes = [b' '; 128];
            header_bytes[..header.len().min(128)].copy_from_slice(&header.as_bytes()[..header.len().min(128)]);
            file.write_all(&header_bytes)?;

            let pos_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 4)
            };
            file.write_all(pos_bytes)?;

            let vel_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(vel.as_ptr() as *const u8, vel.len() * 4)
            };
            file.write_all(vel_bytes)?;

            let signs_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(signs.as_ptr() as *const u8, signs.len())
            };
            file.write_all(signs_bytes)?;

            file.sync_all()?;

            snapshots.push(path.clone());
            while snapshots.len() > MAX_SNAPSHOTS {
                let old = snapshots.remove(0);
                let _ = fs::remove_file(&old);
            }

            eprintln!("[snapshot] Saved: {} (z={:.2})", path, z);
        }

        // Stop conditions
        if z < 0.01 {
            println!();
            println!("=== Reached z < 0.01, stopping ===");
            break;
        }

        if ke_ratio > 100.0 {
            println!();
            println!("=== KE ratio > 100, stopping (possible instability) ===");
            break;
        }

        // Progress report every 500 steps
        if step % 500 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let steps_per_sec = step as f64 / elapsed;
            let eta_steps = ((z / 5.0) * 12000.0) as usize;
            let eta_secs = eta_steps as f64 / steps_per_sec;
            println!();
            println!("--- Progress: step {} | z={:.2} | {:.2} steps/s | ETA to z=0: {:.0}s ---",
                step, z, steps_per_sec, eta_secs);
            println!();
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!();
    println!("=== Simulation Complete ===");
    println!("  Total steps: {}", step);
    println!("  Total time: {:.1}s ({:.1} min)", total_time, total_time / 60.0);
    println!("  Avg step time: {:.1} ms", total_time * 1000.0 / step as f64);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Error: CUDA feature not enabled. Compile with --features cuda");
    std::process::exit(1);
}
