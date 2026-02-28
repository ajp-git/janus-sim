//! Janus 85M particle simulation with TreePM + Morton + warp-coherent
//!
//! Production run using optimized step_treepm_gpu_morton:
//!   - Morton ordering: 7.4x speedup
//!   - Warp-coherent kernel: 3x additional
//!   - Expected: ~35s/step @ 85M → 5 days for 12000 steps
//!
//! Output: /app/output/85M_treepm_YYYY-MM-DD/

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_PARTICLES: usize = 85_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;  // Per FIX-012: theta=0.7 obligatoire
const DT: f64 = 0.01;
const FRAME_INTERVAL: usize = 500;
const SNAPSHOT_INTERVAL: usize = 1000;
const MAX_SNAPSHOTS: usize = 20;  // Keep last 20 snapshots
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 12000;
const VIRIAL_FACTOR: f64 = 0.5;  // 0.5 for large N (prevents premature collapse)

#[cfg(all(feature = "cuda", feature = "cufft"))]
struct RenderJob {
    step: usize,
    pos: Vec<f32>,
    signs: Vec<i8>,
    box_size: f64,
    seg: f64,
    ke_ratio: f64,
    redshift: f64,
    render_data_dir: String,
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn render_thread(rx: mpsc::Receiver<RenderJob>) {
    while let Ok(job) = rx.recv() {
        let path = format!("{}/step_{:06}.bin", job.render_data_dir, job.step);

        let mut file = match File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[render] Failed to create {}: {}", path, e);
                continue;
            }
        };

        // Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + redshift(f64) + n(u32)
        let n = (job.pos.len() / 3) as u32;
        let _ = file.write_all(&(job.step as u32).to_le_bytes());
        let _ = file.write_all(&job.box_size.to_le_bytes());
        let _ = file.write_all(&job.seg.to_le_bytes());
        let _ = file.write_all(&job.ke_ratio.to_le_bytes());
        let _ = file.write_all(&job.redshift.to_le_bytes());
        let _ = file.write_all(&n.to_le_bytes());

        // pos: N×3×f32
        let pos_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(job.pos.as_ptr() as *const u8, job.pos.len() * 4)
        };
        let _ = file.write_all(pos_bytes);

        // signs: N×i8
        let signs_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(job.signs.as_ptr() as *const u8, job.signs.len())
        };
        let _ = file.write_all(signs_bytes);

        eprintln!("[data] step_{:06}.bin saved (z={:.2})", job.step, job.redshift);
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(
    step: usize,
    pos: &[f32],
    vel: &[f32],
    signs: &[i8],
    eta: f64,
    redshift: f64,
    scale_factor: f64,
    snapshots_dir: &str,
    snapshots: &mut Vec<String>,
) -> std::io::Result<()> {
    let path = format!("{}/snapshot_{:06}.bin", snapshots_dir, step);

    let mut file = File::create(&path)?;

    // Header (128 bytes, padded)
    let header = format!("step={} time={:.3} eta={} z={:.4} a={:.6} n={}\n",
        step, step as f64 * DT, eta, redshift, scale_factor, N_PARTICLES);
    let mut header_bytes = [b' '; 128];
    header_bytes[..header.len().min(128)].copy_from_slice(&header.as_bytes()[..header.len().min(128)]);
    file.write_all(&header_bytes)?;

    // pos: 85M × 3 × f32
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 4)
    };
    file.write_all(pos_bytes)?;

    // vel: 85M × 3 × f32
    let vel_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vel.as_ptr() as *const u8, vel.len() * 4)
    };
    file.write_all(vel_bytes)?;

    // signs: 85M × i8
    let signs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(signs.as_ptr() as *const u8, signs.len())
    };
    file.write_all(signs_bytes)?;

    file.sync_all()?;

    // Add to list and rotate
    snapshots.push(path.clone());
    while snapshots.len() > MAX_SNAPSHOTS {
        let old = snapshots.remove(0);
        let _ = fs::remove_file(&old);
        eprintln!("[snapshot] Deleted old: {}", old);
    }

    eprintln!("[snapshot] Saved: {} (z={:.2}, {:.2} GB)", path, redshift,
        (128 + N_PARTICLES * 3 * 4 * 2 + N_PARTICLES) as f64 / 1e9);

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 85M TreePM Production Run                              ║");
    println!("║   Morton + Warp-coherent (optim-warpcoherent-v1.0)            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Calculate particle split based on eta
    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);
    let r_cut = box_size / 16.0;

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_PARTICLES as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_PARTICLES as f64);
    println!("  η = {}", ETA);
    println!("  θ = {} (FIX-012 validated)", THETA);
    println!("  r_cut = {:.2} Mpc (box/16)", r_cut);
    println!("  dt = {}", DT);
    println!("  box = {:.2} Mpc", box_size);
    println!("  integrator = TreePM (Morton + warp-coherent)");
    println!("  virial_factor = {} (prevents premature collapse)", VIRIAL_FACTOR);
    println!("  frames every {} steps", FRAME_INTERVAL);
    println!("  snapshots every {} steps", SNAPSHOT_INTERVAL);
    println!();

    // Setup cosmological expansion
    println!("--- Cosmological Expansion Setup ---");
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let n_steps_to_z0 = TOTAL_STEPS as f64;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("  z_init = {:.2}", z_init_actual);
    println!("  a_init = {:.6}", a_init);
    println!("  H_init = {:.6}", h_init);
    println!("  τ_start = {:.6}", cosmo.tau_start);
    println!("  τ_end = {:.6}", cosmo.tau_end);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!("  Expected steps to z=0: {}", TOTAL_STEPS);
    println!();

    // Estimate runtime
    let estimated_step_ms = 35_000.0;  // ~35s/step based on extrapolation
    let estimated_hours = (TOTAL_STEPS as f64 * estimated_step_ms) / (1000.0 * 3600.0);
    let estimated_days = estimated_hours / 24.0;
    println!("Estimated runtime:");
    println!("  ~{:.0}s/step × {} steps = {:.1} hours ({:.1} days)",
        estimated_step_ms / 1000.0, TOTAL_STEPS, estimated_hours, estimated_days);
    println!();

    // Create dated output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_base = format!("/app/output/85M_treepm_{}", date);
    let frames_dir = format!("{}/frames", output_base);
    let snapshots_dir = format!("{}/snapshots", output_base);
    let render_data_dir = format!("{}/render_data", output_base);

    fs::create_dir_all(&frames_dir)?;
    fs::create_dir_all(&snapshots_dir)?;
    fs::create_dir_all(&render_data_dir)?;

    // Create CSV file for time series
    let csv_path = format!("{}/time_series.csv", output_base);
    let mut csv_file = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,s_max,step_time_ms")?;

    println!("Output directory: {}", output_base);
    println!("CSV: {}", csv_path);
    println!();

    // Create simulation with proper virialization
    println!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new_with_virial_factor(n_positive, n_negative, box_size, VIRIAL_FACTOR)?;
    sim.set_theta(THETA);
    println!("  θ = {}", THETA);

    // Get initial KE
    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;
    println!();
    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", seg0);
    println!();

    // Start render thread
    let (tx, rx) = mpsc::channel::<RenderJob>();
    let render_handle = thread::spawn(move || render_thread(rx));

    // Tracking
    let start_time = Instant::now();
    let mut snapshots: Vec<String> = Vec::new();
    let mut step = 0usize;
    let mut current_tau = cosmo.tau_start;
    let mut s_max = 0.0f64;
    let mut s_max_step = 0usize;
    let mut s_max_z = Z_INIT;

    println!("Starting simulation loop...");
    println!("  Step        z     KE/KE₀      Seg     S_max    ms/step");
    println!("---------------------------------------------------------------");

    loop {
        let step_start = Instant::now();

        // Get cosmological parameters at current tau
        let (a, h) = if current_tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(current_tau)
        } else {
            (1.0, 0.0)
        };
        let z = 1.0 / a - 1.0;

        // Effective dtau_per_dt (0 after reaching z=0)
        let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };

        // TreePM step with Morton ordering + warp-coherent kernel
        sim.step_treepm_gpu_morton(DT, r_cut, h, dtau_eff)?;
        step += 1;
        current_tau += dtau_cosmo;

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        // Calculate metrics
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        let ke_ratio = ke / ke0;

        // Track S_max
        if seg > s_max {
            s_max = seg;
            s_max_step = step;
            s_max_z = z.max(0.0);
        }

        // Print progress every 100 steps or at key events
        if step % 100 == 0 || step <= 5 {
            println!("{:6}   {:.3}   {:.4}   {:.4}   {:.4}   {:.0}",
                step, z.max(0.0), ke_ratio, seg, s_max, step_ms);
        }

        // Write to CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6},{:.6},{:.1}",
            step, step as f64 * DT, z.max(0.0), a, h, ke, ke_ratio, seg, s_max, step_ms)?;

        // Flush every 50 steps
        if step % 50 == 0 {
            csv_file.flush()?;
        }

        // Render frame
        if step % FRAME_INTERVAL == 0 {
            let pos = sim.get_positions()?;
            let signs = sim.get_signs()?;

            let job = RenderJob {
                step,
                pos,
                signs,
                box_size: sim.box_size(),
                seg,
                ke_ratio,
                redshift: z.max(0.0),
                render_data_dir: render_data_dir.clone(),
            };

            if tx.send(job).is_err() {
                eprintln!("[warning] Render thread died");
            }
        }

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            let signs = sim.get_signs()?;

            save_snapshot(step, &pos, &vel, &signs, ETA, z.max(0.0), a,
                &snapshots_dir, &mut snapshots)?;
        }

        // Stop conditions
        if step >= TOTAL_STEPS {
            println!();
            println!("=== Reached {} steps ===", TOTAL_STEPS);
            break;
        }

        if z <= 0.0 && step > TOTAL_STEPS / 2 {
            println!();
            println!("=== Reached z=0 ===");
            break;
        }
    }

    // Final summary
    let total_time = start_time.elapsed();
    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   SIMULATION COMPLETE                                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Results:");
    println!("  Total steps: {}", step);
    println!("  Runtime: {:.1} hours ({:.1} days)",
        total_time.as_secs_f64() / 3600.0,
        total_time.as_secs_f64() / 86400.0);
    println!("  Average: {:.1} ms/step",
        total_time.as_secs_f64() * 1000.0 / step as f64);
    println!();
    println!("  S_max = {:.4} at step {} (z = {:.2})", s_max, s_max_step, s_max_z);
    println!("  Final KE/KE₀ = {:.4}", sim.kinetic_energy()? / ke0);
    println!();
    println!("Output: {}", output_base);

    // Cleanup
    drop(tx);
    let _ = render_handle.join();

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
    std::process::exit(1);
}
