//! Janus 85M particle simulation with cosmological expansion
//!
//! - Friedmann Janus equations coupled to N-body
//! - Hubble friction cools velocities → segregation emerges
//! - Renders every 200 steps (PNG 3840x1280, 3 views via Python)
//! - Snapshots every 200 steps
//!
//! Output structure: /app/output/85M_YYYY-MM-DD/frames/ and /snapshots/

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_PARTICLES: usize = 85_000_000;
const ETA: f64 = 1.045;
const DT: f64 = 0.01;
const RENDER_INTERVAL: usize = 200;
const SNAPSHOT_INTERVAL: usize = 200;
const MAX_SNAPSHOTS: usize = 50;
const Z_INIT: f64 = 5.0;  // Start at z=5 (post-recombination)

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

    // Header (128 bytes, padded with spaces)
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

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 85M Simulation — With Cosmological Expansion           ║");
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
    println!("  θ = 0.5");
    println!("  dt = {}", DT);
    println!("  box = {:.2}", box_size);
    println!("  integrator = DKD + Hubble friction");
    println!("  render every {} steps", RENDER_INTERVAL);
    println!("  snapshot every {} steps", SNAPSHOT_INTERVAL);
    println!();

    // Setup cosmological expansion
    println!("--- Cosmological Expansion Setup ---");
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    // dtau_per_dt converts N-body time to conformal time
    // We want to reach z=0 in roughly 10000-15000 steps
    let n_steps_to_z0 = 12000.0;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    // Initial cosmological state
    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("  z_init = {:.2}", z_init_actual);
    println!("  a_init = {:.6}", a_init);
    println!("  H_init = {:.6}", h_init);
    println!("  τ_start = {:.6}", cosmo.tau_start);
    println!("  τ_end = {:.6}", cosmo.tau_end);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!("  Expected steps to z=0: ~{:.0}", n_steps_to_z0);
    println!();

    // Create dated output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_base = format!("/app/output/85M_{}", date);
    let frames_dir = format!("{}/frames", output_base);
    let snapshots_dir = format!("{}/snapshots", output_base);
    let render_data_dir = format!("{}/render_data", output_base);

    fs::create_dir_all(&frames_dir)?;
    fs::create_dir_all(&snapshots_dir)?;
    fs::create_dir_all(&render_data_dir)?;

    // Create CSV file for time series
    let csv_path = format!("{}/time_series.csv", output_base);
    let mut csv_file = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;

    println!("Output directory: {}", output_base);
    println!("CSV: {}", csv_path);
    println!();

    // Create simulation
    println!("Creating simulation...");
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(0.5);  // θ = 0.5 non-négociable pour publication
    println!("  θ = 0.5");

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

    // Compute initial forces (required for DKD with cold ICs)
    println!("Computing initial forces...");
    sim.compute_forces()?;
    let acc_sum = sim.acceleration_sum()?;
    println!("  Σ|acc| = {:.4e}", acc_sum);
    println!();

    println!("Starting simulation loop...");
    println!();

    loop {
        let step_start = Instant::now();

        // Get cosmological parameters at current tau
        let (a, h) = if current_tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(current_tau)
        } else {
            (1.0, 0.0)  // Post-expansion: a=1, H=0
        };
        let z = 1.0 / a - 1.0;

        // Effective dtau_per_dt (0 after reaching z=0)
        let dtau_eff = if current_tau <= cosmo.tau_end { dtau_per_dt } else { 0.0 };

        // DKD step with Hubble friction
        sim.step_dkd(DT, h, dtau_eff)?;
        step += 1;
        current_tau += dtau_cosmo;

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        // Calculate metrics every step
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        let ke_ratio = ke / ke0;

        println!("step {:06} | z={:.2} | a={:.4} | H={:.4} | KE/KE₀={:.4} | S={:.3e} | {:.1} ms",
            step, z.max(0.0), a, h, ke_ratio, seg, step_ms);

        // Write to CSV
        writeln!(csv_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.1}",
            step, step as f64 * DT, z.max(0.0), a, h, ke, ke_ratio, seg, step_ms)?;

        // Flush every 10 steps
        if step % 10 == 0 {
            csv_file.flush()?;
        }

        if step == 1 {
            let acc_sum = sim.acceleration_sum()?;
            println!();
            println!("✓ Step 1 confirmed: {:.1} ms/step", step_ms);
            println!("  z = {:.2}, a = {:.4}, H = {:.4}", z, a, h);
            println!("  KE = {:.4e}, Σ|acc| = {:.4e}", ke, acc_sum);
            println!();
        }

        // Check if we need to render or snapshot
        let need_render = step % RENDER_INTERVAL == 0;
        let need_snapshot = step % SNAPSHOT_INTERVAL == 0;

        if need_render {
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

            let _ = tx.send(job);
        }

        if need_snapshot {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            let signs = sim.get_signs()?;

            save_snapshot(step, &pos, &vel, &signs, ETA, z.max(0.0), a, &snapshots_dir, &mut snapshots)?;
        }

        // Optional: stop at z < 0.01
        // if z < 0.01 {
        //     println!("\n=== Reached z < 0.01, stopping ===");
        //     break;
        // }
    }

    // Cleanup (unreachable in infinite loop)
    #[allow(unreachable_code)]
    {
        drop(tx);
        render_handle.join().unwrap();
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Error: CUDA feature not enabled. Compile with --features cuda");
    std::process::exit(1);
}
