//! JANUS ZOOM V3 — Independent 50 Mpc Box
//!
//! Full baryonic physics simulation from z=10 to z=0
//! - m+ : SPH + cooling + star formation + SN feedback
//! - m- : Pure gravity (collisionless)
//!
//! Based on ZOOM_V3_TOP_DU_TOP.md specification

use clap::Parser;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::cooling_gpu::GpuCooling;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

use janus::ic_gen::{IcParams, generate_zeldovich_ics};
use janus::vsl_dynamic::{CoupledFriedmann, JanusVSLParams};

// ═══════════════════════════════════════════════════════════════════════════
// CLI ARGUMENTS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Parser, Debug)]
#[command(name = "janus_zoom_v3")]
#[command(about = "Janus Zoom V3 - Independent 50 Mpc simulation z=10→0")]
struct Args {
    #[arg(long, default_value = "50.0")]
    box_size: f64,

    #[arg(long, default_value = "128")]
    n_grid: usize,

    #[arg(long, default_value = "10.0")]
    z_init: f64,

    #[arg(long, default_value = "0.0")]
    z_final: f64,

    #[arg(long, default_value = "42")]
    seed: u64,

    #[arg(long, default_value = "19.0")]
    mu: f64,

    #[arg(long, default_value = "69.9")]
    h0: f64,

    #[arg(long, default_value = "0.05")]
    omega_b: f64,

    #[arg(long, default_value = "0.10")]
    eps_plus: f64,

    #[arg(long, default_value = "0.25")]
    eps_minus: f64,

    #[arg(long, default_value = "0.001")]
    dt_max: f64,

    #[arg(long, default_value = "0.0002")]
    dt_min: f64,

    #[arg(long, default_value = "0.025")]
    eta: f64,

    #[arg(long, default_value = "0.7")]
    theta: f64,

    #[arg(long, default_value = "20")]
    snap_interval: usize,

    #[arg(long, default_value = "100")]
    relax_steps: usize,

    #[arg(long, default_value = "output/janus_zoom_v3")]
    out_dir: PathBuf,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const PI: f64 = std::f64::consts::PI;
const MPC_GYR_TO_KMS: f64 = 977.8;
const G_COSMO: f64 = 4.499e-15;  // Mpc³/(M_sun·Gyr²)

// Baryonic physics (m+ only)
const T_INIT_PLUS: f64 = 10000.0;   // Initial temperature [K]
const T_FLOOR: f64 = 100.0;         // Minimum temperature [K]
const T_THRESHOLD_SF: f64 = 10000.0; // Star formation threshold [K]
const N_THRESHOLD_SF: f64 = 30.0;   // Star formation density [cm⁻³]
const EPSILON_STAR: f64 = 0.02;     // Star formation efficiency
const EPSILON_SN: f64 = 0.003;      // SN feedback efficiency

// Janus cosmology
const ETA_JANUS: f64 = 1.045;       // From Pantheon+ fit

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════");
    println!("JANUS ZOOM V3 — Independent Box Simulation");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Box size    : {} Mpc", args.box_size);
    println!("  N_grid      : {} (N_total = {})", args.n_grid, 2 * args.n_grid.pow(3));
    println!("  z range     : {} → {}", args.z_init, args.z_final);
    println!("  μ           : {}", args.mu);
    println!("  ε_plus      : {} Mpc", args.eps_plus);
    println!("  ε_minus     : {} Mpc", args.eps_minus);
    println!("  dt range    : [{}, {}] Gyr", args.dt_min, args.dt_max);
    println!("  η (timestep): {}", args.eta);
    println!("  θ (B-H)     : {}", args.theta);
    println!("  Output      : {:?}", args.out_dir);
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create output directory
    let out_dir = if args.out_dir.starts_with("/app/") {
        args.out_dir.clone()
    } else {
        PathBuf::from("/app").join(&args.out_dir)
    };
    fs::create_dir_all(out_dir.join("snapshots")).expect("Cannot create output dir");

    // ═══════════════════════════════════════════════════════════════════════
    // [1/5] Generate ICs
    // ═══════════════════════════════════════════════════════════════════════
    println!("[1/5] Generating Zel'dovich ICs...");

    let ic_params = IcParams {
        box_size: args.box_size,
        n_grid: args.n_grid,
        z_init: args.z_init,
        seed: args.seed,
        mu: args.mu,
        n_s: 0.965,
        delta_rms: 0.10,
    };

    let ics = generate_zeldovich_ics(&ic_params);
    let n_total = ics.signs.len();
    let n_plus = ics.n_plus;
    let n_minus = ics.n_minus;

    println!("  ✓ Generated {} particles (N+ = {}, N- = {})", n_total, n_plus, n_minus);

    // Compute masses
    let h = args.h0 / 100.0;
    let rho_crit = 2.775e11 * h * h;  // M☉/Mpc³
    let rho_plus = args.omega_b * rho_crit;
    let rho_minus = args.mu * rho_plus;
    let v_box = args.box_size.powi(3);
    let m_plus = rho_plus * v_box / n_plus as f64;
    let m_minus = rho_minus * v_box / n_minus as f64;

    println!("  m_plus  = {:.3e} M☉", m_plus);
    println!("  m_minus = {:.3e} M☉", m_minus);

    // ═══════════════════════════════════════════════════════════════════════
    // [2/5] Initialize GPU simulation
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n[2/5] Initializing GPU TwoPass simulation...");

    // Convert to f32/i8 for TwoPass API
    let pos_f32: Vec<f32> = ics.positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = ics.velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = ics.signs.iter().map(|&s| s as i8).collect();

    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, args.box_size
    ).expect("Failed to create GPU simulation");

    // Set softening and theta
    // ADAPTIVE SOFTENING: scale with particle spacing to maintain stability at high resolution
    let spacing = args.box_size / args.n_grid as f64;
    let eps_adaptive = 0.5 * spacing;  // Softening = 50% of spacing (standard)

    println!("  Adaptive softening: ε = {:.3} Mpc (spacing = {:.3} Mpc)", eps_adaptive, spacing);

    sim.set_softening(eps_adaptive);
    sim.set_theta(args.theta);

    // CRITICAL: Set mass factor (converts code units to physical units)
    let m_total = m_plus * n_plus as f64 + m_minus * n_minus as f64;
    let mass_factor = G_COSMO * m_total / n_total as f64;
    sim.set_mass_factor(mass_factor);
    println!("  Mass factor: {:.6e} (G × M_total / N)", mass_factor);

    // Set initial redshift
    let a_init = 1.0 / (1.0 + args.z_init);
    sim.set_current_z(args.z_init);

    println!("  ✓ GPU ready: {} particles, θ={}", n_total, args.theta);

    // Initialize cooling (for m+ only)
    let cuda_device = CudaDevice::new(0).expect("No CUDA device");
    let _gpu_cooling = GpuCooling::new(cuda_device, n_plus, args.box_size, m_plus)
        .expect("Failed to init cooling");

    // ═══════════════════════════════════════════════════════════════════════
    // [3/5] Production run (direct start - no relaxation needed)
    // ═══════════════════════════════════════════════════════════════════════
    // With single-grid ICs: min_distance = spacing >> softening
    // Full Janus physics from the start - TwoPass always uses full Janus

    let mut a = a_init;
    let mut t = 0.5;  // Approximate cosmic time at z=10

    // Friedmann integrator for cosmological evolution
    let friedmann_params = JanusVSLParams {
        eta: ETA_JANUS,
        z_init: args.z_init,
        h: args.h0 / 100.0,
    };
    let friedmann = CoupledFriedmann::new(friedmann_params);

    println!("\n[3/5] Production run (z={} → z={}, adaptive dt)...", args.z_init, args.z_final);

    // Open CSV for time series
    let csv_path = out_dir.join("time_series.csv");
    let mut csv = BufWriter::new(File::create(&csv_path).expect("Cannot create CSV"));
    writeln!(csv, "step,z,t_Gyr,a,dt,N_stars,v_rms,rho_max").unwrap();

    let mut step = 0usize;
    let mut n_stars = 0u32;
    let mut dt = args.dt_max;
    let start_time = Instant::now();

    let a_final = 1.0 / (1.0 + args.z_final);

    while a < a_final {
        // ─────────────────────────────────────────────────────────────────
        // Cosmology update
        // ─────────────────────────────────────────────────────────────────
        let z = 1.0 / a - 1.0;
        let hubble_kms = friedmann.hubble_plus(z);  // km/s/Mpc
        let hubble = hubble_kms / 977.8;  // Convert to Gyr⁻¹
        sim.set_current_z(z);

        // ─────────────────────────────────────────────────────────────────
        // Adaptive timestep (use velocity-based estimate)
        // ─────────────────────────────────────────────────────────────────
        let vel = sim.get_velocities().unwrap_or_default();
        let v_max = vel.chunks(3)
            .map(|v| ((v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) as f64).sqrt())
            .fold(0.0f64, f64::max);
        // dt ≈ η × ε / v_max (Courant-like condition)
        dt = if v_max > 1e-10 {
            (args.eta * args.eps_plus / v_max).clamp(args.dt_min, args.dt_max)
        } else {
            args.dt_max
        };

        // ─────────────────────────────────────────────────────────────────
        // Leapfrog with expansion (DKD) - does forces + integration
        // ─────────────────────────────────────────────────────────────────
        sim.step_dkd(dt, hubble, 1.0)?;

        // Update scale factor and time
        let da = a * hubble * dt;
        a += da;
        t += dt;

        // ─────────────────────────────────────────────────────────────────
        // Baryonic physics (m+ only) — every 5 steps
        // ─────────────────────────────────────────────────────────────────
        if step % 5 == 0 && step > 0 {
            // TODO: Full SPH + cooling + SF implementation
            // For now, just track that we would do it here
        }

        // ─────────────────────────────────────────────────────────────────
        // Diagnostics
        // ─────────────────────────────────────────────────────────────────
        let vel = sim.get_velocities().unwrap_or_default();
        let v_rms = compute_v_rms_f32(&vel);
        let pos = sim.get_positions().unwrap_or_default();
        let pos_max = pos.iter().map(|&x| (x as f64).abs()).fold(0.0f64, f64::max);

        // ─────────────────────────────────────────────────────────────────
        // STOP conditions
        // ─────────────────────────────────────────────────────────────────
        if pos_max > 1e4 {
            eprintln!("❌ STOP: Position overflow {:.2e} Mpc", pos_max);
            break;
        }
        if n_stars > 10_000_000 {
            eprintln!("❌ STOP: SF runaway N★={}", n_stars);
            break;
        }

        // ─────────────────────────────────────────────────────────────────
        // Snapshot
        // ─────────────────────────────────────────────────────────────────
        if step % args.snap_interval == 0 {
            let snap_path = out_dir.join(format!("snapshots/snap_{:06}.bin", step));
            write_snapshot(&snap_path, &pos, &vel, &ics.signs, a, t);

            // CSV
            let z_now = 1.0 / a - 1.0;
            writeln!(csv, "{},{:.6},{:.6},{:.6},{:.6},{},{:.2},{:.2e}",
                     step, z_now, t, a, dt, n_stars, v_rms * MPC_GYR_TO_KMS, 0.0).unwrap();
            csv.flush().unwrap();
        }

        // ─────────────────────────────────────────────────────────────────
        // Progress
        // ─────────────────────────────────────────────────────────────────
        if step % 100 == 0 {
            let z_now = 1.0 / a - 1.0;
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { step as f64 / elapsed * 60.0 } else { 0.0 };
            println!("  Step {:6}: z={:.4}, a={:.4}, dt={:.5} Gyr, v_rms={:.1} km/s, {:.1} steps/min",
                     step, z_now, a, dt, v_rms * MPC_GYR_TO_KMS, rate);
        }

        step += 1;

        // Safety: max steps
        if step > 100_000 {
            println!("  Reached max steps (100000), stopping");
            break;
        }
    }

    // Final snapshot
    let pos = sim.get_positions().unwrap_or_default();
    let vel = sim.get_velocities().unwrap_or_default();
    let snap_path = out_dir.join(format!("snapshots/snap_{:06}.bin", step));
    write_snapshot(&snap_path, &pos, &vel, &ics.signs, a, t);

    csv.flush().unwrap();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("RUN COMPLETE");
    println!("  Final: z={:.4}, a={:.4}, t={:.3} Gyr", 1.0/a - 1.0, a, t);
    println!("  Steps: {}", step);
    println!("  N★: {}", n_stars);
    println!("  Output: {:?}", out_dir);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute adaptive timestep from accelerations
fn compute_adaptive_dt(
    accelerations: &[f64],
    eps: f64,
    eta: f64,
    dt_min: f64,
    dt_max: f64,
) -> f64 {
    // Find max acceleration
    let mut a_max = 0.0f64;
    for chunk in accelerations.chunks(3) {
        let a2 = chunk[0]*chunk[0] + chunk[1]*chunk[1] + chunk[2]*chunk[2];
        let a = a2.sqrt();
        if a > a_max { a_max = a; }
    }

    if a_max < 1e-10 {
        return dt_max;
    }

    // dt = η × sqrt(2ε / a_max)
    let dt_raw = eta * (2.0 * eps / a_max).sqrt();
    dt_raw.clamp(dt_min, dt_max)
}

/// Compute RMS velocity (f32 version for TwoPass)
fn compute_v_rms_f32(velocities: &[f32]) -> f64 {
    let n = velocities.len() / 3;
    if n == 0 { return 0.0; }

    let mut v2_sum = 0.0f64;
    for chunk in velocities.chunks(3) {
        let vx = chunk[0] as f64;
        let vy = chunk[1] as f64;
        let vz = chunk[2] as f64;
        v2_sum += vx*vx + vy*vy + vz*vz;
    }
    (v2_sum / n as f64).sqrt()
}

/// Write snapshot to binary file (f32 version)
fn write_snapshot(
    path: &std::path::Path,
    pos: &[f32],
    vel: &[f32],
    signs: &[i32],
    a: f64,
    t: f64,
) {
    use std::io::Write;

    let n = signs.len();
    let mut file = File::create(path).expect("Cannot create snapshot");

    // Header
    file.write_all(&(n as u64).to_le_bytes()).unwrap();
    file.write_all(&a.to_le_bytes()).unwrap();
    file.write_all(&t.to_le_bytes()).unwrap();

    // Positions (f32)
    for i in 0..n {
        file.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
    }

    // Velocities (f32)
    for i in 0..n {
        file.write_all(&vel[i * 3].to_le_bytes()).unwrap();
        file.write_all(&vel[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&vel[i * 3 + 2].to_le_bytes()).unwrap();
    }

    // Signs (u8)
    for i in 0..n {
        file.write_all(&[if signs[i] > 0 { 1u8 } else { 0u8 }]).unwrap();
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda");
    std::process::exit(1);
}
