//! Janus 8M Run — θ=0.7 (twopass GPU Karras)
//! Target: validate S_max ≈ 0.459 vs reference

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;  // Accepted for publication (<3% force error)
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 15000;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 8M — θ=0.7 (twopass GPU Karras)                        ║");
    println!("║   Target: S_max ≈ 0.459                                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {}, N- = {}", n_positive, n_negative);
    println!("  η = {}", ETA);
    println!("  θ = {} (<3% force error)", THETA);
    println!("  dt = {}", DT);
    println!("  box = {:.2}", box_size);
    println!();

    // Cosmological expansion
    println!("--- Cosmological Expansion ---");
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let n_steps_to_z0 = 12000.0;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    println!("  z_init = {:.2}", 1.0/a_init - 1.0);
    println!("  a_init = {:.6}", a_init);
    println!("  H_init = {:.6}", h_init);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d_%H%M").to_string();
    let output_dir = format!("/app/output/8M_theta07_{}", date);
    fs::create_dir_all(&output_dir)?;
    println!("Output: {}\n", output_dir);

    // Create simulation
    println!("Creating simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    println!("Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Time series file
    let ts_filename = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_filename)?);
    writeln!(ts_file, "step,time,redshift,scale_factor,hubble,ke,segregation,step_time_ms")?;

    let mut tau = cosmo.tau_start;
    let mut seg_max = 0.0f64;

    println!("Starting simulation...\n");
    println!("{:>6} | {:>6} | {:>8} | {:>12} | {:>8} | {:>6}",
        "step", "z", "S", "S_max", "KE", "ms");
    println!("{}", "-".repeat(65));

    for step in 1..=MAX_STEPS {
        let t_step = Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_dkd(DT, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;

        let step_time = t_step.elapsed().as_millis();
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        seg_max = seg_max.max(seg);

        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6e},{}",
            step, step as f64 * DT, z, a, h, ke, seg, step_time)?;

        if step % 100 == 0 || step <= 10 {
            ts_file.flush()?;
            println!("{:>6} | {:>6.2} | {:>8.4e} | {:>12.4e} | {:>8.2e} | {:>6}",
                step, z, seg, seg_max, ke, step_time);
        }

        if ke.is_nan() || ke.is_infinite() {
            println!("\n=== KE explosion, stopping ===");
            break;
        }

        if z < 0.01 {
            println!("\n=== Reached z ≈ 0 ===");
            break;
        }
    }

    ts_file.flush()?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("RESULTS:");
    println!("  S_max = {:.6}", seg_max);
    println!("  Target: 0.459 (±10%)");
    if seg_max > 0.4 && seg_max < 0.52 {
        println!("  ✓ VALIDATED");
    } else {
        println!("  ✗ OUT OF RANGE");
    }
    println!("  Output: {}", ts_filename);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
