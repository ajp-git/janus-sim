//! Quick 8M test - 50 steps only to verify physics
//! Expected: ~2 hours total

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::time::Instant;

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 50;  // Only 50 steps for quick validation

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 8M Quick Test — θ=0.7 (50 steps only)                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  θ = {} (<3% force error)", THETA);
    println!("  Steps = {} (quick validation)", MAX_STEPS);
    println!();

    // Cosmological expansion
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);
    let n_steps_to_z0 = 12000.0;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    // Create simulation
    println!("Creating simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_theta(THETA);
    println!("Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    let mut tau = cosmo.tau_start;
    let mut seg_max = 0.0f64;
    let mut total_time = 0u128;

    println!("Starting {} steps...\n", MAX_STEPS);
    println!("{:>6} | {:>6} | {:>10} | {:>10} | {:>8} | {:>6}",
        "step", "z", "S", "S_max", "KE", "ms");
    println!("{}", "-".repeat(70));

    for step in 1..=MAX_STEPS {
        let t_step = Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_dkd(DT, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;

        let step_time = t_step.elapsed().as_millis();
        total_time += step_time;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        seg_max = seg_max.max(seg);

        println!("{:>6} | {:>6.2} | {:>10.4e} | {:>10.4e} | {:>8.2e} | {:>6}",
            step, z, seg, seg_max, ke, step_time);

        if ke.is_nan() || ke.is_infinite() {
            println!("\n=== KE explosion, stopping ===");
            break;
        }
    }

    let avg_ms = total_time / MAX_STEPS as u128;
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("RESULTS ({} steps):", MAX_STEPS);
    println!("  S_max after {} steps: {:.6}", MAX_STEPS, seg_max);
    println!("  Average step time: {} ms", avg_ms);
    println!("  Full run estimate (12000 steps): {:.1} hours",
        (avg_ms as f64 * 12000.0) / 3600000.0);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { println!("CUDA required"); }
