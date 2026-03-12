//! Exact February 2M reference reproduction
//! Uses GpuNBodySimulation::new() + virialize() (not new_with_state + virialize_sampled)

use std::fs::{File, create_dir_all};
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const N_POSITIVE: usize = 1_000_000;
const N_NEGATIVE: usize = 1_000_000;
const BOX_SIZE: f64 = 271.0;
const Z_INIT: f64 = 5.0;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 10000;

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  February 2M Reference - EXACT REPRODUCTION");
    println!("  Using GpuNBodySimulation::new() + virialize()");
    println!("═══════════════════════════════════════════════════════════\n");

    let output_dir = "/app/output/ref_2M_february";
    create_dir_all(output_dir).expect("Failed to create output dir");

    println!("Parameters:");
    println!("  N+ = {}, N- = {}", N_POSITIVE, N_NEGATIVE);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  steps = {}\n", TOTAL_STEPS);

    // Use GpuNBodySimulation::new() - EXACTLY like February
    println!("Creating simulation with GpuNBodySimulation::new()...");
    let mut sim = GpuNBodySimulation::new(N_POSITIVE, N_NEGATIVE, BOX_SIZE)
        .expect("Failed to create simulation");

    sim.set_theta(THETA);

    // Use virialize() - EXACTLY like February (not virialize_sampled)
    println!("\nVirializing with virialize() (full PE calculation)...");
    sim.virialize().expect("Virialization failed");

    // Setup cosmology - February convention
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (TOTAL_STEPS as f64);
    // February convention: dtau_per_dt = range / (10000 * dt)
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * DT);

    println!("\nCosmology:");
    println!("  η = {}", eta);
    println!("  τ_start = {:.4}, τ_end = {:.4}", cosmo.tau_start, cosmo.tau_end);
    println!("  dtau_cosmo = {:.6}", dtau_cosmo);
    println!("  dtau_per_dt = {:.6}", dtau_per_dt);

    // CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke_ratio,seg,step_ms").unwrap();

    // Initial state
    let ke_0 = sim.kinetic_energy().expect("KE failed");
    let seg_0 = sim.segregation_distance().expect("Seg failed");

    writeln!(csv, "0,{:.4},{:.6},{:.6},0", Z_INIT, 1.0, seg_0).unwrap();

    println!("\n══════════════════════════════════════════════════");
    println!("  Starting February-style 2M run");
    println!("══════════════════════════════════════════════════\n");
    println!("Step 0: z={:.2}, KE/KE₀=1.000, Seg={:.4}", Z_INIT, seg_0);

    let start = Instant::now();
    let mut ke_max = 1.0f64;
    let mut seg_max = seg_0;
    let mut seg_max_step = 0usize;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        // Get cosmological parameters
        let current_tau = cosmo.tau_start + (step as f64) * dtau_cosmo;
        let (a, h) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;

        // Step - February style
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        let ke = sim.kinetic_energy().expect("KE failed");
        let ke_ratio = ke / ke_0;
        let seg = sim.segregation_distance().expect("Seg failed");

        ke_max = ke_max.max(ke_ratio);
        if seg > seg_max {
            seg_max = seg;
            seg_max_step = step;
        }

        writeln!(csv, "{},{:.4},{:.6},{:.6},{:.1}", step, z, ke_ratio, seg, step_ms).unwrap();

        if step % 500 == 0 {
            let rate = step as f64 / start.elapsed().as_secs_f64();
            let eta_min = (TOTAL_STEPS - step) as f64 / rate / 60.0;
            println!("Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}, Seg_max={:.4} ({:.1} steps/s, ETA {:.0}min)",
                     step, z, ke_ratio, seg, seg_max, rate, eta_min);
        }

        // Early warning
        if step == 1000 && seg_max < 0.01 {
            println!("\n⚠️  WARNING: Seg_max = {:.4} < 0.01 at step 1000", seg_max);
        }
    }

    csv.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n══════════════════════════════════════════════════");
    println!("  February 2M Reference Complete");
    println!("══════════════════════════════════════════════════");
    println!("  Time: {:.1}min ({:.0} ms/step)", elapsed / 60.0, elapsed * 1000.0 / TOTAL_STEPS as f64);
    println!("  Seg_0: {:.4}, Seg_max: {:.4} @ step {}", seg_0, seg_max, seg_max_step);
    println!("  KE max: {:.3}", ke_max);

    if seg_max > 0.3 {
        println!("\n  ✓ SUCCESS: February behavior reproduced!");
    } else {
        println!("\n  ❌ Still no segregation growth");
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
