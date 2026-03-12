//! Test VRAM pour déterminer N_max
//! Mesure séparément : création ICs, virialization, simulation
//! Usage: cargo run --release --features cuda --bin test_vram_nmax

use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[cfg(feature = "cuda")]
const N_TEST: usize = 12_000_000;  // 12M - à ajuster selon résultat
#[cfg(feature = "cuda")]
const Z_INIT: f64 = 5.0;
#[cfg(feature = "cuda")]
const THETA: f64 = 0.7;
#[cfg(feature = "cuda")]
const DT: f64 = 0.01;
#[cfg(feature = "cuda")]
const TEST_STEPS: usize = 100;
#[cfg(feature = "cuda")]
const ETA: f64 = 1.045;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_positive = N_TEST / 2;
    let n_negative = N_TEST / 2;

    // FIX-015: box = n_side × 2.15 Mpc
    let n_side = (N_TEST as f64).cbrt().round() as usize;
    let box_size = n_side as f64 * 2.15;
    let softening = 0.65;  // 0.3 × spacing

    println!("═══════════════════════════════════════════════════════════");
    println!("  Test VRAM + Timing - N = {} ({:.1}M)", N_TEST, N_TEST as f64 / 1e6);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Paramètres:");
    println!("  N+ = {}, N- = {}", n_positive, n_negative);
    println!("  n_side = {}", n_side);
    println!("  Box = {:.1} Mpc", box_size);
    println!("  Softening = {} Mpc", softening);
    println!("  θ = {}", THETA);
    println!("  Test steps = {}", TEST_STEPS);
    println!();

    // =========================================================================
    // TIMING: Création ICs (new() - placement des particules)
    // =========================================================================
    println!(">>> TIMING: Création ICs...");
    let t_ics_start = Instant::now();

    let mut sim = GpuNBodySimulation::new(n_positive, n_negative, box_size)?;
    sim.set_softening(softening);
    sim.set_theta(THETA);

    let t_ics = t_ics_start.elapsed();
    println!("    Création ICs: {:.1}s", t_ics.as_secs_f64());

    // =========================================================================
    // TIMING: Virialization (PE full - CPU tree O(N log N))
    // =========================================================================
    println!(">>> TIMING: Virialization (PE full)...");
    let t_virial_start = Instant::now();

    sim.virialize()?;

    let t_virial = t_virial_start.elapsed();
    println!("    Virialization: {:.1}s ({:.1} min)",
        t_virial.as_secs_f64(), t_virial.as_secs_f64() / 60.0);

    // =========================================================================
    // Total ICs time
    // =========================================================================
    let t_total_ics = t_ics.as_secs_f64() + t_virial.as_secs_f64();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  TOTAL ICs (création + virialization): {:.1}s ({:.1} min)",
        t_total_ics, t_total_ics / 60.0);
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Cosmologie
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    // FIX-016: dtau_per_dt = τ_range / (TOTAL_STEPS × DT)
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * DT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / 10000.0;

    println!("Cosmologie:");
    println!("  dtau_per_dt = {:.6} (FIX-016)", dtau_per_dt);
    println!();

    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation_distance()?;

    println!("Initial: KE₀ = {:.4e}, Seg₀ = {:.4}", ke0, seg0);
    println!();

    // =========================================================================
    // TIMING: Simulation steps
    // =========================================================================
    println!(">>> TIMING: {} test steps...", TEST_STEPS);
    println!(">>> Mesurer nvidia-smi MAINTENANT <<<");
    println!();

    let t_sim_start = Instant::now();
    let mut tau = cosmo.tau_start;

    for step in 1..=TEST_STEPS {
        let (a, h) = cosmo.get_params_at_tau(tau);
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)?;
        tau += dtau_per_step;

        if step % 10 == 0 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation_distance()?;
            let z = 1.0 / a - 1.0;
            println!("  Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}",
                step, z, ke/ke0, seg);
        }
    }

    let t_sim = t_sim_start.elapsed();
    let ms_per_step = t_sim.as_millis() as f64 / TEST_STEPS as f64;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  RÉSUMÉ TIMING - N = {:.1}M", N_TEST as f64 / 1e6);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Création ICs:     {:.1}s", t_ics.as_secs_f64());
    println!("  Virialization:    {:.1}s ({:.1} min)", t_virial.as_secs_f64(), t_virial.as_secs_f64() / 60.0);
    println!("  Total ICs:        {:.1}s ({:.1} min)", t_total_ics, t_total_ics / 60.0);
    println!();
    println!("  Simulation:       {:.0} ms/step", ms_per_step);
    println!("  ETA 10000 steps:  {:.1} heures", ms_per_step * 10000.0 / 3600000.0);
    println!();
    println!(">>> Vérifier nvidia-smi pour VRAM <<<");

    // Pause pour lire nvidia-smi
    std::thread::sleep(std::time::Duration::from_secs(5));

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
