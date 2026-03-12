//! Test VRAM TreePM pour déterminer N_max
//! Modifiez N_TEST pour tester différentes valeurs
//! Usage: cargo run --release --features cuda,cufft --bin test_treepm_nmax

use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// ============================================
// MODIFIER ICI POUR TESTER DIFFÉRENTS N
// ============================================
#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_TEST: usize = 30_000_000;  // 30M test PM 512³
// ============================================

#[cfg(all(feature = "cuda", feature = "cufft"))]
const Z_INIT: f64 = 5.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const THETA: f64 = 0.7;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const DT: f64 = 0.01;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const TEST_STEPS: usize = 100;  // Quick test 30M PM 512³
#[cfg(all(feature = "cuda", feature = "cufft"))]
const ETA: f64 = 1.045;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SOFTENING: f64 = 0.65;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // FIX-015: box = n_side × 2.15 Mpc
    let n_side = (N_TEST as f64).cbrt().round() as usize;
    let box_size = n_side as f64 * 2.15;
    let r_cut = box_size / 16.0;

    let n_positive = N_TEST / 2;
    let n_negative = N_TEST / 2;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Test VRAM TreePM - N = {} ({:.0}M)", N_TEST, N_TEST as f64 / 1e6);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Paramètres:");
    println!("  N+ = {}, N- = {}", n_positive, n_negative);
    println!("  n_side = {}", n_side);
    println!("  Box = {:.1} Mpc", box_size);
    println!("  r_cut = {:.1} Mpc", r_cut);
    println!("  Test steps = {}", TEST_STEPS);
    println!();

    // =========================================================================
    // TIMING: Création ICs
    // =========================================================================
    println!(">>> TIMING: Création ICs...");
    let t_ics_start = Instant::now();

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_softening(SOFTENING);
    sim.set_theta(THETA);

    let t_ics = t_ics_start.elapsed();
    println!("    Création ICs: {:.1}s", t_ics.as_secs_f64());

    // =========================================================================
    // TIMING: Virialization
    // =========================================================================
    // n_sample proportionnel à N: au moins 0.5% de N/2, minimum 10000
    let n_sample = ((N_TEST / 2) as f64 * 0.005).max(10000.0) as usize;
    println!(">>> TIMING: Virialization (sampled n={})...", n_sample);
    let t_virial_start = Instant::now();

    sim.virialize_sampled(n_sample)?;

    let t_virial = t_virial_start.elapsed();
    println!("    Virialization: {:.1}s ({:.1} min)",
        t_virial.as_secs_f64(), t_virial.as_secs_f64() / 60.0);

    let t_total_ics = t_ics.as_secs_f64() + t_virial.as_secs_f64();
    println!();
    println!("  TOTAL ICs: {:.1}s ({:.1} min)", t_total_ics, t_total_ics / 60.0);
    println!();

    // Cosmologie
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    // FIX-016: dtau_per_dt = τ_range / (10000 × DT) - référence février
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / 10000.0;
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * DT);

    println!("Cosmologie:");
    println!("  dtau_per_dt = {:.6} (FIX-016 référence 10000 steps)", dtau_per_dt);
    println!();

    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;

    println!("Initial: KE₀ = {:.4e}, Seg₀ = {:.4}", ke0, seg0);
    println!();

    // =========================================================================
    // Simulation test
    // =========================================================================
    println!(">>> Running {} test steps...", TEST_STEPS);
    println!(">>> MESURER nvidia-smi MAINTENANT <<<");
    println!();

    let start = Instant::now();
    let mut tau = cosmo.tau_start;

    for step in 1..=TEST_STEPS {
        let (a, h) = cosmo.get_params_at_tau(tau);
        sim.step_treepm_gpu_morton(DT, r_cut, h, dtau_per_dt)?;
        tau += dtau_per_step;

        if step % 10 == 0 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;
            let z = 1.0 / a - 1.0;
            println!("  Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}",
                step, z, ke/ke0, seg);
        }
    }

    let elapsed = start.elapsed();
    let ms_per_step = elapsed.as_millis() as f64 / TEST_STEPS as f64;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  RÉSUMÉ TreePM N = {:.0}M", N_TEST as f64 / 1e6);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Création ICs:     {:.1}s", t_ics.as_secs_f64());
    println!("  Virialization:    {:.1}s ({:.1} min)", t_virial.as_secs_f64(), t_virial.as_secs_f64() / 60.0);
    println!("  Total ICs:        {:.1}s ({:.1} min)", t_total_ics, t_total_ics / 60.0);
    println!();
    println!("  Simulation:       {:.0} ms/step", ms_per_step);
    println!("  ETA 10000 steps:  {:.1} heures", ms_per_step * 10000.0 / 3600000.0);
    println!();
    println!(">>> VÉRIFIER nvidia-smi POUR VRAM <<<");

    // Pause pour nvidia-smi
    std::thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda and cufft features");
}
