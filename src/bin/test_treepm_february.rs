//! Test TreePM + ICs février
//! Vérifie stabilité avant scaling à 40-50M
//! Usage: cargo run --release --features cuda,cufft --bin test_treepm_february

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// Test parameters - 500K
#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_PARTICLES: usize = 500_000;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const Z_INIT: f64 = 5.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const THETA: f64 = 0.7;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const DT: f64 = 0.01;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const TOTAL_STEPS: usize = 2000;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const ETA: f64 = 1.045;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SOFTENING: f64 = 0.65;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // FIX-015: box = n_side × 2.15 Mpc
    let n_side = (N_PARTICLES as f64).cbrt().round() as usize;
    let box_size = n_side as f64 * 2.15;
    let r_cut = box_size / 16.0;

    // ICs février: 50% positive, 50% negative
    let n_positive = N_PARTICLES / 2;
    let n_negative = N_PARTICLES / 2;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Test TreePM + ICs février - N = {} ({:.0}K)", N_PARTICLES, N_PARTICLES as f64 / 1e3);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Paramètres:");
    println!("  N+ = {}, N- = {}", n_positive, n_negative);
    println!("  n_side = {}", n_side);
    println!("  Box = {:.1} Mpc", box_size);
    println!("  r_cut = {:.1} Mpc (box/16)", r_cut);
    println!("  Softening = {} Mpc", SOFTENING);
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  steps = {}", TOTAL_STEPS);
    println!();

    // =========================================================================
    // TIMING: Création ICs
    // =========================================================================
    println!(">>> TIMING: Création ICs (new() - positifs d'abord)...");
    let t_ics_start = Instant::now();

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, box_size)?;
    sim.set_softening(SOFTENING);
    sim.set_theta(THETA);

    let t_ics = t_ics_start.elapsed();
    println!("    Création ICs: {:.1}s", t_ics.as_secs_f64());

    // =========================================================================
    // TIMING: Virialization
    // =========================================================================
    println!(">>> TIMING: Virialization (sampled n=10000)...");
    let t_virial_start = Instant::now();

    sim.virialize_sampled(10000)?;

    let t_virial = t_virial_start.elapsed();
    println!("    Virialization: {:.1}s", t_virial.as_secs_f64());

    let t_total_ics = t_ics.as_secs_f64() + t_virial.as_secs_f64();
    println!();
    println!("  TOTAL ICs: {:.1}s", t_total_ics);
    println!();

    // Cosmologie - FIX-016
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    // FIX-016: dtau_per_dt = τ_range / (TOTAL_STEPS × DT)
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);

    println!("Cosmologie:");
    println!("  η = {}", ETA);
    println!("  τ_start = {:.4}, τ_end = {:.4}", cosmo.tau_start, cosmo.tau_end);
    println!("  dtau_per_dt = {:.6} (FIX-016)", dtau_per_dt);
    println!();

    // Output
    let output_dir = "/app/output/treepm_500k_test";
    fs::create_dir_all(output_dir)?;
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path)?;
    writeln!(csv, "step,z,a,H,KE,KE_ratio,Seg,Seg_max")?;

    // Initial state
    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;

    println!("Initial: KE₀ = {:.4e}, Seg₀ = {:.4}", ke0, seg0);
    println!();

    // =========================================================================
    // Simulation loop
    // =========================================================================
    println!(">>> Running {} steps (TreePM)...", TOTAL_STEPS);
    println!();

    let start = Instant::now();
    let mut tau = cosmo.tau_start;
    let mut seg_max = seg0;
    let mut ke_max = 1.0f64;

    for step in 0..=TOTAL_STEPS {
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        let ke = sim.kinetic_energy()?;
        let ke_ratio = ke / ke0;
        let seg = sim.segregation()?;

        if seg > seg_max { seg_max = seg; }
        if ke_ratio > ke_max { ke_max = ke_ratio; }

        writeln!(csv, "{},{:.4},{:.6},{:.6},{:.4e},{:.4},{:.4},{:.4}",
            step, z, a, h, ke, ke_ratio, seg, seg_max)?;

        // Report and check criteria
        if step == 0 || step == 5 || step == 100 || step == 500 || step % 500 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let ms_per_step = if step > 0 { elapsed * 1000.0 / step as f64 } else { 0.0 };

            println!("Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}, Seg_max={:.4} ({:.0}ms/step)",
                step, z, ke_ratio, seg, seg_max, ms_per_step);

            // Critères PASS/FAIL
            if step == 5 {
                if ke_ratio < 0.95 || ke_ratio > 1.05 {
                    println!("  ⚠️  Step 5: KE/KE₀={:.3} hors [0.95, 1.05]", ke_ratio);
                } else {
                    println!("  ✓ Step 5: KE/KE₀ stable");
                }
            }
            if step == 100 {
                if ke_ratio >= 0.95 {
                    println!("  ⚠️  Step 100: KE/KE₀={:.3} devrait être < 0.95 (friction)", ke_ratio);
                } else {
                    println!("  ✓ Step 100: friction Hubble active (KE décroît)");
                }
            }
        }

        // FAIL criterion
        if ke_ratio > 10.0 && step < 100 {
            println!("\n🛑 FAIL: KE/KE₀ = {:.2} > 10 avant step 100 - TreePM instable!", ke_ratio);
            break;
        }

        // Step (except last)
        if step < TOTAL_STEPS {
            sim.step_treepm_gpu(DT, r_cut, h, dtau_per_dt)?;
            tau += dtau_per_step;
        }
    }

    let total_time = start.elapsed();
    let ms_per_step = total_time.as_millis() as f64 / TOTAL_STEPS as f64;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  RÉSUMÉ - TreePM 500K");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  ICs:              {:.1}s", t_total_ics);
    println!("  Simulation:       {:.0}ms/step", ms_per_step);
    println!("  Runtime:          {:.1}min", total_time.as_secs_f64() / 60.0);
    println!();
    println!("  KE/KE₀ max:       {:.3}", ke_max);
    println!("  Seg_max:          {:.4}", seg_max);
    println!();

    // Verdict
    let pass = ke_max < 10.0 && seg_max > 0.05;
    if pass {
        println!("  ✅ PASS - TreePM stable avec ICs février");
        println!("     → Procéder à ÉTAPE 2 (mesure N_max)");
    } else {
        println!("  ❌ FAIL - TreePM instable ou pas de ségrégation");
        if ke_max >= 10.0 {
            println!("     KE/KE₀ max = {:.2} >= 10.0", ke_max);
        }
        if seg_max <= 0.05 {
            println!("     Seg_max = {:.4} <= 0.05", seg_max);
        }
    }

    println!();
    println!("Output: {}", csv_path);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda and cufft features: cargo run --release --features cuda,cufft --bin test_treepm_february");
}
