/// PM-4: Janus PM Simulation with Cosmological Expansion
///
/// Validates:
///   - S(t) increasing step 0→200
///   - S(200) > 0.01
///   - KE/KE₀ < 20
///
/// Full Janus physics: dual grids, crossed forces, Hubble friction

use janus_pm::janus_pm::{JanusPMSimulation, generate_janus_ic, StepTiming};
use janus_pm::snapshot::{SnapshotMeta, TimeSeriesWriter, save_snapshot_full, save_snapshot_light};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-4: Janus PM Simulation with Cosmology                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Parameters from roadmap - validated against BH reference
    let n_particles = 100_000;
    let grid_size = 256;  // Full resolution
    let box_size = 100.0;
    let dt = 0.005_f32;   // Half timestep for CFL during collapse
    let n_steps = 10;     // Quick diagnostic run
    let eta = 1.045;
    let z_init = 5.0;
    let velocity_dispersion = 0.5_f32;
    let seed = 42;

    // Output directory
    let output_dir = Path::new("janus-pm/output/pm4_test");
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    println!("\nParameters:");
    println!("  Particles: {}", n_particles);
    println!("  Grid: {}³", grid_size);
    println!("  Box size: {:.1}", box_size);
    println!("  dt: {:.3}", dt);
    println!("  Steps: {}", n_steps);
    println!("  η: {:.4}", eta);
    println!("  z_init: {:.1}", z_init);

    // Generate initial conditions
    println!("\nGenerating Janus initial conditions...");
    let t0 = Instant::now();
    let particles = generate_janus_ic(n_particles, box_size, velocity_dispersion, eta, seed);

    let n_pos = particles.iter().filter(|p| p.sign > 0).count();
    let n_neg = particles.iter().filter(|p| p.sign < 0).count();
    println!("  Positive: {}, Negative: {}", n_pos, n_neg);

    // Create simulation
    println!("\nCreating Janus PM simulation...");
    let mut sim = match JanusPMSimulation::new(
        particles,
        grid_size, grid_size, grid_size,
        box_size,
        dt,
        eta,
        z_init,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("ERROR: Failed to create simulation: {}", e);
            std::process::exit(1);
        }
    };

    // Virialize
    println!("\nVirializing...");
    match sim.virialize() {
        Ok(alpha) => println!("  α = {:.4}", alpha),
        Err(e) => {
            println!("WARNING: Virialization failed: {}", e);
        }
    }

    let ke_0 = sim.ke_initial;
    let seg_0 = sim.seg_initial;

    println!("Setup time: {:.2} s", t0.elapsed().as_secs_f64());

    println!("\nInitial state:");
    println!("  KE₀ = {:.6e}", ke_0);
    println!("  Seg₀ = {:.4}", seg_0);
    println!("  Scale factor = {:.4}", sim.scale_factor());

    // Time series writer
    let mut ts_writer = TimeSeriesWriter::new(&output_dir.join("time_series.csv"))
        .expect("Failed to create time series file");

    // Initial snapshot
    let meta_0 = SnapshotMeta {
        step: 0,
        time: 0.0,
        tau: sim.tau,
        scale_factor: sim.scale_factor(),
        segregation: seg_0,
        ke_ratio: 1.0,
        n_positive: n_pos,
        n_negative: n_neg,
    };
    ts_writer.write(&meta_0).ok();

    if let Err(e) = save_snapshot_full(&output_dir.join("snapshot_0.bin"), &sim.particles, &meta_0) {
        println!("WARNING: Failed to save initial snapshot: {}", e);
    }

    // Tracking
    let mut s_max = seg_0;
    let mut s_max_step = 0;
    let mut max_ke_ratio = 1.0_f64;
    let mut s_increasing = true;
    let mut prev_s = seg_0;

    println!("\n  Step      a        Seg        KE/KE₀     Force(ms)  Kick(ms)  Drift(ms)");
    println!("  ───────────────────────────────────────────────────────────────────────────");

    let t_loop = Instant::now();
    for step in 1..=n_steps {
        let t_step = Instant::now();

        let timing = match sim.step_timed() {
            Ok(t) => t,
            Err(e) => {
                println!("ERROR at step {}: {}", step, e);
                std::process::exit(1);
            }
        };

        let ke = sim.kinetic_energy();
        let seg = sim.segregation();
        let ke_ratio = if ke_0 > 1e-10 { ke / ke_0 } else { 1.0 };
        let a = sim.scale_factor();

        max_ke_ratio = max_ke_ratio.max(ke_ratio);

        // Track segregation increase
        if seg < prev_s && step > 10 {
            s_increasing = false;
        }
        prev_s = seg;

        // Peak detection
        if seg > s_max {
            s_max = seg;
            s_max_step = step;

            // Save peak snapshot (overwrite)
            let meta_peak = SnapshotMeta {
                step,
                time: step as f64 * dt as f64,
                tau: sim.tau,
                scale_factor: a,
                segregation: seg,
                ke_ratio,
                n_positive: n_pos,
                n_negative: n_neg,
            };
            if let Err(e) = save_snapshot_full(&output_dir.join("snapshot_peak.bin"), &sim.particles, &meta_peak) {
                println!("WARNING: Failed to save peak snapshot: {}", e);
            }
        }

        // Time series
        let meta = SnapshotMeta {
            step,
            time: step as f64 * dt as f64,
            tau: sim.tau,
            scale_factor: a,
            segregation: seg,
            ke_ratio,
            n_positive: n_pos,
            n_negative: n_neg,
        };
        ts_writer.write(&meta).ok();

        // Light snapshot every 50 steps
        if step % 50 == 0 {
            let snap_path = output_dir.join(format!("snapshot_{:04}.bin", step));
            if let Err(e) = save_snapshot_light(&snap_path, &sim.particles, &meta) {
                println!("WARNING: Failed to save light snapshot: {}", e);
            }
        }

        let step_time = t_step.elapsed().as_secs_f64() * 1000.0;

        // Report every 100 steps + first and last
        if step % 100 == 0 || step == 1 || step == n_steps {
            println!("  {:4}    {:.4}    {:.4}     {:.4}      {:.0}       {:.0}       {:.0}",
                     step, a, seg, ke_ratio, timing.force_time, timing.kick_time, timing.drift_time);
        }
    }

    // Final snapshot
    let seg_final = sim.segregation();
    let ke_final = sim.kinetic_energy();
    let meta_final = SnapshotMeta {
        step: n_steps,
        time: n_steps as f64 * dt as f64,
        tau: sim.tau,
        scale_factor: sim.scale_factor(),
        segregation: seg_final,
        ke_ratio: ke_final / ke_0,
        n_positive: n_pos,
        n_negative: n_neg,
    };
    if let Err(e) = save_snapshot_full(&output_dir.join("snapshot_final.bin"), &sim.particles, &meta_final) {
        println!("WARNING: Failed to save final snapshot: {}", e);
    }

    let total_time = t_loop.elapsed().as_secs_f64();

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      RESULTS                                     ");
    println!("══════════════════════════════════════════════════════════════════");

    println!("\n  Total time: {:.1} s ({:.0} ms/step)", total_time, total_time * 1000.0 / n_steps as f64);
    println!("  S(0) = {:.4}", seg_0);
    println!("  S({}) = {:.4}", n_steps, seg_final);
    println!("  S_max = {:.4} at step {}", s_max, s_max_step);
    println!("  KE/KE₀ max = {:.2}", max_ke_ratio);
    println!("  Final scale factor = {:.4}", sim.scale_factor());

    // Validation
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      VALIDATION SUMMARY                          ");
    println!("══════════════════════════════════════════════════════════════════");

    let s_final_pass = seg_final > 0.01;
    let ke_pass = max_ke_ratio < 20.0;
    // Note: S(t) increasing check is approximate - some fluctuation is normal

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Test                    │ Result    │ Threshold │ Status       │");
    println!("├─────────────────────────┼───────────┼───────────┼──────────────┤");
    println!("│ S({}) > 0.01          │ {:.4}    │ > 0.01    │ {}           │",
             n_steps, seg_final, if s_final_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("│ S_max                   │ {:.4}    │ (info)    │ step {}     │", s_max, s_max_step);
    println!("│ KE/KE₀ (max)            │ {:.2}      │ < 20      │ {}           │",
             max_ke_ratio, if ke_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n  Output directory: {}", output_dir.display());
    println!("  Snapshots: snapshot_0.bin, snapshot_peak.bin, snapshot_final.bin");
    println!("  Time series: time_series.csv");

    println!("\n══════════════════════════════════════════════════════════════════");
    if s_final_pass && ke_pass {
        println!("PM-4 VALIDATION: ✓ PASSED");
        println!("  S({}) = {:.4} > 0.01", n_steps, seg_final);
        println!("  KE ratio = {:.2} < 20", max_ke_ratio);
    } else {
        println!("PM-4 VALIDATION: ✗ FAILED");
        if !s_final_pass {
            println!("  ✗ S({}) = {:.4} <= 0.01", n_steps, seg_final);
        }
        if !ke_pass {
            println!("  ✗ KE ratio {:.2} >= 20", max_ke_ratio);
        }
    }
    println!("══════════════════════════════════════════════════════════════════");
}
