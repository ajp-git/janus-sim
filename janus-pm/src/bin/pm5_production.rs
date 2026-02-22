/// PM-5: Janus PM Production Run — GPU-Only
///
/// Usage: pm5_production [N_PARTICLES] [GRID_SIZE] [N_STEPS]
///   Default: 150M particles, 512³ grid, 1000 steps
///
/// Memory budget (150M on GPU):
///   - Positions f64: 3.6 GB
///   - Velocities f32: 1.8 GB
///   - Signs i8: 0.15 GB
///   - FFT grids: ~2.0 GB
///   - Total GPU: ~9.0 GB < 12 GB

use janus_pm::gpu_simulation::{JanusPMGpu, Particle, generate_janus_ic};
use std::path::Path;
use std::time::Instant;
use std::env;
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command-line arguments
    let n_particles: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(150_000_000);  // Default 150M

    let grid_size: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let n_steps: usize = args.get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-5: Janus PM Production — GPU-Only                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Production parameters
    let box_size = 500.0;
    let dt = 0.005_f32;
    let eta = 1.045;
    let z_init = 5.0;
    let velocity_dispersion = 0.5_f32;
    let seed = 42;

    // Output directory with timestamp
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S");
    let output_dir = Path::new("janus-pm/output").join(format!("pm5_{}", timestamp));
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    println!("\nParameters:");
    println!("  Particles: {} ({:.0}M)", n_particles, n_particles as f64 / 1e6);
    println!("  Grid: {}³", grid_size);
    println!("  Box size: {:.1}", box_size);
    println!("  dt: {:.3}", dt);
    println!("  Steps: {}", n_steps);
    println!("  η: {:.4}", eta);
    println!("  z_init: {:.1}", z_init);
    println!("  Output: {}", output_dir.display());

    // Memory estimation (GPU)
    let particle_mem_gb = (n_particles as f64 * (3.0 * 8.0 + 3.0 * 4.0 + 1.0)) / 1e9;
    let grid_mem_gb = (grid_size as f64).powi(3) * 8.0 * 14.0 / 1e9;  // ~14 buffers
    println!("\nGPU Memory estimate:");
    println!("  Particles: {:.2} GB", particle_mem_gb);
    println!("  Grids + FFT: {:.2} GB", grid_mem_gb);
    println!("  Total: {:.2} GB", particle_mem_gb + grid_mem_gb);

    // Generate initial conditions (CPU)
    println!("\nGenerating {} particles (CPU)...", n_particles);
    let t0 = Instant::now();
    let particles = generate_janus_ic(n_particles, box_size, velocity_dispersion, eta, seed);
    println!("  IC generation: {:.1} s", t0.elapsed().as_secs_f64());

    let n_pos = particles.iter().filter(|p| p.sign > 0).count();
    let n_neg = n_particles - n_pos;
    println!("  Positive: {} ({:.2}%)", n_pos, 100.0 * n_pos as f64 / n_particles as f64);
    println!("  Negative: {} ({:.2}%)", n_neg, 100.0 * n_neg as f64 / n_particles as f64);

    // Create GPU simulation
    println!("\nCreating GPU simulation...");
    let t1 = Instant::now();
    let mut sim = match JanusPMGpu::new(
        particles,
        grid_size, grid_size, grid_size,
        box_size,
        dt,
        eta,
        z_init,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to create simulation: {}", e);
            std::process::exit(1);
        }
    };
    println!("  GPU setup: {:.1} s", t1.elapsed().as_secs_f64());

    // Virialize
    println!("\nVirializing (α = 4.57 hardcoded)...");
    match sim.virialize() {
        Ok(alpha) => println!("  α = {:.4}", alpha),
        Err(e) => eprintln!("WARNING: Virialization failed: {}", e),
    }

    let ke_0 = sim.ke_initial;
    let seg_0 = sim.seg_initial;

    println!("\nInitial state:");
    println!("  KE₀ = {:.6e}", ke_0);
    println!("  Seg₀ = {:.6}", seg_0);
    println!("  Scale factor = {:.4}", sim.scale_factor());

    // Time series CSV
    let ts_path = output_dir.join("time_series.csv");
    let mut ts_file = std::fs::File::create(&ts_path).expect("Failed to create time series file");
    writeln!(ts_file, "step,time,tau,scale_factor,segregation,ke_ratio,n_positive,n_negative").ok();
    writeln!(ts_file, "0,0.0,{},{},{},1.0,{},{}", sim.tau, sim.scale_factor(), seg_0, n_pos, n_neg).ok();

    // Save initial snapshot
    save_checkpoint(&output_dir.join("snapshot_0.bin"), &sim, n_pos, n_neg);

    // Tracking
    let mut s_max = seg_0;
    let mut s_max_step = 0;
    let mut max_ke_ratio = 1.0_f64;

    println!("\n  Step      a        Seg        KE/KE₀     ms/step");
    println!("  ─────────────────────────────────────────────────────");

    let t_loop = Instant::now();
    let mut last_report = Instant::now();

    for step in 1..=n_steps {
        let t_step = Instant::now();

        if let Err(e) = sim.step() {
            eprintln!("ERROR at step {}: {}", step, e);
            save_checkpoint(&output_dir.join("snapshot_error.bin"), &sim, n_pos, n_neg);
            std::process::exit(1);
        }

        let ke = match sim.kinetic_energy() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("ERROR computing KE at step {}: {}", step, e);
                continue;
            }
        };

        let seg = match sim.segregation() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR computing segregation at step {}: {}", step, e);
                continue;
            }
        };

        let ke_ratio = if ke_0 > 1e-10 { ke / ke_0 } else { 1.0 };
        let a = sim.scale_factor();
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;

        max_ke_ratio = max_ke_ratio.max(ke_ratio);

        // Time series (every step)
        let time = step as f64 * dt as f64;
        writeln!(ts_file, "{},{},{},{},{},{},{},{}", step, time, sim.tau, a, seg, ke_ratio, n_pos, n_neg).ok();

        // Peak detection
        if seg > s_max {
            s_max = seg;
            s_max_step = step;
            save_checkpoint(&output_dir.join("checkpoint_peak.bin"), &sim, n_pos, n_neg);
        }

        // Full checkpoint every 500 steps (for resume)
        if step % 500 == 0 {
            let ckpt_path = output_dir.join(format!("checkpoint_{:04}.bin", step));
            save_checkpoint(&ckpt_path, &sim, n_pos, n_neg);
            println!("    [Checkpoint saved: {}]", ckpt_path.file_name().unwrap().to_string_lossy());
        }

        // Light snapshot every 200 steps (subsample 1M, for visualization)
        if step % 200 == 0 {
            let snap_path = output_dir.join(format!("snapshot_{:04}.bin", step));
            save_snapshot_light(&snap_path, &sim, 1_000_000);
        }

        // Report every 10 steps or every 30 seconds
        let should_report = step % 10 == 0 || step == 1 || step == n_steps
            || last_report.elapsed().as_secs() > 30;

        if should_report {
            let elapsed = t_loop.elapsed().as_secs_f64();
            let eta_min = if step > 0 {
                (elapsed / step as f64) * (n_steps - step) as f64 / 60.0
            } else { 0.0 };

            println!("  {:4}    {:.4}    {:.6}   {:.4}     {:.0}   (ETA {:.0}m)",
                     step, a, seg, ke_ratio, step_ms, eta_min);
            last_report = Instant::now();
        }

        // Early stop if KE explodes
        if ke_ratio > 50.0 {
            eprintln!("\n⚠ KE explosion detected at step {}: KE/KE₀ = {:.1}", step, ke_ratio);
            save_checkpoint(&output_dir.join("snapshot_ke_explosion.bin"), &sim, n_pos, n_neg);
            break;
        }
    }

    // Final snapshot
    let seg_final = sim.segregation().unwrap_or(0.0);
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    save_checkpoint(&output_dir.join("snapshot_final.bin"), &sim, n_pos, n_neg);

    let total_time = t_loop.elapsed().as_secs_f64();

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      PM-5 RESULTS                                ");
    println!("══════════════════════════════════════════════════════════════════");

    println!("\n  Total time: {:.1} min ({:.0} ms/step)", total_time / 60.0, total_time * 1000.0 / n_steps as f64);
    println!("  S(0) = {:.6}", seg_0);
    println!("  S({}) = {:.6}", n_steps, seg_final);
    println!("  S_max = {:.6} at step {}", s_max, s_max_step);
    println!("  KE/KE₀ max = {:.2}", max_ke_ratio);
    println!("  Final scale factor = {:.4}", sim.scale_factor());

    // Validation
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("                      VALIDATION                                  ");
    println!("══════════════════════════════════════════════════════════════════");

    let s_pass = seg_final > 0.01 || s_max > 0.05;
    let ke_pass = max_ke_ratio < 20.0;

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Test                    │ Result    │ Threshold │ Status       │");
    println!("├─────────────────────────┼───────────┼───────────┼──────────────┤");
    println!("│ Segregation             │ {:.6}  │ > 0.01    │ {}           │",
             seg_final.max(s_max), if s_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("│ KE/KE₀ (max)            │ {:.2}      │ < 20      │ {}           │",
             max_ke_ratio, if ke_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n  Output: {}", output_dir.display());

    if s_pass && ke_pass {
        println!("\n✓ PM-5 VALIDATION: PASSED");
    } else {
        println!("\n✗ PM-5 VALIDATION: FAILED");
    }
    println!("══════════════════════════════════════════════════════════════════");
}

/// Save full checkpoint (positions + velocities, for resume)
fn save_checkpoint(path: &Path, sim: &JanusPMGpu, n_pos: usize, n_neg: usize) {
    use std::io::BufWriter;
    use byteorder::{LittleEndian, WriteBytesExt};

    let (pos_x, pos_y, pos_z, signs) = match sim.download_all_positions() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("WARNING: Failed to download positions: {}", e);
            return;
        }
    };

    let (vel_x, vel_y, vel_z) = match sim.download_all_velocities() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("WARNING: Failed to download velocities: {}", e);
            return;
        }
    };

    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: Failed to create checkpoint file: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    // Header (extended with ke_initial)
    writer.write_u64::<LittleEndian>(sim.n_particles as u64).ok();
    writer.write_u64::<LittleEndian>(n_pos as u64).ok();
    writer.write_u64::<LittleEndian>(n_neg as u64).ok();
    writer.write_u64::<LittleEndian>(sim.step as u64).ok();
    writer.write_f64::<LittleEndian>(sim.tau).ok();
    writer.write_f64::<LittleEndian>(sim.scale_factor()).ok();
    writer.write_f64::<LittleEndian>(sim.segregation().unwrap_or(0.0)).ok();
    writer.write_f64::<LittleEndian>(sim.kinetic_energy().unwrap_or(0.0) / sim.ke_initial).ok();
    writer.write_f64::<LittleEndian>(sim.ke_initial).ok();  // NEW: save ke_initial for resume

    // Positions, velocities and signs (interleaved)
    for i in 0..sim.n_particles {
        writer.write_f64::<LittleEndian>(pos_x[i]).ok();
        writer.write_f64::<LittleEndian>(pos_y[i]).ok();
        writer.write_f64::<LittleEndian>(pos_z[i]).ok();
        writer.write_f32::<LittleEndian>(vel_x[i]).ok();
        writer.write_f32::<LittleEndian>(vel_y[i]).ok();
        writer.write_f32::<LittleEndian>(vel_z[i]).ok();
        writer.write_i8(signs[i]).ok();
    }
}

/// Save light snapshot (subsampled, f32 positions)
fn save_snapshot_light(path: &Path, sim: &JanusPMGpu, max_particles: usize) {
    use std::io::BufWriter;
    use byteorder::{LittleEndian, WriteBytesExt};

    let subsample = (sim.n_particles / max_particles).max(1);

    let positions = match sim.download_positions_subsampled(subsample) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("WARNING: Failed to download positions: {}", e);
            return;
        }
    };

    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: Failed to create snapshot file: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    // Header (simplified)
    writer.write_u64::<LittleEndian>(positions.len() as u64).ok();
    writer.write_u64::<LittleEndian>(sim.step as u64).ok();
    writer.write_f64::<LittleEndian>(sim.scale_factor()).ok();
    writer.write_f64::<LittleEndian>(sim.segregation().unwrap_or(0.0)).ok();

    // Positions (f32) and signs
    for (x, y, z, sign) in &positions {
        writer.write_f32::<LittleEndian>(*x).ok();
        writer.write_f32::<LittleEndian>(*y).ok();
        writer.write_f32::<LittleEndian>(*z).ok();
        writer.write_i8(*sign).ok();
    }
}
