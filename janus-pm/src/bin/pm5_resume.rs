//! PM-5 Resume: Continue simulation from checkpoint
//!
//! Usage: pm5_resume <checkpoint.bin> <additional_steps> [output_dir]

use std::path::Path;
use std::io::{BufReader, BufWriter};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use chrono::Local;

use janus_pm::gpu_simulation::JanusPMGpu;

// Same config as original run
const GRID_SIZE: usize = 256;
const BOX_SIZE: f64 = 500.0;
const DT: f64 = 0.005;
const ETA: f64 = 1.045;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: pm5_resume <checkpoint.bin> <additional_steps> [output_dir]");
        std::process::exit(1);
    }

    let checkpoint_path = Path::new(&args[1]);
    let additional_steps: usize = args[2].parse().expect("Invalid steps");

    let output_dir = if args.len() > 3 {
        std::path::PathBuf::from(&args[3])
    } else {
        std::path::PathBuf::from(format!(
            "janus-pm/output/pm5_resume_{}",
            Local::now().format("%Y-%m-%d_%H%M%S")
        ))
    };
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PM-5 Resume: Continue from Checkpoint                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Load checkpoint
    println!("Loading checkpoint: {}", checkpoint_path.display());
    let (mut sim, n_pos, n_neg, start_step) = load_checkpoint(checkpoint_path)
        .expect("Failed to load checkpoint");

    let n_particles = sim.n_particles;
    let total_steps = start_step + additional_steps;

    println!("  Particles: {} ({:.1}M)", n_particles, n_particles as f64 / 1e6);
    println!("  Start step: {}", start_step);
    println!("  Additional steps: {}", additional_steps);
    println!("  Total steps: {}", total_steps);
    println!("  Output: {}", output_dir.display());
    println!();

    // Initial state
    let ke0 = sim.ke_initial;
    let seg0 = sim.segregation().unwrap_or(0.0);
    let a0 = sim.scale_factor();

    println!("Resume state:");
    println!("  KE₀ = {:.6e}", ke0);
    println!("  Seg = {:.6}", seg0);
    println!("  Scale factor = {:.4}", a0);
    println!();

    // Create time series file
    let ts_path = output_dir.join("time_series.csv");
    let ts_file = std::fs::File::create(&ts_path).expect("Failed to create time_series.csv");
    let mut ts_writer = BufWriter::new(ts_file);
    writeln!(ts_writer, "step,time,tau,scale_factor,segregation,ke_ratio,n_positive,n_negative").ok();

    // Tracking
    let mut s_max = seg0;
    let mut s_max_step = start_step;

    println!("  Step      a        Seg        KE/KE₀     ms/step");
    println!("  ─────────────────────────────────────────────────────");

    let run_start = std::time::Instant::now();

    for step in (start_step + 1)..=total_steps {
        let step_start = std::time::Instant::now();

        if let Err(e) = sim.step() {
            eprintln!("ERROR at step {}: {}", step, e);
            save_checkpoint(&output_dir.join("checkpoint_error.bin"), &sim, n_pos, n_neg);
            break;
        }

        let step_ms = step_start.elapsed().as_millis();

        // Compute metrics
        let seg = sim.segregation().unwrap_or(0.0);
        let ke = sim.kinetic_energy().unwrap_or(0.0);
        let ke_ratio = ke / ke0;
        let a = sim.scale_factor();
        let t = sim.time();

        // Track max segregation
        if seg > s_max {
            s_max = seg;
            s_max_step = step;
            save_checkpoint(&output_dir.join("checkpoint_peak.bin"), &sim, n_pos, n_neg);
        }

        // Write time series
        writeln!(ts_writer, "{},{},{},{},{},{},{},{}",
            step, t, sim.tau, a, seg, ke_ratio, n_pos, n_neg).ok();

        // Report every 10 steps, or at milestones
        let local_step = step - start_step;
        if local_step % 10 == 0 || local_step % 500 == 0 {
            let remaining = total_steps - step;
            let eta_min = (remaining as f64 * step_ms as f64 / 60000.0) as u32;
            println!("  {:5}    {:.4}    {:.6}   {:.4}     {}   (ETA {}m)",
                step, a, seg, ke_ratio, step_ms, eta_min);
        }

        // Report S(t) every 500 steps
        if local_step % 500 == 0 {
            println!();
            println!("  >>> S({}) = {:.6}  |  S_max = {:.6} at step {}",
                step, seg, s_max, s_max_step);
            println!();
        }

        // Light snapshot every 100 steps
        if local_step % 100 == 0 {
            let snap_path = output_dir.join(format!("snapshot_{:04}.bin", step));
            save_snapshot_light(&snap_path, &sim, 1_000_000);
        }

        // KE explosion check
        if ke_ratio > 20.0 {
            eprintln!("WARNING: KE explosion at step {} (KE/KE₀ = {:.2})", step, ke_ratio);
            save_checkpoint(&output_dir.join("checkpoint_ke_explosion.bin"), &sim, n_pos, n_neg);
            break;
        }
    }

    // Final checkpoint
    let runtime = run_start.elapsed().as_secs_f64();
    println!();
    println!("Run completed in {:.1}s ({:.1} min)", runtime, runtime / 60.0);

    save_checkpoint(&output_dir.join("checkpoint_final.bin"), &sim, n_pos, n_neg);

    // Summary
    let final_seg = sim.segregation().unwrap_or(0.0);
    let final_ke = sim.kinetic_energy().unwrap_or(ke0) / ke0;
    let final_a = sim.scale_factor();
    let final_z = 1.0 / final_a - 1.0;

    println!();
    println!("Final state:");
    println!("  Steps: {} → {}", start_step, sim.step);
    println!("  a = {:.4} (z = {:.2})", final_a, final_z);
    println!("  S(final) = {:.6}", final_seg);
    println!("  S_max = {:.6} at step {}", s_max, s_max_step);
    println!("  KE/KE₀ = {:.4}", final_ke);

    // Save summary
    let summary = serde_json::json!({
        "resumed_from": checkpoint_path.display().to_string(),
        "start_step": start_step,
        "final_step": sim.step,
        "additional_steps": additional_steps,
        "final_scale_factor": final_a,
        "final_redshift": final_z,
        "final_segregation": final_seg,
        "max_segregation": s_max,
        "max_seg_step": s_max_step,
        "final_ke_ratio": final_ke,
        "runtime_seconds": runtime
    });
    std::fs::write(
        output_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap()
    ).ok();
}

/// Load checkpoint with velocities
fn load_checkpoint(path: &Path) -> Result<(JanusPMGpu, usize, usize, usize), String> {
    use std::fs::File;

    let file = File::open(path).map_err(|e| format!("Failed to open: {}", e))?;
    let file_size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut reader = BufReader::new(file);

    // Read header
    let n_particles = reader.read_u64::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let n_pos = reader.read_u64::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let n_neg = reader.read_u64::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let step = reader.read_u64::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let tau = reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
    let _scale_factor = reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
    let _segregation = reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
    let _ke_ratio = reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
    let ke_initial = reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;

    println!("  Header: n={}, step={}, tau={:.4}, ke0={:.4e}", n_particles, step, tau, ke_initial);

    // Check if this is a checkpoint with velocities
    // Old format: 8*8 + n*(3*8 + 1) = 64 + 25n bytes
    // New format: 9*8 + n*(3*8 + 3*4 + 1) = 72 + 37n bytes
    let expected_old = 64 + n_particles * 25;
    let expected_new = 72 + n_particles * 37;

    let has_velocities = file_size as usize >= expected_new;

    if !has_velocities {
        return Err(format!(
            "Checkpoint missing velocities. File size {} < expected {} for {} particles with velocities.",
            file_size, expected_new, n_particles
        ));
    }

    println!("  Reading {} particles with velocities...", n_particles);

    // Read particle data
    let mut pos_x = Vec::with_capacity(n_particles);
    let mut pos_y = Vec::with_capacity(n_particles);
    let mut pos_z = Vec::with_capacity(n_particles);
    let mut vel_x = Vec::with_capacity(n_particles);
    let mut vel_y = Vec::with_capacity(n_particles);
    let mut vel_z = Vec::with_capacity(n_particles);
    let mut signs = Vec::with_capacity(n_particles);

    for _ in 0..n_particles {
        pos_x.push(reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?);
        pos_y.push(reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?);
        pos_z.push(reader.read_f64::<LittleEndian>().map_err(|e| e.to_string())?);
        vel_x.push(reader.read_f32::<LittleEndian>().map_err(|e| e.to_string())?);
        vel_y.push(reader.read_f32::<LittleEndian>().map_err(|e| e.to_string())?);
        vel_z.push(reader.read_f32::<LittleEndian>().map_err(|e| e.to_string())?);
        signs.push(reader.read_i8().map_err(|e| e.to_string())?);
    }

    println!("  Creating GPU simulation...");

    let sim = JanusPMGpu::new_from_checkpoint(
        pos_x, pos_y, pos_z,
        vel_x, vel_y, vel_z,
        signs,
        GRID_SIZE,
        BOX_SIZE,
        DT,
        ETA,
        tau,
        step,
        ke_initial,
    )?;

    Ok((sim, n_pos, n_neg, step))
}

/// Save checkpoint with velocities
fn save_checkpoint(path: &Path, sim: &JanusPMGpu, n_pos: usize, n_neg: usize) {
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
            eprintln!("WARNING: Failed to create checkpoint: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    // Header
    writer.write_u64::<LittleEndian>(sim.n_particles as u64).ok();
    writer.write_u64::<LittleEndian>(n_pos as u64).ok();
    writer.write_u64::<LittleEndian>(n_neg as u64).ok();
    writer.write_u64::<LittleEndian>(sim.step as u64).ok();
    writer.write_f64::<LittleEndian>(sim.tau).ok();
    writer.write_f64::<LittleEndian>(sim.scale_factor()).ok();
    writer.write_f64::<LittleEndian>(sim.segregation().unwrap_or(0.0)).ok();
    writer.write_f64::<LittleEndian>(sim.kinetic_energy().unwrap_or(0.0) / sim.ke_initial).ok();
    writer.write_f64::<LittleEndian>(sim.ke_initial).ok();

    // Data
    for i in 0..sim.n_particles {
        writer.write_f64::<LittleEndian>(pos_x[i]).ok();
        writer.write_f64::<LittleEndian>(pos_y[i]).ok();
        writer.write_f64::<LittleEndian>(pos_z[i]).ok();
        writer.write_f32::<LittleEndian>(vel_x[i]).ok();
        writer.write_f32::<LittleEndian>(vel_y[i]).ok();
        writer.write_f32::<LittleEndian>(vel_z[i]).ok();
        writer.write_i8(signs[i]).ok();
    }

    println!("  Saved checkpoint: {}", path.display());
}

/// Save light snapshot (subsampled, f32 positions)
fn save_snapshot_light(path: &Path, sim: &JanusPMGpu, max_particles: usize) {
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
            eprintln!("WARNING: Failed to create snapshot: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    // Header
    writer.write_u64::<LittleEndian>(positions.len() as u64).ok();
    writer.write_u64::<LittleEndian>(sim.step as u64).ok();
    writer.write_f64::<LittleEndian>(sim.scale_factor()).ok();
    writer.write_f64::<LittleEndian>(sim.segregation().unwrap_or(0.0)).ok();

    // Positions and signs
    for (x, y, z, sign) in positions {
        writer.write_f32::<LittleEndian>(x).ok();
        writer.write_f32::<LittleEndian>(y).ok();
        writer.write_f32::<LittleEndian>(z).ok();
        writer.write_i8(sign).ok();
    }
}
