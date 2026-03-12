//! Resume V10 simulation from snapshot with θ=0.8
//! Usage: cargo run --release --features "cuda cufft" --bin resume_v10

use std::fs::{self, File};
use std::io::{Read, Write, BufWriter, BufReader};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const L_BOX: f64 = 200.0;
const EPSILON: f64 = 0.18;
const H_HUBBLE: f64 = 0.012;
const R_CUT: f64 = 30.0;
const DTAU_PER_DT: f64 = 0.0;
const K_MIN: usize = 2;

// RESUME CONFIG - θ=0.8 for faster BH
const THETA: f64 = 0.8;
const DT: f64 = 0.01;
const START_STEP: usize = 2010;  // After 10 warm-up steps
const TOTAL_STEPS: usize = 3000;  // Run to step 3000 first
const SNAPSHOT_INTERVAL: usize = 500;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let snapshot_path = "/app/output/janus_v10_highres/snapshots/snap_002000.bin";
    let output_dir = "/app/output/janus_v10_resume";

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  JANUS V10 RESUME — θ=0.8                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Loading: {}", snapshot_path);
    println!("θ = {} (faster BH)", THETA);
    println!("dt = {}", DT);
    println!("Resume from step {} → {}", START_STEP, TOTAL_STEPS);
    println!();

    // Create output directory
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    // Load snapshot
    let (positions, signs, n_particles) = load_v10_snapshot(snapshot_path)
        .expect("Failed to load snapshot");

    println!("Loaded {} particles", n_particles);
    let n_pos = signs.iter().filter(|&&s| s > 0).count();
    let n_neg = n_particles - n_pos;
    println!("  N+ = {}, N- = {}", n_pos, n_neg);

    // Initialize velocities to zero (will generate thermal velocities)
    let velocities = vec![0.0f32; n_particles * 3];

    // Create GPU simulation
    println!("\nInitializing GPU simulation...");
    let t0 = Instant::now();

    let mut sim = GpuNBodyTwoPass::with_custom_ics(positions, velocities, signs, L_BOX)
        .expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(K_MIN);

    println!("  GPU initialized in {:.1}s", t0.elapsed().as_secs_f64());

    // Quick virialize to give particles some velocity
    println!("\nRe-virialization...");
    if let Err(e) = sim.virialize_pm() {
        println!("  Warning: virialize failed: {}", e);
    }

    // Run a few steps to build up KE before measuring KE0
    println!("\nWarm-up (10 steps to build KE)...");
    for _ in 0..10 {
        let _ = sim.step_treepm_gpu(DT, R_CUT, H_HUBBLE, DTAU_PER_DT);
    }

    let ke0 = sim.kinetic_energy().unwrap_or(1e10).max(1e10);
    let seg0 = sim.segregation().unwrap_or(0.0);

    println!("  KE₀ = {:.4e} (after warm-up)", ke0);
    println!("  Seg₀ = {:.4}", seg0);

    // Time series file
    let mut ts_file = File::create(format!("{}/time_series.csv", output_dir)).unwrap();
    writeln!(ts_file, "step,KE,KE_ratio,segregation,theta,dt").unwrap();

    println!("\n--- Running steps {} → {} ---\n", START_STEP + 1, TOTAL_STEPS);
    let run_start = Instant::now();

    for step in (START_STEP + 1)..=TOTAL_STEPS {
        let t0 = Instant::now();

        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, H_HUBBLE, DTAU_PER_DT) {
            println!("  ERROR at step {}: {}", step, e);
            break;
        }

        let step_ms = t0.elapsed().as_millis();

        // Log every 100 steps
        if step % 100 == 0 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let ke_ratio = ke / ke0;
            let seg = sim.segregation().unwrap_or(0.0);

            println!("  TreePM GPU step {}: PM + BH = {}ms", step, step_ms);
            println!("  Step {}: KE={:.2e}, Seg={:.4}, θ={:.2}, dt={:.4} ({} ms)",
                     step, ke, seg, THETA, DT, step_ms);

            writeln!(ts_file, "{},{:.6e},{:.6},{:.1},{:.3}",
                     step, ke, seg, THETA, DT).unwrap();

            // No auto-stop on KE ratio (unreliable when starting from zero velocities)
        }

        // Save snapshots
        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
            println!("  → Saving snapshot: {}", snap_path);
            save_snapshot(&sim, &snap_path);
        }
    }

    let total_time = run_start.elapsed().as_secs_f64();
    let steps_done = TOTAL_STEPS - START_STEP;

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Resume complete!");
    println!("  Steps: {} → {}", START_STEP, TOTAL_STEPS);
    println!("  Total time: {:.1} hours", total_time / 3600.0);
    println!("  Average: {:.1} sec/step", total_time / steps_done as f64);
    println!("═══════════════════════════════════════════════════════════════");

    // Final state
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let seg_final = sim.segregation().unwrap_or(0.0);

    println!();
    println!("Final: KE={:.4e}, Seg={:.4}", ke_final, seg_final);

    // Save summary
    let summary = format!(r#"{{
  "resumed_from": "{}",
  "start_step": {},
  "end_step": {},
  "theta": {},
  "dt": {},
  "n_particles": {},
  "total_time_hours": {:.2},
  "avg_step_sec": {:.2},
  "KE_final": {:.6e},
  "segregation_final": {:.4}
}}"#, snapshot_path, START_STEP, TOTAL_STEPS, THETA, DT, n_particles,
        total_time / 3600.0, total_time / steps_done as f64, ke_final, seg_final);

    fs::write(format!("{}/summary.json", output_dir), summary).unwrap();
    println!("\nResults saved to {}/", output_dir);
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn load_v10_snapshot(path: &str) -> Result<(Vec<f32>, Vec<i8>, usize), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);

    // Skip 8-byte header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).map_err(|e| e.to_string())?;

    // Read particles: x(f32), y(f32), z(f32), sign(f32) = 16 bytes each
    let mut positions = Vec::new();
    let mut signs = Vec::new();

    let mut buf = [0u8; 16];
    while reader.read_exact(&mut buf).is_ok() {
        let x = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(buf[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(buf[8..12].try_into().unwrap());
        let sign_f32 = f32::from_le_bytes(buf[12..16].try_into().unwrap());

        positions.push(x);
        positions.push(y);
        positions.push(z);
        signs.push(if sign_f32 > 0.0 { 1i8 } else { -1i8 });
    }

    let n = signs.len();
    Ok((positions, signs, n))
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &str) {
    let (positions, _, signs) = sim.get_particles().expect("get_particles failed");

    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    // 8-byte header (zeros)
    writer.write_all(&[0u8; 8]).unwrap();

    // Write particles: x, y, z, sign as f32
    let n = positions.len() / 3;
    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        let sign = if signs[i] > 0 { 1.0f32 } else { -1.0f32 };

        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    println!("CUDA + cuFFT required. Use: cargo run --release --features \"cuda cufft\" --bin resume_v10");
}
