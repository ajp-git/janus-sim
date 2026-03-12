//! Production BH 12M - Pure Barnes-Hut with snapshots
//!
//! Paramètres from PROMPT_ZELDOVICH_THEN_BH12M.md:
//! - N = 12M, Box = 492 Mpc
//! - ICs = new() positifs d'abord puis négatifs
//! - virialize_sampled(80000)
//! - Steps = 20000, Snapshots every 20 steps
//! - dtau_per_dt = tau_range / (20000 × 0.01)

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// RUN 2 parameters (from PROMPT)
const N_PARTICLES: usize = 12_000_000;
const BOX_SIZE: f64 = 492.0;          // Mpc (n_side=229, spacing=2.15 Mpc)
const Z_INIT: f64 = 5.0;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 20000;
const SOFTENING: f64 = 0.65;          // Mpc
const ETA: f64 = 1.045;

const SNAPSHOT_INTERVAL: usize = 20;  // 1000 snapshots total
const N_SAMPLE_VIRIALIZE: usize = 80000;  // 0.5% of N/2

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  RUN 2: Production BH 12M");
    println!("═══════════════════════════════════════════════════════════\n");

    let output_dir = "/app/output/production_bh_12m";
    let snapshots_dir = format!("{}/snapshots", output_dir);
    fs::create_dir_all(&snapshots_dir).expect("Failed to create snapshots dir");

    // Save PID for monitoring
    let pid = std::process::id();
    fs::write(format!("{}/pid.txt", output_dir), pid.to_string()).ok();

    println!("Output: {}", output_dir);
    println!("PID: {}", pid);
    println!("\nParameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  Spacing = {:.2} Mpc (FIX-015)", BOX_SIZE / (N_PARTICLES as f64).cbrt());
    println!("  Steps = {}", TOTAL_STEPS);
    println!("  Snapshots = {} (every {} steps)", TOTAL_STEPS / SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL);
    println!("  θ = {}, softening = {} Mpc", THETA, SOFTENING);
    println!("  dt = {}", DT);
    println!("  virialize_sampled({})", N_SAMPLE_VIRIALIZE);
    println!();

    // Initialize simulation using new() - positives first convention
    println!("Initializing GPU simulation...");
    let init_start = Instant::now();

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;

    let mut sim = GpuNBodySimulation::new(n_positive, n_negative, BOX_SIZE)
        .expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);

    println!("  Init time: {:.1}s", init_start.elapsed().as_secs_f64());
    println!("  N = {} ({} + / {} -)", n_positive + n_negative, n_positive, n_negative);

    // Virialization with sampled PE (FIX-007)
    println!("\nVirializing (sampled, n={})...", N_SAMPLE_VIRIALIZE);
    let virial_start = Instant::now();
    sim.virialize_sampled(N_SAMPLE_VIRIALIZE).expect("virialize_sampled failed");
    println!("  Virialization time: {:.1}s", virial_start.elapsed().as_secs_f64());

    // Setup cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    // FIX-016: dtau_per_dt = tau_range / (TOTAL_STEPS × DT)
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (TOTAL_STEPS as f64 * DT);

    println!("\nCosmology:");
    println!("  η = {}", ETA);
    println!("  τ_start = {:.4}, τ_end = {:.4}", cosmo.tau_start, cosmo.tau_end);
    println!("  dtau_per_dt = {:.6} (FIX-016)", dtau_per_dt);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke,ke_ratio,seg,step_ms").unwrap();

    // Initial state
    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = sim.segregation_distance().expect("segregation failed");

    writeln!(csv, "0,{:.4},{:.6e},{:.6},{:.6},0", Z_INIT, ke_0, 1.0, seg_0).unwrap();

    // Save initial snapshot
    save_snapshot(&sim, 0, &snapshots_dir, n_positive, N_PARTICLES);

    println!("\nInitial state:");
    println!("  KE₀ = {:.4e}", ke_0);
    println!("  Seg₀ = {:.4}", seg_0);

    // Verify step 5 stability
    println!("\n══════════════════════════════════════════════════");
    println!("  Starting simulation ({} steps, ~29h estimated)", TOTAL_STEPS);
    println!("══════════════════════════════════════════════════\n");

    let mut tau = cosmo.tau_start;
    let start = Instant::now();
    let mut seg_max = seg_0;
    let mut z_at_seg_max = Z_INIT;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        tau += dtau_per_step;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        let step_ms = step_start.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy().expect("kinetic_energy failed");
        let ke_ratio = ke / ke_0;
        let seg = sim.segregation_distance().expect("segregation failed");

        if seg > seg_max {
            seg_max = seg;
            z_at_seg_max = z;
        }

        writeln!(csv, "{},{:.4},{:.6e},{:.6},{:.6},{:.0}", step, z, ke, ke_ratio, seg, step_ms).unwrap();

        // Save snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &snapshots_dir, n_positive, N_PARTICLES);
        }

        // Progress logging
        if step == 5 {
            println!(">>> STEP 5 CHECK <<<");
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            if ke_ratio > 1.05 {
                println!("  ⚠️ WARNING: KE/KE₀ > 1.05 at step 5");
            } else {
                println!("  ✓ PASS: KE/KE₀ < 1.05");
            }
        }

        if step <= 10 || step % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            let eta_h = (TOTAL_STEPS - step) as f64 / rate / 3600.0;
            println!("Step {:5}: z={:.2}, KE/KE₀={:.4}, Seg={:.4} ({:.0} ms, ETA {:.1}h)",
                     step, z, ke_ratio, seg, step_ms, eta_h);
        }

        // Milestone logging
        if step == 100 || step == 2800 || step == 4200 || step == 10000 || step == 20000 {
            println!("\n>>> MILESTONE step {} <<<", step);
            println!("  z = {:.2}", z);
            println!("  KE/KE₀ = {:.4}", ke_ratio);
            println!("  Seg = {:.4} (max = {:.4} @ z={:.2})", seg, seg_max, z_at_seg_max);
            csv.flush().unwrap();
        }

        // Auto-stop on explosion
        if ke_ratio > 50.0 && step > 100 {
            println!("\n❌ AUTO-STOP: KE/KE₀ = {:.1} > 50 at step {}", ke_ratio, step);
            break;
        }
    }

    csv.flush().unwrap();

    let elapsed = start.elapsed();
    let final_ke = sim.kinetic_energy().unwrap();
    let final_seg = sim.segregation_distance().unwrap();

    println!("\n══════════════════════════════════════════════════");
    println!("  RUN 2 COMPLETE");
    println!("══════════════════════════════════════════════════");
    println!("  Runtime: {:.1}h ({:.1}s/step)", elapsed.as_secs_f64() / 3600.0, elapsed.as_secs_f64() / TOTAL_STEPS as f64);
    println!("  KE/KE₀ final: {:.4}", final_ke / ke_0);
    println!("  Seg₀ = {:.4}", seg_0);
    println!("  Seg_max = {:.4} @ z={:.2}", seg_max, z_at_seg_max);
    println!("  Seg final = {:.4}", final_seg);
    println!("  Snapshots saved: {}", TOTAL_STEPS / SNAPSHOT_INTERVAL);

    // Save summary JSON
    let summary = format!(r#"{{
  "model": "Production BH 12M",
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {},
  "box_size": {},
  "steps_completed": {},
  "ke_0": {:.6e},
  "ke_final": {:.6e},
  "ke_ratio_final": {:.6},
  "seg_0": {:.6},
  "seg_max": {:.6},
  "z_at_seg_max": {:.4},
  "seg_final": {:.6},
  "runtime_hours": {:.2}
}}"#,
        N_PARTICLES, n_positive, n_negative, ETA, BOX_SIZE, TOTAL_STEPS,
        ke_0, final_ke, final_ke / ke_0,
        seg_0, seg_max, z_at_seg_max, final_seg,
        elapsed.as_secs_f64() / 3600.0
    );
    fs::write(format!("{}/summary.json", output_dir), &summary).unwrap();

    println!("\nOutput: {}", output_dir);
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodySimulation, step: usize, dir: &str, n_positive: usize, n_total: usize) {
    let filename = format!("{}/snap_{:06}.bin", dir, step);
    let positions = sim.get_positions().expect("get_positions failed");

    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Header: [n_particles: u64, step: u64, reserved: u64] = 24 bytes
    writer.write_all(&(n_total as u64).to_le_bytes()).unwrap();
    writer.write_all(&(step as u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();

    // Positions as f32 + sign as f32 (compact format from PROMPT)
    // sign = +1.0 for positive, -1.0 for negative
    for i in 0..n_total {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let sign: f32 = if i < n_positive { 1.0 } else { -1.0 };

        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires cuda feature");
}
