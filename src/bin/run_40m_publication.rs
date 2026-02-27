//! 40M Publication Run — Morton+WarpCoherent + Hubble Friction
//! z=5 → z=1.5, 6000 steps, snapshots every 100 steps

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

const N: usize = 40_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const STEPS: usize = 6000;
const SNAPSHOT_INTERVAL: usize = 100;
const BOX_SIZE: f64 = 736.8;

// Cosmology: z=5 to z=1.5
const Z_INIT: f64 = 5.0;
const Z_FINAL: f64 = 1.5;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = "/app/output/40M_v3_2026-02-27";
    let snapshot_dir = format!("{}/snapshots", output_dir);

    fs::create_dir_all(&snapshot_dir)?;

    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║   40M PUBLICATION RUN — Morton + WarpCoherent + HUBBLE         ║");
    eprintln!("║   z={} → z={}, {} steps, snapshots every {}              ║", Z_INIT, Z_FINAL, STEPS, SNAPSHOT_INTERVAL);
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Output: {}", output_dir);
    eprintln!();

    // Initialize simulation
    let n_positive = (N as f64 / (1.0 + ETA)) as usize;
    let n_negative = N - n_positive;

    eprintln!("Parameters:");
    eprintln!("  N = {} ({:.0}M)", N, N as f64 / 1e6);
    eprintln!("  N+ = {}, N- = {}", n_positive, n_negative);
    eprintln!("  η = {}", ETA);
    eprintln!("  θ = {}", THETA);
    eprintln!("  dt = {}", DT);
    eprintln!("  box = {:.1} Mpc", BOX_SIZE);
    eprintln!();

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);

    // Open CSV for time series
    let csv_path = format!("{}/time_series.csv", output_dir);
    let csv_file = File::create(&csv_path)?;
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "step,time,z,a,H,seg,ke_ratio,step_time_ms")?;

    // Cosmology with proper Hubble friction from Friedmann integration
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    // Find tau values for z=5 and z=1.5 by searching history
    let a_init = 1.0 / (1.0 + Z_INIT);
    let a_final = 1.0 / (1.0 + Z_FINAL);

    // Find tau for z_init (smallest a)
    let tau_z_init = cosmo.history.iter()
        .min_by(|s1, s2| {
            let d1 = (s1.a - a_init).abs();
            let d2 = (s2.a - a_init).abs();
            d1.partial_cmp(&d2).unwrap()
        })
        .map(|s| s.tau)
        .unwrap_or(cosmo.tau_start);

    // Find tau for z_final (larger a)
    let tau_z_final = cosmo.history.iter()
        .min_by(|s1, s2| {
            let d1 = (s1.a - a_final).abs();
            let d2 = (s2.a - a_final).abs();
            d1.partial_cmp(&d2).unwrap()
        })
        .map(|s| s.tau)
        .unwrap_or(cosmo.tau_end);

    let dtau_per_step = (tau_z_final - tau_z_init) / STEPS as f64;
    let mut tau_current = tau_z_init;

    // Get initial H for display
    let (_, h_init) = cosmo.get_params_at_tau(tau_z_init);
    let (_, h_final) = cosmo.get_params_at_tau(tau_z_final);

    eprintln!("Cosmology (Janus Friedmann with Hubble friction):");
    eprintln!("  a_init = {:.4} (z={}), H_init = {:.4}", a_init, Z_INIT, h_init);
    eprintln!("  a_final = {:.4} (z={}), H_final = {:.4}", a_final, Z_FINAL, h_final);
    eprintln!("  tau_init = {:.6}, tau_final = {:.6}", tau_z_init, tau_z_final);
    eprintln!("  dtau/step = {:.8}", dtau_per_step);
    eprintln!();

    // Get initial segregation
    let pos_cpu = sim.get_positions()?;
    let signs_cpu = sim.get_signs()?;
    let seg_0 = compute_segregation(&pos_cpu, &signs_cpu, n_positive, n_negative);
    let ke_0 = 1.0; // Normalized

    eprintln!("Initial state:");
    eprintln!("  Seg_0 = {:.6}", seg_0);
    eprintln!();

    // Save initial snapshot
    save_snapshot(&snapshot_dir, 0, &pos_cpu, &signs_cpu)?;
    eprintln!("Saved snapshot_00000.bin");

    let run_start = Instant::now();
    let mut last_seg = seg_0;
    let mut max_seg = seg_0;
    let mut max_seg_step = 0;

    eprintln!("Starting {} steps...", STEPS);
    eprintln!();

    for step in 1..=STEPS {
        let step_start = Instant::now();

        // Get cosmological parameters from Friedmann integration
        let (a, hubble) = cosmo.get_params_at_tau(tau_current);
        let z = 1.0 / a - 1.0;

        // Hubble friction: dtau_per_dt = dtau/dt where dt is simulation time
        // The kick uses: friction = -H * v * dtau_per_dt * dt = -H * v * dtau
        let dtau_per_dt = dtau_per_step / DT;

        // Step with Morton+WarpCoherent + Hubble friction
        sim.step_dkd_morton_warpcoherent(DT, hubble, dtau_per_dt)?;

        // Advance conformal time
        tau_current += dtau_per_step;

        let step_time_ms = step_start.elapsed().as_millis();

        // Compute diagnostics every step
        let pos_cpu = sim.get_positions()?;
        let signs_cpu = sim.get_signs()?;
        let seg = compute_segregation(&pos_cpu, &signs_cpu, n_positive, n_negative);
        let ke_ratio = 1.0; // TODO: compute actual KE ratio

        if seg > max_seg {
            max_seg = seg;
            max_seg_step = step;
        }

        // Write CSV (now includes H)
        writeln!(csv, "{},{:.6},{:.4},{:.6},{:.6},{:.6},{:.4},{}",
            step, tau_current, z, a, hubble, seg, ke_ratio, step_time_ms)?;

        // Flush CSV every step
        csv.flush()?;

        // Save snapshot every SNAPSHOT_INTERVAL steps
        if step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&snapshot_dir, step, &pos_cpu, &signs_cpu)?;

            let elapsed = run_start.elapsed().as_secs_f64() / 3600.0;
            let eta = elapsed / step as f64 * (STEPS - step) as f64;

            eprintln!("[{:5}/{:5}] z={:.3} H={:.4} Seg={:.4} (max={:.4}@{}) | {:.1}s/step | {:.1}h elapsed, {:.1}h ETA",
                step, STEPS, z, hubble, seg, max_seg, max_seg_step,
                step_time_ms as f64 / 1000.0, elapsed, eta);
        }

        last_seg = seg;
    }

    let total_time = run_start.elapsed().as_secs_f64() / 3600.0;

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("                      RUN COMPLETE                              ");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("  Total time: {:.2} hours", total_time);
    eprintln!("  Final segregation: {:.4}", last_seg);
    eprintln!("  Max segregation: {:.4} at step {}", max_seg, max_seg_step);
    eprintln!("  Snapshots saved: {}", STEPS / SNAPSHOT_INTERVAL + 1);
    eprintln!();

    Ok(())
}

fn compute_segregation(pos: &[f32], signs: &[i8], n_pos: usize, n_neg: usize) -> f64 {
    let n = pos.len() / 3;

    // Compute COM for each population
    let (mut com_pos, mut com_neg) = ([0.0f64; 3], [0.0f64; 3]);

    for i in 0..n {
        let x = pos[i * 3] as f64;
        let y = pos[i * 3 + 1] as f64;
        let z = pos[i * 3 + 2] as f64;

        if signs[i] > 0 {
            com_pos[0] += x;
            com_pos[1] += y;
            com_pos[2] += z;
        } else {
            com_neg[0] += x;
            com_neg[1] += y;
            com_neg[2] += z;
        }
    }

    com_pos[0] /= n_pos as f64;
    com_pos[1] /= n_pos as f64;
    com_pos[2] /= n_pos as f64;
    com_neg[0] /= n_neg as f64;
    com_neg[1] /= n_neg as f64;
    com_neg[2] /= n_neg as f64;

    // Distance between COMs
    let dx = com_pos[0] - com_neg[0];
    let dy = com_pos[1] - com_neg[1];
    let dz = com_pos[2] - com_neg[2];

    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn save_snapshot(dir: &str, step: usize, pos: &[f32], signs: &[i8]) -> std::io::Result<()> {
    let path = format!("{}/snapshot_{:05}.bin", dir, step);
    let mut file = BufWriter::new(File::create(&path)?);

    let n = pos.len() / 3;

    // Write header: N (u64)
    file.write_all(&(n as u64).to_le_bytes())?;

    // Write particles: x, y, z (f32), sign (i8) = 13 bytes each
    for i in 0..n {
        file.write_all(&pos[i * 3].to_le_bytes())?;
        file.write_all(&pos[i * 3 + 1].to_le_bytes())?;
        file.write_all(&pos[i * 3 + 2].to_le_bytes())?;
        file.write_all(&[signs[i] as u8])?;
    }

    file.flush()?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("CUDA required"); }
