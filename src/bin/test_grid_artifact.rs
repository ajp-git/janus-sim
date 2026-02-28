//! Decisive anti-regression test for grid artifact
//!
//! Context that previously produced grid artifacts:
//! - GPU Barnes-Hut θ=0.5
//! - 100-mode Zel'dovich ICs (built into GpuNBodyTwoPass)
//! - Hubble friction (CosmoInterpolator, z_init=5)
//! - 1M particles, 1000 steps
//!
//! If no grid → GPU BH θ=0.5 is production-ready
//! If grid appears → TreePM + cuFFT remains necessary

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

const N: usize = 1_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.5;  // Critical: θ=0.5 was the problematic setting
const DT: f64 = 0.01;
const BOX_SIZE: f64 = 215.4;  // Scaled for 1M
const Z_INIT: f64 = 5.0;
const STEPS: usize = 1000;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = "/app/output/grid_artifact_test";
    let snapshot_dir = format!("{}/snapshots", output_dir);
    let render_dir = format!("{}/render_data", output_dir);
    fs::create_dir_all(&snapshot_dir)?;
    fs::create_dir_all(&render_dir)?;

    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║     DECISIVE ANTI-REGRESSION TEST: GRID ARTIFACT               ║");
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Configuration (exact context that produced grid):");
    eprintln!("  N = {} (1M)", N);
    eprintln!("  θ = {} (Barnes-Hut opening angle)", THETA);
    eprintln!("  ICs = 100-mode Zel'dovich (multi-mode, non-integer k)");
    eprintln!("  Hubble friction = ON (CosmoInterpolator, z_init={})", Z_INIT);
    eprintln!("  Steps = {}", STEPS);
    eprintln!("  Output = {}", output_dir);
    eprintln!();

    let n_positive = (N as f64 / (1.0 + ETA)) as usize;
    let n_negative = N - n_positive;

    eprintln!("Initializing GPU simulation...");
    let init_start = Instant::now();
    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);
    eprintln!("  Init time: {:.2?}", init_start.elapsed());

    // Cosmology setup
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    let a_init = 1.0 / (1.0 + Z_INIT);
    let tau_z_init = cosmo.history.iter()
        .min_by(|s1, s2| (s1.a - a_init).abs().partial_cmp(&(s2.a - a_init).abs()).unwrap())
        .map(|s| s.tau).unwrap();

    // Compute dtau for ~6000 steps to z=0 (same as production runs)
    let dtau_per_step = (-tau_z_init) / 6000.0;
    let mut tau_current = tau_z_init;

    // Get initial state
    let seg_0 = sim.segregation()?;
    let ke_0 = sim.kinetic_energy()?;
    eprintln!();
    eprintln!("Initial state:");
    eprintln!("  KE₀ = {:.4e}", ke_0);
    eprintln!("  Seg₀ = {:.6}", seg_0);

    // Save initial snapshot
    save_render_data(&sim, &render_dir, 0, BOX_SIZE, seg_0, 1.0, Z_INIT)?;
    eprintln!("  Saved step_000000.bin");

    // Run simulation
    eprintln!();
    eprintln!("Running {} steps with Hubble friction...", STEPS);
    let run_start = Instant::now();

    for step in 1..=STEPS {
        let (a, hubble) = cosmo.get_params_at_tau(tau_current);
        let z = 1.0 / a - 1.0;
        let dtau_per_dt = dtau_per_step / DT;

        sim.step_dkd_morton_warpcoherent(DT, hubble, dtau_per_dt)?;
        tau_current += dtau_per_step;

        // Progress every 100 steps
        if step % 100 == 0 {
            let elapsed = run_start.elapsed().as_secs_f64();
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;
            eprintln!("  Step {:4}/{}: z={:.2}, KE/KE₀={:.4}, Seg={:.4}, {:.1}s",
                     step, STEPS, z, ke / ke_0, seg, elapsed);
        }

        // Save critical frames: 500 and 1000
        if step == 500 || step == 1000 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;
            let (a, _) = cosmo.get_params_at_tau(tau_current);
            let z = 1.0 / a - 1.0;
            save_render_data(&sim, &render_dir, step, BOX_SIZE, seg, ke / ke_0, z)?;
            eprintln!("  >>> SAVED step_{:06}.bin (critical frame)", step);
        }
    }

    let elapsed = run_start.elapsed().as_secs_f64();
    let ke_final = sim.kinetic_energy()?;
    let seg_final = sim.segregation()?;

    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║                     RESULTS                                    ║");
    eprintln!("╚════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Steps completed: {}", STEPS);
    eprintln!("  Runtime: {:.1}s ({:.1}ms/step)", elapsed, elapsed * 1000.0 / STEPS as f64);
    eprintln!("  KE/KE₀ = {:.4}", ke_final / ke_0);
    eprintln!("  Seg: {:.6} → {:.6} ({:+.2}%)", seg_0, seg_final,
             (seg_final - seg_0) / seg_0 * 100.0);
    eprintln!();
    eprintln!("Critical frames saved:");
    eprintln!("  - {}/step_000000.bin (initial)", render_dir);
    eprintln!("  - {}/step_000500.bin (mid-run)", render_dir);
    eprintln!("  - {}/step_001000.bin (final)", render_dir);
    eprintln!();
    eprintln!(">>> RENDER THESE FRAMES AND CHECK FOR GRID ARTIFACTS <<<");
    eprintln!();
    eprintln!("If NO grid → GPU BH θ=0.5 is PRODUCTION-READY");
    eprintln!("If grid appears → TreePM + cuFFT remains necessary");

    Ok(())
}

#[cfg(feature = "cuda")]
fn save_render_data(
    sim: &GpuNBodyTwoPass,
    dir: &str,
    step: usize,
    box_size: f64,
    seg: f64,
    ke_ratio: f64,
    redshift: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let pos = sim.get_positions()?;
    let signs = sim.get_signs()?;

    let path = format!("{}/step_{:06}.bin", dir, step);
    let mut file = BufWriter::new(File::create(&path)?);

    // Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + redshift(f64) + n(u32)
    let n = (pos.len() / 3) as u32;
    file.write_all(&(step as u32).to_le_bytes())?;
    file.write_all(&box_size.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    file.write_all(&ke_ratio.to_le_bytes())?;
    file.write_all(&redshift.to_le_bytes())?;
    file.write_all(&n.to_le_bytes())?;

    // Positions: N×3×f32
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 4)
    };
    file.write_all(pos_bytes)?;

    // Signs: N×i8
    let signs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(signs.as_ptr() as *const u8, signs.len())
    };
    file.write_all(signs_bytes)?;

    file.flush()?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires CUDA support. Build with --features cuda");
}
