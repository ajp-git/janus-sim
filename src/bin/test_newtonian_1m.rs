//! Test 1M particles with PURE NEWTONIAN gravity (all masses positive)
//! Diagnostic to check if grid artifact is from Barnes-Hut or Janus ICs

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{BufWriter, Write};

const N: usize = 1_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const BOX_SIZE: f64 = 215.4;
const Z_INIT: f64 = 5.0;
const Z_FINAL: f64 = 1.5;
const STEPS: usize = 500;
const SNAPSHOT_INTERVAL: usize = 100;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = "/app/output/1M_newtonian_test";
    let snapshot_dir = format!("{}/snapshots", output_dir);
    fs::create_dir_all(&snapshot_dir)?;

    eprintln!("1M NEWTONIAN TEST (all masses positive)");
    eprintln!("Diagnostic: check if grid artifact is from Barnes-Hut");
    eprintln!();

    // Almost all particles positive (N+ = N-1, N- = 1) → quasi-Newtonian
    // Can't use N-=0 due to CUDA kernel constraints
    let n_positive = N - 1;
    let n_negative = 1;

    let mut sim = GpuNBodyTwoPass::new(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);

    // Cosmology (for Hubble friction timing)
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    let a_init = 1.0 / (1.0 + Z_INIT);
    let a_final = 1.0 / (1.0 + Z_FINAL);

    let tau_z_init = cosmo.history.iter()
        .min_by(|s1, s2| (s1.a - a_init).abs().partial_cmp(&(s2.a - a_init).abs()).unwrap())
        .map(|s| s.tau).unwrap();
    let tau_z_final = cosmo.history.iter()
        .min_by(|s1, s2| (s1.a - a_final).abs().partial_cmp(&(s2.a - a_final).abs()).unwrap())
        .map(|s| s.tau).unwrap();

    let dtau_per_step = (tau_z_final - tau_z_init) / 6000.0;
    let mut tau_current = tau_z_init;

    // Save initial snapshot
    let pos = sim.get_positions()?;
    let signs = sim.get_signs()?;
    save_snapshot(&snapshot_dir, 0, &pos, &signs)?;
    eprintln!("Saved snapshot_00000.bin");

    // Run simulation
    eprintln!("Running {} steps (Newtonian only)...", STEPS);
    for step in 1..=STEPS {
        let (_, hubble) = cosmo.get_params_at_tau(tau_current);
        let dtau_per_dt = dtau_per_step / DT;
        sim.step_dkd_morton_warpcoherent(DT, hubble, dtau_per_dt)?;
        tau_current += dtau_per_step;

        if step % SNAPSHOT_INTERVAL == 0 {
            let pos = sim.get_positions()?;
            let signs = sim.get_signs()?;
            save_snapshot(&snapshot_dir, step, &pos, &signs)?;

            let (a, _) = cosmo.get_params_at_tau(tau_current);
            let z = 1.0 / a - 1.0;
            eprintln!("  Step {}/{} z={:.3}", step, STEPS, z);
        }
    }

    eprintln!();
    eprintln!("Done! Check frame_00500 for grid artifact.");

    Ok(())
}

fn save_snapshot(dir: &str, step: usize, pos: &[f32], signs: &[i8]) -> std::io::Result<()> {
    let path = format!("{}/snapshot_{:05}.bin", dir, step);
    let mut file = BufWriter::new(File::create(&path)?);
    let n = pos.len() / 3;
    file.write_all(&(n as u64).to_le_bytes())?;
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
