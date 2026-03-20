//! JANUS V16 RESUME — Continue from snapshot
//! Loads last snapshot and continues simulation

use std::fs::{self, File};
use std::io::{Read, Write, BufWriter};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const L_BOX: f64 = 500.0;
const ETA: f64 = 1.045;
const EPSILON: f64 = 0.25;
const THETA: f64 = 0.7;
const R_CUT: f64 = 40.0;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const STEPS: usize = 5000;
const SNAP_INT: usize = 7;
const LOG_INT: usize = 10;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn load_snapshot(path: &str) -> (Vec<f32>, Vec<f32>, Vec<i8>, usize) {
    let mut f = File::open(path).expect("Cannot open snapshot");
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8).unwrap();
    let n = u64::from_le_bytes(buf8) as usize;

    let mut data = vec![0u8; n * 28];
    f.read_exact(&mut data).unwrap();

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for i in 0..n {
        let off = i * 28;
        let x = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let y = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
        let z = f32::from_le_bytes([data[off+8], data[off+9], data[off+10], data[off+11]]);
        let vx = f32::from_le_bytes([data[off+12], data[off+13], data[off+14], data[off+15]]);
        let vy = f32::from_le_bytes([data[off+16], data[off+17], data[off+18], data[off+19]]);
        let vz = f32::from_le_bytes([data[off+20], data[off+21], data[off+22], data[off+23]]);
        let m = f32::from_le_bytes([data[off+24], data[off+25], data[off+26], data[off+27]]);

        pos.push(x); pos.push(y); pos.push(z);
        vel.push(vx); vel.push(vy); vel.push(vz);
        signs.push(if m > 0.0 { 1i8 } else { -1i8 });
    }

    (pos, vel, signs, n)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snap(sim: &GpuNBodyTwoPass, path: &str, n: usize) {
    let pos = sim.get_positions().unwrap();
    let vel = sim.get_velocities().unwrap();
    let signs = sim.get_signs().unwrap();

    let mut f = BufWriter::new(File::create(path).unwrap());
    f.write_all(&(n as u64).to_le_bytes()).unwrap();

    for i in 0..n {
        let m: f32 = if signs[i] > 0 { 1.0 } else { -1.0 };
        f.write_all(&pos[3*i].to_le_bytes()).unwrap();
        f.write_all(&pos[3*i+1].to_le_bytes()).unwrap();
        f.write_all(&pos[3*i+2].to_le_bytes()).unwrap();
        f.write_all(&vel[3*i].to_le_bytes()).unwrap();
        f.write_all(&vel[3*i+1].to_le_bytes()).unwrap();
        f.write_all(&vel[3*i+2].to_le_bytes()).unwrap();
        f.write_all(&m.to_le_bytes()).unwrap();
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let out = "/app/output/janus_v16_500Mpc_30M_kmin10";

    // Find last snapshot
    let snap_dir = format!("{}/snapshots", out);
    let mut snaps: Vec<_> = fs::read_dir(&snap_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "bin").unwrap_or(false))
        .collect();
    snaps.sort_by_key(|e| e.path());

    let last_snap = snaps.last().expect("No snapshots found");
    let snap_name = last_snap.file_name().to_string_lossy().to_string();
    let start_step: usize = snap_name
        .replace("snap_", "")
        .replace(".bin", "")
        .parse()
        .unwrap();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  JANUS V16 RESUME — from step {}                       ║", start_step);
    println!("╚═══════════════════════════════════════════════════════════╝");

    let snap_path = last_snap.path();
    println!("Loading {}...", snap_path.display());
    let (pos, vel, signs, n) = load_snapshot(snap_path.to_str().unwrap());

    let np = signs.iter().filter(|&&s| s > 0).count();
    let nm = n - np;
    println!("  N={} N+={} N-={}", n, np, nm);

    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    // Calculate current tau/z from start_step
    let tau_now = cosmo.tau_start + (start_step as f64) * DT * dtau;
    let (a_now, h_now) = cosmo.get_params_at_tau(tau_now);
    let z_now = 1.0 / a_now - 1.0;
    println!("  Resuming at step={} z={:.3} a={:.5}", start_step, z_now, a_now);

    println!("Init GPU...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, L_BOX).unwrap();
    sim.set_theta(THETA);
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(2);

    let ke0 = 1.3275e3;  // Original KE0 from V16 start
    let seg_now = sim.segregation().unwrap_or(0.0);
    let ke_now = sim.kinetic_energy().unwrap_or(0.0);
    println!("KE={:.4e} KE/KE0={:.4e} Seg={:.4}", ke_now, ke_now/ke0, seg_now);

    // Append to time_series.csv
    let mut ts = fs::OpenOptions::new()
        .append(true)
        .open(format!("{}/time_series.csv", out))
        .unwrap();

    println!("\n{:>5} {:>7} {:>7} {:>10} {:>7} {:>6} {:>6} {:>5}",
             "Step", "z", "a", "KE/KE0", "Seg", "σ_P", "L_J", "ms");
    println!("{}", "─".repeat(65));

    let start = Instant::now();

    for step in (start_step + 1)..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau;
        let (a, h) = if tau <= cosmo.tau_end { cosmo.get_params_at_tau(tau) } else { (1.0, 0.0) };
        let z = if a > 0.0 { 1.0/a - 1.0 } else { 0.0 };

        let t0 = Instant::now();
        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, h, dtau) {
            println!("ERROR step {}: {}", step, e);
            break;
        }
        let ms = t0.elapsed().as_millis();

        // Snapshot
        if step % SNAP_INT == 0 {
            let snap_path = format!("{}/snapshots/snap_{:06}.bin", out, step);
            save_snap(&sim, &snap_path, n);
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (step - start_step) as f64 / elapsed;
            let remaining = (STEPS - step) as f64 / rate / 3600.0;
            println!("  → {} | ETA: {:.1}h", snap_path.split('/').last().unwrap(), remaining);
        }

        // Log
        if step % LOG_INT == 0 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let seg = sim.segregation().unwrap_or(0.0);

            writeln!(ts, "{},{:.4},{:.5},{:.5},{:.4e},{:.4},{:.4},{:.2},{:.2}",
                     step, z, a, h, ke/ke0, seg, 0.0, 0.0, 0.0).unwrap();

            println!("{:5}   {:.3} {:.5}    {:.3e}  {:.4}      -      - {:5}",
                     step, z, a, ke/ke0, seg, ms);
        }

        // Stop at z=0
        if z <= 0.001 {
            println!("\nReached z ≈ 0, stopping.");
            break;
        }
    }

    println!("\nV16 RESUME complete.");
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    println!("This binary requires --features cuda,cufft");
}
