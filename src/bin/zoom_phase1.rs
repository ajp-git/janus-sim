//! Zoom Phase 1 — Particle splitting from existing snapshot
//!
//! Loads snap_001500.bin from v13, extracts 60 Mpc region, splits particles,
//! adds high-k Zel'dovich perturbations (position only), preserves velocities.
//!
//! CRITICAL: Velocities are PRESERVED from parent snapshot.
//! Only positions get small offsets + Zel'dovich displacement.

use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Read, Write, BufWriter};
use std::time::Instant;
use clap::Parser;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[derive(Parser)]
#[command(name = "zoom_phase1")]
struct Args {
    /// Path to source snapshot (e.g., snap_001500.bin)
    #[arg(long)]
    snapshot: String,

    /// Box size of source snapshot in Mpc
    #[arg(long, default_value = "500.0")]
    source_box: f64,

    /// Zoom region size in Mpc
    #[arg(long, default_value = "60.0")]
    zoom_box: f64,

    /// Number of splits per particle (e.g., 8 or 27)
    #[arg(long, default_value = "20")]
    n_split: usize,

    /// Minimum k mode for Zel'dovich perturbations
    #[arg(long, default_value = "20")]
    k_min: usize,

    /// Softening length in Mpc
    #[arg(long, default_value = "0.14")]
    eps: f64,

    /// Zel'dovich amplitude (position perturbation only)
    #[arg(long, default_value = "0.548")]
    amp: f64,

    /// Output directory
    #[arg(long)]
    output: String,

    /// Number of steps to run
    #[arg(long, default_value = "3000")]
    steps: usize,

    /// Starting redshift (from snapshot)
    #[arg(long, default_value = "1.63")]
    z_start: f64,

    /// Random seed
    #[arg(long, default_value = "12345")]
    seed: u64,

    /// Center X for zoom region (in source coordinates, -250 to +250)
    #[arg(long)]
    center_x: Option<f64>,

    /// Center Y for zoom region
    #[arg(long)]
    center_y: Option<f64>,

    /// Center Z for zoom region
    #[arg(long)]
    center_z: Option<f64>,
}

const ETA: f64 = 1.045;
const THETA: f64 = 0.5;
const DT: f64 = 0.01;
const SNAP_INT: usize = 100;
const LOG_INT: usize = 10;

/// Load binary snapshot: header u64 N, then N × 28 bytes (7 × f32)
fn load_snapshot(path: &str) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<i8>) {
    let mut file = File::open(path).expect("Cannot open snapshot");
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).unwrap();
    let n = u64::from_le_bytes(buf) as usize;

    let mut data = vec![0u8; n * 28];
    file.read_exact(&mut data).unwrap();

    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    let mut signs = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 28;
        let x = f32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as f64;
        let y = f32::from_le_bytes(data[offset+4..offset+8].try_into().unwrap()) as f64;
        let z = f32::from_le_bytes(data[offset+8..offset+12].try_into().unwrap()) as f64;
        let vx = f32::from_le_bytes(data[offset+12..offset+16].try_into().unwrap()) as f64;
        let vy = f32::from_le_bytes(data[offset+16..offset+20].try_into().unwrap()) as f64;
        let vz = f32::from_le_bytes(data[offset+20..offset+24].try_into().unwrap()) as f64;
        let sign_f32 = f32::from_le_bytes(data[offset+24..offset+28].try_into().unwrap());

        pos.push([x, y, z]);
        vel.push([vx, vy, vz]);
        signs.push(if sign_f32 > 0.0 { 1i8 } else { -1i8 });
    }

    (pos, vel, signs)
}

/// Extract particles in zoom region and apply particle splitting
/// CRITICAL: Velocities are PRESERVED exactly from parent!
fn extract_and_split(
    pos: &[[f64; 3]],
    vel: &[[f64; 3]],
    signs: &[i8],
    source_box: f64,
    zoom_box: f64,
    center: [f64; 3],
    n_split: usize,
    eps: f64,
    seed: u64,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<i8>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let half_zoom = zoom_box / 2.0;

    // Gaussian offset for daughter positions: σ = 0.3 × ε
    let sigma_pos = 0.3 * eps;
    let normal = Normal::new(0.0, sigma_pos).unwrap();

    let mut new_pos = Vec::new();
    let mut new_vel = Vec::new();
    let mut new_signs = Vec::new();

    let mut n_parent = 0;

    for i in 0..pos.len() {
        // Check if particle is in zoom region
        let dx = pos[i][0] - center[0];
        let dy = pos[i][1] - center[1];
        let dz = pos[i][2] - center[2];

        // Periodic wrap for distance check
        let dx = if dx > source_box/2.0 { dx - source_box }
                 else if dx < -source_box/2.0 { dx + source_box }
                 else { dx };
        let dy = if dy > source_box/2.0 { dy - source_box }
                 else if dy < -source_box/2.0 { dy + source_box }
                 else { dy };
        let dz = if dz > source_box/2.0 { dz - source_box }
                 else if dz < -source_box/2.0 { dz + source_box }
                 else { dz };

        if dx.abs() <= half_zoom && dy.abs() <= half_zoom && dz.abs() <= half_zoom {
            n_parent += 1;

            // Create n_split daughter particles
            for _ in 0..n_split {
                // Position: parent + small Gaussian offset
                let ox: f64 = normal.sample(&mut rng);
                let oy: f64 = normal.sample(&mut rng);
                let oz: f64 = normal.sample(&mut rng);

                // Map to zoom box coordinates (centered at origin)
                let new_x = dx + ox;
                let new_y = dy + oy;
                let new_z = dz + oz;

                // Velocity: PRESERVED EXACTLY from parent!
                new_pos.push([new_x, new_y, new_z]);
                new_vel.push(vel[i]);
                new_signs.push(signs[i]);
            }
        }
    }

    println!("  Extracted {} parent particles from zoom region", n_parent);
    println!("  Created {} daughter particles ({}× split)", new_pos.len(), n_split);

    (new_pos, new_vel, new_signs)
}

/// Add high-k Zel'dovich perturbations to POSITIONS ONLY
/// CRITICAL: Do NOT modify velocities - they come from the evolved snapshot!
fn add_zeldovich_position_perturbations(
    pos: &mut [[f64; 3]],
    zoom_box: f64,
    k_min: usize,
    amp: f64,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed + 999);
    let n = pos.len();

    // Number of modes to add
    let k_max = k_min + 10;  // Small range of high-k modes
    let n_modes = 50;  // Number of random mode directions

    println!("  Adding Zel'dovich position perturbations (k={}-{}, {} modes)", k_min, k_max, n_modes);

    // Generate random mode directions and phases
    let mut modes: Vec<([f64; 3], f64, f64)> = Vec::with_capacity(n_modes);
    for _ in 0..n_modes {
        // Random k magnitude in [k_min, k_max]
        let k_mag = k_min as f64 + rng.gen::<f64>() * (k_max - k_min) as f64;

        // Random direction on unit sphere
        let theta = rng.gen::<f64>() * 2.0 * PI;
        let phi = (1.0 - 2.0 * rng.gen::<f64>()).acos();
        let kx = k_mag * phi.sin() * theta.cos();
        let ky = k_mag * phi.sin() * theta.sin();
        let kz = k_mag * phi.cos();

        // Random phase
        let phase = rng.gen::<f64>() * 2.0 * PI;

        // Amplitude scaled by k^(-3/2) for realistic power spectrum
        let mode_amp = amp * (k_min as f64 / k_mag).powf(1.5);

        modes.push(([kx, ky, kz], phase, mode_amp));
    }

    // Apply perturbations to each particle
    let cell = zoom_box / (n as f64).cbrt();
    for p in pos.iter_mut() {
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut dz = 0.0;

        for (k_vec, phase, mode_amp) in &modes {
            // k · r
            let k_dot_r = k_vec[0] * p[0] / zoom_box * 2.0 * PI
                        + k_vec[1] * p[1] / zoom_box * 2.0 * PI
                        + k_vec[2] * p[2] / zoom_box * 2.0 * PI;

            let sin_term = (k_dot_r + phase).sin();

            // Displacement along k direction (Zel'dovich)
            let k_mag = (k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]).sqrt();
            if k_mag > 0.0 {
                dx += mode_amp * k_vec[0] / k_mag * sin_term * cell;
                dy += mode_amp * k_vec[1] / k_mag * sin_term * cell;
                dz += mode_amp * k_vec[2] / k_mag * sin_term * cell;
            }
        }

        p[0] += dx;
        p[1] += dy;
        p[2] += dz;
    }

    // Report RMS displacement
    let rms: f64 = pos.iter()
        .map(|p| p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        .sum::<f64>() / n as f64;
    println!("  RMS position after Zel'dovich: {:.3} Mpc", rms.sqrt());
}

/// Compute segregation metrics: σ_P (polarity fluctuation) and L_J (Jeans length)
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_metrics(sim: &GpuNBodyTwoPass, g: usize, l_box: f64) -> (f64, f64) {
    let (pos, _, signs) = match sim.get_particles() { Ok(d) => d, Err(_) => return (0.0, 0.0) };
    let cell = l_box / g as f64;
    let mut rp = vec![0.0f64; g*g*g];
    let mut rm = vec![0.0f64; g*g*g];

    for i in 0..pos.len()/3 {
        let x = pos[i*3] as f64 + l_box/2.0;
        let y = pos[i*3+1] as f64 + l_box/2.0;
        let z = pos[i*3+2] as f64 + l_box/2.0;
        let ix = ((x/cell) as usize).min(g-1);
        let iy = ((y/cell) as usize).min(g-1);
        let iz = ((z/cell) as usize).min(g-1);
        let idx = ix + iy*g + iz*g*g;
        if signs[i] > 0 { rp[idx] += 1.0; } else { rm[idx] += 1.0; }
    }

    // Polarity field: P = (n+ - n-)/(n+ + n-)
    let pf: Vec<f64> = rp.iter().zip(&rm).map(|(&p, &m)| if p+m > 0.0 { (p-m)/(p+m) } else { 0.0 }).collect();
    let pm: f64 = pf.iter().sum::<f64>() / pf.len() as f64;
    let sp = (pf.iter().map(|&p| (p-pm).powi(2)).sum::<f64>() / pf.len() as f64).sqrt();

    // Gradient magnitude for L_J
    let mut gs = 0.0;
    for iz in 0..g { for iy in 0..g { for ix in 0..g {
        let gx = (pf[((ix+1)%g) + iy*g + iz*g*g] - pf[((ix+g-1)%g) + iy*g + iz*g*g]) / (2.0*cell);
        let gy = (pf[ix + ((iy+1)%g)*g + iz*g*g] - pf[ix + ((iy+g-1)%g)*g + iz*g*g]) / (2.0*cell);
        let gz = (pf[ix + iy*g + ((iz+1)%g)*g*g] - pf[ix + iy*g + ((iz+g-1)%g)*g*g]) / (2.0*cell);
        gs += (gx*gx + gy*gy + gz*gz).sqrt();
    }}}
    let mg = gs / (g*g*g) as f64;
    let lj = if mg > 0.0 { sp / mg } else { 0.0 };

    (sp, lj)
}

/// Find center of mass of m- particles (for centering zoom on antimatter concentration)
fn find_antimatter_com(pos: &[[f64; 3]], signs: &[i8], box_size: f64) -> [f64; 3] {
    // Use minimum image convention for periodic COM
    let mut sum_cos = [0.0f64; 3];
    let mut sum_sin = [0.0f64; 3];
    let mut n_minus = 0usize;

    for (i, &sign) in signs.iter().enumerate() {
        if sign < 0 {
            for d in 0..3 {
                let angle = 2.0 * PI * pos[i][d] / box_size;
                sum_cos[d] += angle.cos();
                sum_sin[d] += angle.sin();
            }
            n_minus += 1;
        }
    }

    if n_minus == 0 {
        return [0.0, 0.0, 0.0];
    }

    let mut com = [0.0f64; 3];
    for d in 0..3 {
        let avg_cos = sum_cos[d] / n_minus as f64;
        let avg_sin = sum_sin[d] / n_minus as f64;
        let angle = avg_sin.atan2(avg_cos);
        com[d] = angle * box_size / (2.0 * PI);
    }

    com
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let args = Args::parse();

    let zoom_box = args.zoom_box;
    let eps = args.eps;
    let amp = args.amp;
    let steps = args.steps;
    let out = &args.output;
    let k_min = args.k_min;
    let n_split = args.n_split;
    let z_start = args.z_start;

    // r_cut scaled to zoom box
    let r_cut = 0.09 * zoom_box;

    fs::create_dir_all(format!("{}/snapshots", out)).unwrap();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  ZOOM PHASE 1 — Particle Splitting from Snapshot          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("Source: {}", args.snapshot);
    println!("Source box: {} Mpc → Zoom box: {} Mpc", args.source_box, zoom_box);
    println!("n_split={} k_min={} ε={:.3} amp={:.3}", n_split, k_min, eps, amp);

    // Load source snapshot
    println!("\nLoading snapshot...");
    let t0 = Instant::now();
    let (src_pos, src_vel, src_signs) = load_snapshot(&args.snapshot);
    println!("  Loaded {} particles in {:.1}s", src_pos.len(), t0.elapsed().as_secs_f64());

    // Find zoom center (COM of m- if not specified, or use args)
    let center = if args.center_x.is_some() || args.center_y.is_some() || args.center_z.is_some() {
        // User specified at least one coordinate - use explicit center
        [args.center_x.unwrap_or(0.0), args.center_y.unwrap_or(0.0), args.center_z.unwrap_or(0.0)]
    } else {
        // Auto-find: use antimatter COM
        println!("\nFinding antimatter COM for zoom centering...");
        let com = find_antimatter_com(&src_pos, &src_signs, args.source_box);
        println!("  m- COM: ({:.1}, {:.1}, {:.1})", com[0], com[1], com[2]);
        com
    };

    // Extract and split particles
    println!("\nExtracting zoom region and splitting particles...");
    println!("  Center: ({:.1}, {:.1}, {:.1}), Size: {} Mpc", center[0], center[1], center[2], zoom_box);
    let (mut pos, vel, signs) = extract_and_split(
        &src_pos, &src_vel, &src_signs,
        args.source_box, zoom_box, center,
        n_split, eps, args.seed
    );

    let n = pos.len();
    if n < 1000 {
        println!("ERROR: Only {} particles in zoom region - too few!", n);
        println!("Try a larger zoom_box or different center.");
        return;
    }

    // Add high-k Zel'dovich perturbations (POSITION ONLY!)
    println!("\nAdding high-k Zel'dovich perturbations (position only)...");
    add_zeldovich_position_perturbations(&mut pos, zoom_box, k_min, amp, args.seed);

    // Count particles by sign
    let np = signs.iter().filter(|&&s| s > 0).count();
    let nm = n - np;
    println!("\nFinal particle counts: N+={} N-={} Total={}", np, nm, n);

    // Velocity statistics (should be preserved from snapshot)
    let v_rms: f64 = vel.iter()
        .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        .sum::<f64>() / n as f64;
    println!("Velocity RMS: {:.4} (preserved from z={:.2} snapshot)", v_rms.sqrt(), z_start);

    // Convert to f32 for GPU
    let pos_f32: Vec<f32> = pos.iter().flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let vel_f32: Vec<f32> = vel.iter().flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
    let signs_i8: Vec<i8> = signs.clone();

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, zoom_box).unwrap();
    sim.set_theta(THETA);
    sim.set_softening(eps);
    sim.set_pm_k_min(2);

    // NO virialization! Velocities are already correct from evolved snapshot.
    // Virialization would destroy the bulk flow.
    println!("SKIP virialization — velocities preserved from evolved snapshot");

    // Setup cosmology from z_start
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, z_start);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / (steps as f64 * DT);
    let (a0, h0) = cosmo.get_params_at_tau(cosmo.tau_start);
    println!("Starting from z={:.3} (a={:.4}, H={:.4})", z_start, a0, h0);

    // Create CSV for time series
    let mut csv = BufWriter::new(File::create(format!("{}/time_series.csv", out)).unwrap());
    writeln!(csv, "step,z,a,H,KE_ratio,seg,sigma_P,L_J").unwrap();

    // Initial KE for ratio
    let ke0 = sim.kinetic_energy().unwrap().max(1e-20);
    let seg0 = sim.segregation().unwrap();
    println!("Initial: KE0={:.4e}, Seg0={:.4}", ke0, seg0);

    // Initial snapshot
    save_snapshot(&sim, &format!("{}/snapshots/snap_{:06}.bin", out, 0));
    let (sp0, lj0) = compute_metrics(&sim, 32, zoom_box);
    writeln!(csv, "0,{:.4},{:.5},{:.5},1.0,{:.4},{:.4},{:.2}", z_start, a0, h0, seg0, sp0, lj0).unwrap();

    println!("\n{:>6} {:>7} {:>10} {:>8} {:>7} {:>6} {:>5}", "Step", "z", "KE/KE0", "Seg", "σ_P", "L_J", "ms");
    println!("{}", "─".repeat(55));

    let start_time = Instant::now();
    let mut ke_max = 1.0f64;

    for step in 1..=steps {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau;
        let (a, h) = if tau <= cosmo.tau_end { cosmo.get_params_at_tau(tau) } else { (1.0, 0.0) };
        let z = if a > 0.0 { 1.0 / a - 1.0 } else { 0.0 };

        // Integrate with TreePM + Hubble friction
        let t0 = Instant::now();
        if let Err(e) = sim.step_treepm_gpu(DT, r_cut, h, dtau) {
            println!("ERROR step {}: {}", step, e);
            break;
        }
        let ms = t0.elapsed().as_millis();

        // Logging
        if step % LOG_INT == 0 {
            let ke = sim.kinetic_energy().unwrap();
            let ke_ratio = ke / ke0;
            if ke_ratio > ke_max { ke_max = ke_ratio; }
            let seg = sim.segregation().unwrap();

            let (sp, lj) = if step % SNAP_INT == 0 { compute_metrics(&sim, 32, zoom_box) } else { (0.0, 0.0) };

            writeln!(csv, "{},{:.4},{:.5},{:.5},{:.4e},{:.4},{:.4},{:.2}",
                step, z, a, h, ke_ratio, seg, sp, lj).unwrap();

            if step % SNAP_INT == 0 {
                println!("{:>6} {:>7.3} {:>10.3e} {:>8.4} {:>7.4} {:>6.1} {:>5}",
                    step, z, ke_ratio, seg, sp, lj, ms);
                save_snapshot(&sim, &format!("{}/snapshots/snap_{:06}.bin", out, step));
                csv.flush().unwrap();
            }

            // Early stop if KE explodes
            if ke_ratio > 100.0 {
                println!("\n⚠ KE explosion at step {} (ratio={:.2e}) — stopping", step, ke_ratio);
                break;
            }
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!("\n══════════════════════════════════════════════════════════");
    println!("Completed {} steps in {:.1}s ({:.2} steps/s)", steps, elapsed, steps as f64 / elapsed);

    // Final metrics
    let seg_final = sim.segregation().unwrap();
    let ke_final = sim.kinetic_energy().unwrap() / ke0;
    let (sp_final, lj_final) = compute_metrics(&sim, 32, zoom_box);
    println!("Final: Seg={:.4}, KE/KE0={:.3}, σ_P={:.4}, L_J={:.1}", seg_final, ke_final, sp_final, lj_final);

    // Save summary
    let summary = format!(r#"{{
  "n": {},
  "zoom_box": {},
  "source_box": {},
  "n_split": {},
  "k_min": {},
  "eps": {},
  "amp": {},
  "z_start": {},
  "steps": {},
  "seg_final": {},
  "sigma_P": {},
  "L_J": {},
  "KE_max": {},
  "KE_final": {},
  "time_s": {:.1}
}}"#, n, zoom_box, args.source_box, n_split, k_min, eps, amp, z_start, steps, seg_final, sp_final, lj_final, ke_max, ke_final, elapsed);
    fs::write(format!("{}/summary.json", out), summary).unwrap();
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &str) {
    let (pos, vel, signs) = sim.get_particles().unwrap();
    let n = signs.len();

    let mut file = File::create(path).unwrap();
    file.write_all(&(n as u64).to_le_bytes()).unwrap();

    for i in 0..n {
        let x = pos[i * 3] as f32;
        let y = pos[i * 3 + 1] as f32;
        let z = pos[i * 3 + 2] as f32;
        let vx = vel[i * 3] as f32;
        let vy = vel[i * 3 + 1] as f32;
        let vz = vel[i * 3 + 2] as f32;
        let sign = signs[i] as f32;

        file.write_all(&x.to_le_bytes()).unwrap();
        file.write_all(&y.to_le_bytes()).unwrap();
        file.write_all(&z.to_le_bytes()).unwrap();
        file.write_all(&vx.to_le_bytes()).unwrap();
        file.write_all(&vy.to_le_bytes()).unwrap();
        file.write_all(&vz.to_le_bytes()).unwrap();
        file.write_all(&sign.to_le_bytes()).unwrap();
    }
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
}
