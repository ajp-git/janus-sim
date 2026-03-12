//! JANUS V10 — High Resolution Segregation Scale Measurement
//!
//! Goal: Measure intrinsic Janus segregation scale D_Janus
//! Method: 256³ grid analysis on L=200 Mpc box
//!
//! OPTIMIZATIONS:
//! - Dynamic theta: 0.7 → 0.9 → 1.1 as clustering develops
//! - Larger initial timestep (dt=0.02 for step<1000)

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// V10 PARAMETERS (corrected based on LHS stable runs)
const L_BOX: f64 = 200.0;      // Mpc - smaller box for higher resolution
const K_MIN: usize = 2;        // Avoid global mode collapse (k_min=1 too violent)
const ETA: f64 = 1.06;
const H_HUBBLE: f64 = 0.012;   // LHS stable range: 0.01-0.015
const ALPHA_IC: f64 = 1.6;     // Velocity scaling
const EPSILON: f64 = 0.18;     // Mpc
const TOTAL_STEPS: usize = 3000;  // Reduced: domain structure converges by step 3000
const SNAPSHOT_STEPS: [usize; 4] = [500, 1000, 2000, 3000];  // Include post-relaxation snapshot

// TreePM parameters
const R_CUT: f64 = 30.0;       // TreePM split scale
const DTAU_PER_DT: f64 = 0.0;

// P(k) parameters for ICs
const Z_INIT: f64 = 5.0;
const K_CUT: f64 = 0.25;       // Mpc⁻¹
const PK_INDEX: f64 = -2.0;    // P(k) ∝ k^n
const AMPLITUDE: f64 = 0.02;

// GPU test sizes (N_GRID values -> N = N_GRID³)
const TEST_GRIDS: [usize; 4] = [200, 229, 252, 271];  // ≈ 8M, 12M, 16M, 20M

// STABILIZATION: Conservative parameters for 20M particles
fn get_theta(step: usize) -> f64 {
    // Safer opening angle during relaxation
    if step < 500 { 0.5 }
    else { 0.7 }
}

fn get_dt(step: usize) -> f64 {
    // Small timestep during relaxation to prevent instability
    if step < 200 { 0.003 }       // Very conservative: relaxation
    else if step < 1000 { 0.005 } // Conservative: settling
    else { 0.01 }                  // Normal: science phase
}

// PM skip optimization not available (no BH-only API)
// All steps use full TreePM

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  JANUS V10 — HIGH RESOLUTION (OPTIMIZED)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Stabilization enabled:");
    println!("  • Conservative θ: 0.5 (step<500) → 0.7");
    println!("  • Adaptive dt: 0.003 → 0.005 → 0.01");
    println!("  • Dynamic relaxation (no virialization)");
    println!();

    let output_dir = "/app/output/janus_v10_highres";
    fs::create_dir_all(format!("{}/snapshots", output_dir)).unwrap();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: GPU LIMIT DETECTION
    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 1: GPU LIMIT DETECTION                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut n_grid_max = 0usize;
    let mut step_time_ms = 0u128;

    for &ng in &TEST_GRIDS {
        let n = ng * ng * ng;
        println!("Testing N_GRID = {} → N = {}M...", ng, n / 1_000_000);

        match test_gpu_capacity(ng) {
            Ok(time_ms) => {
                println!("  ✓ N = {}M: {:.1} sec/step", n / 1_000_000, time_ms as f64 / 1000.0);
                n_grid_max = ng;
                step_time_ms = time_ms;
            }
            Err(e) => {
                println!("  ✗ N = {}M: FAILED ({})", n / 1_000_000, e);
                break;
            }
        }
    }

    if n_grid_max == 0 {
        println!("\n✗ No valid GPU configuration found!");
        return;
    }

    let n_max = n_grid_max * n_grid_max * n_grid_max;

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  N_GRID = {} → N = {}M particles", n_grid_max, n_max / 1_000_000);
    println!("  Estimated time (optimized): {:.1} hours for {} steps",
             (step_time_ms as f64 * TOTAL_STEPS as f64 * 0.7) / 3_600_000.0,  // ~70% with dynamic theta
             TOTAL_STEPS);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Save GPU limits
    let mut f = File::create(format!("{}/gpu_limits.txt", output_dir)).unwrap();
    writeln!(f, "N_GRID = {}", n_grid_max).unwrap();
    writeln!(f, "N = {}", n_max).unwrap();
    writeln!(f, "N_millions = {:.1}", n_max as f64 / 1_000_000.0).unwrap();
    writeln!(f, "L_box = {} Mpc", L_BOX).unwrap();
    writeln!(f, "optimized = true").unwrap();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: SCIENCE RUN
    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 2: HIGH RESOLUTION SCIENCE RUN (OPTIMIZED)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    println!("Parameters:");
    println!("  N_GRID = {} → N = {} ({:.1}M)", n_grid_max, n_max, n_max as f64 / 1e6);
    println!("  L = {} Mpc", L_BOX);
    println!("  ε = {} Mpc", EPSILON);
    println!("  η = {}", ETA);
    println!("  H = {}", H_HUBBLE);
    println!("  Steps = {}", TOTAL_STEPS);
    println!("  Snapshots at: {:?}", SNAPSHOT_STEPS);
    println!();

    // Save parameters
    let params_json = format!(r#"{{
  "version": "V10_optimized",
  "description": "High resolution with dynamic theta and PM skip",
  "N_GRID": {},
  "N": {},
  "L_box_mpc": {},
  "epsilon_mpc": {},
  "eta": {},
  "H": {},
  "alpha_IC": {},
  "k_min": {},
  "total_steps": {},
  "snapshot_steps": {:?},
  "grid_analysis": 256,
  "optimizations": {{
    "dynamic_theta": [0.7, 0.9, 1.1],
    "dt_initial": 0.02,
    "dt_final": 0.01
  }}
}}"#, n_grid_max, n_max, L_BOX, EPSILON, ETA, H_HUBBLE, ALPHA_IC, K_MIN, TOTAL_STEPS, SNAPSHOT_STEPS);

    fs::write(format!("{}/params.json", output_dir), params_json).unwrap();

    // Generate ICs
    println!("Generating Zel'dovich ICs (k_min={})...", K_MIN);
    let t0 = Instant::now();
    let (pos, vel, signs) = generate_zeldovich_ics(42, n_grid_max);
    println!("  ICs generated in {:.1}s", t0.elapsed().as_secs_f64());

    // Convert to GPU format
    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize GPU
    println!("Initializing GPU simulation...");
    let t0 = Instant::now();
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX)
        .expect("Failed to create GPU simulation");
    println!("  GPU initialized in {:.1}s", t0.elapsed().as_secs_f64());

    // Configure simulation
    sim.set_theta(get_theta(0));
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(K_MIN);

    // NO VIRIALIZATION - Zel'dovich ICs are cosmological perturbations
    // The system will relax naturally during the first ~500 steps
    // (Virialization is for bound systems like halos, not linear perturbation fields)
    println!("Using Zel'dovich ICs directly (α_IC = {}):", ALPHA_IC);
    println!("  No virialization (cosmological perturbations, not virialized halos)");
    println!("  System will relax naturally during first ~500 steps");

    // Get initial state (Zel'dovich ICs)
    let ke0 = sim.kinetic_energy().unwrap_or(0.0).max(1e-20);
    let seg0 = sim.segregation().unwrap_or(0.0);

    println!();
    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  Seg₀ = {:.4}", seg0);
    println!();

    // Open time series file
    let mut ts_file = File::create(format!("{}/time_series.csv", output_dir)).unwrap();
    writeln!(ts_file, "step,KE,KE_ratio,segregation,theta,dt").unwrap();

    // Run simulation
    println!("Running {} steps (optimized)...", TOTAL_STEPS);
    let run_start = Instant::now();
    let mut snapshot_idx = 0;
    let mut last_theta = get_theta(0);

    for step in 1..=TOTAL_STEPS {
        // Dynamic theta
        let theta = get_theta(step);
        if theta != last_theta {
            sim.set_theta(theta);
            println!("  [Step {}] θ changed to {:.1}", step, theta);
            last_theta = theta;
        }

        let dt = get_dt(step);

        let t0 = Instant::now();

        // Full TreePM step
        if let Err(e) = sim.step_treepm_gpu(dt, R_CUT, H_HUBBLE, DTAU_PER_DT) {
            println!("  Step {} error: {}", step, e);
            break;
        }

        let step_ms = t0.elapsed().as_millis();

        // Log progress every 100 steps
        if step % 100 == 0 || step <= 5 {
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let seg = sim.segregation().unwrap_or(0.0);

            println!("  Step {}: KE={:.2e}, Seg={:.4}, θ={:.2}, dt={:.4} ({} ms)",
                     step, ke, seg, theta, dt, step_ms);

            writeln!(ts_file, "{},{:.6e},{:.6},{:.1},{:.3}",
                     step, ke, seg, theta, dt).unwrap();

            // No auto-stop based on KE ratio for cosmological runs
            // KE will grow as structures form - this is expected
        }

        // Save snapshots
        if snapshot_idx < SNAPSHOT_STEPS.len() && step == SNAPSHOT_STEPS[snapshot_idx] {
            let snap_path = format!("{}/snapshots/snap_{:06}.bin", output_dir, step);
            println!("  → Saving snapshot: {}", snap_path);
            save_snapshot(&sim, &snap_path);
            snapshot_idx += 1;
        }
    }

    let total_time = run_start.elapsed().as_secs_f64();
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Simulation complete!");
    println!("  Total time: {:.1} hours", total_time / 3600.0);
    println!("  Average: {:.1} sec/step", total_time / TOTAL_STEPS as f64);
    println!("═══════════════════════════════════════════════════════════════");

    // Final state
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let seg_final = sim.segregation().unwrap_or(0.0);
    println!();
    println!("Final state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  KE_final = {:.4e}", ke_final);
    println!("  Seg_final = {:.4}", seg_final);
    println!();

    // Save final summary
    let summary = format!(r#"{{
  "total_time_hours": {:.2},
  "avg_step_time_sec": {:.2},
  "KE_initial": {:.6e},
  "KE_final": {:.6e},
  "segregation_initial": {:.4},
  "segregation_final": {:.4},
  "snapshots_saved": {},
  "alpha_IC": {}
}}"#,
        total_time / 3600.0,
        total_time / TOTAL_STEPS as f64,
        ke0, ke_final,
        seg0, seg_final,
        snapshot_idx, ALPHA_IC
    );
    fs::write(format!("{}/run_summary.json", output_dir), summary).unwrap();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  V10 SIMULATION COMPLETE                                     ║");
    println!("║  Run analysis with: python3 scripts/analysis_v10.py          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn test_gpu_capacity(ng: usize) -> Result<u128, String> {
    // Generate minimal ICs for testing
    let (pos, vel, signs) = generate_zeldovich_ics(123, ng);

    // Convert to GPU format
    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Try to create simulation
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX)
        .map_err(|e| format!("{}", e))?;

    sim.set_theta(0.7);
    sim.set_softening(EPSILON);
    sim.set_pm_k_min(K_MIN);

    // Run 3 steps to test stability
    let t0 = Instant::now();
    for _ in 0..3 {
        sim.step_treepm_gpu(0.01, R_CUT, H_HUBBLE, DTAU_PER_DT)
            .map_err(|e| format!("{}", e))?;
    }
    let elapsed = t0.elapsed().as_millis() / 3;

    Ok(elapsed)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &str) {
    let (positions, _, signs) = sim.get_particles().expect("get_particles failed");

    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    // Write header: step (as u64)
    let step: u64 = 0; // Will be parsed from filename
    writer.write_all(&step.to_le_bytes()).unwrap();

    // Write particles: x, y, z, sign as f32
    let n = positions.len() / 3;
    for i in 0..n {
        let x = positions[i * 3] as f32;
        let y = positions[i * 3 + 1] as f32;
        let z = positions[i * 3 + 2] as f32;
        let sign = signs[i] as f32;
        writer.write_all(&x.to_le_bytes()).unwrap();
        writer.write_all(&y.to_le_bytes()).unwrap();
        writer.write_all(&z.to_le_bytes()).unwrap();
        writer.write_all(&sign.to_le_bytes()).unwrap();
    }
}

fn generate_zeldovich_ics(seed: u64, ng: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let ng3 = ng * ng * ng;
    let cell_size = L_BOX / ng as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(ng);

    let mut psi_x = vec![0.0f64; ng3];
    let mut psi_y = vec![0.0f64; ng3];
    let mut psi_z = vec![0.0f64; ng3];

    let dk = 2.0 * PI / L_BOX;
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); ng3];

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let kxi = if ix <= ng/2 { ix as i32 } else { ix as i32 - ng as i32 };
                let kyi = if iy <= ng/2 { iy as i32 } else { iy as i32 - ng as i32 };
                let kzi = if iz <= ng/2 { iz as i32 } else { iz as i32 - ng as i32 };

                let k_idx = (kxi.abs().max(kyi.abs()).max(kzi.abs())) as usize;
                let kx = kxi as f64 * dk;
                let ky = kyi as f64 * dk;
                let kz = kzi as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();

                let idx = ix + ng * (iy + ng * iz);

                if k_idx < K_MIN || k < 1e-10 {
                    delta_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }

                let pk = k.powf(PK_INDEX) * (-((k / K_CUT).powi(2))).exp();
                let amplitude = (pk.max(0.0)).sqrt() * AMPLITUDE;

                let re: f64 = normal.sample(&mut rng);
                let im: f64 = normal.sample(&mut rng);
                delta_k[idx] = Complex::new(re * amplitude, im * amplitude);
            }
        }
    }

    let mut psi_x_k = vec![Complex::new(0.0, 0.0); ng3];
    let mut psi_y_k = vec![Complex::new(0.0, 0.0); ng3];
    let mut psi_z_k = vec![Complex::new(0.0, 0.0); ng3];

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let kxi = if ix <= ng/2 { ix as i32 } else { ix as i32 - ng as i32 };
                let kyi = if iy <= ng/2 { iy as i32 } else { iy as i32 - ng as i32 };
                let kzi = if iz <= ng/2 { iz as i32 } else { iz as i32 - ng as i32 };

                let kx = kxi as f64 * dk;
                let ky = kyi as f64 * dk;
                let kz = kzi as f64 * dk;
                let k2 = kx*kx + ky*ky + kz*kz;

                let idx = ix + ng * (iy + ng * iz);
                if k2 > 1e-10 {
                    let factor = Complex::new(0.0, -1.0) / k2;
                    psi_x_k[idx] = factor * kx * delta_k[idx];
                    psi_y_k[idx] = factor * ky * delta_k[idx];
                    psi_z_k[idx] = factor * kz * delta_k[idx];
                }
            }
        }
    }

    fn ifft_3d(data: &mut Vec<Complex<f64>>, ng: usize, ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>) {
        for iy in 0..ng {
            for ix in 0..ng {
                let mut row: Vec<Complex<f64>> = (0..ng).map(|iz| data[ix + ng * (iy + ng * iz)]).collect();
                ifft.process(&mut row);
                for iz in 0..ng { data[ix + ng * (iy + ng * iz)] = row[iz]; }
            }
        }
        for iz in 0..ng {
            for ix in 0..ng {
                let mut row: Vec<Complex<f64>> = (0..ng).map(|iy| data[ix + ng * (iy + ng * iz)]).collect();
                ifft.process(&mut row);
                for iy in 0..ng { data[ix + ng * (iy + ng * iz)] = row[iy]; }
            }
        }
        for iz in 0..ng {
            for iy in 0..ng {
                let mut row: Vec<Complex<f64>> = (0..ng).map(|ix| data[ix + ng * (iy + ng * iz)]).collect();
                ifft.process(&mut row);
                for ix in 0..ng { data[ix + ng * (iy + ng * iz)] = row[ix]; }
            }
        }
    }

    ifft_3d(&mut psi_x_k, ng, &ifft);
    ifft_3d(&mut psi_y_k, ng, &ifft);
    ifft_3d(&mut psi_z_k, ng, &ifft);

    let norm = 1.0 / (ng3 as f64);
    for i in 0..ng3 {
        psi_x[i] = psi_x_k[i].re * norm;
        psi_y[i] = psi_y_k[i].re * norm;
        psi_z[i] = psi_z_k[i].re * norm;
    }

    let mut positions = Vec::with_capacity(ng3 * 3);
    let mut velocities = Vec::with_capacity(ng3 * 3);
    let mut signs = Vec::with_capacity(ng3);

    let n_plus = (ng3 as f64 / (1.0 + ETA)) as usize;
    let mut sign_vec: Vec<i32> = (0..ng3).map(|i| if i < n_plus { 1 } else { -1 }).collect();
    sign_vec.shuffle(&mut rng);

    let d_plus = 1.0 / (1.0 + Z_INIT);
    let f_growth = 1.0;
    let h_factor = 100.0 * (1.0 + Z_INIT).powf(1.5);

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let idx = ix + ng * (iy + ng * iz);

                let q_x = (ix as f64 + 0.5) * cell_size;
                let q_y = (iy as f64 + 0.5) * cell_size;
                let q_z = (iz as f64 + 0.5) * cell_size;

                let dx = psi_x[idx] * d_plus * ALPHA_IC;
                let dy = psi_y[idx] * d_plus * ALPHA_IC;
                let dz = psi_z[idx] * d_plus * ALPHA_IC;

                let x = (q_x + dx).rem_euclid(L_BOX);
                let y = (q_y + dy).rem_euclid(L_BOX);
                let z = (q_z + dz).rem_euclid(L_BOX);

                let v_scale = h_factor * f_growth * d_plus * 0.001 * ALPHA_IC;
                let vx = psi_x[idx] * v_scale;
                let vy = psi_y[idx] * v_scale;
                let vz = psi_z[idx] * v_scale;

                positions.push(x);
                positions.push(y);
                positions.push(z);
                velocities.push(vx);
                velocities.push(vy);
                velocities.push(vz);
                signs.push(sign_vec[idx]);
            }
        }
    }

    (positions, velocities, signs)
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires features: cuda, cufft");
    eprintln!("Run with: cargo run --release --features cuda,cufft --bin janus_v10_highres");
}
