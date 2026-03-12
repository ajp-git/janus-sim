//! JANUS V9 — GPU MAX PARTICLE SEARCH
//!
//! Phase 1: Find maximum stable particle count
//! Phase 2: Run scientific simulation with N_max
//!
//! Usage: cargo run --release --features "cuda cufft" --bin v9_gpu_search

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 1: GPU LIMIT DETECTION
// ═══════════════════════════════════════════════════════════════════════════

const TEST_STEPS: usize = 50;
const PARTICLE_COUNTS: [usize; 13] = [
    2_000_000,
    4_000_000,
    6_000_000,
    8_000_000,
    10_000_000,
    12_000_000,
    14_000_000,
    16_000_000,
    18_000_000,
    20_000_000,
    24_000_000,
    28_000_000,
    32_000_000,
];

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2: SCIENCE RUN PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const L_BOX: f64 = 300.0;       // Mpc
const EPSILON: f64 = 0.18;      // Softening
const ETA: f64 = 1.06;          // Mass ratio
const HUBBLE: f64 = 0.006;      // Expansion rate
const ALPHA_IC: f64 = 1.6;      // IC asymmetry
const K_MIN: usize = 1;         // No filtering
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 10000;
const THETA: f64 = 0.7;
const Z_INIT: f64 = 5.0;

const SNAPSHOT_STEPS: [usize; 5] = [1000, 3000, 5000, 7000, 10000];

// Cosmological P(k) parameters
const K_PIVOT: f64 = 0.02;  // Mpc⁻¹
const N_S: f64 = 0.96;      // spectral index
const K_CUT: f64 = 0.3;     // high-k cutoff

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::friedmann::{JanusParams, CosmoInterpolator};

    let base_dir = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/app/output/v9_science".to_string());
    fs::create_dir_all(&base_dir)?;

    println!("════════════════════════════════════════════════════════════════");
    println!("  JANUS V9 — GPU MAX PARTICLE SEARCH + SCIENCE RUN");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    // =========================================================================
    // PHASE 1: GPU LIMIT DETECTION
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 1: GPU LIMIT DETECTION                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut n_max: usize = 0;
    let mut last_success: usize = 0;

    for &n in &PARTICLE_COUNTS {
        println!("Testing N = {} ({:.1}M)...", n, n as f64 / 1e6);

        let result = test_particle_count(n, TEST_STEPS);

        match result {
            Ok(timing) => {
                println!("  ✓ SUCCESS: {:.0} ms/step", timing);
                last_success = n;
                n_max = n;
            }
            Err(e) => {
                println!("  ✗ FAILED: {}", e);
                println!("  Maximum stable: N = {} ({:.1}M)", last_success, last_success as f64 / 1e6);
                break;
            }
        }
        println!();
    }

    if n_max == 0 {
        eprintln!("ERROR: Could not find any stable particle count!");
        return Err("No stable N found".into());
    }

    // Save GPU limits
    let limits_file = format!("{}/gpu_limits.txt", base_dir);
    let mut f = File::create(&limits_file)?;
    writeln!(f, "N_max = {}", n_max)?;
    writeln!(f, "N_max_millions = {:.1}", n_max as f64 / 1e6)?;
    writeln!(f, "L_box = {} Mpc", L_BOX)?;
    writeln!(f, "GPU = RTX 3060 12GB")?;
    println!();
    println!("GPU limits saved to: {}", limits_file);

    // =========================================================================
    // PHASE 2: SCIENCE RUN
    // =========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PHASE 2: SCIENCE RUN                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Use 90% of max to be safe
    let n_science = (n_max as f64 * 0.9) as usize;
    // Round to nearest million
    let n_science = (n_science / 1_000_000) * 1_000_000;

    println!("Science run parameters:");
    println!("  N = {} ({:.1}M)", n_science, n_science as f64 / 1e6);
    println!("  L = {} Mpc", L_BOX);
    println!("  ε = {} Mpc", EPSILON);
    println!("  η = {}", ETA);
    println!("  H = {}", HUBBLE);
    println!("  α_IC = {}", ALPHA_IC);
    println!("  k_min = {}", K_MIN);
    println!("  Steps = {}", TOTAL_STEPS);
    println!("  Snapshots at: {:?}", SNAPSHOT_STEPS);
    println!();

    // Create output directories
    let snap_dir = format!("{}/snapshots", base_dir);
    fs::create_dir_all(&snap_dir)?;

    // Generate ICs
    println!("Generating Zel'dovich ICs (k_min={})...", K_MIN);
    let t_init = Instant::now();

    let n_grid = (n_science as f64).cbrt().round() as usize;
    let (positions, velocities, signs) = generate_cosmological_ics(
        42, // seed
        ETA,
        K_MIN,
        ALPHA_IC,
        n_grid,
        L_BOX,
    );

    // Convert to GPU format
    let pos_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize simulation
    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX)?;
    sim.set_softening(EPSILON);
    sim.set_theta(THETA);

    // Virialize (sample 10% for speed)
    println!("Virializing...");
    sim.virialize_sampled(n_grid * n_grid * 10)?;  // ~10% sample

    let t_init_done = t_init.elapsed();
    println!("Initialization: {:.1}s", t_init_done.as_secs_f64());

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = dtau_per_step / DT;

    // Initial state
    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;

    println!();
    println!("Initial state:");
    println!("  KE₀ = {:.4e}", ke0);
    println!("  Seg₀ = {:.4}", seg0);
    println!();

    // Run simulation
    println!("Running {} steps...", TOTAL_STEPS);
    let t_sim = Instant::now();
    let mut tau = cosmo.tau_start;

    // Results tracking
    let mut results: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    for step in 1..=TOTAL_STEPS {
        let (a, h) = cosmo.get_params_at_tau(tau);
        sim.step_treepm_gpu(DT, a, h, dtau_per_dt)?;
        tau += dtau_per_step;

        // Progress every 100 steps
        if step % 100 == 0 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;
            let z = 1.0 / a - 1.0;
            println!("  Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}",
                step, z, ke/ke0, seg);
            results.push((step, z, ke/ke0, seg, a));
        }

        // Save snapshots
        if SNAPSHOT_STEPS.contains(&step) {
            let snap_path = format!("{}/snap_{:06}.bin", snap_dir, step);
            save_snapshot(&sim, &snap_path, step)?;
            println!("  → Saved snapshot: {}", snap_path);
        }
    }

    let t_sim_done = t_sim.elapsed();
    println!();
    println!("Simulation complete!");
    println!("  Total time: {:.1} min", t_sim_done.as_secs_f64() / 60.0);
    println!("  Rate: {:.0} ms/step", t_sim_done.as_millis() as f64 / TOTAL_STEPS as f64);

    // Save time series
    let ts_path = format!("{}/time_series.csv", base_dir);
    let mut ts_file = File::create(&ts_path)?;
    writeln!(ts_file, "step,z,ke_ratio,seg,a")?;
    for (step, z, ke_ratio, seg, a) in &results {
        writeln!(ts_file, "{},{:.4},{:.6},{:.6},{:.6}", step, z, ke_ratio, seg, a)?;
    }
    println!("Time series saved to: {}", ts_path);

    // Save run info
    let info_path = format!("{}/run_info.json", base_dir);
    let mut info_file = File::create(&info_path)?;
    writeln!(info_file, "{{")?;
    writeln!(info_file, "  \"n_particles\": {},", n_science)?;
    writeln!(info_file, "  \"box_size_mpc\": {},", L_BOX)?;
    writeln!(info_file, "  \"epsilon\": {},", EPSILON)?;
    writeln!(info_file, "  \"eta\": {},", ETA)?;
    writeln!(info_file, "  \"hubble\": {},", HUBBLE)?;
    writeln!(info_file, "  \"alpha_ic\": {},", ALPHA_IC)?;
    writeln!(info_file, "  \"k_min\": {},", K_MIN)?;
    writeln!(info_file, "  \"total_steps\": {},", TOTAL_STEPS)?;
    writeln!(info_file, "  \"theta\": {},", THETA)?;
    writeln!(info_file, "  \"z_init\": {},", Z_INIT)?;
    writeln!(info_file, "  \"runtime_min\": {:.2}", t_sim_done.as_secs_f64() / 60.0)?;
    writeln!(info_file, "}}")?;
    println!("Run info saved to: {}", info_path);

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  V9 COMPLETE");
    println!("════════════════════════════════════════════════════════════════");
    println!("Output directory: {}", base_dir);
    println!("Snapshots: {:?}", SNAPSHOT_STEPS);

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn test_particle_count(n: usize, steps: usize) -> Result<f64, String> {
    use janus::friedmann::{JanusParams, CosmoInterpolator};

    // Calculate grid size
    let n_grid = (n as f64).cbrt().round() as usize;
    let actual_n = n_grid * n_grid * n_grid;

    // Generate simple ICs for test
    let (positions, velocities, signs) = generate_cosmological_ics(
        12345, // seed
        1.045, // eta
        2,     // k_min
        1.5,   // alpha_ic
        n_grid,
        300.0, // box size
    );

    let pos_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Try to create simulation
    let mut sim = match GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, 300.0) {
        Ok(s) => s,
        Err(e) => return Err(format!("Init failed: {}", e)),
    };

    sim.set_softening(0.2);
    sim.set_theta(0.7);

    // Quick virialize (small sample)
    if let Err(e) = sim.virialize_sampled(n_grid * 10) {
        return Err(format!("Virialize failed: {}", e));
    }

    // Cosmology
    let params = JanusParams::from_eta(1.045);
    let cosmo = CosmoInterpolator::new(&params, 5.0);
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / 10000.0;
    let dtau_per_dt = dtau_per_step / 0.01;

    // Run test steps
    let t_start = Instant::now();
    let mut tau = cosmo.tau_start;

    for _ in 0..steps {
        let (a, h) = cosmo.get_params_at_tau(tau);
        if let Err(e) = sim.step_treepm_gpu(0.01, a, h, dtau_per_dt) {
            return Err(format!("Step failed: {}", e));
        }
        tau += dtau_per_step;
    }

    let elapsed = t_start.elapsed();
    let ms_per_step = elapsed.as_millis() as f64 / steps as f64;

    Ok(ms_per_step)
}

// ═══════════════════════════════════════════════════════════════════════════
// IC GENERATION
// ═══════════════════════════════════════════════════════════════════════════

fn generate_cosmological_ics(
    seed: u64,
    eta: f64,
    k_min: usize,
    alpha_ic: f64,
    n_grid: usize,
    box_size: f64,
) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    use rand::prelude::*;
    use rustfft::{FftPlanner, num_complex::Complex};
    use std::f64::consts::PI;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let ng = n_grid;
    let ng3 = ng * ng * ng;
    let cell = box_size / ng as f64;

    // Assign signs based on eta
    let n_positive = (ng3 as f64 / (1.0 + eta)) as usize;
    let mut signs: Vec<i32> = vec![1; n_positive];
    signs.extend(vec![-1; ng3 - n_positive]);
    signs.shuffle(&mut rng);

    // Generate P(k) field
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(ng);

    let mut phi_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); ng3];

    for kx in 0..ng {
        for ky in 0..ng {
            for kz in 0..ng {
                let ikx = if kx <= ng/2 { kx as i32 } else { kx as i32 - ng as i32 };
                let iky = if ky <= ng/2 { ky as i32 } else { ky as i32 - ng as i32 };
                let ikz = if kz <= ng/2 { kz as i32 } else { kz as i32 - ng as i32 };

                let k_idx = (ikx.abs() as usize).max(iky.abs() as usize).max(ikz.abs() as usize);
                if k_idx < k_min { continue; }

                let k_phys = 2.0 * PI / box_size * ((ikx*ikx + iky*iky + ikz*ikz) as f64).sqrt();
                if k_phys > K_CUT || k_phys < 1e-10 { continue; }

                // Cosmological P(k) ~ k^0.96 / (1 + (k/k_pivot)^4)
                let k_ratio = k_phys / K_PIVOT;
                let pk = k_phys.powf(N_S) / (1.0 + k_ratio.powi(4));
                let amp = pk.sqrt() * 0.05;

                let phase: f64 = rng.random::<f64>() * 2.0 * PI;
                let idx = kx + ng * (ky + ng * kz);
                phi_k[idx] = Complex::new(amp * phase.cos(), amp * phase.sin());
            }
        }
    }

    // 3D IFFT
    let mut phi_x = phi_k.clone();
    for iz in 0..ng {
        for iy in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|ix| phi_x[ix + ng*(iy + ng*iz)]).collect();
            ifft.process(&mut row);
            for ix in 0..ng {
                phi_x[ix + ng*(iy + ng*iz)] = row[ix];
            }
        }
    }
    for iz in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iy| phi_x[ix + ng*(iy + ng*iz)]).collect();
            ifft.process(&mut row);
            for iy in 0..ng {
                phi_x[ix + ng*(iy + ng*iz)] = row[iy];
            }
        }
    }
    for iy in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iz| phi_x[ix + ng*(iy + ng*iz)]).collect();
            ifft.process(&mut row);
            for iz in 0..ng {
                phi_x[ix + ng*(iy + ng*iz)] = row[iz];
            }
        }
    }

    let phi_real: Vec<f64> = phi_x.iter().map(|c| c.re / ng3 as f64).collect();

    let mut positions = vec![0.0; ng3 * 3];
    let mut velocities = vec![0.0; ng3 * 3];

    let growth_factor = 1.0 / (1.0 + Z_INIT);

    for ix in 0..ng {
        for iy in 0..ng {
            for iz in 0..ng {
                let idx = ix + ng * (iy + ng * iz);

                let x0 = (ix as f64 + 0.5) * cell - box_size / 2.0;
                let y0 = (iy as f64 + 0.5) * cell - box_size / 2.0;
                let z0 = (iz as f64 + 0.5) * cell - box_size / 2.0;

                let ixp = (ix + 1) % ng;
                let ixm = (ix + ng - 1) % ng;
                let iyp = (iy + 1) % ng;
                let iym = (iy + ng - 1) % ng;
                let izp = (iz + 1) % ng;
                let izm = (iz + ng - 1) % ng;

                let dphi_dx = (phi_real[ixp + ng*(iy + ng*iz)] - phi_real[ixm + ng*(iy + ng*iz)]) / (2.0 * cell);
                let dphi_dy = (phi_real[ix + ng*(iyp + ng*iz)] - phi_real[ix + ng*(iym + ng*iz)]) / (2.0 * cell);
                let dphi_dz = (phi_real[ix + ng*(iy + ng*izp)] - phi_real[ix + ng*(iy + ng*izm)]) / (2.0 * cell);

                // Asymmetric amplitude: δ- = α_IC × δ+
                let amp_factor = if signs[idx] > 0 { 1.0 } else { alpha_ic };

                let disp_x = -dphi_dx * growth_factor * box_size * amp_factor;
                let disp_y = -dphi_dy * growth_factor * box_size * amp_factor;
                let disp_z = -dphi_dz * growth_factor * box_size * amp_factor;

                positions[3*idx]     = wrap(x0 + disp_x, box_size);
                positions[3*idx + 1] = wrap(y0 + disp_y, box_size);
                positions[3*idx + 2] = wrap(z0 + disp_z, box_size);

                let vel_scale = 0.1;
                velocities[3*idx]     = disp_x * vel_scale;
                velocities[3*idx + 1] = disp_y * vel_scale;
                velocities[3*idx + 2] = disp_z * vel_scale;
            }
        }
    }

    (positions, velocities, signs)
}

fn wrap(x: f64, l: f64) -> f64 {
    let half = l / 2.0;
    if x > half { x - l }
    else if x < -half { x + l }
    else { x }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &str, step: usize) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufWriter;

    let (positions, _, signs) = sim.get_particles()?;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header: step number
    writer.write_all(&(step as u64).to_le_bytes())?;

    let n = positions.len() / 3;
    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        let sign = signs[i] as f32;
        writer.write_all(&x.to_le_bytes())?;
        writer.write_all(&y.to_le_bytes())?;
        writer.write_all(&z.to_le_bytes())?;
        writer.write_all(&sign.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires cuda and cufft features");
    eprintln!("Run with: cargo run --release --features \"cuda cufft\" --bin v9_gpu_search");
}
