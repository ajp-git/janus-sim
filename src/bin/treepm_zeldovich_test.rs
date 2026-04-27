//! TreePM + Zel'dovich ICs validation test
//!
//! ÉTAPE 1: 100K particles with Zel'dovich ICs + virialized velocities
//!
//! Key elements:
//! - Positions: grid + sinusoidal Zel'dovich displacement (cosmological perturbations)
//! - Velocities: random, scaled by virial_factor = 0.8 (validated on 100K)
//!
//! Criterion: Segregation onset between z=3.0 and z=2.0

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_GRID: usize = 46;  // 46³ ≈ 97K particles
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 2000;

// Zel'dovich parameters (proven values from run_8m_zeldovich.rs)
const ZELDOVICH_AMPLITUDE: f64 = 1e-3;
const ZELDOVICH_LAMBDA: f64 = 100.0;  // wavelength in Mpc
const ZELDOVICH_SIGMA: f64 = 0.1;     // amplitude std dev
const VIRIAL_FACTOR: f64 = 0.8;       // validated on 100K (KE/KE₀ = 8.8)

/// Generate Zel'dovich ICs with virialized velocities
/// - Positions: grid + sinusoidal Zel'dovich displacement
/// - Velocities: random, scaled by virial_velocity = sqrt(N/box) × virial_factor
fn generate_zeldovich_ics(n_total: usize, box_size: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_per_axis = (n_total as f64).powf(1.0/3.0).ceil() as usize;
    let cell_size = box_size / n_per_axis as f64;
    let box_half = box_size / 2.0;
    let normal = Normal::new(0.0, ZELDOVICH_SIGMA).unwrap();
    let k = 2.0 * PI / ZELDOVICH_LAMBDA;

    // Virialized velocity scale (same formula as GpuNBodyTwoPass::new)
    let virial_velocity = ((n_total as f64) / box_size).sqrt() * VIRIAL_FACTOR;

    println!("Generating Zel'dovich ICs (sinusoidal + virialized)...");
    println!("  Grid: {}³ = {} cells", n_per_axis, n_per_axis.pow(3));
    println!("  Box: {:.1} Mpc, cell_size = {:.3} Mpc", box_size, cell_size);
    println!("  Zel'dovich: A = {:.0e}, λ = {:.1} Mpc, σ = {:.3}",
        ZELDOVICH_AMPLITUDE, ZELDOVICH_LAMBDA, ZELDOVICH_SIGMA);
    println!("  Virialization: virial_velocity = {:.4} (factor = {:.2})",
        virial_velocity, VIRIAL_FACTOR);

    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n_total);

    let n_positive = (n_total as f64 / (1.0 + ETA)) as usize;

    let mut count = 0;
    let mut max_disp = 0.0f64;

    'outer: for ix in 0..n_per_axis {
        for iy in 0..n_per_axis {
            for iz in 0..n_per_axis {
                if count >= n_total { break 'outer; }

                // Grid position (centered box)
                let x0 = (ix as f64 + 0.5) * cell_size - box_half;
                let y0 = (iy as f64 + 0.5) * cell_size - box_half;
                let z0 = (iz as f64 + 0.5) * cell_size - box_half;

                // Random phase and amplitude for each particle
                let phase_x = rng.random::<f64>() * 2.0 * PI;
                let phase_y = rng.random::<f64>() * 2.0 * PI;
                let phase_z = rng.random::<f64>() * 2.0 * PI;
                let amp: f64 = normal.sample(&mut rng);

                // Zel'dovich displacement (sinusoidal)
                let dx = ZELDOVICH_AMPLITUDE * amp * (k * x0 + phase_x).sin();
                let dy = ZELDOVICH_AMPLITUDE * amp * (k * y0 + phase_y).sin();
                let dz = ZELDOVICH_AMPLITUDE * amp * (k * z0 + phase_z).sin();

                let d = (dx*dx + dy*dy + dz*dz).sqrt();
                if d > max_disp { max_disp = d; }

                // Apply displacement
                let mut x = x0 + dx;
                let mut y = y0 + dy;
                let mut z = z0 + dz;

                // Periodic wrap
                if x > box_half { x -= box_size; }
                if x < -box_half { x += box_size; }
                if y > box_half { y -= box_size; }
                if y < -box_half { y += box_size; }
                if z > box_half { z -= box_size; }
                if z < -box_half { z += box_size; }

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);

                // Virialized velocities (random direction, scaled magnitude)
                let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
                let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
                let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
                velocities.push(vx as f32);
                velocities.push(vy as f32);
                velocities.push(vz as f32);

                // Sign assignment based on eta
                let sign = if count < n_positive { 1i8 } else { -1i8 };
                signs.push(sign);
                count += 1;
            }
        }
    }

    // Shuffle signs for random spatial distribution
    signs.shuffle(&mut rng);

    let actual_pos = signs.iter().filter(|&&s| s > 0).count();
    let actual_neg = signs.iter().filter(|&&s| s < 0).count();
    println!("  Max displacement: {:.6} Mpc", max_disp);
    println!("  Generated: N+ = {}, N- = {}", actual_pos, actual_neg);

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   TreePM + Zel'dovich ICs Validation (100K, cold start)        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_total = N_GRID * N_GRID * N_GRID;
    let box_size = 100.0 * (n_total as f64 / 100_000.0).powf(1.0/3.0);
    let r_cut = box_size / 16.0;

    println!("Parameters:");
    println!("  N = {} ({}³ grid)", n_total, N_GRID);
    println!("  η = {}", ETA);
    println!("  θ = {}", THETA);
    println!("  box = {:.2} Mpc", box_size);
    println!("  r_cut = {:.2} Mpc", r_cut);
    println!("  steps = {}", TOTAL_STEPS);
    println!("  ICs = Zel'dovich + virialized (virial_factor = {})", VIRIAL_FACTOR);
    println!();

    // Generate ICs
    let t0 = Instant::now();
    let (positions, velocities, signs) = generate_zeldovich_ics(n_total, box_size, 42);
    println!("  IC generation: {:.2}s\n", t0.elapsed().as_secs_f64());

    // Cosmology setup
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = dtau_cosmo / DT;

    println!("Cosmology:");
    println!("  z_init = {:.2}", Z_INIT);
    println!("  dτ/dt = {:.6}\n", dtau_per_dt);

    // Output directory
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S");
    let output_dir = format!("/app/output/treepm_zeldovich_{}", timestamp);
    fs::create_dir_all(&output_dir)?;
    println!("Output: {}\n", output_dir);

    // Create simulation with custom ICs
    println!("Creating simulation with Zel'dovich ICs...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(positions, velocities, signs, box_size)?;
    sim.set_theta(THETA);

    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation()?;
    println!("  KE₀ = {:.4e}", ke_init);
    println!("  S₀ = {:.6}\n", seg_init);

    // Time series file
    let ts_path = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_path)?);
    writeln!(ts_file, "step,z,ke,ke_ratio,segregation,ms_per_step")?;
    writeln!(ts_file, "0,{:.4},{:.6e},1.0,{:.6},0", Z_INIT, ke_init, seg_init)?;

    println!("Running {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE_ref    Seg      ms/step");
    println!("-------------------------------------------------");

    let mut tau = cosmo.tau_start;
    let mut ke_ref: Option<f64> = None;
    let mut seg_max = 0.0f64;
    let mut step_at_seg_max = 0;
    let mut z_at_seg_max = Z_INIT;
    let mut onset_step: Option<usize> = None;
    let mut onset_z: Option<f64> = None;

    let start = Instant::now();

    for step in 1..=TOTAL_STEPS {
        let t_step = Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        // TreePM step with Morton ordering
        sim.step_treepm_gpu(DT, r_cut, h, dtau_per_dt)?;
        tau += dtau_per_dt * DT;

        let step_ms = t_step.elapsed().as_millis();

        // Measure every 10 steps
        if step % 10 == 0 || step <= 10 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;

            // Set KE reference at step 1 (after first kick)
            if ke_ref.is_none() && ke > 1e-10 {
                ke_ref = Some(ke);
                println!(">>> KE_ref set to {:.4e} at step {}", ke, step);
            }

            let ke_ratio = ke_ref.map(|r| ke / r).unwrap_or(1.0);

            // Track max segregation
            if seg > seg_max {
                seg_max = seg;
                step_at_seg_max = step;
                z_at_seg_max = z;
            }

            // Detect onset (S > 0.05)
            if onset_step.is_none() && seg > 0.05 {
                onset_step = Some(step);
                onset_z = Some(z);
                println!(">>> ONSET: S > 0.05 at step {} (z = {:.2})", step, z);
            }

            // Log to CSV
            writeln!(ts_file, "{},{:.4},{:.6e},{:.2},{:.6},{}",
                     step, z, ke, ke_ratio, seg, step_ms)?;
            ts_file.flush()?;

            // Print progress
            if step <= 10 || step % 100 == 0 {
                println!("{:5}  {:.2}    {:8.2}  {:.4}     {:4}",
                         step, z, ke_ratio, seg, step_ms);
            }

            // Safety: stop if KE explodes
            if ke_ratio > 1000.0 {
                println!("\n⚠️ KE/KE_ref > 1000 — stopping\n");
                break;
            }
        }
    }

    let runtime_min = start.elapsed().as_secs_f64() / 60.0;
    let ms_per_step = start.elapsed().as_millis() as f64 / TOTAL_STEPS as f64;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║   VALIDATION RESULTS                                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Runtime: {:.1} minutes ({:.0} ms/step)", runtime_min, ms_per_step);
    println!();
    println!("Results:");
    println!("  S_max = {:.4} at step {} (z = {:.2})", seg_max, step_at_seg_max, z_at_seg_max);

    if let (Some(step), Some(z)) = (onset_step, onset_z) {
        println!("  Onset: step {} (z = {:.2})", step, z);

        // Validation criterion: onset z ∈ [2.0, 3.0]
        if z >= 2.0 && z <= 3.0 {
            println!("\n✅ VALIDATION PASSED — onset z = {:.2} ∈ [2.0, 3.0]", z);
            println!("   Ready to integrate into 85M binary");
        } else if z > 3.0 {
            println!("\n⚠️ WARNING — onset too early (z = {:.2} > 3.0)", z);
            println!("   May indicate insufficient perturbation amplitude");
        } else {
            println!("\n⚠️ WARNING — onset too late (z = {:.2} < 2.0)", z);
            println!("   May indicate excessive perturbation amplitude");
        }
    } else {
        println!("  Onset: NOT DETECTED (S never exceeded 0.05)");
        println!("\n❌ VALIDATION FAILED — Insufficient segregation");
        println!("   Try: increase ZELDOVICH_AMPLITUDE or ZELDOVICH_SIGMA");
    }

    println!("\nCSV: {}", ts_path);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires 'cuda' and 'cufft' features:");
    eprintln!("  cargo run --release --features cuda,cufft --bin treepm_zeldovich_test");
}
