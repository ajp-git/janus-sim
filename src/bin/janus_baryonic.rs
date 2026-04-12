//! Janus N-body avec Physique Baryonique
//!
//! Intègre refroidissement radiatif et pression thermique
//! pour permettre la formation de structures dans le cadre Janus.
//!
//! IMPORTANT: Vitesses initiales = thermiques (pas gravitationnelles)
//!   v_thermal = sqrt(k_B × T / (μ × m_p)) ≈ 12 km/s à T=10,000 K
//!
//! Usage:
//!   cargo run --release --features cuda --bin janus_baryonic -- \
//!     --n 5000000 --box 500 --steps 4000 --t-init 10000
//!
//! Resume from snapshot:
//!   cargo run --release --features cuda --bin janus_baryonic -- \
//!     --resume-from /path/to/snap_00900.bin --start-step 900 \
//!     --snapshot-every 10 --alert-threshold 50

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::sph_pressure_gpu::GpuSphPressure;
use janus::baryonic::cooling::apply_cooling;
use janus::baryonic::pressure::{K_B_OVER_MP_CODE, MU_MOL};
use std::fs::{self, File};
use std::io::{Write, BufWriter, Read as IoRead, BufReader};
use std::time::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

// Grid for density computation
const N_CELLS: usize = 64;

fn main() {
    #[cfg(feature = "cuda")]
    {
        run_baryonic_simulation();
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: This binary requires --features cuda");
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn run_baryonic_simulation() {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let n_particles: usize = args.iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);

    let box_size: f64 = args.iter()
        .position(|a| a == "--box")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(500.0);

    let steps: usize = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);

    let dt: f64 = args.iter()
        .position(|a| a == "--dt")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.005);

    let t_init: f64 = args.iter()
        .position(|a| a == "--t-init")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e4);

    let eta: f64 = args.iter()
        .position(|a| a == "--eta")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.045);

    let mu: f64 = args.iter()
        .position(|a| a == "--mu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(19.0);

    let snapshot_every: usize = args.iter()
        .position(|a| a == "--snapshot-every")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let output_dir: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone())
        .unwrap_or_else(|| "/app/output/janus_baryonic".to_string());

    // Resume from snapshot
    let resume_from: Option<String> = args.iter()
        .position(|a| a == "--resume-from")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone());

    let start_step: usize = args.iter()
        .position(|a| a == "--start-step")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Alert threshold for ρ+_max
    let alert_threshold: f64 = args.iter()
        .position(|a| a == "--alert-threshold")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);

    // High-frequency snapshots after this step
    let snapshot_every_after: usize = args.iter()
        .position(|a| a == "--snapshot-every-after")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Snapshot frequency after threshold step
    let snapshot_freq_after: usize = args.iter()
        .position(|a| a == "--snapshot-freq-after")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Initial density perturbation amplitude (0-100%)
    let delta_init: f64 = args.iter()
        .position(|a| a == "--delta-init")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    // Compute particle counts
    let n_positive = (n_particles as f64 / (1.0 + eta)) as usize;
    let n_negative = n_particles - n_positive;

    // Create output directories
    fs::create_dir_all(format!("{}/snapshots", output_dir)).expect("Failed to create snapshots dir");

    // Save config
    let config = format!(r#"{{
  "n_particles": {},
  "n_positive": {},
  "n_negative": {},
  "eta": {:.4},
  "mu": {:.1},
  "box_size": {:.1},
  "steps": {},
  "dt": {},
  "t_init": {:.0},
  "snapshot_every": {}
}}"#, n_particles, n_positive, n_negative, eta, mu, box_size, steps, dt, t_init, snapshot_every);
    fs::write(format!("{}/config.json", output_dir), &config).expect("Failed to write config");

    println!("================================================================");
    println!("  JANUS N-BODY + PHYSIQUE BARYONIQUE");
    println!("================================================================");
    println!("  N = {} ({} m+ / {} m-)", n_particles, n_positive, n_negative);
    println!("  eta = {:.4}, mu = {:.1}", eta, mu);
    println!("  Box = {} Mpc", box_size);
    println!("  Steps = {}, dt = {} Gyr", steps, dt);
    println!("  T_init = {:.0} K", t_init);
    println!("  Output: {}", output_dir);
    if resume_from.is_some() {
        println!("  RESUME from step {}", start_step);
    }
    println!("  Snapshot every {} steps", snapshot_every);
    println!("  Alert threshold: ρ+_max > {:.0}", alert_threshold);
    println!("================================================================\n");

    let init_start = Instant::now();

    // Either load from snapshot or generate fresh ICs
    let (positions, velocities, signs, mut temperatures) = if let Some(ref snapshot_path) = resume_from {
        println!("Loading from snapshot...");
        match load_snapshot(snapshot_path) {
            Ok((pos, vel, sgn, temps)) => {
                let n = pos.len() / 3;
                println!("  ✓ Loaded {} particles", n);
                println!("  T_mean = {:.0} K", temps.iter().sum::<f64>() / n as f64);
                (pos, vel, sgn, temps)
            }
            Err(e) => {
                eprintln!("Failed to load snapshot: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Initialize with THERMAL velocities (not gravitational)
        println!("Initializing with thermal velocities...");

        // Thermal velocity dispersion: sigma_v = sqrt(k_B × T / (μ × m_p))
        // K_B_OVER_MP_CODE = 8.7e-9 (Mpc/Gyr)²/K
        let sigma_v = (K_B_OVER_MP_CODE * t_init / MU_MOL).sqrt();
        let v_rms_expected = sigma_v * 3.0_f64.sqrt();
        println!("  T_init = {:.0} K", t_init);
        println!("  sigma_v = {:.6} Mpc/Gyr = {:.2} km/s", sigma_v, sigma_v * 978.0);
        println!("  v_rms attendu = {:.6} Mpc/Gyr = {:.2} km/s", v_rms_expected, v_rms_expected * 978.0);

        // Generate initial conditions
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, sigma_v).unwrap();
        let half_box = box_size / 2.0;

        // Positions: uniform random
        // Velocities: Maxwell-Boltzmann thermal
        // Signs: first n_positive are +1, rest are -1
        let mut positions: Vec<f64> = Vec::with_capacity(n_particles * 3);
        let mut velocities: Vec<f64> = Vec::with_capacity(n_particles * 3);
        let mut signs: Vec<i32> = Vec::with_capacity(n_particles);

        // Perturbation wavelength (box/4 creates ~8 overdense regions)
        let k_pert = 2.0 * std::f64::consts::PI * 4.0 / box_size;
        let delta_amp = delta_init / 100.0;  // Convert percentage to fraction
        println!("  δ_init = {:.2}% (amplitude {:.4})", delta_init, delta_amp);

        for i in 0..n_particles {
            // Position: uniform in box with density perturbations
            use rand::Rng;
            let mut x = rng.gen::<f64>() * box_size - half_box;
            let mut y = rng.gen::<f64>() * box_size - half_box;
            let mut z = rng.gen::<f64>() * box_size - half_box;

            // Apply sinusoidal displacement to create density perturbations
            // Displacement ∝ -δ × sin(kx) creates density contrast ∝ δ × cos(kx)
            if delta_amp > 0.0 {
                let disp_scale = delta_amp * box_size / (4.0 * k_pert);
                x += disp_scale * (k_pert * x).sin();
                y += disp_scale * (k_pert * y).sin();
                z += disp_scale * (k_pert * z).sin();
                // Re-wrap to box
                if x > half_box { x -= box_size; }
                if x < -half_box { x += box_size; }
                if y > half_box { y -= box_size; }
                if y < -half_box { y += box_size; }
                if z > half_box { z -= box_size; }
                if z < -half_box { z += box_size; }
            }

            positions.push(x);
            positions.push(y);
            positions.push(z);

            // Velocity: Maxwell-Boltzmann thermal distribution
            let vx = normal.sample(&mut rng);
            let vy = normal.sample(&mut rng);
            let vz = normal.sample(&mut rng);
            velocities.push(vx);
            velocities.push(vy);
            velocities.push(vz);

            // Sign: +1 for first n_positive, -1 for rest
            signs.push(if i < n_positive { 1 } else { -1 });
        }

        // Verify v_rms
        let v2_sum: f64 = velocities.chunks(3)
            .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            .sum();
        let v_rms_init = (v2_sum / n_particles as f64).sqrt();
        let v_rms_kms_init = v_rms_init * 978.0;
        println!("  v_rms mesuré = {:.6} Mpc/Gyr = {:.2} km/s", v_rms_init, v_rms_kms_init);

        // VALIDATION: v_rms must be < 100 km/s
        if v_rms_kms_init > 100.0 {
            eprintln!("\n*** ERREUR: v_rms = {:.1} km/s > 100 km/s ***", v_rms_kms_init);
            eprintln!("    Bug dans l'initialisation des vitesses thermiques!");
            std::process::exit(1);
        }
        println!("  ✓ Validation: v_rms < 100 km/s\n");

        let temps = vec![t_init; n_particles];
        (positions, velocities, signs, temps)
    };

    // Recompute particle counts from actual data
    let n_particles = signs.len();
    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    let n_negative = n_particles - n_positive;

    let mut gpu_sim = match GpuNBodySimulation::new_with_state(
        n_positive, n_negative, box_size,
        positions, velocities, signs
    ) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("Failed to initialize GPU simulation: {}", e);
            std::process::exit(1);
        }
    };

    println!("GPU init: {:.2?}", init_start.elapsed());

    // Initialize SPH pressure calculator for m+ particles only
    // (m- particles don't have baryonic physics in this simplified model)
    let mass_plus = 1e10;  // M_sun per particle (approximate)
    let device = cudarc::driver::CudaDevice::new(0).expect("Failed to get CUDA device");
    let mut sph_pressure = GpuSphPressure::new(device, n_positive, mass_plus, box_size)
        .expect("Failed to create SPH pressure calculator");
    println!("SPH pressure GPU initialized for {} particles", n_positive);

    // Evolution CSV (append mode if resuming)
    let csv_path = format!("{}/evolution.csv", output_dir);
    let csv_file = if start_step > 0 {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)
            .expect("Failed to open evolution.csv for append")
    } else {
        File::create(&csv_path).expect("Failed to create evolution.csv")
    };
    let mut csv = BufWriter::new(csv_file);
    if start_step == 0 {
        writeln!(csv, "step,time_gyr,rho_plus_max,rho_minus_max,t_mean,t_min,v_rms_kms,segregation").unwrap();
    }

    // Mean densities
    let mean_rho_plus = n_positive as f64 / box_size.powi(3);
    let mean_rho_minus = n_negative as f64 / box_size.powi(3);

    let half_box = box_size / 2.0;
    let cell_size = box_size / N_CELLS as f64;
    let cell_vol = cell_size.powi(3);

    println!("\nStarting evolution with baryonic physics...\n");
    let sim_start = Instant::now();
    let mut alert_triggered = false;

    for step in start_step..=steps {
        // Get positions from GPU (interleaved: [x0, y0, z0, x1, y1, z1, ...])
        let pos = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to get positions: {}", e);
                break;
            }
        };
        let vel = match gpu_sim.get_velocities() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to get velocities: {}", e);
                break;
            }
        };
        let signs = match gpu_sim.get_signs() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to get signs: {}", e);
                break;
            }
        };

        // Compute density on grid (separate m+ and m-)
        let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        let mut counts_minus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];

        for i in 0..n_particles {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];

            let ix = ((px + half_box) / cell_size) as usize;
            let iy = ((py + half_box) / cell_size) as usize;
            let iz = ((pz + half_box) / cell_size) as usize;

            if ix < N_CELLS && iy < N_CELLS && iz < N_CELLS {
                let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;
                if signs[i] > 0 {
                    counts_plus[idx] += 1;
                } else {
                    counts_minus[idx] += 1;
                }
            }
        }

        // Find max density ratios
        let rho_plus_max = counts_plus.iter().cloned().max().unwrap_or(0) as f64
            / (mean_rho_plus * cell_vol);
        let rho_minus_max = counts_minus.iter().cloned().max().unwrap_or(0) as f64
            / (mean_rho_minus * cell_vol);

        // Temperature stats
        let t_mean: f64 = temperatures.iter().sum::<f64>() / n_particles as f64;
        let t_min: f64 = temperatures.iter().cloned().fold(f64::MAX, f64::min);

        // Velocity RMS (km/s)
        let v_rms: f64 = (0..n_particles)
            .map(|i| {
                let vx = vel[i * 3];
                let vy = vel[i * 3 + 1];
                let vz = vel[i * 3 + 2];
                vx * vx + vy * vy + vz * vz
            })
            .sum::<f64>() / n_particles as f64;
        let v_rms_kms = v_rms.sqrt() * 978.0; // Mpc/Gyr -> km/s

        // Segregation (COM distance)
        let segregation = compute_segregation(&pos, &signs, n_positive, n_negative, box_size);

        // Write CSV (flush every write for real-time monitoring)
        writeln!(csv, "{},{:.4},{:.2},{:.2},{:.0},{:.0},{:.1},{:.4}",
            step, step as f64 * dt,
            rho_plus_max, rho_minus_max,
            t_mean, t_min, v_rms_kms, segregation
        ).unwrap();
        csv.flush().unwrap();

        // Progress output (every 40 steps for production)
        let progress_interval = if start_step > 0 { 10 } else { 40 };
        if step % progress_interval == 0 {
            let elapsed = sim_start.elapsed().as_secs_f64();
            let steps_done = step - start_step;
            let eta_sec = if steps_done > 0 { elapsed * (steps - step) as f64 / steps_done as f64 } else { 0.0 };
            println!("[Step {:5}] t={:.2}Gyr ρ+={:6.1} ρ-={:6.1} T={:.0}/{:.0}K v={:.0}km/s S={:.3} | ETA {:.0}s",
                step, step as f64 * dt, rho_plus_max, rho_minus_max, t_mean, t_min, v_rms_kms, segregation, eta_sec);
        }

        // ALERT: Check threshold
        if rho_plus_max > alert_threshold && !alert_triggered {
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║  🚨 ALERT: ρ+_max = {:.1} > {:.0} at step {}  🚨", rho_plus_max, alert_threshold, step);
            println!("║  STRUCTURE FORMATION IN PROGRESS!                            ║");
            println!("╚══════════════════════════════════════════════════════════════╝\n");
            alert_triggered = true;
        }

        // Save snapshot (with adaptive frequency)
        let current_snapshot_freq = if snapshot_every_after > 0 && step >= snapshot_every_after {
            snapshot_freq_after
        } else {
            snapshot_every
        };
        if step % current_snapshot_freq == 0 {
            save_snapshot_v2(&output_dir, step, &pos, &vel, &signs, &temperatures);
        }

        if step == steps { break; }

        // === PHYSICS STEP ===

        // 1. N-body gravity (GPU)
        if let Err(e) = gpu_sim.step(dt) {
            eprintln!("GPU step failed: {}", e);
            break;
        }

        // 2. SPH Pressure forces (GPU) - m+ particles only
        // Extract m+ positions and temperatures
        let pos_plus: Vec<f64> = (0..n_particles)
            .filter(|&i| signs[i] > 0)
            .flat_map(|i| vec![pos[i*3], pos[i*3+1], pos[i*3+2]])
            .collect();
        let temp_plus: Vec<f64> = (0..n_particles)
            .filter(|&i| signs[i] > 0)
            .map(|i| temperatures[i])
            .collect();

        // Compute pressure accelerations
        let pressure_acc = match sph_pressure.compute_pressure_accelerations(&pos_plus, &temp_plus) {
            Ok(acc) => acc,
            Err(e) => {
                eprintln!("SPH pressure failed: {}", e);
                vec![0.0; n_positive * 3]
            }
        };

        // Apply pressure kick to m+ particle velocities
        // Need to update velocities through GPU sim
        let mut vel_updated = vel.clone();
        let mut plus_idx = 0;
        for i in 0..n_particles {
            if signs[i] > 0 {
                vel_updated[i*3] += pressure_acc[plus_idx*3] * dt;
                vel_updated[i*3+1] += pressure_acc[plus_idx*3+1] * dt;
                vel_updated[i*3+2] += pressure_acc[plus_idx*3+2] * dt;
                plus_idx += 1;
            }
        }
        // Upload updated velocities back to GPU
        if let Err(e) = gpu_sim.set_velocities(&vel_updated) {
            eprintln!("Failed to set velocities: {}", e);
        }

        // 3. Apply cooling to all particles based on local overdensity
        // Get updated positions
        let pos_new = match gpu_sim.get_positions() {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Recompute density for cooling
        let mut total_counts = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
        for i in 0..n_particles {
            let px = pos_new[i * 3];
            let py = pos_new[i * 3 + 1];
            let pz = pos_new[i * 3 + 2];

            let ix = ((px + half_box) / cell_size) as usize;
            let iy = ((py + half_box) / cell_size) as usize;
            let iz = ((pz + half_box) / cell_size) as usize;

            if ix < N_CELLS && iy < N_CELLS && iz < N_CELLS {
                let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;
                total_counts[idx] += 1;
            }
        }

        let mean_count = n_particles as f64 / (N_CELLS * N_CELLS * N_CELLS) as f64;

        // Apply cooling to each particle
        for i in 0..n_particles {
            let px = pos_new[i * 3];
            let py = pos_new[i * 3 + 1];
            let pz = pos_new[i * 3 + 2];

            let ix = ((px + half_box) / cell_size) as usize;
            let iy = ((py + half_box) / cell_size) as usize;
            let iz = ((pz + half_box) / cell_size) as usize;

            if ix < N_CELLS && iy < N_CELLS && iz < N_CELLS {
                let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;
                let overdensity = total_counts[idx] as f64 / mean_count;
                // TODO: Pass actual redshift when cosmological evolution is added
                temperatures[i] = apply_cooling(temperatures[i], overdensity.max(1.0), 0.0, dt);
            }
        }
    }

    csv.flush().unwrap();

    let total_time = sim_start.elapsed();
    println!("\n================================================================");
    println!("  SIMULATION COMPLETE");
    println!("  Total time: {:.1} s ({:.2} min)",
        total_time.as_secs_f64(), total_time.as_secs_f64() / 60.0);
    println!("  Output: {}", output_dir);
    println!("================================================================");

    // Final summary
    let pos = match gpu_sim.get_positions() {
        Ok(p) => p,
        Err(_) => return,
    };
    let signs = match gpu_sim.get_signs() {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut counts_plus = vec![0u32; N_CELLS * N_CELLS * N_CELLS];
    for i in 0..n_particles {
        if signs[i] > 0 {
            let px = pos[i * 3];
            let py = pos[i * 3 + 1];
            let pz = pos[i * 3 + 2];

            let ix = ((px + half_box) / cell_size) as usize;
            let iy = ((py + half_box) / cell_size) as usize;
            let iz = ((pz + half_box) / cell_size) as usize;

            if ix < N_CELLS && iy < N_CELLS && iz < N_CELLS {
                let idx = ix * N_CELLS * N_CELLS + iy * N_CELLS + iz;
                counts_plus[idx] += 1;
            }
        }
    }

    let rho_plus_final = counts_plus.iter().cloned().max().unwrap_or(0) as f64
        / (mean_rho_plus * cell_vol);
    let t_final: f64 = temperatures.iter().sum::<f64>() / n_particles as f64;

    println!("\nFinal state:");
    println!("  ρ+_max/ρ̄+ = {:.1}", rho_plus_final);
    println!("  T_mean = {:.0} K", t_final);

    // Success criteria from the plan
    if rho_plus_final > 100.0 {
        println!("\n★★★ EFFONDREMENT CONFIRMÉ ★★★");
        println!("  ρ+_max > 100 → Formation de structures!");
    } else if rho_plus_final > 10.0 {
        println!("\n★ STRUCTURES ÉMERGENTES");
        println!("  ρ+_max > 10 → Continuer avec plus de steps");
    } else if rho_plus_final > 3.0 {
        println!("\n• SIGNAL FAIBLE");
        println!("  ρ+_max > 3 → Tester paramètres différents");
    } else {
        println!("\n• RÉSULTAT NÉGATIF");
        println!("  ρ+_max < 3 → Publier limites");
    }
}

fn compute_segregation(
    pos: &[f64], signs: &[i32],
    n_positive: usize, n_negative: usize,
    box_size: f64
) -> f64 {
    let half_box = box_size / 2.0;

    // Find reference particles
    let mut ref_plus = [0.0f64; 3];
    let mut ref_minus = [0.0f64; 3];
    let mut found_plus = false;
    let mut found_minus = false;

    for i in 0..signs.len() {
        if signs[i] > 0 && !found_plus {
            ref_plus = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
            found_plus = true;
        }
        if signs[i] < 0 && !found_minus {
            ref_minus = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
            found_minus = true;
        }
        if found_plus && found_minus { break; }
    }

    // Compute COMs with minimum image convention
    let mut com_plus = [0.0f64; 3];
    let mut com_minus = [0.0f64; 3];

    for i in 0..signs.len() {
        let px = pos[i * 3];
        let py = pos[i * 3 + 1];
        let pz = pos[i * 3 + 2];

        if signs[i] > 0 {
            let mut dx = px - ref_plus[0];
            let mut dy = py - ref_plus[1];
            let mut dz = pz - ref_plus[2];
            if dx > half_box { dx -= box_size; }
            if dx < -half_box { dx += box_size; }
            if dy > half_box { dy -= box_size; }
            if dy < -half_box { dy += box_size; }
            if dz > half_box { dz -= box_size; }
            if dz < -half_box { dz += box_size; }
            com_plus[0] += ref_plus[0] + dx;
            com_plus[1] += ref_plus[1] + dy;
            com_plus[2] += ref_plus[2] + dz;
        } else {
            let mut dx = px - ref_minus[0];
            let mut dy = py - ref_minus[1];
            let mut dz = pz - ref_minus[2];
            if dx > half_box { dx -= box_size; }
            if dx < -half_box { dx += box_size; }
            if dy > half_box { dy -= box_size; }
            if dy < -half_box { dy += box_size; }
            if dz > half_box { dz -= box_size; }
            if dz < -half_box { dz += box_size; }
            com_minus[0] += ref_minus[0] + dx;
            com_minus[1] += ref_minus[1] + dy;
            com_minus[2] += ref_minus[2] + dz;
        }
    }

    if n_positive > 0 {
        com_plus[0] /= n_positive as f64;
        com_plus[1] /= n_positive as f64;
        com_plus[2] /= n_positive as f64;
    }
    if n_negative > 0 {
        com_minus[0] /= n_negative as f64;
        com_minus[1] /= n_negative as f64;
        com_minus[2] /= n_negative as f64;
    }

    // Distance between COMs (with periodic boundary)
    let mut dx = com_plus[0] - com_minus[0];
    let mut dy = com_plus[1] - com_minus[1];
    let mut dz = com_plus[2] - com_minus[2];
    if dx > half_box { dx -= box_size; }
    if dx < -half_box { dx += box_size; }
    if dy > half_box { dy -= box_size; }
    if dy < -half_box { dy += box_size; }
    if dz > half_box { dz -= box_size; }
    if dz < -half_box { dz += box_size; }

    (dx * dx + dy * dy + dz * dz).sqrt() / box_size
}

fn save_snapshot_v2(
    output_dir: &str, step: usize,
    pos: &[f64], vel: &[f64], signs: &[i32], temperatures: &[f64]
) {
    let path = format!("{}/snapshots/snap_{:05}.bin", output_dir, step);
    let mut file = match File::create(&path) {
        Ok(f) => BufWriter::new(f),
        Err(e) => {
            eprintln!("Failed to create snapshot {}: {}", path, e);
            return;
        }
    };

    let n = (pos.len() / 3) as u32;
    // Version marker: negative n means v2 format with velocities
    let version_marker: i32 = -2;
    file.write_all(&version_marker.to_le_bytes()).unwrap();
    file.write_all(&n.to_le_bytes()).unwrap();

    for i in 0..n as usize {
        // Position
        file.write_all(&pos[i * 3].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&pos[i * 3 + 2].to_le_bytes()).unwrap();
        // Velocity
        file.write_all(&vel[i * 3].to_le_bytes()).unwrap();
        file.write_all(&vel[i * 3 + 1].to_le_bytes()).unwrap();
        file.write_all(&vel[i * 3 + 2].to_le_bytes()).unwrap();
        // Sign and temperature
        file.write_all(&(signs[i] as f64).to_le_bytes()).unwrap();
        file.write_all(&temperatures[i].to_le_bytes()).unwrap();
    }
}

/// Load snapshot - supports both v1 (no velocities) and v2 (with velocities) formats
fn load_snapshot(path: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>), String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    // Read first 4 bytes - could be version marker or particle count
    let mut header_bytes = [0u8; 4];
    reader.read_exact(&mut header_bytes).map_err(|e| format!("Read error: {}", e))?;
    let header = i32::from_le_bytes(header_bytes);

    let (n, has_velocities) = if header == -2 {
        // V2 format with velocities
        let mut n_bytes = [0u8; 4];
        reader.read_exact(&mut n_bytes).map_err(|e| format!("Read error: {}", e))?;
        (u32::from_le_bytes(n_bytes) as usize, true)
    } else {
        // V1 format (header is particle count)
        (header as usize, false)
    };

    let mut positions = Vec::with_capacity(n * 3);
    let mut velocities = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);
    let mut temperatures = Vec::with_capacity(n);

    for _ in 0..n {
        let mut buf = [0u8; 8];

        // x, y, z
        reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        positions.push(f64::from_le_bytes(buf));
        reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        positions.push(f64::from_le_bytes(buf));
        reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        positions.push(f64::from_le_bytes(buf));

        // vx, vy, vz (only in v2)
        if has_velocities {
            reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
            velocities.push(f64::from_le_bytes(buf));
            reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
            velocities.push(f64::from_le_bytes(buf));
            reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
            velocities.push(f64::from_le_bytes(buf));
        } else {
            velocities.push(0.0);
            velocities.push(0.0);
            velocities.push(0.0);
        }

        // sign (stored as f64)
        reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        let sign_f64 = f64::from_le_bytes(buf);
        signs.push(if sign_f64 > 0.0 { 1 } else { -1 });

        // temperature
        reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        temperatures.push(f64::from_le_bytes(buf));
    }

    println!("  Loaded {} particles from {} (format v{})",
             n, path, if has_velocities { 2 } else { 1 });
    if !has_velocities {
        println!("  ⚠ WARNING: No velocities in snapshot - using zero velocities");
    }
    Ok((positions, velocities, signs, temperatures))
}
