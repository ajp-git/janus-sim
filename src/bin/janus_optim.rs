//! JANUS OPTIM — Optimization binary using YAML config
//!
//! Usage: cargo run --release --features cuda,cufft --bin janus_optim -- --config optim/tour1/config_run_A.yaml
//!
//! This binary is designed for trichotomy parameter search with:
//! - YAML configuration files
//! - JSONL metrics output
//! - Early stopping conditions
//! - Reproducible ICs (same seed = same ICs)

use clap::Parser;
use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

use janus::config::JanusConfig;
use janus::metrics::{StepMetrics, MetricsWriter};
use janus::early_stop::check_basic_early_stop;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};

#[derive(Parser)]
#[command(name = "janus_optim")]
#[command(about = "Janus optimization run with YAML config")]
struct Args {
    /// Path to YAML configuration file
    #[arg(short, long)]
    config: PathBuf,
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    let args = Args::parse();

    // Load configuration
    let config = match JanusConfig::from_yaml(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: Failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    let box_size = config.simulation.box_size_mpc;
    let n_particles = config.simulation.n_particles;
    let n_steps = config.simulation.n_steps;
    let z_start = config.simulation.z_start;
    let z_end = config.simulation.z_end;
    let seed = config.simulation.seed;
    let theta = config.simulation.theta;
    let eta = config.physics.eta;
    if eta > 1.0 {
        eprintln!("WARNING: eta={:.2} > 1.0 gives unphysical cosmology (a_init > 1, H < 0)", eta);
        eprintln!("         Results may not reflect standard Janus structure formation.");
        eprintln!("         Recommend eta in [0.5, 1.0] for optimization runs.");
    }
    let lambda_base = config.physics.lambda_base_mpc;
    let r_smooth = config.physics.r_smooth_mpc;
    let n_cells = config.pm_grid.n_cells;
    let k_min = config.pm_grid.k_min;
    let output_dir = PathBuf::from("/app").join(&config.output.dir);
    let metrics_interval = config.output.metrics_every_steps;

    // Compute derived parameters
    let (n_positive, n_negative) = config.particle_counts();
    let softening = config.softening();
    let r_cut = 2.0 * box_size / n_cells as f64;  // TreePM split radius
    let dt = 0.01;  // Fixed timestep

    // Create output directories
    fs::create_dir_all(output_dir.join("snapshots")).ok();

    println!("======================================================================");
    println!("JANUS OPTIM — Trichotomy Optimization Run");
    println!("======================================================================");
    println!("Config: {}", args.config.display());
    println!("\nSimulation:");
    println!("  N = {} ({} + / {} -)", n_particles, n_positive, n_negative);
    println!("  box = {:.1} Mpc", box_size);
    println!("  steps = {}", n_steps);
    println!("  z = {:.1} -> {:.1}", z_start, z_end);
    println!("  seed = {}", seed);
    println!("\nPhysics:");
    println!("  eta = {:.4}", eta);
    println!("  lambda_base = {:.1} Mpc", lambda_base);
    println!("  r_smooth = {:.1} Mpc", r_smooth);
    println!("  theta = {:.2}", theta);
    println!("  softening = {:.3} Mpc", softening);
    println!("  r_cut = {:.1} Mpc", r_cut);
    println!("\nOutput: {:?}", output_dir);

    // Initialize cosmology
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, z_start);
    let dtau = (cosmo.tau_end - cosmo.tau_start) / (n_steps as f64 * dt);
    let (a0, h0) = cosmo.get_params_at_tau(cosmo.tau_start);
    println!("\nCosmology:");
    println!("  tau = [{:.4}, {:.4}]", cosmo.tau_start, cosmo.tau_end);
    println!("  dtau/dt = {:.4}", dtau);
    println!("  a_init = {:.5}, H_init = {:.5}", a0, h0);

    // Generate ICs
    println!("\nGenerating ICs...");
    let t0 = Instant::now();
    let ng = (n_particles as f64).powf(1.0/3.0).round() as usize;
    let (pos, vel, signs) = generate_ics(seed, ng, box_size, eta, k_min);
    println!("  Done in {:.1}s (grid {}^3 = {})", t0.elapsed().as_secs_f64(), ng, ng*ng*ng);

    let np = signs.iter().filter(|&&s| s > 0).count();
    let nm = signs.len() - np;
    println!("  N+ = {}, N- = {} (ratio {:.4})", np, nm, nm as f64 / np as f64);

    // Convert to f32 for GPU
    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = vel.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize GPU simulation
    println!("\nInitializing GPU...");
    let mut sim = match GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, box_size) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to init GPU: {}", e);
            std::process::exit(1);
        }
    };
    sim.set_theta(theta);
    sim.set_softening(softening);
    sim.set_pm_k_min(k_min);
    sim.set_lambda_base(config.physics.lambda_base_mpc);

    let ke0 = sim.kinetic_energy().unwrap_or(1e-20).max(1e-20);
    let seg0 = sim.segregation().unwrap_or(0.0);
    println!("  KE0 = {:.4e}", ke0);
    println!("  Seg0 = {:.4}", seg0);

    // Open output files
    let mut ts_file = BufWriter::new(
        File::create(output_dir.join("time_series.csv")).unwrap()
    );
    writeln!(ts_file, "step,z,a,H,KE_ratio,segregation,dcom_x,dcom_y,dcom_z,dcom_mag,v_rms,v_max,ms").unwrap();

    let mut metrics_writer = MetricsWriter::new(output_dir.join("metrics.jsonl")).unwrap();

    // Save initial snapshot
    if config.output.save_snapshots {
        save_snapshot(&sim, &output_dir.join("snapshots/snap_000000.bin"), signs.len());
    }

    // Initial metrics
    let m0 = StepMetrics::from_basic(0, z_start, seg0, 0.0, 0.0, 1.0);
    metrics_writer.write(&m0).unwrap();

    // Save config to output
    config.to_yaml(output_dir.join("config.yaml")).ok();

    println!("\n{:>6} {:>7} {:>7} {:>10} {:>8} {:>6}",
             "Step", "z", "a", "KE/KE0", "Seg", "ms");
    println!("{}", "-".repeat(55));

    let sim_start = Instant::now();
    let mut stop_reason: Option<String> = None;
    let mut steps_completed = 0;

    for step in 1..=n_steps {
        let step_start = Instant::now();

        // Cosmological parameters
        let tau = cosmo.tau_start + (step as f64) * dt * dtau;
        let (a, h) = if tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { 1.0/a - 1.0 } else { 0.0 };

        // Step simulation
        if let Err(e) = sim.step_treepm_gpu(dt, r_cut, h, dtau) {
            stop_reason = Some(format!("Simulation error: {}", e));
            break;
        }

        let ms = step_start.elapsed().as_millis();
        let ke = sim.kinetic_energy().unwrap_or(0.0);
        let seg = sim.segregation().unwrap_or(0.0);
        let ke_ratio = ke / ke0;

        // Estimate velocities (approximate from KE)
        let v_rms = (2.0 * ke / (n_particles as f64 * 1e10)).sqrt() * 300.0;  // rough km/s
        let v_max = v_rms * 3.0;  // estimate

        steps_completed = step;

        // Compute ΔCOM
        let (dcom_x, dcom_y, dcom_z, dcom_mag) = match compute_dcom(&sim) {
            Some(dcom) => dcom,
            None => (0.0, 0.0, 0.0, 0.0),
        };

        // Write time series
        writeln!(ts_file, "{},{:.4},{:.5},{:.5},{:.4e},{:.4},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{}",
                 step, z, a, h, ke_ratio, seg, dcom_x, dcom_y, dcom_z, dcom_mag, v_rms, v_max, ms).unwrap();

        // Basic early stopping check
        if let Some(reason) = check_basic_early_stop(step as u32, ke_ratio, v_max, v_rms) {
            stop_reason = Some(reason);
            break;
        }

        // Metrics and logging at intervals
        if step % metrics_interval == 0 {
            let m = StepMetrics::from_basic(step as u32, z, seg, v_rms, v_max, ke_ratio);
            metrics_writer.write(&m).unwrap();

            println!("{:>6} {:>7.3} {:>7.5} {:>10.3e} {:>8.4} {:>6}",
                     step, z, a, ke_ratio, seg, ms);

            ts_file.flush().unwrap();
        }

        // Save snapshots
        if config.output.save_snapshots {
            // Mode 1: Save every N steps (for video generation)
            if let Some(interval) = config.output.snapshot_every_steps {
                if step % interval == 0 {
                    let path = output_dir.join(format!("snapshots/snap_{:06}.bin", step));
                    save_snapshot(&sim, &path, signs.len());
                    if step % (interval * 10) == 0 {
                        println!("  -> saved snapshot {} at z={:.2}", step, z);
                    }
                }
            } else {
                // Mode 2: Save at specific redshifts
                for &target_z in &config.output.snapshot_redshifts {
                    if (z - target_z).abs() < 0.1 && step % 50 == 0 {
                        let path = output_dir.join(format!("snapshots/snap_{:06}.bin", step));
                        save_snapshot(&sim, &path, signs.len());
                        println!("  -> saved snapshot at z={:.2}", z);
                        break;
                    }
                }
            }
        }
    }

    // Final metrics
    let seg_final = sim.segregation().unwrap_or(0.0);
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let ke_ratio_final = ke_final / ke0;

    let total_time = sim_start.elapsed().as_secs_f64();

    println!("\n======================================================================");
    if let Some(ref reason) = stop_reason {
        println!("STOPPED: {}", reason);
    } else {
        println!("COMPLETE");
    }
    println!("======================================================================");
    println!("Steps: {} / {}", steps_completed, n_steps);
    println!("Time: {:.1} min ({:.0} ms/step)", total_time / 60.0, 1000.0 * total_time / steps_completed as f64);
    println!("Segregation: {:.4} -> {:.4}", seg0, seg_final);
    println!("KE ratio: {:.4e}", ke_ratio_final);

    // Save summary
    let summary = serde_json::json!({
        "config": args.config.to_string_lossy(),
        "eta": eta,
        "lambda_base_mpc": lambda_base,
        "n_particles": n_particles,
        "steps_completed": steps_completed,
        "steps_total": n_steps,
        "seg_initial": seg0,
        "seg_final": seg_final,
        "ke_ratio_final": ke_ratio_final,
        "time_seconds": total_time,
        "stop_reason": stop_reason,
    });
    fs::write(output_dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).ok();

    if let Some(reason) = stop_reason {
        // Write abort to log
        let mut log = File::create(output_dir.join("run.log")).unwrap();
        writeln!(log, "ABORT: {}", reason).unwrap();
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_ics(seed: u64, ng: usize, box_size: f64, eta: f64, k_min: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = ng * ng * ng;
    let cell = box_size / ng as f64;
    let half_box = box_size / 2.0;

    let mut pos = Vec::with_capacity(n * 3);
    let mut vel = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    // Compute n_positive from eta
    let n_positive = (n as f64 / (1.0 + eta)).round() as usize;

    // Generate grid positions with small perturbation
    let noise = Normal::new(0.0, 0.1 * cell).unwrap();

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let _idx = iz * ng * ng + iy * ng + ix;

                // Grid position with noise
                let x = (ix as f64 + 0.5) * cell - half_box + rng.sample(noise);
                let y = (iy as f64 + 0.5) * cell - half_box + rng.sample(noise);
                let z = (iz as f64 + 0.5) * cell - half_box + rng.sample(noise);

                pos.push(x);
                pos.push(y);
                pos.push(z);

                // Small random velocity
                vel.push(rng.sample(noise) * 10.0);
                vel.push(rng.sample(noise) * 10.0);
                vel.push(rng.sample(noise) * 10.0);

                // Random sign assignment
                signs.push(if rng.random_bool(n_positive as f64 / n as f64) { 1 } else { -1 });
            }
        }
    }

    // Add coherent perturbation (Zel'dovich-like) for structure
    let k = 2.0 * PI / box_size * k_min as f64;
    let amplitude = 0.001 * box_size;

    for i in 0..n {
        let x = pos[i * 3];
        let disp = amplitude * (k * x).sin();
        pos[i * 3] += disp;
        vel[i * 3] += disp * 50.0;  // velocity from displacement
    }

    (pos, vel, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::Path, n: usize) {
    use std::io::BufWriter;

    let pos = match sim.get_positions() {
        Ok(p) => p,
        Err(_) => return,
    };
    let vel = match sim.get_velocities() {
        Ok(v) => v,
        Err(_) => return,
    };
    let signs = match sim.get_signs() {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut f = BufWriter::new(File::create(path).unwrap());
    // Write header: n particles
    // Format v2: n(u32) + n×(3×f32 pos + 3×f32 vel + i8 sign)
    f.write_all(&(n as u32).to_le_bytes()).unwrap();
    for i in 0..n {
        // Position
        f.write_all(&pos[i*3].to_le_bytes()).unwrap();
        f.write_all(&pos[i*3+1].to_le_bytes()).unwrap();
        f.write_all(&pos[i*3+2].to_le_bytes()).unwrap();
        // Velocity
        f.write_all(&vel[i*3].to_le_bytes()).unwrap();
        f.write_all(&vel[i*3+1].to_le_bytes()).unwrap();
        f.write_all(&vel[i*3+2].to_le_bytes()).unwrap();
        // Sign
        f.write_all(&(signs[i] as i8).to_le_bytes()).unwrap();
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_dcom(sim: &GpuNBodyTwoPass) -> Option<(f64, f64, f64, f64)> {
    let pos = sim.get_positions().ok()?;
    let signs = sim.get_signs().ok()?;
    let n = signs.len();

    let mut com_plus = [0.0f64; 3];
    let mut com_minus = [0.0f64; 3];
    let mut n_plus = 0usize;
    let mut n_minus = 0usize;

    for i in 0..n {
        let x = pos[i*3] as f64;
        let y = pos[i*3+1] as f64;
        let z = pos[i*3+2] as f64;

        if signs[i] > 0 {
            com_plus[0] += x;
            com_plus[1] += y;
            com_plus[2] += z;
            n_plus += 1;
        } else {
            com_minus[0] += x;
            com_minus[1] += y;
            com_minus[2] += z;
            n_minus += 1;
        }
    }

    if n_plus == 0 || n_minus == 0 {
        return None;
    }

    com_plus[0] /= n_plus as f64;
    com_plus[1] /= n_plus as f64;
    com_plus[2] /= n_plus as f64;
    com_minus[0] /= n_minus as f64;
    com_minus[1] /= n_minus as f64;
    com_minus[2] /= n_minus as f64;

    let dx = com_plus[0] - com_minus[0];
    let dy = com_plus[1] - com_minus[1];
    let dz = com_plus[2] - com_minus[2];
    let mag = (dx*dx + dy*dy + dz*dz).sqrt();

    Some((dx, dy, dz, mag))
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("ERROR: This binary requires --features cuda,cufft");
    std::process::exit(1);
}
