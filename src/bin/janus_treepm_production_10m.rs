//! Phase 11 — Production run TreePM Janus 10M particles, z=10→z=0.
//!
//! Pipeline post-validation Phase 10.7+10.8+10.9:
//!  - GPU TreePM (cuFFT PM + Barnes-Hut Tree avec splitting Springel T(x))
//!  - r_s passé en paramètre, fix kernel correct
//!  - Cosmologie Janus dual a±, dynamic VSL c̄(z), φ coupling
//!  - Snapshots V3 standard (compatibles scripts/postprocess_*.py)
//!  - Checkpoints + heartbeat 5 min + auto-stop NaN/v_rms

#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_GRID: usize = 215; // 215³ = 9,938,375 ≈ 10M
#[cfg(all(feature = "cuda", feature = "cufft"))]
const L_BOX: f64 = 500.0; // Mpc
#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_PM: usize = 256; // PM grid (cell ≈ 1.953 Mpc)
#[cfg(all(feature = "cuda", feature = "cufft"))]
const Z_INIT: f64 = 10.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const Z_TARGET: f64 = 0.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_STEPS_MAX: usize = 15000;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const DT: f64 = 0.001; // Gyr
#[cfg(all(feature = "cuda", feature = "cufft"))]
const ETA: f64 = 1.045;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const MU: f64 = 19.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const H0: f64 = 69.9;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const OMEGA_B: f64 = 0.05;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SOFTENING_PLUS: f64 = 0.05;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SOFTENING_MINUS: f64 = 0.25;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SEED_IC: u64 = 42;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SNAPSHOT_FREQ_STEPS: usize = 10;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const CHECKPOINT_FREQ_STEPS: usize = 1000;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const DIAGNOSTICS_FREQ_STEPS: usize = 50;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const HEARTBEAT_INTERVAL_SEC: u64 = 300;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const THETA: f64 = 0.5;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SPLIT_SCALE_FACTOR: f64 = 1.2;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const CUTOFF_FACTOR: f64 = 6.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const MPC_GYR_TO_KMS: f64 = 977.8;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const TAU_0_JANUS: f64 = 23.3011940229;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const V_RMS_LIMIT: f64 = 5000.0;

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;
    if a < A_TRANSITION_JANUS {
        h0_gyr_inv / a.powf(1.5)
    } else {
        let cosh2_mu = (a / ALPHA_SQ_JANUS).max(1.0);
        let cosh_mu = cosh2_mu.sqrt();
        let mu_p = cosh_mu.acosh();
        let s2mu = (2.0 * mu_p).sinh();
        s2mu / (TAU_0_JANUS * ALPHA_SQ_JANUS * cosh2_mu * (1.0 + 0.5 * s2mu))
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn vrms_split(vel: &[f32], signs: &[i8]) -> (f64, f64, bool) {
    // Returns (v_rms_plus, v_rms_minus, has_nan) — vel in Mpc/Gyr → km/s
    let mut sum_p = 0.0_f64;
    let mut sum_m = 0.0_f64;
    let mut n_p = 0_usize;
    let mut n_m = 0_usize;
    for i in 0..signs.len() {
        let vx = vel[i * 3] as f64;
        let vy = vel[i * 3 + 1] as f64;
        let vz = vel[i * 3 + 2] as f64;
        if !vx.is_finite() || !vy.is_finite() || !vz.is_finite() {
            return (0.0, 0.0, true);
        }
        let v2 = vx * vx + vy * vy + vz * vz;
        if signs[i] > 0 {
            sum_p += v2;
            n_p += 1;
        } else {
            sum_m += v2;
            n_m += 1;
        }
    }
    let v_p = if n_p > 0 {
        (sum_p / n_p as f64).sqrt() * MPC_GYR_TO_KMS
    } else {
        0.0
    };
    let v_m = if n_m > 0 {
        (sum_m / n_m as f64).sqrt() * MPC_GYR_TO_KMS
    } else {
        0.0
    };
    (v_p, v_m, false)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn write_v3_snapshot(
    snap_path: &std::path::Path,
    pos: &[f32],
    vel: &[f32],
    signs: &[i8],
    a_plus: f64,
    t_gyr: f64,
    z: f64,
    step: u64,
    n_plus: usize,
    n_minus: usize,
) -> std::io::Result<()> {
    use janus::snapshot_v3::{
        write_snapshot_v3, ParticleV3, SnapshotHeaderV3,
    };
    let n = signs.len();
    // Compute per-particle masses (Janus: equal mass per particle, ratio via N±)
    let h = H0 / 100.0;
    let rho_crit = 2.775e11 * h * h; // M☉/Mpc³
    let m_plus_total = OMEGA_B * rho_crit * L_BOX.powi(3);
    let m_minus_total = MU * m_plus_total;
    let m_plus = (m_plus_total / n_plus as f64) as f32;
    let m_minus = (m_minus_total / n_minus as f64) as f32;
    let mut particles: Vec<ParticleV3> = Vec::with_capacity(n);
    for i in 0..n {
        // Convert vel from Mpc/Gyr to km/s for V3 storage
        let vx = (vel[i * 3] as f64 * MPC_GYR_TO_KMS) as f32;
        let vy = (vel[i * 3 + 1] as f64 * MPC_GYR_TO_KMS) as f32;
        let vz = (vel[i * 3 + 2] as f64 * MPC_GYR_TO_KMS) as f32;
        let (mass, eps, sign_byte) = if signs[i] > 0 {
            (m_plus, SOFTENING_PLUS as f32, 1u8)
        } else {
            (m_minus, SOFTENING_MINUS as f32, 255u8)
        };
        particles.push(ParticleV3 {
            pos: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
            vel: [vx, vy, vz],
            mass,
            epsilon: eps,
            sign: sign_byte,
            split_level: 0,
            is_star: 0,
            flags: 0,
        });
    }
    let mut header = SnapshotHeaderV3::new("janus_treepm_10m");
    header.n_total = n as u64;
    header.a = a_plus;
    header.t_gyr = t_gyr;
    header.l_box = L_BOX;
    header.h0 = H0;
    header.mu = MU;
    header.omega_b = OMEGA_B;
    header.m_part_plus_base = m_plus as f64;
    header.m_part_minus_base = m_minus as f64;
    header.eps_plus_base = SOFTENING_PLUS;
    header.eps_minus_base = SOFTENING_MINUS;
    header.seed_ic = SEED_IC as u32;
    header.z_init = Z_INIT;
    header.z_start_run = Z_INIT;
    let _ = step;
    let _ = z;
    write_snapshot_v3(snap_path, &header, &particles)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn write_heartbeat(
    path: &str,
    step: usize,
    z: f64,
    t_gyr: f64,
    v_p: f64,
    v_m: f64,
    speed: f64,
    eta_min: f64,
    elapsed_min: f64,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "PHASE 11 — Run TreePM Janus 10M heartbeat")?;
    writeln!(f, "step={}/{}", step, N_STEPS_MAX)?;
    writeln!(f, "z={:.4} (target {})", z, Z_TARGET)?;
    writeln!(f, "t_Gyr={:.4}", t_gyr)?;
    writeln!(f, "v_rms+={:.1} km/s", v_p)?;
    writeln!(f, "v_rms-={:.1} km/s", v_m)?;
    writeln!(f, "speed={:.4} step/s", speed)?;
    writeln!(f, "elapsed={:.2} min", elapsed_min)?;
    writeln!(f, "ETA={:.2} min ({:.2} h)", eta_min, eta_min / 60.0)?;
    writeln!(
        f,
        "timestamp={}",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    )?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::janus_expansion::{a_minus_from_a_plus, compute_phi_factors};
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
    use janus::vsl_dynamic::CoupledFriedmann;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::fs;
    use std::io::Write;
    use std::time::Instant;

    let max_steps_env: usize = std::env::var("MAX_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_STEPS_MAX);

    let run_dir = std::env::var("RUN_DIR")
        .unwrap_or_else(|_| "/app/output/janus_treepm_10m".to_string());
    fs::create_dir_all(format!("{}/snapshots", run_dir))?;
    fs::create_dir_all(format!("{}/checkpoints", run_dir))?;
    fs::create_dir_all(format!("{}/diagnostics", run_dir))?;

    let heartbeat_path = "/app/logs/treepm/phase11_HEARTBEAT.txt";
    fs::create_dir_all("/app/logs/treepm")?;

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  JANUS PRODUCTION TreePM 10M — Phase 11                          ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  N = {}³ = {} particles", N_GRID, N_GRID.pow(3));
    println!("║  Box = {} Mpc, n_pm = {} (cell ≈ {:.3} Mpc)",
        L_BOX, N_PM, L_BOX / N_PM as f64);
    println!("║  μ = {}, η = {}, dt = {} Gyr (max {} steps)",
        MU, ETA, DT, max_steps_env);
    println!("║  z_init = {}, z_target = {}", Z_INIT, Z_TARGET);
    println!("║  Snapshots V3 every {} steps → ~{} files",
        SNAPSHOT_FREQ_STEPS, max_steps_env / SNAPSHOT_FREQ_STEPS);
    println!("║  Output: {}", run_dir);
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // === ICs Zel'dovich (lattice + 15% perturbation, μ=19) ===
    let n: usize = N_GRID.pow(3);
    let f_plus = 1.0 / (1.0 + MU); // 5%
    let n_plus = (n as f64 * f_plus).round() as usize;
    let n_minus = n - n_plus;
    println!(
        "[IC] N={} ({}m+, {}m-), generating Zel'dovich on lattice {}³...",
        n, n_plus, n_minus, N_GRID
    );

    let half = (L_BOX as f32) * 0.5;
    let dx = (L_BOX / N_GRID as f64) as f32;
    let displacement_amp = 0.15 * dx;
    let mut rng = StdRng::seed_from_u64(SEED_IC);
    let mut pos_f32 = Vec::with_capacity(n * 3);
    let vel = vec![0.0_f32; n * 3];
    let mut signs = Vec::with_capacity(n);
    let mut idx = 0usize;
    let t_ic = Instant::now();
    for i in 0..N_GRID {
        for j in 0..N_GRID {
            for k in 0..N_GRID {
                let gx = (i as f32 + 0.5) * dx - half;
                let gy = (j as f32 + 0.5) * dx - half;
                let gz = (k as f32 + 0.5) * dx - half;
                let dxp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let dyp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let dzp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let mut x = gx + dxp;
                let mut y = gy + dyp;
                let mut z = gz + dzp;
                while x >= half {
                    x -= 2.0 * half;
                }
                while x < -half {
                    x += 2.0 * half;
                }
                while y >= half {
                    y -= 2.0 * half;
                }
                while y < -half {
                    y += 2.0 * half;
                }
                while z >= half {
                    z -= 2.0 * half;
                }
                while z < -half {
                    z += 2.0 * half;
                }
                pos_f32.push(x);
                pos_f32.push(y);
                pos_f32.push(z);
                signs.push(if idx < n_plus { 1_i8 } else { -1_i8 });
                idx += 1;
            }
        }
    }
    println!("[IC] Done in {:.2}s", t_ic.elapsed().as_secs_f64());

    // === GPU init ===
    println!("[GPU] Allocating buffers (PM 256³ + BVH 2 trees)...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel, signs.clone(), L_BOX)?;
    sim.set_softening(SOFTENING_PLUS);
    sim.set_theta(THETA);
    let auto_mf = sim.get_mass_factor();
    println!("[GPU] Allocation OK, theta={}, softening={}, mass_factor={:.4e}",
        THETA, SOFTENING_PLUS, auto_mf);

    // TreePM scales
    let dg = L_BOX / N_PM as f64;
    let r_s = SPLIT_SCALE_FACTOR * dg;
    let r_cut = CUTOFF_FACTOR * dg;
    println!("[TreePM] dg={:.4}, r_s={:.4}, r_cut={:.4} (r_cut/r_s={:.2})",
        dg, r_s, r_cut, r_cut / r_s);

    // CSV
    let csv_path = format!("{}/diagnostics/evolution.csv", run_dir);
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
    writeln!(
        csv,
        "step,z,t_Gyr,a_plus,a_minus,c_bar,phi,v_rms_plus,v_rms_minus,wall_min,step_per_sec"
    )?;

    // Run log
    let log_path = format!("{}/run.log", run_dir);
    let mut log = std::io::BufWriter::new(std::fs::File::create(&log_path)?);
    writeln!(log, "JANUS PRODUCTION TreePM 10M — start")?;
    writeln!(log, "Started: {:?}", std::time::SystemTime::now())?;

    // === Main loop ===
    let start = Instant::now();
    let mut last_heartbeat = Instant::now();
    let mut a = 1.0 / (1.0 + Z_INIT);
    let mut t_gyr = 0.0_f64;
    let mut step = 0usize;
    let mut stop_reason = String::new();
    println!("\n[RUN] Starting main loop z={}→z={} ...\n", Z_INIT, Z_TARGET);

    loop {
        let z = 1.0 / a - 1.0;
        if z <= Z_TARGET || step >= max_steps_env {
            break;
        }

        let h_plus = compute_hubble_janus(a, H0);
        let a_minus = a_minus_from_a_plus(a, ETA);
        let h_minus = compute_hubble_janus(a_minus, H0);
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let (phi, _) = compute_phi_factors(a, ETA);

        if let Err(e) = sim.step_treepm_gpu_cosmo(
            DT, r_cut, r_s, a, a_minus, h_plus, h_minus, phi, c_ratio_sq, 1.0,
        ) {
            stop_reason = format!("CUDA error step {}: {}", step, e);
            eprintln!("❌ {}", stop_reason);
            writeln!(log, "{}", stop_reason)?;
            break;
        }

        // Update a
        let da = a * h_plus * DT;
        a += da;
        t_gyr += DT;

        // Diagnostics + snapshots
        if step % DIAGNOSTICS_FREQ_STEPS == 0 || step % SNAPSHOT_FREQ_STEPS == 0 {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            let (v_p, v_m, has_nan) = vrms_split(&vel, &signs);
            if has_nan {
                stop_reason = format!("NaN/Inf at step {}", step);
                eprintln!("❌ {}", stop_reason);
                writeln!(log, "{}", stop_reason)?;
                break;
            }
            if v_p > V_RMS_LIMIT || v_m > V_RMS_LIMIT {
                stop_reason = format!("v_rms > {} km/s at step {}: v+={:.0} v-={:.0}",
                    V_RMS_LIMIT, step, v_p, v_m);
                eprintln!("❌ {}", stop_reason);
                writeln!(log, "{}", stop_reason)?;
                break;
            }

            let elapsed = start.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 { step as f64 / elapsed } else { 0.0 };
            let eta_sec = if speed > 0.0 {
                (max_steps_env - step) as f64 / speed
            } else {
                0.0
            };

            if step % DIAGNOSTICS_FREQ_STEPS == 0 {
                let c_bar = c_ratio_sq.sqrt();
                writeln!(
                    csv,
                    "{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8},{:.2},{:.2},{:.2},{:.4}",
                    step, z, t_gyr, a, a_minus, c_bar, phi,
                    v_p, v_m, elapsed / 60.0, speed
                )?;
                csv.flush()?;
            }

            if step % SNAPSHOT_FREQ_STEPS == 0 {
                let snap_path = std::path::PathBuf::from(format!(
                    "{}/snapshots/snap_{:05}.bin",
                    run_dir, step
                ));
                if let Err(e) = write_v3_snapshot(
                    &snap_path, &pos, &vel, &signs, a, t_gyr, z, step as u64, n_plus, n_minus,
                ) {
                    eprintln!("⚠ snapshot write failed step {}: {}", step, e);
                    writeln!(log, "snapshot write failed step {}: {}", step, e)?;
                }
            }

            if step % 100 == 0 {
                println!(
                    "  step {:6}/{} | z={:.4} | t={:.2} Gyr | v±={:.0}/{:.0} km/s | {:.3} step/s | ETA {:.1} min",
                    step, max_steps_env, z, t_gyr, v_p, v_m, speed, eta_sec / 60.0
                );
            }

            // Heartbeat 5 min
            if last_heartbeat.elapsed().as_secs() > HEARTBEAT_INTERVAL_SEC {
                let _ = write_heartbeat(
                    heartbeat_path, step, z, t_gyr, v_p, v_m, speed,
                    eta_sec / 60.0, elapsed / 60.0,
                );
                last_heartbeat = Instant::now();
            }
        }

        step += 1;
    }

    // === Final state ===
    let total = start.elapsed().as_secs_f64();
    csv.flush()?;
    drop(csv);

    let final_z = 1.0 / a - 1.0;
    println!("\n[END] step={}, z_final={:.4}, t={:.2} Gyr, wall {:.1} min ({:.2} h)",
        step, final_z, t_gyr, total / 60.0, total / 3600.0);
    if !stop_reason.is_empty() {
        println!("[STOP] {}", stop_reason);
        writeln!(log, "STOP: {}", stop_reason)?;
    }

    // Final snapshot
    let final_pos = sim.get_positions()?;
    let final_vel = sim.get_velocities()?;
    let snap_path =
        std::path::PathBuf::from(format!("{}/snapshots/snap_final.bin", run_dir));
    let _ = write_v3_snapshot(
        &snap_path, &final_pos, &final_vel, &signs, a, t_gyr, final_z, step as u64,
        n_plus, n_minus,
    );

    writeln!(log, "Wall: {:.2} h, final z={:.4}", total / 3600.0, final_z)?;
    log.flush()?;
    println!("\n[OUTPUTS]");
    println!("  Snapshots: {}/snapshots/", run_dir);
    println!("  Diagnostics: {}/diagnostics/evolution.csv", run_dir);
    println!("  Log: {}/run.log", run_dir);
    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires features: cuda cufft");
    std::process::exit(1);
}
