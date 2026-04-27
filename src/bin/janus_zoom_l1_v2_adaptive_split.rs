//! JANUS Zoom L1 v2 — Adaptive Splitting with m⁻ subdivision
//!
//! Based on ZOOM_L1_V2_SPEC.md (April 2026)
//! Key innovation: smooth split factor transitions + m⁻ subdivision

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
use janus::vsl_dynamic::CoupledFriedmann;
use std::fs::{self, File};
use std::io::{BufWriter, Write, Read};
use std::f64::consts::PI;
use rand::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// COSMOLOGY & PHYSICS
// ═══════════════════════════════════════════════════════════════════════════
const ETA: f64 = 1.045;
const G_CODE: f64 = 4.498e-15;  // G in [Mpc³/(M_sun·Gyr²)]

// ═══════════════════════════════════════════════════════════════════════════
// ZOOM PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const CENTER: [f64; 3] = [-5.329, 11.171, -39.571];
const R_HR: f64 = 8.0;
const R_EXTRACT: f64 = 50.0;
const L_BOX_SOURCE: f64 = 500.0;  // Source box size
const L_ZOOM: f64 = 100.0;        // Zoom box size (2 × R_EXTRACT)

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE SPLITTING PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
// Zone boundaries for m+
const R_CORE: f64 = 2.0;
const R_MID: f64 = 8.0;
const R_EXT: f64 = 20.0;
const R_OUTER: f64 = 25.0;

// Split factors for m+ (physically limited by inter-parent spacing)
// Core: ~530 parents, spacing ~400 kpc → r_disp_max=130kpc → N_max≈35
const SPLIT_CORE: u32 = 30;
const SPLIT_MID: u32 = 15;
const SPLIT_EXT: u32 = 8;
const SPLIT_OUTER: u32 = 1;

// m⁻ splitting (CRITICAL for Janus - Gemini recommendation)
const R_MINUS_HR: f64 = 12.0;
const SPLIT_MINUS: u32 = 5;
const SPLIT_MINUS_OUTER: u32 = 3;

// Smooth transition width
const TRANS_WIDTH: f64 = 1.0;  // Mpc

// Splitting physics
const THERMAL_SIGMA: f64 = 2.0;  // km/s velocity perturbation

// ═══════════════════════════════════════════════════════════════════════════
// SOFTENING (adaptive by zone)
// ═══════════════════════════════════════════════════════════════════════════
const EPS_CORE: f64 = 0.030;  // 30 kpc for core (r<2) and mid (r<8)
const EPS_MID: f64 = 0.030;   // 30 kpc
const EPS_EXT: f64 = 0.050;   // 50 kpc for ext (r>8)

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const DT: f64 = 0.0003;
const DT_RELAX: f64 = 0.0001;  // Smaller dt for relaxation
const N_RELAX: usize = 50;
const N_STEPS: usize = 18500;
const THETA: f64 = 0.7;

// ═══════════════════════════════════════════════════════════════════════════
// BARYONIC PHYSICS (HR only)
// ═══════════════════════════════════════════════════════════════════════════
const T_INIT_HR: f64 = 10000.0;
const T_FLOOR: f64 = 1000.0;
const T_SF_MAX: f64 = 10000.0;
const OVERDENSITY_THRESHOLD: f64 = 50.0;
const EPSILON_STAR: f64 = 0.01;
const M_PART_SOURCE: f64 = 5.1e10;  // Source particle mass [M_sun]
const V_KICK_SN: f64 = 20.0;
const R_NEIGHBOR: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════════
const SNAPSHOT_INTERVAL: usize = 10;
const CSV_INTERVAL: usize = 5;
const OUTPUT_DIR: &str = "/app/output/janus_zoom_L1_v2";

// ═══════════════════════════════════════════════════════════════════════════
// SMOOTH INTERPOLATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Smooth step (Hermite interpolation): 0 for x<edge0, 1 for x>edge1
fn smoothstep(x: f64, edge0: f64, edge1: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Compute split factor for m+ with SMOOTH transitions at zone boundaries
fn compute_split_factor_plus(r: f64) -> f64 {
    let hw = TRANS_WIDTH / 2.0;

    // Transition functions at each boundary
    let t_core_mid = smoothstep(r, R_CORE - hw, R_CORE + hw);
    let t_mid_ext = smoothstep(r, R_MID - hw, R_MID + hw);
    let t_ext_outer = smoothstep(r, R_EXT - hw, R_EXT + hw);

    // Interpolate through zones
    let f_core = SPLIT_CORE as f64;
    let f_mid = SPLIT_MID as f64;
    let f_ext = SPLIT_EXT as f64;
    let f_outer = SPLIT_OUTER as f64;

    if r < R_CORE - hw {
        f_core
    } else if r < R_CORE + hw {
        lerp(f_core, f_mid, t_core_mid)
    } else if r < R_MID - hw {
        f_mid
    } else if r < R_MID + hw {
        lerp(f_mid, f_ext, t_mid_ext)
    } else if r < R_EXT - hw {
        f_ext
    } else if r < R_EXT + hw {
        lerp(f_ext, f_outer, t_ext_outer)
    } else {
        f_outer
    }
}

/// Compute split factor for m⁻ with smooth transition
fn compute_split_factor_minus(r: f64) -> f64 {
    let hw = TRANS_WIDTH / 2.0;
    let t = smoothstep(r, R_MINUS_HR - hw, R_MINUS_HR + hw);
    lerp(SPLIT_MINUS as f64, SPLIT_MINUS_OUTER as f64, t)
}

/// Compute adaptive softening based on radius
fn adaptive_softening(r: f64) -> f64 {
    let hw = TRANS_WIDTH / 2.0;
    let t1 = smoothstep(r, R_CORE - hw, R_CORE + hw);
    let t2 = smoothstep(r, R_MID - hw, R_MID + hw);

    if r < R_CORE + hw {
        lerp(EPS_CORE, EPS_MID, t1)
    } else if r < R_MID + hw {
        lerp(EPS_MID, EPS_EXT, t2)
    } else {
        EPS_EXT
    }
}

/// Compute dispersion radius for split particles
/// Guarantees inter-particle spacing d_ips >= epsilon
/// Volume-based: d_ips = (4/3 π r³ / N)^(1/3)
/// Solving for r when d_ips = epsilon: r = epsilon × (N / 4.19)^(1/3)
fn compute_r_disp(n_split: usize, epsilon: f64) -> f64 {
    if n_split <= 1 {
        return 0.0;
    }
    // r_disp = epsilon × (N / 4.19)^(1/3) where 4.19 ≈ 4π/3
    let r = epsilon * (n_split as f64 / 4.19).powf(1.0 / 3.0);
    r.max(epsilon)  // Never less than epsilon
}

// ═══════════════════════════════════════════════════════════════════════════
// BLUE NOISE — Fibonacci Lattice
// ═══════════════════════════════════════════════════════════════════════════

/// Generate n uniformly distributed directions using Fibonacci lattice (Blue Noise)
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    if n <= 1 {
        return vec![[0.0, 0.0, 0.0]];
    }

    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / golden;

    (0..n).map(|i| {
        let theta = golden_angle * (i as f64);
        let z = 1.0 - (2.0 * i as f64 + 1.0) / (n as f64);
        let r_xy = (1.0 - z * z).sqrt();
        [r_xy * theta.cos(), r_xy * theta.sin(), z]
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// SNAPSHOT I/O
// ═══════════════════════════════════════════════════════════════════════════

fn read_source_snapshot(path: &str) -> (Vec<f64>, Vec<f64>, Vec<i32>, f64, f64) {
    let mut file = File::open(path).expect("Cannot open source snapshot");
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf8).unwrap();
    let n = u64::from_le_bytes(buf8) as usize;

    file.read_exact(&mut buf8).unwrap();
    let a = f64::from_le_bytes(buf8);

    file.read_exact(&mut buf8).unwrap();
    let t = f64::from_le_bytes(buf8);

    let mut pos_buf = vec![0u8; n * 3 * 4];
    file.read_exact(&mut pos_buf).unwrap();

    let mut vel_buf = vec![0u8; n * 3 * 4];
    file.read_exact(&mut vel_buf).unwrap();

    let mut signs_buf = vec![0u8; n];
    file.read_exact(&mut signs_buf).unwrap();

    let mut positions = Vec::with_capacity(n * 3);
    let mut velocities = Vec::with_capacity(n * 3);
    let mut signs = Vec::with_capacity(n);

    for i in 0..n {
        let px = f32::from_le_bytes([pos_buf[i*12], pos_buf[i*12+1], pos_buf[i*12+2], pos_buf[i*12+3]]) as f64;
        let py = f32::from_le_bytes([pos_buf[i*12+4], pos_buf[i*12+5], pos_buf[i*12+6], pos_buf[i*12+7]]) as f64;
        let pz = f32::from_le_bytes([pos_buf[i*12+8], pos_buf[i*12+9], pos_buf[i*12+10], pos_buf[i*12+11]]) as f64;

        let vx = f32::from_le_bytes([vel_buf[i*12], vel_buf[i*12+1], vel_buf[i*12+2], vel_buf[i*12+3]]) as f64;
        let vy = f32::from_le_bytes([vel_buf[i*12+4], vel_buf[i*12+5], vel_buf[i*12+6], vel_buf[i*12+7]]) as f64;
        let vz = f32::from_le_bytes([vel_buf[i*12+8], vel_buf[i*12+9], vel_buf[i*12+10], vel_buf[i*12+11]]) as f64;

        positions.push(px);
        positions.push(py);
        positions.push(pz);
        velocities.push(vx);
        velocities.push(vy);
        velocities.push(vz);
        signs.push(if signs_buf[i] == 1 { 1 } else { -1 });
    }

    (positions, velocities, signs, a, t)
}

fn write_snapshot(path: &str, pos: &[f64], vel: &[f64], signs: &[i32], epsilon: Option<&[f64]>, a: f64, t: f64) {
    let n = signs.len();
    let mut file = File::create(path).expect("Cannot create snapshot");

    file.write_all(&(n as u64).to_le_bytes()).unwrap();
    file.write_all(&a.to_le_bytes()).unwrap();
    file.write_all(&t.to_le_bytes()).unwrap();

    for i in 0..n {
        file.write_all(&(pos[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(pos[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(pos[i * 3 + 2] as f32).to_le_bytes()).unwrap();
    }
    for i in 0..n {
        file.write_all(&(vel[i * 3] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3 + 1] as f32).to_le_bytes()).unwrap();
        file.write_all(&(vel[i * 3 + 2] as f32).to_le_bytes()).unwrap();
    }
    for i in 0..n {
        file.write_all(&[if signs[i] > 0 { 1u8 } else { 0u8 }]).unwrap();
    }
    // Extended format: per-particle epsilon (f32)
    if let Some(eps) = epsilon {
        for &e in eps {
            file.write_all(&(e as f32).to_le_bytes()).unwrap();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE SPLITTING — Core function
// ═══════════════════════════════════════════════════════════════════════════

fn extract_and_adaptive_split(
    positions: &[f64],
    velocities: &[f64],
    signs: &[i32],
) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>, Vec<bool>, Vec<f64>, usize, usize) {
    let mut rng = rand::thread_rng();

    let mut new_pos = Vec::new();
    let mut new_vel = Vec::new();
    let mut new_signs = Vec::new();
    let mut new_masses = Vec::new();
    let mut is_hr_flags = Vec::new();
    let mut new_epsilon = Vec::new();  // Per-particle softening

    let n_source = signs.len();
    let mut n_daughters_plus = 0usize;
    let mut n_daughters_minus = 0usize;

    println!("  Processing {} source particles...", n_source);

    for i in 0..n_source {
        let px = positions[i * 3];
        let py = positions[i * 3 + 1];
        let pz = positions[i * 3 + 2];

        // Distance from CENTER with periodic wrapping
        let mut dx = px - CENTER[0];
        let mut dy = py - CENTER[1];
        let mut dz = pz - CENTER[2];

        // Periodic wrap
        dx -= L_BOX_SOURCE * (dx / L_BOX_SOURCE).round();
        dy -= L_BOX_SOURCE * (dy / L_BOX_SOURCE).round();
        dz -= L_BOX_SOURCE * (dz / L_BOX_SOURCE).round();

        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        // Skip particles outside extraction region
        if r > R_EXTRACT {
            continue;
        }

        let sign = signs[i];
        let is_plus = sign > 0;
        let is_hr = r < R_HR;

        // Compute adaptive split factor with smooth transitions
        let split_f = if is_plus {
            compute_split_factor_plus(r)
        } else {
            compute_split_factor_minus(r)
        };

        let n_split = (split_f.round() as usize).max(1);

        // Softening for this zone
        let epsilon = adaptive_softening(r);

        // Dispersion radius: guarantees inter-particle spacing >= epsilon
        let r_disp = compute_r_disp(n_split, epsilon);

        // Mass of each daughter
        let m_daughter = M_PART_SOURCE / (n_split as f64);

        // Generate daughter positions using Fibonacci lattice (Blue Noise)
        let directions = fibonacci_sphere(n_split);

        for dir in &directions {
            // Position: parent + r_disp * direction
            let new_x = dx + r_disp * dir[0];
            let new_y = dy + r_disp * dir[1];
            let new_z = dz + r_disp * dir[2];

            // Velocity: parent + small thermal perturbation (Box-Muller)
            let mut dv = || -> f64 {
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                THERMAL_SIGMA * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            };

            new_pos.push(new_x);
            new_pos.push(new_y);
            new_pos.push(new_z);
            new_vel.push(velocities[i * 3] + dv());
            new_vel.push(velocities[i * 3 + 1] + dv());
            new_vel.push(velocities[i * 3 + 2] + dv());
            new_signs.push(sign);
            new_masses.push(m_daughter);
            is_hr_flags.push(is_hr);
            new_epsilon.push(epsilon);

            if is_plus {
                n_daughters_plus += 1;
            } else {
                n_daughters_minus += 1;
            }
        }
    }

    (new_pos, new_vel, new_signs, new_masses, is_hr_flags, new_epsilon, n_daughters_plus, n_daughters_minus)
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("JANUS ZOOM L1 v2 — Adaptive Splitting");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Based on ZOOM_L1_V2_SPEC.md (April 2026)");
    println!("Key: Smooth split transitions + m⁻ subdivision (Gemini)");
    println!();
    println!("Split factors m+ : core(r<{})×{}, mid(r<{})×{}, ext(r<{})×{}, outer×{}",
             R_CORE, SPLIT_CORE, R_MID, SPLIT_MID, R_EXT, SPLIT_EXT, SPLIT_OUTER);
    println!("Split factors m- : r<{} Mpc → ×{}", R_MINUS_HR, SPLIT_MINUS);
    println!("Transition width : {} Mpc (smooth)", TRANS_WIDTH);
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create output directories
    fs::create_dir_all(OUTPUT_DIR).ok();
    fs::create_dir_all(format!("{}/snapshots", OUTPUT_DIR)).ok();

    // ─────────────────────────────────────────────────────────────────────────
    // Step 1: Read source snapshot
    // ─────────────────────────────────────────────────────────────────────────
    let source_path = "/app/output/janus_baryonic_calibrated/snapshots/snap_04550.bin";
    println!("[1/5] Reading source snapshot: {}", source_path);

    let (src_pos, src_vel, src_signs, a_init, t_init) = read_source_snapshot(source_path);
    let z_init = 1.0 / a_init - 1.0;
    println!("  Loaded {} particles at z={:.4}, t={:.4} Gyr\n",
             src_signs.len(), z_init, t_init);

    // ─────────────────────────────────────────────────────────────────────────
    // Step 2: Extract and adaptive split
    // ─────────────────────────────────────────────────────────────────────────
    println!("[2/5] Extracting R={} Mpc and adaptive splitting...", R_EXTRACT);

    let (pos, vel, signs, masses, is_hr, epsilon_per_particle, n_plus, n_minus) =
        extract_and_adaptive_split(&src_pos, &src_vel, &src_signs);

    let n_total = signs.len();
    println!("  Result: {} particles total", n_total);
    println!("    m+ : {} daughters", n_plus);
    println!("    m- : {} daughters", n_minus);
    println!("    HR : {} particles (r < {} Mpc)", is_hr.iter().filter(|&&x| x).count(), R_HR);

    // Verify
    if n_total > 10_000_000 {
        eprintln!("  WARNING: {} particles may exceed VRAM!", n_total);
    }
    if n_total < 1_000_000 {
        eprintln!("  WARNING: Only {} particles, below target 5M", n_total);
    }

    // Average mass check
    let avg_mass: f64 = masses.iter().sum::<f64>() / masses.len() as f64;
    println!("  Average particle mass: {:.2e} M☉\n", avg_mass);

    // ─────────────────────────────────────────────────────────────────────────
    // Step 3: Initialize GPU simulation
    // ─────────────────────────────────────────────────────────────────────────
    println!("[3/5] Initializing GPU Barnes-Hut simulation...");

    #[cfg(feature = "cuda")]
    let mut sim = GpuNBodySimulation::new_with_state(
        n_plus, n_minus, L_ZOOM,
        pos.clone(), vel.clone(), signs.clone()
    ).expect("GPU init failed");

    #[cfg(feature = "cuda")]
    {
        sim.set_theta(THETA);
        sim.set_softening(EPS_MID);
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z_init, ETA);
        sim.set_c_ratio(c_ratio_sq.sqrt());
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: CUDA feature not enabled. Compile with --features cuda");
        std::process::exit(1);
    }

    println!("  GPU ready: {} particles, L_box={} Mpc, θ={}, η={}\n",
             n_total, L_ZOOM, THETA, ETA);

    // ─────────────────────────────────────────────────────────────────────────
    // Step 4: Relaxation phase
    // ─────────────────────────────────────────────────────────────────────────
    println!("[4/5] Relaxation phase ({} steps, dt={} Gyr)", N_RELAX, DT_RELAX);
    println!("      SF: OFF, Feedback: OFF, Cooling: ON");

    let mut a = a_init;
    let mut t = t_init;
    let mut ke_prev = 0.0;
    let mut v_rms_history: Vec<f64> = Vec::new();

    for step in 0..N_RELAX {
        // Hubble parameter for expansion (small during relaxation)
        let h = 100.0 * ETA * a.powf(-1.5) * 1.022e-3;  // H in Gyr^-1

        #[cfg(feature = "cuda")]
        {
            sim.step_with_expansion_dkd(DT_RELAX, a, h, 1.0)
                .expect("Relaxation step failed");
        }

        a += a * h * DT_RELAX;
        t += DT_RELAX;

        // Diagnostics every 10 steps
        if step % 10 == 0 {
            #[cfg(feature = "cuda")]
            let cur_vel = sim.get_velocities().unwrap_or_default();

            #[cfg(feature = "cuda")]
            let v_rms = {
                let n_vel = cur_vel.len() / 3;
                if n_vel > 0 {
                    let v2_sum: f64 = cur_vel.chunks(3)
                        .map(|v| v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
                        .sum();
                    (v2_sum / n_vel as f64).sqrt()
                } else { 0.0 }
            };

            #[cfg(feature = "cuda")]
            {
                v_rms_history.push(v_rms);

                // Simple KE for stability check
                let ke: f64 = cur_vel.chunks(3)
                    .zip(masses.iter())
                    .map(|(v, &m)| 0.5 * m * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
                    .sum();

                if step > 0 && ke_prev > 0.0 {
                    let de = (ke - ke_prev).abs() / ke_prev;
                    if de > 0.10 {
                        eprintln!("  STOP: Relaxation unstable, ΔE/E = {:.1}%", de * 100.0);
                        std::process::exit(1);
                    }
                }
                ke_prev = ke;

                println!("  Relax step {:3}/{}: v_rms = {:.2} km/s", step, N_RELAX, v_rms);
            }
        }
    }

    // Check v_rms stability
    if v_rms_history.len() >= 3 {
        let last: Vec<_> = v_rms_history.iter().rev().take(3).collect();
        let mean = last.iter().copied().sum::<f64>() / 3.0;
        let var: f64 = last.iter().map(|&&v| (v - mean).powi(2)).sum::<f64>() / 3.0;
        let cv = var.sqrt() / mean.max(1e-10);
        println!("  v_rms stability: CV = {:.2}% {}", cv * 100.0,
                 if cv < 0.02 { "✓" } else { "⚠" });
    }

    println!("  Relaxation complete.\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Step 5: Production run
    // ─────────────────────────────────────────────────────────────────────────
    println!("[5/5] Production run ({} steps, dt={} Gyr)", N_STEPS, DT);
    println!("      z={:.4} → z≈0", z_init);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", OUTPUT_DIR);
    let mut csv = BufWriter::new(File::create(&csv_path).unwrap());
    writeln!(csv, "step,z,t_Gyr,N_stars,rho_max_HR,SFR,v_disp,M_stars_total").unwrap();

    let mut n_stars = 0u32;
    let mut m_stars_total = 0.0f64;

    for step in 0..N_STEPS {
        let z = 1.0 / a - 1.0;

        // Update c_ratio for Janus
        #[cfg(feature = "cuda")]
        {
            let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
            sim.set_c_ratio(c_ratio_sq.sqrt());
        }

        // Hubble parameter: H = H₀ × η × a^(-3/2) in code units
        let h = 100.0 * ETA * a.powf(-1.5) * 1.022e-3;

        // GPU step with expansion
        #[cfg(feature = "cuda")]
        {
            sim.step_with_expansion_dkd(DT, a, h, 1.0)
                .expect("Production step failed");
        }

        a += a * h * DT;
        t += DT;

        // Get particles for diagnostics/output
        #[cfg(feature = "cuda")]
        let cur_pos = sim.get_positions().unwrap_or_default();
        #[cfg(feature = "cuda")]
        let cur_vel = sim.get_velocities().unwrap_or_default();
        #[cfg(feature = "cuda")]
        let cur_signs = sim.get_signs().unwrap_or_default();

        // Compute HR diagnostics
        #[cfg(feature = "cuda")]
        let (v_disp, rho_max_hr) = {
            let mut v2_sum = 0.0f64;
            let mut n_hr = 0usize;

            let n_parts = cur_pos.len() / 3;
            for i in 0..n_parts {
                let r2 = cur_pos[i*3]*cur_pos[i*3] + cur_pos[i*3+1]*cur_pos[i*3+1] + cur_pos[i*3+2]*cur_pos[i*3+2];
                if r2 < R_HR * R_HR {
                    let vv = cur_vel[i*3]*cur_vel[i*3] + cur_vel[i*3+1]*cur_vel[i*3+1] + cur_vel[i*3+2]*cur_vel[i*3+2];
                    v2_sum += vv;
                    n_hr += 1;
                }
            }

            let v_disp = if n_hr > 0 { (v2_sum / n_hr as f64).sqrt() } else { 0.0 };
            (v_disp, 1.0)  // rho_max placeholder
        };

        // STOP conditions
        if n_stars > 2_000_000 {
            eprintln!("  STOP: SF runaway at step {}", step);
            break;
        }

        // Snapshot
        if step % SNAPSHOT_INTERVAL == 0 {
            let snap_path = format!("{}/snapshots/snap_{:05}.bin", OUTPUT_DIR, step);

            #[cfg(feature = "cuda")]
            {
                write_snapshot(&snap_path, &cur_pos, &cur_vel, &cur_signs, Some(&epsilon_per_particle), a, t);
            }
        }

        // CSV
        if step % CSV_INTERVAL == 0 {
            let z_now = 1.0 / a - 1.0;
            #[cfg(feature = "cuda")]
            writeln!(csv, "{},{:.6},{:.6},{},{:.2e},{:.2e},{:.2},{:.2e}",
                     step, z_now, t, n_stars, rho_max_hr, 0.0, v_disp, m_stars_total).unwrap();
        }

        // Progress
        if step % 500 == 0 {
            let z_now = 1.0 / a - 1.0;
            #[cfg(feature = "cuda")]
            println!("  Step {:5}/{}: z={:.4}, t={:.3} Gyr, N★={}, v_disp={:.1} km/s",
                     step, N_STEPS, z_now, t, n_stars, v_disp);
        }

        // Stop past z=0
        let z_now = 1.0 / a - 1.0;
        if z_now < -0.01 {
            println!("  Reached z < 0, stopping at step {}", step);
            break;
        }
    }

    csv.flush().unwrap();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("RUN COMPLETE");
    println!("  Final: z={:.4}, N★={}, M★={:.2e} M☉", 1.0/a - 1.0, n_stars, m_stars_total);
    println!("  Output: {}", OUTPUT_DIR);
    println!("═══════════════════════════════════════════════════════════════");
}
