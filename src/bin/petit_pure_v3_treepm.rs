//! Petit Pure v3 — TreePM Validation Run
//!
//! Validates that corrected TreePM (erfc splitting) reproduces v2 results
//! without k=8 octree resonance artifact.
//!
//! - N = 2M (μ=8: 222k m+, 1.78M m-)
//! - Box = 500 Mpc
//! - TreePM: PM 256³, r_cut = 20 Mpc
//! - λ = 0 (pure anti-Newton 1/r²)
//! - 2000 steps, snapshots every 10

use rand::prelude::*;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

const MU: f64 = 8.0;
const N_TOTAL: usize = 2_000_000;
const BOX_SIZE: f64 = 500.0;
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.005;
const STEPS: usize = 2000;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.01;
const ETA: f64 = 1.0;  // Pure Petit (no Janus correction)
const SEED: u64 = 42;
const SNAPSHOT_INTERVAL: usize = 10;
const R_CUT: f64 = 20.0;  // TreePM cutoff

// Check k=8 at this step (z≈3.5)
const K8_CHECK_STEP: usize = 125;

#[cfg(feature = "cuda")]
fn main() {
    println!("================================================================");
    println!("  Petit Pure v3 — TreePM Validation (erfc splitting)");
    println!("================================================================");
    println!("  Validating TreePM eliminates k=8 artifact while preserving physics");
    println!("================================================================");

    let n_positive = (N_TOTAL as f64 / (1.0 + MU)) as usize;
    let n_negative = N_TOTAL - n_positive;

    println!("  N_total = {} (2M)", N_TOTAL);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_TOTAL as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_TOTAL as f64);
    println!("  μ = N⁻/N⁺ = {:.2}", n_negative as f64 / n_positive as f64);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  PM Grid = 256³");
    println!("  r_cut = {} Mpc (TreePM)", R_CUT);
    println!("  λ = 0 (pure anti-Newton)");
    println!("  Steps = {}", STEPS);
    println!("================================================================");
    println!();

    // Generate uniform random ICs
    println!("Generating uniform random ICs...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let half_box = BOX_SIZE / 2.0;

    let mut pos_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut vel_f32: Vec<f32> = Vec::with_capacity(N_TOTAL * 3);
    let mut signs_i8: Vec<i8> = Vec::with_capacity(N_TOTAL);

    for _ in 0..n_positive {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(1);
    }

    for _ in 0..n_negative {
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        pos_f32.push((rng.random::<f64>() * BOX_SIZE - half_box) as f32);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        vel_f32.push(0.0);
        signs_i8.push(-1);
    }

    println!("  Generated {} particles", N_TOTAL);

    // Setup output
    let base_dir = std::path::Path::new("/app/output/petit_pure_v3_treepm");
    let snap_dir = base_dir.join("snapshots");
    fs::create_dir_all(&snap_dir).expect("Failed to create output dir");

    let mut ts_file = BufWriter::new(
        File::create(base_dir.join("time_series.csv")).expect("Failed to create CSV")
    );
    writeln!(ts_file, "step,z,a,P,void_frac,wall_frac").unwrap();

    // Initialize simulation
    println!("Initializing GPU simulation...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        pos_f32, vel_f32, signs_i8, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);

    println!("  θ = {} (Barnes-Hut opening angle)", THETA);
    println!("  λ₀ = 0.0 (pure anti-Newton)");

    // Cosmology
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start) / (STEPS as f64 * DT);

    let start = Instant::now();

    println!();
    println!("Starting TreePM evolution (r_cut={} Mpc)...", R_CUT);
    println!();

    let mut k8_check_passed = false;

    for step in 0..=STEPS {
        let tau = cosmo.tau_start + (step as f64) * DT * dtau_per_dt;
        let (a, h) = if tau <= cosmo.tau_end {
            cosmo.get_params_at_tau(tau)
        } else {
            (1.0, 0.0)
        };
        let z = if a > 0.0 { (1.0 / a - 1.0).max(0.0) } else { 0.0 };

        if step > 0 {
            sim.set_current_z(z);
            sim.step_treepm_gpu(DT, R_CUT, h, dtau_per_dt)
                .expect("TreePM step failed");
        }

        if step % SNAPSHOT_INTERVAL == 0 {
            let purity = sim.local_purity(32).unwrap_or(0.0);
            let (void_frac, wall_frac) = compute_void_wall_fractions(&sim);

            writeln!(ts_file, "{},{:.4},{:.6},{:.4},{:.4},{:.4}",
                     step, z, a, purity, void_frac, wall_frac).unwrap();

            let elapsed = start.elapsed().as_secs_f64();
            let rate = if step > 0 { step as f64 / elapsed } else { 0.0 };
            let eta_sec = if rate > 0.0 { (STEPS - step) as f64 / rate } else { 0.0 };

            println!("  step {:4} | z={:.2} | P={:.3} | void={:.1}% | wall={:.1}% | {:.1}s ({:.2} step/s) ETA {:.0}s",
                     step, z, purity, void_frac * 100.0, wall_frac * 100.0, elapsed, rate, eta_sec);

            save_snapshot(&sim, &snap_dir, step, z);

            // K=8 check at step 125
            if step == K8_CHECK_STEP {
                println!();
                println!("  ========== K=8 CHECK AT STEP {} (z={:.2}) ==========", step, z);
                k8_check_passed = check_k8_artifact(&sim, &snap_dir, step);
                println!("  =====================================================");
                println!();
            }
        }
    }

    ts_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();

    // Final validation
    println!();
    println!("================================================================");
    println!("  VALIDATION RESULTS");
    println!("================================================================");

    let final_purity = sim.local_purity(32).unwrap_or(0.0);
    let (void_frac, wall_frac) = compute_void_wall_fractions(&sim);

    println!("  Final Purity (P):     {:.4}", final_purity);
    println!("  Void Fraction:        {:.1}%", void_frac * 100.0);
    println!("  Wall Fraction:        {:.1}%", wall_frac * 100.0);
    println!("  K=8 Check at step {}: {}", K8_CHECK_STEP, if k8_check_passed { "PASS" } else { "FAIL" });
    println!();

    // Check criteria
    let p_ok = final_purity > 0.95;
    let void_ok = void_frac > 0.30;

    println!("  Criteria:");
    println!("    P > 0.95:        {} ({})", if p_ok { "PASS" } else { "FAIL" }, final_purity);
    println!("    void > 30%:      {} ({:.1}%)", if void_ok { "PASS" } else { "FAIL" }, void_frac * 100.0);
    println!("    k=8 spike < 10×: {}", if k8_check_passed { "PASS" } else { "FAIL" });
    println!();

    let all_pass = p_ok && void_ok && k8_check_passed;

    if all_pass {
        println!("  ✓ ALL CRITERIA PASSED — TreePM validation successful!");
        println!("  → Ready to launch 20M TreePM production run");
    } else {
        println!("  ✗ VALIDATION FAILED — Check results above");
    }

    println!();
    println!("  Runtime: {:.1}s ({:.2} step/s)", elapsed, STEPS as f64 / elapsed);
    println!("  Output: {:?}", base_dir);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn compute_void_wall_fractions(sim: &GpuNBodyTwoPass) -> (f64, f64) {
    // Get particle data
    let (positions, _, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return (0.0, 0.0),
    };

    let n = signs.len();
    let n_grid = 32usize;
    let cell_size = BOX_SIZE / n_grid as f64;
    let half_box = BOX_SIZE / 2.0;

    // Count particles per cell
    let mut pos_counts = vec![0u32; n_grid * n_grid * n_grid];
    let mut neg_counts = vec![0u32; n_grid * n_grid * n_grid];

    for i in 0..n {
        let x = ((positions[i * 3] as f64 + half_box) / cell_size) as usize;
        let y = ((positions[i * 3 + 1] as f64 + half_box) / cell_size) as usize;
        let z = ((positions[i * 3 + 2] as f64 + half_box) / cell_size) as usize;

        let x = x.min(n_grid - 1);
        let y = y.min(n_grid - 1);
        let z = z.min(n_grid - 1);

        let idx = x * n_grid * n_grid + y * n_grid + z;

        if signs[i] > 0 {
            pos_counts[idx] += 1;
        } else {
            neg_counts[idx] += 1;
        }
    }

    // Classify cells
    let mut n_void = 0;
    let mut n_wall = 0;
    let total_cells = n_grid * n_grid * n_grid;

    for idx in 0..total_cells {
        let total = pos_counts[idx] + neg_counts[idx];
        if total == 0 {
            n_void += 1;
        } else {
            let pos_frac = pos_counts[idx] as f64 / total as f64;
            // Wall = mixed cell (neither pure positive nor pure negative)
            if pos_frac > 0.1 && pos_frac < 0.9 {
                n_wall += 1;
            }
        }
    }

    let void_frac = n_void as f64 / total_cells as f64;
    let wall_frac = n_wall as f64 / total_cells as f64;

    (void_frac, wall_frac)
}

#[cfg(feature = "cuda")]
fn check_k8_artifact(sim: &GpuNBodyTwoPass, snap_dir: &std::path::PathBuf, step: usize) -> bool {
    use std::collections::HashMap;

    // Get particle positions
    let (positions, _, _signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => {
            println!("    ERROR: Could not get particle data");
            return false;
        }
    };

    let n = positions.len() / 3;
    let n_grid = 64usize;
    let cell_size = BOX_SIZE / n_grid as f64;
    let half_box = BOX_SIZE / 2.0;

    // Compute density field
    let mut density = vec![0.0f64; n_grid * n_grid * n_grid];

    for i in 0..n {
        let x = ((positions[i * 3] as f64 + half_box) / cell_size) as usize;
        let y = ((positions[i * 3 + 1] as f64 + half_box) / cell_size) as usize;
        let z = ((positions[i * 3 + 2] as f64 + half_box) / cell_size) as usize;

        let x = x.min(n_grid - 1);
        let y = y.min(n_grid - 1);
        let z = z.min(n_grid - 1);

        let idx = x * n_grid * n_grid + y * n_grid + z;
        density[idx] += 1.0;
    }

    // Convert to contrast
    let mean_density: f64 = density.iter().sum::<f64>() / density.len() as f64;
    if mean_density > 0.0 {
        for d in density.iter_mut() {
            *d = (*d - mean_density) / mean_density;
        }
    }

    // Simple FFT power spectrum (3D)
    // For simplicity, compute power at axial modes only
    let mut power_axial = HashMap::new();

    // DFT along each axis for axial modes (k, 0, 0), (0, k, 0), (0, 0, k)
    for k in 1..n_grid/2 {
        let mut p_x = 0.0f64;
        let mut p_y = 0.0f64;
        let mut p_z = 0.0f64;

        // Mode (k, 0, 0)
        let mut re = 0.0;
        let mut im = 0.0;
        for ix in 0..n_grid {
            let phase = -2.0 * std::f64::consts::PI * (k * ix) as f64 / n_grid as f64;
            for iy in 0..n_grid {
                for iz in 0..n_grid {
                    let idx = ix * n_grid * n_grid + iy * n_grid + iz;
                    re += density[idx] * phase.cos();
                    im += density[idx] * phase.sin();
                }
            }
        }
        p_x = re * re + im * im;

        // Mode (0, k, 0)
        re = 0.0;
        im = 0.0;
        for iy in 0..n_grid {
            let phase = -2.0 * std::f64::consts::PI * (k * iy) as f64 / n_grid as f64;
            for ix in 0..n_grid {
                for iz in 0..n_grid {
                    let idx = ix * n_grid * n_grid + iy * n_grid + iz;
                    re += density[idx] * phase.cos();
                    im += density[idx] * phase.sin();
                }
            }
        }
        p_y = re * re + im * im;

        // Mode (0, 0, k)
        re = 0.0;
        im = 0.0;
        for iz in 0..n_grid {
            let phase = -2.0 * std::f64::consts::PI * (k * iz) as f64 / n_grid as f64;
            for ix in 0..n_grid {
                for iy in 0..n_grid {
                    let idx = ix * n_grid * n_grid + iy * n_grid + iz;
                    re += density[idx] * phase.cos();
                    im += density[idx] * phase.sin();
                }
            }
        }
        p_z = re * re + im * im;

        power_axial.insert(k, (p_x + p_y + p_z) / 3.0);
    }

    // Get k=8 power and neighbors
    let p_k8 = *power_axial.get(&8).unwrap_or(&0.0);
    let p_k7 = *power_axial.get(&7).unwrap_or(&1.0);
    let p_k9 = *power_axial.get(&9).unwrap_or(&1.0);
    let p_neighbors = (p_k7 + p_k9) / 2.0;

    let k8_ratio = if p_neighbors > 0.0 { p_k8 / p_neighbors } else { 0.0 };

    // Also compute diagonal power for comparison
    // Diagonal mode (5, 5, 4) has |k| ≈ 8.1
    let mut p_diag = 0.0;
    let diag_modes = [(5, 5, 4), (5, 4, 5), (4, 5, 5), (6, 6, 0), (6, 0, 6), (0, 6, 6)];

    for (kx, ky, kz) in diag_modes.iter() {
        let mut re = 0.0;
        let mut im = 0.0;
        for ix in 0..n_grid {
            let phase_x = -2.0 * std::f64::consts::PI * (*kx * ix) as f64 / n_grid as f64;
            for iy in 0..n_grid {
                let phase_y = -2.0 * std::f64::consts::PI * (*ky * iy) as f64 / n_grid as f64;
                for iz in 0..n_grid {
                    let phase_z = -2.0 * std::f64::consts::PI * (*kz * iz) as f64 / n_grid as f64;
                    let phase = phase_x + phase_y + phase_z;
                    let idx = ix * n_grid * n_grid + iy * n_grid + iz;
                    re += density[idx] * phase.cos();
                    im += density[idx] * phase.sin();
                }
            }
        }
        p_diag += re * re + im * im;
    }
    p_diag /= diag_modes.len() as f64;

    let axial_diag_ratio = if p_diag > 0.0 { p_k8 / p_diag } else { 0.0 };

    println!("    k=8 axial power:     {:.2e}", p_k8);
    println!("    k=7,9 neighbor avg:  {:.2e}", p_neighbors);
    println!("    k=8 spike ratio:     {:.1}×", k8_ratio);
    println!("    Diagonal power:      {:.2e}", p_diag);
    println!("    Axial/Diagonal:      {:.1}", axial_diag_ratio);

    // Success criteria: k=8 spike < 10× AND axial/diagonal > 0.3
    let spike_ok = k8_ratio < 10.0;
    let isotropy_ok = axial_diag_ratio > 0.3 && axial_diag_ratio < 3.0;

    println!();
    println!("    k=8 spike < 10×:     {} ({:.1}×)", if spike_ok { "PASS" } else { "FAIL" }, k8_ratio);
    println!("    0.3 < axial/diag < 3: {} ({:.1})", if isotropy_ok { "PASS" } else { "FAIL" }, axial_diag_ratio);

    spike_ok && isotropy_ok
}

#[cfg(feature = "cuda")]
fn save_snapshot(sim: &GpuNBodyTwoPass, path: &std::path::PathBuf, step: usize, z: f64) {
    let (positions, velocities, signs) = match sim.get_particles() {
        Ok(data) => data,
        Err(_) => return,
    };

    let n = signs.len();
    let snap_path = path.join(format!("snap_{:05}.bin", step));

    let file = match File::create(&snap_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut writer = BufWriter::new(file);

    let _ = writer.write_all(&(n as u32).to_le_bytes());
    let _ = writer.write_all(&(BOX_SIZE as f32).to_le_bytes());
    let _ = writer.write_all(&(step as u32).to_le_bytes());
    let _ = writer.write_all(&(z as f32).to_le_bytes());

    for i in 0..n {
        let _ = writer.write_all(&positions[i*3].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+1].to_le_bytes());
        let _ = writer.write_all(&positions[i*3+2].to_le_bytes());
        let _ = writer.write_all(&velocities[i*3].to_le_bytes());
        let _ = writer.write_all(&velocities[i*3+1].to_le_bytes());
        let _ = writer.write_all(&velocities[i*3+2].to_le_bytes());
        let _ = writer.write_all(&(signs[i] as i8).to_le_bytes());
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda cufft");
    std::process::exit(1);
}
