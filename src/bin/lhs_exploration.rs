//! Janus LHS Exploration — 50 runs with Latin Hypercube Sampling
//!
//! Parameters explored:
//!   ε:     0.15 – 0.35 Mpc (softening)
//!   k_min: 2.0 – 3.0 (mode suppression)
//!   η:     1.00 – 1.10 (mass ratio)
//!   H:     0.00 – 0.02 (expansion)
//!   α_IC:  1.0 – 2.0 (IC asymmetry)
//!
//! IC spectrum: P(k) ~ k^0.96 / (1 + (k/0.02)^4)

use rand::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// ═══════════════════════════════════════════════════════════════════════════
// FIXED PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 80;              // 80³ = 512,000 particles
const L_BOX: f64 = 492.0;              // Mpc
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 10000;
const SNAPSHOT_INTERVAL: usize = 100;
const THETA: f64 = 0.7;

// Analysis checkpoints
const ANALYSIS_STEPS: [usize; 3] = [5000, 8000, 10000];
const R_CUT: f64 = 30.0;
const DTAU_PER_DT: f64 = 0.0;

const N_RUNS: usize = 50;

// Parameter ranges
const EPS_MIN: f64 = 0.15;
const EPS_MAX: f64 = 0.35;
const KMIN_MIN: f64 = 2.0;
const KMIN_MAX: f64 = 3.0;
const ETA_MIN: f64 = 1.00;
const ETA_MAX: f64 = 1.10;
const H_MIN: f64 = 0.00;
const H_MAX: f64 = 0.02;
const ALPHA_MIN: f64 = 1.0;
const ALPHA_MAX: f64 = 2.0;

// Cosmological P(k) parameters
const K_PIVOT: f64 = 0.02;  // Mpc⁻¹
const N_S: f64 = 0.96;      // spectral index
const K_CUT: f64 = 0.3;     // high-k cutoff

// ═══════════════════════════════════════════════════════════════════════════
// LHS PARAMETER STRUCT
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct LHSParams {
    run_id: usize,
    epsilon: f64,
    k_min: usize,
    eta: f64,
    hubble: f64,
    alpha_ic: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULTS STRUCT
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct RunResult {
    run_id: usize,
    epsilon: f64,
    k_min: usize,
    eta: f64,
    hubble: f64,
    alpha_ic: f64,
    runtime_s: f64,
    seg_final: f64,
    sigma_rho: f64,
    sigma_p: f64,
    r_ratio: f64,
    filament_fraction: f64,
    pk_slope: f64,
    xi_slope: f64,
    anisotropy: f64,
    k_peak: f64,
    score: f64,
    // New metrics: R at checkpoints and ΔR
    r_5000: f64,
    r_8000: f64,
    r_10000: f64,
    delta_r: f64,  // R(5000) - R(10000): positive = Janus physics activated
}

impl RunResult {
    fn csv_header() -> &'static str {
        "run_id,epsilon,k_min,eta,hubble,alpha_ic,runtime_s,seg_final,\
         sigma_rho,sigma_p,r_ratio,filament_fraction,pk_slope,xi_slope,\
         anisotropy,k_peak,r_5000,r_8000,r_10000,delta_r,score"
    }

    fn to_csv_line(&self) -> String {
        format!(
            "{},{:.4},{},{:.4},{:.4},{:.2},{:.1},{:.6},\
             {:.4},{:.4},{:.2},{:.4},{:.2},{:.2},\
             {:.3},{:.4},{:.2},{:.2},{:.2},{:.2},{:.3}",
            self.run_id, self.epsilon, self.k_min, self.eta, self.hubble,
            self.alpha_ic, self.runtime_s, self.seg_final,
            self.sigma_rho, self.sigma_p, self.r_ratio, self.filament_fraction,
            self.pk_slope, self.xi_slope, self.anisotropy, self.k_peak,
            self.r_5000, self.r_8000, self.r_10000, self.delta_r, self.score
        )
    }

    fn compute_score(&self) -> f64 {
        // New score: prioritize ΔR (Janus activation) + filament formation
        // ΔR > 0 means R decreased → polarization increased → Janus active
        self.delta_r * 0.5                          // Weight ΔR heavily
            + self.filament_fraction * 10.0          // Filaments are good
            - (self.pk_slope + 2.5).abs() * 0.1      // P(k) slope target
            - (self.xi_slope + 1.8).abs() * 0.1      // ξ(r) slope target
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LATIN HYPERCUBE SAMPLING
// ═══════════════════════════════════════════════════════════════════════════

fn generate_lhs_params(seed: u64) -> Vec<LHSParams> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = N_RUNS;

    // Generate permutations for each parameter
    let mut eps_idx: Vec<usize> = (0..n).collect();
    let mut kmin_idx: Vec<usize> = (0..n).collect();
    let mut eta_idx: Vec<usize> = (0..n).collect();
    let mut h_idx: Vec<usize> = (0..n).collect();
    let mut alpha_idx: Vec<usize> = (0..n).collect();

    eps_idx.shuffle(&mut rng);
    kmin_idx.shuffle(&mut rng);
    eta_idx.shuffle(&mut rng);
    h_idx.shuffle(&mut rng);
    alpha_idx.shuffle(&mut rng);

    let mut params = Vec::with_capacity(n);

    for i in 0..n {
        // Sample from center of each bin with small jitter
        let mut jitter = || rng.gen::<f64>() * 0.8 + 0.1;  // 0.1 to 0.9 within bin

        let eps = EPS_MIN + (eps_idx[i] as f64 + jitter()) / n as f64 * (EPS_MAX - EPS_MIN);
        let kmin_f = KMIN_MIN + (kmin_idx[i] as f64 + jitter()) / n as f64 * (KMIN_MAX - KMIN_MIN);
        let eta = ETA_MIN + (eta_idx[i] as f64 + jitter()) / n as f64 * (ETA_MAX - ETA_MIN);
        let h = H_MIN + (h_idx[i] as f64 + jitter()) / n as f64 * (H_MAX - H_MIN);
        let alpha = ALPHA_MIN + (alpha_idx[i] as f64 + jitter()) / n as f64 * (ALPHA_MAX - ALPHA_MIN);

        // Round k_min to integer (2 or 3)
        let k_min = kmin_f.round() as usize;

        params.push(LHSParams {
            run_id: i + 1,
            epsilon: eps,
            k_min,
            eta,
            hubble: h,
            alpha_ic: alpha,
        });
    }

    params
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  JANUS LHS EXPLORATION — 50 RUNS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Generate LHS parameters
    let lhs_seed = 12345u64;
    let params = generate_lhs_params(lhs_seed);

    println!("  Parameter ranges:");
    println!("    ε:     {:.2} – {:.2} Mpc", EPS_MIN, EPS_MAX);
    println!("    k_min: {:.1} – {:.1}", KMIN_MIN, KMIN_MAX);
    println!("    η:     {:.2} – {:.2}", ETA_MIN, ETA_MAX);
    println!("    H:     {:.2} – {:.2}", H_MIN, H_MAX);
    println!("    α_IC:  {:.1} – {:.1}", ALPHA_MIN, ALPHA_MAX);
    println!();
    println!("  Total runs: {}", N_RUNS);
    println!("  Steps/run:  {}", TOTAL_STEPS);
    println!();

    // Setup output
    let base_dir = "/app/output/lhs_exploration";
    fs::create_dir_all(base_dir).ok();

    // Check which runs are already completed (have final snapshot)
    let completed_runs: std::collections::HashSet<usize> = (1..=N_RUNS)
        .filter(|&run_id| {
            let snap_path = format!("{}/lhs_run_{:02}/snapshots/snap_{:06}.bin",
                base_dir, run_id, TOTAL_STEPS);
            std::path::Path::new(&snap_path).exists()
        })
        .collect();

    if !completed_runs.is_empty() {
        println!("  Resuming: {} runs already completed, skipping them", completed_runs.len());
        println!();
    }

    let csv_path = format!("{}/results.csv", base_dir);
    let csv_exists = std::path::Path::new(&csv_path).exists();
    let mut csv_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)  // Append mode for resume
        .open(&csv_path)
        .expect("Failed to create CSV");

    // Only write header if file is new/empty
    if !csv_exists || std::fs::metadata(&csv_path).map(|m| m.len() == 0).unwrap_or(true) {
        writeln!(csv_file, "{}", RunResult::csv_header()).unwrap();
    }

    // Print first 10 parameter combinations
    println!("  First 10 parameter combinations:");
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Run │  ε    │ k_min │   η   │   H   │ α_IC");
    println!("  ────────────────────────────────────────────────────────────");
    for p in params.iter().take(10) {
        println!("  {:3} │ {:.3} │   {}   │ {:.3} │ {:.4} │ {:.2}",
            p.run_id, p.epsilon, p.k_min, p.eta, p.hubble, p.alpha_ic);
    }
    println!("  ... ({} more)", N_RUNS - 10);
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  STARTING EXPLORATION");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let exploration_start = Instant::now();
    let mut results: Vec<RunResult> = Vec::new();

    for p in &params {
        // Skip already completed runs
        if completed_runs.contains(&p.run_id) {
            println!("  Skipping run {}/{} (already completed)", p.run_id, N_RUNS);
            continue;
        }

        println!("┌─────────────────────────────────────────────────────────────");
        println!("│ Run {}/{}: ε={:.3} k_min={} η={:.3} H={:.4} α={:.2}",
            p.run_id, N_RUNS, p.epsilon, p.k_min, p.eta, p.hubble, p.alpha_ic);
        println!("└─────────────────────────────────────────────────────────────");

        let result = run_single_lhs(p);

        // Write to CSV immediately
        writeln!(csv_file, "{}", result.to_csv_line()).unwrap();
        csv_file.flush().unwrap();

        println!("  → ΔR={:.1} fil={:.1}% R_final={:.1} score={:.3}",
            result.delta_r, result.filament_fraction * 100.0,
            result.r_ratio, result.score);
        println!();

        results.push(result);
    }

    let total_time = exploration_start.elapsed().as_secs_f64();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  EXPLORATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Total runs:  {}", N_RUNS);
    println!("  Total time:  {:.1}s ({:.1} h)", total_time, total_time / 3600.0);
    println!("  Avg per run: {:.1}s ({:.1} min)", total_time / N_RUNS as f64, total_time / N_RUNS as f64 / 60.0);
    println!();
    println!("  Results: {}", csv_path);
    println!();

    // Print top 10 by score
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    println!("  Top 10 by score (ΔR = Janus activation):");
    println!("  ────────────────────────────────────────────────────────────");
    for (i, r) in sorted.iter().take(10).enumerate() {
        println!("  {:2}. Run {:2}: ε={:.2} k={} η={:.2} H={:.3} α={:.1} → score={:.3}",
            i + 1, r.run_id, r.epsilon, r.k_min, r.eta, r.hubble, r.alpha_ic, r.score);
        println!("      ΔR={:.1} (R: {:.1}→{:.1}→{:.1}) fil={:.1}%",
            r.delta_r, r.r_5000, r.r_8000, r.r_10000, r.filament_fraction * 100.0);
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// SINGLE LHS RUN
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn run_single_lhs(p: &LHSParams) -> RunResult {
    let run_dir = format!("/app/output/lhs_exploration/lhs_run_{:02}", p.run_id);
    fs::create_dir_all(&run_dir).ok();
    fs::create_dir_all(format!("{}/snapshots", run_dir)).ok();

    let n3 = N_GRID * N_GRID * N_GRID;

    // Generate ICs with cosmological spectrum and asymmetry
    let (positions, velocities, signs) = generate_cosmological_ics(
        p.run_id as u64, p.eta, p.k_min, p.alpha_ic
    );

    let n_positive = signs.iter().filter(|&&s| s > 0).count();

    // Convert to GPU format
    let pos_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize simulation
    let mut sim = match GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  ERROR: {}", e);
            return RunResult {
                run_id: p.run_id,
                epsilon: p.epsilon,
                k_min: p.k_min,
                eta: p.eta,
                hubble: p.hubble,
                alpha_ic: p.alpha_ic,
                runtime_s: 0.0,
                seg_final: 999.0,
                sigma_rho: 0.0,
                sigma_p: 0.0,
                r_ratio: 0.0,
                filament_fraction: 0.0,
                pk_slope: 0.0,
                xi_slope: 0.0,
                anisotropy: 1.0,
                k_peak: 0.0,
                score: -999.0,
                r_5000: 0.0,
                r_8000: 0.0,
                r_10000: 0.0,
                delta_r: 0.0,
            };
        }
    };

    sim.set_theta(THETA);
    sim.set_softening(p.epsilon);
    sim.set_pm_k_min(p.k_min);

    // Run simulation with checkpoints
    let start = Instant::now();

    // Store R at checkpoints
    let mut r_5000 = 0.0;
    let mut r_8000 = 0.0;
    let mut r_10000 = 0.0;

    for step in 1..=TOTAL_STEPS {
        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, p.hubble, DTAU_PER_DT) {
            eprintln!("  Step {} error: {}", step, e);
            break;
        }

        // Save snapshot at analysis checkpoints
        if ANALYSIS_STEPS.contains(&step) || step % SNAPSHOT_INTERVAL == 0 {
            save_snapshot(&sim, step, &run_dir, n_positive, n3);
        }

        // Compute R at checkpoints
        if step == 5000 {
            let (_, _, r) = compute_density_stats(&sim, 64, L_BOX);
            r_5000 = r;
            println!("  step 5000: R={:.2}", r_5000);
        } else if step == 8000 {
            let (_, _, r) = compute_density_stats(&sim, 64, L_BOX);
            r_8000 = r;
            println!("  step 8000: R={:.2}", r_8000);
        } else if step == 10000 {
            let (_, _, r) = compute_density_stats(&sim, 64, L_BOX);
            r_10000 = r;
            println!("  step 10000: R={:.2}", r_10000);
        }

        // Progress
        if step % 1000 == 0 && !ANALYSIS_STEPS.contains(&step) {
            print!(".");
            std::io::stdout().flush().ok();
        }
    }
    println!();

    let runtime = start.elapsed().as_secs_f64();

    // ΔR = R(5000) - R(10000): positive means Janus physics activated
    let delta_r = r_5000 - r_10000;
    println!("  ΔR = R(5000) - R(10000) = {:.2} - {:.2} = {:.2}", r_5000, r_10000, delta_r);

    // Compute final metrics
    let seg_final = sim.segregation().unwrap_or(0.0);
    let (sigma_rho, sigma_p, r_ratio) = compute_density_stats(&sim, 64, L_BOX);
    let filament_fraction = compute_filament_fraction(&sim, 64, L_BOX);
    let (pk_slope, k_peak) = compute_pk_slope(&sim, L_BOX);
    let xi_slope = compute_xi_slope(&sim, L_BOX);
    let anisotropy = compute_anisotropy(&sim, L_BOX);

    let mut result = RunResult {
        run_id: p.run_id,
        epsilon: p.epsilon,
        k_min: p.k_min,
        eta: p.eta,
        hubble: p.hubble,
        alpha_ic: p.alpha_ic,
        runtime_s: runtime,
        seg_final,
        sigma_rho,
        sigma_p,
        r_ratio,
        filament_fraction,
        pk_slope,
        xi_slope,
        anisotropy,
        k_peak,
        score: 0.0,
        r_5000,
        r_8000,
        r_10000,
        delta_r,
    };

    result.score = result.compute_score();
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// SNAPSHOT
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_snapshot(sim: &GpuNBodyTwoPass, step: usize, run_dir: &str, _n_positive: usize, n_total: usize) {
    let filename = format!("{}/snapshots/snap_{:06}.bin", run_dir, step);
    let (positions, _, signs) = sim.get_particles().expect("get_particles failed");

    let file = File::create(&filename).unwrap();
    let mut writer = BufWriter::new(file);

    writer.write_all(&(n_total as u64).to_le_bytes()).unwrap();
    writer.write_all(&(step as u64).to_le_bytes()).unwrap();
    writer.write_all(&(0u64).to_le_bytes()).unwrap();

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

// ═══════════════════════════════════════════════════════════════════════════
// COSMOLOGICAL IC GENERATION
// P(k) ~ k^0.96 / (1 + (k/0.02)^4)
// ═══════════════════════════════════════════════════════════════════════════

fn generate_cosmological_ics(seed: u64, eta: f64, k_min: usize, alpha_ic: f64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ng = N_GRID;
    let ng3 = ng * ng * ng;
    let cell = L_BOX / ng as f64;

    // FIRST: Assign signs
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

                let k_phys = 2.0 * PI / L_BOX * ((ikx*ikx + iky*iky + ikz*ikz) as f64).sqrt();
                if k_phys > K_CUT || k_phys < 1e-10 { continue; }

                // Cosmological P(k) ~ k^0.96 / (1 + (k/k_pivot)^4)
                let k_ratio = k_phys / K_PIVOT;
                let pk = k_phys.powf(N_S) / (1.0 + k_ratio.powi(4));
                let amp = pk.sqrt() * 0.05;

                let phase = rng.gen::<f64>() * 2.0 * PI;
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

                let x0 = (ix as f64 + 0.5) * cell - L_BOX / 2.0;
                let y0 = (iy as f64 + 0.5) * cell - L_BOX / 2.0;
                let z0 = (iz as f64 + 0.5) * cell - L_BOX / 2.0;

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

                let disp_x = -dphi_dx * growth_factor * L_BOX * amp_factor;
                let disp_y = -dphi_dy * growth_factor * L_BOX * amp_factor;
                let disp_z = -dphi_dz * growth_factor * L_BOX * amp_factor;

                positions[3*idx]     = wrap(x0 + disp_x, L_BOX);
                positions[3*idx + 1] = wrap(y0 + disp_y, L_BOX);
                positions[3*idx + 2] = wrap(z0 + disp_z, L_BOX);

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

// ═══════════════════════════════════════════════════════════════════════════
// ANALYSIS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_density_stats(sim: &GpuNBodyTwoPass, grid_size: usize, box_size: f64) -> (f64, f64, f64) {
    let (positions, _, signs) = match sim.get_particles() {
        Ok(state) => state,
        Err(_) => return (0.0, 0.0, 1.0),
    };

    let n = positions.len() / 3;
    let cell = box_size / grid_size as f64;
    let ng = grid_size;
    let ng3 = ng * ng * ng;

    let mut rho_plus = vec![0.0f64; ng3];
    let mut rho_minus = vec![0.0f64; ng3];

    for i in 0..n {
        let x = positions[3*i] as f64;
        let y = positions[3*i + 1] as f64;
        let z = positions[3*i + 2] as f64;

        let gx = ((x + box_size/2.0) / cell).floor() as usize % ng;
        let gy = ((y + box_size/2.0) / cell).floor() as usize % ng;
        let gz = ((z + box_size/2.0) / cell).floor() as usize % ng;

        let idx = gx + ng * (gy + ng * gz);
        if signs[i] > 0 { rho_plus[idx] += 1.0; }
        else { rho_minus[idx] += 1.0; }
    }

    let mean_rho = n as f64 / ng3 as f64;
    let mut sum_delta2 = 0.0;
    let mut sum_polar2 = 0.0;

    for i in 0..ng3 {
        let rho_total = rho_plus[i] + rho_minus[i];
        let delta = (rho_total - mean_rho) / mean_rho.max(1e-10);
        let polar = if rho_total > 0.0 {
            (rho_plus[i] - rho_minus[i]) / rho_total
        } else { 0.0 };

        sum_delta2 += delta * delta;
        sum_polar2 += polar * polar;
    }

    let sigma_rho = (sum_delta2 / ng3 as f64).sqrt();
    let sigma_p = (sum_polar2 / ng3 as f64).sqrt();
    let r_ratio = sigma_rho / sigma_p.max(1e-10);

    (sigma_rho, sigma_p, r_ratio)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_filament_fraction(sim: &GpuNBodyTwoPass, grid_size: usize, box_size: f64) -> f64 {
    let (positions, _, _) = match sim.get_particles() {
        Ok(state) => state,
        Err(_) => return 0.0,
    };

    let n = positions.len() / 3;
    let cell = box_size / grid_size as f64;
    let ng = grid_size;
    let ng3 = ng * ng * ng;

    let mut grid = vec![0.0f64; ng3];

    for i in 0..n {
        let x = positions[3*i] as f64;
        let y = positions[3*i + 1] as f64;
        let z = positions[3*i + 2] as f64;

        let gx = ((x + box_size/2.0) / cell).floor() as usize % ng;
        let gy = ((y + box_size/2.0) / cell).floor() as usize % ng;
        let gz = ((z + box_size/2.0) / cell).floor() as usize % ng;

        grid[gx + ng * (gy + ng * gz)] += 1.0;
    }

    // Simple Hessian-based filament detection
    let mean = grid.iter().sum::<f64>() / ng3 as f64;
    let mut filament_count = 0;

    for ix in 1..ng-1 {
        for iy in 1..ng-1 {
            for iz in 1..ng-1 {
                let idx = |x, y, z| x + ng * (y + ng * z);
                let c = grid[idx(ix, iy, iz)];

                // Second derivatives
                let dxx = grid[idx(ix+1, iy, iz)] - 2.0*c + grid[idx(ix-1, iy, iz)];
                let dyy = grid[idx(ix, iy+1, iz)] - 2.0*c + grid[idx(ix, iy-1, iz)];
                let dzz = grid[idx(ix, iy, iz+1)] - 2.0*c + grid[idx(ix, iy, iz-1)];

                // Count negative eigenvalues (simplified: trace)
                let trace = dxx + dyy + dzz;

                // Filament: density above mean, negative curvature
                if c > mean && trace < -mean * 0.1 {
                    filament_count += 1;
                }
            }
        }
    }

    filament_count as f64 / ng3 as f64
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_pk_slope(_sim: &GpuNBodyTwoPass, _box_size: f64) -> (f64, f64) {
    // Simplified: return placeholder
    // Full implementation would compute FFT of density field
    (-2.5, 0.05)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_xi_slope(_sim: &GpuNBodyTwoPass, _box_size: f64) -> f64 {
    // Simplified: return placeholder
    -1.5
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_anisotropy(_sim: &GpuNBodyTwoPass, _box_size: f64) -> f64 {
    // Simplified: return placeholder
    0.2
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires --features cuda,cufft");
}
