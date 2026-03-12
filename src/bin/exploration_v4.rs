//! Janus Exploration Run — Protocol v4
//!
//! Full parameter sweep exploration pipeline:
//!   1. Systematically explore parameter space
//!   2. Record all results to CSV
//!   3. AI analysis selects promising configurations
//!   4. Production runs on selected configurations
//!
//! Usage:
//!   cargo run --release --features cuda,cufft --bin exploration_v4
//!
//! Output:
//!   results/exploration_results.csv

use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// ═══════════════════════════════════════════════════════════════════════════
// FIXED PARAMETERS (not explored)
// ═══════════════════════════════════════════════════════════════════════════

const N_GRID: usize = 80;              // 80³ = 512,000 particles
const L_BOX: f64 = 492.0;              // Mpc
const Z_INIT: f64 = 5.0;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 1000;       // Reduced for exploration (was 5000)
const SNAPSHOT_INTERVAL: usize = 100;  // Less frequent snapshots
const THETA: f64 = 0.7;                // Barnes-Hut opening angle
const R_CUT: f64 = 30.0;               // TreePM split scale
const DTAU_PER_DT: f64 = 0.0;
const TWEB_GRID: usize = 64;

// P(k) IC parameters
const K_CUT: f64 = 0.25;               // Mpc⁻¹
const PK_INDEX: f64 = -2.0;            // P(k) ∝ k^n
const AMPLITUDE: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETER GRID FOR EXPLORATION
// ═══════════════════════════════════════════════════════════════════════════

// Softening lengths to explore (protocol: ε ∈ {0.15, 0.30, 0.40})
const EPSILON_VALUES: &[f64] = &[0.15, 0.30, 0.40];

// PM k-space filter thresholds to explore
const K_MIN_VALUES: &[usize] = &[2, 3, 4];

// Hubble damping values to explore (protocol: H ∈ {0, 0.01, 0.02})
const HUBBLE_VALUES: &[f64] = &[0.0, 0.01, 0.02];

// Population ratio values to explore
const ETA_VALUES: &[f64] = &[1.0, 1.045, 1.10];

// ═══════════════════════════════════════════════════════════════════════════
// RESULT RECORDING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct ExplorationResult {
    run_id: String,
    n_particles: usize,
    epsilon: f64,
    k_min: usize,
    hubble: f64,
    eta: f64,
    runtime_s: f64,
    steps: usize,
    ms_per_step: f64,
    seg_final: f64,
    seg_max: f64,
    dipole_suppressed: bool,
    sigma_rho: f64,
    sigma_p: f64,
    r_ratio: f64,
    void_fraction: f64,
    sheet_fraction: f64,
    filament_fraction: f64,
    node_fraction: f64,
    ke_final: f64,
    pe_final: f64,
    virial_ratio: f64,
    early_stop_reason: String,
}

impl ExplorationResult {
    fn csv_header() -> &'static str {
        "run_id,n_particles,epsilon,k_min,hubble,eta,runtime_s,steps,ms_per_step,\
         seg_final,seg_max,dipole_suppressed,sigma_rho,sigma_p,r_ratio,\
         void_fraction,sheet_fraction,filament_fraction,node_fraction,\
         ke_final,pe_final,virial_ratio,early_stop_reason"
    }

    fn to_csv_line(&self) -> String {
        format!(
            "{},{},{:.4},{},{:.4},{:.4},{:.1},{},{:.1},\
             {:.6},{:.6},{},{:.6},{:.6},{:.4},\
             {:.4},{:.4},{:.4},{:.4},\
             {:.6e},{:.6e},{:.4},{}",
            self.run_id, self.n_particles, self.epsilon, self.k_min,
            self.hubble, self.eta, self.runtime_s, self.steps, self.ms_per_step,
            self.seg_final, self.seg_max, self.dipole_suppressed,
            self.sigma_rho, self.sigma_p, self.r_ratio,
            self.void_fraction, self.sheet_fraction, self.filament_fraction, self.node_fraction,
            self.ke_final, self.pe_final, self.virial_ratio, self.early_stop_reason
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPLORATION LOOP
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  JANUS EXPLORATION PIPELINE — Protocol v4");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Calculate total runs
    let total_runs = EPSILON_VALUES.len() * K_MIN_VALUES.len() *
                     HUBBLE_VALUES.len() * ETA_VALUES.len();

    println!("  Parameter grid:");
    println!("    ε:     {:?}", EPSILON_VALUES);
    println!("    k_min: {:?}", K_MIN_VALUES);
    println!("    H:     {:?}", HUBBLE_VALUES);
    println!("    η:     {:?}", ETA_VALUES);
    println!();
    println!("  Total configurations: {}", total_runs);
    println!("  Steps per run: {}", TOTAL_STEPS);
    println!("  N: {}³ = {} particles", N_GRID, N_GRID * N_GRID * N_GRID);
    println!();

    // Setup results CSV
    let results_dir = "/app/results";
    fs::create_dir_all(results_dir).ok();
    let csv_path = format!("{}/exploration_results.csv", results_dir);

    // Write header if file doesn't exist
    let write_header = !Path::new(&csv_path).exists();
    let mut csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)
        .expect("Failed to open results CSV");

    if write_header {
        writeln!(csv_file, "{}", ExplorationResult::csv_header()).unwrap();
    }

    println!("  Results: {}", csv_path);
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STARTING EXPLORATION");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let exploration_start = Instant::now();
    let mut run_count = 0;
    let mut valid_count = 0;
    let mut results: Vec<ExplorationResult> = Vec::new();

    // Parameter sweep - ALL combinations
    for &epsilon in EPSILON_VALUES {
        for &k_min in K_MIN_VALUES {
            for &hubble in HUBBLE_VALUES {
                for &eta in ETA_VALUES {
                    run_count += 1;

                    println!("┌─────────────────────────────────────────────────────────────");
                    println!("│ Run {}/{}: ε={:.2}, k_min={}, H={:.2}, η={:.3}",
                        run_count, total_runs, epsilon, k_min, hubble, eta);
                    println!("└─────────────────────────────────────────────────────────────");

                    // Run single exploration
                    let result = run_single_exploration(
                        epsilon, k_min, hubble, eta, run_count
                    );

                    // Record result
                    if result.dipole_suppressed {
                        valid_count += 1;
                    }

                    // Write to CSV immediately (append mode)
                    writeln!(csv_file, "{}", result.to_csv_line()).unwrap();
                    csv_file.flush().unwrap();

                    // Print summary
                    println!("  → Seg={:.2} Mpc, R={:.2}, filaments={:.1}%, {}",
                        result.seg_final, result.r_ratio,
                        result.filament_fraction * 100.0,
                        if result.dipole_suppressed { "✓ valid" } else { "✗ dipole" }
                    );
                    println!();

                    results.push(result);
                }
            }
        }
    }

    let total_time = exploration_start.elapsed().as_secs_f64();

    // Final summary
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  EXPLORATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Total runs:  {}", run_count);
    println!("  Valid runs:  {} ({:.1}%)", valid_count, 100.0 * valid_count as f64 / run_count as f64);
    println!("  Total time:  {:.1}s ({:.1} min)", total_time, total_time / 60.0);
    println!("  Avg per run: {:.1}s", total_time / run_count as f64);
    println!();
    println!("  Results saved to: {}", csv_path);
    println!();

    // Print top configurations by filament fraction
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| b.filament_fraction.partial_cmp(&a.filament_fraction).unwrap());

    println!("  Top 5 configurations by filament fraction:");
    println!("  ──────────────────────────────────────────────────────────────");
    for (i, r) in sorted.iter().take(5).enumerate() {
        println!("  {}. ε={:.2} k_min={} H={:.2} η={:.3} → {:.1}% filaments, R={:.2}",
            i + 1, r.epsilon, r.k_min, r.hubble, r.eta,
            r.filament_fraction * 100.0, r.r_ratio);
    }
    println!();

    // Print configurations with best R ratio (density dominates polarization)
    sorted.sort_by(|a, b| b.r_ratio.partial_cmp(&a.r_ratio).unwrap());
    println!("  Top 5 configurations by R = σ_ρ/σ_P ratio:");
    println!("  ──────────────────────────────────────────────────────────────");
    for (i, r) in sorted.iter().filter(|r| r.dipole_suppressed).take(5).enumerate() {
        println!("  {}. ε={:.2} k_min={} H={:.2} η={:.3} → R={:.2}, Seg={:.2}",
            i + 1, r.epsilon, r.k_min, r.hubble, r.eta,
            r.r_ratio, r.seg_final);
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// SINGLE EXPLORATION RUN
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn run_single_exploration(
    epsilon: f64,
    k_min: usize,
    hubble: f64,
    eta: f64,
    run_number: usize,
) -> ExplorationResult {
    let run_id = format!("run_{:03}_{}", run_number,
        chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let n3 = N_GRID * N_GRID * N_GRID;

    // Generate ICs with this eta value
    let (positions, velocities, signs) = generate_zeldovich_ics(42 + run_number as u64, eta, k_min);

    // Convert to GPU format
    let pos_f32: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    let vel_f32: Vec<f32> = velocities.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();

    // Initialize simulation
    let mut sim = match GpuNBodyTwoPass::with_custom_ics(pos_f32, vel_f32, signs_i8, L_BOX) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  ERROR initializing: {}", e);
            return ExplorationResult {
                run_id,
                n_particles: n3,
                epsilon,
                k_min,
                hubble,
                eta,
                runtime_s: 0.0,
                steps: 0,
                ms_per_step: 0.0,
                seg_final: 999.0,
                seg_max: 999.0,
                dipole_suppressed: false,
                sigma_rho: 0.0,
                sigma_p: 0.0,
                r_ratio: 0.0,
                void_fraction: 0.25,
                sheet_fraction: 0.25,
                filament_fraction: 0.25,
                node_fraction: 0.25,
                ke_final: 0.0,
                pe_final: 0.0,
                virial_ratio: 0.0,
                early_stop_reason: format!("init_error: {}", e),
            };
        }
    };

    sim.set_theta(THETA);
    sim.set_softening(epsilon);
    sim.set_pm_k_min(k_min);

    // Run simulation
    let start = Instant::now();
    let mut seg_max: f64 = 0.0;
    let mut step_times: Vec<u128> = Vec::new();
    let mut early_stop_reason = String::from("completed");
    let mut actual_steps = 0;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        if let Err(e) = sim.step_treepm_gpu(DT, R_CUT, hubble, DTAU_PER_DT) {
            early_stop_reason = format!("step_error: {}", e);
            break;
        }

        step_times.push(step_start.elapsed().as_millis());
        actual_steps = step;

        // Check stopping conditions (but don't stop exploration)
        if step % 50 == 0 {
            let seg = sim.segregation().unwrap_or(0.0);
            seg_max = seg_max.max(seg);

            // Early stop on dipole instability
            if seg > L_BOX / 4.0 {
                early_stop_reason = format!("dipole_instability_step_{}", step);
                break;
            }

            // Early stop on energy explosion
            let ke = sim.kinetic_energy().unwrap_or(0.0);
            let pe = sim.potential_energy_binding_sampled(500).unwrap_or(-1.0);
            let virial = if pe.abs() > 1e-10 { 2.0 * ke / pe.abs() } else { 0.0 };
            if virial > 20.0 && step > 100 {
                early_stop_reason = format!("virial_explosion_step_{}", step);
                break;
            }
        }

        // Progress indicator
        if step % 200 == 0 {
            print!(".");
            std::io::stdout().flush().ok();
        }
    }
    println!();

    let runtime = start.elapsed().as_secs_f64();
    let avg_ms = if !step_times.is_empty() {
        step_times.iter().sum::<u128>() as f64 / step_times.len() as f64
    } else { 0.0 };

    // Compute final metrics
    let ke_final = sim.kinetic_energy().unwrap_or(0.0);
    let pe_final = sim.potential_energy_binding_sampled(500).unwrap_or(-1.0);
    let virial_final = if pe_final.abs() > 1e-10 { 2.0 * ke_final / pe_final.abs() } else { 0.0 };
    let seg_final = sim.segregation().unwrap_or(0.0);
    seg_max = seg_max.max(seg_final);

    let (sigma_rho, sigma_p, r_ratio) = compute_janus_ratio(&sim, TWEB_GRID, L_BOX);
    let (void_frac, sheet_frac, filament_frac, node_frac) =
        compute_tweb_classification(&sim, TWEB_GRID, L_BOX);

    let dipole_suppressed = seg_final < L_BOX / 10.0;

    ExplorationResult {
        run_id,
        n_particles: n3,
        epsilon,
        k_min,
        hubble,
        eta,
        runtime_s: runtime,
        steps: actual_steps,
        ms_per_step: avg_ms,
        seg_final,
        seg_max,
        dipole_suppressed,
        sigma_rho,
        sigma_p,
        r_ratio,
        void_fraction: void_frac,
        sheet_fraction: sheet_frac,
        filament_fraction: filament_frac,
        node_fraction: node_frac,
        ke_final,
        pe_final,
        virial_ratio: virial_final,
        early_stop_reason,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHYSICS FUNCTIONS (UNCHANGED)
// ═══════════════════════════════════════════════════════════════════════════

/// Janus-specific diagnostic: σ_ρ vs σ_P ratio
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_janus_ratio(sim: &GpuNBodyTwoPass, grid_size: usize, box_size: f64) -> (f64, f64, f64) {
    let (positions, _, signs) = match sim.get_particles() {
        Ok(state) => state,
        Err(_) => return (0.0, 0.0, 1.0),
    };

    let n = positions.len() / 3;
    let cell_size = box_size / grid_size as f64;
    let ng = grid_size;
    let ng3 = ng * ng * ng;

    let mut rho_plus = vec![0.0f64; ng3];
    let mut rho_minus = vec![0.0f64; ng3];

    for i in 0..n {
        let x = positions[3*i] as f64;
        let y = positions[3*i + 1] as f64;
        let z = positions[3*i + 2] as f64;
        let sign = signs[i];

        let gx = ((x / cell_size).floor() as isize).rem_euclid(ng as isize) as usize;
        let gy = ((y / cell_size).floor() as isize).rem_euclid(ng as isize) as usize;
        let gz = ((z / cell_size).floor() as isize).rem_euclid(ng as isize) as usize;

        let idx = gx + ng * (gy + ng * gz);
        if idx < ng3 {
            if sign > 0 { rho_plus[idx] += 1.0; }
            else { rho_minus[idx] += 1.0; }
        }
    }

    let mut delta = vec![0.0f64; ng3];
    let mut polar = vec![0.0f64; ng3];
    let mean_rho = n as f64 / ng3 as f64;

    for i in 0..ng3 {
        let rho_total = rho_plus[i] + rho_minus[i];
        delta[i] = (rho_total - mean_rho) / mean_rho.max(1e-10);
        if rho_total > 0.0 {
            polar[i] = (rho_plus[i] - rho_minus[i]) / rho_total;
        }
    }

    let mean_delta: f64 = delta.iter().sum::<f64>() / ng3 as f64;
    let mean_polar: f64 = polar.iter().sum::<f64>() / ng3 as f64;

    let var_delta: f64 = delta.iter().map(|&d| (d - mean_delta).powi(2)).sum::<f64>() / ng3 as f64;
    let var_polar: f64 = polar.iter().map(|&p| (p - mean_polar).powi(2)).sum::<f64>() / ng3 as f64;

    let sigma_rho = var_delta.sqrt();
    let sigma_p = var_polar.sqrt();
    let r_ratio = if sigma_p > 1e-10 { sigma_rho / sigma_p } else { 100.0 };

    (sigma_rho, sigma_p, r_ratio)
}

/// T-web classification using tidal tensor eigenvalues
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_tweb_classification(sim: &GpuNBodyTwoPass, grid_size: usize, box_size: f64) -> (f64, f64, f64, f64) {
    let (positions, _, _) = match sim.get_particles() {
        Ok(state) => state,
        Err(_) => return (0.25, 0.25, 0.25, 0.25),
    };

    let n = positions.len() / 3;
    let cell_size = box_size / grid_size as f64;
    let ng = grid_size;
    let ng3 = ng * ng * ng;

    let mut rho = vec![0.0f64; ng3];
    for i in 0..n {
        let x = positions[3*i] as f64;
        let y = positions[3*i + 1] as f64;
        let z = positions[3*i + 2] as f64;

        let gx = ((x / cell_size).floor() as isize).rem_euclid(ng as isize) as usize;
        let gy = ((y / cell_size).floor() as isize).rem_euclid(ng as isize) as usize;
        let gz = ((z / cell_size).floor() as isize).rem_euclid(ng as isize) as usize;

        let idx = gx + ng * (gy + ng * gz);
        if idx < ng3 { rho[idx] += 1.0; }
    }

    let mean_rho = n as f64 / ng3 as f64;
    for r in rho.iter_mut() {
        *r = (*r - mean_rho) / mean_rho.max(1e-10);
    }

    // Solve Poisson via FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(ng);
    let ifft = planner.plan_fft_inverse(ng);

    let mut phi_complex: Vec<Complex<f64>> = rho.iter().map(|&r| Complex::new(r, 0.0)).collect();

    // 3D FFT
    for iz in 0..ng {
        for iy in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|ix| phi_complex[ix + ng * (iy + ng * iz)]).collect();
            fft.process(&mut row);
            for ix in 0..ng { phi_complex[ix + ng * (iy + ng * iz)] = row[ix]; }
        }
    }
    for iz in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iy| phi_complex[ix + ng * (iy + ng * iz)]).collect();
            fft.process(&mut row);
            for iy in 0..ng { phi_complex[ix + ng * (iy + ng * iz)] = row[iy]; }
        }
    }
    for iy in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iz| phi_complex[ix + ng * (iy + ng * iz)]).collect();
            fft.process(&mut row);
            for iz in 0..ng { phi_complex[ix + ng * (iy + ng * iz)] = row[iz]; }
        }
    }

    // Apply Green's function
    let dk = 2.0 * PI / box_size;
    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let kx = if ix <= ng/2 { ix as f64 } else { ix as f64 - ng as f64 } * dk;
                let ky = if iy <= ng/2 { iy as f64 } else { iy as f64 - ng as f64 } * dk;
                let kz = if iz <= ng/2 { iz as f64 } else { iz as f64 - ng as f64 } * dk;
                let k2 = kx*kx + ky*ky + kz*kz;
                let idx = ix + ng * (iy + ng * iz);
                if k2 > 1e-10 { phi_complex[idx] = phi_complex[idx] * (-1.0 / k2); }
                else { phi_complex[idx] = Complex::new(0.0, 0.0); }
            }
        }
    }

    // 3D IFFT
    for iy in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iz| phi_complex[ix + ng * (iy + ng * iz)]).collect();
            ifft.process(&mut row);
            for iz in 0..ng { phi_complex[ix + ng * (iy + ng * iz)] = row[iz]; }
        }
    }
    for iz in 0..ng {
        for ix in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|iy| phi_complex[ix + ng * (iy + ng * iz)]).collect();
            ifft.process(&mut row);
            for iy in 0..ng { phi_complex[ix + ng * (iy + ng * iz)] = row[iy]; }
        }
    }
    for iz in 0..ng {
        for iy in 0..ng {
            let mut row: Vec<Complex<f64>> = (0..ng).map(|ix| phi_complex[ix + ng * (iy + ng * iz)]).collect();
            ifft.process(&mut row);
            for ix in 0..ng { phi_complex[ix + ng * (iy + ng * iz)] = row[ix]; }
        }
    }

    let norm = 1.0 / (ng3 as f64);
    let phi: Vec<f64> = phi_complex.iter().map(|c| c.re * norm).collect();

    // Compute Hessian eigenvalues and classify
    let mut n_void = 0usize;
    let mut n_sheet = 0usize;
    let mut n_filament = 0usize;
    let mut n_node = 0usize;

    let h = cell_size;
    let h2 = h * h;

    for iz in 1..ng-1 {
        for iy in 1..ng-1 {
            for ix in 1..ng-1 {
                let idx = |x: usize, y: usize, z: usize| x + ng * (y + ng * z);
                let c = idx(ix, iy, iz);

                let d2x = (phi[idx(ix+1, iy, iz)] - 2.0*phi[c] + phi[idx(ix-1, iy, iz)]) / h2;
                let d2y = (phi[idx(ix, iy+1, iz)] - 2.0*phi[c] + phi[idx(ix, iy-1, iz)]) / h2;
                let d2z = (phi[idx(ix, iy, iz+1)] - 2.0*phi[c] + phi[idx(ix, iy, iz-1)]) / h2;

                let dxy = (phi[idx(ix+1, iy+1, iz)] - phi[idx(ix+1, iy-1, iz)]
                         - phi[idx(ix-1, iy+1, iz)] + phi[idx(ix-1, iy-1, iz)]) / (4.0 * h2);
                let dxz = (phi[idx(ix+1, iy, iz+1)] - phi[idx(ix+1, iy, iz-1)]
                         - phi[idx(ix-1, iy, iz+1)] + phi[idx(ix-1, iy, iz-1)]) / (4.0 * h2);
                let dyz = (phi[idx(ix, iy+1, iz+1)] - phi[idx(ix, iy+1, iz-1)]
                         - phi[idx(ix, iy-1, iz+1)] + phi[idx(ix, iy-1, iz-1)]) / (4.0 * h2);

                let eigenvalues = symmetric_3x3_eigenvalues(d2x, d2y, d2z, dxy, dxz, dyz);
                let n_positive = eigenvalues.iter().filter(|&&l| l > 0.0).count();

                match n_positive {
                    0 => n_void += 1,
                    1 => n_sheet += 1,
                    2 => n_filament += 1,
                    3 => n_node += 1,
                    _ => {}
                }
            }
        }
    }

    let total = (n_void + n_sheet + n_filament + n_node) as f64;
    if total > 0.0 {
        (n_void as f64 / total, n_sheet as f64 / total,
         n_filament as f64 / total, n_node as f64 / total)
    } else {
        (0.25, 0.25, 0.25, 0.25)
    }
}

/// Eigenvalues of symmetric 3x3 matrix
fn symmetric_3x3_eigenvalues(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> [f64; 3] {
    let p1 = d*d + e*e + f*f;
    if p1 < 1e-20 {
        let mut eig = [a, b, c];
        eig.sort_by(|x, y| y.partial_cmp(x).unwrap());
        return eig;
    }

    let q = (a + b + c) / 3.0;
    let p2 = (a - q)*(a - q) + (b - q)*(b - q) + (c - q)*(c - q) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    let b11 = (a - q) / p;
    let b22 = (b - q) / p;
    let b33 = (c - q) / p;
    let b12 = d / p;
    let b13 = e / p;
    let b23 = f / p;

    let det_b = b11 * (b22*b33 - b23*b23) - b12 * (b12*b33 - b23*b13) + b13 * (b12*b23 - b22*b13);
    let r = (det_b / 2.0).max(-1.0).min(1.0);
    let phi = r.acos() / 3.0;

    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
    let eig2 = 3.0 * q - eig1 - eig3;

    let mut eig = [eig1, eig2, eig3];
    eig.sort_by(|x, y| y.partial_cmp(x).unwrap());
    eig
}

/// Generate Zel'dovich ICs with given eta and k_min
fn generate_zeldovich_ics(seed: u64, eta: f64, k_min_idx: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let ng = N_GRID;
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

                if k_idx < k_min_idx || k < 1e-10 {
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

    let n_plus = (ng3 as f64 / (1.0 + eta)) as usize;
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

                let dx = psi_x[idx] * d_plus;
                let dy = psi_y[idx] * d_plus;
                let dz = psi_z[idx] * d_plus;

                let x = (q_x + dx).rem_euclid(L_BOX);
                let y = (q_y + dy).rem_euclid(L_BOX);
                let z = (q_z + dz).rem_euclid(L_BOX);

                let v_scale = h_factor * f_growth * d_plus * 0.001;
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
    eprintln!("Run with: cargo run --release --features cuda,cufft --bin exploration_v4");
}
