//! Exhaustive Parameter Scan — 4,500 runs
//! Search for Janus configurations that produce GROWTH in structures
//!
//! Metric: signal = ρ+_max(step=200) / ρ+_max(step=0)
//!   signal > 2.0 → density doubled → real physical signal
//!   signal > 3.0 → strong growth → priority candidate ★
//!   signal < 1.2 → nothing happening → eliminate

use std::fs::{self, File, OpenOptions};
use std::io::{Write, BufWriter, BufRead};
use std::time::Instant;
use std::collections::HashSet;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

// Scan parameters
const MU_VALUES: &[f64] = &[8.0, 10.0, 12.0, 14.0, 16.0, 19.0, 24.0, 32.0, 48.0, 64.0];
const LAMBDA_VALUES: &[f64] = &[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0];
const DELTA_VALUES: &[f64] = &[1.0, 5.0, 10.0, 20.0, 50.0];  // percent
const BOX_VALUES: &[f64] = &[20.0, 50.0, 100.0];  // Mpc
const SEEDS: &[u64] = &[1, 2, 3];

// Simulation parameters
const N: usize = 50_000;
const STEPS: usize = 200;
const DT: f64 = 0.005;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.3;
const R_CUT: f64 = 5.0;
const N_CELLS: usize = 8;

// Physics
const G_COSMO: f64 = 4.499e-15;
const RHO_CRIT: f64 = 1.36e11;
const OMEGA_B: f64 = 0.05;

#[derive(Clone)]
struct RunResult {
    mu: f64,
    lambda: f64,
    delta: f64,
    box_size: f64,
    seed: u64,
    rho_0: f64,      // ρ+_max at step 0
    rho_200: f64,    // ρ+_max at step 200
    signal: f64,     // rho_200 / rho_0
    corr: f64,
    runtime_s: f64,
}

#[cfg(feature = "cuda")]
fn main() {
    let total_runs = MU_VALUES.len() * LAMBDA_VALUES.len() * DELTA_VALUES.len()
                   * BOX_VALUES.len() * SEEDS.len();

    println!("================================================================");
    println!("  EXHAUSTIVE PARAMETER SCAN v2 — {} runs", total_runs);
    println!("  Metric: signal = ρ+_max(200) / ρ+_max(0)");
    println!("================================================================");
    println!("  μ values: {:?}", MU_VALUES);
    println!("  λ values: {:?}", LAMBDA_VALUES);
    println!("  δ values: {:?}%", DELTA_VALUES);
    println!("  Box sizes: {:?} Mpc", BOX_VALUES);
    println!("  Seeds: {:?}", SEEDS);
    println!("  N = {}, Steps = {}", N, STEPS);
    println!("  Thresholds: signal > 2.0 (candidate), > 3.0 (priority ★)");
    println!("================================================================\n");

    // Setup output
    let output_dir = "/app/output/exhaustive_scan_v2";
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    let csv_path = format!("{}/scan_results.csv", output_dir);

    // Load already completed runs
    let mut done: HashSet<String> = HashSet::new();
    if let Ok(file) = File::open(&csv_path) {
        let reader = std::io::BufReader::new(file);
        for line in reader.lines().skip(1) {
            if let Ok(l) = line {
                let parts: Vec<&str> = l.split(',').collect();
                if parts.len() >= 5 {
                    let key = format!("{},{},{},{},{}", parts[0], parts[1], parts[2], parts[3], parts[4]);
                    done.insert(key);
                }
            }
        }
    }

    let already_done = done.len();
    if already_done > 0 {
        println!("RESUMING: {} runs already completed, {} remaining\n", already_done, total_runs - already_done);
    } else {
        // Create new CSV with header
        let mut csv_file = BufWriter::new(
            File::create(&csv_path).expect("Failed to create CSV")
        );
        writeln!(csv_file, "mu,lambda,delta,box,seed,rho_0,rho_200,signal,corr,runtime_s").unwrap();
        csv_file.flush().unwrap();
    }

    let scan_start = Instant::now();
    let mut completed = already_done;
    let mut candidates: Vec<RunResult> = Vec::new();
    let mut priority_count = 0;

    // Main scan loop
    for &mu in MU_VALUES {
        for &lambda in LAMBDA_VALUES {
            for &delta in DELTA_VALUES {
                for &box_size in BOX_VALUES {
                    for &seed in SEEDS {
                        // Skip already completed
                        let key = format!("{},{},{},{},{}", mu, lambda, delta, box_size, seed);
                        if done.contains(&key) {
                            continue;
                        }

                        completed += 1;

                        let result = run_single(mu, lambda, delta, box_size, seed);

                        // Append to CSV immediately
                        {
                            let mut csv_append = OpenOptions::new()
                                .append(true)
                                .open(&csv_path)
                                .expect("Failed to open CSV for append");
                            writeln!(csv_append, "{},{},{},{},{},{:.3},{:.3},{:.3},{:.4},{:.1}",
                                result.mu, result.lambda, result.delta, result.box_size,
                                result.seed, result.rho_0, result.rho_200, result.signal,
                                result.corr, result.runtime_s).unwrap();
                        }

                        // Check if candidate based on GROWTH
                        let is_priority = result.signal > 3.0;
                        let is_candidate = result.signal > 2.0;

                        if is_priority {
                            priority_count += 1;
                            candidates.push(result.clone());
                            candidates.sort_by(|a, b| b.signal.partial_cmp(&a.signal).unwrap());
                        } else if is_candidate {
                            candidates.push(result.clone());
                            candidates.sort_by(|a, b| b.signal.partial_cmp(&a.signal).unwrap());
                        }

                        // Progress update
                        let elapsed = scan_start.elapsed().as_secs_f64();
                        let rate = completed as f64 / elapsed;
                        let eta_min = (total_runs - completed) as f64 / rate / 60.0;

                        let status = if is_priority { "★★★" }
                                    else if is_candidate { "★" }
                                    else if result.signal < 1.2 { "✗" }
                                    else { "" };

                        print!("\r[{:4}/{:4}] μ={:2} λ={:4.1} δ={:2}% box={:3} | ρ₀={:.2} ρ₂₀₀={:.2} signal={:.2} {} | ",
                               completed, total_runs, mu as i32, lambda, delta as i32,
                               box_size as i32, result.rho_0, result.rho_200, result.signal, status);
                        print!("Best: {:.2} | Candidates: {} (★{})",
                               candidates.first().map(|c| c.signal).unwrap_or(0.0),
                               candidates.len(), priority_count);
                        print!(" | ETA: {:.0}min   ", eta_min);
                        std::io::stdout().flush().unwrap();
                    }
                }
            }
        }
    }

    println!("\n\n================================================================");
    println!("  SCAN COMPLETE — {} runs in {:.1} min",
             total_runs, scan_start.elapsed().as_secs_f64() / 60.0);
    println!("================================================================");

    println!("\nTOP 10 CANDIDATES (signal > 2.0):");
    println!("{:-<90}", "");
    println!("{:<6} {:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8}",
             "μ", "λ", "δ%", "Box", "Seed", "ρ₀", "ρ₂₀₀", "Signal", "Corr");
    println!("{:-<90}", "");

    for c in candidates.iter().take(10) {
        let marker = if c.signal > 3.0 { "★" } else { "" };
        println!("{:<6} {:<6.1} {:<6} {:<6} {:<6} {:<8.2} {:<8.2} {:<8.2} {:<8.4} {}",
                 c.mu as i32, c.lambda, c.delta as i32, c.box_size as i32,
                 c.seed, c.rho_0, c.rho_200, c.signal, c.corr, marker);
    }

    println!("{:-<90}", "");
    println!("\nTotal candidates (signal > 2.0): {}", candidates.len());
    println!("Priority candidates (signal > 3.0): {}", priority_count);
    println!("\nResults saved to: {}", csv_path);
}

#[cfg(feature = "cuda")]
fn run_single(mu: f64, lambda: f64, delta_pct: f64, box_size: f64, seed: u64) -> RunResult {
    let start = Instant::now();

    // Calculate particle counts
    let n_plus = (N as f64 / (1.0 + mu)) as usize;
    let _n_minus = N - n_plus;

    // Mass factor
    let rho_plus = OMEGA_B * RHO_CRIT;
    let rho_total = rho_plus * (1.0 + mu);
    let m_total = rho_total * box_size.powi(3);
    let mass_factor = G_COSMO * m_total / N as f64;

    // Generate ICs
    let n_per_dim = (N as f64).powf(1.0 / 3.0).ceil() as usize;
    let cell_size = box_size / n_per_dim as f64;
    let half_box = box_size / 2.0;
    let delta_frac = delta_pct / 100.0;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, delta_frac * cell_size).unwrap();

    let mut positions = Vec::with_capacity(N * 3);
    let mut velocities = Vec::with_capacity(N * 3);
    let mut signs = Vec::with_capacity(N);

    let mut count = 0;
    'outer: for ix in 0..n_per_dim {
        for iy in 0..n_per_dim {
            for iz in 0..n_per_dim {
                if count >= N { break 'outer; }

                let x = -half_box + (ix as f64 + 0.5) * cell_size + normal.sample(&mut rng);
                let y = -half_box + (iy as f64 + 0.5) * cell_size + normal.sample(&mut rng);
                let z = -half_box + (iz as f64 + 0.5) * cell_size + normal.sample(&mut rng);

                let x = ((x + half_box) % box_size) - half_box;
                let y = ((y + half_box) % box_size) - half_box;
                let z = ((z + half_box) % box_size) - half_box;

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);
                velocities.push(0.0f32);
                velocities.push(0.0f32);
                velocities.push(0.0f32);

                // P(+) = 1/(1+μ)
                let sign: i8 = if rng.gen::<f64>() < 1.0 / (1.0 + mu) { 1 } else { -1 };
                signs.push(sign);
                count += 1;
            }
        }
    }

    // Initialize simulation
    let mut sim = match GpuNBodyTwoPass::with_custom_ics(
        positions, velocities, signs, box_size
    ) {
        Ok(s) => s,
        Err(_) => return RunResult {
            mu, lambda, delta: delta_pct, box_size, seed,
            rho_0: 0.0, rho_200: 0.0, signal: 0.0, corr: 0.0, runtime_s: 0.0
        }
    };

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(lambda);
    sim.set_mass_factor(mass_factor);

    // Measure rho_max at step 0
    let rho_0 = match sim.get_particles() {
        Ok((pos, _, sgn)) => compute_rho_max(&pos, &sgn, box_size, N_CELLS),
        Err(_) => 1.0
    };

    // Run simulation
    for _step in 0..STEPS {
        sim.set_current_z(0.0);
        if let Err(_) = sim.step_treepm_gpu(DT, R_CUT, 0.0, 1.0) {
            break;
        }
    }

    // Measure rho_max at step 200 and correlation
    let (rho_200, corr) = match sim.get_particles() {
        Ok((pos, _, sgn)) => {
            let rho = compute_rho_max(&pos, &sgn, box_size, N_CELLS);
            let c = compute_correlation(&pos, &sgn, box_size, N_CELLS);
            (rho, c)
        },
        Err(_) => (1.0, 0.0)
    };

    // Calculate signal (growth factor)
    let signal = if rho_0 > 0.0 { rho_200 / rho_0 } else { 1.0 };

    RunResult {
        mu, lambda, delta: delta_pct, box_size, seed,
        rho_0, rho_200, signal, corr,
        runtime_s: start.elapsed().as_secs_f64()
    }
}

#[cfg(feature = "cuda")]
fn compute_rho_max(positions: &[f32], signs: &[i8], box_size: f64, n_cells: usize) -> f64 {
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n_cells_cubed = n_cells * n_cells * n_cells;
    let n = signs.len();

    let mut n_plus_grid = vec![0u32; n_cells_cubed];

    for i in 0..n {
        if signs[i] <= 0 { continue; }

        let x = ((positions[i*3] as f64 + half_box) % box_size) / cell_size;
        let y = ((positions[i*3+1] as f64 + half_box) % box_size) / cell_size;
        let z = ((positions[i*3+2] as f64 + half_box) % box_size) / cell_size;

        let ix = (x as usize).min(n_cells - 1);
        let iy = (y as usize).min(n_cells - 1);
        let iz = (z as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        n_plus_grid[idx] += 1;
    }

    let total_plus: u64 = n_plus_grid.iter().map(|&x| x as u64).sum();
    let mean_plus = total_plus as f64 / n_cells_cubed as f64;
    let max_plus = *n_plus_grid.iter().max().unwrap_or(&0) as f64;

    if mean_plus > 0.0 { max_plus / mean_plus } else { 1.0 }
}

#[cfg(feature = "cuda")]
fn compute_correlation(positions: &[f32], signs: &[i8], box_size: f64, n_cells: usize) -> f64 {
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n_cells_cubed = n_cells * n_cells * n_cells;
    let n = signs.len();

    let mut n_plus_grid = vec![0u32; n_cells_cubed];
    let mut n_minus_grid = vec![0u32; n_cells_cubed];

    for i in 0..n {
        let x = ((positions[i*3] as f64 + half_box) % box_size) / cell_size;
        let y = ((positions[i*3+1] as f64 + half_box) % box_size) / cell_size;
        let z = ((positions[i*3+2] as f64 + half_box) % box_size) / cell_size;

        let ix = (x as usize).min(n_cells - 1);
        let iy = (y as usize).min(n_cells - 1);
        let iz = (z as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        if signs[i] > 0 {
            n_plus_grid[idx] += 1;
        } else {
            n_minus_grid[idx] += 1;
        }
    }

    let mean_plus = n_plus_grid.iter().map(|&x| x as f64).sum::<f64>() / n_cells_cubed as f64;
    let mean_minus = n_minus_grid.iter().map(|&x| x as f64).sum::<f64>() / n_cells_cubed as f64;

    let delta_plus: Vec<f64> = n_plus_grid.iter()
        .map(|&x| if mean_plus > 0.0 { (x as f64 - mean_plus) / mean_plus } else { 0.0 }).collect();
    let delta_minus: Vec<f64> = n_minus_grid.iter()
        .map(|&x| if mean_minus > 0.0 { (x as f64 - mean_minus) / mean_minus } else { 0.0 }).collect();

    let cov: f64 = delta_plus.iter().zip(delta_minus.iter())
        .map(|(dp, dm)| dp * dm).sum::<f64>() / n_cells_cubed as f64;
    let var_plus: f64 = delta_plus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;
    let var_minus: f64 = delta_minus.iter().map(|d| d.powi(2)).sum::<f64>() / n_cells_cubed as f64;

    if var_plus > 0.0 && var_minus > 0.0 {
        cov / (var_plus.sqrt() * var_minus.sqrt())
    } else {
        0.0
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda cufft");
    std::process::exit(1);
}
