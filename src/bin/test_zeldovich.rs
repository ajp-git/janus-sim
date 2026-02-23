//! Jour 3 — Anti-correlated ICs test
//!
//! Grid: 32³, Box: 400 Mpc
//! Single-mode perturbation with anti-correlated negative masses
//!
//! Compare Run B (Janus α=1) vs Yukawa runs (C-F)

use rand::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;

// Physical parameters
const N_GRID: usize = 64;      // 64³ = 262144 particles
const L_BOX: f64 = 400.0;      // Mpc

// Simulation parameters
const DT: f64 = 0.005;
const N_STEPS: usize = 500;
const MEASURE_INTERVAL: usize = 50;
const THETA: f64 = 1.5;        // Same as test_anisotropic

/// Generate ICs with single-mode perturbation (like test_anisotropic)
/// but with anti-correlated negative masses
fn generate_zeldovich_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let n3 = N_GRID * N_GRID * N_GRID;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating ICs with anti-correlated perturbation...");
    println!("  Grid: {}³ = {} particles", N_GRID, n3);
    println!("  Box: {} Mpc", L_BOX);

    // Single-mode perturbation (like test_anisotropic)
    let kx = 2.0 * PI / L_BOX;
    let amplitude = 0.002 * L_BOX;  // 0.2% - linear regime

    let spacing = L_BOX / N_GRID as f64;
    let half_box = L_BOX / 2.0;
    let noise_amp = spacing * 0.001;  // Tiny noise

    println!("  Perturbation: A = {:.4} Mpc ({:.2}% of box)", amplitude, amplitude / L_BOX * 100.0);
    println!("  kx = 2π/box = {:.6}", kx);

    // Generate particles on grid with perturbation
    // η ~ 1.0 → equal positive and negative masses
    let n_positive = n3 / 2;
    let n_negative = n3 - n_positive;

    let mut positions = Vec::with_capacity(n3 * 3);
    let velocities = vec![0.0f64; n3 * 3];  // Start at rest
    let mut signs = Vec::with_capacity(n3);

    for iz in 0..N_GRID {
        for iy in 0..N_GRID {
            for ix in 0..N_GRID {
                let idx = iz * N_GRID * N_GRID + iy * N_GRID + ix;

                // Grid position centered around [-box/2, box/2] (like test_anisotropic)
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                // Tiny noise to break degeneracy
                let nx: f64 = (rng.random::<f64>() - 0.5) * noise_amp;
                let ny: f64 = (rng.random::<f64>() - 0.5) * noise_amp;
                let nz: f64 = (rng.random::<f64>() - 0.5) * noise_amp;

                // Single-mode perturbation along x
                // For positive masses: x = x0 + A × sin(kx × x0)
                // For negative masses: x = x0 - A × sin(kx × x0) (anti-correlation)
                let sign = if idx < n_positive { 1 } else { -1 };
                let sign_factor = sign as f64;

                let x = x0 + sign_factor * amplitude * (kx * x0).sin() + nx;
                let y = y0 + ny;
                let z = z0 + nz;

                positions.push(x);
                positions.push(y);
                positions.push(z);
                signs.push(sign);
            }
        }
    }

    println!("  Total particles: {} ({}+ and {}-)", n3, n_positive, n_negative);

    (positions, velocities, signs)
}



#[cfg(feature = "cuda")]
fn run_simulation(
    name: &str,
    n_positive: usize,
    n_negative: usize,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    signs: Vec<i32>,
    kx: f64,
    use_yukawa: bool,
    epsilon: f64,
    r_c: f64,
) -> Vec<(usize, f64, f64)> {
    println!("\n=== {} ===", name);
    if use_yukawa {
        println!("  Yukawa: ε={}, r_c={} Mpc", epsilon, r_c);
    } else {
        println!("  Janus α=1 (cross_factor=-1.0)");
    }

    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        L_BOX,
        positions,
        velocities,
        signs,
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);

    let mut results = Vec::new();

    // Initial measurement
    let pos = sim.get_positions().expect("get_positions failed");
    let delta_k = compute_mode_amplitude(&pos, kx);
    let sigma_x = compute_sigma_x_simple(&pos);
    results.push((0, delta_k, sigma_x));
    println!("  Step 0: δ_k = {:.6e}, σ_x = {:.2} Mpc", delta_k, sigma_x);

    let start = Instant::now();

    for step in 1..=N_STEPS {
        if use_yukawa {
            sim.step_with_yukawa(DT, epsilon, r_c).expect("Yukawa step failed");
        } else {
            sim.step_with_cross_factor(DT, -1.0).expect("Step failed");
        }

        if step % MEASURE_INTERVAL == 0 {
            let pos = sim.get_positions().expect("get_positions failed");
            let delta_k = compute_mode_amplitude(&pos, kx);
            let sigma_x = compute_sigma_x_simple(&pos);
            results.push((step, delta_k, sigma_x));

            let elapsed = start.elapsed().as_secs_f64();
            let rate = step as f64 / elapsed;
            println!("  Step {}: δ_k = {:.6e}, σ_x = {:.2} Mpc ({:.1} steps/s)",
                     step, delta_k, sigma_x, rate);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("  Completed in {:.1}s ({:.1} steps/s)", elapsed, N_STEPS as f64 / elapsed);

    results
}

/// Compute Fourier mode amplitude (like test_anisotropic)
fn compute_mode_amplitude(positions: &[f64], kx: f64) -> f64 {
    let n = positions.len() / 3;
    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;

    for i in 0..n {
        let x = positions[i * 3];
        let phase = kx * x;
        sum_cos += phase.cos();
        sum_sin += phase.sin();
    }

    let delta_re = sum_cos / n as f64;
    let delta_im = sum_sin / n as f64;

    (delta_re * delta_re + delta_im * delta_im).sqrt()
}

/// Compute spatial dispersion σ_x
fn compute_sigma_x_simple(positions: &[f64]) -> f64 {
    let n = positions.len() / 3;
    let mut sum_x = 0.0f64;
    let mut sum_x2 = 0.0f64;

    for i in 0..n {
        let x = positions[i * 3];
        sum_x += x;
        sum_x2 += x * x;
    }

    let mean_x = sum_x / n as f64;
    let var_x = sum_x2 / n as f64 - mean_x * mean_x;
    var_x.sqrt()
}

#[cfg(feature = "cuda")]
fn main() {
    println!("==============================================");
    println!("Jour 3 — Anti-correlated ICs Test");
    println!("              5 Runs: B-F                     ");
    println!("==============================================\n");

    let seed = 42u64;

    // Generate ICs
    let (positions, velocities, signs) = generate_zeldovich_ics(seed);
    let n3 = N_GRID * N_GRID * N_GRID;
    let n_positive = n3 / 2;
    let n_negative = n3 - n_positive;
    let kx = 2.0 * PI / L_BOX;

    // Run B: Janus α=1
    let results_b = run_simulation(
        "Run B (Janus α=1)",
        n_positive, n_negative,
        positions.clone(),
        velocities.clone(),
        signs.clone(),
        kx,
        false, 0.0, 0.0,
    );

    // Run C: Yukawa ε=0.3, r_c=40 Mpc
    let results_c = run_simulation(
        "Run C (Yukawa ε=0.3, r_c=40)",
        n_positive, n_negative,
        positions.clone(),
        velocities.clone(),
        signs.clone(),
        kx,
        true, 0.3, 40.0,
    );

    // Run D: Yukawa ε=0.3, r_c=10 Mpc
    let results_d = run_simulation(
        "Run D (Yukawa ε=0.3, r_c=10)",
        n_positive, n_negative,
        positions.clone(),
        velocities.clone(),
        signs.clone(),
        kx,
        true, 0.3, 10.0,
    );

    // Run E: Yukawa ε=0.7, r_c=40 Mpc
    let results_e = run_simulation(
        "Run E (Yukawa ε=0.7, r_c=40)",
        n_positive, n_negative,
        positions.clone(),
        velocities.clone(),
        signs.clone(),
        kx,
        true, 0.7, 40.0,
    );

    // Run F: Yukawa ε=0.7, r_c=10 Mpc
    let results_f = run_simulation(
        "Run F (Yukawa ε=0.7, r_c=10)",
        n_positive, n_negative,
        positions.clone(),
        velocities.clone(),
        signs.clone(),
        kx,
        true, 0.7, 10.0,
    );

    // Write CSV output
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let output_dir = format!("/app/output/zeldovich_{}", timestamp);
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    let csv_path = format!("{}/results.csv", output_dir);
    let mut file = File::create(&csv_path).expect("Failed to create CSV");

    writeln!(file, "step,delta_k_B,delta_k_C,delta_k_D,delta_k_E,delta_k_F,sigma_x_B,sigma_x_C,sigma_x_D,sigma_x_E,sigma_x_F").unwrap();
    for i in 0..results_b.len() {
        writeln!(file, "{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.4},{:.4},{:.4},{:.4},{:.4}",
                 results_b[i].0,
                 results_b[i].1, results_c[i].1, results_d[i].1, results_e[i].1, results_f[i].1,
                 results_b[i].2, results_c[i].2, results_d[i].2, results_e[i].2, results_f[i].2).unwrap();
    }

    println!("\n=== SUMMARY ===");
    println!("CSV written to: {}", csv_path);

    // Compare final results
    let (_, dk_init, _) = results_b[0];
    let (_, dk_b_final, _) = results_b.last().unwrap();
    let (_, dk_c_final, _) = results_c.last().unwrap();
    let (_, dk_d_final, _) = results_d.last().unwrap();
    let (_, dk_e_final, _) = results_e.last().unwrap();
    let (_, dk_f_final, _) = results_f.last().unwrap();

    let growth_b = (dk_b_final / dk_init - 1.0) * 100.0;
    let growth_c = (dk_c_final / dk_init - 1.0) * 100.0;
    let growth_d = (dk_d_final / dk_init - 1.0) * 100.0;
    let growth_e = (dk_e_final / dk_init - 1.0) * 100.0;
    let growth_f = (dk_f_final / dk_init - 1.0) * 100.0;

    println!("\nδ_k growth after {} steps:", N_STEPS);
    println!("  Run B (Janus α=1):         {:+.1}%", growth_b);
    println!("  Run C (Yukawa ε=0.3 r=40): {:+.1}%", growth_c);
    println!("  Run D (Yukawa ε=0.3 r=10): {:+.1}%", growth_d);
    println!("  Run E (Yukawa ε=0.7 r=40): {:+.1}%", growth_e);
    println!("  Run F (Yukawa ε=0.7 r=10): {:+.1}%", growth_f);

    println!("\nRatios vs Run B:");
    println!("  C/B: {:.3}", growth_c / growth_b);
    println!("  D/B: {:.3}", growth_d / growth_b);
    println!("  E/B: {:.3}", growth_e / growth_b);
    println!("  F/B: {:.3}", growth_f / growth_b);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature. Compile with:");
    eprintln!("  cargo build --release --features cuda --bin test_zeldovich");
}
