//! ΛCDM Control Test - Standard gravity, nearly single species
//! 99.9% m+, 0.1% m- to avoid edge case bug

use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[cfg(feature = "cuda")]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;

const N: usize = 500_000;
const BOX_SIZE: f64 = 50.0;
const DT: f64 = 0.002;
const STEPS: usize = 300;
const THETA: f64 = 0.7;
const SOFTENING: f64 = 0.2;
const DELTA_INIT: f64 = 0.10;

const G_COSMO: f64 = 4.499e-15;
const RHO_CRIT: f64 = 1.36e11;
const OMEGA_M: f64 = 0.30;

const CSV_INTERVAL: usize = 10;
const R_CUT: f64 = 3.0;
const N_CELLS: usize = 16;

#[cfg(feature = "cuda")]
fn main() {
    println!("================================================================");
    println!("  ΛCDM-LIKE CONTROL TEST — Nearly Pure m+");
    println!("================================================================");
    println!("  N = {} ({:.1}K)", N, N as f64 / 1e3);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  99.9% m+, 0.1% m- (avoid N-=0 bug)");
    println!("================================================================\n");

    let m_total = OMEGA_M * RHO_CRIT * BOX_SIZE.powi(3);
    let mass_factor = G_COSMO * m_total / N as f64;

    println!("m_total = {:.3e} M_sun", m_total);
    println!("mass_factor = {:.4e}\n", mass_factor);

    // Generate ICs
    println!("Generating perturbed grid ICs...");
    let n_per_dim = (N as f64).powf(1.0 / 3.0).ceil() as usize;
    let cell_size = BOX_SIZE / n_per_dim as f64;
    let half_box = BOX_SIZE / 2.0;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, DELTA_INIT * cell_size).unwrap();

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

                let x = ((x + half_box) % BOX_SIZE) - half_box;
                let y = ((y + half_box) % BOX_SIZE) - half_box;
                let z = ((z + half_box) % BOX_SIZE) - half_box;

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);
                velocities.push(0.0f32);
                velocities.push(0.0f32);
                velocities.push(0.0f32);

                // 99.9% m+, 0.1% m-
                let sign: i8 = if rng.gen::<f64>() < 0.999 { 1 } else { -1 };
                signs.push(sign);
                count += 1;
            }
        }
    }

    let n_plus: usize = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = N - n_plus;
    println!("  N+ = {} ({:.2}%), N- = {} ({:.2}%)\n",
             n_plus, 100.0 * n_plus as f64 / N as f64,
             n_minus, 100.0 * n_minus as f64 / N as f64);

    // Output
    let output_dir = "/app/output/lcdm_control";
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    let mut csv_file = BufWriter::new(
        File::create(format!("{}/time_series.csv", output_dir)).unwrap()
    );
    writeln!(csv_file, "step,t_gyr,rho_max_ratio,v_mean_kms").unwrap();

    println!("Initializing GPU...");
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        positions, velocities, signs, BOX_SIZE
    ).expect("Failed to create simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    sim.set_lambda_0(0.0);
    sim.set_mass_factor(mass_factor);

    let start = Instant::now();
    println!("\nRunning (pure gravity, ~ΛCDM)...\n");

    let mut t_gyr = 0.0;
    let mut prev_rho_max = 0.0;

    for step in 0..=STEPS {
        if step > 0 {
            sim.set_current_z(0.0);
            sim.step_treepm_gpu(DT, R_CUT, 0.0, 1.0)
                .expect("TreePM step failed");
            t_gyr += DT;
        }

        if step % CSV_INTERVAL == 0 {
            let (pos, vel, sgn) = sim.get_particles().unwrap();
            let (rho_max, v_mean) = compute_metrics(&pos, &vel, &sgn, BOX_SIZE, N_CELLS);

            writeln!(csv_file, "{},{:.4},{:.2},{:.1}", step, t_gyr, rho_max, v_mean).unwrap();

            let growth = if prev_rho_max > 0.0 {
                (rho_max - prev_rho_max) / prev_rho_max * 100.0
            } else { 0.0 };

            let alert = if rho_max > 10.0 {
                ">>> STRUCTURES <<<"
            } else if rho_max > 5.0 {
                "** clustering **"
            } else if rho_max > 2.0 {
                "* growth *"
            } else if growth > 5.0 {
                "↑"
            } else {
                ""
            };

            println!("  step {:3} | t={:.3} Gyr | ρmax/ρ̄={:.2} ({:+.1}%) | <v>={:.0} km/s {}",
                     step, t_gyr, rho_max, growth, v_mean, alert);

            prev_rho_max = rho_max;
        }
    }

    csv_file.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let (pos, vel, sgn) = sim.get_particles().unwrap();
    let (rho_max, v_mean) = compute_metrics(&pos, &vel, &sgn, BOX_SIZE, N_CELLS);

    println!("\n================================================================");
    println!("  CONTROL TEST COMPLETE");
    println!("================================================================");
    println!("  Final t: {:.2} Gyr", t_gyr);
    println!("  ρ_max/ρ̄: {:.2} {}", rho_max,
             if rho_max > 2.0 { "✓ GROWTH!" } else { "✗ no growth" });
    println!("  <v> = {:.0} km/s {}", v_mean,
             if v_mean > 10.0 { "✓ moving" } else { "✗ static" });
    println!("  Runtime: {:.1}s", elapsed);
    println!("================================================================");
}

#[cfg(feature = "cuda")]
fn compute_metrics(positions: &[f32], velocities: &[f32], signs: &[i8],
                   box_size: f64, n_cells: usize) -> (f64, f64) {
    let cell_size = box_size / n_cells as f64;
    let half_box = box_size / 2.0;
    let n_cells_cubed = n_cells * n_cells * n_cells;
    let n = signs.len();

    let mut grid = vec![0u32; n_cells_cubed];
    let mut v_sum = 0.0f64;

    for i in 0..n {
        if signs[i] <= 0 { continue; }  // Only count m+

        let x = ((positions[i*3] as f64 + half_box) % box_size) / cell_size;
        let y = ((positions[i*3+1] as f64 + half_box) % box_size) / cell_size;
        let z = ((positions[i*3+2] as f64 + half_box) % box_size) / cell_size;

        let ix = (x as usize).min(n_cells - 1);
        let iy = (y as usize).min(n_cells - 1);
        let iz = (z as usize).min(n_cells - 1);
        let idx = ix * n_cells * n_cells + iy * n_cells + iz;

        grid[idx] += 1;

        let vmag = (velocities[i*3].powi(2) + velocities[i*3+1].powi(2) +
                   velocities[i*3+2].powi(2)).sqrt() as f64 * 977.8;
        v_sum += vmag;
    }

    let n_plus: usize = signs.iter().filter(|&&s| s > 0).count();
    let v_mean = if n_plus > 0 { v_sum / n_plus as f64 } else { 0.0 };
    let mean_density = n_plus as f64 / n_cells_cubed as f64;
    let max_density = *grid.iter().max().unwrap_or(&0) as f64;
    let rho_max = if mean_density > 0.0 { max_density / mean_density } else { 1.0 };

    (rho_max, v_mean)
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda cufft");
    std::process::exit(1);
}
