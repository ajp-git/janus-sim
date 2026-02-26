/// Janus 8M Run — Zel'dovich ICs (v=0, no virialization)
/// Based on working test_8m_debug code

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Normal, Distribution};
use std::fs::{self, File};
use std::io::{Write, BufWriter};

const N_PARTICLES: usize = 8_000_000;
const ETA: f64 = 1.045;
const THETA: f64 = 0.5;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 15000;

const ZELDOVICH_AMPLITUDE: f64 = 1e-3;
const ZELDOVICH_LAMBDA: f64 = 100.0;
const ZELDOVICH_SIGMA: f64 = 0.1;

fn generate_zeldovich_ics(n_total: usize, box_size: f64, seed: u64, eta: f64)
    -> (Vec<f64>, Vec<f64>, Vec<i32>)
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_per_axis = (n_total as f64).powf(1.0/3.0).ceil() as usize;
    let cell_size = box_size / n_per_axis as f64;
    let box_half = box_size / 2.0;
    let normal = Normal::new(0.0, ZELDOVICH_SIGMA).unwrap();
    let k = 2.0 * std::f64::consts::PI / ZELDOVICH_LAMBDA;

    println!("  Grid: {}³ = {} cells, cell_size = {:.3}", n_per_axis, n_per_axis.pow(3), cell_size);
    println!("  Zel'dovich: amplitude = {:.0e}, lambda = {:.1}, sigma = {:.3}",
        ZELDOVICH_AMPLITUDE, ZELDOVICH_LAMBDA, ZELDOVICH_SIGMA);

    let mut positions = Vec::with_capacity(n_total * 3);
    let mut velocities = Vec::with_capacity(n_total * 3);
    let mut signs = Vec::with_capacity(n_total);

    let mut count = 0;
    'outer: for ix in 0..n_per_axis {
        for iy in 0..n_per_axis {
            for iz in 0..n_per_axis {
                if count >= n_total { break 'outer; }
                let x0 = (ix as f64 + 0.5) * cell_size - box_half;
                let y0 = (iy as f64 + 0.5) * cell_size - box_half;
                let z0 = (iz as f64 + 0.5) * cell_size - box_half;

                let phase_x = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let phase_y = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let phase_z = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
                let amp: f64 = normal.sample(&mut rng);
                let dx = ZELDOVICH_AMPLITUDE * amp * (k * x0 + phase_x).sin();
                let dy = ZELDOVICH_AMPLITUDE * amp * (k * y0 + phase_y).sin();
                let dz = ZELDOVICH_AMPLITUDE * amp * (k * z0 + phase_z).sin();

                let mut x = x0 + dx;
                let mut y = y0 + dy;
                let mut z = z0 + dz;
                if x > box_half { x -= box_size; }
                if x < -box_half { x += box_size; }
                if y > box_half { y -= box_size; }
                if y < -box_half { y += box_size; }
                if z > box_half { z -= box_size; }
                if z < -box_half { z += box_size; }

                positions.push(x);
                positions.push(y);
                positions.push(z);
                velocities.push(0.0);
                velocities.push(0.0);
                velocities.push(0.0);
                let sign = if rng.random::<f64>() < (1.0 / (1.0 + eta)) { 1 } else { -1 };
                signs.push(sign);
                count += 1;
            }
        }
    }

    let actual_pos = signs.iter().filter(|&&s| s == 1).count();
    let actual_neg = signs.iter().filter(|&&s| s == -1).count();
    println!("  Generated: N+ = {}, N- = {} (random assignment)", actual_pos, actual_neg);
    println!("  Velocities: v = 0 (cold start, no virialization)");

    (positions, velocities, signs)
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 8M — Zel'dovich ICs (v=0, no virialization)            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;
    let box_size = 100.0 * (N_PARTICLES as f64 / 100_000.0).powf(1.0/3.0);

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_PARTICLES as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_PARTICLES as f64);
    println!("  η = {}", ETA);
    println!("  θ = {}", THETA);
    println!("  dt = {}", DT);
    println!("  box = {:.2}", box_size);
    println!("  integrator = DKD + Hubble friction");
    println!();

    // Setup cosmological expansion
    println!("--- Cosmological Expansion Setup ---");
    let janus_params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&janus_params, Z_INIT);

    let n_steps_to_z0 = 12000.0;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    let (a_init, h_init) = cosmo.get_params_at_tau(cosmo.tau_start);
    let z_init_actual = 1.0 / a_init - 1.0;

    println!("  z_init = {:.2}", z_init_actual);
    println!("  a_init = {:.6}", a_init);
    println!("  H_init = {:.6}", h_init);
    println!("  τ range = [{:.4}, {:.4}]", cosmo.tau_start, cosmo.tau_end);
    println!("  dτ/dt = {:.6}", dtau_per_dt);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = format!("/app/output/8M_zeldovich_{}", date);
    fs::create_dir_all(&output_dir)?;
    println!("Output directory: {}\n", output_dir);

    // Generate Zel'dovich ICs
    println!("Generating Zel'dovich ICs (v=0)...");
    let t0 = std::time::Instant::now();
    let (positions, velocities, signs) = generate_zeldovich_ics(N_PARTICLES, box_size, 12345, ETA);
    println!("  Generated in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Create simulation with pre-generated state
    println!("Creating simulation...");
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive, n_negative, box_size,
        positions, velocities, signs
    )?;
    sim.set_theta(THETA);
    println!("  Created in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Initial diagnostics
    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation_distance()?;
    println!("Initial state:");
    println!("  KE₀ = {:.4e} (should be ~0 for cold start)", ke_init);
    println!("  S₀ = {:.6}", seg_init);
    println!();

    // Time series file
    let ts_filename = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_filename)?);
    writeln!(ts_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms")?;

    let mut ke_ref: Option<f64> = None;
    let mut tau = cosmo.tau_start;
    let mut seg_max = 0.0f64;

    println!("Starting simulation loop...\n");

    for step in 1..=MAX_STEPS {
        let t_step = std::time::Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;

        let step_time = t_step.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation_distance()?;

        if ke_ref.is_none() && ke > 0.0 {
            ke_ref = Some(ke);
            println!("  KE_ref set to {:.4e} at step {}", ke, step);
        }

        let ke_ratio = ke_ref.map(|r| ke / r).unwrap_or(1.0);
        seg_max = seg_max.max(seg);

        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke_ratio, seg, step_time)?;

        // Flush every step for debugging
        ts_file.flush()?;

        if step % 100 == 0 || step <= 10 {
            println!("step {:06} | z={:.2} | a={:.4} | H={:.4} | KE/KE_ref={:.4} | S={:.3e} | S_max={:.3e} | {} ms",
                step, z, a, h, ke_ratio, seg, seg_max, step_time as u64);
        }

        if ke.is_nan() || ke.is_infinite() {
            println!("\n=== KE is NaN/Inf, stopping (numerical explosion) ===");
            break;
        }

        if z < 0.01 {
            println!("\n=== Reached z ≈ 0, simulation complete ===");
            break;
        }
    }

    ts_file.flush()?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Final results:");
    println!("  S_max = {:.6}", seg_max);
    println!("  Output: {}", ts_filename);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled!");
}
