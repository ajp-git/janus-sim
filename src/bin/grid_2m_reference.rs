//! Case A at 2M: Uniform random ICs, box=271 Mpc
//! Reference run to validate dtau fix against February results

use rand::prelude::*;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};

// 2M reference parameters (February convention)
const N_PARTICLES: usize = 2_000_000;
const BOX_SIZE: f64 = 271.0;           // Mpc (same as February 2M reference)
const Z_INIT: f64 = 5.0;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const TOTAL_STEPS: usize = 10000;      // Full z=5 → z=0
const SOFTENING: f64 = 0.65;           // Mpc
const SEED: u64 = 42;

const RENDER_INTERVAL: usize = 200;    // Every 200 steps

fn generate_uniform_random_ics(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let half_box = BOX_SIZE / 2.0;

    println!("Generating uniform random ICs (Case A - February reference)");
    println!("  N = {}", N_PARTICLES);
    println!("  Box = {} Mpc", BOX_SIZE);

    let mut positions = Vec::with_capacity(N_PARTICLES * 3);
    let mut velocities = Vec::with_capacity(N_PARTICLES * 3);
    let mut signs = Vec::with_capacity(N_PARTICLES);

    let virial_velocity = ((N_PARTICLES as f64) / BOX_SIZE).sqrt() * 0.3;

    for _ in 0..N_PARTICLES {
        positions.push(rng.gen::<f64>() * BOX_SIZE - half_box);
        positions.push(rng.gen::<f64>() * BOX_SIZE - half_box);
        positions.push(rng.gen::<f64>() * BOX_SIZE - half_box);

        signs.push(if rng.gen::<bool>() { 1 } else { -1 });

        velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
        velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
        velocities.push((rng.gen::<f64>() - 0.5) * virial_velocity);
    }

    let n_positive = signs.iter().filter(|&&s| s > 0).count();
    println!("  N+ = {}, N- = {}", n_positive, N_PARTICLES - n_positive);

    (positions, velocities, signs, n_positive)
}

fn compute_coms(positions: &[f64], signs: &[i32]) -> ([f64; 3], [f64; 3]) {
    let n = positions.len() / 3;
    let mut sum_pos = [0.0f64; 3];
    let mut sum_neg = [0.0f64; 3];
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;

    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        if signs[i] > 0 {
            sum_pos[0] += x;
            sum_pos[1] += y;
            sum_pos[2] += z;
            n_pos += 1;
        } else {
            sum_neg[0] += x;
            sum_neg[1] += y;
            sum_neg[2] += z;
            n_neg += 1;
        }
    }

    let com_pos = [sum_pos[0]/n_pos as f64, sum_pos[1]/n_pos as f64, sum_pos[2]/n_pos as f64];
    let com_neg = [sum_neg[0]/n_neg as f64, sum_neg[1]/n_neg as f64, sum_neg[2]/n_neg as f64];

    (com_pos, com_neg)
}

fn compute_segregation(positions: &[f64], signs: &[i32]) -> f64 {
    let (com_pos, com_neg) = compute_coms(positions, signs);
    let dx = com_pos[0] - com_neg[0];
    let dy = com_pos[1] - com_neg[1];
    let dz = com_pos[2] - com_neg[2];
    (dx*dx + dy*dy + dz*dz).sqrt() / BOX_SIZE
}

fn write_render_data(
    path: &str,
    positions: &[f64],
    signs: &[i32],
    step: usize,
    box_size: f64,
    seg: f64,
    ke_ratio: f64,
    redshift: f64,
) -> std::io::Result<()> {
    let n = positions.len() / 3;
    let mut file = BufWriter::new(File::create(path)?);

    file.write_all(&(step as u32).to_le_bytes())?;
    file.write_all(&box_size.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    file.write_all(&ke_ratio.to_le_bytes())?;
    file.write_all(&redshift.to_le_bytes())?;
    file.write_all(&(n as u32).to_le_bytes())?;

    for i in 0..n {
        file.write_all(&(positions[i * 3] as f32).to_le_bytes())?;
        file.write_all(&(positions[i * 3 + 1] as f32).to_le_bytes())?;
        file.write_all(&(positions[i * 3 + 2] as f32).to_le_bytes())?;
    }

    for i in 0..n {
        file.write_all(&(signs[i] as i8).to_le_bytes())?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  2M Reference Run — Uniform Random ICs (Case A)");
    println!("  Validating dtau fix against February 2026 results");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create output directory
    let output_dir = "/app/output/ref_2M_uniform";
    create_dir_all(output_dir).expect("Failed to create output dir");
    let render_dir = format!("{}/render_data", output_dir);
    create_dir_all(&render_dir).expect("Failed to create render_data dir");

    println!("Output: {}", output_dir);
    println!("Parameters:");
    println!("  N = {}", N_PARTICLES);
    println!("  Box = {} Mpc", BOX_SIZE);
    println!("  θ = {}", THETA);
    println!("  softening = {} Mpc", SOFTENING);
    println!("  dt = {}", DT);
    println!("  steps = {}", TOTAL_STEPS);
    println!("  seed = {}\n", SEED);

    // Generate ICs
    let (positions, velocities, signs, n_positive) = generate_uniform_random_ics(SEED);

    let n_total = signs.len();
    let n_negative = n_total - n_positive;

    // Initialize simulation
    println!("\nInitializing GPU simulation...");
    let mut sim = GpuNBodySimulation::new_with_state(
        n_positive,
        n_negative,
        BOX_SIZE,
        positions.clone(),
        velocities,
        signs.clone(),
    ).expect("Failed to create GPU simulation");

    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);

    // Virialization
    println!("\nVirializing with PE_binding method...");
    let n_sample = (n_total / 100).max(1000).min(10000);
    sim.virialize_sampled(n_sample).expect("virialize_sampled failed");
    println!("  ✓ Virialization complete");

    // Setup cosmology - FEBRUARY CONVENTION
    let eta = 1.045;
    let params = JanusParams::from_eta(eta);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);

    // dtau_per_step: for advancing tau each step
    let dtau_per_step = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    // dtau_per_dt: FIXED convention - 10000 steps cover z=5→0
    let dtau_per_dt = (cosmo.tau_end - cosmo.tau_start).abs() / (10000.0 * DT);

    println!("\nCosmology (February convention):");
    println!("  η = {}", eta);
    println!("  τ_start = {:.4}, τ_end = {:.4}", cosmo.tau_start, cosmo.tau_end);
    println!("  dtau_per_step = {:.6}", dtau_per_step);
    println!("  dtau_per_dt = {:.6} (friction parameter)", dtau_per_dt);

    // Open CSV
    let csv_path = format!("{}/time_series.csv", output_dir);
    let mut csv = File::create(&csv_path).expect("Failed to create CSV");
    writeln!(csv, "step,z,ke_ratio,seg,step_ms").unwrap();

    // Initial state
    let pos = sim.get_positions().expect("get_positions failed");
    let ke_0 = sim.kinetic_energy().expect("kinetic_energy failed");
    let seg_0 = compute_segregation(&pos, &signs);

    writeln!(csv, "0,{:.4},{:.6},{:.6},0", Z_INIT, 1.0, seg_0).unwrap();

    let render_path = format!("{}/step_{:06}.bin", render_dir, 0);
    write_render_data(&render_path, &pos, &signs, 0, BOX_SIZE, seg_0, 1.0, Z_INIT)
        .expect("Failed to write render_data");

    println!("\n══════════════════════════════════════════════════");
    println!("  Starting 2M reference run");
    println!("  Expected: Seg should GROW from ~0.001 to ~0.3-0.7");
    println!("══════════════════════════════════════════════════\n");
    println!("Step 0: z={:.2}, KE/KE₀=1.000, Seg={:.4}", Z_INIT, seg_0);

    let mut tau = cosmo.tau_start;
    let start = Instant::now();
    let mut ke_ratio_max = 1.0f64;
    let mut seg_max = seg_0;
    let mut seg_max_step = 0usize;

    for step in 1..=TOTAL_STEPS {
        let step_start = Instant::now();

        tau += dtau_per_step;
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)
            .expect("Step failed");

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

        let ke = sim.kinetic_energy().expect("kinetic_energy failed");
        let ke_ratio = ke / ke_0;
        let pos = sim.get_positions().expect("get_positions failed");
        let seg = compute_segregation(&pos, &signs);

        ke_ratio_max = ke_ratio_max.max(ke_ratio);
        if seg > seg_max {
            seg_max = seg;
            seg_max_step = step;
        }

        writeln!(csv, "{},{:.4},{:.6},{:.6},{:.1}", step, z, ke_ratio, seg, step_ms).unwrap();

        // Render at intervals
        if step % RENDER_INTERVAL == 0 {
            let render_path = format!("{}/step_{:06}.bin", render_dir, step);
            write_render_data(&render_path, &pos, &signs, step, BOX_SIZE, seg, ke_ratio, z)
                .expect("Failed to write render_data");
        }

        // Progress every 500 steps
        if step % 500 == 0 {
            let rate = step as f64 / start.elapsed().as_secs_f64();
            let eta_min = (TOTAL_STEPS - step) as f64 / rate / 60.0;
            println!("Step {}: z={:.2}, KE/KE₀={:.3}, Seg={:.4}, Seg_max={:.4} ({:.1} steps/s, ETA {:.0}min)",
                     step, z, ke_ratio, seg, seg_max, rate, eta_min);
        }

        // Check for segregation growth
        if step == 1000 && seg_max < 0.01 {
            println!("\n⚠️  WARNING at step 1000: Seg_max = {:.4} < 0.01", seg_max);
            println!("    February reference had Seg > 0.1 at this point");
        }
    }

    csv.flush().unwrap();

    let elapsed = start.elapsed().as_secs_f64();
    let final_ke = sim.kinetic_energy().expect("kinetic_energy failed") / ke_0;
    let final_pos = sim.get_positions().expect("get_positions failed");
    let final_seg = compute_segregation(&final_pos, &signs);

    println!("\n══════════════════════════════════════════════════");
    println!("  2M Reference Run Complete");
    println!("══════════════════════════════════════════════════");
    println!("  Total time: {:.1}s ({:.1} ms/step)", elapsed, elapsed * 1000.0 / TOTAL_STEPS as f64);
    println!("  Seg_0: {:.4}", seg_0);
    println!("  Seg_max: {:.4} @ step {}", seg_max, seg_max_step);
    println!("  Seg_final: {:.4}", final_seg);
    println!("  KE/KE₀ max: {:.3}, final: {:.3}", ke_ratio_max, final_ke);
    println!("\n  VERDICT:");
    if seg_max > 0.3 {
        println!("  ✓ SUCCESS: Seg_max > 0.3 — February behavior reproduced!");
    } else if seg_max > 0.1 {
        println!("  ~ PARTIAL: Seg_max > 0.1 but < 0.3");
    } else if seg_max > 0.01 {
        println!("  ⚠️  MARGINAL: Some growth detected but Seg_max < 0.1");
    } else {
        println!("  ❌ FAIL: No segregation growth (Seg_max < 0.01)");
    }
    println!("\n  Output: {}", output_dir);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires the 'cuda' feature. Compile with:");
    eprintln!("  cargo build --release --features cuda --bin grid_2m_reference");
}
