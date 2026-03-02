/// Janus 30M Production Run — Pure Barnes-Hut
///
/// N_max BH pur = 32M (FIX-013), production à 30M (marge 5%)
/// TreePM plus efficace en mémoire mais instable (discontinuité r_cut)
///
/// Parameters:
///   N = 30_000_000
///   box = 690.0 Mpc (densité 8M validé : spacing ~2.15 Mpc)
///   θ = 0.7 (FIX-012 validated)
///   softening = 0.1 Mpc
///   dt = 0.01
///   integrator = Pure BH (step_with_expansion_dkd_gpu)

#[cfg(feature = "cuda")]
use janus::nbody_gpu::GpuNBodySimulation;
#[cfg(feature = "cuda")]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::time::Instant;

const N_PARTICLES: usize = 30_000_000;
const BOX_SIZE: f64 = 690.0;  // Spacing ~2.22 Mpc (densité 8M validé)
const SOFTENING: f64 = 0.1;   // Default, validated on 8M

const ETA: f64 = 1.045;
const THETA: f64 = 0.7;  // FIX-012 validated
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const MAX_STEPS: usize = 15000;
const FRAME_INTERVAL: usize = 20;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Janus 30M — Pure Barnes-Hut Production                       ║");
    println!("║   N_max=32M RTX 3060 (FIX-013), 5% margin                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_positive = (N_PARTICLES as f64 / (1.0 + ETA)) as usize;
    let n_negative = N_PARTICLES - n_positive;

    println!("Parameters:");
    println!("  N = {} ({:.1}M)", N_PARTICLES, N_PARTICLES as f64 / 1e6);
    println!("  N+ = {} ({:.1}%)", n_positive, 100.0 * n_positive as f64 / N_PARTICLES as f64);
    println!("  N- = {} ({:.1}%)", n_negative, 100.0 * n_negative as f64 / N_PARTICLES as f64);
    println!("  η = {}", ETA);
    println!("  θ = {} (FIX-012 validated)", THETA);
    println!("  softening = {} Mpc", SOFTENING);
    println!("  dt = {}", DT);
    println!("  box = {:.2} Mpc (spacing = {:.2} Mpc)", BOX_SIZE, BOX_SIZE / (N_PARTICLES as f64).powf(1.0/3.0));
    println!("  integrator = Pure BH + DKD + Hubble");
    println!("  ICs = virialize_sampled(10000)");
    println!("  frames every {} steps", FRAME_INTERVAL);
    println!();

    // Cosmological expansion
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
    println!("  Expected steps to z=0: {}", n_steps_to_z0 as usize);
    println!();

    // Output directory
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = format!("/app/output/30M_bh_{}", date);
    let frames_dir = format!("{}/frames", output_dir);
    let render_data_dir = format!("{}/render_data", output_dir);
    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(&frames_dir)?;
    fs::create_dir_all(&render_data_dir)?;
    println!("Output directory: {}\n", output_dir);

    // Create simulation with BH-only
    println!("Creating BH simulation (30M particles)...");
    let t0 = Instant::now();
    let mut sim = GpuNBodySimulation::new_bvh_only(n_positive, n_negative, BOX_SIZE)?;
    sim.set_theta(THETA);
    sim.set_softening(SOFTENING);
    println!("  Created in {:.2}s", t0.elapsed().as_secs_f64());
    println!("  VRAM usage: check nvidia-smi\n");

    // virialize_sampled(10000) — exact same as 8M validated run
    println!("Applying virialize_sampled(10000)...");
    let t0 = Instant::now();
    sim.virialize_sampled(10000)?;
    println!("  Virialized in {:.2}s\n", t0.elapsed().as_secs_f64());

    // Initial state
    let ke_init = sim.kinetic_energy()?;
    let seg_init = sim.segregation_distance()?;
    println!("Initial state (after virialization):");
    println!("  KE₀ = {:.4e}", ke_init);
    println!("  S₀ = {:.6}", seg_init);
    println!();

    // Time series file
    let ts_filename = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_filename)?);
    writeln!(ts_file, "step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,seg_max,step_time_ms")?;

    let ke_ref = ke_init;
    let mut tau = cosmo.tau_start;
    let mut seg_max = seg_init;

    // Save step 0 render data
    save_render_data(&sim, 0, seg_init, 1.0, Z_INIT, BOX_SIZE, &render_data_dir)?;

    println!("Starting simulation loop...\n");
    println!("  Step        z     KE/KE₀      Seg     S_max    ms/step");
    println!("---------------------------------------------------------------");

    for step in 1..=MAX_STEPS {
        let t_step = Instant::now();

        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = 1.0 / a - 1.0;

        // Pure BH step with Hubble friction
        sim.step_with_expansion_dkd_gpu(DT, a, h, dtau_per_dt)?;
        tau += DT * dtau_per_dt;

        let step_time = t_step.elapsed().as_millis() as f64;
        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation_distance()?;

        let ke_ratio = ke / ke_ref;
        seg_max = seg_max.max(seg);

        // Write CSV every step
        writeln!(ts_file, "{},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6},{:.6e},{:.6e},{:.1}",
            step, step as f64 * DT, z, a, h, ke, ke_ratio, seg, seg_max, step_time)?;

        // Flush every 10 steps
        if step % 10 == 0 {
            ts_file.flush()?;
        }

        // Progress every 100 steps or first 20
        if step <= 20 || step % 100 == 0 {
            println!("  {:5}   {:.3}   {:7.4}   {:.4}   {:.4}   {:6.0}",
                step, z, ke_ratio, seg, seg_max, step_time);
        }

        // Validation at step 20
        if step == 20 {
            println!("\n=== VALIDATION @ step 20 ===");
            println!("  KE/KE₀ = {:.4} (expected < 5)", ke_ratio);
            if ke_ratio > 5.0 {
                println!("  FAIL: KE/KE₀ > 5 — physics invalid, stopping");
                break;
            } else {
                println!("  PASS: KE/KE₀ < 5 — continuing production run");
            }
            println!();
        }

        // Save render data every FRAME_INTERVAL steps
        if step % FRAME_INTERVAL == 0 {
            save_render_data(&sim, step, seg, ke_ratio, z.max(0.0), BOX_SIZE, &render_data_dir)?;
        }

        // Stop conditions
        if ke.is_nan() || ke.is_infinite() || ke_ratio > 100.0 {
            println!("\n=== STOPPING: KE explosion (KE/KE₀ = {:.1}) ===", ke_ratio);
            break;
        }

        if z < 0.01 {
            println!("\n=== Reached z ≈ 0, simulation complete ===");
            break;
        }
    }

    ts_file.flush()?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Results:");
    println!("  S_max = {:.6}", seg_max);
    println!("  Output: {}", ts_filename);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(feature = "cuda")]
fn save_render_data(
    sim: &GpuNBodySimulation,
    step: usize,
    seg: f64,
    ke_ratio: f64,
    z: f64,
    box_size: f64,
    render_data_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let pos = sim.get_positions()?;
    let signs = sim.get_signs()?;

    let path = format!("{}/step_{:06}.bin", render_data_dir, step);
    let mut file = File::create(&path)?;

    // Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + z(f64) + n(u32)
    let n = (pos.len() / 3) as u32;
    file.write_all(&(step as u32).to_le_bytes())?;
    file.write_all(&box_size.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    file.write_all(&ke_ratio.to_le_bytes())?;
    file.write_all(&z.to_le_bytes())?;
    file.write_all(&n.to_le_bytes())?;

    // Convert f64 positions to f32 for render data
    let pos_f32: Vec<f32> = pos.iter().map(|&x| x as f32).collect();
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(pos_f32.as_ptr() as *const u8, pos_f32.len() * 4)
    };
    file.write_all(pos_bytes)?;

    // Convert i32 signs to i8
    let signs_i8: Vec<i8> = signs.iter().map(|&s| s as i8).collect();
    file.write_all(&signs_i8.iter().map(|&s| s as u8).collect::<Vec<u8>>())?;

    eprintln!("[data] step_{:06}.bin saved (z={:.2})", step, z);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled!");
}
