//! Test box size configurations for optimal filament visibility
//! Config A: box=585 Mpc (same density as 20M reference)
//! Config B: box=400 Mpc (higher resolution)

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::env;

const N_GRID: usize = 46;  // 46³ ≈ 97K
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 2500;
const VIRIAL_FACTOR: f64 = 0.8;
const N_S: f64 = 0.96;
const K0: f64 = 0.02;

fn ifft_3d(data: &mut [Complex<f64>], ifft: &std::sync::Arc<dyn rustfft::Fft<f64>>, n: usize) -> Vec<f64> {
    let n3 = n * n * n;
    for iy in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|iz| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for iz in 0..n { data[iz * n * n + iy * n + ix] = slice[iz]; }
        }
    }
    for iz in 0..n {
        for ix in 0..n {
            let mut slice: Vec<Complex<f64>> = (0..n).map(|iy| data[iz * n * n + iy * n + ix]).collect();
            ifft.process(&mut slice);
            for iy in 0..n { data[iz * n * n + iy * n + ix] = slice[iy]; }
        }
    }
    for iz in 0..n {
        for iy in 0..n {
            let base = iz * n * n + iy * n;
            let mut slice: Vec<Complex<f64>> = data[base..base+n].to_vec();
            ifft.process(&mut slice);
            for ix in 0..n { data[base + ix] = slice[ix]; }
        }
    }
    let norm = 1.0 / (n3 as f64);
    data.iter().map(|c| c.re * norm).collect()
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_zeldovich_ics(n_total: usize, box_size: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
    let n_grid = (n_total as f64).powf(1.0/3.0).ceil() as usize;
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = StdRng::seed_from_u64(seed);

    let dk = 2.0 * PI / box_size;
    let half_n = n_grid / 2;
    let spacing = box_size / n_grid as f64;
    let half_box = box_size / 2.0;
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;
    let amplitude = 0.01;

    let mut delta_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let normal = Normal::new(0.0, 1.0).unwrap();

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };
                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k = (kx*kx + ky*ky + kz*kz).sqrt();
                if k < 1e-10 { continue; }
                let pk = k.powf(N_S) / (1.0 + (k / K0).powi(4));
                let sigma_k = pk.sqrt() * amplitude * d_growth;
                let re: f64 = rng.sample(&normal) * sigma_k;
                let im: f64 = rng.sample(&normal) * sigma_k;
                delta_k[idx] = Complex::new(re, im);
            }
        }
    }

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..=half_n {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let iz_conj = if iz == 0 { 0 } else { n_grid - iz };
                let iy_conj = if iy == 0 { 0 } else { n_grid - iy };
                let ix_conj = if ix == 0 { 0 } else { n_grid - ix };
                let idx_conj = iz_conj * n_grid * n_grid + iy_conj * n_grid + ix_conj;
                if idx < idx_conj { delta_k[idx_conj] = delta_k[idx].conj(); }
            }
        }
    }

    let mut psi_x_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_y_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];
    let mut psi_z_k: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n3];

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };
                let kx = kx_idx as f64 * dk;
                let ky = ky_idx as f64 * dk;
                let kz = kz_idx as f64 * dk;
                let k2 = kx*kx + ky*ky + kz*kz;
                if k2 < 1e-20 { continue; }
                let minus_i = Complex::new(0.0, -1.0);
                psi_x_k[idx] = minus_i * kx * delta_k[idx] / k2;
                psi_y_k[idx] = minus_i * ky * delta_k[idx] / k2;
                psi_z_k[idx] = minus_i * kz * delta_k[idx] / k2;
            }
        }
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);
    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    let mut max_disp = 0.0f64;
    for i in 0..n3 {
        let d = (psi_x[i]*psi_x[i] + psi_y[i]*psi_y[i] + psi_z[i]*psi_z[i]).sqrt();
        if d > max_disp { max_disp = d; }
    }
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-10 { target_disp / max_disp } else { 1.0 };

    let virial_velocity = ((n3 as f64) / box_size).sqrt() * VIRIAL_FACTOR;
    let n_positive = (n3 as f64 / (1.0 + ETA)) as usize;

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs: Vec<i8> = Vec::with_capacity(n3);

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;
                let mut x = x0 + psi_x[idx] * scale;
                let mut y = y0 + psi_y[idx] * scale;
                let mut z = z0 + psi_z[idx] * scale;
                while x > half_box { x -= box_size; }
                while x < -half_box { x += box_size; }
                while y > half_box { y -= box_size; }
                while y < -half_box { y += box_size; }
                while z > half_box { z -= box_size; }
                while z < -half_box { z += box_size; }
                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);
                let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
                let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
                let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
                velocities.push(vx as f32);
                velocities.push(vy as f32);
                velocities.push(vz as f32);
                signs.push(1i8);
            }
        }
    }

    for i in 0..n_positive { signs[i] = 1; }
    for i in n_positive..n3 { signs[i] = -1; }
    signs.shuffle(&mut rng);

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn save_render_data(step: usize, pos: &[f32], signs: &[i8], box_size: f64, seg: f64, ke_ratio: f64, z: f64, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let n = (pos.len() / 3) as u32;
    file.write_all(&(step as u32).to_le_bytes())?;
    file.write_all(&box_size.to_le_bytes())?;
    file.write_all(&seg.to_le_bytes())?;
    file.write_all(&ke_ratio.to_le_bytes())?;
    file.write_all(&z.to_le_bytes())?;
    file.write_all(&n.to_le_bytes())?;
    let pos_bytes: &[u8] = unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 4) };
    file.write_all(pos_bytes)?;
    let signs_bytes: &[u8] = unsafe { std::slice::from_raw_parts(signs.as_ptr() as *const u8, signs.len()) };
    file.write_all(signs_bytes)?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn run_config(config_name: &str, box_size: f64, output_dir: &str) -> Result<(f64, f64, Option<f64>), Box<dyn std::error::Error>> {
    println!("\n======================================================================");
    println!("Config {}: box = {:.0} Mpc", config_name, box_size);
    println!("======================================================================\n");

    let n_total = N_GRID * N_GRID * N_GRID;
    let r_cut = box_size / 16.0;

    println!("  N = {} ({}³)", n_total, N_GRID);
    println!("  box = {:.1} Mpc", box_size);
    println!("  r_cut = {:.2} Mpc", r_cut);
    println!("  density = {:.2e} particles/Mpc³", n_total as f64 / box_size.powi(3));

    let (positions, velocities, signs) = generate_zeldovich_ics(n_total, box_size, 42);

    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / TOTAL_STEPS as f64;
    let dtau_per_dt = dtau_cosmo / DT;

    let mut sim = GpuNBodyTwoPass::with_custom_ics(positions, velocities, signs, box_size)?;
    sim.set_theta(THETA);

    let ke_init = sim.kinetic_energy()?;
    let mut tau = cosmo.tau_start;
    let mut seg_max = 0.0f64;
    let mut z_at_seg_max = Z_INIT;
    let mut onset_z: Option<f64> = None;

    let ts_path = format!("{}/time_series.csv", output_dir);
    let mut ts_file = BufWriter::new(File::create(&ts_path)?);
    writeln!(ts_file, "step,z,ke_ratio,segregation")?;

    println!("\n  Running {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE₀    Seg");
    println!("  --------------------------------");

    for step in 1..=TOTAL_STEPS {
        let (a, h) = cosmo.get_params_at_tau(tau);
        let z = (1.0 / a - 1.0).max(0.0);

        sim.step_treepm_gpu(DT, r_cut, h, dtau_per_dt)?;
        tau += dtau_per_dt * DT;

        if step % 10 == 0 || step == 1 {
            let ke = sim.kinetic_energy()?;
            let seg = sim.segregation()?;
            let ke_ratio = ke / ke_init;

            if seg > seg_max {
                seg_max = seg;
                z_at_seg_max = z;
            }

            if onset_z.is_none() && seg > 0.05 {
                onset_z = Some(z);
                println!("  >>> ONSET at step {} (z = {:.2})", step, z);
            }

            writeln!(ts_file, "{},{:.4},{:.4},{:.6}", step, z, ke_ratio, seg)?;

            if step % 100 == 0 || step <= 10 {
                println!("  {:4}   {:.2}   {:6.2}   {:.4}", step, z, ke_ratio, seg);
            }

            // Save render data at key steps
            if step == 1000 || step == 2000 {
                let pos = sim.get_positions()?;
                let signs = sim.get_signs()?;
                let path = format!("{}/step_{:04}.bin", output_dir, step);
                save_render_data(step, &pos, &signs, box_size, seg, ke_ratio, z, &path)?;
                println!("  >>> Saved {}", path);
            }
        }
    }

    ts_file.flush()?;

    println!("\n  Results:");
    println!("    S_max = {:.4} at z = {:.2}", seg_max, z_at_seg_max);
    if let Some(z) = onset_z {
        println!("    Onset z = {:.2}", z);
    } else {
        println!("    Onset: NOT DETECTED");
    }

    Ok((seg_max, z_at_seg_max, onset_z))
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Box Size Configuration Test (100K)                           ║");
    println!("║   Config A: 585 Mpc (20M density)  vs  Config B: 400 Mpc       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S");

    // Config A: 585 Mpc (same density as 20M reference)
    let dir_a = format!("/app/output/config_A_{}", timestamp);
    fs::create_dir_all(&dir_a)?;
    let (seg_max_a, z_max_a, onset_a) = run_config("A (585 Mpc)", 585.0, &dir_a)?;

    // Config B: 400 Mpc (higher resolution)
    let dir_b = format!("/app/output/config_B_{}", timestamp);
    fs::create_dir_all(&dir_b)?;
    let (seg_max_b, z_max_b, onset_b) = run_config("B (400 Mpc)", 400.0, &dir_b)?;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║   COMPARISON RESULTS                                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Config A (585 Mpc, 20M density):");
    println!("  S_max = {:.4} at z = {:.2}", seg_max_a, z_max_a);
    println!("  Onset z = {:?}", onset_a);
    let valid_a = onset_a.map(|z| z >= 2.0 && z <= 3.0).unwrap_or(false) && seg_max_a > 0.3;
    println!("  Valid: {} (onset ∈ [2,3] && S_max > 0.3)", if valid_a { "✅" } else { "❌" });

    println!("\nConfig B (400 Mpc, high res):");
    println!("  S_max = {:.4} at z = {:.2}", seg_max_b, z_max_b);
    println!("  Onset z = {:?}", onset_b);
    let valid_b = onset_b.map(|z| z >= 2.0 && z <= 3.0).unwrap_or(false) && seg_max_b > 0.3;
    println!("  Valid: {} (onset ∈ [2,3] && S_max > 0.3)", if valid_b { "✅" } else { "❌" });

    println!("\nRender data saved:");
    println!("  Config A: {}/step_1000.bin, step_2000.bin", dir_a);
    println!("  Config B: {}/step_1000.bin, step_2000.bin", dir_b);

    if valid_a && valid_b {
        if seg_max_b > seg_max_a {
            println!("\n>>> RECOMMENDATION: Config B (400 Mpc) - higher S_max");
        } else {
            println!("\n>>> RECOMMENDATION: Config A (585 Mpc) - matches 20M density");
        }
    } else if valid_b {
        println!("\n>>> RECOMMENDATION: Config B (400 Mpc)");
    } else if valid_a {
        println!("\n>>> RECOMMENDATION: Config A (585 Mpc)");
    } else {
        println!("\n>>> WARNING: Neither config passed validation criteria");
    }

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires --features cuda,cufft");
}
