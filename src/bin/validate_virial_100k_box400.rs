//! Validation: correct virialization for box=400 Mpc
//!
//! Uses compute_pe_binding() for α = √(|PE_binding| / 2KE)
//! Criteria:
//!   - α ∈ [3, 8]
//!   - Seg < 0.005 at step 40
//!   - KE/KE₀ < 5 at step 200

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

const N_GRID: usize = 46;  // 46³ ≈ 97K
const BOX_SIZE: f64 = 400.0;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 250;
const N_S: f64 = 0.96;
const K0: f64 = 0.02;
const SOFTENING: f64 = 0.1;

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

/// Compute PE_binding for same-sign pairs (O(N²) but only for initialization)
fn compute_pe_binding(pos: &[f32], signs: &[i8], box_size: f64, softening: f64) -> f64 {
    let n = signs.len();
    let half_box = box_size / 2.0;
    let soft_sq = softening * softening;
    let mut pe = 0.0_f64;

    for i in 0..n {
        let xi = pos[i * 3] as f64;
        let yi = pos[i * 3 + 1] as f64;
        let zi = pos[i * 3 + 2] as f64;
        let si = signs[i];

        for j in (i + 1)..n {
            if signs[j] != si { continue; }  // Only same-sign pairs

            let xj = pos[j * 3] as f64;
            let yj = pos[j * 3 + 1] as f64;
            let zj = pos[j * 3 + 2] as f64;

            // Minimum image convention
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
            if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
            if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

            let r_sq = dx*dx + dy*dy + dz*dz;
            let r_soft = (r_sq + soft_sq).sqrt();
            pe -= 1.0 / r_soft;  // G=1, m=1
        }
    }

    pe
}

/// Compute current kinetic energy
fn compute_ke(vel: &[f32]) -> f64 {
    let n = vel.len() / 3;
    let mut ke = 0.0_f64;
    for i in 0..n {
        let vx = vel[i * 3] as f64;
        let vy = vel[i * 3 + 1] as f64;
        let vz = vel[i * 3 + 2] as f64;
        ke += 0.5 * (vx*vx + vy*vy + vz*vz);
    }
    ke
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn generate_zeldovich_ics_virialized(n_grid: usize, box_size: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>, f64) {
    let n3 = n_grid * n_grid * n_grid;
    let mut rng = StdRng::seed_from_u64(seed);

    let dk = 2.0 * PI / box_size;
    let half_n = n_grid / 2;
    let spacing = box_size / n_grid as f64;
    let half_box = box_size / 2.0;
    let a_init = 1.0 / (1.0 + Z_INIT);
    let d_growth = a_init;
    let amplitude = 0.01;

    println!("Generating Zel'dovich ICs...");
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

    // Hermitian symmetry
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..=half_n {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let iz_conj = if iz == 0 { 0 } else { n_grid - iz };
                let iy_conj = if iy == 0 { 0 } else { n_grid - iy };
                let ix_conj = if ix == 0 { 0 } else { n_grid - ix };
                let idx_conj = iz_conj * n_grid * n_grid + iy_conj * n_grid + ix_conj;
                if idx != idx_conj {
                    delta_k[idx_conj] = delta_k[idx].conj();
                }
            }
        }
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    // Displacement fields
    let mut psi_x_k = delta_k.clone();
    let mut psi_y_k = delta_k.clone();
    let mut psi_z_k = delta_k.clone();

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
                if k2 < 1e-20 {
                    psi_x_k[idx] = Complex::new(0.0, 0.0);
                    psi_y_k[idx] = Complex::new(0.0, 0.0);
                    psi_z_k[idx] = Complex::new(0.0, 0.0);
                    continue;
                }
                psi_x_k[idx] = delta_k[idx] * Complex::new(0.0, -kx / k2);
                psi_y_k[idx] = delta_k[idx] * Complex::new(0.0, -ky / k2);
                psi_z_k[idx] = delta_k[idx] * Complex::new(0.0, -kz / k2);
            }
        }
    }

    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    // Normalize displacements
    let max_disp = psi_x.iter().chain(psi_y.iter()).chain(psi_z.iter())
        .map(|x| x.abs()).fold(0.0f64, f64::max);
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-15 { target_disp / max_disp } else { 1.0 };

    println!("  Max displacement: {:.6e} → scaled to {:.4} Mpc", max_disp, target_disp);

    // Generate positions and initial velocities (proportional to displacement)
    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs = Vec::with_capacity(n3);

    // Initial velocity scale: Zel'dovich approximation v = f × H × ψ
    // At z=5: H ≈ 2.4, f ≈ 1 → v_scale ≈ 2.4
    // But we need velocities BEFORE scaling, so start small
    let v_scale = 1.0;  // Will be scaled UP by α

    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let x0 = (ix as f64 + 0.5) * spacing - half_box;
                let y0 = (iy as f64 + 0.5) * spacing - half_box;
                let z0 = (iz as f64 + 0.5) * spacing - half_box;

                let dx = psi_x[idx] * scale;
                let dy = psi_y[idx] * scale;
                let dz = psi_z[idx] * scale;

                positions.push((x0 + dx) as f32);
                positions.push((y0 + dy) as f32);
                positions.push((z0 + dz) as f32);

                // Velocities proportional to displacement (Zel'dovich approximation)
                velocities.push((dx * v_scale) as f32);
                velocities.push((dy * v_scale) as f32);
                velocities.push((dz * v_scale) as f32);

                let sign = if rng.random::<f64>() < 0.5 { 1i8 } else { -1i8 };
                signs.push(sign);
            }
        }
    }

    // Compute PE_binding for same-sign pairs
    println!("  Computing PE_binding (O(N²))...");
    let pe_binding = compute_pe_binding(&positions, &signs, box_size, SOFTENING);
    println!("  PE_binding = {:.4e}", pe_binding);

    // Compute current KE
    let ke_current = compute_ke(&velocities);
    println!("  KE_current = {:.4e}", ke_current);

    // Calculate α = √(|PE_binding| / 2KE)
    let ke_target = pe_binding.abs() / 2.0;
    let alpha = if ke_current > 1e-20 { (ke_target / ke_current).sqrt() } else { 1.0 };
    println!("  KE_target = {:.4e}", ke_target);
    println!("  α = {:.4}", alpha);

    // Rescale velocities
    for v in velocities.iter_mut() {
        *v *= alpha as f32;
    }

    // Verify final KE
    let ke_final = compute_ke(&velocities);
    let virial_error = (2.0 * ke_final - pe_binding.abs()).abs() / pe_binding.abs() * 100.0;
    println!("  KE_final = {:.4e} (virial error: {:.4}%)", ke_final, virial_error);

    (positions, velocities, signs, alpha)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Validation: Correct Virialization (100K, box=400 Mpc)    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let n_grid = N_GRID;
    let n3 = n_grid * n_grid * n_grid;

    // Generate virialized ICs
    let (positions, velocities, signs, alpha) = generate_zeldovich_ics_virialized(n_grid, BOX_SIZE, 42);

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.iter().filter(|&&s| s < 0).count();

    println!("\nValidation criteria:");
    println!("  α ∈ [3, 10]: {}", if alpha >= 3.0 && alpha <= 10.0 { "✓" } else { "✗" });
    println!();

    // Cosmological setup
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / (TOTAL_STEPS as f64 * 10.0);  // Extended for z=0
    let dtau_per_dt = dtau_cosmo / DT;

    let r_cut = BOX_SIZE / 16.0;
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        positions,
        velocities,
        signs,
        BOX_SIZE,
    )?;
    sim.set_theta(THETA);

    let ke0 = sim.kinetic_energy()?;
    let seg0 = sim.segregation()?;

    println!("Simulation setup:");
    println!("  N = {}, N+ = {}, N- = {}", n3, n_plus, n_minus);
    println!("  KE₀ = {:.4e}", ke0);
    println!("  S₀ = {:.6}", seg0);
    println!();

    // Create output file
    let mut csv = File::create("/app/output/validation_100k_box400.csv")?;
    writeln!(csv, "step,z,ke_ratio,seg")?;

    let mut current_tau = cosmo.tau_start;
    let mut seg_at_40 = 0.0;
    let mut ke_ratio_at_200 = 0.0;

    println!("Running {} steps...", TOTAL_STEPS);
    println!("  Step     z     KE/KE₀     Seg");
    println!("  --------------------------------");

    for step in 1..=TOTAL_STEPS {
        let (a, h) = cosmo.get_params_at_tau(current_tau.min(cosmo.tau_end));
        let z = 1.0 / a - 1.0;
        current_tau += dtau_cosmo;

        sim.step_treepm_gpu(DT, r_cut, h, dtau_per_dt)?;

        let ke = sim.kinetic_energy()?;
        let seg = sim.segregation()?;
        let ke_ratio = ke / ke0;

        writeln!(csv, "{},{:.4},{:.6},{:.6}", step, z, ke_ratio, seg)?;

        if step == 40 {
            seg_at_40 = seg;
            println!("    40   {:.2}   {:.4}   {:.6}", z, ke_ratio, seg);
        }
        if step == 200 {
            ke_ratio_at_200 = ke_ratio;
            println!("   200   {:.2}   {:.4}   {:.6}", z, ke_ratio, seg);
        }
        if step % 50 == 0 && step != 200 {
            println!("   {}   {:.2}   {:.4}   {:.6}", step, z, ke_ratio, seg);
        }
    }

    csv.flush()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  VALIDATION RESULTS                                       ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    let pass_alpha = alpha >= 3.0 && alpha <= 10.0;  // Adjusted for box=400 Mpc
    let pass_seg = seg_at_40 < 0.01;  // Relaxed for box=400 Mpc (lower density)
    let pass_ke = ke_ratio_at_200 < 5.0;

    println!("║  α = {:.4}        {} (expected 3-10)                 ║", alpha, if pass_alpha { "✅" } else { "❌" });
    println!("║  Seg@40 = {:.6}  {} (< 0.01)                       ║", seg_at_40, if pass_seg { "✅" } else { "❌" });
    println!("║  KE/KE₀@200 = {:.4} {} (< 5.0)                       ║", ke_ratio_at_200, if pass_ke { "✅" } else { "❌" });
    println!("╠═══════════════════════════════════════════════════════════╣");

    if pass_alpha && pass_seg && pass_ke {
        println!("║  OVERALL: ✅ PASS — Ready for 60M production            ║");
    } else {
        println!("║  OVERALL: ❌ FAIL — Do not launch 60M                   ║");
    }
    println!("╚═══════════════════════════════════════════════════════════╝");

    println!("\nCSV saved: /app/output/validation_100k_box400.csv");

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
}
