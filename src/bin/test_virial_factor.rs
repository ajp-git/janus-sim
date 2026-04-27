//! Quick test of virial_factor on box=400 Mpc
//! Goal: find factor that gives Seg < 0.005 at step 40 AND onset z ∈ [2.0, 3.0]

#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
#[cfg(all(feature = "cuda", feature = "cufft"))]
use janus::friedmann::{JanusParams, CosmoInterpolator};
use rand::prelude::*;
use rand_distr::Normal;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::env;

const N_GRID: usize = 46;  // 46³ ≈ 97K
const BOX_SIZE: f64 = 400.0;
const ETA: f64 = 1.045;
const THETA: f64 = 0.7;
const DT: f64 = 0.01;
const Z_INIT: f64 = 5.0;
const TOTAL_STEPS: usize = 2500;
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
fn generate_zeldovich_ics(n_grid: usize, box_size: f64, virial_factor: f64, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<i8>) {
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

    // IFFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid);

    // Compute displacement in x
    let mut psi_x_k = delta_k.clone();
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                let kx = kx_idx as f64 * dk;
                let k2 = {
                    let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                    let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };
                    let ky = ky_idx as f64 * dk;
                    let kz = kz_idx as f64 * dk;
                    kx*kx + ky*ky + kz*kz
                };
                if k2 < 1e-20 { psi_x_k[idx] = Complex::new(0.0, 0.0); continue; }
                psi_x_k[idx] = delta_k[idx] * Complex::new(0.0, -kx / k2);
            }
        }
    }
    let psi_x = ifft_3d(&mut psi_x_k, &ifft, n_grid);

    let mut psi_y_k = delta_k.clone();
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                let ky = ky_idx as f64 * dk;
                let k2 = {
                    let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                    let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };
                    let kx = kx_idx as f64 * dk;
                    let kz = kz_idx as f64 * dk;
                    kx*kx + ky*ky + kz*kz
                };
                if k2 < 1e-20 { psi_y_k[idx] = Complex::new(0.0, 0.0); continue; }
                psi_y_k[idx] = delta_k[idx] * Complex::new(0.0, -ky / k2);
            }
        }
    }
    let psi_y = ifft_3d(&mut psi_y_k, &ifft, n_grid);

    let mut psi_z_k = delta_k.clone();
    for iz in 0..n_grid {
        for iy in 0..n_grid {
            for ix in 0..n_grid {
                let idx = iz * n_grid * n_grid + iy * n_grid + ix;
                let kz_idx = if iz <= half_n { iz as i32 } else { iz as i32 - n_grid as i32 };
                let kz = kz_idx as f64 * dk;
                let k2 = {
                    let kx_idx = if ix <= half_n { ix as i32 } else { ix as i32 - n_grid as i32 };
                    let ky_idx = if iy <= half_n { iy as i32 } else { iy as i32 - n_grid as i32 };
                    let kx = kx_idx as f64 * dk;
                    let ky = ky_idx as f64 * dk;
                    kx*kx + ky*ky + kz*kz
                };
                if k2 < 1e-20 { psi_z_k[idx] = Complex::new(0.0, 0.0); continue; }
                psi_z_k[idx] = delta_k[idx] * Complex::new(0.0, -kz / k2);
            }
        }
    }
    let psi_z = ifft_3d(&mut psi_z_k, &ifft, n_grid);

    // Normalize
    let max_disp = psi_x.iter().chain(psi_y.iter()).chain(psi_z.iter())
        .map(|x| x.abs()).fold(0.0f64, f64::max);
    let target_disp = spacing * 0.3;
    let scale = if max_disp > 1e-15 { target_disp / max_disp } else { 1.0 };

    let virial_velocity = scale * box_size * virial_factor;
    println!("  virial_velocity = {:.4} (factor = {:.2})", virial_velocity, virial_factor);

    let mut positions = Vec::with_capacity(n3 * 3);
    let mut velocities = Vec::with_capacity(n3 * 3);
    let mut signs = Vec::with_capacity(n3);

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

                let x = x0 + dx;
                let y = y0 + dy;
                let z = z0 + dz;

                positions.push(x as f32);
                positions.push(y as f32);
                positions.push(z as f32);

                let vx = dx * virial_velocity / scale.max(1e-10);
                let vy = dy * virial_velocity / scale.max(1e-10);
                let vz = dz * virial_velocity / scale.max(1e-10);

                velocities.push(vx as f32);
                velocities.push(vy as f32);
                velocities.push(vz as f32);

                let sign = if rng.random::<f64>() < 0.5 { 1i8 } else { -1i8 };
                signs.push(sign);
            }
        }
    }

    (positions, velocities, signs)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn run_test(virial_factor: f64) -> (f64, f64, f64) {
    println!("\n=== Testing virial_factor = {} ===", virial_factor);

    let n_grid = N_GRID;
    let n3 = n_grid * n_grid * n_grid;

    let (positions, velocities, signs) = generate_zeldovich_ics(n_grid, BOX_SIZE, virial_factor, 42);

    let n_plus = signs.iter().filter(|&&s| s > 0).count();
    let n_minus = signs.iter().filter(|&&s| s < 0).count();

    // Cosmological setup
    let params = JanusParams::from_eta(ETA);
    let cosmo = CosmoInterpolator::new(&params, Z_INIT);
    let n_steps_to_z0 = TOTAL_STEPS as f64;
    let dtau_cosmo = (cosmo.tau_end - cosmo.tau_start) / n_steps_to_z0;
    let dtau_per_dt = dtau_cosmo / DT;

    let r_cut = BOX_SIZE / 16.0;
    let mut sim = GpuNBodyTwoPass::with_custom_ics(
        positions.clone(),
        velocities,
        signs.clone(),
        BOX_SIZE,
    ).expect("Failed to create simulation");
    sim.set_theta(THETA);

    let ke0 = sim.kinetic_energy().unwrap();
    let seg0 = sim.segregation().unwrap();

    println!("  N = {}, N+ = {}, N- = {}", n3, n_plus, n_minus);
    println!("  KE₀ = {:.4e}, S₀ = {:.6}", ke0, seg0);

    // Store seg values to find onset
    let mut seg_history: Vec<(usize, f64, f64)> = Vec::new();
    let mut seg_at_40 = 0.0;
    let mut s_max = seg0;
    let mut current_tau = cosmo.tau_start;

    for step in 1..=TOTAL_STEPS {
        // Get cosmological parameters
        let (a, h) = cosmo.get_params_at_tau(current_tau);
        let z = 1.0 / a - 1.0;
        let dtau_eff = dtau_per_dt;
        current_tau += dtau_cosmo;

        sim.step_treepm_gpu(DT, r_cut, h, dtau_eff).expect("step failed");

        let seg = sim.segregation().unwrap();
        s_max = s_max.max(seg);

        if step == 40 {
            seg_at_40 = seg;
            println!("  Step 40: z={:.2}, Seg={:.6}", z, seg);
        }

        seg_history.push((step, z, seg));

        if step % 100 == 0 {
            println!("  Step {}: z={:.2}, Seg={:.6}", step, z, seg);
        }

        // Early stop if we've passed z=2
        if z < 1.5 {
            println!("  Reached z={:.2}, stopping early", z);
            break;
        }
    }

    // Find onset redshift (where seg first exceeds 0.02)
    let onset_threshold = 0.02;
    let mut onset_z = -1.0;
    for (_, z, seg) in &seg_history {
        if *seg > onset_threshold {
            onset_z = *z;
            break;
        }
    }

    println!("  Result: Seg@40={:.6}, onset_z={:.2}", seg_at_40, onset_z);
    (virial_factor, seg_at_40, onset_z)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Virial Factor Calibration Test (box=400 Mpc, N≈100K)     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let factors: Vec<f64> = env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();

    let factors = if factors.is_empty() {
        vec![1.2, 1.5]
    } else {
        factors
    };

    let mut results = Vec::new();

    for &vf in &factors {
        let (factor, seg_40, onset_z) = run_test(vf);
        results.push((factor, seg_40, onset_z));
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  RESULTS SUMMARY                                          ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Factor    Seg@40    onset_z    PASS?                     ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    for (factor, seg_40, onset_z) in &results {
        let pass_seg = *seg_40 < 0.005;
        let pass_onset = *onset_z >= 2.0 && *onset_z <= 3.0;
        let status = if pass_seg && pass_onset { "✅ PASS" } else { "❌ FAIL" };
        let seg_mark = if pass_seg { "✓" } else { "✗" };
        let onset_mark = if pass_onset { "✓" } else { "✗" };
        println!("║  {:.1}      {:.6}   {:.2}       {} (seg:{}, z:{})  ║",
            factor, seg_40, onset_z, status, seg_mark, onset_mark);
    }
    println!("╚═══════════════════════════════════════════════════════════╝");
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("This binary requires --features cuda,cufft");
}
