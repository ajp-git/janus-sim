//! Phase 10.9 — Mini-run TreePM Janus, N=10K Zel'dovich, 15000 steps z=10→z=0.
//!
//! Validates the Phase 10.7 fixes (r_s parameter + Springel T(x)) over a full
//! cosmological evolution. Snapshots every 20 steps for full evolution film.
//!
//! Auto-STOP on:
//!  - NaN/Inf in positions or velocities
//!  - v_rms+ or v_rms- > 5000 km/s
//!  - GPU/CUDA crash (propagated as Result error)
//!
//! Final REPORT.md compares metrics vs Barnes-Hut historical baseline:
//!  - Corr(δ⁺, δ⁻) ≈ -0.07
//!  - σ8 ≈ 0.70 (proxy: delta_grid_rms_plus)
//!  - t₀ ≈ 15.87 Gyr
//!  - PAS de pic à |k|=4, 8, 16 (octree resonance check)

#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_GRID: usize = 22; // 22³ = 10648
#[cfg(all(feature = "cuda", feature = "cufft"))]
const L_BOX: f64 = 100.0; // Mpc
#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_PM: usize = 64;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const Z_INIT: f64 = 10.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const Z_FINAL: f64 = 0.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const N_STEPS_MAX: usize = 15000;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const DT: f64 = 0.001; // Gyr
#[cfg(all(feature = "cuda", feature = "cufft"))]
const ETA: f64 = 1.045;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const MU: f64 = 19.0;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const H0: f64 = 69.9; // km/s/Mpc
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SOFTENING: f64 = 0.05;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SEED_IC: u64 = 42;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const SNAPSHOT_INTERVAL: usize = 20;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const METRIC_INTERVAL: usize = 5;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const THETA: f64 = 0.5;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const MPC_GYR_TO_KMS: f64 = 977.8;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const TAU_0_JANUS: f64 = 23.3011940229;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;
#[cfg(all(feature = "cuda", feature = "cufft"))]
const V_RMS_LIMIT: f64 = 5000.0; // km/s

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;
    if a < A_TRANSITION_JANUS {
        h0_gyr_inv / a.powf(1.5)
    } else {
        let cosh2_mu = (a / ALPHA_SQ_JANUS).max(1.0);
        let cosh_mu = cosh2_mu.sqrt();
        let mu_p = cosh_mu.acosh();
        let s2mu = (2.0 * mu_p).sinh();
        s2mu / (TAU_0_JANUS * ALPHA_SQ_JANUS * cosh2_mu * (1.0 + 0.5 * s2mu))
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_overdensity_grids(
    pos: &[f32],
    signs: &[i8],
    n_grid: usize,
    l_box: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_cells = n_grid * n_grid * n_grid;
    let cell = l_box / n_grid as f64;
    let half = (l_box * 0.5) as f32;
    let mut rho_plus = vec![0.0_f64; n_cells];
    let mut rho_minus = vec![0.0_f64; n_cells];
    let n_part = signs.len();
    let mut n_p = 0_usize;
    let mut n_m = 0_usize;
    for i in 0..n_part {
        let x = ((pos[i * 3] + half) as f64).rem_euclid(l_box) / cell;
        let y = ((pos[i * 3 + 1] + half) as f64).rem_euclid(l_box) / cell;
        let z = ((pos[i * 3 + 2] + half) as f64).rem_euclid(l_box) / cell;
        let ix = (x.floor() as usize) % n_grid;
        let iy = (y.floor() as usize) % n_grid;
        let iz = (z.floor() as usize) % n_grid;
        let fx = x - x.floor();
        let fy = y - y.floor();
        let fz = z - z.floor();
        let wx = [1.0 - fx, fx];
        let wy = [1.0 - fy, fy];
        let wz = [1.0 - fz, fz];
        let target: &mut Vec<f64> = if signs[i] > 0 {
            n_p += 1;
            &mut rho_plus
        } else {
            n_m += 1;
            &mut rho_minus
        };
        for ai in 0..2 {
            let ii = (ix + ai) % n_grid;
            for aj in 0..2 {
                let jj = (iy + aj) % n_grid;
                for ak in 0..2 {
                    let kk = (iz + ak) % n_grid;
                    target[ii * n_grid * n_grid + jj * n_grid + kk] += wx[ai] * wy[aj] * wz[ak];
                }
            }
        }
    }
    let mean_p = n_p as f64 / n_cells as f64;
    let mean_m = n_m as f64 / n_cells as f64;
    let delta_plus: Vec<f64> = if mean_p > 0.0 {
        rho_plus.iter().map(|r| r / mean_p - 1.0).collect()
    } else {
        vec![0.0; n_cells]
    };
    let delta_minus: Vec<f64> = if mean_m > 0.0 {
        rho_minus.iter().map(|r| r / mean_m - 1.0).collect()
    } else {
        vec![0.0; n_cells]
    };
    (delta_plus, delta_minus)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let xa = a[i] - mean_a;
        let xb = b[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da > 0.0 && db > 0.0 {
        num / (da.sqrt() * db.sqrt())
    } else {
        0.0
    }
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_vrms_split(vel: &[f32], signs: &[i8]) -> (f64, f64) {
    // vel in Mpc/Gyr, returned in km/s
    let mut sum_p = 0.0_f64;
    let mut sum_m = 0.0_f64;
    let mut n_p = 0_usize;
    let mut n_m = 0_usize;
    for i in 0..signs.len() {
        let v2 = (vel[i * 3] as f64).powi(2)
            + (vel[i * 3 + 1] as f64).powi(2)
            + (vel[i * 3 + 2] as f64).powi(2);
        if signs[i] > 0 {
            sum_p += v2;
            n_p += 1;
        } else {
            sum_m += v2;
            n_m += 1;
        }
    }
    let v_p = if n_p > 0 {
        (sum_p / n_p as f64).sqrt() * MPC_GYR_TO_KMS
    } else {
        0.0
    };
    let v_m = if n_m > 0 {
        (sum_m / n_m as f64).sqrt() * MPC_GYR_TO_KMS
    } else {
        0.0
    };
    (v_p, v_m)
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn compute_power_spectrum(delta: &[f64], n_grid: usize, l_box: f64, n_bins: usize) -> Vec<(f64, f64, usize)> {
    use rustfft::{num_complex::Complex64, FftPlanner};
    let mut data: Vec<Complex64> = delta.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_grid);
    // Do 3D FFT (3 passes)
    for ix in 0..n_grid {
        for iy in 0..n_grid {
            let s = ix * n_grid * n_grid + iy * n_grid;
            let mut row: Vec<Complex64> = data[s..s + n_grid].to_vec();
            fft.process(&mut row);
            data[s..s + n_grid].copy_from_slice(&row);
        }
    }
    for ix in 0..n_grid {
        for iz in 0..n_grid {
            let mut col: Vec<Complex64> = (0..n_grid)
                .map(|iy| data[ix * n_grid * n_grid + iy * n_grid + iz])
                .collect();
            fft.process(&mut col);
            for iy in 0..n_grid {
                data[ix * n_grid * n_grid + iy * n_grid + iz] = col[iy];
            }
        }
    }
    for iy in 0..n_grid {
        for iz in 0..n_grid {
            let mut col: Vec<Complex64> = (0..n_grid)
                .map(|ix| data[ix * n_grid * n_grid + iy * n_grid + iz])
                .collect();
            fft.process(&mut col);
            for ix in 0..n_grid {
                data[ix * n_grid * n_grid + iy * n_grid + iz] = col[ix];
            }
        }
    }
    let k_fund = 2.0 * std::f64::consts::PI / l_box;
    let k_nyq = std::f64::consts::PI * n_grid as f64 / l_box;
    let dk = k_nyq / n_bins as f64;
    let mut sum = vec![0.0_f64; n_bins];
    let mut counts = vec![0_usize; n_bins];
    for ix in 0..n_grid {
        let kxi = if ix <= n_grid / 2 {
            ix as i32
        } else {
            ix as i32 - n_grid as i32
        };
        for iy in 0..n_grid {
            let kyi = if iy <= n_grid / 2 {
                iy as i32
            } else {
                iy as i32 - n_grid as i32
            };
            for iz in 0..n_grid {
                let kzi = if iz <= n_grid / 2 {
                    iz as i32
                } else {
                    iz as i32 - n_grid as i32
                };
                let k = ((kxi * kxi + kyi * kyi + kzi * kzi) as f64).sqrt() * k_fund;
                if k < 1e-10 || k > k_nyq {
                    continue;
                }
                let bin = ((k / dk).floor() as usize).min(n_bins - 1);
                let id = ix * n_grid * n_grid + iy * n_grid + iz;
                sum[bin] += data[id].norm_sqr();
                counts[bin] += 1;
            }
        }
    }
    let mut out = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        let kc = (b as f64 + 0.5) * dk;
        let p = if counts[b] > 0 {
            sum[b] / counts[b] as f64
        } else {
            0.0
        };
        out.push((kc, p, counts[b]));
    }
    out
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn cross_power(delta_a: &[f64], delta_b: &[f64], n_grid: usize, l_box: f64, n_bins: usize) -> Vec<f64> {
    use rustfft::{num_complex::Complex64, FftPlanner};
    let to_fft = |delta: &[f64]| -> Vec<Complex64> {
        let mut data: Vec<Complex64> = delta.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_grid);
        for ix in 0..n_grid {
            for iy in 0..n_grid {
                let s = ix * n_grid * n_grid + iy * n_grid;
                let mut row: Vec<Complex64> = data[s..s + n_grid].to_vec();
                fft.process(&mut row);
                data[s..s + n_grid].copy_from_slice(&row);
            }
        }
        for ix in 0..n_grid {
            for iz in 0..n_grid {
                let mut col: Vec<Complex64> = (0..n_grid)
                    .map(|iy| data[ix * n_grid * n_grid + iy * n_grid + iz])
                    .collect();
                fft.process(&mut col);
                for iy in 0..n_grid {
                    data[ix * n_grid * n_grid + iy * n_grid + iz] = col[iy];
                }
            }
        }
        for iy in 0..n_grid {
            for iz in 0..n_grid {
                let mut col: Vec<Complex64> = (0..n_grid)
                    .map(|ix| data[ix * n_grid * n_grid + iy * n_grid + iz])
                    .collect();
                fft.process(&mut col);
                for ix in 0..n_grid {
                    data[ix * n_grid * n_grid + iy * n_grid + iz] = col[ix];
                }
            }
        }
        data
    };
    let fa = to_fft(delta_a);
    let fb = to_fft(delta_b);
    let k_fund = 2.0 * std::f64::consts::PI / l_box;
    let k_nyq = std::f64::consts::PI * n_grid as f64 / l_box;
    let dk = k_nyq / n_bins as f64;
    let mut sum = vec![0.0_f64; n_bins];
    let mut counts = vec![0_usize; n_bins];
    for ix in 0..n_grid {
        let kxi = if ix <= n_grid / 2 {
            ix as i32
        } else {
            ix as i32 - n_grid as i32
        };
        for iy in 0..n_grid {
            let kyi = if iy <= n_grid / 2 {
                iy as i32
            } else {
                iy as i32 - n_grid as i32
            };
            for iz in 0..n_grid {
                let kzi = if iz <= n_grid / 2 {
                    iz as i32
                } else {
                    iz as i32 - n_grid as i32
                };
                let k = ((kxi * kxi + kyi * kyi + kzi * kzi) as f64).sqrt() * k_fund;
                if k < 1e-10 || k > k_nyq {
                    continue;
                }
                let bin = ((k / dk).floor() as usize).min(n_bins - 1);
                let id = ix * n_grid * n_grid + iy * n_grid + iz;
                sum[bin] += (fa[id] * fb[id].conj()).re;
                counts[bin] += 1;
            }
        }
    }
    let mut out = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        let p = if counts[b] > 0 {
            sum[b] / counts[b] as f64
        } else {
            0.0
        };
        out.push(p);
    }
    out
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn write_snapshot_bin(
    path: &std::path::Path,
    pos: &[f32],
    vel: &[f32],
    signs: &[i8],
    a_plus: f64,
    t_gyr: f64,
    z: f64,
    step: usize,
) -> std::io::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut bw = std::io::BufWriter::new(f);
    // Compact custom format: 64-byte header + N×(3f32 pos + 3f32 vel + 1i8 sign)
    let n = signs.len() as u32;
    bw.write_all(b"JANSMINI")?;            // 8 bytes magic
    bw.write_all(&n.to_le_bytes())?;        // 4 bytes N
    bw.write_all(&(step as u32).to_le_bytes())?; // 4 bytes step
    bw.write_all(&z.to_le_bytes())?;        // 8 bytes z
    bw.write_all(&a_plus.to_le_bytes())?;   // 8 bytes a
    bw.write_all(&t_gyr.to_le_bytes())?;    // 8 bytes t_gyr
    bw.write_all(&[0u8; 24])?;              // 24 bytes pad (total 64)
    for i in 0..signs.len() {
        bw.write_all(&pos[i * 3].to_le_bytes())?;
        bw.write_all(&pos[i * 3 + 1].to_le_bytes())?;
        bw.write_all(&pos[i * 3 + 2].to_le_bytes())?;
        bw.write_all(&vel[i * 3].to_le_bytes())?;
        bw.write_all(&vel[i * 3 + 1].to_le_bytes())?;
        bw.write_all(&vel[i * 3 + 2].to_le_bytes())?;
        bw.write_all(&[signs[i] as u8])?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::janus_expansion::a_minus_from_a_plus;
    use janus::janus_expansion::compute_phi_factors;
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
    use janus::vsl_dynamic::CoupledFriedmann;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::fs;
    use std::io::Write;
    use std::time::Instant;

    let out_dir = "/app/output/janus_minirun_treepm_15k";
    fs::create_dir_all(format!("{}/snapshots", out_dir))?;

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  JANUS MINI-RUN TreePM 15K — Phase 10.9 validation              ║");
    println!("║  N=10K Zel'dovich, 15000 steps, z=10→z=0, snapshots every 20    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // === ICs Zel'dovich (lattice + 15% perturbation, 5% m+ / 95% m-) ===
    let n: usize = N_GRID.pow(3);
    let half = (L_BOX as f32) * 0.5;
    let dx = (L_BOX / N_GRID as f64) as f32;
    let displacement_amp = 0.15 * dx;
    let mut rng = StdRng::seed_from_u64(SEED_IC);
    let mut pos_f32 = Vec::with_capacity(n * 3);
    let mut vel = vec![0.0_f32; n * 3];
    let mut signs = Vec::with_capacity(n);
    let n_plus = (n as f64 / (1.0 + MU)).round() as usize; // ≈ 5%
    let mut idx = 0;
    for i in 0..N_GRID {
        for j in 0..N_GRID {
            for k in 0..N_GRID {
                let gx = (i as f32 + 0.5) * dx - half;
                let gy = (j as f32 + 0.5) * dx - half;
                let gz = (k as f32 + 0.5) * dx - half;
                let dxp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let dyp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let dzp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let mut x = gx + dxp;
                let mut y = gy + dyp;
                let mut z = gz + dzp;
                while x >= half {
                    x -= 2.0 * half;
                }
                while x < -half {
                    x += 2.0 * half;
                }
                while y >= half {
                    y -= 2.0 * half;
                }
                while y < -half {
                    y += 2.0 * half;
                }
                while z >= half {
                    z -= 2.0 * half;
                }
                while z < -half {
                    z += 2.0 * half;
                }
                pos_f32.push(x);
                pos_f32.push(y);
                pos_f32.push(z);
                signs.push(if idx < n_plus { 1_i8 } else { -1_i8 });
                idx += 1;
            }
        }
    }
    println!("[IC] N={}, n+={}, n-={} (μ={})", n, n_plus, n - n_plus, MU);

    // === GPU init ===
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32, vel, signs.clone(), L_BOX)?;
    // Use auto-default mass_factor (= G × Ω_m=0.3 × ρ_crit × V / N)
    // Auto-default already propagates correctly through the full TreePM kernel chain.
    let auto_mf = sim.get_mass_factor();
    sim.set_softening(SOFTENING);
    sim.set_theta(THETA);
    println!(
        "[GPU] sim allocated, n_pm={}, theta={}, mass_factor={:.4e} (auto)",
        N_PM, THETA, auto_mf
    );

    // TreePM parameters (PhotoNs canonical: r_s = 1.2·Δg, r_cut = 5·r_s)
    let dg = L_BOX / N_PM as f64;
    let r_s = 1.2 * dg;
    let r_cut = 5.0 * r_s;
    println!("[TreePM] dg={:.4}, r_s={:.4}, r_cut={:.4}", dg, r_s, r_cut);

    // CSV
    let csv_path = format!("{}/evolution.csv", out_dir);
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
    writeln!(
        csv,
        "step,z,t_Gyr,a,a_minus,c_bar,phi,v_rms_plus,v_rms_minus,corr_delta,sigma8_proxy,max_pos,max_vel"
    )?;

    // Run log
    let log_path = format!("{}/run.log", out_dir);
    let mut log = std::io::BufWriter::new(std::fs::File::create(&log_path)?);
    writeln!(log, "JANUS MINI-RUN TreePM 15K — start")?;

    // === Loop z=10 → z=0 ===
    let start = Instant::now();
    let mut a = 1.0 / (1.0 + Z_INIT);
    let mut t_gyr = 0.0_f64;
    let mut step = 0_usize;
    let mut stop_reason = String::new();
    let n_bins_pk = 16;

    loop {
        let z = 1.0 / a - 1.0;
        if z <= Z_FINAL || step >= N_STEPS_MAX {
            break;
        }

        let h_plus = compute_hubble_janus(a, H0);
        let a_minus = a_minus_from_a_plus(a, ETA);
        let h_minus = compute_hubble_janus(a_minus, H0);
        let c_ratio_sq = CoupledFriedmann::c_ratio_sq_at_z(z, ETA);
        let c_bar = c_ratio_sq.sqrt();
        let (phi, _phi_inv) = compute_phi_factors(a, ETA);

        // Step
        let res = sim.step_treepm_gpu_cosmo(
            DT, r_cut, r_s, a, a_minus, h_plus, h_minus, phi, c_ratio_sq, 1.0,
        );
        if let Err(e) = res {
            stop_reason = format!("CUDA error at step {}: {}", step, e);
            eprintln!("❌ {}", stop_reason);
            break;
        }

        // Update a (peculiar convention)
        let da = a * h_plus * DT;
        a += da;
        t_gyr += DT;

        // Metrics
        if step % METRIC_INTERVAL == 0 || step % SNAPSHOT_INTERVAL == 0 {
            let pos = sim.get_positions()?;
            let vel = sim.get_velocities()?;
            // NaN check
            let mut has_nan = false;
            let mut max_pos = 0.0_f32;
            let mut max_vel = 0.0_f32;
            for v in pos.iter() {
                if !v.is_finite() {
                    has_nan = true;
                    break;
                }
                let av = v.abs();
                if av > max_pos {
                    max_pos = av;
                }
            }
            if !has_nan {
                for v in vel.iter() {
                    if !v.is_finite() {
                        has_nan = true;
                        break;
                    }
                    let av = v.abs();
                    if av > max_vel {
                        max_vel = av;
                    }
                }
            }
            if has_nan {
                stop_reason = format!("NaN/Inf detected at step {}", step);
                eprintln!("❌ {}", stop_reason);
                break;
            }

            let (v_p, v_m) = compute_vrms_split(&vel, &signs);
            if v_p > V_RMS_LIMIT || v_m > V_RMS_LIMIT {
                stop_reason = format!(
                    "v_rms exceeded {} km/s at step {}: v+={:.0}, v-={:.0}",
                    V_RMS_LIMIT, step, v_p, v_m
                );
                eprintln!("❌ {}", stop_reason);
                break;
            }

            // Density grids
            let (delta_p, delta_m) = compute_overdensity_grids(&pos, &signs, 64, L_BOX);
            let corr = correlation(&delta_p, &delta_m);
            let sigma_proxy = {
                let n_cells = delta_p.len() as f64;
                let m: f64 = delta_p.iter().sum::<f64>() / n_cells;
                let var: f64 = delta_p.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n_cells;
                var.sqrt()
            };

            writeln!(
                csv,
                "{},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8},{:.2},{:.2},{:.6},{:.6},{:.4},{:.4}",
                step, z, t_gyr, a, a_minus, c_bar, phi, v_p, v_m, corr, sigma_proxy, max_pos, max_vel
            )?;
            csv.flush()?;

            if step % 100 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let speed = if elapsed > 0.0 {
                    step as f64 / elapsed
                } else {
                    0.0
                };
                let eta_min = if speed > 0.0 {
                    ((N_STEPS_MAX - step) as f64 / speed) / 60.0
                } else {
                    0.0
                };
                println!(
                    "  step {:6} | z={:.4} | v±={:.0}/{:.0} | corr={:+.4} | σ={:.3} | {:.0} step/s | ETA {:.1} min",
                    step, z, v_p, v_m, corr, sigma_proxy, speed, eta_min
                );
            }

            // Snapshot
            if step % SNAPSHOT_INTERVAL == 0 {
                let snap_path = std::path::PathBuf::from(format!(
                    "{}/snapshots/snap_{:06}.bin",
                    out_dir, step
                ));
                write_snapshot_bin(&snap_path, &pos, &vel, &signs, a, t_gyr, z, step)?;
            }
        }

        step += 1;
    }

    let total = start.elapsed().as_secs_f64();
    csv.flush()?;
    drop(csv);

    let final_z = 1.0 / a - 1.0;
    println!("\n[END] step={}, z_final={:.4}, t={:.2} Gyr, {:.2} min wall",
        step, final_z, t_gyr, total / 60.0);
    if !stop_reason.is_empty() {
        println!("[STOP] {}", stop_reason);
        writeln!(log, "STOP: {}", stop_reason)?;
    }

    // === Final snapshot + P(k) check ===
    let final_pos = sim.get_positions()?;
    let final_vel = sim.get_velocities()?;
    let snap_path =
        std::path::PathBuf::from(format!("{}/snapshots/snap_final.bin", out_dir));
    write_snapshot_bin(&snap_path, &final_pos, &final_vel, &signs, a, t_gyr, final_z, step)?;
    let (delta_p_final, delta_m_final) =
        compute_overdensity_grids(&final_pos, &signs, 64, L_BOX);
    let pk_plus = compute_power_spectrum(&delta_p_final, 64, L_BOX, n_bins_pk);
    let pk_minus = compute_power_spectrum(&delta_m_final, 64, L_BOX, n_bins_pk);
    let pk_cross = cross_power(&delta_p_final, &delta_m_final, 64, L_BOX, n_bins_pk);
    let final_corr = correlation(&delta_p_final, &delta_m_final);

    // === REPORT.md ===
    let report_path = format!("{}/REPORT.md", out_dir);
    let mut rep = std::io::BufWriter::new(std::fs::File::create(&report_path)?);
    writeln!(
        rep,
        "# Phase 10.9 — Mini-run TreePM 15K Janus z=10→z=0\n"
    )?;
    writeln!(rep, "**Generated** : {:?}", std::time::SystemTime::now())?;
    writeln!(rep, "**Branch** : feat/treepm-jpp-port (Phase 10.7+10.8 fixes)")?;
    writeln!(rep, "**Setup** : N={} ({} m+, {} m-), L={} Mpc, n_pm={}, μ={}, η={}",
        n, n_plus, n - n_plus, L_BOX, N_PM, MU, ETA)?;
    writeln!(rep, "**dt** : {} Gyr (fixed)", DT)?;
    writeln!(rep, "**TreePM** : r_s={:.4} Mpc, r_cut={:.4} Mpc, θ={}", r_s, r_cut, THETA)?;
    writeln!(rep, "**softening** : {} Mpc", SOFTENING)?;
    writeln!(rep)?;
    writeln!(rep, "## Run summary\n")?;
    writeln!(rep, "- Steps completed : **{}**", step)?;
    writeln!(rep, "- Final z         : **{:.4}**", final_z)?;
    writeln!(rep, "- Cosmic time     : **{:.2} Gyr**", t_gyr)?;
    writeln!(rep, "- Wall time       : {:.2} min", total / 60.0)?;
    if !stop_reason.is_empty() {
        writeln!(rep, "- **STOP reason** : {}", stop_reason)?;
    }
    writeln!(rep)?;
    writeln!(rep, "## Final state\n")?;
    let (v_p_f, v_m_f) = compute_vrms_split(&final_vel, &signs);
    writeln!(rep, "| Metric | Value | Reference (Barnes-Hut hist.) |")?;
    writeln!(rep, "|---|---|---|")?;
    writeln!(rep, "| Corr(δ⁺, δ⁻) | **{:+.4}** | ≈ -0.07 |", final_corr)?;
    let sigma_final: f64 = {
        let n_cells = delta_p_final.len() as f64;
        let m: f64 = delta_p_final.iter().sum::<f64>() / n_cells;
        let var: f64 =
            delta_p_final.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n_cells;
        var.sqrt()
    };
    writeln!(rep, "| σ_8 proxy (rms δ⁺) | **{:.4}** | ≈ 0.70 (Mpc/h scale-dependent) |", sigma_final)?;
    writeln!(rep, "| t₀ (cosmic time) | **{:.2} Gyr** | ≈ 15.87 Gyr |", t_gyr)?;
    writeln!(rep, "| v_rms+ | {:.1} km/s | < 5000 |", v_p_f)?;
    writeln!(rep, "| v_rms- | {:.1} km/s | < 5000 |", v_m_f)?;
    writeln!(rep)?;
    writeln!(rep, "## Power spectrum P(k) at z=z_final\n")?;
    writeln!(rep, "| bin | k_c [1/Mpc] | P_+(k) | P_-(k) | P_×(k) |")?;
    writeln!(rep, "|---|---|---|---|---|")?;
    for b in 0..n_bins_pk {
        writeln!(
            rep,
            "| {} | {:.4} | {:.3e} | {:.3e} | {:+.3e} |",
            b, pk_plus[b].0, pk_plus[b].1, pk_minus[b].1, pk_cross[b]
        )?;
    }
    writeln!(rep)?;
    writeln!(rep, "## Test critique : pic à |k|=4, 8, 16 ?\n")?;
    // P(k) at index k = 4*k_fund, 8*k_fund, 16*k_fund (nearest bin)
    let k_fund = 2.0 * std::f64::consts::PI / L_BOX;
    let dk_pk = (std::f64::consts::PI * 64.0 / L_BOX) / n_bins_pk as f64;
    let bins_check: [usize; 3] = [
        ((4.0 * k_fund) / dk_pk).floor() as usize,
        ((8.0 * k_fund) / dk_pk).floor() as usize,
        ((16.0 * k_fund) / dk_pk).floor() as usize,
    ];
    let mut peak_alert = false;
    for &b in &bins_check {
        if b >= n_bins_pk {
            continue;
        }
        // Peak detection: ratio P[b] / median(P[b-1], P[b+1])
        let p_self = pk_plus[b].1.max(1e-30);
        let p_prev = if b > 0 { pk_plus[b - 1].1 } else { p_self };
        let p_next = if b + 1 < n_bins_pk {
            pk_plus[b + 1].1
        } else {
            p_self
        };
        let neighbor = (p_prev + p_next) * 0.5;
        let ratio = if neighbor > 1e-30 {
            p_self / neighbor
        } else {
            1.0
        };
        let alert = ratio > 1.5;
        if alert {
            peak_alert = true;
        }
        writeln!(
            rep,
            "- bin {} (k≈{:.4}): P=P_+={:.3e}, ratio vs neighbors={:.2} {}",
            b,
            pk_plus[b].0,
            p_self,
            ratio,
            if alert {
                "⚠ POSSIBLE RESONANCE"
            } else {
                "OK"
            }
        )?;
    }
    if peak_alert {
        writeln!(
            rep,
            "\n⚠ **ALERT** : isolated peak detected at k=4/8/16. Octree resonance possible."
        )?;
    } else {
        writeln!(
            rep,
            "\n✅ **No isolated peak** at k=4/8/16. No octree resonance signature."
        )?;
    }
    writeln!(rep)?;
    writeln!(rep, "## CSV evolution\n")?;
    writeln!(rep, "Saved at `{}/evolution.csv`", out_dir)?;
    writeln!(rep, "Snapshots in `{}/snapshots/snap_*.bin`", out_dir)?;
    writeln!(rep)?;
    writeln!(rep, "## Verdict\n")?;
    if stop_reason.is_empty()
        && !peak_alert
        && final_corr < 0.0
        && v_p_f < V_RMS_LIMIT
        && v_m_f < V_RMS_LIMIT
    {
        writeln!(rep, "🟢 **PASS** — pipeline stable, Janus segregation correct (Corr<0), no resonance, v_rms within bounds.")?;
    } else {
        writeln!(rep, "🟡 **CHECK** — see flagged metrics above.")?;
    }
    rep.flush()?;
    drop(rep);

    println!("\n[REPORT] {}", report_path);
    println!("[CSV]    {}", csv_path);
    println!("[SNAPS]  {}/snapshots/", out_dir);

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires features: cuda cufft");
    std::process::exit(1);
}
