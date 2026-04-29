//! Phase 10A.5a Ewald — Validation force-field cross-correlation r(k) :
//! TreePM Janus GPU vs PP-direct **avec sommation Ewald** sur Zel'dovich N=10K.
//!
//! Phase 10.8 variant : remplace pp_direct_janus (MIC seul) par
//! pp_direct_forces_janus_ewald (rigoureusement périodique). Apples-to-apples
//! avec le GPU TreePM nativement périodique (PM via FFT + Tree borné).
//!
//! Setup identique : 22³=10648 particules lattice + 15% dx, 5% m+ / 95% m- (μ=19),
//! L=100, n_pm=64, r_cut=9.375, r_s=1.875, θ=0.5, softening=0.05, cosmo neutre.
//!
//! Métrique: cross-correlation r(k). Critère: min r(k) > 0.99.

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rustfft::{num_complex::Complex64, FftPlanner};

    println!("=== Phase 10A.5a — GPU TreePM Janus vs PP-direct (Zel'dovich N=10K) ===");

    let n_per_dim: usize = 22;
    let n: usize = n_per_dim.pow(3);
    let l = 100.0_f64;
    let n_pm = 64;
    let dg = l / n_pm as f64;
    let r_cut = 6.0 * dg;
    let r_s = 1.2 * dg;
    let theta = 0.5;
    let softening = 0.05;
    let half = l as f32 * 0.5;

    // Zel'dovich-like ICs: lattice + 15% dx random displacement
    let mut rng = StdRng::seed_from_u64(42);
    let dx = (l / n_per_dim as f64) as f32;
    let displacement_amp = 0.15 * dx;
    let mut pos_f32 = Vec::with_capacity(n * 3);
    let mut pos_f64: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    let mut vel = vec![0.0_f32; n * 3];
    let mut signs = Vec::with_capacity(n);

    // 5% m+ / 95% m- mix (Janus μ=19)
    let n_plus = (n as f64 * 0.05).round() as usize;
    println!("  N={}, n_plus={}, n_minus={}", n, n_plus, n - n_plus);

    let mut idx = 0;
    for i in 0..n_per_dim {
        for j in 0..n_per_dim {
            for k in 0..n_per_dim {
                let gx = (i as f32 + 0.5) * dx - half;
                let gy = (j as f32 + 0.5) * dx - half;
                let gz = (k as f32 + 0.5) * dx - half;
                let dxp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let dyp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let dzp = (rng.random::<f32>() - 0.5) * 2.0 * displacement_amp;
                let mut x = gx + dxp;
                let mut y = gy + dyp;
                let mut z = gz + dzp;
                while x >= half { x -= 2.0 * half; }
                while x < -half { x += 2.0 * half; }
                while y >= half { y -= 2.0 * half; }
                while y < -half { y += 2.0 * half; }
                while z >= half { z -= 2.0 * half; }
                while z < -half { z += 2.0 * half; }
                pos_f32.push(x);
                pos_f32.push(y);
                pos_f32.push(z);
                pos_f64.push((x as f64, y as f64, z as f64));
                signs.push(if idx < n_plus { 1_i8 } else { -1_i8 });
                idx += 1;
            }
        }
    }

    // ========== GPU forces ==========
    println!("Computing GPU TreePM Janus forces...");
    let t0 = std::time::Instant::now();
    let mut sim = GpuNBodyTwoPass::with_custom_ics(pos_f32.clone(), vel, signs.clone(), l)?;
    sim.set_mass_factor(1.0);
    sim.set_softening(softening as f64);
    sim.set_theta(theta);

    // Cosmologie neutre pour comparison directe avec PP
    let a_plus = 1.0_f64;
    let a_minus = 1.0_f64;
    let h_plus = 0.0_f64;
    let h_minus = 0.0_f64;
    let phi = 1.0_f64;
    let c_ratio_sq = 1.0_f64;
    let repulsion_scale = 1.0_f64;
    let dt = 0.0_f64; // dt=0 → force computation only, no drift

    sim.step_treepm_gpu_cosmo(
        dt, r_cut, r_s, a_plus, a_minus, h_plus, h_minus,
        phi, c_ratio_sq, repulsion_scale,
    )?;
    // Récupérer les accélérations (acc field)
    let acc_gpu = sim.get_acc()?;
    println!("  GPU: {:.2}s", t0.elapsed().as_secs_f64());

    // ========== PP-direct Ewald (CPU reference) ==========
    println!("Computing PP-direct Ewald (N²={:.2e} pairs × ~1500 images)...", (n*n) as f64);
    let t0 = std::time::Instant::now();
    let mass_f64 = vec![1.0_f64; n];
    let signs_i32: Vec<i32> = signs.iter().map(|&s| s as i32).collect();
    let coupling = janus::treepm::janus::JanusCoupling {
        phi,
        c_ratio_sq,
        repulsion_scale,
    };
    // n_real = 4, n_fourier = 4 — converged to machine precision per
    // test_ewald_convergence (Phase 10.8). Default Hernquist & Bouchet 1991.
    let acc_pp = janus::treepm::pp_reference::pp_direct_forces_janus_ewald(
        &pos_f64,
        &mass_f64,
        &signs_i32,
        l,
        softening as f64,
        softening as f64,
        1.0,
        &coupling,
        4,
        4,
    );
    println!("  PP-Ewald: {:.2}s", t0.elapsed().as_secs_f64());

    // ========== Force-field cross-correlation per k ==========
    println!("Computing cross-correlation per k...");
    let cic_dep = |weights: &[f64]| -> Vec<Complex64> {
        let n_cells = n_pm * n_pm * n_pm;
        let mut grid = vec![0.0_f64; n_cells];
        let cell = l / n_pm as f64;
        for i in 0..n {
            let x = ((pos_f64[i].0 + l*0.5).rem_euclid(l)) / cell;
            let y = ((pos_f64[i].1 + l*0.5).rem_euclid(l)) / cell;
            let z = ((pos_f64[i].2 + l*0.5).rem_euclid(l)) / cell;
            let ix = x.floor() as usize;
            let iy = y.floor() as usize;
            let iz = z.floor() as usize;
            let fx = x - ix as f64; let fy = y - iy as f64; let fz = z - iz as f64;
            let wx = [1.0 - fx, fx]; let wy = [1.0 - fy, fy]; let wz = [1.0 - fz, fz];
            for ai in 0..2 { let ii = (ix + ai) % n_pm;
                for aj in 0..2 { let jj = (iy + aj) % n_pm;
                    for ak in 0..2 { let kk = (iz + ak) % n_pm;
                        grid[ii*n_pm*n_pm + jj*n_pm + kk] += wx[ai]*wy[aj]*wz[ak]*weights[i];
                    } } }
        }
        grid.into_iter().map(|x| Complex64::new(x, 0.0)).collect()
    };
    let fft_3d = |grid: &mut Vec<Complex64>| {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_pm);
        for ix in 0..n_pm { for iy in 0..n_pm {
            let s = ix*n_pm*n_pm + iy*n_pm;
            let mut row: Vec<Complex64> = grid[s..s+n_pm].to_vec();
            fft.process(&mut row);
            grid[s..s+n_pm].copy_from_slice(&row);
        } }
        for ix in 0..n_pm { for iz in 0..n_pm {
            let mut col: Vec<Complex64> = (0..n_pm).map(|iy| grid[ix*n_pm*n_pm+iy*n_pm+iz]).collect();
            fft.process(&mut col);
            for iy in 0..n_pm { grid[ix*n_pm*n_pm+iy*n_pm+iz] = col[iy]; }
        } }
        for iy in 0..n_pm { for iz in 0..n_pm {
            let mut col: Vec<Complex64> = (0..n_pm).map(|ix| grid[ix*n_pm*n_pm+iy*n_pm+iz]).collect();
            fft.process(&mut col);
            for ix in 0..n_pm { grid[ix*n_pm*n_pm+iy*n_pm+iz] = col[ix]; }
        } }
    };

    let f_g_x: Vec<f64> = (0..n).map(|i| acc_gpu[i*3] as f64).collect();
    let f_g_y: Vec<f64> = (0..n).map(|i| acc_gpu[i*3+1] as f64).collect();
    let f_g_z: Vec<f64> = (0..n).map(|i| acc_gpu[i*3+2] as f64).collect();
    let f_p_x: Vec<f64> = (0..n).map(|i| acc_pp[i].0).collect();
    let f_p_y: Vec<f64> = (0..n).map(|i| acc_pp[i].1).collect();
    let f_p_z: Vec<f64> = (0..n).map(|i| acc_pp[i].2).collect();

    let mut g_gx = cic_dep(&f_g_x); fft_3d(&mut g_gx);
    let mut g_gy = cic_dep(&f_g_y); fft_3d(&mut g_gy);
    let mut g_gz = cic_dep(&f_g_z); fft_3d(&mut g_gz);
    let mut g_px = cic_dep(&f_p_x); fft_3d(&mut g_px);
    let mut g_py = cic_dep(&f_p_y); fft_3d(&mut g_py);
    let mut g_pz = cic_dep(&f_p_z); fft_3d(&mut g_pz);

    let n_bins = 16;
    let k_fund = 2.0 * std::f64::consts::PI / l;
    let k_nyq = std::f64::consts::PI * n_pm as f64 / l;
    let dk = k_nyq / n_bins as f64;
    let mut cross = vec![0.0_f64; n_bins];
    let mut pow_g = vec![0.0_f64; n_bins];
    let mut pow_p = vec![0.0_f64; n_bins];
    let mut counts = vec![0_usize; n_bins];

    for ix in 0..n_pm {
        let kxi = if ix <= n_pm/2 { ix as i32 } else { ix as i32 - n_pm as i32 };
        for iy in 0..n_pm {
            let kyi = if iy <= n_pm/2 { iy as i32 } else { iy as i32 - n_pm as i32 };
            for iz in 0..n_pm {
                let kzi = if iz <= n_pm/2 { iz as i32 } else { iz as i32 - n_pm as i32 };
                let k = ((kxi*kxi + kyi*kyi + kzi*kzi) as f64).sqrt() * k_fund;
                if k < 1e-10 || k > k_nyq { continue; }
                let bin = ((k/dk).floor() as usize).min(n_bins-1);
                let id = ix*n_pm*n_pm + iy*n_pm + iz;
                let dot = (g_gx[id]*g_px[id].conj()).re + (g_gy[id]*g_py[id].conj()).re + (g_gz[id]*g_pz[id].conj()).re;
                let pg = g_gx[id].norm_sqr() + g_gy[id].norm_sqr() + g_gz[id].norm_sqr();
                let pp = g_px[id].norm_sqr() + g_py[id].norm_sqr() + g_pz[id].norm_sqr();
                cross[bin] += dot;
                pow_g[bin] += pg;
                pow_p[bin] += pp;
                counts[bin] += 1;
            }
        }
    }

    println!();
    println!("--- r(k) per bin ---");
    println!("  {:<6} {:<10} {:<10} {:<10}", "bin", "k", "r(k)", "|F_g|/|F_p|");
    let mut min_r: f64 = 1.0;
    for b in 0..n_bins {
        if counts[b] == 0 { continue; }
        let r = cross[b] / (pow_g[b] * pow_p[b]).sqrt().max(1e-30);
        let mag = (pow_g[b]/pow_p[b].max(1e-30)).sqrt();
        min_r = min_r.min(r);
        let kc = (b as f64 + 0.5) * dk;
        println!("  {:<6} {:<10.5} {:<10.4} {:<10.4}", b, kc, r, mag);
    }
    println!();
    println!("  min r(k) = {:.4}", min_r);

    if min_r > 0.99 {
        println!("✅ Phase 10A.5a Ewald PASS — min r(k) > 0.99");
        Ok(())
    } else if min_r > 0.95 {
        println!("⚠ Phase 10A.5a Ewald GO_WITH_CAVEAT — min r(k) ∈ (0.95, 0.99]");
        Ok(())
    } else {
        eprintln!("❌ Phase 10A.5a Ewald FAIL — min r(k) = {:.4} < 0.95", min_r);
        std::process::exit(1)
    }
}

/// PP-direct N² avec couplage Janus.
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn pp_direct_janus(
    pos: &[(f64,f64,f64)], mass: &[f64], signs: &[i32],
    box_size: f64, softening: f64, g_phys: f64,
    c_ratio_sq: f64, phi: f64, repulsion_scale: f64,
) -> Vec<(f64,f64,f64)> {
    let n = pos.len();
    let mut acc = vec![(0.0,0.0,0.0); n];
    let eps2 = softening * softening;
    let half_l = box_size * 0.5;
    let cross_minus_plus = c_ratio_sq * (1.0/phi) * repulsion_scale;
    let cross_plus_minus = phi * repulsion_scale;
    for i in 0..n {
        let (xi, yi, zi) = pos[i];
        let si = signs[i];
        let (mut ax, mut ay, mut az) = (0.0_f64, 0.0_f64, 0.0_f64);
        for j in 0..n {
            if i == j { continue; }
            let (xj, yj, zj) = pos[j];
            let sj = signs[j];
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            if dx > half_l { dx -= box_size; }
            if dx < -half_l { dx += box_size; }
            if dy > half_l { dy -= box_size; }
            if dy < -half_l { dy += box_size; }
            if dz > half_l { dz -= box_size; }
            if dz < -half_l { dz += box_size; }
            let r2 = dx*dx + dy*dy + dz*dz + eps2;
            let r = r2.sqrt();
            let inv_r3 = 1.0 / (r * r2);
            // Sign factor (Phase 5 convention)
            let sign_factor = if si == sj {
                1.0 // self-attraction
            } else if si > 0 {
                -cross_plus_minus // m+ feels m- repulsive scaled
            } else {
                -cross_minus_plus // m- feels m+ repulsive scaled
            };
            let factor = sign_factor * g_phys * mass[j] * inv_r3;
            ax += factor * dx;
            ay += factor * dy;
            az += factor * dz;
        }
        acc[i] = (ax, ay, az);
    }
    acc
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires features: cuda cufft");
    std::process::exit(1);
}
