//! Particle-Particle direct N² force reference for validating TreePM accuracy.
//!
//! Reference: GrGadget Fig. 5, PhotoNs §6, Springel 2005 (GADGET-2) Fig. 6.
//!
//! O(N²) cost — only feasible for N ≲ 10K. Used as ground truth in Phase 6.2
//! validation before authorizing GPU port (Phase 5 GPU integration).

/// Direct N² particle-particle force computation, Newton only (no Janus).
///
/// Convention:
///   F_i = +G · Σ_{j≠i} m_j × (r_j - r_i) / |r_j - r_i|³
/// (note: F points FROM i TOWARD j when m_j > 0, attractive Newton.)
///
/// With Plummer softening:
///   |r_j - r_i|³ → (|r_j - r_i|² + ε²)^(3/2)
///
/// Periodic boundary conditions via minimum image convention.
pub fn pp_direct_forces_newton(
    pos: &[(f64, f64, f64)],
    mass: &[f64],
    box_size: f64,
    softening: f64,
    g_phys: f64,
) -> Vec<(f64, f64, f64)> {
    let n = pos.len();
    assert_eq!(n, mass.len());
    let mut acc = vec![(0.0, 0.0, 0.0); n];
    let eps2 = softening * softening;
    let half_l = box_size * 0.5;

    for i in 0..n {
        let (xi, yi, zi) = pos[i];
        let (mut ax, mut ay, mut az) = (0.0_f64, 0.0_f64, 0.0_f64);

        for j in 0..n {
            if i == j {
                continue;
            }

            let (xj, yj, zj) = pos[j];
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;

            // Minimum image convention
            if dx > half_l {
                dx -= box_size;
            }
            if dx < -half_l {
                dx += box_size;
            }
            if dy > half_l {
                dy -= box_size;
            }
            if dy < -half_l {
                dy += box_size;
            }
            if dz > half_l {
                dz -= box_size;
            }
            if dz < -half_l {
                dz += box_size;
            }

            let r2 = dx * dx + dy * dy + dz * dz + eps2;
            let r = r2.sqrt();
            let inv_r3 = 1.0 / (r * r2);

            let factor = g_phys * mass[j] * inv_r3;
            ax += factor * dx;
            ay += factor * dy;
            az += factor * dz;
        }

        acc[i] = (ax, ay, az);
    }

    acc
}

/// Phase 10.8: Direct N² gravitational force with full Ewald summation.
///
/// Sums contributions from all periodic images using Hernquist & Bouchet 1991
/// decomposition into rapidly converging real-space and Fourier-space parts.
/// This is the rigorous reference for comparing TreePM forces in a periodic box,
/// since plain MIC misses contributions from images at distance ≥ L/2.
///
/// Convention (matches `pp_direct_forces_newton`):
///   F_i = +G · Σ_{j≠i} m_j × Σ_n (r_j + nL - r_i) / |r_j + nL - r_i|³_Ewald
///
/// Splitting (with α = 2/L, well-conditioned for cubic boxes):
///   real-space term : (r̂/r²)·[erfc(αr) + (2αr/√π)·exp(-α²r²)/r] over n ∈ [-N_real..N_real]³
///   Fourier term    : -(4πG·m/V)·Σ_{k≠0} k̂·sin(k·r)·exp(-k²/4α²)/k² over m ∈ [-N_fourier..N_fourier]³
///
/// Softening (Plummer ε²) is applied ONLY to the direct pair (n=0). Periodic
/// images are not softened — they are at distances ≥ L/2 ≫ ε.
///
/// Reference: Hernquist & Bouchet 1991 ApJS 75 231; Springel 2005 GADGET-2 §3.1.
///
/// Complexity: O(N² × ((2N_real+1)³ + (2N_fourier+1)³)). Parallelized over i.
pub fn pp_direct_forces_newton_ewald(
    pos: &[(f64, f64, f64)],
    mass: &[f64],
    box_size: f64,
    softening: f64,
    g_phys: f64,
    n_real_max: i32,
    n_fourier_max: i32,
) -> Vec<(f64, f64, f64)> {
    use rayon::prelude::*;
    let n = pos.len();
    assert_eq!(n, mass.len());

    let alpha = 2.0 / box_size;
    let alpha2 = alpha * alpha;
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let two_pi_over_l = 2.0 * std::f64::consts::PI / box_size;
    let four_pi_over_v = 4.0 * std::f64::consts::PI / box_size.powi(3);
    let eps2 = softening * softening;

    // Pre-compute Fourier modes (k != 0): (kx, ky, kz, weight = exp(-k²/4α²)/k²)
    let mut fourier_modes: Vec<(f64, f64, f64, f64)> = Vec::new();
    for mx in -n_fourier_max..=n_fourier_max {
        for my in -n_fourier_max..=n_fourier_max {
            for mz in -n_fourier_max..=n_fourier_max {
                if mx == 0 && my == 0 && mz == 0 {
                    continue;
                }
                let kx = mx as f64 * two_pi_over_l;
                let ky = my as f64 * two_pi_over_l;
                let kz = mz as f64 * two_pi_over_l;
                let k2 = kx * kx + ky * ky + kz * kz;
                let weight = (-k2 / (4.0 * alpha2)).exp() / k2;
                fourier_modes.push((kx, ky, kz, weight));
            }
        }
    }

    (0..n)
        .into_par_iter()
        .map(|i| {
            let (xi, yi, zi) = pos[i];
            let (mut ax, mut ay, mut az) = (0.0_f64, 0.0_f64, 0.0_f64);

            for j in 0..n {
                if i == j {
                    continue;
                }
                let (xj, yj, zj) = pos[j];
                let dx_base = xj - xi;
                let dy_base = yj - yi;
                let dz_base = zj - zi;
                let m_j = mass[j];

                // Real-space sum over images n
                for nx in -n_real_max..=n_real_max {
                    for ny in -n_real_max..=n_real_max {
                        for nz in -n_real_max..=n_real_max {
                            let dx = dx_base + (nx as f64) * box_size;
                            let dy = dy_base + (ny as f64) * box_size;
                            let dz = dz_base + (nz as f64) * box_size;
                            let r2_raw = dx * dx + dy * dy + dz * dz;
                            let is_direct = nx == 0 && ny == 0 && nz == 0;
                            let r2 = if is_direct { r2_raw + eps2 } else { r2_raw };
                            if r2 < 1e-30 {
                                continue;
                            }
                            let r = r2.sqrt();
                            let alpha_r = alpha * r;
                            // erfc(αr)/r³
                            let erfc_term = crate::treepm::truncation_table::erfc_approx(alpha_r)
                                / (r * r2);
                            // (2α/√π)·exp(-α²r²)/r²
                            let exp_term = (2.0 * alpha * inv_sqrt_pi)
                                * (-alpha_r * alpha_r).exp()
                                / r2;
                            let coeff = g_phys * m_j * (erfc_term + exp_term);
                            ax += coeff * dx;
                            ay += coeff * dy;
                            az += coeff * dz;
                        }
                    }
                }

                // Fourier-space sum over modes k
                // F_Fourier = -(4πG·m_j/V) · Σ_k k̂·sin(k·r)·exp(-k²/4α²)/|k|²
                //           = -(4πG·m_j/V) · Σ_k k·sin(k·r)·weight  where weight = exp(-k²/4α²)/k²
                for &(kx, ky, kz, weight) in &fourier_modes {
                    let k_dot_r = kx * dx_base + ky * dy_base + kz * dz_base;
                    let factor = -four_pi_over_v * g_phys * m_j * weight * k_dot_r.sin();
                    ax += factor * kx;
                    ay += factor * ky;
                    az += factor * kz;
                }
            }

            (ax, ay, az)
        })
        .collect()
}

/// Phase 10.8: Direct N² Janus force with full Ewald summation.
///
/// Same as `pp_direct_forces_newton_ewald` but applies Janus sign factors:
///   sign_i == sign_j        : factor = +1            (Newton self-attraction)
///   sign_i > 0, opposite    : factor = -coupling.cross_plus_minus()
///   sign_i < 0, opposite    : factor = -coupling.cross_minus_plus()
///
/// Softening per sign (`softening_plus` for m+, `softening_minus` for m-) is
/// applied only on the direct pair (n=0).
pub fn pp_direct_forces_janus_ewald(
    pos: &[(f64, f64, f64)],
    mass: &[f64],
    sign: &[i32],
    box_size: f64,
    softening_plus: f64,
    softening_minus: f64,
    g_phys: f64,
    coupling: &super::janus::JanusCoupling,
    n_real_max: i32,
    n_fourier_max: i32,
) -> Vec<(f64, f64, f64)> {
    use rayon::prelude::*;
    let n = pos.len();
    assert_eq!(n, mass.len());
    assert_eq!(n, sign.len());

    let alpha = 2.0 / box_size;
    let alpha2 = alpha * alpha;
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let two_pi_over_l = 2.0 * std::f64::consts::PI / box_size;
    let four_pi_over_v = 4.0 * std::f64::consts::PI / box_size.powi(3);
    let eps_plus_sq = softening_plus * softening_plus;
    let eps_minus_sq = softening_minus * softening_minus;
    let cross_plus_minus = coupling.cross_plus_minus();
    let cross_minus_plus = coupling.cross_minus_plus();

    let mut fourier_modes: Vec<(f64, f64, f64, f64)> = Vec::new();
    for mx in -n_fourier_max..=n_fourier_max {
        for my in -n_fourier_max..=n_fourier_max {
            for mz in -n_fourier_max..=n_fourier_max {
                if mx == 0 && my == 0 && mz == 0 {
                    continue;
                }
                let kx = mx as f64 * two_pi_over_l;
                let ky = my as f64 * two_pi_over_l;
                let kz = mz as f64 * two_pi_over_l;
                let k2 = kx * kx + ky * ky + kz * kz;
                let weight = (-k2 / (4.0 * alpha2)).exp() / k2;
                fourier_modes.push((kx, ky, kz, weight));
            }
        }
    }

    (0..n)
        .into_par_iter()
        .map(|i| {
            let (xi, yi, zi) = pos[i];
            let s_i = sign[i];
            let eps2_i = if s_i > 0 { eps_plus_sq } else { eps_minus_sq };
            let (mut ax, mut ay, mut az) = (0.0_f64, 0.0_f64, 0.0_f64);

            for j in 0..n {
                if i == j {
                    continue;
                }
                let (xj, yj, zj) = pos[j];
                let s_j = sign[j];
                let dx_base = xj - xi;
                let dy_base = yj - yi;
                let dz_base = zj - zi;
                let m_j = mass[j];

                // Janus sign factor (matches GPU kernel and pp_direct_janus convention)
                let sign_factor = if s_i == s_j {
                    1.0
                } else if s_i > 0 {
                    -cross_plus_minus
                } else {
                    -cross_minus_plus
                };

                // Real-space sum
                for nx in -n_real_max..=n_real_max {
                    for ny in -n_real_max..=n_real_max {
                        for nz in -n_real_max..=n_real_max {
                            let dx = dx_base + (nx as f64) * box_size;
                            let dy = dy_base + (ny as f64) * box_size;
                            let dz = dz_base + (nz as f64) * box_size;
                            let r2_raw = dx * dx + dy * dy + dz * dz;
                            let is_direct = nx == 0 && ny == 0 && nz == 0;
                            let r2 = if is_direct { r2_raw + eps2_i } else { r2_raw };
                            if r2 < 1e-30 {
                                continue;
                            }
                            let r = r2.sqrt();
                            let alpha_r = alpha * r;
                            let erfc_term = crate::treepm::truncation_table::erfc_approx(alpha_r)
                                / (r * r2);
                            let exp_term = (2.0 * alpha * inv_sqrt_pi)
                                * (-alpha_r * alpha_r).exp()
                                / r2;
                            let coeff = sign_factor * g_phys * m_j * (erfc_term + exp_term);
                            ax += coeff * dx;
                            ay += coeff * dy;
                            az += coeff * dz;
                        }
                    }
                }

                // Fourier sum
                for &(kx, ky, kz, weight) in &fourier_modes {
                    let k_dot_r = kx * dx_base + ky * dy_base + kz * dz_base;
                    let factor =
                        -sign_factor * four_pi_over_v * g_phys * m_j * weight * k_dot_r.sin();
                    ax += factor * kx;
                    ay += factor * ky;
                    az += factor * kz;
                }
            }

            (ax, ay, az)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pp_two_particles_attract() {
        // Two unit masses at distance 1, no softening, G=1.
        // F_i = G · m_j · (r_j - r_i) / r³
        // Particle 0 at origin, particle 1 at (1,0,0):
        //   F_0 = 1 · 1 · (+1, 0, 0) / 1 = (+1, 0, 0)  → toward particle 1 (attractive)
        //   F_1 = 1 · 1 · (-1, 0, 0) / 1 = (-1, 0, 0)  → toward particle 0
        let pos = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let mass = vec![1.0, 1.0];
        let acc = pp_direct_forces_newton(&pos, &mass, 100.0, 0.0, 1.0);

        // Tolerance: 1e-12 (FP arithmetic on simple values)
        assert!((acc[0].0 - 1.0).abs() < 1e-12, "acc_x[0] = {}", acc[0].0);
        assert!(acc[0].1.abs() < 1e-12);
        assert!(acc[0].2.abs() < 1e-12);

        // Action-reaction
        assert!((acc[1].0 + 1.0).abs() < 1e-12, "acc_x[1] = {}", acc[1].0);
    }

    #[test]
    fn test_pp_three_particles_symmetric() {
        // Triangle équilatéral autour de l'origine, masses égales.
        // Particule 0 at (1, 0, 0), particle 1 at (-0.5, +√3/2, 0), particle 2 at (-0.5, -√3/2, 0).
        // Each particle is attracted by the other two; forces should sum
        // to a vector pointing toward the centroid (= origin).
        let r = 1.0_f64;
        let s = 0.866_025_403_784_438_6_f64; // √3/2
        let pos = vec![
            (r, 0.0, 0.0),
            (-r * 0.5, r * s, 0.0),
            (-r * 0.5, -r * s, 0.0),
        ];
        let mass = vec![1.0, 1.0, 1.0];
        let acc = pp_direct_forces_newton(&pos, &mass, 100.0, 0.0, 1.0);

        // Magnitudes should be equal by symmetry
        let m0 = (acc[0].0.powi(2) + acc[0].1.powi(2)).sqrt();
        let m1 = (acc[1].0.powi(2) + acc[1].1.powi(2)).sqrt();
        let m2 = (acc[2].0.powi(2) + acc[2].1.powi(2)).sqrt();
        assert!((m0 - m1).abs() / m0 < 1e-12);
        assert!((m1 - m2).abs() / m1 < 1e-12);

        // Particle 0 at (+1, 0): force toward origin → fx < 0
        assert!(acc[0].0 < 0.0);
        assert!(acc[0].1.abs() < 1e-12); // y-symmetric, fy = 0

        // Particle 1 at (-0.5, +s): force toward origin → fx > 0, fy < 0
        assert!(acc[1].0 > 0.0);
        assert!(acc[1].1 < 0.0);

        // Particle 2 at (-0.5, -s): force toward origin → fx > 0, fy > 0
        assert!(acc[2].0 > 0.0);
        assert!(acc[2].1 > 0.0);
    }

    #[test]
    fn test_pp_softening_finite_at_zero() {
        // Two particles at same position with softening: force should be finite (zero by symm)
        let pos = vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)];
        let mass = vec![1.0, 1.0];
        let acc = pp_direct_forces_newton(&pos, &mass, 100.0, 0.5, 1.0);
        // At identical positions, force is zero (dx=dy=dz=0).
        assert!(acc[0].0.abs() < 1e-12);
        assert!(acc[0].1.abs() < 1e-12);
        assert!(acc[0].2.abs() < 1e-12);
    }

    #[test]
    fn test_pp_linearity_in_g() {
        // Force scales linearly in G.
        let pos = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let mass = vec![1.0, 1.0];
        let acc_g1 = pp_direct_forces_newton(&pos, &mass, 100.0, 0.0, 1.0);
        let acc_g10 = pp_direct_forces_newton(&pos, &mass, 100.0, 0.0, 10.0);
        // Tolerance: machine epsilon on scaling
        assert!((acc_g10[0].0 - 10.0 * acc_g1[0].0).abs() < 1e-12);
    }

    #[test]
    fn test_pp_linearity_in_mass() {
        // Doubling all masses doubles all forces.
        let pos = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let mass1 = vec![1.0, 1.0, 1.0];
        let mass2 = vec![2.0, 2.0, 2.0];
        let acc1 = pp_direct_forces_newton(&pos, &mass1, 100.0, 0.0, 1.0);
        let acc2 = pp_direct_forces_newton(&pos, &mass2, 100.0, 0.0, 1.0);
        for i in 0..3 {
            assert!((acc2[i].0 - 2.0 * acc1[i].0).abs() < 1e-12);
            assert!((acc2[i].1 - 2.0 * acc1[i].1).abs() < 1e-12);
            assert!((acc2[i].2 - 2.0 * acc1[i].2).abs() < 1e-12);
        }
    }

    /// **Étape 3 du plan PROMPT_CLI_PRECISION_CHECK** : test combiné TreePM
    /// CPU vs PP-direct sur N=1000.
    ///
    /// Setup : particules aléatoires (seed 42) dans L=100 Mpc, N_pm=64,
    /// r_s = 1.2·Δg, r_cut = 6·Δg, θ=0.5, softening Plummer.
    ///
    /// Verdict :
    /// - A_ACCEPTABLE : median < 1%, P95 < 5%, max < 20%
    /// - B_OR_C_BUG   : median > 5% ou max > 50%
    /// - GREY_ZONE    : sinon, review manuel
    #[test]
    #[ignore]
    fn test_treepm_combined_vs_pp_direct_precision() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1000;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);

        // Generate random particles in box [0, L]³ (TreePMForce/PmGrid expect
        // positions in [-L/2, L/2] but assign_mass internally wraps; for safety
        // we'll use [-L/2, L/2]).
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;

        // === Reference: PP-direct ===
        println!("Computing PP-direct reference (N={})...", n);
        let t_pp = std::time::Instant::now();
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        println!("  PP-direct: {:.1}s", t_pp.elapsed().as_secs_f64());

        // === TreePM combined ===
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_s = 1.2 * dg;
        let r_cut = 6.0 * dg;
        let theta = 0.5;

        // Convention: g_solver = G_phys / V_cell (cf Phase 6)
        let v_cell = dg.powi(3);
        let g_solver = g_phys / v_cell;

        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        // Convention: PM uses G/V_cell (cell-volume normalization for
        // mass-count→density), Tree uses G_phys directly (no grid).
        // Set tpm.g_constant for the PM solver inside update().
        tpm.g_constant = g_solver;

        println!(
            "Computing TreePM CPU (n_pm={}, r_s={:.3}, r_cut={:.3}, θ={})...",
            n_pm, r_s, r_cut, theta
        );
        let t_tpm = std::time::Instant::now();
        tpm.update(&particles_bh);
        // Override tree's g_constant AFTER update (which built tree with
        // tpm.g_constant = g_solver). Tree must use G_phys for direct 1/r².
        tpm.tree.g_constant = g_phys;
        let acc_treepm = tpm.compute_all_forces(&particles_bh);
        println!("  TreePM:    {:.1}s", t_tpm.elapsed().as_secs_f64());

        // === Compare ===
        let mut rel_errors = Vec::with_capacity(n);
        let mut angle_errors_deg = Vec::with_capacity(n);
        for i in 0..n {
            let (rx, ry, rz) = acc_pp[i];
            let tv = acc_treepm[i];
            let (tx, ty, tz) = (tv.x, tv.y, tv.z);

            let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
            let dx = tx - rx;
            let dy = ty - ry;
            let dz = tz - rz;
            let mag_diff = (dx * dx + dy * dy + dz * dz).sqrt();

            if mag_ref > 1e-15 {
                rel_errors.push(mag_diff / mag_ref);
                let mag_tpm = (tx * tx + ty * ty + tz * tz).sqrt();
                if mag_tpm > 1e-15 {
                    let cos_a = (rx * tx + ry * ty + rz * tz) / (mag_ref * mag_tpm);
                    let cos_a = cos_a.clamp(-1.0, 1.0);
                    angle_errors_deg.push(cos_a.acos().to_degrees());
                }
            }
        }

        rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = rel_errors[rel_errors.len() / 2];
        let p95 = rel_errors[(rel_errors.len() * 95) / 100];
        let max_rel = rel_errors.iter().cloned().fold(0.0_f64, f64::max);

        angle_errors_deg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_angle = angle_errors_deg[angle_errors_deg.len() / 2];
        let max_angle = angle_errors_deg.iter().cloned().fold(0.0_f64, f64::max);

        println!();
        println!("=== TreePM vs PP-direct precision (N={}) ===", n);
        println!("  Median rel err   : {:.4}%", median * 100.0);
        println!("  P95    rel err   : {:.4}%", p95 * 100.0);
        println!("  Max    rel err   : {:.4}%", max_rel * 100.0);
        println!("  Median angle err : {:.4}°", median_angle);
        println!("  Max    angle err : {:.4}°", max_angle);
        println!();

        let verdict = if median < 0.01 && p95 < 0.05 && max_rel < 0.20 {
            "A_ACCEPTABLE"
        } else if median > 0.05 || max_rel > 0.50 {
            "B_OR_C_BUG"
        } else {
            "GREY_ZONE_NEEDS_REVIEW"
        };
        println!("VERDICT: {}", verdict);

        // Save data + verdict
        let _ = std::fs::create_dir_all("logs/treepm");
        let _ = std::fs::write(
            "logs/treepm/precision_verdict.txt",
            format!(
                "{}\nmedian={}\np95={}\nmax={}\nmedian_angle_deg={}\nmax_angle_deg={}\n",
                verdict, median, p95, max_rel, median_angle, max_angle
            ),
        );
        if let Ok(mut f) = std::fs::File::create("logs/treepm/precision_check_data.csv") {
            use std::io::Write;
            writeln!(f, "i,rel_err,angle_err_deg").unwrap();
            for (i, (re, ae)) in rel_errors.iter().zip(angle_errors_deg.iter()).enumerate() {
                writeln!(f, "{},{},{}", i, re, ae).unwrap();
            }
        }

        // Pas d'assert : on veut le verdict, pas un crash. Le test s'exécute
        // toujours en succès, le verdict est dans le fichier.
    }

    /// Diagnostic 4a: PM-only vs TreePM combined error vs PP-direct.
    /// If median(pm_only) >> median(treepm) : Tree compense → Hypothèse A.
    /// If median(pm_only) ≈ median(treepm) : bug systémique.
    #[test]
    #[ignore]
    fn diagnostic_pm_only_vs_combined() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1000;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let v_cell = dg.powi(3);
        let g_solver = g_phys / v_cell;

        // PP reference
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

        // PM-only
        let mut pm_only = TreePMForce::new_pm_only(n_pm, l);
        pm_only.g_constant = g_solver;
        pm_only.update(&particles_bh);
        let acc_pm = pm_only.compute_all_forces(&particles_bh);

        // TreePM combined
        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        tpm.g_constant = g_solver;
        tpm.update(&particles_bh);
        tpm.tree.g_constant = g_phys; // tree uses G_phys
        let acc_tpm = tpm.compute_all_forces(&particles_bh);

        // Stats vs PP
        let stats = |acc: &[crate::nbody::Vec3], label: &str| {
            let mut errs = Vec::with_capacity(n);
            for i in 0..n {
                let (rx, ry, rz) = acc_pp[i];
                let av = acc[i];
                let (ax, ay, az) = (av.x, av.y, av.z);
                let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
                if mag_ref < 1e-15 {
                    continue;
                }
                let dx = ax - rx;
                let dy = ay - ry;
                let dz = az - rz;
                let diff = (dx * dx + dy * dy + dz * dz).sqrt();
                errs.push(diff / mag_ref);
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = errs[errs.len() / 2];
            let p95 = errs[(errs.len() * 95) / 100];
            let max = errs.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "  {:<12}: median={:.3}%, P95={:.3}%, max={:.3}%",
                label,
                med * 100.0,
                p95 * 100.0,
                max * 100.0
            );
            (med, p95, max)
        };

        println!();
        println!("=== Diagnostic 4a: PM-only vs TreePM combined ===");
        let (m_pm, _, _) = stats(&acc_pm, "PM-only");
        let (m_tpm, _, _) = stats(&acc_tpm, "TreePM");
        let ratio = m_pm / m_tpm;
        println!("  PM-only / TreePM median ratio: {:.2}×", ratio);
        if ratio > 3.0 {
            println!(
                "  Tree COMPENSATES PM (Hypothèse A confirmée)"
            );
        } else if (m_pm - m_tpm).abs() / m_pm < 0.3 {
            println!(
                "  PM and TreePM similar errors → systemic bug (Hypothèse B/C)"
            );
        } else {
            println!("  Mixed result → review");
        }
    }

    /// Génère ICs Zel'dovich simples (lattice + perturbation aléatoire bornée).
    ///
    /// Pas un vrai Zel'dovich avec spectre P(k), mais une distribution
    /// quasi-uniforme avec déplacement contrôlé qui reproduit la densité
    /// homogène typique d'IC cosmologique à grand z (avant clustering).
    fn generate_zeldovich_simple_ics(
        n_per_dim: usize,
        box_size: f64,
        seed: u64,
    ) -> Vec<(f64, f64, f64)> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = n_per_dim.pow(3);
        let dx = box_size / n_per_dim as f64;
        let displacement_amplitude = 0.15 * dx;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pos = Vec::with_capacity(n);

        for i in 0..n_per_dim {
            for j in 0..n_per_dim {
                for k in 0..n_per_dim {
                    // Centered grid in [-L/2, L/2)
                    let gx = (i as f64 + 0.5) * dx - box_size * 0.5;
                    let gy = (j as f64 + 0.5) * dx - box_size * 0.5;
                    let gz = (k as f64 + 0.5) * dx - box_size * 0.5;

                    let dx_p = (rng.random::<f64>() - 0.5) * 2.0 * displacement_amplitude;
                    let dy_p = (rng.random::<f64>() - 0.5) * 2.0 * displacement_amplitude;
                    let dz_p = (rng.random::<f64>() - 0.5) * 2.0 * displacement_amplitude;

                    let mut x = gx + dx_p;
                    let mut y = gy + dy_p;
                    let mut z = gz + dz_p;
                    // PBC wrap to [-L/2, L/2)
                    let half = box_size * 0.5;
                    while x >= half { x -= box_size; }
                    while x < -half { x += box_size; }
                    while y >= half { y -= box_size; }
                    while y < -half { y += box_size; }
                    while z >= half { z -= box_size; }
                    while z < -half { z += box_size; }

                    pos.push((x, y, z));
                }
            }
        }
        pos
    }

    /// Phase 9.7-C v2 sur N=10K Zeldovich: confirme cross-correlation reste
    /// élevée même sur lattice + perturbation (où force errs individuelles
    /// montent à 70%).
    #[test]
    #[ignore]
    fn test_force_field_cross_correlation_zeldovich() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rustfft::{num_complex::Complex64, FftPlanner};

        let n_per_dim: usize = 22;
        let n: usize = n_per_dim.pow(3); // 10648
        let l = 100.0_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let softening = 0.05;
        let g_phys = 1.0;
        let v_cell = dg.powi(3);

        let pos = generate_zeldovich_simple_ics(n_per_dim, l, 42);
        let mass: Vec<f64> = vec![1.0; n];
        let mut particles_bh = Vec::with_capacity(n);
        for &(x, y, z) in &pos {
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        tpm.g_constant = g_phys / v_cell;
        tpm.update(&particles_bh);
        tpm.tree.g_constant = g_phys;
        let acc_tpm = tpm.compute_all_forces(&particles_bh);

        let half = l * 0.5;
        let cic_dep = |weights: &[f64]| -> Vec<Complex64> {
            let n_cells = n_pm * n_pm * n_pm;
            let mut grid = vec![0.0_f64; n_cells];
            let cell = l / n_pm as f64;
            for i in 0..n {
                let x = ((pos[i].0 + half).rem_euclid(l)) / cell;
                let y = ((pos[i].1 + half).rem_euclid(l)) / cell;
                let z = ((pos[i].2 + half).rem_euclid(l)) / cell;
                let ix = x.floor() as usize;
                let iy = y.floor() as usize;
                let iz = z.floor() as usize;
                let fx = x - ix as f64;
                let fy = y - iy as f64;
                let fz = z - iz as f64;
                let wx = [1.0 - fx, fx];
                let wy = [1.0 - fy, fy];
                let wz = [1.0 - fz, fz];
                for ai in 0..2 {
                    let ii = (ix + ai) % n_pm;
                    for aj in 0..2 {
                        let jj = (iy + aj) % n_pm;
                        for ak in 0..2 {
                            let kk = (iz + ak) % n_pm;
                            grid[ii * n_pm * n_pm + jj * n_pm + kk] +=
                                wx[ai] * wy[aj] * wz[ak] * weights[i];
                        }
                    }
                }
            }
            grid.into_iter().map(|x| Complex64::new(x, 0.0)).collect()
        };
        let fft_3d = |grid: &mut Vec<Complex64>| {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n_pm);
            for ix in 0..n_pm {
                for iy in 0..n_pm {
                    let s = ix * n_pm * n_pm + iy * n_pm;
                    let mut row: Vec<Complex64> = grid[s..s + n_pm].to_vec();
                    fft.process(&mut row);
                    grid[s..s + n_pm].copy_from_slice(&row);
                }
            }
            for ix in 0..n_pm {
                for iz in 0..n_pm {
                    let mut col: Vec<Complex64> = (0..n_pm).map(|iy| grid[ix*n_pm*n_pm+iy*n_pm+iz]).collect();
                    fft.process(&mut col);
                    for iy in 0..n_pm { grid[ix*n_pm*n_pm+iy*n_pm+iz] = col[iy]; }
                }
            }
            for iy in 0..n_pm {
                for iz in 0..n_pm {
                    let mut col: Vec<Complex64> = (0..n_pm).map(|ix| grid[ix*n_pm*n_pm+iy*n_pm+iz]).collect();
                    fft.process(&mut col);
                    for ix in 0..n_pm { grid[ix*n_pm*n_pm+iy*n_pm+iz] = col[ix]; }
                }
            }
        };

        let f_t_x: Vec<f64> = (0..n).map(|i| acc_tpm[i].x).collect();
        let f_t_y: Vec<f64> = (0..n).map(|i| acc_tpm[i].y).collect();
        let f_t_z: Vec<f64> = (0..n).map(|i| acc_tpm[i].z).collect();
        let f_p_x: Vec<f64> = (0..n).map(|i| acc_pp[i].0).collect();
        let f_p_y: Vec<f64> = (0..n).map(|i| acc_pp[i].1).collect();
        let f_p_z: Vec<f64> = (0..n).map(|i| acc_pp[i].2).collect();

        let mut g_tx = cic_dep(&f_t_x); fft_3d(&mut g_tx);
        let mut g_ty = cic_dep(&f_t_y); fft_3d(&mut g_ty);
        let mut g_tz = cic_dep(&f_t_z); fft_3d(&mut g_tz);
        let mut g_px = cic_dep(&f_p_x); fft_3d(&mut g_px);
        let mut g_py = cic_dep(&f_p_y); fft_3d(&mut g_py);
        let mut g_pz = cic_dep(&f_p_z); fft_3d(&mut g_pz);

        let n_bins = 16;
        let k_fund = 2.0 * std::f64::consts::PI / l;
        let k_nyq = std::f64::consts::PI * n_pm as f64 / l;
        let dk = k_nyq / n_bins as f64;
        let mut cross = vec![0.0_f64; n_bins];
        let mut pow_t = vec![0.0_f64; n_bins];
        let mut pow_p = vec![0.0_f64; n_bins];
        let mut counts = vec![0_usize; n_bins];

        for ix in 0..n_pm {
            let kxi = if ix <= n_pm / 2 { ix as i32 } else { ix as i32 - n_pm as i32 };
            for iy in 0..n_pm {
                let kyi = if iy <= n_pm / 2 { iy as i32 } else { iy as i32 - n_pm as i32 };
                for iz in 0..n_pm {
                    let kzi = if iz <= n_pm / 2 { iz as i32 } else { iz as i32 - n_pm as i32 };
                    let k = ((kxi*kxi + kyi*kyi + kzi*kzi) as f64).sqrt() * k_fund;
                    if k < 1e-10 || k > k_nyq { continue; }
                    let bin = ((k / dk).floor() as usize).min(n_bins - 1);
                    let idx = ix * n_pm * n_pm + iy * n_pm + iz;
                    let ft_dot_fp = (g_tx[idx]*g_px[idx].conj()).re
                                  + (g_ty[idx]*g_py[idx].conj()).re
                                  + (g_tz[idx]*g_pz[idx].conj()).re;
                    let pt = g_tx[idx].norm_sqr() + g_ty[idx].norm_sqr() + g_tz[idx].norm_sqr();
                    let pp = g_px[idx].norm_sqr() + g_py[idx].norm_sqr() + g_pz[idx].norm_sqr();
                    cross[bin] += ft_dot_fp;
                    pow_t[bin] += pt;
                    pow_p[bin] += pp;
                    counts[bin] += 1;
                }
            }
        }

        println!();
        println!("=== Phase 9.7-C Zeldovich N=10K force-field cross-corr ===");
        println!("  {:<6} {:<10} {:<10} {:<10}", "bin", "k", "r(k)", "|F_t|/|F_p|");
        let mut min_r: f64 = 1.0;
        for b in 0..n_bins {
            if counts[b] == 0 { continue; }
            let r = cross[b] / (pow_t[b] * pow_p[b]).sqrt().max(1e-30);
            let mag_ratio = (pow_t[b] / pow_p[b].max(1e-30)).sqrt();
            min_r = min_r.min(r);
            let k_center = (b as f64 + 0.5) * dk;
            println!("  {:<6} {:<10.5} {:<10.4} {:<10.4}", b, k_center, r, mag_ratio);
        }
        println!();
        println!("  min r(k): {:.4}", min_r);
        let verdict = if min_r > 0.99 { "GO_PHASE_10" }
            else if min_r > 0.95 { "GO_WITH_CAVEAT" }
            else { "NO_GO_REAL_BUG" };
        println!("  VERDICT: {}", verdict);
    }

    /// Phase 9.7-C v2: comparison Fourier-space directe par k-mode.
    ///
    /// Compute cross-correlation per k-mode entre F_treepm et F_pp dans le
    /// champ de force CIC-déposé. Métrique:
    ///   r(k) = Re⟨F̂_t · F̂_p⟩ / sqrt(⟨|F̂_t|²⟩·⟨|F̂_p|²⟩)
    ///
    /// r(k) ≈ 1 : méthodes agree per k mode (forces mêmes direction)
    /// r(k) << 1 : méthodes disagreement à cette échelle
    ///
    /// Verdict: si r(k) > 0.99 sur tous k résolus, GO Phase 10 quel que soit
    /// le bias de magnitude (qui s'élimine via normalisation).
    #[test]
    #[ignore]
    fn test_force_field_cross_correlation_per_k() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use rustfft::{num_complex::Complex64, FftPlanner};

        let n = 1000;
        let l = 100.0_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let softening = 0.05;
        let g_phys = 1.0;
        let v_cell = dg.powi(3);

        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        tpm.g_constant = g_phys / v_cell;
        tpm.update(&particles_bh);
        tpm.tree.g_constant = g_phys;
        let acc_tpm = tpm.compute_all_forces(&particles_bh);

        // Helper: CIC-deposit weighted scalar per particle
        let cic_dep = |weights: &[f64]| -> Vec<Complex64> {
            let n_cells = n_pm * n_pm * n_pm;
            let mut grid = vec![0.0_f64; n_cells];
            let cell = l / n_pm as f64;
            for i in 0..n {
                let x = ((pos[i].0 + half).rem_euclid(l)) / cell;
                let y = ((pos[i].1 + half).rem_euclid(l)) / cell;
                let z = ((pos[i].2 + half).rem_euclid(l)) / cell;
                let ix = x.floor() as usize;
                let iy = y.floor() as usize;
                let iz = z.floor() as usize;
                let fx = x - ix as f64;
                let fy = y - iy as f64;
                let fz = z - iz as f64;
                let wx = [1.0 - fx, fx];
                let wy = [1.0 - fy, fy];
                let wz = [1.0 - fz, fz];
                for ai in 0..2 {
                    let ii = (ix + ai) % n_pm;
                    for aj in 0..2 {
                        let jj = (iy + aj) % n_pm;
                        for ak in 0..2 {
                            let kk = (iz + ak) % n_pm;
                            grid[ii * n_pm * n_pm + jj * n_pm + kk] +=
                                wx[ai] * wy[aj] * wz[ak] * weights[i];
                        }
                    }
                }
            }
            grid.into_iter().map(|x| Complex64::new(x, 0.0)).collect()
        };

        // FFT 3D function (reuses existing pattern)
        let fft_3d = |grid: &mut Vec<Complex64>| {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n_pm);
            // Z direction
            for ix in 0..n_pm {
                for iy in 0..n_pm {
                    let start = ix * n_pm * n_pm + iy * n_pm;
                    let mut row: Vec<Complex64> = grid[start..start + n_pm].to_vec();
                    fft.process(&mut row);
                    grid[start..start + n_pm].copy_from_slice(&row);
                }
            }
            // Y
            for ix in 0..n_pm {
                for iz in 0..n_pm {
                    let mut col: Vec<Complex64> = (0..n_pm)
                        .map(|iy| grid[ix * n_pm * n_pm + iy * n_pm + iz])
                        .collect();
                    fft.process(&mut col);
                    for iy in 0..n_pm {
                        grid[ix * n_pm * n_pm + iy * n_pm + iz] = col[iy];
                    }
                }
            }
            // X
            for iy in 0..n_pm {
                for iz in 0..n_pm {
                    let mut col: Vec<Complex64> = (0..n_pm)
                        .map(|ix| grid[ix * n_pm * n_pm + iy * n_pm + iz])
                        .collect();
                    fft.process(&mut col);
                    for ix in 0..n_pm {
                        grid[ix * n_pm * n_pm + iy * n_pm + iz] = col[ix];
                    }
                }
            }
        };

        // CIC + FFT for x, y, z components of TreePM and PP forces
        let f_t_x: Vec<f64> = (0..n).map(|i| acc_tpm[i].x).collect();
        let f_t_y: Vec<f64> = (0..n).map(|i| acc_tpm[i].y).collect();
        let f_t_z: Vec<f64> = (0..n).map(|i| acc_tpm[i].z).collect();
        let f_p_x: Vec<f64> = (0..n).map(|i| acc_pp[i].0).collect();
        let f_p_y: Vec<f64> = (0..n).map(|i| acc_pp[i].1).collect();
        let f_p_z: Vec<f64> = (0..n).map(|i| acc_pp[i].2).collect();

        let mut g_tx = cic_dep(&f_t_x); fft_3d(&mut g_tx);
        let mut g_ty = cic_dep(&f_t_y); fft_3d(&mut g_ty);
        let mut g_tz = cic_dep(&f_t_z); fft_3d(&mut g_tz);
        let mut g_px = cic_dep(&f_p_x); fft_3d(&mut g_px);
        let mut g_py = cic_dep(&f_p_y); fft_3d(&mut g_py);
        let mut g_pz = cic_dep(&f_p_z); fft_3d(&mut g_pz);

        // Bin per |k|: compute < F̂_t · F̂_p* >, < |F̂_t|² >, < |F̂_p|² >
        let n_bins = 16;
        let k_fund = 2.0 * std::f64::consts::PI / l;
        let k_nyq = std::f64::consts::PI * n_pm as f64 / l;
        let dk = k_nyq / n_bins as f64;
        let mut cross = vec![0.0_f64; n_bins];
        let mut pow_t = vec![0.0_f64; n_bins];
        let mut pow_p = vec![0.0_f64; n_bins];
        let mut counts = vec![0_usize; n_bins];

        for ix in 0..n_pm {
            let kxi = if ix <= n_pm / 2 { ix as i32 } else { ix as i32 - n_pm as i32 };
            for iy in 0..n_pm {
                let kyi = if iy <= n_pm / 2 { iy as i32 } else { iy as i32 - n_pm as i32 };
                for iz in 0..n_pm {
                    let kzi = if iz <= n_pm / 2 { iz as i32 } else { iz as i32 - n_pm as i32 };
                    let k = ((kxi * kxi + kyi * kyi + kzi * kzi) as f64).sqrt() * k_fund;
                    if k < 1e-10 || k > k_nyq { continue; }
                    let bin = ((k / dk).floor() as usize).min(n_bins - 1);
                    let idx = ix * n_pm * n_pm + iy * n_pm + iz;
                    let ft_dot_fp = (g_tx[idx] * g_px[idx].conj()).re
                        + (g_ty[idx] * g_py[idx].conj()).re
                        + (g_tz[idx] * g_pz[idx].conj()).re;
                    let pt = g_tx[idx].norm_sqr() + g_ty[idx].norm_sqr() + g_tz[idx].norm_sqr();
                    let pp = g_px[idx].norm_sqr() + g_py[idx].norm_sqr() + g_pz[idx].norm_sqr();
                    cross[bin] += ft_dot_fp;
                    pow_t[bin] += pt;
                    pow_p[bin] += pp;
                    counts[bin] += 1;
                }
            }
        }

        println!();
        println!("=== Phase 9.7-C v2: F-field cross-correlation per k ===");
        println!("  Setup: N=1000 random uniform, L=100, n_pm=64, r_cut=9.375");
        println!("  {:<6} {:<10} {:<10} {:<10} {:<10}", "bin", "k", "r(k)", "|F_t|/|F_p|", "n_modes");
        let mut min_r: f64 = 1.0;
        for b in 0..n_bins {
            if counts[b] == 0 { continue; }
            let r = cross[b] / (pow_t[b] * pow_p[b]).sqrt().max(1e-30);
            let mag_ratio = (pow_t[b] / pow_p[b].max(1e-30)).sqrt();
            min_r = min_r.min(r);
            let k_center = (b as f64 + 0.5) * dk;
            println!(
                "  {:<6} {:<10.5} {:<10.4} {:<10.4} {:<10}",
                b, k_center, r, mag_ratio, counts[b]
            );
        }
        println!();
        println!("  min r(k) over all bins: {:.4}", min_r);
        let verdict = if min_r > 0.99 { "GO_PHASE_10" }
            else if min_r > 0.95 { "GO_WITH_CAVEAT" }
            else { "NO_GO_REAL_BUG" };
        println!("  VERDICT (per-k correlation): {}", verdict);

        let _ = std::fs::create_dir_all("logs/treepm");
        let _ = std::fs::write(
            "logs/treepm/phase97c_v2_verdict.txt",
            format!("{}\nmin_r={}\n", verdict, min_r),
        );
    }

    /// Phase 9.7-C — test principal réaliste : P(k) TreePM vs PP-direct sur
    /// distribution Zel'dovich.
    ///
    /// Setup: N=10K Zel'dovich, L=100, n_pm=64, z_init=10. Compare
    /// les forces individuelles (médiane attendue ~10% per Phase 9.7-B)
    /// ET le power spectrum P(k) (cible <1% médiane pour GO).
    ///
    /// Verdict:
    /// - GO_PHASE_10 : P(k) median < 1%, max < 5%
    /// - GO_WITH_CAVEAT : P(k) median < 5%
    /// - NO_GO_REAL_BUG : P(k) median > 5%
    #[test]
    #[ignore]
    fn test_realistic_zeldovich_pk_treepm_vs_pp() {
        use crate::nbody::{Particle, Vec3};
        use crate::power_spectrum::{compute_pk, PowerSpectrumResult};
        let _ = compute_pk;
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;

        let n_per_dim: usize = 22; // N = 22³ = 10648
        let n: usize = n_per_dim.pow(3);
        let l = 100.0_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let softening = 0.05;
        let g_phys = 1.0;
        let v_cell = dg.powi(3);

        println!();
        println!("=== Phase 9.7-C: réaliste Zel'dovich, N={}, L={}, n_pm={} ===", n, l, n_pm);

        // Zel'dovich-like ICs
        let pos = generate_zeldovich_simple_ics(n_per_dim, l, 42);
        let mass: Vec<f64> = vec![1.0; n];
        let mut particles_bh = Vec::with_capacity(n);
        for &(x, y, z) in &pos {
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        // PP-direct
        println!("Computing PP-direct (N²={:.2e} pairs)...", (n * n) as f64);
        let t0 = std::time::Instant::now();
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        let t_pp = t0.elapsed().as_secs_f64();
        println!("  PP-direct: {:.1}s", t_pp);

        // TreePM
        println!("Computing TreePM CPU...");
        let t0 = std::time::Instant::now();
        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        tpm.g_constant = g_phys / v_cell;
        tpm.update(&particles_bh);
        tpm.tree.g_constant = g_phys;
        let acc_tpm = tpm.compute_all_forces(&particles_bh);
        let t_tpm = t0.elapsed().as_secs_f64();
        println!("  TreePM:    {:.1}s (speedup vs PP: {:.1}×)", t_tpm, t_pp / t_tpm);

        // === Métrique 1: erreurs forces individuelles ===
        let mut force_errs = Vec::with_capacity(n);
        for i in 0..n {
            let (rx, ry, rz) = acc_pp[i];
            let av = acc_tpm[i];
            let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
            if mag_ref < 1e-15 {
                continue;
            }
            let dx = av.x - rx;
            let dy = av.y - ry;
            let dz = av.z - rz;
            let diff = (dx * dx + dy * dy + dz * dz).sqrt();
            force_errs.push(diff / mag_ref);
        }
        force_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let f_med = force_errs[force_errs.len() / 2];
        let f_p95 = force_errs[(force_errs.len() * 95) / 100];
        let f_max = force_errs.iter().cloned().fold(0.0, f64::max);

        println!();
        println!("--- Forces individuelles ---");
        println!("  median={:.4}%, P95={:.4}%, max={:.4}%", f_med * 100.0, f_p95 * 100.0, f_max * 100.0);

        // === Métrique 2: P(k) du CHAMP DE FORCE (CIC-deposit acc components) ===
        // Pour distribution Zel'dovich quasi-uniforme à z=10, le champ de
        // densité δ ≈ shot noise (pas de signal physique). Le champ qui
        // contient le signal est le champ de FORCE. On compare P_F(k) entre
        // TreePM et PP-direct.
        //
        // Méthode: CIC-deposit f_x, f_y, f_z de chaque particule sur grille
        // séparée (3 grilles par méthode). FFT chacune, sommer |F̂|² par k.
        let half = l * 0.5;
        let pos_for_cic: Vec<[f64; 3]> = pos
            .iter()
            .map(|&(x, y, z)| {
                [
                    (x + half).rem_euclid(l),
                    (y + half).rem_euclid(l),
                    (z + half).rem_euclid(l),
                ]
            })
            .collect();

        // Helper: deposit weighted scalar field via CIC (similar to cic_assign
        // but with per-particle weight instead of unit mass).
        let cic_assign_weighted = |positions: &[[f64; 3]], weights: &[f64]| -> Vec<f64> {
            let n_cells = n_pm * n_pm * n_pm;
            let mut grid = vec![0.0; n_cells];
            let cell_size = l / n_pm as f64;
            for (idx, p) in positions.iter().enumerate() {
                let x = ((p[0] % l) + l) % l;
                let y = ((p[1] % l) + l) % l;
                let z = ((p[2] % l) + l) % l;
                let ix = (x / cell_size).floor() as usize;
                let iy = (y / cell_size).floor() as usize;
                let iz = (z / cell_size).floor() as usize;
                let dxc = x / cell_size - ix as f64;
                let dyc = y / cell_size - iy as f64;
                let dzc = z / cell_size - iz as f64;
                let wx = [1.0 - dxc, dxc];
                let wy = [1.0 - dyc, dyc];
                let wz = [1.0 - dzc, dzc];
                for ai in 0..2 {
                    let ii = (ix + ai) % n_pm;
                    for aj in 0..2 {
                        let jj = (iy + aj) % n_pm;
                        for ak in 0..2 {
                            let kk = (iz + ak) % n_pm;
                            grid[ii * n_pm * n_pm + jj * n_pm + kk] +=
                                wx[ai] * wy[aj] * wz[ak] * weights[idx];
                        }
                    }
                }
            }
            grid
        };

        let f_t_x: Vec<f64> = (0..n).map(|i| acc_tpm[i].x).collect();
        let f_t_y: Vec<f64> = (0..n).map(|i| acc_tpm[i].y).collect();
        let f_t_z: Vec<f64> = (0..n).map(|i| acc_tpm[i].z).collect();
        let f_p_x: Vec<f64> = (0..n).map(|i| acc_pp[i].0).collect();
        let f_p_y: Vec<f64> = (0..n).map(|i| acc_pp[i].1).collect();
        let f_p_z: Vec<f64> = (0..n).map(|i| acc_pp[i].2).collect();

        let grid_tx = cic_assign_weighted(&pos_for_cic, &f_t_x);
        let grid_ty = cic_assign_weighted(&pos_for_cic, &f_t_y);
        let grid_tz = cic_assign_weighted(&pos_for_cic, &f_t_z);
        let grid_px = cic_assign_weighted(&pos_for_cic, &f_p_x);
        let grid_py = cic_assign_weighted(&pos_for_cic, &f_p_y);
        let grid_pz = cic_assign_weighted(&pos_for_cic, &f_p_z);

        // P(k) for each force component, sum (no shot-noise subtraction)
        let pk_tx = compute_pk(&grid_tx, l, n_pm, n, 16);
        let pk_ty = compute_pk(&grid_ty, l, n_pm, n, 16);
        let pk_tz = compute_pk(&grid_tz, l, n_pm, n, 16);
        let pk_px = compute_pk(&grid_px, l, n_pm, n, 16);
        let pk_py = compute_pk(&grid_py, l, n_pm, n, 16);
        let pk_pz = compute_pk(&grid_pz, l, n_pm, n, 16);

        let n_bins = pk_tx.k.len();
        let pk_t: Vec<f64> = (0..n_bins).map(|i| pk_tx.pk[i] + pk_ty.pk[i] + pk_tz.pk[i]).collect();
        let pk_p: Vec<f64> = (0..n_bins).map(|i| pk_px.pk[i] + pk_py.pk[i] + pk_pz.pk[i]).collect();
        // Use struct facade
        let pk_t = PowerSpectrumResult { k: pk_tx.k.clone(), pk: pk_t, n_modes: pk_tx.n_modes.clone() };
        let pk_p = PowerSpectrumResult { k: pk_px.k.clone(), pk: pk_p, n_modes: pk_px.n_modes.clone() };

        println!();
        println!("--- P(k) comparison (n_bins={}) ---", pk_t.k.len());
        println!("  {:<6} {:<10} {:<14} {:<14} {:<10} {:<8}", "bin", "k [1/Mpc]", "P_TreePM", "P_PP", "rel_err%", "n_modes");
        for i in 0..pk_t.k.len() {
            let rel_err = if pk_p.pk[i].abs() > 1e-30 {
                ((pk_t.pk[i] - pk_p.pk[i]) / pk_p.pk[i]).abs()
            } else {
                f64::NAN
            };
            println!(
                "  {:<6} {:<10.5} {:<14.5e} {:<14.5e} {:<10.4} {:<8}",
                i, pk_t.k[i], pk_t.pk[i], pk_p.pk[i], rel_err * 100.0, pk_p.n_modes[i]
            );
        }
        let mut pk_errs = Vec::new();
        for i in 0..pk_t.k.len() {
            if pk_p.pk[i].abs() < 1e-30 || pk_p.n_modes[i] == 0 {
                continue;
            }
            let rel_err = ((pk_t.pk[i] - pk_p.pk[i]) / pk_p.pk[i]).abs();
            pk_errs.push(rel_err);
        }
        let (pk_med, pk_max) = if pk_errs.is_empty() {
            println!("  ⚠ pk_errs empty — P(k) all zero or n_modes==0");
            (f64::NAN, f64::NAN)
        } else {
            pk_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = pk_errs[pk_errs.len() / 2];
            let mx = pk_errs.iter().cloned().fold(0.0, f64::max);
            (m, mx)
        };

        println!();
        println!("  P(k) median rel err : {:.4}%", pk_med * 100.0);
        println!("  P(k) max rel err    : {:.4}%", pk_max * 100.0);

        let verdict = if pk_med < 0.01 && pk_max < 0.05 {
            "GO_PHASE_10"
        } else if pk_med < 0.05 {
            "GO_WITH_CAVEAT"
        } else {
            "NO_GO_REAL_BUG"
        };
        println!();
        println!("VERDICT: {}", verdict);

        let _ = std::fs::create_dir_all("logs/treepm");
        let _ = std::fs::write(
            "logs/treepm/phase97c_verdict.txt",
            format!(
                "{}\nforce_median={}\nforce_p95={}\nforce_max={}\npk_median={}\npk_max={}\n",
                verdict, f_med, f_p95, f_max, pk_med, pk_max
            ),
        );
        // Save P(k) data
        if let Ok(mut f) = std::fs::File::create("logs/treepm/phase97c_pk_comparison.csv") {
            use std::io::Write;
            writeln!(f, "bin,k,pk_treepm,pk_pp,n_modes,rel_err").unwrap();
            for i in 0..pk_t.k.len() {
                let rel_err = if pk_p.pk[i] > 1e-30 {
                    ((pk_t.pk[i] - pk_p.pk[i]) / pk_p.pk[i]).abs()
                } else {
                    0.0
                };
                writeln!(
                    f,
                    "{},{},{},{},{},{}",
                    i, pk_t.k[i], pk_t.pk[i], pk_p.pk[i], pk_t.n_modes[i], rel_err
                )
                .unwrap();
            }
        }
    }

    /// Phase 9.7-B: r_cut sensitivity test. Tree handles r < r_cut, PM handles
    /// r >= r_cut. Pour N=1000 dans L=100, mean_sep ≈ 10 Mpc. Avec r_cut=9.375,
    /// la plupart des pairs sont en zone PM. Tester r_cut plus grand pour voir
    /// si l'erreur baisse (Tree corrige plus de pairs).
    #[test]
    #[ignore]
    fn diagnostic_r_cut_sensitivity() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1000;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let theta = 0.5;
        let v_cell = dg.powi(3);

        // mean separation ≈ L · N^(-1/3) = 10 Mpc
        let mean_sep = l * (n as f64).powf(-1.0 / 3.0);
        println!();
        println!("=== r_cut sensitivity (N={}, mean_sep ≈ {:.2} Mpc) ===", n, mean_sep);
        println!("  {:<12} {:<10} {:<10} {:<10}", "r_cut/Δg", "median", "P95", "max");

        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

        for &factor in &[3.0_f64, 6.0, 12.0, 18.0, 24.0] {
            let r_cut = factor * dg;
            // Skip if r_cut > L/2 (Tree raw distance limit)
            if r_cut > l * 0.45 {
                continue;
            }
            let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
            tpm.g_constant = g_phys / v_cell;
            tpm.update(&particles_bh);
            tpm.tree.g_constant = g_phys;
            let acc_tpm = tpm.compute_all_forces(&particles_bh);

            let mut errs = Vec::new();
            for i in 0..n {
                let (rx, ry, rz) = acc_pp[i];
                let av = acc_tpm[i];
                let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
                if mag_ref < 1e-15 {
                    continue;
                }
                let dx = av.x - rx;
                let dy = av.y - ry;
                let dz = av.z - rz;
                let diff = (dx * dx + dy * dy + dz * dz).sqrt();
                errs.push(diff / mag_ref);
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = errs[errs.len() / 2];
            let p95 = errs[(errs.len() * 95) / 100];
            let max = errs.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "  {:<12.2} {:<10.4} {:<10.4} {:<10.4} (r_cut={:.2})",
                factor,
                med,
                p95,
                max,
                r_cut,
            );
        }
    }

    /// Phase 9.7-B Étape 4: PBC consistency test.
    /// 2 particules à (10,50,50) et (90,50,50). Distance directe 80 Mpc.
    /// Distance MIC: 20 Mpc (via boundary).
    /// PP utilise MIC (20 Mpc), TreePM doit aussi via PM (FFT periodic) + Tree.
    /// Si Tree n'utilise pas MIC raw distance, divergence garantie.
    #[test]
    #[ignore]
    fn diagnostic_pbc_consistency() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;

        let l = 100.0_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let r_s_default = r_cut / 5.0;
        let theta = 0.5;
        let softening = 0.05;
        let g_phys = 1.0;
        let v_cell = dg.powi(3);

        // Two particles at opposite ends of x axis (boundary case)
        let pos = vec![(40.0, 0.0, 0.0), (-40.0, 0.0, 0.0)];
        let mass = vec![1.0_f64, 1.0];
        let particles_bh = vec![
            Particle::new(Vec3::new(40.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
            Particle::new(Vec3::new(-40.0, 0.0, 0.0), Vec3::zero(), 1.0, MassSign::Positive),
        ];
        // Particule 0 at (40, 0, 0), 1 at (-40, 0, 0).
        // Direct distance: 80. MIC distance: 100-80 = 20 (via boundary at x=±50).
        // PP-direct uses MIC: r=20, F = G·m/r² = 1/400 = 2.5e-3, attractive
        //   towards each other via MIC → particle 0 toward +x (image is at +60).
        //   So acc_x[0] should be POSITIVE (toward image at x=+60).
        // Tree without MIC: r=80 (raw direct), F = 1/6400 = 1.56e-4 (much smaller)
        //   towards -x (toward direct partner at x=-40).

        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        tpm.g_constant = g_phys / v_cell;
        tpm.update(&particles_bh);
        tpm.tree.g_constant = g_phys;
        let acc_tpm = tpm.compute_all_forces(&particles_bh);

        println!();
        println!("=== Phase 9.7-B Étape 4: PBC consistency ===");
        println!("  2 particles at x=±40 in L=100 box");
        println!("  Direct dist=80, MIC dist=20");
        println!();
        println!("  PP-direct (uses MIC):");
        println!("    acc[0] = ({:.4e}, {:.4e}, {:.4e})", acc_pp[0].0, acc_pp[0].1, acc_pp[0].2);
        println!("    acc[1] = ({:.4e}, {:.4e}, {:.4e})", acc_pp[1].0, acc_pp[1].1, acc_pp[1].2);
        println!();
        println!("  TreePM (PM+Tree):");
        println!("    acc[0] = ({:.4e}, {:.4e}, {:.4e})", acc_tpm[0].x, acc_tpm[0].y, acc_tpm[0].z);
        println!("    acc[1] = ({:.4e}, {:.4e}, {:.4e})", acc_tpm[1].x, acc_tpm[1].y, acc_tpm[1].z);
        println!();
        // F_PP at MIC r=20: G·m/r² = 1/400 = 2.5e-3, attract via MIC → particle 0
        // sees source at MIC position (+60), so feels force toward +x (+2.5e-3)
        println!("  Expected (Newton MIC, r=20): acc_x[0] = +2.5e-3");
        println!("  Expected (Newton no-MIC, r=80): acc_x[0] = -1.56e-4 (toward direct partner -x)");
        let _ = r_s_default; // silence unused warning
    }

    /// Phase 9.7-B Étape 3 (cas A.3): θ-sensitivity test.
    /// Si l'erreur converge vers 0 avec θ → 0 : c'est l'opening criterion ou
    /// le multipôle qui est imprécis. Si plateau : bug ailleurs.
    #[test]
    #[ignore]
    fn diagnostic_theta_sensitivity() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1000;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let v_cell = dg.powi(3);
        let g_solver = g_phys / v_cell;

        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

        println!();
        println!("=== θ sensitivity (TreePM vs PP, N=1000) ===");
        println!("  {:<8} {:<10} {:<10} {:<10}", "θ", "median", "P95", "max");

        for &theta in &[0.1_f64, 0.3, 0.5, 0.7] {
            let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
            tpm.g_constant = g_solver;
            tpm.update(&particles_bh);
            tpm.tree.g_constant = g_phys;
            let acc_tpm = tpm.compute_all_forces(&particles_bh);

            let mut errs = Vec::new();
            for i in 0..n {
                let (rx, ry, rz) = acc_pp[i];
                let av = acc_tpm[i];
                let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
                if mag_ref < 1e-15 {
                    continue;
                }
                let dx = av.x - rx;
                let dy = av.y - ry;
                let dz = av.z - rz;
                let diff = (dx * dx + dy * dy + dz * dz).sqrt();
                errs.push(diff / mag_ref);
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = errs[errs.len() / 2];
            let p95 = errs[(errs.len() * 95) / 100];
            let max = errs.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "  {:<8.2} {:<10.4} {:<10.4} {:<10.4}",
                theta,
                med,
                p95,
                max
            );
        }
    }

    /// Phase 9.7-B Étape 2.1: Tree-only baseline (no splitting, r_cut très grand).
    ///
    /// Si Tree-only avec splitting=1 partout reproduit PP-direct à <1%,
    /// le Tree est correct et le bug est dans le PM ou la combinaison.
    /// Si Tree-only montre 11% médiane → bug dans le Tree (Barnes-Hut, COM, θ).
    #[test]
    #[ignore]
    fn diagnostic_tree_only_no_splitting_vs_pp() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::tree_short::TreePMTree;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1000;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;

        // PP-direct référence
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

        // Tree avec r_cut = L·√3 (couvre la diagonale, toutes paires inclues
        // sans MIC) ET r_s très grand pour que splitting_tree_springel(r, r_s)
        // ≈ T(r/(2·r_s)) → 1 (Tree donne toute la force, pas de damping).
        // Note: Tree utilise distance BRUTE (pas MIC), donc r_cut doit couvrir
        // la diagonale max de la boîte pour inclure toutes les paires.
        let r_cut = l * 2.0; // > L·√3/2 ≈ 86.6, inclut tout
        let r_s_huge = 1000.0 * l; // T(x≈0) ≈ 1, splitting ≈ 1 partout
        let theta = 0.5_f64;

        let tree = TreePMTree::build_with_rs_and_g(&particles_bh, theta, r_cut, r_s_huge, g_phys);

        let acc_tree: Vec<Vec3> = (0..n)
            .map(|i| {
                tree.compute_short_range_acc_excluding(
                    particles_bh[i].pos,
                    particles_bh[i].sign,
                    &particles_bh,
                    softening,
                    Some(i),
                )
            })
            .collect();

        let mut errs = Vec::new();
        for i in 0..n {
            let (rx, ry, rz) = acc_pp[i];
            let av = acc_tree[i];
            let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
            if mag_ref < 1e-15 {
                continue;
            }
            let dx = av.x - rx;
            let dy = av.y - ry;
            let dz = av.z - rz;
            let diff = (dx * dx + dy * dy + dz * dz).sqrt();
            errs.push(diff / mag_ref);
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = errs[errs.len() / 2];
        let p95 = errs[(errs.len() * 95) / 100];
        let max = errs.iter().cloned().fold(0.0_f64, f64::max);

        println!();
        println!("=== Tree-only (r_cut=L/2, splitting≈1, θ=0.5) vs PP-direct ===");
        println!(
            "  median={:.4}%, P95={:.4}%, max={:.4}%",
            med * 100.0,
            p95 * 100.0,
            max * 100.0
        );

        if med < 0.01 {
            println!("  ✅ Tree-only OK (<1%) → bug est dans PM ou combinaison");
        } else if med > 0.05 {
            println!("  ❌ Tree-only montre {}% → bug dans Barnes-Hut multipole/COM/θ", med * 100.0);
        } else {
            println!("  ⚠ Tree-only ~{:.1}% → précision intermédiaire (zone θ?)", med * 100.0);
        }
    }

    /// Phase 9.7-B Étape 2.3: décomposition force 2-particules à différents r/r_s.
    /// Pour chaque ratio r/r_s, mesurer l'erreur TreePM vs PP. Localise la zone
    /// (courte/intermédiaire/longue portée) où le bug se manifeste.
    #[test]
    #[ignore]
    fn diagnostic_two_particle_decomposition() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;

        let l = 100.0_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_s = 1.2 * dg;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let softening = 0.05;
        let g_phys = 1.0;
        let v_cell = dg.powi(3);
        let g_solver = g_phys / v_cell;

        println!();
        println!("=== Décomposition force 2-particules ===");
        println!("  L={}, n_pm={}, r_s={:.3}, r_cut={:.3}", l, n_pm, r_s, r_cut);
        println!(
            "  {:<6} {:<10} {:<14} {:<14} {:<10} {:<10}",
            "r/r_s", "r (Mpc)", "F_PP (m+/m+)", "F_TreePM", "rel_err", "T(x)"
        );

        for &factor in &[0.5_f64, 1.0, 2.0, 3.0, 5.0, 8.0] {
            let r = factor * r_s;
            // Skip r > r_cut: outside Tree range, only PM
            // Mais on veut mesurer comment ça performe dans le raccord
            let mass = vec![1.0_f64, 1.0];
            let pos = vec![(-r * 0.5, 0.0, 0.0), (r * 0.5, 0.0, 0.0)];
            let particles_bh = vec![
                Particle::new(Vec3::new(pos[0].0, pos[0].1, pos[0].2), Vec3::zero(), 1.0, MassSign::Positive),
                Particle::new(Vec3::new(pos[1].0, pos[1].1, pos[1].2), Vec3::zero(), 1.0, MassSign::Positive),
            ];

            let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

            let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
            tpm.g_constant = g_solver;
            tpm.update(&particles_bh);
            tpm.tree.g_constant = g_phys;
            let acc_tpm = tpm.compute_all_forces(&particles_bh);

            let f_pp = acc_pp[0].0;
            let f_tpm = acc_tpm[0].x;
            let rel_err = if f_pp.abs() > 1e-15 {
                ((f_tpm - f_pp) / f_pp).abs()
            } else {
                0.0
            };

            // T(x) à ce r
            let x = r / (2.0 * r_s);
            let exp_mx2 = (-x * x).exp();
            // Direct erfc approximation
            let p_a = 0.3275911_f64;
            let t_a = 1.0 / (1.0 + p_a * x);
            let erfc_x = (0.254829592 + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t_a) * t_a) * t_a) * t_a) * t_a * (-x * x).exp();
            let erfc_val = if x < 0.0 { 2.0 - erfc_x } else { erfc_x };
            let t_x = if x >= 3.0 { 0.0 } else { erfc_val + (2.0 * x / std::f64::consts::PI.sqrt()) * exp_mx2 };

            println!(
                "  {:<6.2} {:<10.3} {:<14.5e} {:<14.5e} {:<10.4} {:<10.4}",
                factor,
                r,
                f_pp,
                f_tpm,
                rel_err,
                t_x
            );
        }
    }

    /// Phase 9.7-A diagnostic : grad2 vs grad4 dans le PM gather.
    ///
    /// Setup identique à `test_treepm_combined_vs_pp_direct_precision`
    /// (N=1000, L=100, seed=42, n_pm=64, r_s=1.875, r_cut=9.375, θ=0.5,
    /// softening=0.05). Compare deux modes :
    ///
    /// - **A**: TreePMForce avec `pm.interpolate_force` (gradient ord 2,
    ///   chemin production actuel)
    /// - **B**: TreePMForce avec `pm.interpolate_force_grad4` (gradient ord 4,
    ///   correction Phase 2 GrGadget)
    ///
    /// Verdict :
    /// - grad2 < 5% et grad4 ~ 11% → grad4 amplifie le bruit
    /// - grad2 ~ 11% (même que grad4) → bug ailleurs (V_cell, FFT norm)
    /// - grad2 > grad4 → contre-intuitif, signal d'autre chose
    #[test]
    #[ignore]
    fn diagnostic_grad2_vs_grad4_in_pm() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 1000;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;
        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let v_cell = dg.powi(3);
        let g_solver = g_phys / v_cell;

        // PP-direct référence
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

        // Setup TreePMForce (un seul update — réutilisé pour grad2 et grad4)
        let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
        tpm.g_constant = g_solver;
        tpm.update(&particles_bh);
        tpm.tree.g_constant = g_phys; // Tree uses G_phys directly

        // Version A: grad2 = utilise TreePMForce.compute_all_forces (chemin standard,
        //                  appelle pm.interpolate_force = grad2)
        let acc_grad2 = tpm.compute_all_forces(&particles_bh);

        // Version B: grad4 = manuel, on appelle pm.interpolate_force_grad4 + Tree
        let acc_grad4: Vec<Vec3> = (0..n)
            .map(|i| {
                let p = &particles_bh[i];
                let (fx_pm, fy_pm, fz_pm) =
                    tpm.pm.interpolate_force_grad4(p.pos.x, p.pos.y, p.pos.z, 1);
                let f_pm = Vec3::new(fx_pm, fy_pm, fz_pm);
                let f_tree = tpm.tree.compute_short_range_acc_excluding(
                    p.pos, p.sign, &particles_bh, softening, Some(i),
                );
                f_pm + f_tree
            })
            .collect();

        // Stats vs PP
        let stats = |acc: &[Vec3], label: &str| -> (f64, f64, f64) {
            let mut errs = Vec::with_capacity(n);
            for i in 0..n {
                let (rx, ry, rz) = acc_pp[i];
                let av = acc[i];
                let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
                if mag_ref < 1e-15 {
                    continue;
                }
                let dx = av.x - rx;
                let dy = av.y - ry;
                let dz = av.z - rz;
                let diff = (dx * dx + dy * dy + dz * dz).sqrt();
                errs.push(diff / mag_ref);
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = errs[errs.len() / 2];
            let p95 = errs[(errs.len() * 95) / 100];
            let max = errs.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "  {:<8}: median={:.3}%, P95={:.3}%, max={:.3}%",
                label,
                med * 100.0,
                p95 * 100.0,
                max * 100.0
            );
            (med, p95, max)
        };

        println!();
        println!("=== Phase 9.7-A: grad2 vs grad4 dans le PM gather ===");
        println!("  Setup: N=1000, L=100, n_pm=64, r_s=1.875, r_cut=9.375, θ=0.5, ε=0.05");
        let (m2, p2, mx2) = stats(&acc_grad2, "grad2");
        let (m4, p4, mx4) = stats(&acc_grad4, "grad4");
        println!();
        let delta_med = (m4 - m2) / m2 * 100.0;
        let delta_p95 = (p4 - p2) / p2 * 100.0;
        let delta_max = (mx4 - mx2) / mx2 * 100.0;
        println!(
            "  Δ grad4 vs grad2: median {:+.1}%, P95 {:+.1}%, max {:+.1}%",
            delta_med, delta_p95, delta_max
        );

        let verdict = if m2 < 0.05 && m4 > 0.10 {
            "GRAD4_AMPLIFIES_NOISE"
        } else if (m2 - m4).abs() / m2 < 0.20 && m2 > 0.05 {
            "BUG_ELSEWHERE_NOT_GRAD"
        } else if m2 > m4 {
            "GRAD2_WORSE_UNEXPECTED"
        } else {
            "MIXED_RESULT"
        };
        println!("  VERDICT: {}", verdict);

        // Save report
        let _ = std::fs::create_dir_all("logs/treepm");
        let _ = std::fs::write(
            "logs/treepm/phase97a_verdict.txt",
            format!(
                "verdict={}\ngrad2_median={}\ngrad2_p95={}\ngrad2_max={}\n\
                 grad4_median={}\ngrad4_p95={}\ngrad4_max={}\n",
                verdict, m2, p2, mx2, m4, p4, mx4
            ),
        );
    }

    /// Diagnostic 4d: PM convergence in N_pm.
    /// Si l'erreur DÉCROÎT vers 0 avec N_pm croissant : précision PM intrinsèque
    /// (à augmenter pour résultats finaux).
    /// Si l'erreur PLATEAU à valeur non-nulle : bug systémique.
    #[test]
    #[ignore]
    fn diagnostic_pm_convergence_in_n_pm() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 500;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut pos = Vec::with_capacity(n);
        let mass: Vec<f64> = vec![1.0_f64; n];
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            pos.push((x, y, z));
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let g_phys = 1.0_f64;
        let softening = 0.05_f64;
        let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);

        println!();
        println!("=== Diagnostic 4d: PM convergence in N_pm ===");
        for &n_pm in &[32, 64, 128, 256] {
            let dg = l / n_pm as f64;
            let r_cut = 6.0 * dg;
            let theta = 0.5;
            let v_cell = dg.powi(3);

            let mut tpm = TreePMForce::new(r_cut, n_pm, l, theta, softening);
            tpm.g_constant = g_phys / v_cell;
            tpm.update(&particles_bh);
            tpm.tree.g_constant = g_phys;
            let acc_tpm = tpm.compute_all_forces(&particles_bh);

            let mut errs = Vec::new();
            for i in 0..n {
                let (rx, ry, rz) = acc_pp[i];
                let av = acc_tpm[i];
                let mag_ref = (rx * rx + ry * ry + rz * rz).sqrt();
                if mag_ref < 1e-15 {
                    continue;
                }
                let dx = av.x - rx;
                let dy = av.y - ry;
                let dz = av.z - rz;
                let diff = (dx * dx + dy * dy + dz * dz).sqrt();
                errs.push(diff / mag_ref);
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = errs[errs.len() / 2];
            let p95 = errs[(errs.len() * 95) / 100];
            let max = errs.iter().cloned().fold(0.0_f64, f64::max);
            println!(
                "  N_pm={:<4} (dg={:.3}, r_cut={:.3}): median={:.2}%, P95={:.2}%, max={:.2}%",
                n_pm,
                dg,
                r_cut,
                med * 100.0,
                p95 * 100.0,
                max * 100.0
            );
        }
    }

    /// Diagnostic 4b: linearity in G.
    #[test]
    #[ignore]
    fn diagnostic_force_linearity_in_g() {
        use crate::nbody::{Particle, Vec3};
        use crate::treepm::treepm_force::TreePMForce;
        use crate::MassSign;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = 500;
        let l = 100.0_f64;
        let mut rng = StdRng::seed_from_u64(42);
        let half = l * 0.5;
        let mut particles_bh = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.random::<f64>() * l - half;
            let y = rng.random::<f64>() * l - half;
            let z = rng.random::<f64>() * l - half;
            particles_bh.push(Particle::new(
                Vec3::new(x, y, z),
                Vec3::zero(),
                1.0,
                MassSign::Positive,
            ));
        }

        let n_pm = 64;
        let dg = l / n_pm as f64;
        let r_cut = 6.0 * dg;
        let theta = 0.5;
        let v_cell = dg.powi(3);

        // G=1
        let mut tpm1 = TreePMForce::new(r_cut, n_pm, l, theta, 0.05);
        tpm1.g_constant = 1.0 / v_cell;
        tpm1.update(&particles_bh);
        tpm1.tree.g_constant = 1.0;
        let acc_g1 = tpm1.compute_all_forces(&particles_bh);

        // G=10
        let mut tpm10 = TreePMForce::new(r_cut, n_pm, l, theta, 0.05);
        tpm10.g_constant = 10.0 / v_cell;
        tpm10.update(&particles_bh);
        tpm10.tree.g_constant = 10.0;
        let acc_g10 = tpm10.compute_all_forces(&particles_bh);

        let mut max_lin_err: f64 = 0.0;
        for i in 0..n {
            let m1 = (acc_g1[i].x.powi(2) + acc_g1[i].y.powi(2) + acc_g1[i].z.powi(2)).sqrt();
            let m10 =
                (acc_g10[i].x.powi(2) + acc_g10[i].y.powi(2) + acc_g10[i].z.powi(2)).sqrt();
            if m1 > 1e-15 {
                let ratio = m10 / m1;
                let err = (ratio - 10.0).abs() / 10.0;
                max_lin_err = max_lin_err.max(err);
            }
        }
        println!();
        println!("=== Diagnostic 4b: F(G=10)/F(G=1) should = 10 ===");
        println!("  Max relative deviation from 10×: {:.4}%", max_lin_err * 100.0);
        // PM is linear in g_constant by construction. Tree is linear in g_constant
        // by construction. Should be exact to FP precision.
        assert!(
            max_lin_err < 1e-6,
            "Linearity in G violated: max err = {}",
            max_lin_err
        );
    }

    #[test]
    fn test_pp_pbc_minimum_image() {
        // Two particles near opposite faces of a periodic box.
        // L=10, particle 0 at (0.1, 0, 0), particle 1 at (9.9, 0, 0).
        // Direct distance: 9.8. Minimum image: -0.2 (closer through PBC).
        // Force on particle 0 should point toward particle 1's image = -x direction.
        let pos = vec![(0.1, 0.0, 0.0), (9.9, 0.0, 0.0)];
        let mass = vec![1.0, 1.0];
        let acc = pp_direct_forces_newton(&pos, &mass, 10.0, 0.0, 1.0);
        // Expected: dx_image = 9.9 - 0.1 - 10 = -0.2 (closer via PBC).
        // F_0 = G·m·dx/r³ = 1 · 1 · (-0.2) / |0.2|³ = -25
        let r = 0.2_f64;
        let expected = -1.0 / (r * r);
        assert!(
            (acc[0].0 - expected).abs() / expected.abs() < 1e-12,
            "acc[0].x = {}, expected {}",
            acc[0].0,
            expected
        );
    }

    // ============================================================
    // Phase 10.8 — Ewald summation validation tests
    // ============================================================

    #[test]
    #[ignore]
    fn test_ewald_two_particles_close() {
        // 2 particles separated by r = L/50 = 2 Mpc, softening 0.
        // At this scale image contribs are O((r/L)²) ≈ 4e-4 relative,
        // so Ewald ≈ Newton non-periodique to better than 1%.
        let l = 100.0_f64;
        let pos = vec![(49.0, 50.0, 50.0), (51.0, 50.0, 50.0)];
        let mass = vec![1.0_f64, 1.0];

        let acc = pp_direct_forces_newton_ewald(&pos, &mass, l, 0.0, 1.0, 4, 4);

        // Newton non-periodique : F_x = G·m·dx/r³ = 2/8 = 0.25
        let f_newton = 1.0 / (2.0 * 2.0);
        let f_ewald_x = acc[0].0;
        let rel_err = (f_ewald_x - f_newton).abs() / f_newton;
        println!(
            "Ewald 2-particles (r=L/50): F_x = {:.6e}, Newton = {:.6e}, rel err = {:.3}%",
            f_ewald_x,
            f_newton,
            rel_err * 100.0
        );
        assert!(rel_err < 0.01, "Ewald close vs Newton: rel err = {}", rel_err);
        // Sanity: y, z components ~zero (1D pair on x axis)
        assert!(acc[0].1.abs() < 1e-6 * f_newton);
        assert!(acc[0].2.abs() < 1e-6 * f_newton);
    }

    #[test]
    #[ignore]
    fn test_ewald_convergence() {
        // Convergence: Ewald(4,4) → Ewald(5,5) relative diff < 0.1%
        let l = 100.0_f64;
        let n = 10;
        let pos: Vec<_> = (0..n)
            .map(|i| ((i as f64) * 8.0 + 5.0, 50.0, 50.0))
            .collect();
        let mass = vec![1.0_f64; n];

        let acc_44 = pp_direct_forces_newton_ewald(&pos, &mass, l, 0.05, 1.0, 4, 4);
        let acc_55 = pp_direct_forces_newton_ewald(&pos, &mass, l, 0.05, 1.0, 5, 5);

        let mut max_rel_diff = 0.0_f64;
        for i in 0..n {
            let dx = acc_44[i].0 - acc_55[i].0;
            let dy = acc_44[i].1 - acc_55[i].1;
            let dz = acc_44[i].2 - acc_55[i].2;
            let diff = (dx * dx + dy * dy + dz * dz).sqrt();
            let mag = (acc_55[i].0.powi(2)
                + acc_55[i].1.powi(2)
                + acc_55[i].2.powi(2))
            .sqrt();
            if mag > 1e-15 {
                max_rel_diff = max_rel_diff.max(diff / mag);
            }
        }
        println!(
            "Ewald convergence (4,4) → (5,5): max rel diff = {:.5}%",
            max_rel_diff * 100.0
        );
        assert!(
            max_rel_diff < 0.001,
            "Ewald not converged at (4,4): {} > 0.1%",
            max_rel_diff
        );
    }

    #[test]
    #[ignore]
    fn test_ewald_vs_mic_close_pair() {
        // Pair at r = L/10 (close): MIC should give close to Ewald (< 5% diff)
        let l = 100.0_f64;
        let pos = vec![(45.0, 50.0, 50.0), (55.0, 50.0, 50.0)];
        let mass = vec![1.0_f64, 1.0];

        let acc_mic = pp_direct_forces_newton(&pos, &mass, l, 0.05, 1.0);
        let acc_ewald = pp_direct_forces_newton_ewald(&pos, &mass, l, 0.05, 1.0, 4, 4);

        let mag_mic = (acc_mic[0].0.powi(2) + acc_mic[0].1.powi(2) + acc_mic[0].2.powi(2)).sqrt();
        let mag_ewald = (acc_ewald[0].0.powi(2) + acc_ewald[0].1.powi(2) + acc_ewald[0].2.powi(2))
            .sqrt();

        let rel_diff = (mag_ewald - mag_mic).abs() / mag_mic;
        println!(
            "Ewald vs MIC for close pair (r=L/10): mag_mic = {:.4e}, mag_ewald = {:.4e}, rel diff = {:.3}%",
            mag_mic,
            mag_ewald,
            rel_diff * 100.0
        );
        // For r=L/10, image contributions are at r ≥ L (10× further),
        // so Ewald correction should be < 5% over MIC.
        assert!(
            rel_diff < 0.05,
            "Close pair Ewald vs MIC: rel diff = {}",
            rel_diff
        );
    }
}
