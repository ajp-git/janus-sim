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
}
