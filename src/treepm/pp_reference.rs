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
}
