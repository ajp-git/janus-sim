//! Phase 10.5 — Décomposition diagnostique PM/Tree GPU.
//!
//! Sur 2 paires synthétiques m+/m+ à distance r ∈ {0.5, 1, 1.5, 2, 3, 5, 8} × r_s,
//! mesurer 3 forces GPU :
//!  1. PM-only (compute_pm_only_janus)
//!  2. Tree-only (compute_tree_only_janus)
//!  3. Total = PM + Tree (step_treepm_gpu_cosmo with dt=0)
//!
//! Comparer aux attentes analytiques :
//!   F_PP(r) = G·m·r̂/(r²+ε²)^(3/2) (Newton + Plummer)
//!   F_PM_expected(r) = F_PP × [1 - T(r/(2·r_s))]
//!   F_Tree_expected(r) = F_PP × T(r/(2·r_s))
//!
//! Le pattern des ratios identifie la cause:
//!  Cas A : ratio_PM constant ~0.85 → bug PM normalisation
//!  Cas B : ratio_Tree constant ~0.85 → bug Tree double sign
//!  Cas C : variations avec r/r_s → mismatch raccord
//!  Cas D : tous ~1.0 → effet collectif (N grand uniquement)

#[cfg(all(feature = "cuda", feature = "cufft"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use janus::nbody_gpu_twopass::GpuNBodyTwoPass;
    use std::f64::consts::PI;

    println!("=== Phase 10.5 — GPU PM/Tree Decomposition Diagnostic ===");

    let l = 100.0_f64;
    let n_pm = 64;
    let dg = l / n_pm as f64;
    let r_s = 1.2 * dg;
    let r_cut = 6.0 * dg;
    let theta = 0.5;
    let softening = 0.05_f64;
    let g_phys = 1.0_f64;

    // Cosmologie neutre pour analyse de précision pure
    let phi = 1.0_f64;
    let c_ratio_sq = 1.0_f64;
    let repulsion_scale = 1.0_f64;

    println!("Setup: r_s={:.3}, r_cut={:.3}, n_pm={}", r_s, r_cut, n_pm);
    println!();
    println!("{:<6} {:<8} {:<14} {:<14} {:<14} {:<14} {:<10} {:<10} {:<10}",
        "r/r_s", "r [Mpc]", "F_PP", "F_PM_GPU", "F_Tree_GPU", "F_Total_GPU",
        "rPM", "rTree", "rTotal");
    println!("{}", "-".repeat(120));

    // Ligne d'analyse: stocker les valeurs pour verdict
    let mut all_ratios_pm = Vec::new();
    let mut all_ratios_tree = Vec::new();
    let mut all_ratios_total = Vec::new();

    let make_setup = |r: f64| -> (Vec<f32>, Vec<f32>, Vec<i8>) {
        // 2 m+ paires + 20 filler particles loin (10 m+, 10 m-) pour BVH
        let mut pos = vec![
            -0.5 * r as f32, 0.0, 0.0,
             0.5 * r as f32, 0.0, 0.0,
        ];
        let mut vel = vec![0.0_f32; 6];
        let mut signs = vec![1_i8, 1_i8];
        for k in 0..10 {
            let off = 30.0 + k as f32 * 5.0;
            pos.extend_from_slice(&[0.0, off, 0.0]);
            vel.extend_from_slice(&[0.0, 0.0, 0.0]);
            signs.push(1);
        }
        for k in 0..10 {
            let off = -30.0 - k as f32 * 5.0;
            pos.extend_from_slice(&[0.0, off, 0.0]);
            vel.extend_from_slice(&[0.0, 0.0, 0.0]);
            signs.push(-1);
        }
        (pos, vel, signs)
    };

    for &factor in &[0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0] {
        let r = factor * r_s;

        // === Référence analytique ===
        let r2_soft = r * r + softening * softening;
        let f_pp = g_phys * 1.0 / r2_soft; // |F| sur particule 0 vers particule 1

        let x = r / (2.0 * r_s);
        let t_split = if x < 3.0 {
            erfc_approx(x) + (2.0 * x / PI.sqrt()) * (-x * x).exp()
        } else {
            0.0
        };
        let f_pm_expected = f_pp * (1.0 - t_split);
        let f_tree_expected = f_pp * t_split;

        // === Mesure 1: PM-only ===
        let (pos, vel, signs) = make_setup(r);
        let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, l)?;
        sim.set_mass_factor(1.0);
        sim.set_softening(softening);
        sim.set_theta(theta);
        sim.compute_pm_only_janus(r_s, phi, c_ratio_sq, repulsion_scale)?;
        let acc_pm = sim.get_acc()?;
        let f_pm_x = acc_pm[0]; // x-component of force on particle 0
        // Force should point toward +x (attraction to particle at +r/2)
        // |F| ≈ acc_pm_x (ignoring tiny y/z from filler)

        // === Mesure 2: Tree-only ===
        let (pos, vel, signs) = make_setup(r);
        let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, l)?;
        sim.set_mass_factor(1.0);
        sim.set_softening(softening);
        sim.set_theta(theta);
        sim.compute_tree_only_janus(r_cut, r_s, phi, c_ratio_sq, repulsion_scale)?;
        let acc_tree = sim.get_acc()?;
        let f_tree_x = acc_tree[0];

        // === Mesure 3: Total (full pipeline) ===
        let (pos, vel, signs) = make_setup(r);
        let mut sim = GpuNBodyTwoPass::with_custom_ics(pos, vel, signs, l)?;
        sim.set_mass_factor(1.0);
        sim.set_softening(softening);
        sim.set_theta(theta);
        sim.step_treepm_gpu_cosmo(
            0.0, r_cut, r_s,
            1.0, 1.0, 0.0, 0.0,
            phi, c_ratio_sq, repulsion_scale,
        )?;
        let acc_total = sim.get_acc()?;
        let f_total_x = acc_total[0];

        // Magnitudes (signed, attraction positive vers +x for particle 0 at -r/2)
        let f_pm_mag = f_pm_x as f64;
        let f_tree_mag = f_tree_x as f64;
        let f_total_mag = f_total_x as f64;

        let r_pm = if f_pm_expected > 1e-12 { f_pm_mag / f_pm_expected } else { 1.0 };
        let r_tree = if f_tree_expected > 1e-12 { f_tree_mag / f_tree_expected } else { 1.0 };
        let r_total = if f_pp > 1e-12 { f_total_mag / f_pp } else { 1.0 };

        all_ratios_pm.push(r_pm);
        all_ratios_tree.push(r_tree);
        all_ratios_total.push(r_total);

        println!("{:<6.2} {:<8.3} {:<14.4e} {:<14.4e} {:<14.4e} {:<14.4e} {:<10.4} {:<10.4} {:<10.4}",
            factor, r, f_pp, f_pm_mag, f_tree_mag, f_total_mag,
            r_pm, r_tree, r_total);
    }

    // === Verdict ===
    println!();
    println!("=== Analyse ===");
    let mean_pm: f64 = all_ratios_pm.iter().sum::<f64>() / all_ratios_pm.len() as f64;
    let mean_tree: f64 = all_ratios_tree.iter().sum::<f64>() / all_ratios_tree.len() as f64;
    let mean_total: f64 = all_ratios_total.iter().sum::<f64>() / all_ratios_total.len() as f64;
    let var_pm: f64 = all_ratios_pm.iter().map(|x| (x - mean_pm).powi(2)).sum::<f64>() / all_ratios_pm.len() as f64;
    let var_tree: f64 = all_ratios_tree.iter().map(|x| (x - mean_tree).powi(2)).sum::<f64>() / all_ratios_tree.len() as f64;
    println!("  ratio_PM    : mean = {:.4}, std = {:.4}", mean_pm, var_pm.sqrt());
    println!("  ratio_Tree  : mean = {:.4}, std = {:.4}", mean_tree, var_tree.sqrt());
    println!("  ratio_Total : mean = {:.4}", mean_total);

    let pm_constant_low = (mean_pm - 0.85).abs() < 0.10 && var_pm.sqrt() < 0.10;
    let tree_constant_low = (mean_tree - 0.85).abs() < 0.10 && var_tree.sqrt() < 0.10;
    let pm_close_one = (mean_pm - 1.0).abs() < 0.10;
    let tree_close_one = (mean_tree - 1.0).abs() < 0.10;
    let pm_variable = var_pm.sqrt() > 0.15;
    let tree_variable = var_tree.sqrt() > 0.15;

    println!();
    if pm_constant_low && tree_close_one {
        println!("VERDICT: Cas A — bug côté PM (mean ratio_PM ≈ {:.2} constant)", mean_pm);
        println!("→ Probable: g_constant scaling, FFT norm, ou facteur 4πG");
    } else if tree_constant_low && pm_close_one {
        println!("VERDICT: Cas B — bug côté Tree (mean ratio_Tree ≈ {:.2} constant)", mean_tree);
        println!("→ Probable: double sign_factor, ou splitting factor mal appliqué");
    } else if pm_variable || tree_variable {
        println!("VERDICT: Cas C — bug zone de raccord PM/Tree (ratios variables)");
        println!("→ Probable: T(x) GPU vs CPU mismatch");
    } else if pm_close_one && tree_close_one {
        println!("VERDICT: Cas D — paradoxe (tous ratios ≈ 1.0 sur 2 paires)");
        println!("→ Effet collectif N>2, à investiguer avec N=10, 100, 1000");
    } else {
        println!("VERDICT: pattern atypique — rPM={:.3} rTree={:.3} rTotal={:.3}",
                 mean_pm, mean_tree, mean_total);
        println!("→ Analyse manuelle requise");
    }

    Ok(())
}

/// Abramowitz §7.1.26 erfc approx (same as truncation_table.rs)
#[cfg(all(feature = "cuda", feature = "cufft"))]
fn erfc_approx(x: f64) -> f64 {
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (a1 + (a2 + (a3 + (a4 + a5 * t) * t) * t) * t) * t * (-x_abs * x_abs).exp();
    1.0 - sign * y
}

#[cfg(not(all(feature = "cuda", feature = "cufft")))]
fn main() {
    eprintln!("Requires features: cuda cufft");
    std::process::exit(1);
}
