# PROMPT CLI — Vérification précision TreePM combiné vs PP-direct

**Mission** : avant de porter Phase 5 sur GPU, vérifier que le pipeline TreePM CPU intégré (PM + Tree combinés) atteint une précision conforme à la littérature (GrGadget Fig. 5, PhotoNs §6) sur un test à 1000 particules. Le résultat de ce test détermine si on porte sur GPU ou si on debug d'abord.

**Contexte** : Phase 9 a livré 81/81 tests passants, mais le test PM-only `test_pm_force_single_source_grad4` (Phase 6.1) montre `54%/30%/12% err à 6/12/24 dg`. Trois hypothèses possibles :

1. **Hypothèse A (acceptable)** : c'est PM-only sans Tree, donc à r ~ r_cut, l'erreur PM seule est intrinsèquement élevée. La somme PM+Tree (TreePM) est précise.
2. **Hypothèse B (bug)** : il manque un facteur de normalisation (4πG, FFT norm, ou cell_volume) qui se manifestera aussi en TreePM combiné.
3. **Hypothèse C (bug subtil)** : la déconvolution CIC W⁻² est mal placée dans le pipeline.

Ce test tranche entre A, B, et C.

---

## Règles d'opération

- **Mode autonome** : aucune question sauf si bloqueur algo grave. Tous les choix d'implémentation sont déjà spécifiés ci-dessous.
- **Pas de modification du pipeline existant** : ajouter un nouveau test isolé, ne rien changer aux modules `treepm/`.
- **Branche** : continuer sur `feat/treepm-jpp-port` (même branche que Phase 1-9).
- **Logs** : créer `logs/treepm/precision_check_$(date +%F).log` avec sortie complète des tests.
- **Commit final** : `treepm-precision-check: TreePM CPU vs PP-direct, verdict <A|B|C>`.

---

## Étape 1 — Localiser le pipeline CPU intégré

Identifier la fonction CPU qui combine PM + Tree pour calculer la force totale sur un set de particules. C'est probablement quelque part dans `src/treepm/` (vu qu'il y a 1865 LOC selon Phase 1 audit). Si elle n'existe pas en CPU pur (parce que tout passait par GPU avant), créer un wrapper minimal qui :

1. Construit `PmGrid` avec N_pm cellules pour la boîte
2. Dépose les masses CIC (en mass count, conformément à la convention identifiée Phase 6)
3. Résout Poisson avec `g_constant = G_phys / V_cell`
4. Interpole la force PM via `interpolate_force_grad4`
5. Construit l'arbre Barnes-Hut (ou tree existant) avec opening angle θ=0.5
6. Calcule la force Tree à courte portée avec splitting `T(r/2r_s) = erfc(...) + (2x/√π)·exp(-x²)`
7. Somme `F_total = F_PM + F_tree`

Si ce wrapper existe déjà, l'utiliser tel quel. S'il n'existe pas, le créer dans `src/treepm/cpu_pipeline.rs` en mode minimal — il sert seulement à ce test, pas à la production.

**Documenter dans le log** : path exact de la fonction utilisée, et si elle a été créée ou existait déjà.

---

## Étape 2 — Implémenter le test de référence PP-direct

Dans `src/treepm/pp_reference.rs` (nouveau module), implémenter un calcul de force direct N² sans aucune approximation :

```rust
/// Direct N² particle-particle force computation, Newton only (no Janus).
/// 
/// Reference for validating TreePM accuracy. O(N²) so only feasible for N ≤ ~10K.
/// 
/// Convention : 
///   F_i = -G · Σ_{j≠i} m_j × (r_i - r_j) / |r_i - r_j|³
/// 
/// With Plummer softening :
///   |r_i - r_j|³ → (|r_i - r_j|² + ε²)^(3/2)
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
            if i == j { continue; }
            
            let (xj, yj, zj) = pos[j];
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            
            // Minimum image convention
            if dx >  half_l { dx -= box_size; }
            if dx < -half_l { dx += box_size; }
            if dy >  half_l { dy -= box_size; }
            if dy < -half_l { dy += box_size; }
            if dz >  half_l { dz -= box_size; }
            if dz < -half_l { dz += box_size; }
            
            let r2 = dx*dx + dy*dy + dz*dz + eps2;
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
```

**Tests obligatoires** pour valider PP-direct lui-même :

```rust
#[test]
fn test_pp_two_particles_attract() {
    let pos = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
    let mass = vec![1.0, 1.0];
    let acc = pp_direct_forces_newton(&pos, &mass, 100.0, 0.0, 1.0);
    
    // F_0 sur 0 = +G·m_1 · (r_1 - r_0) / r³ = (1, 0, 0)
    assert!((acc[0].0 - 1.0).abs() < 1e-12, "acc_x[0] = {}", acc[0].0);
    assert!(acc[0].1.abs() < 1e-12);
    assert!(acc[0].2.abs() < 1e-12);
    
    // Action-reaction : F_0 sur 1 = -F_0 sur 0
    assert!((acc[1].0 + 1.0).abs() < 1e-12, "acc_x[1] = {}", acc[1].0);
}

#[test]
fn test_pp_three_particles_symmetric() {
    // Triangle équilatéral autour de l'origine, masses égales
    let r = 1.0_f64;
    let pos = vec![
        (r, 0.0, 0.0),
        (-r * 0.5, r * 0.866025403784, 0.0),
        (-r * 0.5, -r * 0.866025403784, 0.0),
    ];
    let mass = vec![1.0, 1.0, 1.0];
    let acc = pp_direct_forces_newton(&pos, &mass, 100.0, 0.0, 1.0);
    
    // Par symétrie : magnitudes égales pour les 3 particules
    let m0 = (acc[0].0.powi(2) + acc[0].1.powi(2)).sqrt();
    let m1 = (acc[1].0.powi(2) + acc[1].1.powi(2)).sqrt();
    let m2 = (acc[2].0.powi(2) + acc[2].1.powi(2)).sqrt();
    
    assert!((m0 - m1).abs() < 1e-12);
    assert!((m1 - m2).abs() < 1e-12);
    
    // Chaque particule pointe vers le centre de masse (origine)
    // acc_0 = -|acc_0| · r̂_0 , donc acc_0.x < 0
    assert!(acc[0].0 < 0.0);
    assert!(acc[1].0 > 0.0);  // particule 1 vers -x ... non, vers origine donc +x si elle est en (-0.5, 0.866)
    // En réalité : particule 1 en (-0.5, 0.866), centre en (0,0), donc force vers +x, -y
    assert!(acc[1].0 > 0.0);
    assert!(acc[1].1 < 0.0);
}
```

---

## Étape 3 — Test combiné TreePM vs PP-direct

Le test principal. Setup déterministe pour reproductibilité.

```rust
#[test]
#[ignore]  // long, run with --ignored
fn test_treepm_combined_vs_pp_direct_precision() {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    // Setup : 1000 particules aléatoires dans L=100 Mpc
    let n = 1000;
    let l = 100.0_f64;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    
    let mut pos = Vec::with_capacity(n);
    let mass = vec![1.0_f64; n];
    for _ in 0..n {
        let x = rng.gen::<f64>() * l;
        let y = rng.gen::<f64>() * l;
        let z = rng.gen::<f64>() * l;
        pos.push((x, y, z));
    }
    
    let g_phys = 1.0_f64;
    let softening = 0.05_f64;  // Plummer
    
    // Référence PP-direct
    let acc_pp = pp_direct_forces_newton(&pos, &mass, l, softening, g_phys);
    
    // TreePM combiné
    let n_pm = 64;          // Δg = 100/64 = 1.5625 Mpc
    let r_s = 1.2 * (l / n_pm as f64);   // 1.875 Mpc
    let r_cut = 6.0 * (l / n_pm as f64); // 9.375 Mpc
    let theta = 0.5;
    
    let acc_treepm = compute_treepm_total_forces_cpu(
        &pos, &mass, l, n_pm,
        r_s, r_cut, theta, softening, g_phys,
    );
    
    // Comparaison particule par particule
    let mut rel_errors = Vec::with_capacity(n);
    let mut abs_errors = Vec::with_capacity(n);
    let mut angle_errors_deg = Vec::with_capacity(n);
    
    for i in 0..n {
        let (rx, ry, rz) = acc_pp[i];
        let (tx, ty, tz) = acc_treepm[i];
        
        let mag_ref = (rx*rx + ry*ry + rz*rz).sqrt();
        let dx = tx - rx;
        let dy = ty - ry;
        let dz = tz - rz;
        let mag_diff = (dx*dx + dy*dy + dz*dz).sqrt();
        
        if mag_ref > 1e-15 {
            rel_errors.push(mag_diff / mag_ref);
            abs_errors.push(mag_diff);
            
            // Angle entre les deux vecteurs
            let mag_tpm = (tx*tx + ty*ty + tz*tz).sqrt();
            if mag_tpm > 1e-15 {
                let cos_a = (rx*tx + ry*ty + rz*tz) / (mag_ref * mag_tpm);
                let cos_a = cos_a.clamp(-1.0, 1.0);
                angle_errors_deg.push(cos_a.acos().to_degrees());
            }
        }
    }
    
    // Statistiques
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
    
    // Sauvegarder le détail dans CSV pour inspection
    save_precision_data(
        "logs/treepm/precision_check_data.csv",
        &rel_errors,
        &angle_errors_deg,
    );
    
    // CRITÈRES DE VERDICT (NE PAS ASSERTER ICI — on veut voir les chiffres)
    // Verdict A (acceptable) : median < 1%, P95 < 5%, max < 20%
    // Verdict B/C (bug)      : median > 5% ou max > 50%
    // 
    // Ces seuils sont basés sur GrGadget Fig. 5 et PhotoNs Tab. 2.
    
    let verdict = if median < 0.01 && p95 < 0.05 && max_rel < 0.20 {
        "A_ACCEPTABLE"
    } else if median > 0.05 || max_rel > 0.50 {
        "B_OR_C_BUG"
    } else {
        "GREY_ZONE_NEEDS_REVIEW"
    };
    
    println!("VERDICT: {}", verdict);
    
    // Écrire le verdict dans un fichier pour le rapport final
    std::fs::write(
        "logs/treepm/precision_verdict.txt",
        format!("{}\nmedian={}\np95={}\nmax={}\nmedian_angle_deg={}\n",
                verdict, median, p95, max_rel, median_angle),
    ).unwrap();
    
    // Pas d'assert : on veut que le test passe pour pouvoir analyser
    // Le verdict est dans le fichier, pas dans assert
}

fn save_precision_data(path: &str, rel_errors: &[f64], angle_errors_deg: &[f64]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "rel_err,angle_err_deg").unwrap();
    for (re, ae) in rel_errors.iter().zip(angle_errors_deg.iter()) {
        writeln!(f, "{},{}", re, ae).unwrap();
    }
}
```

---

## Étape 4 — Tests de diagnostic complémentaires

Si verdict = `B_OR_C_BUG`, exécuter ces tests pour localiser le problème :

### 4a. Test PM-only vs TreePM combiné

Si le PM-only seul donne 54% d'erreur mais TreePM combiné < 1%, l'hypothèse A est confirmée et on continue.

```rust
#[test]
#[ignore]
fn diagnostic_pm_only_vs_combined() {
    // Même setup que test précédent (N=1000, L=100, seed=42)
    // Calculer trois forces sur les mêmes particules :
    //   acc_pp     : référence PP direct
    //   acc_pm_only: PM seul (sans Tree)
    //   acc_treepm : PM + Tree combiné
    
    // Pour chaque particule, calculer rel_err vs PP pour les deux méthodes
    // 
    // Si median(pm_only) >> median(treepm) : Hypothèse A (Tree compense le PM imprécis à courte portée)
    // Si median(pm_only) ≈ median(treepm) : bug dans le PM qui n'est pas compensé
    
    // Output : tableau comparatif imprimé + sauvegarde CSV
}
```

### 4b. Test échelle : forces vs G

Si on multiplie G par 10, les forces doivent être multipliées par 10 EXACTEMENT. Test linéarité.

```rust
#[test]
#[ignore]
fn diagnostic_force_linearity_in_g() {
    // Même IC, même setup
    // Calculer F avec G=1.0
    // Calculer F avec G=10.0
    // Vérifier F(G=10) == 10 × F(G=1) à eps machine près
    // Si non : bug d'unités
}
```

### 4c. Test échelle : forces vs masse

```rust
#[test]
#[ignore]
fn diagnostic_force_linearity_in_mass() {
    // Doubler toutes les masses → forces doublent (puisque tous les m_j sont doublés)
}
```

### 4d. Test convergence en N_pm

```rust
#[test]
#[ignore]
fn diagnostic_pm_convergence_in_n_pm() {
    // Même IC à N_pm = 32, 64, 128, 256
    // Pour chaque N_pm, mesurer median rel_err vs PP-direct
    // Si l'erreur DÉCROÎT vers 0 : bug à corriger en augmentant la résolution
    // Si l'erreur PLATEAU à valeur non-nulle : bug systématique d'unités/normalisation
}
```

---

## Étape 5 — Rapport final

Créer `logs/treepm/precision_check_report.md` avec :

```markdown
# Rapport vérification précision TreePM CPU

**Date** : <DATE>
**Commit** : <HASH>
**Setup** : N=1000, L=100 Mpc, seed=42, N_pm=64, r_s=1.2·Δg, r_cut=6·Δg, θ=0.5

## Résultats principaux

| Métrique | Valeur |
|---|---|
| Median rel err | <X>% |
| P95 rel err | <X>% |
| Max rel err | <X>% |
| Median angle err | <X>° |

## Verdict : <A | B | C | grey>

[explication 2-3 lignes]

## Diagnostics (si verdict B ou C)

[résultats des tests 4a-4d]

## Recommandation

- Si A : GO pour port GPU Phase 5, plan d'origine valide
- Si B : bug de normalisation localisé en <module>, fix à appliquer avant port GPU
- Si C : bug de placement CIC dans le pipeline, à investiguer
- Si grey : analyse manuelle nécessaire avant décision
```

---

## Critères d'arrêt automatique

CLI s'arrête et demande l'humain UNIQUEMENT si :

1. **Pipeline CPU intégré inexistant et infaisable à créer en < 4h** (auquel cas, documenter l'obstacle et stopper)
2. **PP-direct sur N=1000 prend > 30 minutes** (problème de performance, indique un bug d'implémentation)
3. **Tests de validation PP-direct (test_pp_two_particles_attract et test_pp_three_particles_symmetric) échouent** (bug dans la référence elle-même)

Tout autre cas : continuer en autonomie, rapporter dans le rapport final.

---

## Estimation de durée

| Étape | Durée estimée |
|---|---|
| 1. Localiser/créer pipeline CPU | 1-2h |
| 2. PP-direct + ses tests | 30 min |
| 3. Test combiné principal | 1h |
| 4. Tests de diagnostic | 1h (seulement si verdict B/C) |
| 5. Rapport | 30 min |
| **TOTAL** | **3-5h** |

---

## Référence physique pour les seuils

- **GrGadget Fig. 5** (Quintana-Miranda 2023) : TreePM avec corrections gradient ord 4 + CIC + Laplacien continu donne médiane < 0.5%, max < 3% sur tests à particule unique
- **PhotoNs Tab. 2** (Wang & Meng 2021) : speedup de la table ne dégrade pas la précision en-dessous de 0.1% médiane
- **Springel 2005 (GADGET-2) Fig. 6** : TreePM standard donne médiane ~0.5%, max ~5% pour θ=0.5

Notre cible **median < 1%, P95 < 5%** est donc cohérente avec l'état de l'art en cosmologie.

---

**Fin du prompt. Bonne route, CLI.**
