# PLAN D'ACTION CLI — TreePM Janus haute performance

**Mission** : Connecter le module TreePM existant (`src/treepm/`) au pipeline Janus, l'optimiser à l'état de l'art GPU (techniques PhotoNs-GPU 2021), valider via une suite de tests Rust autonome, et arriver à un **run de validation 1M particules de z=10 à z=0** qui démontre la cohérence avec les résultats Barnes-Hut précédents.

**Mode de fonctionnement** : CLI travaille en autonomie totale. Pas de questions humaines sauf si **problème algorithmique grave** détecté (cf. §0.5 critères d'arrêt).

**Cible matérielle** : RTX 3060 12 GB sur Linux Mint (machine `ajp@inspiron`).

**Cible scientifique** : reproduire à minimum les résultats Barnes-Hut établis :
- Corr(δ⁺, δ⁻) ≈ −0.07
- P_×(k) < 0 sur toute la gamme k résolue
- Diff/Pois ≈ 3 (scale-invariance)
- σ8 ≈ 0.70
- t₀ ≈ 15.87 Gyr
- Run stable jusqu'à z=0 avec μ=19, η=1.045, dynamic VSL

---

## 0. RÈGLES D'OPÉRATION CLI

### 0.1 Discipline de test obligatoire

**Aucune sous-tâche ne valide sans test Rust passant en `cargo test --release`.**

Pattern obligatoire pour chaque sous-tâche :
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_<feature>_<property>() {
        // Arrange
        let input = ...;
        let tol = 1e-10; // tolérance JUSTIFIÉE par commentaire
        
        // Act
        let output = function_under_test(input);
        
        // Assert
        assert!((output - expected).abs() < tol,
                "Expected {}, got {}, diff {}", expected, output, (output-expected).abs());
    }
}
```

**Règles strictes** :
- JAMAIS `assert_eq!` sur des floats — toujours `(a - b).abs() < tol`
- `tol` doit être justifié par un commentaire (ordre du schéma, machine epsilon, etc.)
- Tests en mode release pour mesurer la vraie performance
- Tests CUDA dans `#[cfg(feature = "cuda")]` séparés des tests CPU

### 0.2 Logging et traçabilité

Structure des logs créée par CLI :
```
logs/treepm/
├── phase1_audit_<DATE>.log
├── phase2_corrections_<DATE>.log
├── phase3_gpu_optims_<DATE>.log
├── phase4_parameters_<DATE>.log
├── phase5_janus_<DATE>.log
├── phase6_validation_newton_<DATE>.log
├── phase7_validation_janus_<DATE>.log
├── phase8_benchmarks_<DATE>.log
├── phase9_decision.md
└── benchmarks/
    ├── kernel_p2p_<N>_<config>.json
    ├── pm_solver_<N_pm>.json
    └── full_step_<N>.json
```

Chaque log contient :
- Timestamp et commande exacte
- Sortie complète (`cargo test --release -- --nocapture`)
- Métriques mesurées avec valeurs cibles
- Décision : VALIDÉ / À REFAIRE / BLOQUÉ

### 0.3 Commits Git atomiques

Un commit par sous-étape validée :
```
treepm-phase<N>: <étape> — <résumé 1 ligne>

- <bullet 1>
- <bullet 2>

Tests: <passed>/<total>
Benchmark: <metric> = <value> (target: <target>)
```

Branche dédiée : `feature/treepm-integration`. Merge sur `main` uniquement après phase 9 décision GO.

### 0.4 Gestion de l'incertitude

Si CLI rencontre **un choix d'implémentation non spécifié** dans ce plan :
1. Choisir la convention la plus conservative (DP > SP, ordre élevé > bas, plus de tests > moins)
2. Documenter le choix dans le log de la phase
3. Continuer

Si CLI rencontre **une ambiguïté entre sources** (ex. PhotoNs dit `r_s = 1.2·Δg`, Bagla dit `1.25`) :
1. Suivre PhotoNs-GPU par défaut (ce plan est calibré PhotoNs)
2. Documenter
3. Ajouter un test paramétrique qui balaie les deux valeurs

### 0.5 Critères d'arrêt automatique

CLI s'arrête et **demande à l'humain** UNIQUEMENT si :

1. **Erreur de force > 5%** dans le test particule unique (§6.1) après 3 tentatives de fix
2. **Speedup GPU < 2×** par rapport au CPU dans le benchmark P2P (§3.6) — indique problème majeur
3. **P(k) Newton TreePM diverge de >5%** du tree Newton seul à mêmes paramètres (§6.2)
4. **Mini-run Janus 100K crashe** ou produit Corr(δ⁺,δ⁻) > 0 (§7.2) — bug physique
5. **Violation de conservation d'énergie > 10%** sur 1000 steps (§7.3)
6. **OOM CUDA** détecté en dehors des seuils prévus du §3.5

Tout autre incident (test isolé qui échoue, warning compilateur, performance sous cible mais > seuil critique) : CLI tente 3 fixes automatiques puis documente et passe à la phase suivante avec le statut **FLAG**.

### 0.6 Format de sortie pour humain

À la fin du plan, CLI produit `phase9_decision.md` avec :
- Résultats numériques de chaque phase (tableau)
- Liste des FLAGS ouverts
- Verdict : GO / NO-GO / GO-AVEC-RÉSERVES pour le run 1M
- Si GO : commande exacte à lancer, ETA, monitoring conseillé
- Si NO-GO : liste des bloqueurs et actions de remédiation

---

## 1. PHASE 1 — AUDIT PRÉ-VOL (read-only, 1h)

### 1.1 Inventaire du module TreePM existant

```bash
cd ~/janus-sim  # ou path projet
git checkout -b feature/treepm-integration

# Inventaire fichiers
echo "=== Module treepm ===" > logs/treepm/phase1_audit_$(date +%F).log
find src/treepm -type f \( -name "*.rs" -o -name "*.cu" \) \
    -exec echo "--- {} ---" \; -exec wc -l {} \; \
    >> logs/treepm/phase1_audit_$(date +%F).log

# Inventaire API publique
echo "=== API publique ===" >> logs/treepm/phase1_audit_$(date +%F).log
grep -rn "pub fn\|pub struct\|pub enum" src/treepm/ \
    >> logs/treepm/phase1_audit_$(date +%F).log

# TODO/FIXME/HACK
echo "=== Dette technique ===" >> logs/treepm/phase1_audit_$(date +%F).log
grep -rn "TODO\|FIXME\|HACK\|XXX" src/treepm/ \
    >> logs/treepm/phase1_audit_$(date +%F).log
```

**Livrable** : `audit/01_treepm_inventory.md`
- Tableau fichier × LOC × rôle
- Liste des fonctions/structs publiques avec signature
- Liste des TODO/FIXME ouverts

### 1.2 Cartographie du flux de force actuel

CLI doit identifier dans le code :
- Où est calculée la force actuellement (Barnes-Hut pur dans `src/nbody/...`)
- Comment elle s'intègre dans le pas DKD (`src/integrator/...` ou équivalent)
- Format des données d'entrée/sortie pour la force
- Allocations CUDA actuelles (kernels, transferts H↔D)
- Convention vélocité utilisée (`v_pec = a·ẋ_co` ou autre)

**Livrable** : `audit/02_force_pipeline.md` avec :
- Diagramme ASCII du flux actuel
- Liste des fonctions point d'entrée
- Convention vélocité documentée (CRITIQUE pour cohérence Janus)

### 1.3 Audit des 3 corrections numériques GrGadget

Vérifier la présence et la correction des 3 points dans le module TreePM existant :

**(a) Ordre du gradient** (GrGadget Eq. 19 vs 20)
```bash
grep -rn "gradient\|finite_diff\|FINITE_DIFF\|grad" src/treepm/
```

Détection :
- Pattern `(phi[i+1] - phi[i]) / h` → **ordre 1** (À UPGRADER)
- Pattern `(phi[i+1] - phi[i-1]) / (2*h)` → **ordre 2** (À UPGRADER)
- Pattern `(8*(phi[i+1]-phi[i-1]) - (phi[i+2]-phi[i-2])) / (12*h)` → **ordre 4** (OK)

**(b) Correction CIC en Fourier** (Sefusatti et al. 2016)
```bash
grep -rn "CIC\|cic\|sinc\|cic_correction\|window" src/treepm/
```

Détection : présence/absence de la fonction `W_CIC(k) = sinc²(πk/N)` appliquée à la déconvolution.

**(c) Laplacien discret** (GrGadget Eq. 21 vs 22)
```bash
grep -rn "laplacian\|Laplacian\|k2\|k_squared\|kernel_poisson" src/treepm/
```

Détection :
- `sin²(πk/N)` → **ancienne forme Gevolution** (À CORRIGER)
- `k_x² + k_y² + k_z²` × `(2π/L)²` → **forme continue** (OK)

**Livrable** : `audit/03_grgadget_corrections.md` avec statut PRÉSENT/ABSENT/À-CORRIGER pour chaque point.

### 1.4 Inventaire CUDA et capabilities GPU

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvcc --version
cat /proc/driver/nvidia/version

# Vérifier shared memory limit (crucial pour optim PhotoNs)
nvidia-smi --query-gpu=name --format=csv,noheader
```

RTX 3060 attendu :
- **Compute capability 8.6** (Ampere)
- **VRAM : 12 GB** (11 GB utilisable après safety margin)
- **Shared memory par SM : 100 KB** (configurable jusqu'à 100KB sur Ampere, vs 48 KB defaults)
- **Registers par thread : 255 max**
- **L1 cache : 128 KB par SM**
- **28 SMs × 128 cores = 3584 CUDA cores**
- **TFlops SP : ~13** (réel ~10-11 en charge)

**Livrable** : `audit/04_gpu_capabilities.md`

### 1.5 Critère de fin Phase 1

Phase 1 est validée si :
- ✅ Inventaire complet livré
- ✅ Statut des 3 corrections GrGadget connu
- ✅ Convention vélocité du code Janus documentée
- ✅ Capacités GPU confirmées

Aucun code modifié pendant la Phase 1. Commit : `treepm-phase1: audit complet — bilan dans audit/`

---

## 2. PHASE 2 — CORRECTIONS NUMÉRIQUES (jour 1, 4-6h)

Cette phase corrige les 3 points GrGadget identifiés au §1.3, AVANT toute optimisation GPU. C'est la fondation correcte.

### 2.1 Gradient ordre 4

**Implémentation** dans `src/treepm/gradient.rs` :

```rust
/// 4th order central finite difference for d/dx on a regular periodic grid.
/// 
/// Reference: GrGadget Eq. (20), originally from Springel 2005.
/// Error: O(h^4)
/// 
/// # Arguments
/// * `field` - 3D field stored in row-major order [i*n*n + j*n + k]
/// * `(i, j, k)` - cell index where to evaluate
/// * `n` - number of cells per dimension
/// * `h` - cell size
#[inline(always)]
pub fn grad4_x(field: &[f64], i: usize, j: usize, k: usize, 
                n: usize, h: f64) -> f64 {
    let im2 = (i + n - 2) % n;
    let im1 = (i + n - 1) % n;
    let ip1 = (i + 1) % n;
    let ip2 = (i + 2) % n;
    let idx = |a: usize, b: usize, c: usize| -> usize { a * n * n + b * n + c };
    
    (8.0 * (field[idx(ip1,j,k)] - field[idx(im1,j,k)])
     - (field[idx(ip2,j,k)] - field[idx(im2,j,k)])) / (12.0 * h)
}

// Idem pour grad4_y et grad4_z, symétrie évidente
```

**Tests obligatoires** dans `src/treepm/gradient.rs` :

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    
    #[test]
    fn test_grad4_constant_field_zero() {
        // Constant field → gradient = 0 partout (à la machine epsilon près)
        let n = 32;
        let field = vec![3.14159_f64; n*n*n];
        let mut max_err = 0.0_f64;
        for i in 0..n { for j in 0..n { for k in 0..n {
            let g = grad4_x(&field, i, j, k, n, 1.0);
            max_err = max_err.max(g.abs());
        }}}
        // Tolerance: machine epsilon × magnitude × n_ops
        assert!(max_err < 1e-13, "max_err = {}", max_err);
    }
    
    #[test]
    fn test_grad4_linear_field_constant() {
        // Field f(x) = x → gradient = 1 (au centre, loin des bords PBC)
        let n = 64;
        let l = 1.0_f64;
        let h = l / n as f64;
        let mut field = vec![0.0; n*n*n];
        for i in 0..n { for j in 0..n { for k in 0..n {
            field[i*n*n + j*n + k] = i as f64 * h;
        }}}
        // Centre du domaine, à au moins 4 cellules des bords PBC
        let g = grad4_x(&field, n/2, n/2, n/2, n, h);
        assert!((g - 1.0).abs() < 1e-12, "Expected 1.0, got {}", g);
    }
    
    #[test]
    fn test_grad4_sine_convergence_order4() {
        // f(x) = sin(2πx/L) → ∂f/∂x = (2π/L)·cos(2πx/L)
        // Erreur doit décroître en h^4
        let l = 1.0_f64;
        let mut errors = Vec::new();
        let resolutions = [32_usize, 64, 128];
        
        for &n in &resolutions {
            let h = l / n as f64;
            let mut field = vec![0.0; n*n*n];
            for i in 0..n { for j in 0..n { for k in 0..n {
                let x = i as f64 * h;
                field[i*n*n + j*n + k] = (2.0 * PI * x / l).sin();
            }}}
            
            let mut max_err = 0.0_f64;
            // Évite les bords PBC pour test propre
            for i in 4..n-4 {
                let x = i as f64 * h;
                let exact = (2.0 * PI / l) * (2.0 * PI * x / l).cos();
                let g = grad4_x(&field, i, n/2, n/2, n, h);
                max_err = max_err.max((g - exact).abs());
            }
            errors.push(max_err);
        }
        
        // Vérifier convergence O(h^4) : err(2h)/err(h) ≈ 16
        let ratio_32_64 = errors[0] / errors[1];
        let ratio_64_128 = errors[1] / errors[2];
        
        // Tolérance : on doit voir au moins 12× (proche de 16)
        assert!(ratio_32_64 > 12.0 && ratio_32_64 < 20.0,
                "Convergence n=32→64 ratio = {} (expected ~16)", ratio_32_64);
        assert!(ratio_64_128 > 12.0 && ratio_64_128 < 20.0,
                "Convergence n=64→128 ratio = {} (expected ~16)", ratio_64_128);
    }
    
    #[test]
    fn test_grad4_periodic_consistency() {
        // f périodique : gradient au bord doit être cohérent avec PBC
        let n = 32;
        let l = 1.0_f64;
        let h = l / n as f64;
        let mut field = vec![0.0; n*n*n];
        for i in 0..n { for j in 0..n { for k in 0..n {
            let x = i as f64 * h;
            field[i*n*n + j*n + k] = (2.0 * PI * x / l).cos();
        }}}
        // Gradient en i=0 (bord) doit être bien défini grâce au modulo
        let g0 = grad4_x(&field, 0, n/2, n/2, n, h);
        let g_n = grad4_x(&field, n-1, n/2, n/2, n, h);
        // Par symétrie pour cos : g(0) ≈ 0, g(L-h) ≈ 0
        assert!(g0.abs() < 1e-2, "g(0) = {}", g0);
        assert!(g_n.abs() < 1e-2, "g(L-h) = {}", g_n);
    }
}
```

**Critère de validation §2.1** : 4/4 tests passent en `cargo test --release`. Convergence O(h⁴) confirmée numériquement.

### 2.2 Correction CIC en Fourier

Le déposit CIC convolue la densité avec un kernel triangle. En Fourier ça équivaut à multiplier par `W_CIC(k) = sinc²(πk/N)` par dimension. Pour récupérer la vraie densité spectrale, il faut **déconvoluer** par ce facteur.

Application **DEUX FOIS** : une à l'aller (sur ρ̂), une au retour (sur Φ̂ avant interpolation de la force).

**Implémentation** dans `src/treepm/cic_correction.rs` :

```rust
/// CIC window function inverse, applied per-dimension.
/// 
/// Reference: Sefusatti et al. 2016 (MNRAS 460, 3624), 
///            GrGadget §3.3.1
/// 
/// # Arguments
/// * `(kx, ky, kz)` - integer Fourier mode indices in [-N/2, N/2)
/// * `n` - grid size per dimension
/// 
/// # Returns
/// 1 / W_CIC(k)² where W_CIC(k) = sinc(πk/N)
#[inline(always)]
pub fn cic_window_inv_squared(kx: i32, ky: i32, kz: i32, n: usize) -> f64 {
    let n_f = n as f64;
    
    let inv_sinc = |k: i32| -> f64 {
        if k == 0 {
            1.0
        } else {
            let arg = std::f64::consts::PI * (k as f64) / n_f;
            arg / arg.sin()
        }
    };
    
    let w = inv_sinc(kx) * inv_sinc(ky) * inv_sinc(kz);
    w * w  // CIC = 2nd order, donc deux facteurs sinc⁻¹
}
```

**Tests obligatoires** :

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cic_window_at_zero() {
        // k=0 mode : pas de correction
        let n = 64;
        let w = cic_window_inv_squared(0, 0, 0, n);
        assert!((w - 1.0).abs() < 1e-15, "Got {}", w);
    }
    
    #[test]
    fn test_cic_window_at_nyquist() {
        // À k = N/2, sinc(π/2) = 2/π, donc inv_sinc = π/2
        // w = (π/2)^6 (3 dimensions × 2 facteurs)
        let n = 64;
        let w = cic_window_inv_squared((n/2) as i32, 0, 0, n);
        let expected = (std::f64::consts::PI / 2.0).powi(2); // 1D Nyquist²
        // Ici seulement kx Nyquist, ky=kz=0
        assert!((w - expected).abs() < 1e-13, 
                "Expected {}, got {}", expected, w);
    }
    
    #[test]
    fn test_cic_roundtrip_density() {
        // Test d'intégration : déposer une distribution sinusoïdale
        // de fréquence connue, FFT, déconvoluer, IFFT, comparer.
        // Voir tests/integration/test_cic_roundtrip.rs (avec rustfft)
        // L'erreur relative doit être < 1% sauf exactement à Nyquist
        // [Test intégration livré séparément en §6]
    }
}
```

### 2.3 Laplacien forme continue

```rust
/// Continuous-form discrete Laplacian factor for periodic Poisson solver.
/// Reference: GrGadget Eq. (22).
/// 
/// Returns the factor to divide ρ̂ by (with sign) to obtain Φ̂:
/// Φ̂(k) = -ρ̂(k) / k² × (4πG)
/// 
/// # Arguments
/// * `(kx, ky, kz)` - integer Fourier mode indices
/// * `l` - box size (physical units)
/// 
/// # Returns
/// k² in physical units. Returns 1.0 for k=0 (zero mode handled separately).
#[inline(always)]
pub fn k_squared_continuous(kx: i32, ky: i32, kz: i32, l: f64) -> f64 {
    if kx == 0 && ky == 0 && kz == 0 {
        return 1.0;  // sentinel: zero mode set to 0 in caller
    }
    let two_pi_over_l = 2.0 * std::f64::consts::PI / l;
    let k2 = (kx*kx + ky*ky + kz*kz) as f64;
    k2 * two_pi_over_l * two_pi_over_l
}
```

**Tests** :

```rust
#[test]
fn test_k2_zero_mode_sentinel() {
    let k2 = k_squared_continuous(0, 0, 0, 100.0);
    assert!(k2 == 1.0); // sentinel
}

#[test]
fn test_k2_known_modes() {
    let l = 100.0_f64;
    // Mode (1,0,0) : k² = (2π/L)²
    let k2 = k_squared_continuous(1, 0, 0, l);
    let expected = (2.0 * std::f64::consts::PI / l).powi(2);
    assert!((k2 - expected).abs() / expected < 1e-14);
    
    // Mode (3,4,12) : k² = (2π/L)² × (9+16+144) = (2π/L)² × 169
    let k2 = k_squared_continuous(3, 4, 12, l);
    let expected = (2.0 * std::f64::consts::PI / l).powi(2) * 169.0;
    assert!((k2 - expected).abs() / expected < 1e-14);
}
```

### 2.4 Solveur de Poisson assemblé

Une fois 2.1, 2.2, 2.3 validés indépendamment, assembler le solveur :

```rust
/// Solve ∇²Φ = 4πGρ on a periodic grid using FFT.
/// 
/// Pipeline:
/// 1. CIC correction on density: ρ̂ /= W_CIC²
/// 2. Poisson solver: Φ̂ = -4πG·ρ̂ / k²  (k² continuous form)
/// 3. CIC correction on potential: Φ̂ /= W_CIC²
/// 4. Inverse FFT
pub fn solve_poisson_corrected(
    rho: &mut [f32],  // input/output: density → potential
    n: usize,
    l: f64,
    g: f64,
) -> Result<(), PoissonError> {
    // Implémentation utilisant cuFFT en mode in-place complex-to-real
    // [Détails à compléter par CLI selon stack rustfft / cufft existante]
    todo!()
}
```

**Test d'intégration §2.4** : Test analytique avec source connue.

```rust
#[test]
fn test_poisson_point_mass() {
    // Test : déposer une masse ponctuelle au centre, résoudre Poisson,
    // comparer Φ(r) avec -GM/r à différentes distances.
    let n = 128;
    let l = 100.0_f64;
    let m = 1.0_f64;
    let g = 1.0_f64;
    
    let mut rho = vec![0.0_f32; n*n*n];
    let cell_size = l / n as f64;
    deposit_cic(&mut rho, n, l, &[(l/2.0, l/2.0, l/2.0, m as f32)]);
    
    solve_poisson_corrected(&mut rho, n, l, g).unwrap();
    
    // Test 1 : à r = L/4 (proche), tolérance plus large à cause du CIC + PBC
    let r_close = l / 4.0;
    let i_close = n * 3 / 4;
    let phi_close = rho[i_close * n * n + (n/2) * n + (n/2)] as f64;
    let phi_exact_close = -g * m / r_close;
    let rel_err_close = (phi_close - phi_exact_close).abs() / phi_exact_close.abs();
    assert!(rel_err_close < 0.05, 
            "Close: rel_err = {} ({})", rel_err_close, phi_close);
    
    // Test 2 : à r = 3L/8 (loin), tolérance plus stricte
    // Les effets PBC dominent ici (image de la masse à L/2 de l'autre côté)
    // mais l'erreur du PM seul sans PBC self-interaction est < 2%
    // Sur ce test, on vérifie que le rapport entre 2 distances obéit à 1/r
    let r_mid = l * 3.0 / 8.0;
    let i_mid = n * 7 / 8;
    let phi_mid = rho[i_mid * n * n + (n/2) * n + (n/2)] as f64;
    
    // Test du rapport phi_close/phi_mid ≈ r_mid/r_close
    // (élimine les effets PBC qui sont approximativement constants)
    let ratio_measured = phi_close / phi_mid;
    let ratio_expected = r_mid / r_close;
    let rel_err_ratio = ((ratio_measured - ratio_expected) / ratio_expected).abs();
    assert!(rel_err_ratio < 0.02,
            "Ratio test: measured {} vs expected {}, err {}",
            ratio_measured, ratio_expected, rel_err_ratio);
}
```

### 2.5 Critère de fin Phase 2

Phase 2 validée si :
- ✅ Tous les tests unitaires (gradient, CIC, Laplacien) passent
- ✅ Test d'intégration `test_poisson_point_mass` passe avec err < 5%
- ✅ Convergence O(h⁴) du gradient confirmée numériquement

**Commit** : `treepm-phase2: corrections numériques (gradient ord 4, CIC, Laplacien)`

---

## 3. PHASE 3 — OPTIMISATIONS GPU CRITIQUES (jour 1-2, 6-8h)

C'est le cœur de la performance. Les 7 optimisations PhotoNs-GPU appliquées dans l'ordre. Chaque optim est validée séparément par benchmark avant de passer à la suivante.

### 3.0 Politique de précision SP/DP par module

Décision de précision arrêtée pour ce projet, conformément aux benchmarks PhotoNs-GPU §3 et à l'expérience GADGET-4 :

| Module | Précision | Justification |
|---|---|---|
| **Kernel P2P** (sh_pos_j, dx, r2, idr, acc local) | **SP (float32)** | RTX 3060 SP = 13 TFlops vs DP = 0.2 TFlops (×65). PhotoNs montre que SP suffit avec table T[4] à err < 1e-6 |
| **Table truncation T(x), E(x)** | **SP** | Interpolation Taylor 4 termes sur 512 points → précision SP saturée |
| **Tree COM, multipoles, opening criterion** | **DP (float64)** | Accumulation sur ~10⁵ leaves : DP évite drift d'erreur sur chaîne de sommes |
| **PM solver (FFT, ρ, Φ)** | **SP** (DP optionnel) | cuFFT SP largement suffisant pour TreePM, ×2 mémoire si DP. `feature = "pm-double"` pour activer DP |
| **Drift/Kick global** | **DP** sur position_co (cosmological coords), **SP** sur vel | Position cumule erreurs sur 10⁴ steps ; vélocité reset partiellement à chaque kick |
| **Diagnostics (P(k), Corr, σ8)** | **DP** | Statistiques globales : DP obligatoire pour stabilité numérique |

**Implémentation Rust** : utiliser `f32` partout par défaut, et `f64` aux interfaces critiques (positions cosmologiques, accumulation tree, FFT post-processing).

```rust
pub struct ParticleArrays {
    // Positions en DP — accumulation cosmologique
    pub pos_x: Vec<f64>,
    pub pos_y: Vec<f64>,
    pub pos_z: Vec<f64>,
    
    // Vélocités en SP — reset partiel à chaque kick
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub vel_z: Vec<f32>,
    
    // Accélération en SP — output du kernel GPU
    pub acc_x: Vec<f32>,
    pub acc_y: Vec<f32>,
    pub acc_z: Vec<f32>,
    
    // Masse |m| en SP — toujours positive
    pub mass: Vec<f32>,
    
    // Signe Janus en i8 — économise mémoire et permet sign_factor rapide
    pub sign: Vec<i8>,
}
```

**Tests obligatoires** :
- À chaque transfert SP↔DP, ajouter une assertion de cohérence (perte < epsilon machine acceptée)
- Test dédié de roundtrip DP→SP→DP sur les positions

```rust
#[test]
fn test_position_dp_to_sp_roundtrip_loss() {
    // Positions cosmologiques typiques : x ∈ [0, 1000] Mpc/h
    let positions_dp: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
    let positions_sp: Vec<f32> = positions_dp.iter().map(|&x| x as f32).collect();
    let positions_back: Vec<f64> = positions_sp.iter().map(|&x| x as f64).collect();
    
    // Erreur maximale doit rester < 1e-4 Mpc/h sur la gamme typique
    // (epsilon SP ≈ 1.2e-7 × magnitude ≈ 1.2e-4 sur x=1000)
    let max_err = positions_dp.iter().zip(positions_back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "DP→SP→DP loss = {} Mpc/h", max_err);
}
```

### 3.1 Layout SoA pour les positions GPU

**Principe** : sur GPU, les threads d'un warp accèdent à la mémoire de manière coalescée seulement si les données sont contigües. Stocker les particules en `Structure of Arrays` au lieu de `Array of Structures`.

**Avant** (AoS, mauvais pour GPU) :
```rust
struct Particle { pos: [f32; 3], vel: [f32; 3], ... }
let particles: Vec<Particle> = ...;
```

**Après** (SoA, bon pour GPU) :
```rust
struct ParticleArrays {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,
    // ...
}
```

**Implémentation** : module `src/treepm/gpu_layout.rs`. La transformation AoS↔SoA se fait sur CPU **après** construction de l'arbre (les particules sont déjà réordonnées par Morton/Hilbert dans l'arbre, donc la transformation est triviale).

**Tests** :

```rust
#[test]
fn test_aos_to_soa_roundtrip() {
    let n = 1000;
    let aos: Vec<[f32; 3]> = (0..n).map(|i| {
        [i as f32, (i*2) as f32, (i*3) as f32]
    }).collect();
    
    let soa = aos_to_soa(&aos);
    assert_eq!(soa.x.len(), n);
    
    let back = soa_to_aos(&soa);
    for i in 0..n {
        assert_eq!(aos[i], back[i]);
    }
}

#[test]
fn test_soa_layout_alignment() {
    let n = 1000;
    let soa = ParticleArrays::new(n);
    // Vérifier alignement 32-byte (taille warp) pour CUDA coalescing
    assert_eq!(soa.pos_x.as_ptr() as usize % 32, 0);
    assert_eq!(soa.pos_y.as_ptr() as usize % 32, 0);
    assert_eq!(soa.pos_z.as_ptr() as usize % 32, 0);
}
```

**Benchmark §3.1** : Mesurer le temps de copie 1M particules H→D dans les deux layouts.
- Cible AoS : ~5 ms
- Cible SoA : ~3 ms (gain ~40% mesuré dans PhotoNs-GPU)

### 3.2 Construction d'une task list ordonnée (TaskGroup)

**Principe PhotoNs-GPU** : après construction de l'arbre, on dérive une liste de paires `<i_leaf, j_leaf>` de leaves qui doivent interagir en P2P. Cette liste est ensuite **réordonnée** par `i_leaf` pour que toutes les tâches concernant le même `i_leaf` soient contiguës. On crée un index secondaire `GroupIdx` qui donne `(start, length)` pour chaque groupe.

**Avantage** : un block CUDA traite un groupe entier (toutes les paires `<i_fixed, j_*>`), ce qui maximise la réutilisation des données `i` en registres et permet le pre-fetch des `j` en shared memory.

**Implémentation** dans `src/treepm/task_list.rs` :

```rust
/// Pair of leaves that must interact in P2P.
#[derive(Clone, Copy, Debug)]
pub struct P2PTask {
    pub i_leaf: u32,
    pub j_leaf: u32,
}

/// Indexing structure pointing into a sorted task list.
/// For each `i_leaf` value, gives the (start, length) of the contiguous
/// range of tasks in the sorted list.
pub struct GroupIdx {
    pub i_leaves: Vec<u32>,        // unique i_leaf values (sorted)
    pub starts: Vec<u32>,          // start offset in tasks array
    pub lengths: Vec<u32>,         // number of j_leaves for this i_leaf
}

/// Build P2P task list from tree walk, then sort by i_leaf.
pub fn build_task_list(tree: &Tree, r_cut: f64) -> (Vec<P2PTask>, GroupIdx) {
    let mut tasks = Vec::with_capacity(tree.n_leaves * 50);  // estim
    
    // Pour chaque leaf, trouver les leaves voisines à distance < r_cut
    for i_leaf in 0..tree.n_leaves {
        let neighbors = tree.find_neighbor_leaves(i_leaf as u32, r_cut);
        for j_leaf in neighbors {
            tasks.push(P2PTask { i_leaf: i_leaf as u32, j_leaf });
        }
    }
    
    // Tri par i_leaf (utiliser thrust côté GPU ou rayon::par_sort côté CPU)
    tasks.sort_by_key(|t| t.i_leaf);
    
    // Construction GroupIdx
    let group_idx = build_group_idx(&tasks);
    
    (tasks, group_idx)
}
```

**Tests** :

```rust
#[test]
fn test_task_list_sorted_by_i() {
    let tree = mock_tree_4leaves();
    let (tasks, _) = build_task_list(&tree, 10.0);
    
    for w in tasks.windows(2) {
        assert!(w[0].i_leaf <= w[1].i_leaf, "Tasks not sorted by i_leaf");
    }
}

#[test]
fn test_group_idx_consistency() {
    let tree = mock_tree_4leaves();
    let (tasks, idx) = build_task_list(&tree, 10.0);
    
    for k in 0..idx.i_leaves.len() {
        let start = idx.starts[k] as usize;
        let len = idx.lengths[k] as usize;
        let i_expected = idx.i_leaves[k];
        for offset in 0..len {
            assert_eq!(tasks[start + offset].i_leaf, i_expected);
        }
    }
}

#[test]
fn test_task_list_completeness() {
    // Toutes les paires <i,j> à r < r_cut doivent être dans la liste
    let n_particles = 1000;
    let particles = mock_random_particles(n_particles, 100.0);
    let tree = build_tree(&particles);
    let r_cut = 5.0;
    
    let (tasks, _) = build_task_list(&tree, r_cut);
    
    // Vérification brute force : compter les vrais voisins
    let mut expected = 0;
    for i in 0..tree.n_leaves {
        for j in 0..tree.n_leaves {
            if tree.leaf_distance(i, j) < r_cut as f64 {
                expected += 1;
            }
        }
    }
    
    // Tolerance: les leaves voisines exactes doivent matcher
    assert!((tasks.len() as i64 - expected as i64).abs() < expected as i64 / 100,
            "Got {} tasks, expected ~{}", tasks.len(), expected);
}
```

**Benchmark §3.2** : Construction task list pour 1M particules.
- Cible : < 100 ms

### 3.3 Kernel P2P optimisé (block-per-group, thread-per-particle)

C'est le kernel central. Voici la structure CUDA cible :

```cuda
// Kernel CUDA central : P2P pour TreePM Janus
//
// Pattern: 1 block CUDA = 1 task group (= 1 i_leaf donné)
//          1 thread = 1 particule dans i_leaf
//          j_leaves prefetchées en shared memory par chunks
//
// Reference: PhotoNs-GPU §3.2 (Wang & Meng 2021)
__global__ void p2p_kernel_janus(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int8_t* __restrict__ sign,    // +1 for m+, -1 for m-
    const float* __restrict__ mass,
    float* acc_x,
    float* acc_y,
    float* acc_z,
    const Leaf* leaves,
    const P2PTask* task_group,    // sorted, GroupIdx-indexed
    const GroupIdx group_idx,
    int n_groups,
    float g_const,
    float r_cut,
    float r_s,
    const float* __restrict__ erfc_table,  // cf. §3.4
    const float* __restrict__ exp_table,
    int table_size
) {
    extern __shared__ float sh_pos_j[];  // [3 × MAX_LEAF_SIZE]
    
    for (int b = blockIdx.x; b < n_groups; b += gridDim.x) {
        int i_leaf = group_idx.i_leaves[b];
        Leaf li = leaves[i_leaf];
        int t = threadIdx.x;
        
        // Charger position et propriétés de la particule i en registres
        float3 pos_i;
        int8_t sign_i;
        float mass_i;
        if (t < li.np) {
            int gi = li.start + t;
            pos_i.x = pos_x[gi];
            pos_i.y = pos_y[gi];
            pos_i.z = pos_z[gi];
            sign_i = sign[gi];
            mass_i = mass[gi];
        }
        
        float3 acc = {0.0f, 0.0f, 0.0f};
        
        int j_start = group_idx.starts[b];
        int j_end = j_start + group_idx.lengths[b];
        
        for (int task_idx = j_start; task_idx < j_end; task_idx++) {
            int j_leaf = task_group[task_idx].j_leaf;
            Leaf lj = leaves[j_leaf];
            
            __syncthreads();
            // Prefetch positions de j_leaf en shared memory
            if (t < lj.np) {
                int gj = lj.start + t;
                sh_pos_j[3*t]   = pos_x[gj];
                sh_pos_j[3*t+1] = pos_y[gj];
                sh_pos_j[3*t+2] = pos_z[gj];
                // sign et mass aussi
                // [stockage dans tableaux shared additionnels]
            }
            __syncthreads();
            
            // Calcul P2P
            if (t < li.np) {
                for (int j = 0; j < lj.np; j++) {
                    float dx = sh_pos_j[3*j]   - pos_i.x;
                    float dy = sh_pos_j[3*j+1] - pos_i.y;
                    float dz = sh_pos_j[3*j+2] - pos_i.z;
                    
                    // Minimum image convention (sécurité)
                    // Pour task list correctement construite avec r_cut < L/4,
                    // ce wrap est presque toujours un no-op, mais le coût est
                    // faible (3 fma + 3 round) et il garantit la correctness
                    // quelle que soit la valeur de r_cut.
                    dx -= box_size * rintf(dx * inv_box_size);
                    dy -= box_size * rintf(dy * inv_box_size);
                    dz -= box_size * rintf(dz * inv_box_size);
                    
                    float r2 = dx*dx + dy*dy + dz*dz;
                    float r = sqrtf(r2);
                    
                    if (r > 0.0f && r < r_cut) {
                        float idr = rsqrtf(r2);
                        
                        // Truncation function via interpolation table (§3.4)
                        float pref = truncation_pref(r, r_s, erfc_table, exp_table, table_size);
                        
                        // Sign factor for Janus:
                        //   m+/m+ : +
                        //   m-/m- : + (attractif entre masses négatives, mais inverse signe acc)
                        //   m+/m- : -
                        // Convention: F sur i = sign_i × sign_j × G m_j × pref × r̂ / r²
                        float sign_factor = sign_i * sign_j[j];
                        float grav = sign_factor * g_const * mass_j[j] * idr * idr * idr * pref;
                        
                        acc.x = fmaf(grav, dx, acc.x);
                        acc.y = fmaf(grav, dy, acc.y);
                        acc.z = fmaf(grav, dz, acc.z);
                    }
                }
            }
        }
        
        // Accumulate dans la mémoire globale
        if (t < li.np) {
            int gi = li.start + t;
            atomicAdd(&acc_x[gi], acc.x);
            atomicAdd(&acc_y[gi], acc.y);
            atomicAdd(&acc_z[gi], acc.z);
        }
    }
}
```

**Note importante sur les signes Janus** :

Convention validée dans le code Barnes-Hut existant (cf. discussion Petit 2016 p.35-36) :
- `m⁺` interagit avec `m⁺` : Newton standard, attractif (sign_factor = +1)
- `m⁻` interagit avec `m⁻` : Newton appliqué à masses négatives, attractif entre elles (sign_factor = +1, mais l'attraction effective vient du fait que les deux masses sont négatives)
- `m⁺` interagit avec `m⁻` : anti-Newton, répulsif (sign_factor = -1)

**Critique** : ce sign_factor doit être **identique** au schéma Barnes-Hut existant, sinon résultats divergents. CLI doit lire le code Barnes-Hut Janus actuel et reproduire **exactement** sa convention. Test au §5.

**Convention de stockage mass/sign (CRITIQUE)** :

Tout au long du pipeline TreePM (Phase 3, 5, 6, 7), la convention de stockage est :
- `mass[i]` : **valeur absolue** de la masse, toujours `> 0`
- `sign[i]` : `i8` valant `+1` (pour m⁺) ou `-1` (pour m⁻)

Le calcul de force utilise alors :
```
sign_factor = sign[i] × sign[j]    (∈ {+1, -1})
acc[i] += sign_factor × G × mass[j] × ... × r̂_ij / r²
```

**Pourquoi cette convention ?**
- Évite les masses négatives en SP (qui peuvent introduire des NaN dans `sqrt`, `log`, etc.)
- Le `sign_factor` est calculable en une instruction (multiplication d'entiers)
- Compatible avec le stockage compact (i8 vs f32 économise 24 bits/particule)

**Si le code Barnes-Hut existant utilise une autre convention** (ex. `mass[i]` signé directement, sans `sign[i]`), CLI doit :
1. Documenter la convention BH dans `audit/05_janus_conventions.md`
2. Convertir au layout TreePM (mass, sign) à l'entrée du pipeline TreePM
3. Reconvertir au layout BH à la sortie (si on garde BH en parallèle pour comparaison)
4. Ajouter un test `test_mass_sign_roundtrip` qui vérifie l'invariance de `mass × sign`

**Tests CUDA** (avec features `#[cfg(feature = "cuda")]`) :

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_p2p_kernel_two_particles() {
    // Setup minimal : 2 particules m+, distance r=1, masse 1, G=1
    // F_attendue = 1/r² = 1
    let positions = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
    let signs = vec![1_i8, 1_i8];
    let masses = vec![1.0_f32, 1.0_f32];
    
    let acc = run_p2p_kernel(&positions, &signs, &masses, 
                             /*g=*/1.0, /*r_cut=*/10.0, /*r_s=*/2.0);
    
    // Particule 0 attirée vers +x par particule 1
    // Force sur 0 = +G·m1/r² · (r1-r0)/|r1-r0| = (1, 0, 0)
    // Mais avec splitting r_s=2, la force est suppressée par la function de coupure
    // [Calcul de la valeur exacte attendue avec erfc + Gaussienne]
    let r = 1.0_f64;
    let r_s = 2.0_f64;
    let pref = truncation_function_exact(r, r_s);
    let f_expected = 1.0 * pref;
    
    assert!((acc[0].0 - f_expected).abs() < 1e-4, 
            "F_x = {}, expected {}", acc[0].0, f_expected);
}

#[test]
#[cfg(feature = "cuda")]
fn test_p2p_kernel_janus_repulsion() {
    // m+ et m- à distance 1 : doivent se REPOUSSER
    let positions = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
    let signs = vec![1_i8, -1_i8];  // un de chaque
    let masses = vec![1.0_f32, 1.0_f32];
    
    let acc = run_p2p_kernel(&positions, &signs, &masses, 1.0, 10.0, 2.0);
    
    // Particule m+ doit être REPOUSSÉE par m- (force vers -x)
    assert!(acc[0].0 < 0.0, "Expected repulsion, got F_x = {}", acc[0].0);
    
    // Symétrie Newton 3 : F sur 1 = -F sur 0
    assert!((acc[0].0 + acc[1].0).abs() < 1e-5, "Action-reaction violated");
}
```

### 3.4 Table d'interpolation erfc/exp

**Principe (PhotoNs-GPU §3.3)** : sur GPU, `erfc()` direct coûte ~5.5× les flops d'un add/mul. Pré-calcul d'une table de 512 points avec développement de Taylor à 5 termes.

La fonction de troncation à représenter :

$$T(x) = \text{erfc}(x) + \frac{x}{\sqrt{\pi}} \cdot 2x \cdot e^{-x^2} \quad \text{où } x = r/(2r_s)$$

Sa série de Taylor autour de $x_i$ (PhotoNs Eq. 2) :

$$T(x) = T_i + \epsilon^2 E_i \cdot T^{[1]} - \epsilon (\epsilon^2-1) E_i^2 \cdot T^{[2]} + O(\epsilon^5)$$

où $\epsilon = x - x_i$ et $E_i = -4 e^{-x_i^2}/\sqrt{\pi}$.

**Implémentation** dans `src/treepm/cuda/truncation_table.cu` :

```cuda
// Table pré-calculée sur CPU à l'initialisation
// 512 points dans [0, 3] (r/2rs ∈ [0, 3] couvre r_cut = 6 r_s)
__constant__ float c_T_table[512];   // Ti
__constant__ float c_E_table[512];   // Ei

__device__ __forceinline__
float truncation_pref(float r, float r_s, int order = 4) {
    float x = r / (2.0f * r_s);
    if (x >= 3.0f) return 0.0f;  // beyond cutoff
    
    // Index dans la table
    float fidx = x * (511.0f / 3.0f);
    int idx = __float2int_rd(fidx);
    float eps = x - (idx * 3.0f / 511.0f);
    
    // Lecture table (en shared memory pour vitesse, sinon constant memory)
    float Ti = c_T_table[idx];
    float Ei = c_E_table[idx];
    
    float eps2 = eps * eps;
    float pref = Ti;                                     // T[0]
    pref = fmaf(eps2 * Ei, 1.0f, pref);                  // T[1]
    
    if (order >= 2) {
        float Ei2 = Ei * Ei;
        pref = fmaf(-eps * (eps2 - 1.0f) * Ei2, 1.0f, pref);  // T[2]
    }
    
    if (order >= 4) {
        // T[3] et T[4] - voir PhotoNs Eq. 2
        // ... (14 flops total grâce à FMA + reuse)
    }
    
    return pref;
}
```

**Génération de la table sur CPU** :

```rust
/// Pré-calcul de la table d'interpolation pour la fonction de troncation.
/// 
/// Table de 512 points dans [0, 3], pour x = r/(2 r_s) ∈ [0, 3].
/// 
/// Dérivation de T(x) à partir de PhotoNs-GPU Eq. (1) :
///   F_s(r) = -(r̂/r²) × T(r ; r_s)
///   T(r) = erfc(r/2r_s) + (r/(r_s√π)) × exp(-r²/4r_s²)
/// 
/// Avec x ≡ r/(2 r_s), on a r = 2x·r_s, r²/(4r_s²) = x², r/(r_s√π) = 2x/√π.
/// Donc :
///   T(x) = erfc(x) + (2x/√π) × exp(-x²)
/// 
/// Et E(x) = -4·exp(-x²)/√π (PhotoNs Eq. 2).
/// 
/// Reference: PhotoNs-GPU Eq. (1) et (2), Wang & Meng 2021
pub fn build_truncation_table() -> (Vec<f32>, Vec<f32>) {
    let n = 512;
    let mut t_table = vec![0.0_f32; n];
    let mut e_table = vec![0.0_f32; n];
    
    let sqrt_pi = std::f64::consts::PI.sqrt();
    
    for i in 0..n {
        let x = (i as f64) * 3.0 / (n as f64 - 1.0);
        let exp_mx2 = (-x * x).exp();
        
        // T(x) = erfc(x) + (2x/√π) × exp(-x²)
        let erfc_x = libm::erfc(x);
        let ti = erfc_x + (2.0 * x / sqrt_pi) * exp_mx2;
        
        // E(x) = -4·exp(-x²) / √π
        let ei = -4.0 * exp_mx2 / sqrt_pi;
        
        t_table[i] = ti as f32;
        e_table[i] = ei as f32;
    }
    
    (t_table, e_table)
}

/// Valeurs de référence pour validation des tests :
/// - T(0) = 1.0           (erfc(0)=1, terme exp s'annule)
/// - T(1) ≈ 0.5724        (erfc(1)≈0.1573 + 2/(√π·e)≈0.4151)
/// - T(2) ≈ 0.04601       (erfc(2)≈0.00468 + 4/(√π·e⁴)≈0.04133)
/// - T(3) ≈ 4.40e-4
/// - E(0) ≈ -2.2568       (= -4/√π)
/// - E(1) ≈ -0.8302
/// - E(2) ≈ -0.04133
```

**Tests** :

```rust
#[test]
fn test_truncation_table_at_zero() {
    let (t, e) = build_truncation_table();
    // x=0 : T(0) = erfc(0) + 0 = 1, E(0) = -4/√π ≈ -2.2568
    assert!((t[0] - 1.0).abs() < 1e-5,
            "T(0) = {}, expected 1.0", t[0]);
    let expected_e = -4.0 / std::f64::consts::PI.sqrt();
    assert!((e[0] as f64 - expected_e).abs() < 1e-4,
            "E(0) = {}, expected {}", e[0], expected_e);
}

#[test]
fn test_truncation_table_known_points() {
    let (t, e) = build_truncation_table();
    let n = 512;
    
    // Index pour x=1 : i = 1.0 × 511 / 3 ≈ 170
    let idx_1 = (1.0 * (n - 1) as f64 / 3.0).round() as usize;
    let x_actual = idx_1 as f64 * 3.0 / (n - 1) as f64;
    // Calculer T(x_actual) exact pour comparer
    let exp_mx2 = (-x_actual * x_actual).exp();
    let t_exact = libm::erfc(x_actual) + 
                  (2.0 * x_actual / std::f64::consts::PI.sqrt()) * exp_mx2;
    assert!((t[idx_1] as f64 - t_exact).abs() < 1e-5,
            "T at x≈1 differs by {}", (t[idx_1] as f64 - t_exact).abs());
    
    // À x=1 : T(1) ≈ 0.5724 (référence analytique)
    assert!((t[idx_1] as f64 - 0.5724).abs() < 0.01,
            "T(1) = {}, expected ~0.5724", t[idx_1]);
}

#[test]
fn test_truncation_interpolation_accuracy() {
    let (t_table, e_table) = build_truncation_table();
    
    // Comparer T(x) interpolé avec T(x) exact en plusieurs points
    let test_points = vec![0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
    for &x in &test_points {
        let interp = interpolate_truncation(x, &t_table, &e_table, 4);
        let exact = truncation_exact(x);
        let rel_err = (interp - exact).abs() / exact.abs();
        // PhotoNs claim T[4] adequate for DP : err < 1e-7
        assert!(rel_err < 1e-6, "x={}, err={}", x, rel_err);
    }
}

#[test]
fn test_truncation_at_cutoff() {
    let (t, e) = build_truncation_table();
    // x=3 : T(3) ≈ 0 (tail of erfc + Gaussian)
    let interp = interpolate_truncation(3.0, &t, &e, 4);
    assert!(interp.abs() < 1e-5);
}
```

**Benchmark §3.4** : Mesurer le temps GPU d'évaluation de la fonction de troncation pour 10⁹ appels.
- Naïf (erfc direct) : ~10 s sur RTX 3060 (estimation)
- Table T[2] : ~3 s (cible)
- Table T[4] : ~4 s (cible)

### 3.5 FMA et reuse de variables

PhotoNs-GPU réduit le kernel central de **39 flops à 14 flops** par interaction grâce à :
- Fused Multiply-Add (`fmaf` en CUDA) partout où possible
- Reuse de variables (`r2`, `idr`, `idr3`)
- Élimination des branches dans la boucle interne

**Pattern à suivre** dans le kernel :

```cuda
// AVANT (39 flops, naïf)
float dx = pos_j.x - pos_i.x;
float dy = pos_j.y - pos_i.y;
float dz = pos_j.z - pos_i.z;
float r2 = dx*dx + dy*dy + dz*dz;
float r = sqrtf(r2);
float r3 = r*r*r;
float pref = compute_pref(r, r_s);
float f_mag = G * m_j * pref / r3;
float fx = f_mag * dx;
float fy = f_mag * dy;
float fz = f_mag * dz;
acc.x += fx;
acc.y += fy;
acc.z += fz;

// APRÈS (14 flops avec FMA + rsqrt + table)
float dx = pos_j.x - pos_i.x;             // 1 sub
float dy = pos_j.y - pos_i.y;             // 1 sub
float dz = pos_j.z - pos_i.z;             // 1 sub
float r2 = fmaf(dx, dx, fmaf(dy, dy, dz*dz));  // 5 flops (3 fma + 1 mul)
float idr = rsqrtf(r2);                   // 1 intrinsic
float idr3 = idr * idr * idr;             // 2 mul
float pref = truncation_pref(r2, r_s);    // 4 flops (avec table)
float grav = g_m * idr3 * pref;           // 2 mul
acc.x = fmaf(grav, dx, acc.x);            // 1 fma
acc.y = fmaf(grav, dy, acc.y);            // 1 fma
acc.z = fmaf(grav, dz, acc.z);            // 1 fma
// Total : ~14-16 flops
```

**Test §3.5** : pas de test fonctionnel direct (le résultat doit être numériquement identique au pattern naïf à epsilon SP près). Test indirect via §6 (force vs particule unique).

**Benchmark §3.5** : Comparer kernel naïf vs kernel optimisé sur 10M interactions.
- Naïf : ~30 ms (cible)
- Optimisé : ~10-12 ms (gain 2.5-3×, conforme PhotoNs)

### 3.6 Bilan Phase 3 : benchmark complet kernel P2P

À la fin de la phase 3, faire tourner le kernel optimisé complet sur des configurations de test :

| N_particles | Config | Cible time/step kernel |
|---|---|---|
| 100K | uniforme | < 50 ms |
| 1M | uniforme | < 200 ms |
| 10M | clusterisé | < 3 s |

**Critères de validation Phase 3** :
- ✅ Tous les tests unitaires (3.1, 3.2, 3.3, 3.4) passent
- ✅ Benchmarks 3.5 et 3.6 atteignent les cibles (tolérance 50%)
- ✅ Speedup mesuré vs naïf > 2× (sinon FLAG)

**Commit** : `treepm-phase3: optimisations GPU PhotoNs-GPU intégrées`

---


## 4. PHASE 4 — PARAMÈTRES TREEPM (jour 2, 2-3h)

Choix des paramètres optimaux basé sur GreeM (Ishiyama 2009) + PhotoNs-GPU (Wang & Meng 2021).

### 4.1 Paramètres canoniques

| Paramètre | Symbole | Valeur | Source |
|---|---|---|---|
| Cellule mesh | Δg = L/N_pm | dépend de L, N_pm | — |
| Split scale | r_s | 1.2 × Δg | PhotoNs §2 |
| Cutoff radius | r_cut | 6 × Δg | PhotoNs §2 |
| Opening angle | θ | 0.5 | GreeM (z<10) |
| Particles per leaf | N_leaf | 10 | GreeM, PhotoNs |
| Group threshold | N_crit | 300 | GreeM |
| PM grid (1M, L=250 Mpc) | N_pm | 256³ | calcul §1.4 |
| PM grid (10M, L=500 Mpc) | N_pm | 512³ | calcul §1.4 |

### 4.2 Validation par sweep paramétrique

Pour confirmer ces choix, CLI doit faire un sweep paramétrique sur une configuration test (1M particules, snapshot Janus à z=5) :

```rust
#[test]
#[ignore]  // long test, run avec --ignored
fn parameter_sweep_1m() {
    let particles = load_test_snapshot("snapshots/janus_1m_z5.bin");
    
    let configs = vec![
        // (r_s_factor, r_cut_factor, theta, label)
        (1.0, 4.0, 0.5, "rs1.0_rc4.0_t0.5"),
        (1.2, 6.0, 0.5, "rs1.2_rc6.0_t0.5"),  // canonique
        (1.5, 6.0, 0.5, "rs1.5_rc6.0_t0.5"),
        (1.2, 8.0, 0.5, "rs1.2_rc8.0_t0.5"),
        (1.2, 6.0, 0.3, "rs1.2_rc6.0_t0.3"),  // précision +
        (1.2, 6.0, 0.7, "rs1.2_rc6.0_t0.7"),  // vitesse +
    ];
    
    for (rs_f, rc_f, theta, label) in configs {
        let result = run_treepm_step(&particles, rs_f, rc_f, theta);
        let force_err = compute_force_error_vs_pp(&result, &particles);
        let time_ms = result.timing_ms;
        
        // Save to JSON
        save_benchmark(&format!("logs/treepm/benchmarks/sweep_{}.json", label),
                       force_err, time_ms);
    }
}
```

**Critères de validation Phase 4** :
- ✅ Sweep complété sans crash
- ✅ Configuration canonique (rs1.2, rc6.0, t0.5) atteint force_err < 0.5% sur 95% des particules
- ✅ Configuration canonique au moins 2× plus rapide que la plus lente

**Commit** : `treepm-phase4: paramètres canoniques validés par sweep`

---

## 5. PHASE 5 — EXTENSION JANUS (jour 2-3, 4-6h)

Maintenant que le TreePM Newton fonctionne, on l'étend à la physique Janus. **Cette phase est la plus délicate** : il faut garantir cohérence avec le code Barnes-Hut Janus existant.

### 5.1 Lecture impérative du code Janus existant

CLI doit **AVANT TOUT** lire et documenter le code Barnes-Hut Janus actuel :

```bash
# Localiser le calcul de force Janus
grep -rn "sign_i.*sign_j\|sign_factor\|janus_force\|cross_coupling" src/ \
    > audit/05_janus_conventions.md

# Documenter :
# - Convention de signe pour les 3 cas d'interaction
# - Convention de softening (asymétrique m+/m- ?)
# - Drift/kick avec dual scale factors a+(t), a-(t)
# - Application du dynamic VSL c_ratio_sq(t) = a+/a-
```

**Livrable** : `audit/05_janus_conventions.md` qui répond à :
1. Comment est encodé le signe `m+`/`m-` dans le code (enum, i8, bool, mass négatif) ?
2. Le softening est-il symétrique entre populations ou non ?
3. Où exactement est appliqué le c_ratio_sq dans le step DKD ?
4. Le drift utilise-t-il a+ ou a- pour les particules m- ?

### 5.2 Extension de la source PM : ρ_eff = ρ⁺ - ρ⁻

Pour la partie PM, la modification est minime :

```rust
/// Compute the effective density source for the PM Poisson solver.
/// 
/// In Janus cosmology with bimetric gravity, the PM source is:
///   ρ_eff(x) = ρ⁺(x) - ρ⁻(x)
/// 
/// Particles m+ contribute with sign +1, particles m- with sign -1.
pub fn compute_effective_density_cic(
    particles: &ParticleArrays,
    n_pm: usize,
    l: f64,
) -> Vec<f32> {
    let mut rho = vec![0.0_f32; n_pm * n_pm * n_pm];
    let cell_size = (l / n_pm as f64) as f32;
    
    for i in 0..particles.n {
        let sign = particles.sign[i] as f32;  // +1 or -1
        let mass = particles.mass[i];
        let signed_mass = sign * mass;
        
        deposit_cic_single(
            &mut rho, n_pm, cell_size,
            particles.pos_x[i], particles.pos_y[i], particles.pos_z[i],
            signed_mass,
        );
    }
    
    rho
}
```

**Application de la force PM aux particules** :

Une fois `Φ` résolu via `∇²Φ = 4πG·ρ_eff`, la force PM sur une particule m+ est `-∇Φ`, sur une particule m- c'est **+∇Φ** (signe inverse car masse négative).

```rust
pub fn apply_pm_force_with_signs(
    particles: &mut ParticleArrays,
    grad_phi: &GradientField,
    n_pm: usize,
    l: f64,
) {
    for i in 0..particles.n {
        let sign = particles.sign[i] as f32;  // +1 or -1
        let (gx, gy, gz) = interpolate_cic_gradient(
            grad_phi, n_pm, l,
            particles.pos_x[i], particles.pos_y[i], particles.pos_z[i],
        );
        
        // F = -∇Φ × sign  (pour m-, masse négative inverse la force)
        particles.acc_x[i] -= sign * gx;
        particles.acc_y[i] -= sign * gy;
        particles.acc_z[i] -= sign * gz;
    }
}
```

### 5.3 Tests de cohérence Janus PM

```rust
#[test]
fn test_pm_two_plus_attract() {
    // Deux particules m+ à distance L/4 doivent s'attirer
    let n_pm = 64;
    let l = 100.0_f64;
    let mut particles = ParticleArrays::new(2);
    particles.pos_x = vec![25.0, 75.0];
    particles.pos_y = vec![50.0, 50.0];
    particles.pos_z = vec![50.0, 50.0];
    particles.sign = vec![1, 1];
    particles.mass = vec![1.0, 1.0];
    
    let rho = compute_effective_density_cic(&particles, n_pm, l);
    let phi = solve_poisson_corrected(&rho, n_pm, l, 1.0);
    let grad = compute_gradient(&phi, n_pm, l);
    apply_pm_force_with_signs(&mut particles, &grad, n_pm, l);
    
    // Particule 0 doit être attirée vers +x (vers particule 1)
    assert!(particles.acc_x[0] > 0.0,
            "Two m+ should attract, got acc_x[0] = {}", particles.acc_x[0]);
    assert!(particles.acc_x[1] < 0.0,
            "Two m+ should attract, got acc_x[1] = {}", particles.acc_x[1]);
}

#[test]
fn test_pm_plus_minus_repel() {
    // m+ et m- à distance L/4 doivent se repousser
    let n_pm = 64;
    let l = 100.0_f64;
    let mut particles = ParticleArrays::new(2);
    particles.pos_x = vec![25.0, 75.0];
    particles.pos_y = vec![50.0, 50.0];
    particles.pos_z = vec![50.0, 50.0];
    particles.sign = vec![1, -1];  // un de chaque
    particles.mass = vec![1.0, 1.0];
    
    let rho = compute_effective_density_cic(&particles, n_pm, l);
    let phi = solve_poisson_corrected(&rho, n_pm, l, 1.0);
    let grad = compute_gradient(&phi, n_pm, l);
    apply_pm_force_with_signs(&mut particles, &grad, n_pm, l);
    
    // m+ (idx 0) doit être REPOUSSÉ (vers -x, loin de m-)
    assert!(particles.acc_x[0] < 0.0,
            "m+ should be repelled by m-, got acc_x[0] = {}", particles.acc_x[0]);
    // m- (idx 1) doit être REPOUSSÉ aussi (vers +x, loin de m+)
    assert!(particles.acc_x[1] > 0.0,
            "m- should be repelled by m+, got acc_x[1] = {}", particles.acc_x[1]);
}

#[test]
fn test_pm_two_minus_attract() {
    // Deux m- doivent s'attirer entre elles (Petit p.36)
    let n_pm = 64;
    let l = 100.0_f64;
    let mut particles = ParticleArrays::new(2);
    particles.pos_x = vec![25.0, 75.0];
    particles.pos_y = vec![50.0, 50.0];
    particles.pos_z = vec![50.0, 50.0];
    particles.sign = vec![-1, -1];
    particles.mass = vec![1.0, 1.0];
    
    let rho = compute_effective_density_cic(&particles, n_pm, l);
    let phi = solve_poisson_corrected(&rho, n_pm, l, 1.0);
    let grad = compute_gradient(&phi, n_pm, l);
    apply_pm_force_with_signs(&mut particles, &grad, n_pm, l);
    
    // m- (idx 0) doit être attiré vers +x (vers m- en idx 1)
    assert!(particles.acc_x[0] > 0.0, "Two m- should attract");
    assert!(particles.acc_x[1] < 0.0, "Two m- should attract");
}
```

### 5.4 Drift/Kick avec dual scale factor et VSL

C'est le point le plus délicat. CLI doit reproduire **exactement** le schéma DKD du code Barnes-Hut Janus existant.

D'après les données mémoire (chat précédent) :
- Dynamic VSL : `c_ratio_sq(t) = a⁺(t) / a⁻(t)`
- η = 1.045 (Pantheon+ fit)
- μ = 19 (canonical Petit value, contrainte de platitude)

```rust
/// Drift step for Janus with dual scale factors.
/// 
/// Convention vélocité (à confirmer depuis le code Barnes-Hut existant):
///   v_pec = a · ẋ_co  (peculiar velocity, GADGET-2/4 convention)
/// 
/// Drift: x_co(t+dt) = x_co(t) + v_pec(t+dt/2) · dt / a²
pub fn drift_janus(
    particles: &mut ParticleArrays,
    a_plus: f64,
    a_minus: f64,
    dt: f64,
) {
    for i in 0..particles.n {
        let sign = particles.sign[i];
        let a = if sign > 0 { a_plus } else { a_minus };
        let inv_a2 = (1.0 / (a * a)) as f32;
        
        particles.pos_x[i] += particles.vel_x[i] * inv_a2 * dt as f32;
        particles.pos_y[i] += particles.vel_y[i] * inv_a2 * dt as f32;
        particles.pos_z[i] += particles.vel_z[i] * inv_a2 * dt as f32;
    }
}

/// Kick step for Janus with dual scale factors.
/// 
/// Avec VSL dynamique c_ratio_sq(t) = a⁺/a⁻, le facteur de couplage croisé
/// est modulé. La force totale sur m+ contient :
///   - F⁺⁺ Newton (PM + Tree avec sign_factor=+1)
///   - F⁺⁻ anti-Newton modulé par c_ratio_sq
/// 
/// [À VÉRIFIER précisément avec code Barnes-Hut existant]
pub fn kick_janus(
    particles: &mut ParticleArrays,
    a_plus: f64,
    a_minus: f64,
    h_plus: f64,  // Hubble for m+
    h_minus: f64, // Hubble for m-
    dt: f64,
) {
    for i in 0..particles.n {
        let sign = particles.sign[i];
        let (a, h) = if sign > 0 { (a_plus, h_plus) } else { (a_minus, h_minus) };
        let inv_a = (1.0 / a) as f32;
        let h_drag = (h * dt) as f32;
        
        // Hubble drag: v -= H·v·dt
        particles.vel_x[i] *= 1.0 - h_drag;
        particles.vel_y[i] *= 1.0 - h_drag;
        particles.vel_z[i] *= 1.0 - h_drag;
        
        // Force kick: v += a · F · dt (où a est le scale factor, pas l'accélération)
        // Note: convention dépend de la définition de l'accélération computée
        // [À CONFIRMER depuis code Barnes-Hut]
        particles.vel_x[i] += inv_a * particles.acc_x[i] * dt as f32;
        particles.vel_y[i] += inv_a * particles.acc_y[i] * dt as f32;
        particles.vel_z[i] += inv_a * particles.acc_z[i] * dt as f32;
    }
}
```

### 5.5 Test critique de cohérence Barnes-Hut ↔ TreePM

Test de référence : faire **un step** de TreePM Janus et un step de Barnes-Hut Janus sur les **mêmes initial conditions**. Comparer les accélérations particule par particule.

```rust
#[test]
#[ignore]  // long, GPU required
fn test_treepm_vs_barnes_hut_janus_one_step() {
    // Charger un snapshot Janus de 100K particules à z=5
    let particles = load_snapshot("snapshots/janus_100k_z5.bin");
    
    // Faire UN step avec Barnes-Hut existant
    let mut p_bh = particles.clone();
    let acc_bh = barnes_hut_janus_step(&mut p_bh);
    
    // Faire UN step avec TreePM
    let mut p_tpm = particles.clone();
    let acc_tpm = treepm_janus_step(&mut p_tpm);
    
    // Comparer les accélérations
    let mut max_rel_err = 0.0_f64;
    let mut median_rel_err = Vec::new();
    
    for i in 0..particles.n {
        let mag_bh = (acc_bh.x[i].powi(2) + acc_bh.y[i].powi(2) + acc_bh.z[i].powi(2)).sqrt();
        if mag_bh < 1e-10 { continue; }  // skip nearly-zero
        
        let dx = (acc_bh.x[i] - acc_tpm.x[i]) as f64;
        let dy = (acc_bh.y[i] - acc_tpm.y[i]) as f64;
        let dz = (acc_bh.z[i] - acc_tpm.z[i]) as f64;
        let diff = (dx*dx + dy*dy + dz*dz).sqrt();
        let rel_err = diff / mag_bh as f64;
        
        max_rel_err = max_rel_err.max(rel_err);
        median_rel_err.push(rel_err);
    }
    median_rel_err.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = median_rel_err[median_rel_err.len() / 2];
    
    // TreePM avec r_a=1.5 L/N_pm doit donner forces sub-pourcent vs BH
    assert!(median < 0.01,
            "Median rel err = {} (target < 1%)", median);
    // Max peut être ~5% sur quelques particules (zones de raccord)
    assert!(max_rel_err < 0.05,
            "Max rel err = {} (target < 5%)", max_rel_err);
}
```

**Critères de validation Phase 5** :
- ✅ Tests §5.3 passent (3 cas Janus PM corrects)
- ✅ `audit/05_janus_conventions.md` complet et cohérent
- ✅ Test §5.5 : médiane d'erreur relative < 1%, max < 5% sur 100K particules
- ✅ Conservation d'énergie sur 100 steps : drift < 1%

**Commit** : `treepm-phase5: extension Janus (sign-flip source PM, dual a±, VSL)`

---


## 6. PHASE 6 — VALIDATION NEWTON (jour 3, 2-3h)

Avant de tester Janus en intégration, valider que le TreePM en mode Newton pur donne les bonnes forces. Deux tests canoniques.

### 6.1 Test particule unique (GrGadget Fig. 5)

**Protocole** :
- 1 particule massive (M=1) au centre d'une boîte L=1000 Mpc/h, N_pm=256
- Particules tests sans masse à différentes distances de 1 Δg à L/4
- Calculer la force avec TreePM, comparer à F_exact = G·M·r̂/r²

```rust
#[test]
#[ignore]
fn test_force_single_source() {
    let l = 1000.0_f64;
    let n_pm = 256;
    let dx_pm = l / n_pm as f64;  // 3.9 Mpc/h
    
    // 1 source au centre, 100 test particles à différentes distances
    let mut particles = ParticleArrays::new(101);
    particles.pos_x[0] = (l/2.0) as f32;
    particles.pos_y[0] = (l/2.0) as f32;
    particles.pos_z[0] = (l/2.0) as f32;
    particles.mass[0] = 1.0;
    particles.sign[0] = 1;
    
    let distances: Vec<f32> = (1..=100).map(|i| (i as f32) * dx_pm as f32 * 0.5).collect();
    for (k, &d) in distances.iter().enumerate() {
        particles.pos_x[k+1] = (l/2.0) as f32 + d;
        particles.pos_y[k+1] = (l/2.0) as f32;
        particles.pos_z[k+1] = (l/2.0) as f32;
        particles.mass[k+1] = 0.0;  // test particle
        particles.sign[k+1] = 1;
    }
    
    let acc = compute_force_treepm_newton(&particles, n_pm, l, 1.0);
    
    // Pour chaque test particle, comparer F mesurée à F exacte
    let mut errors = Vec::new();
    for k in 0..100 {
        let d = distances[k] as f64;
        let f_exact = -1.0 / (d * d);  // F sur test (vers source) = -G·M/r² (radial)
        let f_meas = acc.x[k+1] as f64;
        let rel_err = ((f_meas - f_exact) / f_exact).abs();
        errors.push((d, rel_err));
    }
    
    // Save data pour figure
    save_force_test_data("logs/treepm/benchmarks/force_single_source.csv", &errors);
    
    // Critères :
    // - Loin de la source (r > 5 Δg) : err < 1%
    // - Zone de raccord (r ~ Δg) : err peut atteindre 5%
    let large_distance_errors: Vec<f64> = errors.iter()
        .filter(|(d, _)| *d > 5.0 * dx_pm)
        .map(|(_, e)| *e)
        .collect();
    
    let max_far_err = large_distance_errors.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_far_err < 0.01,
            "Max err at far distances = {}, target < 1%", max_far_err);
}
```

### 6.2 Test power spectrum Newton TreePM vs PP direct

**Protocole** :
- Snapshot 1M particules avec ICs Zel'dovich à z=10
- Calculer P(k) avec :
  - Méthode 1 : TreePM Newton complet
  - Méthode 2 : PM seul (Tree off)
  - Méthode 3 : Direct PP (référence, lent mais exact)

```rust
#[test]
#[ignore]  // long
fn test_pk_treepm_vs_pp_newton() {
    let particles = load_snapshot("snapshots/zeldovich_1m_z10.bin");
    let l = 250.0_f64;  // 1M dans 250 Mpc/h
    let n_pm = 256;
    
    // Faire 100 steps avec chaque méthode
    let mut p_treepm = particles.clone();
    for _ in 0..100 {
        treepm_newton_step(&mut p_treepm, n_pm, l);
    }
    
    let mut p_pmonly = particles.clone();
    for _ in 0..100 {
        pmonly_newton_step(&mut p_pmonly, n_pm, l);
    }
    
    // PP direct trop coûteux pour 1M, on prend 100K sub-sample
    let p_pp_subset = subsample(&particles, 100_000);
    let mut p_pp = p_pp_subset.clone();
    for _ in 0..100 {
        pp_direct_step(&mut p_pp);
    }
    
    let pk_treepm = compute_power_spectrum(&p_treepm, n_pm, l);
    let pk_pmonly = compute_power_spectrum(&p_pmonly, n_pm, l);
    let pk_pp = compute_power_spectrum(&p_pp, /*pour comparaison aux grandes échelles*/);
    
    // TreePM doit matcher PP à small scales (k > 0.1 h/Mpc)
    // TreePM doit matcher PM à large scales (k < 0.05 h/Mpc)
    
    let high_k_err = compare_pk(&pk_treepm, &pk_pp, 0.1, 0.5);
    let low_k_err = compare_pk(&pk_treepm, &pk_pmonly, 0.01, 0.05);
    
    assert!(high_k_err < 0.05, "P(k) high-k err = {} vs PP", high_k_err);
    assert!(low_k_err < 0.02, "P(k) low-k err = {} vs PM-only", low_k_err);
}
```

**Critères de validation Phase 6** :
- ✅ Test §6.1 : err < 1% loin de la source
- ✅ Test §6.2 : P(k) match PP à small scale, PM à large scale, < 5% partout

**Commit** : `treepm-phase6: validation Newton (force + P(k))`

---

## 7. PHASE 7 — VALIDATION JANUS (jour 3, 3-4h)

Maintenant que TreePM Newton est validé, on teste l'extension Janus sur des configurations dont les résultats sont connus.

### 7.1 Mini-run Janus 100K particules

Utiliser des ICs Janus existantes (snapshot 100K déjà testé en Barnes-Hut) et faire un mini-run TreePM jusqu'à z=2 (~500 steps).

```rust
#[test]
#[ignore]  // long, GPU
fn test_janus_minirun_100k_z10_to_z2() {
    let particles = load_snapshot("snapshots/janus_ics_100k.bin");
    
    let mut p = particles.clone();
    let mut diagnostics = Vec::new();
    
    // Run TreePM Janus avec dynamic VSL et μ=19
    let config = JanusConfig {
        mu: 19.0,
        eta: 1.045,
        z_init: 10.0,
        z_target: 2.0,
        n_steps: 500,
        dynamic_vsl: true,
    };
    
    for step in 0..config.n_steps {
        treepm_janus_step(&mut p, &config);
        
        if step % 50 == 0 {
            let diag = compute_janus_diagnostics(&p);
            diagnostics.push((step, diag));
        }
    }
    
    save_diagnostics("logs/treepm/benchmarks/janus_100k_z10_z2.csv", &diagnostics);
    
    // Critères de validation Janus à z=2 :
    let final_diag = &diagnostics.last().unwrap().1;
    
    // Anti-corrélation m+/m- (signature Janus)
    assert!(final_diag.corr_plus_minus < -0.03,
            "Corr(δ+,δ-) = {} (expected < -0.03)", final_diag.corr_plus_minus);
    
    // P_×(k) < 0 sur la gamme k résolue
    assert!(final_diag.frac_negative_px > 0.8,
            "Frac k with P_x<0 = {} (expected > 80%)", final_diag.frac_negative_px);
    
    // Pas de runaway de v_rms
    assert!(final_diag.v_rms_plus < 5000.0 && final_diag.v_rms_minus < 5000.0,
            "v_rms runaway: v+ = {}, v- = {}", 
            final_diag.v_rms_plus, final_diag.v_rms_minus);
}
```

### 7.2 Conservation d'énergie

Sur le run 100K, vérifier la conservation d'énergie totale.

**Définition de l'énergie totale Janus** : pour un système bimétrique m+/m-, la définition n'est pas unique. On adopte la convention suivante (cohérente avec le code Barnes-Hut existant — à vérifier) :

```
E_tot = KE⁺ + KE⁻ + W⁺⁺ + W⁻⁻ + W⁺⁻
```

où :
- `KE⁺ = ½ Σ_i m_i⁺ v_i²` (énergie cinétique des m+, positive)
- `KE⁻ = ½ Σ_j m_j⁻ v_j²` (avec convention `m_j⁻ > 0` mais comptée avec signe Janus, donc en pratique on mesure `½ Σ_j |m_j⁻| v_j²` séparément)
- `W⁺⁺ = -½ Σ_{i≠i'} G m_i⁺ m_{i'}⁺ / r_{ii'}` (auto-énergie m+, négative)
- `W⁻⁻ = -½ Σ_{j≠j'} G m_j⁻ m_{j'}⁻ / r_{jj'}` (auto-énergie m-, attractive entre m-)
- `W⁺⁻ = +Σ_{i,j} G m_i⁺ m_j⁻ / r_{ij}` (énergie d'interaction croisée, **positive** car force répulsive = énergie potentielle augmente quand on rapproche)

**Critère de conservation** : drift relatif `|E(t)/E(0) - 1| < 5%` sur 500 steps.

**Note critique** : Janus avec dynamic VSL peut formellement violer la conservation d'énergie stricte (le c qui varie injecte/dissipe de l'énergie). On accepte donc une dérive modérée, mais qui doit rester monotone (pas d'oscillation pathologique).

```rust
fn check_energy_conservation(diagnostics: &[(usize, JanusDiagnostics)]) -> bool {
    let e0 = diagnostics[0].1.total_energy;
    let e_final = diagnostics.last().unwrap().1.total_energy;
    let drift = ((e_final - e0) / e0).abs();
    
    // Tolérance : 5% sur 500 steps avec leapfrog DKD + VSL
    drift < 0.05
}

/// Compute total Janus energy for diagnostic.
/// 
/// Direct sum O(N²) — only run on small samples for diagnostics.
fn compute_total_energy_janus(particles: &ParticleArrays, g: f64) -> f64 {
    let mut ke = 0.0;
    let mut w_pp = 0.0;
    let mut w_mm = 0.0;
    let mut w_pm = 0.0;
    
    for i in 0..particles.n {
        let v2 = (particles.vel_x[i].powi(2) + 
                  particles.vel_y[i].powi(2) + 
                  particles.vel_z[i].powi(2)) as f64;
        let m_abs = particles.mass[i].abs() as f64;
        ke += 0.5 * m_abs * v2;
    }
    
    // Direct N² gravitational sum — only feasible for N < 100K
    for i in 0..particles.n {
        for j in (i+1)..particles.n {
            let dx = (particles.pos_x[j] - particles.pos_x[i]) as f64;
            let dy = (particles.pos_y[j] - particles.pos_y[i]) as f64;
            let dz = (particles.pos_z[j] - particles.pos_z[i]) as f64;
            let r = (dx*dx + dy*dy + dz*dz).sqrt();
            if r < 1e-10 { continue; }
            
            let mi = particles.mass[i].abs() as f64;
            let mj = particles.mass[j].abs() as f64;
            let si = particles.sign[i] as f64;
            let sj = particles.sign[j] as f64;
            
            let phi_ij = -g * mi * mj / r;  // pure Newton
            
            if si > 0.0 && sj > 0.0 {
                w_pp += phi_ij;          // m+/m+ : attractif (W < 0)
            } else if si < 0.0 && sj < 0.0 {
                w_mm += phi_ij;          // m-/m- : attractif entre eux (W < 0)
            } else {
                w_pm += -phi_ij;         // m+/m- : répulsif (W > 0, signe inversé)
            }
        }
    }
    
    ke + w_pp + w_mm + w_pm
}
```

### 7.3 Comparaison avec Barnes-Hut sur même run

Refaire le même mini-run avec Barnes-Hut Janus, comparer les diagnostics finaux :

```rust
#[test]
#[ignore]
fn test_janus_treepm_vs_barnes_hut_full_run() {
    let particles = load_snapshot("snapshots/janus_ics_100k.bin");
    
    let mut p_bh = particles.clone();
    let mut p_tpm = particles.clone();
    
    let config = JanusConfig::canonical_test();  // 500 steps, μ=19, etc.
    
    for _ in 0..config.n_steps {
        barnes_hut_janus_step(&mut p_bh, &config);
        treepm_janus_step(&mut p_tpm, &config);
    }
    
    let diag_bh = compute_janus_diagnostics(&p_bh);
    let diag_tpm = compute_janus_diagnostics(&p_tpm);
    
    // Tolérances : les deux méthodes ne donnent pas exactement les mêmes
    // résultats à cause des différences de discrétisation, mais doivent être proches.
    
    let corr_diff = (diag_bh.corr_plus_minus - diag_tpm.corr_plus_minus).abs();
    assert!(corr_diff < 0.02, "Corr diff = {}", corr_diff);
    
    let sigma8_diff = ((diag_bh.sigma8 - diag_tpm.sigma8) / diag_bh.sigma8).abs();
    assert!(sigma8_diff < 0.05, "σ8 rel diff = {}", sigma8_diff);
}
```

**Critères de validation Phase 7** :
- ✅ Mini-run 100K complète sans crash
- ✅ Anti-corrélation m+/m- préservée
- ✅ P_×(k) < 0 sur >80% des modes
- ✅ Conservation d'énergie < 5% sur 500 steps
- ✅ Diagnostics Janus TreePM cohérents avec Barnes-Hut (Corr diff < 0.02)

**Commit** : `treepm-phase7: validation Janus 100K confirme physique correcte`

---

## 8. PHASE 8 — BENCHMARKS PERFORMANCE (jour 4, 2-3h)

Mesurer la performance TreePM aux différentes échelles cibles, comparer aux prédictions du modèle PhotoNs-GPU.

### 8.1 Benchmarks scaling

```rust
#[test]
#[ignore]
fn benchmark_treepm_scaling() {
    let configs = vec![
        // (N, L_Mpc, N_pm, label)
        (100_000, 100.0, 128, "100k_L100_Npm128"),
        (1_000_000, 250.0, 256, "1M_L250_Npm256"),
        (10_000_000, 500.0, 512, "10M_L500_Npm512"),
    ];
    
    for (n, l, n_pm, label) in configs {
        let particles = generate_random_uniform(n, l);
        
        // Mesurer 10 steps en moyenne
        let mut times = Vec::new();
        for _ in 0..10 {
            let t0 = std::time::Instant::now();
            treepm_janus_step_minimal(&particles, l, n_pm);
            let dt = t0.elapsed().as_secs_f64();
            times.push(dt);
        }
        
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let std = (times.iter().map(|t| (t-mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
        
        save_benchmark(&format!("logs/treepm/benchmarks/scaling_{}.json", label),
                       n, l, n_pm, mean, std);
    }
}
```

**Cibles de performance attendues sur RTX 3060** (basées sur PhotoNs-GPU × ratio 0.4) :

| N | L (Mpc/h) | N_pm | Cible time/step |
|---|---|---|---|
| 100K | 100 | 128 | < 100 ms |
| 1M | 250 | 256 | < 500 ms |
| 10M | 500 | 512 | < 5 s |

### 8.2 Profiling GPU avec nvprof / nsight

```bash
# Profiling du kernel P2P sur 1M particules
ncu --set full --target-processes all \
    --export logs/treepm/profile_1m \
    cargo test --release benchmark_treepm_scaling -- --ignored

# Analyser :
# - Occupancy (cible: > 60%)
# - Memory throughput (cible: > 500 GB/s)
# - Achieved compute throughput (cible: > 3 TFlops sur 13 théoriques)
```

### 8.3 Mémoire mesurée vs prédite

```rust
#[test]
#[ignore]
fn measure_memory_usage() {
    for &n in &[100_000, 1_000_000, 10_000_000] {
        let mem_before = cuda_memory_used();
        let _particles = allocate_treepm_state(n);
        let mem_after = cuda_memory_used();
        
        let mem_used_gb = (mem_after - mem_before) as f64 / 1e9;
        let predicted_gb = predict_memory(n);
        
        let rel_diff = ((mem_used_gb - predicted_gb) / predicted_gb).abs();
        assert!(rel_diff < 0.20, 
                "N={}: predicted {} GB, used {} GB", n, predicted_gb, mem_used_gb);
    }
}
```

**Critères de validation Phase 8** :
- ✅ Time/step à 100K : < 200 ms (tolérance 2× sur cible)
- ✅ Time/step à 1M : < 1 s
- ✅ Mémoire mesurée match prédiction à ±20%
- ✅ GPU occupancy > 50% (sinon FLAG)

**Commit** : `treepm-phase8: benchmarks complets et profiling GPU`

---

## 9. PHASE 9 — DÉCISION GO/NO-GO RUN 1M (jour 4)

Phase finale : CLI compile tous les résultats des phases 1-8 et produit un rapport pour humain.

### 9.1 Génération du rapport

CLI génère `logs/treepm/phase9_decision.md` avec la structure suivante :

```markdown
# Décision GO/NO-GO — Run TreePM Janus 1M particules z=10→0

**Date** : <DATE>
**Branch** : feature/treepm-integration
**Commit head** : <HASH>

## Résumé exécutif

**Verdict** : GO / NO-GO / GO-AVEC-RÉSERVES

[Une phrase de justification]

## Tableaux de résultats par phase

### Phase 1 — Audit
- Module TreePM existant : <résumé>
- Corrections GrGadget initiales : <statut>
- GPU capabilities : <résumé>

### Phase 2 — Corrections numériques
| Test | Résultat | Tolérance | Status |
|---|---|---|---|
| Gradient ord 4 convergence | ratio 16.0 | > 12 | ✅ |
| CIC roundtrip | err 0.5% | < 1% | ✅ |
| Poisson point mass | err 3.2% | < 5% | ✅ |

### Phase 3 — Optimisations GPU
| Optim | Avant | Après | Speedup | Status |
|---|---|---|---|---|
| AoS → SoA | 5 ms | 3 ms | 1.7× | ✅ |
| Task list reorder | naïf | + | + | ✅ |
| Table erfc/exp | 10 s | 4 s | 2.5× | ✅ |
| FMA + reuse | 30 ms | 12 ms | 2.5× | ✅ |
| **Total kernel** | <ref> | <new> | <gain> | ✅/⚠️ |

### Phase 4-5 — Paramètres et Janus
[...]

### Phase 6-7 — Validation
| Test | Résultat | Cible | Status |
|---|---|---|---|
| Force vs PP (10 Δg) | 0.8% | < 1% | ✅ |
| P(k) Newton match | 1.2% | < 5% | ✅ |
| Janus 100K corr | -0.058 | < -0.03 | ✅ |
| Énergie conservation | 2.1% | < 5% | ✅ |

### Phase 8 — Performance
| N | Time/step mesuré | Cible | Status |
|---|---|---|---|
| 100K | 85 ms | < 200 ms | ✅ |
| 1M | 0.6 s | < 1 s | ✅ |
| 10M | 4.2 s | < 5 s | ✅ |

## Liste des FLAGS ouverts

[Liste des points qui ont nécessité un workaround]

## Recommandation pour le run 1M

### Si GO :
**Configuration recommandée** :
```toml
n_particles = 1_000_000
n_plus = 50_000        # 5% de m+
n_minus = 950_000      # 95% de m- (ratio μ=19)
box_size_mpc = 250.0
n_pm = 256
mu = 19.0
eta = 1.045
z_init = 10.0
z_target = 0.0
# Pour z=10→0, on parcourt a=1/11 → 1, soit 11× expansion.
# Précédent run (z=4→0) utilisait 30000 steps. Pour z=10→0, on
# augmente proportionnellement : ~50000 steps avec dt adaptatif.
# Si on choisit fixed-step pour stabilité : ~15000 steps de dt fixe.
n_steps = 15000        # adaptive : peut être réduit à ~10000
dynamic_vsl = true
softening_plus = 0.05  # Mpc/h
softening_minus = 0.25 # Mpc/h (5× selon convention existante)
output_freq_steps = 500
checkpoint_freq_steps = 2000
```

**Commande de lancement** :
```bash
cargo run --release --features cuda --bin janus_run -- \
    --config configs/treepm_validation_1m.toml \
    --output runs/treepm_1m_$(date +%F)/
```

**ETA** : ~55 minutes pour 15000 steps (~0.22 s/step) ; jusqu'à 3h si steps adaptatifs avec dt court à z élevé

**Monitoring** :
```bash
# Dans une autre fenêtre, suivre l'évolution
tail -f runs/treepm_1m_*/log.txt | grep -E "step|Corr|sigma8"
nvidia-smi -l 5
```

**Critères de succès du run 1M** :
- Run termine à z=0 sans crash
- Corr(δ+,δ-) finale entre -0.05 et -0.10
- σ8 finale entre 0.65 et 0.75
- t₀ entre 14 et 17 Gyr
- v_rms+ et v_rms- < 3000 km/s à z=0

### Si NO-GO :
**Bloqueurs identifiés** :
[Liste des problèmes]

**Actions de remédiation** :
[Plan pour résoudre]
```

### 9.2 Critères automatiques GO / NO-GO / GO-AVEC-RÉSERVES

```rust
fn determine_decision(reports: &[PhaseReport]) -> Decision {
    let critical_failures = reports.iter()
        .filter(|r| r.has_critical_failure())
        .count();
    
    let flags = reports.iter()
        .map(|r| r.flags.len())
        .sum::<usize>();
    
    if critical_failures > 0 {
        Decision::NoGo {
            reasons: collect_critical_failures(reports),
        }
    } else if flags > 3 {
        Decision::GoWithReservations {
            flags: collect_all_flags(reports),
        }
    } else {
        Decision::Go {
            recommended_config: build_canonical_config(),
        }
    }
}
```

### 9.3 Critère de fin du plan

CLI s'arrête après avoir :
- ✅ Généré `phase9_decision.md`
- ✅ Préparé la config TOML pour le run (si GO)
- ✅ Vérifié l'espace disque disponible (~10 GB pour snapshots)
- ✅ Pushé toutes les modifications sur la branche `feature/treepm-integration`

**Commit final** : `treepm-phase9: décision <GO|NO-GO> + rapport complet`

**CLI s'arrête ici et attend l'humain.** Pas de lancement automatique du run 1M.

---

## ANNEXES

### A. Estimation des temps d'exécution par phase

| Phase | Estimation | Bloquant possible |
|---|---|---|
| 1. Audit | 1h | Non (pure lecture) |
| 2. Corrections numériques | 4-6h | Test convergence O(h⁴) capricieux |
| 3. Optims GPU | 6-8h | Table erfc/exp à calibrer |
| 4. Paramètres | 2-3h | Sweep peut être long |
| 5. Janus extension | 4-6h | Conventions Barnes-Hut à respecter ! |
| 6. Validation Newton | 2-3h | Génération snapshots Zel'dovich |
| 7. Validation Janus | 3-4h | Mini-run 100K = ~10 min × N validations |
| 8. Benchmarks | 2-3h | Profiling GPU |
| 9. Décision | 1h | Compilation rapport |
| **TOTAL** | **~25-37h** | **2-4 jours autonomes** |

### B. Liste de contrôle des références

À chaque commit, vérifier que les sources suivantes ont été correctement utilisées :

- [x] **PhotoNs-GPU** (Wang & Meng 2021, RAA) — Phase 3 essentiellement
- [x] **GADGET-4** (Springel 2021) — Architecture générale, Phase 1 audit
- [x] **GreeM** (Ishiyama 2009) — Paramètres Phase 4
- [x] **GrGadget** (Quintana-Miranda 2023) — Corrections Phase 2 + tests Phase 6
- [x] **Bagla 2002** — Splitting de force (référence secondaire)
- [x] **Springel 2005 (GADGET-2)** — Force splitting erfc (référence secondaire)
- [x] **Sefusatti 2016** — Correction CIC (référence Phase 2)
- [x] **Petit 2014, 2018** — Physique Janus (Phase 5)

### C. Glossaire

- **DKD** : Drift-Kick-Drift, schéma d'intégration symplectique
- **CIC** : Cloud-In-Cell, méthode d'interpolation particule↔grille
- **PBC** : Periodic Boundary Conditions
- **PM** : Particle-Mesh, méthode FFT pour la force longue portée
- **TreePM** : Hybride Tree (court) + PM (long)
- **P2P** : Particle-Particle, calcul direct de force entre paires
- **M2L** : Multipole-to-Local, FMM operator
- **VSL** : Variable Speed of Light
- **SoA / AoS** : Structure of Arrays / Array of Structures (layout mémoire)
- **FMA** : Fused Multiply-Add (`a*b + c` en une instruction)
- **SP / DP** : Single / Double Precision
- **SM** : Streaming Multiprocessor (unité GPU)

### D. En cas de blocage

Si CLI rencontre un problème non couvert par ce plan :

1. **Vérifier** que le problème ne vient pas d'une mauvaise interprétation locale
2. **Documenter** précisément dans `logs/treepm/blocker_<DATE>.md` :
   - Ce qui était tenté
   - L'erreur exacte
   - Les 3 fixes essayés
   - Hypothèses sur la cause
3. **Continuer** avec la phase suivante si possible (FLAG la phase bloquée)
4. **Ne pas abandonner** le plan global pour un seul blocage

Si plusieurs blocages graves : émettre un signal humain via `logs/treepm/STOP.md` et attendre.

---

**Fin du plan. Bonne route, CLI.**

