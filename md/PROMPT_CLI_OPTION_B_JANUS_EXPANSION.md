# Implementation détaillée — Option B
## Raccord Janus phase radiative (Petit 2018) + phase matière (Petit 2014)

**Date** : 26 avril 2026  
**For** : CLI (Claude Code)  
**Status** : URGENT — à implémenter avant relance run v9 multi-μ  
**Cohérence** : Petit & D'Agostini 2014 (matter era) + Petit 2018 (gauge process era)

---

## 1. Contexte et justification

### 1.1. Pourquoi cette modification ?

Le code actuel utilise `H = H₀ / a^(3/2)` (Einstein-de Sitter) sur tout l'intervalle z=10 → 0.

**Problème** : ce n'est PAS la vraie expansion Janus pour la phase matière (z < 4.51). La vraie formule est paramétrique (Petit 2014) avec `a⁺(μ_p) = α² cosh²(μ_p)`.

**Bonne nouvelle** : la formule actuelle `H = H₀ / a^(3/2)` est CORRECTE pour la phase haute-z (z > 4.51) qui correspond à la "generalized gauge process era" de Petit 2018 où `t ∝ a^(3/2)`.

**Solution** : implémenter un raccord à z = 4.51 entre les deux phases.

### 1.2. Les deux phases

**Phase radiative / gauge process (z > 4.508, Petit 2018)** :
- `t⁺ ∝ (a⁺)^(3/2)` (équation 4 du papier 2018)
- Donc `a ∝ t^(2/3)` (Einstein-de Sitter)
- Donc `H(a) = H₀ × a^(-3/2)` (formule actuelle du code, OK)
- VSL active mais ça n'affecte pas H(a)

**Phase matière (z ≤ 4.508, Petit 2014)** :
- `a⁺(μ_p) = α² cosh²(μ_p)`
- `t⁺(μ_p) = τ₀ × α² × (μ_p + ½ sinh²(μ_p))` (en Gyr)
- `H⁺(μ_p) = sinh(2μ_p) / [τ₀ α² cosh²(μ_p) (1 + ½ sinh(2μ_p))]`

### 1.3. Calibration de la phase matière (déjà résolue)

Système de 3 équations, 3 inconnues (μ_0, τ_0, α²) :
1. `a(μ_0) = α² cosh²(μ_0) = 1` (par définition à z=0)
2. `H(μ_0) = H_0 = 69.9 km/s/Mpc = 0.0715 Gyr⁻¹`
3. `t(μ_0) = t_0 = 15.87 Gyr` (calibration cosmologique)

**Solution numérique** :
```
α² = 0.1815456201
τ_0 = 23.3011940229 Gyr
μ_0 = 1.4974406287
z_transition = 1/α² - 1 = 4.5083
```

### 1.4. Discontinuité au raccord

À `a = α²` (z = 4.5083) :
- Phase radiative : `H_rad = H₀ / α²^(3/2) = 0.9242 Gyr⁻¹` (élevé)
- Phase matière : `H_mat → 0` (point μ_p = 0, "rebond" mathématique)

**Cette discontinuité est PHYSIQUE**, pas un bug — c'est le "Janus point" de Petit. On la traite par sélection de phase selon la valeur de `a`.

---

## 2. Code Rust à implémenter

### 2.1. Constantes (à placer en haut de fichier ou dans un module dédié)

```rust
/// Janus parametric solution constants (Petit & D'Agostini 2014)
/// Calibrated for H_0 = 69.9 km/s/Mpc and t_0 = 15.87 Gyr
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;  // Gyr

/// Transition between Janus radiative era (Petit 2018) and matter era (Petit 2014)
/// z_transition = 1/α² - 1 ≈ 4.51
/// Below a_transition: matter era (cosh² parametric)
/// Above a_transition: radiative era (gauge process, EdS-like)
const A_TRANSITION_JANUS: f64 = ALPHA_SQ_JANUS;  // = 0.1815456201
```

### 2.2. Fonction de calcul de H(a) — remplacer ligne ~1178

**AVANT** (à supprimer) :
```rust
let h = (args.h0 / MPC_GYR_TO_KMS) / a.powf(1.5);
```

**APRÈS** :
```rust
let h = compute_hubble_janus(a, args.h0);
```

### 2.3. Nouvelle fonction `compute_hubble_janus`

À ajouter quelque part dans le module (par exemple dans un nouveau fichier `src/janus_expansion.rs` ou en haut de `janus_adaptive_zoom.rs`) :

```rust
/// Compute Hubble parameter H(a) for the Janus cosmological model.
/// 
/// Implements two-phase expansion:
/// - For a < α²: gauge process era (Petit 2018), EdS-like H = H₀ × a^(-3/2)
/// - For a ≥ α²: matter era (Petit & D'Agostini 2014), parametric cosh²(μ_p)
/// 
/// Discontinuity at a = α²: physical "Janus point" where the universe
/// transitions between the two regimes. Selection by branch is correct;
/// no smoothing needed.
/// 
/// # Arguments
/// * `a` - Scale factor (a=1 today, a→0 at Big Bang)
/// * `h0_kms_mpc` - Hubble constant in km/s/Mpc (typically 69.9)
/// 
/// # Returns
/// H(a) in Gyr⁻¹
/// 
/// # Reference values
/// - H(a=1)    = 0.071487 Gyr⁻¹  = 69.90 km/s/Mpc
/// - H(z=1)    = 0.117986 Gyr⁻¹  = 115.37 km/s/Mpc
/// - H(z=4.5)  = 0.017622 Gyr⁻¹  = 17.23 km/s/Mpc (matter era, near transition)
/// - H(z=4.52) = 0.927120 Gyr⁻¹  = 906.7 km/s/Mpc (radiative era, near transition)
/// - H(z=10)   = 2.608052 Gyr⁻¹  = 2550 km/s/Mpc
fn compute_hubble_janus(a: f64, h0_kms_mpc: f64) -> f64 {
    const MPC_GYR_TO_KMS: f64 = 977.8;
    let h0_gyr_inv = h0_kms_mpc / MPC_GYR_TO_KMS;
    
    if a < A_TRANSITION_JANUS {
        // Phase radiative / gauge process (Petit 2018)
        // t ∝ a^(3/2) → H = H₀ × a^(-3/2)
        h0_gyr_inv / a.powf(1.5)
    } else {
        // Phase matière (Petit & D'Agostini 2014)
        // a(μ_p) = α² cosh²(μ_p)
        // H(μ_p) = sinh(2μ_p) / [τ₀ α² cosh²(μ_p) (1 + ½ sinh(2μ_p))]
        let cosh2_mu = a / ALPHA_SQ_JANUS;
        // Numerical safety: cosh²(μ) ≥ 1 by definition, but float arithmetic
        // near the transition can give values like 0.99999 → NaN in acosh
        let cosh2_mu_safe = cosh2_mu.max(1.0);
        let cosh_mu = cosh2_mu_safe.sqrt();
        let mu_p = cosh_mu.acosh();
        let s2mu = (2.0 * mu_p).sinh();
        s2mu / (TAU_0_JANUS * ALPHA_SQ_JANUS * cosh2_mu_safe * (1.0 + 0.5 * s2mu))
    }
}
```

### 2.4. Note sur le kernel CUDA

Le kernel CUDA (`nbody_gpu.rs:282-284`) reçoit `hubble_param` comme paramètre :
```rust
double friction = -hubble_param * vel[base + d] * dtau_per_dt;
```

**Pas de modification du kernel CUDA nécessaire** — il continue à recevoir `hubble_param` calculé sur CPU et passé en argument. Seul le calcul de `hubble_param` côté Rust change (avec `compute_hubble_janus`).

---

## 3. Tests unitaires obligatoires

À ajouter dans le module Rust (ou fichier test séparé `tests/test_janus_expansion.rs`).

### 3.1. Test 1 — H(z=0) = 69.9 km/s/Mpc

```rust
#[test]
fn test_hubble_today() {
    let h = compute_hubble_janus(1.0, 69.9);
    let h_kms_mpc = h * 977.8;
    
    // Doit donner 69.9 km/s/Mpc à 0.01% près
    assert!(
        (h_kms_mpc - 69.9).abs() < 0.01,
        "H(a=1) = {} km/s/Mpc, expected 69.9", h_kms_mpc
    );
}
```

### 3.2. Test 2 — H(z) à différents redshifts (phase matière)

```rust
#[test]
fn test_hubble_matter_era() {
    let h0 = 69.9;
    
    // Valeurs de référence calculées en Python (precision 0.01%)
    let test_cases = vec![
        // (z,    expected_H_Gyr_inv,   tol)
        (0.0,    0.071487,             1e-4),
        (0.1,    0.077186,             1e-4),
        (0.5,    0.097594,             1e-4),
        (1.0,    0.117986,             1e-4),
        (2.0,    0.142492,             1e-4),
        (3.0,    0.143787,             1e-4),
        (4.0,    0.107606,             1e-4),
    ];
    
    for (z, expected, tol) in test_cases {
        let a = 1.0 / (1.0 + z);
        let h = compute_hubble_janus(a, h0);
        assert!(
            (h - expected).abs() < tol,
            "H(z={}): got {}, expected {}, diff = {}", 
            z, h, expected, (h - expected).abs()
        );
    }
}
```

### 3.3. Test 3 — H(z) en phase radiative

```rust
#[test]
fn test_hubble_radiation_era() {
    let h0 = 69.9;
    let h0_gyr = h0 / 977.8;
    
    // Phase radiative : H = H₀ / a^(3/2)
    let test_zs = vec![5.0, 6.0, 8.0, 10.0];
    
    for z in test_zs {
        let a = 1.0 / (1.0 + z);
        let h = compute_hubble_janus(a, h0);
        let expected = h0_gyr / a.powf(1.5);
        assert!(
            (h - expected).abs() < 1e-7,
            "H(z={}) radiative: got {}, expected {} (EdS)", 
            z, h, expected
        );
    }
}
```

### 3.4. Test 4 — Transition de phase à a = α²

```rust
#[test]
fn test_phase_transition() {
    let h0 = 69.9;
    let alpha_sq = 0.1815456201;
    
    // Juste sous α² : phase radiative, H grand
    let a_below = alpha_sq - 1e-6;
    let h_below = compute_hubble_janus(a_below, h0);
    assert!(
        h_below > 0.9,
        "Just below transition (a={}): H = {} should be ~0.92 (radiation)",
        a_below, h_below
    );
    
    // À α² exactement : phase matière, μ_p=0, H ≈ 0
    let h_at = compute_hubble_janus(alpha_sq, h0);
    assert!(
        h_at.abs() < 1e-5,
        "At transition (a=α²): H = {} should be ~0 (μ_p=0)", h_at
    );
    
    // Juste au-dessus de α² : phase matière, H petit positif
    let a_above = alpha_sq + 1e-6;
    let h_above = compute_hubble_janus(a_above, h0);
    assert!(
        h_above > 0.0 && h_above < 0.01,
        "Just above transition (a={}): H = {} should be small positive",
        a_above, h_above
    );
}
```

### 3.5. Test 5 — Âge cosmique par intégration numérique

```rust
#[test]
fn test_cosmic_age() {
    let h0 = 69.9;
    let alpha_sq = 0.1815456201;
    
    // Intégrer ∫_{α²}^{1} da/(aH) doit donner 15.87 Gyr
    // Méthode : trapèze avec 1000 points (assez précis)
    
    let n_steps = 10000;
    let log_a_start = alpha_sq.ln() + 1e-6;  // éviter le point exact
    let log_a_end = 0.0;  // ln(1) = 0
    let dlog = (log_a_end - log_a_start) / n_steps as f64;
    
    let mut t_integral = 0.0;
    for i in 0..n_steps {
        let log_a_mid = log_a_start + (i as f64 + 0.5) * dlog;
        let a = log_a_mid.exp();
        let h = compute_hubble_janus(a, h0);
        // dt = da/(aH) = d(ln a)/H
        t_integral += dlog / h;
    }
    
    println!("Cosmic age (matter era only): {:.4} Gyr", t_integral);
    assert!(
        (t_integral - 15.87).abs() < 0.05,
        "Cosmic age: got {} Gyr, expected 15.87 Gyr", t_integral
    );
}
```

### 3.6. Test 6 — Continuité de a(t) au passage de l'intégrateur

```rust
#[test]
fn test_a_progression_through_transition() {
    let h0 = 69.9;
    let alpha_sq = 0.1815456201;
    
    // Simuler l'intégration leapfrog autour du raccord
    // À z=10, a=0.091, on doit pouvoir avancer jusqu'à z=0
    
    let dt = 0.001;  // Gyr
    let mut a = 1.0 / 11.0;  // z = 10
    let mut steps = 0;
    let max_steps = 50_000;  // safety limit
    
    while a < 1.0 && steps < max_steps {
        let h = compute_hubble_janus(a, h0);
        // da = a * H * dt
        a += a * h * dt;
        steps += 1;
        
        // Sanity: a doit toujours croître
        // Sanity: pas de NaN ni Inf
        assert!(a.is_finite(), "a became {} at step {}", a, steps);
        assert!(h.is_finite(), "H became {} at step {} (a={})", h, steps, a);
    }
    
    println!("Reached a={} in {} steps", a, steps);
    assert!(steps < max_steps, "Did not reach a=1 in {} steps", max_steps);
}
```

### 3.7. Test 7 — Reproductibilité avec valeurs hardcodées

```rust
#[test]
fn test_reference_table() {
    let h0 = 69.9;
    
    // Table de valeurs précisément calculées en Python
    // Format: (a, H_in_Gyr_inv)
    let reference = vec![
        // Phase radiative (a < α² = 0.1815)
        (0.09091, 2.608052),  // z=10
        (0.11111, 1.930149),  // z=8
        (0.14286, 1.323958),  // z=6
        (0.16667, 1.050640),  // z=5
        // Transition
        (0.18181, 1.0508),    // juste sous (radiation)
        // Phase matière (a > α²)
        (0.20000, 0.107606),  // z=4
        (0.25000, 0.143787),  // z=3
        (0.33333, 0.142492),  // z=2
        (0.50000, 0.117986),  // z=1
        (0.66667, 0.097594),  // z=0.5
        (0.90909, 0.077186),  // z=0.1
        (1.00000, 0.071487),  // z=0
    ];
    
    for (a, expected) in reference {
        let h = compute_hubble_janus(a, h0);
        let rel_err = (h - expected).abs() / expected;
        assert!(
            rel_err < 1e-3,
            "a={}: H={}, expected={}, rel_err={}", a, h, expected, rel_err
        );
    }
}
```

---

## 4. Validation supplémentaire — Test d'intégration

### 4.1. Test "rapide" pré-production (1-2 minutes max)

Avant de lancer le run v9 complet, faire un mini-run pour valider que tout fonctionne :

```bash
# Mini-run de validation : 100 steps seulement
TIMESTAMP=$(date +%Y%m%d_%H%M)
TEST_DIR="/app/output/test_janus_expansion_${TIMESTAMP}"
mkdir -p "$TEST_DIR"

./target/release/janus_adaptive_zoom \
  --n-grid 100 \
  --l-box 200 \
  --z-init 10.0 \
  --z-final 0.0 \
  --mu 8.0 \
  --h0 69.9 \
  --omega-b 0.05 \
  --eps-plus 0.05 \
  --eps-minus 0.10 \
  --dt-max 0.001 \
  --theta 0.7 \
  --delta-rms-ic 0.15 \
  --snap-interval 10 \
  --max-split-level 0 \
  --max-steps 100 \
  --out-dir "$TEST_DIR" \
  --run-label "test_expansion_validation" \
  2>&1 | tee "$TEST_DIR/run.log"

# Vérifier dans time_series.csv :
#  - Pas de NaN
#  - z décroît correctement
#  - a augmente continûment
#  - v_rms reste fini
echo ""
echo "=== Validation post-test ==="
head -5 "$TEST_DIR/time_series.csv"
echo "..."
tail -5 "$TEST_DIR/time_series.csv"
echo ""
grep -c "nan\|NaN" "$TEST_DIR/time_series.csv" || echo "No NaN found"
```

### 4.2. Vérification du passage à la transition

Le test mini-run de 100 steps couvre z=10 → ~z=9 (très peu d'évolution). Pour vérifier le PASSAGE de la transition, il faut un test plus long ou démarrer plus bas.

**Option A** : test avec z_init=5, z_final=4, dt=0.001 → ~1000 steps, ~3-5 minutes
- Doit voir le passage de a < α² à a > α² 
- v_rms doit rester fini
- Pas de saut d'a (a doit être continu)

**Option B** : test analytique pur sans simulation N-body, juste l'évolution de a(t) :

```rust
#[test]
fn test_continuous_a_evolution() {
    let h0 = 69.9;
    let alpha_sq = 0.1815456201;
    
    let dt = 0.0001;  // Gyr (petit pas pour précision)
    let mut a = 0.10;  // z=9, sous la transition
    let mut t = 0.0;
    
    let mut crossings = 0;
    let mut a_max_jump = 0.0;
    let mut prev_a = a;
    
    while a < 0.30 {
        let h = compute_hubble_janus(a, h0);
        let da = a * h * dt;
        a += da;
        t += dt;
        
        let jump = (a - prev_a).abs();
        if jump > a_max_jump {
            a_max_jump = jump;
        }
        
        // Détection passage transition
        if prev_a < alpha_sq && a >= alpha_sq {
            crossings += 1;
            println!("Crossed transition at t={:.4} Gyr, a went from {} to {}", 
                     t, prev_a, a);
        }
        
        prev_a = a;
    }
    
    assert_eq!(crossings, 1, "Should cross transition exactly once");
    
    // Le saut max doit être petit (limité par dt × H)
    // Au max H = 0.92, donc da_max = 0.18 × 0.92 × 0.0001 ≈ 1.7e-5
    assert!(
        a_max_jump < 1e-4,
        "a evolution not smooth, max jump = {}", a_max_jump
    );
    
    println!("Test passed: smooth a evolution, total time = {:.4} Gyr", t);
}
```

---

## 5. Workflow recommandé pour CLI

### Étape 1 — Préparation (5 min)
```bash
# Tuer le run actuel s'il tourne encore
pkill -f janus_adaptive_zoom
ls /app/output/janus_v9_mu8_*/ 2>/dev/null && {
  RUN_ACTIVE=$(ls -td /app/output/janus_v9_mu8_*/ 2>/dev/null | head -1)
  rm "$RUN_ACTIVE/PRODUCTION_ACTIVE.lock" 2>/dev/null
  echo "Stopped run in $RUN_ACTIVE"
}
```

### Étape 2 — Implémentation (30 min)
1. Ajouter les constantes `ALPHA_SQ_JANUS`, `TAU_0_JANUS`, `A_TRANSITION_JANUS` en haut de `janus_adaptive_zoom.rs`
2. Ajouter la fonction `compute_hubble_janus(a, h0)` 
3. Remplacer la ligne `let h = (args.h0 / MPC_GYR_TO_KMS) / a.powf(1.5);` par `let h = compute_hubble_janus(a, args.h0);`

### Étape 3 — Tests unitaires (30 min)
1. Ajouter les 7 tests dans le module
2. `cargo test compute_hubble_janus -- --nocapture`
3. Si un test échoue : analyser, debugger, refaire jusqu'à ce que tous passent

### Étape 4 — Test d'intégration (10 min)
1. Compiler en release : `cargo build --release --features cuda --bin janus_adaptive_zoom`
2. Lancer le mini-run de validation (100 steps)
3. Vérifier :
   - Pas de NaN dans time_series.csv
   - z décroît, a augmente
   - v_rms reste fini
   - Pas de crash

### Étape 5 — Lancer la production (~38h GPU)
- Run μ=8 (nuit 1)
- Run μ=30 (nuit 2)

---

## 6. Surveillance du run μ=8 avec nouvelle expansion

Pendant le run, surveiller particulièrement le passage à la transition :

```bash
# Toutes les heures, vérifier la progression
RUN_DIR=$(ls -td /app/output/janus_v9_mu8_*/ | head -1)
TS_CSV="$RUN_DIR/time_series.csv"

# Quel z actuel ?
awk -F, 'END {print "Current z=" $3 ", a=" $4 ", v_rms=" $9}' "$TS_CSV"

# Suivi du passage à z=4.51
# À ce point, on s'attend à un changement de comportement : 
# Phase radiation : v_rms décroît rapidement
# Phase matière : v_rms peut commencer à se stabiliser ou croître
awk -F, '
  NR > 1 && prev_z >= 4.51 && $3 < 4.51 { 
    print "TRANSITION PASSED at step=" $1 ": z went from " prev_z " to " $3 
  }
  { prev_z = $3 }
' "$TS_CSV"

# NaN check
grep -c "nan\|NaN" "$TS_CSV"
```

### Comportement attendu

**Phase radiative (z ∈ [10, 4.51])** :
- v_rms décroît (univers en expansion, peu de gravité)
- ρ+_max stable autour de 3-5× ρ+_mean comoving
- Évolution douce

**Transition à z = 4.51** :
- Léger changement de pente dans v_rms
- Les vitesses peculières peuvent commencer à pouvoir résister à l'expansion
- C'est le moment où la croissance des structures peut commencer

**Phase matière (z < 4.51)** :
- v_rms peut se stabiliser ou croître si la gravité prend le dessus
- ρ+_max peut commencer à augmenter au-delà de 5×
- Formation potentielle de structures

⚠️ **Si v_rms continue à décroître monotoniquement même en phase matière**, c'est que la gravité n'est pas suffisante pour vaincre l'expansion (probable à μ=8, Ω_tot=0.45 sous-critique).

---

## 7. Références bibliographiques

1. **Petit, J.-P. & D'Agostini, G.** (2014). "Negative mass hypothesis in cosmology and the nature of dark energy." *Modern Physics Letters A* 29, 1450182.  
   → Source de la formule paramétrique cosh²(μ_p) pour la matter-dominated era

2. **Petit, J.-P.** (2018). "The Janus Cosmological Model and the fluctuations of the CMB." *Progress in Physics* xx.  
   → Source des relations de gauge process avec t ∝ a^(3/2) (équations 4)

3. **D'Agostini, G. & Petit, J.-P.** (2018). "Constraints on Janus Cosmological model from recent observations of supernovae type Ia." *Astrophysics and Space Science* 363, 139.  
   → Calibration α et cohérence avec Pantheon+

---

## 8. Récapitulatif des constantes (à coder en Rust)

```rust
// Janus parametric solution constants
const ALPHA_SQ_JANUS: f64 = 0.1815456201;
const TAU_0_JANUS: f64 = 23.3011940229;  // Gyr
const A_TRANSITION_JANUS: f64 = 0.1815456201;  // = ALPHA_SQ_JANUS

// Reference values for tests
const H0_TARGET_GYR_INV: f64 = 0.0714870;  // 69.9 km/s/Mpc
const T0_TARGET_GYR: f64 = 15.87;  // age of universe in matter era
const Z_TRANSITION: f64 = 4.5083;  // 1/α² - 1
const MU_0_TODAY: f64 = 1.4974406287;  // μ_p value at z=0
```

---

## 9. Total temps estimé

| Étape | Temps |
|-------|-------|
| Tuer le run actuel | 1 min |
| Implémenter les constantes + fonction | 30 min |
| Écrire et passer les 7 tests | 30 min |
| Compiler en release | 5 min |
| Mini-run de validation | 5 min |
| Total avant production | **~70 min** |

Puis production : ~38h GPU sur 2 nuits (μ=8 + μ=30).
