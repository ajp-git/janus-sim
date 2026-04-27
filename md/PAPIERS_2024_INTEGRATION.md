# Intégration des papiers 2024 dans le code Janus

**Sources** : Petit-Margnat-Zejli (arXiv:2412.04644v3, déc. 2024) + Petit-Zejli (mai 2024).
**Cible** : avant lancement du run de production μ=8.
**Effort estimé** : +2h sur les 70 min déjà prévues pour H(z).

---

## Résumé exécutif — 3 modifications physiques + 1 monitoring

| # | Modif | Priorité | Effort | Impact run |
|---|-------|----------|--------|------------|
| 1 | Facteur de couplage cube `Φ = (ā/a)³` sur forces inter-espèces | **A — critique** | 45 min | Modifie l'amplitude des forces, peut tout changer |
| 2 | Tracking de `E = ρ⁺c²a⁺³ + ρ⁻c̄²a⁻³` dans CSV | **A — critique** | 15 min | Diagnostic essentiel, doit rester constant |
| 3 | Floor de pression m⁻ à T_reion = 10⁴ K (sans cooling) | **B — défer** | 60 min | Résout le runaway m⁻ à z<0.19 |
| 4 | Calcul lentille gravitationnelle négative (post-prod) | **C — bonus** | 90 min | Prédiction falsifiable forte pour preprint |

**Décision recommandée** : faire **A** maintenant (1+2), reporter **B** après le run μ=8 si runaway, et **C** en post-production sur les snapshots.

---

## Modif #1 — Facteur de couplage `(ā/a)³`

### Source

Eq. 92-93 du long papier :

```
R_μν − ½ g_μν R   =  χ ( T_μν   + √(|ḡ|/|g|) · T̃_μν  )
R̄_μν − ½ ḡ_μν R̄  = −χ̄ ( T̄_μν  + √(|g|/|ḡ|) · T̃̄_μν )
```

Eq. 95-96 + 2.7 (papier court) : pour métriques FLRW homogènes,

```
√(|ḡ|/|g|) = (ā/a)³        Φ(t) = (ā(t) / a(t))³
√(|g|/|ḡ|) = (a/ā)³        φ(t) = (a(t) / ā(t))³ = 1/Φ(t)
```

**C'est le cube du rapport des facteurs d'échelle, pas le rapport linéaire.**

### Pourquoi c'est important

La force inter-espèces qu'une particule m⁻ exerce sur une particule m⁺ doit être modulée par `Φ(t) = (ā/a)³`. Symétriquement, m⁺ → m⁻ par `1/Φ(t)`.

À z=4 avec η=1.045 : ā/a ≈ 1.04 → Φ ≈ 1.13 (effet 13%)
À z=0  avec η=1.045 : ā/a ≈ 1.00 → Φ ≈ 1.00 (effet nul)

Si dans le code actuel on utilise un couplage symétrique sans cube, on **sous-estime** la répulsion inter-espèces de ~13% à grands z, là où la ségrégation se met en place. C'est peut-être pourquoi l'effet observé est si subtil.

### Modification code

Dans `cosmology.rs` (ou équivalent), ajouter une fonction :

```rust
/// Facteur de couplage croisé Janus : Φ(t) = (ā/a)³
/// Utilisé pour moduler la force qu'exerce m⁻ sur m⁺.
/// Le facteur inverse 1/Φ module la force m⁺ sur m⁻.
#[inline]
pub fn phi_coupling(a_plus: f64, a_minus: f64) -> f64 {
    let ratio = a_minus / a_plus;
    ratio * ratio * ratio
}

#[inline]
pub fn phi_coupling_inv(a_plus: f64, a_minus: f64) -> f64 {
    let ratio = a_plus / a_minus;
    ratio * ratio * ratio
}
```

Dans la boucle de calcul des forces (kernel CUDA `compute_cross_forces` ou équivalent) :

```rust
// AVANT (ce qu'on a probablement aujourd'hui)
let f_cross = G * m_other / r2;     // répulsive si signes opposés

// APRÈS
let phi = phi_coupling(a_plus, a_minus);    // sur particule m+ depuis m-
// ou
let phi_inv = phi_coupling_inv(a_plus, a_minus);  // sur particule m- depuis m+

let f_cross = G * m_other * phi / r2;       // m+ subissant m-
let f_cross = G * m_other * phi_inv / r2;   // m- subissant m+
```

**Note importante** : le facteur Φ est passé au GPU comme un scalaire (pas modifier le kernel, juste passer 2 floats dans la struct CosmoParams).

### Test unitaire

```rust
#[test]
fn test_phi_coupling_is_cube() {
    // À z=0 avec ā = a → Φ = 1
    assert!((phi_coupling(1.0, 1.0) - 1.0).abs() < 1e-12);
    // ā/a = 2 → Φ = 8
    assert!((phi_coupling(1.0, 2.0) - 8.0).abs() < 1e-12);
    // Symétrie : φ × 1/φ = 1
    let a_p = 0.5;
    let a_m = 0.6;
    assert!((phi_coupling(a_p, a_m) * phi_coupling_inv(a_p, a_m) - 1.0).abs() < 1e-12);
}

#[test]
fn test_phi_typical_values() {
    // À η=1.045, z=4 : ā/a ≈ 1.04 → Φ ≈ 1.125
    let a_p_z4 = 0.2;
    let a_m_z4 = 0.208;  // 1.04 × a_plus
    let phi = phi_coupling(a_p_z4, a_m_z4);
    assert!((phi - 1.124864).abs() < 1e-4);
}
```

---

## Modif #2 — Tracking de l'énergie totale `E`

### Source

Eq. 96 long papier / eq. 2.8 papier court :

```
E = ρ⁺ · c²  · a⁺³  +  ρ⁻ · c̄² · a⁻³  =  constante
```

Cet invariant **doit** être conservé strictement dans toute simulation Janus correcte. C'est la conservation d'énergie bimétrique généralisée.

**E doit être négatif** pour avoir l'accélération de a⁺ observée (eq. 98a-b). Avec μ=8 : ρ⁻/ρ⁺ = 8 et c̄/c > 1, donc E_neg > E_pos en module → E < 0. ✓

### Modification code

Dans le module qui écrit `time_series.csv`, ajouter le calcul :

```rust
fn compute_total_energy(
    rho_plus: f64,    // densité moyenne m+ comobile
    rho_minus: f64,   // densité moyenne m- comobile (signée < 0)
    c_plus: f64,      // c (constante, = 1 en unités code)
    c_minus: f64,     // c̄(t), VSL dynamique
    a_plus: f64,
    a_minus: f64,
) -> f64 {
    let e_plus  = rho_plus  * c_plus  * c_plus  * a_plus.powi(3);
    let e_minus = rho_minus * c_minus * c_minus * a_minus.powi(3);
    e_plus + e_minus
}
```

Ajouter au header CSV : `E_total, E_plus, E_minus, E_drift_pct`

```
E_drift_pct = (E_total(t) - E_total(t=0)) / E_total(t=0) × 100
```

### Critère de validation

| z | E_drift attendu |
|---|-----------------|
| 4.0 → 0 | < 0.1 % sur tout le run |
| Si > 1 % | bug dans l'intégration ou les forces |
| Si > 5 % | run invalide, à arrêter |

### Test unitaire

```rust
#[test]
fn test_E_negative_with_mu_canonical() {
    // μ=8 canonique : ρ⁻ ≫ ρ⁺ en module
    let rho_p = 1.0;
    let rho_m = -8.0;       // μ=8
    let c_p = 1.0;
    let c_m = 1.05;          // VSL c̄ > c
    let a_p = 0.5;
    let a_m = 0.5;
    let e = compute_total_energy(rho_p, rho_m, c_p, c_m, a_p, a_m);
    assert!(e < 0.0, "E doit être négatif pour expansion accélérée");
}
```

---

## Modif #3 — Floor de pression m⁻ à T_reion (B, à différer)

### Source

Section 11 long papier :

> *"As soon as their temperature causes hydrogen reionization, their contraction will cease. […] their cooling time is then large compared to the age of the universe, which means that these objects will no longer evolve."*

### Diagnostic du runaway actuel

Notre run principal montrait un runaway m⁻ à z<0.19 — c'est bien le collapse physique sans plancher thermique. Le papier dit explicitement que ce collapse **doit s'arrêter** quand T atteint la réionisation H (~10⁴ K), puis figé pour le reste de l'histoire cosmique.

### Approche

Pas de SPH complète sur m⁻ (on a vu que ça déstabilise — Options A/B abandonnées). Juste un **floor de pression** :

```rust
const T_REION_MINUS: f64 = 1e4;  // K, plancher thermique m-

// Dans la boucle SPH minimale m- (juste pression, pas cooling, pas SF)
for particle in m_minus_particles {
    let T = max(T_REION_MINUS, particle.T);  // floor
    let p_thermal = (k_B / m_p) * particle.rho * T;
    // Force de pression : -∇p / ρ
    apply_pressure_gradient(particle, p_thermal);
}
```

**Différence cruciale avec Option A/B précédentes** : pas de cooling radiatif, pas de star formation, juste un floor T → pression. C'est minimaliste et stable.

### Critère de succès

Run μ=8 jusqu'à z=0 sans crash, ρ⁻_max plafonne au lieu de diverger.

### Pourquoi reporter ?

- Le run μ=8 actuel n'a pas encore tourné — on ne sait pas si le runaway revient
- Si μ=8 tourne propre jusqu'à z=0, on n'a pas besoin de cette modif
- Plus simple d'ajouter en réaction à un problème observé qu'en spéculatif

---

## Modif #4 — Lentille gravitationnelle négative (C, post-production)

### Source

Section 13 long papier + figure 14. Prédiction falsifiable, utilise les métriques eq. 122 (Schwarzschild intérieur m⁻) et eq. 128.

Magnitude attenuée d'une source d'arrière-plan en **anneau** autour du centre du vide m⁻ — pas atténuation uniforme, anneau.

### Pour le preprint

C'est une signature qui distingue Janus de tout autre modèle. Si on a un gros vide m⁻ identifié dans nos snapshots z=0, on peut :

1. Localiser le centre + rayon du conglomérat m⁻
2. Tracer les géodésiques d'une source d'arrière-plan
3. Calculer Δm(b) = atténuation en fonction du paramètre d'impact b
4. Produire la figure de l'anneau d'atténuation

C'est un script Python post-prod, pas une modif du code de simu. À faire sur le snapshot final z=0 du run μ=8.

---

## Workflow CLI proposé

### Étape 0 — Avant tout

CLI termine d'abord les 70 min H(z) : implémentation + 7 tests + mini-run validation. **NE PAS lancer μ=8 production tant que les modifs A ne sont pas en place.**

### Étape 1 — Modif #1 cube (45 min)

1. Ajouter `phi_coupling()` et `phi_coupling_inv()` dans `cosmology.rs`
2. Modifier le calcul des forces inter-espèces dans le kernel CUDA (juste 2 multiplications)
3. Passer `a_plus, a_minus` au kernel via `CosmoParams`
4. Tests unitaires Φ (cube, symétrie, valeurs typiques)

### Étape 2 — Modif #2 énergie (15 min)

1. Ajouter `compute_total_energy()`
2. Étendre le header CSV
3. Calculer et logger à chaque snapshot
4. Test unitaire E < 0 pour μ=8

### Étape 3 — Mini-run validation combiné (10 min)

100 steps avec H(z) + Φ + E tracking. Critères :

- H(z) : transition Janus point sans NaN ✓ (déjà testé)
- Φ : valeur affichée à chaque step, doit être proche de 1 mais pas exactement 1
- E_drift : < 0.1% sur 100 steps
- E_total : < 0 (signe correct)

### Étape 4 — Production μ=8

Si tous les diagnostics OK : lancer la production avec configuration vsl_petit_production (qu'on connaît) **+ ces 2 modifs**.

### Étape 5 — Décision sur Modif #3

Pendant que μ=8 tourne (à mi-run, vers z~1.5), examiner v_rms⁻ et ρ⁻_max :

- Si stable → pas besoin de Modif #3, on poursuit
- Si signes de runaway → préparer Modif #3 en parallèle, prête à appliquer pour μ=30

---

## Prompt prêt pour CLI

```
Lire PAPIERS_2024_INTEGRATION.md en entier.

Ordre d'exécution :
1. Terminer d'abord H(z) Janus 2014+2018 (déjà en cours, ~70 min)
2. Implémenter Modif #1 (couplage cube Φ = (ā/a)³) — 45 min
3. Implémenter Modif #2 (tracking E_total) — 15 min
4. Tests unitaires des deux modifs — déjà décrits dans le doc
5. Mini-run combiné 100 steps avec validation
6. STOP — me rapporter les résultats avant de lancer μ=8 production

Pour chaque modif :
- Afficher le diff exact des fichiers modifiés
- Lancer les tests unitaires dédiés
- Confirmer "GO" avant la suivante

Modifs #3 et #4 NE PAS implémenter maintenant — décision après run μ=8.
```

---

## Références

- arXiv:2412.04644v3 — Petit, Margnat, Zejli (2024). « A bimetric cosmological model based on Andreï Sakharov's twin universe approach »
- Petit, Zejli (mai 2024). « Janus Cosmological Model — Mathematically & Physically Consistent »
- Eq. clés : 92-93 (champ couplé), 95-96 (compatibilité FLRW), 122-129 (Schwarzschild bimétrique), 2.7 papier court (forme cube)

---

*Document préparé pour intégration avant production μ=8 — avril 2026*
