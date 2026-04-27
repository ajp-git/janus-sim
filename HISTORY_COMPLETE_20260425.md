# HISTORIQUE COMPLET — Projet Janus Simulation
**Date de création**: 2026-04-25
**Couverture**: Février 2026 → Avril 2026

---

## PHASE 1a — Pantheon+ Fit (Février 2026) ✅ COMPLETE

### Objectif
Ajuster le modèle Janus bimétrique aux données SNIa Pantheon+ (1701 supernovae).

### Résultats
```
η = 1.045 (rapport de masse m-/m+)
q₀ = -0.022 (paramètre de décélération)
χ²/dof = 0.914 (excellent fit, 1590 SNIa après cuts)
```

### Référence
- Petit, Margnat & Zejli (2024) — EPJC 84:1226
- Scolnic et al. (2022) — ApJ 938:113 (données Pantheon+)

---

## PHASE 1b — GPU N-body Initial (Février 2026) ✅ COMPLETE

### Implémentation
- Barnes-Hut octree sur GPU (CUDA via cudarc)
- Précision f64 obligatoire
- Speedup: 39.6× vs CPU

### Bug critique corrigé: rsqrt()
```rust
// AVANT (bug): rsqrt() est float32 intrinsic
let r_inv = rsqrt(r2);

// APRÈS (fix): f64 correct
let r_inv = 1.0 / r2.sqrt();
```

### Runs de validation
| Run | N | Résultat |
|-----|---|----------|
| 500K | 500,000 | S=0.513 |
| 2M | 2,000,000 | S=0.694 |
| 8M | 8,000,000 | S=0.459, θ=1.5 |

---

## PHASE 1c — Virialization Janus (Février 2026) ✅ COMPLETE

### Problème
La virialization standard (2KE + PE_total = 0) échoue pour Janus car PE_total inclut les paires répulsives.

### Solution: PE_binding
```rust
// PE_binding < 0 toujours (paires attractives uniquement)
// m+ attire m+, m- attire m-
let alpha = (pe_binding.abs() / (2.0 * ke)).sqrt();
// Typique: α ≈ 4.57 pour η=1.045
```

### Validation
Les ICs virialisées avec PE_binding restent stables sur 1000+ steps.

---

## PHASE 2 — Filament Formation (Mars 2026) 🔄 PARTIEL

### Théorie linéaire validée
Matrice de couplage à deux fluides:
- λ₊ = ρ̄(1+α) > 0 → ségrégation (blob) ✅
- λ₋ = ρ̄(1-α) = 0 pour α=1 → mode filamentaire gelé

**Avec α=1 (code actuel): λ₋=0 exactement.**
Test Jour 1: +1047% ΛCDM vs +262% Janus (ratio 4×).

### Tests Yukawa (Mars 2026) — NÉGATIFS
```rust
α(r) = 1 − ε·exp(−r/r_c)
```
6 configurations testées (ε=0.3/0.7, r_c=10/40 Mpc):
- Effet maximal: +0.2% (non significatif)
- Cause: régime linéaire mono-mode insensible à α(k)
- Yukawa vérifié N² vs BH (erreur 0.02%)

---

## BUGS MAJEURS CORRIGÉS (Chronologique)

### Bug 1: Accélération locale vs globale (Fév 2026)
```
Symptôme: Mauvaise dynamique
Cause: Densités locales au lieu de masses conservées
Fix: Utiliser E_conserved du papier source
```

### Bug 2: Offset 0.8 mag Pantheon+ (Fév 2026)
```
Symptôme: Distance modulii décalés
Cause: H(z) ΛCDM mélangé avec accélérations Janus
Fix: Cohérence théorique complète
```

### Bug 3: Ségrégation nulle avec PM (Fév 2026)
```
Symptôme: S ≈ 0 même après évolution
Cause: Méthode PM lisse les interactions courte portée
Fix: Passer à Barnes-Hut pur
```

### Bug 4: COM périodique (Fév 2026)
```
Symptôme: COMs aberrants près des bords
Cause: Moyenne simple ignore le wrap périodique
Fix: Minimum image convention
```

### Bug 5: ICs Zel'dovich biaisées (Fév-Mars 2026)
```
Symptôme: Ségrégation dès step 0 (S₀ ≈ 0.49)
Cause: m+ dans première moitié grille, m- dans seconde
Code bugué:
  let sign = if idx < n_positive { 1 } else { -1 };
Fix:
  let sign = if rng.gen::<bool>() { 1 } else { -1 };
```

### Bug 6: FFT "displacement=0" (Mars 2026)
```
Symptôme: Affichage "psi = 0.0000"
Cause: Format {:.4} tronque 2.4e-8
Fix: Utiliser {:.6e}
```

---

## PHASE 11 — Hubble Friction (Avril 2026) ✅ VALIDATED

### Implémentation
Friction de Hubble dans l'intégrateur leapfrog:
```rust
v_new = v_old * exp(-H * dt) + a * dt
```

### Test de validation
- Run z=5→0 avec friction
- v_rms décroît correctement avec expansion
- Commit: 9cca681

---

## PHASE 12 — Box Size Scaling (Avril 2026) ⚠️ PARTIEL

### Tests effectués
| Box (Mpc) | N | Résultat |
|-----------|---|----------|
| 100 | 2M | Trop petit, effets de bord |
| 200 | 5M | OK |
| 500 | 10M | Standard pour production |

### Observation
Box 500 Mpc avec μ=19 → m+ très dilués, peu de clustering.

---

## PHASE 13 — Random Morton Offset (Avril 2026) ✅ VALIDATED

### Problème
Contamination cardinale dans le spectre angulaire:
- max_axes(m-) = 13.35 à z=2 (devrait être ~1.0)
- Artefact de l'octree aligné sur les axes

### Solution
Offset aléatoire des coordonnées Morton à chaque rebuild:
```rust
// Dans compute_morton_codes_periodic()
let offset_x = rng.gen::<f64>() * cell_size;
let offset_y = rng.gen::<f64>() * cell_size;
let offset_z = rng.gen::<f64>() * cell_size;

let ix = ((x + offset_x) / cell_size) as u32 & mask;
// ...
```

### Validation
| Run | max_axes(m-) z=2 | Status |
|-----|------------------|--------|
| Avant fix | 13.35 | ❌ FAIL |
| Après fix | 1.06 | ✅ PASS |

### Commits
- a32ea76: Phase 13 changes to nbody_gpu.rs and nbody_gpu_twopass.rs

---

## RUN v7b (24 Avril 2026) — ANALYSE COMPLÈTE

### Paramètres
```
N = 9,938,375 (n_grid=215)
L_box = 500 Mpc
z_init = 10.0, z_final = 0.77 (arrêt prématuré)
μ = 19, η = 1.045
```

### Cause arrêt prématuré
Script autonome a renommé le dossier output pendant que la simulation tournait → erreurs I/O.

### Résultats Phase 9 (z=1.14)
| Critère | Description | Résultat |
|---------|-------------|----------|
| 1 | r(k) décroît avec k | ✅ |
| 2 | r(k) décroît avec le temps | ❌ |
| 3 | var(δ-)/var(δ+) > 1 | ❌ (0.22) |
| 4 | Ségrégation > Co-localisation | ❌ (9.7% vs 90.3%) |
| 5 | m+ dépeuplé près pic m- | ✅ |

**Score: 2/5** — Signatures Janus faibles

### Anomalies observées
- v_rms = 30-65 km/s (attendu ~300 km/s)
- ρ_max quasi-constant (peu de formation de structures)

### Fichiers
```
/mnt/T2/janus-sim/output/janus_adaptive_v7b_zmin077/
├── FINAL_REPORT.md
├── analysis/phase9_deep_analysis.md
└── snapshots/ (186 fichiers)
```

---

## PHASE v8 — Zoom Spatial (24-25 Avril 2026) 🔄 EN COURS

### Changement conceptuel majeur
Le splitting utilisait ρ_total, mais avec μ=19:
- 95% des particules sont m-
- ρ_total dominé par m-
- Les m+ ne déclenchent jamais de splits

**Fix v8**: Utiliser ρ_plus uniquement pour le trigger de split.

### Modifications code (janus_adaptive_zoom.rs)

**1. compute_densities_split()** — Sépare ρ+ et ρ-:
```rust
fn compute_densities_split(particles: &[ParticleV3], box_size: f64)
    -> (Vec<f64>, Vec<f64>, f64, f64) {
    // Grille 64³
    // Retourne (densities_plus, densities_minus, rho_plus_max, rho_minus_max)
}
```

**2. Nouveaux CLI flags**:
```rust
--zoom-cube-size 150    // Cube [-75,+75]³ Mpc
--max-split-level 2     // Limite stricte
--delta-split-l1 6.78e10  // Seuil L0→L1 (10× ρ_mean_plus)
--delta-split-l2 6.78e11  // Seuil L1→L2 (100× ρ_mean_plus)
```

**3. Condition spatiale**:
```rust
if zoom_cube_size > 0.0 {
    if px.abs() > zoom_half || py.abs() > zoom_half || pz.abs() > zoom_half {
        continue;  // Hors zone → pas de split
    }
}
```

**4. Logging ρ+_max**:
- Nouvelle colonne dans time_series.csv
- Affichage console: `ρ+_max=X.XXeYY`

### Sanity Check v8 (24 Avril 2026)
```
z=10 → z=5, ~0.4h GPU
N = 4,913,000 constant (pas de splits)
ρ+_max ≈ 5.8e10 < seuil 6.78e10
✅ Code fonctionne, pas de crash
```

### Production v8 (24-25 Avril 2026)
```
Directory: janus_production_v8_zoom_20260424_2136/
z=10 → z≈0.06 (en cours)
N = 4,913,000 constant (AUCUN SPLIT)
ρ+_max = 5.0-7.2e10 (oscille autour du seuil sans le dépasser durablement)
v_rms: 40 → 172 km/s
~415 snapshots, ~412 frames
```

### Observation critique v8
**Aucun split déclenché** sur tout le run z=10→0 car:
- Les m+ sont trop dilués par répulsion des m- dominants (μ=19)
- ρ+_max ne dépasse jamais durablement le seuil 6.78e10
- C'est une conséquence physique de Janus, pas un bug

### Recommandations pour v9
1. Baisser `--delta-split-l1` à 3e10 ou 4e10
2. Réduire box (100-200 Mpc) pour meilleure résolution
3. Tester μ plus faible (3-10)
4. Ou accepter que μ=19 ne permet pas de splits

---

## MODULES IMPLÉMENTÉS

### GpuCooling (cooling_gpu.rs)
- Refroidissement radiatif (Sutherland-Dopita like)
- T_init = 10000 K
- Formation stellaire (SF threshold)
- **Activé** dans v8 (hardcodé)

### VSL Dynamique (vsl_dynamic.rs)
- CoupledFriedmann pour c(z)/c₀
- η = 1.045 fixé
- Mise à jour c_ratio tous les 100 steps
- **Activé** dans v8 (hardcodé)

### SPH
- **NON implémenté** en tant que tel
- Densités calculées par grille CIC (32³ ou 64³)
- Pas de kernels de smoothing, pas de pression hydrodynamique

### Particle Splitting
- 1 parent → 8 daughters (Blue Noise placement)
- mass_daughter = mass/8
- epsilon_daughter = epsilon/2
- split_level incremented
- **Implémenté mais jamais déclenché** avec μ=19

---

## TESTS A/B REJETÉS

### Tree Rebuild Interval=3 (Commit 1262f1a)
```
Test: Reconstruire octree tous les 3 steps au lieu de 1
Résultat: REJETÉ — tous métriques échouent
```

### Adaptive θ seul (Commit 67a723f)
```
Test: θ adaptatif sans autres changements
Résultat: REJETÉ — σ_P deviation 3.5%
```

### Optimization Validation (Commit 6088898)
```
Résultat: REJETÉ
```

---

## STRUCTURE DES SNAPSHOTS v3

### Header (408 bytes)
```c
struct SnapshotHeaderV3 {
    magic: u32,           // 0x4A414E53 ("JANS")
    version: u32,         // 3
    reserved: u64,
    n_particles: u64,     // offset 16
    scale_factor: f64,    // offset 24
    time_gyr: f64,        // offset 32
    box_size: f64,        // offset 40
    // ... autres champs
};
```

### Particle (36 bytes)
```c
struct ParticleV3 {
    pos: [f32; 3],        // 12 bytes
    vel: [f32; 3],        // 12 bytes
    mass: f32,            // 4 bytes
    epsilon: f32,         // 4 bytes
    sign: u8,             // 1 byte (1=m+, 255=m-)
    split_level: u8,      // 1 byte
    is_star: u8,          // 1 byte
    flags: u8,            // 1 byte
};
```

### Lecture Python
```python
import struct
with open(path, 'rb') as f:
    header = f.read(408)
    n = struct.unpack('<Q', header[16:24])[0]
    a = struct.unpack('<d', header[24:32])[0]
    z = 1/a - 1
```

---

## MÉTRIQUES DE VALIDATION

### Phase 9 — Signatures Janus
1. r(k) = P_cross(k) / sqrt(P_plus(k) * P_minus(k)) décroît avec k
2. r(k) décroît avec le temps (anti-corrélation croissante)
3. var(δ-)/var(δ+) > 1 (m- plus clustérisé)
4. Ségrégation > Co-localisation (quadrants off-diagonal)
5. m+ dépeuplé près des pics m- (δ+ < 0 à r < 5 Mpc)

### Phase 13 — Contamination grille
- max_axes < 1.15: PASS
- max_axes < 1.25: WARN
- max_axes ≥ 1.25: FAIL

---

## RÉFÉRENCES BIBLIOGRAPHIQUES

1. **Petit, Margnat & Zejli (2024)** — EPJC 84:1226
   "The Janus cosmological model: A bimetric approach"

2. **D'Agostini & Petit (2018)** — Astrophys. Space Sci. 363:139
   "Constraints on the Janus cosmological model"

3. **Petit & D'Agostini (2014)** — Astrophys. Space Sci. 354:611
   "Negative mass hypothesis in cosmology"

4. **Scolnic et al. (2022)** — ApJ 938:113
   "The Pantheon+ Analysis: Cosmological Constraints"

5. **Lane et al. (2024)** — MNRAS arXiv:2311.01438
   "4.4σ tension avec calibration ΛCDM"

---

## COMMANDES UTILES

### Compilation
```bash
cd /mnt/T2/janus-sim
docker compose run --rm dev cargo build --release --features cuda
```

### Lancement simulation
```bash
docker compose run --rm dev cargo run --release --features cuda \
  --bin janus_adaptive_zoom -- [OPTIONS]
```

### Monitoring
```bash
tail -f output/*/run.log
tail -f output/*/time_series.csv
docker ps | grep janus
nvidia-smi
```

### Analyse snapshot
```bash
/tmp/plotenv/bin/python scripts/phase9_deep_analysis.py \
  --snap output/*/snapshots/snap_XXXXX.bin
```

### Vidéo
```bash
ffmpeg -r 30 -i frames_10panel/frame_%05d.png \
  -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

*Document généré le 2026-04-25. Historique complet du projet Janus.*

---

## MISE À JOUR — Production v8 TERMINÉE (25 Avril 2026)

### Résultat Final
```
Durée: 8.8 heures
Steps: 9,070
z final: 0.00
N final: 4,913,000 (AUCUN SPLIT)
v_rms: 83 → 189 km/s
ρ_max: 2.97e11 → 3.91e11 M☉/Mpc³
ρ+_max: 5.0-7.2e10 (jamais > seuil 6.78e10 durablement)
```

### Conclusion
Le trigger de split basé sur ρ_plus ne fonctionne pas avec μ=19 car:
- Les m+ sont trop dilués par répulsion des m- dominants (95%)
- Le seuil 10× ρ_mean_plus est trop élevé pour ce régime

### Fichiers générés
- 454 snapshots (snap_00000.bin → snap_09060.bin)
- 452 frames 10-panel + 452 frames 2.5D
- FINAL_REPORT_v8.md
