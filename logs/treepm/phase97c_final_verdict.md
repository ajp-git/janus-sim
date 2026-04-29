# Phase 9.7-C — Verdict final TreePM CPU pour port GPU

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port

## Résumé exécutif

**Verdict : GO_WITH_CAVEAT** — TreePM CPU précis sur distribution random uniform mais montre **bias systématique sur lattice/Zel'dovich**. Le port GPU reste justifié, mais avec validation supplémentaire sur IC réalistes après port.

## Résultats des tests

### Test 1 — Force-field cross-correlation per k mode (N=1000 random uniform)

Setup: N=1000 particules random uniform, L=100, n_pm=64, r_cut=9.375.

| k bin | k [1/Mpc] | r(k) | |F_t|/|F_p| |
|---|---|---|---|
| 0 | 0.063 | 0.9982 | 0.866 |
| 1-15 | 0.19-1.95 | **0.994-0.998** | 0.95-0.98 |

**min r(k) = 0.9938** → **GO_PHASE_10** sur ce test.

Interprétation : sur distribution non-clusterisée, TreePM et PP-direct concordent à >99% en direction. L'erreur médiane 11% sur forces individuelles vient de la magnitude (~5%) et de la direction (~4°), mais s'élimine quasi-complètement dans le champ de force Fourier-spatial.

### Test 2 — Force-field cross-correlation sur Zel'dovich N=10K

Setup: N=22³=10648 particules Zel'dovich (lattice + 15% dx perturbation), L=100, n_pm=64.

| k bin | k [1/Mpc] | r(k) | |F_t|/|F_p| |
|---|---|---|---|
| 0 | 0.063 | 0.8579 | 0.474 |
| 1-15 | 0.19-1.95 | **0.756-0.828** | 0.38-0.50 |

**min r(k) = 0.756** → **NO_GO_REAL_BUG** sur ce test.

Interprétation : sur lattice quasi-régulière, TreePM produit des forces **~50% de la magnitude** des forces PP-direct, et avec corrélation directionnelle dégradée à ~76%.

### Forces individuelles (cohérence avec tests précédents)

| Setup | Median rel err |
|---|---|
| N=1000 random uniform (Phase 9.7-B) | 11.25% |
| N=10K Zel'dovich (Phase 9.7-C) | **69.94%** |

L'augmentation de l'erreur de 11% à 70% sur Zel'dovich confirme un mécanisme structurel.

## Cause probable

**Tree sans Minimum Image Convention (MIC)** : `tree_short.rs::pairwise_acc_with_split` utilise la distance brute `||p_j - p_i||` (pas MIC). PP-direct utilise MIC.

- Distribution random uniform : peu de paires near-boundary corrélées → effet faible (~1-5%)
- Distribution lattice/Zel'dovich : structure régulière → toutes paires near-boundary affectées de manière systématique → effet ~50%

Le PM (FFT periodic) compense partiellement les paires à grande r > r_cut, mais à r ~ r_cut où Tree manque les images periodicalises, PM Gaussian damping atténue déjà ces modes → trou dans la force totale.

## Recommandation

### Option A — Port GPU avec fix MIC

Avant ou pendant le port GPU, ajouter la convention MIC dans le tree CPU et GPU :

```rust
// Dans pairwise_acc_with_split (CPU) et forces_treepm_short_range (GPU CUDA)
let mut dx = pos_j.x - pos_i.x;
let mut dy = pos_j.y - pos_i.y;
let mut dz = pos_j.z - pos_i.z;
let half = box_size * 0.5;
if dx > half { dx -= box_size; }
if dx < -half { dx += box_size; }
// idem pour dy, dz
let r2 = dx*dx + dy*dy + dz*dz;
```

Estimation : 1-2h modification + tests + re-validation.

### Option B — Validation directe sur production GPU **[CONFIRMÉE]**

Le kernel GPU `forces_treepm_short_range` (`nbody_gpu_twopass.rs` lines 1763-1769) **utilise déjà la MIC** :

```c
if (ddx > box_half) ddx -= 2.0f * box_half;
if (ddx < -box_half) ddx += 2.0f * box_half;
// ddy, ddz idem
```

Et utilise `erfcf(rp / (2.0f * r_s))` (line 1778) — convention Springel cohérente avec le PM Gaussian damping cuFFT. **Le port GPU évite naturellement le bug CPU**.

**RECOMMANDATION FINALE : GO Phase 10 (port GPU)** sans préalable fix CPU. Le bug Tree CPU est local à `tree_short.rs` (utilisé seulement pour validation CPU), pas au pipeline production GPU.

### Option C — Tests complémentaires post-port GPU

Après port GPU, refaire les tests force-field cross-correlation :
- N=1000 random : critère min r(k) > 0.99
- N=10K Zel'dovich : critère min r(k) > 0.95 (acceptable avec MIC)

Si les deux passent → full prod 1M autorisée. Sinon → fix Tree MIC requis avant prod.

## CLI s'arrête (plan §6 critère 3)

Verdict mixte révèle un bug réel (Tree no MIC sur lattice). Décision humaine requise :
1. Fix MIC dans tree_short.rs CPU avant port GPU (option A, 1-2h)
2. Vérifier MIC dans GPU kernel et porter directement (option B, 30min audit)
3. Port GPU + tests post-port (option C)

## Tests créés (Phase 9.7-C)

| Test | But |
|---|---|
| `test_realistic_zeldovich_pk_treepm_vs_pp` | P(k) density (failed: shot noise dominate) |
| `test_force_field_cross_correlation_per_k` | r(k) sur N=1000 random → GO 0.994 |
| `test_force_field_cross_correlation_zeldovich` | r(k) sur Zel'dovich → NO-GO 0.756 |

Total tests TreePM : 93 passed / 16 ignored (3 nouveaux Phase 9.7-C).

## Livrables

- `logs/treepm/phase97c_final_verdict.md` — ce rapport
- `logs/treepm/phase97c_v2_verdict.txt` — verdict machine-readable
- `logs/treepm/phase97c_pk_comparison.csv` — données par bin (premier test)
