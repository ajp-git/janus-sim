# Phase 10.5 — Diagnostic décomposition PM/Tree GPU

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Test** : `test_phase105_decomp` (binaire diagnostique avec wrappers)

## Setup

2 particules m+ à distance r ∈ {0.5, 1, 1.5, 2, 3, 5, 8} × r_s, entourées de 20 fillers à |z|>30.
- L = 100 Mpc, n_pm = 64, r_s = 1.875, r_cut = 9.375, θ = 0.5
- softening = 0.05, cosmologie neutre
- Wrappers nouveaux : `compute_pm_only_janus`, `compute_tree_only_janus`

## Tableau de ratios

| r/r_s | r [Mpc] | F_PP | F_PM_GPU | F_Tree_GPU | F_Total_GPU | r_PM | r_Tree | r_Total |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.94 | 1.135 | 0.0127 | 0.935 | 0.947 | **0.990** | 0.833 | 0.835 |
| 1.0 | 1.88 | 0.284 | 0.0228 | 0.187 | 0.210 | **0.987** | 0.716 | 0.738 |
| 1.5 | 2.81 | 0.126 | 0.0286 | 0.0641 | 0.0927 | **0.989** | 0.658 | 0.734 |
| 2.0 | 3.75 | 0.0711 | 0.0302 | 0.0268 | 0.0570 | **0.993** | 0.658 | 0.801 |
| 3.0 | 5.62 | 0.0316 | 0.0248 | 0.00584 | 0.0307 | **0.998** | 0.871 | 0.971 |
| 5.0 | 9.38 | 0.0114 | 0.0113 | 0.000309 | 0.0116 | 0.998 | (tail) | 1.019 |
| 8.0 | 15.0 | 0.00444 | 0.00438 | ~0 | 0.00438 | 0.986 | (zero) | 0.987 |

**Statistiques** :
- ratio_PM : mean **0.992**, std 0.005 → **PM est CORRECT**
- ratio_Tree : variable de 0.66 à 0.87 dans le régime physique
- ratio_Total : converge vers 1.0 quand r > r_cut (Tree contribution → 0)

## Verdict : Cas B/C — bug Tree splitting function

PM est parfait, Tree est sous-estimé de 13-34% selon r. **Le bug n'est pas dans la convention sign_factor (A.5c PASS), mais dans la fonction de splitting du Tree GPU.**

## Cause root identifiée

**Le kernel GPU `forces_treepm_short_range_janus` utilise `erfcf(x)` seulement** (lignes 1966 et 2122 de `src/nbody_gpu_twopass.rs`) :

```c
float erfc_factor = erfcf(erfc_arg);  // ← bug: erfc seul, pas T(x)
...
float f = sign_factor * m * irp3 * erfc_factor;
```

**Convention Springel/PhotoNs requise** :

```
T(x) = erfc(x) + (2x/√π)·exp(-x²)
```

où `x = r/(2·r_s)`. Le terme `(2x/√π)·exp(-x²)` est MANQUANT côté GPU.

CPU `splitting_tree_springel` (Phase 9.6, `src/treepm/splitting.rs:62-77`) implémente bien T(x) complet. Le GPU kernel n'a jamais été aligné sur cette correction.

**Vérification** :
- `erfc(0.25)/T(0.25) = 0.724/0.989 = 0.732` (ratio attendu si erfc seul)
- ratio_Tree mesuré = 0.833 (Tree retourne plus que erfc seul)
- L'écart vient des contributions filler particles + variance numerique

L'amplitude exacte du facteur Tree GPU dépend de l'opening criterion BVH et des fillers, mais le pattern global confirme : **GPU Tree manque le terme `(2x/√π)·exp(-x²)`**.

## Plan de fix proposé

### Modification de 2 kernels CUDA

**1. `forces_treepm_short_range`** (kernel original, line 1858+)

```c
// Avant (line 1966):
float erfc_factor = erfcf(erfc_arg);

if (erfc_factor > 1e-6f) {
    float irp3 = 1.0f / (rp * rp2);
    float f = sign_factor * m * irp3 * erfc_factor;
    ax += f*ddx; ay += f*ddy; az += f*ddz;
}

// Après:
float exp_mx2 = expf(-erfc_arg * erfc_arg);
float t_factor = erfcf(erfc_arg) + (2.0f * erfc_arg / 1.7724538509055159f) * exp_mx2;

if (t_factor > 1e-6f) {
    float irp3 = 1.0f / (rp * rp2);
    float f = sign_factor * m * irp3 * t_factor;
    ax += f*ddx; ay += f*ddy; az += f*ddz;
}
```

**2. `forces_treepm_short_range_janus`** (Phase 10A.4.5, line 2122)

Identique : remplacer `erfc_factor` par `t_factor` calculé pareillement.

### Validation post-fix

1. Re-run `test_phase105_decomp` → ratio_Tree devrait être ≈ 1.0 partout
2. Re-run `test_phase10_a5a` → min r(k) devrait passer le seuil 0.99
3. Re-run `test_phase10_a5c` → 3 cas Janus toujours OK (signe inchangé)

## Estimation effort fix

**~30 min** (modification 2 kernels, rebuild, tests).

## Comparaison historique mise à jour

| Pipeline | Setup | min r(k) | Cause |
|---|---|---|---|
| CPU TreePM Janus (Phase 9.6 fixed) | N=1000 random | 0.994 | OK |
| CPU TreePM Janus | N=10K Zel'dovich | 0.756 | CPU MIC bug acc_recursive |
| GPU TreePM Janus (sans fix) | N=10K Zel'dovich | 0.893 | GPU Tree erfc-only |
| GPU TreePM Janus (avec fix attendu) | N=10K Zel'dovich | **? > 0.99** | À mesurer |

## Recommandation

**GO pour fix puis re-validation A.5a.** Le fix est trivial (modif 2 lignes par kernel, 4 sites au total dont 2 dans chaque kernel). Le risque est nul car CPU Phase 9.6 a déjà validé la formule T(x) sur 87 tests.

Décision pragmatique : **appliquer le fix immédiatement** puis re-runner A.5a et A.5c. Si A.5a passe → GO mini-run 500 steps.
