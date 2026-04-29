---
name: Phase 10.6 audit synthesis
description: 2 bugs concurrents Tree GPU (r_s hardcoded + erfc seul). Fix Phase 10.5 partiel a aggravé. Plan de fix combinant les deux corrections, NON APPLIQUÉ — décision AJP.
type: project
---

# Phase 10.6 — Synthèse audit kernel GPU TreePM

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Commit baseline** : 8f99e3f (post-revert Phase 10.5)
**Détails ligne par ligne** : [phase106_kernel_audit.md](phase106_kernel_audit.md)

---

## Ce que le kernel fait réellement

`F_short_GPU(r) = sign_factor · G·m / r² · erfc(r / (2·r_cut/3))`

Avec `r_s_kernel = r_cut/3` hardcodé en kernel et `erfc` seul (sans terme
Springel `(2x/√π)·exp(-x²)`).

## Ce que le kernel devrait faire (cohérent PM + Springel)

`F_short_correct(r) = sign_factor · G·m / r² · T(r / (2·r_s_host))`

avec `T(x) = erfc(x) + (2x/√π)·exp(-x²)` et `r_s_host` = même r_s utilisé
par le PM (configuré PhotoNs canonical : r_s = 1.2·Δg, r_cut = 6·Δg = 5·r_s).

## Divergences identifiées

| # | Divergence | Sévérité |
|---|---|---|
| 1 | `r_s = r_cut/3` hardcodé dans kernel (lignes 1882 et 2038), ignore le r_s host | 🔴 critique |
| 2 | `erfcf(x)` au lieu de `T(x) = erfc(x) + (2x/√π)·exp(-x²)` (lignes 1966 et 2122) | 🔴 critique |

**Impact combiné** : Tree GPU rend `F_PP × erfc(0.6·x_host)` au lieu de
`F_PP × T(x_host)`. Sous-estimation systématique de la force courte portée.

Le PM est correct (validé : ratio_PM ≈ 0.99 ± 0.005), reçoit r_s en paramètre
et applique la bonne smoothing Gaussienne.

## Pourquoi le fix Phase 10.5 a échoué

Phase 10.5 a fixé UNIQUEMENT le bug #2 (ajout du terme `(2x/√π)·exp(-x²)`),
sans toucher au bug #1.

Pré-fix Tree : `F_short = F_PP × erfc(0.6·x_host)`  → sous-estimation
Post-fix Tree : `F_short = F_PP × T(0.6·x_host)`  → sur-estimation à r/r_s>1

```
ratio_Tree pré-fix  : erfc(0.6·x) / T(x)  → 0.66-0.87 (sous)
ratio_Tree post-fix : T(0.6·x) / T(x)     → 1.0-2.94 (sur)
```

L'asymétrie (sur-correction plus violente que sous-correction) explique la
légère régression A.5a : 0.8926 → 0.8784.

## Plan de fix proposé — NON APPLIQUÉ

### Fix 1 : Passer `r_s` en paramètre au kernel Tree

**Fichiers** : `src/nbody_gpu_twopass.rs`

#### 1.a Modifier signature kernel `forces_treepm_short_range` (ligne 1858)

**Avant** :
```c
extern "C" __global__ void forces_treepm_short_range(
    ..., int n_all_signed, float theta_soft_packed, float rcut_boxhalf_packed
) {
    ...
    float r_cut = ...;
    float r_s = r_cut / 3.0f;       // ← hardcoded, à supprimer
```

**Après** :
```c
extern "C" __global__ void forces_treepm_short_range(
    ..., int n_all_signed, float theta_soft_packed, float rcut_boxhalf_packed,
    float r_s                       // ← nouveau param
) {
    ...
    float r_cut = ...;
    // r_s reçu en paramètre, pas dérivé
```

#### 1.b Modifier signature kernel `forces_treepm_short_range_janus` (ligne 2015)

Identique : ajouter `float r_s` en paramètre, supprimer `float r_s = r_cut / 3.0f`
ligne 2038.

⚠️ **Vérifier limite cudarc 12 args** : le kernel Janus a déjà 11 args, ajouter
`r_s` → 12 args = limite. Si dépassement, packer r_s avec un autre param ou
utiliser version 13-args via `cudarc 0.10+`.

Alternative : packer r_s avec r_cut sur le `rcut_boxhalf_packed` qui n'utilise
que 32 bits sur 32 (16 r_cut + 16 box_half). Trouver un emplacement libre :
- créer `rcut_rs_packed: f32` (upper 16: r_cut, lower 16: r_s) en plus de
  `boxhalf` séparé, ou
- changer le packing pour rcut+rs séparé du box_half.

#### 1.c Modifier wrappers Rust qui lancent les kernels

Sites à modifier :
- `compute_short_range_forces` (ligne 4309) : prend r_cut, doit aussi prendre r_s
- `compute_short_range_forces_janus` (ligne 4494) : idem
- `compute_short_range_forces_reuse_tree` (ligne 4606) : idem
- `step_treepm_hybrid` (ligne 4945) : déjà a r_s implicite, propager
- `step_treepm_gpu_cosmo` (ligne 5260) : déjà a r_s en paramètre, le passer au kernel
- `compute_tree_only_janus` (ligne 4905) : ajouter r_s en paramètre

### Fix 2 : Appliquer formule Springel `T(x)` complète

**Fichiers** : `src/nbody_gpu_twopass.rs`

#### 2.a Kernel `forces_treepm_short_range` (ligne 1965-1973)

**Avant** :
```c
float erfc_arg = rp / (2.0f * r_s);
float erfc_factor = erfcf(erfc_arg);
if (erfc_factor > 1e-6f) {
    float irp3 = 1.0f / (rp * rp2);
    float f = sign_factor * m * irp3 * erfc_factor;
    ...
}
```

**Après** :
```c
float erfc_arg = rp / (2.0f * r_s);
float exp_mx2 = expf(-erfc_arg * erfc_arg);
float t_factor = erfcf(erfc_arg) + (2.0f * erfc_arg / 1.7724538509055159f) * exp_mx2;
if (t_factor > 1e-6f) {
    float irp3 = 1.0f / (rp * rp2);
    float f = sign_factor * m * irp3 * t_factor;
    ...
}
```

#### 2.b Kernel `forces_treepm_short_range_janus` (ligne 2121-2128)

Identique au 2.a.

### Important : Fix 1 ET Fix 2 doivent être appliqués ensemble

Appliquer seulement Fix 2 → Phase 10.5 (régression mesurée).
Appliquer seulement Fix 1 → kernel rend `F_PP × erfc(x_host)`. Différent du Tree
attendu mais mieux que pré-fix sur certaines plages.
**Les deux ensemble** : kernel rend `F_PP × T(x_host)` ✓ correct.

## Validation post-fix prévue

| Test | Critère | Attendu après fix combiné |
|---|---|---|
| `test_phase105_decomp` ratio_Tree | constant ≈ 1.0 partout | 1.000 ± 0.05 (incluant BVH multipôle) |
| `test_phase105_decomp` ratio_Total | ≈ 1.0 partout | 0.99-1.01 |
| `test_phase10_a5a` min r(k) | > 0.99 | ~0.99-0.995 |
| `test_phase10_a5c` 3 cas Janus | PASS | 3/3 |

Le test r/r_s = 5 et 8 sont en queue (T(x)→0), ratio mesuré non significatif.

## Estimation effort fix

| Tâche | Durée |
|---|---|
| Modifier 2 signatures kernel + ajouter r_s | 30 min |
| Repacker `rcut_boxhalf_packed` ou ajouter param 13 (selon limite cudarc) | 30 min |
| Mettre à jour 5+ wrappers Rust | 45 min |
| Appliquer formule T(x) dans 2 kernels | 10 min |
| Build + tests A.5d, A.5c, A.5a, decomp | 30 min |
| **TOTAL** | **~2h30** |

## Risques

| Risque | Mitigation |
|---|---|
| Limite cudarc 12 args | Repacker rcut_boxhalf_packed → rcut_rs_packed (16+16 bits) + box_half séparé |
| Précision f32 sur (2x/√π)·exp(-x²) à grand x | OK : terme tend vers 0 exp, pas de catastrophic cancellation |
| Régression sur tests historiques `step_treepm_gpu` | Possible si appelants ne passent pas le bon r_s. Audit complet de 5 wrappers |
| BVH multipôle introduit erreur résiduelle non liée au fix | Existe pré-fix ; fix isole l'erreur Tree pure |

## Décision attendue d'AJP

| Option | Description |
|---|---|
| **GO Fix 1 + 2** | Appliquer les deux fixes ensemble. ETA 2h30. Validation par 4 tests. Le risque est calculé. |
| **GO Fix 2 + Fix r_s côté caller** | Au lieu de passer r_s au kernel, configurer le host pour appeler avec r_cut tel que `r_cut/3 = r_s_désiré`. Plus simple mais break PhotoNs canonical r_cut/r_s = 5. |
| **NO-GO** | Pivot CPU TreePM Janus : porter pipeline GPU vers validation CPU stricte avant tout run. ETA 1-2 jours. Mini-run 500 steps reporté. |
| **GO with caveat** | Accepter A.5a actuel (0.89), produire mini-run avec biais ~10% documenté. Risque scientifique sur préprint. |

**Recommandation CLI** : **GO Fix 1 + 2** (option 1). Le diagnostic est complet,
les corrections sont chirurgicales, validation prévue par tests. ETA ~2h30
incluant validation. Si succès, mini-run 500 steps z=10→z=5 immédiatement
derrière. Si échec, STOP et reporter à AJP.

## État actuel

- Branche `feat/treepm-jpp-port` à commit `8f99e3f` (baseline restaurée)
- Working tree clean (revert Phase 10.5 fix appliqué via git checkout)
- 3 tests baselines reproduits :
  - decomp ratio_Tree {0.5, 1, 1.5, 2, 3} = {0.833, 0.716, 0.658, 0.658, 0.871}
  - A.5a min r(k) = 0.8926
  - A.5c 3/3 PASS
- Audit ligne par ligne disponible : `phase106_kernel_audit.md`
