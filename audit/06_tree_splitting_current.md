# Audit 06 — Tree splitting current state (Phase 9.6)

**Date** : 2026-04-29

## Forme exacte du splitting actuel

**Formule** : `splitting_pm_weight(r, r_cut) = (r/r_cut)^4` pour `r < r_cut`, `1.0` pour `r >= r_cut`.
**Tree weight** : `splitting_tree_weight(r, r_cut) = 1 - splitting_pm_weight(r, r_cut)`.

Source : `src/treepm/tree_short.rs:17-31`.

C'est le **polynomial Bagla 2002** (smooth derivative, simple FFT-incompatible).

## Application

Pattern : multiplicatif sur la force complète Newton (avec softening Plummer) :

```rust
// src/treepm/tree_short.rs:241-246
let inv_r3_soft = 1.0 / (r2_soft * r2_soft.sqrt());
let acc_full = r_vec * (interaction * mass_j * inv_r3_soft);
let tree_weight = splitting_tree_weight(r, r_cut);
acc_full * tree_weight
```

`F_tree = F_Newton × (1 - (r/r_cut)^4)` avec cutoff dur à `r >= r_cut`.

## Paramètres

Le tree reçoit **`r_cut`** (et non `r_s`) :
- `TreePMTree::build_with_g(particles, theta, r_cut, g_constant)` line 49
- `pairwise_acc_with_split(..., r_cut, g_constant)` line 220

`r_s` n'apparaît **PAS** dans `tree_short.rs`. Il est utilisé uniquement dans `treepm_force.rs` :
- `TreePMForce::new()` calcule `r_s = r_cut / 3.0` (ligne 38)
- `r_s` passé à `pm.solve_poisson_with_splitting(g, Some(r_s))` pour la Gaussian PM
- `r_cut` passé au tree

Donc convention : `r_cut = 3·r_s` actuel, vs PhotoNs canonique `r_cut = 5·r_s`. Différent mais pas critique pour le fix.

## Cas limite

À `r >= r_cut` : force tree = 0 (cutoff dur, line 229-231).
À `r < r_cut` : force tree = `F_Newton × (1 - (r/r_cut)^4)`.

## Callers identifiés

```bash
grep -rn "compute_short_range\|tree_short" src/treepm/ src/bin/*.rs
```

Résultats principaux :
- `src/treepm/treepm_force.rs:103`: `tree.compute_short_range_acc_excluding(...)`
- `src/treepm/tree_short.rs:159, 166, 220`: méthodes du tree
- `src/nbody_gpu_twopass.rs:1670, 1772`: kernel CUDA `forces_treepm_short_range` (GPU, hors scope CPU fix)

Pour le fix CPU, **un seul caller** côté Tree CPU : `treepm_force.rs::compute_force_excluding`.

## Stratégie de fix

**Option 1 (préférée)** : Garder `r_cut` en paramètre, calculer `r_s = r_cut / 3.0` (ou autre ratio) en interne, utiliser `splitting_tree_springel(r, r_s)`. Cutoff effectif à `r >= 3·r_s` (= `r_cut`).

**Implication** : avec convention actuelle `r_cut = 3·r_s`, le cutoff Springel à `x = r/(2·r_s) = 3` se produit à `r = 6·r_s = 2·r_cut`. Le hard cutoff Tree à `r >= r_cut` doit être ajusté à `r >= 6·r_s`.

Solution propre : changer l'API pour passer `r_s` au tree (signature modifiée), ce qui rend la dépendance au splitting explicite.

**Implémentation** : modifier `TreePMTree` pour stocker `r_s` au lieu (ou en plus) de `r_cut`, et utiliser `splitting_tree_springel(r, r_s)` dans `pairwise_acc_with_split`.

## Conclusion Audit Étape 1

- Splitting actuel : polynomial `1 - (r/r_cut)^4`, simple multiplicatif, cutoff dur à r_cut
- Fix : remplacer par `splitting_tree_springel(r, r_s)` (Springel 2005 convention)
- Modifications nécessaires : `splitting.rs` (nouvelle fonction), `tree_short.rs` (call), `treepm_force.rs` (passer r_s au lieu de r_cut)
- Pas de modification GPU (kernel CUDA `forces_treepm_short_range`) — hors scope Phase 9.6
