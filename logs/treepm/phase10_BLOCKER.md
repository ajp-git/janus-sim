# Phase 10 BLOCKER — A.5a fail, STOP avant mini-run

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Last commit** : (pending — A.5c PASS, A.5a FAIL)

## Status global

| Test | Status |
|---|---|
| A.5d smoke 10 steps | ✅ PASS (commit bacbb50) |
| **A.5c 3 cas Janus** | ✅ **PASS** (m+/m+ attractif, m+/m- répulsif, m-/m- attractif) |
| **A.5a r(k) GPU vs PP** | ❌ **FAIL** (min r(k) = 0.8926 < 0.95) |
| A.5b GPU vs CPU | non lancé (CPU TreePM Janus n'existe pas) |
| Mini-run 500 steps | **NON LANCÉ** (STOP per mandat) |

## A.5a détail

Setup : Zel'dovich N=10648 (22³ lattice + 15% dx perturbation), L=100 Mpc,
n_pm=64, r_cut=9.375, r_s=1.875, θ=0.5, softening=0.05, mix 5% m+ / 95% m-,
cosmologie neutre (a=1, H=0, phi=1, c̄²=1, repulsion=1).

**Résultats r(k) per bin :**

```
bin    k          r(k)       |F_g|/|F_p|
0      0.063      0.9999     0.826
1      0.188      0.938      1.096
2      0.314      0.909      0.998
3      0.440      0.893      0.873   ← min r(k)
4      0.565      0.903      0.809
5      0.691      0.900      0.785
6      0.817      0.903      0.786
7      0.942      0.910      0.782
8      1.068      0.919      0.795
9      1.194      0.932      0.815
10     1.319      0.970      0.862
11     1.445      0.975      0.867
12     1.571      0.936      0.883
13     1.696      0.948      0.853
14     1.822      0.942      0.915
15     1.948      0.980      0.846
```

**min r(k) = 0.8926** (à k=0.44 1/Mpc, bin 3)

## Comparaison historique

| Test | Setup | min r(k) |
|---|---|---|
| Phase 9.7-C CPU TreePM Janus on N=1000 random | random uniform | 0.994 |
| Phase 9.7-C CPU TreePM Janus on N=10K Zel'dovich | lattice | 0.756 |
| **Phase 10 GPU TreePM Janus on N=10K Zel'dovich** | lattice | **0.893** |

Le GPU améliore vs CPU (0.893 vs 0.756) grâce à la MIC correcte dans
le tree GPU. Mais reste sous le seuil 0.95.

## Cause probable

Phase 9.7-B avait conclu que le 11% médiane sur N=1000 random est la
**limite intrinsèque PM CPU** (CIC + finite N_pm + multipôle Tree).
Sur Zel'dovich lattice, l'erreur structurelle s'amplifie même avec MIC
correcte.

Hypothèses :
1. **Précision intrinsèque PM** : N_pm=64 trop bas pour N=10648 (PhotoNs
   recommande N_pm³ ≈ N_part, donc N_pm=22 minimum, mais ici 64 → marge).
2. **CIC f32 vs f64** : GPU utilise SP partout, perte de précision sur
   distribution dense.
3. **Cross-coupling subtle bug** : PP et GPU appliquent la même convention
   par construction (vérifié), mais une erreur d'index ou de signe pourrait
   exister.
4. **Sign factor sur le tree** : `forces_treepm_short_range_janus` utilise
   `sign_factor` pondéré, mais le COM des cellules est calculé avec sign
   factor déjà inclus dans le BVH (build_single_sign_tree). Possible double-
   compte de signe.

## Vérification réussie : 3 cas Janus PASS

A.5c confirme que les SIGNES (direction) des accélérations sont corrects :
- m+/m+ attractif ✅
- m+/m- répulsif (Janus signature) ✅
- m-/m- attractif ✅

Donc la PHYSIQUE Janus est correctement implémentée. C'est juste la
PRÉCISION sur distribution lattice qui est limitée.

## Recommandation

**STOP comme demandé par AJP.** Le critère min r(k) > 0.95 n'est pas
atteint. Lancer le mini-run sur cette base aurait des accélérations
biaisées, propageant le 10% d'erreur dans la physique Janus.

### Options pour AJP

1. **Investiguer hypothèse 4** (sign factor double-compte) : audit du
   tree build_single_sign_tree + forces_treepm_short_range_janus pour
   trouver bug subtil. ETA 2-3h.

2. **Augmenter N_pm à 128 ou 256** sur le test, voir si r(k) converge :
   Phase 9.7-B avait montré plateau 10-11% sur CPU, donc probablement
   pareil GPU. Mais worth checking.

3. **Accepter GO_WITH_CAVEAT à r(k)=0.89** : Springel 2005 et PhotoNs
   recommandent r(k) > 0.99 pour preprint quantitatif. 0.89 suggère un
   biais ~10% sur les forces, qui se propagerait à σ8, Corr, etc.
   Probablement pas suffisant pour un préprint scientifique.

4. **Pivot CPU TreePM Janus** : porter le code GPU vers CPU validation
   stricte avant tout run. ETA 1-2 jours.

Aucune décision unilatérale. CLI s'arrête conformément au mandat.

## Heartbeat à jour, branche propre

7 commits Phase 10. Pipeline GPU compile, fonctionne, applique le bon
couplage Janus en signe, mais précision quantitative insuffisante pour
le préprint.
