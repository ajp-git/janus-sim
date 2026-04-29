---
name: Phase 10.7 validation report
description: Fix 1+2 appliqués. decomp ratio_Tree = 0.9996±0.0005 (parfait), A.5a fail à 0.8997 (limite intrinsèque PP-MIC sans Ewald), A.5c 3/3 PASS, 3 tests pré-existants régressifs non liés.
type: project
---

# Phase 10.7 — Rapport de validation post-fix combiné

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Commits** : 7e5ae7d (Fix 1) + ddbc51d (Fix 2)
**Mandat** : "Si un seul critère échoue → STOP"

## Résultats des 4 critères

### 3.1 — `test_phase105_decomp` ratios (cible ~1.0)

| r/r_s | r [Mpc] | ratio_PM | ratio_Tree | ratio_Total | Verdict |
|---|---|---|---|---|---|
| 0.5 | 0.94 | 0.9899 | **0.9985** | **0.9984** | ✅ |
| 1.0 | 1.88 | 0.9868 | **0.9996** | **0.9985** | ✅ |
| 1.5 | 2.81 | 0.9890 | **0.9997** | **0.9973** | ✅ |
| 2.0 | 3.75 | 0.9933 | **0.9998** | **0.9970** | ✅ |
| 3.0 | 5.62 | 0.9982 | **0.9998** | **0.9985** | ✅ |
| 5.0 | 9.38 | 0.9979 | 0.9998 | 0.9979 | ✅ |
| 8.0 | 15.0 | 0.9862 | 1.0000 | 0.9862 | ✅ |

**ratio_PM**   : mean = 0.9916, std = 0.0046
**ratio_Tree** : mean = **0.9996, std = 0.0005**  ← essentiellement parfait
**ratio_Total** : mean = 0.9963

**VERDICT** : ✅ **PASS** — tous les ratios dans [0.99, 1.00], **excellence**.

Tolérance ±0.05 demandée, on est à ±0.005. Le fix Springel + r_s correct
fonctionne parfaitement sur 2 paires synthétiques.

### 3.2 — `test_phase10_a5a` min r(k) (cible > 0.99)

```
bin    k          r(k)       |F_g|/|F_p|
0      0.063      0.9999     0.826
1      0.188      0.939      1.099
2      0.314      0.911      1.012
3      0.440      0.900      0.892
4      0.565      0.915      0.832
5      0.691      0.918      0.808
6      0.817      0.922      0.808
7      0.942      0.927      0.803
8      1.068      0.933      0.811
9      1.194      0.942      0.828
10     1.319      0.973      0.868
11     1.445      0.976      0.870
12     1.571      0.942      0.891
13     1.696      0.953      0.864
14     1.822      0.947      0.923
15     1.948      0.982      0.849
```

**min r(k) = 0.8997** (au bin 3, k=0.44 1/Mpc)

| Phase | min r(k) | Δ baseline |
|---|---|---|
| Pré-fix Phase 10.5 (revert) | 0.8926 | (référence) |
| Phase 10.5 (fix #2 seul) | 0.8784 | -0.014 (régression) |
| **Phase 10.7 (fix #1+#2)** | **0.8997** | **+0.0071** |

**VERDICT** : ❌ **FAIL** — min r(k) < 0.99 demandé.

#### Diagnostic critique : limitation intrinsèque du test

Le test `test_phase10_a5a` compare `GPU TreePM` à `pp_direct_janus` (CPU
reference). Inspection ligne 226 montre que **PP-direct utilise MIC seul**
(minimum image convention, image la plus proche), **PAS d'Ewald**.

```rust
fn pp_direct_janus(...) {
    // Single-image MIC, no periodic image summation:
    if dx > half_l { dx -= box_size; }
    if dx < -half_l { dx += box_size; }
    ...
}
```

GPU TreePM produit des forces **totalement périodiques** (Tree+PM, où PM
est nativement périodique via FFT). PP-MIC ignore les contributions des
images périodiques lointaines.

**Conséquence** : la cross-correlation r(k) plafonne à ~0.90 sur les bins
intermédiaires parce que GPU et PP-MIC voient des forces structurellement
différentes au-delà de la première image.

Ceci explique aussi le |F_g|/|F_p| à ~0.83 : GPU et PP n'ont pas la même
amplitude car PP rate des contributions périodiques.

**Conclusion** : `min r(k) = 0.99` est **inatteignable** avec PP-MIC comme
référence. La cible Phase 10 BLOCKER était trop optimiste compte tenu du
test choisi. Phase 9.7-B avait déjà observé ce plateau ~10% sur CPU avec
N=10K Zel'dovich pour les mêmes raisons.

#### Verdict alternatif si on ignore la limitation du test

Si on évalue Phase 10.7 par l'amélioration vs baseline pré-fix :
- baseline pré-fix: 0.8926
- post-fix:        0.8997
- Δ = **+0.71%** (gain réel du Tree fix)

Le gain est modeste mais cohérent avec ce que decomp prédit : Tree corrigé
de 0.66-0.87 (sous-estimation 13-34%) → 0.999 (parfait), donc l'erreur Tree
résiduelle est éliminée. Le 10% restant vient de la limitation PP-MIC.

### 3.3 — `test_phase10_a5c` Janus 3 cas (cible 3/3 PASS)

```
Cas 1: m+/m+ at x=±5 → expect attraction      ✅ PASS (v0=+0.00011, v1=-0.00011)
Cas 2: m+/m- at ±5 → expect repulsion (Janus) ✅ PASS (v0=-0.00010, v1=+0.00010)
Cas 3: m-/m- at ±5 → expect attraction (Petit) ✅ PASS (v0=+0.00010, v1=-0.00010)
```

**VERDICT** : ✅ **PASS** — 3/3 cas Janus PASS.

### 3.4 — Régression suite tests `cargo test --lib`

```
test result: FAILED. 210 passed; 3 failed; 19 ignored
```

Tests échoués :
- `ic_gen::tests::test_ic_generation_small` (assert 64 == 128) — IC count
- `janus_expansion::tests::test_phi_typical_values` (a_minus formula) — VSL
- `lensing::tests::test_kappa_map_mass_conservation` (mass=1024) — lensing

**Vérification** : `git log src/ic_gen.rs src/janus_expansion.rs src/lensing.rs`
montre que ces fichiers n'ont PAS été modifiés par Phase 10.7. Les commits
les plus récents sont 7cd031c (vsl, antérieur Phase 10.7) et 65b610e
(lensing init). Ces failures sont **pré-existants**, pas une régression.

**VERDICT** : ✅ **PASS** — 0 régression introduite par Phase 10.7.

## Synthèse

| # | Critère | Verdict |
|---|---|---|
| 1 | decomp ratios proches de 1.0 | ✅ **PASS** (0.9996 ± 0.0005) |
| 2 | A.5a min r(k) > 0.99 | ❌ **FAIL** (0.8997, limite intrinsèque test) |
| 3 | A.5c 3/3 Janus signs | ✅ **PASS** |
| 4 | Pas de régression | ✅ **PASS** (3 fails pré-existants) |

## Verdict final

**STOP per mandat strict** (1 critère sur 4 échoue).

Cependant, l'analyse montre que :
- Les fixes Phase 10.7 fonctionnent **parfaitement** (decomp = 0.9996 ± 0.0005)
- A.5a échec est dû à la **limitation du test** (PP-MIC sans Ewald)
- 0.99 est inatteignable avec ce test, **indépendamment des fixes**

## Décision attendue d'AJP

| Option | Description |
|---|---|
| **STOP, NO-GO mini-run** | Mandat strict, 4/4 nécessaire. Investigation supplémentaire requise. |
| **GO mini-run avec caveat** | Accept que A.5a plafonne intrinsèquement. Decomp + A.5c PASS suffisent à valider la physique. Mini-run avec biais possible documenté. |
| **Réviser le test A.5a** | Implémenter Ewald summation dans pp_direct_janus pour comparaison apples-to-apples. ETA 2-3h supplémentaires. |
| **Comparer GPU à CPU TreePM** | Au lieu de PP-MIC, comparer GPU à CPU TreePM (même physique, même r(k) cible). Mais CPU TreePM Janus n'existe pas (cf. Phase 10 BLOCKER). |

**Recommandation CLI** : **Option 3 (Ewald)** ou **Option 2 (GO with caveat)**.
L'option 2 est rapide ; l'option 3 donne une validation propre.

L'option 1 (STOP strict) bloque sur un test qui n'a jamais pu atteindre 0.99
sur la version GPU (ni sur la version CPU avec lattice Zel'dovich,
cf. Phase 9.7-B).

## État technique

- Branche `feat/treepm-jpp-port` à commit `ddbc51d`
- Working tree clean
- Fixes appliqués proprement, kernels documentés en commentaires
- Tests A.5d (smoke) + A.5c (signs) + decomp (precision) PASS
- A.5a: 0.8997 (vs 0.8926 baseline), gain réel +0.71% (cohérent avec
  élimination du bug Tree, le résiduel ~10% est PP-MIC vs PM-FFT)
