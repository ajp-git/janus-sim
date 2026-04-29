---
name: Phase 10.5 post-fix STOP
description: Springel T(x) fix appliqué dans 2 kernels GPU, A.5a FAIL (0.8784 < 0.99), STOP per mandat strict
type: project
---

# Phase 10.5 post-fix — STOP

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Mandat** : "Si un seul test échoue, STOP"

## Fix appliqué

`src/nbody_gpu_twopass.rs` — 2 kernels CUDA modifiés :
- `forces_treepm_short_range` (line 1967-1970)
- `forces_treepm_short_range_janus` (line 2125-2128)

```c
// AVANT
float erfc_factor = erfcf(erfc_arg);
... f = sign_factor * m * irp3 * erfc_factor;

// APRÈS (Springel 2005 GADGET-2 Eq. 17 complet)
float exp_mx2 = expf(-erfc_arg * erfc_arg);
float t_factor = erfcf(erfc_arg) + (2.0f * erfc_arg / 1.7724538509055159f) * exp_mx2;
... f = sign_factor * m * irp3 * t_factor;
```

## Résultats des 3 critères

| Test | Critère | Résultat | Status |
|---|---|---|---|
| test_phase105_decomp | ratio_Tree → 1.0 partout | mean=5.65, std=10.31 (variable) | ❌ FAIL |
| test_phase10_a5a | min r(k) > 0.99 | min r(k) = **0.8784** | ❌ FAIL |
| test_phase10_a5c | 3 cas Janus PASS | 3/3 PASS | ✅ PASS |

## Détail decomp (post-fix)

| r/r_s | F_PP | F_Tree_GPU | ratio_Tree | ratio_Total |
|---|---|---|---|---|
| 0.5 | 1.135 | 1.130 | **1.007** | 1.007 |
| 1.0 | 0.284 | 0.245 | 1.143 | 1.085 |
| 1.5 | 0.126 | 0.118 | 1.209 | 1.158 |
| 2.0 | 0.0711 | 0.0607 | 1.492 | 1.279 |
| 3.0 | 0.0316 | 0.0197 | 2.940 | 1.410 |
| 5.0 | 0.0114 | 0.00205 | 30.85 (tail) | 1.173 |
| 8.0 | 0.00444 | 2.6e-5 | 1.000 (zero/zero) | 0.992 |

**Pattern** : Tree donne le bon résultat à r/r_s = 0.5 (où T≈1, Newton dominant)
mais sur-corrige aux distances intermédiaires. La sur-correction croît avec r/r_s.

## Détail A.5a (post-fix)

```
bin    k          r(k)       |F_g|/|F_p|
0      0.063      1.0000     0.840
1      0.188      0.949      1.196
2      0.314      0.918      1.271
3      0.440      0.878      1.161   ← min r(k) = 0.8784
4      0.565      0.901      1.007
...
```

**min r(k) = 0.8784** (était 0.8926 pré-fix).
Le fix DÉGRADE légèrement la métrique r(k) sur Zel'dovich N=10K.

## Hypothèses pour la sur-correction

1. **Force = F_PP · T(x) seulement aux grandes distances**
   La formule Springel Eq.17 est :
   ```
   F_short = -G·m·r̂/r² · [erfc(x) + (2x/√π)·exp(-x²)]    avec x=r/(2·r_s)
   ```
   À r/r_s=0.5 (x=0.25), T=0.989 → F_short ≈ F_PP. ✓
   À r/r_s=3 (x=1.5), T=0.222 → F_short ≈ 0.22·F_PP. Mais kernel donne 2.94·attendu.

2. **Possible contamination BVH** : les 20 fillers à |y|>30 sont à r/r_s>15
   donc T→0 dans l'évaluation directe. Mais BVH peut grouper fillers + particle 1
   dans un cell avec COM intermédiaire, et l'opening criterion θ=0.5 peut accepter
   le multipôle au lieu de descendre.

3. **Possible facteur de normalisation différent** : le kernel pré-fix utilisait
   erfcf(x) seul. Avec la diagnostic mesure de 0.66-0.87, l'écart au facteur erfc
   suggère qu'AVANT le fix, le Tree ne calculait PAS exactement F_PP·erfc(x) ;
   il y avait peut-être un autre facteur compensateur. Le fix a peut-être
   créé une sur-correction en ajoutant le terme manquant à un kernel qui
   n'était pas erfc-only.

## Décision

**STOP** comme demandé par mandat strict ("Si un seul test échoue, STOP").

A.5a 0.8784 < 0.99 → FAIL.
decomp ratio_Tree variable → FAIL.

Le mini-run 500 steps n'est PAS lancé.

## Investigation future suggérée

1. **Vérifier si le kernel pré-fix utilisait vraiment erfc(x) seul** : git diff
   sur ce kernel + comparaison avec splitting_tree_springel CPU pour s'assurer
   que la convention de la PM correspondante est cohérente.

2. **Valider PM-only** : le diagnostic montre ratio_PM=0.99 partout (PM correct).
   Donc si Tree devait être F_PP·T(x), le total serait F_PP·(1-T+T)=F_PP, soit
   ratio_Total=1.0. Mais on a 1.16 mean. Contradiction → le kernel PM ne calcule
   peut-être pas F_PP·(1-T) mais une autre forme.

3. **Audit théorique** : revisiter la dérivation PM/Tree splitting du code en
   comparant aux références Springel 2005 Eq.17 + PhotoNs §2 + GADGET-4 paper.

4. **Test à N=2 sans fillers** : si les fillers contaminent BVH, le test devient
   ambigu. Mais N=2 fait planter cudarc (BVH par-sign nécessite N≥10/sign).
   Solution : test PP-direct vs Tree-only avec petit N et pas de PM.

## Heartbeat

5 commits Phase 10.5 :
- `7cd031c fix(vsl): Correct a_minus formula to match Petit 2014`
- (pending: Phase 10.5 fix Springel T(x) full)

Branche : phase13-octree-offset (feat/treepm-jpp-port linéarisée)
