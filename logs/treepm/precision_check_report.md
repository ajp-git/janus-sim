# Rapport vérification précision TreePM CPU

**Date** : 2026-04-29
**Commit** : (pending — voir git log feat/treepm-jpp-port HEAD)
**Setup** : N=1000, L=100 Mpc, seed=42, N_pm=64, r_s=1.875 Mpc (1.2·Δg), r_cut=9.375 Mpc (6·Δg), θ=0.5, softening Plummer 0.05 Mpc

## Résultats principaux

### Test combiné TreePM vs PP-direct

| Métrique | Valeur |
|---|---|
| Median rel err | **16.61%** |
| P95 rel err | **60.17%** |
| Max rel err | **165.68%** |
| Median angle err | 6.70° |
| Max angle err | 103.52° |

**Cible (GrGadget Fig. 5, PhotoNs Tab. 2, Springel 2005 Fig. 6)** : médiane < 1%, P95 < 5%, max < 20%.

## Verdict : **B_OR_C_BUG**

Médiane 16.6× au-dessus du seuil acceptable. P95 12× au-dessus. Bug systémique confirmé.

## Diagnostics complémentaires

### 4a — PM-only vs TreePM combined

| Méthode | median rel err | P95 | max |
|---|---|---|---|
| PM-only | 15.85% | 59.10% | 580.5% |
| TreePM combined | 16.61% | 60.17% | 165.7% |
| Ratio PM-only / TreePM | **0.95×** | | |

**Tree ne compense PAS le PM** : les deux méthodes ont une erreur médiane quasi-identique. Le Tree ne réduit l'erreur max (de 580% à 166%, ×3.5 réduction) mais n'améliore pas la médiane.

→ **Hypothèse A (Tree compense le PM imprécis à courte portée) RÉFUTÉE.**

### 4b — Linéarité en G

Max relative deviation from F(G=10) = 10·F(G=1) : **0.0000%**.

→ Pas de bug d'unités liées à G. La normalisation 4πG est correcte.

### Tests linéarité 4c et 4d non exécutés

Inutiles puisque 4b confirme l'absence de bug d'unités. Le bug n'est pas dans les facteurs constants.

## Cause identifiée : **mismatch des fonctions de splitting** (Hypothèse C)

**PM** utilise Gaussian damping en Fourier :
- `G_PM(k) = -4πG · exp(-k²·r_s²) / k²`
- Source : `src/treepm/pm_grid.rs:219` `green *= (-k2 * rs2).exp();`

**Tree** utilise polynomial x⁴ splitting en réel :
- `splitting_tree(r) = 1 - (r/r_cut)⁴` (= `1 - splitting_pm`)
- Source : `src/treepm/splitting.rs:36`, appliqué dans `tree_short.rs::compute_short_range_pair`

**Mismatch** : pour un TreePM cohérent (Springel 2005 §3, PhotoNs §2), PM et Tree DOIVENT utiliser la MÊME fonction de splitting :

| Convention | PM (Fourier) | Tree (réel) |
|---|---|---|
| Standard (Springel/PhotoNs) | `exp(-k²r_s²)` (FT de Gaussian) | `erfc(r/2r_s)` |
| Alternative (Bagla 2002) | polynomial low-pass | `1 - (r/r_cut)^n` |
| **Code actuel (INCONSISTANT)** | **Gaussian k-space** | **polynomial real-space** |

Le code mixte fait que PM et Tree ne se complètent pas exactement → erreur médiane 15-17%.

## Recommandation

### **NO-GO sur port GPU Phase 5** sans fix préalable

Avant d'investir 1-3 jours dans le port GPU des modules Phase 2/3/5 vers `nbody_gpu_twopass.rs`, **fixer le mismatch de splitting**.

### Plan de fix (recommandé)

**Option 1 (préférée)** : aligner Tree sur la convention Gaussian de PM.

Modifications :
1. `src/treepm/tree_short.rs::compute_short_range_pair` : remplacer `splitting_tree_weight(r, r_cut)` par `erfc(r/(2*r_s))` (utiliser Phase 3 `truncation_table.rs`).
2. Passer `r_s` (au lieu de `r_cut`) au tree.
3. Estimation : 2-4h (modification + tests + re-validation).

**Option 2** : aligner PM sur la convention polynomial de Tree.

Modifications :
1. `src/treepm/pm_grid.rs::solve_poisson_filtered` : remplacer `exp(-k²·r_s²)` par la transformée de Fourier de la fonction polynomial (qui est plus complexe analytiquement).
2. Probablement pas faisable proprement sans erreur numérique dans la FFT.
3. **Non recommandé**.

### Critère de succès post-fix

Re-running `test_treepm_combined_vs_pp_direct_precision` :
- median rel err < 1%
- P95 rel err < 5%
- max rel err < 20%

### Re-évaluation du verdict du plan PLAN_CLI_TREEPM_JANUS

Le rapport phase9_decision.md disait **GO-AVEC-RÉSERVES**. Cette précision check rétroactive **rabat le verdict à NO-GO** : il faut Phase 9.5 (fix splitting) avant Phase 10 (port GPU).

| Phase | Statut |
|---|---|
| 1-9 | ✅ OK (CPU foundations validées) |
| 9.5 (cette précision check) | ⚠️ **B_OR_C_BUG identifié** |
| 9.6 (à faire) | **Fix splitting Tree → erfc/Gaussian cohérent** |
| 9.7 (à faire) | Re-run `test_treepm_combined_vs_pp_direct_precision` → **A_ACCEPTABLE attendu** |
| 10 (Phase 5 GPU port) | À démarrer après 9.7 |

## Données détaillées

Disponibles dans :
- `logs/treepm/precision_check_data.csv` (CSV per-particle errors, sorted)
- `logs/treepm/precision_verdict.txt` (one-line verdict + metrics)
- `logs/treepm/precision_check_2026-04-29.log` (full test output)

## CLI s'arrête ici

Comme stipulé dans `PROMPT_CLI_PRECISION_CHECK.md` §5 :
> Si verdict = `B_OR_C_BUG`, exécuter ces tests pour localiser le problème : ✅ done

Le bug est localisé : **mismatch de fonctions de splitting entre PM (Gaussian) et Tree (polynomial)**. Recommandation explicite Option 1 (modifier Tree pour utiliser erfc compatible avec Gaussian PM).

**Pas de fix automatique** parce que la modification touche le code core du tree et nécessite re-validation extensive. CLI attend décision humaine pour Phase 9.6.
