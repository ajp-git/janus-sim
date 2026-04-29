# Phase 9.6 — Splitting fix : rapport final

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Commits** : `89e3895` (step 2) → `a3da4eb` (step 3) → `5075b44` (step 4)

## Résumé

Mismatch splitting Tree(polynomial)/PM(Gaussian) corrigé en alignant le Tree
sur la convention Springel 2005 (`erfc + 2x/√π·exp(-x²)`).

**Résultat** : amélioration substantielle mesurée (median -32%, P95 -45%, ratio
PM/TreePM passe de 0.95× à 1.41×, donc Tree compense maintenant).
**Verdict reste B_OR_C_BUG** : un bug latent supplémentaire empêche le verdict A.

## Résultats numériques

| Métrique | Avant Phase 9.6 | Après Phase 9.6 | Cible (A) | Status |
|---|---|---|---|---|
| Median rel err | 16.61% | **11.25%** | < 1% | ❌ |
| P95 rel err | 60.17% | **33.03%** | < 5% | ❌ |
| Max rel err | 165.7% | **113.1%** | < 20% | ❌ |
| Median angle err | 6.70° | **4.13°** | < 1° | ❌ |
| Max angle err | 103.5° | **81.5°** | < 5° | ❌ |
| ratio PM-only/TreePM | 0.95× | **1.41×** | > 1× (Tree compense) | ✅ |

**Évolution** : amélioration de 32-45% sur toutes les métriques. Le splitting
fix résout une partie du problème mais pas tout.

## Modifications de code

- `src/treepm/splitting.rs` : ajout `splitting_tree_springel(r, r_s)`,
  fonctions polynomial marquées DEPRECATED dans la doc.
- `src/treepm/tree_short.rs` : `TreePMTree` gagne champ `r_s`, nouvelle méthode
  `build_with_rs_and_g`, `pairwise_acc_with_split` utilise Springel au lieu de
  polynomial.
- `src/treepm/treepm_force.rs` : `TreePMForce::new` met `r_s = r_cut/5`
  (PhotoNs canonical, était `r_cut/3` mismatched), `update()` propage `r_s` au tree.

## Tests

- 5 nouveaux tests sur `splitting_tree_springel` (Phase 9.6 step 2)
- 1 test `test_tree_uses_springel_splitting` (Phase 9.6 step 3)
- 1 nouveau diagnostic `diagnostic_pm_convergence_in_n_pm` (Phase 9.6 step 4)
- Total tests TreePM : 87 → **94 passés** / 0 failed / 6 ignored

Aucun ancien test cassé (87 originaux passent toujours).

## Diagnostic du bug latent restant

### Convergence en N_pm

| N_pm | dg | r_cut | median | P95 | max |
|---|---|---|---|---|---|
| 32 | 3.13 | 18.75 | 11.69% | 60.7% | 249% |
| 64 | 1.56 | 9.38 | 11.24% | 32.6% | 94% |
| 128 | 0.78 | 4.69 | 10.65% | 31.0% | 77% |
| 256 | 0.39 | 2.34 | 10.35% | 29.6% | 78% |

**Plateau à ~10-11%** avec N_pm croissant. Pas un problème de résolution.

### Hypothèses pour bug latent

1. **Mismatch softening** : PP-direct softens toutes paires (Plummer ε=0.05).
   TreePM Tree softens, mais PM ne softens PAS (CIC smoothing seulement).
   Pour des paires à r ~ ε, le softening différent peut introduire ~5-10% biais.

2. **Convention de normalisation V_cell** : `ρ_grid` est mass-count (pas density).
   Le code utilise `g_solver = G_phys / V_cell` pour compenser, mais la cohérence
   exacte avec le forward FFT (factor implicite V_cell) n'est pas vérifiée.

3. **Interpolate_force_grad4** : gradient ordre 4 sur Φ amplifie les fluctuations
   haute fréquence. La déconvolution CIC ×2 amplifie encore plus à haute k.
   Tester revertir gradient ord 2 pour voir.

4. **Convention Janus** : tree applique `interaction = (sign_i == sign_j ? 1 : -1) × g`.
   Pour test Newton (tous m+), c'est `+1·g`. PP applique `+G·m_j·r̂/r²`. Devrait
   être identique. À vérifier qu'aucune subtilité n'introduit -1.

## Re-évaluation du verdict global

| Phase | Status |
|---|---|
| 1-9 | ✅ OK |
| 9.5 | ⚠ B_OR_C_BUG identifié (16% median) |
| **9.6 (cette phase)** | ⚠ **Amélioration partielle (-32% median), B persistant** |
| 9.7 (à faire) | Investigation softening + V_cell + grad ord 2/4 |
| 10 (port GPU) | **NO-GO maintenu** — fixer Phase 9.7 d'abord |

## Recommandation

**STOP** automatique per plan §5 critère 2 : verdict B persistant après fix
splitting → bug latent supplémentaire à investiguer.

### Plan d'investigation Phase 9.7

1. **Test softening cohérent** : exécuter PP-direct AVEC softening=0 ET TreePM
   AVEC softening=0 (Tree sans Plummer). Si les deux donnent le même résultat,
   le softening était la cause.

2. **Test g_solver alternative** : passer `g_solver = G_phys` (pas `/V_cell`),
   et vérifier que la magnitude des forces correspond à PP-direct. L'erreur
   absolue donnera le facteur multiplicatif manquant.

3. **Test grad ord 2 vs ord 4** : utiliser `interpolate_force` (grad2) au lieu
   de `interpolate_force_grad4` dans le PM gather. Si grad2 donne meilleurs
   résultats, c'est que grad4 amplifie le bruit.

4. **Test tree-only vs PM-only** : isoler chaque composant pour identifier
   où vient le 11% résiduel.

Estimation Phase 9.7 : 4-6h selon complexité du bug.

### Décision GO/NO-GO

- **NO-GO** sur port GPU Phase 5/10 sans Phase 9.7.
- **GO** pour continuer investigation CPU autonome (Phase 9.7), bug localisable.
- **GO partiel possible** si AJP accepte un TreePM avec ~10% precision pour
  applications qualitatives (visualisations, ordre de grandeur). Pas pour
  preprint quantitatif.

## CLI s'arrête ici (plan §5)

Code modifié, tests passants (94/94), bug systémique localisé mais non résolu.
En attente décision humaine pour :
- Phase 9.7 (investigation continuation)
- Acceptation 11% precision (use case dépendant)
- Pivot vers autre algorithme (DKD direct N², ou TreePM existant GPU malgré le bug)
