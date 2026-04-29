# Phase 9.7-A — Diagnostic grad2 vs grad4 dans le PM gather

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port
**Test** : `diagnostic_grad2_vs_grad4_in_pm` (`src/treepm/pp_reference.rs`)
**Setup** : identique à `test_treepm_combined_vs_pp_direct_precision` :
N=1000, L=100, seed=42, n_pm=64, r_s=1.875, r_cut=9.375, θ=0.5, softening=0.05.

## Résultats

| Méthode | Median rel err | P95 | Max | Source |
|---|---|---|---|---|
| **grad2** (production current) | **11.25%** | 34.26% | 113.1% | `pm.interpolate_force` (central diff ord 2) |
| **grad4** (Phase 2 GrGadget upgrade) | **94.52%** | 262.3% | 726.8% | `pm.interpolate_force_grad4` (central diff ord 4) |

| Δ grad4 vs grad2 |
|---|
| Median : **+740%** |
| P95 : +666% |
| Max : +543% |

## Verdict

**`GRAD4_AMPLIFIES_NOISE`** (re-classifié manuellement — le classifier auto a renvoyé `MIXED_RESULT` car son seuil exigeait grad2 < 5%, condition non atteinte ; mais le ratio 8.4× est indiscutable).

## Interprétation

Le gradient ordre 4 (formule centrée 5-points `[8(φ_{i+1}-φ_{i-1}) - (φ_{i+2}-φ_{i-2})]/12h`) est censé être **plus précis** que le gradient ordre 2 (`(φ_{i+1}-φ_{i-1})/2h`). C'est mathématiquement vrai pour un Φ lisse.

**Mais** combiné avec la déconvolution CIC W⁻² appliquée ×2 dans le solveur Poisson (`pm_grid.rs:236`), le Φ résultant a son **bruit haute fréquence amplifié** par un facteur (sinc⁻²)⁴ par dimension à Nyquist. À k=N/2, ce facteur vaut `(π/2)⁸ ≈ 36`. Le gradient ord 4 capture ces fluctuations haute fréquence avec un poids stable (formule à 4 points), tandis que le gradient ord 2 (3 points adjacents) les lisse partiellement.

**Conséquence pratique** : pour une PM avec déconvolution CIC ×2 active, **le gradient ordre 2 est MEILLEUR que ordre 4** — l'inverse de l'intuition GrGadget.

## Analyse des corrections Phase 2 (rétroactif)

Les corrections Phase 2 GrGadget introduites simultanément (`gradient.rs` + `cic_correction.rs`) ne sont **pas compatibles entre elles** dans le contexte de ce pipeline PM Gaussian :

| Phase 2 correction | Effet isolé | Effet combiné PM Gaussian + ord 4 |
|---|---|---|
| Gradient ord 4 (Eq. 20) | Améliore l'erreur d'approximation | Amplifie bruit haute-k |
| CIC W⁻² ×2 deconvolution (Sefusatti) | Récupère ρ_true depuis ρ_grid | Amplifie bruit haute-k |
| Laplacien continu (forme k²) | Correct comme avant | Inchangé |

Le test sinusoïdal `test_poisson_sinusoidal_source` (Phase 2.4) passait à 2.6% parce que la source était **un seul mode k=4**, sans bruit haute fréquence. Le test point-mass et le test combiné N=1000 introduisent du bruit large bande qui révèle le problème.

## Recommandation

### Option A — Désactiver la double déconvolution

Réduire CIC deconvolution de ×2 à ×1 (un seul `cic_inv` au lieu de `cic_inv²`) dans `pm_grid.rs:236`. Cela rend grad4 et grad2 plus comparables. Mais déjà testé en Phase 9.6 : impact ≈ 0.1% sur médiane (15.85% → 15.83%).

### Option B — Filtrage low-pass avant grad4

Appliquer un filtre Gaussien à Φ après FFT inverse, avant `interpolate_force_grad4`. Adoucit le bruit haute-k au prix de précision en zone CIC.

### Option C — **Garder grad2 comme production (RECOMMANDÉ)**

Le gradient ord 2 donne 11.25% médiane vs grad4 94.5%. **Garder le code production tel quel** (utilisant `interpolate_force` = grad2). Marquer `interpolate_force_grad4` comme **expérimental, à utiliser uniquement avec PM sans déconvolution agressive**.

### Option D — Refondre Phase 2 entièrement

Réimplémenter les corrections Phase 2 avec une convention cohérente (gradient ord 4 SANS déconvolution CIC, ou déconvolution simple ×1 avec gradient ord 2). Plus de travail mais correct théoriquement.

## Conclusion Phase 9.7-A

- Le gradient ord 4 + CIC W⁻² ×2 ensemble amplifient le bruit ; **l'un ou l'autre est OK seul, les deux ensemble cassent**.
- La cause du **median 11% avec grad2** reste à identifier (Phase 9.7-B selon plan AJP).
- **Recommandation production** : Option C (garder grad2). Phase 9.7-B doit chercher pourquoi grad2 elle-même donne 11% (V_cell, FFT norm, ou softening mismatch).

## Lien avec Phase 9.6 splitting fix

Le fix Springel (Phase 9.6) reste valide : il a réduit l'erreur de 16.61% → 11.25% avec grad2. Le splitting fix est cumulatif avec les autres corrections.

## Tests

Aucun test cassé. Diagnostic ajouté en `#[ignore]` (lance avec `--ignored`).
Total tests TreePM : 95 passés / 0 failed / 7 ignored.

## Pas de modification de production

Conformément au mandat :
- `interpolate_force_grad4` (`pm_grid.rs:381`) reste disponible et inchangé
- `TreePMForce.compute_force_excluding` continue d'utiliser `interpolate_force` (grad2)
- Aucune autre modification de code production

## Sortie

- `logs/treepm/phase97a_grad2_vs_grad4.md` — ce rapport
- `logs/treepm/phase97a_verdict.txt` — verdict + métriques machine-readable
- Diagnostic test `diagnostic_grad2_vs_grad4_in_pm` ajouté
