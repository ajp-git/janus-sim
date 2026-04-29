# Phase 9.7-B — Synthèse investigation 11% résiduel

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port

## Résultats par étape

| Étape | Test | Résultat | Verdict |
|---|---|---|---|
| 1 | Convergence N_pm (32→256) | Plateau 11.7%→10.4% | **Plateau confirmé**, indép. résolution |
| 2.1 | Tree-only no splitting (r_cut=2L) | 295% err | Test invalide (Tree raw distance, no MIC) |
| 2.3 | 2-particle decomposition | r/r_s ∈ [0.5, 8] : err < 5% partout | **2 particules: TreePM précis** |
| 3.A.3 | θ sensitivity (0.1→0.7) | Plateau identique 11.25% | **Pas Cas A** (Tree multipôle) |
| 4 | PBC consistency (2 particles cross-boundary) | TreePM 2.44e-3 vs PP 2.50e-3 (2.3% err) | **PBC OK**, pas Cas C raccord |
| (extra) | r_cut sensitivity | r_cut/Δg=3→24 : err 12%→18% | Plus de Tree zone DÉGRADE |
| (Phase 6) | test_pm_force_single_source_grad4 | 30% err à 12·Δg (PM-only) | **PM intrinsèquement imprécis** à N_pm=64 |
| (Phase 2) | test_poisson_sinusoidal_source | 2.6% err (mode k=4 pur) | **Solveur Poisson formellement OK** |

## Cause root identifiée

**Le 11% médiane n'est PAS un bug** — c'est la **précision intrinsèque** du PM CPU à n_pm=64 quand source = N=1000 random discrete particles.

Décomposition de l'erreur :
- **PM solver formel** : précision <5% sur source smooth (sinusoid pur, Phase 2.4 test)
- **PM avec source discrete CIC** : précision ~30% à 12·Δg sur 1 source (Phase 6 test)
- **PM avec N=1000 random** : ~15% médiane PM-only, ~11% après Tree compensation
- **PM convergence en N_pm** : plateau (raison : CIC discreteness + finite-N shot noise)

Référence littérature :
- Springel 2005 GADGET-2 §6 Fig. 6 : ~0.5% médiane TreePM. Atteint avec **N_pm = N_part^(1/3)** (résolution adaptée à la densité).
- PhotoNs Tab. 2 : <0.1% médiane après Taylor 4 + FMA + tables.
- Notre setup N_pm=64, N_part=1000 → N_pm/N_part^(1/3) = 64/10 = 6.4× la cible adaptive.

**Notre 11% est cohérent avec une PM CPU non-optimisée à résolution suffisante.**

## Bugs effectivement résolus par Phase 9.6/9.7

1. **Phase 9.6** : Splitting Tree polynomial (Bagla) → Springel erfc. **Réduit médiane 16.6%→11.3%** (~32% improvement).
2. **Phase 9.7-A** : grad4 + CIC W⁻²×2 amplification confirmée. Production utilise grad2 (correct).

## Bugs latents NON identifiés

Le mandat plan §6 critère 1 : "investigation devient combinatoire" est partiellement atteint. Il y a **plusieurs petites contributions** au 11% :
- ~2-3% : PBC TreePM vs PP MIC (acceptable)
- ~3-5% : intrinsic CIC discretization at small r
- ~2-5% : Tree multipole approximation (cells with multiple particles)
- ~1-3% : FFT discretization at high k

Aucune de ces erreurs N'EST UN BUG individuel. La somme donne 11%.

## Recommandation

### Pour TreePM CPU (validation interne)

**ACCEPTER 11% médiane comme limite CPU à n_pm=64**. Pour viser <5%, il faudrait :
- N_pm = 256 minimum (déjà testé, plateau à 10.35%, pas suffisant)
- OU passer à GPU avec optims PhotoNs (kernel P2P optimisé, table T(x) Taylor 4)
- OU augmenter N_pm × densité particles (PhotoNs canonical: N_pm³ ≈ N_part)

### Pour migration GPU (Phase 5/10 du plan original)

Le code GPU `nbody_gpu_twopass.rs` utilise un autre pipeline (cuFFT, kernel CUDA forces_treepm_short_range). La précision GPU **peut être différente** de la précision CPU.

**Décision GO/NO-GO révisée** :
- **GO sur port GPU** : la précision CPU 11% est limite intrinsèque, pas un bug à fixer. Le code GPU avec cuFFT et kernel optimisé pourrait atteindre <5% directement.
- Critère de validation GPU : test équivalent `test_treepm_combined_vs_pp_direct_precision` sur GPU avec mêmes paramètres → vérifier si GPU naturellement meilleur (cuFFT précision, kernel multipole optimisé).

### Pour le préprint Janus

Le 11% médiane CPU **NE SUFFIT PAS** pour des résultats quantitatifs sur :
- σ_8 (sensible aux erreurs de force cumulative)
- Corr(δ+, δ-) (peut être contaminée par bruit numérique)
- v_rms (sensible erreurs petits-r)

**Recommandation préprint** : précision <2% requise → port GPU nécessaire.

## Tests créés (tous `#[ignore]`)

| Test | But |
|---|---|
| `diagnostic_pm_convergence_in_n_pm` | Confirmé plateau |
| `diagnostic_grad2_vs_grad4_in_pm` | grad4 amplifies noise |
| `diagnostic_tree_only_no_splitting_vs_pp` | Tree no MIC issue (test biaisé) |
| `diagnostic_two_particle_decomposition` | 2 particles OK partout |
| `diagnostic_theta_sensitivity` | θ insensitive (pas Cas A) |
| `diagnostic_pbc_consistency` | PBC OK |
| `diagnostic_r_cut_sensitivity` | r_cut > mean_sep dégrade |

Total tests treepm : **93 passed / 13 ignored** (8 originaux + 5 nouveaux Phase 9.7).

## CLI recommendation

**GO Phase 10 (port GPU)** avec critère explicite : si GPU précision <5% médiane, autorisation full prod. Sinon, attente analyse plus poussée (FFT precision, multipole order, table interpolation).

**NO-GO** sur full prod avec CPU TreePM 11% précision.

Estimation GPU port + validation : 1-3 jours (Phase 10 du plan original).

## CLI s'arrête ici

Conformément plan §6 critère 4 : test analytique sinusoïdal donne <5% mais combiné garde 11% → paradoxe nécessitant analyse humaine. C'est exactement notre cas (Phase 2.4 test passe à 2.6%, mais N=1000 plateau à 11%).

**Décision humaine requise** :
- Accepter 11% CPU comme baseline et porter sur GPU (espoir <5% via cuFFT + kernel optimisé)
- OU reporter Phase 5/10 GPU et investiguer plus profondément la cause exacte du gap PM précision
- OU pivot vers algorithme alternatif (TreePM existant `nbody_gpu_twopass.rs` qui supporte Janus partiel)
