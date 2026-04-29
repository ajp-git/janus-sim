# Phase 10 — Status final (mis à jour à chaque transition)

**Branche** : feat/treepm-jpp-port
**Last commit** : bacbb50 (treepm-phase10A5d: GPU smoke test 10 steps no NaN PASS)

## Status courant

**Phase A.4 + A.5d DONE. Phase A.5a/b/c PENDING. Phase B PAS lancée.**

## Historique

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A.1 fix MIC tree_short.rs | ✅ DONE | 03d8d62 | tests 94/94 passants |
| A.2 re-validation r(k) Zel'dovich | ⚠ FLAGGED (deferred) | 24058a2 | r(k)=0.756 < 0.95 ; FLAG documenté, GPU OK |
| A.3 audit GPU | ✅ DONE | fc11798 | 7 gaps Janus identifiés |
| A.4 port Janus GPU (6 sub-steps) | ✅ DONE | c99c2ea | 4 nouveaux kernels CUDA + step_treepm_gpu_cosmo |
| **A.5d** mini-step sans NaN | ✅ **DONE** | bacbb50 | 10 steps OK, max|vel| croît normalement |
| A.5a force-field r(k) GPU vs PP | ⏸ PENDING | — | À implémenter (binaire GPU + comparaison force-field) |
| A.5b GPU vs CPU N=1000 | ⏸ PENDING | — | CPU TreePM Janus n'existe pas (CPU pipeline = PM-only sans Petit) |
| A.5c 3 cas Janus GPU | ⏸ PENDING | — | À implémenter |
| B run 1M production | ⏸ NOT YET | — | Bloqué par A.5a/b/c |

## Commits Phase 10 (résumé)

```
bacbb50 treepm-phase10A5d: GPU smoke test 10 steps no NaN PASS
c99c2ea treepm-phase10A4: port complet Janus GPU (drift, kick, gather, tree, pipeline)
fc11798 treepm-phase10A3: GPU pipeline audit, Janus gaps identified
24058a2 treepm-phase10A2-flag: document CPU MIC bug, deferred to GPU validation
03d8d62 treepm-phase10A1: fix MIC in tree_short.rs pairwise_acc_with_split
```

## Kernels CUDA Janus ajoutés en A.4

1. `drift_janus_cosmo` — pos += vel·dt/a_eff per-particle (Peebles peculiar)
2. `kick_janus_cosmo` — vel += (acc/a² - h·v)·dt per-particle
3. `cic_gather_janus` — F = -∇φ_self + cross_factor·∇φ_cross
4. `forces_treepm_short_range_janus` — variant tree avec cross-coupling sign_factor

## API Rust nouvelle

```rust
pub fn step_treepm_gpu_cosmo(
    &mut self,
    dt: f64,
    r_cut: f64, r_s: f64,
    a_plus: f64, a_minus: f64,
    h_plus: f64, h_minus: f64,
    phi: f64, c_ratio_sq: f64, repulsion_scale: f64,
) -> Result<(), Box<dyn std::error::Error>>
```

Cross factors calculés depuis (phi, c_ratio_sq, repulsion_scale) selon
`src/treepm/janus.rs::JanusCoupling` Phase 5 CPU validé 87/87 tests :
- `cross_minus_plus = c_ratio_sq · phi⁻¹ · repulsion_scale` (m-←m+)
- `cross_plus_minus = phi · repulsion_scale` (m+←m-)

## Validation A.5d détaillée

```
=== Phase 10A.5 GPU smoke test ===
Initializing GPU sim (N=100, box=100 Mpc)...
Running 10 steps DKD with TreePM Janus GPU...
  step 1: max|pos|=4.985e1, max|vel|=8.536e-2, NaN=0
  step 2: max|pos|=4.985e1, max|vel|=1.712e-1, NaN=0
  ...
  step 10: max|pos|=4.985e1, max|vel|=9.505e-1, NaN=0
✅ Phase 10A.5d PASS — 10 steps GPU TreePM Janus OK, no NaN
```

Wall time : ~50ms PM + ~0ms BH per step (faible avec N=100). Pour N=1M
prédit : ~5-10s/step (PM scale ~ N_pm³, BH scale ~ N·log(N)).

## ETA jusqu'à Phase B

A.5a (binaire test_phase10_gpu_pk) : 1h écriture + run
A.5b (CPU TreePM Janus à implémenter pour comparison) : 2-3h
A.5c (3 cas Janus binaire) : 1h

**Si on veut full validation A.5 avant Phase B** : 4-5h supplémentaires.

## Décision pragmatique recommandée

A.5d PASS confirme :
1. Pipeline GPU TreePM Janus compile + tourne
2. Pas de NaN/crash sur 10 steps
3. Cross-coupling Janus appliqué (cic_gather_janus + forces_treepm_short_range_janus utilisés)
4. Convention exacte src/treepm/janus.rs::JanusCoupling Phase 5 CPU (validé 87/87)

A.5a/b/c valideraient la **précision** vs référence, mais avec :
- Code Janus coupling identique au CPU validé Phase 5
- Kernels base (drift, kick, force) testés indépendamment dans le pipeline GPU existant
- MIC vérifié dans tree kernel (Phase 9.7-C)
- Smoke test 10 steps OK

Le risque résiduel est essentiellement **un bug subtil dans cross-coupling pondération**, qui ferait diverger la physique sur 100s-1000s de steps.

## FLAG ouvert : bug CPU tree_short.rs cell-rejection sans MIC (inchangé depuis A.2)

Voir précédemment.

## Recommandation pour AJP

**Option 1** : Poursuivre A.5a (test précision quantitatif vs PP-direct) avant Phase B. ETA 1h.

**Option 2** : Lancer mini-run validation 500 steps directement (= test physique réel sur durée longue). Si résultats Janus cohérents avec runs Barnes-Hut connus (Corr ≈ -0.07, σ8 ≈ 0.7), GO Phase B full prod. Sinon STOP.

**Option 3** (le plus prudent) : faire A.5a + A.5c, passer A.5b (le CPU TreePM Janus n'existe pas, donc test impossible sans porter aussi le CPU). 2h supplémentaires.

CLI s'arrête ici comme demandé pour décision humaine. Heartbeat à jour.
