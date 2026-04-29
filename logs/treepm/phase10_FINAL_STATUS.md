# Phase 10 — Status final (mis à jour à chaque transition)

**Branche** : feat/treepm-jpp-port

## Status courant

**Phase A.3 — Audit pipeline GPU** : IN PROGRESS

## Historique

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A.1 fix MIC tree_short.rs | DONE | 03d8d62 | tests 94/94 passants |
| A.2 re-validation r(k) Zel'dovich | FLAGGED (deferred) | — | r(k)=0.756 < 0.95, voir FLAG |
| A.3 audit GPU | IN PROGRESS | — | — |
| A.4 port Janus GPU (6 sub-steps) | PENDING | — | — |
| A.5 validation triple | PENDING | — | — |
| B run 1M production | PENDING | — | — |

## ETA jusqu'à Phase B

A.3 : 1h
A.4 : 4-8h
A.5 : 1.5h
**Total Phase A** : 7-11h
**Phase B launch** : aujourd'hui ou demain

## FLAG ouvert : bug CPU tree_short.rs cell-rejection sans MIC

**Phase de découverte** : A.2 (re-validation force-field r(k) Zel'dovich)
**Symptôme** : r(k) Zel'dovich N=10K = 0.756 (cible 0.95)
**Cause root** : `acc_recursive` (src/treepm/tree_short.rs:230-231) utilise `r_to_cell` brut (pas MIC) pour le cell-rejection. Les cellules MIC-proches mais raw-far sont skippées avant `pairwise_acc_with_split` (qui lui a la MIC depuis A.1).

**Décision AJP** : NON FIXÉ. Le bug est local au pipeline CPU. Le pipeline production GPU (`nbody_gpu_twopass.rs::forces_treepm_short_range:1763-1769`) utilise la MIC partout, donc le bug n'affecte pas la production.

**Dette technique** : à fixer dans une phase future si le pipeline CPU est utilisé pour autre chose qu'audits ponctuels. Estimation 1-6h selon hypothèse 1 (cell-rejection MIC seul) vs 2 (refonte build avec PBC).

**Validation alternative** : Phase A.5 renforcée (GPU vs PP-direct sur Zel'dovich + GPU vs CPU sur N=1000 random où CPU est correct).
