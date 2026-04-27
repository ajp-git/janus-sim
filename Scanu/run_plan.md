# Plan du μ-scan — N+ constant ≈ 40K, z=10→2

Objectif : trouver la valeur de μ la plus cohérente avec la physique Janus prédite
par Petit (ρ⁻ >> ρ⁺, ségrégation spatiale, r(k) négatif).

## Design expérimental

**Invariants** :
- Box = 500 Mpc (grand pour statistique)
- N+ ≈ 40 000 (comparable entre runs)
- z_init = 10, z_final = 2
- dt_max = 0.001 Gyr
- θ = 0.7
- Ω_b = 0.05
- h0 = 69.9

**Variables** : seulement μ et N_total (ajusté pour maintenir N+ ≈ 40K)

## Liste des runs

| Run | μ | N_total | N+ | N- | N_grid | ~temps est. |
|-----|---|---------|-----|-----|--------|-------------|
| s01 | 4 | 200,000 | 40,000 | 160,000 | 58 | 10 min |
| s02 | 8 | 360,000 | 40,000 | 320,000 | 71 | 15 min |
| s03 | 19 | 800,000 | 40,000 | 760,000 | 93 | 30 min |
| s04 | 50 | 2,040,000 | 40,000 | 2,000,000 | 127 | 60 min |
| s05 | 100 | 4,040,000 | 40,000 | 4,000,000 | 159 | 2h |
| s06 | 200 | 8,040,000 | 40,000 | 8,000,000 | 200 | 4h |
| s07 | 500 | 20,040,000 | 40,000 | 20,000,000 | 272 | 8h |

**Total : ~16-18h** pour le scan complet.

Si s07 (μ=500) prend trop : skipper et s'arrêter à μ=200.

## Pipeline pour chaque run

```bash
MU=<valeur>
N_TOTAL=<N_plus × (1+μ)>
N_GRID=$(python3 -c "print(round(${N_TOTAL}**0.333))")
OUTDIR=/app/output/janus_mu_scan/mu_${MU}

./target/release/janus_adaptive_zoom \
  --n-grid ${N_GRID} --l-box 500 --z-init 10.0 --z-final 2.0 \
  --snap-interval 50 --steps-check 50 \
  --h0 69.9 --mu ${MU} --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir ${OUTDIR} --run-label mu_${MU} \
  2>&1 | tee ${OUTDIR}/run.log

# Analyse immédiate après chaque run
python3 mu_scan_analyzer.py \
  --run-dir ${OUTDIR} --mu ${MU} \
  --out-json /app/output/janus_mu_scan/results_mu_${MU}.json \
  --out-png /app/output/janus_mu_scan/figure_mu_${MU}.png
```

## À la fin du scan

```bash
python3 mu_scan_aggregator.py \
  --json-dir /app/output/janus_mu_scan/ \
  --out-png /app/output/janus_mu_scan/SUMMARY.png \
  --out-md  /app/output/janus_mu_scan/SUMMARY.md
```

Me fournir SUMMARY.png + SUMMARY.md. Je détermine la meilleure valeur de μ
et on lance un run de validation complet (z=10→0) à cette valeur pour le preprint.

## Critères de décision

Le meilleur μ est celui qui minimise le score composite :

1. **corr(δ+, δ-)** proche de la cible historique −0.07
2. **r(k) mid < 0** (signature Janus)
3. **var(δ-)/var(δ+) élevé** (ségrégation active)
4. **v/c < 0.05** (stabilité)

Si plusieurs μ ont un score proche, privilégier celui avec **le plus grand rapport var(δ-)/var(δ+)** — c'est la signature de ρ⁻ >> ρ⁺ de Petit.
