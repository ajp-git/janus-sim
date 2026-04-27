# Phase 2 — Janus in vitro avec Hubble ON

## Objectif

Trouver une configuration où la VRAIE physique Janus (avec Hubble friction 
active) produit des structures observables. Le Test 1B nous a confirmé 
que le moteur peut collapser. Maintenant il faut identifier la combinaison 
{boîte, résolution, μ} qui permet ce collapse même avec freinage cosmologique.

## Stratégie

Dichotomie sur 3 axes principaux :
1. **Boîte** : 50, 100, 200 Mpc
2. **Résolution effective des m+** : déterminée par N et fraction f+ = 1/(1+μ)
3. **μ** : 5, 10, 19

L'idée : à boîte plus petite, les m+ sont plus proches en moyenne et la 
gravité intra-m+ peut vaincre la répulsion locale du fond m-.

## Pre-requis (obligatoire avant le run)

```bash
# Reverter le code après Test 1B
cd /mnt/T2/janus-sim
git checkout src/bin/janus_adaptive_zoom.rs
git status  # vérifier "nothing to commit, working tree clean"

# Recompiler
cargo build --release --features cuda 2>&1 | tail -5
```

Confirmer dans la sortie compilation qu'il n'y a aucun warning de modification.

## Run 1 — Janus in vitro minimal (1h GPU)

Configuration la plus prometteuse selon les analyses des 4 IA :

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTDIR="/app/output/janus_invitro_L100_${TIMESTAMP}"

mkdir -p "$OUTDIR"
touch "$OUTDIR/PRODUCTION_ACTIVE.lock"

echo "=== JANUS IN VITRO L=100 Mpc ===" > "$OUTDIR/README.txt"
echo "Goal: structures with REAL Hubble friction in small box" >> "$OUTDIR/README.txt"
echo "Started: $(date)" >> "$OUTDIR/README.txt"

./target/release/janus_adaptive_zoom \
  --n-grid 170 --l-box 100 --z-init 4.0 --z-final 0.0 \
  --snap-interval 30 --steps-check 100 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --eps-plus 0.05 --eps-minus 0.10 \
  --zoom-cube-size 0 \
  --max-split-level 0 \
  --out-dir "$OUTDIR" \
  --run-label "invitro_L100_mu19" \
  2>&1 | tee "$OUTDIR/run.log" &

PID=$!
echo "$PID" > "$OUTDIR/PID"
```

**Paramètres** :
- N = 5M (170³), L = 100 Mpc
- Spacing moyen 0.59 Mpc (vs 2.92 Mpc pour le Test 1B v8)
- N+ ≈ 250K particules positives (5% à μ=19)
- Spacing m+ ≈ 1.6 Mpc (vs 17 Mpc pour Test 1B 500K en 500 Mpc)
- **Résolution m+ ×10 par rapport à v8**
- z=4→0 = ~4500 steps
- Hubble friction = 1.0 (vraie physique)

**Estimation temps** : 4-6h GPU (ressemble à v8 mais boîte 5× plus petite 
demande plus de Tree calls car structures plus locales)

## Critères de réussite

À z=2 (~2h dans le run), vérifier dans le CSV :

**Réussite forte** :
- ρ+_max > 6.78e10 (10× ρ_mean+) durablement
- v_rms en croissance soutenue (>200 km/s)
- Structures visibles dans les snapshots

→ Le couple (μ=19, L=100 Mpc) marche. Phase 2 réussie.

**Réussite partielle** :
- ρ+_max oscille autour de 5e10
- v_rms croît mais stagne

→ Boîte encore trop grande. Tester L=50 Mpc dans Run 2.

**Échec** :
- ρ+_max plafonne <2e10
- v_rms stagne <100 km/s

→ μ=19 trop fort même à L=100 Mpc. Tester μ=10 dans Run 3.

## Run 2 — Si Run 1 partiel : L=50 Mpc

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 170 --l-box 50 --z-init 4.0 --z-final 0.0 \
  --snap-interval 30 --steps-check 100 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --eps-plus 0.025 --eps-minus 0.05 \
  --out-dir "/app/output/janus_invitro_L50_${TIMESTAMP}" \
  --run-label "invitro_L50_mu19"
```

Spacing moyen 0.29 Mpc, N+ ≈ 250K avec spacing 0.81 Mpc — résolution m+ 
×35 par rapport à v8.

## Run 3 — Si Run 1+2 échouent : μ=10

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 170 --l-box 100 --z-init 4.0 --z-final 0.0 \
  --snap-interval 30 --steps-check 100 \
  --h0 69.9 --mu 10.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir "/app/output/janus_invitro_L100_mu10_${TIMESTAMP}" \
  --run-label "invitro_L100_mu10"
```

À μ=10, fraction m+ = 9% (vs 5% à μ=19). Plus de matière positive disponible 
pour s'effondrer.

## Analyse post-run

Pour chaque run réussi :

1. **Validation grille Phase 13** sur 5 z (z=4, 2, 1, 0.5, 0)
2. **Phase 9 deep analysis** sur snap z=0
3. **Vidéo MP4** de l'évolution
4. **Profils radiaux** des m- denses (signature alvéolaire)
5. **Profils radiaux** des m+ denses (halos formés)
6. **r(k) cross** entre m+ et m-

Comparer ces signatures aux 7 prédictions de Petit dans 
`/mnt/user-data/outputs/janus_petit_comparison/`.

## Ce qu'on cherche

**Structure alvéolaire de Petit** :
- Conglomérats m- sphéroïdaux de ~30-50% de la boîte (15-50 Mpc dans une 
  boîte 100 Mpc)
- Parois m+ entre les conglomérats m-
- Cœurs vides de m+ dans les conglomérats m-
- r(k) négatif à plusieurs échelles

Si on trouve ça à L=100 Mpc avec Hubble ON, c'est **publiable**.

## Limites strictes

- **Pas de Hubble friction OFF** dans cette phase (c'était juste Test 1B)
- **Pas de splits** pour rester simple (les comprendre dans Phase 3 si nécessaire)
- **Pas de baryonic SF** (le seuil 30 cm⁻³ est inatteignable, on n'attend pas d'étoiles)
- **Pas plus de 3 runs** avant analyse ensemble (stop budget GPU)

## Décision après Phase 2

Si **Run 1 réussit (L=100, μ=19)** → préparer le preprint :
> "First N-body simulation of Janus Cosmological Model with proper cosmological 
> expansion: emergence of alveolar structure at sub-100 Mpc scales"

Si **L=50 ou μ=10 nécessaires** → preprint plus modeste :
> "Conditions for structure formation in the Janus Cosmological Model: 
> resolution and mass-ratio constraints"

Si **rien ne marche** → le modèle Janus tel que formulé est peut-être 
incompatible avec la formation de structures sans physique additionnelle. 
Contact direct avec Petit dans ce cas.

## Budget total Phase 2

- Run 1 : 4-6h GPU
- Analyse : 1h
- Décision Run 2 ou 3 si nécessaire : 4-6h GPU
- Total worst case : 12h GPU + 2h analyse = **1 journée**
