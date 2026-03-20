# Tâche : Run ΛCDM contrôle
# Objectif : montrer r(k) ≈ 0 vs r(k) ≈ -0.5 Janus

## Contexte

Les runs Janus (V13/V14/V16) montrent P_×(k) < 0 et r(k) ≈ -0.5 à toutes
les échelles. Pour que ce résultat soit publiable, il faut un run ΛCDM avec
exactement le même pipeline qui montre r(k) ≈ 0 — prouvant que l'anti-
corrélation vient de la physique Janus, pas d'un artefact numérique.

## Paramètres — identiques à V14, une seule modification

| Paramètre | V14 (Janus) | ΛCDM contrôle |
|-----------|-------------|---------------|
| N | 3M | **3M** (identique) |
| BOX | 500 Mpc | **500 Mpc** (identique) |
| k_min ICs | 20 | **20** (identique) |
| ε | 0.25 Mpc | **0.25 Mpc** (identique) |
| Steps | 5000 | **5000** (identique) |
| snapshots | /5 | **/5** (identique) |
| η | 1.045 | **N/A** |
| **Force m+/m−** | **répulsion** | **→ attraction** |

**Une seule modification physique :**
Toutes les particules s'attirent — pas de répulsion croisée m+/m−.
Deux façons d'implémenter :

**Option A (la plus simple) :** Mettre toutes les mass_sign = +1.0
→ Simulation ΛCDM matière noire froide standard

**Option B :** Garder deux populations mais changer le signe de la force
croisée : F_cross = −G|m+||m−|/r² (attraction au lieu de répulsion)
→ Teste spécifiquement l'effet du signe de la répulsion

**Recommandation : Option A.** Plus propre, directement comparable à ΛCDM.

## Étape 1 — Créer le binaire ΛCDM

```bash
# Partir du source V14
cp /mnt/T2/janus-sim/src/bin/janus_v14_3M_kmin20.rs \
   /mnt/T2/janus-sim/src/bin/lcdm_control_3M_kmin20.rs
```

**Modification dans `lcdm_control_3M_kmin20.rs` :**

Trouver la ligne qui initialise les mass_sign (probablement dans la
génération des ICs) et mettre toutes les masses à +1.0 :

```rust
// AVANT (Janus) — alternance +1/-1 selon η
let mass_sign = if rng.gen::<f32>() < eta_frac { -1.0f32 } else { 1.0f32 };

// APRÈS (ΛCDM) — toutes positives
let mass_sign = 1.0f32;
```

Changer aussi le répertoire de sortie :
```rust
// AVANT
"/app/output/janus_v14_500Mpc_3M_kmin20"
// APRÈS
"/app/output/lcdm_control_500Mpc_3M_kmin20"
```

**Ne rien toucher d'autre** — même ICs, même H(z), même friction de Hubble,
même softening. Le but est que la seule différence soit l'absence de
répulsion croisée.

## Étape 2 — Cargo.toml

```bash
grep "lcdm_control" /mnt/T2/janus-sim/Cargo.toml || \
cat >> /mnt/T2/janus-sim/Cargo.toml << 'EOF'

[[bin]]
name = "lcdm_control_3M_kmin20"
path = "src/bin/lcdm_control_3M_kmin20.rs"
EOF
```

## Étape 3 — Compiler

```bash
cd /mnt/T2/janus-sim
docker compose run --rm dev cargo build --release \
  --features "cuda cufft" \
  --bin lcdm_control_3M_kmin20 2>&1 | tail -5
```

## Étape 4 — Créer le répertoire et lancer

```bash
mkdir -p /mnt/T2/janus-sim/output/lcdm_control_500Mpc_3M_kmin20/snapshots

nohup docker compose run --rm dev \
  ./target/release/lcdm_control_3M_kmin20 \
  2>&1 | tee /mnt/T2/janus-sim/output/lcdm_control_500Mpc_3M_kmin20/run.log &

echo "PID: $!"
tail -f /mnt/T2/janus-sim/output/lcdm_control_500Mpc_3M_kmin20/run.log
```

## Étape 5 — Vérifications critiques

**À step 10 :**
```bash
tail -5 /mnt/T2/janus-sim/output/lcdm_control_500Mpc_3M_kmin20/run.log
# Vérifier :
# - KE_ratio stable (pas d'explosion)
# - Seg ≈ 0 (pas de ségrégation — toutes masses identiques)
# - Pas de NaN
```

**À step 500 :**
```bash
python check_structure.py \
  --snap .../lcdm_control.../snapshots/snap_000500.bin \
  --step 500 --z 3.39
# Attendu : structure filamentaire diffuse (web cosmique standard)
# PAS de damier bleu/rouge — une seule population
```

**À step 5000 (z=0) — analyse P(k) :**
```bash
python compute_pk_janus.py \
  --snap .../lcdm_control.../snapshots/snap_005000.bin \
  --z 0.0 --grid 256
# Résultat attendu :
# - P_×(k) ≈ 0 (pas d'anti-corrélation — une seule espèce)
# - r(k) ≈ 0
# - P(k) forme classique ΛCDM
# - Ratio P_LCDM_sim/P_ΛCDM_analytique ≈ 1
```

## Estimation

| Métrique | Valeur |
|----------|--------|
| VRAM | ~1.5 GB (3M, pas de Barnes-Hut croisé) |
| Temps/step | ~10-15s (moins de calcul sans répulsion) |
| ETA | ~20-25h |
| Snapshots | ~1000 (interval=5) |
| Espace disque | ~30M × 28 bytes × 1000 = ~80 GB |

## Ce que le run ΛCDM va produire

Après le run, relancer `paper_figures.py` en ajoutant le run ΛCDM :

La figure comparative finale :

```
r(k) Janus  ≈ -0.5  (bleu dans la heatmap)
r(k) ΛCDM   ≈  0.0  (blanc dans la heatmap)
```

C'est la figure la plus convaincante du preprint — deux univers simulés
avec le même pipeline, la seule différence étant le signe de la force
croisée. L'un produit r(k) ≈ 0, l'autre r(k) ≈ -0.5.

## Note importante

Le run ΛCDM ne produira PAS de vraie cosmologie ΛCDM (pas de matière
noire séparée, pas d'énergie noire explicite). C'est un run de contrôle
numérique — il montre que l'anti-corrélation P_× vient de la répulsion
Janus, pas d'un biais dans le code.

Pour une vraie comparaison ΛCDM, il faudrait des ICs Boltzmann (MUSIC2)
et un code hydro (Gadget-4) — hors scope pour ce preprint.
