# Tâche : Run V15 — Filaments Janus persistants
# 15M particules, 500 Mpc, k_min=20

## Diagnostic

Deux runs précédents :
- **V13** (15M, k_min=3) : résolution dx=2.0 Mpc, mais modes ICs λ_min=167 Mpc
  → filaments jamais semés → 4 méga-halos sphériques
- **V14** (3M, k_min=20) : modes ICs λ_min=25 Mpc, sous-structure visible à z=3-4
  → filaments apparus PUIS DISSOUS à z=0 → résolution dx=3.5 Mpc insuffisante

**Cause de la dissolution :** avec seulement 3M particules, les filaments de 25 Mpc
ne contiennent que ~500 particules en section transverse — trop peu pour résister
à la répulsion Janus pendant 10 Gyr.

**Solution V15 :** combiner la résolution de V13 (dx=2.0 Mpc, 15M particules)
avec les modes ICs de V14 (k_min=20). Les filaments seront 8× plus denses
en particules → survivent jusqu'à z=0.

## Paramètres V15

| Paramètre | V13 | V14 | **V15 cible** |
|-----------|-----|-----|---------------|
| N | 15M | 3M | **15M** |
| BOX | 500 Mpc | 500 Mpc | **500 Mpc** |
| k_min ICs | 3 | 20 | **20** |
| k_min PM | 2 | 2 | **2** (inchangé) |
| ε | 0.25 Mpc | 0.25 Mpc | **0.25 Mpc** |
| η | 1.045 | 1.045 | **1.045** |
| snapshots | /100 | /5 | **/5** |
| ETA | ~100h | ~25h | **~100h** |

## Étape 0 — Test de stabilité 200 steps (~2h)

Avant de lancer 100h, tester la stabilité avec 200 steps :

```bash
cd /mnt/T2/janus-sim

# Modifier le binaire V15 (voir Étape 1)
# puis lancer seulement 200 steps :
docker compose run --rm dev \
  ./target/release/janus_v15_500Mpc_15M --steps 200 \
  2>&1 | tail -30
```

**Critères de validation à step 200 :**
- KE/KE₀ < 10⁸ (même ordre que V13 au même step)
- Seg > 0.01 (ségrégation qui démarre)
- Pas d'explosion (pas de NaN dans les positions)

Si OK → lancer le run complet immédiatement.

## Étape 1 — Créer le binaire V15

```bash
# Trouver le source du run V13 (15M particules)
grep -rl "15_000_000\|15000000" /mnt/T2/janus-sim/src/bin/*.rs | head -5

# Trouver le source du run V14 (pour récupérer la logique k_min ICs)
ls /mnt/T2/janus-sim/src/bin/janus_v14*.rs
```

Copier le source V14 et ajuster pour 15M :

```bash
cp /mnt/T2/janus-sim/src/bin/janus_v14_3M_kmin20.rs \
   /mnt/T2/janus-sim/src/bin/janus_v15_500Mpc_15M.rs
```

**3 modifications dans `janus_v15_500Mpc_15M.rs` :**

**Modification 1 — N particules : 3M → 15M**
```rust
// AVANT (V14)
const N_GRID: usize = 144;   // 144³ = 2,985,984 ≈ 3M
// APRÈS (V15)
const N_GRID: usize = 246;   // 246³ = 14,886,936 ≈ 15M
// Note : 248³=15,252,992, 246³=14,886,936 — prendre la valeur la plus proche de 15M
// Vérifier : println!("{}", 246_usize.pow(3));
```

**Modification 2 — Répertoire de sortie**
```rust
// AVANT
"/app/output/janus_v14_500Mpc_3M_kmin20"
// APRÈS
"/app/output/janus_v15_500Mpc_15M_kmin20"
```

**Modification 3 — Snapshot interval : garder à 5**
```rust
// Vérifier que SNAPSHOT_INTERVAL = 5 (comme V14)
// Si c'est 100 (comme V13), changer en 5
const SNAPSHOT_INTERVAL: usize = 5;
```

**Ne pas toucher :**
- `k_min_ic = 20` (déjà dans V14, conserver)
- `sim.set_pm_k_min(2)` (force PM, conserver)
- `EPSILON = 0.25`, `ETA = 1.045`, `H_friction = 0.01`

## Étape 2 — Ajouter le binaire dans Cargo.toml

```bash
# Vérifier si le binaire est déclaré
grep "janus_v15" /mnt/T2/janus-sim/Cargo.toml

# Si absent, ajouter :
cat >> /mnt/T2/janus-sim/Cargo.toml << 'EOF'

[[bin]]
name = "janus_v15_500Mpc_15M"
path = "src/bin/janus_v15_500Mpc_15M.rs"
EOF
```

## Étape 3 — Compiler

```bash
cd /mnt/T2/janus-sim
docker compose run --rm dev cargo build --release \
  --features "cuda cufft" \
  --bin janus_v15_500Mpc_15M 2>&1 | tail -10
```

Vérifier que la compilation réussit sans erreur.

## Étape 4 — Créer le répertoire et lancer

```bash
mkdir -p /mnt/T2/janus-sim/output/janus_v15_500Mpc_15M_kmin20/snapshots

# Test 200 steps
docker compose run --rm dev \
  ./target/release/janus_v15_500Mpc_15M --steps 200 2>&1 | tee /tmp/v15_test.log

# Vérifier stabilité
tail -5 /tmp/v15_test.log
grep "step.*200" /tmp/v15_test.log
```

Si stable → lancer le run complet en arrière-plan :

```bash
nohup docker compose run --rm dev \
  ./target/release/janus_v15_500Mpc_15M \
  2>&1 | tee /mnt/T2/janus-sim/output/janus_v15_500Mpc_15M_kmin20/run.log &

echo "PID: $!"
```

## Étape 5 — Surveillance (checkpoints critiques)

**Step 100 (z≈4.6, ~30 min) :**
```bash
grep "^100," \
  /mnt/T2/janus-sim/output/janus_v15_500Mpc_15M_kmin20/time_series.csv
# Attendre : KE_ratio < 10⁹, Seg > 0.001
```

**Step 500 (z≈3.4, ~2.5h) — premier check filaments :**
```bash
python check_structure.py \
  --snap /mnt/T2/janus-sim/output/janus_v15_500Mpc_15M_kmin20/snapshots/snap_000500.bin \
  --step 500 --z 3.39
# Uploader l'image — on doit voir de la sous-structure inter-halos
```

**Step 1500 (z≈1.6, ~7.5h) — filaments persistants ?**
```bash
python check_structure.py \
  --snap /mnt/T2/janus-sim/output/janus_v15_500Mpc_15M_kmin20/snapshots/snap_001500.bin \
  --step 1500 --z 1.634
# Si filaments visibles ici → succès
# Si déjà dissous → même problème que V14, envisager V16
```

**Step 3000 (z≈0.4, ~15h) — état avancé**

## Estimation ressources

| Ressource | Valeur |
|-----------|--------|
| Temps/step estimé | ~70-120s (15M, Barnes-Hut) |
| Steps totaux | 5000 |
| ETA totale | ~100-170h |
| Snapshots | ~1000 (tous les 5 steps) |
| Espace disque | ~1000 × 400MB = **~400 GB** |

⚠️ **Vérifier l'espace disponible sur /mnt/T2 avant de lancer :**
```bash
df -h /mnt/T2
```
Si < 500 GB libres → réduire snapshot_interval à 10 ou 20.

## Si V15 montre encore dissolution des filaments

Plan B — V16 (15M, k_min=10, λ_min=50 Mpc) :
Les filaments de 50 Mpc sont 8× plus larges → résistent mieux à la répulsion Janus.
Modifier k_min_ic = 10 dans le source et relancer.

Plan C — Analyser les snapshots V15 à z=2-3 pour la vidéo :
Les filaments existent à z=3-4 même s'ils disparaissent à z=0.
Une vidéo montrant leur naissance et dissolution est scientifiquement valide.
