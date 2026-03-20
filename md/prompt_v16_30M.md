# Tâche : Run V16 — Filaments Janus persistants
# 30M particules, 500 Mpc, k_min=20

## Pourquoi 30M

| Run | N | VRAM | dx | Filaments |
|-----|---|------|----|-----------|
| V13 | 15M | 4.3 GB | 2.0 Mpc | Absents (k_min=3) |
| V14 | 3M | 1.2 GB | 3.5 Mpc | Apparus z=3-4, dissous z=0 |
| V15 | 15M | 4.3 GB | 2.0 Mpc | En cours → killé |
| **V16** | **30M** | **~10 GB** | **1.6 Mpc** | **Objectif** |

RTX 3060 = 12 GB VRAM. V15 n'utilisait que 4.3 GB — 8 GB libres inutilisés.
30M particules ≈ 10 GB — rentre avec 2 GB de marge.
Filaments 2× plus denses en particules → résistent mieux à la répulsion Janus.

## Étape 0 — Killer V15

```bash
# Trouver le container
docker ps --filter "name=janus" --format "{{.Names}}"

# Killer
docker stop <nom_container>
```

## Étape 1 — Créer le source V16

```bash
# Partir du source V15 (qui a déjà k_min=20 et snapshot_interval=5)
cp /mnt/T2/janus-sim/src/bin/janus_v15_500Mpc_15M.rs \
   /mnt/T2/janus-sim/src/bin/janus_v16_500Mpc_30M.rs
```

**2 modifications dans `janus_v16_500Mpc_30M.rs` :**

**Modification 1 — N_GRID : 246 → 310**
```rust
// AVANT (V15 — 15M)
const N_GRID: usize = 246;   // 246³ = 14,886,936 ≈ 15M
// APRÈS (V16 — 30M)
const N_GRID: usize = 310;   // 310³ = 29,791,000 ≈ 30M
```

Vérifier :
```bash
python3 -c "print(f'310³ = {310**3:,} particules')"
# Attendu : 29,791,000 ≈ 30M ✓
```

**Modification 2 — Répertoire de sortie**
```rust
// AVANT
"/app/output/janus_v15_500Mpc_15M_kmin20"
// APRÈS
"/app/output/janus_v16_500Mpc_30M_kmin20"
```

**Ne rien toucher d'autre** — k_min_ic=20, set_pm_k_min(2), ε=0.25,
η=1.045, H=0.01, snapshot_interval=5 restent identiques à V15.

## Étape 2 — Cargo.toml

```bash
grep "janus_v16" /mnt/T2/janus-sim/Cargo.toml || \
cat >> /mnt/T2/janus-sim/Cargo.toml << 'EOF'

[[bin]]
name = "janus_v16_500Mpc_30M"
path = "src/bin/janus_v16_500Mpc_30M.rs"
EOF
```

## Étape 3 — Compiler

```bash
cd /mnt/T2/janus-sim
docker compose run --rm dev cargo build --release \
  --features "cuda cufft" \
  --bin janus_v16_500Mpc_30M 2>&1 | tail -5
```

## Étape 4 — Vérifier l'espace disque

```bash
df -h /mnt/T2
# 30M particules × 30 bytes × 1000 snapshots ≈ 900 GB
# Si < 1 TB libre → changer snapshot_interval=10 (450 GB)
# Si < 500 GB libre → changer snapshot_interval=20 (225 GB)
```

Ajuster `SNAPSHOT_INTERVAL` dans le source si nécessaire avant de compiler.

## Étape 5 — Test 50 steps

```bash
mkdir -p /mnt/T2/janus-sim/output/janus_v16_500Mpc_30M_kmin20/snapshots

docker compose run --rm dev \
  ./target/release/janus_v16_500Mpc_30M --steps 50 \
  2>&1 | tail -20
```

**Critères de validation :**
- Pas de OOM (Out Of Memory) sur la GPU
- KE_ratio > 0 et < 10¹² (pas de NaN)
- `nvidia-smi` montre < 11 GB utilisés

```bash
# Dans un autre terminal pendant le test :
watch -n 2 nvidia-smi
```

## Étape 6 — Lancer le run complet

```bash
nohup docker compose run --rm dev \
  ./target/release/janus_v16_500Mpc_30M \
  2>&1 | tee /mnt/T2/janus-sim/output/janus_v16_500Mpc_30M_kmin20/run.log &

echo "PID: $!"
tail -f /mnt/T2/janus-sim/output/janus_v16_500Mpc_30M_kmin20/run.log
```

## Checkpoints critiques

**Step 50 (~1h) — vérifier VRAM et stabilité :**
```bash
nvidia-smi
# Attendu : ~10 GB utilisés, GPU à 100%
```

**Step 500 (z≈3.4, ~8h) — premiers filaments :**
```bash
python check_structure.py \
  --snap /mnt/T2/janus-sim/output/janus_v16_500Mpc_30M_kmin20/snapshots/snap_000500.bin \
  --step 500 --z 3.39
# Si sous-structure inter-halos visible → on est sur la bonne voie
```

**Step 1500 (z≈1.6, ~24h) — filaments persistants ?**
```bash
python check_structure.py \
  --snap /mnt/T2/janus-sim/output/janus_v16_500Mpc_30M_kmin20/snapshots/snap_001500.bin \
  --step 1500 --z 1.63
# C'est le test décisif — filaments visibles ici = succès
```

## Estimations

| Métrique | Valeur |
|----------|--------|
| VRAM estimée | ~10 GB / 12 GB |
| dx résolution | 1.6 Mpc |
| λ_min ICs | 25 Mpc (filaments de 25-250 Mpc semés) |
| Temps/step estimé | ~25s (PM) + ~25s (BH) = ~50s |
| ETA totale | ~70h |
| Snapshots (interval=5) | ~1000 |
| Espace disque | ~900 GB (vérifier avant !) |

## Si OOM au test 50 steps

Réduire légèrement N_GRID :
```rust
const N_GRID: usize = 295;   // 295³ = 25,672,375 ≈ 26M
```
Recompiler et retester. 26M particules = ~9.2 GB VRAM.
