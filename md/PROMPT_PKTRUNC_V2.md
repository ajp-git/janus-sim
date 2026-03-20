# MISSION : Relancer 12M avec fenêtre P(k) élargie
# Modification mineure — changer uniquement k_min et k_max
# Lis ce fichier en entier avant toute action.

---

## ACTION IMMÉDIATE

### 1. Stopper le run actuel

```bash
docker stop 7e8c1959476f
nvidia-smi  # vérifier GPU libéré
```

### 2. Modifier k_min et k_max dans le code

Trouver dans le code de génération des ICs Zel'dovich :

```rust
// ANCIEN (trop restrictif — 1.2% modes conservés)
let k_min = 2.0 * PI / 150.0;
let k_max = 2.0 * PI / 15.0;

// NOUVEAU (fenêtre élargie — plus de détail)
let k_min = 2.0 * PI / 200.0;  // supprime λ > 200 Mpc
let k_max = 2.0 * PI / 8.0;    // garde jusqu'à λ = 8 Mpc
```

**Effet attendu :**
- Structures à l'échelle 8-200 Mpc dans la boîte 492 Mpc
- Plus de détail fin (filaments ~10-30 Mpc)
- Toujours plusieurs structures distinctes (pas de mode unique dominant)

### 3. Vérifier les autres paramètres — NE RIEN CHANGER D'AUTRE

```
N               = 12_000_000  (garder)
Box             = 492 Mpc     (garder)
Softening       = 0.65 Mpc    (garder)
Steps           = 20000       (garder)
dtau_per_dt     = tau_range / (20000 × 0.01)  ← FIX-016 (garder)
SNAPSHOT_INTERVAL = 20        (garder)
virialize_sampled(80000)      (garder)
Output = /mnt/T2/janus-sim/output/production_pktrunc_12m_v2/
```

### 4. Générer image step 0 avant de lancer

Générer une image de densité des ICs (step 0) pour vérifier visuellement :
- Plusieurs structures visibles à différentes échelles ?
- Pas de mode unique dominant ?
- corr(idx, z) < 0.02 ?

Si OK → lancer le run 12M.

### 5. Checklist avant lancement

```
□ nvidia-smi → GPU propre
□ k_min = 2π/200 dans le code
□ k_max = 2π/8 dans le code
□ dtau_per_dt = tau_range / (20000 × 0.01)
□ Image step 0 vérifiée → structures multi-échelles
□ corr(idx, z) < 0.02
□ KE/KE₀ step 5 < 1.05
□ git commit -m "feat: P(k) k_min=2pi/200 k_max=2pi/8"
□ Container ID sauvegardé
```

### 6. Mettre à jour RUNS.md

```
### Run: production_pktrunc_12m (STOPPÉ)
Raison: fenêtre P(k) trop restrictive (1.2% modes, k_min=2π/150, k_max=2π/15)
Remplacé par: production_pktrunc_12m_v2

### Run: production_pktrunc_12m_v2
Date: 2026-03-05
N=12M, Box=492 Mpc, Steps=20000
k_min = 2π/200 Mpc, k_max = 2π/8 Mpc
dtau_per_dt = tau_range / (20000 × 0.01)
Container: [ID]
ETA: ~65h
```

---

## RÈGLES ABSOLUES

```
JAMAIS  : docker stop $(docker ps -q)  sans filtre --filter name=
TOUJOURS : docker compose run --rm dev ... depuis janus-sim/
TOUJOURS : nvidia-smi avant tout lancement
TOUJOURS : vérifier image step 0 avant run 12M
```
