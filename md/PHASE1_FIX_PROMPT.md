# Phase 1 — Fix périodicité niveau 1

## Portée du fix
Wrapper les 12 calculs de distance identifiés dans l'audit. Pas de modification
de `reduce_com`. Changement minimal et testable.

## Étape 1 — Helper device function

Ajouter en tête de `CUDA_KERNEL_SRC` (juste après `#define SOFTENING_MINUS_RATIO 5.0`)
une fonction helper :

```cuda
__device__ inline double minimum_image(double d, double box_size, double box_half) {
    if (d >  box_half) d -= box_size;
    if (d < -box_half) d += box_size;
    return d;
}
```

## Étape 2 — Modifier les 4 signatures de kernels

Ajouter `double box_size` en paramètre à chaque kernel qui calcule des distances :

1. `compute_forces_simple`
2. `compute_forces_bvh`
3. `compute_forces_bvh_cross`
4. `compute_forces_bvh_yukawa`

Ajouter en tête de chaque kernel :

```cuda
double box_half = 0.5 * box_size;
```

## Étape 3 — Patcher les 12 calculs de distance

Pour chaque occurrence listée dans l'audit (lignes 118-121, 138-141, 158-161,
723-726, 741-744, 759-762, 832-835, 848-851, 865-868, 940-943, 956-959, 984-987),
remplacer :

```cuda
double dx = cx - px;
double dy = cy - py;
double dz = cz - pz;
```

par :

```cuda
double dx = minimum_image(cx - px, box_size, box_half);
double dy = minimum_image(cy - py, box_size, box_half);
double dz = minimum_image(cz - pz, box_size, box_half);
```

Garder les noms de variables (dx/dy/dz, dpx/dpy/dpz, dmx/dmy/dmz) identiques.

## Étape 4 — Modifier les 4 sites de lancement des kernels

Dans le code Rust qui appelle `.launch()` pour chaque kernel, ajouter 
l'argument `box_size` :

```rust
force_kernel.launch(cfg, (
    &self.pos,
    &self.signs,
    &self.bvh_node_data,
    // ... autres args ...
    self.theta,
    self.softening,
    self.c_ratio_sq,
    self.box_size,  // NOUVEAU — à la fin de la liste d'args
))?;
```

Vérifier que `self.box_size` existe déjà comme champ de la struct 
(sinon l'ajouter, initialisé lors de `new()` avec la valeur `l_box`).

## Étape 5 — Compilation

```bash
cargo build --release --features cuda 2>&1 | tee build.log
```

Si erreur : me remonter le log. Ne pas tenter de corriger.

## Étape 6 — Test unitaire simple : 2 particules opposées

Créer `tests/test_periodicity.rs` :

```rust
#[test]
fn test_minimum_image_force() {
    // 2 particules sur l'axe X à (+200, 0, 0) et (-200, 0, 0), L=500
    // Sans minimum image : distance = 400 Mpc
    // Avec minimum image : distance = 100 Mpc
    // Force attendue : ×16 plus forte (1/r²)
    
    // Construire un mini-run 2 particules, 1 step
    // Lire l'accélération
    // Vérifier : |a| > valeur sans fix, cohérente avec G*m/100² (pas /400²)
}
```

Ou plus simplement : script Python post-run qui vérifie sur snapshot.

## Étape 7 — Test empirique coquilles radiales

Identique au test de l'audit précédent. Lancer :

```bash
./target/release/janus_adaptive_zoom \
  --n-grid 50 --l-box 100 --z-init 10.0 --z-final 9.8 \
  --snap-interval 10 --steps-check 999 \
  --h0 69.9 --mu 19.0 --omega-b 0.05 \
  --dt-max 0.001 --theta 0.7 \
  --out-dir /app/output/test_periodic_fix \
  --run-label test_periodic_fix
```

Puis python :

```python
# shell_analysis.py
import struct, numpy as np, sys

def load(p):
    with open(p, 'rb') as f:
        # ... (code standard)
        pass

snap = load('/app/output/test_periodic_fix/snapshots/snap_00020.bin')
pos = snap['pos']
vel = snap['vel']
r = np.sqrt((pos**2).sum(axis=1))
v_rad = (pos * vel).sum(axis=1) / (r + 1e-10)

print(f"{'shell':<15} {'N':<10} {'v_rms':<12} {'<v_r>':<12}")
for r_lo, r_hi in [(0,20),(20,30),(30,40),(40,50)]:
    mask = (r >= r_lo) & (r < r_hi)
    if mask.sum() == 0: continue
    v_rms = np.sqrt((vel[mask]**2).sum(axis=1).mean())
    v_r = v_rad[mask].mean()
    print(f"[{r_lo},{r_hi}]     {mask.sum():<10} {v_rms:<12.1f} {v_r:<12.1f}")
```

**Critère GO** :
- `|<v_r>|` < 50 km/s dans toutes les coquilles (peut être 10-20 km/s typique)
- `v_rms` varie de moins de 20% entre coquilles
- Pas de tendance monotone centre→bord

**Critère NO-GO** :
- `<v_r>` < -100 km/s partout → fix n'a pas marché
- `v_rms` bord >> v_rms centre → effondrement persistant

## Étape 8 — Rapport

Me fournir :
1. Le build.log (doit compiler sans warning)
2. Le diff des changements (via git diff)
3. Le tableau des coquilles radiales
4. Verdict GO/NO-GO

**Ne pas lancer la production 30h avant mon feu vert.**

## Notes importantes

- **Ne pas modifier `reduce_com`** pour l'instant. Si coquilles radiales sont OK,
  on reste en niveau 1. Sinon on attaque niveau 2.
- **Ne pas toucher** au softening (ε=0.05 m+, ε=0.25 m- via ratio 5)
- **Ne pas toucher** aux ICs random + CIC
- **Ne pas toucher** au splitting adaptatif
- **Ne pas toucher** à la physique baryonique
