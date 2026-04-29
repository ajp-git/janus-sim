# Audit 03 — Corrections numériques GrGadget

**Date** : 2026-04-29
**Référence** : Quintana-Miranda et al. 2023 (GrGadget MNRAS)

## (a) Ordre du gradient — **À UPGRADER**

**Statut actuel** : `(phi[i+1] - phi[i-1]) / (2*h)` → **ordre 2**

Localisation :
- `src/treepm/pm_grid.rs` lignes 328-348 (`interpolate_force`, gather CPU)
- `src/treepm/pm_grid_gpu.rs` lignes 180-195 (idem GPU)
- `src/nbody_gpu_twopass.rs` lignes 235-242 (kernel CUDA `cic_gather`)

Pattern observé :
```rust
let dphi_dx = (phi[ci_p + ...] - phi[ci_m + ...]) / (2.0 * h);
```

**Action requise** : Implémenter gradient ordre 4 (GrGadget Eq. 20) :
```
∂φ/∂x ≈ (8·(φ_{i+1} - φ_{i-1}) - (φ_{i+2} - φ_{i-2})) / (12·h)
```

Erreur passe de O(h²) à O(h⁴).

## (b) Correction CIC en Fourier — **ABSENT**

**Statut actuel** : Aucune fonction `W_CIC(k) = sinc²(πk/N)` détectée dans le module TreePM.

Recherche faite :
```bash
grep -rn "CIC\|sinc\|cic_correction\|window" src/treepm/
```

Résultat : seulement les commentaires CIC pour scatter/gather, AUCUNE déconvolution Fourier (Sefusatti et al. 2016, GrGadget §3.3.1).

**Action requise** : 
1. Implémenter `cic_window_inv_squared(kx, ky, kz, n) = 1/W_CIC²(k)`
2. Appliquer **DEUX FOIS** dans le solveur Poisson :
   - À l'aller : `ρ̂ /= W_CIC²` avant Green's function
   - Au retour : `Φ̂ /= W_CIC²` avant inverse FFT

## (c) Laplacien forme — **OK ✅ (forme continue)**

**Statut actuel** : `(ki² + kj² + kk²) × dk²` avec `dk = 2π/L` → **forme continue** (correcte)

Localisation : `src/treepm/pm_grid.rs` lignes 197-198 :
```rust
let k2 = (ki * ki + kj * kj + kk * kk) * dk * dk;
// avec dk = 2.0 * PI / self.box_size  (ligne 185)
```

C'est la forme `k_x² + k_y² + k_z²` × `(2π/L)²` recommandée par GrGadget Eq. 22.
Pas de `sin²(πk/N)` (forme Gevolution ancienne, à éviter).

**Conclusion (c)** : Aucune action requise.

## Résumé Phase 1.3

| Correction | Statut | Action |
|---|---|---|
| (a) Gradient ordre | **ordre 2** détecté | UPGRADE → ordre 4 (Phase 2.1) |
| (b) CIC déconvolution | **ABSENT** | ADD → `W_CIC⁻²` (Phase 2.2) |
| (c) Laplacien continu | **OK ✅** | Aucune action |

Les corrections (a) et (b) seront implémentées dans Phase 2.

## Note sur le gradient ordre 4 dans le pipeline TreePM JPP

Le passage du gradient ordre 2 → ordre 4 doit être appliqué **dans tous les paths** :

1. **CPU** :
   - `src/treepm/pm_grid.rs` `interpolate_force()`
   - `src/treepm/pm_grid_gpu.rs` `interpolate_force()` CPU fallback path

2. **GPU** :
   - Kernel CUDA `cic_gather` dans `src/nbody_gpu_twopass.rs` lignes 235-242

Pour cohérence avec les tests Phase 2.1, le module Rust isolé `src/treepm/gradient.rs` sera créé avec convergence O(h⁴) testée.
