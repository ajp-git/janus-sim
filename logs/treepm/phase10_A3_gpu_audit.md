# Phase 10 A.3 — Audit pipeline GPU `nbody_gpu_twopass.rs`

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port

## 1. Kernels CUDA existants (33 kernels, ~5519 LOC)

| Ligne | Kernel | Rôle | État Janus |
|---|---|---|---|
| 33 | `drift_f32` | `pos += vel·dt` simple, PBC wrap | **Newton uniforme** (pas /a, pas dual a±) |
| 50 | `kick_f32` | `vel += (acc + friction)·dt`, friction=−H·v·dtau | **Newton single H** (pas dual h±) |
| 67 | `add_pm_forces` | `acc += pm_forces` | neutre |
| 98 | `cic_scatter` | CIC → ρ_plus, ρ_minus séparés (i8 sign) | **Janus partiel** (sign-aware mais sans cross-coupling) |
| 162 | `cic_gather` | F = −∇φ_attract + ∇φ_repel (factor 1.0) | **Janus partiel** (factor 1.0, pas φ ni c̄²) |
| 260 | `reset_f64_grid` | clear ρ | neutre |
| 266 | `extract_by_sign` | split particles par sign | neutre |
| 302 | `compute_morton_f32` | Morton codes pour BVH | neutre |
| 328-340 | `reorder_*` | reorder particles | neutre |
| 404-485 | `build_bvh_tp/init_leaves_tp/reduce_tp` | Karras BVH | neutre |
| 567 | `forces_twopass_overwrite` | BH force pure | Newton |
| 668 | `forces_twopass_accumulate` | BH force accumulate | Newton |
| 775 | `forces_twopass_warpcoherent` | BH warp-coalesced | Newton |
| 911 | `forces_twopass_warpcoherent_screening` | Yukawa screening | Janus + Yukawa |
| 1049-1339 | `forces_twopass_shmem_*`, `forces_direct_n2` | autres BH variants | Newton |
| **1670** | **`forces_treepm_short_range`** | **TreePM tree avec MIC + erfc Springel** | **Janus signs + erfc**, mais cross-coupling factor 1.0 |
| 1422-1437 | `reset_*` | clear arrays | neutre |
| 1521-1575 | `radix_*` | radix sort GPU | neutre |

## 2. Pipeline step DKD existant

### 2.a `step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt)` (line 4486)

Flux DKD :
1. **D1** : `drift_f32(half_dt)` — drift simple `pos += vel·dt/2`, PBC wrap
2. **CIC scatter** : `cic_scatter` → `rho_plus`, `rho_minus` (signed grid)
3. **Poisson cuFFT** : `cufft_ffi::solve_device(rho_plus → phi_plus)` et `(rho_minus → phi_minus)` avec Gaussian damping `exp(-k²·r_s²)`, `r_s = r_cut/3`
4. **CIC gather** : `cic_gather` → `pm_forces` avec F = -∇φ_attract + ∇φ_repel (factor 1.0)
5. **BH short-range** : `compute_short_range_forces(r_cut)` → `forces_treepm_short_range` (avec erfc + MIC)
6. **add_pm_forces** : `acc += pm_forces`
7. **K** : `kick_f32(dt)` — `vel += (acc - hubble·v·dtau_per_dt)·dt` single-H
8. **D2** : `drift_f32(half_dt)`

### 2.b Variantes
- `step_treepm_hybrid` : CIC sur CPU (legacy)
- `step_treepm_gpu_cached` : tree caching A/B
- `step_treepm_gpu_morton` : Morton reordering pour BH speedup

## 3. Conventions actuelles

| Aspect | Convention | Source |
|---|---|---|
| Position | `pos[3*tid + d]` f32, comoving Mpc, [-box_half, box_half] | `drift_f32:41-46` |
| Vélocité | `vel[3*tid + d]` f32 | (NB: pas peculiar Peebles, juste vel proper) |
| Sign | `signs_all[i]` `signed char` (i8) ±1 | `cic_scatter:100`, `cic_gather:164`, `forces_treepm_short_range:1672` |
| Mass | implicit unit masses (mass_factor scalaire global) | `cic_scatter:104`, `set_mass_factor` |
| Drift | `pos += vel·dt` (PAS /a) | `drift_f32:41-43` |
| Kick | `vel += (acc + friction)·dt`, friction = −hubble·v·dtau_per_dt (single H) | `kick_f32:60-62` |
| g_constant | `g_solver = mass_factor / cell_vol` | `step_treepm_gpu:4503` |
| Splitting r_s | `r_cut/3` (PAS `r_cut/5` PhotoNs canonical) | `step_treepm_gpu:4563`, `forces_treepm_short_range:1694` |
| MIC short-range | OUI dans tree forces | `forces_treepm_short_range:1741-1746, 1763-1769` |
| MIC PM | implicite par périodicité FFT | cuFFT |

## 4. Manques pour Janus complet

### Critique pour préprint

| Manque | Description | Phase à porter |
|---|---|---|
| **drift dual a±** | `drift_f32` n'a pas de a, pas peculiar /a per-sign | A.4.1 |
| **kick dual a±, h±** | `kick_f32` single hubble, pas /a² per-sign | A.4.2 |
| **Cross-coupling Petit** | `cic_gather` line 247-249 utilise factor 1.0 dans -∇φ_a + ∇φ_r ; manque `c_ratio_sq · phi_inv` (m-←m+) et `phi` (m+←m-) | A.4.4 |
| **Tree cross-coupling** | `forces_treepm_short_range` ligne 1709 `sign_factor = ±1` (pas pondéré par cross_minus_plus / cross_plus_minus) | A.4.5 |
| **Convention vélocité** | `vel` traité comme proper velocity ; pour Janus Peebles peculiar `v_pec = a·dx_co/dt` il faut /a dans drift et /a² dans kick | A.4.1, A.4.2 |
| **Splitting r_s = r_cut/3 vs r_cut/5** | Phase 9.6 décision : r_cut/5 PhotoNs canonical | A.4.6 |
| **dynamic VSL c̄²(t) = a⁺/a⁻** | actuellement `current_z` set externally mais pas utilisé dans force kernels | A.4.5 |

### Non-critique (peuvent rester)

- Mass factor scalaire global (vs per-particle masses) : OK pour μ=19 où masses uniformes par sign
- Précision SP partout (vs DP pour accumulation tree) : optimisation, déjà décidé Phase 3.0

## 5. Pipeline target post-A.4

```rust
pub fn step_treepm_gpu_cosmo(
    &mut self,
    dt: f64,
    a_plus: f64, a_minus: f64,
    h_plus: f64, h_minus: f64,
    coupling: &JanusCoupling,  // phi, c_ratio_sq, repulsion_scale
) -> Result<...>
```

Avec :
- D1 : `drift_janus_cosmo(a_plus, a_minus, half_dt)` — per-particle /a
- Force : CIC scatter (signs OK), Poisson, CIC gather Janus (avec cross-coupling), BH short-range Janus
- K : `kick_janus_cosmo(a_plus, a_minus, h_plus, h_minus, dt)` — per-particle /a², h
- D2 : drift_janus_cosmo

## 6. État production GPU TreePM Newton

`step_treepm_gpu` est **fonctionnel pour Newton uniforme** :
- Compile avec `--features cuda,cufft`
- Utilisé dans `janus_mu19_production.rs` ligne 135
- Tests existants : `treepm_2m_production`, `petit_pure_v3_treepm`, etc.

**Pour Janus complet, gaps identifiés au §4.**

## Critère de fin A.3 : ✅ Audit complet, 7 manques pour Janus identifiés

**Transition vers A.4** : port physique en 6 sub-steps, chaque commité séparément.
