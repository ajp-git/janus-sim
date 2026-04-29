# Audit 02 — Cartographie flux force JPP actuel

**Date** : 2026-04-29

## Pipeline actuel JPP (Barnes-Hut pur, NON TreePM)

```
janus_jpp_production.rs (binaire prod)
  │
  └─> GpuNBodySimulation::step_with_expansion_dkd_gpu_cosmo(dt, a_plus, a_minus, h_plus, h_minus)
      ├─ drift_only_cosmo  (D1: dt/2)         [src/nbody_gpu.rs:293]
      ├─ build_gpu_tree    (Karras BVH)        [src/nbody_gpu.rs:2658]
      │   ├─ compute_morton_codes
      │   ├─ par_sort_unstable_by_key (CPU)
      │   ├─ reorder_positions, reorder_signs, reorder_velocities, reorder_masses
      │   ├─ build_bvh_internal
      │   ├─ init_leaves (b_max=0)
      │   └─ reduce_com (BMAX bottom-up b_max(parent) = max(...))
      ├─ compute_forces_bvh                    [src/nbody_gpu.rs:907]
      │   └─ MAC: b_max² < theta²·r²    (theta=0.7, BMAX squared)
      ├─ kick_only_cosmo  (K: full dt)         [src/nbody_gpu.rs:324]
      │   └─ acc/a² - H·v
      └─ drift_only_cosmo  (D2: dt/2)
```

## Convention vélocité (CRITIQUE pour port TreePM)

**Convention Peebles, peculiar (documentée src/nbody_gpu.rs:3131-3138) :**

- `pos[i]` : comoving Mpc, fixed box [-L/2, L/2]
- `vel[i]` : peculiar proper velocity `v_pec = a · dx_co/dt`  [unités Mpc/Gyr]
- `acc[i]` : bare comoving accel `G·m / r_co²`  (no a-factor) — output de `compute_forces_bvh`

**Equations of motion :**
```
dx_co/dt   = v_pec / a
dv_pec/dt  = -H · v_pec  -  G·m / (a² · r_co²)  =  acc / a²  -  H · v_pec
```

## Per-particle a, H selection by sign

**Drift** (`drift_only_cosmo`, src/nbody_gpu.rs:306-310) :
```c
double a_eff = (signs[tid] > 0) ? a_plus : a_minus;
double inv_a = 1.0 / a_eff;
pos[base]     += vel[base]     * dt * inv_a;
```

**Kick** (`kick_only_cosmo`, src/nbody_gpu.rs:338-345) :
```c
double a_eff = (signs[tid] > 0) ? a_plus : a_minus;
double h_eff = (signs[tid] > 0) ? h_plus : h_minus;
double inv_a2 = 1.0 / (a_eff * a_eff);
double grav = acc[base + d] * inv_a2;
double friction = -h_eff * vel[base + d];
vel[base + d] += (grav + friction) * dt;
```

## Cross-coupling Janus φ, c̄²

Dans `step_with_expansion_dkd_gpu_cosmo` (src/nbody_gpu.rs:3162-3164) :
```rust
let phi_inv = 1.0 / self.phi;
let cross_minus_plus = self.c_ratio_sq * phi_inv * self.repulsion_scale;
let cross_plus_minus = self.phi * self.repulsion_scale;
```

Convention force dans `compute_forces_bvh` (src/nbody_gpu.rs:982, 1000) :
```c
// Force m- ← m+ : interaction = -cross_minus_plus  (m+ contribute, m- target)
// Force m+ ← m- : interaction = -cross_plus_minus  (m- contribute, m+ target)
double interaction = (my_sign > 0) ? 1.0 : -cross_minus_plus;  // for mass_plus contrib
// puis
double interaction = (my_sign < 0) ? 1.0 : -cross_plus_minus;  // for mass_minus contrib
```

Donc la convention est :
- `m+ ← m+` : factor +1 (attraction Newton)
- `m- ← m-` : factor +1 (attraction entre m- via Newton avec masse positive stockée)
- `m- ← m+` : factor `-cross_minus_plus = -c̄²·φ⁻¹·repulsion_scale` (répulsion)
- `m+ ← m-` : factor `-cross_plus_minus = -φ·repulsion_scale` (répulsion)

## Convention de stockage mass/sign (BARNES-HUT existant)

- `signs[i]` : `i32` valant `1` (m+) ou `-1` (m-) — **NOT `i8` ni `u1`**
  - Mais le snapshot V3 utilise `u1` avec `1` pour m+ et `255` (= −1 en wrap u8) pour m-
  - Conversion implicite à la lecture
- `masses[i]` : `f64` en valeur absolue (positive)
- Convention force utilise `masses[i]` × signe dérivé du `signs[]` pour tous les calculs

## Pipeline TreePM existant (non-Janus)

Pour comparaison, le pipeline TreePM dans `nbody_gpu_twopass.rs` :

```
janus_mu19_production.rs
  │
  └─> GpuNBodyTwoPass::step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt)
      ├─ drift_f32 (D1: dt/2)             [pos += vel*dt, NO /a]
      ├─ cic_scatter → rho_plus, rho_minus
      ├─ cufft_ffi::solve_device pour rho_plus → phi_plus
      ├─ cufft_ffi::solve_device pour rho_minus → phi_minus
      │   (Gaussian damping exp(-k²r_s²) avec r_s = r_cut/3)
      ├─ cic_gather → pm_forces
      │   F = -∇φ_attract + ∇φ_repel (factors 1.0)
      ├─ compute_short_range_forces (BH avec r_cut)
      │   forces_treepm_short_range avec erfc(r/(2r_s))
      ├─ add_pm_forces : acc += pm_forces
      ├─ kick_f32 (K: full dt)
      │   vel += (acc + friction)*dt, friction = -hubble·v·dtau
      └─ drift_f32 (D2: dt/2)
```

## Différences à porter pour TreePM JPP

| Aspect | Barnes-Hut JPP | TreePM existant | À porter |
|---|---|---|---|
| Drift | `/a_eff` per-particle | `pos += vel*dt` | Remplacer `drift_f32` par version cosmo |
| Kick | `acc/a_eff² - h_eff·v` per-sign | `acc + friction`, single hubble | Remplacer `kick_f32` par version cosmo |
| Cross-couplage | `cross_minus_plus, cross_plus_minus` (φ et c̄²) | factor 1.0 | Modifier `cic_gather` et `forces_treepm_short_range` |
| Conversions signs | `i32` ±1 | `i8` ou `signed char` ±1 | Vérifier interface |

## Conclusion 1.2

Convention vélocité : **Peebles peculiar** `v_pec = a·dx_co/dt`. Confirmée sur `nbody_gpu.rs` lignes 3131-3138. À reproduire **exactement** dans le port TreePM JPP.
