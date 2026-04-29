# Audit 01 — Inventaire module TreePM

**Date** : 2026-04-29
**Branche** : feat/treepm-jpp-port (acting as feature/treepm-integration)

## Inventaire fichiers × LOC × rôle

| Fichier | LOC | Rôle |
|---|---|---|
| `src/treepm/mod.rs` | 49 | Module entry, constants R_CUT_FACTOR=16, PM_GRID_SIZE=256 |
| `src/treepm/pm_grid.rs` | 461 | PmGrid CPU avec rustfft. Solve Poisson + CIC + interpolate force |
| `src/treepm/pm_grid_gpu.rs` | 260 | PmGridGpu wrapper (utilisable avec/sans cuFFT) |
| `src/treepm/cufft_ffi.rs` | 384 | CuFFTPoisson FFI wrapper for libcufft_wrapper.so |
| `src/treepm/splitting.rs` | 88 | Splitting polynomial x⁴ (PM/Tree weights) |
| `src/treepm/tree_short.rs` | 356 | TreePMTree CPU short-range with r_cut |
| `src/treepm/treepm_force.rs` | 267 | TreePMForce orchestrator (CPU) |
| **Total** | **1865** | |

## API publique consolidée

### `mod.rs`
- `pub const R_CUT_FACTOR: f64 = 16.0` — r_cut = box_size / 16
- `pub const PM_GRID_SIZE: usize = 256` — N_pm = 256³
- `pub fn compute_r_cut(box_size: f64) -> f64`

### `splitting.rs`
- `pub fn splitting_pm(r: f64, r_cut: f64) -> f64` — polynomial x⁴
- `pub fn splitting_tree(r: f64, r_cut: f64) -> f64` — 1 - splitting_pm

### `tree_short.rs`
- `pub struct TreePMTree { theta, r_cut, g_constant, ... }`
- `pub fn splitting_pm_weight(r, r_cut)` (legacy, == splitting_pm)
- `pub fn splitting_tree_weight(r, r_cut)`

### `pm_grid.rs` — PmGrid (CPU rustfft)
- `pub fn new(grid_size, box_size) -> Self`
- `pub fn clear()`
- `pub fn assign_mass(x, y, z, mass, sign: i8)` — CIC scatter
- `pub fn solve_poisson(g_constant)` — pure Poisson
- `pub fn solve_poisson_with_splitting(g_constant, r_s: Option<f64>)` — with Gaussian damping
- `pub fn solve_poisson_with_k_filter(g_constant, k_min: usize)` — k-mode filter
- `pub fn interpolate_force(x, y, z, sign) -> (f64, f64, f64)` — CIC gather + central diff
- `pub fn memory_bytes() -> usize`

### `pm_grid_gpu.rs` — PmGridGpu (GPU via cuFFT)
- Same API as PmGrid but uses cuFFT internally

### `cufft_ffi.rs` — CuFFTPoisson
- `pub fn new(grid_size, box_size) -> Result<Self, String>`
- `pub fn solve(rho: &[f64], g_constant, r_s) -> Result<Vec<f64>, String>` (CPU↔GPU transfer)
- `pub fn solve_filtered(rho, g_constant, r_s, k_min)`
- `solve_device(rho_ptr, phi_ptr, ...)` and `solve_device_filtered` — device-to-device, no copy

### `treepm_force.rs` — TreePMForce (orchestrator)
- `pub fn new(r_cut, grid_size, box_size, theta, softening) -> Self`
- `pub fn new_pm_only(grid_size, box_size) -> Self`
- `pub fn update(particles)` — rebuild tree + assign mass
- `pub fn compute_force(pos, sign, particles) -> Vec3`
- `pub fn compute_force_excluding(pos, sign, particles, exclude_idx)`
- `pub fn compute_all_forces(particles) -> Vec<Vec3>`

## Dette technique (TODO/FIXME/HACK)

**Aucun TODO/FIXME/HACK dans `src/treepm/`.** Module clean.

## GPU pipeline parallèle (`src/nbody_gpu_twopass.rs`)

5519 LOC. Implémentation GPU complète :

- **4 méthodes step_treepm_*** :
  - `step_treepm_hybrid(dt, pm_grid: &mut PmGrid, r_cut, hubble, dtau_per_dt)` — line 4354 — CIC sur CPU
  - `step_treepm_gpu(dt, r_cut, hubble, dtau_per_dt)` — line 4486 — full GPU, cuFFT, primary
  - `step_treepm_gpu_cached(...)` — line 4657 — tree caching A/B
  - `step_treepm_gpu_morton(...)` — line 4825 — Morton reordering

- **Kernels CUDA TreePM** (string `CUDA_KERNEL_SRC`) :
  - `drift_f32(pos, vel, dt, box_half, n)` — line 33 — simple drift `pos += vel*dt`
  - `kick_f32(vel, acc, dt, n, hubble, dtau_per_dt)` — line 50 — `vel += (acc + friction)*dt`, friction = -hubble·v·dtau
  - `cic_scatter(pos, signs, rho_plus, rho_minus, n, grid, ...)` — line 98 — CIC scatter sur 2 grilles
  - `cic_gather(pos, signs, phi_plus, phi_minus, pm_forces, ...)` — line 162 — CIC gather + central diff (ordre 2)
  - `add_pm_forces(acc, pm_forces, n)` — line 67
  - `forces_treepm_short_range(...)` — line 1670 — BH short-range avec r_cut + erfc

## Relation au pipeline JPP actuel

`janus_jpp_production.rs` ligne 907 utilise actuellement :
```rust
gpu_sim.step_with_expansion_dkd_gpu_cosmo(DT, a, a_minus, h, h_minus)
```
qui est **Barnes-Hut pur** dans `src/nbody_gpu.rs` ligne 3142 (NOT TreePM).

`janus_mu19_production.rs` ligne 135 utilise :
```rust
sim.step_treepm_gpu(DT, R_CUT, state.h_plus, 1.0)
```
qui est TreePM mais signature single-hubble (pas de couplage Janus complet).

## Conclusion Phase 1.1

Module TreePM **présent, complet, production-ready pour Newton**. Il manque pour JPP :
1. Couplage Janus φ et c̄² dans cic_gather + forces_treepm_short_range
2. Per-particle a_plus/a_minus dans drift_f32 et kick_f32
3. Per-particle h_plus/h_minus dans kick_f32
4. Pas de TODO résidu dans le code TreePM existant
