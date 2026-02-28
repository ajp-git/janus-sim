# Janus N-body Simulation Runs

This file documents all simulation runs with their parameters and results.
**Always consult this file before launching a new run.**

---

## Run Configuration Template

```
Run: [NAME]
Date: YYYY-MM-DD
Status: [completed | running | interrupted]

Parameters:
  N particles: X
  eta: 1.045
  z_init: 5.0
  H0: 76.0 km/s/Mpc (implicit in Janus equations)
  theta: 0.5
  dt: 0.01
  box_size: auto (100 × (N/100K)^(1/3))
  integrator: DKD + Hubble friction

Cosmological:
  tau_start: X
  tau_end: X
  dtau_per_dt: X
  Expected steps to z=0: ~12000

Results:
  S_max: X at step Y (z ≈ Z)
  KE/KE0 final: X
  Total steps: X
  Runtime: X hours

Notes:
  [Any relevant observations]
```

---

## Completed Runs

### Run: 2M_production (Reference)
Date: 2026-02-XX
Status: **completed**

Parameters:
  N particles: 2,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  dt: 0.01
  integrator: DKD + Hubble friction (step_with_expansion_dkd)

Results:
  - Segregation starts increasing: step ~1100-1200 (z ≈ 2.4)
  - **S_max = 0.694 at z = 1.8**
  - Spontaneous spatial segregation confirmed

Notes:
  - This is the reference run that validates Janus segregation
  - Hubble friction is ESSENTIAL for segregation to emerge
  - Without expansion (H=0), virialized system never segregates

---

### Run: 8M_full
Date: 2026-02-XX
Status: **completed**

Parameters:
  N particles: 8,000,000
  eta: 1.045
  z_init: 5.0
  theta: 1.5
  dt: 0.005
  integrator: step_with_expansion_dkd_gpu

Results:
  - S_max = 0.459
  - Note: theta=1.5 vs theta=0.7 for 2M — not directly comparable

---

## Completed Validation Runs

### Run: TreePM_validation_100K
Date: 2026-02-28
Status: **completed** ✅

Parameters:
  N particles: 100,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  r_cut: box/16 (6.25 Mpc)
  dt: 0.01
  steps: 5000
  integrator: TreePM (BH short-range + cuFFT PM long-range)

Results:
  - **S_max = 0.659 at step 1300 (z = 1.88)**
  - Grid artifacts: **absent** (human validated)
  - Performance: ~46ms/step (PM 8ms + BH 38ms)

Validation Criteria:
  - [x] S_max > 0.4 (got 0.659)
  - [x] z @ S_max ≈ 1.8 (got 1.88)
  - [x] No grid artifacts (visual inspection passed)

Notes:
  - TreePM production-ready after this validation
  - Centered box convention [-L/2, +L/2] confirmed correct
  - PM uses dual-grid (rho_plus, rho_minus) per FIX-009
  - Frames saved: output/treepm_validation/frame_{1000,3000}.png

---

### Run: TreePM_2M_production
Date: 2026-02-28
Status: **killed** (step ~2500) ⚠️ **PHYSICALLY INVALID**

Parameters:
  N particles: 2,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  r_cut: box/16 (16.97 Mpc)
  dt: 0.01
  steps: 12000
  integrator: TreePM (step_treepm_gpu) — OLD, not Morton
  virial_velocity: sqrt(N/box) × 0.3 ← **TOO COLD**

⚠️ **PHYSICS PROBLEM DETECTED**:
```
KE/KE₀ = 850 (should be ~1.9 max with proper Hubble friction)
Segregation onset: step 200 (z=4.69) vs expected step 1200 (z=2.4)
```

**Diagnosis**: Gravitational collapse, NOT cosmological segregation
  - virial_velocity = sqrt(2M/271) × 0.3 = 25.7 × 0.3 = 7.7 km/s
  - System was "cold" (insufficient KE to resist collapse)
  - Collapsed before Hubble friction could play its role
  - Reference 2M run used different IC generation

**Root cause**: virial_velocity factor 0.3 gives insufficient velocities
for large N in large box. System not properly virialized → immediate collapse.

**Action**: Run killed at step ~2500. Data steps 100-2500 conserved for documentation.

Final State (step ~2500):
  - z ≈ 2.5
  - S_max = 0.5261 (at step 1500, z=3.08) ← NOT PHYSICAL
  - KE/KE₀ ≈ 850 ← COLLAPSE SIGNATURE

Notes:
  - **Killed**: virial_factor=0.3 invalide, données suffisantes
  - Partial data in output/treepm_2m_production/
  - 85M production uses corrected virial_factor=0.5

---

### Run: treepm_zeldovich_100K (Validation)
Date: 2026-02-28
Status: **completed** ✅

Parameters:
  N particles: 97,336 (46³ grid)
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  r_cut: box/16 (6.19 Mpc)
  dt: 0.01
  steps: 2000
  box_size: 99.10 Mpc
  integrator: TreePM (step_treepm_gpu_morton)

  **ICs: Zel'dovich + virialized**
    - Positions: grid + sinusoidal (A=1e-3, λ=100 Mpc, σ=0.1)
    - Velocities: random, virial_factor = 0.8

Results:
  - **Onset at z = 2.46** ✅ (criterion: z ∈ [2.0, 3.0])
  - S_max = 0.9255 at step 1820 (z = 0.09)
  - Final KE/KE₀ = 28 (stable, no collapse)
  - Runtime: 0.6 min (18 ms/step)

Notes:
  - This validates Zel'dovich ICs for 85M production run
  - CSV: output/treepm_zeldovich_2026-02-28_212851/time_series.csv

---

## Current Run

### Run: 85M_treepm_production (FINAL)
Date: 2026-02-28
Status: **ready to launch** (Zel'dovich ICs validated)

Parameters:
  N particles: 85,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7 (FIX-012 validated)
  r_cut: box/16 (~59 Mpc)
  dt: 0.01
  steps: 12000
  box_size: ~947 Mpc (auto: 100 × (85M/100K)^(1/3))
  integrator: TreePM (step_treepm_gpu_morton)
  kernel: Morton + warp-coherent (optim-warpcoherent-v1.0)

  **ICs: Zel'dovich + virialized** (validated 2026-02-28)
    - Positions: grid + sinusoidal displacement (A=1e-3, λ=100 Mpc, σ=0.1)
    - Velocities: random, scaled by virial_factor = 0.8

Validation runs:
  - vf=0.3 on 2M (uniform): ❌ KE/KE₀ = 850 (collapse)
  - vf=0.8 on 100K (uniform): ✅ KE/KE₀ = 8.8 (stable)
  - **Zel'dovich + vf=0.8 on 100K**: ✅ onset z = 2.46, S_max = 0.93

Output:
  - frames every 500 steps
  - snapshots every 1000 steps (last 20 kept)
  - time_series.csv with: step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,s_max,step_time_ms

Expected Performance:
  - ~35s/step (extrapolated from 4M benchmark: 1626ms → 35s via O(N log N))
  - Total runtime: ~5 days on RTX 3060

Binary: src/bin/janus_85m_treepm.rs
Command:
```bash
docker compose run --rm dev cargo run --release --features cuda,cufft \
  --bin janus_85m_treepm
```

Notes:
  - Uses warp-coherent kernel (22x faster than baseline)
  - Uses Zel'dovich ICs (validated: onset z=2.46 on 100K)
  - First 85M TreePM production run with cosmological ICs
  - Will replace 85M_expansion (which uses old BH code)

---

### Run: 85M_expansion (SUPERSEDED)
Date: 2026-02-25
Status: **superseded** by 85M_treepm_production

Parameters:
  N particles: 85,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.5
  dt: 0.01
  box_size: 947.27
  integrator: DKD + Hubble friction (step_dkd with hubble, dtau_per_dt)

Cosmological:
  tau_start: (from CosmoInterpolator at z=5)
  tau_end: (at z=0)
  dtau_per_dt: calculated for ~12000 steps to z=0
  Expected steps to z=0: ~12000

Output:
  - frames/frame_XXXXXX.png every 200 steps
  - snapshots/snapshot_XXXXXX.bin every 200 steps
  - time_series.csv with columns:
    step,time,redshift,scale_factor,hubble,ke,ke_ratio,segregation,step_time_ms

Expected Results:
  - Segregation should start around step ~1100-1200 (z ≈ 2.4)
  - S_max expected similar to 2M run, possibly higher due to better resolution

Notes:
  - First 85M run WITH cosmological expansion
  - Previous 85M run (without expansion) showed S ≈ 1e-4 constant — system virialized, no segregation

---

## Failed/Interrupted Runs

### Run: 85M_no_expansion (INVALID)
Date: 2026-02-25
Status: **interrupted** (replaced by 85M_expansion)

Parameters:
  N particles: 85,000,000
  eta: 1.045
  integrator: step_dkd(dt, 0.0, 0.0) — **NO HUBBLE FRICTION**

Results:
  - S ≈ 1e-4 constant over 5800 steps
  - System virialized, never segregates

**Why it failed:**
  - Without Hubble friction (H=0), the virialized system is in thermal equilibrium
  - Kinetic energy prevents gravitational collapse
  - Segregation requires Hubble friction to cool velocities over cosmological time

---

## Key Lessons

1. **Hubble friction is ESSENTIAL** for segregation
   - Without expansion: S stays at ~1e-4 forever
   - With expansion: S grows from ~0.01 to ~0.7

2. **Configuration must include:**
   - z_init (typically 5.0)
   - Proper dtau_per_dt for N-body ↔ conformal time coupling
   - step_dkd(dt, hubble, dtau_per_dt) with hubble > 0

3. **Reference values:**
   - Segregation onset: z ≈ 2.4 (step ~1100-1200 for 12000 steps total)
   - S_max: 0.6-0.7 at z ≈ 1.8
   - KE should decrease as universe expands (Hubble cooling)

---

## Code Locations

- **Friedmann solver:** `src/friedmann.rs` (CosmoInterpolator, JanusParams)
- **N-body GPU (twopass):** `src/nbody_gpu_twopass.rs` (step_dkd)
- **85M binary:** `src/bin/janus_85m.rs`
- **Reference run (8M):** `src/bin/run_8m_full.rs`
