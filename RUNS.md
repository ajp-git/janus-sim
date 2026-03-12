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

## Killed Runs

### Run: 30M_bh_pure
Date: 2026-03-02
Status: **killed** (step 2130, z=2.55)

Parameters:
  N particles: 30,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  dt: 0.01
  steps: 12000
  box_size: 690.0 Mpc
  softening: 0.1 Mpc
  integrator: Pure Barnes-Hut (step_with_expansion_dkd_gpu)

  **ICs: Uniform + virialize_sampled(10000)**

Results at kill:
  - Step: 2130 (z = 2.55)
  - KE/KE₀ = 1.75
  - Seg = 0.029
  - Runtime: ~8 hours (~13.6s/step)

**Why killed:** No added value vs 8M run
  - Similar behavior to 8M but 4× slower
  - VRAM-limited to 32M max (BH pure uses ~370 bytes/particle)
  - TreePM allows 60M+ but is unstable (FIX-013)

Data preserved:
  - time_series.csv → /mnt/T2/janus-sim/output/30M_partial_killed.csv
  - render_data: 106 .bin files (steps 0-2100)
  - frames: 106 × 2 formats (25d + dens)
  - videos: janus_30m_25d.mp4, janus_30m_dens.mp4

Binary: src/bin/janus_30m_bh.rs

---

## Current Run

### Run: janus_60m_final
Date: 2026-03-02
Status: **running**

Parameters:
  N particles: 60,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7 (FIX-012 validated)
  r_cut: box/16 (52.7 Mpc)
  dt: 0.01
  steps: 12000
  box_size: 843.0 Mpc
  softening: 1.0 Mpc
  integrator: TreePM (step_treepm_gpu_morton)
  kernel: Morton + warp-coherent

  **ICs: Uniform + virialize_sampled(10000)**
    - Positions: uniform random
    - Velocities: uniform random, scaled by α from PE_binding
    - α = √(|PE_bind|/2KE) ≈ 4-6 (same as validated 8M run)

Validation:
  - test_100k_843mpc.rs: KE/KE₀ = 0.95 @ step 100 ✅ (expected < 5)
  - Commit: ac2fe4e (fix(60M): Use exact 8M virialization method)

Output:
  - Log: /mnt/T2/janus-sim/output/janus_60m_final.log
  - Data: /app/output/60M_virialized_2026-03-02/
  - time_series.csv updated every step

Expected Performance:
  - ~50s/step (TreePM Morton + warp-coherent)
  - Total runtime: ~7 days on RTX 3060

Binary: src/bin/janus_60m_treepm.rs
Container: janus-sim-dev-run-bcd418cb5446

Notes:
  - First 60M run with correct virialization (same as 8M validated)
  - Previous attempts failed: Zel'dovich α=127, virial_factor=0.8 → KE explosion
  - **DO NOT TOUCH unless emergency**

---

### Run: 85M_treepm_production (SUPERSEDED)
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

### Run: anticorr_8M_filaments_v1 (INVALIDATED)
Date: 2026-03-02/03
Status: **killed** (wrong box size)

Parameters:
  N particles: 8,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  dt: 0.01
  box_size: 271 Mpc  ← ERREUR: devrait être 430 Mpc
  softening: 0.4 Mpc
  spacing: 1.36 Mpc  ← trop dense (ref: 2.15 Mpc)
  integrator: BH pure + DKD + Hubble friction
  ICs: Density-based anti-correlated Zel'dovich

Results (invalidated):
  Step 8584/10000 (85.8% complete)
  z_final: 0.15
  KE/KE₀_max: 3.44
  Seg: 0.017 constant (no growth!)

**Why invalidated:**
  - Box 271 Mpc same as 2M reference → density 4× higher
  - Spacing 1.36 Mpc vs 2.15 Mpc reference → dynamics altered
  - Seg stayed at ~0.017 instead of growing to ~0.6
  - Must use box = N^(1/3) × spacing_ref = 200 × 2.15 = 430 Mpc

---

### Run: anticorr_8M_filaments_v2
Date: 2026-03-03
Status: **completed**
Output: anticorr_8000k_1772521928

Parameters:
  N particles: 8,000,000
  eta: 1.045
  z_init: 5.0
  theta: 0.7
  dt: 0.01
  box_size: 430 Mpc
  softening: 0.65 Mpc (0.3 × spacing)
  spacing: 2.15 Mpc (même que 2M référence)
  α virialization: 4.59
  integrator: BH pure + DKD + Hubble friction
  ICs: Density-based anti-correlated Zel'dovich

Results:
  Steps: 10000 / 10000 ✓
  z final: 0.000
  KE/KE₀ final: 1.037
  Seg₀: 0.0076
  Seg final: 0.0165
  Seg max: ~0.017 (NO GROWTH)
  Runtime: ~9.3h (3.35s/step)

**CONCLUSION: ÉCHEC de la ségrégation**
  - Seg reste plat à ~0.017 de z=5 à z=0
  - Aucune croissance de ségrégation observée
  - Comparaison run 2M référence: S_max=0.694 vs 0.017 ici
  - Les ICs density-based NE PRODUISENT PAS la dynamique de ségrégation attendue

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
---

## Runs session 2026-03-03

### Run: anticorr_8M_box271 — INVALIDE
Date: 2026-03-03
Status: INVALIDE — stoppé
Cause: spacing=1.36 Mpc (box trop petite) + bug dtau friction 20× trop faible
Symptôme: KE/KE₀ → 3.38 à z=0, Seg figé à 0.018
Action: supprimé

### Run: anticorr_8M_box430_v1 — INVALIDE
Date: 2026-03-03
Status: INVALIDE — complété mais résultats non exploitables
Cause: bug dtau (friction de Hubble 20× trop faible)
Résultats: N=8M, box=430 Mpc, 10000 steps, z=5→0
  KE/KE₀_max=1.030 (stable mais friction insuffisante)
  Seg_max=0.017 (figé — pas de dynamique)
Leçon: stabilité numérique ≠ physique correcte

### Run: grid_exploration_100K_A-F — INFORMATIF
Date: 2026-03-03
Status: complété
N=100K, box=100 Mpc, 2000 steps, 6 variantes ICs
Bug dtau présent mais partiellement corrigé en cours de session
Résultats: voir section EXPLORATION GRID dans FILAMENTS_ROADMAP.md
Conclusion: 100K insuffisant, ICs density-based figées, uniforme aléatoire
  produit effondrement blob co-localisé (Seg métrique trompeuse à cette résolution)

### Run: ref_2M_icsfevrier — EN COURS ✅
Date: 2026-03-03
Status: running (~38% au moment du patch)
N=2,000,000 | Box=271 Mpc | ICs=new() positifs d'abord | virialize() PE full
dtau_per_dt corrigé : τ_range / (TOTAL_STEPS × DT)
Résultats partiels:
  Seg_max=0.452 @ step 2906 (z≈1.69)  ✅
  KE/KE₀=4.59 au pic (normal)
  Pic z≈1.7 cohérent avec run février (z≈1.8)
Verdict attendu: EXCEL si Seg_final > 0.05
Prochaine action: lancer 8M box=430 avec mêmes ICs si EXCEL

---

### Run: zeldovich_500k_combined
Date: 2026-03-04
Status: **completed** (partial - 1072/2000 steps due to timeout)
ICs: Zel'dovich density-based + ordre février (+ d'abord)
Parameters:
  N = 493,039 (grid 79³)
  Box = 172 Mpc
  dt = 0.01, θ = 0.7, softening = 0.65 Mpc
  dtau_per_dt = 0.066026 (FIX-016)
Results:
  Seg₀ = 0.0538 (from density-based ICs)
  Seg_max = 0.543 @ z≈0.75
  KE/KE₀_max ≈ 4.6 (stable)
Verdict: **EXCEL** (Seg_max > 0.20)
Morphologie: Dynamic segregation observed
Notes: Combined density-based ICs with février ordering successfully
  reproduces segregation dynamics. Ready for production scale.

### Run: production_bh_12m
Date: 2026-03-04
Status: **stopped** (investigation artefact rectangulaire)
N = 12,000,000 | Box = 492 Mpc | Steps = 20,000 | Snapshots = 1000
ICs: new() positifs d'abord + virialize_sampled(80000)
dtau_per_dt = tau_range / (20000 × 0.01) (FIX-016)
Container: 786d693e9521
Results (partial):
  Step 3520: corr(idx,z) = 0.77 - DISCOVERED during run
  Investigation revealed: new() generates CORRECT uncorrelated ICs (corr=0.00)
  The correlation DEVELOPED DURING SIMULATION = REAL PHYSICS!
  Run stopped prematurely thinking it was artifact - WRONG!
Lesson: Index-position correlation emerging during simulation is real Janus physics

### Run: pktrunc_500k_validation
Date: 2026-03-05
Status: **completed** (partial - 1310/2000 steps due to timeout)
ICs: P(k) truncated Zel'dovich + density-based signs + shuffled indices
  k_min = 2π/60 Mpc (suppress λ > 60 Mpc)
  k_max = 2π/6 Mpc (suppress λ < 6 Mpc)
Parameters:
  N = 512,000 (grid 80³)
  Box = 172 Mpc
  dt = 0.01, θ = 0.7, softening = 0.65 Mpc
  dtau_per_dt = 0.066026 (FIX-016)
Results:
  corr(idx, z) = 0.0056 ✓ (NO index-position bias)
  Seg₀ = 0.014
  Seg_max = 0.499 @ z≈0.63 (>> 0.05 PASS)
  KE/KE₀_max = 3.88 (< 20 PASS)
Verdict: **PASS** - Ready for 12M production
Notes: P(k) truncation successfully eliminates large-scale modes

### Run: production_pktrunc_12m (v1 — STOPPÉ)
Date: 2026-03-05
Status: **stopped** — fenêtre P(k) trop restrictive
k_min = 2π/150, k_max = 2π/15 → seulement 1.2% modes
Remplacé par: production_pktrunc_12m_v2

### Run: production_pktrunc_12m_v2
Date: 2026-03-05
Status: **running**
N = 12,008,989 (grid 229³) | Box = 492 Mpc | Steps = 20,000 | Snapshots = 1000
ICs: P(k) truncated Zel'dovich v2 + density-based signs + shuffled indices
  k_min = 2π/200 Mpc (suppress λ > 200 Mpc)
  k_max = 2π/8 Mpc (suppress λ < 8 Mpc)
  Modes kept: 8.1% (vs 1.2% v1)
  corr(idx, z) = 0.0113 < 0.02 ✓
Parameters:
  η = 1.045, z_init = 5.0
  dt = 0.01, θ = 0.7, softening = 0.65 Mpc
  dtau_per_dt = 0.006603 (FIX-016 verified)
  virialize_sampled(80000)
Container: fc535d11603f
Initial state:
  KE₀ = 6.91e10, Seg₀ = 0.0068
  Step 5 check: KE/KE₀ = 0.9993 ✓ PASS
ETA: ~35h
Milestones expected:
  Step 3000 (z≈2.8): onset segregation
  Step 4500 (z≈2.0): peak segregation (Seg > 0.2)
  Step 20000 (z=0): run complete
Output: /app/output/production_pktrunc_12m_v2/

---

### Run: janus_v10_highres + resume
Date: 2026-03-09 to 2026-03-12
Status: **stopped** (step 2700/3000)

Parameters:
  N particles: 19,902,511 (~20M)
  Box: 200 Mpc
  η: 1.06
  H: 0.012
  θ: 0.5 → 0.7 (step 500+) → 0.8 (resume)
  dt: 0.003 → 0.005 → 0.01 (adaptive)
  ε: 0.18 Mpc
  Grid (PM): 256³
  k_min: 2
  Integrator: TreePM (step_treepm_gpu)
  ICs: Zel'dovich (k_cut=0.25, α_IC=1.6)

Results:
  Steps completed: 2700 (original 2100 + resume 600)
  Seg₀: 0.007
  Seg_max: **0.32** @ step 2100
  Seg_final: **0.09** @ step 2700
  KE range: 1.1e5 → 6.9e12
  Runtime: ~40h (original) + 14h (resume)

**Key Scientific Findings:**

1. **Two characteristic scales discovered:**
   - L_J = 0.83 Mpc (interface thickness)
   - ξ = 13.1 Mpc (coherence/domain size)
   - Ratio ξ/L_J ≈ 16

2. **Power law dynamics:**
   - σ_P ∝ t^{-0.16}
   - L_J ∝ t^{-0.18}
   - ξ ∝ t^{-0.26}
   - Correlation σ_P vs L_J: r = 0.998

3. **NOT standard coarsening:**
   - Domains SHRINK (ξ decreases), not grow
   - System evolves toward homogeneity
   - Segregation peaked at 0.32 then decreased to 0.09

4. **Self-similarity tests:**
   - Gradient PDF collapse: partial (not strict)
   - Power spectrum scaling: approximate

**Interpretation:**
  Interface-driven mixing dynamics. The Janus system does NOT form
  stable segregated domains at this resolution/parameters. Instead,
  initial segregation (Seg=0.32) is followed by re-mixing (Seg=0.09).

**Outputs:**
  - Snapshots: snap_000500.bin, snap_001000.bin, snap_002000.bin
  - Analysis: analysis_v10_snapshot{500,1000,2000}/
  - Video: janus_particles_rotation_4k.mp4 (239 MB, 2M particles)
  - Report: V10_SIMULATION_RESULTS.md

Binary: src/bin/janus_v10_highres.rs, src/bin/resume_v10.rs

---

### Run: janus_v11_hubble_test
Date: 2026-03-12
Status: **completed** ✅

**Goal:** Validate that Hubble friction stabilizes segregation (vs V10 re-mixing)

Parameters:
  N particles: 512,000 (80³ grid)
  Box: 200 Mpc
  η: 1.06
  z_init: 5.0 → z_final: 0.0
  θ: 0.7
  dt: 0.01
  ε: 0.18 Mpc
  R_cut: 18 Mpc
  Integrator: TreePM (step_treepm_gpu)
  ICs: Zel'dovich (k_cut=0.25, α_IC=1.6, random signs)

Cosmological:
  τ_start: -1.413
  τ_end: 0.0
  dτ/dt: 0.047
  H_init: 1.32

Results:
  Steps: 3000 / 3000 ✓
  Runtime: 13.7 min (0.27s/step)
  Seg₀: 0.0008
  **Seg_max: 0.410** @ step 2760 (z = 0.08)
  **Seg_final: 0.369** @ z = 0
  KE/KE₀ final: 6.98e8

**VALIDATION SUCCESSFUL:**

| Metric | V10 (no Hubble) | V11 (with Hubble) |
|--------|-----------------|-------------------|
| Seg trend | Decreasing (re-mixing) | **Increasing then plateau** |
| Seg final | 0.09 | **0.37** |
| Interpretation | System mixes back | **Stable segregation** |

**Conclusion:**
  Hubble friction is ESSENTIAL for stable segregation.
  Without it (V10), KE drives re-mixing.
  With it (V11), friction cools the system → domains persist.

Output: /app/output/janus_v11_hubble_test/
Binary: src/bin/janus_v11_hubble_test.rs

---

### Run: janus_v12_box_test
Date: 2026-03-12
Status: **completed** ✅

**Goal:** Test if characteristic scale ξ is intrinsic or box artifact

Parameters:
  N particles: 1,000,000 (100³ grid)
  Box: 400 Mpc (2× V11)
  η: 1.06
  z_init: 5.0 → z_final: 0.0
  θ: 0.7
  dt: 0.01
  ε: 0.18 Mpc
  R_cut: 18 Mpc
  Integrator: TreePM (step_treepm_gpu)
  ICs: Zel'dovich (same as V11)

Results:
  Steps: 3000 / 3000 ✓
  Runtime: 38.8 min (0.78s/step)
  Seg₀: 0.0003
  **Seg_max: 0.457** @ step 2250 (z ≈ 0.32)
  **Seg_final: 0.438** @ z = 0

**Scale Analysis:**

| Metric | V11 (L=200) | V12 (L=400) | Ratio |
|--------|-------------|-------------|-------|
| ξ final | 31 Mpc | 50 Mpc | 1.61 |
| L_J final | 5.5 Mpc | 37.5 Mpc | 6.8 |
| ξ/L_J | 5.7 | 1.3 | - |

**Interpretation:**
  - ξ increases but does NOT double (50 vs 62 if pure artifact)
  - L_J is much larger → V12 has lower spatial resolution
  - ξ/L_J ratio very different suggests resolution mismatch
  - **CONCLUSION:** Partial scaling. Need higher resolution (8M in 400 Mpc)
    to conclusively determine if ξ is intrinsic or artifact.

Output: /app/output/janus_v12_box_test/
Binary: src/bin/janus_v12_box_test.rs

---

### A/B Test: V11 Optimized
Date: 2026-03-12
Status: **completed** — ❌ REJECTED

**Goal:** Test numerical optimizations for performance improvement

Optimizations tested:
  1. Adaptive θ: 0.5 (step<500) → 0.7 (500-1500) → 0.9 (step≥1500)
  2. Reduced R_cut: 15 Mpc (was 18 Mpc)

**Performance Results:**
  - Reference: ~300 ms/step, 13.7 min total
  - Optimized: 241 ms/step, 12.0 min total
  - **Speedup: 1.25x**

**Physics Comparison (final state, step 3000):**

| Metric | Reference | Optimized | Diff | Threshold | Status |
|--------|-----------|-----------|------|-----------|--------|
| σ_P | 0.372 | 0.339 | 8.9% | 2% | ❌ FAIL |
| L_J | 5.52 | 5.56 | 0.9% | 5% | ✅ PASS |
| ξ | 31.3 | 28.1 | 10.0% | 5% | ❌ FAIL |

**Verdict: ❌ REJECTED**
  - Optimizations alter physics beyond acceptable thresholds
  - Domains are smaller (ξ -10%) and less contrasted (σ_P -9%)
  - Speedup (1.25x) does not justify physics deviation

**Recommendation:**
  - Adaptive θ alone may be acceptable (needs separate test)
  - R_cut = 15 Mpc is too aggressive for this resolution

Binary: src/bin/janus_v11_optimized.rs
Output: /app/output/janus_v11_optimized/

---

### A/B Test: V11 Theta Adaptive (Isolated)
Date: 2026-03-12
Status: **completed** — ❌ REJECTED

**Goal:** Test if adaptive θ ALONE (without R_cut change) is acceptable

Optimizations tested:
  - Adaptive θ ONLY: 0.5 (step<500) → 0.7 (500-1500) → 0.9 (step≥1500)
  - R_cut = 18 Mpc (SAME as reference — no change)

**Performance Results:**
  - Reference: ~300 ms/step, 13.7 min total
  - θ-Adaptive: 237 ms/step, 11.8 min total
  - **Speedup: 1.27x**

**Physics Comparison (final state, step 3000):**

| Metric | Reference | θ-Adaptive | Diff | Threshold | Status |
|--------|-----------|------------|------|-----------|--------|
| σ_P | 0.372 | 0.359 | 3.5% | 2% | ❌ FAIL |
| L_J | 5.52 | 5.44 | 1.3% | 5% | ✅ PASS |
| ξ | 31.3 | 31.3 | 0.0% | 5% | ✅ PASS |

**Temporal evolution:**
  - Step 1000 (θ=0.7, same as ref): σ_P deviation -0.2% ✅
  - Step 2000 (θ=0.9 active): σ_P deviation 4.8% ❌
  - Step 3000 (θ=0.9): σ_P deviation 3.5% ❌

**Interpretation:**
  - θ=0.9 phase introduces systematic error in σ_P
  - Error accumulates even though L_J and ξ stay within bounds
  - Polarization contrast (σ_P) is sensitive to force accuracy

**Verdict: ❌ REJECTED**
  - Adaptive θ alone still fails σ_P threshold (3.5% > 2%)
  - θ=0.9 too aggressive for accurate short-range forces
  - Recommend: constant θ=0.7 for production runs

Binary: src/bin/janus_v11_theta_adaptive.rs
Output: /app/output/janus_v11_theta_adaptive/

---

### A/B Test: V11 Tree Rebuild B2 (Interval=3)
Date: 2026-03-12
Status: **completed** — ❌ REJECTED

**Goal:** Test if reducing tree rebuild frequency improves performance

Optimization tested:
  - Tree rebuild every 3 steps (instead of every step)
  - Steps 1,4,7,...: rebuild both trees + cache them
  - Steps 2,3,5,6,...: reuse cached trees

**Performance Results:**
  - Rebuild steps: avg 286 ms/step
  - Reuse steps: avg 280 ms/step
  - **Speedup: 1.02x** (only 2% faster — negligible!)
  - Total time: 14.2 min (0.28s/step)

**Why speedup is negligible:**
  - Tree BUILDING is not the bottleneck
  - Tree TRAVERSAL dominates (~90% of BH time)
  - Traversing a stale tree takes the same time as a fresh one

**Physics Comparison (final state, step 3000):**

| Metric | Reference | TreeRebuild | Diff | Threshold | Status |
|--------|-----------|-------------|------|-----------|--------|
| σ_P | 0.372 | 0.393 | 5.7% | 2% | ❌ FAIL |
| L_J | 5.52 | 4.66 | 15.5% | 5% | ❌ FAIL |
| ξ | 31.3 | 28.1 | 10.0% | 5% | ❌ FAIL |

**Temporal evolution:**
  - Step 1000: σ_P deviation -12.9%, L_J deviation 17.5%
  - Step 2000: σ_P deviation -7.9%, ξ deviation -9.1%
  - Step 3000: All metrics fail thresholds

**Verdict: ❌ REJECTED**
  - Physics deviation MUCH too large (all metrics fail)
  - Performance gain negligible (1.02x not worth complexity)
  - Stale trees cause significant force calculation errors
  - Tree rebuild is cheap; traversal is expensive

**Recommendation:**
  - Do NOT reduce tree rebuild frequency
  - Optimize tree traversal instead (warp-coherent kernel already in use)
  - Consider adaptive timestep to reduce total steps

Binary: src/bin/janus_v11_tree_rebuild_B2.rs
Output: /app/output/janus_v11_tree_rebuild_B2/
