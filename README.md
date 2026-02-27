# Janus Cosmological Model — GPU N-body Simulation

Numerical validation of Jean-Pierre Petit's Janus bimetric cosmological model.  
Independent research project — February 2026.

---

## Overview

This project provides a full numerical validation pipeline for the Janus model:

1. **Phase 1a** — Fit of 1590 Type Ia supernovae (Pantheon+ catalog) using the exact analytical formula from D'Agostini & Petit (2018)
2. **Phase 1b** — CPU Barnes-Hut N-body simulation (1M particles, validated)
3. **Phase 1c** — GPU CUDA Barnes-Hut N-body simulation with virialized initial conditions
4. **Phase 2** — Filament formation analysis: linear theory + Yukawa screening tests

The Janus model replaces dark matter and dark energy with a bimetric framework featuring two coupled metrics — one for positive masses, one for negative masses — with specific interaction rules in the Newtonian limit.

---

## Physical Model

### Janus Interaction Rules (Newtonian Limit)

| Pair | Interaction | Sign |
|------|------------|------|
| Positive ↔ Positive | Attraction | +G |
| Negative ↔ Negative | Attraction | +G |
| Positive ↔ Negative | **Repulsion** | −G |

This eliminates the "runaway" problem present in earlier negative mass models (Bondi 1957).

### Single Free Parameter

**η = |ρ̄₀|/ρ₀** — ratio of negative to positive mass density

- η = 1 : perfectly symmetric universe (no net acceleration)
- η = 1.045 : our Pantheon+ fit — universe "just barely" accelerating
- q₀ = (1−η)/(1+η) : deceleration parameter

### Coupled Friedmann Equations

From Petit & D'Agostini (2014):

```
ä  = −1.5 · E / a²     (positive sector)
ā̈  = +1.5 · E / ā²     (negative sector)
```

where E = Ω₊ − Ω₋ is conserved. When E < 0 (η > 1): positive sector accelerates, negative sector decelerates.

---

## Phase 1a — Pantheon+ Fit Results

**Dataset**: 1590 Type Ia supernovae, Pantheon+ SH0ES 2022  
**Formula** (D'Agostini & Petit 2018, eq. 5):

```
μ = 5·log₁₀[z + z²·(1−q₀) / (1 + q₀·z + √(1 + 2·q₀·z))] + cst
```

| Parameter | Value | Note |
|-----------|-------|------|
| η | **1.045** | Single free parameter |
| q₀ | **−0.022 ± 0.015** | Near-flat acceleration |
| χ²/dof | **0.607** | Excellent fit |

---

## Phase 1c — N-body Simulation Results

### GPU Implementation

| Feature | Detail |
|---------|--------|
| Language | Rust + CUDA (cudarc) |
| Precision | f64 throughout |
| Tree | Linear octree, GPU Karras build |
| Hardware | NVIDIA RTX 3060 12GB |
| Speedup | **39.6×** vs CPU baseline |

**Performance** (RTX 3060, f64):

| N | θ | ms/step |
|---|---|---------|
| 2M | 0.7 | 2,370 |
| 2M | 1.5 | 398 |
| 8M | 1.5 | ~11,400 |

### Production Runs

| Run | N | θ | S_max | z(S_max) | Runtime |
|-----|---|---|-------|----------|---------|
| 500K | 500,000 | 0.7 | 0.513 | ~1.8 | 4h |
| 2M | 2,000,000 | 0.7 | **0.694** | 1.8 | ~14h |
| 8M | 8,000,000 | 1.5 | 0.459 | 2.07 | 3h |

Spontaneous spatial segregation S_max = 0.694 at z=1.8 with 2M particles (+35% vs 500K).

> Note: 8M run used θ=1.5 vs θ=0.7 for 2M — a matched comparison requires a θ=0.7 run at 8M.

---

## Phase 2 — Filament Formation Analysis

### Problem Identified

All runs produce a **spherical blob** — not a cosmic filament network. Linear perturbation theory explains why.

### Linear Theory

Two-fluid coupling matrix with cross-repulsion α:

```
M = [ ρ̄₊     −α·ρ̄₋ ]
    [ −α·ρ̄₊   ρ̄₋   ]

λ₊ = ρ̄(1+α) > 0  →  segregation (blob)     ✅ observed
λ₋ = ρ̄(1−α)      →  filament growth mode
```

**With α=1 (current code): λ₋ = 0 exactly.**

The filament mode is frozen — regardless of density asymmetry (ρ₋/ρ₊ = 2.23 for η=1.045), cosmological expansion H(t), or non-linear effects. This is a fundamental property of the α=1 model, not a numerical artifact.

See `janus_linear_analysis.md` (769 lines, 22 sections) for the complete derivation, validated by 5 independent AI systems.

### Experimental Validation — Anisotropic Mode Test

Single-mode perturbation δ(x) = A·sin(kₓx):

| Metric | Run A (α=0) | Run B (α=1) | Ratio |
|--------|-------------|-------------|-------|
| δk growth | +1047% | +262% | **4.0× suppressed** |
| σx collapse | −7.35% | −3.35% | **2.2× suppressed** |

α=1 suppresses ~75% of anisotropic growth. Consistent with λ₋ ≈ 0. The residual 25% growth comes from non-linear effects at A=10% amplitude.

### Yukawa Screening Tests

Proposed solution:

```rust
// Scale-dependent coupling — 3 lines in the CUDA kernel
let alpha_r = 1.0 - epsilon * (-r / r_c).exp();
let interaction = if sign_i == sign_j { 1.0 } else { -alpha_r };
```

This preserves Janus symmetry at large scales (r ≫ r_c) while restoring effective gravity at structural scales. Corresponds to a massive mediator — compatible with bimetric relativity.

**Results** (6 parameter sets, N²-verified — not a Barnes-Hut artifact):

| Run | ε | r_c | α(5 Mpc) | Δ vs Janus |
|-----|---|-----|----------|------------|
| C | 0.3 | 40 Mpc | 0.74 | ~0% |
| D | 0.3 | 10 Mpc | 0.82 | ~0% |
| E | 0.7 | 40 Mpc | 0.38 | ~0% |
| F | 0.7 | 10 Mpc | 0.57 | ~0% |

**Finding**: On a uniform grid, all ±pairs are at the same distance (~18 Mpc), so α(r) acts as a uniform scale factor — no differential effect on mode growth.

**Conclusion**: Yukawa requires non-uniform ICs (Zel'dovich with anti-correlation δ₋ = −δ₊) to produce a measurable differential effect.

### Open Question for J.-P. Petit

Is α=1 a fundamental constraint of the Janus action? If so, the model cannot produce filaments via linear gravitational instability — a significant tension with observed large-scale structure (SDSS cosmic web). If a massive mediator between sectors is permitted, α(k) Yukawa is a natural extension.

---

## Key Bugs Fixed

| Bug | Root Cause | Impact |
|-----|-----------|--------|
| Wrong acceleration equations | Local densities instead of conserved E | Incorrect dynamics |
| Analytical/numerical inconsistency | Mixed ΛCDM H(z) with Janus accelerations | 0.4–0.8 mag offset |
| Particle-Mesh failure | PM smooths short-range interactions | Zero segregation |
| COM with periodic BC | Simple average ignores wrapping | Invalid metric |
| GPU rsqrt() in f64 | Float intrinsic, implicit promotion | CPU/GPU divergence |
| Artificial initial segregation | Different COM references | Seg₀ = 0.49 (150× too high) |
| Janus virialization | Standard PE_total > 0 for mixed system | KE_target < 0 |

### Janus Virialization

Standard (2KE + PE_total = 0) fails: with η ≈ 1, repulsive +/− pairs dominate → PE_total > 0.

**Solution**: virialize on PE_binding (same-sign pairs only):
```
α = √(|PE_binding| / (2·KE))   →   α = 4.57
```

---

## Roadmap

### Completed ✅
- Phase 1a: Pantheon+ fit (η=1.045, χ²/dof=0.607)
- GPU Barnes-Hut with 39.6× speedup (Karras tree build)
- Virialized ICs (PE_binding method)
- Hubble friction (z=5 → z=0)
- Production runs: 500K, 2M, 8M particles
- Video pipeline (3-panel, 1080p)
- Linear analysis: α=1 → λ₋=0 → filaments impossible (`janus_linear_analysis.md`)
- Anisotropic mode test: 4× suppression confirmed experimentally
- Yukawa kernel implementation + N² diagnostic

### In Progress 🔄
- Zel'dovich ICs with P(k) and anti-correlation δ₋ = −δ₊
- Run 16–32M with Yukawa + Zel'dovich ICs

### Planned
- Morphological analysis: ξ(r), Minkowski functionals
- Contact J.-P. Petit with full experimental + theoretical results
- η=1.0 edge case test

---

## TreePM Architecture (New in v1.0)

### Motivation

Barnes-Hut θ=0.7 shows grid artifacts with Janus +/− interactions due to aggressive cell approximation at intermediate distances. TreePM provides a cleaner split: FFT for long-range, Tree for short-range.

### Architecture

```
Force_total(i) = Force_PM_longrange(i)  + Force_Tree_shortrange(i)
                 (FFT, r > r_cut)         (BH, r < r_cut)

PM dual-grid (FIX-009):
  ρ⁺ grid for positive masses
  ρ⁻ grid for negative masses

  F_on_+ = -∇φ⁺ + ∇φ⁻  (attracted by +, repelled by -)
  F_on_- = -∇φ⁻ + ∇φ⁺  (attracted by -, repelled by +)

Splitting:
  PM weight = (r/r_cut)⁴  → 0 at r=0, 1 at r≥r_cut
  Tree weight = 1 - PM weight
  k-space: G(k) × exp(-k²r_s²)  with r_s = r_cut/3
```

### Validation Results

| Test | Result |
|------|--------|
| PM isotropy (64³) | σ = 0.12° < 2° ✓ |
| PM isotropy (128³) | σ = 0.03° < 2° ✓ |
| Force continuity at r_cut | 8.9% jump < 10% ✓ |
| All 4 Janus sign combinations | Correct ✓ |
| KE stability (100 steps) | KE/KE₀ = 1.000 ✓ |
| Memory (256³ grid) | 512 MB < 2 GB ✓ |

### Performance (CPU rustfft + Rayon)

| N | PM (s) | Force (s) | Total (ms/step) |
|---|--------|-----------|-----------------|
| 10K | 0.051 | 0.012 | 63 |
| 50K | 0.073 | 0.226 | 298 |
| 100K | 0.120 | 0.897 | 1017 |

Note: GPU cuFFT would significantly accelerate PM phase.

### Files

```
src/treepm/
├── mod.rs           # Module declaration
├── pm_grid.rs       # Dual-grid PM with Gaussian splitting
├── splitting.rs     # Real-space x⁴ splitting function
├── tree_short.rs    # Barnes-Hut with r_cut cutoff
└── treepm_force.rs  # Combined PM + Tree force calculator

src/bin/
├── treepm_benchmark.rs  # Performance benchmarking
└── treepm_validate.rs   # Physics validation run

tests/
├── treepm_physics_8p.rs  # 8-particle Janus sign tests
├── treepm_isotropy.rs    # PM isotropy validation
└── treepm_continuity.rs  # Force continuity at r_cut
```

See `TREEPM_ROADMAP.md` for full implementation details.

---

## Project Structure

```
janus-sim/
├── src/
│   ├── lib.rs                    # Constants, Janus interaction rules
│   ├── friedmann.rs              # Coupled FLRW + CosmoInterpolator
│   ├── nbody.rs                  # CPU N-body (Barnes-Hut, Rayon)
│   ├── nbody_gpu.rs              # GPU N-body (CUDA, f64, Yukawa kernel)
│   ├── analysis.rs               # χ² fitting on Pantheon+
│   ├── treepm/                   # TreePM hybrid force calculator (NEW)
│   │   ├── mod.rs
│   │   ├── pm_grid.rs            # Dual-grid PM with Gaussian splitting
│   │   ├── splitting.rs          # Real-space splitting functions
│   │   ├── tree_short.rs         # BH tree with r_cut cutoff
│   │   └── treepm_force.rs       # Combined PM + Tree
│   └── bin/
│       ├── friedmann.rs          # Friedmann solver + SNIa fit
│       ├── nbody_overnight.rs    # GPU production binary
│       ├── treepm_benchmark.rs   # TreePM performance (NEW)
│       ├── treepm_validate.rs    # TreePM physics validation (NEW)
│       └── ...
├── scripts/
│   ├── render_overnight.py       # 3-panel frame renderer
│   └── batch_render.py           # Parallel batch rendering
├── data/
│   └── Pantheon+SH0ES.dat
├── output/                       # Results (not tracked)
├── VALIDATION_RULES.md
├── KNOWN_FIXES.md
├── janus_roadmap.md              # Full roadmap with results per day
├── janus_linear_analysis.md      # Linear perturbation theory (769 lines)
├── Cargo.toml
├── docker-compose.yml
└── README.md
```

---

## Installation

```bash
# Prerequisites: Docker, NVIDIA driver, nvidia-container-toolkit
git clone https://github.com/YOUR_USERNAME/janus-nbody.git
cd janus-nbody
nvidia-smi
docker compose build
```

---

## Usage

```bash
# Pantheon+ fit
docker compose run --rm dev cargo run --release --bin friedmann

# Production N-body (2M particles)
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- --n 2000000 --eta 1.045 --dt 0.005 --steps 6000

# Anisotropic mode test
docker compose run --rm dev cargo run --release --features cuda --bin test_anisotropic

# Yukawa diagnostic (N² vs BH)
docker compose run --rm dev cargo run --release --features cuda --bin test_yukawa_n2
```

---

## References

1. **Petit, J.-P., Margnat, S. & Zejli, H.** (2024). *The Janus Cosmological Model*. Eur. Phys. J. C 84, 1226.
2. **D'Agostini, G. & Petit, J.-P.** (2018). *Constraints on Janus Cosmological model from recent observations of supernovae type Ia*. Astrophys. Space Sci. 363, 139.
3. **Petit, J.-P. & D'Agostini, G.** (2014). *Negative mass hypothesis in cosmology and the nature of dark energy*. Astrophys. Space Sci. 354, 611.
4. **Scolnic, D. et al.** (2022). *The Pantheon+ Analysis*. ApJ 938, 113.
5. **Lane, Z.G. et al.** (2024). *ΛCDM calibration bias in Pantheon+*. MNRAS. arXiv:2311.01438.
6. **Bédorf, J. et al.** (2012). *A sparse octree gravitational N-body code*. J. Comput. Phys. 231, 2825.
7. **Peebles, P.J.E.** (1980). *The Large-Scale Structure of the Universe*. Princeton UP.

---

## License

MIT

## Contact

- **Jean-Pierre Petit** (Janus model author): jean-pierre.petit@manaty.net
- **Hicham Zejli** (co-author 2024): hicham.zejli@manaty.net
