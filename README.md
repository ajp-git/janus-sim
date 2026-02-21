# Janus Cosmological Model — GPU N-body Simulation

Numerical validation of Jean-Pierre Petit's Janus bimetric cosmological model.  
Independent research project — February 2026.

---

## Overview

This project provides a full numerical validation pipeline for the Janus model:

1. **Phase 1a** — Fit of 1590 Type Ia supernovae (Pantheon+ catalog) using the exact analytical formula from D'Agostini & Petit (2018)
2. **Phase 1b** — CPU Barnes-Hut N-body simulation (1M particles, validated)
3. **Phase 1c** — GPU CUDA Barnes-Hut N-body simulation with virialized initial conditions

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

From Petit & D'Agostini (2014), the acceleration equations are:

```
ä  = −1.5 · E / a²     (positive sector)
ā̈  = +1.5 · E / ā²     (negative sector)
```

where E = Ω₊ − Ω₋ is conserved (computed once at t=0).  
When E < 0 (η > 1): positive sector accelerates, negative sector decelerates.

---

## Phase 1a — Pantheon+ Fit Results

**Dataset**: Pantheon+ SH0ES 2022 — 1590 Type Ia supernovae with full covariance matrix  
**Method**: Exact analytical formula (D'Agostini & Petit 2018, eq.5):

```
μ = 5·log₁₀[z + z²·(1−q₀) / (1 + q₀·z + √(1 + 2·q₀·z))] + cst
```

**Results**:

| Parameter | Value | Note |
|-----------|-------|------|
| η (optimal) | **1.045** | Single free parameter |
| q₀ | **−0.022 ± 0.015** | Near-flat acceleration |
| χ²/dof | **0.914** | Excellent fit |
| Dataset | Pantheon+ (1590 SNIa) | vs JLA (740 SNIa) in 2018 paper |

**Comparison with D'Agostini & Petit (2018)**:

| Paper | Dataset | N SNIa | q₀ | η |
|-------|---------|--------|----|---|
| D'Agostini 2018 | JLA | 740 | −0.087 | ~1.19 |
| **This work** | **Pantheon+** | **1590** | **−0.022** | **1.045** |

**4.4σ tension explained**: Lane et al. (2024) documented a ΛCDM calibration bias in the SALT2 parameters of Pantheon+. Since SALT2 standardization is performed assuming ΛCDM, any alternative model using these corrected magnitudes inherits a systematic offset. Our value η=1.045 is therefore a conservative lower bound.

---

## Phase 1b/1c — N-body Simulation

### Algorithm

- **Barnes-Hut** O(N log N) tree algorithm (Bédorf 2012 GPU implementation)
- **Leapfrog integrator** (kick-drift-kick, symplectic)
- **Plummer softening**: 1/(r² + ε²)^(3/2) for energy conservation
- **Periodic boundary conditions** with minimum image convention

### GPU Implementation

| Feature | Detail |
|---------|--------|
| Language | Rust + CUDA (cudarc crate) |
| Precision | f64 throughout |
| CPU/GPU validation | 0% difference (synchronized seeds) |
| Tree structure | Linear octree (Bédorf-style) |
| Hardware | NVIDIA RTX 3060 12GB |

### Bugs Identified and Fixed

During development, several critical bugs were identified and corrected:

| Bug | Root Cause | Impact |
|-----|-----------|--------|
| Wrong acceleration equations | Used local densities instead of conserved E | Incorrect Janus dynamics |
| Analytical/numerical inconsistency | Mixed standard Friedmann H(z) with Janus accelerations | 0.4–0.8 mag systematic offset |
| Particle-Mesh method failure | PM smooths short-range interactions | Zero segregation observed |
| COM with periodic BC | Simple average ignores particle wrapping | Segregation metric invalid |
| GPU rsqrt() in f64 | rsqrt() is float intrinsic — implicit promotion | Precision loss, CPU/GPU divergence |
| Artificial initial segregation | Different reference particles for COM+ and COM− | Seg₀ ≈ 0.49 (150× overestimate) |
| Virialization with total PE | Total PE > 0 for mixed Janus system | 2KE + PE = 0 impossible |

### Key Insight: Janus Virialization

Standard virialization (2KE + PE_total = 0) fails for Janus systems because:
- With η ≈ 1, repulsive +/− pairs dominate → PE_total > 0
- KE_target = −PE_total/2 < 0 → impossible

**Solution**: virialize using PE_binding (same-sign pairs only):
```
α = √(|PE_binding| / (2·KE))
```
PE_binding < 0 always (attractive interactions only), giving α ≈ 4.57 for our parameters.

### Validation Results

**Test: 500K particles, η=1.045, IC virialized**

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Seg₀ | 0.49 (artificial) | 0.0024 ✅ |
| KE/KE₀ at step 200 | ~50 | 1.0012 ✅ |
| Segregation trend | −1.85% | +265% ✅ |
| Virial error | N/A | 0.0000% ✅ |

### Current Runs (2026-02-21)

Three simultaneous runs with virialized IC and corrected COM metric:

| Run | N | Steps | Est. duration | Status |
|-----|---|-------|--------------|--------|
| run_lo | 100K | 10 000 | ~5h | Running |
| run_mid | 500K | 10 000 | ~5h | Running |
| run_hi | 2M | 10 000 | ~22h | Running |

---

## Validation Framework

All physics functions are tested before use. See `VALIDATION_RULES.md` for complete rules.

### Mandatory Tests

```rust
// Segregation metric — trivial case
// 4 particles+ at (10,0,0), 4 particles− at (−10,0,0) → distance = 20.0 exactly

// Periodic BC — minimum image convention
// Particles at x=+49 and x=−49, box=100 → distance = 2, not 98

// Janus forces — sign verification
// Mass+ at origin, Mass− at (1,0,0) → repulsion (force toward −x)

// Virialization — Janus mode
// PE_binding < 0, virial error < 1% after rescaling

// CosmoInterpolator — synchronization check
// a(tau_start) = 1/(1+z_init), a(tau_end) = 1.0, H_end = √Ω₊
```

### Auto-Stop Conditions

Simulations stop automatically if:
- **KE/KE₀ > 50**: energy instability
- **Segregation decreases for 500 consecutive steps**: unphysical behavior

---

## Roadmap

### Completed ✅
- Phase 1a: Pantheon+ fit (publishable)
- Barnes-Hut CPU/GPU with 0% validation error
- Bug fixes: rsqrt, COM periodic, acceleration equations
- Virialized initial conditions (PE_binding method)
- Corrected COM reference (common origin for both populations)

### In Progress 🔄
- Convergence study: 100K / 500K / 2M (current overnight runs)
- Understanding Run A behavior (segregation peak then decline)

### Planned
- **Tâche 2**: Hubble friction — couple expansion factor a(t) from friedmann.rs into leapfrog kernel
- **Tâche 3**: Full convergence study 100K → 2M, criterion < 10% between N and 2N
- **Tâche 4**: Two-point correlation function ξ(r) via Corrfunc, qualitative comparison with SDSS DR7
- **Tâche 5**: Test at η=1.0 (theoretical limit) to characterize the quasi-symmetric regime

---

## Project Structure

```
janus-sim/
├── src/
│   ├── lib.rs              # Constants and Janus interaction rules
│   ├── friedmann.rs        # Coupled FLRW integration (RK4) + CosmoInterpolator
│   ├── nbody.rs            # CPU N-body (Barnes-Hut, Rayon parallel)
│   ├── nbody_gpu.rs        # GPU N-body (CUDA, f64, virialization)
│   ├── analysis.rs         # χ² fitting on Pantheon+ data
│   └── bin/
│       ├── friedmann.rs    # Friedmann solver + SNIa fit binary
│       ├── nbody.rs        # CPU N-body binary
│       └── nbody_overnight.rs  # GPU production binary
├── data/
│   └── Pantheon+SH0ES.dat  # SNIa data (not included, see Scolnic 2022)
├── output/                 # Results (not tracked by git)
│   └── YYYY-MM-DD_run_*/
│       ├── snapshots/      # HDF5 particle data
│       ├── frames/         # 4K PNG visualization
│       ├── time_series.csv # Step metrics
│       └── summary.json    # Final results
├── VALIDATION_RULES.md     # Mandatory test rules for all physics functions
├── janus_roadmap.md        # Detailed roadmap with code (Tâches 1–5)
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

# Verify CUDA
nvidia-smi

# Build
docker compose build
```

---

## Usage

### Phase 1a — Friedmann + Pantheon+ Fit

```bash
docker compose run --rm dev cargo run --release --bin friedmann
```

### Phase 1c — GPU N-body Simulation

```bash
# Quick validation test (100K particles, 200 steps)
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- \
  --n 100000 --eta 1.045 --dt 0.01 --steps 200 \
  --output /app/output/test

# Production run (500K particles, 10000 steps)
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- \
  --n 500000 --eta 1.045 --dt 0.01 --steps 10000 \
  --output /app/output/2026-02-21_run_mid
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n` | Total number of particles | 100000 |
| `--eta` | Density ratio ρ̄/ρ | 1.045 |
| `--dt` | Time step (dimensionless) | 0.01 |
| `--steps` | Number of integration steps | 1000 |
| `--output` | Output directory | output/ |

### Video Assembly

```bash
ffmpeg -framerate 24 -i output/run/frames/frame_%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  output/run/janus_simulation.mp4
```

---

## Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA CUDA | RTX 3060 12GB |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB SSD | 1 TB NVMe |

**Performance** (RTX 3060, f64):

| N | Time/step | 10K steps |
|---|-----------|-----------|
| 100K | ~1.9s | ~5h |
| 500K | ~1.9s | ~5h |
| 2M | ~7.8s | ~22h |

---

## References

1. **Petit, J.-P., Margnat, S. & Zejli, H.** (2024). *The Janus Cosmological Model*. Eur. Phys. J. C 84, 1226. [DOI:10.1140/epjc/s10052-024-13589-8](https://doi.org/10.1140/epjc/s10052-024-13589-8)

2. **D'Agostini, G. & Petit, J.-P.** (2018). *Constraints on Janus Cosmological model from recent observations of supernovae type Ia*. Astrophys. Space Sci. 363, 139.

3. **Petit, J.-P. & D'Agostini, G.** (2014). *Negative mass hypothesis in cosmology and the nature of dark energy*. Astrophys. Space Sci. 354, 611.

4. **Scolnic, D. et al.** (2022). *The Pantheon+ Analysis: The Full Data Set and Light-Curve Release*. ApJ 938, 113.

5. **Lane, Z.G. et al.** (2024). *ΛCDM calibration bias in Pantheon+*. MNRAS. arXiv:2311.01438.

6. **Bédorf, J. et al.** (2012). *A sparse octree gravitational N-body code*. J. Comput. Phys. 231, 2825.

---

## License

MIT

---

## Contact

- **Jean-Pierre Petit** (Janus model author): jean-pierre.petit@manaty.net
- **Hicham Zejli** (co-author 2024 paper): hicham.zejli@manaty.net
