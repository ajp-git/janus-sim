# Janus Cosmological Model — N-body Simulation

Numerical validation of Jean-Pierre Petit's Janus bimetric cosmological model.

## Description

This project implements a GPU-accelerated N-body simulation to validate the predictions of the Janus model, which replaces dark matter and dark energy with negative masses interacting according to specific interaction rules.

### Janus Interaction Rules (Newtonian Limit)
- **Positive mass attracts positive mass**: classical Newtonian gravity
- **Negative mass attracts negative mass**: symmetry (attraction)
- **Positive and negative masses repel each other**: anti-gravity (eliminates runaway)

### Single Free Parameter
- **η = |ρ̄₀|/ρ₀**: ratio of negative to positive density
- H₀ = 70 km/s/Mpc (consistent with Janus)

## Phase 1a Results — Pantheon+ Fit

Fit on 1701 Type Ia supernovae from the Pantheon+ catalog (Scolnic et al. 2022):

| Parameter | Value |
|-----------|-------|
| Optimal η | **1.045** |
| χ²/dof | **0.914** |
| q₀ (deceleration) | -0.022 |

The Janus model reproduces SNIa observations with **only 1 free parameter** (vs 2 for ΛCDM: Ωm, ΩΛ).

## References

1. **Petit, J.-P., Margnat, S. & Zejli, H.** (2024). *The Janus Cosmological Model*. Eur. Phys. J. C 84, 1226. [DOI:10.1140/epjc/s10052-024-13589-8](https://doi.org/10.1140/epjc/s10052-024-13589-8)

2. **D'Agostini, G. & Petit, J.-P.** (2018). *Constraints on Janus Cosmological model from recent observations of supernovae type Ia*. Astrophys. Space Sci. 363, 139.

3. **Petit, J.-P. & D'Agostini, G.** (2014). *Negative mass hypothesis in cosmology and the nature of dark energy*. Astrophys. Space Sci. 354, 611.

4. **Zejli, H.** (2024). *The Janus Cosmological Model — Technical Book*. 233 pages. [januscosmologicalmodel.com](https://januscosmologicalmodel.com)

## Hardware Requirements

- **GPU**: NVIDIA with CUDA (RTX 3060 12GB or higher recommended)
- **RAM**: 32 GB minimum
- **Storage**: SSD recommended for snapshots

## Installation

```bash
# Clone the repository
git clone https://github.com/ajp-git/janus-sim.git
cd janus-sim

# Verify CUDA
nvidia-smi

# Build with Docker
docker compose build
```

## Usage

### Phase 1a — Friedmann + Pantheon+ Fit

```bash
docker compose run --rm friedmann
```

### Phase 1b/c — GPU N-body Simulation

```bash
# Quick test (100K particles, 100 steps)
docker compose run --rm dev cargo run --release --features cuda --bin nbody_overnight -- \
  --n 100000 --eta 1.045 --dt 0.01 --steps 100 --output /app/output/test

# Production run (500K particles, 10000 steps)
docker compose run --rm dev cargo run --release --features cuda --bin nbody_overnight -- \
  --n 500000 --eta 1.045 --dt 0.01 --steps 10000 --output /app/output/run
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n` | Total number of particles | 100000 |
| `--eta` | Density ratio ρ̄/ρ | 1.045 |
| `--dt` | Time step | 0.01 |
| `--steps` | Number of steps | 1000 |
| `--output` | Output directory | output/ |

## Project Structure

```
janus-sim/
├── src/
│   ├── lib.rs           # Constants and Janus interaction rules
│   ├── friedmann.rs     # Coupled FLRW integration
│   ├── nbody.rs         # CPU N-body (Barnes-Hut)
│   ├── nbody_gpu.rs     # GPU N-body (CUDA)
│   └── analysis.rs      # χ² fitting
├── scripts/             # Python visualization
├── data/                # Pantheon+ data (not included)
└── output/              # Results (not included)
```

## Validation

The code follows strict validation rules defined in `VALIDATION_RULES.md`:

- Mandatory trivial test for each physics function
- Energy conservation (KE/KE₀ < 50)
- Expected increasing segregation
- Virialized initial conditions (2KE + PE_bind = 0)

## Auto-Stop Conditions

Simulations automatically stop if:
- **KE/KE₀ > 50**: Energy instability detected
- **Segregation decreases for 500 consecutive steps**: Unphysical behavior

## Output

Each simulation produces:
- `snapshots/`: Binary particle data at each step
- `frames/`: 4K PNG visualization frames
- `time_series.csv`: Step-by-step metrics
- `summary.json`: Final results

### Video Assembly

```bash
ffmpeg -framerate 24 -i output/run/frames/frame_%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p output/run/janus_simulation.mp4
```

## License

MIT

## Contact

- Jean-Pierre Petit: jean-pierre.petit@manaty.net
- Hicham Zejli: hicham.zejli@manaty.net
