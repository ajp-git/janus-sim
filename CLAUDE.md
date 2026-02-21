# Janus Cosmological Model — Simulation Project

## OPERATING MODE

Claude operates in **fully autonomous** mode on this server.
- Write and modify code without asking for confirmation
- Compile, launch simulations, read results
- Download necessary observational data
- Debug and fix errors autonomously
- Never ask "would you like me to..." — just do it

---

## CRITICAL: READ FIRST

1. **Read `VALIDATION_RULES.md`** at the start of every session
2. **Every new physics function** must have a trivial test before use
3. **Never launch simulations** without explicit user instruction
4. **Report results** and wait for instruction after each task

---

## DOCKER RULES — SHARED SERVER

**Other Docker containers are running on this server. Never touch them.**

### ALWAYS
- Use `docker compose` from the `janus-sim/` folder
- Check `docker ps` before any global action
- Use `--rm` for ephemeral containers
- Explicitly name the service: `docker compose run --rm dev`

### NEVER
```bash
# These commands are FORBIDDEN — they affect the entire server
docker stop $(docker ps -q)
docker rm -f $(docker ps -aq)
docker system prune
docker network rm ...
docker volume rm ...
```

### Containers of this project (the only ones to manipulate)
- `janus-sim-dev` (main development container)

---

## Environment

**Ubuntu Server 24/7**
- CPU: x86_64 Linux
- RAM: 32GB
- GPU: NVIDIA RTX 3060 12GB VRAM (Ampere, sm_86)
- Interface: Claude CLI on host (not in Docker)
- Results: `./output/` (mounted in containers)

**Stack**
- Docker + nvidia-container-toolkit (GPU → containers)
- Rust compiled in container (CUDA 12.3 image)
- CUDA via `cudarc` for GPU N-body kernels (f64 precision)

---

## Current Project Status (February 2026)

### Phase 1a — Pantheon+ Fit ✅ COMPLETE
- **η = 1.045** (single free parameter)
- **q₀ = -0.022** (near-flat acceleration)
- **χ²/dof = 0.914** (excellent fit on 1590 SNIa)

### Phase 1b — CPU Barnes-Hut ✅ COMPLETE
- 0% CPU/GPU deviation (validated with synchronized seeds)

### Phase 1c — GPU N-body 🔄 IN PROGRESS
- Virialized initial conditions implemented
- Auto-stop conditions active
- Current runs: 100K, 500K, 2M particles (10,000 steps each)

---

## Key Technical Insights

### Janus Virialization (CRITICAL)
Standard virialization (2KE + PE_total = 0) **fails** for Janus systems:
- With η ≈ 1, repulsive +/− pairs dominate → PE_total > 0
- KE_target = −PE_total/2 < 0 → impossible

**Solution**: Use PE_binding (same-sign pairs only):
```rust
// PE_binding < 0 always (attractive pairs only)
let alpha = (pe_binding.abs() / (2.0 * ke)).sqrt();
// Typical alpha ≈ 4.57 for η=1.045
```

### Bugs Fixed (lessons learned)
| Bug | Root Cause | Lesson |
|-----|-----------|--------|
| Wrong acceleration | Local densities instead of conserved E | Verify vs source paper |
| 0.8 mag offset | Mixed Friedmann H(z) with Janus accelerations | Theoretical consistency first |
| Zero segregation | PM method smooths short-range | Validate method on known case |
| COM periodic error | Simple average ignores wrap | Use minimum image convention |
| GPU rsqrt() | rsqrt() is float intrinsic | Use 1.0/sqrt() for f64 |
| Seg₀ ≈ 0.49 | Different COM references | Common origin for both populations |

---

## Project Commands

```bash
# Build the image
docker compose build

# Friedmann + Pantheon+ fit (CPU)
docker compose run --rm dev cargo run --release --bin friedmann

# GPU N-body simulation
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- \
  --n 500000 --eta 1.045 --dt 0.01 --steps 10000 \
  --output /app/output/run_name

# Interactive shell for debugging
docker compose run --rm dev

# Check simulation progress
tail -20 output/*/run.log
ls output/*/snapshots/ | wc -l
```

---

## Simulation Parameters

| Parameter | Description | Validated Value |
|-----------|-------------|-----------------|
| `--n` | Total particles | 100K-2M |
| `--eta` | ρ̄/ρ ratio | 1.045 |
| `--dt` | Time step | 0.01 |
| `--steps` | Integration steps | 10000 |
| `--output` | Output directory | /app/output/... |

### Auto-Stop Conditions
- **KE/KE₀ > 50**: Energy instability → stop
- **Seg decreases 500 consecutive steps**: Unphysical → stop

---

## Project Structure

```
janus-sim/
├── CLAUDE.md              ← read first (this file)
├── VALIDATION_RULES.md    ← mandatory tests for physics functions
├── janus_roadmap.md       ← detailed task roadmap (Tasks 1-5)
├── README.md              ← full project documentation
├── Cargo.toml
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── lib.rs             ← constants + Janus interaction rules
│   ├── friedmann.rs       ← coupled FLRW + RK4 + CosmoInterpolator
│   ├── nbody.rs           ← CPU N-body (Barnes-Hut, Rayon parallel)
│   ├── nbody_gpu.rs       ← GPU N-body (CUDA f64, virialization)
│   ├── analysis.rs        ← χ² fitting on Pantheon+
│   └── bin/
│       ├── friedmann.rs       ← Friedmann solver + SNIa fit
│       ├── nbody.rs           ← CPU N-body binary
│       └── nbody_overnight.rs ← GPU production binary
├── data/
│   └── Pantheon+SH0ES.dat ← SNIa data (download from Scolnic 2022)
└── output/                ← simulation results (not in git)
    └── YYYY-MM-DD_run_*/
        ├── snapshots/     ← particle data per step
        ├── frames/        ← 4K PNG visualization
        ├── time_series.csv
        └── run.log
```

---

## Roadmap

### Completed ✅
- [x] Phase 1a: Pantheon+ fit (η=1.045, χ²/dof=0.914)
- [x] Barnes-Hut CPU/GPU with 0% validation error
- [x] Bug fixes: rsqrt, COM periodic, acceleration equations
- [x] Virialized IC (PE_binding method, virial error 0.0000%)
- [x] Corrected COM reference (common origin)
- [x] Auto-stop conditions

### In Progress 🔄
- [ ] Convergence study: 100K / 500K / 2M (overnight runs)
- [ ] Understanding segregation dynamics at η=1.045

### Planned (see janus_roadmap.md)
- [ ] Task 2: Hubble friction (couple a(t) from friedmann.rs)
- [ ] Task 3: Full convergence study (criterion < 10%)
- [ ] Task 4: Two-point correlation ξ(r) via Corrfunc
- [ ] Task 5: Test η=1.0 (theoretical limit case)

---

## Key References

1. **Petit, Margnat & Zejli (2024)** — EPJC 84:1226 — current reference
2. **D'Agostini & Petit (2018)** — Astrophys. Space Sci. 363:139 — exact μ(z) formula
3. **Petit & D'Agostini (2014)** — Astrophys. Space Sci. 354:611 — Friedmann equations
4. **Scolnic et al. (2022)** — ApJ 938:113 — Pantheon+ data
5. **Lane et al. (2024)** — MNRAS arXiv:2311.01438 — ΛCDM calibration bias

---

## Observational Data

- SNIa Pantheon+: https://pantheonplussh0es.github.io/
- Large-scale structure SDSS: https://www.sdss.org/dr16/
- Rotation curves SPARC: http://astroweb.cwru.edu/SPARC/

---

## Technical Notes

- **f64 everywhere**: GPU kernels use double precision
- **Barnes-Hut θ = 0.5**: Balance speed/accuracy for N-body
- **Leapfrog integrator**: Symplectic, time-reversible
- **Plummer softening**: ε = 0.1 (prevents close encounters)
- **Periodic BC**: Minimum image convention for all distances
