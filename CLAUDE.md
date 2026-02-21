# Janus Cosmological Model — Simulation Project

## OPERATING MODE

Claude operates in **fully autonomous** mode on this server.
- Write and modify code without asking for confirmation
- Compile, launch simulations, read results
- Download necessary observational data
- Debug and fix errors autonomously
- Never ask "would you like me to..." — just do it

---

## DOCKER RULES — SHARED SERVER

**Other Docker containers are running on this server. Never touch them.**

### ALWAYS
- Use `docker compose` from the `janus-sim/` folder
- Check `docker ps` before any global action
- Use `--rm` for ephemeral containers
- Explicitly name the service: `docker compose run --rm friedmann`

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
- `janus-friedmann`
- `janus-nbody`
- `janus-dev`

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
- CUDA via `cudarc` for GPU N-body kernels

---

## Host Prerequisites (check at startup)

```bash
nvidia-smi                    # NVIDIA drivers OK?
docker --version              # Docker installed?
docker compose version        # Compose available?
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
# GPU visible in Docker?
```

If something is missing:
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Project Commands

```bash
# Build the image (first time ~5-10 min)
docker compose build

# Step 1 — Friedmann FLRW (CPU)
docker compose run --rm friedmann

# Step 2 — N-body GPU
docker compose run --rm nbody -- --n 1000000 --steps 1000

# Interactive shell for debugging
docker compose run --rm dev

# Results
ls output/
```

---

## Scientific Objective

Numerically validate Jean-Pierre Petit's Janus cosmological model
by reproducing **real observables** (not illustrations).

### The Janus Model
Coupled bimetric model (Petit & D'Agostini 2014, EPJC 2024).
Two metrics g+ and g- coexist on the same manifold M4.
Replaces dark matter and dark energy with negative masses.

### Interaction Rules (Newtonian limit)
- mass+ attracts mass+ → classical Newton
- mass- attracts mass- → attraction (symmetry)
- mass+ and mass- repel each other → anti-Newton (runaway eliminated)

### Coupled FLRW Conservation Equation
```
rho*c^2*a^3 + rho_bar*c_bar^2*a_bar^3 = E = constant,  E < 0
```
Negative total energy → dominant negative sector → cosmic acceleration.

### Single Free Parameter
eta = |rho_bar_0|/rho_0  (negative/positive density ratio)
H0 = 70 km/s/Mpc (consistent with Janus, vs 67 for LCDM)

---

## Roadmap

### Phase 1 — Local Validation (current objective)

**1a — Friedmann + Pantheon+ fit** (CPU)
- Integrate coupled Janus FLRW equations
- Scan eta, minimize chi^2 on ~1700 SNIa Pantheon+
- Output: magnitude-redshift curve + optimal eta

**1b — N-body 1M particles** (RTX 3060)
- Verify +/- spatial segregation
- Estimated time: ~1 min/run

**1c — N-body 10M particles** (RTX 3060)
- Measure two-point correlation function
- Compare with SDSS DR16
- Estimated time: ~10 min/run

**1d — N-body 100M particles** (RTX 3060)
- Power spectrum vs observations
- Estimated time: ~2h/run

### Phase 2 — Submission to Petit/Zejli
- Public GitHub + technical report
- Friedmann curves + N-body maps + statistics
- Contact: jean-pierre.petit@manaty.net / hicham.zejli@manaty.net

### Phase 3 — Distributed Computing (if validation OK)
- Cross-platform Rust client + network layer
- BOINC-like server (Axum/Rust)
- Public dashboard for aggregated results
- This server becomes the central node

---

## Key References

1. Petit & D'Agostini (2014) — Astrophys. Space Sci. 354:611
2. D'Agostini & Petit (2018) — Astrophys. Space Sci. 363:139 — fit 740 SNIa, 1 free parameter
3. Petit, Margnat & Zejli (2024) — EPJC 84:1226 — current reference
4. Zejli (2024) — technical book 233p — januscosmologicalmodel.com
5. Petit (1995) — Astrophys. Space Sci. 226:273 — original simulations

## Observational Data

- SNIa Pantheon+: https://pantheonplussh0es.github.io/
- Large-scale structure SDSS: https://www.sdss.org/dr16/
- Rotation curves SPARC: http://astroweb.cwru.edu/SPARC/

---

## Project Structure

```
janus-sim/
├── CLAUDE.md              ← read first
├── Cargo.toml
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── lib.rs             ← constants + Janus interaction rules
│   ├── friedmann.rs       ← coupled FLRW + RK4
│   ├── nbody.rs           ← N-body
│   ├── analysis.rs        ← chi^2 fitting
│   └── bin/
│       ├── friedmann.rs   ← magnitude-redshift binary
│       └── nbody.rs       ← GPU N-body binary
├── data/                  ← obs. data (downloaded automatically)
├── output/                ← simulation results
└── tests/
```

## Technical Notes

- 1 free parameter model → 1D scan on eta for SNIa
- Opposite masses mutually exclude in dense regions
- Symplectic integrator (Leapfrog/Yoshida 4) mandatory
- Barnes-Hut O(N log N) mandatory beyond 100K particles
