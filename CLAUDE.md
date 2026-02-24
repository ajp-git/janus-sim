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
docker stop $(docker ps -q)
docker rm -f $(docker ps -aq)
docker system prune
docker network rm ...
docker volume rm ...
```

---

## Environment

**Ubuntu Server 24/7**
- CPU: x86_64 Linux
- RAM: 32GB
- GPU: NVIDIA RTX 3060 12GB VRAM (Ampere, sm_86)
- Interface: Claude CLI on host (not in Docker)
- Results: `./output/` (mounted in containers)

**Stack**
- Docker + nvidia-container-toolkit
- Rust compiled in container (CUDA 12.3)
- CUDA via `cudarc` (f64 precision)

---

## Current Project Status (February 2026)

### Phase 1a — Pantheon+ Fit ✅ COMPLETE
- η = 1.045, q₀ = -0.022, χ²/dof = 0.914 (1590 SNIa)

### Phase 1b/1c — GPU N-body ✅ COMPLETE
- GPU speedup: 39.6× vs CPU
- Production runs: 500K (S=0.513), 2M (S=0.694), 8M (S=0.459, θ=1.5)
- Virialized ICs (PE_binding method, α≈4.57)
- Hubble friction implemented (z=5→z=0)

### Phase 2 — Filament Formation 🔄 IN PROGRESS

#### Théorie linéaire (VALIDÉE)
La matrice de couplage à deux fluides donne :
- λ₊ = ρ̄(1+α) > 0 → ségrégation (blob) ✅ observée
- λ₋ = ρ̄(1-α) = 0 pour α=1 → mode filamentaire gelé

**Avec α=1 (code actuel) : λ₋=0 exactement.**
Validé expérimentalement : α=1 supprime ~75% de la croissance anisotrope
(test Jour 1 : +1047% ΛCDM vs +262% Janus, ratio 4×).

#### Tests Yukawa (TERMINÉS — négatifs)
α(r) = 1 − ε·exp(−r/r_c) implémenté dans le kernel CUDA.
6 configurations testées (ε=0.3/0.7, r_c=10/40 Mpc) :
→ Effet maximal : +0.2% (non significatif physiquement)
→ Cause : régime linéaire mono-mode insensible à α(k)
→ Yukawa vérifié N² vs BH (erreur 0.02%)

#### Run Jour 4 — ICs Zel'dovich ✅ CORRIGÉ
**Bug critique dans les ICs (corrigé) :**
- Ancienne version : `let sign = if idx < n_positive { 1 } else { -1 };`
  → m+ dans première moitié grille (z bas), m- dans seconde (z haut)
  → ségrégation artificielle dès step 0
- **Fix appliqué :** `let sign = if rng.gen::<bool>() { 1 } else { -1 };`
  → signes aléatoires, même déplacement ψ pour tous

**FFT fonctionne correctement (24 fév 2026) :**
- `max(|psi|) = 2.4e-8 Mpc` après IFFT (très petit mais non-nul)
- Scaling factor = 25M× appliqué → displacement final ~0.94 Mpc (30% cell)
- L'affichage "0.0000" était un problème de format (`{:.4}` → `{:.6e}`)
- Simulation tourne : S₀=0.35 → S₁₀₀=0.45 (ségrégation croissante ✅)

**Remarque σ constant :** Normal pour 2M particules - la distribution bulk
ne change pas de forme, seuls les COMs se séparent.

---

## Bugs Fixed (lessons learned)

| Bug | Root Cause | Lesson |
|-----|-----------|--------|
| Wrong acceleration | Local densities instead of conserved E | Verify vs source paper |
| 0.8 mag offset | Mixed ΛCDM H(z) with Janus accelerations | Theoretical consistency first |
| Zero segregation | PM method smooths short-range | Validate method on known case |
| COM periodic error | Simple average ignores wrap | Use minimum image convention |
| GPU rsqrt() | rsqrt() is float intrinsic | Use 1.0/sqrt() for f64 |
| Seg₀ ≈ 0.49 | Different COM references | Common origin for both populations |
| IC Zel'dovich biais | m+/m- dans moitiés opposées grille | Signes aléatoires obligatoires ✅ |
| FFT "displacement=0" | Format {:.4} tronque 2.4e-8 | Utiliser {:.6e} ✅ |

---

## Janus Virialization (CRITICAL)

Standard virialization (2KE + PE_total = 0) **fails** for Janus:
```rust
// PE_binding < 0 always (attractive pairs only)
let alpha = (pe_binding.abs() / (2.0 * ke)).sqrt();
// Typical alpha ≈ 4.57 for η=1.045
```

---

## Résultats CSV disponibles

```
Jour 1 ✅ : output/aniso_test_*/combined.csv
            step, delta_k_A, delta_k_B, sigma_x_A, sigma_x_B
Jour 2 ✅ : output/yukawa_*/combined_6runs.csv
            step, delta_k_A..F, sigma_x_A..F
Jour 3 ✅ : output/zeldovich_*/results.csv
            step, delta_k_B..F, sigma_x_B..F
Jour 4 ✅ : output/jour4_corrected_*/evolution.csv (FFT corrigé)
```

---

## Project Commands

**`--features cuda` EST OBLIGATOIRE pour toutes les simulations N-body.**
Sans ce flag, le code compile en mode CPU (100× plus lent) et ne détecte pas le GPU.

```bash
# Build
docker compose build

# GPU N-body simulation (--features cuda OBLIGATOIRE)
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- \
  --n 2000000 --eta 1.045 --dt 0.005 --steps 3000 \
  --output /app/output/run_name

# Interactive debug shell
docker compose run --rm dev

# Check progress
tail -f output/*/evolution.csv
```

---

## Simulation Parameters

| Parameter | Validated Value |
|-----------|----------------|
| η | 1.045 |
| dt | 0.005 |
| θ (Barnes-Hut) | 0.7 (précis) / 1.2 (rapide) |
| box | 400 Mpc |
| z_init | 10 |
| Snapshot interval | 100 steps |
| CSV interval | 10 steps |

---

## Key References

1. **Petit, Margnat & Zejli (2024)** — EPJC 84:1226
2. **D'Agostini & Petit (2018)** — Astrophys. Space Sci. 363:139
3. **Petit & D'Agostini (2014)** — Astrophys. Space Sci. 354:611
4. **Scolnic et al. (2022)** — ApJ 938:113
5. **Lane et al. (2024)** — MNRAS arXiv:2311.01438
