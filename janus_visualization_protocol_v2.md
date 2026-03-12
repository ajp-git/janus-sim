
# Janus Simulation Visualization & Diagnostics
## Updated Protocol (including cross-correlation diagnostic)

This document defines visualization and diagnostic outputs for Janus simulations.
It extends the previous pipeline with an additional **cross-correlation diagnostic**
used to detect genuine Janus segregation.

Runs context:

```
Particles: 500k (exploration) → 12M (production)
Steps: 10000
Box: 492 Mpc
```

---
# PART 1 — Diagnostic Images

Each run produces:

```
run_XX_web.png
run_XX_polarization.png
run_XX_slice.png
run_XX_crosscorr.png
run_XX_wow.png
```

Total for 50 runs:

```
50 × 5 = 250 images
```

---
# 1 Cosmic Web Density

Projection of total density:

```
ρ_total = ρ+ + ρ-
```

Processing:

- Gaussian smoothing σ ≈ 1–2 pixels
- log10 projection
- XY projection

Purpose:

Detect

- voids
- sheets
- filaments
- nodes

---
# 2 Polarization Map

Definition:

```
P = (ρ+ − ρ−) / (ρ+ + ρ−)
```

Color coding:

```
blue  → negative mass dominance
white → neutral interface
red   → positive mass dominance
```

Purpose:

Detect

- Janus interfaces
- species segregation
- filament polarity

---
# 3 Diagnostic Slice

Thin slice through the box.

Example:

```
XZ slice at Y ≈ 0
thickness ≈ 5 Mpc
```

Purpose:

Detect

- dipole
- slab instabilities
- grid artefacts

---
# 4 Cross‑Correlation Map (NEW)

Definition:

```
C(x) = ρ+(x) × ρ−(x)
```

Interpretation:

| Region | Meaning |
|------|------|
| high C | both species overlap |
| low C | segregation region |
| ring structures | Janus interfaces |

If the Janus mechanism operates correctly:

- nodes become **anti‑correlated**
- filaments become **interfaces**
- voids remain neutral

This diagnostic is often the clearest indicator that the **Janus repulsion is dynamically active**.

---
# Python Example

```
rho_plus  = grid_plus
rho_minus = grid_minus

cross = rho_plus * rho_minus

proj = cross.sum(axis=2)

proj = np.log10(proj + 1)
```

---
# PART 2 — "Wow" Rendering

Illustris‑style rendering used for interpretation and presentation.

Steps:

```
projection
log compression
contrast stretch
gamma correction
```

Example:

```
proj = grid.sum(axis=2)
proj = gaussian_filter(proj,1)
proj = np.log10(proj + 1)

proj -= proj.min()
proj /= proj.max()

proj = proj**0.6
```

Colormap:

```
inferno
```

---
# PART 3 — The Checkerboard (Damier) Pattern

Some Janus simulations display a **checkerboard-like pattern**:

- alternating positive‑mass and negative‑mass domains
- quasi‑periodic spacing
- domain scale ≈ 100–120 Mpc

This pattern can arise from two different mechanisms.

---
## 1 Numerical origin (box mode)

If the dominant wavelength corresponds to:

```
λ ≈ L / n
```

then the pattern may simply be the **fundamental box mode** selected by the simulation.

Indicators:

- strong peak in P(k)
- periodic tiling of structures
- nodes positioned on a grid

This effect is common in small cosmological boxes.

---
## 2 Physical Janus instability

However, Janus gravity can also produce a **domain instability**:

positive and negative masses repel each other, which can produce:

```
+ domain | interface | − domain
```

If this instability operates, the system may evolve toward:

```
alternating large-scale domains
```

similar to:

- spinodal decomposition
- phase separation
- plasma two‑species instabilities

In that case the domain size becomes a **physical scale** of the model.

---
# How to distinguish the two cases

Test 1 — box scaling

Run the simulation with a different box size.

If the pattern wavelength scales with box size:

```
λ ∝ L
```

→ numerical artifact.

If λ remains constant:

```
λ ≈ constant
```

→ physical instability.

---
Test 2 — power spectrum

If the peak appears at

```
k ≈ k_min
```

→ box mode.

If the peak appears at a fixed physical scale

```
k ≈ constant
```

→ physical scale of Janus dynamics.

---
Test 3 — cross‑correlation

The cross‑correlation map should show:

- interfaces along filaments
- anti‑correlated nodes

If the checkerboard pattern survives in the cross‑correlation map,
it is likely a **true dynamical segregation pattern**.

---
# PART 4 — Key metrics

Track during runs:

```
R = σρ / σP
filament_fraction
Seg = |COM+ − COM−|
```

Expected Janus behaviour:

```
R: 38 → 3–5
filament_fraction: >30%
Seg: <0.3 Mpc
```

---
# End of document
