
# Janus Cosmological Simulation – 50‑Run Exploration Plan (Latin Hypercube)

## Objective
Systematically explore the parameter space of the Janus cosmological model using an efficient **Latin Hypercube Sampling (LHS)** strategy.  
This avoids the inefficiency of full factorial grids while still covering the parameter space evenly.

Total runs: **50**

---

# Fixed Simulation Parameters

These remain identical for all runs.

```
N_particles      = 500000
steps            = 5000
theta_BH         = 0.7
dt               = 0.01
z_init           = 5
sign_assignment  = random
IC_spectrum      = cosmological_approx
snapshot_interval = 100
```

Approximate cosmological spectrum used for IC:

```
P(k) ~ k^0.96 / (1 + (k / 0.02)^4)
```

---

# Parameters Explored

| Parameter | Meaning | Range |
|-----------|--------|-------|
| ε | gravitational softening | 0.15 – 0.35 Mpc |
| k_min | suppression of large modes | 2.0 – 3.0 |
| η | negative / positive mass ratio | 1.00 – 1.10 |
| H | expansion coefficient | 0.00 – 0.02 |
| α_IC | asymmetry of fluctuations | 1.0 – 2.0 |

Where:

```
δ_minus = α_IC × δ_plus
```

---

# Latin Hypercube Strategy

Each parameter range is divided into **50 equal intervals**.  
For each run:

1. One value is drawn from each interval.
2. The order is shuffled independently for each parameter.
3. Each run uses a unique combination.

This guarantees:

* uniform coverage
* minimal parameter correlation
* efficient exploration with limited runs

---

# Example of Generated Runs

| Run | ε | k_min | η | H | α_IC |
|----|----|----|----|----|----|
1 | 0.17 | 2.10 | 1.02 | 0.005 | 1.10 |
2 | 0.32 | 2.82 | 1.06 | 0.012 | 1.70 |
3 | 0.21 | 2.34 | 1.04 | 0.018 | 1.50 |
4 | 0.29 | 2.63 | 1.08 | 0.007 | 1.30 |
5 | 0.18 | 2.91 | 1.01 | 0.014 | 1.90 |
6 | 0.24 | 2.22 | 1.05 | 0.010 | 1.40 |
7 | 0.33 | 2.47 | 1.09 | 0.003 | 1.20 |
8 | 0.20 | 2.76 | 1.03 | 0.017 | 1.60 |
9 | 0.31 | 2.04 | 1.07 | 0.015 | 1.80 |
10 | 0.27 | 2.55 | 1.00 | 0.009 | 1.20 |

Continue generation until **50 runs**.

---

# Recommended Output Structure

```
output/
   lhs_run_01/
   lhs_run_02/
   ...
   lhs_run_50/
```

Each folder contains:

```
snapshots/
time_series.csv
analysis/
frames/
```

---

# Metrics to Record

For each run compute:

```
Seg_final
sigma_rho
sigma_P
R = sigma_rho / sigma_P
filament_fraction
filament_width
P(k)_slope
xi(r)_slope
anisotropy
k_peak
runtime
```

---

# Automatic Scoring

Suggested evaluation score:

```
score =
 + filament_fraction
 - |P(k)_slope + 2.5|
 - |xi_slope + 1.8|
 - |filament_width - 2|
 - Seg_final
```

Higher score indicates better agreement with observed cosmic web properties.

---

# Scientific Goals

This exploration aims to determine:

1. Whether **η controls filament structure**.
2. Whether **k_min sets the void scale**.
3. Whether **α_IC determines matter segregation**.
4. Whether **H stabilizes or suppresses polarization**.
5. Whether the Janus model can reproduce:

```
filament width ≈ 1–3 Mpc
P(k) slope ≈ −2 to −3
xi(r) slope ≈ −1.5 to −2
```

---

# Expected Outcomes

The study should identify:

* regions of parameter space producing a realistic cosmic web
* sensitivity of structure formation to Janus parameters
* potential observable predictions of the model

---

End of document.
