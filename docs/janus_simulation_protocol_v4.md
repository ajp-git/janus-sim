# Janus N-body Exploration Protocol (v4)

Testing the emergence of cosmic web structures in a Janus two-species
gravity model, while accounting for real hardware constraints, runtime
benchmarking, and standardized visualization.

------------------------------------------------------------------------

# Physical Model

Two species of particles:

(+) (−)

Interaction rules:

  Interaction   Force
  ------------- ----------------
  \+ +          attraction (G)
  − −           attraction (G)
  \+ −          repulsion (G)

This corresponds to the **Janus gravity hypothesis (Petit)**.

Interaction symmetry is fixed:

G\_{+-} = G

There is **no free coupling parameter** in the strict Janus model.

------------------------------------------------------------------------

# Population Asymmetry

The only physical free parameter is the population ratio:

η = N− / N+

Example value used in simulations:

η ≈ 1.045

This asymmetry can influence polarization growth and segregation
dynamics.

------------------------------------------------------------------------

# Numerical Setup

Simulation box

L = 492 Mpc

Particle count

exploration runs: N ≈ 500k\
production runs: N ≈ 8M -- 18M

Softening length

ε ∈ {0.15, 0.30, 0.40} Mpc

Time step

dt = 0.01

Barnes--Hut opening angle

θ = 0.7

TreePM split scale

r_cut ≈ 30 Mpc

------------------------------------------------------------------------

# Gravity Solver

Two solver configurations are used depending on particle count.

### Exploration runs

TreePM GPU solver

PM grid → long-range gravity\
Barnes--Hut GPU → short-range forces

Typical benchmark:

512k particles → \~0.4--0.5 s / step

### Production runs

Barnes--Hut GPU only

TreePM becomes unstable beyond \~2M particles on RTX 3060 due to GPU
VRAM limitations.

------------------------------------------------------------------------

# Large-Scale Mode Filter

To suppress artificial box-scale instabilities, large Fourier modes are
removed in the PM solver.

Removed modes:

k = 0 monopole\
k = 1 dipole\
k = 2 quadrupole

Implementation:

k_min = 3

Purpose:

Prevent artificial dipole phase separation driven by box-scale modes in
periodic simulations.

------------------------------------------------------------------------

# Cosmological Expansion

Optional weak Hubble-like damping term.

Velocity equation:

dv/dt = g − 2Hv

Test values:

H ∈ {0, 0.01, 0.02}

Effects:

H = 0 → pure Newtonian collapse\
H \> 0 → slows isotropic collapse and promotes anisotropic collapse.

------------------------------------------------------------------------

# Initial Conditions

Zel'dovich displacement field with Gaussian random phases.

Power spectrum:

P(k) ∝ k\^n

Cutoff:

k_cut ≈ 0.25 Mpc⁻¹

Particles start from a perturbed regular grid.

------------------------------------------------------------------------

# Polarization Field

Define polarization field:

P = (ρ+ − ρ−) / (ρ+ + ρ−)

Important diagnostics:

σ(P) → polarization growth\
σ(δ) → total density growth

Both quantities should be monitored during simulations.

------------------------------------------------------------------------

# Dipole Instability Metric

Global segregation is tracked using:

Seg = \|COM+ − COM−\|

Interpretation:

Seg ≈ 0 → well-mixed populations\
Seg large → dipole phase separation

Successful runs should maintain:

Seg ≪ L

------------------------------------------------------------------------

# Cosmic Web Detection

Structures are classified using the **Tidal Tensor (T-web)** method.

Tidal tensor:

T_ij = ∂²Φ / ∂x_i ∂x_j

Eigenvalues:

λ1 ≥ λ2 ≥ λ3

Classification:

0 eigenvalues \> threshold → void\
1 eigenvalue \> threshold → sheet\
2 eigenvalues \> threshold → filament\
3 eigenvalues \> threshold → node

This method is more robust than density-Hessian classification.

------------------------------------------------------------------------

# Filament Formation Diagnostic

Monitor density variance:

σ²(δ)

Nonlinear regime typically begins when:

σ(δ) ≈ 0.5 -- 1

Expected structural sequence:

voids → sheets → filaments → nodes

------------------------------------------------------------------------

# Virialization Diagnostic

Monitor gravitational binding through:

PE_binding / PE_total

This ratio indicates when structures approach stable gravitational
equilibrium.

------------------------------------------------------------------------

# Runtime Benchmark

Each exploration run records:

particle count\
solver type\
number of steps\
time per step\
total runtime

Example benchmark:

512k particles → \~0.47 s / step

TreePM scaling approximately follows:

T ≈ N log N

This allows estimating runtime of large simulations.

------------------------------------------------------------------------

# Full Run Decision Rule

Production simulations are launched only if runtime is reasonable.

Guideline:

estimated runtime \< 5 days → full run allowed\
5--10 days → optional depending on resources\
\> 10 days → exploration runs only

------------------------------------------------------------------------

# Parameter Exploration

Due to hardware limits, exploration uses:

20--40 simulations

Parameters scanned:

ε (softening length)\
H (expansion strength)\
k_min (mode filter threshold)\
η (population ratio)

Large parameter sweeps (\~300 runs) are impractical on current hardware.

------------------------------------------------------------------------

# Early Stopping Criteria

Simulation stops early if:

dipole instability detected\
kinetic energy diverges\
density variance saturates

This prevents wasting compute time on unstable runs.

------------------------------------------------------------------------

# Image Rendering Specification

Each snapshot produces two images:

1.  diagnostic panels
2.  cinematic cosmic web render

Snapshots saved every:

20 simulation steps

------------------------------------------------------------------------

# Diagnostic Panels

Purpose: scientific monitoring.

Layout:

2 × 3 panels

Top row

ρ+ density projection (XY)\
ρ+ density projection (XZ)\
ρ+ density projection (YZ)

Bottom row

ρ− density projection (XY)\
ρ− density projection (XZ)\
ρ− density projection (YZ)

Resolution:

3840 × 2160 (4K)

Density scale:

log10(ρ)

Smoothing:

Gaussian σ ≈ 1--2 pixels

Color mapping:

ρ+ → red/orange\
ρ− → blue/cyan

------------------------------------------------------------------------

# Cosmic Web Rendering

Purpose: visualisation of large-scale structure.

Projection:

integrated density (XY)

Resolution:

3840 × 2160

Dynamic range:

log10(ρ)

Color scheme:

background → black\
filaments → gold / white\
nodes → bright white cores

Visualization inspired by:

Millennium Simulation\
Illustris visualizations

------------------------------------------------------------------------

# Frame Output

Image format:

PNG

Naming convention:

frame_000000.png\
frame_000020.png\
frame_000040.png

Diagnostic panels:

panel_000020.png

Cosmic web render:

web_000020.png

------------------------------------------------------------------------

# Video Assembly

Target resolution:

3840 × 2160 (4K)

Frame rate:

20 fps

Example command:

ffmpeg -framerate 20 -i web\_%06d.png -c:v libx264 -pix_fmt yuv420p
cosmic_web.mp4

------------------------------------------------------------------------

# Simulation Pipeline

1)  exploration runs (\~500k particles)\
2)  measure runtime and structural evolution\
3)  tune parameters\
4)  estimate runtime of large simulations\
5)  production runs (8M--18M particles)\
6)  generate cinematic cosmic web renderings

------------------------------------------------------------------------

# Exploration Strategy

Exploration runs do NOT stop at the first successful configuration.

The exploration pipeline is:

1. **Exploration runs** — systematically sweep parameter space
2. **Results aggregation** — collect all metrics into a single CSV
3. **AI selection** — analyze results to identify promising configurations
4. **Production runs** — run selected configurations at high resolution

This approach ensures:

- No promising configuration is missed due to early stopping
- The full parameter landscape is mapped
- AI-assisted analysis can identify non-obvious patterns
- Reproducible selection criteria for production runs

------------------------------------------------------------------------

# Run Result Recording

Each exploration run records the following metrics:

## Identification

| Field | Description |
|-------|-------------|
| run_id | Unique identifier (timestamp) |
| N | Total particle count |
| epsilon | Softening length (Mpc) |
| k_min | PM mode filter threshold |
| H | Hubble damping parameter |
| eta | Population ratio N-/N+ |

## Runtime

| Field | Description |
|-------|-------------|
| runtime_s | Total runtime (seconds) |
| steps | Number of simulation steps |
| ms_per_step | Average time per step |

## Segregation

| Field | Description |
|-------|-------------|
| seg_final | Final segregation |COM+ - COM-| (Mpc) |
| seg_max | Maximum segregation during run |
| dipole_suppressed | Boolean: seg_final < L/10 |

## Structure Formation

| Field | Description |
|-------|-------------|
| sigma_rho | Density variance σ(δ) |
| sigma_P | Polarization variance σ(P) |
| R_ratio | σ_ρ / σ_P (>1 favors filaments) |

## T-web Classification

| Field | Description |
|-------|-------------|
| void_fraction | Fraction of cells classified as void |
| sheet_fraction | Fraction of cells classified as sheet |
| filament_fraction | Fraction of cells classified as filament |
| node_fraction | Fraction of cells classified as node |

## Energy

| Field | Description |
|-------|-------------|
| KE_final | Final kinetic energy |
| PE_final | Final potential energy (binding) |
| virial_ratio | 2 KE / |PE| |

## Output

All runs are aggregated into:

```
results/exploration_results.csv
```

This CSV can be analyzed programmatically to:

- Rank configurations by filament fraction
- Filter by dipole suppression criterion
- Identify optimal R_ratio values
- Select candidates for production runs

------------------------------------------------------------------------

# Output

Each run produces:

particle snapshots\
density maps\
polarization maps\
cosmic web classification\
power spectrum

------------------------------------------------------------------------

# Scientific Goal

Determine whether **Janus gravity naturally produces
void--filament--node structures** comparable to the observed cosmic web
while avoiding dipole phase separation.
