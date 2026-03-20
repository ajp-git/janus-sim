# Janus Cosmology — Research Plan (V13)

This document defines the next simulation campaign and analysis pipeline for the Janus bimetric cosmology project.

The goal is to test whether the Janus model naturally produces the observed cosmic web structure.

---

# 1. Scientific Objectives

Key questions:

1. Does the Janus model produce a cosmic web?
2. Does segregation reach a universal equilibrium?
3. What are the characteristic scales of the structures?
4. Are these scales compatible with observations?

Key observables:

σ_P(z) — polarization dispersion  
ξ(z) — domain correlation length  
L_J(z) — interface thickness  

Power spectrum:

P(k)

Two-point correlation:

ξ(r)

---

# 2. Simulation Runs

## Run 1 — Production Run

Simulation name:

janus_v13_cosmic_web

Purpose:

Test formation of cosmic web with realistic parameters.

Parameters:

Particles ≈ 5,000,000  
Box size L = 200 Mpc  

Initial redshift z_init = 10  
Final redshift z = 0  

Integrator:

step_with_expansion_dkd_gpu()

Physics:

η = 1.045  
α_IC = 1.6  
ε = 0.18 Mpc  

TreePM:

θ = 0.7  
R_cut = 18 Mpc  

dt = 0.01

Steps:

5000

Outputs:

Snapshots every 100 steps  
Logs every 10 steps

---

## Run 2 — Resolution Convergence

Simulation name:

janus_v13_convergence

Purpose:

Check numerical convergence.

Parameters:

Particles ≈ 2,000,000  
Box size L = 200 Mpc  

Same cosmological parameters as Run 1.

Steps:

5000

Outputs:

Snapshots every 100 steps  
Logs every 10 steps

Compare with Run 1:

σ_P  
ξ  
L_J  
P(k)

---

## Run 3 — Large Box Test

Simulation name:

janus_v13_large_box

Purpose:

Test emergence of large-scale structures.

Parameters:

Particles ≈ 5,000,000  
Box size L = 400 Mpc  

Same cosmological parameters.

Steps:

5000

Outputs:

Snapshots every 100 steps  
Logs every 10 steps

---

# 3. Analysis Pipeline

Perform analysis on the following snapshots:

z ≈ 5  
z ≈ 3  
z ≈ 2  
z ≈ 1  
z ≈ 0

---

# 4. Polarization Statistics

Compute:

Polarization field:

P(x) = (ρ⁺ − ρ⁻)/(ρ⁺ + ρ⁻)

Metrics:

σ_P = std(P)

Segregation fraction:

|P| > 0.5

Plot:

σ_P(z)

---

# 5. Domain Statistics

Segment polarized domains.

Compute:

Median domain diameter  
Domain count  
Domain volume distribution

Characteristic scale:

ξ

Plot:

ξ(z)

---

# 6. Interface Thickness

Compute interface width from gradient field.

L_J ≈ σ_P / √⟨|∇P|²⟩

Plot:

L_J(z)

---

# 7. Power Spectrum

Compute density power spectrum:

P(k)

Steps:

1. Deposit particles on grid
2. FFT density field
3. Compute isotropic P(k)

Compare with ΛCDM expectations.

Look for peaks corresponding to:

100 Mpc — superstructures  
30–50 Mpc — void scale  
5–10 Mpc — filament scale

---

# 8. Correlation Function

Compute two-point correlation:

ξ(r)

Plot:

log ξ(r) vs r

Look for correlation length:

~5–10 Mpc

---

# 9. Cosmic Web Classification

Compute tidal tensor / Hessian classification.

Classify cells into:

Void  
Sheet  
Filament  
Node

Compute volume fractions.

Compare with ΛCDM typical values:

Void ≈ 70–80%  
Filaments ≈ 10–15%

---

# 10. Visualization

Produce:

2D slices:

Density  
Polarization

3D visualizations:

Polarized domains  
Density filaments

Generate 4K rotation video.

---

# 11. Observational Comparison

Compare simulation statistics with galaxy surveys.

Datasets:

SDSS  
DESI

Compare:

Power spectrum P(k)

Large scale structure scales.

---

# 12. Key Scientific Tests

The Janus model will be supported if:

1. Segregation reaches stable plateau

σ_P ≈ constant

2. Domain size stabilizes

ξ ≈ 30–50 Mpc

3. Interface thickness corresponds to filament scale

L_J ≈ 5–10 Mpc

4. Density power spectrum shows realistic cosmic web structure.

---

# End of Research Plan