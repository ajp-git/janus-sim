# Janus Cosmological Model - V10 High-Resolution Simulation Results

## Executive Summary

This document presents the first high-resolution N-body simulation results for the **Janus bimetric cosmological model** (Petit & D'Agostini 2014, EPJC 2024). The simulation tracks 20 million particles (10M positive mass, 10M negative mass) in a 200 Mpc periodic box, using a TreePM hybrid method with GPU acceleration.

**Key finding**: The simulation reveals a characteristic **interface-driven mixing dynamics** where segregated matter/antimatter domains evolve toward homogeneity following specific power laws.

---

## 1. Simulation Parameters

| Parameter | Value |
|-----------|-------|
| N particles | 19,902,511 (≈20M) |
| Box size | L = 200 Mpc |
| Grid (PM) | 256³ |
| η parameter | 1.045 (from Pantheon+ fit) |
| θ (Barnes-Hut) | 0.70 |
| Softening | ε = 0.1 Mpc |
| Timestep | dt = 0.003 → 0.01 (adaptive) |
| Steps analyzed | 500, 1000, 2000 |

### Janus Interaction Rules (from Petit 2014)

```
Same sign:    F = -G m₁ m₂ / r²  (attractive)
Opposite sign: F = +G |m₁ m₂| / r²  (repulsive)
```

---

## 2. Measured Quantities

### 2.1 Polarization Field

The polarization field P(x) measures local matter/antimatter asymmetry:

```
P(x) = (ρ₊ - ρ₋) / (ρ₊ + ρ₋)
```

where ρ₊ and ρ₋ are CIC-interpolated densities on a 256³ grid.

### 2.2 Key Metrics Evolution

| Step | σ_P | L_J (Mpc) | ξ (Mpc) | ξ/L_J | KE_ratio |
|------|-----|-----------|---------|-------|----------|
| 500 | 0.881 | 1.08 | 19.3 | 17.9 | 0.08 |
| 1000 | 0.827 | 0.99 | 18.0 | 18.2 | 0.30 |
| 2000 | 0.700 | 0.83 | 13.1 | 15.8 | 0.20 |

Where:
- **σ_P** = standard deviation of polarization (segregation strength)
- **L_J** = gradient scale: √(⟨P²⟩ / ⟨|∇P|²⟩) — interface thickness
- **ξ** = correlation length from structure function fit — domain size
- **KE_ratio** = kinetic energy ratio between populations

---

## 3. Two Characteristic Scales Discovered

### 3.1 Interface Scale L_J

L_J represents the **thickness of domain walls** between positive and negative mass regions:

```
L_J = √(⟨P²⟩ / ⟨|∇P|²⟩)
```

Measured values: **L_J ≈ 0.8 - 1.1 Mpc**

### 3.2 Coherence Scale ξ

ξ represents the **domain/cluster size**, measured from the structure function:

```
S(r) = ⟨|P(x+r) - P(x)|²⟩ ≈ S_∞ (1 - e^{-r/ξ})
```

Measured values: **ξ ≈ 13 - 19 Mpc**

### 3.3 Scale Ratio

The ratio ξ/L_J ≈ 16-18 indicates domains are about **16× larger than their interfaces**.

---

## 4. Power Law Dynamics

### 4.1 Temporal Evolution

All characteristic quantities follow power laws:

| Quantity | Power Law | Exponent |
|----------|-----------|----------|
| σ_P(t) | ∝ t^α | α = **-0.160** |
| L_J(t) | ∝ t^β | β = **-0.184** |
| ξ(t) | ∝ t^γ | γ = **-0.257** |

### 4.2 Key Relationship: σ_P ∝ L_J

**Correlation coefficient r = 0.998** (nearly perfect)

Linear fit: σ_P = 0.73 × L_J + 0.09

**Interpretation**: Polarization decay is directly tied to interface thickness, NOT diffusive (diffusive would give α = -0.5).

### 4.3 NOT Standard Coarsening

In classical phase separation (Cahn-Hilliard), domains **grow** with ξ ∝ t^{+1/3}.

Here, domains **shrink**: ξ ∝ t^{-0.26}

**The system evolves toward homogeneity, not segregation.**

---

## 5. Structure Function Analysis

The structure function S(r) = ⟨|P(x+r) - P(x)|²⟩ reveals:

1. **r < L_J**: S(r) ∝ r² (smooth/correlated regime)
2. **r > L_J**: S(r) saturates toward 2⟨P²⟩
3. **Saturation scale**: r(90% saturation) ≈ 40 Mpc

This confirms L_J as the interface thickness and ξ as the coherence length.

---

## 6. Self-Similarity Tests

### 6.1 Gradient PDF Test

PDF(|∇P| / σ_P) for different steps:

| Step | ⟨|∇P|/σ_P⟩ | std |
|------|------------|-----|
| 500 | 0.675 | 0.635 |
| 1000 | 0.806 | 0.615 |
| 2000 | 1.058 | 0.573 |

**Result**: Mean shifts significantly → **NOT strictly self-similar**

### 6.2 Power Spectrum Scaling

Testing P(k,t) = f(k·L_J(t)):

**Result**: Partial collapse at intermediate scales, divergence at high k → **Approximate scaling only**

---

## 7. Physical Interpretation

### 7.1 Mixing Mechanism

The negative exponents suggest an **interface-driven mixing** process:

1. Initial segregation creates +/- domains
2. Repulsive forces at interfaces drive mixing
3. Interface thickness L_J shrinks as gradients steepen
4. Domain size ξ shrinks as mixing progresses
5. σ_P decreases proportionally to L_J

### 7.2 Toward Equilibrium?

The power law decay suggests the system may approach a **quasi-homogeneous equilibrium** state where:
- σ_P → 0 (complete mixing)
- P(x) → 0 everywhere (local neutrality)

This would be a **fundamental prediction** different from JPP's expectation of stable segregated domains.

---

## 8. Comparison with Petit's Publications

### 8.1 Expected from Theory (Petit 2014, 2024)

- Matter and antimatter should **segregate** due to gravitational repulsion
- Large-scale voids should form, filled with negative mass
- Galaxy clusters should be positive mass islands in a negative mass sea

### 8.2 Simulation Results (This Work)

- Initial segregation occurs (σ_P = 0.88 at step 500)
- But segregation **decreases** over time (σ_P = 0.70 at step 2000)
- Domains **shrink**, not grow
- System evolves toward **homogeneity**

### 8.3 Possible Explanations

1. **Simulation too short**: Need more steps to reach stable configuration
2. **Missing physics**: No Hubble expansion, no initial perturbations from inflation
3. **Box too small**: L = 200 Mpc may constrain large-scale dynamics
4. **Resolution effects**: 20M particles may not resolve all scales
5. **Fundamental difference**: Discrete N-body differs from continuum theory

---

## 9. Key Questions for Analysis

1. **Is the mixing behavior physical or numerical?**
   - Does the discrete N-body approach correctly capture Janus dynamics?
   - Would continuum methods (e.g., SPH) give different results?

2. **What is the equilibrium state?**
   - Will σ_P → 0 (complete mixing) or saturate at finite value?
   - Is there a characteristic Janus segregation scale?

3. **How does Hubble expansion affect the dynamics?**
   - Current simulation has no cosmological expansion
   - Would a(t) coupling change the power law exponents?

4. **Comparison with JPP's analytical predictions**:
   - Are the measured L_J ≈ 1 Mpc and ξ ≈ 15 Mpc consistent with theory?
   - What scales does JPP predict for domain sizes?

---

## 10. Technical Notes

### 10.1 Numerical Method

- **TreePM hybrid**: PM for long-range (k < k_Nyquist), Barnes-Hut tree for short-range
- **GPU acceleration**: CUDA kernels for tree traversal (RTX 3060, 12GB VRAM)
- **Precision**: f64 throughout
- **Time integration**: Leapfrog (symplectic)

### 10.2 Diagnostics

- CIC density interpolation on 256³ grid
- Polarization field computed every 100 steps
- Snapshots saved every 500 steps

### 10.3 Computational Cost

- ~85 seconds per step at step 2000
- Total runtime: ~40 hours for 2000 steps
- GPU memory: 3.9 GB

---

## 11. Data Files

All data available in `/mnt/T2/janus-sim/output/janus_v10_highres/`:

- `snapshots/snap_XXXXXX.bin` - Particle data (x, y, z, sign)
- `time_series.csv` - KE, segregation metrics per step
- `analysis_v10_snapshot*/` - Analysis outputs, plots, polarization fields

---

## 12. References

1. **Petit, J.-P. & D'Agostini, G.** (2014). Negative mass hypothesis in cosmology and the nature of dark energy. *Astrophys. Space Sci.* 354, 611-615.

2. **Petit, J.-P., D'Agostini, G. & Debergh, N.** (2024). Reconstruction of Cosmological Models Based on the Janus Paradigm. *Eur. Phys. J. C* 84, 1226.

3. **D'Agostini, G. & Petit, J.-P.** (2018). Constraints on Janus Cosmological model from recent observations of supernovae type Ia. *Astrophys. Space Sci.* 363, 139.

4. **Scolnic, D. et al.** (2022). The Pantheon+ Analysis: The Full Data Set and Light-curve Release. *ApJ* 938, 113.

---

## 13. Summary Table

| Metric | Value | Interpretation |
|--------|-------|----------------|
| L_J | 0.83 Mpc | Interface thickness |
| ξ | 13.1 Mpc | Domain size |
| ξ/L_J | 15.8 | Domain/interface ratio |
| σ_P exponent | -0.16 | Polarization decay rate |
| L_J exponent | -0.18 | Interface shrinking rate |
| σ_P ∝ L_J | r = 0.998 | Perfect correlation |
| Self-similar | Partial | Approximate only |

**Main conclusion**: The Janus N-body simulation shows **mixing dynamics** (not coarsening), with characteristic scales L_J ≈ 1 Mpc and ξ ≈ 15 Mpc, evolving via power laws toward homogeneity.

---

*Document generated: March 2026*
*Simulation: V10 High-Resolution (20M particles)*
*Analysis by: Claude Code + Human supervision*
