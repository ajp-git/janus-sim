#!/usr/bin/env python3
"""Test coarsening: ξ ∝ t^{1/3} ?"""

import numpy as np
import matplotlib.pyplot as plt

# Data from time_series.csv (only non-zero xi points)
# Run 1: L=200 Mpc
run1 = np.array([
    # step, z, a, xi
    [0, 5.0, 0.167, 3.12],
    [100, 4.63, 0.178, 6.25],
    [200, 4.28, 0.189, 25.0],
    [300, 3.96, 0.202, 100.0],  # saturated at L/2
])

# Run 2: L=500 Mpc (full data)
run2 = np.array([
    [0, 5.0, 0.167, 7.81],
    [100, 4.63, 0.178, 78.12],
    [200, 4.28, 0.189, 250.0],  # saturated
    [300, 3.96, 0.202, 250.0],
    [2300, 0.94, 0.514, 117.19],  # starts decreasing
    [2400, 0.88, 0.531, 109.38],
    [2500, 0.82, 0.549, 109.38],
    [2600, 0.77, 0.566, 101.56],
    [2700, 0.71, 0.584, 101.56],
    [2900, 0.62, 0.619, 93.75],
    [3400, 0.41, 0.708, 85.94],
    [4300, 0.15, 0.871, 78.12],
    [5000, 0.0, 1.0, 78.12],
])

# Cosmic time: t ∝ a^{3/2} in matter domination
# More precisely, in Janus: t = ∫ da / (a H(a))
# For simplicity, use t ∝ a^{3/2}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: ξ(z)
ax = axes[0]
ax.plot(run1[:, 1], run1[:, 3], 'b.-', label='Run1 L=200', markersize=8)
ax.plot(run2[:, 1], run2[:, 3], 'r.-', label='Run2 L=500', markersize=8)
ax.axhline(100, color='b', ls='--', alpha=0.5, label='L/2=100')
ax.axhline(250, color='r', ls='--', alpha=0.5, label='L/2=250')
ax.set_xlabel('z')
ax.set_ylabel('ξ [Mpc]')
ax.set_title('Correlation length ξ(z)')
ax.legend()
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Panel 2: ξ(a) log-log for coarsening test
ax = axes[1]
t1 = run1[:, 2]**1.5  # t ∝ a^{3/2}
t2 = run2[:, 2]**1.5

# Only non-saturated points
mask1 = run1[:, 3] < 90  # below saturation
mask2_early = run2[:, 3] < 240  # early non-saturated

ax.loglog(t1[mask1], run1[mask1, 3], 'b.-', label='Run1 (non-sat)', markersize=8)

# Run2: early growth then late decrease
early_mask = (run2[:, 2] < 0.2) & (run2[:, 3] < 240)
late_mask = run2[:, 2] > 0.5

if np.sum(early_mask) > 1:
    ax.loglog(t2[early_mask], run2[early_mask, 3], 'r.-', label='Run2 early', markersize=8)
if np.sum(late_mask) > 1:
    ax.loglog(t2[late_mask], run2[late_mask, 3], 'r.--', label='Run2 late (decreasing)', markersize=8, alpha=0.7)

# Reference: t^{1/3} coarsening
t_ref = np.linspace(0.05, 0.3, 50)
xi_coarsening = 10 * (t_ref / 0.1)**(1/3)
ax.loglog(t_ref, xi_coarsening, 'g-', lw=2, label='t^{1/3} coarsening')

# Fit early Run1
if np.sum(mask1) >= 2:
    log_t = np.log(t1[mask1])
    log_xi = np.log(run1[mask1, 3])
    coeffs = np.polyfit(log_t, log_xi, 1)
    slope = coeffs[0]
    ax.set_title(f'Coarsening test: Run1 slope = {slope:.2f}')
    print(f"Run1 early growth: ξ ∝ t^{{{slope:.2f}}}")

ax.set_xlabel('t ∝ a^{3/2}')
ax.set_ylabel('ξ [Mpc]')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# Panel 3: ξ/L normalized
ax = axes[2]
ax.plot(run1[:, 2], run1[:, 3]/200, 'b.-', label='Run1 ξ/L', markersize=8)
ax.plot(run2[:, 2], run2[:, 3]/500, 'r.-', label='Run2 ξ/L', markersize=8)
ax.axhline(0.5, color='gray', ls='--', label='ξ = L/2 (saturation)')
ax.set_xlabel('a')
ax.set_ylabel('ξ / L')
ax.set_title('Normalized ξ/L — box artifact test')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/T2/janus-sim/output/coarsening_xi_test.png', dpi=150)
print("Saved: output/coarsening_xi_test.png")

# Analysis
print("\n" + "="*60)
print("COARSENING ANALYSIS")
print("="*60)

print("""
OBSERVATIONS:
1. Run1: ξ grows from 3→100 Mpc between a=0.17→0.20, then SATURATES at L/2
2. Run2: ξ grows to 250 Mpc (L/2), then DECREASES to 78 Mpc at z=0

EARLY GROWTH (a < 0.2):
- Both runs show rapid ξ growth
- Consistent with gravitational coarsening ξ ∝ t^α

LATE BEHAVIOR (a > 0.5):
- Run1: saturated at L/2 (box artifact)
- Run2: ξ DECREASES from 250→78 Mpc ← ANTI-coarsening!

INTERPRETATION:
The decrease in ξ at late times in Run2 suggests:
1. NOT standard coarsening (would increase)
2. Janus physics may BREAK large-scale correlations
3. Or: numerical artifact in ξ calculation when structures sharpen

KEY INSIGHT:
ξ measures correlation length of PARITY FIELD P = (ρ+ - ρ-)/(ρ+ + ρ-)
As segregation proceeds, P → ±1 locally (pure + or - regions)
This can DECREASE ξ if domains fragment rather than merge

CONCLUSION:
→ Need L=500, N=15M to test if σ_P=0.15 is real or resolution artifact
→ If σ_P increases with N, current Run2 is under-resolved
→ If σ_P stays low, it's real physics: large box = more diffuse segregation
""")
