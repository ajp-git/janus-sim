#!/usr/bin/env python3
"""Analyze ξ(z) and L_J(z) for coarsening vs box-size artifact"""

import numpy as np
import matplotlib.pyplot as plt

# Run 1: L=200 Mpc, 1M particles
run1_data = """
0,5.0000,0.16667,0.0003,0.5992,2.79,3.12
100,4.6309,0.17759,0.0008,0.6868,2.88,6.25
200,4.2840,0.18925,0.0156,0.7718,3.35,25.00
300,3.9616,0.20155,0.0880,0.6711,5.14,100.00
400,3.6638,0.21442,0.0988,0.7404,5.00,100.00
500,3.3900,0.22779,0.0604,0.7480,5.10,100.00
600,3.1390,0.24160,0.0349,0.7541,5.19,100.00
700,2.9091,0.25581,0.0794,0.7348,5.30,100.00
800,2.6986,0.27037,0.1009,0.7104,5.36,100.00
900,2.5057,0.28525,0.0744,0.6850,5.43,100.00
1000,2.3288,0.30041,0.1060,0.6690,5.39,100.00
1100,2.1663,0.31583,0.1032,0.6418,5.52,100.00
1200,2.0168,0.33147,0.1100,0.6176,5.60,100.00
1300,1.8791,0.34733,0.1601,0.5979,5.66,100.00
1400,1.7520,0.36338,0.1359,0.5758,5.79,100.00
1500,1.6344,0.37960,0.1366,0.5541,5.90,100.00
1600,1.5254,0.39598,0.1750,0.5356,6.04,100.00
1700,1.4242,0.41251,0.1630,0.5231,6.03,100.00
1800,1.3301,0.42917,0.1147,0.5163,5.94,100.00
1900,1.2423,0.44596,0.1504,0.5109,5.86,100.00
2000,1.1604,0.46287,0.1433,0.5019,5.96,100.00
"""

# Run 2: L=500 Mpc, 5M particles
run2_data = """
0,5.0000,0.16667,0.0002,0.2386,6.95,7.81
100,4.6309,0.17759,0.0053,0.4264,12.38,78.12
200,4.2840,0.18925,0.0546,0.8762,17.37,250.00
300,3.9616,0.20155,0.1641,0.8869,19.13,250.00
400,3.6638,0.21442,0.1483,0.8851,19.19,250.00
500,3.3900,0.22779,0.1619,0.8946,19.54,250.00
600,3.1390,0.24160,0.1944,0.9018,19.36,250.00
700,2.9091,0.25581,0.1776,0.8967,18.79,250.00
800,2.6986,0.27037,0.1942,0.8828,18.31,250.00
900,2.5057,0.28525,0.1965,0.8576,17.69,250.00
1000,2.3288,0.30041,0.1948,0.8248,17.09,250.00
1100,2.1663,0.31583,0.2169,0.7883,16.82,250.00
1200,2.0168,0.33147,0.2005,0.7518,16.78,250.00
1300,1.8791,0.34733,0.2090,0.7145,16.96,250.00
1400,1.7520,0.36338,0.2288,0.6789,17.54,250.00
1500,1.6344,0.37960,0.2148,0.6478,17.97,250.00
1600,1.5254,0.39598,0.2269,0.6165,18.82,250.00
1700,1.4242,0.41251,0.2315,0.5876,19.78,250.00
1800,1.3301,0.42917,0.2264,0.5606,20.77,250.00
1900,1.2423,0.44596,0.2372,0.5363,21.81,250.00
2000,1.1604,0.46287,0.2399,0.5145,22.79,250.00
2100,1.0839,0.47988,0.2512,0.4942,23.91,250.00
2200,1.0121,0.49699,0.2567,0.4746,24.98,250.00
2300,0.9448,0.51420,0.2738,0.4580,25.99,117.19
2400,0.8815,0.53149,0.2740,0.4406,27.12,109.38
2500,0.8219,0.54887,0.2786,0.4263,28.13,109.38
2600,0.7658,0.56632,0.2968,0.4112,29.23,101.56
2700,0.7128,0.58384,0.3023,0.4004,30.05,101.56
2800,0.6627,0.60143,0.2990,0.3873,30.98,101.56
2900,0.6153,0.61909,0.3268,0.3756,31.64,93.75
3000,0.5704,0.63680,0.3344,0.3648,32.75,93.75
3100,0.5277,0.65457,0.3282,0.3560,33.52,93.75
3200,0.4872,0.67240,0.3545,0.3464,34.00,93.75
3300,0.4487,0.69027,0.3451,0.3381,34.80,93.75
3400,0.4120,0.70820,0.3676,0.3293,35.69,85.94
3500,0.3771,0.72617,0.3768,0.3219,36.39,85.94
3600,0.3438,0.74418,0.3892,0.3139,37.13,85.94
3700,0.3119,0.76223,0.3924,0.3077,37.61,85.94
3800,0.2815,0.78033,0.4012,0.3061,35.57,85.94
3900,0.2524,0.79846,0.4096,0.3030,34.72,85.94
4000,0.2245,0.81663,0.4160,0.3002,33.69,85.94
4100,0.1978,0.83483,0.4157,0.2965,34.10,85.94
4200,0.1722,0.85307,0.4222,0.2912,34.04,85.94
4300,0.1477,0.87134,0.4171,0.2878,34.12,78.12
4400,0.1241,0.88964,0.4273,0.2859,33.69,78.12
4500,0.1014,0.90796,0.4208,0.2810,34.12,78.12
4600,0.0795,0.92632,0.4240,0.2821,32.85,78.12
4700,0.0585,0.94470,0.4272,0.2760,33.86,78.12
4800,0.0383,0.96311,0.4257,0.2732,33.74,78.12
4900,0.0188,0.98154,0.4243,0.2729,32.38,78.12
5000,0.0000,1.00000,0.4279,0.2698,32.60,78.12
"""

def parse_data(data_str):
    lines = [l.strip() for l in data_str.strip().split('\n') if l.strip()]
    step, z, a, seg, sigma_P, L_J, xi = [], [], [], [], [], [], []
    for line in lines:
        parts = line.split(',')
        step.append(int(parts[0]))
        z.append(float(parts[1]))
        a.append(float(parts[2]))
        seg.append(float(parts[3]))
        sigma_P.append(float(parts[4]))
        L_J.append(float(parts[5]))
        xi.append(float(parts[6]))
    return np.array(step), np.array(z), np.array(a), np.array(seg), np.array(sigma_P), np.array(L_J), np.array(xi)

s1, z1, a1, seg1, sp1, lj1, xi1 = parse_data(run1_data)
s2, z2, a2, seg2, sp2, lj2, xi2 = parse_data(run2_data)

# Cosmic time t ∝ a^(3/2) in matter domination
t1 = a1**1.5
t2 = a2**1.5

# Normalize xi by box size
L1, L2 = 200, 500
xi1_norm = xi1 / L1
xi2_norm = xi2 / L2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: ξ(z) raw
ax = axes[0, 0]
ax.plot(z1, xi1, 'b.-', label=f'Run1 L=200 Mpc, 1M', markersize=4)
ax.plot(z2, xi2, 'r.-', label=f'Run2 L=500 Mpc, 5M', markersize=4)
ax.axhline(100, color='b', ls='--', alpha=0.5, label='L/2 = 100 (Run1)')
ax.axhline(250, color='r', ls='--', alpha=0.5, label='L/2 = 250 (Run2)')
ax.set_xlabel('z')
ax.set_ylabel('ξ [Mpc]')
ax.set_title('Correlation length ξ(z)')
ax.legend(fontsize=8)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Panel 2: ξ/L normalized - test for box artifact
ax = axes[0, 1]
ax.plot(z1, xi1_norm, 'b.-', label='Run1 ξ/L', markersize=4)
ax.plot(z2, xi2_norm, 'r.-', label='Run2 ξ/L', markersize=4)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='Saturation at L/2')
ax.set_xlabel('z')
ax.set_ylabel('ξ / L')
ax.set_title('Normalized ξ/L — box artifact test')
ax.legend(fontsize=8)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Panel 3: L_J(z) - the robust scale
ax = axes[1, 0]
ax.plot(z1, lj1, 'b.-', label=f'Run1 L_J (mean={np.mean(lj1[3:]):.1f} Mpc)', markersize=4)
ax.plot(z2, lj2, 'r.-', label=f'Run2 L_J (mean late={np.mean(lj2[-20:]):.1f} Mpc)', markersize=4)
ax.axhline(4, color='green', ls='--', lw=2, label='Filament thickness 1-5 Mpc')
ax.axhline(5, color='green', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('z')
ax.set_ylabel('L_J [Mpc]')
ax.set_title('Jeans length L_J(z)')
ax.legend(fontsize=8)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Panel 4: ξ vs cosmic time - coarsening test
ax = axes[1, 1]
# Only non-saturated part for Run2
mask2 = xi2 < 240  # below saturation
ax.loglog(t2[mask2], xi2[mask2], 'r.-', label='Run2 (non-saturated)', markersize=6)

# Fit power law if enough points
if np.sum(mask2) > 5:
    t_fit = t2[mask2]
    xi_fit = xi2[mask2]
    # Filter out early times
    late_mask = t_fit > 0.3
    if np.sum(late_mask) > 3:
        coeffs = np.polyfit(np.log(t_fit[late_mask]), np.log(xi_fit[late_mask]), 1)
        slope = coeffs[0]
        t_line = np.linspace(t_fit[late_mask].min(), t_fit[late_mask].max(), 50)
        xi_line = np.exp(coeffs[1]) * t_line**slope
        ax.loglog(t_line, xi_line, 'r--', label=f'Fit: ξ ∝ t^{{{slope:.2f}}}')

# Reference lines
t_ref = np.linspace(0.1, 1.0, 50)
ax.loglog(t_ref, 80 * (t_ref/0.5)**0.33, 'g--', alpha=0.5, label='t^{1/3} coarsening')
ax.loglog(t_ref, 80 * (t_ref/0.5)**(-0.5), 'm--', alpha=0.5, label='t^{-1/2} (anti-coarsening)')

ax.set_xlabel('t ∝ a^{3/2}')
ax.set_ylabel('ξ [Mpc]')
ax.set_title('Coarsening test: ξ(t)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/mnt/T2/janus-sim/output/coarsening_analysis.png', dpi=150)
print("Saved: output/coarsening_analysis.png")

# Summary statistics
print("\n" + "="*60)
print("COARSENING ANALYSIS SUMMARY")
print("="*60)

print(f"\nRun 1 (L=200 Mpc, 1M particles):")
print(f"  L_J mean (z<4): {np.mean(lj1[3:]):.2f} ± {np.std(lj1[3:]):.2f} Mpc")
print(f"  ξ saturates at L/2 = 100 Mpc from z≈4")
print(f"  → BOX ARTIFACT: ξ limited by box size")

print(f"\nRun 2 (L=500 Mpc, 5M particles):")
print(f"  L_J early (z>1): {np.mean(lj2[3:22]):.1f} ± {np.std(lj2[3:22]):.1f} Mpc")
print(f"  L_J late (z<0.5): {np.mean(lj2[-20:]):.1f} ± {np.std(lj2[-20:]):.1f} Mpc")
print(f"  ξ: saturates at 250 (z=4→1), then DECREASES to {xi2[-1]:.0f} Mpc")
print(f"  → ANTI-COARSENING: domains shrink at late times?")

# Check if ξ decrease is real physics or numerical
xi_late = xi2[-20:]
z_late = z2[-20:]
if xi_late[-1] < xi_late[0] * 0.9:
    print(f"\n⚠ ξ decreased by {100*(1-xi_late[-1]/xi_late[0]):.0f}% from z={z_late[0]:.2f} to z=0")
    print("  Possible causes:")
    print("  1. Real physics: Janus segregation breaks large-scale correlations")
    print("  2. Numerical: algorithm artifact in ξ calculation")
    print("  3. Resolution: grid too coarse for late-time structure")

print(f"\n{'='*60}")
print("KEY FINDING: L_J is RESOLUTION-DEPENDENT")
print(f"  Run1 (1M): L_J ≈ 5.5 Mpc")
print(f"  Run2 (5M): L_J ≈ 18→33 Mpc")
print("  This suggests L_J is NOT converged yet!")
print("  Need higher resolution to find true intrinsic scale.")
print("="*60)
