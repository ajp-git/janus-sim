#!/usr/bin/env python3
"""Analyze fine μ scan results (5M particles)"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUT_DIR = Path("/mnt/T2/janus-sim/output")
PLOT_DIR = OUT_DIR / "scan_analysis"
PLOT_DIR.mkdir(exist_ok=True)

# Collect fine scan results
fine_results = []
for mu in [20, 24, 28, 32, 36, 40, 48]:
    summary_file = OUT_DIR / f"scan_mu_{mu}_5M" / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            fine_results.append(data)

# Also load coarse scan results (2M)
coarse_results = []
for mu in [4, 16, 32, 64]:
    summary_file = OUT_DIR / f"scan_mu_{mu}_2M" / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            coarse_results.append(data)

# Extract arrays
fine_mu = np.array([r['mu'] for r in fine_results])
fine_void = np.array([r['void_frac'] for r in fine_results]) * 100
fine_wall = np.array([r['wall_frac'] for r in fine_results]) * 100
fine_blobs = np.array([r['n_blobs'] for r in fine_results])
fine_reff = np.array([r['r_eff_mean'] for r in fine_results])

coarse_mu = np.array([r['mu'] for r in coarse_results]) if coarse_results else np.array([])
coarse_void = np.array([r['void_frac'] for r in coarse_results]) * 100 if coarse_results else np.array([])

# Plot 1: void fraction vs μ (comparing resolutions)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# void fraction
ax = axes[0, 0]
ax.plot(fine_mu, fine_void, 'bo-', markersize=10, linewidth=2, label='5M particles')
if len(coarse_mu) > 0:
    ax.plot(coarse_mu, coarse_void, 'rs--', markersize=8, linewidth=1.5, label='2M particles')
ax.axhspan(70, 80, alpha=0.2, color='green', label='Target (70-80%)')
ax.set_xlabel('μ (N⁻/N⁺)', fontsize=12)
ax.set_ylabel('Void fraction (%)', fontsize=12)
ax.set_title('Void Fraction vs Mass Ratio', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(40, 90)

# wall fraction
ax = axes[0, 1]
ax.plot(fine_mu, fine_wall, 'go-', markersize=10, linewidth=2)
ax.set_xlabel('μ (N⁻/N⁺)', fontsize=12)
ax.set_ylabel('Wall fraction (%)', fontsize=12)
ax.set_title('Wall Fraction vs Mass Ratio', fontsize=14)
ax.grid(True, alpha=0.3)

# n_blobs
ax = axes[1, 0]
ax.bar(fine_mu, fine_blobs, width=3, color='purple', alpha=0.7)
ax.set_xlabel('μ (N⁻/N⁺)', fontsize=12)
ax.set_ylabel('Number of m⁺ blobs', fontsize=12)
ax.set_title('Structure Count vs Mass Ratio', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# r_eff
ax = axes[1, 1]
ax.plot(fine_mu, fine_reff, 'mo-', markersize=10, linewidth=2)
ax.set_xlabel('μ (N⁻/N⁺)', fontsize=12)
ax.set_ylabel('Mean effective radius (Mpc)', fontsize=12)
ax.set_title('Blob Size vs Mass Ratio', fontsize=14)
ax.grid(True, alpha=0.3)

plt.suptitle('Fine μ Scan: 5M particles, 500 Mpc box, z=5→0', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'scan_mu_fine_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOT_DIR / 'scan_mu_fine_analysis.png'}")

# Summary statistics
print("\n=== FINE SCAN SUMMARY ===")
print(f"μ range: {fine_mu.min()} - {fine_mu.max()}")
print(f"void fraction: {fine_void.min():.1f}% - {fine_void.max():.1f}%")
print(f"wall fraction: {fine_wall.min():.1f}% - {fine_wall.max():.1f}%")
print(f"\nBest μ for target 70-80% void:")
target_diff = np.abs(fine_void - 75)
best_idx = np.argmin(target_diff)
print(f"  μ={fine_mu[best_idx]} → void={fine_void[best_idx]:.1f}% (closest to 75%)")

# Find μ* by interpolation
if fine_void.max() >= 70:
    # Linear interpolation to find μ where void=70%
    for i in range(len(fine_void) - 1):
        if fine_void[i] < 70 <= fine_void[i+1]:
            mu_star = fine_mu[i] + (70 - fine_void[i]) / (fine_void[i+1] - fine_void[i]) * (fine_mu[i+1] - fine_mu[i])
            print(f"  μ* ≈ {mu_star:.0f} (interpolated for void=70%)")
            break
else:
    print(f"  Note: void fraction saturates at {fine_void.max():.1f}% (below 70% target)")
    print(f"  May need larger μ or different physics parameters")
