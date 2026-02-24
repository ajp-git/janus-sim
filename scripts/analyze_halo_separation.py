#!/usr/bin/env python3
"""Analyze if m+ and m- dense regions are spatially separated or overlapping"""

import numpy as np
import struct
import sys

def read_snapshot(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        scale_factor = struct.unpack('<d', f.read(8))[0]
        segregation = struct.unpack('<d', f.read(8))[0]

        positions = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            x, y, z = struct.unpack('<fff', f.read(12))
            sign = struct.unpack('<b', f.read(1))[0]
            positions[i] = [x, y, z]
            signs[i] = sign

    return positions, signs, step

def analyze_separation(path, n_grid=32, box_size=400.0):
    print(f"Reading {path}...")
    pos, signs, step = read_snapshot(path)
    n = len(signs)

    mask_plus = signs > 0
    mask_minus = signs < 0
    pos_plus = pos[mask_plus]
    pos_minus = pos[mask_minus]

    print(f"Step {step}: {np.sum(mask_plus)} m+, {np.sum(mask_minus)} m-")

    # Grid density for m+ and m- separately
    cell_size = box_size / n_grid
    pos_shifted = pos + box_size / 2

    def compute_density(positions):
        density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
        pos_s = positions + box_size / 2
        ix = np.clip((pos_s[:, 0] / cell_size).astype(int), 0, n_grid - 1)
        iy = np.clip((pos_s[:, 1] / cell_size).astype(int), 0, n_grid - 1)
        iz = np.clip((pos_s[:, 2] / cell_size).astype(int), 0, n_grid - 1)
        np.add.at(density, (ix, iy, iz), 1)
        return density

    rho_plus = compute_density(pos_plus)
    rho_minus = compute_density(pos_minus)
    rho_total = rho_plus + rho_minus

    mean_plus = np.mean(rho_plus)
    mean_minus = np.mean(rho_minus)

    print(f"\n=== Density Analysis ===")
    print(f"Mean density: m+={mean_plus:.1f}, m-={mean_minus:.1f} per cell")

    # Find top 20 densest cells for each type
    flat_plus = rho_plus.flatten()
    flat_minus = rho_minus.flatten()
    flat_total = rho_total.flatten()

    top_plus_idx = np.argsort(flat_plus)[-20:][::-1]
    top_minus_idx = np.argsort(flat_minus)[-20:][::-1]

    print(f"\n=== Top 10 densest m+ cells ===")
    for i, idx in enumerate(top_plus_idx[:10]):
        iz, iy, ix = np.unravel_index(idx, (n_grid, n_grid, n_grid))
        n_plus_here = rho_plus[iz, iy, ix]
        n_minus_here = rho_minus[iz, iy, ix]
        ratio = n_plus_here / (n_plus_here + n_minus_here) * 100 if (n_plus_here + n_minus_here) > 0 else 0
        print(f"  {i+1}. Cell ({ix},{iy},{iz}): {int(n_plus_here)} m+, {int(n_minus_here)} m- ({ratio:.1f}% m+)")

    print(f"\n=== Top 10 densest m- cells ===")
    for i, idx in enumerate(top_minus_idx[:10]):
        iz, iy, ix = np.unravel_index(idx, (n_grid, n_grid, n_grid))
        n_plus_here = rho_plus[iz, iy, ix]
        n_minus_here = rho_minus[iz, iy, ix]
        ratio = n_minus_here / (n_plus_here + n_minus_here) * 100 if (n_plus_here + n_minus_here) > 0 else 0
        print(f"  {i+1}. Cell ({ix},{iy},{iz}): {int(n_minus_here)} m-, {int(n_plus_here)} m+ ({ratio:.1f}% m-)")

    # Check overlap: are the densest m+ cells the same as densest m- cells?
    overlap = len(set(top_plus_idx[:10]) & set(top_minus_idx[:10]))
    print(f"\n=== Spatial Separation ===")
    print(f"Overlap: {overlap}/10 top cells are shared")

    if overlap > 7:
        print("→ m+ and m- halos are COINCIDENT (same locations)")
    elif overlap < 3:
        print("→ m+ and m- halos are SEPARATED (different locations)")
    else:
        print("→ Partial separation")

    # Correlation between m+ and m- density fields
    corr = np.corrcoef(flat_plus, flat_minus)[0, 1]
    print(f"\nCorrelation(ρ_+, ρ_-) = {corr:.4f}")
    if corr > 0.9:
        print("→ Very high correlation: m+ and m- trace the SAME structures")
    elif corr < 0.5:
        print("→ Low correlation: m+ and m- are forming DIFFERENT structures")

    # Density contrast evolution
    print(f"\n=== Density Contrast ===")
    delta_plus = (rho_plus - mean_plus) / mean_plus
    delta_minus = (rho_minus - mean_minus) / mean_minus

    print(f"max(δρ/ρ) m+: {np.max(delta_plus):.1f}")
    print(f"max(δρ/ρ) m-: {np.max(delta_minus):.1f}")

    # Check cells where m+ dominates vs m- dominates
    dominance = (rho_plus - rho_minus) / (rho_total + 1)  # +1 to avoid div by 0
    print(f"\nCells where m+ dominates (>60%): {np.sum(dominance > 0.2)}")
    print(f"Cells where m- dominates (>60%): {np.sum(dominance < -0.2)}")
    print(f"Mixed cells (40-60%): {np.sum(np.abs(dominance) < 0.2)}")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/T2/janus-sim/output/jour4_corrected_1771892736/snapshots/snap_09100.bin"
    analyze_separation(path)
