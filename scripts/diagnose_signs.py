#!/usr/bin/env python3
"""Diagnose sign distribution and clustering in Janus snapshots"""

import numpy as np
import struct
import sys

def read_snapshot(path):
    """Read snapshot binary file"""
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

    return positions, signs, step, scale_factor, segregation


def analyze_signs(path):
    """Analyze sign distribution"""
    print(f"Reading {path}...")
    pos, signs, step, a, seg = read_snapshot(path)
    n = len(signs)

    n_plus = np.sum(signs > 0)
    n_minus = np.sum(signs < 0)
    n_zero = np.sum(signs == 0)

    print(f"\n=== Step {step} Sign Analysis ===")
    print(f"Total particles: {n:,}")
    print(f"  m+ (sign > 0): {n_plus:,} ({100*n_plus/n:.2f}%)")
    print(f"  m- (sign < 0): {n_minus:,} ({100*n_minus/n:.2f}%)")
    print(f"  zero signs:    {n_zero:,}")

    print(f"\nFirst 20 signs: {signs[:20].tolist()}")
    print(f"Last 20 signs:  {signs[-20:].tolist()}")

    # Check if signs are constant (all same)
    unique_signs = np.unique(signs)
    print(f"\nUnique sign values: {unique_signs.tolist()}")

    if len(unique_signs) == 1:
        print("⚠️  BUG: ALL SIGNS ARE IDENTICAL!")
        return

    # Check spatial distribution of signs
    print("\n=== Spatial Distribution ===")

    # Divide box into octants
    center = np.mean(pos, axis=0)
    for octant in range(8):
        ox = 1 if octant & 1 else -1
        oy = 1 if octant & 2 else -1
        oz = 1 if octant & 4 else -1

        mask = ((pos[:, 0] - center[0]) * ox > 0) & \
               ((pos[:, 1] - center[1]) * oy > 0) & \
               ((pos[:, 2] - center[2]) * oz > 0)

        n_oct = np.sum(mask)
        n_plus_oct = np.sum(signs[mask] > 0)
        frac_plus = 100 * n_plus_oct / n_oct if n_oct > 0 else 0

        sign_str = f"({'+' if ox > 0 else '-'}{'+' if oy > 0 else '-'}{'+' if oz > 0 else '-'})"
        print(f"  Octant {sign_str}: {n_oct:,} particles, {frac_plus:.1f}% m+")

    # Check clustering by computing local density ratio
    print("\n=== Density in High-Density Regions ===")

    # Grid analysis
    n_grid = 32
    box_size = 400.0
    cell_size = box_size / n_grid
    pos_shifted = pos + box_size / 2

    # Count particles per cell
    ix = np.clip((pos_shifted[:, 0] / cell_size).astype(int), 0, n_grid - 1)
    iy = np.clip((pos_shifted[:, 1] / cell_size).astype(int), 0, n_grid - 1)
    iz = np.clip((pos_shifted[:, 2] / cell_size).astype(int), 0, n_grid - 1)

    cell_idx = ix * n_grid * n_grid + iy * n_grid + iz

    # Find densest cells
    unique_cells, counts = np.unique(cell_idx, return_counts=True)
    top_cells = unique_cells[np.argsort(counts)[-10:]]  # Top 10 densest

    print("Top 10 densest cells:")
    for cell in top_cells[::-1]:
        mask = cell_idx == cell
        n_cell = np.sum(mask)
        n_plus_cell = np.sum(signs[mask] > 0)
        frac_plus = 100 * n_plus_cell / n_cell if n_cell > 0 else 0
        print(f"  Cell {cell}: {n_cell} particles, {frac_plus:.1f}% m+")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_signs.py <snapshot.bin>")
        sys.exit(1)

    analyze_signs(sys.argv[1])
