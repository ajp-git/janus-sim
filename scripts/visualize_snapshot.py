#!/usr/bin/env python3
"""Visualize Janus snapshot with 3 projections (xy, xz, yz)"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

def read_snapshot(path):
    """Read snapshot binary file"""
    with open(path, 'rb') as f:
        # Header: 32 bytes
        n = struct.unpack('<Q', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        scale_factor = struct.unpack('<d', f.read(8))[0]
        segregation = struct.unpack('<d', f.read(8))[0]

        # Particle data: x,y,z (f32), sign (i8) per particle
        positions = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            x, y, z = struct.unpack('<fff', f.read(12))
            sign = struct.unpack('<b', f.read(1))[0]
            positions[i] = [x, y, z]
            signs[i] = sign

    return positions, signs, step, scale_factor, segregation

def visualize_3panel(path, output_path, subsample=10):
    """Create 3-panel projection visualization"""
    print(f"Reading {path}...")
    pos, signs, step, a, seg = read_snapshot(path)
    n = len(signs)

    print(f"  N={n}, step={step}, a={a:.4f}, S={seg:.4f}")

    # Subsample for faster plotting
    idx = np.arange(0, n, subsample)
    pos_sub = pos[idx]
    signs_sub = signs[idx]

    # Separate + and - particles
    mask_plus = signs_sub > 0
    mask_minus = signs_sub < 0

    pos_plus = pos_sub[mask_plus]
    pos_minus = pos_sub[mask_minus]

    print(f"  Plotting {len(pos_plus)} m+ (blue), {len(pos_minus)} m- (red)")

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

    projections = [
        (0, 1, 'X', 'Y', 'XY projection'),
        (0, 2, 'X', 'Z', 'XZ projection'),
        (1, 2, 'Y', 'Z', 'YZ projection'),
    ]

    for ax, (ix, iy, xlabel, ylabel, title) in zip(axes, projections):
        ax.set_facecolor('white')

        # Plot m- (red) first, then m+ (blue) on top
        ax.scatter(pos_minus[:, ix], pos_minus[:, iy],
                   c='red', s=0.1, alpha=0.3, rasterized=True)
        ax.scatter(pos_plus[:, ix], pos_plus[:, iy],
                   c='blue', s=0.1, alpha=0.3, rasterized=True)

        ax.set_xlabel(f'{xlabel} [Mpc]', fontsize=12)
        ax.set_ylabel(f'{ylabel} [Mpc]', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Step {step} | S = {seg:.4f} | N = {n:,}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python visualize_snapshot.py <snapshot.bin> <output.png> [subsample]")
        sys.exit(1)

    snap_path = sys.argv[1]
    out_path = sys.argv[2]
    subsample = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    visualize_3panel(snap_path, out_path, subsample)
