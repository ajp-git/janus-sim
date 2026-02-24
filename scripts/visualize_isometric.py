#!/usr/bin/env python3
"""Isometric 3D visualization + density contrast analysis for Janus snapshots"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def compute_density_grid(positions, box_size, n_grid=32):
    """Compute density on a grid using CIC (Cloud-in-Cell)"""
    # Shift positions to [0, box_size] range
    pos_shifted = positions + box_size / 2

    # Grid cell size
    cell_size = box_size / n_grid

    # Initialize density grid
    density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    # Simple NGP (Nearest Grid Point) for speed
    ix = np.clip((pos_shifted[:, 0] / cell_size).astype(int), 0, n_grid - 1)
    iy = np.clip((pos_shifted[:, 1] / cell_size).astype(int), 0, n_grid - 1)
    iz = np.clip((pos_shifted[:, 2] / cell_size).astype(int), 0, n_grid - 1)

    # Count particles in each cell
    np.add.at(density, (ix, iy, iz), 1)

    return density


def analyze_density(positions, signs, box_size=400.0, n_grid=32):
    """Analyze density contrast for m+ and m- separately"""

    mask_plus = signs > 0
    mask_minus = signs < 0

    pos_plus = positions[mask_plus]
    pos_minus = positions[mask_minus]

    n_plus = len(pos_plus)
    n_minus = len(pos_minus)

    print(f"\n=== Density Analysis (grid {n_grid}³) ===")
    print(f"N+ = {n_plus:,}, N- = {n_minus:,}")

    # Compute density grids
    rho_plus = compute_density_grid(pos_plus, box_size, n_grid)
    rho_minus = compute_density_grid(pos_minus, box_size, n_grid)

    # Mean density (particles per cell)
    mean_plus = n_plus / (n_grid ** 3)
    mean_minus = n_minus / (n_grid ** 3)

    print(f"\nMean density per cell:")
    print(f"  m+: {mean_plus:.2f} particles/cell")
    print(f"  m-: {mean_minus:.2f} particles/cell")

    # Density contrast δρ/ρ = (ρ - ρ̄) / ρ̄ = ρ/ρ̄ - 1
    delta_plus = (rho_plus - mean_plus) / mean_plus
    delta_minus = (rho_minus - mean_minus) / mean_minus

    max_delta_plus = np.max(delta_plus)
    max_delta_minus = np.max(delta_minus)

    print(f"\nMax density contrast δρ/ρ:")
    print(f"  m+: {max_delta_plus:.2f} ({max_delta_plus + 1:.1f}× mean)")
    print(f"  m-: {max_delta_minus:.2f} ({max_delta_minus + 1:.1f}× mean)")

    # Fraction in dense halos (> 10× mean)
    threshold = 10.0

    # Count particles in cells with density > threshold × mean
    dense_mask_plus = rho_plus > threshold * mean_plus
    dense_mask_minus = rho_minus > threshold * mean_minus

    n_dense_cells_plus = np.sum(dense_mask_plus)
    n_dense_cells_minus = np.sum(dense_mask_minus)

    particles_in_dense_plus = np.sum(rho_plus[dense_mask_plus])
    particles_in_dense_minus = np.sum(rho_minus[dense_mask_minus])

    frac_plus = particles_in_dense_plus / n_plus * 100
    frac_minus = particles_in_dense_minus / n_minus * 100

    print(f"\nFraction in dense halos (ρ > {threshold:.0f}× mean):")
    print(f"  m+: {frac_plus:.2f}% ({int(particles_in_dense_plus):,} particles in {n_dense_cells_plus} cells)")
    print(f"  m-: {frac_minus:.2f}% ({int(particles_in_dense_minus):,} particles in {n_dense_cells_minus} cells)")

    # Additional stats
    print(f"\nDensity distribution (percentiles):")
    print(f"  m+ : p50={np.percentile(rho_plus, 50):.1f}, p90={np.percentile(rho_plus, 90):.1f}, p99={np.percentile(rho_plus, 99):.1f}, max={np.max(rho_plus):.0f}")
    print(f"  m- : p50={np.percentile(rho_minus, 50):.1f}, p90={np.percentile(rho_minus, 90):.1f}, p99={np.percentile(rho_minus, 99):.1f}, max={np.max(rho_minus):.0f}")

    return {
        'max_delta_plus': max_delta_plus,
        'max_delta_minus': max_delta_minus,
        'frac_dense_plus': frac_plus,
        'frac_dense_minus': frac_minus,
    }


def visualize_isometric(path, output_path, subsample=20):
    """Create isometric 3D visualization"""
    print(f"Reading {path}...")
    pos, signs, step, a, seg = read_snapshot(path)
    n = len(signs)

    print(f"  N={n}, step={step}, a={a:.4f}, S={seg:.4f}")

    # Subsample for plotting
    idx = np.arange(0, n, subsample)
    pos_sub = pos[idx]
    signs_sub = signs[idx]

    # Separate + and - particles
    mask_plus = signs_sub > 0
    mask_minus = signs_sub < 0

    pos_plus = pos_sub[mask_plus]
    pos_minus = pos_sub[mask_minus]

    print(f"  Plotting {len(pos_plus)} m+ (blue), {len(pos_minus)} m- (orange/red)")

    # Create isometric figure
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Plot m- first (orange/red), then m+ (blue)
    ax.scatter(pos_minus[:, 0], pos_minus[:, 1], pos_minus[:, 2],
               c='orangered', s=0.5, alpha=0.5, rasterized=True, label='m-')
    ax.scatter(pos_plus[:, 0], pos_plus[:, 1], pos_plus[:, 2],
               c='blue', s=0.5, alpha=0.5, rasterized=True, label='m+')

    # Set equal aspect ratio
    max_range = 200  # box is 400 Mpc, centered at 0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_xlabel('X [Mpc]', fontsize=12)
    ax.set_ylabel('Y [Mpc]', fontsize=12)
    ax.set_zlabel('Z [Mpc]', fontsize=12)

    # Isometric view angle
    ax.view_init(elev=25, azim=45)

    ax.set_title(f'Step {step} | S = {seg:.4f} | N = {n:,}', fontsize=14)

    # Legend
    ax.legend(loc='upper right', markerscale=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

    return pos, signs, step


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python visualize_isometric.py <snapshot.bin> <output.png> [subsample]")
        sys.exit(1)

    snap_path = sys.argv[1]
    out_path = sys.argv[2]
    subsample = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    # Generate visualization
    pos, signs, step = visualize_isometric(snap_path, out_path, subsample)

    # Density analysis
    analyze_density(pos, signs, box_size=400.0, n_grid=32)
