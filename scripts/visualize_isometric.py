#!/usr/bin/env python3
"""
Isometric 3D visualization + density contrast analysis for Janus snapshots

New snapshot format (janus_85m.rs):
  Header: 128 bytes text "step=X time=X.XXX eta=X n=XXXXXXXX\n" + padding
  pos:    N × 3 × f32
  vel:    N × 3 × f32
  signs:  N × i8

WARNING: 3D scatter plot will be slow for 85M+ particles.
         Use heavy subsampling (e.g., 1000) or use visualize_snapshot.py instead.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def read_snapshot(path):
    """Read new-format snapshot binary file"""
    with open(path, 'rb') as f:
        # Header: 128 bytes text
        header = f.read(128).decode('utf-8', errors='ignore').strip()

        # Parse header: "step=X time=X.XXX eta=X n=XXXXXXXX"
        parts = {}
        for part in header.split():
            if '=' in part:
                k, v = part.split('=', 1)
                parts[k] = v

        n = int(parts.get('n', 0))
        step = int(parts.get('step', 0))
        eta = float(parts.get('eta', 1.045))
        time = float(parts.get('time', 0))

        print(f"  Header: step={step}, time={time:.3f}, eta={eta}, n={n:,}")

        # pos: N × 3 × f32
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)

        # vel: N × 3 × f32 (skip for visualization)
        f.seek(n * 3 * 4, 1)

        # signs: N × i8
        signs = np.frombuffer(f.read(n), dtype=np.int8)

        # Compute segregation
        pos_plus = pos[signs > 0]
        pos_minus = pos[signs <= 0]
        com_plus = pos_plus.mean(axis=0) if len(pos_plus) > 0 else np.zeros(3)
        com_minus = pos_minus.mean(axis=0) if len(pos_minus) > 0 else np.zeros(3)
        box_size = (pos.max() - pos.min()) * 1.05
        seg = np.linalg.norm(com_plus - com_minus) / box_size

    return pos, signs, step, time, seg


def compute_density_grid(positions, box_size, n_grid=32):
    """Compute density on a grid using NGP (Nearest Grid Point)"""
    # Shift positions to [0, box_size] range
    pos_shifted = positions + box_size / 2

    # Grid cell size
    cell_size = box_size / n_grid

    # Initialize density grid
    density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    # NGP assignment
    ix = np.clip((pos_shifted[:, 0] / cell_size).astype(int), 0, n_grid - 1)
    iy = np.clip((pos_shifted[:, 1] / cell_size).astype(int), 0, n_grid - 1)
    iz = np.clip((pos_shifted[:, 2] / cell_size).astype(int), 0, n_grid - 1)

    # Count particles in each cell
    np.add.at(density, (ix, iy, iz), 1)

    return density


def analyze_density(positions, signs, box_size=None, n_grid=32):
    """Analyze density contrast for m+ and m- separately"""
    if box_size is None:
        box_size = (positions.max() - positions.min()) * 1.05

    mask_plus = signs > 0
    mask_minus = signs <= 0

    pos_plus = positions[mask_plus]
    pos_minus = positions[mask_minus]

    n_plus = len(pos_plus)
    n_minus = len(pos_minus)

    print(f"\n=== Density Analysis (grid {n_grid}³) ===")
    print(f"N+ = {n_plus:,}, N- = {n_minus:,}")
    print(f"Box size = {box_size:.2f}")

    # Compute density grids
    rho_plus = compute_density_grid(pos_plus, box_size, n_grid)
    rho_minus = compute_density_grid(pos_minus, box_size, n_grid)

    # Mean density (particles per cell)
    mean_plus = n_plus / (n_grid ** 3)
    mean_minus = n_minus / (n_grid ** 3)

    print(f"\nMean density per cell:")
    print(f"  m+: {mean_plus:.2f} particles/cell")
    print(f"  m-: {mean_minus:.2f} particles/cell")

    # Density contrast δρ/ρ
    delta_plus = (rho_plus - mean_plus) / mean_plus
    delta_minus = (rho_minus - mean_minus) / mean_minus

    max_delta_plus = np.max(delta_plus)
    max_delta_minus = np.max(delta_minus)

    print(f"\nMax density contrast δρ/ρ:")
    print(f"  m+: {max_delta_plus:.2f} ({max_delta_plus + 1:.1f}× mean)")
    print(f"  m-: {max_delta_minus:.2f} ({max_delta_minus + 1:.1f}× mean)")

    # Fraction in dense halos (> 10× mean)
    threshold = 10.0

    dense_mask_plus = rho_plus > threshold * mean_plus
    dense_mask_minus = rho_minus > threshold * mean_minus

    n_dense_cells_plus = np.sum(dense_mask_plus)
    n_dense_cells_minus = np.sum(dense_mask_minus)

    particles_in_dense_plus = np.sum(rho_plus[dense_mask_plus])
    particles_in_dense_minus = np.sum(rho_minus[dense_mask_minus])

    frac_plus = particles_in_dense_plus / n_plus * 100 if n_plus > 0 else 0
    frac_minus = particles_in_dense_minus / n_minus * 100 if n_minus > 0 else 0

    print(f"\nFraction in dense halos (ρ > {threshold:.0f}× mean):")
    print(f"  m+: {frac_plus:.2f}% ({int(particles_in_dense_plus):,} particles in {n_dense_cells_plus} cells)")
    print(f"  m-: {frac_minus:.2f}% ({int(particles_in_dense_minus):,} particles in {n_dense_cells_minus} cells)")

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
    """Create isometric 3D visualization (WARNING: slow for 85M+)"""
    print(f"Reading {path}...")
    pos, signs, step, time, seg = read_snapshot(path)
    n = len(signs)

    print(f"  N={n:,}, step={step}, time={time:.3f}, S={seg:.6f}")

    if n > 1_000_000 and subsample < 100:
        print(f"  WARNING: N={n:,} is large. Consider subsample >= 100 for speed.")

    # Subsample for plotting
    idx = np.arange(0, n, subsample)
    pos_sub = pos[idx]
    signs_sub = signs[idx]

    # Separate + and - particles
    mask_plus = signs_sub > 0
    mask_minus = signs_sub <= 0

    pos_plus = pos_sub[mask_plus]
    pos_minus = pos_sub[mask_minus]

    print(f"  Plotting {len(pos_plus):,} m+ (blue), {len(pos_minus):,} m- (red)")

    # Create isometric figure
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Plot m- first (red), then m+ (blue)
    ax.scatter(pos_minus[:, 0], pos_minus[:, 1], pos_minus[:, 2],
               c='orangered', s=0.5, alpha=0.5, rasterized=True, label='m-')
    ax.scatter(pos_plus[:, 0], pos_plus[:, 1], pos_plus[:, 2],
               c='blue', s=0.5, alpha=0.5, rasterized=True, label='m+')

    # Auto-detect box range
    box_half = (pos.max() - pos.min()) / 2 * 1.05
    ax.set_xlim(-box_half, box_half)
    ax.set_ylim(-box_half, box_half)
    ax.set_zlim(-box_half, box_half)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    # Isometric view angle
    ax.view_init(elev=25, azim=45)

    ax.set_title(f'Step {step} | t={time:.2f} | S = {seg:.4e} | N = {n:,}', fontsize=14)
    ax.legend(loc='upper right', markerscale=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

    return pos, signs, step


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python visualize_isometric.py <snapshot.bin> <output.png> [subsample]")
        print("  subsample: default=20, use 100-1000 for 85M particles")
        sys.exit(1)

    snap_path = sys.argv[1]
    out_path = sys.argv[2]
    subsample = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    # Generate visualization
    pos, signs, step = visualize_isometric(snap_path, out_path, subsample)

    # Density analysis
    analyze_density(pos, signs, n_grid=32)
