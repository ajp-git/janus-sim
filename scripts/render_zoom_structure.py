#!/usr/bin/env python3
"""
Render structure images for zoom simulation snapshots.
Compares initial (step 0) vs final snapshot.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path

def load_snapshot(path):
    """Load binary snapshot."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    signs = np.sign(data[:, 6])
    return pos, signs, n

def make_density_grid(pos, signs, box, grid_size, sign_filter=None):
    """Create 2D density grid (XY projection, full Z)."""
    half = box / 2

    # Filter by sign if requested
    if sign_filter is not None:
        mask = signs == sign_filter
        pos = pos[mask]

    # Normalize to [0, grid_size]
    x = (pos[:, 0] + half) / box * grid_size
    y = (pos[:, 1] + half) / box * grid_size

    # Clip to grid bounds
    x = np.clip(x, 0, grid_size - 1).astype(int)
    y = np.clip(y, 0, grid_size - 1).astype(int)

    # Create density grid
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    np.add.at(grid, (y, x), 1.0)

    return grid

def render_comparison(snap0_path, snap_final_path, output_path, box, grid_size=512, sigma=0.5):
    """Render comparison of initial vs final snapshot."""

    print(f"Loading {snap0_path}...")
    pos0, signs0, n0 = load_snapshot(snap0_path)
    n_plus0 = np.sum(signs0 > 0)
    n_minus0 = np.sum(signs0 < 0)
    print(f"  N={n0:,} (N+={n_plus0:,}, N-={n_minus0:,})")

    print(f"Loading {snap_final_path}...")
    pos_f, signs_f, n_f = load_snapshot(snap_final_path)
    n_plus_f = np.sum(signs_f > 0)
    n_minus_f = np.sum(signs_f < 0)
    print(f"  N={n_f:,} (N+={n_plus_f:,}, N-={n_minus_f:,})")

    # Create density grids
    print("Creating density grids...")

    # Step 0
    grid0_plus = gaussian_filter(make_density_grid(pos0, signs0, box, grid_size, +1), sigma)
    grid0_minus = gaussian_filter(make_density_grid(pos0, signs0, box, grid_size, -1), sigma)

    # Final step
    grid_f_plus = gaussian_filter(make_density_grid(pos_f, signs_f, box, grid_size, +1), sigma)
    grid_f_minus = gaussian_filter(make_density_grid(pos_f, signs_f, box, grid_size, -1), sigma)

    # Normalize for display
    def normalize(g):
        if g.max() > 0:
            return np.log10(g + 1) / np.log10(g.max() + 1)
        return g

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    extent = [-box/2, box/2, -box/2, box/2]

    # Row 0: Step 0 (ICs)
    axes[0, 0].imshow(normalize(grid0_plus), origin='lower', extent=extent, cmap='Reds')
    axes[0, 0].set_title(f'Step 0 — m+ (N={n_plus0:,})', fontsize=12)
    axes[0, 0].set_xlabel('X [Mpc]')
    axes[0, 0].set_ylabel('Y [Mpc]')

    axes[0, 1].imshow(normalize(grid0_minus), origin='lower', extent=extent, cmap='Blues')
    axes[0, 1].set_title(f'Step 0 — m- (N={n_minus0:,})', fontsize=12)
    axes[0, 1].set_xlabel('X [Mpc]')

    # Overlay for step 0
    rgb0 = np.zeros((grid_size, grid_size, 3))
    rgb0[:, :, 0] = normalize(grid0_plus)  # Red = m+
    rgb0[:, :, 2] = normalize(grid0_minus)  # Blue = m-
    axes[0, 2].imshow(rgb0, origin='lower', extent=extent)
    axes[0, 2].set_title('Step 0 — Overlay (Red=m+, Blue=m-)', fontsize=12)
    axes[0, 2].set_xlabel('X [Mpc]')

    # Row 1: Final step
    axes[1, 0].imshow(normalize(grid_f_plus), origin='lower', extent=extent, cmap='Reds')
    axes[1, 0].set_title(f'Step 4000 — m+ (N={n_plus_f:,})', fontsize=12)
    axes[1, 0].set_xlabel('X [Mpc]')
    axes[1, 0].set_ylabel('Y [Mpc]')

    axes[1, 1].imshow(normalize(grid_f_minus), origin='lower', extent=extent, cmap='Blues')
    axes[1, 1].set_title(f'Step 4000 — m- (N={n_minus_f:,})', fontsize=12)
    axes[1, 1].set_xlabel('X [Mpc]')

    # Overlay for final
    rgb_f = np.zeros((grid_size, grid_size, 3))
    rgb_f[:, :, 0] = normalize(grid_f_plus)  # Red = m+
    rgb_f[:, :, 2] = normalize(grid_f_minus)  # Blue = m-
    axes[1, 2].imshow(rgb_f, origin='lower', extent=extent)
    axes[1, 2].set_title('Step 4000 — Overlay (Red=m+, Blue=m-)', fontsize=12)
    axes[1, 2].set_xlabel('X [Mpc]')

    # Add main title
    fig.suptitle(f'Zoom Phase 1: z=3.39→0 | Box={box} Mpc | N={n_f:,} particles', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap0', required=True, help='Path to step 0 snapshot')
    parser.add_argument('--snap-final', required=True, help='Path to final snapshot')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--box', type=float, default=80.0, help='Box size in Mpc')
    parser.add_argument('--grid', type=int, default=512, help='Grid size')
    parser.add_argument('--sigma', type=float, default=0.5, help='Gaussian smoothing sigma')
    args = parser.parse_args()

    render_comparison(args.snap0, args.snap_final, args.output, args.box, args.grid, args.sigma)

if __name__ == "__main__":
    main()
