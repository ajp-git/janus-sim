#!/usr/bin/env python3
"""Generate density projection plots for Janus simulation runs."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import argparse

def load_snapshot(path):
    """Load binary snapshot: header (u32 n) + n*(3*f32 + i8)."""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        pos = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            x, y, z = struct.unpack('<fff', f.read(12))
            sign = struct.unpack('<b', f.read(1))[0]
            pos[i] = [x, y, z]
            signs[i] = sign
    return pos, signs

def plot_density(run_dir, slice_thickness=20, output_path=None):
    """Generate density projection plot."""
    run_dir = Path(run_dir)

    # Find latest snapshot
    snaps = sorted(run_dir.glob("snapshots/snap_*.bin"))
    if not snaps:
        print(f"No snapshots found in {run_dir}")
        return

    snap_path = snaps[-1]
    print(f"Loading {snap_path}")
    pos, signs = load_snapshot(snap_path)

    # Detect box size from positions
    box_size = max(pos.max() - pos.min(), 150.0)
    box_half = box_size / 2 if pos.min() < 0 else 0

    # Adjust coordinates if centered
    if pos.min() < -box_size/4:
        # Coordinates are centered [-L/2, L/2]
        pass
    else:
        # Shift to centered
        pos = pos - box_size/2

    # Select slice in z
    mask = np.abs(pos[:, 2]) < slice_thickness / 2
    pos_slice = pos[mask]
    signs_slice = signs[mask]

    pos_plus = pos_slice[signs_slice > 0]
    pos_minus = pos_slice[signs_slice < 0]

    print(f"  Slice: {len(pos_slice)} particles ({len(pos_plus)} m+, {len(pos_minus)} m-)")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compute COM
    com_plus = pos[signs > 0].mean(axis=0) if (signs > 0).any() else np.zeros(3)
    com_minus = pos[signs < 0].mean(axis=0) if (signs < 0).any() else np.zeros(3)
    dcom = np.sqrt(((com_plus - com_minus)**2).sum())
    seg = (signs_slice > 0).sum() / max(len(signs_slice), 1)

    # Plot 1: All particles scatter
    ax = axes[0]
    extent = box_size / 2
    ax.scatter(pos_minus[:, 0], pos_minus[:, 1], s=0.1, c='blue', alpha=0.3, label='m-')
    ax.scatter(pos_plus[:, 0], pos_plus[:, 1], s=0.1, c='red', alpha=0.3, label='m+')
    ax.scatter([com_plus[0]], [com_plus[1]], marker='*', s=200, c='red', edgecolor='black', zorder=10)
    ax.scatter([com_minus[0]], [com_minus[1]], marker='*', s=200, c='blue', edgecolor='black', zorder=10)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title(f'All particles (|z| < {slice_thickness/2:.0f} Mpc)')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)

    # Plot 2: 2D histogram m+
    ax = axes[1]
    bins = 128
    h_plus, xedges, yedges = np.histogram2d(pos_plus[:, 0], pos_plus[:, 1],
                                             bins=bins, range=[[-extent, extent], [-extent, extent]])
    im = ax.imshow(h_plus.T, origin='lower', extent=[-extent, extent, -extent, extent],
                   cmap='Reds', vmin=0)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('m+ density')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='count')

    # Plot 3: 2D histogram m-
    ax = axes[2]
    h_minus, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 1],
                                    bins=bins, range=[[-extent, extent], [-extent, extent]])
    im = ax.imshow(h_minus.T, origin='lower', extent=[-extent, extent, -extent, extent],
                   cmap='Blues', vmin=0)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('m- density')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='count')

    run_name = run_dir.name
    plt.suptitle(f'{run_name}: z=0 density projection\n|ΔCOM| = {dcom:.1f} Mpc', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    else:
        plt.savefig(run_dir / "density_z0.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {run_dir / 'density_z0.png'}")

    plt.close()
    return dcom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Run directory')
    parser.add_argument('--slice-thickness', type=float, default=20, help='Slice thickness in Mpc')
    parser.add_argument('--output', help='Output path')
    args = parser.parse_args()

    plot_density(args.run, args.slice_thickness, args.output)
