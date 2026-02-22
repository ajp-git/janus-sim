#!/usr/bin/env python3
"""
Render PM-5 snapshots as 3-panel visualization (XY, XZ, YZ projections).
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_light_snapshot(path):
    """Read light snapshot (subsampled, f32 positions)."""
    with open(path, 'rb') as f:
        n_particles = struct.unpack('<Q', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        scale_factor = struct.unpack('<d', f.read(8))[0]
        segregation = struct.unpack('<d', f.read(8))[0]

        # Read positions (f32) and signs (i8)
        data = np.frombuffer(f.read(n_particles * 13), dtype=np.uint8)
        data = data.reshape(n_particles, 13)

        x = np.frombuffer(data[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(data[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(data[:, 8:12].tobytes(), dtype=np.float32)
        signs = np.frombuffer(data[:, 12:13].tobytes(), dtype=np.int8)

    return {
        'n': n_particles,
        'step': step,
        'a': scale_factor,
        'seg': segregation,
        'x': x, 'y': y, 'z': z,
        'signs': signs
    }

def render_3panel(data, output_path, box_size=500.0):
    """Render 3-panel XY, XZ, YZ projection."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')

    x, y, z = data['x'], data['y'], data['z']
    signs = data['signs']

    pos_mask = signs > 0
    neg_mask = signs < 0

    projections = [
        (x, y, 'X', 'Y', 'XY Projection'),
        (x, z, 'X', 'Z', 'XZ Projection'),
        (y, z, 'Y', 'Z', 'YZ Projection'),
    ]

    for ax, (px, py, xlabel, ylabel, title) in zip(axes, projections):
        ax.set_facecolor('black')

        # Plot negative first (behind)
        ax.scatter(px[neg_mask], py[neg_mask],
                   c='red', s=0.05, alpha=0.4, marker='.', rasterized=True)
        # Plot positive on top
        ax.scatter(px[pos_mask], py[pos_mask],
                   c='cyan', s=0.05, alpha=0.4, marker='.', rasterized=True)

        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel, color='white', fontsize=12)
        ax.set_ylabel(ylabel, color='white', fontsize=12)
        ax.set_title(title, color='white', fontsize=14)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    # Main title
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    suptitle = f"PM-5: Step {data['step']:04d}  |  a = {data['a']:.4f}  |  Seg = {data['seg']:.6f}  |  N = {data['n']:,} ({n_pos:,} + / {n_neg:,} -)"
    fig.suptitle(suptitle, color='white', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"  {output_path.name}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python render_3panel.py <input_dir> <output_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.mkdir(parents=True, exist_ok=True)

    # Find light snapshots (< 100 MB)
    snapshots = sorted(input_path.glob("snapshot_[0-9]*.bin"))
    light_snaps = [s for s in snapshots if s.stat().st_size < 100_000_000]

    print(f"Rendering {len(light_snaps)} snapshots...")

    for snap in light_snaps:
        out_file = output_path / f"frame_{snap.stem.split('_')[1]}.png"
        try:
            data = read_light_snapshot(snap)
            render_3panel(data, out_file)
        except Exception as e:
            print(f"  Error {snap.name}: {e}")

if __name__ == "__main__":
    main()
