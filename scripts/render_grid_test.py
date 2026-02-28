#!/usr/bin/env python3
"""
Render grid artifact test frames
Binary format: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + redshift(f64) + n(u32) + pos(n*3*f32) + signs(n*i8)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import sys

def read_snapshot(path):
    """Read binary snapshot file with redshift"""
    with open(path, 'rb') as f:
        step = struct.unpack('<I', f.read(4))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
        redshift = struct.unpack('<d', f.read(8))[0]
        n = struct.unpack('<I', f.read(4))[0]

        pos_f32 = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32)
        pos = pos_f32.reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    return step, box_size, seg, ke_ratio, redshift, pos, signs

def render_frame(snapshot_path, output_path, sample_frac=0.05):
    """Render a single frame with higher resolution for grid detection"""
    step, box_size, seg, ke_ratio, redshift, pos, signs = read_snapshot(snapshot_path)

    # Subsample for visualization (5% of 1M = 50K points)
    n = len(signs)
    sample_size = int(n * sample_frac)
    np.random.seed(42)  # Reproducible
    indices = np.random.choice(n, sample_size, replace=False)

    pos_s = pos[indices]
    signs_s = signs[indices]

    # Separate positive and negative
    pos_plus = pos_s[signs_s > 0]
    pos_minus = pos_s[signs_s < 0]

    # Create figure with 3 views (higher resolution for artifact detection)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='black')

    views = [
        ('XY projection', 0, 1),
        ('XZ projection', 0, 2),
        ('YZ projection', 1, 2),
    ]

    for ax, (name, ix, iy) in zip(axes, views):
        ax.set_facecolor('black')

        # Plot particles (smaller dots for better artifact visibility)
        ax.scatter(pos_plus[:, ix], pos_plus[:, iy], s=0.1, alpha=0.5, c='#4488ff', marker='.', rasterized=True)
        ax.scatter(pos_minus[:, ix], pos_minus[:, iy], s=0.1, alpha=0.5, c='#ff4444', marker='.', rasterized=True)

        half = box_size / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.set_title(f'{name}', color='white', fontsize=16)
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_xlabel('Mpc', color='white')
        ax.set_ylabel('Mpc', color='white')

    # Info text with prominent display
    title = f'Step {step}  |  z={redshift:.2f}  |  Seg={seg:.4f}  |  N={n:,}  |  θ=0.5'
    fig.suptitle(title, color='white', fontsize=18, y=0.98)

    # Add grid artifact warning
    fig.text(0.5, 0.02, 'CHECK FOR GRID/SHEET ARTIFACTS (horizontal/vertical lines)',
             ha='center', color='yellow', fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, facecolor='black', bbox_inches='tight')
    plt.close()

    return step, redshift

def main():
    if len(sys.argv) < 2:
        print("Usage: render_grid_test.py <render_data_dir> [output_dir]")
        sys.exit(1)

    render_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else render_dir.parent / 'frames'
    output_dir.mkdir(exist_ok=True, parents=True)

    snapshots = sorted(render_dir.glob('step_*.bin'))
    print(f"Found {len(snapshots)} snapshots in {render_dir}")

    for snap_path in snapshots:
        out_path = output_dir / f'frame_{snap_path.stem}.png'
        step, z = render_frame(snap_path, out_path)
        print(f"Rendered step {step} (z={z:.2f}) -> {out_path.name}")

    print(f"\nDone! Check frames in {output_dir} for grid artifacts")

if __name__ == '__main__':
    main()
