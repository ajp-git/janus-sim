#!/usr/bin/env python3
"""
Render GPU snapshots to PNG frames
Input: Binary files with header + positions + signs
Output: PNG frames showing particle distribution
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import sys

def read_snapshot(path):
    """Read binary snapshot file"""
    with open(path, 'rb') as f:
        # Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + n(u32)
        step = struct.unpack('<I', f.read(4))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
        n = struct.unpack('<I', f.read(4))[0]

        # Positions: N*3*f32
        pos_f32 = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32)
        pos = pos_f32.reshape(n, 3)

        # Signs: N*i8
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    return step, box_size, seg, ke_ratio, pos, signs

def render_frame(snapshot_path, output_path, sample_frac=0.1):
    """Render a single frame"""
    step, box_size, seg, ke_ratio, pos, signs = read_snapshot(snapshot_path)

    # Subsample for visualization
    n = len(signs)
    sample_size = int(n * sample_frac)
    indices = np.random.choice(n, sample_size, replace=False)

    pos_s = pos[indices]
    signs_s = signs[indices]

    # Separate positive and negative
    pos_plus = pos_s[signs_s > 0]
    pos_minus = pos_s[signs_s < 0]

    # Create figure with 3 views
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')

    views = [
        ('XY', 0, 1),
        ('XZ', 0, 2),
        ('YZ', 1, 2),
    ]

    for ax, (name, ix, iy) in zip(axes, views):
        ax.set_facecolor('black')

        # Plot particles (blue=+, red=-)
        ax.scatter(pos_plus[:, ix], pos_plus[:, iy], s=0.2, alpha=0.3, c='#4488ff', marker='.')
        ax.scatter(pos_minus[:, ix], pos_minus[:, iy], s=0.2, alpha=0.3, c='#ff4444', marker='.')

        half = box_size / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.set_title(f'{name}', color='white', fontsize=14)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    # Add info text
    fig.suptitle(f'Step {step}  |  KE/KE₀={ke_ratio:.4f}  |  Seg={seg:.4f}  |  N={n:,}',
                 color='white', fontsize=16, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()

    return step

def main():
    if len(sys.argv) < 2:
        print("Usage: render_gpu_snapshots.py <render_data_dir> [output_dir]")
        sys.exit(1)

    render_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else render_dir.parent / 'frames'
    output_dir.mkdir(exist_ok=True)

    # Find all snapshot files
    snapshots = sorted(render_dir.glob('step_*.bin'))
    print(f"Found {len(snapshots)} snapshots in {render_dir}")
    print(f"Output to {output_dir}")

    for i, snap_path in enumerate(snapshots):
        out_path = output_dir / f'frame_{snap_path.stem}.png'
        step = render_frame(snap_path, out_path)
        print(f"[{i+1}/{len(snapshots)}] Rendered step {step}")

    print(f"\nDone! {len(snapshots)} frames saved to {output_dir}")

if __name__ == '__main__':
    main()
