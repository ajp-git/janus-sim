#!/usr/bin/env python3
"""
Render PM-5 light snapshots to PNG images.

Usage:
    python render_snapshot.py <snapshot.bin> <output.png>
    python render_snapshot.py <snapshot_dir> <output_dir>  # batch mode
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
        positions = []
        signs = []
        for _ in range(n_particles):
            x = struct.unpack('<f', f.read(4))[0]
            y = struct.unpack('<f', f.read(4))[0]
            z = struct.unpack('<f', f.read(4))[0]
            sign = struct.unpack('<b', f.read(1))[0]
            positions.append((x, y, z))
            signs.append(sign)

    return {
        'n': n_particles,
        'step': step,
        'a': scale_factor,
        'seg': segregation,
        'pos': np.array(positions),
        'signs': np.array(signs)
    }

def render_snapshot(data, output_path, box_size=500.0):
    """Render XY projection with color by sign."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')

    pos = data['pos']
    signs = data['signs']

    # Separate positive and negative
    pos_mask = signs > 0
    neg_mask = signs < 0

    # Plot with small alpha for density visualization
    ax.scatter(pos[pos_mask, 0], pos[pos_mask, 1],
               c='cyan', s=0.1, alpha=0.3, marker='.')
    ax.scatter(pos[neg_mask, 0], pos[neg_mask, 1],
               c='red', s=0.1, alpha=0.3, marker='.')

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    title = f"Step {data['step']:04d}  a={data['a']:.4f}  Seg={data['seg']:.6f}"
    ax.set_title(title, color='white', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python render_snapshot.py <input> <output>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if input_path.is_file():
        # Single file mode
        data = read_light_snapshot(input_path)
        render_snapshot(data, output_path)
    elif input_path.is_dir():
        # Batch mode
        output_path.mkdir(parents=True, exist_ok=True)
        snapshots = sorted(input_path.glob("snapshot_[0-9]*.bin"))
        print(f"Found {len(snapshots)} snapshots")

        for snap in snapshots:
            # Skip full snapshots (too large)
            if snap.stat().st_size > 100_000_000:  # > 100 MB
                continue

            out_file = output_path / f"{snap.stem}.png"
            try:
                data = read_light_snapshot(snap)
                render_snapshot(data, out_file)
            except Exception as e:
                print(f"  Error with {snap}: {e}")
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
