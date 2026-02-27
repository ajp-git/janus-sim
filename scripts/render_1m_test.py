#!/usr/bin/env python3
"""
Quick render of 1M jitter test to verify no grid pattern.
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

# Parameters
SNAPSHOT_DIR = Path("/mnt/T2/janus-sim/output/1M_jitter_test/snapshots")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/1M_jitter_test/frames")
BOX_SIZE = 215.4
PARTICLE_SIZE = 13
GRID_SIZE = 512


def read_all_particles(filepath):
    """Read all particles for 1M (fits in memory)"""
    with open(filepath, 'rb') as f:
        n_total = struct.unpack('<Q', f.read(8))[0]
        data = f.read(n_total * PARTICLE_SIZE)

        chunk_data = np.frombuffer(data, dtype=np.uint8).reshape(-1, PARTICLE_SIZE)
        pos_bytes = chunk_data[:, :12].tobytes()
        positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
        signs = chunk_data[:, 12].astype(np.int8)

    return positions, signs, n_total


def create_density_grid(positions, signs, grid_size=512, box_size=215.4):
    """Create density grids for + and - particles"""
    half = box_size / 2
    cell_size = box_size / grid_size

    density_plus = np.zeros((grid_size, grid_size), dtype=np.float32)
    density_minus = np.zeros((grid_size, grid_size), dtype=np.float32)

    ix = np.clip(((positions[:, 0] + half) / cell_size).astype(np.int32), 0, grid_size - 1)
    iy = np.clip(((positions[:, 1] + half) / cell_size).astype(np.int32), 0, grid_size - 1)

    mask_pos = signs > 0
    mask_neg = signs < 0

    np.add.at(density_plus, (ix[mask_pos], iy[mask_pos]), 1)
    np.add.at(density_minus, (ix[mask_neg], iy[mask_neg]), 1)

    return density_plus, density_minus


def render_frame(positions, signs, step, n_total, output_path):
    """Render frame with density projection"""

    plus_xy, minus_xy = create_density_grid(positions, signs, GRID_SIZE, BOX_SIZE)

    # Apply log scaling and smoothing
    sigma = 0.5

    def process_grid(g):
        g = gaussian_filter(g.astype(np.float32), sigma=sigma)
        g = np.log1p(g * 10)
        p99 = np.percentile(g, 99.9)
        p50 = np.percentile(g, 50)
        if p99 > p50:
            g = np.clip((g - p50) / (p99 - p50), 0, 1)
        return g

    plus_xy = process_grid(plus_xy)
    minus_xy = process_grid(minus_xy)

    # Create figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='white')

    # Layout: main + side panels
    ax_main = fig.add_axes([0.02, 0.08, 0.62, 0.84], facecolor='black')
    ax_plus = fig.add_axes([0.66, 0.52, 0.32, 0.40], facecolor='black')
    ax_minus = fig.add_axes([0.66, 0.08, 0.32, 0.40], facecolor='black')

    # Colors
    color_plus = np.array([0.2, 0.5, 1.0])   # Blue
    color_minus = np.array([1.0, 0.25, 0.25]) # Red

    # Main panel: RGB composite
    grid_size = plus_xy.shape[0]
    rgb = np.zeros((grid_size, grid_size, 3))
    rgb[:, :, 0] = minus_xy * 0.8           # Red channel
    rgb[:, :, 2] = plus_xy * 0.8            # Blue channel
    rgb[:, :, 1] = np.minimum(plus_xy, minus_xy) * 0.2  # Green = overlap
    rgb = np.clip(rgb, 0, 1)
    rgb = np.power(rgb, 1.2)

    ax_main.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', aspect='equal', interpolation='bilinear',
                   extent=[-BOX_SIZE/2, BOX_SIZE/2, -BOX_SIZE/2, BOX_SIZE/2])
    ax_main.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_main.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_main.set_xlabel('X [Mpc]', color='black', fontsize=14)
    ax_main.set_ylabel('Y [Mpc]', color='black', fontsize=14)
    ax_main.tick_params(colors='black', labelsize=10)
    for spine in ax_main.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)

    # Right top: Masses+
    rgb_plus = np.zeros((grid_size, grid_size, 3))
    rgb_plus[:, :, 0] = plus_xy * color_plus[0]
    rgb_plus[:, :, 1] = plus_xy * color_plus[1]
    rgb_plus[:, :, 2] = plus_xy * color_plus[2]
    rgb_plus = np.power(np.clip(rgb_plus, 0, 1), 1.2)

    ax_plus.imshow(np.transpose(rgb_plus, (1, 0, 2)), origin='lower', aspect='equal', interpolation='bilinear',
                   extent=[-BOX_SIZE/2, BOX_SIZE/2, -BOX_SIZE/2, BOX_SIZE/2])
    ax_plus.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_plus.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_plus.axis('off')
    ax_plus.set_title('Masses+ (blue)', color='#2266dd', fontsize=16, pad=10, fontweight='bold')

    # Right bottom: Masses-
    rgb_minus = np.zeros((grid_size, grid_size, 3))
    rgb_minus[:, :, 0] = minus_xy * color_minus[0]
    rgb_minus[:, :, 1] = minus_xy * color_minus[1]
    rgb_minus[:, :, 2] = minus_xy * color_minus[2]
    rgb_minus = np.power(np.clip(rgb_minus, 0, 1), 1.2)

    ax_minus.imshow(np.transpose(rgb_minus, (1, 0, 2)), origin='lower', aspect='equal', interpolation='bilinear',
                    extent=[-BOX_SIZE/2, BOX_SIZE/2, -BOX_SIZE/2, BOX_SIZE/2])
    ax_minus.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_minus.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_minus.axis('off')
    ax_minus.set_title('Masses- (red)', color='#dd2222', fontsize=16, pad=10, fontweight='bold')

    # Title
    title = f"Janus 1M Jitter Test — Step {step:04d} | N = {n_total:,}"
    fig.text(0.5, 0.97, title, ha='center', va='top', fontsize=32,
             color='black', fontweight='bold')

    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for snap_name in ["snapshot_00000.bin", "snapshot_00100.bin"]:
        snap_path = SNAPSHOT_DIR / snap_name
        if not snap_path.exists():
            print(f"Skip {snap_name} (not found)")
            continue

        step = int(snap_name.split('_')[1].split('.')[0])
        output_path = OUTPUT_DIR / f"frame_{step:05d}.png"

        print(f"Rendering {snap_name}...")
        positions, signs, n_total = read_all_particles(snap_path)
        print(f"  N = {n_total:,}, N+ = {(signs > 0).sum():,}, N- = {(signs < 0).sum():,}")

        render_frame(positions, signs, step, n_total, output_path)
        print(f"  Saved: {output_path}")

    print("\nDone! Check frames for grid pattern.")


if __name__ == "__main__":
    main()
