#!/usr/bin/env python3
"""
FORMAT B — Publication: 3 projections XY|XZ|YZ
Panneaux fond noir, fond global blanc
"""

import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def load_render_data(path):
    """Load binary render_data file."""
    with open(path, 'rb') as f:
        step = struct.unpack('<I', f.read(4))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
        redshift = struct.unpack('<d', f.read(8))[0]
        n = struct.unpack('<I', f.read(4))[0]
        positions = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)
    return {
        'step': step, 'box_size': box_size, 'seg': seg,
        'ke_ratio': ke_ratio, 'z': redshift, 'n': n,
        'pos': positions, 'signs': signs
    }


def create_projection(positions, signs, box_size, axis1, axis2, grid_size=1024):
    """Create density grids for a specific projection."""
    half = box_size / 2
    cell = box_size / grid_size

    p1 = positions[:, axis1]
    p2 = positions[:, axis2]

    ix = np.clip(((p1 + half) / cell).astype(np.int32), 0, grid_size - 1)
    iy = np.clip(((p2 + half) / cell).astype(np.int32), 0, grid_size - 1)

    grid_plus = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid_minus = np.zeros((grid_size, grid_size), dtype=np.float32)

    mask_pos = signs > 0
    mask_neg = signs < 0

    np.add.at(grid_plus, (iy[mask_pos], ix[mask_pos]), 1)
    np.add.at(grid_minus, (iy[mask_neg], ix[mask_neg]), 1)

    return grid_plus, grid_minus


def render_pub(data, output_path, grid_size=1024):
    """Render publication format: 3 projections with distinct points on black background."""
    print(f"Creating projections ({grid_size}x{grid_size})...")
    box_size = data['box_size']
    half = box_size / 2

    # XY projection (axes 0, 1)
    xy_plus, xy_minus = create_projection(data['pos'], data['signs'], box_size, 0, 1, grid_size)
    # XZ projection (axes 0, 2)
    xz_plus, xz_minus = create_projection(data['pos'], data['signs'], box_size, 0, 2, grid_size)
    # YZ projection (axes 1, 2)
    yz_plus, yz_minus = create_projection(data['pos'], data['signs'], box_size, 1, 2, grid_size)

    # Create figure with white background
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

    projections = [
        (xy_plus, xy_minus, 'XY', 'X [Mpc]', 'Y [Mpc]'),
        (xz_plus, xz_minus, 'XZ', 'X [Mpc]', 'Z [Mpc]'),
        (yz_plus, yz_minus, 'YZ', 'Y [Mpc]', 'Z [Mpc]'),
    ]

    for ax, (gp, gm, title, xlabel, ylabel) in zip(axes, projections):
        ax.set_facecolor('black')

        # Create RGB - high contrast density on black
        rgb = np.zeros((grid_size, grid_size, 3))

        # Strong log scaling for contrast
        gp_log = np.log1p(gp * 2)
        gm_log = np.log1p(gm * 2)

        # Percentile normalization for contrast
        p99_p = np.percentile(gp_log, 99) if gp_log.max() > 0 else 1
        p99_m = np.percentile(gm_log, 99) if gm_log.max() > 0 else 1
        p10_p = np.percentile(gp_log, 10)
        p10_m = np.percentile(gm_log, 10)

        gp_norm = np.clip((gp_log - p10_p) / (p99_p - p10_p + 1e-10), 0, 1)
        gm_norm = np.clip((gm_log - p10_m) / (p99_m - p10_m + 1e-10), 0, 1)

        # Gamma correction for contrast
        gp_norm = np.power(gp_norm, 1.5)
        gm_norm = np.power(gm_norm, 1.5)

        # Pure blue for +, pure red for -, black background
        rgb[:, :, 2] = gp_norm  # Blue
        rgb[:, :, 0] = gm_norm  # Red
        # No green - pure blue/red separation
        rgb = np.clip(rgb, 0, 1)

        ax.imshow(rgb, origin='lower', extent=[-half, half, -half, half])
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=9)

    # Title
    n_millions = data['n'] / 1e6
    title = f"Janus {n_millions:.0f}M — Step {data['step']:,} — z = {data['z']:.2f} — "
    title += f"Seg = {data['seg']:.4f}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python render_pub.py <input.bin> [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.bin', '_pub.png')

    print(f"Loading {input_path}...")
    data = load_render_data(input_path)
    print(f"  Step: {data['step']}, z={data['z']:.2f}, N={data['n']:,}")
    render_pub(data, output_path)
