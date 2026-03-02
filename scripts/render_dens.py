#!/usr/bin/env python3
"""
FORMAT C — Densité colormap
2 panneaux : ρ+ (colormap bleu) | ρ- (colormap rouge)
histogram2d sur grille 1024×1024, normalisation log(1+ρ)
Panneaux fond noir, fond global blanc
"""

import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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


def create_blue_cmap():
    """Blue colormap: black -> blue -> cyan -> white."""
    colors = ['#000000', '#000033', '#0033aa', '#0066ff', '#33aaff', '#99ddff', '#ffffff']
    return LinearSegmentedColormap.from_list('blue_density', colors, N=256)


def create_red_cmap():
    """Red colormap: black -> red -> orange -> white."""
    colors = ['#000000', '#330000', '#aa3300', '#ff3300', '#ff6633', '#ffaa66', '#ffffff']
    return LinearSegmentedColormap.from_list('red_density', colors, N=256)


def render_dens(data, output_path, grid_size=1024):
    """Render density colormap format."""
    print(f"Creating density histograms ({grid_size}x{grid_size})...")
    box_size = data['box_size']
    half = box_size / 2
    positions = data['pos']
    signs = data['signs']

    # Separate positive and negative
    mask_pos = signs > 0
    mask_neg = signs < 0

    pos_plus = positions[mask_pos]
    pos_minus = positions[mask_neg]

    # Create 2D histograms (XY projection)
    bins = np.linspace(-half, half, grid_size + 1)

    hist_plus, _, _ = np.histogram2d(pos_plus[:, 0], pos_plus[:, 1], bins=[bins, bins])
    hist_minus, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 1], bins=[bins, bins])

    # Log normalization
    dens_plus = np.log1p(hist_plus.T)
    dens_minus = np.log1p(hist_minus.T)

    # Normalize to [0, 1]
    if dens_plus.max() > 0:
        dens_plus = dens_plus / dens_plus.max()
    if dens_minus.max() > 0:
        dens_minus = dens_minus / dens_minus.max()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

    # Colormaps
    cmap_blue = create_blue_cmap()
    cmap_red = create_red_cmap()

    # Left: ρ+ (blue)
    ax = axes[0]
    ax.set_facecolor('black')
    im_plus = ax.imshow(dens_plus, origin='lower', extent=[-half, half, -half, half],
                        cmap=cmap_blue, vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel('X [Mpc]', fontsize=12)
    ax.set_ylabel('Y [Mpc]', fontsize=12)
    ax.set_title(f'ρ+ (N = {mask_pos.sum():,})', fontsize=14, fontweight='bold', color='#0066ff')
    ax.tick_params(labelsize=10)
    cbar_plus = plt.colorbar(im_plus, ax=ax, shrink=0.8, pad=0.02)
    cbar_plus.set_label('log(1+ρ) normalized', fontsize=10)

    # Right: ρ- (red)
    ax = axes[1]
    ax.set_facecolor('black')
    im_minus = ax.imshow(dens_minus, origin='lower', extent=[-half, half, -half, half],
                         cmap=cmap_red, vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel('X [Mpc]', fontsize=12)
    ax.set_ylabel('Y [Mpc]', fontsize=12)
    ax.set_title(f'ρ− (N = {mask_neg.sum():,})', fontsize=14, fontweight='bold', color='#ff3300')
    ax.tick_params(labelsize=10)
    cbar_minus = plt.colorbar(im_minus, ax=ax, shrink=0.8, pad=0.02)
    cbar_minus.set_label('log(1+ρ) normalized', fontsize=10)

    # Title
    n_millions = data['n'] / 1e6
    title = f"Janus {n_millions:.0f}M — Step {data['step']:,} — z = {data['z']:.2f} — "
    title += f"Seg = {data['seg']:.4f}"
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python render_dens.py <input.bin> [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.bin', '_dens.png')

    print(f"Loading {input_path}...")
    data = load_render_data(input_path)
    print(f"  Step: {data['step']}, z={data['z']:.2f}, N={data['n']:,}")
    render_dens(data, output_path)
