#!/usr/bin/env python3
"""
Render 2.5D isometric + 2 side panels from render_data binary files.

Layout:
  ┌─────────────────────────┬───────────┐
  │                         │  Masses+  │
  │   Vue isométrique 2.5D  │  (bleu)   │
  │   azimuth=30°, elev=60° ├───────────┤
  │   ALL particles         │  Masses−  │
  │                         │  (rouge)  │
  └─────────────────────────┴───────────┘

Usage:
    python render_25d.py /path/to/step_001800.bin /path/to/output.png
"""

import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Isometric projection angles
AZIMUTH = 30
ELEVATION = 60


def rotation_matrix(azim_deg, elev_deg):
    """Create rotation matrix for isometric projection."""
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)

    Rz = np.array([
        [np.cos(azim), -np.sin(azim), 0],
        [np.sin(azim),  np.cos(azim), 0],
        [0,             0,            1]
    ])

    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(elev), -np.sin(elev)],
        [0, np.sin(elev),  np.cos(elev)]
    ])

    return Rx @ Rz


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
        'step': step,
        'box_size': box_size,
        'seg': seg,
        'ke_ratio': ke_ratio,
        'z': redshift,
        'n': n,
        'pos': positions,
        'signs': signs
    }


def create_density_grids(data, grid_size=1024):
    """Create isometric and XY density grids for + and - particles."""
    R = rotation_matrix(AZIMUTH, ELEVATION)
    box_size = data['box_size']
    positions = data['pos']
    signs = data['signs']
    half_box = box_size / 2

    # Compute projected box extent for isometric
    corners = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ]) * half_box

    corners_rot = corners @ R.T
    x_extent = max(abs(corners_rot[:, 0].min()), abs(corners_rot[:, 0].max())) * 1.02
    y_extent = max(abs(corners_rot[:, 1].min()), abs(corners_rot[:, 1].max())) * 1.02
    extent = max(x_extent, y_extent)
    cell_iso = (2 * extent) / grid_size
    cell_xy = box_size / grid_size

    # Apply rotation for isometric
    rotated = positions @ R.T

    # Convert to grid indices - isometric
    ix_iso = np.clip(((rotated[:, 0] + extent) / cell_iso).astype(np.int32), 0, grid_size - 1)
    iy_iso = np.clip(((rotated[:, 1] + extent) / cell_iso).astype(np.int32), 0, grid_size - 1)

    # Convert to grid indices - XY projection
    ix_xy = np.clip(((positions[:, 0] + half_box) / cell_xy).astype(np.int32), 0, grid_size - 1)
    iy_xy = np.clip(((positions[:, 1] + half_box) / cell_xy).astype(np.int32), 0, grid_size - 1)

    # Separate grids
    iso_plus = np.zeros((grid_size, grid_size), dtype=np.float32)
    iso_minus = np.zeros((grid_size, grid_size), dtype=np.float32)
    xy_plus = np.zeros((grid_size, grid_size), dtype=np.float32)
    xy_minus = np.zeros((grid_size, grid_size), dtype=np.float32)

    mask_pos = signs > 0
    mask_neg = signs < 0

    np.add.at(iso_plus, (iy_iso[mask_pos], ix_iso[mask_pos]), 1)
    np.add.at(iso_minus, (iy_iso[mask_neg], ix_iso[mask_neg]), 1)
    np.add.at(xy_plus, (iy_xy[mask_pos], ix_xy[mask_pos]), 1)
    np.add.at(xy_minus, (iy_xy[mask_neg], ix_xy[mask_neg]), 1)

    return {
        'iso_plus': iso_plus,
        'iso_minus': iso_minus,
        'xy_plus': xy_plus,
        'xy_minus': xy_minus,
        'extent': extent,
        'n_plus': mask_pos.sum(),
        'n_minus': mask_neg.sum()
    }


def process_density(g, sigma=0.8):
    """Apply smoothing, log scaling and normalization."""
    g = gaussian_filter(g.astype(np.float32), sigma=sigma)
    g = np.log1p(g * 30)
    p99 = np.percentile(g, 99.5)
    p05 = np.percentile(g, 5)
    if p99 > p05:
        g = np.clip((g - p05) / (p99 - p05), 0, 1)
    return g


def render_25d(data, output_path, grid_size=1024):
    """Render 2.5D isometric + 2 side panels."""
    print(f"Creating density grids ({grid_size}x{grid_size})...")
    grids = create_density_grids(data, grid_size)

    print("Processing densities...")
    iso_plus = process_density(grids['iso_plus'])
    iso_minus = process_density(grids['iso_minus'])
    xy_plus = process_density(grids['xy_plus'])
    xy_minus = process_density(grids['xy_minus'])
    extent = grids['extent']
    box_size = data['box_size']

    # Create figure with 3-panel layout
    fig = plt.figure(figsize=(20, 12), facecolor='black')

    # Main panel (left 70%)
    ax_main = fig.add_axes([0.02, 0.08, 0.65, 0.84], facecolor='black')
    # Right top: Masses+ (blue)
    ax_plus = fig.add_axes([0.70, 0.52, 0.28, 0.40], facecolor='black')
    # Right bottom: Masses- (red)
    ax_minus = fig.add_axes([0.70, 0.08, 0.28, 0.40], facecolor='black')

    # Colors
    color_plus = np.array([0.2, 0.5, 1.0])   # Blue
    color_minus = np.array([1.0, 0.25, 0.25]) # Red

    # === MAIN PANEL: Isometric combined ===
    rgb_main = np.zeros((grid_size, grid_size, 3))
    rgb_main[:, :, 0] = iso_minus * 0.9  # Red
    rgb_main[:, :, 2] = iso_plus * 0.9   # Blue
    rgb_main[:, :, 1] = np.minimum(iso_plus, iso_minus) * 0.3  # Green for overlap
    rgb_main = np.clip(rgb_main, 0, 1)
    rgb_main = np.power(rgb_main, 1.1)

    ax_main.imshow(rgb_main, origin='lower', aspect='equal',
                   extent=[-extent, extent, -extent, extent])
    ax_main.set_xlim(-extent, extent)
    ax_main.set_ylim(-extent, extent)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_color('white')
        spine.set_linewidth(0.5)

    # Label
    ax_main.text(0.02, 0.98, "Isometric 2.5D", transform=ax_main.transAxes,
                 color='white', fontsize=12, va='top', fontweight='bold')

    # === RIGHT TOP: Masses+ (XY projection) ===
    rgb_plus = np.zeros((grid_size, grid_size, 3))
    rgb_plus[:, :, 0] = xy_plus * color_plus[0]
    rgb_plus[:, :, 1] = xy_plus * color_plus[1]
    rgb_plus[:, :, 2] = xy_plus * color_plus[2]
    rgb_plus = np.clip(np.power(rgb_plus, 1.1), 0, 1)

    half = box_size / 2
    ax_plus.imshow(rgb_plus, origin='lower', aspect='equal',
                   extent=[-half, half, -half, half])
    ax_plus.set_xlim(-half, half)
    ax_plus.set_ylim(-half, half)
    ax_plus.set_xticks([])
    ax_plus.set_yticks([])
    for spine in ax_plus.spines.values():
        spine.set_color(color_plus)
        spine.set_linewidth(2)

    ax_plus.text(0.5, 0.97, f"Masses+ (N={grids['n_plus']:,})", transform=ax_plus.transAxes,
                 color='white', fontsize=11, ha='center', va='top', fontweight='bold')

    # === RIGHT BOTTOM: Masses- (XY projection) ===
    rgb_minus = np.zeros((grid_size, grid_size, 3))
    rgb_minus[:, :, 0] = xy_minus * color_minus[0]
    rgb_minus[:, :, 1] = xy_minus * color_minus[1]
    rgb_minus[:, :, 2] = xy_minus * color_minus[2]
    rgb_minus = np.clip(np.power(rgb_minus, 1.1), 0, 1)

    ax_minus.imshow(rgb_minus, origin='lower', aspect='equal',
                    extent=[-half, half, -half, half])
    ax_minus.set_xlim(-half, half)
    ax_minus.set_ylim(-half, half)
    ax_minus.set_xticks([])
    ax_minus.set_yticks([])
    for spine in ax_minus.spines.values():
        spine.set_color(color_minus)
        spine.set_linewidth(2)

    ax_minus.text(0.5, 0.97, f"Masses− (N={grids['n_minus']:,})", transform=ax_minus.transAxes,
                  color='white', fontsize=11, ha='center', va='top', fontweight='bold')

    # Title with metrics
    n_millions = data['n'] / 1e6
    title = f"Janus {n_millions:.0f}M — Step {data['step']:,} — z = {data['z']:.2f} — "
    title += f"KE/KE₀ = {data['ke_ratio']:.4f} — Seg = {data['seg']:.4f}"
    fig.suptitle(title, fontsize=16, color='white', y=0.97, fontweight='bold')

    # Footer
    fig.text(0.5, 0.02, 'Blue = positive mass | Red = negative mass | Box = {:.0f} Mpc'.format(box_size),
             ha='center', fontsize=11, color='gray')

    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python render_25d.py <input.bin> [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.bin', '_25d.png')

    print(f"Loading {input_path}...")
    data = load_render_data(input_path)
    print(f"  Step: {data['step']}, z={data['z']:.2f}, N={data['n']:,}")
    render_25d(data, output_path)
