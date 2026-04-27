#!/usr/bin/env python3
"""
render_stars.py - Janus visualization with star formation

Renders 4K frames showing:
- m+ (gas) as blue
- m- as red
- Stars (sink particles) as bright white points 3× larger

Usage:
    python render_stars.py <snapshot.bin> [output.png]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import sys
from pathlib import Path

# Import snapshot reader
from snapshot_reader import (
    read_snapshot_fast, get_particle_masks, count_particles,
    TYPE_GAS_PLUS, TYPE_SINK_STAR, TYPE_MASS_MINUS
)

# Configuration
W, H = 3840, 2160  # 4K
DPI = 100
GRID_RES = 512
N_CELLS = 64

# Colors
BG_COLOR = '#0a0a15'
TEXT_COLOR = '#ccccdd'
GAS_PLUS_COLOR = '#4db8ff'    # Blue cyan
MASS_MINUS_COLOR = '#ff5533'  # Orange red
STAR_COLOR = '#ffffff'         # Bright white
STAR_GLOW = '#ffffaa'         # Yellow glow


def project_density(pos, box, grid_res, ax0=0, ax1=1, sigma=1.0):
    """2D density projection with gaussian smoothing"""
    half = box / 2.0
    x = ((pos[:, ax0] + half) / box * grid_res).astype(int)
    y = ((pos[:, ax1] + half) / box * grid_res).astype(int)
    x = np.clip(x, 0, grid_res - 1)
    y = np.clip(y, 0, grid_res - 1)

    grid = np.zeros((grid_res, grid_res))
    np.add.at(grid, (y, x), 1)

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma, mode='wrap')
    return grid


def compute_purity(pos, signs, box, n_cells):
    """Compute purity map P = (n+ - n-) / (n+ + n-)"""
    cell_size = box / n_cells
    half = box / 2.0

    n_plus = np.zeros((n_cells, n_cells), dtype=np.int32)
    n_minus = np.zeros((n_cells, n_cells), dtype=np.int32)

    x = ((pos[:, 0] + half) / box * n_cells).astype(int).clip(0, n_cells - 1)
    y = ((pos[:, 1] + half) / box * n_cells).astype(int).clip(0, n_cells - 1)

    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus[x[i], y[i]] += 1
        else:
            n_minus[x[i], y[i]] += 1

    total = n_plus + n_minus
    purity = np.zeros_like(total, dtype=float)
    mask = total > 0
    purity[mask] = (n_plus[mask] - n_minus[mask]) / total[mask]

    return purity, n_plus, n_minus


def render_frame(snap_path, output_path=None):
    """Render 4K frame with star visualization"""

    # Load snapshot
    print(f"Loading {snap_path}...")
    snap = read_snapshot_fast(str(snap_path))
    counts = count_particles(snap)

    print(f"  N={snap.n:,}, z={snap.z:.3f}, box={snap.box_size} Mpc")
    print(f"  Gas m+: {counts['n_gas_plus']:,}")
    print(f"  Stars: {counts['n_stars']:,}")
    print(f"  m-: {counts['n_mass_minus']:,}")

    # Get masks
    gas_mask, star_mask, minus_mask = get_particle_masks(snap)

    pos_gas = snap.positions[gas_mask]
    pos_stars = snap.positions[star_mask]
    pos_minus = snap.positions[minus_mask]

    box = snap.box_size
    half_box = box / 2.0
    extent = [-half_box, half_box, -half_box, half_box]

    # Create figure
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=BG_COLOR)

    # Header
    header = (f"Janus VSL  |  z={snap.z:.3f}  |  "
              f"N_gas={counts['n_gas_plus']:,}  |  "
              f"N_stars={counts['n_stars']:,}  |  "
              f"N_m-={counts['n_mass_minus']:,}")
    fig.suptitle(header, fontsize=24, fontweight='bold', color='white', y=0.97)

    # === Panel 1: XY projection with stars ===
    ax1 = fig.add_axes([0.02, 0.35, 0.45, 0.58])
    ax1.set_facecolor(BG_COLOR)

    # Density layers
    if len(pos_minus) > 0:
        dens_minus = project_density(pos_minus, box, GRID_RES, 0, 1, sigma=2.0)
        dens_minus = np.log10(dens_minus + 1)
        ax1.imshow(dens_minus, extent=extent, origin='lower', cmap='Reds',
                   alpha=0.7, vmin=0, vmax=dens_minus.max() * 0.8)

    if len(pos_gas) > 0:
        dens_gas = project_density(pos_gas, box, GRID_RES, 0, 1, sigma=1.5)
        dens_gas = np.log10(dens_gas + 1)
        ax1.imshow(dens_gas, extent=extent, origin='lower', cmap='Blues',
                   alpha=0.8, vmin=0, vmax=dens_gas.max() * 0.8)

    # Stars as scatter (3× larger, bright white with glow)
    if len(pos_stars) > 0:
        # Glow layer (larger, semi-transparent yellow)
        ax1.scatter(pos_stars[:, 0], pos_stars[:, 1],
                    s=30, c=STAR_GLOW, alpha=0.3, marker='o', edgecolors='none')
        # Core layer (smaller, bright white)
        ax1.scatter(pos_stars[:, 0], pos_stars[:, 1],
                    s=10, c=STAR_COLOR, alpha=0.9, marker='o', edgecolors='none')

    ax1.set_xlim(-half_box, half_box)
    ax1.set_ylim(-half_box, half_box)
    ax1.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=12)
    ax1.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=12)
    ax1.set_title('XY Projection (Blue=m+gas, Red=m-, White=stars)',
                  color=TEXT_COLOR, fontsize=14)
    ax1.tick_params(colors=TEXT_COLOR)
    for spine in ax1.spines.values():
        spine.set_color(TEXT_COLOR)
    ax1.set_aspect('equal')

    # === Panel 2: XZ projection with stars ===
    ax2 = fig.add_axes([0.52, 0.35, 0.45, 0.58])
    ax2.set_facecolor(BG_COLOR)

    if len(pos_minus) > 0:
        dens_minus_xz = project_density(pos_minus, box, GRID_RES, 0, 2, sigma=2.0)
        dens_minus_xz = np.log10(dens_minus_xz + 1)
        ax2.imshow(dens_minus_xz, extent=extent, origin='lower', cmap='Reds',
                   alpha=0.7, vmin=0, vmax=dens_minus_xz.max() * 0.8)

    if len(pos_gas) > 0:
        dens_gas_xz = project_density(pos_gas, box, GRID_RES, 0, 2, sigma=1.5)
        dens_gas_xz = np.log10(dens_gas_xz + 1)
        ax2.imshow(dens_gas_xz, extent=extent, origin='lower', cmap='Blues',
                   alpha=0.8, vmin=0, vmax=dens_gas_xz.max() * 0.8)

    if len(pos_stars) > 0:
        ax2.scatter(pos_stars[:, 0], pos_stars[:, 2],
                    s=30, c=STAR_GLOW, alpha=0.3, marker='o', edgecolors='none')
        ax2.scatter(pos_stars[:, 0], pos_stars[:, 2],
                    s=10, c=STAR_COLOR, alpha=0.9, marker='o', edgecolors='none')

    ax2.set_xlim(-half_box, half_box)
    ax2.set_ylim(-half_box, half_box)
    ax2.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=12)
    ax2.set_ylabel('Z [Mpc]', color=TEXT_COLOR, fontsize=12)
    ax2.set_title('XZ Projection', color=TEXT_COLOR, fontsize=14)
    ax2.tick_params(colors=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(TEXT_COLOR)
    ax2.set_aspect('equal')

    # === Panel 3: Purity map ===
    ax3 = fig.add_axes([0.02, 0.05, 0.28, 0.25])
    ax3.set_facecolor(BG_COLOR)

    # Combine gas and stars for purity (both are positive mass)
    pos_plus = np.vstack([pos_gas, pos_stars]) if len(pos_stars) > 0 else pos_gas
    signs_plus = np.ones(len(pos_plus), dtype=np.int8)
    signs_minus = -np.ones(len(pos_minus), dtype=np.int8)

    all_pos = np.vstack([pos_plus, pos_minus]) if len(pos_minus) > 0 else pos_plus
    all_signs = np.concatenate([signs_plus, signs_minus]) if len(pos_minus) > 0 else signs_plus

    purity, n_plus, n_minus = compute_purity(all_pos, all_signs, box, N_CELLS)

    im = ax3.imshow(purity.T, extent=extent, origin='lower', cmap='RdBu',
                    vmin=-1, vmax=1)
    ax3.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=10)
    ax3.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=10)
    ax3.set_title('Purity P=(n+-n-)/(n++n-)', color=TEXT_COLOR, fontsize=12)
    ax3.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color(TEXT_COLOR)

    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cbar.set_label('Purity', color=TEXT_COLOR, fontsize=10)

    # === Panel 4: Star histogram ===
    ax4 = fig.add_axes([0.38, 0.05, 0.28, 0.25])
    ax4.set_facecolor(BG_COLOR)

    if len(pos_stars) > 0:
        # Star distribution histogram
        star_r = np.sqrt(pos_stars[:, 0]**2 + pos_stars[:, 1]**2 + pos_stars[:, 2]**2)
        ax4.hist(star_r, bins=50, color=STAR_COLOR, alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Distance from center [Mpc]', color=TEXT_COLOR, fontsize=10)
        ax4.set_ylabel('N stars', color=TEXT_COLOR, fontsize=10)
        ax4.set_title(f'Star radial distribution (N={len(pos_stars):,})',
                      color=TEXT_COLOR, fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No stars formed yet', ha='center', va='center',
                 transform=ax4.transAxes, color=TEXT_COLOR, fontsize=14)
        ax4.set_title('Star radial distribution', color=TEXT_COLOR, fontsize=12)

    ax4.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax4.spines.values():
        spine.set_color(TEXT_COLOR)

    # === Panel 5: Stats ===
    ax5 = fig.add_axes([0.72, 0.05, 0.25, 0.25])
    ax5.set_facecolor(BG_COLOR)
    ax5.axis('off')

    # Compute global P
    total = n_plus + n_minus
    weights = total.flatten()
    purities = np.abs(purity.flatten())
    P_global = np.sum(purities * weights) / np.sum(weights) if np.sum(weights) > 0 else 0

    stats_text = f"""
    Box: {box:.0f} Mpc
    z = {snap.z:.4f}

    N total: {snap.n:,}
    N gas m+: {counts['n_gas_plus']:,}
    N stars: {counts['n_stars']:,}
    N m-: {counts['n_mass_minus']:,}

    Global P: {P_global:.4f}
    Star fraction: {100*counts['n_stars']/max(1,counts['n_gas_plus']+counts['n_stars']):.3f}%
    """

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, color=TEXT_COLOR,
             fontsize=12, verticalalignment='top', family='monospace')

    # Save
    if output_path is None:
        output_path = Path(snap_path).with_suffix('.png')

    plt.savefig(output_path, dpi=DPI, facecolor=BG_COLOR, bbox_inches='tight',
                pad_inches=0.1)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python render_stars.py <snapshot.bin> [output.png]")
        sys.exit(1)

    snap_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    render_frame(snap_path, output_path)
