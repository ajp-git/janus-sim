#!/usr/bin/env python3
"""
render_7panel.py - Janus 7-panel visualization
Format matching scan μ frames for production runs

Panels:
- Top left: XY projection (m+ blue, m- red)
- Middle left: XZ projection
- Bottom left: velocity distribution v+ and v-
- Center: 3D scatter (subsample ~50k)
- Top right: purity map 2D (-1 to +1)
- Middle right: |purity| map (0 to 1)
- Bottom right: log density map

Usage: python render_7panel.py <snap.bin> [output.png]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import struct
import sys
import os

# Configuration
MU = 64
TOTAL_STEPS = 2000
N_CELLS = 32
GRID_RES = 256
SAMPLE_3D = 50000

# Colors
BG_COLOR = '#0a0a15'
TEXT_COLOR = '#ccccdd'
PLUS_COLOR = '#4db8ff'
MINUS_COLOR = '#ff5533'

def read_snapshot(path):
    """Read binary snapshot (25 bytes per particle)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]

        pos = np.zeros((n, 3), dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            pos[i] = struct.unpack('<fff', f.read(12))
            vel[i] = struct.unpack('<fff', f.read(12))
            signs[i] = struct.unpack('<b', f.read(1))[0]

    return n, box, step, z, pos, vel, signs


def compute_metrics(pos, vel, signs, box, n_cells=N_CELLS):
    """Compute P, void_frac, wall_frac, velocities"""
    cell_size = box / n_cells
    half_box = box / 2.0

    n_plus_grid = np.zeros((n_cells, n_cells, n_cells), dtype=np.int32)
    n_minus_grid = np.zeros((n_cells, n_cells, n_cells), dtype=np.int32)

    ix = (((pos[:, 0] + half_box) % box) / cell_size).astype(int).clip(0, n_cells-1)
    iy = (((pos[:, 1] + half_box) % box) / cell_size).astype(int).clip(0, n_cells-1)
    iz = (((pos[:, 2] + half_box) % box) / cell_size).astype(int).clip(0, n_cells-1)

    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus_grid[ix[i], iy[i], iz[i]] += 1
        else:
            n_minus_grid[ix[i], iy[i], iz[i]] += 1

    total = n_plus_grid + n_minus_grid
    purity = np.zeros_like(total, dtype=float)
    mask = total > 0
    purity[mask] = (n_plus_grid[mask] - n_minus_grid[mask]) / total[mask]

    # P = weighted average of |purity|
    P = np.sum(np.abs(purity) * total) / np.sum(total) if np.sum(total) > 0 else 0

    # Void and wall fractions
    occupied = total > 0
    n_occupied = np.sum(occupied)
    frac_minus = np.zeros_like(total, dtype=float)
    frac_plus = np.zeros_like(total, dtype=float)
    frac_minus[occupied] = n_minus_grid[occupied] / total[occupied]
    frac_plus[occupied] = n_plus_grid[occupied] / total[occupied]

    void_frac = np.sum(frac_minus[occupied] > 0.90) / n_occupied if n_occupied > 0 else 0
    wall_frac = np.sum(frac_plus[occupied] > 0.90) / n_occupied if n_occupied > 0 else 0

    # Velocities
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
    n_plus = np.sum(signs > 0)
    n_minus = np.sum(signs < 0)
    v_plus = np.mean(v_mag[signs > 0]) if n_plus > 0 else 0
    v_minus = np.mean(v_mag[signs < 0]) if n_minus > 0 else 0

    return P, void_frac, wall_frac, v_plus, v_minus, purity, n_plus_grid, n_minus_grid


def project_density(pos, box, grid_res, ax0=0, ax1=1, sigma=1.5):
    """2D density projection"""
    half = box / 2
    a = ((pos[:, ax0] + half) / box * grid_res).astype(int).clip(0, grid_res-1)
    b = ((pos[:, ax1] + half) / box * grid_res).astype(int).clip(0, grid_res-1)
    grid = np.zeros((grid_res, grid_res))
    np.add.at(grid, (b, a), 1)
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma, mode='wrap')
    return grid


def render_frame(snap_path, output_path=None):
    """Render 7-panel frame"""
    print(f"Loading {snap_path}...")
    n, box, step, z, pos, vel, signs = read_snapshot(snap_path)

    pos_plus = pos[signs > 0]
    pos_minus = pos[signs < 0]
    n_plus = len(pos_plus)
    n_minus = len(pos_minus)

    print(f"  N={n:,}, N+={n_plus:,}, N-={n_minus:,}, z={z:.3f}")

    P, void_frac, wall_frac, v_plus, v_minus, purity_3d, n_plus_grid, n_minus_grid = \
        compute_metrics(pos, vel, signs, box)

    print(f"  P={P:.3f}, void={void_frac*100:.1f}%, wall={wall_frac*100:.1f}%")

    # Create figure
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100, facecolor=BG_COLOR)

    # Header
    header = f'Janus μ={MU}  |  λ₀=0  |  z={z:.3f}  |  P={P:.3f}  |  void={void_frac*100:.1f}%  |  wall={wall_frac*100:.1f}%  |  step {step}/{TOTAL_STEPS}'
    fig.suptitle(header, fontsize=24, fontweight='bold', color='white', y=0.98)

    half_box = box / 2
    extent = [-half_box, half_box, -half_box, half_box]

    # === LEFT COLUMN: Projections ===

    # Panel 1: XY projection
    ax1 = fig.add_axes([0.01, 0.67, 0.18, 0.28])
    proj_plus_xy = project_density(pos_plus, box, GRID_RES, 0, 1)
    proj_minus_xy = project_density(pos_minus, box, GRID_RES, 0, 1)
    ax1.imshow(proj_plus_xy, origin='lower', cmap='Blues',
               norm=LogNorm(vmin=1, vmax=proj_plus_xy.max()+1), extent=extent, alpha=0.7)
    ax1.imshow(proj_minus_xy, origin='lower', cmap='Reds',
               norm=LogNorm(vmin=1, vmax=proj_minus_xy.max()+1), extent=extent, alpha=0.5)
    ax1.set_title('X-Y Projection', color=TEXT_COLOR, fontsize=11)
    ax1.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax1.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax1.set_facecolor(BG_COLOR)

    # Panel 2: XZ projection
    ax2 = fig.add_axes([0.01, 0.36, 0.18, 0.28])
    proj_plus_xz = project_density(pos_plus, box, GRID_RES, 0, 2)
    proj_minus_xz = project_density(pos_minus, box, GRID_RES, 0, 2)
    ax2.imshow(proj_plus_xz, origin='lower', cmap='Blues',
               norm=LogNorm(vmin=1, vmax=proj_plus_xz.max()+1), extent=extent, alpha=0.7)
    ax2.imshow(proj_minus_xz, origin='lower', cmap='Reds',
               norm=LogNorm(vmin=1, vmax=proj_minus_xz.max()+1), extent=extent, alpha=0.5)
    ax2.set_title('X-Z Projection', color=TEXT_COLOR, fontsize=11)
    ax2.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax2.set_ylabel('Z [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax2.set_facecolor(BG_COLOR)

    # Panel 3: Velocity distribution
    ax3 = fig.add_axes([0.01, 0.05, 0.18, 0.28])
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
    v_plus_arr = v_mag[signs > 0]
    v_minus_arr = v_mag[signs < 0]
    if len(v_plus_arr) > 0 and len(v_minus_arr) > 0:
        v_max = np.percentile(v_mag, 99)
        bins = np.linspace(0, v_max, 50)
        ax3.hist(v_plus_arr, bins=bins, alpha=0.6, color=PLUS_COLOR,
                 label=f'm+ <v>={v_plus:.0f}', density=True)
        ax3.hist(v_minus_arr, bins=bins, alpha=0.6, color=MINUS_COLOR,
                 label=f'm- <v>={v_minus:.0f}', density=True)
        ax3.legend(fontsize=8, facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
    ax3.set_xlabel('|v| [Mpc/Gyr]', color=TEXT_COLOR, fontsize=9)
    ax3.set_ylabel('Density', color=TEXT_COLOR, fontsize=9)
    ax3.set_title('Velocity Distribution', color=TEXT_COLOR, fontsize=11)
    ax3.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax3.set_facecolor(BG_COLOR)

    # === CENTER: 3D scatter ===
    ax_3d = fig.add_axes([0.21, 0.05, 0.40, 0.88], projection='3d')
    ax_3d.set_facecolor(BG_COLOR)

    # Subsample for 3D
    n_sample_plus = min(SAMPLE_3D // 2, len(pos_plus))
    n_sample_minus = min(SAMPLE_3D // 2, len(pos_minus))
    idx_p = np.random.choice(len(pos_plus), n_sample_plus, replace=False) if len(pos_plus) > n_sample_plus else np.arange(len(pos_plus))
    idx_m = np.random.choice(len(pos_minus), n_sample_minus, replace=False) if len(pos_minus) > n_sample_minus else np.arange(len(pos_minus))

    ax_3d.scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 1], pos_plus[idx_p, 2],
                  s=0.5, alpha=0.4, c=PLUS_COLOR, rasterized=True)
    ax_3d.scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 1], pos_minus[idx_m, 2],
                  s=0.5, alpha=0.4, c=MINUS_COLOR, rasterized=True)

    # Box edges
    corners = half_box * np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                    [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    for e in edges:
        ax_3d.plot3D(*zip(corners[e[0]], corners[e[1]]), 'w-', alpha=0.3, linewidth=0.5)

    ax_3d.set_xlim(-half_box, half_box)
    ax_3d.set_ylim(-half_box, half_box)
    ax_3d.set_zlim(-half_box, half_box)
    ax_3d.view_init(elev=25, azim=35)
    ax_3d.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax_3d.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax_3d.set_zlabel('Z [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax_3d.set_title(f'3D View (N+={n_plus:,}, N-={n_minus:,})', color=TEXT_COLOR, fontsize=12)
    ax_3d.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False

    # === RIGHT COLUMN: Purity and density maps ===

    # Panel 5: Purity map (signed)
    ax5 = fig.add_axes([0.63, 0.67, 0.16, 0.28])
    purity_2d = np.mean(purity_3d, axis=2)  # Project along z
    im5 = ax5.imshow(purity_2d.T, origin='lower', cmap='RdBu',
                     norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
                     extent=extent)
    ax5.set_title('Purity (m+/m-)', color=TEXT_COLOR, fontsize=11)
    ax5.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax5.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax5.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax5.set_facecolor(BG_COLOR)
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

    # Panel 6: |Purity| map
    ax6 = fig.add_axes([0.63, 0.36, 0.16, 0.28])
    im6 = ax6.imshow(np.abs(purity_2d.T), origin='lower', cmap='hot',
                     vmin=0, vmax=1, extent=extent)
    ax6.set_title('|Purity|', color=TEXT_COLOR, fontsize=11)
    ax6.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax6.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax6.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax6.set_facecolor(BG_COLOR)
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

    # Panel 7: Log density map
    ax7 = fig.add_axes([0.63, 0.05, 0.16, 0.28])
    total_2d = np.sum(n_plus_grid + n_minus_grid, axis=2)
    im7 = ax7.imshow(total_2d.T + 1, origin='lower', cmap='viridis',
                     norm=LogNorm(), extent=extent)
    ax7.set_title('Log Density (total)', color=TEXT_COLOR, fontsize=11)
    ax7.set_xlabel('X [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax7.set_ylabel('Y [Mpc]', color=TEXT_COLOR, fontsize=9)
    ax7.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax7.set_facecolor(BG_COLOR)
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cbar7.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

    # === SIDEBAR: Stats ===
    sidebar_text = f"""
μ = {MU}
N+ = {n_plus:,}
N- = {n_minus:,}

z = {z:.3f}
P = {P:.3f}

void = {void_frac*100:.1f}%
wall = {wall_frac*100:.1f}%

<v+> = {v_plus:.1f}
<v-> = {v_minus:.1f}
"""
    fig.text(0.82, 0.5, sidebar_text, fontsize=14, color=TEXT_COLOR,
             family='monospace', va='center', ha='left',
             bbox=dict(boxstyle='round', facecolor=BG_COLOR, edgecolor='#333'))

    # Save
    if output_path is None:
        output_path = snap_path.replace('.bin', '.png')

    plt.savefig(output_path, dpi=100, facecolor=BG_COLOR, edgecolor='none',
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_7panel.py <snap.bin> [output.png]")
        sys.exit(1)

    snap_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    render_frame(snap_path, output_path)
