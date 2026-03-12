#!/usr/bin/env python3
"""
render_frame.py v6 — Frame 4K Janus : 6 panneaux XY/XZ/YZ × ρ+/ρ−
Usage: python3 render_frame.py <snap.bin> [out.png] [z] [seg] [ke] [t_gyr]
"""

import sys, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

BOX_MPC      = 492.0
GRID_RES     = 768
SMOOTH_SIGMA = 1.4

BG      = '#06060f'
C_TEXT  = '#aaaacc'
C_TITLE = '#ffffff'
C_GRID  = '#1a1a2e'
C_PLUS  = '#4db8ff'
C_MINUS = '#ff5533'

def read_snapshot(path):
    with open(path, 'rb') as f:
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        data = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)
    sign  = data[:, 3]
    pos_p = data[sign >  0, :3]
    pos_m = data[sign <  0, :3]
    return n, step, pos_p, pos_m

def density_map(pos, box, res, sigma, ax0=0, ax1=1):
    """Projection 2D avec wrapping périodique correct."""
    half = box / 2.0

    # Wrapping périodique correct (modulo)
    a = (pos[:, ax0] + half) % box - half
    b = (pos[:, ax1] + half) % box - half

    xi = np.clip(((a + half) / box * res).astype(int), 0, res - 1)
    yi = np.clip(((b + half) / box * res).astype(int), 0, res - 1)

    grid = np.zeros((res, res), dtype=np.float64)
    np.add.at(grid, (yi, xi), 1)

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma, mode='constant', cval=0)

    return grid

def make_cmap(r, g, b):
    return LinearSegmentedColormap.from_list('', [
        BG,
        (r*0.25, g*0.25, b*0.25),
        (r*0.55, g*0.55, b*0.55),
        (r,      g,      b     ),
        (1.0,    1.0,    1.0   ),
    ])

CMAP_PLUS  = make_cmap(0.30, 0.72, 1.00)
CMAP_MINUS = make_cmap(1.00, 0.33, 0.20)

PROJECTIONS = [
    (0, 1, 'X', 'Y', 'XY'),
    (0, 2, 'X', 'Z', 'XZ'),
    (1, 2, 'Y', 'Z', 'YZ'),
]

def render(snap_path, out_path, z_cosmo=None, seg=None, ke=None, t_gyr=None):

    print(f"Lecture {snap_path}...")
    n, step, pos_p, pos_m = read_snapshot(snap_path)
    print(f"  step={step}  N+={len(pos_p):,}  N-={len(pos_m):,}")

    # Précalculer les 6 grilles
    print("Calcul grilles...")
    grids = {}
    for ax0, ax1, lx, ly, name in PROJECTIONS:
        grids[('p', name)] = density_map(pos_p, BOX_MPC, GRID_RES, SMOOTH_SIGMA, ax0, ax1)
        grids[('m', name)] = density_map(pos_m, BOX_MPC, GRID_RES, SMOOTH_SIGMA, ax0, ax1)

    # Échelle log commune à tous les panneaux
    vmin = 0.1
    vmax = max(g.max() for g in grids.values())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ext  = [-BOX_MPC/2, BOX_MPC/2, -BOX_MPC/2, BOX_MPC/2]

    # ── Figure 4K ─────────────────────────────────────────────────────────────
    DPI = 200
    fig = plt.figure(figsize=(3840/DPI, 2160/DPI), dpi=DPI, facecolor=BG)

    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.10, 0.45, 0.45],
        hspace=0.08, wspace=0.05,
        left=0.05, right=0.95,
        top=0.97, bottom=0.04
    )

    # ── Header ────────────────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(BG); ax_h.axis('off')

    parts = []
    if z_cosmo is not None: parts.append(f'z = {z_cosmo:.3f}')
    if seg     is not None: parts.append(f'Seg = {seg:.4f}')
    if ke      is not None: parts.append(f'KE/KE₀ = {ke:.3f}')
    parts.append(f'Step {step:06d}')
    parts.append(f'N = {n/1e6:.2f}M  │  Box = {BOX_MPC:.0f} Mpc  │  η = 1.045')

    ax_h.text(0.5, 0.75,
              'JANUS COSMOLOGICAL SIMULATION  —  N-Body Gravitational Segregation',
              ha='center', va='center', color=C_TITLE,
              fontsize=17, fontweight='bold', fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.5, 0.20,
              '    ·    '.join(parts),
              ha='center', va='center', color=C_TEXT,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)

    # Labels lignes
    ax_h.text(0.01, -0.35, f'Masses +   N₊ = {len(pos_p):,}',
              ha='left', va='center', color=C_PLUS,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.01, -1.55, f'Masses −   N₋ = {len(pos_m):,}',
              ha='left', va='center', color=C_MINUS,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)

    # ── 6 panneaux ────────────────────────────────────────────────────────────
    last_im = None
    for row, (pop, cmap, color) in enumerate([('p', CMAP_PLUS,  C_PLUS),
                                               ('m', CMAP_MINUS, C_MINUS)]):
        for col, (ax0, ax1, lx, ly, name) in enumerate(PROJECTIONS):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.set_facecolor(BG)

            g = np.clip(grids[(pop, name)], vmin, vmax)
            im = ax.imshow(g, origin='lower', cmap=cmap,
                           norm=norm, interpolation='gaussian', extent=ext)
            last_im = (im, cmap, color)

            ax.set_title(f'Projection {name}',
                         color=color if row == 0 else C_TEXT,
                         fontsize=9, pad=4, fontfamily='monospace')
            ax.set_xlabel(f'{lx} (Mpc)', color=C_TEXT, fontsize=8)
            ax.set_ylabel(f'{ly} (Mpc)', color=C_TEXT, fontsize=8)
            ax.tick_params(colors=C_TEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(C_GRID)

            # Colorbar seulement sur le panneau de droite
            if col == 2:
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                cb.set_label('Density', color=C_TEXT, fontsize=7)
                cb.ax.yaxis.set_tick_params(color=C_TEXT, labelsize=6)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_TEXT)
                cb.outline.set_edgecolor(C_GRID)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"Sauvegarde {out_path}...")
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"Done → {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 render_frame.py <snap.bin> [out.png] [z] [seg] [ke] [t_gyr]")
        sys.exit(1)
    snap  = sys.argv[1]
    out   = sys.argv[2] if len(sys.argv) > 2 else snap.replace('.bin', '.png')
    z_c   = float(sys.argv[3]) if len(sys.argv) > 3 else None
    seg   = float(sys.argv[4]) if len(sys.argv) > 4 else None
    ke    = float(sys.argv[5]) if len(sys.argv) > 5 else None
    t_gyr = float(sys.argv[6]) if len(sys.argv) > 6 else None
    render(snap, out, z_c, seg, ke, t_gyr)
