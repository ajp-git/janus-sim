#!/usr/bin/env python3
"""
Auto-render frames for petit_pure_20m_treepm_v3 — V4 style (6 panels)
Watches snapshot directory and renders new frames as they appear.

Snapshot format: 16-byte header + N×25 bytes
  Header: u32 N, f32 box, u32 step, f32 z
  Particle: f32 x,y,z, f32 vx,vy,vz, i8 sign
"""

import os, sys, time, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from pathlib import Path

# Config
SNAP_DIR = Path("/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3/snapshots")
FRAME_DIR = Path("/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3/frames")
GRID_RES = 512
SMOOTH_SIGMA = 1.8
BOX_SIZE = 500.0

# Colors (V4 style)
BG = '#06060f'
C_TEXT = '#aaaacc'
C_TITLE = '#ffffff'
C_PLUS = '#4db8ff'
C_MINUS = '#ff5533'

def make_cmap(r, g, b):
    return LinearSegmentedColormap.from_list('', [
        BG,
        (r*0.25, g*0.25, b*0.25),
        (r*0.55, g*0.55, b*0.55),
        (r, g, b),
        (1.0, 1.0, 1.0),
    ])

CMAP_PLUS = make_cmap(0.30, 0.72, 1.00)
CMAP_MINUS = make_cmap(1.00, 0.33, 0.20)

PROJECTIONS = [
    (0, 1, 'X', 'Y', 'XY'),
    (0, 2, 'X', 'Z', 'XZ'),
    (1, 2, 'Y', 'Z', 'YZ'),
]

def read_snapshot_fast(path, n_sample=2_000_000):
    """Read snapshot with sampling for faster rendering"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]

    particle_size = 25
    header_size = 16
    n_read = min(n, n_sample)

    # Random sampling
    if n > n_sample:
        indices = np.random.choice(n, n_read, replace=False)
        indices.sort()
    else:
        indices = np.arange(n)

    pos = np.zeros((n_read, 3), dtype=np.float32)
    signs = np.zeros(n_read, dtype=np.int8)

    with open(path, 'rb') as f:
        for i, idx in enumerate(indices):
            f.seek(header_size + idx * particle_size)
            pos[i, 0] = struct.unpack('<f', f.read(4))[0]
            pos[i, 1] = struct.unpack('<f', f.read(4))[0]
            pos[i, 2] = struct.unpack('<f', f.read(4))[0]
            f.read(12)  # skip velocities
            signs[i] = struct.unpack('<b', f.read(1))[0]

    return n, box, step, z, pos, signs

def density_map(pos, box, res, sigma, ax0=0, ax1=1):
    """2D density projection"""
    half_box = box / 2
    a = (pos[:, ax0] + half_box) % box
    b = (pos[:, ax1] + half_box) % box

    ix = np.clip((a / box * res).astype(int), 0, res - 1)
    iy = np.clip((b / box * res).astype(int), 0, res - 1)

    grid = np.zeros((res, res), dtype=np.float64)
    np.add.at(grid, (iy, ix), 1)

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma, mode='wrap')

    return grid

def render_frame(snap_path, out_path):
    """Render a single frame — V4 style (6 panels)"""
    print(f"  Reading {snap_path.name}...")
    n, box, step, z, pos, signs = read_snapshot_fast(snap_path)

    pos_plus = pos[signs > 0]
    pos_minus = pos[signs < 0]

    print(f"    step={step}, z={z:.2f}, N+={len(pos_plus):,}, N-={len(pos_minus):,}")

    # Compute 6 density maps
    print("    Computing grids...")
    grids = {}
    for ax0, ax1, lx, ly, name in PROJECTIONS:
        grids[('p', name)] = density_map(pos_plus, box, GRID_RES, SMOOTH_SIGMA, ax0, ax1)
        grids[('m', name)] = density_map(pos_minus, box, GRID_RES, SMOOTH_SIGMA, ax0, ax1)

    # Common scale
    vmin = 0.5
    vmax = max(g.max() for g in grids.values())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ext = [-box/2, box/2, -box/2, box/2]

    # Create 4K figure
    DPI = 200
    fig = plt.figure(figsize=(3840/DPI, 2160/DPI), dpi=DPI, facecolor=BG)

    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.10, 0.45, 0.45],
        hspace=0.08, wspace=0.05,
        left=0.05, right=0.95,
        top=0.97, bottom=0.04
    )

    # Header
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(BG)
    ax_h.axis('off')

    ax_h.text(0.5, 0.75,
              'JANUS COSMOLOGICAL SIMULATION — TreePM v3 (erfc splitting)',
              ha='center', va='center', color=C_TITLE,
              fontsize=17, fontweight='bold', fontfamily='monospace',
              transform=ax_h.transAxes)

    info = f'z = {z:.3f}    ·    Step {step:05d}    ·    N = {n/1e6:.1f}M    ·    Box = {box:.0f} Mpc    ·    μ = 8'
    ax_h.text(0.5, 0.20, info,
              ha='center', va='center', color=C_TEXT,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)

    # Row labels
    ax_h.text(0.01, -0.35, f'm+   N₊ = {len(pos_plus):,}',
              ha='left', va='center', color=C_PLUS,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.01, -1.55, f'm−   N₋ = {len(pos_minus):,}',
              ha='left', va='center', color=C_MINUS,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)

    # 6 panels
    for row, (pop, cmap, color) in enumerate([('p', CMAP_PLUS, C_PLUS),
                                               ('m', CMAP_MINUS, C_MINUS)]):
        for col, (ax0, ax1, lx, ly, name) in enumerate(PROJECTIONS):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.set_facecolor(BG)

            g = np.clip(grids[(pop, name)], vmin, vmax)
            ax.imshow(g, origin='lower', cmap=cmap,
                      norm=norm, interpolation='gaussian', extent=ext)

            # Axes
            ax.set_xlim(-box/2, box/2)
            ax.set_ylim(-box/2, box/2)
            ax.set_aspect('equal')

            # Labels
            if row == 1:
                ax.set_xlabel(f'{lx} [Mpc]', color=C_TEXT, fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f'{ly} [Mpc]', color=C_TEXT, fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(colors=C_TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#333355')

            # Panel title
            ax.text(0.98, 0.96, name, ha='right', va='top',
                    color=color, fontsize=10, fontweight='bold',
                    transform=ax.transAxes)

    plt.savefig(out_path, dpi=DPI, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"    Saved {out_path.name}")

def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  Auto-render V4 style for petit_pure_20m_treepm_v3")
    print("="*60)
    print(f"  Watching: {SNAP_DIR}")
    print(f"  Output:   {FRAME_DIR}")
    print("="*60)

    rendered = set()

    while True:
        # Find new snapshots
        if SNAP_DIR.exists():
            snaps = sorted(SNAP_DIR.glob("snap_*.bin"))

            for snap in snaps:
                if snap.name in rendered:
                    continue

                # Check if file is complete (> 100 bytes)
                if snap.stat().st_size < 100:
                    continue

                # Extract step number
                step = int(snap.stem.split('_')[1])
                out_path = FRAME_DIR / f"frame_{step:05d}.png"

                if out_path.exists():
                    rendered.add(snap.name)
                    continue

                try:
                    render_frame(snap, out_path)
                    rendered.add(snap.name)
                except Exception as e:
                    print(f"    Error rendering {snap.name}: {e}")

        time.sleep(5)

if __name__ == "__main__":
    main()
