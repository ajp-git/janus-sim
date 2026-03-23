#!/usr/bin/env python3 -u
"""
Render Janus simulation snapshots to 4K video frames in real-time.

Usage:
  python3 render_frames.py --snap-dir output/run/snapshots --output-dir output/run/frames --box 500 --watch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
import time
import sys

# Cosmology for redshift calculation
def z_from_step(step, n_steps=3000, z_start=5.0, z_end=0.0):
    """Approximate redshift from step number."""
    frac = step / n_steps
    # Linear interpolation in scale factor
    a_start = 1.0 / (1 + z_start)
    a_end = 1.0 / (1 + z_end)
    a = a_start + frac * (a_end - a_start)
    return 1.0 / a - 1.0

def cosmic_time_gyr(z):
    """Approximate cosmic time in Gyr (Janus cosmology)."""
    # Simplified: t ≈ 13.8 * (1 - (1+z)^(-1.5)) for matter-dominated
    H0 = 70  # km/s/Mpc
    t_H = 978 / H0  # Gyr
    return t_H * (1 - (1 + z)**(-1.5)) * 0.67

def load_snapshot(path, box_size):
    """Load binary snapshot: header u32 + n×(3×f32 pos + 3×f32 vel + i8 sign)."""
    with open(path, 'rb') as f:
        # Read header
        n = np.frombuffer(f.read(4), dtype='<u4')[0]

        # Read particle data: 25 bytes per particle
        # pos(12) + vel(12) + sign(1) = 25 bytes
        raw = f.read(n * 25)

    # Parse structured data
    dt = np.dtype([('pos', '<f4', 3), ('vel', '<f4', 3), ('sign', 'i1')])
    particles = np.frombuffer(raw, dtype=dt)

    return particles

def compute_metrics(particles, box_size):
    """Compute segregation S and COM separation."""
    pos = particles['pos']
    sign = particles['sign']

    mask_p = sign > 0
    mask_m = sign < 0

    n_pos = np.sum(mask_p)
    n_neg = np.sum(mask_m)

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    # COM with periodic wrapping (minimum image)
    def periodic_com(positions, box):
        # Use angle method for periodic COM
        theta = 2 * np.pi * positions / box
        cos_mean = np.mean(np.cos(theta), axis=0)
        sin_mean = np.mean(np.sin(theta), axis=0)
        com = box * np.arctan2(sin_mean, cos_mean) / (2 * np.pi)
        return com

    com_p = periodic_com(pos[mask_p], box_size)
    com_m = periodic_com(pos[mask_m], box_size)

    # Minimum image distance
    delta = com_p - com_m
    delta = delta - box_size * np.round(delta / box_size)
    dcom = np.linalg.norm(delta)

    # Segregation (simplified)
    # S = (mean distance to opposite) / (mean distance to same) - 1
    # Approximate with COM distance / (box/4)
    S = dcom / (box_size / 4)
    S = min(S, 1.0)

    return S, dcom

def render_frame(snap_path, output_path, box_size, step, n_steps=3000):
    """Render a single 4K frame with 3-panel layout."""

    particles = load_snapshot(snap_path, box_size)
    pos = particles['pos']
    sign = particles['sign']

    # Positions are in [-box/2, box/2], shift to [0, box]
    pos = pos + box_size / 2

    mask_p = sign > 0
    mask_m = sign < 0

    z = z_from_step(step, n_steps)
    t_gyr = cosmic_time_gyr(z)
    S, dcom = compute_metrics(particles, box_size)

    # Subsample for plotting (max 500k points per panel)
    max_points = 500000

    def subsample(mask):
        indices = np.where(mask)[0]
        if len(indices) > max_points:
            indices = np.random.choice(indices, max_points, replace=False)
        return indices

    idx_p = subsample(mask_p)
    idx_m = subsample(mask_m)

    # Project to 2D (use X-Y plane, integrate Z)
    # For density, bin into 2D histogram
    nbins = 800

    hist_p, xe, ye = np.histogram2d(pos[idx_p, 0], pos[idx_p, 1],
                                     bins=nbins, range=[[0, box_size], [0, box_size]])
    hist_m, _, _ = np.histogram2d(pos[idx_m, 0], pos[idx_m, 1],
                                   bins=nbins, range=[[0, box_size], [0, box_size]])

    # Log density
    hist_p = np.log10(hist_p + 1)
    hist_m = np.log10(hist_m + 1)
    hist_c = hist_p + hist_m

    # Create figure: 3840×2160 at 200 dpi
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor='black')

    # GridSpec: 3 columns for panels, bottom row for info bar
    gs = GridSpec(2, 3, figure=fig, height_ratios=[5, 1],
                  hspace=0.05, wspace=0.02,
                  left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Colormaps
    cmap_blue = plt.cm.Blues
    cmap_red = plt.cm.Reds
    cmap_combined = plt.cm.magma

    vmax_p = np.percentile(hist_p, 99.5)
    vmax_m = np.percentile(hist_m, 99.5)
    vmax_c = np.percentile(hist_c, 99.5)

    # Panel 1: m+ (Blue)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(hist_p.T, origin='lower', extent=[0, box_size, 0, box_size],
               cmap=cmap_blue, vmin=0, vmax=vmax_p, aspect='equal')
    ax1.set_title('m+ (positive mass)', color='cyan', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_facecolor('black')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    # Panel 2: Combined
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hist_c.T, origin='lower', extent=[0, box_size, 0, box_size],
               cmap=cmap_combined, vmin=0, vmax=vmax_c, aspect='equal')
    ax2.set_title('Combined density', color='white', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_facecolor('black')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # Panel 3: m- (Red)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(hist_m.T, origin='lower', extent=[0, box_size, 0, box_size],
               cmap=cmap_red, vmin=0, vmax=vmax_m, aspect='equal')
    ax3.set_title('m- (negative mass)', color='orange', fontsize=14, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_facecolor('black')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    # Info bar at bottom
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.set_facecolor('black')
    ax_info.axis('off')

    # Progress bar
    progress = step / n_steps
    bar_y = 0.7
    bar_height = 0.15
    # Background
    ax_info.add_patch(plt.Rectangle((0.05, bar_y), 0.9, bar_height,
                                      facecolor='#333333', edgecolor='gray'))
    # Progress
    ax_info.add_patch(plt.Rectangle((0.05, bar_y), 0.9 * progress, bar_height,
                                      facecolor='#00aa00', edgecolor=None))

    # Labels
    ax_info.text(0.02, bar_y + bar_height/2, f'z={z_from_step(0, n_steps):.1f}',
                 color='white', fontsize=12, va='center', ha='left')
    ax_info.text(0.98, bar_y + bar_height/2, f'z={z_from_step(n_steps, n_steps):.1f}',
                 color='white', fontsize=12, va='center', ha='right')

    # Metrics text
    info_text = f'Redshift: {z:.2f}  |  S = {S:.3f}  |  ΔCOM = {dcom:.1f} Mpc  |  t = {t_gyr:.2f} Gyr  |  Step {step}/{n_steps}'
    ax_info.text(0.5, 0.25, info_text, color='white', fontsize=14,
                 va='center', ha='center', fontweight='bold')

    # Title
    ax_info.text(0.5, 0.95, 'JANUS COSMOLOGICAL MODEL — Bimetric Matter Segregation',
                 color='yellow', fontsize=16, va='top', ha='center', fontweight='bold')

    # Save
    plt.savefig(output_path, dpi=200, facecolor='black',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return z, S, dcom

def main():
    parser = argparse.ArgumentParser(description='Render Janus snapshots to 4K frames')
    parser.add_argument('--snap-dir', required=True, help='Directory containing snapshots')
    parser.add_argument('--output-dir', required=True, help='Output directory for frames')
    parser.add_argument('--box', type=float, default=500.0, help='Box size in Mpc')
    parser.add_argument('--resolution', nargs=2, type=int, default=[3840, 2160], help='Resolution')
    parser.add_argument('--watch', action='store_true', help='Watch mode: wait for new snapshots')
    parser.add_argument('--n-steps', type=int, default=3000, help='Total simulation steps')
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    if not snap_dir.exists():
        snap_dir = snap_dir / 'snapshots'
    if not snap_dir.exists():
        # Create and wait
        snap_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = set()

    # Find already rendered frames
    for f in output_dir.glob('frame_*.png'):
        try:
            step = int(f.stem.split('_')[1])
            rendered.add(step)
        except:
            pass

    print(f"[RENDER] Watching {snap_dir}")
    print(f"[RENDER] Output: {output_dir}")
    print(f"[RENDER] Already rendered: {len(rendered)} frames")
    print(f"[RENDER] Resolution: {args.resolution[0]}x{args.resolution[1]}")

    last_report = time.time()

    while True:
        # Find all snapshots
        snaps = sorted(snap_dir.glob('snap_*.bin'))

        for snap_path in snaps:
            try:
                step = int(snap_path.stem.split('_')[1])
            except:
                continue

            if step in rendered:
                continue

            # Check file is complete (not still being written)
            size1 = snap_path.stat().st_size
            time.sleep(0.5)
            size2 = snap_path.stat().st_size
            if size1 != size2:
                continue  # Still writing

            output_path = output_dir / f'frame_{step:06d}.png'

            try:
                t0 = time.time()
                z, S, dcom = render_frame(snap_path, output_path, args.box, step, args.n_steps)
                dt = time.time() - t0
                print(f"[RENDER] Step {step:5d} -> frame_{step:06d}.png  z={z:.2f} S={S:.3f} ({dt:.1f}s)")
                rendered.add(step)
            except Exception as e:
                print(f"[RENDER] ERROR on step {step}: {e}")

        if not args.watch:
            break

        # Report progress
        if time.time() - last_report > 60:
            print(f"[RENDER] {len(rendered)} frames rendered, watching...")
            last_report = time.time()

        time.sleep(2)

    print(f"[RENDER] Done. Total frames: {len(rendered)}")

if __name__ == '__main__':
    main()
