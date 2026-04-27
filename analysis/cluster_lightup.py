#!/usr/bin/env python3
"""
cluster_lightup.py — 4K Isometric Renderer for Janus Cluster Formation

Reads GBIN v2 binary files produced by cluster_extractor.rs
Renders 4K frames showing m+/m- segregation and proto-star formation
"""

import argparse
import struct
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Video Parameters ===
WIDTH, HEIGHT = 3840, 2160
DPI = 200
FPS = 24

# === Cluster center (relative coords in GBIN are already centered) ===
CENTER = np.array([0.0, 0.0, 0.0])
RADIUS = 25.0

# === Fixed isometric view ===
ELEV, AZIM = 30, 45

# === Color palette ===
COLOR_MINUS = '#0a3d6b'   # Deep blue - m-
COLOR_GAS   = '#3d0a00'   # Very dark red - m+ gas
COLOR_STAR  = 'white'     # Established proto-star
COLOR_NEW   = '#ffff44'   # New star flash
COLOR_HALO  = '#ff8800'   # Orange halo around flash


def read_gbin(path: Path) -> dict:
    """Read GBIN v2 binary file produced by cluster_extractor"""
    with open(path, 'rb') as f:
        # Header 48 bytes
        magic, z, t, step = struct.unpack('<IfII', f.read(16))
        if magic != 0x4742494E:
            raise ValueError(f"Bad magic in {path}: 0x{magic:08X}")

        n_plus, n_minus, n_proto, n_new = struct.unpack('<IIII', f.read(16))
        cx, cy, cz, radius = struct.unpack('<ffff', f.read(16))

        # m+ particles - structured array (20 bytes each)
        dtype_plus = np.dtype([
            ('x',  '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('od', '<f4'),
            ('is_star', 'u1'), ('is_new', 'u1'),
            ('pad', 'u1', 2)
        ])
        plus = np.frombuffer(f.read(n_plus * 20), dtype=dtype_plus).copy()

        # m- particles - xyz only (12 bytes each)
        dtype_minus = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4')
        ])
        minus = np.frombuffer(f.read(n_minus * 12), dtype=dtype_minus).copy()

    return dict(
        z=z, t=t, step=step,
        plus=plus, minus=minus,
        n_proto=n_proto, n_new=n_new,
        center=(cx, cy, cz), radius=radius
    )


def render_frame(args):
    """Render a single frame (for parallel processing)"""
    gbin_path, frame_idx, frames_dir = args

    try:
        data = read_gbin(gbin_path)
    except Exception as e:
        print(f"Error reading {gbin_path}: {e}")
        return None

    plus = data['plus']
    minus = data['minus']
    z, t = data['z'], data['t']

    fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.grid(False)

    # Remove axis panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.view_init(elev=ELEV, azim=AZIM)

    # m- : deep blue transparent - ALL particles
    if len(minus) > 0:
        ax.scatter(minus['x'], minus['y'], minus['z'],
                   c=COLOR_MINUS, alpha=0.10, s=0.3,
                   linewidths=0, rasterized=True)

    # m+ gas : very dark red
    mask_gas = plus['is_star'] == 0
    if mask_gas.sum() > 0:
        ax.scatter(plus['x'][mask_gas],
                   plus['y'][mask_gas],
                   plus['z'][mask_gas],
                   c=COLOR_GAS, alpha=0.12, s=0.3,
                   linewidths=0, rasterized=True)

    # Established proto-stars : white, size proportional to overdensity
    mask_star = (plus['is_star'] == 1) & (plus['is_new'] == 0)
    if mask_star.sum() > 0:
        sizes = np.clip(plus['od'][mask_star] * 0.5, 2, 20)
        ax.scatter(plus['x'][mask_star],
                   plus['y'][mask_star],
                   plus['z'][mask_star],
                   c=COLOR_STAR, alpha=0.80, s=sizes,
                   linewidths=0)

    # NEW proto-stars : yellow flash + orange halo
    mask_new = plus['is_new'] == 1
    if mask_new.sum() > 0:
        # Halo first (behind)
        ax.scatter(plus['x'][mask_new],
                   plus['y'][mask_new],
                   plus['z'][mask_new],
                   c=COLOR_HALO, alpha=0.35, s=200,
                   linewidths=0)
        # Central flash
        ax.scatter(plus['x'][mask_new],
                   plus['y'][mask_new],
                   plus['z'][mask_new],
                   c=COLOR_NEW, alpha=1.0, s=60,
                   edgecolors='white', linewidths=0.5,
                   zorder=10)

    # Title with statistics
    n_s = data['n_proto']
    n_n = data['n_new']
    m_stellar = n_s * 5.1e11  # M_sun per particle

    title_line1 = f"Janus Cluster  |  z = {z:.3f}  |  t = {t:.2f} Gyr"
    title_line2 = f"M* = {m_stellar:.2e} M\u2609  |  +{n_n} new stellar populations"
    ax.set_title(f"{title_line1}\n{title_line2}",
                 color='white', fontsize=15, pad=12, fontfamily='monospace')

    # Discrete axes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('#444444')
        axis.set_tick_params(colors='#333333')
    ax.set_xlabel('X [Mpc]', color='#555555', fontsize=9)
    ax.set_ylabel('Y [Mpc]', color='#555555', fontsize=9)
    ax.set_zlabel('Z [Mpc]', color='#555555', fontsize=9)

    # FIXED limits - same bbox throughout video
    ax.set_xlim(-RADIUS, RADIUS)
    ax.set_ylim(-RADIUS, RADIUS)
    ax.set_zlim(-RADIUS, RADIUS)

    out_path = frames_dir / f'frame_{frame_idx:05d}.png'
    plt.savefig(out_path, dpi=DPI, facecolor='black',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return (frame_idx, z, n_s, n_n)


def main():
    parser = argparse.ArgumentParser(description='Render Janus cluster formation video')
    parser.add_argument('--gbin-dir', required=True, help='Directory with GBIN files')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=cpu_count(), help='Number of parallel workers')
    parser.add_argument('--skip-render', action='store_true', help='Skip rendering, just encode video')
    args = parser.parse_args()

    gbin_dir = Path(args.gbin_dir)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # List GBIN files - sort by step number for chronological order
    # Filename pattern: cluster_z{z:.3f}_s{step:05d}.gbin
    def get_step(path):
        name = path.stem  # cluster_z0.417_s04795
        parts = name.split('_s')
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    files = sorted(gbin_dir.glob('cluster_*.gbin'), key=get_step)
    print(f"Found {len(files)} GBIN files to render (sorted by step)")

    if not args.skip_render:
        # Prepare render jobs
        jobs = [(f, i, frames_dir) for i, f in enumerate(files)]

        # Parallel rendering
        print(f"Rendering with {args.workers} workers...")
        with Pool(args.workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(render_frame, jobs)):
                if result is not None:
                    frame_idx, z, n_s, n_n = result
                    if (i + 1) % 20 == 0 or (i + 1) == len(jobs):
                        print(f"  [{(i+1)/len(jobs)*100:5.1f}%] "
                              f"z={z:.3f} | N*={n_s:,} | +{n_n}")
                results.append(result)

        print(f"\nRendered {len(results)} frames")

    # Assemble video
    video_path = out_dir / 'janus_cluster_2.5D_4K.mp4'
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {FPS} '
        f'-pattern_type glob -i "{frames_dir}/frame_*.png" '
        f'-c:v libx264 -crf 15 -preset slow '
        f'-pix_fmt yuv420p -movflags +faststart '
        f'"{video_path}"'
    )

    print(f"\nEncoding video: {video_path}")
    os.system(ffmpeg_cmd)

    print(f"\nDone! Video: {video_path}")


if __name__ == '__main__':
    main()
