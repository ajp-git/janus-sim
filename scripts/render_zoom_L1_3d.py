#!/usr/bin/env python3
"""
JANUS Zoom-L1 — 2.5D Renderer with ALL particles
3840×2160, rotating view
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
import os
from pathlib import Path
import pandas as pd

OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
FRAME_DIR = f"{OUTPUT_DIR}/frames_3d"
CSV_PATH = f"{OUTPUT_DIR}/time_series.csv"

R_VIEW = 20.0  # Mpc view radius
M_PART_HR = 5.1e10


def read_snapshot(path):
    """Read JSNP snapshot"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int8)
        signs[signs > 1] = -1
    return pos, vel, signs, a, t


def render_3d_frame(snap_path, step, total_frames=18500):
    """Render 2.5D frame with all particles"""

    pos, vel, signs, a, t = read_snapshot(snap_path)
    z = 1/a - 1

    # Masks
    plus_mask = signs > 0
    minus_mask = signs < 0

    pos_plus = pos[plus_mask]
    pos_minus = pos[minus_mask]

    # Filter to view radius
    r_plus = np.sqrt(pos_plus[:, 0]**2 + pos_plus[:, 1]**2 + pos_plus[:, 2]**2)
    r_minus = np.sqrt(pos_minus[:, 0]**2 + pos_minus[:, 1]**2 + pos_minus[:, 2]**2) if len(pos_minus) > 0 else np.array([])

    in_view_plus = r_plus < R_VIEW
    in_view_minus = r_minus < R_VIEW if len(r_minus) > 0 else np.array([], dtype=bool)

    pos_plus_view = pos_plus[in_view_plus]
    pos_minus_view = pos_minus[in_view_minus] if len(pos_minus) > 0 else np.array([]).reshape(0, 3)

    # Get star count from CSV
    try:
        df = pd.read_csv(CSV_PATH)
        row = df[df['step'] <= step].iloc[-1]
        n_stars = int(row['N_stars_HR'])
        rho_hr = row['rho_max_HR']
    except:
        n_stars = 0
        rho_hr = 0

    # Identify "stars" as slowest particles in HR zone (r < 8 Mpc)
    vel_plus = vel[plus_mask]
    v_mag = np.sqrt(vel_plus[:, 0]**2 + vel_plus[:, 1]**2 + vel_plus[:, 2]**2)
    hr_mask = r_plus < 8.0

    star_pos = None
    if n_stars > 0 and np.sum(hr_mask) > n_stars:
        v_hr = v_mag[hr_mask]
        pos_hr = pos_plus[hr_mask]
        star_idx = np.argsort(v_hr)[:n_stars]
        star_pos = pos_hr[star_idx]

    # Rotation angle
    azim = 45 + (step / total_frames) * 360

    # Create figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Plot m+ particles (red, small, transparent)
    if len(pos_plus_view) > 0:
        # Subsample if too many
        n_plot = min(50000, len(pos_plus_view))
        idx = np.random.choice(len(pos_plus_view), n_plot, replace=False) if len(pos_plus_view) > n_plot else np.arange(len(pos_plus_view))
        ax.scatter(pos_plus_view[idx, 0], pos_plus_view[idx, 1], pos_plus_view[idx, 2],
                   c='red', alpha=0.08, s=0.3, rasterized=True)

    # Plot m- particles (blue, if any)
    if len(pos_minus_view) > 0:
        ax.scatter(pos_minus_view[:, 0], pos_minus_view[:, 1], pos_minus_view[:, 2],
                   c='blue', alpha=0.3, s=2)

    # Plot stars (white, prominent)
    if star_pos is not None and len(star_pos) > 0:
        ax.scatter(star_pos[:, 0], star_pos[:, 1], star_pos[:, 2],
                   c='white', alpha=0.95, s=8, edgecolors='yellow', linewidths=0.3)

    # View settings
    ax.view_init(elev=25, azim=azim)
    ax.set_xlim(-R_VIEW, R_VIEW)
    ax.set_ylim(-R_VIEW, R_VIEW)
    ax.set_zlim(-R_VIEW, R_VIEW)

    # Style
    ax.set_xlabel('X [Mpc]', color='white', fontsize=10)
    ax.set_ylabel('Y [Mpc]', color='white', fontsize=10)
    ax.set_zlabel('Z [Mpc]', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)

    # Remove panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    # Title
    title = f"JANUS Zoom-L1 | z={z:.3f} | t={t:.2f} Gyr | N★={n_stars} | ρ_HR={rho_hr:.0f}"
    ax.set_title(title, color='white', fontsize=14, pad=20)

    plt.tight_layout()
    return fig


def render_worker(snap_path):
    """Worker for parallel rendering"""
    step = int(Path(snap_path).stem.split('_')[1])
    out_path = f"{FRAME_DIR}/frame_{step:05d}.png"
    if os.path.exists(out_path):
        return f"Skip {step}"
    try:
        fig = render_3d_frame(snap_path, step)
        fig.savefig(out_path, facecolor='black', edgecolor='none', dpi=200)
        plt.close(fig)
        return f"Done {step}"
    except Exception as e:
        return f"Error {step}: {e}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--step', type=int)
    args = parser.parse_args()

    os.makedirs(FRAME_DIR, exist_ok=True)
    snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))

    if args.test or args.step:
        if args.step:
            snap_path = Path(SNAP_DIR) / f'snap_{args.step:05d}.bin'
        else:
            snap_path = snaps[-1]
        step = int(snap_path.stem.split('_')[1])
        print(f"Rendering 3D step {step}...")
        fig = render_3d_frame(str(snap_path), step)
        out_path = f"{FRAME_DIR}/frame_{step:05d}.png"
        fig.savefig(out_path, facecolor='black', edgecolor='none')
        plt.close(fig)
        print(f"Saved: {out_path}")

    elif args.all:
        from multiprocessing import Pool, cpu_count
        snap_paths = [str(s) for s in snaps]
        print(f"Rendering {len(snap_paths)} 3D frames with {cpu_count()} processes...")
        with Pool(cpu_count()) as p:
            for i, result in enumerate(p.imap_unordered(render_worker, snap_paths)):
                if i % 50 == 0:
                    print(f"[{i}/{len(snap_paths)}] {result}")
        print("Done!")


if __name__ == '__main__':
    main()
