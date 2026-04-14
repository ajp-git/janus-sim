#!/usr/bin/env python3
"""
Batch render all snapshots with publication layout for video.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import struct
from pathlib import Path
import pandas as pd
import os
from multiprocessing import Pool
import sys

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
CSV_PATH = f"{OUTPUT_DIR}/time_series.csv"
FRAME_DIR = f"{OUTPUT_DIR}/frames_publication"

R_HR = 8.0
R_VIEW = 2.5
EPSILON = 0.03
M_PART_HR = 5.1e10

os.makedirs(FRAME_DIR, exist_ok=True)

# Load CSV once
DF = pd.read_csv(CSV_PATH)


def read_snapshot(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int8)
        signs[signs > 1] = -1
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    is_plus = signs > 0
    is_hr = r < R_HR
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
    return {'n': n, 'a': a, 't': t, 'pos': pos, 'vel': vel, 'signs': signs,
            'is_plus': is_plus, 'is_hr': is_hr, 'r': r, 'v_mag': v_mag}


def compute_local_density(pos, k=8):
    tree = cKDTree(pos)
    distances, _ = tree.query(pos, k=k+1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    return 1.0 / (mean_dist**3 + 1e-10)


def compute_radial_profile(pos, signs, n_bins=40):
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    is_plus = signs > 0
    r_bins = np.logspace(-2, np.log10(R_HR * 1.2), n_bins + 1)
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])
    rho_plus = np.zeros(n_bins)
    rho_minus = np.zeros(n_bins)
    for i in range(n_bins):
        r_in, r_out = r_bins[i], r_bins[i + 1]
        shell_mask = (r >= r_in) & (r < r_out)
        vol = 4/3 * np.pi * (r_out**3 - r_in**3)
        rho_plus[i] = np.sum(shell_mask & is_plus) * M_PART_HR / vol if vol > 0 else 0
        rho_minus[i] = np.sum(shell_mask & ~is_plus) * M_PART_HR / vol if vol > 0 else 0
    return r_mid, rho_plus, rho_minus


def render_frame(snap_path):
    step = int(Path(snap_path).stem.split('_')[1])
    out_path = f"{FRAME_DIR}/frame_{step:05d}.png"

    if os.path.exists(out_path):
        return f"[SKIP] {step}"

    try:
        data = read_snapshot(snap_path)
        z = 1 / data['a'] - 1
        t_gyr = data['t']
        pos = data['pos']
        signs = data['signs']
        is_plus = data['is_plus']
        is_hr = data['is_hr']
        r = data['r']
        v_mag = data['v_mag']

        # Get star count
        df_step = DF[DF['step'] <= step]
        n_stars = int(df_step['N_stars_HR'].iloc[-1]) if len(df_step) > 0 else 0

        # Identify stars
        hr_plus_mask = is_plus & is_hr
        star_pos = None
        if n_stars > 0 and np.sum(hr_plus_mask) > n_stars:
            v_hr = v_mag[hr_plus_mask]
            pos_hr_all = pos[hr_plus_mask]
            star_idx = np.argsort(v_hr)[:n_stars]
            star_pos = pos_hr_all[star_idx]

        # Radial profile
        r_mid, rho_plus, rho_minus = compute_radial_profile(pos[is_hr], signs[is_hr])

        # Particles for 2.5D
        in_view_mask = is_plus & is_hr & (r < R_VIEW)
        pos_view = pos[in_view_mask]
        r_view = r[in_view_mask]

        # Outer halo
        mask_outer = (r_view >= 1.0) & (r_view < R_VIEW)
        pos_outer = pos_view[mask_outer]
        if len(pos_outer) > 40000:
            pos_outer = pos_outer[np.random.choice(len(pos_outer), 40000, replace=False)]

        # Core
        mask_core = r_view < 1.0
        pos_core = pos_view[mask_core]
        if len(pos_core) > 30000:
            pos_core = pos_core[np.random.choice(len(pos_core), 30000, replace=False)]

        # Core colors
        if len(pos_core) > 10:
            density_core = compute_local_density(pos_core, k=8)
            log_dens = np.log10(density_core + 1e-10)
            vmin, vmax = np.percentile(log_dens, [5, 98])
            norm_dens = np.clip((log_dens - vmin) / (vmax - vmin + 1e-10), 0, 1)
            colors_core = plt.cm.hot(norm_dens)
        else:
            colors_core = 'orange'

        # Stars in view
        stars_in_view = None
        if star_pos is not None:
            r_stars = np.sqrt(star_pos[:, 0]**2 + star_pos[:, 1]**2 + star_pos[:, 2]**2)
            stars_in_view = star_pos[r_stars < R_VIEW]

        central_star = None
        if stars_in_view is not None and len(stars_in_view) > 0:
            r_sv = np.sqrt(stars_in_view[:, 0]**2 + stars_in_view[:, 1]**2 + stars_in_view[:, 2]**2)
            central_star = stars_in_view[np.argmin(r_sv)]

        # ═══════════════════════════════════════════════════════════════════
        # CREATE FIGURE
        # ═══════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(38.4, 21.6), dpi=100, facecolor='black')

        ax3d = fig.add_axes([0.00, 0.02, 0.62, 0.94], projection='3d')
        ax_rho = fig.add_axes([0.66, 0.54, 0.32, 0.38])
        ax_nst = fig.add_axes([0.66, 0.08, 0.32, 0.38])

        # 3D setup
        ax3d.set_facecolor('black')
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax3d.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax3d.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax3d.grid(False)

        # Plot outer halo
        if len(pos_outer) > 0:
            ax3d.scatter(pos_outer[:, 0], pos_outer[:, 1], pos_outer[:, 2],
                        c='darkred', s=1.5, alpha=0.2, zorder=1, rasterized=True)

        # Plot core
        if len(pos_core) > 0:
            ax3d.scatter(pos_core[:, 0], pos_core[:, 1], pos_core[:, 2],
                        c=colors_core, s=4.0, alpha=0.7, zorder=2, rasterized=True)

        # Stars
        if stars_in_view is not None and len(stars_in_view) > 0:
            ax3d.scatter(stars_in_view[:, 0], stars_in_view[:, 1], stars_in_view[:, 2],
                        c='white', s=25, alpha=1.0, zorder=10)

        if central_star is not None:
            ax3d.scatter([central_star[0]], [central_star[1]], [central_star[2]],
                        c='yellow', s=120, alpha=1.0, zorder=11,
                        edgecolors='white', linewidths=0.5)

        ax3d.view_init(elev=32, azim=225)
        ax3d.set_xlim(-R_VIEW, R_VIEW)
        ax3d.set_ylim(-R_VIEW, R_VIEW)
        ax3d.set_zlim(-1.8, 3.0)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_xlabel('X [Mpc]', color='white', fontsize=13, labelpad=12)
        ax3d.set_ylabel('Y [Mpc]', color='white', fontsize=13, labelpad=12)
        ax3d.set_zlabel('Z [Mpc]', color='white', fontsize=13, labelpad=12)
        ax3d.tick_params(colors='white', labelsize=10)
        ax3d.xaxis.line.set_color('#555555')
        ax3d.yaxis.line.set_color('#555555')
        ax3d.zaxis.line.set_color('#555555')

        # Radial profile
        ax_rho.set_facecolor('#0a0a0a')
        for spine in ax_rho.spines.values():
            spine.set_edgecolor('#444444')
        ax_rho.loglog(r_mid, rho_plus + 1e3, color='#ff4444', lw=2.0, label='ρ⁺(r)')
        ax_rho.loglog(r_mid, rho_minus + 1e3, color='#4488ff', lw=2.0, label='ρ⁻(r)')
        ax_rho.axvline(EPSILON, color='gray', ls=':', lw=1.0, alpha=0.6)
        ax_rho.axvline(R_HR, color='white', ls='--', lw=1.0, alpha=0.4)
        ax_rho.set_xlabel('r [Mpc]', color='white', fontsize=12)
        ax_rho.set_ylabel('ρ [M☉/Mpc³]', color='white', fontsize=12)
        ax_rho.set_title('Radial Density Profile', color='white', fontsize=13, pad=8)
        ax_rho.tick_params(colors='white', labelsize=10)
        ax_rho.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
        ax_rho.grid(True, alpha=0.15, color='gray')
        ax_rho.set_xlim(0.01, 10)
        ax_rho.set_ylim(1e3, 1e15)
        ax_rho.annotate('m⁻ absent\nr < 0.3 Mpc', xy=(0.08, 1e8), xytext=(0.2, 1e6),
                       color='#6699ff', fontsize=11, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='#6699ff', lw=1.5))

        # N★(z)
        ax_nst.set_facecolor('#0a0a0a')
        for spine in ax_nst.spines.values():
            spine.set_edgecolor('#444444')
        if len(df_step) > 1:
            ax_nst.plot(df_step['z'], df_step['N_stars_HR'], color='#ffaa00', lw=2.0)
            ax_nst.scatter([z], [n_stars], c='red', s=80, zorder=10, edgecolors='white', linewidths=1)
        ax_nst.set_xlabel('z (redshift)', color='white', fontsize=12)
        ax_nst.set_ylabel('N★ (star count)', color='white', fontsize=12)
        ax_nst.set_title('Star Formation History', color='white', fontsize=13, pad=8)
        ax_nst.tick_params(colors='white', labelsize=10)
        ax_nst.grid(True, alpha=0.15, color='gray')
        ax_nst.set_xlim(0.5, 0)
        ax_nst.set_ylim(0, 2500)

        # Title
        M_stars = n_stars * M_PART_HR
        fig.text(0.5, 0.97,
                f"JANUS Zoom-L1 Baryonic | z={z:.3f} | t={t_gyr:.2f} Gyr | "
                f"N★={n_stars} | M★≈{M_stars:.1e} M☉",
                ha='center', va='top', fontsize=18, color='white', fontweight='bold')
        fig.text(0.5, 0.945,
                "High-resolution region (8 Mpc) — Complete m⁻ segregation — Proto-stellar formation",
                ha='center', va='top', fontsize=13, color='#aaaaaa', style='italic')

        plt.savefig(out_path, dpi=100, facecolor='black')
        plt.close(fig)

        return f"[OK] {step}"
    except Exception as e:
        return f"[ERR] {step}: {e}"


def main():
    snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))
    print(f"Found {len(snaps)} snapshots")
    print(f"Output: {FRAME_DIR}")
    sys.stdout.flush()

    with Pool(4) as pool:
        for i, result in enumerate(pool.imap(render_frame, [str(s) for s in snaps])):
            if i % 20 == 0:
                print(f"[{i}/{len(snaps)}] {result}")
                sys.stdout.flush()

    print("\nDone!")


if __name__ == '__main__':
    main()
