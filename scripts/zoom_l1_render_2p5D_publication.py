#!/usr/bin/env python3
"""
JANUS Zoom-L1 — Publication-quality 2.5D + panels (4K)
Layout: 2.5D view (left) + radial profile + N★(z) (right column)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import struct
from pathlib import Path
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
CSV_PATH = f"{OUTPUT_DIR}/time_series.csv"

R_HR = 8.0          # HR region radius [Mpc]
R_VIEW = 2.5        # View radius for 2.5D [Mpc]
EPSILON = 0.03      # Softening [Mpc]
M_PART_HR = 5.1e10  # HR particle mass [M_sun]


def read_snapshot(path):
    """Read JSNP v1 snapshot"""
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

    return {
        'n': n, 'a': a, 't': t,
        'pos': pos, 'vel': vel, 'signs': signs,
        'is_plus': is_plus, 'is_hr': is_hr,
        'r': r, 'v_mag': v_mag
    }


def compute_local_density(pos, k=8):
    """Compute local density using k-nearest neighbors"""
    tree = cKDTree(pos)
    distances, _ = tree.query(pos, k=k+1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    density = 1.0 / (mean_dist**3 + 1e-10)
    return density


def compute_radial_profile(pos, signs, n_bins=40):
    """Compute radial density profiles for m+ and m-"""
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    is_plus = signs > 0

    r_bins = np.logspace(-2, np.log10(R_HR * 1.2), n_bins + 1)
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])  # geometric mean

    rho_plus = np.zeros(n_bins)
    rho_minus = np.zeros(n_bins)

    for i in range(n_bins):
        r_in, r_out = r_bins[i], r_bins[i + 1]
        shell_mask = (r >= r_in) & (r < r_out)
        vol = 4/3 * np.pi * (r_out**3 - r_in**3)

        n_plus = np.sum(shell_mask & is_plus)
        n_minus = np.sum(shell_mask & ~is_plus)

        rho_plus[i] = n_plus * M_PART_HR / vol if vol > 0 else 0
        rho_minus[i] = n_minus * M_PART_HR / vol if vol > 0 else 0

    return r_mid, rho_plus, rho_minus


def main():
    # Find latest snapshot
    snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))
    if not snaps:
        print("No snapshots found")
        return

    snap_path = snaps[-1]
    step = int(snap_path.stem.split('_')[1])
    print(f"Reading {snap_path}")

    # Load data
    data = read_snapshot(str(snap_path))
    z = 1 / data['a'] - 1
    t_gyr = data['t']

    pos = data['pos']
    vel = data['vel']
    signs = data['signs']
    is_plus = data['is_plus']
    is_hr = data['is_hr']
    r = data['r']
    v_mag = data['v_mag']

    # Load CSV for time series
    try:
        df = pd.read_csv(CSV_PATH)
        df_final = df[df['step'] <= step]
        n_stars = int(df_final['N_stars_HR'].iloc[-1]) if len(df_final) > 0 else 0
    except Exception as e:
        print(f"CSV error: {e}")
        df = None
        df_final = None
        n_stars = 0

    # Identify stars as slowest HR m+ particles
    hr_plus_mask = is_plus & is_hr
    star_pos = None
    if n_stars > 0 and np.sum(hr_plus_mask) > n_stars:
        v_hr = v_mag[hr_plus_mask]
        pos_hr_all = pos[hr_plus_mask]
        star_idx = np.argsort(v_hr)[:n_stars]
        star_pos = pos_hr_all[star_idx]

    # ═══════════════════════════════════════════════════════════════════════
    # PREPARE DATA FOR PLOTS
    # ═══════════════════════════════════════════════════════════════════════

    # Radial profile
    print("Computing radial profiles...")
    r_mid, rho_plus, rho_minus = compute_radial_profile(pos[is_hr], signs[is_hr])

    # Particles for 2.5D view
    in_view_mask = is_plus & is_hr & (r < R_VIEW)
    pos_view = pos[in_view_mask]
    r_view = r[in_view_mask]

    print(f"Particles in view (r < {R_VIEW} Mpc): {len(pos_view)}")

    # Layer 1 — Outer halo (1-2.5 Mpc)
    mask_outer = (r_view >= 1.0) & (r_view < R_VIEW)
    pos_outer = pos_view[mask_outer]
    n_outer_max = 40000
    if len(pos_outer) > n_outer_max:
        idx = np.random.choice(len(pos_outer), n_outer_max, replace=False)
        pos_outer = pos_outer[idx]
    print(f"Outer halo: {len(pos_outer)} particles")

    # Layer 2 — Dense core (r < 1 Mpc)
    mask_core = r_view < 1.0
    pos_core = pos_view[mask_core]
    n_core_max = 30000
    if len(pos_core) > n_core_max:
        idx = np.random.choice(len(pos_core), n_core_max, replace=False)
        pos_core = pos_core[idx]
    print(f"Dense core: {len(pos_core)} particles")

    # Compute local density for core — log-scale normalization
    print("Computing local densities...")
    density_core = compute_local_density(pos_core, k=8)

    print(f"DEBUG density_core: min={density_core.min():.3e}, "
          f"max={density_core.max():.3e}, "
          f"p10={np.percentile(density_core,10):.3e}, "
          f"p99={np.percentile(density_core,99):.3e}, "
          f"N={len(density_core)}")

    # Log-scale normalization for better gradient
    log_dens = np.log10(density_core + 1e-10)
    vmin = np.percentile(log_dens, 5)
    vmax = np.percentile(log_dens, 98)
    norm_dens = np.clip((log_dens - vmin) / (vmax - vmin), 0, 1)
    colors_core = plt.cm.hot(norm_dens)

    print(f"DEBUG log_dens: vmin={vmin:.2f}, vmax={vmax:.2f}, "
          f"norm range=[{norm_dens.min():.3f}, {norm_dens.max():.3f}]")

    # Stars in view
    stars_in_view = None
    if star_pos is not None:
        r_stars = np.sqrt(star_pos[:, 0]**2 + star_pos[:, 1]**2 + star_pos[:, 2]**2)
        stars_in_view = star_pos[r_stars < R_VIEW]
        print(f"Stars in view: {len(stars_in_view)}")

    # Central star
    central_star = None
    if stars_in_view is not None and len(stars_in_view) > 0:
        r_stars_view = np.sqrt(stars_in_view[:, 0]**2 + stars_in_view[:, 1]**2 + stars_in_view[:, 2]**2)
        central_idx = np.argmin(r_stars_view)
        central_star = stars_in_view[central_idx]

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE FIGURE
    # ═══════════════════════════════════════════════════════════════════════

    figsize = (38.4, 21.6)  # 4K 16:9
    dpi = 100

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='black')

    # Utiliser add_axes() pour contrôle précis (GridSpec ne fonctionne pas avec 3D)
    ax3d = fig.add_axes([0.00, 0.02, 0.62, 0.94], projection='3d')
    ax_rho = fig.add_axes([0.66, 0.54, 0.32, 0.38])
    ax_nst = fig.add_axes([0.66, 0.08, 0.32, 0.38])

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL LEFT — 2.5D VIEW
    # ═══════════════════════════════════════════════════════════════════════

    ax3d.set_facecolor('black')
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax3d.yaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax3d.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax3d.grid(False)

    # Layer 1 — Outer halo
    ax3d.scatter(
        pos_outer[:, 0], pos_outer[:, 1], pos_outer[:, 2],
        c='darkred', s=1.5, alpha=0.2, zorder=1, rasterized=True
    )

    # Layer 2 — Dense core (pre-computed colors with log normalization)
    ax3d.scatter(
        pos_core[:, 0], pos_core[:, 1], pos_core[:, 2],
        c=colors_core, s=4.0, alpha=0.7, zorder=2, rasterized=True
    )

    # Stars
    if stars_in_view is not None and len(stars_in_view) > 0:
        ax3d.scatter(
            stars_in_view[:, 0], stars_in_view[:, 1], stars_in_view[:, 2],
            c='white', s=25, alpha=1.0, zorder=10
        )

    # Central star
    if central_star is not None:
        ax3d.scatter(
            [central_star[0]], [central_star[1]], [central_star[2]],
            c='yellow', s=120, alpha=1.0, zorder=11,
            edgecolors='white', linewidths=0.5
        )

    ax3d.view_init(elev=32, azim=225)
    ax3d.set_xlim(-R_VIEW, R_VIEW)
    ax3d.set_ylim(-R_VIEW, R_VIEW)
    ax3d.set_zlim(-1.8, 3.0)  # asymétrique → remonte l'objet
    ax3d.set_box_aspect([1, 1, 1])

    ax3d.set_xlabel('X [Mpc]', color='white', fontsize=13, labelpad=12)
    ax3d.set_ylabel('Y [Mpc]', color='white', fontsize=13, labelpad=12)
    ax3d.set_zlabel('Z [Mpc]', color='white', fontsize=13, labelpad=12)
    ax3d.tick_params(colors='white', labelsize=10)
    ax3d.xaxis.line.set_color('#555555')
    ax3d.yaxis.line.set_color('#555555')
    ax3d.zaxis.line.set_color('#555555')

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL TOP RIGHT — Radial Density Profile
    # ═══════════════════════════════════════════════════════════════════════

    ax_rho.set_facecolor('#0a0a0a')
    for spine in ax_rho.spines.values():
        spine.set_edgecolor('#444444')

    # Add small offset to avoid log(0)
    rho_plus_plot = rho_plus + 1e3
    rho_minus_plot = rho_minus + 1e3

    ax_rho.loglog(r_mid, rho_plus_plot, color='#ff4444', lw=2.0, label='ρ⁺(r)')
    ax_rho.loglog(r_mid, rho_minus_plot, color='#4488ff', lw=2.0, label='ρ⁻(r)')

    ax_rho.axvline(EPSILON, color='gray', ls=':', lw=1.0, alpha=0.6, label=f'ε={EPSILON} Mpc')
    ax_rho.axvline(R_HR, color='white', ls='--', lw=1.0, alpha=0.4, label='r_HR=8 Mpc')

    ax_rho.set_xlabel('r [Mpc]', color='white', fontsize=12)
    ax_rho.set_ylabel('ρ [M☉/Mpc³]', color='white', fontsize=12)
    ax_rho.set_title('Radial Density Profile', color='white', fontsize=13, pad=8)
    ax_rho.tick_params(colors='white', labelsize=10)
    ax_rho.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
    ax_rho.grid(True, alpha=0.15, color='gray')
    ax_rho.set_xlim(0.01, 10)
    ax_rho.set_ylim(1e3, 1e15)

    # Annotation for m⁻ absence
    ax_rho.annotate('m⁻ absent\nr < 0.3 Mpc',
                    xy=(0.08, 1e8),
                    xytext=(0.2, 1e6),
                    color='#6699ff', fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#6699ff', lw=1.5))

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL BOTTOM RIGHT — N★(z)
    # ═══════════════════════════════════════════════════════════════════════

    ax_nst.set_facecolor('#0a0a0a')
    for spine in ax_nst.spines.values():
        spine.set_edgecolor('#444444')

    if df_final is not None and len(df_final) > 1:
        ax_nst.plot(df_final['z'], df_final['N_stars_HR'], color='#ffaa00', lw=2.0)
        ax_nst.scatter([z], [n_stars], c='red', s=80, zorder=10, edgecolors='white', linewidths=1)

    ax_nst.set_xlabel('z (redshift)', color='white', fontsize=12)
    ax_nst.set_ylabel('N★ (star count)', color='white', fontsize=12)
    ax_nst.set_title('Star Formation History', color='white', fontsize=13, pad=8)
    ax_nst.tick_params(colors='white', labelsize=10)
    ax_nst.grid(True, alpha=0.15, color='gray')
    ax_nst.set_xlim(0.5, 0)
    ax_nst.set_ylim(0, max(2500, n_stars * 1.2))

    # Annotation current state
    ax_nst.annotate(f'z={z:.3f}\nN★={n_stars}',
                    xy=(z, n_stars),
                    xytext=(z + 0.15, n_stars * 0.7),
                    color='#ffaa00', fontsize=11,
                    arrowprops=dict(arrowstyle='->', color='#ffaa00', lw=1.5))

    # ═══════════════════════════════════════════════════════════════════════
    # GLOBAL TITLE
    # ═══════════════════════════════════════════════════════════════════════

    M_stars = n_stars * M_PART_HR

    fig.text(0.5, 0.97,
             f"JANUS Zoom-L1 Baryonic | z={z:.3f} | t={t_gyr:.2f} Gyr | "
             f"N★={n_stars} | M★≈{M_stars:.1e} M☉",
             ha='center', va='top', fontsize=18, color='white', fontweight='bold')

    fig.text(0.5, 0.945,
             "High-resolution region (8 Mpc) — Complete m⁻ segregation — Proto-stellar formation",
             ha='center', va='top', fontsize=13, color='#aaaaaa', style='italic')

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════

    out_path = f"{OUTPUT_DIR}/render_2p5D_publication.png"
    plt.savefig(out_path, dpi=dpi, facecolor='black')
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
