#!/usr/bin/env python3
"""
JANUS Zoom-L1 — Standalone 2.5D High Resolution Render (4K)
Focuses on the inner 4 Mpc core with density-colored particles.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import struct
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
CSV_PATH = f"{OUTPUT_DIR}/time_series.csv"

R_HR = 8.0          # HR region radius [Mpc]
R_VIEW = 4.0        # View radius for this render [Mpc] (zoom ×2)
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
    """Compute local density using k-nearest neighbors distance"""
    tree = cKDTree(pos)
    distances, _ = tree.query(pos, k=k+1)  # k+1 because point itself is included
    # Density proxy: 1 / (mean distance to k neighbors)^3
    mean_dist = np.mean(distances[:, 1:], axis=1)  # exclude self
    density = 1.0 / (mean_dist**3 + 1e-10)
    return density


def main():
    # Find latest snapshot (closest to z=0)
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
    is_plus = data['is_plus']
    is_hr = data['is_hr']
    r = data['r']
    v_mag = data['v_mag']

    # Get star count from CSV
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        df_final = df[df['step'] <= step]
        n_stars = int(df_final['N_stars_HR'].iloc[-1]) if len(df_final) > 0 else 0
    except:
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
    # FILTER DATA FOR VIEW REGION — TWO LAYERS
    # ═══════════════════════════════════════════════════════════════════════

    # HR m+ particles in view
    in_view_mask = is_plus & is_hr & (r < R_VIEW)
    pos_view = pos[in_view_mask]
    r_view = r[in_view_mask]

    print(f"Particles in view (r < {R_VIEW} Mpc): {len(pos_view)}")

    # Layer 1 — Outer halo (1-4 Mpc)
    mask_outer = (r_view >= 1.0) & (r_view < R_VIEW)
    pos_outer = pos_view[mask_outer]
    n_outer_max = 50000
    if len(pos_outer) > n_outer_max:
        idx = np.random.choice(len(pos_outer), n_outer_max, replace=False)
        pos_outer = pos_outer[idx]
    print(f"Outer halo (1-4 Mpc): {len(pos_outer)} particles")

    # Layer 2 — Dense core (r < 1 Mpc)
    mask_core = r_view < 1.0
    pos_core = pos_view[mask_core]
    n_core_max = 40000
    if len(pos_core) > n_core_max:
        idx = np.random.choice(len(pos_core), n_core_max, replace=False)
        pos_core = pos_core[idx]
    print(f"Dense core (r < 1 Mpc): {len(pos_core)} particles")

    # Compute local density for core coloring
    print("Computing local densities for core...")
    density_core = compute_local_density(pos_core, k=8)
    density_core_norm = (density_core - density_core.min()) / (density_core.max() - density_core.min() + 1e-10)

    # Stars in view
    stars_in_view = None
    if star_pos is not None:
        r_stars = np.sqrt(star_pos[:, 0]**2 + star_pos[:, 1]**2 + star_pos[:, 2]**2)
        stars_in_view = star_pos[r_stars < R_VIEW]
        print(f"Stars in view: {len(stars_in_view)}")

    # Find central/most massive star (closest to origin)
    central_star = None
    if stars_in_view is not None and len(stars_in_view) > 0:
        r_stars_view = np.sqrt(stars_in_view[:, 0]**2 + stars_in_view[:, 1]**2 + stars_in_view[:, 2]**2)
        central_idx = np.argmin(r_stars_view)
        central_star = stars_in_view[central_idx]

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE FIGURE
    # ═══════════════════════════════════════════════════════════════════════

    figsize = (38.4, 21.6)  # 4K 16:9
    dpi = 100               # → 3840×2160 px

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Remove panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT DATA — TWO LAYERS
    # ═══════════════════════════════════════════════════════════════════════

    # Layer 1 — Outer halo diffus (1-4 Mpc)
    ax.scatter(
        pos_outer[:, 0], pos_outer[:, 1], pos_outer[:, 2],
        c='darkred', s=1.5, alpha=0.25, zorder=1,
        rasterized=True
    )

    # Layer 2 — Dense core (r < 1 Mpc) colored by density
    ax.scatter(
        pos_core[:, 0], pos_core[:, 1], pos_core[:, 2],
        c=density_core_norm, cmap='hot', s=2.5, alpha=0.5, zorder=2,
        rasterized=True
    )

    # Stars — bright white points
    if stars_in_view is not None and len(stars_in_view) > 0:
        ax.scatter(
            stars_in_view[:, 0], stars_in_view[:, 1], stars_in_view[:, 2],
            c='white', s=15, alpha=1.0, zorder=10
        )

    # Central star — highlighted
    if central_star is not None:
        ax.scatter(
            [central_star[0]], [central_star[1]], [central_star[2]],
            c='yellow', s=80, alpha=1.0, zorder=20,
            edgecolors='white', linewidths=0.5
        )

    # ═══════════════════════════════════════════════════════════════════════
    # VIEW SETTINGS
    # ═══════════════════════════════════════════════════════════════════════

    ax.view_init(elev=35, azim=225)

    ax.set_xlim(-R_VIEW, R_VIEW)
    ax.set_ylim(-R_VIEW, R_VIEW)
    ax.set_zlim(-R_VIEW, R_VIEW)

    # Axes styling
    ax.tick_params(colors='white', labelsize=11)
    ax.set_xlabel('X [Mpc]', color='white', fontsize=13, labelpad=10)
    ax.set_ylabel('Y [Mpc]', color='white', fontsize=13, labelpad=10)
    ax.set_zlabel('Z [Mpc]', color='white', fontsize=13, labelpad=10)

    ax.xaxis.line.set_color('gray')
    ax.yaxis.line.set_color('gray')
    ax.zaxis.line.set_color('gray')

    # ═══════════════════════════════════════════════════════════════════════
    # TITLES
    # ═══════════════════════════════════════════════════════════════════════

    n_stars_view = len(stars_in_view) if stars_in_view is not None else 0
    M_stars = n_stars * M_PART_HR

    fig.text(0.5, 0.97,
             f"JANUS Zoom-L1 | z={z:.3f} | t={t_gyr:.2f} Gyr | "
             f"N★={n_stars} | M★≈{M_stars:.1e} M☉ | r_HR=8 Mpc",
             ha='center', va='top', fontsize=16, color='white')

    fig.text(0.5, 0.93,
             "Région HR 8 Mpc — m⁻ absent (ségrégation complète)",
             ha='center', va='top', fontsize=13, color='#aaaaaa', style='italic')

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE — fill canvas
    # ═══════════════════════════════════════════════════════════════════════

    # Agrandir l'axe 3D pour remplir le canvas
    fig.subplots_adjust(left=-0.12, right=1.12, bottom=-0.08, top=1.05)

    out_path = f"{OUTPUT_DIR}/render_2p5D_4K.png"
    plt.savefig(out_path, dpi=dpi, facecolor='black')
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
