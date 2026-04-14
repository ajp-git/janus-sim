#!/usr/bin/env python3
"""
JANUS Zoom-L1 Renderer v2 — 9-panel layout (3×3)
4K resolution (3840×2160), supports black/white backgrounds
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import struct
import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
CSV_PATH = f"{OUTPUT_DIR}/time_series.csv"

R_HR = 8.0          # HR region radius [Mpc]
R_VIEW = 50.0       # Global view radius [Mpc]
EPSILON = 0.03      # Softening [Mpc]
M_PART_HR = 5.1e10  # HR particle mass [M_sun]


def read_snapshot(path):
    """Read JSNP v1 snapshot (basic format)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]

        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int8)
        signs[signs > 1] = -1

    # Derive HR and star flags based on position and velocity
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    is_plus = signs > 0
    is_hr = r < R_HR

    # Stars: identify from CSV count - use slowest HR particles
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)

    return {
        'n': n, 'a': a, 't': t,
        'pos': pos, 'vel': vel, 'signs': signs,
        'is_plus': is_plus, 'is_hr': is_hr,
        'r': r, 'v_mag': v_mag
    }


def compute_radial_profiles(data, n_bins=30):
    """Compute radial density and velocity dispersion profiles"""
    r_bins = np.logspace(-2, np.log10(R_HR), n_bins + 1)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    r = data['r']
    v_mag = data['v_mag']
    is_plus = data['is_plus']
    is_minus = ~is_plus

    rho_plus = np.zeros(n_bins)
    rho_minus = np.zeros(n_bins)
    v_disp = np.zeros(n_bins)

    for i in range(n_bins):
        r_in, r_out = r_bins[i], r_bins[i + 1]
        shell_mask = (r >= r_in) & (r < r_out)
        vol = 4/3 * np.pi * (r_out**3 - r_in**3)

        n_plus = np.sum(shell_mask & is_plus)
        n_minus = np.sum(shell_mask & is_minus)

        rho_plus[i] = n_plus * M_PART_HR / vol if vol > 0 else 0
        rho_minus[i] = n_minus * M_PART_HR / vol if vol > 0 else 0

        if n_plus > 10:
            v_shell = v_mag[shell_mask & is_plus]
            v_disp[i] = np.std(v_shell)

    return r_mid, rho_plus, rho_minus, v_disp


def render_frame(snap_path, step, white_bg=False):
    """Render a single frame with 9-panel layout"""

    # Reset matplotlib state to avoid cmap contamination
    plt.close('all')
    plt.rcdefaults()
    plt.set_cmap('hot')  # Force global cmap to 'hot'

    # Color scheme
    # Spatial panels (row 1) are ALWAYS black background with 'hot' cmap
    SPATIAL_BGCOLOR = 'black'
    STAR_COLOR = 'white'
    STAR_SIZE = 8
    CIRCLE_COLOR = 'white'

    # Only figure bg and row 2/3 axes change
    if white_bg:
        FIG_BGCOLOR = 'white'
        AXES_BGCOLOR = 'white'
        TEXT_COLOR = 'black'
        GRID_ALPHA = 0.3
        SPINE_COLOR = 'black'
    else:
        FIG_BGCOLOR = 'black'
        AXES_BGCOLOR = '#111111'
        TEXT_COLOR = 'white'
        GRID_ALPHA = 0.2
        SPINE_COLOR = 'gray'

    # Forcer le cmap par défaut — protège contre toute
    # modification globale de rcParams
    plt.rcParams['image.cmap'] = 'hot'
    _hot = plt.get_cmap('hot').copy()

    # Read data
    data = read_snapshot(snap_path)
    z = 1 / data['a'] - 1
    t_gyr = data['t']

    pos = data['pos']
    vel = data['vel']
    is_plus = data['is_plus']
    is_hr = data['is_hr']
    r = data['r']
    v_mag = data['v_mag']

    # Read CSV for time series and star count
    try:
        df = pd.read_csv(CSV_PATH)
        df_up_to = df[df['step'] <= step]
        n_stars = int(df_up_to['N_stars_HR'].iloc[-1]) if len(df_up_to) > 0 else 0
        rho_max = df_up_to['rho_max_HR'].iloc[-1] if len(df_up_to) > 0 else 0
    except Exception as e:
        print(f"Warning: Could not read CSV: {e}")
        df = None
        df_up_to = None
        n_stars = 0
        rho_max = 0

    # Identify stars as slowest HR m+ particles
    hr_plus_mask = is_plus & is_hr
    star_pos = None
    if n_stars > 0 and np.sum(hr_plus_mask) > n_stars:
        v_hr = v_mag[hr_plus_mask]
        pos_hr = pos[hr_plus_mask]
        star_idx = np.argsort(v_hr)[:n_stars]
        star_pos = pos_hr[star_idx]

    # Positions for m+ particles
    pos_plus = pos[is_plus]

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE FIGURE
    # ═══════════════════════════════════════════════════════════════════════
    plt.rcParams['figure.facecolor'] = FIG_BGCOLOR
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR
    plt.rcParams['xtick.color'] = TEXT_COLOR
    plt.rcParams['ytick.color'] = TEXT_COLOR

    fig = plt.figure(figsize=(38.4, 21.6), dpi=100, facecolor=FIG_BGCOLOR)

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 1 — Spatial views
    # ═══════════════════════════════════════════════════════════════════════

    # Panel [0,0] — Density m+ XY global (50 Mpc)
    ax00 = fig.add_subplot(3, 3, 1)
    ax00.set_facecolor(SPATIAL_BGCOLOR)

    in_view = r[is_plus] < R_VIEW
    pos_view = pos_plus[in_view]

    H, xedges, yedges = np.histogram2d(
        pos_view[:, 0], pos_view[:, 1],
        bins=512, range=[[-R_VIEW, R_VIEW], [-R_VIEW, R_VIEW]]
    )
    H = np.log1p(H.T)
    vmax = np.percentile(H[H > 0], 95) if np.any(H > 0) else 1

    ax00.imshow(H, extent=[-R_VIEW, R_VIEW, -R_VIEW, R_VIEW],
                cmap=_hot, norm=Normalize(vmin=0, vmax=vmax), origin='lower')

    # HR circle (r_HR = 8 Mpc)
    circle = plt.Circle(
        (0, 0), R_HR,
        fill=False,
        color='white',   # toujours blanc (fond toujours noir)
        linestyle='--',
        linewidth=1.2,
        alpha=0.7
    )
    ax00.add_patch(circle)
    ax00.set_aspect('equal')  # nécessaire pour cercle non-elliptique

    ax00.set_xlim(-R_VIEW, R_VIEW)
    ax00.set_ylim(-R_VIEW, R_VIEW)
    ax00.set_xlabel('X [Mpc]', fontsize=10)
    ax00.set_ylabel('Y [Mpc]', fontsize=10)
    ax00.set_title('Density m+ XY (50 Mpc)', fontsize=12, color=TEXT_COLOR)
    for spine in ax00.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # Panel [0,1] — Zoom HR XY + proto-stars
    ax01 = fig.add_subplot(3, 3, 2)
    ax01.set_facecolor(SPATIAL_BGCOLOR)

    in_hr = r[is_plus] < R_HR
    pos_hr_plus = pos_plus[in_hr]

    H_hr, _, _ = np.histogram2d(
        pos_hr_plus[:, 0], pos_hr_plus[:, 1],
        bins=256, range=[[-R_HR, R_HR], [-R_HR, R_HR]]
    )
    H_hr = np.log1p(H_hr.T)
    vmax_hr = np.percentile(H_hr[H_hr > 0], 95) if np.any(H_hr > 0) else 1

    ax01.imshow(H_hr, extent=[-R_HR, R_HR, -R_HR, R_HR],
                cmap=_hot, norm=Normalize(vmin=0, vmax=vmax_hr), origin='lower')

    if star_pos is not None and len(star_pos) > 0:
        ax01.scatter(star_pos[:, 0], star_pos[:, 1],
                     c=STAR_COLOR, s=STAR_SIZE, alpha=0.9, zorder=10)

    ax01.set_xlim(-R_HR, R_HR)
    ax01.set_ylim(-R_HR, R_HR)
    ax01.set_xlabel('X [Mpc]', fontsize=10)
    ax01.set_ylabel('Y [Mpc]', fontsize=10)
    ax01.set_title(f'Zoom HR XY 16 Mpc | N★={n_stars}', fontsize=12, color=TEXT_COLOR)
    for spine in ax01.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # Panel [0,2] — Zoom HR XZ + proto-stars
    ax02 = fig.add_subplot(3, 3, 3)
    ax02.set_facecolor(SPATIAL_BGCOLOR)

    H_xz, _, _ = np.histogram2d(
        pos_hr_plus[:, 0], pos_hr_plus[:, 2],
        bins=256, range=[[-R_HR, R_HR], [-R_HR, R_HR]]
    )
    H_xz = np.log1p(H_xz.T)

    ax02.imshow(H_xz, extent=[-R_HR, R_HR, -R_HR, R_HR],
                cmap=_hot, norm=Normalize(vmin=0, vmax=vmax_hr), origin='lower')

    if star_pos is not None and len(star_pos) > 0:
        ax02.scatter(star_pos[:, 0], star_pos[:, 2],
                     c=STAR_COLOR, s=STAR_SIZE, alpha=0.9, zorder=10)

    ax02.set_xlim(-R_HR, R_HR)
    ax02.set_ylim(-R_HR, R_HR)
    ax02.set_xlabel('X [Mpc]', fontsize=10)
    ax02.set_ylabel('Z [Mpc]', fontsize=10)
    ax02.set_title('Zoom HR XZ 16 Mpc', fontsize=12, color=TEXT_COLOR)
    for spine in ax02.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 2 — Instantaneous physics
    # ═══════════════════════════════════════════════════════════════════════

    # Compute radial profiles
    r_mid, rho_plus, rho_minus, v_disp_prof = compute_radial_profiles(data)

    # Panel [1,0] — Radial Density Profile
    ax10 = fig.add_subplot(3, 3, 4)
    ax10.set_facecolor(AXES_BGCOLOR)

    ax10.loglog(r_mid, rho_plus + 1e3, 'r-', linewidth=2, label='ρ+')
    ax10.loglog(r_mid, rho_minus + 1e3, 'b-', linewidth=2, label='ρ-')
    ax10.axvline(EPSILON, color='gray', linestyle=':', linewidth=1.5, label=f'ε={EPSILON}')
    ax10.axvline(R_HR, color=TEXT_COLOR, linestyle='--', linewidth=1, alpha=0.5, label='r_HR')

    ax10.set_xlim(0.01, R_HR * 1.2)
    ax10.set_ylim(1e3, 1e15)
    ax10.set_xlabel('r [Mpc]', fontsize=10)
    ax10.set_ylabel('ρ [M☉/Mpc³]', fontsize=10)
    ax10.set_title('Radial Density Profile', fontsize=12, color=TEXT_COLOR)
    ax10.legend(fontsize=9, loc='upper right',
                facecolor=AXES_BGCOLOR, labelcolor=TEXT_COLOR, edgecolor=SPINE_COLOR)
    ax10.grid(True, alpha=GRID_ALPHA)
    for spine in ax10.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # Panel [1,1] — Velocity Dispersion
    ax11 = fig.add_subplot(3, 3, 5)
    ax11.set_facecolor(AXES_BGCOLOR)

    ax11.semilogx(r_mid, v_disp_prof, 'g-', linewidth=2)
    ax11.axvline(R_HR, color=TEXT_COLOR, linestyle='--', linewidth=1, alpha=0.5)

    ax11.set_xlim(0.01, R_HR * 1.2)
    ax11.set_ylim(0, 300)
    ax11.set_xlabel('r [Mpc]', fontsize=10)
    ax11.set_ylabel('σ_v [km/s]', fontsize=10)
    ax11.set_title('Velocity Dispersion σ_v(r)', fontsize=12, color=TEXT_COLOR)
    ax11.grid(True, alpha=GRID_ALPHA)
    for spine in ax11.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # Panel [1,2] — SFR(z)
    ax12 = fig.add_subplot(3, 3, 6)
    ax12.set_facecolor(AXES_BGCOLOR)

    if df_up_to is not None and len(df_up_to) > 1:
        ax12.semilogy(df_up_to['z'], df_up_to['SFR_HR'] + 1, 'c-', linewidth=2)
        ax12.scatter([z], [df_up_to['SFR_HR'].iloc[-1] + 1], c='red', s=50, zorder=10)

    ax12.set_xlim(0.5, 0)
    ax12.set_ylim(1e8, 1e14)
    ax12.set_xlabel('z', fontsize=10)
    ax12.set_ylabel('SFR [M☉/Gyr]', fontsize=10)
    ax12.set_title('Star Formation Rate', fontsize=12, color=TEXT_COLOR)
    ax12.grid(True, alpha=GRID_ALPHA)
    for spine in ax12.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 3 — Time series + 2.5D
    # ═══════════════════════════════════════════════════════════════════════

    # Panel [2,0] — Star Count Evolution N★(z)
    ax20 = fig.add_subplot(3, 3, 7)
    ax20.set_facecolor(AXES_BGCOLOR)

    if df_up_to is not None and len(df_up_to) > 1:
        ax20.plot(df_up_to['z'], df_up_to['N_stars_HR'], 'm-', linewidth=2)
        ax20.scatter([z], [n_stars], c='red', s=50, zorder=10)

    ax20.set_xlim(0.5, 0)
    ax20.set_ylim(0, max(2500, n_stars * 1.2))
    ax20.set_xlabel('z', fontsize=10)
    ax20.set_ylabel('N★', fontsize=10)
    ax20.set_title('Star Count Evolution', fontsize=12, color=TEXT_COLOR)
    ax20.grid(True, alpha=GRID_ALPHA)
    for spine in ax20.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # Panel [2,1] — Max HR Density ρ_max(z)
    ax21 = fig.add_subplot(3, 3, 8)
    ax21.set_facecolor(AXES_BGCOLOR)

    if df_up_to is not None and len(df_up_to) > 1:
        ax21.semilogy(df_up_to['z'], df_up_to['rho_max_HR'] + 1, 'orange', linewidth=2)
        ax21.scatter([z], [rho_max + 1], c='red', s=50, zorder=10)

    ax21.set_xlim(0.5, 0)
    ax21.set_ylim(1, 1e4)
    ax21.set_xlabel('z', fontsize=10)
    ax21.set_ylabel('ρ_max_HR', fontsize=10)
    ax21.set_title('Max HR Density (baryon cycle)', fontsize=12, color=TEXT_COLOR)
    ax21.grid(True, alpha=GRID_ALPHA)
    for spine in ax21.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # Panel [2,2] — 2.5D Isometric
    ax22 = fig.add_subplot(3, 3, 9, projection='3d', facecolor=AXES_BGCOLOR)

    # Colors depend on background mode
    particle_color_3d = 'salmon' if white_bg else 'red'
    star_color_3d = 'darkred' if white_bg else 'white'

    # Subsample HR particles
    hr_plus_idx = np.where(is_plus & is_hr)[0]
    n_plot = min(5000, len(hr_plus_idx))
    if n_plot > 0:
        sample_idx = np.random.choice(hr_plus_idx, n_plot, replace=False)
        pos_3d = pos[sample_idx]

        ax22.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2],
                     c=particle_color_3d, alpha=0.2 if white_bg else 0.3,
                     s=0.3 if white_bg else 0.5, rasterized=True)

    if star_pos is not None and len(star_pos) > 0:
        ax22.scatter(star_pos[:, 0], star_pos[:, 1], star_pos[:, 2],
                     c=star_color_3d, alpha=0.9, s=4, zorder=10)

    ax22.set_xlim(-R_HR, R_HR)
    ax22.set_ylim(-R_HR, R_HR)
    ax22.set_zlim(-R_HR, R_HR)
    ax22.set_xlabel('X', fontsize=9, color=TEXT_COLOR)
    ax22.set_ylabel('Y', fontsize=9, color=TEXT_COLOR)
    ax22.set_zlabel('Z', fontsize=9, color=TEXT_COLOR)
    ax22.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax22.xaxis.pane.fill = False
    ax22.yaxis.pane.fill = False
    ax22.zaxis.pane.fill = False
    ax22.grid(False)
    ax22.set_title('3D HR region', fontsize=12, color=TEXT_COLOR, pad=10)
    ax22.view_init(elev=25, azim=45 + step * 0.5)

    # ═══════════════════════════════════════════════════════════════════════
    # GLOBAL TITLE
    # ═══════════════════════════════════════════════════════════════════════
    M_stars = n_stars * M_PART_HR  # Total stellar mass
    fig.suptitle(
        f"JANUS Zoom-L1 | z={z:.3f} | t={t_gyr:.2f} Gyr | "
        f"N★={n_stars} | M★≈{M_stars:.1e} M☉ | ρ_HR={rho_max:.0f} | "
        f"m_part=5×10¹⁰ M☉ | r_HR=8 Mpc",
        fontsize=14, color=TEXT_COLOR, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


def main():
    parser = argparse.ArgumentParser(description='Zoom-L1 Renderer v2')
    parser.add_argument('--step', type=int, help='Render specific step')
    parser.add_argument('--white-bg', action='store_true', help='Use white background')
    parser.add_argument('--test', action='store_true', help='Test mode: render latest snapshot')
    parser.add_argument('--both', action='store_true', help='Generate both black and white versions')
    args = parser.parse_args()

    # Find latest snapshot
    snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))
    if not snaps:
        print("No snapshots found")
        return

    if args.step:
        snap_path = Path(SNAP_DIR) / f'snap_{args.step:05d}.bin'
        if not snap_path.exists():
            print(f"Snapshot not found: {snap_path}")
            return
    else:
        snap_path = snaps[-1]

    step = int(snap_path.stem.split('_')[1])
    print(f"Rendering step {step} from {snap_path}")

    if args.both or args.test:
        # Generate both versions
        print("Generating black background version...")
        fig_black = render_frame(str(snap_path), step, white_bg=False)
        out_black = f"{OUTPUT_DIR}/frame_test_black.png"
        fig_black.savefig(out_black, facecolor=fig_black.get_facecolor(), edgecolor='none', dpi=100)
        plt.close(fig_black)
        print(f"Saved: {out_black}")

        print("Generating white background version...")
        fig_white = render_frame(str(snap_path), step, white_bg=True)
        out_white = f"{OUTPUT_DIR}/frame_test_white.png"
        fig_white.savefig(out_white, facecolor=fig_white.get_facecolor(), edgecolor='none', dpi=100)
        plt.close(fig_white)
        print(f"Saved: {out_white}")
    else:
        # Single version
        fig = render_frame(str(snap_path), step, white_bg=args.white_bg)
        suffix = 'white' if args.white_bg else 'black'
        out_path = f"{OUTPUT_DIR}/frame_{step:05d}_{suffix}.png"
        fig.savefig(out_path, facecolor='white' if args.white_bg else 'black',
                    edgecolor='none', dpi=100)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
