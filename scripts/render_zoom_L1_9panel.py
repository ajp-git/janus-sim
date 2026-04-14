#!/usr/bin/env python3
"""
JANUS Zoom-L1 Renderer — 9-panel layout (3×3)
3840×2160, DPI=200
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import struct
import sys
import os
from pathlib import Path
import pandas as pd

# Constants
L_ZOOM = 120.0  # Mpc (box extracted)
R_HR = 8.0      # Mpc
R_VIEW = 50.0   # Mpc for global view
R_ZOOM = 16.0   # Mpc for zoom view
EPSILON_HR = 0.03  # Mpc
M_PART_HR = 5.1e10  # M_sun

OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
FRAME_DIR = f"{OUTPUT_DIR}/frames"
CSV_PATH = f"{OUTPUT_DIR}/time_series.csv"


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


def cic_density(pos, L, ngrid, center=None):
    """Cloud-in-Cell density on grid"""
    if center is not None:
        pos = pos - center

    # Wrap to box
    pos = pos % L

    grid = np.zeros((ngrid, ngrid), dtype=np.float32)
    cell = L / ngrid

    for p in pos:
        x, y = p[0], p[1]
        ix = int(x / cell)
        iy = int(y / cell)

        fx = x / cell - ix
        fy = y / cell - iy

        ix = ix % ngrid
        iy = iy % ngrid
        ixp = (ix + 1) % ngrid
        iyp = (iy + 1) % ngrid

        grid[iy, ix] += (1 - fx) * (1 - fy)
        grid[iy, ixp] += fx * (1 - fy)
        grid[iyp, ix] += (1 - fx) * fy
        grid[iyp, ixp] += fx * fy

    return grid


def compute_radial_profile(pos, vel, signs, r_bins):
    """Compute radial density and velocity dispersion profiles"""
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)

    plus_mask = signs > 0
    minus_mask = signs < 0

    rho_plus = np.zeros(len(r_bins) - 1)
    rho_minus = np.zeros(len(r_bins) - 1)
    v_disp = np.zeros(len(r_bins) - 1)

    for i in range(len(r_bins) - 1):
        r_in, r_out = r_bins[i], r_bins[i + 1]
        shell_mask = (r >= r_in) & (r < r_out)
        vol = 4/3 * np.pi * (r_out**3 - r_in**3)

        n_plus = np.sum(shell_mask & plus_mask)
        n_minus = np.sum(shell_mask & minus_mask)

        rho_plus[i] = n_plus * M_PART_HR / vol if vol > 0 else 0
        rho_minus[i] = n_minus * M_PART_HR / vol if vol > 0 else 0

        if n_plus > 10:
            v_shell = v_mag[shell_mask & plus_mask]
            v_disp[i] = np.std(v_shell)

    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    return r_mid, rho_plus, rho_minus, v_disp


def render_frame(snap_path, step, total_frames=None):
    """Render a single frame"""

    # Read snapshot
    pos, vel, signs, a, t = read_snapshot(snap_path)
    z = 1/a - 1
    n = len(signs)

    # Identify populations
    plus_mask = signs > 0
    minus_mask = signs < 0

    pos_plus = pos[plus_mask]
    pos_minus = pos[minus_mask]
    vel_plus = vel[plus_mask]

    # Identify HR particles (r < R_HR from center)
    r_plus = np.sqrt(pos_plus[:, 0]**2 + pos_plus[:, 1]**2 + pos_plus[:, 2]**2)
    hr_mask = r_plus < R_HR

    # Stars: high velocity dispersion particles in dense regions
    # For now, use particles with very low velocity as "settled" proto-stars
    v_mag = np.sqrt(vel_plus[:, 0]**2 + vel_plus[:, 1]**2 + vel_plus[:, 2]**2)

    # Read CSV for time series data
    try:
        df = pd.read_csv(CSV_PATH)
        df_up_to = df[df['step'] <= step]
        n_stars = df_up_to['N_stars_HR'].iloc[-1] if len(df_up_to) > 0 else 0
        rho_hr = df_up_to['rho_max_HR'].iloc[-1] if len(df_up_to) > 0 else 0
    except:
        df = None
        n_stars = 0
        rho_hr = 0

    # Create figure
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor='black')

    # Super title
    title = f"JANUS Zoom-L1 | z={z:.3f} | t={t:.2f} Gyr | N★={int(n_stars)} | ρ_HR={rho_hr:.0f} | m_part=5×10¹⁰ M☉ | r_HR=8 Mpc"
    fig.suptitle(title, fontsize=14, color='white', y=0.98)

    # ═══════════════════════════════════════════════════════════════
    # ROW 1 — Global views
    # ═══════════════════════════════════════════════════════════════

    # Panel 1: XY density m+ — 50 Mpc
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_facecolor('black')

    # Filter to 50 Mpc region
    in_view = r_plus < R_VIEW
    pos_view = pos_plus[in_view]

    # CIC density
    grid1 = cic_density(pos_view[:, :2] + R_VIEW, 2*R_VIEW, 512)
    grid1 = np.maximum(grid1, 0.1)

    im1 = ax1.imshow(np.log10(grid1), extent=[-R_VIEW, R_VIEW, -R_VIEW, R_VIEW],
                     cmap='inferno', origin='lower', vmin=-0.5, vmax=2.5)

    # HR zone circle
    circle1 = Circle((0, 0), R_HR, fill=False, color='white', linestyle='--', linewidth=1)
    ax1.add_patch(circle1)

    ax1.set_xlim(-R_VIEW, R_VIEW)
    ax1.set_ylim(-R_VIEW, R_VIEW)
    ax1.set_xlabel('X [Mpc]', color='white', fontsize=8)
    ax1.set_ylabel('Y [Mpc]', color='white', fontsize=8)
    ax1.set_title('Density m+ (50 Mpc)', color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=7)

    # Panel 2: XY density zoom 16 Mpc
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_facecolor('black')

    in_zoom = r_plus < R_ZOOM
    pos_zoom = pos_plus[in_zoom]

    grid2 = cic_density(pos_zoom[:, :2] + R_ZOOM, 2*R_ZOOM, 256)
    grid2 = np.maximum(grid2, 0.1)

    im2 = ax2.imshow(np.log10(grid2), extent=[-R_ZOOM, R_ZOOM, -R_ZOOM, R_ZOOM],
                     cmap='inferno', origin='lower', vmin=-0.5, vmax=3)

    # Mark proto-stars (low velocity particles in HR zone)
    if n_stars > 0:
        # Use slowest particles as proxy for stars
        v_sorted_idx = np.argsort(v_mag[hr_mask])[:int(n_stars)]
        star_pos = pos_plus[hr_mask][v_sorted_idx]
        ax2.scatter(star_pos[:, 0], star_pos[:, 1], c='white', s=3, alpha=0.8)

    ax2.set_xlim(-R_ZOOM, R_ZOOM)
    ax2.set_ylim(-R_ZOOM, R_ZOOM)
    ax2.set_xlabel('X [Mpc]', color='white', fontsize=8)
    ax2.set_ylabel('Y [Mpc]', color='white', fontsize=8)
    ax2.set_title('Zoom 16 Mpc + proto-stars', color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=7)

    # Panel 3: 2.5D isometric
    ax3 = fig.add_subplot(3, 3, 3, projection='3d', facecolor='black')
    ax3.set_facecolor('black')

    # Rotation angle
    if total_frames:
        azim = 45 + (step / total_frames) * 360
    else:
        azim = 45 + step * 0.5

    # Subsample for 3D
    subsample = max(1, len(pos_zoom) // 5000)
    pos_3d = pos_zoom[::subsample]

    ax3.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2],
                c='red', alpha=0.15, s=0.5)

    # Proto-stars in 3D
    if n_stars > 0 and len(star_pos) > 0:
        ax3.scatter(star_pos[:, 0], star_pos[:, 1], star_pos[:, 2],
                    c='white', alpha=0.9, s=4)

    ax3.view_init(elev=30, azim=azim)
    ax3.set_xlim(-R_ZOOM, R_ZOOM)
    ax3.set_ylim(-R_ZOOM, R_ZOOM)
    ax3.set_zlim(-R_ZOOM, R_ZOOM)
    ax3.set_xlabel('X', color='white', fontsize=7)
    ax3.set_ylabel('Y', color='white', fontsize=7)
    ax3.set_zlabel('Z', color='white', fontsize=7)
    ax3.tick_params(colors='white', labelsize=6)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.set_title('2.5D Isometric', color='white', fontsize=10)

    # ═══════════════════════════════════════════════════════════════
    # ROW 2 — Science HR
    # ═══════════════════════════════════════════════════════════════

    # Compute radial profiles
    r_bins = np.logspace(-2, 1, 30)  # 0.01 to 10 Mpc
    r_mid, rho_plus_prof, rho_minus_prof, v_disp_prof = compute_radial_profile(
        pos, vel, signs, r_bins)

    # Panel 4: Radial density profile
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_facecolor('black')

    ax4.loglog(r_mid, rho_plus_prof + 1e5, 'r-', linewidth=1.5, label='ρ+')
    ax4.loglog(r_mid, rho_minus_prof + 1e5, 'b-', linewidth=1.5, label='ρ-')
    ax4.axvline(EPSILON_HR, color='gray', linestyle=':', label=f'ε={EPSILON_HR}')
    ax4.axvline(R_HR, color='white', linestyle='--', alpha=0.5, label='R_HR')

    ax4.set_xlim(0.01, 10)
    ax4.set_ylim(1e5, 1e15)
    ax4.set_xlabel('r [Mpc]', color='white', fontsize=8)
    ax4.set_ylabel('ρ [M☉/Mpc³]', color='white', fontsize=8)
    ax4.set_title('Radial Density Profile', color='white', fontsize=10)
    ax4.tick_params(colors='white', labelsize=7)
    ax4.legend(fontsize=7, loc='upper right', facecolor='black', labelcolor='white')
    for spine in ax4.spines.values():
        spine.set_color('white')

    # Panel 5: Velocity dispersion profile
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_facecolor('black')

    ax5.semilogx(r_mid, v_disp_prof, 'g-', linewidth=1.5)
    ax5.axvline(R_HR, color='white', linestyle='--', alpha=0.5)

    ax5.set_xlim(0.01, 10)
    ax5.set_ylim(0, 300)
    ax5.set_xlabel('r [Mpc]', color='white', fontsize=8)
    ax5.set_ylabel('σ_v [km/s]', color='white', fontsize=8)
    ax5.set_title('Velocity Dispersion', color='white', fontsize=10)
    ax5.tick_params(colors='white', labelsize=7)
    for spine in ax5.spines.values():
        spine.set_color('white')

    # Panel 6: SFR history
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.set_facecolor('black')

    if df is not None and len(df_up_to) > 1:
        ax6.semilogy(df_up_to['z'], df_up_to['SFR_HR'] + 1, 'c-', linewidth=1)
        ax6.scatter([z], [df_up_to['SFR_HR'].iloc[-1] + 1], c='red', s=30, zorder=10)

    # Madau peak z=1.9
    ax6.axvline(1.9, color='yellow', linestyle='--', alpha=0.5, label='Madau peak')

    ax6.set_xlim(0.5, 0)
    ax6.set_ylim(1e9, 1e14)
    ax6.set_xlabel('z', color='white', fontsize=8)
    ax6.set_ylabel('SFR [M☉/Gyr]', color='white', fontsize=8)
    ax6.set_title('Star Formation Rate', color='white', fontsize=10)
    ax6.tick_params(colors='white', labelsize=7)
    for spine in ax6.spines.values():
        spine.set_color('white')

    # ═══════════════════════════════════════════════════════════════
    # ROW 3 — Time evolution
    # ═══════════════════════════════════════════════════════════════

    # Panel 7: N★(z)
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_facecolor('black')

    if df is not None and len(df_up_to) > 1:
        ax7.plot(df_up_to['z'], df_up_to['N_stars_HR'], 'm-', linewidth=1.5)
        ax7.scatter([z], [n_stars], c='red', s=30, zorder=10)

    ax7.set_xlim(0.5, 0)
    ax7.set_ylim(0, max(2000, n_stars * 1.2))
    ax7.set_xlabel('z', color='white', fontsize=8)
    ax7.set_ylabel('N★', color='white', fontsize=8)
    ax7.set_title('Star Count Evolution', color='white', fontsize=10)
    ax7.tick_params(colors='white', labelsize=7)
    for spine in ax7.spines.values():
        spine.set_color('white')

    # Panel 8: ρ_max_HR(z)
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.set_facecolor('black')

    if df is not None and len(df_up_to) > 1:
        ax8.semilogy(df_up_to['z'], df_up_to['rho_max_HR'] + 1, 'orange', linewidth=1.5)
        ax8.scatter([z], [rho_hr + 1], c='red', s=30, zorder=10)

    ax8.set_xlim(0.5, 0)
    ax8.set_ylim(1, 1e4)
    ax8.set_xlabel('z', color='white', fontsize=8)
    ax8.set_ylabel('ρ_max_HR', color='white', fontsize=8)
    ax8.set_title('Max HR Density (baryon cycle)', color='white', fontsize=10)
    ax8.tick_params(colors='white', labelsize=7)
    for spine in ax8.spines.values():
        spine.set_color('white')

    # Panel 9: v_disp(z)
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_facecolor('black')

    if df is not None and len(df_up_to) > 1:
        ax9.plot(df_up_to['z'], df_up_to['v_disp_HR'], 'lime', linewidth=1.5)
        ax9.scatter([z], [df_up_to['v_disp_HR'].iloc[-1]], c='red', s=30, zorder=10)

    ax9.set_xlim(0.5, 0)
    ax9.set_ylim(0, 300)
    ax9.set_xlabel('z', color='white', fontsize=8)
    ax9.set_ylabel('σ_v [km/s]', color='white', fontsize=8)
    ax9.set_title('Velocity Dispersion (virialization)', color='white', fontsize=10)
    ax9.tick_params(colors='white', labelsize=7)
    for spine in ax9.spines.values():
        spine.set_color('white')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def render_one_worker(snap_path):
    """Worker function for parallel rendering (must be at module level for pickle)"""
    step = int(Path(snap_path).stem.split('_')[1])
    out_path = f"{FRAME_DIR}/frame_{step:05d}.png"
    if os.path.exists(out_path):
        return f"Skip {step}"
    try:
        fig = render_frame(str(snap_path), step, 18500)
        fig.savefig(out_path, facecolor='black', edgecolor='none')
        plt.close(fig)
        return f"Done {step}"
    except Exception as e:
        return f"Error {step}: {e}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, help='Render specific step')
    parser.add_argument('--all', action='store_true', help='Render all existing')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--test', action='store_true', help='Test on latest snapshot')
    args = parser.parse_args()

    os.makedirs(FRAME_DIR, exist_ok=True)

    # Get all snapshots
    snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))
    if not snaps:
        print("No snapshots found")
        return

    total_frames = 18500  # Total expected

    if args.test or args.step:
        # Single frame
        if args.step:
            snap_path = Path(SNAP_DIR) / f'snap_{args.step:05d}.bin'
        else:
            snap_path = snaps[-1]

        step = int(snap_path.stem.split('_')[1])
        print(f"Rendering step {step}...")

        fig = render_frame(str(snap_path), step, total_frames)
        out_path = f"{FRAME_DIR}/frame_{step:05d}.png"
        fig.savefig(out_path, facecolor='black', edgecolor='none')
        plt.close(fig)
        print(f"Saved: {out_path}")

    elif args.all:
        # Render all existing
        from multiprocessing import Pool, cpu_count

        snap_paths = [str(s) for s in snaps]
        print(f"Rendering {len(snap_paths)} frames with {cpu_count()} processes...")
        with Pool(cpu_count()) as p:
            for i, result in enumerate(p.imap_unordered(render_one_worker, snap_paths)):
                if i % 50 == 0:
                    print(f"[{i}/{len(snap_paths)}] {result}")
        print("Done!")

    elif args.daemon:
        import time
        print("Starting render daemon (Ctrl+C to stop)...")
        rendered = set()

        while True:
            snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))
            for snap_path in snaps:
                step = int(snap_path.stem.split('_')[1])
                if step in rendered:
                    continue

                out_path = f"{FRAME_DIR}/frame_{step:05d}.png"
                if os.path.exists(out_path):
                    rendered.add(step)
                    continue

                try:
                    fig = render_frame(str(snap_path), step, total_frames)
                    fig.savefig(out_path, facecolor='black', edgecolor='none')
                    plt.close(fig)
                    rendered.add(step)
                    print(f"Rendered step {step}")
                except Exception as e:
                    print(f"Error step {step}: {e}")

            time.sleep(300)  # 5 minutes


if __name__ == '__main__':
    main()
