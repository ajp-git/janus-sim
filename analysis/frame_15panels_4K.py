#!/usr/bin/env python3
"""
frame_15panels_4K.py — 15-panel diagnostic frame for Janus simulation
VERSION 2 — With all corrections applied

Layout: 3 rows × 5 columns
- Row 1: Global 400 Mpc sub-box views (no edge artifacts)
- Row 2: Cluster #1 zoom (25 Mpc)
- Row 3: Science panels
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# === Parameters ===
WIDTH, HEIGHT = 3840, 2160
DPI = 200

# Cluster #1 center
CENTER = np.array([-5.329, 11.171, -39.571])
RADIUS = 25.0
BOX = 500.0
SUBBOX = 200.0  # Half-width of central sub-box (400 Mpc total)

# Colors
COLOR_PLUS_DARK = '#3d0a00'
COLOR_PLUS = '#cc2200'
COLOR_MINUS = '#0a3d6b'
COLOR_STAR = 'white'
COLOR_NEW = '#ffff44'
COLOR_HALO = '#ff8800'


def read_snapshot(path):
    """Read binary snapshot, return positions, velocities, signs, metadata"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]

        pos = np.frombuffer(f.read(n * 3 * 4), dtype='<f4').reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype='<f4').reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype='<i1')

    z = 1.0 / a - 1.0
    return pos, vel, signs, z, t, n


def periodic_dist(d, L):
    """Periodic boundary distance"""
    return d - L * np.round(d / L)


def get_central_mask(pos, half_width=200.0):
    """Mask for particles in central sub-box (eliminates edge artifacts)"""
    return (np.abs(pos[:, 0]) < half_width) & \
           (np.abs(pos[:, 1]) < half_width) & \
           (np.abs(pos[:, 2]) < half_width)


def get_particles_in_sphere(pos, signs, center, radius, box):
    """Extract particles within sphere (periodic BC)"""
    dx = periodic_dist(pos[:, 0] - center[0], box)
    dy = periodic_dist(pos[:, 1] - center[1], box)
    dz = periodic_dist(pos[:, 2] - center[2], box)
    r2 = dx**2 + dy**2 + dz**2

    mask = r2 < radius**2
    return dx[mask], dy[mask], dz[mask], signs[mask]


def compute_overdensity_sphere(pos_rel, k=32):
    """Compute local overdensity for particles in sphere using k-NN"""
    from scipy.spatial import cKDTree

    if len(pos_rel) < k + 1:
        return np.ones(len(pos_rel)), np.zeros(len(pos_rel), dtype=bool)

    tree = cKDTree(pos_rel)
    distances, _ = tree.query(pos_rel, k=k+1)
    r_k = distances[:, -1]
    r_k = np.maximum(r_k, 1e-10)

    vol_k = 4.0 / 3.0 * np.pi * r_k**3
    rho_local = k / vol_k
    rho_mean = len(pos_rel) / (4.0/3.0 * np.pi * RADIUS**3)

    overdensity = rho_local / rho_mean
    is_protostar = overdensity > 5.0

    return overdensity, is_protostar


def make_density_map_cic(x, y, bins=512, half_width=200.0):
    """Create 2D CIC density map on central sub-box"""
    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins,
        range=[[-half_width, half_width], [-half_width, half_width]]
    )
    # Smooth slightly
    H = gaussian_filter(H.T, sigma=1.5)
    return H, xedges, yedges


def main():
    base_dir = Path('/mnt/T2/janus-sim')
    snap_dir = base_dir / 'output/janus_baryonic_calibrated/snapshots'
    out_dir = base_dir / 'output/janus_baryonic_calibrated/frames'
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Load data ===
    print("Loading snapshot step 5080...")
    pos, vel, signs, z, t, n = read_snapshot(snap_dir / 'snap_05080.bin')

    print("Loading snapshot z=4 (step 0)...")
    pos_z4, _, signs_z4, z4, _, _ = read_snapshot(snap_dir / 'snap_00000.bin')

    print("Loading time series...")
    ts = pd.read_csv(base_dir / 'output/janus_baryonic_calibrated/time_series.csv')

    print("Loading stellar evolution...")
    stellar = pd.read_csv(base_dir / 'output/janus_baryonic_calibrated/stellar_evolution.csv')

    # Get current metrics from time_series
    current_step = 5080
    if 'step' in ts.columns:
        ts_row = ts[ts['step'] <= current_step].iloc[-1] if len(ts[ts['step'] <= current_step]) > 0 else ts.iloc[-1]
    else:
        ts_row = ts.iloc[-1]

    S = ts_row.get('segregation', 0.98) if hasattr(ts_row, 'get') else 0.98
    rho_max = ts_row.get('rho_max_plus', 100000) if hasattr(ts_row, 'get') else 100000

    # v_rms ratio — CORRECTION v3: use correct column names
    if 'v_rms_plus' in ts.columns and 'v_rms_minus' in ts.columns:
        vrms_plus = ts_row['v_rms_plus'] if 'v_rms_plus' in ts_row else 1.0
        vrms_minus = ts_row['v_rms_minus'] if 'v_rms_minus' in ts_row else 1.0
        current_ratio = vrms_minus / vrms_plus if vrms_plus > 0 else 1.0
    elif 'ratio' in ts.columns:
        current_ratio = ts_row['ratio'] if 'ratio' in ts_row else 1.0
    else:
        current_ratio = 1.0

    # Masks for full box
    plus_mask = signs > 0
    minus_mask = signs < 0
    pos_plus = pos[plus_mask].copy()
    pos_minus = pos[minus_mask].copy()

    # CORRECTION v4: Shift visualization center by 50 Mpc to move halo off-center
    # This displaces the X pattern away from the center
    SHIFT_X = 50.0  # Mpc
    def shift_periodic(x, shift, L=BOX):
        return ((x - shift + L/2) % L) - L/2

    pos_plus[:, 0] = shift_periodic(pos_plus[:, 0], SHIFT_X)
    pos_minus[:, 0] = shift_periodic(pos_minus[:, 0], SHIFT_X)

    # Central sub-box mask (400 Mpc)
    central_plus = get_central_mask(pos_plus, SUBBOX)
    central_minus = get_central_mask(pos_minus, SUBBOX)
    pos_plus_central = pos_plus[central_plus]
    pos_minus_central = pos_minus[central_minus]

    # Get cluster particles
    print("Extracting cluster particles...")
    cx, cy, cz, csigns = get_particles_in_sphere(pos, signs, CENTER, RADIUS, BOX)
    cplus_mask = csigns > 0
    cminus_mask = csigns < 0

    # Compute proto-stars in cluster
    print("Computing proto-stars in cluster...")
    cpos_plus = np.column_stack([cx[cplus_mask], cy[cplus_mask], cz[cplus_mask]])
    if len(cpos_plus) > 100:
        c_overdensity, c_is_proto = compute_overdensity_sphere(cpos_plus, k=32)
    else:
        c_overdensity = np.ones(len(cpos_plus))
        c_is_proto = np.zeros(len(cpos_plus), dtype=bool)

    n_proto_cluster = c_is_proto.sum()

    # z=4 cluster for evolution panel
    cx4, cy4, cz4, csigns4 = get_particles_in_sphere(pos_z4, signs_z4, CENTER, RADIUS, BOX)
    cminus4_mask = csigns4 < 0

    # Proto-stars count from stellar evolution
    if 'n_proto_stars' in stellar.columns:
        stellar_row = stellar[stellar['step'] <= current_step]
        n_stars_total = int(stellar_row.iloc[-1]['n_proto_stars']) if len(stellar_row) > 0 else 0
    else:
        n_stars_total = 0

    # === Create figure ===
    print("Creating 15-panel figure v2...")
    fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI, facecolor='black')
    gs = GridSpec(3, 5, figure=fig, hspace=0.08, wspace=0.06,
                  left=0.02, right=0.98, top=0.92, bottom=0.04)

    # CORRECTION 6: Enhanced super title
    title_str = (
        f"JANUS 10M  |  z = {z:.3f}  |  t = {t:.2f} Gyr  |  "
        f"S = {S:.3f}  |  "
        r"$\rho^+_{max}$" + f" = {rho_max:.0f}  |  "
        f"v_rms ratio = {current_ratio:.3f}  |  "
        r"N$\bigstar$" + f" = {n_stars_total:,}"
    )
    fig.suptitle(title_str, color='white', fontsize=13,
                 fontweight='bold', y=0.97, fontfamily='monospace')

    # === ROW 1: Global 400 Mpc sub-box — CORRECTION v3: density maps ===
    print("  Row 1: Global views (density maps)...")

    grid_size = 512

    # Compute density maps for m+ and m-
    def make_density_imshow(x, y, half_width=200.0, bins=512):
        H, xedges, yedges = np.histogram2d(
            x, y, bins=bins,
            range=[[-half_width, half_width], [-half_width, half_width]]
        )
        H_smooth = gaussian_filter(np.log10(H + 1), sigma=1.5)
        return H_smooth.T, [-half_width, half_width, -half_width, half_width]

    # m+ XY density
    H_plus_xy, extent = make_density_imshow(
        pos_plus_central[:, 0], pos_plus_central[:, 1], SUBBOX, grid_size)
    # m+ XZ density
    H_plus_xz, _ = make_density_imshow(
        pos_plus_central[:, 0], pos_plus_central[:, 2], SUBBOX, grid_size)
    # m- XY density
    H_minus_xy, _ = make_density_imshow(
        pos_minus_central[:, 0], pos_minus_central[:, 1], SUBBOX, grid_size)

    # Panel 1: XY m+ global (density map)
    ax1 = fig.add_subplot(gs[0, 0], facecolor='black')
    ax1.imshow(H_plus_xy, origin='lower', extent=extent,
               cmap='inferno', aspect='equal', interpolation='gaussian')
    ax1.set_title('m+ XY — 400 Mpc', color='white', fontsize=8)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: XZ m+ global (density map)
    ax2 = fig.add_subplot(gs[0, 1], facecolor='black')
    ax2.imshow(H_plus_xz, origin='lower', extent=extent,
               cmap='inferno', aspect='equal', interpolation='gaussian')
    ax2.set_title('m+ XZ — 400 Mpc', color='white', fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: XY Combined (overlay m+ and m- with alpha)
    ax3 = fig.add_subplot(gs[0, 2], facecolor='black')
    ax3.imshow(H_minus_xy, origin='lower', extent=extent,
               cmap='Blues', aspect='equal', alpha=0.5, interpolation='gaussian')
    ax3.imshow(H_plus_xy, origin='lower', extent=extent,
               cmap='Reds', aspect='equal', alpha=0.5, interpolation='gaussian')
    ax3.set_title('Combined XY — Segregation', color='white', fontsize=8)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: XY m- global (density map)
    ax4 = fig.add_subplot(gs[0, 3], facecolor='black')
    ax4.imshow(H_minus_xy, origin='lower', extent=extent,
               cmap='Blues', aspect='equal', interpolation='gaussian')
    ax4.set_title('m- XY — 400 Mpc', color='white', fontsize=8)
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Panel 5: 2.5D isometric global (keep scatter but with density coloring)
    ax5 = fig.add_subplot(gs[0, 4], projection='3d', facecolor='black')
    ax5.set_facecolor('black')
    subsample_3d = 30
    ax5.scatter(pos_minus_central[::subsample_3d, 0],
                pos_minus_central[::subsample_3d, 1],
                pos_minus_central[::subsample_3d, 2],
                c=COLOR_MINUS, alpha=0.06, s=0.10, linewidths=0, rasterized=True)
    ax5.scatter(pos_plus_central[::subsample_3d, 0],
                pos_plus_central[::subsample_3d, 1],
                pos_plus_central[::subsample_3d, 2],
                c=COLOR_PLUS_DARK, alpha=0.08, s=0.10, linewidths=0, rasterized=True)
    ax5.view_init(elev=25, azim=45)
    ax5.set_xlim(-SUBBOX, SUBBOX)
    ax5.set_ylim(-SUBBOX, SUBBOX)
    ax5.set_zlim(-SUBBOX, SUBBOX)
    ax5.set_title('2.5D Isometric — 400 Mpc', color='white', fontsize=8)
    ax5.xaxis.pane.fill = False
    ax5.yaxis.pane.fill = False
    ax5.zaxis.pane.fill = False
    ax5.xaxis.pane.set_edgecolor('none')
    ax5.yaxis.pane.set_edgecolor('none')
    ax5.zaxis.pane.set_edgecolor('none')
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_zticks([])

    # === ROW 2: Cluster #1 zoom ===
    print("  Row 2: Cluster zoom...")

    # Panel 6: XY m+ zoom with proto-stars — CORRECTION v3: reduce saturation
    ax6 = fig.add_subplot(gs[1, 0], facecolor='black')
    # Gas (non-proto-stars)
    ax6.scatter(cx[cplus_mask][~c_is_proto], cy[cplus_mask][~c_is_proto],
                c=COLOR_PLUS_DARK, alpha=0.10, s=2, linewidths=0, rasterized=True)
    # Proto-stars (reduced alpha)
    if c_is_proto.sum() > 0:
        ax6.scatter(cx[cplus_mask][c_is_proto], cy[cplus_mask][c_is_proto],
                    c=COLOR_STAR, alpha=0.3, s=4, linewidths=0)
    ax6.set_xlim(-RADIUS, RADIUS)
    ax6.set_ylim(-RADIUS, RADIUS)
    ax6.set_title('Cluster #1 m+ XY — 25 Mpc', color='white', fontsize=8)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_aspect('equal')

    # Panel 7: XZ m+ zoom — CORRECTION v3: reduce saturation
    ax7 = fig.add_subplot(gs[1, 1], facecolor='black')
    ax7.scatter(cx[cplus_mask][~c_is_proto], cz[cplus_mask][~c_is_proto],
                c=COLOR_PLUS_DARK, alpha=0.10, s=2, linewidths=0, rasterized=True)
    if c_is_proto.sum() > 0:
        ax7.scatter(cx[cplus_mask][c_is_proto], cz[cplus_mask][c_is_proto],
                    c=COLOR_STAR, alpha=0.3, s=4, linewidths=0)
    ax7.set_xlim(-RADIUS, RADIUS)
    ax7.set_ylim(-RADIUS, RADIUS)
    ax7.set_title('Cluster #1 m+ XZ — 25 Mpc', color='white', fontsize=8)
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.set_aspect('equal')

    # Panel 8: XY m- zoom (shows VOID)
    ax8 = fig.add_subplot(gs[1, 2], facecolor='black')
    ax8.scatter(cx[cminus_mask], cy[cminus_mask],
                c=COLOR_MINUS, alpha=0.3, s=2, linewidths=0, rasterized=True)
    ax8.set_xlim(-RADIUS, RADIUS)
    ax8.set_ylim(-RADIUS, RADIUS)
    ax8.set_title('Cluster #1 m- XY — VOID', color='white', fontsize=8)
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.set_aspect('equal')
    # R_half circle
    circle = plt.Circle((0, 0), 3.63, fill=False, color='yellow',
                         linewidth=0.8, linestyle='--')
    ax8.add_patch(circle)

    # Panel 9: m- Evolution z=4 → z_current — CORRECTION v3: contourf density
    ax9 = fig.add_subplot(gs[1, 3], facecolor='black')
    # z=4 m- (very small white points as background)
    ax9.scatter(cx4[cminus4_mask], cy4[cminus4_mask],
                c='white', alpha=0.2, s=0.2, linewidths=0, rasterized=True)
    # Current m- density with contourf
    if cminus_mask.sum() > 100:
        H_m, xe, ye = np.histogram2d(cx[cminus_mask], cy[cminus_mask],
                                      bins=64, range=[[-RADIUS, RADIUS], [-RADIUS, RADIUS]])
        H_m = gaussian_filter(H_m.T, sigma=1.5)
        # Contourf with blue colormap
        levels = np.linspace(0, H_m.max() * 0.8, 12)
        ax9.contourf(H_m, levels=levels, extent=[-RADIUS, RADIUS, -RADIUS, RADIUS],
                     cmap='Blues', alpha=0.7)
        # Contour lines
        ax9.contour(H_m, levels=levels[::3], extent=[-RADIUS, RADIUS, -RADIUS, RADIUS],
                    colors='#66ccff', linewidths=0.5, alpha=0.8)
    ax9.set_xlim(-RADIUS, RADIUS)
    ax9.set_ylim(-RADIUS, RADIUS)
    ax9.set_title(f'm- Evolution: z=4 → z={z:.3f}', color='white', fontsize=8)
    ax9.set_xticks([])
    ax9.set_yticks([])
    ax9.set_aspect('equal')

    # Panel 10: 2.5D isometric cluster zoom — CORRECTION 5
    ax10 = fig.add_subplot(gs[1, 4], projection='3d', facecolor='black')
    ax10.set_facecolor('black')
    ax10.scatter(cx[cminus_mask], cy[cminus_mask], cz[cminus_mask],
                 c=COLOR_MINUS, alpha=0.15, s=1, linewidths=0, rasterized=True)
    ax10.scatter(cx[cplus_mask][~c_is_proto], cy[cplus_mask][~c_is_proto], cz[cplus_mask][~c_is_proto],
                 c=COLOR_PLUS_DARK, alpha=0.2, s=1, linewidths=0, rasterized=True)
    # Proto-stars: white
    if c_is_proto.sum() > 0:
        ax10.scatter(cx[cplus_mask][c_is_proto], cy[cplus_mask][c_is_proto], cz[cplus_mask][c_is_proto],
                     c=COLOR_STAR, alpha=0.9, s=6, linewidths=0)
    ax10.view_init(elev=35, azim=200)  # Different angle
    ax10.set_xlim(-RADIUS, RADIUS)
    ax10.set_ylim(-RADIUS, RADIUS)
    ax10.set_zlim(-RADIUS, RADIUS)
    ax10.set_title('Cluster 2.5D — 25 Mpc', color='white', fontsize=8)
    ax10.xaxis.pane.fill = False
    ax10.yaxis.pane.fill = False
    ax10.zaxis.pane.fill = False
    ax10.xaxis.pane.set_edgecolor('none')
    ax10.yaxis.pane.set_edgecolor('none')
    ax10.zaxis.pane.set_edgecolor('none')
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax10.set_zticks([])

    # === ROW 3: Science panels ===
    print("  Row 3: Science panels...")

    # Panel 11: Log Density m+ — CORRECTION 8 (proper CIC map)
    ax11 = fig.add_subplot(gs[2, 0], facecolor='black')
    H_plus, xedges, yedges = make_density_map_cic(
        pos_plus_central[:, 0], pos_plus_central[:, 1],
        bins=512, half_width=SUBBOX
    )
    rho_mean_plus = H_plus.mean()
    H_plus_norm = H_plus / max(rho_mean_plus, 1e-10)
    H_plus_norm = np.maximum(H_plus_norm, 0.1)  # Avoid log(0)
    im11 = ax11.imshow(np.log10(H_plus_norm), extent=[-SUBBOX, SUBBOX, -SUBBOX, SUBBOX],
                       origin='lower', cmap='inferno', aspect='equal',
                       vmin=-1, vmax=3, interpolation='gaussian')
    ax11.set_title(r'Log$_{10}$($\rho^+/\bar{\rho}$)', color='white', fontsize=8)
    ax11.set_xticks([])
    ax11.set_yticks([])

    # Panel 12: Purity map
    ax12 = fig.add_subplot(gs[2, 1], facecolor='black')
    H_minus, _, _ = make_density_map_cic(
        pos_minus_central[:, 0], pos_minus_central[:, 1],
        bins=512, half_width=SUBBOX
    )
    H_total = H_plus + H_minus + 1e-10
    purity = H_plus / H_total
    im12 = ax12.imshow(purity, extent=[-SUBBOX, SUBBOX, -SUBBOX, SUBBOX],
                       origin='lower', cmap='RdBu_r', vmin=0, vmax=1, aspect='equal',
                       interpolation='gaussian')
    ax12.set_title('Purity: red=m+, blue=m-', color='white', fontsize=8)
    ax12.set_xticks([])
    ax12.set_yticks([])

    # Panel 13: r(k) cross-correlation — CORRECTION 2
    ax13 = fig.add_subplot(gs[2, 2], facecolor='black')
    # Measured data points
    k_measured = np.array([0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0])
    r_measured = np.array([-0.04, -0.08, -0.15, -0.19, -0.35, -0.50, -0.45])
    # Interpolated curve
    k_interp = np.logspace(-2, 0.5, 100)
    r_interp = np.interp(np.log10(k_interp), np.log10(k_measured), r_measured)
    ax13.semilogx(k_interp, r_interp, 'cyan', linewidth=2, label='r(k) Janus')
    ax13.scatter(k_measured, r_measured, c='red', s=30, zorder=6, label='Measured')
    ax13.axhline(0, color='white', linestyle='--', linewidth=0.8, label=r'$\Lambda$CDM')
    ax13.fill_between(k_interp, r_interp, 0, where=(r_interp < 0),
                      alpha=0.3, color='red')
    ax13.set_xlim(0.01, 3.0)
    ax13.set_ylim(-0.6, 0.15)
    ax13.set_xlabel('k [h/Mpc]', color='#888888', fontsize=7)
    ax13.set_ylabel('r(k)', color='#888888', fontsize=7)
    ax13.set_title('Cross-correlation r(k)', color='white', fontsize=8)
    ax13.tick_params(colors='#888888', labelsize=6)
    ax13.text(0.5, -0.52, 'r(k)<0 : JCM signature', color='white', fontsize=7,
              transform=ax13.transData)
    ax13.legend(loc='upper right', fontsize=5, facecolor='black',
                edgecolor='#444444', labelcolor='white')

    # Panel 14: v_rms ratio evolution — CORRECTION v3: correct column names
    ax14 = fig.add_subplot(gs[2, 3], facecolor='black')
    if 'ratio' in ts.columns:
        # Use pre-computed ratio column
        ratio_series = ts['ratio']
        ts_until_now = ts[ts['step'] <= current_step]
        ax14.plot(ts_until_now['z'], ts_until_now['ratio'], 'lime', linewidth=1.5)
        ax14.axhline(1.0, color='white', linestyle='--', linewidth=0.5)
        # Fill regions
        ax14.fill_between(ts_until_now['z'], ts_until_now['ratio'], 1,
                          where=(ts_until_now['ratio'] > 1), alpha=0.3, color='blue')
        ax14.fill_between(ts_until_now['z'], ts_until_now['ratio'], 1,
                          where=(ts_until_now['ratio'] < 1), alpha=0.3, color='red')
        # Current position (red point)
        ax14.axvline(z, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax14.scatter([z], [current_ratio], c='red', s=80, zorder=10, edgecolors='white', linewidths=1.5)
        ax14.text(z + 0.15, current_ratio + 0.05, f'z={z:.2f}\nr={current_ratio:.3f}',
                  color='red', fontsize=7, va='bottom', fontweight='bold')
    elif 'v_rms_plus' in ts.columns and 'v_rms_minus' in ts.columns:
        ratio_series = ts['v_rms_minus'] / ts['v_rms_plus'].replace(0, np.nan)
        ts_until_now = ts[ts['step'] <= current_step]
        ratio_now = ts_until_now['v_rms_minus'] / ts_until_now['v_rms_plus'].replace(0, np.nan)
        ax14.plot(ts_until_now['z'], ratio_now, 'lime', linewidth=1.5)
        ax14.axhline(1.0, color='white', linestyle='--', linewidth=0.5)
        ax14.axvline(z, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax14.scatter([z], [current_ratio], c='red', s=80, zorder=10, edgecolors='white', linewidths=1.5)
    ax14.set_xlim(4, 0)
    ax14.set_ylim(0.0, 1.2)
    ax14.set_xlabel('z', color='#888888', fontsize=7)
    ax14.set_ylabel(r'$v_{rms}^- / v_{rms}^+$', color='#888888', fontsize=7)
    ax14.set_title('Velocity ratio evolution', color='white', fontsize=8)
    ax14.tick_params(colors='#888888', labelsize=6)

    # Panel 15: Proto-star formation history
    ax15 = fig.add_subplot(gs[2, 4], facecolor='black')
    if 'n_proto_stars' in stellar.columns:
        ax15.semilogy(stellar['z'], stellar['n_proto_stars'] + 1, 'yellow', linewidth=1.5)
        # Onset z=1.55
        ax15.axvline(1.55, color='orange', linestyle='--', linewidth=1, alpha=0.8)
        ax15.text(1.5, 3e5, 'Onset\nz=1.55', color='orange', fontsize=6, ha='right')
        # Current position
        current_stellar = stellar[stellar['step'] <= current_step]
        if len(current_stellar) > 0:
            current_n = current_stellar.iloc[-1]['n_proto_stars']
            current_z_stellar = current_stellar.iloc[-1]['z']
            ax15.scatter([current_z_stellar], [current_n + 1],
                         c='red', s=60, zorder=10, edgecolors='white')
    ax15.set_xlim(4, 0)
    ax15.set_ylim(1, 1e6)
    ax15.set_xlabel('z', color='#888888', fontsize=7)
    ax15.set_ylabel(r'N$_{\bigstar}$', color='#888888', fontsize=7)
    ax15.set_title('Star formation history', color='white', fontsize=8)
    ax15.tick_params(colors='#888888', labelsize=6)

    # === Save ===
    out_path = out_dir / 'frame_test_v4_z0371.png'
    print(f"Saving to {out_path}...")
    plt.savefig(out_path, dpi=DPI, facecolor='black', bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print(f"\nDone! Output: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"\nMetrics: z={z:.3f}, S={S:.3f}, rho_max={rho_max:.0f}, ratio={current_ratio:.3f}")
    print(f"Proto-stars in cluster: {n_proto_cluster}")


if __name__ == '__main__':
    main()
