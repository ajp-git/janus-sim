#!/usr/bin/env python3
"""
Janus Adaptive v2 — Multi-layout Render Daemon
Generates both 10-panel and 3-panel 2.5D frames from v3 snapshots
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import struct
import time
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# === CONFIGURATION ===
SNAP_DIR = Path("/mnt/T2/janus-sim/output/test_phase11_translation/snapshots")
OUT_10P = Path("/mnt/T2/janus-sim/output/test_phase11_translation/frames_10panel")
OUT_2P5D = Path("/mnt/T2/janus-sim/output/test_phase11_translation/frames_2p5d")
OUT_10P.mkdir(exist_ok=True, parents=True)
OUT_2P5D.mkdir(exist_ok=True, parents=True)

RESOLUTION_4K = (3840, 2160)
DPI = 200
ZOOM_SIZE = 100.0  # For 500 Mpc box (larger zoom to see structure better)
GRID_SIZE = 64
SUBSAMPLE = 200000  # Max particles to plot for performance
GLOBAL_RADIUS = 200.0  # Only show particles within this radius from center (removes edge artifacts)
MARGIN = 50.0  # Mpc — forbidden zone on each edge for zoom center search

# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOT READER (V3 format)
# ═══════════════════════════════════════════════════════════════════════════

def read_snapshot_v3_fast(path):
    """Fast reader for v3 snapshots using numpy"""
    with open(path, 'rb') as f:
        # Header (408 bytes)
        header = f.read(408)

        # Parse key fields
        n = struct.unpack('<Q', header[16:24])[0]
        a = struct.unpack('<d', header[24:32])[0]
        t_gyr = struct.unpack('<d', header[32:40])[0]
        l_box = struct.unpack('<d', header[40:48])[0]

        # Particle dtype (36 bytes)
        dt = np.dtype([
            ('pos', '<f4', 3),
            ('vel', '<f4', 3),
            ('mass', '<f4'),
            ('epsilon', '<f4'),
            ('sign', 'u1'),
            ('split_level', 'u1'),
            ('is_star', 'u1'),
            ('flags', 'u1'),
        ])

        particles = np.frombuffer(f.read(n * 36), dtype=dt)

    z = 1.0 / a - 1.0 if a > 0 else 0.0

    return {
        'n': n, 'a': a, 'z': z, 't_gyr': t_gyr, 'l_box': l_box,
        'pos': particles['pos'],
        'vel': particles['vel'],
        'sign': particles['sign'],
        'split_level': particles['split_level'],
        'is_star': particles['is_star'],
        'mass': particles['mass'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def find_density_peak(pos, box_size, n_cells=32):
    """Find position of maximum density in safe zone (away from edges)"""
    half = box_size / 2
    safe_limit = half - MARGIN  # Stay away from edges

    # Filter to safe zone only
    mask_safe = (
        (np.abs(pos[:, 0]) < safe_limit) &
        (np.abs(pos[:, 1]) < safe_limit) &
        (np.abs(pos[:, 2]) < safe_limit)
    )
    pos_safe = pos[mask_safe]

    if len(pos_safe) < 100:
        # Fallback to center if no particles in safe zone
        return np.array([0.0, 0.0, 0.0])

    # Search only in safe region
    bins = np.linspace(-safe_limit, safe_limit, n_cells + 1)
    H, _ = np.histogramdd(pos_safe, bins=[bins, bins, bins])
    idx = np.unravel_index(np.argmax(H), H.shape)
    cell = (2 * safe_limit) / n_cells
    return np.array([bins[idx[0]] + cell/2, bins[idx[1]] + cell/2, bins[idx[2]] + cell/2])


def subsample(arr, n_max):
    """Random subsample for plotting"""
    if len(arr) <= n_max:
        return arr
    idx = np.random.choice(len(arr), n_max, replace=False)
    return arr[idx]


def compute_density_2d(x, y, bins, sigma=1.5):
    """Compute smoothed 2D density"""
    H, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    return gaussian_filter(H.T, sigma=sigma)


def render_10panel(snap_path, step):
    """Render 10-panel layout"""
    data = read_snapshot_v3_fast(str(snap_path))
    pos = data['pos']
    sign = data['sign']
    z = data['z']
    l_box = data['l_box']
    split_level = data['split_level']

    half = l_box / 2

    # Separate by sign (255 = m-)
    is_plus = sign == 1
    is_minus = sign == 255
    pos_plus = pos[is_plus]
    pos_minus = pos[is_minus]

    # Find density peak in m+
    center = find_density_peak(pos_plus, l_box)
    zoom_half = ZOOM_SIZE / 2

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(19.2, 10.8), facecolor='black')
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06, wspace=0.20, hspace=0.20)

    # Title
    n_hr = np.sum(split_level > 0)
    fig.suptitle(f'JANUS ADAPTIVE — Step {step} | z = {z:.3f} | N = {data["n"]:,} | N_HR = {n_hr:,}',
                 color='white', fontsize=14)

    bins_global = np.linspace(-half, half, GRID_SIZE + 1)
    bins_zoom = np.linspace(-zoom_half, zoom_half, GRID_SIZE + 1)

    # === ROW 1: m+ ===
    for i, ax in enumerate(axes[0]):
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_color('gray')
        ax.tick_params(colors='gray', labelsize=7)
        ax.set_clip_on(True)

    # Apply radial mask for global views (removes edge artifacts)
    r_plus = np.sqrt(pos_plus[:,0]**2 + pos_plus[:,1]**2 + pos_plus[:,2]**2)
    pos_plus_inner = pos_plus[r_plus < GLOBAL_RADIUS]

    # 1. XY m+ global (r < 200 Mpc)
    p = subsample(pos_plus_inner, SUBSAMPLE)
    axes[0,0].scatter(p[:,0], p[:,1], s=0.01, c='#4488ff', alpha=0.3, rasterized=True, zorder=1)
    axes[0,0].set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[0,0].set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[0,0].set_title('m+ XY (r<200 Mpc)', color='#4488ff', fontsize=9)
    axes[0,0].set_aspect('equal')
    axes[0,0].set_zorder(10)

    # 2. XZ m+ global (r < 200 Mpc)
    axes[0,1].scatter(p[:,0], p[:,2], s=0.01, c='#4488ff', alpha=0.3, rasterized=True, zorder=1)
    axes[0,1].set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[0,1].set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[0,1].set_title('m+ XZ (r<200 Mpc)', color='#4488ff', fontsize=9)
    axes[0,1].set_aspect('equal')
    axes[0,1].set_zorder(10)

    # 3. XY m+ zoom with m- contours
    zoom_mask_p = (np.abs(pos_plus[:,0] - center[0]) < zoom_half) & \
                  (np.abs(pos_plus[:,1] - center[1]) < zoom_half)
    pz = subsample(pos_plus[zoom_mask_p] - center, min(50000, zoom_mask_p.sum()))
    axes[0,2].scatter(pz[:,0], pz[:,1], s=0.1, c='#4488ff', alpha=0.5, rasterized=True)

    # m- contours
    zoom_mask_m = (np.abs(pos_minus[:,0] - center[0]) < zoom_half) & \
                  (np.abs(pos_minus[:,1] - center[1]) < zoom_half)
    if zoom_mask_m.sum() > 100:
        pm = pos_minus[zoom_mask_m] - center
        H = compute_density_2d(pm[:,0], pm[:,1], bins_zoom)
        if H.max() > 0:
            levels = np.array([0.2, 0.4, 0.6, 0.8]) * H.max()
            axes[0,2].contour(bins_zoom[:-1], bins_zoom[:-1], H, levels=levels,
                             colors='#ff4444', alpha=0.7, linewidths=0.5)

    axes[0,2].set_xlim(-zoom_half, zoom_half)
    axes[0,2].set_ylim(-zoom_half, zoom_half)
    axes[0,2].set_title('m+ XY zoom + m- contours', color='#4488ff', fontsize=9)
    axes[0,2].set_aspect('equal')

    # 4. XZ m+ zoom
    zoom_mask_xz = (np.abs(pos_plus[:,0] - center[0]) < zoom_half) & \
                   (np.abs(pos_plus[:,2] - center[2]) < zoom_half)
    pz_xz = subsample(pos_plus[zoom_mask_xz] - center, min(50000, zoom_mask_xz.sum()))
    axes[0,3].scatter(pz_xz[:,0], pz_xz[:,2], s=0.1, c='#4488ff', alpha=0.5, rasterized=True)
    axes[0,3].set_xlim(-zoom_half, zoom_half)
    axes[0,3].set_ylim(-zoom_half, zoom_half)
    axes[0,3].set_title('m+ XZ zoom', color='#4488ff', fontsize=9)
    axes[0,3].set_aspect('equal')

    # 5. Log density m+ (zoom)
    if zoom_mask_p.sum() > 100:
        H = compute_density_2d(pz[:,0], pz[:,1], bins_zoom)
        im = axes[0,4].imshow(H, extent=[-zoom_half, zoom_half, -zoom_half, zoom_half],
                              origin='lower', cmap='inferno', norm=LogNorm(vmin=1, vmax=max(H.max(), 2)))
        axes[0,4].set_title('ρ(m+) zoom', color='orange', fontsize=9)
    else:
        axes[0,4].set_title('ρ(m+) zoom [N/A]', color='gray', fontsize=9)
    axes[0,4].set_xlim(-zoom_half, zoom_half)
    axes[0,4].set_ylim(-zoom_half, zoom_half)
    axes[0,4].set_aspect('equal')

    # === ROW 2: m- ===
    for i, ax in enumerate(axes[1]):
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_color('gray')
        ax.tick_params(colors='gray', labelsize=7)

    # Apply radial mask for m- global views
    r_minus = np.sqrt(pos_minus[:,0]**2 + pos_minus[:,1]**2 + pos_minus[:,2]**2)
    pos_minus_inner = pos_minus[r_minus < GLOBAL_RADIUS]

    # 6. XY m- global (r < 200 Mpc)
    m = subsample(pos_minus_inner, SUBSAMPLE)
    axes[1,0].scatter(m[:,0], m[:,1], s=0.01, c='#ff4444', alpha=0.3, rasterized=True, zorder=1)
    axes[1,0].set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[1,0].set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[1,0].set_title('m- XY (r<200 Mpc)', color='#ff4444', fontsize=9)
    axes[1,0].set_aspect('equal')
    axes[1,0].set_zorder(10)

    # 7. XZ m- global (r < 200 Mpc)
    axes[1,1].scatter(m[:,0], m[:,2], s=0.01, c='#ff4444', alpha=0.3, rasterized=True, zorder=1)
    axes[1,1].set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[1,1].set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    axes[1,1].set_title('m- XZ (r<200 Mpc)', color='#ff4444', fontsize=9)
    axes[1,1].set_aspect('equal')
    axes[1,1].set_zorder(10)

    # 8. XY m- zoom
    mz = subsample(pos_minus[zoom_mask_m] - center, min(50000, zoom_mask_m.sum())) if zoom_mask_m.sum() > 0 else np.empty((0,3))
    if len(mz) > 0:
        axes[1,2].scatter(mz[:,0], mz[:,1], s=0.1, c='#ff4444', alpha=0.5, rasterized=True)
    axes[1,2].set_xlim(-zoom_half, zoom_half)
    axes[1,2].set_ylim(-zoom_half, zoom_half)
    axes[1,2].set_title('m- XY zoom', color='#ff4444', fontsize=9)
    axes[1,2].set_aspect('equal')

    # 9. XZ m- zoom
    zoom_mask_m_xz = (np.abs(pos_minus[:,0] - center[0]) < zoom_half) & \
                     (np.abs(pos_minus[:,2] - center[2]) < zoom_half)
    mz_xz = subsample(pos_minus[zoom_mask_m_xz] - center, min(50000, zoom_mask_m_xz.sum())) if zoom_mask_m_xz.sum() > 0 else np.empty((0,3))
    if len(mz_xz) > 0:
        axes[1,3].scatter(mz_xz[:,0], mz_xz[:,2], s=0.1, c='#ff4444', alpha=0.5, rasterized=True)
    axes[1,3].set_xlim(-zoom_half, zoom_half)
    axes[1,3].set_ylim(-zoom_half, zoom_half)
    axes[1,3].set_title('m- XZ zoom', color='#ff4444', fontsize=9)
    axes[1,3].set_aspect('equal')

    # 10. Split level distribution (replacing purity)
    split_counts = [np.sum(split_level == i) for i in range(6)]
    colors = ['#888888', '#44ff44', '#ffff44', '#ff8844', '#ff4444', '#ff00ff']
    bars = axes[1,4].bar(range(6), split_counts, color=colors)
    axes[1,4].set_xlabel('Split level', color='white', fontsize=8)
    axes[1,4].set_ylabel('N particles', color='white', fontsize=8)
    axes[1,4].set_title('Split distribution', color='white', fontsize=9)
    axes[1,4].set_yscale('log')
    axes[1,4].set_ylim(1, data['n'])

    # Save
    out_path = OUT_10P / f'frame_{step:05d}.png'
    fig.savefig(out_path, dpi=DPI, facecolor='black')
    plt.close(fig)
    return out_path


def render_2p5d(snap_path, step):
    """Render 2.5D layout: left=combined full height, right=m+ top / m- bottom"""
    from matplotlib.gridspec import GridSpec

    data = read_snapshot_v3_fast(str(snap_path))
    pos = data['pos']
    sign = data['sign']
    z = data['z']
    l_box = data['l_box']

    half = l_box / 2
    is_plus = sign == 1
    is_minus = sign == 255
    pos_plus = pos[is_plus]
    pos_minus = pos[is_minus]

    # Apply radial mask (r < 200 Mpc) to remove edge artifacts
    r_plus = np.sqrt(pos_plus[:,0]**2 + pos_plus[:,1]**2 + pos_plus[:,2]**2)
    r_minus = np.sqrt(pos_minus[:,0]**2 + pos_minus[:,1]**2 + pos_minus[:,2]**2)
    pos_plus_inner = pos_plus[r_plus < GLOBAL_RADIUS]
    pos_minus_inner = pos_minus[r_minus < GLOBAL_RADIUS]

    # Subsample for 3D rendering (from inner region only)
    pp = subsample(pos_plus_inner, 150000)
    pm = subsample(pos_minus_inner, 100000)

    # 4K figure: 3840x2160 at 200 DPI = 19.2 x 10.8 inches
    fig = plt.figure(figsize=(19.2, 10.8), facecolor='black')

    # GridSpec: 2 rows, 3 columns
    # Left panel (combined): spans all rows, 2 columns
    # Right panels: 1 column each, top=m+, bottom=m-
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 2, 1.5],
                  height_ratios=[1, 1], wspace=0.05, hspace=0.08)

    azim = 45 + step * 0.15  # Slow rotation
    elev = 20

    # === LEFT: Combined m+/m- (full height, 2/3 width) ===
    ax_main = fig.add_subplot(gs[:, :2], projection='3d', facecolor='black')
    ax_main.scatter(pm[:,0], pm[:,1], pm[:,2], s=0.15, c='#ff4444', alpha=0.4, rasterized=True)
    ax_main.scatter(pp[:,0], pp[:,1], pp[:,2], s=0.15, c='#44aaff', alpha=0.4, rasterized=True)
    ax_main.set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_main.set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_main.set_zlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_main.set_title(f'JANUS BIMETRIC — Step {step} | z = {z:.2f}',
                      color='white', fontsize=16, pad=10)
    ax_main.xaxis.pane.fill = False
    ax_main.yaxis.pane.fill = False
    ax_main.zaxis.pane.fill = False
    ax_main.xaxis.pane.set_edgecolor('#333333')
    ax_main.yaxis.pane.set_edgecolor('#333333')
    ax_main.zaxis.pane.set_edgecolor('#333333')
    ax_main.tick_params(colors='#666666', labelsize=8)
    ax_main.set_xlabel('X [Mpc]', color='#888888', fontsize=9)
    ax_main.set_ylabel('Y [Mpc]', color='#888888', fontsize=9)
    ax_main.set_zlabel('Z [Mpc]', color='#888888', fontsize=9)
    ax_main.view_init(elev=elev, azim=azim)

    # Legend (show inner region counts)
    ax_main.text2D(0.02, 0.98, f'N (r<{GLOBAL_RADIUS:.0f}) = {len(pos_plus_inner) + len(pos_minus_inner):,}',
                   transform=ax_main.transAxes, color='white', fontsize=11, va='top')
    ax_main.text2D(0.02, 0.94, f'm+ (blue): {len(pos_plus_inner):,}', transform=ax_main.transAxes,
                   color='#44aaff', fontsize=10, va='top')
    ax_main.text2D(0.02, 0.90, f'm- (red): {len(pos_minus_inner):,}', transform=ax_main.transAxes,
                   color='#ff4444', fontsize=10, va='top')

    # === TOP RIGHT: m+ only ===
    ax_plus = fig.add_subplot(gs[0, 2], projection='3d', facecolor='black')
    ax_plus.scatter(pp[:,0], pp[:,1], pp[:,2], s=0.015, c='#44aaff', alpha=0.25, rasterized=True)
    ax_plus.set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_plus.set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_plus.set_zlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_plus.set_title('m+ (baryonic matter)', color='#44aaff', fontsize=11)
    ax_plus.xaxis.pane.fill = False
    ax_plus.yaxis.pane.fill = False
    ax_plus.zaxis.pane.fill = False
    ax_plus.xaxis.pane.set_edgecolor('#222244')
    ax_plus.yaxis.pane.set_edgecolor('#222244')
    ax_plus.zaxis.pane.set_edgecolor('#222244')
    ax_plus.tick_params(colors='#444466', labelsize=6)
    ax_plus.view_init(elev=elev, azim=azim)

    # === BOTTOM RIGHT: m- only ===
    ax_minus = fig.add_subplot(gs[1, 2], projection='3d', facecolor='black')
    ax_minus.scatter(pm[:,0], pm[:,1], pm[:,2], s=0.015, c='#ff4444', alpha=0.25, rasterized=True)
    ax_minus.set_xlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_minus.set_ylim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_minus.set_zlim(-GLOBAL_RADIUS, GLOBAL_RADIUS)
    ax_minus.set_title('m- (negative mass)', color='#ff4444', fontsize=11)
    ax_minus.xaxis.pane.fill = False
    ax_minus.yaxis.pane.fill = False
    ax_minus.zaxis.pane.fill = False
    ax_minus.xaxis.pane.set_edgecolor('#442222')
    ax_minus.yaxis.pane.set_edgecolor('#442222')
    ax_minus.zaxis.pane.set_edgecolor('#442222')
    ax_minus.tick_params(colors='#664444', labelsize=6)
    ax_minus.view_init(elev=elev, azim=azim)

    out_path = OUT_2P5D / f'frame_{step:05d}.png'
    fig.savefig(out_path, dpi=200, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    return out_path


def render_snapshot(snap_path):
    """Render both layouts for a snapshot"""
    step = int(snap_path.stem.split('_')[1])

    # Check if already rendered
    out_10p = OUT_10P / f'frame_{step:05d}.png'
    out_2p5d = OUT_2P5D / f'frame_{step:05d}.png'

    results = []
    if not out_10p.exists():
        try:
            results.append(('10P', render_10panel(snap_path, step)))
        except Exception as e:
            results.append(('10P', f'ERROR: {e}'))

    if not out_2p5d.exists():
        try:
            results.append(('2.5D', render_2p5d(snap_path, step)))
        except Exception as e:
            results.append(('2.5D', f'ERROR: {e}'))

    return step, results


def main():
    print(f"=== Janus Adaptive v2 Render Daemon ===")
    print(f"Snap dir: {SNAP_DIR}")
    print(f"10-panel: {OUT_10P}")
    print(f"2.5D: {OUT_2P5D}")
    print()

    rendered = set()

    while True:
        # Find new snapshots
        snaps = sorted(SNAP_DIR.glob('snap_*.bin'))
        new_snaps = [s for s in snaps if s.name not in rendered]

        for snap in new_snaps:
            step = int(snap.stem.split('_')[1])
            print(f"[{time.strftime('%H:%M:%S')}] Rendering step {step}...", end=' ', flush=True)

            try:
                step, results = render_snapshot(snap)
                for layout, result in results:
                    print(f"{layout}:OK", end=' ')
                print()
            except Exception as e:
                print(f"ERROR: {e}")

            rendered.add(snap.name)

        time.sleep(30)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--oneshot':
        # Process all existing snapshots
        snaps = sorted(SNAP_DIR.glob('snap_*.bin'))
        print(f"Processing {len(snaps)} snapshots...")
        for snap in snaps:
            step, results = render_snapshot(snap)
            print(f"Step {step}: {results}")
    else:
        main()
