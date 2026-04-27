#!/usr/bin/env python3
"""
JANUS 10-Panel 4K Renderer — Layout Original
=============================================
Rangée 1:
  1. XY m+ global
  2. XZ m+ global
  3. XY m+ ZOOM + contours m-
  4. XZ m+ ZOOM
  5. Log Density m+ (zoom)

Rangée 2:
  6. XY m- global
  7. XZ m- global
  8. m- Evolution z_init→z_current (zoom, contours)
  9. XZ m- ZOOM
  10. Purity XZ (±10 Mpc masqué)

Corrections:
- DZ = 4 Mpc fixe
- Bordure Purity XZ: ±10 Mpc masqué gris
- Cellules vides (N < 5) → gris neutre
- Résolution 4K (3840×2160)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import struct
from pathlib import Path
import json

# === CONFIGURATION ===
DZ = 4.0  # Fixed slice thickness [Mpc]
ZOOM_SIZE = 50.0  # Provisional zoom size [Mpc]
MIN_PARTICLES_CELL = 5  # Minimum particles for valid purity
BORDER_MASK = 10.0  # Border mask for XZ purity [Mpc]
GRAY_NEUTRAL = 0.3  # Neutral gray value
RESOLUTION = (3840, 2160)  # 4K
DPI = 200
Z_INIT = 5.0  # Initial redshift for evolution panel

def load_snapshot(path):
    """Load binary snapshot"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)
    return pos, vel, signs, a, t

def find_dominant_halo(pos, signs, box_size, n_cells=32):
    """Find position of maximum m+ density"""
    half = box_size / 2
    bins = np.linspace(-half, half, n_cells + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    pos_plus = pos[signs > 0]
    rho_plus, _ = np.histogramdd(pos_plus, bins=[bins, bins, bins])

    idx = np.unravel_index(np.argmax(rho_plus), rho_plus.shape)
    center = np.array([centers[idx[0]], centers[idx[1]], centers[idx[2]]])

    return center, rho_plus.max()

def get_zoom_particles(pos, signs, center, zoom_size):
    """Extract particles in zoom region"""
    half = zoom_size / 2
    mask = (
        (np.abs(pos[:, 0] - center[0]) < half) &
        (np.abs(pos[:, 1] - center[1]) < half) &
        (np.abs(pos[:, 2] - center[2]) < half)
    )
    return pos[mask] - center, signs[mask]

def compute_density_2d(pos, box_size, n_cells=128, projection='xy'):
    """Compute 2D density histogram"""
    half = box_size / 2
    bins = np.linspace(-half, half, n_cells + 1)

    if projection == 'xy':
        h, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[bins, bins])
    else:  # xz
        h, _, _ = np.histogram2d(pos[:, 0], pos[:, 2], bins=[bins, bins])

    return h.T, bins  # Transpose for imshow

def compute_purity_xz(pos, signs, box_size, n_cells=64):
    """Compute purity map for XZ projection with border masking"""
    half = box_size / 2
    bins = np.linspace(-half, half, n_cells + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    pos_plus = pos[signs > 0]
    pos_minus = pos[signs < 0]

    h_plus, _, _ = np.histogram2d(pos_plus[:, 0], pos_plus[:, 2], bins=[bins, bins])
    h_minus, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 2], bins=[bins, bins])

    total = h_plus + h_minus
    purity = np.abs(h_plus - h_minus) / np.maximum(total, 1)

    # Mask empty cells
    purity[total < MIN_PARTICLES_CELL] = np.nan

    # Apply border mask (±BORDER_MASK Mpc from edges)
    edge_limit = half - BORDER_MASK
    for i, cx in enumerate(centers):
        for j, cz in enumerate(centers):
            if abs(cx) > edge_limit or abs(cz) > edge_limit:
                purity[i, j] = np.nan

    return purity.T, bins

def render_10panel(snap_path, out_path, box_size=300.0, snap_init_path=None):
    """
    Render 10-panel 4K frame with original layout
    """
    pos, vel, signs, a, t = load_snapshot(snap_path)
    z = 1.0/a - 1.0 if a > 0 else 0

    # Find dominant halo for zoom
    halo_center, rho_max = find_dominant_halo(pos, signs, box_size)

    half = box_size / 2
    zoom_half = ZOOM_SIZE / 2

    # Global slices (DZ fixed)
    mask_xy_global = np.abs(pos[:, 2]) < DZ / 2
    mask_xz_global = np.abs(pos[:, 1]) < DZ / 2

    # Zoom region particles
    pos_zoom, signs_zoom = get_zoom_particles(pos, signs, halo_center, ZOOM_SIZE)
    mask_xy_zoom = np.abs(pos_zoom[:, 2]) < DZ / 2
    mask_xz_zoom = np.abs(pos_zoom[:, 1]) < DZ / 2

    # Separate by sign - Global
    pos_global_xy = pos[mask_xy_global]
    signs_global_xy = signs[mask_xy_global]
    pos_global_xz = pos[mask_xz_global]
    signs_global_xz = signs[mask_xz_global]

    xy_plus_global = pos_global_xy[signs_global_xy > 0]
    xy_minus_global = pos_global_xy[signs_global_xy < 0]
    xz_plus_global = pos_global_xz[signs_global_xz > 0]
    xz_minus_global = pos_global_xz[signs_global_xz < 0]

    # Separate by sign - Zoom
    pos_zoom_xy = pos_zoom[mask_xy_zoom]
    signs_zoom_xy = signs_zoom[mask_xy_zoom]
    pos_zoom_xz = pos_zoom[mask_xz_zoom]
    signs_zoom_xz = signs_zoom[mask_xz_zoom]

    xy_plus_zoom = pos_zoom_xy[signs_zoom_xy > 0]
    xy_minus_zoom = pos_zoom_xy[signs_zoom_xy < 0]
    xz_plus_zoom = pos_zoom_xz[signs_zoom_xz > 0]
    xz_minus_zoom = pos_zoom_xz[signs_zoom_xz < 0]

    # Load initial snapshot for evolution panel if available
    if snap_init_path and Path(snap_init_path).exists():
        pos_init, _, signs_init, a_init, _ = load_snapshot(snap_init_path)
        pos_init_zoom, signs_init_zoom = get_zoom_particles(pos_init, signs_init, halo_center, ZOOM_SIZE)
        mask_xy_init = np.abs(pos_init_zoom[:, 2]) < DZ / 2
        xy_minus_init = pos_init_zoom[mask_xy_init][signs_init_zoom[mask_xy_init] < 0]
        has_init = True
    else:
        has_init = False

    # Create figure
    fig = plt.figure(figsize=(RESOLUTION[0]/DPI, RESOLUTION[1]/DPI), dpi=DPI, facecolor='black')

    axes = []
    for row in range(2):
        for col in range(5):
            ax = fig.add_subplot(2, 5, row * 5 + col + 1)
            ax.set_facecolor('black')
            axes.append(ax)

    # Particle sizes
    s_global = max(0.05, min(0.5, 2000 / np.sqrt(len(pos_global_xy) + 1)))
    s_zoom = max(0.3, min(3.0, 3000 / np.sqrt(len(pos_zoom_xy) + 1)))

    # === ROW 1: m+ ===

    # Panel 0: XY m+ global
    axes[0].scatter(xy_plus_global[:, 0], xy_plus_global[:, 1], s=s_global, c='orangered', alpha=0.5)
    axes[0].set_xlim(-half, half)
    axes[0].set_ylim(-half, half)
    axes[0].set_title(f'XY m+ global (N={len(xy_plus_global)})', color='white', fontsize=8)
    # Mark zoom region
    rect = plt.Rectangle((halo_center[0] - zoom_half, halo_center[1] - zoom_half),
                          ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=0.5)
    axes[0].add_patch(rect)

    # Panel 1: XZ m+ global
    axes[1].scatter(xz_plus_global[:, 0], xz_plus_global[:, 2], s=s_global, c='orangered', alpha=0.5)
    axes[1].set_xlim(-half, half)
    axes[1].set_ylim(-half, half)
    axes[1].set_title(f'XZ m+ global (N={len(xz_plus_global)})', color='white', fontsize=8)
    rect = plt.Rectangle((halo_center[0] - zoom_half, halo_center[2] - zoom_half),
                          ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=0.5)
    axes[1].add_patch(rect)

    # Panel 2: XY m+ ZOOM + contours m-
    axes[2].scatter(xy_plus_zoom[:, 0], xy_plus_zoom[:, 1], s=s_zoom, c='orangered', alpha=0.7)
    # Add m- contours
    if len(xy_minus_zoom) > 100:
        rho_minus, bins = compute_density_2d(xy_minus_zoom, ZOOM_SIZE, n_cells=32, projection='xy')
        rho_smooth = gaussian_filter(rho_minus, sigma=1)
        extent = [-zoom_half, zoom_half, -zoom_half, zoom_half]
        levels = np.percentile(rho_smooth[rho_smooth > 0], [50, 75, 90]) if rho_smooth.max() > 0 else [1]
        axes[2].contour(rho_smooth, levels=levels, extent=extent, colors='dodgerblue', linewidths=0.5, alpha=0.7)
    axes[2].set_xlim(-zoom_half, zoom_half)
    axes[2].set_ylim(-zoom_half, zoom_half)
    axes[2].set_title(f'XY m+ zoom + m- contours', color='white', fontsize=8)

    # Panel 3: XZ m+ ZOOM
    axes[3].scatter(xz_plus_zoom[:, 0], xz_plus_zoom[:, 2], s=s_zoom, c='orangered', alpha=0.7)
    axes[3].set_xlim(-zoom_half, zoom_half)
    axes[3].set_ylim(-zoom_half, zoom_half)
    axes[3].set_title(f'XZ m+ zoom (N={len(xz_plus_zoom)})', color='white', fontsize=8)

    # Panel 4: Log Density m+ (zoom)
    if len(xy_plus_zoom) > 10:
        rho_plus, bins = compute_density_2d(xy_plus_zoom, ZOOM_SIZE, n_cells=64, projection='xy')
        extent = [-zoom_half, zoom_half, -zoom_half, zoom_half]
        im = axes[4].imshow(rho_plus + 1, extent=extent, origin='lower',
                           cmap='hot', norm=LogNorm(vmin=1, vmax=max(2, rho_plus.max())))
    axes[4].set_xlim(-zoom_half, zoom_half)
    axes[4].set_ylim(-zoom_half, zoom_half)
    axes[4].set_title('log ρ+ (zoom)', color='white', fontsize=8)

    # === ROW 2: m- ===

    # Panel 5: XY m- global
    axes[5].scatter(xy_minus_global[:, 0], xy_minus_global[:, 1], s=s_global, c='dodgerblue', alpha=0.5)
    axes[5].set_xlim(-half, half)
    axes[5].set_ylim(-half, half)
    axes[5].set_title(f'XY m- global (N={len(xy_minus_global)})', color='white', fontsize=8)
    rect = plt.Rectangle((halo_center[0] - zoom_half, halo_center[1] - zoom_half),
                          ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=0.5)
    axes[5].add_patch(rect)

    # Panel 6: XZ m- global
    axes[6].scatter(xz_minus_global[:, 0], xz_minus_global[:, 2], s=s_global, c='dodgerblue', alpha=0.5)
    axes[6].set_xlim(-half, half)
    axes[6].set_ylim(-half, half)
    axes[6].set_title(f'XZ m- global (N={len(xz_minus_global)})', color='white', fontsize=8)
    rect = plt.Rectangle((halo_center[0] - zoom_half, halo_center[2] - zoom_half),
                          ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=0.5)
    axes[6].add_patch(rect)

    # Panel 7: m- Evolution z_init → z_current (zoom, contours)
    if has_init and len(xy_minus_init) > 100:
        # Initial state contours (gray)
        rho_init, _ = compute_density_2d(xy_minus_init, ZOOM_SIZE, n_cells=32, projection='xy')
        rho_init_smooth = gaussian_filter(rho_init, sigma=1)
        extent = [-zoom_half, zoom_half, -zoom_half, zoom_half]
        levels_init = np.percentile(rho_init_smooth[rho_init_smooth > 0], [50, 75, 90]) if rho_init_smooth.max() > 0 else [1]
        axes[7].contour(rho_init_smooth, levels=levels_init, extent=extent, colors='gray', linewidths=0.5, alpha=0.5, linestyles='dashed')
    # Current state (solid blue)
    axes[7].scatter(xy_minus_zoom[:, 0], xy_minus_zoom[:, 1], s=s_zoom*0.7, c='dodgerblue', alpha=0.6)
    if len(xy_minus_zoom) > 100:
        rho_curr, _ = compute_density_2d(xy_minus_zoom, ZOOM_SIZE, n_cells=32, projection='xy')
        rho_curr_smooth = gaussian_filter(rho_curr, sigma=1)
        levels_curr = np.percentile(rho_curr_smooth[rho_curr_smooth > 0], [50, 75, 90]) if rho_curr_smooth.max() > 0 else [1]
        axes[7].contour(rho_curr_smooth, levels=levels_curr, extent=extent, colors='cyan', linewidths=0.8, alpha=0.8)
    axes[7].set_xlim(-zoom_half, zoom_half)
    axes[7].set_ylim(-zoom_half, zoom_half)
    init_label = f'z={Z_INIT:.0f}' if has_init else 'N/A'
    axes[7].set_title(f'm- evolution ({init_label}→z={z:.2f})', color='white', fontsize=8)

    # Panel 8: XZ m- ZOOM
    axes[8].scatter(xz_minus_zoom[:, 0], xz_minus_zoom[:, 2], s=s_zoom, c='dodgerblue', alpha=0.7)
    axes[8].set_xlim(-zoom_half, zoom_half)
    axes[8].set_ylim(-zoom_half, zoom_half)
    axes[8].set_title(f'XZ m- zoom (N={len(xz_minus_zoom)})', color='white', fontsize=8)

    # Panel 9: Purity XZ (±10 Mpc masqué)
    purity, pbins = compute_purity_xz(pos, signs, box_size, n_cells=64)
    purity_cmap = plt.cm.RdBu_r.copy()
    purity_cmap.set_bad(color=(GRAY_NEUTRAL, GRAY_NEUTRAL, GRAY_NEUTRAL))
    axes[9].imshow(purity, extent=[-half, half, -half, half], origin='lower',
                   cmap=purity_cmap, vmin=0, vmax=1)
    # Border indicators
    edge = half - BORDER_MASK
    for e in [-edge, edge]:
        axes[9].axvline(e, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
        axes[9].axhline(e, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    axes[9].set_xlim(-half, half)
    axes[9].set_ylim(-half, half)
    axes[9].set_title(f'Purity XZ (±{BORDER_MASK:.0f}Mpc masked)', color='white', fontsize=8)

    # Formatting
    for i, ax in enumerate(axes):
        ax.set_aspect('equal')
        ax.tick_params(colors='white', labelsize=5)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.3)

        # Labels
        if i in [0, 5]:
            ax.set_ylabel('Y [Mpc]', color='white', fontsize=7)
        if i in [1, 6, 9]:
            ax.set_ylabel('Z [Mpc]', color='white', fontsize=7)
        if i >= 5:
            ax.set_xlabel('X [Mpc]', color='white', fontsize=7)

    # Super title
    step = int(Path(snap_path).stem.split('_')[1])
    fig.suptitle(f'JANUS 10M | Step {step} | z={z:.3f} | a={a:.4f} | dz={DZ} Mpc | zoom={ZOOM_SIZE:.0f} Mpc',
                 color='white', fontsize=11, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=DPI, facecolor='black', bbox_inches='tight')
    plt.close()

    return z, len(pos), halo_center

def render_zoom_series(snap_path, out_dir, box_size=300.0):
    """Generate zoom series at 20, 50, 100 Mpc centered on dominant halo"""
    global ZOOM_SIZE

    pos, vel, signs, a, t = load_snapshot(snap_path)
    z = 1.0/a - 1.0 if a > 0 else 0

    center, rho_max = find_dominant_halo(pos, signs, box_size)
    print(f"Dominant halo at {center}, ρ_max = {rho_max:.0f}")

    # Save halo info
    halo_info = {
        "center": center.tolist(),
        "rho_max": float(rho_max),
        "z": float(z)
    }
    with open(out_dir / "dominant_halo.json", 'w') as f:
        json.dump(halo_info, f, indent=2)

    # Generate zoom frames
    for zoom_size in [20, 50, 100]:
        ZOOM_SIZE = zoom_size
        out_path = out_dir / f"zoom_{zoom_size}mpc_z{z:.2f}.png"
        render_10panel(snap_path, out_path, box_size)
        print(f"  Saved: {out_path}")

    return center, halo_info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='JANUS 10-Panel 4K Renderer')
    parser.add_argument('snap_path', help='Path to snapshot')
    parser.add_argument('out_path', help='Output path for frame')
    parser.add_argument('--box', type=float, default=300.0, help='Box size [Mpc]')
    parser.add_argument('--zoom', action='store_true', help='Generate zoom series')
    parser.add_argument('--init', type=str, default=None, help='Initial snapshot for evolution')
    args = parser.parse_args()

    snap_path = Path(args.snap_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.zoom:
        render_zoom_series(snap_path, out_path.parent, args.box)
    else:
        z, n, center = render_10panel(snap_path, out_path, args.box, args.init)
        print(f"Rendered: {out_path} (z={z:.3f}, N={n}, halo={center})")
