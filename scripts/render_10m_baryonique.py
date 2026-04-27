#!/usr/bin/env python3
"""
Renderer 10-panels pour JANUS Baryonique 10M
Basé sur render_phase2_nosph.py — même layout exact

Layout:
  Row 1: XY m+ | XZ m+ | XY m+ ZOOM + m- contours | XZ m+ ZOOM | Log Density m+
  Row 2: XY m- | XZ m- | m- Evolution z=5→z | XZ m- ZOOM | Purity XZ (±10 Mpc masqué)

Corrections:
- DZ = 4 Mpc fixe
- Bordure Purity ±10 Mpc → gris neutre
- Format binaire simple (pas JSNP)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from pathlib import Path
import struct
import time
import sys
from scipy.ndimage import gaussian_filter

# Configuration
SNAP_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/snapshots")
OUT_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/frames_10panel")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ZOOM_SIZE = 50.0  # Mpc
DZ = 4.0  # Fixed slice thickness
GRID_SIZE = 64
CONTOUR_BINS = 64
DENSITY_VMIN = 1
DENSITY_VMAX = 100
BOX_SIZE = 300.0
BORDER_MASK = 10.0  # Mpc to mask at edges for Purity
Z_INIT = 5.0

_z_init_cache = None

def read_snapshot(path):
    """Read simple binary format (no JSNP header)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    z = 1.0/a - 1.0 if a > 0 else 0
    return {
        'n': n, 'z': z, 'a': a, 'box': BOX_SIZE,
        'x': pos[:, 0], 'y': pos[:, 1], 'zpos': pos[:, 2],
        'sign': signs
    }

def get_z_init_reference():
    """Get initial snapshot (z=5) for evolution panel"""
    global _z_init_cache
    if _z_init_cache is None:
        z_init_path = SNAP_DIR / 'snap_00000.bin'
        if z_init_path.exists():
            _z_init_cache = read_snapshot(z_init_path)
    return _z_init_cache

def find_density_peak(x, y, z, box):
    """Find position of maximum density"""
    half = box / 2
    bins = np.linspace(-half, half, GRID_SIZE + 1)
    H, edges = np.histogramdd(np.column_stack([x, y, z]), bins=[bins, bins, bins])
    idx = np.unravel_index(np.argmax(H), H.shape)
    cell_size = box / GRID_SIZE
    x0 = edges[0][idx[0]] + cell_size / 2
    y0 = edges[1][idx[1]] + cell_size / 2
    z0 = edges[2][idx[2]] + cell_size / 2
    return x0, y0, z0, H.max()

def compute_density_grid(x, y, x0, y0, zoom_half, nbins=CONTOUR_BINS):
    """Compute 2D density histogram in zoom region"""
    mask = (np.abs(x - x0) < zoom_half) & (np.abs(y - y0) < zoom_half)
    xz, yz = x[mask], y[mask]
    bins = np.linspace(x0-zoom_half, x0+zoom_half, nbins+1)
    bins_y = np.linspace(y0-zoom_half, y0+zoom_half, nbins+1)
    H, _, _ = np.histogram2d(xz, yz, bins=[bins, bins_y])
    return gaussian_filter(H.T, sigma=1.0)

def compute_purity_xz_masked(x, zpos, sign, box):
    """Compute purity XZ with ±BORDER_MASK Mpc masked as gray"""
    half = box / 2
    nbins = 80
    bins = np.linspace(-half, half, nbins + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    mp = sign > 0
    mm = sign < 0

    H_plus, _, _ = np.histogram2d(x[mp], zpos[mp], bins=bins)
    H_minus, _, _ = np.histogram2d(x[mm], zpos[mm], bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        purity = (H_plus - H_minus) / (H_plus + H_minus + 1)
    purity = np.nan_to_num(purity, 0)

    # Mask border ±BORDER_MASK Mpc
    edge_limit = half - BORDER_MASK
    for i, cx in enumerate(centers):
        for j, cz in enumerate(centers):
            if abs(cx) > edge_limit or abs(cz) > edge_limit:
                purity[i, j] = np.nan

    return purity

def render_frame(snap_path, out_path):
    """Render 10-panel frame"""
    snap = read_snapshot(snap_path)
    n, z, box = snap['n'], snap['z'], snap['box']
    half = box / 2
    zoom_half = ZOOM_SIZE / 2

    mp = snap['sign'] > 0
    mm = snap['sign'] < 0
    n_plus, n_minus = mp.sum(), mm.sum()

    # Find density peak in m+
    x0, y0, z0, peak_density = find_density_peak(
        snap['x'][mp], snap['y'][mp], snap['zpos'][mp], box)

    fig = plt.figure(figsize=(20, 10), facecolor='#080810')
    step = int(snap_path.stem.split('_')[1])

    fig.suptitle(f'JANUS Baryonique 10M | Step {step} | z={z:.3f} | N={n/1e6:.1f}M | dz={DZ} Mpc',
                 fontsize=14, color='white', y=0.98)

    gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.2,
                          left=0.03, right=0.97, top=0.92, bottom=0.06)
    c_plus, c_minus = '#ff6644', '#4488ff'  # Orange-red / Blue
    extent = [x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half]
    extent_z = [x0-zoom_half, x0+zoom_half, z0-zoom_half, z0+zoom_half]

    # Subsample for scatter plots (performance)
    max_scatter = 300000
    sub_p = np.random.choice(mp.sum(), min(max_scatter, mp.sum()), replace=False) if mp.sum() > 0 else []
    sub_m = np.random.choice(mm.sum(), min(max_scatter, mm.sum()), replace=False) if mm.sum() > 0 else []

    # ========== ROW 1 - m+ ==========

    # Panel 0: XY m+ global
    ax = fig.add_subplot(gs[0, 0], facecolor='#0a0a14')
    ax.scatter(snap['x'][mp][sub_p], snap['y'][mp][sub_p],
               s=0.02, c=c_plus, alpha=0.4, rasterized=True)
    ax.add_patch(Rectangle((x0-zoom_half, y0-zoom_half), ZOOM_SIZE, ZOOM_SIZE,
                            fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title(f'XY m+ (N={n_plus:,})', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 1: XZ m+ global
    ax = fig.add_subplot(gs[0, 1], facecolor='#0a0a14')
    ax.scatter(snap['x'][mp][sub_p], snap['zpos'][mp][sub_p],
               s=0.02, c=c_plus, alpha=0.4, rasterized=True)
    ax.add_patch(Rectangle((x0-zoom_half, z0-zoom_half), ZOOM_SIZE, ZOOM_SIZE,
                            fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title('XZ m+', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 2: XY m+ ZOOM + m- contours
    ax = fig.add_subplot(gs[0, 2], facecolor='#0a0a14')
    mask_p = (np.abs(snap['x'][mp] - x0) < zoom_half) & (np.abs(snap['y'][mp] - y0) < zoom_half)
    xz_p, yz_p = snap['x'][mp][mask_p], snap['y'][mp][mask_p]
    if len(xz_p) > 80000:
        idx = np.random.choice(len(xz_p), 80000, replace=False)
        xz_p, yz_p = xz_p[idx], yz_p[idx]
    ax.scatter(xz_p, yz_p, s=0.15, c=c_plus, alpha=0.6, rasterized=True)

    # m- contours
    H_m = compute_density_grid(snap['x'][mm], snap['y'][mm], x0, y0, zoom_half, CONTOUR_BINS)
    if H_m.max() > 0:
        levels = [H_m.max() * p for p in [0.2, 0.4, 0.6, 0.8] if H_m.max() * p > 0]
        if levels:
            ax.contour(H_m, levels=levels, extent=extent,
                       colors='#6688ff', linewidths=[0.5, 0.8, 1.0, 1.5][:len(levels)], alpha=0.9)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(y0-zoom_half, y0+zoom_half)
    ax.set_aspect('equal')
    ax.set_title('XY m+ ZOOM + m- contours', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 3: XZ m+ ZOOM
    ax = fig.add_subplot(gs[0, 3], facecolor='#0a0a14')
    mask_pz = (np.abs(snap['x'][mp] - x0) < zoom_half) & (np.abs(snap['zpos'][mp] - z0) < zoom_half)
    xz_pz, zz_pz = snap['x'][mp][mask_pz], snap['zpos'][mp][mask_pz]
    if len(xz_pz) > 80000:
        idx = np.random.choice(len(xz_pz), 80000, replace=False)
        xz_pz, zz_pz = xz_pz[idx], zz_pz[idx]
    ax.scatter(xz_pz, zz_pz, s=0.15, c=c_plus, alpha=0.6, rasterized=True)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(z0-zoom_half, z0+zoom_half)
    ax.set_aspect('equal')
    ax.set_title('XZ m+ ZOOM', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 4: Log Density m+ (zoom)
    ax = fig.add_subplot(gs[0, 4], facecolor='#0a0a14')
    H = compute_density_grid(snap['x'][mp], snap['y'][mp], x0, y0, zoom_half, 64)
    H = np.clip(H, 0.1, None)
    im = ax.imshow(H, extent=extent, origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX),
                   aspect='equal', interpolation='bilinear')
    ax.set_title('Log Density m+', color='orange', fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ========== ROW 2 - m- ==========

    # Panel 5: XY m- global
    ax = fig.add_subplot(gs[1, 0], facecolor='#0a0a14')
    ax.scatter(snap['x'][mm][sub_m], snap['y'][mm][sub_m],
               s=0.02, c=c_minus, alpha=0.4, rasterized=True)
    ax.add_patch(Rectangle((x0-zoom_half, y0-zoom_half), ZOOM_SIZE, ZOOM_SIZE,
                            fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title(f'XY m- (N={n_minus:,})', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 6: XZ m- global
    ax = fig.add_subplot(gs[1, 1], facecolor='#0a0a14')
    ax.scatter(snap['x'][mm][sub_m], snap['zpos'][mm][sub_m],
               s=0.02, c=c_minus, alpha=0.4, rasterized=True)
    ax.add_patch(Rectangle((x0-zoom_half, z0-zoom_half), ZOOM_SIZE, ZOOM_SIZE,
                            fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title('XZ m-', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 7: m- Evolution z=5 → z_current
    ax = fig.add_subplot(gs[1, 2], facecolor='#0a0a14')
    H_now = compute_density_grid(snap['x'][mm], snap['y'][mm], x0, y0, zoom_half, CONTOUR_BINS)

    # Initial state contours (gray dashed)
    z_init_snap = get_z_init_reference()
    if z_init_snap is not None:
        mm_init = z_init_snap['sign'] < 0
        H_init = compute_density_grid(z_init_snap['x'][mm_init], z_init_snap['y'][mm_init],
                                      x0, y0, zoom_half, CONTOUR_BINS)
        if H_init.max() > 0:
            levels_init = [H_init.max() * p for p in [0.3, 0.5, 0.7] if H_init.max() * p > 0]
            if levels_init:
                ax.contour(H_init, levels=levels_init, extent=extent,
                          colors='#666666', linewidths=[0.5, 0.8, 1.0][:len(levels_init)],
                          alpha=0.6, linestyles='dashed')

    # Current state contours (solid)
    if H_now.max() > 0:
        levels_now = [H_now.max() * p for p in [0.3, 0.5, 0.7, 0.9] if H_now.max() * p > 0]
        if levels_now:
            ax.contour(H_now, levels=levels_now, extent=extent,
                      colors=c_minus, linewidths=[0.6, 0.9, 1.2, 1.5][:len(levels_now)], alpha=0.9)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(y0-zoom_half, y0+zoom_half)
    ax.set_aspect('equal')
    ax.set_title(f'm- Evolution: z={Z_INIT:.0f} → z={z:.2f}', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 8: XZ m- ZOOM
    ax = fig.add_subplot(gs[1, 3], facecolor='#0a0a14')
    mask_mz = (np.abs(snap['x'][mm] - x0) < zoom_half) & (np.abs(snap['zpos'][mm] - z0) < zoom_half)
    xz_mz, zz_mz = snap['x'][mm][mask_mz], snap['zpos'][mm][mask_mz]
    if len(xz_mz) > 80000:
        idx = np.random.choice(len(xz_mz), 80000, replace=False)
        xz_mz, zz_mz = xz_mz[idx], zz_mz[idx]
    ax.scatter(xz_mz, zz_mz, s=0.15, c=c_minus, alpha=0.6, rasterized=True)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(z0-zoom_half, z0+zoom_half)
    ax.set_aspect('equal')
    ax.set_title('XZ m- ZOOM', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Panel 9: Purity XZ (±10 Mpc masked)
    ax = fig.add_subplot(gs[1, 4], facecolor='#0a0a14')
    purity = compute_purity_xz_masked(snap['x'], snap['zpos'], snap['sign'], box)

    # Custom colormap with gray for NaN
    purity_cmap = plt.cm.RdBu.copy()
    purity_cmap.set_bad(color='#4a4a4a')  # Gray for masked regions

    im = ax.imshow(purity.T, extent=[-half, half, -half, half], origin='lower',
                   cmap=purity_cmap, vmin=-0.1, vmax=0.1, aspect='equal')

    # Draw border indicators
    edge = half - BORDER_MASK
    for e in [-edge, edge]:
        ax.axvline(e, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(e, color='white', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_title(f'Purity XZ (±{BORDER_MASK:.0f} Mpc masked)', color='white', fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Footer
    fig.text(0.5, 0.01,
             f"Halo: ({x0:.0f}, {y0:.0f}, {z0:.0f}) Mpc | Peak ρ={peak_density:.0f} | zoom={ZOOM_SIZE:.0f} Mpc",
             ha='center', fontsize=10, color='#888888', family='monospace')

    plt.savefig(out_path, dpi=150, facecolor='#080810', edgecolor='none')
    plt.close()

    return z, peak_density, (x0, y0, z0)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        snap_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
        out_path.parent.mkdir(exist_ok=True, parents=True)
        z, rho, halo = render_frame(snap_path, out_path)
        print(f"Rendered: {out_path} (z={z:.3f}, ρ_max={rho:.0f}, halo={halo})")
    elif len(sys.argv) > 1:
        step = int(sys.argv[1])
        snap_path = SNAP_DIR / f'snap_{step:05d}.bin'
        out_path = OUT_DIR / f'frame_{step:05d}.png'
        z, rho, halo = render_frame(snap_path, out_path)
        print(f"Rendered step {step}: z={z:.3f}, ρ_max={rho:.0f}")
    else:
        print("Usage: python render_10m_baryonique.py <step>")
        print("   or: python render_10m_baryonique.py <snap_path> <out_path>")
