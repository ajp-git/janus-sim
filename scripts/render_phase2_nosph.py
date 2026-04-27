#!/usr/bin/env python3
"""Render daemon for VSL Phase 2 NO SPH (asymmetric softening only)"""
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

SNAP_DIR = Path("/mnt/T2/janus-sim/output/vsl_phase2_nosph/snapshots")
OUT_DIR = Path("/mnt/T2/janus-sim/output/vsl_phase2_nosph/frames")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ZOOM_SIZE = 50.0  # 50 Mpc zoom for 500 Mpc box
GRID_SIZE = 64
CONTOUR_BINS = 64
DENSITY_VMIN = 1
DENSITY_VMAX = 50
BOX_SIZE = 500.0

_z4_cache = None

def read_snapshot(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic: {magic}")
        struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        box = struct.unpack('<d', f.read(8))[0]
        dt = np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('sign','i1'),('type','u1')])
        data = np.frombuffer(f.read(n*26), dtype=dt)
    return {'n': n, 'z': z, 'box': box, 'x': data['x'], 'y': data['y'],
            'zpos': data['z'], 'sign': data['sign'], 'type': data['type']}

def get_z4_reference():
    global _z4_cache
    if _z4_cache is None:
        z4_path = SNAP_DIR / 'snap_000000.bin'
        if z4_path.exists():
            _z4_cache = read_snapshot(z4_path)
    return _z4_cache

def find_density_peak(x, y, z, box):
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
    mask = (np.abs(x - x0) < zoom_half) & (np.abs(y - y0) < zoom_half)
    xz, yz = x[mask], y[mask]
    bins = np.linspace(x0-zoom_half, x0+zoom_half, nbins+1)
    bins_y = np.linspace(y0-zoom_half, y0+zoom_half, nbins+1)
    H, _, _ = np.histogram2d(xz, yz, bins=[bins, bins_y])
    return gaussian_filter(H.T, sigma=1.0)

def render_frame(snap_path, out_path):
    snap = read_snapshot(snap_path)
    n, z, box = snap['n'], snap['z'], snap['box']
    half = box / 2
    zoom_half = ZOOM_SIZE / 2

    mp = snap['sign'] > 0
    mm = snap['sign'] < 0
    n_plus, n_minus = mp.sum(), mm.sum()

    x0, y0, z0, peak_density = find_density_peak(snap['x'][mp], snap['y'][mp], snap['zpos'][mp], box)

    fig = plt.figure(figsize=(20, 10), facecolor='#080810')
    step = int(snap_path.stem.split('_')[1])

    # Dynamic c_ratio calculation
    eta = 1.045
    delta = (eta - 1.0) / eta
    c_ratio_sq = (1.0 + max(z, 0)) ** delta

    fig.suptitle(f'Janus Phase 2 NO SPH (asym. softening) | Step {step} | z={z:.3f} | c²={c_ratio_sq:.4f} | N={n//1e6:.1f}M',
                 fontsize=14, color='white', y=0.98)

    gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.2, left=0.03, right=0.97, top=0.92, bottom=0.06)
    c_plus, c_minus = '#4488ff', '#ff4444'
    extent = [x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half]

    # Subsample for scatter plots
    sub_p = np.random.choice(mp.sum(), min(500000, mp.sum()), replace=False)
    sub_m = np.random.choice(mm.sum(), min(500000, mm.sum()), replace=False)

    # Row 1 - Positive mass
    ax = fig.add_subplot(gs[0, 0], facecolor='#0a0a14')
    ax.scatter(snap['x'][mp][sub_p], snap['y'][mp][sub_p], s=0.02, c=c_plus, alpha=0.3, rasterized=True)
    ax.add_patch(Rectangle((x0-zoom_half, y0-zoom_half), ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title(f'XY m+ (N={n_plus:,})', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    ax = fig.add_subplot(gs[0, 1], facecolor='#0a0a14')
    ax.scatter(snap['x'][mp][sub_p], snap['zpos'][mp][sub_p], s=0.02, c=c_plus, alpha=0.3, rasterized=True)
    ax.add_patch(Rectangle((x0-zoom_half, z0-zoom_half), ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title('XZ m+', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Zoom with contours
    ax = fig.add_subplot(gs[0, 2], facecolor='#0a0a14')
    mask_p = (np.abs(snap['x'][mp] - x0) < zoom_half) & (np.abs(snap['y'][mp] - y0) < zoom_half)
    xz_p, yz_p = snap['x'][mp][mask_p], snap['y'][mp][mask_p]
    if len(xz_p) > 80000:
        idx = np.random.choice(len(xz_p), 80000, replace=False)
        xz_p, yz_p = xz_p[idx], yz_p[idx]
    ax.scatter(xz_p, yz_p, s=0.15, c=c_plus, alpha=0.6, rasterized=True)
    H_m = compute_density_grid(snap['x'][mm], snap['y'][mm], x0, y0, zoom_half, CONTOUR_BINS)
    if H_m.max() > 0:
        levels = [H_m.max() * p for p in [0.2, 0.4, 0.6, 0.8] if H_m.max() * p > 0]
        if levels:
            ax.contour(H_m, levels=levels, extent=extent, colors='#ff6666', linewidths=[0.5, 0.8, 1.0, 1.5][:len(levels)], alpha=0.9)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(y0-zoom_half, y0+zoom_half); ax.set_aspect('equal')
    ax.set_title('XY m+ ZOOM + m- contours', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    ax = fig.add_subplot(gs[0, 3], facecolor='#0a0a14')
    mask_pz = (np.abs(snap['x'][mp] - x0) < zoom_half) & (np.abs(snap['zpos'][mp] - z0) < zoom_half)
    xz_pz, zz_pz = snap['x'][mp][mask_pz], snap['zpos'][mp][mask_pz]
    if len(xz_pz) > 80000:
        idx = np.random.choice(len(xz_pz), 80000, replace=False)
        xz_pz, zz_pz = xz_pz[idx], zz_pz[idx]
    ax.scatter(xz_pz, zz_pz, s=0.15, c=c_plus, alpha=0.6, rasterized=True)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(z0-zoom_half, z0+zoom_half); ax.set_aspect('equal')
    ax.set_title('XZ m+ ZOOM', color=c_plus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Density
    ax = fig.add_subplot(gs[0, 4], facecolor='#0a0a14')
    H = compute_density_grid(snap['x'][mp], snap['y'][mp], x0, y0, zoom_half, 64)
    H = np.clip(H, 0.1, None)
    im = ax.imshow(H, extent=extent, origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX), aspect='equal', interpolation='bilinear')
    ax.set_title('Log Density m+', color='orange', fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2 - Negative mass
    ax = fig.add_subplot(gs[1, 0], facecolor='#0a0a14')
    ax.scatter(snap['x'][mm][sub_m], snap['y'][mm][sub_m], s=0.02, c=c_minus, alpha=0.3, rasterized=True)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title(f'XY m- (N={n_minus:,})', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    ax = fig.add_subplot(gs[1, 1], facecolor='#0a0a14')
    ax.scatter(snap['x'][mm][sub_m], snap['zpos'][mm][sub_m], s=0.02, c=c_minus, alpha=0.3, rasterized=True)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect('equal')
    ax.set_title('XZ m-', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Evolution panel
    ax = fig.add_subplot(gs[1, 2], facecolor='#0a0a14')
    H_now = compute_density_grid(snap['x'][mm], snap['y'][mm], x0, y0, zoom_half, CONTOUR_BINS)
    z4 = get_z4_reference()
    if z4 is not None:
        mm_z4 = z4['sign'] < 0
        H_z4 = compute_density_grid(z4['x'][mm_z4], z4['y'][mm_z4], x0, y0, zoom_half, CONTOUR_BINS)
        if H_z4.max() > 0:
            levels_z4 = [H_z4.max() * p for p in [0.3, 0.5, 0.7] if H_z4.max() * p > 0]
            if levels_z4:
                ax.contour(H_z4, levels=levels_z4, extent=extent, colors='#666666', linewidths=[0.5, 0.8, 1.0][:len(levels_z4)], alpha=0.6, linestyles='dashed')
    if H_now.max() > 0:
        levels_now = [H_now.max() * p for p in [0.3, 0.5, 0.7, 0.9] if H_now.max() * p > 0]
        if levels_now:
            ax.contour(H_now, levels=levels_now, extent=extent, colors='#ff4444', linewidths=[0.6, 0.9, 1.2, 1.5][:len(levels_now)], alpha=0.9)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(y0-zoom_half, y0+zoom_half); ax.set_aspect('equal')
    ax.set_title(f'm- Evolution: z=4 -> z={z:.2f}', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    ax = fig.add_subplot(gs[1, 3], facecolor='#0a0a14')
    mask_mz = (np.abs(snap['x'][mm] - x0) < zoom_half) & (np.abs(snap['zpos'][mm] - z0) < zoom_half)
    xz_mz, zz_mz = snap['x'][mm][mask_mz], snap['zpos'][mm][mask_mz]
    if len(xz_mz) > 80000:
        idx = np.random.choice(len(xz_mz), 80000, replace=False)
        xz_mz, zz_mz = xz_mz[idx], zz_mz[idx]
    ax.scatter(xz_mz, zz_mz, s=0.15, c=c_minus, alpha=0.6, rasterized=True)
    ax.set_xlim(x0-zoom_half, x0+zoom_half); ax.set_ylim(z0-zoom_half, z0+zoom_half); ax.set_aspect('equal')
    ax.set_title('XZ m- ZOOM', color=c_minus, fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)

    # Purity
    ax = fig.add_subplot(gs[1, 4], facecolor='#0a0a14')
    bins = np.linspace(-half, half, 80)
    H_plus, _, _ = np.histogram2d(snap['x'][mp], snap['zpos'][mp], bins=bins)
    H_minus, _, _ = np.histogram2d(snap['x'][mm], snap['zpos'][mm], bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        purity = (H_plus - H_minus) / (H_plus + H_minus + 1)
    purity = np.nan_to_num(purity, 0)
    im = ax.imshow(purity.T, extent=[-half, half, -half, half], origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1, aspect='equal')
    ax.set_title('Purity XZ', color='white', fontsize=9)
    ax.tick_params(colors='gray', labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Footer
    com_plus = np.array([snap['x'][mp].mean(), snap['y'][mp].mean(), snap['zpos'][mp].mean()])
    com_minus = np.array([snap['x'][mm].mean(), snap['y'][mm].mean(), snap['zpos'][mm].mean()])
    dz = abs(com_plus[2] - com_minus[2])
    fig.text(0.5, 0.01, f"DZ={dz:.0f} Mpc | Peak rho={peak_density:.0f} | c^2={c_ratio_sq:.4f} | eps_minus=5*eps_plus",
             ha='center', fontsize=11, color='#888888', family='monospace')

    plt.savefig(out_path, dpi=150, facecolor='#080810', edgecolor='none')
    plt.close()

def render_single(step):
    """Render a single frame and return render time in seconds"""
    snap_path = SNAP_DIR / f'snap_{step:06d}.bin'
    out_path = OUT_DIR / f'frame_{step:06d}.png'

    if not snap_path.exists():
        print(f"Snapshot not found: {snap_path}")
        return None

    start = time.time()
    render_frame(snap_path, out_path)
    elapsed = time.time() - start
    return elapsed

def daemon_mode():
    """Run as daemon, rendering all new snapshots"""
    print(f"""
Render Daemon - Phase 2 NO SPH
Output: {OUT_DIR}
""")
    rendered = set()
    while True:
        if SNAP_DIR.exists():
            snaps = sorted(SNAP_DIR.glob('snap_*.bin'))
            for snap_path in snaps:
                snap_num = int(snap_path.stem.split('_')[1])
                # Render every snapshot (they're already every 100 steps)
                if snap_num not in rendered:
                    out_path = OUT_DIR / f'frame_{snap_num:06d}.png'
                    try:
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Rendering snap_{snap_num:06d}...", end=' ', flush=True)
                        render_frame(snap_path, out_path)
                        rendered.add(snap_num)
                        print("OK")
                    except Exception as e:
                        print(f"ERROR: {e}")
        time.sleep(30)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        step = int(sys.argv[1])
        elapsed = render_single(step)
        if elapsed:
            print(f"Rendered step {step} in {elapsed:.1f}s")
    else:
        daemon_mode()
