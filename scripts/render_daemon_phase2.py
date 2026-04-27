#!/usr/bin/env python3
"""
Render daemon for VSL Phase 2 (10M particles, 500 Mpc)
10-panel layout, renders every 5 steps
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
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor

SNAP_DIR = Path("/mnt/T2/janus-sim/output/vsl_phase2/snapshots")
OUT_DIR = Path("/mnt/T2/janus-sim/output/vsl_phase2/frames")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ZOOM_SIZE = 100.0  # 100 Mpc zoom for 500 Mpc box
GRID_SIZE = 64
CONTOUR_BINS = 64
DENSITY_VMIN = 1
DENSITY_VMAX = 500

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
            'zpos': data['z'], 'sign': data['sign']}

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

def render_projection(ax, x, y, half, color, title, s=0.01, alpha=0.2, subsample=200000):
    n = len(x)
    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        x, y = x[idx], y[idx]
    ax.scatter(x, y, s=s, c=color, alpha=alpha, rasterized=True)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_title(title, color=color, fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(colors='gray', labelsize=7)

def compute_density_grid(x, y, x0, y0, zoom_half, nbins=CONTOUR_BINS):
    mask = (np.abs(x - x0) < zoom_half) & (np.abs(y - y0) < zoom_half)
    xz, yz = x[mask], y[mask]
    if len(xz) == 0:
        return np.zeros((nbins, nbins))
    bins = np.linspace(x0-zoom_half, x0+zoom_half, nbins+1)
    bins_y = np.linspace(y0-zoom_half, y0+zoom_half, nbins+1)
    H, _, _ = np.histogram2d(xz, yz, bins=[bins, bins_y])
    return gaussian_filter(H.T, sigma=1.0)

def render_frame(snap_path, out_path):
    try:
        snap = read_snapshot(snap_path)
        n, z, box = snap['n'], snap['z'], snap['box']
        half = box / 2
        zoom_half = ZOOM_SIZE / 2

        mp = snap['sign'] > 0
        mm = snap['sign'] < 0
        n_plus, n_minus = mp.sum(), mm.sum()

        x0, y0, z0, _ = find_density_peak(snap['x'][mp], snap['y'][mp], snap['zpos'][mp], box)

        fig = plt.figure(figsize=(20, 10), facecolor='#080810')
        step = int(snap_path.stem.split('_')[1])
        fig.suptitle(f'Janus VSL Phase 2 | Step {step} | z={z:.3f} | N={n/1e6:.1f}M | Box={box:.0f}Mpc',
                     fontsize=14, color='white', y=0.98)

        gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.2, left=0.03, right=0.97, top=0.92, bottom=0.06)
        c_plus, c_minus = '#4488ff', '#ff4444'

        # Row 1: m+ views
        ax = fig.add_subplot(gs[0, 0], facecolor='#0a0a14')
        render_projection(ax, snap['x'][mp], snap['y'][mp], half, c_plus, f'XY m+ (N={n_plus:,})')
        ax.add_patch(Rectangle((x0-zoom_half, y0-zoom_half), ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=1))

        ax = fig.add_subplot(gs[0, 1], facecolor='#0a0a14')
        render_projection(ax, snap['x'][mp], snap['zpos'][mp], half, c_plus, 'XZ m+')

        ax = fig.add_subplot(gs[0, 2], facecolor='#0a0a14')
        # Zoom m+ with m- contours
        mask_p = (np.abs(snap['x'][mp] - x0) < zoom_half) & (np.abs(snap['y'][mp] - y0) < zoom_half)
        xz_p, yz_p = snap['x'][mp][mask_p], snap['y'][mp][mask_p]
        if len(xz_p) > 80000:
            idx = np.random.choice(len(xz_p), 80000, replace=False)
            xz_p, yz_p = xz_p[idx], yz_p[idx]
        ax.scatter(xz_p, yz_p, s=0.1, c=c_plus, alpha=0.5, rasterized=True)
        H_m = compute_density_grid(snap['x'][mm], snap['y'][mm], x0, y0, zoom_half)
        if H_m.max() > 0:
            levels = [H_m.max() * p for p in [0.3, 0.5, 0.7] if H_m.max() * p > 0]
            if levels:
                ax.contour(H_m, levels=levels, extent=[x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half],
                          colors='#ff6666', linewidths=[0.5, 0.8, 1.0][:len(levels)], alpha=0.8)
        ax.set_xlim(x0-zoom_half, x0+zoom_half)
        ax.set_ylim(y0-zoom_half, y0+zoom_half)
        ax.set_title('XY ZOOM m+ + m- contours', color=c_plus, fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(colors='gray', labelsize=7)

        ax = fig.add_subplot(gs[0, 3], facecolor='#0a0a14')
        mask_p = (np.abs(snap['x'][mp] - x0) < zoom_half) & (np.abs(snap['zpos'][mp] - z0) < zoom_half)
        xz_p, zz_p = snap['x'][mp][mask_p], snap['zpos'][mp][mask_p]
        if len(xz_p) > 80000:
            idx = np.random.choice(len(xz_p), 80000, replace=False)
            xz_p, zz_p = xz_p[idx], zz_p[idx]
        ax.scatter(xz_p, zz_p, s=0.1, c=c_plus, alpha=0.5, rasterized=True)
        ax.set_xlim(x0-zoom_half, x0+zoom_half)
        ax.set_ylim(z0-zoom_half, z0+zoom_half)
        ax.set_title(f'XZ m+ ZOOM ({ZOOM_SIZE:.0f} Mpc)', color=c_plus, fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(colors='gray', labelsize=7)

        ax = fig.add_subplot(gs[0, 4], facecolor='#0a0a14')
        H = compute_density_grid(snap['x'][mp], snap['y'][mp], x0, y0, zoom_half, 64)
        H = np.clip(H, 0.1, None)
        im = ax.imshow(H, extent=[x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half], origin='lower',
                      cmap='inferno', norm=LogNorm(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX), aspect='equal')
        ax.set_title('Log Density m+', color='orange', fontsize=9)
        ax.tick_params(colors='gray', labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 2: m- views
        ax = fig.add_subplot(gs[1, 0], facecolor='#0a0a14')
        render_projection(ax, snap['x'][mm], snap['y'][mm], half, c_minus, f'XY m- (N={n_minus:,})')

        ax = fig.add_subplot(gs[1, 1], facecolor='#0a0a14')
        render_projection(ax, snap['x'][mm], snap['zpos'][mm], half, c_minus, 'XZ m-')

        ax = fig.add_subplot(gs[1, 2], facecolor='#0a0a14')
        # Evolution panel
        H_now = compute_density_grid(snap['x'][mm], snap['y'][mm], x0, y0, zoom_half)
        z4 = get_z4_reference()
        if z4 is not None:
            mm_z4 = z4['sign'] < 0
            H_z4 = compute_density_grid(z4['x'][mm_z4], z4['y'][mm_z4], x0, y0, zoom_half)
            if H_z4.max() > 0:
                levels_z4 = [H_z4.max() * p for p in [0.3, 0.5, 0.7] if H_z4.max() * p > 0]
                if levels_z4:
                    ax.contour(H_z4, levels=levels_z4, extent=[x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half],
                              colors='#666666', linewidths=[0.5, 0.8, 1.0][:len(levels_z4)], alpha=0.6, linestyles='dashed')
        if H_now.max() > 0:
            levels_now = [H_now.max() * p for p in [0.3, 0.5, 0.7, 0.9] if H_now.max() * p > 0]
            if levels_now:
                ax.contour(H_now, levels=levels_now, extent=[x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half],
                          colors='#ff4444', linewidths=[0.6, 0.9, 1.2, 1.5][:len(levels_now)], alpha=0.9)
        ax.set_xlim(x0-zoom_half, x0+zoom_half)
        ax.set_ylim(y0-zoom_half, y0+zoom_half)
        ax.set_title(f'm- Evolution: z=4 (gray) -> z={z:.2f} (red)', color=c_minus, fontsize=9)
        ax.set_facecolor('#0a0a14')
        ax.set_aspect('equal')
        ax.tick_params(colors='gray', labelsize=7)

        ax = fig.add_subplot(gs[1, 3], facecolor='#0a0a14')
        mask_m = (np.abs(snap['x'][mm] - x0) < zoom_half) & (np.abs(snap['zpos'][mm] - z0) < zoom_half)
        xz_m, zz_m = snap['x'][mm][mask_m], snap['zpos'][mm][mask_m]
        if len(xz_m) > 80000:
            idx = np.random.choice(len(xz_m), 80000, replace=False)
            xz_m, zz_m = xz_m[idx], zz_m[idx]
        ax.scatter(xz_m, zz_m, s=0.1, c=c_minus, alpha=0.5, rasterized=True)
        ax.set_xlim(x0-zoom_half, x0+zoom_half)
        ax.set_ylim(z0-zoom_half, z0+zoom_half)
        ax.set_title('XZ m- ZOOM', color=c_minus, fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(colors='gray', labelsize=7)

        ax = fig.add_subplot(gs[1, 4], facecolor='#0a0a14')
        bins = np.linspace(-half, half, 80)
        H_plus, _, _ = np.histogram2d(snap['x'][mp], snap['zpos'][mp], bins=bins)
        H_minus, _, _ = np.histogram2d(snap['x'][mm], snap['zpos'][mm], bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            purity = (H_plus - H_minus) / (H_plus + H_minus + 1)
        purity = np.nan_to_num(purity, 0)
        im = ax.imshow(purity.T, extent=[-half, half, -half, half], origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1, aspect='equal')
        ax.set_title('Purity XZ (m+/m-)', color='white', fontsize=9)
        ax.tick_params(colors='gray', labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.savefig(out_path, dpi=150, facecolor='#080810', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error rendering {snap_path}: {e}")
        return False

def main():
    print("Render daemon for Phase 2 started")
    print(f"Watching: {SNAP_DIR}")
    print(f"Output: {OUT_DIR}")

    rendered = set()

    while True:
        complete_file = SNAP_DIR.parent / "phase2_complete.txt"
        sim_complete = complete_file.exists()

        if SNAP_DIR.exists():
            snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
            for snap_path in snaps:
                snap_name = snap_path.stem
                step = int(snap_name.split('_')[1])
                # Render every 5th snapshot for frames
                if step % 5 != 0:
                    continue
                out_path = OUT_DIR / f"frame_{step:06d}.png"

                if snap_name not in rendered and not out_path.exists():
                    print(f"Rendering step {step}...")
                    if render_frame(snap_path, out_path):
                        rendered.add(snap_name)
                        print(f"  -> {out_path.name}")

        if sim_complete:
            time.sleep(5)
            # Final pass
            if SNAP_DIR.exists():
                snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
                for snap_path in snaps:
                    step = int(snap_path.stem.split('_')[1])
                    if step % 5 != 0:
                        continue
                    out_path = OUT_DIR / f"frame_{step:06d}.png"
                    if not out_path.exists():
                        render_frame(snap_path, out_path)

            n_frames = len(list(OUT_DIR.glob("frame_*.png")))
            print(f"\nRender complete: {n_frames} frames")
            break

        time.sleep(3)

if __name__ == "__main__":
    main()
