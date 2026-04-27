#!/usr/bin/env python3
"""
10-Panel Scientific Renderer v5 for Janus VSL
- High-res contours (64 bins)
- Temporal evolution panel: z=4 contours (gray) + current (red)
- Dynamic zoom following m+ density peak
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from pathlib import Path
import struct
import sys
import json
import re
from scipy.ndimage import gaussian_filter

SNAP_DIR = Path("/mnt/T2/janus-sim/output/janus_vsl_2714mpc_10M/snapshots")
OUT_DIR = Path("/mnt/T2/janus-sim/output/janus_vsl_2714mpc_10M/frames_10panel")
SIM_LOG = SNAP_DIR.parent / 'simulation.log'
NORM_FILE = SNAP_DIR.parent / 'density_norm.json'
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Parse simulation.log for v_rms and segregation values
_sim_metrics = {}
def load_sim_metrics():
    global _sim_metrics
    if _sim_metrics:
        return _sim_metrics
    if not SIM_LOG.exists():
        return {}
    with open(SIM_LOG) as f:
        for line in f:
            # Format: "  step |  z  | ... | v_rms+ | v_rms- | N_stars | Seg"
            m = re.match(r'\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*([\d.]+)', line)
            if m:
                step = int(m.group(1))
                v_rms_plus = int(m.group(2))
                v_rms_minus = int(m.group(3))
                seg = float(m.group(4))
                _sim_metrics[step] = {'v_rms_plus': v_rms_plus, 'v_rms_minus': v_rms_minus, 'seg': seg}
    return _sim_metrics

def get_step_metrics(step):
    """Get v_rms+, v_rms-, segregation for a step"""
    metrics = load_sim_metrics()
    if step in metrics:
        return metrics[step]
    # Find closest step
    steps = sorted(metrics.keys())
    for s in steps:
        if s >= step:
            return metrics[s]
    if steps:
        return metrics[steps[-1]]
    return {'v_rms_plus': 0, 'v_rms_minus': 0, 'seg': 0}

ZOOM_SIZE = 100.0
GRID_SIZE = 64
CONTOUR_BINS = 64

if NORM_FILE.exists():
    with open(NORM_FILE) as f:
        norm = json.load(f)
    DENSITY_VMIN = norm['vmin']
    DENSITY_VMAX = norm['vmax']
else:
    DENSITY_VMIN = 1
    DENSITY_VMAX = 100

_z4_cache = None

def read_snapshot(path):
    with open(path, 'rb') as f:
        f.read(4)
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

def render_projection(ax, x, y, half, color, title, s=0.02, alpha=0.3, subsample=500000):
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
    for spine in ax.spines.values():
        spine.set_color('gray')

def compute_density_grid(x, y, x0, y0, zoom_half, nbins=CONTOUR_BINS):
    mask = (np.abs(x - x0) < zoom_half) & (np.abs(y - y0) < zoom_half)
    xz, yz = x[mask], y[mask]
    bins = np.linspace(x0-zoom_half, x0+zoom_half, nbins+1)
    bins_y = np.linspace(y0-zoom_half, y0+zoom_half, nbins+1)
    H, _, _ = np.histogram2d(xz, yz, bins=[bins, bins_y])
    return gaussian_filter(H.T, sigma=1.0)

def render_zoom_with_contours(ax, x_p, y_p, x_m, y_m, x0, y0, zoom_half, title):
    mask_p = (np.abs(x_p - x0) < zoom_half) & (np.abs(y_p - y0) < zoom_half)
    xz_p, yz_p = x_p[mask_p], y_p[mask_p]
    if len(xz_p) > 80000:
        idx = np.random.choice(len(xz_p), 80000, replace=False)
        xz_p, yz_p = xz_p[idx], yz_p[idx]
    ax.scatter(xz_p, yz_p, s=0.15, c='#4488ff', alpha=0.6, rasterized=True)

    H_m = compute_density_grid(x_m, y_m, x0, y0, zoom_half, CONTOUR_BINS)
    extent = [x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half]
    if H_m.max() > 0:
        levels = [H_m.max() * p for p in [0.2, 0.4, 0.6, 0.8] if H_m.max() * p > 0]
        if levels:
            ax.contour(H_m, levels=levels, extent=extent, colors='#ff6666',
                      linewidths=[0.5, 0.8, 1.0, 1.5][:len(levels)], alpha=0.9)

    ax.set_xlim(x0 - zoom_half, x0 + zoom_half)
    ax.set_ylim(y0 - zoom_half, y0 + zoom_half)
    ax.set_title(title, color='#4488ff', fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(colors='gray', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#444444')

def render_evolution_panel(ax, snap_current, x0, y0, zoom_half, current_z):
    mm_current = snap_current['sign'] < 0
    H_now = compute_density_grid(snap_current['x'][mm_current], snap_current['y'][mm_current],
                                  x0, y0, zoom_half, CONTOUR_BINS)
    extent = [x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half]

    z4 = get_z4_reference()
    if z4 is not None:
        mm_z4 = z4['sign'] < 0
        H_z4 = compute_density_grid(z4['x'][mm_z4], z4['y'][mm_z4], x0, y0, zoom_half, CONTOUR_BINS)
        if H_z4.max() > 0:
            levels_z4 = [H_z4.max() * p for p in [0.3, 0.5, 0.7] if H_z4.max() * p > 0]
            if levels_z4:
                ax.contour(H_z4, levels=levels_z4, extent=extent, colors='#666666',
                          linewidths=[0.5, 0.8, 1.0][:len(levels_z4)], alpha=0.6, linestyles='dashed')

    if H_now.max() > 0:
        levels_now = [H_now.max() * p for p in [0.3, 0.5, 0.7, 0.9] if H_now.max() * p > 0]
        if levels_now:
            ax.contour(H_now, levels=levels_now, extent=extent, colors='#ff4444',
                      linewidths=[0.6, 0.9, 1.2, 1.5][:len(levels_now)], alpha=0.9)

    ax.set_xlim(x0 - zoom_half, x0 + zoom_half)
    ax.set_ylim(y0 - zoom_half, y0 + zoom_half)
    ax.set_title(f'm- Evolution: z=4 (gray) → z={current_z:.2f} (red)', color='#ff4444', fontsize=9)
    ax.set_facecolor('#0a0a14')
    ax.set_aspect('equal')
    ax.tick_params(colors='gray', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8)
    ax.set_ylabel('Y [Mpc]', color='gray', fontsize=8)

def render_zoom(ax, x, y, x0, y0, zoom_half, color, title, s=0.15, alpha=0.6):
    mask = (np.abs(x - x0) < zoom_half) & (np.abs(y - y0) < zoom_half)
    xz, yz = x[mask], y[mask]
    if len(xz) > 80000:
        idx = np.random.choice(len(xz), 80000, replace=False)
        xz, yz = xz[idx], yz[idx]
    ax.scatter(xz, yz, s=s, c=color, alpha=alpha, rasterized=True)
    ax.set_xlim(x0 - zoom_half, x0 + zoom_half)
    ax.set_ylim(y0 - zoom_half, y0 + zoom_half)
    ax.set_title(title, color=color, fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(colors='gray', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#444444')

def render_frame(snap_path, out_path):
    snap = read_snapshot(snap_path)
    n, z, box = snap['n'], snap['z'], snap['box']
    half = box / 2
    zoom_half = ZOOM_SIZE / 2

    mp = snap['sign'] > 0
    mm = snap['sign'] < 0
    is_star = snap['type'] == 1
    n_plus, n_minus, n_stars = mp.sum(), mm.sum(), is_star.sum()

    x0, y0, z0, peak_density = find_density_peak(snap['x'][mp], snap['y'][mp], snap['zpos'][mp], box)

    fig = plt.figure(figsize=(20, 10), facecolor='#080810')
    step = int(snap_path.stem.split('_')[1])
    fig.suptitle(f'Janus VSL (μ=19) | Step {step} | z={z:.3f} | N={n//1e6:.1f}M | Peak@({x0:.0f},{y0:.0f},{z0:.0f})',
                 fontsize=14, color='white', y=0.98)

    gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.2, left=0.03, right=0.97, top=0.92, bottom=0.06)
    c_plus, c_minus = '#4488ff', '#ff4444'

    ax = fig.add_subplot(gs[0, 0], facecolor='#0a0a14')
    render_projection(ax, snap['x'][mp], snap['y'][mp], half, c_plus, f'XY m+ (N={n_plus:,})')
    ax.add_patch(Rectangle((x0-zoom_half, y0-zoom_half), ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Y [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[0, 1], facecolor='#0a0a14')
    render_projection(ax, snap['x'][mp], snap['zpos'][mp], half, c_plus, 'XZ m+')
    ax.add_patch(Rectangle((x0-zoom_half, z0-zoom_half), ZOOM_SIZE, ZOOM_SIZE, fill=False, edgecolor='yellow', linewidth=1.5))
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Z [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[0, 2], facecolor='#0a0a14')
    render_zoom_with_contours(ax, snap['x'][mp], snap['y'][mp], snap['x'][mm], snap['y'][mm], x0, y0, zoom_half, 'XY m+ ZOOM + m- contours')
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Y [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[0, 3], facecolor='#0a0a14')
    render_zoom(ax, snap['x'][mp], snap['zpos'][mp], x0, z0, zoom_half, c_plus, f'XZ m+ ZOOM ({ZOOM_SIZE:.0f} Mpc)')
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Z [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[0, 4], facecolor='#0a0a14')
    H = compute_density_grid(snap['x'][mp], snap['y'][mp], x0, y0, zoom_half, 64)
    H = np.clip(H, 0.1, None)
    im = ax.imshow(H, extent=[x0-zoom_half, x0+zoom_half, y0-zoom_half, y0+zoom_half], origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX), aspect='equal', interpolation='bilinear')
    ax.set_title('Log Density m+ (fixed norm)', color='orange', fontsize=9)
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Y [Mpc]', color='gray', fontsize=8)
    ax.tick_params(colors='gray', labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0], facecolor='#0a0a14')
    render_projection(ax, snap['x'][mm], snap['y'][mm], half, c_minus, f'XY m- (N={n_minus:,})')
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Y [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[1, 1], facecolor='#0a0a14')
    render_projection(ax, snap['x'][mm], snap['zpos'][mm], half, c_minus, 'XZ m-')
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Z [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[1, 2], facecolor='#0a0a14')
    render_evolution_panel(ax, snap, x0, y0, zoom_half, z)

    ax = fig.add_subplot(gs[1, 3], facecolor='#0a0a14')
    render_zoom(ax, snap['x'][mm], snap['zpos'][mm], x0, z0, zoom_half, c_minus, 'XZ m- ZOOM')
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Z [Mpc]', color='gray', fontsize=8)

    ax = fig.add_subplot(gs[1, 4], facecolor='#0a0a14')
    bins = np.linspace(-half, half, 80)
    H_plus, _, _ = np.histogram2d(snap['x'][mp], snap['zpos'][mp], bins=bins)
    H_minus, _, _ = np.histogram2d(snap['x'][mm], snap['zpos'][mm], bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        purity = (H_plus - H_minus) / (H_plus + H_minus + 1)
    purity = np.nan_to_num(purity, 0)
    im = ax.imshow(purity.T, extent=[-half, half, -half, half], origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1, aspect='equal')
    ax.set_title('Purity XZ (±0.1)', color='white', fontsize=9)
    ax.set_xlabel('X [Mpc]', color='gray', fontsize=8); ax.set_ylabel('Z [Mpc]', color='gray', fontsize=8)
    ax.tick_params(colors='gray', labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    com_plus = np.array([snap['x'][mp].mean(), snap['y'][mp].mean(), snap['zpos'][mp].mean()])
    com_minus = np.array([snap['x'][mm].mean(), snap['y'][mm].mean(), snap['zpos'][mm].mean()])
    dz = abs(com_plus[2] - com_minus[2])

    # Get v_rms and segregation from simulation log
    metrics = get_step_metrics(step)
    v_rms_p = metrics['v_rms_plus']
    v_rms_m = metrics['v_rms_minus']
    seg = metrics['seg']

    fig.text(0.5, 0.01, f"ΔZ={dz:.0f} Mpc | v_rms+={v_rms_p} km/s | v_rms-={v_rms_m} km/s | Seg={seg:.4f} | Peak ρ={peak_density:.0f}",
             ha='center', fontsize=11, color='#888888', family='monospace')

    plt.savefig(out_path, dpi=150, facecolor='#080810', edgecolor='none')
    plt.close()
    print(f"  ✓ {out_path.name} (z={z:.2f})")

def main():
    if len(sys.argv) > 1:
        snap_num = int(sys.argv[1])
        snap_path = SNAP_DIR / f'snap_{snap_num:06d}.bin'
        out_path = OUT_DIR / f'frame_{snap_num:06d}.png'
        if snap_path.exists():
            render_frame(snap_path, out_path)
    else:
        snaps = sorted(SNAP_DIR.glob('snap_*.bin'))
        for snap_path in snaps:
            snap_num = int(snap_path.stem.split('_')[1])
            out_path = OUT_DIR / f'frame_{snap_num:06d}.png'
            print(f"[{snap_num:06d}] Rendering...")
            render_frame(snap_path, out_path)

if __name__ == '__main__':
    main()
