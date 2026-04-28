#!/usr/bin/env python3
"""
Publication-quality rendering pipeline for Janus simulation snapshots.

Pipeline (différent du render_daemon_adaptive_v2 diagnostic):
  1. Read V3 snapshot
  2. CIC density on 256³ grid for δ_+, δ_-, δ_total (mass-weighted)
  3. Gaussian smoothing (σ=1 cell ≈ 2 Mpc)
  4. Project 2D (column-integrate along z-axis, or thin slice)
  5. Plot with publication colormap (viridis for log-density,
     RdBu_r for δ_+ vs δ_- contrast)
  6. Export PNG aspect-ratio uniform, no zoom artifacts, suitable for video

Modes:
  - --once    : process all existing snapshots and exit
  - default   : watch snapshots dir, render new ones

Layouts:
  - density   : log10(ρ_total)   (single panel viridis)
  - segregation : δ_+ vs δ_-     (RdBu_r diverging map)
  - dual      : 2 panels [δ_+ blue, δ_- red] with shared colorbar
"""
import argparse
import os
import struct
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

DEFAULT_RUNDIR = "/mnt/T2/janus-sim/output/janus_jpp_production"
DEFAULT_SNAPDIR = f"{DEFAULT_RUNDIR}/snapshots"
DEFAULT_OUTDIR = f"{DEFAULT_RUNDIR}/frames_publication"

N_GRID = 256
SLICE_THICKNESS_MPC = 50.0   # Project along z within ±25 Mpc of the box mid-plane
GAUSSIAN_SIGMA_CELL = 1.0    # Smoothing in cells (~2 Mpc with N_GRID=256, L=500)
DPI = 200
FIG_WIDTH_INCH = 10.0
FIG_HEIGHT_INCH = 10.0

def read_v3(path):
    with open(path, 'rb') as f:
        h = f.read(408)
        n = struct.unpack('<Q', h[16:24])[0]
        a = struct.unpack('<d', h[24:32])[0]
        t_gyr = struct.unpack('<d', h[32:40])[0]
        l_box = struct.unpack('<d', h[40:48])[0]
        dt = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(n * 36), dtype=dt)
    z = 1.0 / a - 1.0 if a > 0 else 0.0
    return n, a, z, t_gyr, l_box, particles

def cic(positions, n_grid, box_size):
    cell = box_size / n_grid
    pos = (positions + box_size / 2.0)
    pos = pos - box_size * np.floor(pos / box_size)
    coords = pos / cell
    i0 = np.floor(coords).astype(np.int64) % n_grid
    d = coords - np.floor(coords)
    i1 = (i0 + 1) % n_grid
    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    for dx in (0, 1):
        wx = d[:, 0] if dx else (1.0 - d[:, 0])
        ix = i1[:, 0] if dx else i0[:, 0]
        for dy in (0, 1):
            wy = d[:, 1] if dy else (1.0 - d[:, 1])
            iy = i1[:, 1] if dy else i0[:, 1]
            for dz in (0, 1):
                wz = d[:, 2] if dz else (1.0 - d[:, 2])
                iz = i1[:, 2] if dz else i0[:, 2]
                np.add.at(rho, (ix, iy, iz), wx*wy*wz)
    return rho

def project_2d_slab(rho, n_grid, box_size, thickness_mpc=None):
    """Project 3D density to 2D (column integration).
    If thickness_mpc set, integrate only within ±thickness/2 around z=0.
    Else full integration along z.
    Returns (n_grid, n_grid) 2D array."""
    if thickness_mpc is None:
        return rho.sum(axis=2)
    cell = box_size / n_grid
    half_slab = thickness_mpc / 2.0
    n_cells_slab = max(1, int(half_slab / cell))
    mid = n_grid // 2
    z_slice = slice(max(0, mid - n_cells_slab), min(n_grid, mid + n_cells_slab + 1))
    return rho[:, :, z_slice].sum(axis=2)

def render_density(snap_path, out_dir, layout='dual', overwrite=False):
    n, a, z, t_gyr, l_box, particles = read_v3(snap_path)
    base = os.path.basename(snap_path).replace('.bin', '')
    out_path = os.path.join(out_dir, f"{base}_pub_{layout}.png")
    if os.path.exists(out_path) and not overwrite:
        return False, "skipped (exists)"

    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)

    rho_plus = cic(pos[is_plus], N_GRID, l_box)
    rho_minus = cic(pos[is_minus], N_GRID, l_box)

    # Gaussian smooth (3D before projection — preserves true smoothing scale)
    rho_plus_sm = gaussian_filter(rho_plus, sigma=GAUSSIAN_SIGMA_CELL)
    rho_minus_sm = gaussian_filter(rho_minus, sigma=GAUSSIAN_SIGMA_CELL)

    # Project to 2D slab
    proj_p = project_2d_slab(rho_plus_sm, N_GRID, l_box, SLICE_THICKNESS_MPC)
    proj_m = project_2d_slab(rho_minus_sm, N_GRID, l_box, SLICE_THICKNESS_MPC)

    extent = [-l_box/2, l_box/2, -l_box/2, l_box/2]

    if layout == 'dual':
        # Two panels: m+ in cool blues, m- in warm reds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*FIG_WIDTH_INCH, FIG_HEIGHT_INCH))
        # Use percentile clipping to avoid extreme values
        for ax, proj, cmap, label, color in [
            (ax1, proj_p, 'Blues', 'm+ (matter)', 'tab:blue'),
            (ax2, proj_m, 'Reds', 'm- (anti-matter)', 'tab:red'),
        ]:
            v_min = np.percentile(proj[proj>0], 5) if (proj > 0).any() else 0
            v_max = np.percentile(proj, 99.5)
            im = ax.imshow(np.log10(np.maximum(proj, v_min)).T,
                          extent=extent, origin='lower', cmap=cmap,
                          vmin=np.log10(max(v_min, 1e-10)), vmax=np.log10(v_max))
            ax.set_title(label, fontsize=14, color=color)
            ax.set_xlabel('x (Mpc)')
            ax.set_ylabel('y (Mpc)')
            ax.set_aspect('equal')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('log₁₀(column density)', fontsize=10)
        fig.suptitle(f'Janus µ=19  z={z:.2f}  t={t_gyr:.2f} Gyr  '
                     f'(slice ±{SLICE_THICKNESS_MPC/2:.0f} Mpc)',
                     fontsize=14, y=1.02)
    elif layout == 'segregation':
        # Single panel showing δ_+ - δ_- (segregation visualization)
        mp = proj_p.mean()
        mm = proj_m.mean()
        delta_p = proj_p / mp - 1.0 if mp > 0 else np.zeros_like(proj_p)
        delta_m = proj_m / mm - 1.0 if mm > 0 else np.zeros_like(proj_m)
        seg = delta_p - delta_m
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH))
        v_max = np.percentile(np.abs(seg), 99)
        im = ax.imshow(seg.T, extent=extent, origin='lower', cmap='RdBu_r',
                      vmin=-v_max, vmax=v_max)
        ax.set_title(f'Janus segregation  δ₊ − δ₋  '
                    f'z={z:.2f}  t={t_gyr:.2f} Gyr', fontsize=14)
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('y (Mpc)')
        ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('δ₊ − δ₋  (red: m+ excess, blue: m- excess)', fontsize=10)
    elif layout == 'density':
        # Single panel total density
        proj_total = proj_p + proj_m
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH))
        v_min = np.percentile(proj_total[proj_total>0], 5) if (proj_total > 0).any() else 0
        v_max = np.percentile(proj_total, 99.5)
        im = ax.imshow(np.log10(np.maximum(proj_total, v_min)).T,
                      extent=extent, origin='lower', cmap='viridis',
                      vmin=np.log10(max(v_min, 1e-10)), vmax=np.log10(v_max))
        ax.set_title(f'Janus total density  z={z:.2f}  t={t_gyr:.2f} Gyr', fontsize=14)
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('y (Mpc)')
        ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log₁₀(column density)', fontsize=10)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return True, out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapdir', default=DEFAULT_SNAPDIR)
    ap.add_argument('--outdir',  default=DEFAULT_OUTDIR)
    ap.add_argument('--layouts', default='dual,segregation,density',
                    help='Comma-separated layouts: dual, segregation, density')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--poll', type=float, default=60.0)
    args = ap.parse_args()

    layouts = [s.strip() for s in args.layouts.split(',')]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== Publication renderer ===")
    print(f"  snapdir : {args.snapdir}")
    print(f"  outdir  : {args.outdir}")
    print(f"  layouts : {layouts}")
    print(f"  N_grid  : {N_GRID}")
    print(f"  slab    : ±{SLICE_THICKNESS_MPC/2:.0f} Mpc")
    print(f"  smooth  : σ={GAUSSIAN_SIGMA_CELL} cells")
    print(f"  DPI     : {DPI}")
    print()

    seen = set()
    while True:
        files = sorted([f for f in os.listdir(args.snapdir) if f.endswith('.bin')])
        new_files = [f for f in files if f not in seen]
        for fname in new_files:
            path = os.path.join(args.snapdir, fname)
            size_a = os.path.getsize(path)
            time.sleep(2)
            if os.path.getsize(path) != size_a:
                continue  # still being written
            for layout in layouts:
                try:
                    t0 = time.time()
                    rendered, msg = render_density(path, args.outdir, layout=layout,
                                                    overwrite=args.overwrite)
                    dt = time.time() - t0
                    if rendered:
                        print(f"  {fname} [{layout}]: {dt:.1f}s → {msg}", flush=True)
                    else:
                        # silent skip
                        pass
                except Exception as e:
                    print(f"  {fname} [{layout}]: ERROR {e}", flush=True)
            seen.add(fname)

        if args.once:
            print("Done (--once)")
            break
        time.sleep(args.poll)

if __name__ == '__main__':
    main()
