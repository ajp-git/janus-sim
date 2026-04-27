#!/usr/bin/env python3
"""
Render frame for Janus Baryonic Calibrated run (cooling-only)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import sys

def read_snapshot(path):
    """Read JSNP binary snapshot"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]

        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    return n, a, t, pos, vel, signs

def main():
    if len(sys.argv) < 3:
        print("Usage: render_cooling_run.py <snapshot> <output.png>")
        sys.exit(1)

    snap_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Reading {snap_path}...")
    n, a, t, pos, vel, signs = read_snapshot(snap_path)
    z = 1/a - 1

    plus_mask = signs > 0
    minus_mask = signs < 0

    print(f"  N = {n:,}, z = {z:.2f}, t = {t:.2f} Gyr")
    print(f"  N+ = {plus_mask.sum():,}, N- = {minus_mask.sum():,}")

    # Compute v_rms
    v2_plus = np.sum(vel[plus_mask]**2, axis=1)
    v2_minus = np.sum(vel[minus_mask]**2, axis=1)
    v_rms_plus = np.sqrt(np.mean(v2_plus)) * 977.8  # km/s
    v_rms_minus = np.sqrt(np.mean(v2_minus)) * 977.8
    ratio = v_rms_minus / v_rms_plus

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # Left: XY projection
    ax1 = fig.add_subplot(121)

    # Subsample for plotting (10M is too many)
    n_plot = min(100000, n)
    idx = np.random.choice(n, n_plot, replace=False)

    pos_sub = pos[idx]
    signs_sub = signs[idx]

    plus_sub = signs_sub > 0
    minus_sub = signs_sub < 0

    ax1.scatter(pos_sub[minus_sub, 0], pos_sub[minus_sub, 1],
                c='blue', s=0.1, alpha=0.3, label='m-')
    ax1.scatter(pos_sub[plus_sub, 0], pos_sub[plus_sub, 1],
                c='red', s=0.1, alpha=0.3, label='m+')

    ax1.set_xlim(-250, 250)
    ax1.set_ylim(-250, 250)
    ax1.set_xlabel('X [Mpc]')
    ax1.set_ylabel('Y [Mpc]')
    ax1.set_title(f'Janus Baryonic (S&D93 Cooling)\nz = {z:.2f}, t = {t:.2f} Gyr')
    ax1.legend(loc='upper right', markerscale=20)
    ax1.set_aspect('equal')

    # Right: Density histogram
    ax2 = fig.add_subplot(122)

    # Compute 2D histogram
    bins = 64
    range_xy = [[-250, 250], [-250, 250]]

    H_plus, xedges, yedges = np.histogram2d(
        pos[plus_mask, 0], pos[plus_mask, 1], bins=bins, range=range_xy)
    H_minus, _, _ = np.histogram2d(
        pos[minus_mask, 0], pos[minus_mask, 1], bins=bins, range=range_xy)

    # Segregation map: (N+ - N-) / (N+ + N-)
    with np.errstate(divide='ignore', invalid='ignore'):
        seg_map = (H_plus - H_minus) / (H_plus + H_minus)
        seg_map = np.nan_to_num(seg_map, nan=0)

    im = ax2.imshow(seg_map.T, origin='lower', extent=[-250, 250, -250, 250],
                    cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2, label='(N+ - N-)/(N+ + N-)')
    ax2.set_xlabel('X [Mpc]')
    ax2.set_ylabel('Y [Mpc]')
    ax2.set_title(f'Segregation Map\nratio = {ratio:.3f}, S = {np.std(seg_map):.3f}')
    ax2.set_aspect('equal')

    # Add text box with metrics
    textstr = '\n'.join([
        f'Step: {Path(snap_path).stem.replace("snap_", "")}',
        f'N = {n:,}',
        f'z = {z:.3f}',
        f'v_rms+ = {v_rms_plus:.0f} km/s',
        f'v_rms- = {v_rms_minus:.0f} km/s',
        f'ratio = {ratio:.4f}',
        f'N_stars = 0 (SF disabled)',
        f'T_mean = 10000 K (cooling off @ low n_H)'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
             fontfamily='monospace', bbox=props)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
