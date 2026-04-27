#!/usr/bin/env python3
"""
Generate zoom series at different scales for Janus 10M validation.
Renders 3 zoom levels: 20, 50, 100 Mpc centered on main halo.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import sys

# Configuration
SNAP_PATH = Path("/mnt/T2/janus-sim/output/run_final_10m/snapshots/snap_01500.bin")
OUT_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/zooms")
OUT_DIR.mkdir(exist_ok=True)

BOX_SIZE = 300.0
DZ = 4.0  # Slice thickness
ZOOM_SIZES = [20, 50, 100]  # Mpc

# Halo position from analysis
HALO_CENTER = np.array([58.6, -2.3, 68.0])

def read_snapshot(path):
    """Read binary snapshot"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    z = 1.0/a - 1.0 if a > 0 else 0
    return {
        'n': n, 'z': z, 'a': a,
        'x': pos[:, 0], 'y': pos[:, 1], 'zpos': pos[:, 2],
        'sign': signs
    }

def render_zoom(snap, zoom_size, center, output_path):
    """Render a single zoom level with 4 panels"""
    half = zoom_size / 2

    # Filter particles in zoom region (XY slice)
    mask_xy = (
        (np.abs(snap['x'] - center[0]) < half) &
        (np.abs(snap['y'] - center[1]) < half) &
        (np.abs(snap['zpos'] - center[2]) < DZ/2)
    )

    # Filter particles in zoom region (XZ slice)
    mask_xz = (
        (np.abs(snap['x'] - center[0]) < half) &
        (np.abs(snap['zpos'] - center[2]) < half) &
        (np.abs(snap['y'] - center[1]) < DZ/2)
    )

    x_xy = snap['x'][mask_xy]
    y_xy = snap['y'][mask_xy]
    sign_xy = snap['sign'][mask_xy]

    x_xz = snap['x'][mask_xz]
    z_xz = snap['zpos'][mask_xz]
    sign_xz = snap['sign'][mask_xz]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')

    # Colors
    color_plus = '#ff6b35'  # Orange for m+
    color_minus = '#4dabf7'  # Blue for m-

    # Panel 1: XY m+ only
    ax = axes[0, 0]
    ax.set_facecolor('black')
    mask_p = sign_xy > 0
    if np.sum(mask_p) > 0:
        ax.scatter(x_xy[mask_p], y_xy[mask_p], c=color_plus, s=0.5, alpha=0.8)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_title(f'XY m+ (N={np.sum(mask_p):,})', color='white', fontsize=14)
    ax.set_xlabel('X [Mpc]', color='white')
    ax.set_ylabel('Y [Mpc]', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Panel 2: XY m- only
    ax = axes[0, 1]
    ax.set_facecolor('black')
    mask_m = sign_xy < 0
    if np.sum(mask_m) > 0:
        ax.scatter(x_xy[mask_m], y_xy[mask_m], c=color_minus, s=0.5, alpha=0.8)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_title(f'XY m- (N={np.sum(mask_m):,})', color='white', fontsize=14)
    ax.set_xlabel('X [Mpc]', color='white')
    ax.set_ylabel('Y [Mpc]', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Panel 3: XZ m+ only
    ax = axes[1, 0]
    ax.set_facecolor('black')
    mask_p = sign_xz > 0
    if np.sum(mask_p) > 0:
        ax.scatter(x_xz[mask_p], z_xz[mask_p], c=color_plus, s=0.5, alpha=0.8)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[2] - half, center[2] + half)
    ax.set_title(f'XZ m+ (N={np.sum(mask_p):,})', color='white', fontsize=14)
    ax.set_xlabel('X [Mpc]', color='white')
    ax.set_ylabel('Z [Mpc]', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Panel 4: XZ m- only
    ax = axes[1, 1]
    ax.set_facecolor('black')
    mask_m = sign_xz < 0
    if np.sum(mask_m) > 0:
        ax.scatter(x_xz[mask_m], z_xz[mask_m], c=color_minus, s=0.5, alpha=0.8)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[2] - half, center[2] + half)
    ax.set_title(f'XZ m- (N={np.sum(mask_m):,})', color='white', fontsize=14)
    ax.set_xlabel('X [Mpc]', color='white')
    ax.set_ylabel('Z [Mpc]', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.suptitle(f'JANUS 10M z=0 | Zoom {zoom_size} Mpc | Center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) | dz={DZ} Mpc',
                 color='white', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, facecolor='black', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_path}")

def main():
    print("=" * 60)
    print("JANUS 10M Zoom Series Generator")
    print("=" * 60)

    print(f"\nReading snapshot: {SNAP_PATH}")
    snap = read_snapshot(SNAP_PATH)
    print(f"  N = {snap['n']:,}, z = {snap['z']:.4f}")
    print(f"  Center: ({HALO_CENTER[0]:.1f}, {HALO_CENTER[1]:.1f}, {HALO_CENTER[2]:.1f}) Mpc")

    print(f"\nGenerating zoom series...")
    for zoom_size in ZOOM_SIZES:
        print(f"\n[Zoom {zoom_size} Mpc]")
        output_path = OUT_DIR / f"zoom_{zoom_size}mpc.png"
        render_zoom(snap, zoom_size, HALO_CENTER, output_path)

    print("\n" + "=" * 60)
    print("ZOOM SERIES COMPLETE")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
