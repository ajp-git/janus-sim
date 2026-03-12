#!/usr/bin/env python3
"""
Janus LHS Visualization Pipeline
Generates diagnostic + wow images for each run
"""

import numpy as np
import struct
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Constants
BOX = 492.0
RES = 1024
HALF = BOX / 2

def load_snapshot(path):
    """Load binary snapshot"""
    with open(path, "rb") as f:
        n, step, _ = struct.unpack("<QQQ", f.read(24))
        data = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)

    pos = data[:, :3]
    sign = data[:, 3]
    return pos, sign, n, step

def make_grids(pos, sign):
    """Create density grids for positive and negative masses"""
    xi = ((pos[:, 0] + HALF) / BOX * RES).astype(int) % RES
    yi = ((pos[:, 1] + HALF) / BOX * RES).astype(int) % RES
    zi = ((pos[:, 2] + HALF) / BOX * RES).astype(int) % RES

    grid_pos = np.zeros((RES, RES, RES), dtype=np.float32)
    grid_neg = np.zeros((RES, RES, RES), dtype=np.float32)

    mask_pos = sign > 0
    mask_neg = sign < 0

    np.add.at(grid_pos, (xi[mask_pos], yi[mask_pos], zi[mask_pos]), 1)
    np.add.at(grid_neg, (xi[mask_neg], yi[mask_neg], zi[mask_neg]), 1)

    return grid_pos, grid_neg

def render_web(grid_total, output_path):
    """Cosmic web projection (XY)"""
    proj = grid_total.sum(axis=2)
    proj = gaussian_filter(proj, 1.5)
    proj = np.log10(proj + 1)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    im = ax.imshow(proj.T, origin='lower', cmap='magma',
                   extent=[-HALF, HALF, -HALF, HALF])
    ax.set_xlabel('X (Mpc)', fontsize=12)
    ax.set_ylabel('Y (Mpc)', fontsize=12)
    ax.set_title('Cosmic Web - Total Density', fontsize=14)
    plt.colorbar(im, ax=ax, label='log₁₀(Σρ)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def render_polarization(grid_pos, grid_neg, output_path):
    """Polarization map P = (ρ+ - ρ-) / (ρ+ + ρ-)"""
    proj_pos = grid_pos.sum(axis=2)
    proj_neg = grid_neg.sum(axis=2)

    proj_pos = gaussian_filter(proj_pos, 2.0)
    proj_neg = gaussian_filter(proj_neg, 2.0)

    total = proj_pos + proj_neg
    P = np.zeros_like(total)
    mask = total > 0
    P[mask] = (proj_pos[mask] - proj_neg[mask]) / total[mask]

    # Custom colormap: blue (neg) - white (neutral) - red (pos)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('polarization', colors)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    im = ax.imshow(P.T, origin='lower', cmap=cmap, vmin=-1, vmax=1,
                   extent=[-HALF, HALF, -HALF, HALF])
    ax.set_xlabel('X (Mpc)', fontsize=12)
    ax.set_ylabel('Y (Mpc)', fontsize=12)
    ax.set_title('Polarization Map P = (ρ⁺ - ρ⁻)/(ρ⁺ + ρ⁻)', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P (blue=neg, red=pos)', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def render_slice(grid_total, output_path, slice_thickness=10):
    """Diagnostic slice (XZ plane, thin slice in Y)"""
    # Take central slice in Y
    y_center = RES // 2
    half_thick = slice_thickness // 2
    slab = grid_total[:, y_center-half_thick:y_center+half_thick, :].sum(axis=1)
    slab = gaussian_filter(slab, 1.0)
    slab = np.log10(slab + 1)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    im = ax.imshow(slab.T, origin='lower', cmap='viridis',
                   extent=[-HALF, HALF, -HALF, HALF])
    ax.set_xlabel('X (Mpc)', fontsize=12)
    ax.set_ylabel('Z (Mpc)', fontsize=12)
    ax.set_title(f'XZ Slice (Y ± {slice_thickness * BOX / RES / 2:.1f} Mpc)', fontsize=14)
    plt.colorbar(im, ax=ax, label='log₁₀(ρ)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def render_wow(grid_total, output_path):
    """Illustris-style wow rendering"""
    proj = grid_total.sum(axis=2)
    proj = gaussian_filter(proj, 1.0)

    # Log scaling
    proj = np.log10(proj + 1)

    # Normalize
    proj -= proj.min()
    proj /= (proj.max() + 1e-10)

    # Gamma correction for contrast
    proj = proj ** 0.6

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax.imshow(proj.T, origin='lower', cmap='inferno')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()

def render_crosscorr(grid_pos, grid_neg, output_path):
    """Cross-correlation map C(x) = ρ+(x) × ρ−(x)

    Interpretation:
    - High C: both species overlap
    - Low C: segregation (Janus active)
    - Ring structures: Janus interfaces
    """
    # Cross-correlation in 3D
    cross = grid_pos * grid_neg

    # Project
    proj = cross.sum(axis=2)
    proj = gaussian_filter(proj, 1.5)

    # Log scale
    proj = np.log10(proj + 1)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    im = ax.imshow(proj.T, origin='lower', cmap='cividis',
                   extent=[-HALF, HALF, -HALF, HALF])
    ax.set_xlabel('X (Mpc)', fontsize=12)
    ax.set_ylabel('Y (Mpc)', fontsize=12)
    ax.set_title('Cross-Correlation C = ρ⁺ × ρ⁻ (low = segregation)', fontsize=14)
    plt.colorbar(im, ax=ax, label='log₁₀(C)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def process_run(run_dir, run_id):
    """Process a single run"""
    snap_path = Path(run_dir) / "snapshots" / "snap_010000.bin"

    if not snap_path.exists():
        print(f"  Snapshot not found: {snap_path}")
        return False

    print(f"  Loading {snap_path}...")
    pos, sign, n, step = load_snapshot(snap_path)
    print(f"  N={n}, step={step}")

    print("  Building density grids...")
    grid_pos, grid_neg = make_grids(pos, sign)
    grid_total = grid_pos + grid_neg

    # Smooth
    grid_pos = gaussian_filter(grid_pos, 1.0)
    grid_neg = gaussian_filter(grid_neg, 1.0)
    grid_total = gaussian_filter(grid_total, 1.0)

    # Output directory
    img_dir = Path(run_dir).parent / "images"
    img_dir.mkdir(exist_ok=True)

    # Generate images
    print("  Rendering web projection...")
    render_web(grid_total, img_dir / f"run_{run_id:02d}_web.png")

    print("  Rendering polarization map...")
    render_polarization(grid_pos, grid_neg, img_dir / f"run_{run_id:02d}_polarization.png")

    print("  Rendering diagnostic slice...")
    render_slice(grid_total, img_dir / f"run_{run_id:02d}_slice.png")

    print("  Rendering cross-correlation...")
    render_crosscorr(grid_pos, grid_neg, img_dir / f"run_{run_id:02d}_crosscorr.png")

    print("  Rendering wow image...")
    render_wow(grid_total, img_dir / f"run_{run_id:02d}_wow.png")

    print(f"  ✓ Run {run_id} complete")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python render_lhs.py <run_id> [base_dir]")
        print("       python render_lhs.py all [base_dir]")
        sys.exit(1)

    base_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/T2/janus-sim/output/lhs_exploration"

    if sys.argv[1] == "all":
        # Process all completed runs
        for run_id in range(1, 51):
            run_dir = Path(base_dir) / f"lhs_run_{run_id:02d}"
            snap = run_dir / "snapshots" / "snap_010000.bin"
            if snap.exists():
                print(f"\n=== Processing Run {run_id} ===")
                process_run(run_dir, run_id)
    else:
        run_id = int(sys.argv[1])
        run_dir = Path(base_dir) / f"lhs_run_{run_id:02d}"
        print(f"\n=== Processing Run {run_id} ===")
        process_run(run_dir, run_id)

if __name__ == "__main__":
    main()
