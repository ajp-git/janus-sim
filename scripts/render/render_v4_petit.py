#!/usr/bin/env python3
"""
4K Composite Frame Renderer for petit_pure_20m_treepm_v3
Adapted for 25-byte particle format (pos + vel + sign)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from pathlib import Path
import struct
import time

# Constants - Petit Pure TreePM v3
MU = 8.0
LAMBDA_0 = 0.0  # Pure anti-Newton
BOX_SIZE = 500.0
TOTAL_STEPS = 2000

def read_snapshot(path, max_particles=None):
    """Read binary snapshot file - 25 bytes/particle format"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]

        # Read all particle data at once (much faster)
        particle_size = 25  # 6×f32 + i8
        n_read = min(n, max_particles) if max_particles else n

        if max_particles and n > max_particles:
            # Random sampling for large files
            indices = np.sort(np.random.choice(n, n_read, replace=False))
            pos = np.zeros((n_read, 3), dtype=np.float32)
            signs = np.zeros(n_read, dtype=np.int8)

            for i, idx in enumerate(indices):
                f.seek(16 + idx * particle_size)
                pos[i, 0] = struct.unpack('<f', f.read(4))[0]
                pos[i, 1] = struct.unpack('<f', f.read(4))[0]
                pos[i, 2] = struct.unpack('<f', f.read(4))[0]
                f.read(12)  # skip velocities
                signs[i] = struct.unpack('<b', f.read(1))[0]
        else:
            # Read all particles
            pos = np.zeros((n, 3), dtype=np.float32)
            signs = np.zeros(n, dtype=np.int8)

            for i in range(n):
                pos[i, 0] = struct.unpack('<f', f.read(4))[0]
                pos[i, 1] = struct.unpack('<f', f.read(4))[0]
                pos[i, 2] = struct.unpack('<f', f.read(4))[0]
                f.read(12)  # skip velocities (vx, vy, vz)
                signs[i] = struct.unpack('<b', f.read(1))[0]

    return pos, signs, step, z, box

def compute_purity_map_2d(pos, signs, box_size, n_cells=64):
    """Compute 2D purity map (XY projection) with periodic wrapping"""
    cell_size = box_size / n_cells
    half_box = box_size / 2.0

    n_plus = np.zeros((n_cells, n_cells), dtype=np.int32)
    n_minus = np.zeros((n_cells, n_cells), dtype=np.int32)

    # Periodic wrap: pos in [-half, half] -> [0, box_size]
    x = (pos[:, 0] + half_box) % box_size
    y = (pos[:, 1] + half_box) % box_size

    ix = np.clip((x / cell_size).astype(int), 0, n_cells - 1)
    iy = np.clip((y / cell_size).astype(int), 0, n_cells - 1)

    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus[ix[i], iy[i]] += 1
        else:
            n_minus[ix[i], iy[i]] += 1

    total = n_plus + n_minus
    purity = np.zeros((n_cells, n_cells))
    mask = total > 0
    purity[mask] = (n_plus[mask] - n_minus[mask]) / total[mask]

    return purity, total

def compute_global_purity(pos, signs, box_size, n_cells=32):
    """Compute global purity metric P with periodic wrapping"""
    cell_size = box_size / n_cells
    half_box = box_size / 2.0
    n_cells_cubed = n_cells ** 3

    n_plus = np.zeros(n_cells_cubed, dtype=np.int32)
    n_minus = np.zeros(n_cells_cubed, dtype=np.int32)

    # Periodic wrap: pos in [-half, half] -> [0, box_size]
    pos_wrapped = (pos + half_box) % box_size
    ix = np.clip((pos_wrapped[:, 0] / cell_size).astype(int), 0, n_cells - 1)
    iy = np.clip((pos_wrapped[:, 1] / cell_size).astype(int), 0, n_cells - 1)
    iz = np.clip((pos_wrapped[:, 2] / cell_size).astype(int), 0, n_cells - 1)
    idx = ix * n_cells * n_cells + iy * n_cells + iz

    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus[idx[i]] += 1
        else:
            n_minus[idx[i]] += 1

    total_weight = 0.0
    weighted_purity = 0.0
    for cell_idx in range(n_cells_cubed):
        np_cell = n_plus[cell_idx]
        nm_cell = n_minus[cell_idx]
        weight = np_cell + nm_cell
        if weight > 0:
            purity = abs(np_cell - nm_cell) / weight
            weighted_purity += purity * weight
            total_weight += weight

    return weighted_purity / total_weight if total_weight > 0 else 0.0

def render_composite(snap_path, output_path):
    """Render 4K composite frame"""
    print(f"  Reading {snap_path.name}...")
    pos, signs, step, z, box = read_snapshot(snap_path, max_particles=2_000_000)

    n_plus = np.sum(signs > 0)
    n_minus = np.sum(signs < 0)
    print(f"    step={step}, z={z:.3f}, N+={n_plus:,}, N-={n_minus:,}")

    P = compute_global_purity(pos, signs, box)
    print(f"    P={P:.3f}")

    # Subsample for visualization
    n = len(signs)
    sample_size = min(200000, n)
    idx = np.random.choice(n, sample_size, replace=False)
    pos_s = pos[idx]
    signs_s = signs[idx]

    pos_plus = pos_s[signs_s > 0]
    pos_minus = pos_s[signs_s < 0]

    half_box = box / 2
    extent = [-half_box, half_box, -half_box, half_box]

    # Create figure - 4K resolution
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)

    # Header
    fig.suptitle(f'Petit Pure TreePM v3  |  μ={MU}, λ₀={LAMBDA_0}  |  z={z:.3f}  |  P={P:.3f}  |  step {step}/{TOTAL_STEPS}',
                 fontsize=28, fontweight='bold', y=0.98)

    # Left column: 3 scatter plots (25% width)
    # X-Y projection
    ax1 = fig.add_axes([0.01, 0.67, 0.23, 0.28])
    ax1.scatter(pos_plus[:, 0], pos_plus[:, 1], s=0.3, alpha=0.3, c='blue', rasterized=True)
    ax1.scatter(pos_minus[:, 0], pos_minus[:, 1], s=0.3, alpha=0.3, c='red', rasterized=True)
    ax1.set_xlim(-half_box, half_box)
    ax1.set_ylim(-half_box, half_box)
    ax1.set_xlabel('X [Mpc]', fontsize=10)
    ax1.set_ylabel('Y [Mpc]', fontsize=10)
    ax1.set_title('X-Y Projection', fontsize=12)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=8)

    # X-Z projection
    ax2 = fig.add_axes([0.01, 0.36, 0.23, 0.28])
    ax2.scatter(pos_plus[:, 0], pos_plus[:, 2], s=0.3, alpha=0.3, c='blue', rasterized=True)
    ax2.scatter(pos_minus[:, 0], pos_minus[:, 2], s=0.3, alpha=0.3, c='red', rasterized=True)
    ax2.set_xlim(-half_box, half_box)
    ax2.set_ylim(-half_box, half_box)
    ax2.set_xlabel('X [Mpc]', fontsize=10)
    ax2.set_ylabel('Z [Mpc]', fontsize=10)
    ax2.set_title('X-Z Projection', fontsize=12)
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=8)

    # Y-Z projection
    ax3 = fig.add_axes([0.01, 0.05, 0.23, 0.28])
    ax3.scatter(pos_plus[:, 1], pos_plus[:, 2], s=0.3, alpha=0.3, c='blue', rasterized=True)
    ax3.scatter(pos_minus[:, 1], pos_minus[:, 2], s=0.3, alpha=0.3, c='red', rasterized=True)
    ax3.set_xlim(-half_box, half_box)
    ax3.set_ylim(-half_box, half_box)
    ax3.set_xlabel('Y [Mpc]', fontsize=10)
    ax3.set_ylabel('Z [Mpc]', fontsize=10)
    ax3.set_title('Y-Z Projection', fontsize=12)
    ax3.set_aspect('equal')
    ax3.tick_params(labelsize=8)

    # Center column: 3D isometric view (50% width)
    ax_3d = fig.add_axes([0.26, 0.05, 0.48, 0.88], projection='3d')

    # Subsample even more for 3D
    sample_3d = min(50000, len(pos_plus)), min(50000, len(pos_minus))
    idx_p = np.random.choice(len(pos_plus), sample_3d[0], replace=False) if len(pos_plus) > sample_3d[0] else np.arange(len(pos_plus))
    idx_m = np.random.choice(len(pos_minus), sample_3d[1], replace=False) if len(pos_minus) > sample_3d[1] else np.arange(len(pos_minus))

    ax_3d.scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 1], pos_plus[idx_p, 2],
                  s=0.8, alpha=0.3, c='blue', rasterized=True)
    ax_3d.scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 1], pos_minus[idx_m, 2],
                  s=0.8, alpha=0.3, c='red', rasterized=True)

    # Wireframe box
    corners = half_box * np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                    [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    for e in edges:
        ax_3d.plot3D(*zip(corners[e[0]], corners[e[1]]), 'k-', alpha=0.3, linewidth=0.5)

    ax_3d.set_xlim(-half_box, half_box)
    ax_3d.set_ylim(-half_box, half_box)
    ax_3d.set_zlim(-half_box, half_box)
    ax_3d.view_init(elev=35, azim=30)
    ax_3d.set_xlabel('X [Mpc]', fontsize=10)
    ax_3d.set_ylabel('Y [Mpc]', fontsize=10)
    ax_3d.set_zlabel('Z [Mpc]', fontsize=10)
    ax_3d.set_title('Isometric View (m+ blue, m- red)', fontsize=14)
    ax_3d.tick_params(labelsize=8)

    # Right column: Purity maps (25% width)
    purity, density = compute_purity_map_2d(pos, signs, box, n_cells=64)
    abs_purity = np.abs(purity)

    # Purity map
    ax4 = fig.add_axes([0.76, 0.67, 0.22, 0.28])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im1 = ax4.imshow(purity.T, origin='lower', extent=extent, cmap='coolwarm_r', norm=norm)
    ax4.set_xlabel('X [Mpc]', fontsize=10)
    ax4.set_ylabel('Y [Mpc]', fontsize=10)
    ax4.set_title('Purity (blue=m+, red=m-)', fontsize=12)
    ax4.tick_params(labelsize=8)
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    # |Purity|
    ax5 = fig.add_axes([0.76, 0.36, 0.22, 0.28])
    im2 = ax5.imshow(abs_purity.T, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=1)
    ax5.set_xlabel('X [Mpc]', fontsize=10)
    ax5.set_ylabel('Y [Mpc]', fontsize=10)
    ax5.set_title('|Purity| (0=mixed, 1=pure)', fontsize=12)
    ax5.tick_params(labelsize=8)
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    # Density (with Gaussian smoothing)
    ax6 = fig.add_axes([0.76, 0.05, 0.22, 0.28])
    log_density = np.log10(density + 1)
    log_density_smooth = gaussian_filter(log_density, sigma=1.5)
    im3 = ax6.imshow(log_density_smooth.T, origin='lower', extent=extent, cmap='viridis')
    ax6.set_xlabel('X [Mpc]', fontsize=10)
    ax6.set_ylabel('Y [Mpc]', fontsize=10)
    ax6.set_title('log₁₀(N+1) Total Density', fontsize=12)
    ax6.tick_params(labelsize=8)
    plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=100, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")

def main():
    snap_dir = Path("/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3/snapshots")
    out_dir = Path("/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3/frames_composite")
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = set()

    print("=" * 60)
    print("  Petit Pure 20M TreePM v3 — Composite Renderer")
    print("=" * 60)
    print(f"  Watching: {snap_dir}")
    print(f"  Output:   {out_dir}")
    print("=" * 60)

    while True:
        # Find all snapshots
        if snap_dir.exists():
            snaps = sorted(snap_dir.glob("snap_*.bin"))

            for snap in snaps:
                if snap.name in processed:
                    continue

                # Wait for file to be fully written (500MB = ~500M bytes)
                if snap.stat().st_size < 400_000_000:
                    continue

                try:
                    output = out_dir / f"frame_{snap.stem}.png"
                    if output.exists():
                        processed.add(snap.name)
                        continue
                    render_composite(snap, output)
                    processed.add(snap.name)
                except Exception as e:
                    print(f"Error processing {snap.name}: {e}")

        # Check if simulation is done (step 2000)
        if any("02000" in s.name for s in snaps if snap_dir.exists()):
            print("Simulation complete!")
            break

        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    main()
