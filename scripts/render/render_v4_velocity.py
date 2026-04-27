#!/usr/bin/env python3
"""
4K Composite Frame Renderer with Velocity Plots
For petit_pure_20m_treepm_v3 (25-byte particle format with velocities)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from pathlib import Path
import struct
import time

# Constants
MU = 8.0
LAMBDA_0 = 0.0
BOX_SIZE = 500.0
TOTAL_STEPS = 2000

def read_snapshot_with_vel(path, max_particles=None):
    """Read binary snapshot file with velocities"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]

        particle_size = 25
        n_read = min(n, max_particles) if max_particles else n

        if max_particles and n > max_particles:
            indices = np.sort(np.random.choice(n, n_read, replace=False))
            pos = np.zeros((n_read, 3), dtype=np.float32)
            vel = np.zeros((n_read, 3), dtype=np.float32)
            signs = np.zeros(n_read, dtype=np.int8)

            for i, idx in enumerate(indices):
                f.seek(16 + idx * particle_size)
                pos[i, 0] = struct.unpack('<f', f.read(4))[0]
                pos[i, 1] = struct.unpack('<f', f.read(4))[0]
                pos[i, 2] = struct.unpack('<f', f.read(4))[0]
                vel[i, 0] = struct.unpack('<f', f.read(4))[0]
                vel[i, 1] = struct.unpack('<f', f.read(4))[0]
                vel[i, 2] = struct.unpack('<f', f.read(4))[0]
                signs[i] = struct.unpack('<b', f.read(1))[0]
        else:
            pos = np.zeros((n, 3), dtype=np.float32)
            vel = np.zeros((n, 3), dtype=np.float32)
            signs = np.zeros(n, dtype=np.int8)

            for i in range(n):
                pos[i, 0] = struct.unpack('<f', f.read(4))[0]
                pos[i, 1] = struct.unpack('<f', f.read(4))[0]
                pos[i, 2] = struct.unpack('<f', f.read(4))[0]
                vel[i, 0] = struct.unpack('<f', f.read(4))[0]
                vel[i, 1] = struct.unpack('<f', f.read(4))[0]
                vel[i, 2] = struct.unpack('<f', f.read(4))[0]
                signs[i] = struct.unpack('<b', f.read(1))[0]

    return pos, vel, signs, step, z, box

def compute_global_purity(pos, signs, box_size, n_cells=32):
    """Compute global purity metric P"""
    cell_size = box_size / n_cells
    half_box = box_size / 2.0
    n_cells_cubed = n_cells ** 3

    n_plus = np.zeros(n_cells_cubed, dtype=np.int32)
    n_minus = np.zeros(n_cells_cubed, dtype=np.int32)

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

def compute_velocity_field(pos, vel, box, n_cells=32):
    """Compute mean velocity field on 2D grid (XY projection)"""
    cell_size = box / n_cells
    half_box = box / 2.0

    vx_sum = np.zeros((n_cells, n_cells))
    vy_sum = np.zeros((n_cells, n_cells))
    counts = np.zeros((n_cells, n_cells))

    x = (pos[:, 0] + half_box) % box
    y = (pos[:, 1] + half_box) % box

    ix = np.clip((x / cell_size).astype(int), 0, n_cells - 1)
    iy = np.clip((y / cell_size).astype(int), 0, n_cells - 1)

    np.add.at(vx_sum, (ix, iy), vel[:, 0])
    np.add.at(vy_sum, (ix, iy), vel[:, 1])
    np.add.at(counts, (ix, iy), 1)

    mask = counts > 0
    vx_mean = np.zeros_like(vx_sum)
    vy_mean = np.zeros_like(vy_sum)
    vx_mean[mask] = vx_sum[mask] / counts[mask]
    vy_mean[mask] = vy_sum[mask] / counts[mask]

    return vx_mean, vy_mean

def render_composite_velocity(snap_path, output_path):
    """Render 4K composite frame with velocity analysis"""
    print(f"  Reading {snap_path.name}...")
    pos, vel, signs, step, z, box = read_snapshot_with_vel(snap_path, max_particles=2_000_000)

    n_plus = np.sum(signs > 0)
    n_minus = np.sum(signs < 0)
    print(f"    step={step}, z={z:.3f}, N+={n_plus:,}, N-={n_minus:,}")

    P = compute_global_purity(pos, signs, box)

    # Velocity magnitudes
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
    v_plus = v_mag[signs > 0]
    v_minus = v_mag[signs < 0]

    # Mean velocities
    v_mean_plus = np.mean(v_plus) if len(v_plus) > 0 else 0
    v_mean_minus = np.mean(v_minus) if len(v_minus) > 0 else 0
    print(f"    P={P:.3f}, <v+>={v_mean_plus:.2f}, <v->={v_mean_minus:.2f}")

    # Subsample for visualization
    n = len(signs)
    sample_size = min(200000, n)
    idx = np.random.choice(n, sample_size, replace=False)
    pos_s = pos[idx]
    vel_s = vel[idx]
    signs_s = signs[idx]

    pos_plus = pos_s[signs_s > 0]
    pos_minus = pos_s[signs_s < 0]

    half_box = box / 2
    extent = [-half_box, half_box, -half_box, half_box]

    # Create figure - 4K resolution with extra row
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)

    # Header
    fig.suptitle(f'Petit Pure TreePM v3  |  μ={MU}, λ₀={LAMBDA_0}  |  z={z:.3f}  |  P={P:.3f}  |  <v+>={v_mean_plus:.1f}  <v->={v_mean_minus:.1f}  |  step {step}/{TOTAL_STEPS}',
                 fontsize=24, fontweight='bold', y=0.98)

    # Left column: 2 projections + velocity histogram
    # X-Y projection
    ax1 = fig.add_axes([0.01, 0.67, 0.18, 0.28])
    ax1.scatter(pos_plus[:, 0], pos_plus[:, 1], s=0.2, alpha=0.3, c='blue', rasterized=True)
    ax1.scatter(pos_minus[:, 0], pos_minus[:, 1], s=0.2, alpha=0.3, c='red', rasterized=True)
    ax1.set_xlim(-half_box, half_box)
    ax1.set_ylim(-half_box, half_box)
    ax1.set_xlabel('X [Mpc]', fontsize=9)
    ax1.set_ylabel('Y [Mpc]', fontsize=9)
    ax1.set_title('X-Y', fontsize=11)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=7)

    # X-Z projection
    ax2 = fig.add_axes([0.01, 0.36, 0.18, 0.28])
    ax2.scatter(pos_plus[:, 0], pos_plus[:, 2], s=0.2, alpha=0.3, c='blue', rasterized=True)
    ax2.scatter(pos_minus[:, 0], pos_minus[:, 2], s=0.2, alpha=0.3, c='red', rasterized=True)
    ax2.set_xlim(-half_box, half_box)
    ax2.set_ylim(-half_box, half_box)
    ax2.set_xlabel('X [Mpc]', fontsize=9)
    ax2.set_ylabel('Z [Mpc]', fontsize=9)
    ax2.set_title('X-Z', fontsize=11)
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=7)

    # Velocity histogram
    ax3 = fig.add_axes([0.01, 0.05, 0.18, 0.28])
    v_max = np.percentile(v_mag, 99)
    bins = np.linspace(0, v_max, 50)
    ax3.hist(v_plus, bins=bins, alpha=0.6, color='blue', label=f'm+ <v>={v_mean_plus:.1f}', density=True)
    ax3.hist(v_minus, bins=bins, alpha=0.6, color='red', label=f'm- <v>={v_mean_minus:.1f}', density=True)
    ax3.set_xlabel('|v| [Mpc/τ]', fontsize=9)
    ax3.set_ylabel('Density', fontsize=9)
    ax3.set_title('Velocity Distribution', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=7)

    # Center column: 3D isometric view
    ax_3d = fig.add_axes([0.21, 0.05, 0.40, 0.88], projection='3d')

    sample_3d = min(50000, len(pos_plus)), min(50000, len(pos_minus))
    idx_p = np.random.choice(len(pos_plus), sample_3d[0], replace=False) if len(pos_plus) > sample_3d[0] else np.arange(len(pos_plus))
    idx_m = np.random.choice(len(pos_minus), sample_3d[1], replace=False) if len(pos_minus) > sample_3d[1] else np.arange(len(pos_minus))

    ax_3d.scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 1], pos_plus[idx_p, 2],
                  s=0.8, alpha=0.3, c='blue', rasterized=True)
    ax_3d.scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 1], pos_minus[idx_m, 2],
                  s=0.8, alpha=0.3, c='red', rasterized=True)

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
    ax_3d.set_xlabel('X [Mpc]', fontsize=9)
    ax_3d.set_ylabel('Y [Mpc]', fontsize=9)
    ax_3d.set_zlabel('Z [Mpc]', fontsize=9)
    ax_3d.set_title('Isometric View (m+ blue, m- red)', fontsize=12)
    ax_3d.tick_params(labelsize=7)

    # Right column: Purity + Velocity field
    # Purity map
    from matplotlib.colors import LogNorm
    n_cells = 64
    cell_size = box / n_cells

    n_plus_grid = np.zeros((n_cells, n_cells), dtype=np.int32)
    n_minus_grid = np.zeros((n_cells, n_cells), dtype=np.int32)
    x = (pos[:, 0] + half_box) % box
    y = (pos[:, 1] + half_box) % box
    ix = np.clip((x / cell_size).astype(int), 0, n_cells - 1)
    iy = np.clip((y / cell_size).astype(int), 0, n_cells - 1)
    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus_grid[ix[i], iy[i]] += 1
        else:
            n_minus_grid[ix[i], iy[i]] += 1
    total = n_plus_grid + n_minus_grid
    purity = np.zeros((n_cells, n_cells))
    mask = total > 0
    purity[mask] = (n_plus_grid[mask] - n_minus_grid[mask]) / total[mask]

    ax4 = fig.add_axes([0.63, 0.67, 0.17, 0.28])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im1 = ax4.imshow(purity.T, origin='lower', extent=extent, cmap='coolwarm_r', norm=norm)
    ax4.set_xlabel('X [Mpc]', fontsize=9)
    ax4.set_ylabel('Y [Mpc]', fontsize=9)
    ax4.set_title('Purity', fontsize=11)
    ax4.tick_params(labelsize=7)
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    # Velocity field (quiver)
    ax5 = fig.add_axes([0.63, 0.36, 0.17, 0.28])
    vx_mean, vy_mean = compute_velocity_field(pos, vel, box, n_cells=16)
    v_field_mag = np.sqrt(vx_mean**2 + vy_mean**2)

    cell_centers = np.linspace(-half_box + box/(2*16), half_box - box/(2*16), 16)
    X, Y = np.meshgrid(cell_centers, cell_centers)

    ax5.quiver(X, Y, vx_mean.T, vy_mean.T, v_field_mag.T, cmap='plasma', scale=v_field_mag.max()*20)
    ax5.set_xlim(-half_box, half_box)
    ax5.set_ylim(-half_box, half_box)
    ax5.set_xlabel('X [Mpc]', fontsize=9)
    ax5.set_ylabel('Y [Mpc]', fontsize=9)
    ax5.set_title('Mean Velocity Field', fontsize=11)
    ax5.set_aspect('equal')
    ax5.tick_params(labelsize=7)

    # Density
    ax6 = fig.add_axes([0.63, 0.05, 0.17, 0.28])
    log_density = np.log10(total + 1)
    log_density_smooth = gaussian_filter(log_density, sigma=1.5)
    im3 = ax6.imshow(log_density_smooth.T, origin='lower', extent=extent, cmap='viridis')
    ax6.set_xlabel('X [Mpc]', fontsize=9)
    ax6.set_ylabel('Y [Mpc]', fontsize=9)
    ax6.set_title('log₁₀(N+1) Density', fontsize=11)
    ax6.tick_params(labelsize=7)
    plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)

    # Extra right column: velocity dispersion + radial velocity
    # Velocity dispersion per species
    ax7 = fig.add_axes([0.82, 0.67, 0.17, 0.28])
    sigma_plus = np.std(v_plus) if len(v_plus) > 0 else 0
    sigma_minus = np.std(v_minus) if len(v_minus) > 0 else 0

    # Radial bins for sigma(r)
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    r_bins = np.linspace(0, half_box * np.sqrt(3), 20)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    sigma_r_plus = []
    sigma_r_minus = []
    for i in range(len(r_bins) - 1):
        mask_r = (r >= r_bins[i]) & (r < r_bins[i+1])
        mask_plus = mask_r & (signs > 0)
        mask_minus = mask_r & (signs < 0)
        sigma_r_plus.append(np.std(v_mag[mask_plus]) if np.sum(mask_plus) > 10 else np.nan)
        sigma_r_minus.append(np.std(v_mag[mask_minus]) if np.sum(mask_minus) > 10 else np.nan)

    ax7.plot(r_centers, sigma_r_plus, 'b-', linewidth=2, label=f'm+ σ={sigma_plus:.1f}')
    ax7.plot(r_centers, sigma_r_minus, 'r-', linewidth=2, label=f'm- σ={sigma_minus:.1f}')
    ax7.set_xlabel('r [Mpc]', fontsize=9)
    ax7.set_ylabel('σ_v [Mpc/τ]', fontsize=9)
    ax7.set_title('Velocity Dispersion vs r', fontsize=11)
    ax7.legend(fontsize=8)
    ax7.tick_params(labelsize=7)
    ax7.grid(alpha=0.3)

    # Radial velocity
    ax8 = fig.add_axes([0.82, 0.36, 0.17, 0.28])
    r_safe = np.maximum(r, 1e-6)
    v_radial = (pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1] + pos[:, 2] * vel[:, 2]) / r_safe

    v_r_plus = v_radial[signs > 0]
    v_r_minus = v_radial[signs < 0]

    bins_vr = np.linspace(np.percentile(v_radial, 1), np.percentile(v_radial, 99), 50)
    ax8.hist(v_r_plus, bins=bins_vr, alpha=0.6, color='blue', label='m+', density=True)
    ax8.hist(v_r_minus, bins=bins_vr, alpha=0.6, color='red', label='m-', density=True)
    ax8.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('v_radial [Mpc/τ]', fontsize=9)
    ax8.set_ylabel('Density', fontsize=9)
    ax8.set_title('Radial Velocity', fontsize=11)
    ax8.legend(fontsize=8)
    ax8.tick_params(labelsize=7)

    # |Purity| map
    ax9 = fig.add_axes([0.82, 0.05, 0.17, 0.28])
    abs_purity = np.abs(purity)
    im4 = ax9.imshow(abs_purity.T, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=1)
    ax9.set_xlabel('X [Mpc]', fontsize=9)
    ax9.set_ylabel('Y [Mpc]', fontsize=9)
    ax9.set_title('|Purity|', fontsize=11)
    ax9.tick_params(labelsize=7)
    plt.colorbar(im4, ax=ax9, fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=100, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")

def main():
    snap_dir = Path("/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3/snapshots")
    out_dir = Path("/mnt/T2/janus-sim/output/petit_pure_20m_treepm_v3/frames_velocity")
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = set()

    print("=" * 60)
    print("  Petit Pure 20M TreePM v3 — Velocity Renderer")
    print("=" * 60)

    while True:
        if snap_dir.exists():
            snaps = sorted(snap_dir.glob("snap_*.bin"))

            for snap in snaps:
                if snap.name in processed:
                    continue

                if snap.stat().st_size < 400_000_000:
                    continue

                try:
                    output = out_dir / f"frame_{snap.stem}.png"
                    if output.exists():
                        processed.add(snap.name)
                        continue
                    render_composite_velocity(snap, output)
                    processed.add(snap.name)
                except Exception as e:
                    print(f"Error processing {snap.name}: {e}")

        if any("02000" in s.name for s in snaps if snap_dir.exists()):
            print("Simulation complete!")
            break

        time.sleep(10)

if __name__ == "__main__":
    main()
