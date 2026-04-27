#!/usr/bin/env python3
"""
Render frames for scan_mu runs (2M particles, 25-byte format)
Same style as petit_pure_20m_treepm_v3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from pathlib import Path
import struct
import sys

BOX_SIZE = 500.0
TOTAL_STEPS = 2000

def read_snapshot(path, max_particles=None):
    """Read binary snapshot file"""
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

def compute_metrics(pos, signs, box, n_cells=32):
    """Compute purity, void_frac, wall_frac"""
    cell_size = box / n_cells
    half_box = box / 2.0

    n_plus = np.zeros((n_cells, n_cells, n_cells), dtype=np.int32)
    n_minus = np.zeros((n_cells, n_cells, n_cells), dtype=np.int32)

    x = ((pos[:, 0] + half_box) % box / cell_size).astype(int).clip(0, n_cells-1)
    y = ((pos[:, 1] + half_box) % box / cell_size).astype(int).clip(0, n_cells-1)
    z = ((pos[:, 2] + half_box) % box / cell_size).astype(int).clip(0, n_cells-1)

    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus[x[i], y[i], z[i]] += 1
        else:
            n_minus[x[i], y[i], z[i]] += 1

    total = n_plus + n_minus
    purity = np.zeros_like(total, dtype=float)
    mask = total > 0
    purity[mask] = np.abs(n_plus[mask] - n_minus[mask]) / total[mask]
    P = np.sum(purity * total) / np.sum(total) if np.sum(total) > 0 else 0

    # Void/wall fractions
    occupied = total > 0
    void_cells = np.sum((n_minus[occupied] / total[occupied]) > 0.90)
    wall_cells = np.sum((n_plus[occupied] / total[occupied]) > 0.90)
    n_occupied = np.sum(occupied)

    void_frac = void_cells / n_occupied if n_occupied > 0 else 0
    wall_frac = wall_cells / n_occupied if n_occupied > 0 else 0

    return P, void_frac, wall_frac

def render_frame(snap_path, output_path, mu):
    """Render a single frame"""
    print(f"  Reading {snap_path.name}...")
    pos, vel, signs, step, z, box = read_snapshot(snap_path)

    n_plus = np.sum(signs > 0)
    n_minus = np.sum(signs < 0)
    P, void_frac, wall_frac = compute_metrics(pos, signs, box)

    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
    v_plus = np.mean(v_mag[signs > 0]) if n_plus > 0 else 0
    v_minus = np.mean(v_mag[signs < 0]) if n_minus > 0 else 0

    print(f"    step={step}, z={z:.3f}, P={P:.3f}, void={void_frac*100:.1f}%")

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

    # Create figure
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)

    # Header
    fig.suptitle(f'Scan μ={mu}  |  λ₀=0  |  z={z:.3f}  |  P={P:.3f}  |  void={void_frac*100:.1f}%  |  wall={wall_frac*100:.1f}%  |  step {step}/{TOTAL_STEPS}',
                 fontsize=24, fontweight='bold', y=0.98)

    # Left: 2 projections + velocity histogram
    ax1 = fig.add_axes([0.01, 0.67, 0.18, 0.28])
    ax1.scatter(pos_plus[:, 0], pos_plus[:, 1], s=0.3, alpha=0.4, c='blue', rasterized=True)
    ax1.scatter(pos_minus[:, 0], pos_minus[:, 1], s=0.3, alpha=0.4, c='red', rasterized=True)
    ax1.set_xlim(-half_box, half_box)
    ax1.set_ylim(-half_box, half_box)
    ax1.set_xlabel('X [Mpc]', fontsize=9)
    ax1.set_ylabel('Y [Mpc]', fontsize=9)
    ax1.set_title('X-Y', fontsize=11)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=7)

    ax2 = fig.add_axes([0.01, 0.36, 0.18, 0.28])
    ax2.scatter(pos_plus[:, 0], pos_plus[:, 2], s=0.3, alpha=0.4, c='blue', rasterized=True)
    ax2.scatter(pos_minus[:, 0], pos_minus[:, 2], s=0.3, alpha=0.4, c='red', rasterized=True)
    ax2.set_xlim(-half_box, half_box)
    ax2.set_ylim(-half_box, half_box)
    ax2.set_xlabel('X [Mpc]', fontsize=9)
    ax2.set_ylabel('Z [Mpc]', fontsize=9)
    ax2.set_title('X-Z', fontsize=11)
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=7)

    # Velocity histogram
    ax3 = fig.add_axes([0.01, 0.05, 0.18, 0.28])
    v_plus_arr = v_mag[signs > 0]
    v_minus_arr = v_mag[signs < 0]
    if len(v_plus_arr) > 0 and len(v_minus_arr) > 0:
        v_max = np.percentile(v_mag, 99)
        bins = np.linspace(0, v_max, 50)
        ax3.hist(v_plus_arr, bins=bins, alpha=0.6, color='blue', label=f'm+ <v>={v_plus:.0f}', density=True)
        ax3.hist(v_minus_arr, bins=bins, alpha=0.6, color='red', label=f'm- <v>={v_minus:.0f}', density=True)
        ax3.legend(fontsize=8)
    ax3.set_xlabel('|v| [Mpc/τ]', fontsize=9)
    ax3.set_ylabel('Density', fontsize=9)
    ax3.set_title('Velocity Distribution', fontsize=11)
    ax3.tick_params(labelsize=7)

    # Center: 3D view
    ax_3d = fig.add_axes([0.21, 0.05, 0.40, 0.88], projection='3d')

    sample_3d_p = min(30000, len(pos_plus))
    sample_3d_m = min(30000, len(pos_minus))
    idx_p = np.random.choice(len(pos_plus), sample_3d_p, replace=False) if len(pos_plus) > sample_3d_p else np.arange(len(pos_plus))
    idx_m = np.random.choice(len(pos_minus), sample_3d_m, replace=False) if len(pos_minus) > sample_3d_m else np.arange(len(pos_minus))

    ax_3d.scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 1], pos_plus[idx_p, 2],
                  s=1.0, alpha=0.4, c='blue', rasterized=True)
    ax_3d.scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 1], pos_minus[idx_m, 2],
                  s=1.0, alpha=0.4, c='red', rasterized=True)

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
    ax_3d.set_title(f'μ={mu}  (N+={n_plus:,}, N-={n_minus:,})', fontsize=12)
    ax_3d.tick_params(labelsize=7)

    # Right: Purity maps
    n_cells = 64
    cell_size = box / n_cells
    n_plus_grid = np.zeros((n_cells, n_cells), dtype=np.int32)
    n_minus_grid = np.zeros((n_cells, n_cells), dtype=np.int32)

    x = ((pos[:, 0] + half_box) % box / cell_size).astype(int).clip(0, n_cells-1)
    y = ((pos[:, 1] + half_box) % box / cell_size).astype(int).clip(0, n_cells-1)

    for i in range(len(signs)):
        if signs[i] > 0:
            n_plus_grid[x[i], y[i]] += 1
        else:
            n_minus_grid[x[i], y[i]] += 1

    total = n_plus_grid + n_minus_grid
    purity_map = np.zeros((n_cells, n_cells))
    mask = total > 0
    purity_map[mask] = (n_plus_grid[mask] - n_minus_grid[mask]) / total[mask]

    ax4 = fig.add_axes([0.63, 0.67, 0.17, 0.28])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im1 = ax4.imshow(purity_map.T, origin='lower', extent=extent, cmap='coolwarm_r', norm=norm)
    ax4.set_xlabel('X [Mpc]', fontsize=9)
    ax4.set_ylabel('Y [Mpc]', fontsize=9)
    ax4.set_title('Purity', fontsize=11)
    ax4.tick_params(labelsize=7)
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_axes([0.63, 0.36, 0.17, 0.28])
    abs_purity = np.abs(purity_map)
    im2 = ax5.imshow(abs_purity.T, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=1)
    ax5.set_xlabel('X [Mpc]', fontsize=9)
    ax5.set_ylabel('Y [Mpc]', fontsize=9)
    ax5.set_title('|Purity|', fontsize=11)
    ax5.tick_params(labelsize=7)
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_axes([0.63, 0.05, 0.17, 0.28])
    log_density = np.log10(total + 1)
    log_density_smooth = gaussian_filter(log_density, sigma=1.5)
    im3 = ax6.imshow(log_density_smooth.T, origin='lower', extent=extent, cmap='viridis')
    ax6.set_xlabel('X [Mpc]', fontsize=9)
    ax6.set_ylabel('Y [Mpc]', fontsize=9)
    ax6.set_title('log₁₀(N+1) Density', fontsize=11)
    ax6.tick_params(labelsize=7)
    plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)

    # Extra stats panel
    ax7 = fig.add_axes([0.82, 0.67, 0.17, 0.28])
    ax7.axis('off')
    stats_text = f"""
    μ = {mu}
    N+ = {n_plus:,}
    N- = {n_minus:,}

    z = {z:.3f}
    P = {P:.4f}

    void_frac = {void_frac*100:.1f}%
    wall_frac = {wall_frac*100:.1f}%

    <v+> = {v_plus:.1f} Mpc/τ
    <v-> = {v_minus:.1f} Mpc/τ
    """
    ax7.text(0.1, 0.9, stats_text, fontsize=14, fontfamily='monospace',
             verticalalignment='top', transform=ax7.transAxes)

    plt.savefig(output_path, dpi=100, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")

def render_run(mu, steps=[0, 500, 1000, 1500, 2000], n_millions=None, box_mpc=None):
    """Render selected frames for a given mu"""
    # Support various naming conventions
    if n_millions and box_mpc:
        run_dir = Path(f"/mnt/T2/janus-sim/output/scan_mu_{mu}_{n_millions}M_{box_mpc}Mpc")
    elif n_millions:
        run_dir = Path(f"/mnt/T2/janus-sim/output/scan_mu_{mu}_{n_millions}M")
    else:
        run_dir = Path(f"/mnt/T2/janus-sim/output/scan_mu_{mu}")

    snap_dir = run_dir / "snapshots"
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(exist_ok=True)

    if not snap_dir.exists():
        print(f"No snapshots for μ={mu}")
        return

    for step in steps:
        snap_path = snap_dir / f"snap_{step:05d}.bin"
        if snap_path.exists():
            # Include mu in filename
            out_path = frame_dir / f"mu{mu}_step{step:05d}.png"
            render_frame(snap_path, out_path, mu)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_scan_mu.py <mu> [--n <millions>] [--box <mpc>] [steps...]")
        print("Example: python render_scan_mu.py 16 0 500 1000 2000")
        print("Example: python render_scan_mu.py 20 --n 5 0 1000 2000")
        print("Example: python render_scan_mu.py 32 --n 5 --box 1000 2000")
        sys.exit(1)

    mu = int(sys.argv[1])

    # Parse --n and --box arguments
    n_millions = None
    box_mpc = None
    steps = [0, 500, 1000, 1500, 2000]

    args = sys.argv[2:]
    if "--n" in args:
        idx = args.index("--n")
        n_millions = int(args[idx + 1])
        args = args[:idx] + args[idx+2:]

    if "--box" in args:
        idx = args.index("--box")
        box_mpc = int(args[idx + 1])
        args = args[:idx] + args[idx+2:]

    if args:
        steps = [int(s) for s in args]

    desc = f"μ={mu}"
    if n_millions:
        desc += f" ({n_millions}M)"
    if box_mpc:
        desc += f" [{box_mpc}Mpc]"
    print(f"Rendering {desc} at steps {steps}")
    render_run(mu, steps, n_millions, box_mpc)
