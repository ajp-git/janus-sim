#!/usr/bin/env python3
"""
Halo Renderer Daemon — Generates detailed halo analysis frames
For each snapshot, renders 4 halos with:
  - XY projection (colored by split_level)
  - XZ projection
  - Radial density profile ρ(r)
  - Histogram of split_levels
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import struct
import time
import argparse
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===
MARGIN = 50.0  # Mpc — zone interdite sur chaque bord
# La boîte fait ±250 Mpc, zone autorisée : [-200, 200] Mpc

# === SNAPSHOT READER (V3 format) ===
def read_snapshot_v3(path):
    """Read v3 snapshot with split_level info"""
    with open(path, 'rb') as f:
        header = f.read(408)

        n = struct.unpack('<Q', header[16:24])[0]
        a = struct.unpack('<d', header[24:32])[0]
        l_box = struct.unpack('<d', header[40:48])[0]
        z = 1.0 / a - 1.0

        # Particle dtype: ParticleV3 uses f32 for positions/velocities
        dt = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'),
            ('mass', '<f4'),
            ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'), ('is_star', 'u1'), ('flags', 'u1'),
        ])

        particles = np.fromfile(f, dtype=dt, count=n)

    pos = np.column_stack([particles['x'], particles['y'], particles['z']])
    vel = np.column_stack([particles['vx'], particles['vy'], particles['vz']])

    return {
        'pos': pos,
        'vel': vel,
        'sign': particles['sign'],
        'split_level': particles['split_level'],
        'mass': particles['mass'],
        'z': z,
        'a': a,
        'l_box': l_box,
        'n': n
    }

def read_analysis_csv(path):
    """Read analysis.csv and extract halo positions per step"""
    halos_by_step = {}

    try:
        with open(path, 'r') as f:
            header = f.readline().strip().split(',')

            # Find column indices
            step_idx = header.index('step') if 'step' in header else 0

            # Look for halo columns (halo0_x, halo0_y, halo0_z, etc.)
            halo_cols = {}
            for i in range(0, 10):  # Up to 10 halos (0-indexed)
                x_col = f'halo{i}_x'
                y_col = f'halo{i}_y'
                z_col = f'halo{i}_z'
                if x_col in header and y_col in header and z_col in header:
                    halo_cols[i] = (header.index(x_col), header.index(y_col), header.index(z_col))

            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue

                try:
                    step = int(parts[step_idx])
                    halos = []

                    for i in sorted(halo_cols.keys()):
                        xi, yi, zi = halo_cols[i]
                        x = float(parts[xi]) if parts[xi] else np.nan
                        y = float(parts[yi]) if parts[yi] else np.nan
                        z = float(parts[zi]) if parts[zi] else np.nan

                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                            halos.append(np.array([x, y, z]))

                    if halos:
                        halos_by_step[step] = halos

                except (ValueError, IndexError):
                    continue

    except FileNotFoundError:
        pass

    return halos_by_step

def find_density_peaks(pos, signs, l_box, n_halos=4, grid_size=64, border=20.0):
    """Find density peaks as halo candidates (fallback if no analysis.csv)

    Args:
        border: Exclude peaks within this distance from box edges (Mpc)
                Default 20 Mpc to avoid edge artifacts
    """
    half = l_box / 2

    # Only use m+ particles for halo detection
    is_plus = signs == 1
    pos_plus = pos[is_plus]

    if len(pos_plus) < 100:
        return []

    # Grid density
    cell = l_box / grid_size
    grid = np.zeros((grid_size, grid_size, grid_size))

    ix = ((pos_plus[:, 0] + half) / cell).astype(int) % grid_size
    iy = ((pos_plus[:, 1] + half) / cell).astype(int) % grid_size
    iz = ((pos_plus[:, 2] + half) / cell).astype(int) % grid_size

    np.add.at(grid, (ix, iy, iz), 1)

    # Find peaks
    from scipy.ndimage import maximum_filter, gaussian_filter
    grid_smooth = gaussian_filter(grid, sigma=1.5)
    local_max = maximum_filter(grid_smooth, size=5)
    peaks = (grid_smooth == local_max) & (grid_smooth > np.percentile(grid_smooth, 99))

    peak_coords = np.argwhere(peaks)
    peak_values = grid_smooth[peaks]

    # Sort by density
    order = np.argsort(peak_values)[::-1]
    peak_coords = peak_coords[order]

    # Border exclusion: skip peaks too close to edges
    inner_min = border
    inner_max = l_box - border

    halos = []
    for i in range(len(peak_coords)):
        if len(halos) >= n_halos:
            break

        cx = (peak_coords[i, 0] + 0.5) * cell - half
        cy = (peak_coords[i, 1] + 0.5) * cell - half
        cz = (peak_coords[i, 2] + 0.5) * cell - half

        # Check if inside inner region (away from borders)
        # Coordinates are in [-half, half], so check |coord| < half - border
        if (abs(cx) > half - border or
            abs(cy) > half - border or
            abs(cz) > half - border):
            continue  # Skip edge peaks

        halos.append(np.array([cx, cy, cz]))

    return halos

def render_halo(data, halo_center, halo_idx, step, out_dir, r_extract=15.0):
    """Render 2x2 analysis figure for one halo"""
    pos = data['pos']
    sign = data['sign']
    split_level = data['split_level']
    mass = data['mass']
    z = data['z']
    l_box = data['l_box']
    half = l_box / 2

    # Skip halos too close to edges (within MARGIN of box boundary)
    safe_limit = half - MARGIN  # 250 - 50 = 200 Mpc
    if (abs(halo_center[0]) > safe_limit or
        abs(halo_center[1]) > safe_limit or
        abs(halo_center[2]) > safe_limit):
        return None  # Halo too close to edge

    # Extract particles within r_extract of halo center (with periodic BC)
    dx = pos[:, 0] - halo_center[0]
    dy = pos[:, 1] - halo_center[1]
    dz = pos[:, 2] - halo_center[2]

    # Periodic boundary
    dx = np.where(dx > half, dx - l_box, dx)
    dx = np.where(dx < -half, dx + l_box, dx)
    dy = np.where(dy > half, dy - l_box, dy)
    dy = np.where(dy < -half, dy + l_box, dy)
    dz = np.where(dz > half, dz - l_box, dz)
    dz = np.where(dz < -half, dz + l_box, dz)

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    mask = r < r_extract

    if np.sum(mask) < 10:
        return None

    # Require minimum m+ particles for a valid halo
    n_plus_local = np.sum((sign[mask] == 1))
    if n_plus_local < 100:
        return None  # Not a real halo, skip

    # Extract local particles
    local_dx = dx[mask]
    local_dy = dy[mask]
    local_dz = dz[mask]
    local_r = r[mask]
    local_sign = sign[mask]
    local_split = split_level[mask]
    local_mass = mass[mask]

    # Only m+ for main plots
    is_plus = local_sign == 1

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='black')
    fig.suptitle(f'Halo {halo_idx+1} — Step {step} | z = {z:.2f}\n'
                 f'Center: ({halo_center[0]:.1f}, {halo_center[1]:.1f}, {halo_center[2]:.1f}) Mpc',
                 color='white', fontsize=14)

    # Colormap for split_level
    max_split = max(6, local_split.max() + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, max_split))
    cmap = ListedColormap(colors)

    # === Top Left: XY projection ===
    ax1 = axes[0, 0]
    ax1.set_facecolor('black')
    if np.sum(is_plus) > 0:
        sc = ax1.scatter(local_dx[is_plus], local_dy[is_plus],
                        c=local_split[is_plus], cmap=cmap, vmin=0, vmax=max_split-1,
                        s=1, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax1, label='Split Level')
    ax1.set_xlim(-r_extract, r_extract)
    ax1.set_ylim(-r_extract, r_extract)
    ax1.set_xlabel('ΔX [Mpc]', color='white')
    ax1.set_ylabel('ΔY [Mpc]', color='white')
    ax1.set_title('XY Projection (m+)', color='#44aaff')
    ax1.tick_params(colors='gray')
    ax1.set_aspect('equal')

    # === Top Right: XZ projection ===
    ax2 = axes[0, 1]
    ax2.set_facecolor('black')
    if np.sum(is_plus) > 0:
        sc = ax2.scatter(local_dx[is_plus], local_dz[is_plus],
                        c=local_split[is_plus], cmap=cmap, vmin=0, vmax=max_split-1,
                        s=1, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax2, label='Split Level')
    ax2.set_xlim(-r_extract, r_extract)
    ax2.set_ylim(-r_extract, r_extract)
    ax2.set_xlabel('ΔX [Mpc]', color='white')
    ax2.set_ylabel('ΔZ [Mpc]', color='white')
    ax2.set_title('XZ Projection (m+)', color='#44aaff')
    ax2.tick_params(colors='gray')
    ax2.set_aspect('equal')

    # === Bottom Left: Radial density profile ===
    ax3 = axes[1, 0]
    ax3.set_facecolor('black')

    r_bins = np.linspace(0, r_extract, 30)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Compute density in shells
    rho_plus = np.zeros(len(r_centers))
    rho_minus = np.zeros(len(r_centers))

    for i in range(len(r_centers)):
        shell_mask = (local_r >= r_bins[i]) & (local_r < r_bins[i+1])
        vol = (4/3) * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)

        m_plus = np.sum(local_mass[shell_mask & is_plus])
        m_minus = np.sum(local_mass[shell_mask & ~is_plus])

        rho_plus[i] = m_plus / vol if vol > 0 else 0
        rho_minus[i] = m_minus / vol if vol > 0 else 0

    ax3.semilogy(r_centers, rho_plus + 1e-10, 'b-', linewidth=2, label='m+')
    ax3.semilogy(r_centers, rho_minus + 1e-10, 'r--', linewidth=2, label='m-')
    ax3.set_xlabel('r [Mpc]', color='white')
    ax3.set_ylabel('ρ [M☉/Mpc³]', color='white')
    ax3.set_title('Radial Density Profile', color='white')
    ax3.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    ax3.tick_params(colors='gray')
    ax3.grid(True, alpha=0.2)

    # === Bottom Right: Split level histogram ===
    ax4 = axes[1, 1]
    ax4.set_facecolor('black')

    split_levels_plus = local_split[is_plus]
    if len(split_levels_plus) > 0:
        bins = np.arange(-0.5, max_split + 0.5, 1)
        ax4.hist(split_levels_plus, bins=bins, color='#44aaff', edgecolor='white', alpha=0.7)

    ax4.set_xlabel('Split Level', color='white')
    ax4.set_ylabel('Count', color='white')
    ax4.set_title(f'Split Level Distribution (N_m+ = {np.sum(is_plus):,})', color='white')
    ax4.tick_params(colors='gray')

    # Stats text
    stats_text = f'N_total: {len(local_r):,}\n'
    stats_text += f'N_m+: {np.sum(is_plus):,}\n'
    stats_text += f'N_m-: {np.sum(~is_plus):,}\n'
    stats_text += f'M_total: {np.sum(local_mass):.2e} M☉'
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
             color='white', fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    plt.tight_layout()

    out_path = out_dir / f'frame_halo{halo_idx+1}_step{step:05d}.png'
    fig.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)

    return out_path

def main():
    parser = argparse.ArgumentParser(description='Halo Renderer Daemon')
    parser.add_argument('--snap-dir', type=str, required=True, help='Snapshot directory')
    parser.add_argument('--analysis', type=str, default=None, help='analysis.csv path')
    parser.add_argument('--out-dir', type=str, required=True, help='Output frames directory')
    parser.add_argument('--n-halos', type=int, default=4, help='Number of halos to render')
    parser.add_argument('--r-extract', type=float, default=15.0, help='Extraction radius [Mpc]')
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"=== Halo Renderer Daemon ===")
    print(f"Snap dir: {snap_dir}")
    print(f"Analysis: {args.analysis}")
    print(f"Out dir: {out_dir}")
    print(f"N halos: {args.n_halos}, R extract: {args.r_extract} Mpc")
    print()

    # Track rendered frames
    rendered = set()
    for f in out_dir.glob('frame_halo*_step*.png'):
        try:
            # Extract step from filename
            step = int(f.stem.split('step')[1])
            halo = int(f.stem.split('halo')[1].split('_')[0])
            rendered.add((halo, step))
        except:
            pass

    print(f"Already rendered: {len(rendered)} halo frames")

    # Track attempted steps to avoid re-processing
    attempted_steps = set()

    # Main daemon loop
    while True:
        # Read analysis.csv for halo positions
        halos_by_step = {}
        if args.analysis and Path(args.analysis).exists():
            halos_by_step = read_analysis_csv(args.analysis)

        # Find snapshots
        snaps = sorted(snap_dir.glob('snap_*.bin'))

        for snap_path in snaps:
            try:
                step = int(snap_path.stem.split('_')[1])
            except:
                continue

            # Skip already attempted steps
            if step in attempted_steps:
                continue

            # Check if all halos for this step are rendered
            all_rendered = all((h+1, step) in rendered for h in range(args.n_halos))
            if all_rendered:
                attempted_steps.add(step)
                continue

            # Read snapshot
            try:
                data = read_snapshot_v3(str(snap_path))
            except Exception as e:
                print(f"[ERROR] Failed to read {snap_path}: {e}")
                continue

            # Get halo positions
            if step in halos_by_step:
                halos = halos_by_step[step][:args.n_halos]
            else:
                # Fallback: find density peaks
                halos = find_density_peaks(data['pos'], data['sign'],
                                          data['l_box'], args.n_halos)

            if not halos:
                continue

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Step {step}: {len(halos)} halos...", end=" ", flush=True)

            for i, halo_center in enumerate(halos):
                if (i+1, step) in rendered:
                    continue

                try:
                    result = render_halo(data, halo_center, i, step, out_dir, args.r_extract)
                    if result:
                        print(f"H{i+1}:OK", end=" ", flush=True)
                        rendered.add((i+1, step))
                except Exception as e:
                    print(f"H{i+1}:ERR({e})", end=" ", flush=True)

            print()
            # Mark step as attempted (even if no halos rendered)
            attempted_steps.add(step)

        time.sleep(30)

if __name__ == '__main__':
    main()
