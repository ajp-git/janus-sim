#!/usr/bin/env python3
"""
Generate visualization images for Janus runs:
1. density_z05.png - slice |z|<10 Mpc with m+ and m- separated
2. skeleton_z05.png - inter-halo filaments colored by length
3. filament_detail.png - density profile of longest filament
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments


def find_snapshot_z05(run_dir: Path, target_frac: float = 0.9) -> Path:
    """Find snapshot closest to z~0.5 (90% of simulation)."""
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {run_dir}")

    steps = [int(s.stem.replace('snap_', '')) for s in snaps]
    max_step = max(steps)
    target_step = int(target_frac * max_step)

    best_snap = min(snaps, key=lambda s: abs(int(s.stem.replace('snap_', '')) - target_step))
    return best_snap


def plot_density_slice(pos: np.ndarray, signs: np.ndarray, box_size: float,
                       output_path: Path, z_slice: float = 10.0):
    """Create density slice with m+ (blue) and m- (red) separated."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Slice mask
    mask = np.abs(pos[:, 2]) < z_slice
    pos_slice = pos[mask]
    signs_slice = signs[mask]

    # m+ particles
    mask_plus = signs_slice > 0
    # m- particles
    mask_minus = signs_slice < 0

    # 2D histograms
    bins = 256
    range_xy = [[-box_size/2, box_size/2], [-box_size/2, box_size/2]]

    h_plus, xedges, yedges = np.histogram2d(
        pos_slice[mask_plus, 0], pos_slice[mask_plus, 1],
        bins=bins, range=range_xy
    )
    h_minus, _, _ = np.histogram2d(
        pos_slice[mask_minus, 0], pos_slice[mask_minus, 1],
        bins=bins, range=range_xy
    )

    # Smooth
    h_plus = gaussian_filter(h_plus, sigma=1)
    h_minus = gaussian_filter(h_minus, sigma=1)

    # Plot m+
    im0 = axes[0].imshow(h_plus.T, origin='lower', extent=[-box_size/2, box_size/2]*2,
                         cmap='Blues', vmin=0, vmax=np.percentile(h_plus, 99))
    axes[0].set_title(f'm+ (N={mask_plus.sum():,})', fontsize=14)
    axes[0].set_xlabel('x [Mpc]')
    axes[0].set_ylabel('y [Mpc]')
    plt.colorbar(im0, ax=axes[0], label='counts')

    # Plot m-
    im1 = axes[1].imshow(h_minus.T, origin='lower', extent=[-box_size/2, box_size/2]*2,
                         cmap='Reds', vmin=0, vmax=np.percentile(h_minus, 99))
    axes[1].set_title(f'm- (N={mask_minus.sum():,})', fontsize=14)
    axes[1].set_xlabel('x [Mpc]')
    axes[1].set_ylabel('y [Mpc]')
    plt.colorbar(im1, ax=axes[1], label='counts')

    # Combined view (m+ blue, m- red)
    rgb = np.zeros((bins, bins, 3))
    if h_plus.max() > 0:
        rgb[:, :, 2] = h_plus.T / h_plus.max()  # Blue channel
    if h_minus.max() > 0:
        rgb[:, :, 0] = h_minus.T / h_minus.max()  # Red channel
    rgb = np.clip(rgb * 2, 0, 1)  # Boost contrast

    axes[2].imshow(rgb, origin='lower', extent=[-box_size/2, box_size/2]*2)
    axes[2].set_title('Combined (blue=m+, red=m-)', fontsize=14)
    axes[2].set_xlabel('x [Mpc]')
    axes[2].set_ylabel('y [Mpc]')

    plt.suptitle(f'Density slice |z| < {z_slice} Mpc at z~0.5', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_skeleton(pos: np.ndarray, signs: np.ndarray, box_size: float,
                  result: dict, output_path: Path):
    """Plot filament skeleton colored by length."""
    fig, ax = plt.subplots(figsize=(10, 10))

    n_cells = 64
    cell_size = box_size / n_cells

    # Background: particle density
    bins = 128
    h, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=bins,
        range=[[-box_size/2, box_size/2]]*2
    )
    h = gaussian_filter(h, sigma=1)
    ax.imshow(h.T, origin='lower', extent=[-box_size/2, box_size/2]*2,
              cmap='Greys', alpha=0.5, vmin=0, vmax=np.percentile(h, 95))

    # Plot halos (m+ in blue circles, m- in red)
    if 'halos_plus' in result and len(result['halos_plus']) > 0:
        halos_plus = np.array(result['halos_plus'])
        ax.scatter(halos_plus[:, 0], halos_plus[:, 1],
                   s=200, c='blue', marker='o', alpha=0.5, label=f"Halos m+ ({len(halos_plus)})")

    if 'halos_minus' in result and len(result['halos_minus']) > 0:
        halos_minus = np.array(result['halos_minus'])
        ax.scatter(halos_minus[:, 0], halos_minus[:, 1],
                   s=200, c='red', marker='o', alpha=0.5, label=f"Halos m- ({len(halos_minus)})")

    # Plot filaments colored by length
    if 'filaments' in result and len(result['filaments']) > 0:
        lengths = [f['length'] for f in result['filaments']]
        max_len = max(lengths) if lengths else 1

        cmap = plt.cm.viridis
        for fil in result['filaments']:
            cells = fil['cells']
            coords = cells * cell_size - box_size / 2
            color = cmap(fil['length'] / max_len)
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color,
                    linewidth=2, alpha=0.8)

        # Colorbar for filament length
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_len))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Filament length [Mpc]', shrink=0.8)

    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_xlabel('x [Mpc]', fontsize=12)
    ax.set_ylabel('y [Mpc]', fontsize=12)
    ax.set_title(f"Inter-halo filaments: n={result['n_filaments_real']}, "
                 f"L_max={result['length_max_real']:.1f} Mpc", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_filament_detail(pos: np.ndarray, signs: np.ndarray, box_size: float,
                         result: dict, output_path: Path):
    """Plot density profile along the longest filament."""
    if 'filaments' not in result or len(result['filaments']) == 0:
        print(f"  No filaments found, skipping detail plot")
        return

    # Find longest filament
    longest = max(result['filaments'], key=lambda f: f['length'])
    cells = longest['cells']
    n_cells = 64
    cell_size = box_size / n_cells
    coords = cells * cell_size - box_size / 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Filament path in 3D projection
    ax = axes[0]
    ax.scatter(pos[::100, 0], pos[::100, 1], s=0.1, c='gray', alpha=0.3)
    ax.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=3, label='Longest filament')
    ax.scatter(coords[0, 0], coords[0, 1], s=100, c='green', marker='o', zorder=5)
    ax.scatter(coords[-1, 0], coords[-1, 1], s=100, c='red', marker='x', zorder=5)
    ax.set_xlabel('x [Mpc]')
    ax.set_ylabel('y [Mpc]')
    ax.set_title(f'Filament path (L={longest["length"]:.1f} Mpc)')
    ax.set_aspect('equal')
    ax.legend()

    # 2. Density profile along filament
    ax = axes[1]
    densities_plus = []
    densities_minus = []
    distances = [0]

    for i, cc in enumerate(coords):
        r = 5.0  # sampling radius
        dists = np.linalg.norm(pos - cc, axis=1)
        mask_near = dists < r

        n_plus = (signs[mask_near] > 0).sum()
        n_minus = (signs[mask_near] < 0).sum()

        vol = (4/3) * np.pi * r**3
        n_total = len(pos)
        rho_mean = n_total / (box_size ** 3)

        densities_plus.append(n_plus / vol / rho_mean)
        densities_minus.append(n_minus / vol / rho_mean)

        if i > 0:
            distances.append(distances[-1] + np.linalg.norm(coords[i] - coords[i-1]))

    ax.plot(distances, densities_plus, 'b-', linewidth=2, label='m+ (overdensity)')
    ax.plot(distances, densities_minus, 'r-', linewidth=2, label='m- (overdensity)')
    ax.axhline(y=1, color='gray', linestyle='--', label='mean density')
    ax.set_xlabel('Distance along filament [Mpc]')
    ax.set_ylabel('Overdensity (ρ/ρ̄)')
    ax.set_title('Density profile along filament')
    ax.legend()
    ax.set_ylim(0, max(max(densities_plus), max(densities_minus)) * 1.2)

    # 3. m+/m- ratio along filament
    ax = axes[2]
    ratios = []
    for dp, dm in zip(densities_plus, densities_minus):
        if dm > 0.1:
            ratios.append(dp / dm)
        else:
            ratios.append(np.nan)

    ax.plot(distances, ratios, 'purple', linewidth=2)
    ax.axhline(y=1, color='gray', linestyle='--', label='equal m+/m-')
    ax.set_xlabel('Distance along filament [Mpc]')
    ax.set_ylabel('m+ / m- ratio')
    ax.set_title('Segregation along filament')
    ax.legend()

    plt.suptitle(f'Longest filament analysis: {longest["length"]:.1f} Mpc', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def process_run(run_dir: Path, box_size: float = 150.0, output_base: Path = None):
    """Process a single run and generate all visualizations."""
    print(f"\n{'='*60}")
    print(f"Processing: {run_dir.name}")
    print(f"{'='*60}")

    # Create output directory
    if output_base:
        viz_dir = output_base / run_dir.name
    else:
        viz_dir = run_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Find snapshot at z~0.5
    try:
        snap_path = find_snapshot_z05(run_dir)
        print(f"  Using snapshot: {snap_path.name}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    # Load data
    try:
        pos, signs, vel = load_snapshot(str(snap_path))
        print(f"  Loaded {len(pos):,} particles")
    except Exception as e:
        print(f"  ERROR loading snapshot: {e}")
        return

    # 1. Density slice
    plot_density_slice(pos, signs, box_size, viz_dir / 'density_z05.png')

    # 2. Detect filaments and plot skeleton
    print("  Detecting filaments...")
    result = detect_interhalos_filaments(
        pos, signs, box_size,
        n_cells=64, halo_mask_radius=10.0, min_filament_length=8.0
    )
    print(f"  Found {result['n_filaments_real']} filaments, L_max={result['length_max_real']:.1f} Mpc")

    plot_skeleton(pos, signs, box_size, result, viz_dir / 'skeleton_z05.png')

    # 3. Filament detail
    plot_filament_detail(pos, signs, box_size, result, viz_dir / 'filament_detail.png')

    print(f"  All visualizations saved to: {viz_dir}")


def main():
    runs = [
        Path('/mnt/T2/janus-sim/output/trichotomie_gpu/tour2/T2_B1_eta0.99_lam9.6_z3.0'),
        Path('/mnt/T2/janus-sim/output/trichotomie_gpu/tour3/P3.2_eta0.99800_lam10.80_z3.300'),
        Path('/mnt/T2/janus-sim/output/trichotomie_gpu/tour3/P3.3_eta0.99600_lam10.80_z3.300'),
        Path('/mnt/T2/janus-sim/output/nuit3/P2_eta088_lambda8_Z1'),
    ]

    output_base = Path('/tmp/janus_visualizations')
    output_base.mkdir(exist_ok=True)

    for run_dir in runs:
        if run_dir.exists():
            process_run(run_dir, output_base=output_base)
        else:
            print(f"\nWARNING: Run not found: {run_dir}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
