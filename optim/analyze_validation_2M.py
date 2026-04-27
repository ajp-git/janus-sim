#!/usr/bin/env python3
"""
Analyse complète du run validation 2M pour publication.
Génère les 9 figures requises selon janus_runs_validation_publication.md
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration
RUN_DIR = Path("/mnt/T2/janus-sim/output/validation_2M_pub")
BOX_SIZE = 200.0  # Mpc
OUTPUT_DIR = RUN_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_snapshot(path):
    """Load binary snapshot."""
    with open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]
        raw = np.frombuffer(f.read(n * 25), dtype=np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'),
            ('sign', '<i1')
        ]))
    # Copy and wrap positions into [0, BOX_SIZE]
    data = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
        ('sign', 'i1')
    ])
    data['x'] = raw['x'] % BOX_SIZE
    data['y'] = raw['y'] % BOX_SIZE
    data['z'] = raw['z'] % BOX_SIZE
    data['vx'] = raw['vx']
    data['vy'] = raw['vy']
    data['vz'] = raw['vz']
    data['sign'] = raw['sign']
    return data

def periodic_distance(dx, box):
    """Minimum image convention."""
    return dx - box * np.round(dx / box)

def periodic_extent(coords, box_size):
    """Extent with periodic correction."""
    if len(coords) == 0:
        return 0.0
    d_direct = coords.max() - coords.min()
    d_periodic = box_size - d_direct
    return min(d_direct, d_periodic)

def compute_density_grid(pos, box, ncells=256):
    """Compute density on grid."""
    edges = np.linspace(0, box, ncells + 1)
    H, _ = np.histogramdd(pos, bins=[edges, edges, edges])
    mean_density = len(pos) / ncells**3
    return H / mean_density if mean_density > 0 else H

def find_halos_fof(pos, linking_length=0.2, min_particles=100):
    """Simple FOF halo finder."""
    tree = cKDTree(pos, boxsize=BOX_SIZE)
    n = len(pos)
    labels = -np.ones(n, dtype=int)
    current_label = 0

    for i in range(n):
        if labels[i] >= 0:
            continue
        # BFS
        queue = [i]
        labels[i] = current_label
        members = [i]

        while queue:
            idx = queue.pop(0)
            neighbors = tree.query_ball_point(pos[idx], linking_length)
            for nb in neighbors:
                if labels[nb] < 0:
                    labels[nb] = current_label
                    queue.append(nb)
                    members.append(nb)

        if len(members) >= min_particles:
            current_label += 1
        else:
            for m in members:
                labels[m] = -1

    return labels

def detect_filaments_grid(density_grid, threshold=2.0, min_cells=20):
    """Detect filaments as elongated structures in density grid."""
    binary = density_grid > threshold
    labeled, n_features = ndimage.label(binary)

    filaments = []
    for i in range(1, n_features + 1):
        cells = np.array(np.where(labeled == i)).T
        if len(cells) < min_cells:
            continue

        # Compute extent with periodic correction
        cell_size = BOX_SIZE / density_grid.shape[0]
        cell_coords = cells * cell_size

        bbox_x = periodic_extent(cell_coords[:, 0], BOX_SIZE)
        bbox_y = periodic_extent(cell_coords[:, 1], BOX_SIZE)
        bbox_z = periodic_extent(cell_coords[:, 2], BOX_SIZE)

        length = max(bbox_x, bbox_y, bbox_z)
        length = min(length, BOX_SIZE / 2)  # Cap at half box

        width = np.median([bbox_x, bbox_y, bbox_z])
        aspect = length / max(width, 1.0)

        mean_density = np.mean(density_grid[labeled == i])

        if aspect >= 2.0 and length >= 10.0:
            filaments.append({
                'cells': cells,
                'length': length,
                'width': width,
                'aspect': aspect,
                'density': mean_density,
                'n_cells': len(cells)
            })

    return sorted(filaments, key=lambda f: f['length'], reverse=True)

def load_time_series():
    """Load evolution data."""
    csv_path = RUN_DIR / "time_series.csv"
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    return {
        'step': data[:, 0],
        'z': data[:, 1],
        'a': data[:, 2],
        'H': data[:, 3],
        'KE': data[:, 4],
        'Seg': data[:, 5],
        'COM_x_plus': data[:, 6],
        'COM_y_plus': data[:, 7],
        'COM_z_plus': data[:, 8],
        'delta_COM': data[:, 9],
    }

# ============= FIGURE 1: Density maps z=0.5 =============
def fig1_density_maps():
    print("Figure 1: Density maps at z≈0.5...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_001450.bin")

    pos_plus = np.column_stack([snap['x'][snap['sign'] > 0],
                                 snap['y'][snap['sign'] > 0],
                                 snap['z'][snap['sign'] > 0]])
    pos_minus = np.column_stack([snap['x'][snap['sign'] < 0],
                                  snap['y'][snap['sign'] < 0],
                                  snap['z'][snap['sign'] < 0]])

    # Project along z (slice of 20 Mpc)
    z_min, z_max = 90, 110  # 20 Mpc slice
    mask_plus = (snap['z'][snap['sign'] > 0] > z_min) & (snap['z'][snap['sign'] > 0] < z_max)
    mask_minus = (snap['z'][snap['sign'] < 0] > z_min) & (snap['z'][snap['sign'] < 0] < z_max)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # m+ density
    h1, xe, ye = np.histogram2d(
        pos_plus[mask_plus, 0], pos_plus[mask_plus, 1],
        bins=200, range=[[0, BOX_SIZE], [0, BOX_SIZE]]
    )
    axes[0].imshow(np.log10(h1.T + 1), origin='lower', extent=[0, BOX_SIZE, 0, BOX_SIZE],
                   cmap='Blues', aspect='equal')
    axes[0].set_title(f'm+ density (N={mask_plus.sum():,})', fontsize=12)
    axes[0].set_xlabel('x [Mpc]')
    axes[0].set_ylabel('y [Mpc]')

    # m- density
    h2, _, _ = np.histogram2d(
        pos_minus[mask_minus, 0], pos_minus[mask_minus, 1],
        bins=200, range=[[0, BOX_SIZE], [0, BOX_SIZE]]
    )
    axes[1].imshow(np.log10(h2.T + 1), origin='lower', extent=[0, BOX_SIZE, 0, BOX_SIZE],
                   cmap='Oranges', aspect='equal')
    axes[1].set_title(f'm- density (N={mask_minus.sum():,})', fontsize=12)
    axes[1].set_xlabel('x [Mpc]')

    # Combined
    combined = np.zeros((200, 200, 3))
    h1_norm = np.log10(h1.T + 1)
    h2_norm = np.log10(h2.T + 1)
    h1_norm = h1_norm / h1_norm.max() if h1_norm.max() > 0 else h1_norm
    h2_norm = h2_norm / h2_norm.max() if h2_norm.max() > 0 else h2_norm
    combined[:, :, 2] = h1_norm  # Blue for m+
    combined[:, :, 0] = h2_norm  # Red for m-

    axes[2].imshow(combined, origin='lower', extent=[0, BOX_SIZE, 0, BOX_SIZE], aspect='equal')
    axes[2].set_title('Combined (blue=m+, red=m-)', fontsize=12)
    axes[2].set_xlabel('x [Mpc]')

    plt.suptitle('Janus Validation 2M — z≈0.55 (η=0.99, λ=9.6 Mpc)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_density_z05.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig1_density_z05.png")

# ============= FIGURE 2: Evolution S(z) and ΔCOM(z) =============
def fig2_evolution():
    print("Figure 2: Evolution S(z) and ΔCOM(z)...")
    ts = load_time_series()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Segregation vs z
    axes[0].plot(ts['z'], ts['Seg'], 'b-', linewidth=2)
    axes[0].axvline(3.0, color='r', linestyle='--', label='z_act=3.0')
    axes[0].set_xlabel('Redshift z', fontsize=12)
    axes[0].set_ylabel('Segregation S', fontsize=12)
    axes[0].set_xlim(5, 0)
    axes[0].legend()
    axes[0].set_title('Segregation evolution')
    axes[0].grid(True, alpha=0.3)

    # ΔCOM vs z
    axes[1].plot(ts['z'], ts['delta_COM'], 'g-', linewidth=2)
    axes[1].axvline(3.0, color='r', linestyle='--', label='z_act=3.0')
    axes[1].set_xlabel('Redshift z', fontsize=12)
    axes[1].set_ylabel('ΔCOM [Mpc]', fontsize=12)
    axes[1].set_xlim(5, 0)
    axes[1].legend()
    axes[1].set_title('COM separation evolution')
    axes[1].grid(True, alpha=0.3)

    # Find max segregation
    idx_max = np.argmax(ts['Seg'])
    z_max = ts['z'][idx_max]
    seg_max = ts['Seg'][idx_max]
    axes[0].scatter([z_max], [seg_max], s=100, c='red', marker='*',
                    label=f'Max S={seg_max:.3f} at z={z_max:.2f}')
    axes[0].legend()

    plt.suptitle('Janus Validation 2M — Dynamical Evolution (η=0.99, λ=9.6 Mpc)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig2_evolution.png")

    return ts

# ============= FIGURE 3: Filament detection =============
def fig3_filaments():
    print("Figure 3: Filament detection at z≈0.5...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_001450.bin")
    pos = np.column_stack([snap['x'], snap['y'], snap['z']])

    # Compute density grid
    density = compute_density_grid(pos, BOX_SIZE, ncells=128)

    # Show density stats
    print(f"  Density grid: min={density.min():.2f}, max={density.max():.2f}, mean={density.mean():.2f}")

    # Detect filaments with lower threshold
    filaments = detect_filaments_grid(density, threshold=1.5, min_cells=15)

    print(f"  Found {len(filaments)} filaments")
    for i, f in enumerate(filaments[:5]):
        print(f"    Filament {i+1}: L={f['length']:.1f} Mpc, aspect={f['aspect']:.1f}, ρ/ρ̄={f['density']:.2f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Density projection
    proj = density.sum(axis=2)
    axes[0].imshow(np.log10(proj.T + 1), origin='lower', extent=[0, BOX_SIZE, 0, BOX_SIZE],
                   cmap='viridis', aspect='equal')
    axes[0].set_title(f'Density projection (log scale)', fontsize=12)
    axes[0].set_xlabel('x [Mpc]')
    axes[0].set_ylabel('y [Mpc]')

    # Filament skeleton
    cell_size = BOX_SIZE / 128
    colors = plt.cm.rainbow(np.linspace(0, 1, min(10, len(filaments))))
    for i, f in enumerate(filaments[:10]):
        cells = f['cells']
        axes[1].scatter(cells[:, 0] * cell_size, cells[:, 1] * cell_size,
                        c=[colors[i]], s=1, alpha=0.5, label=f"F{i+1}: {f['length']:.0f} Mpc")

    axes[1].set_xlim(0, BOX_SIZE)
    axes[1].set_ylim(0, BOX_SIZE)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Detected filaments (N={len(filaments)})', fontsize=12)
    axes[1].set_xlabel('x [Mpc]')
    axes[1].legend(fontsize=8, loc='upper right')

    plt.suptitle('Janus Validation 2M — Filament Detection (z≈0.55)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_filaments.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig3_filaments.png")

    return filaments

# ============= FIGURE 4: Correlation functions =============
def fig4_correlation():
    print("Figure 4: Correlation functions g(r)...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_001450.bin")

    # Subsample for speed
    n_sample = 50000
    idx = np.random.choice(len(snap), min(n_sample, len(snap)), replace=False)

    pos = np.column_stack([snap['x'][idx], snap['y'][idx], snap['z'][idx]])
    signs = snap['sign'][idx]

    pos_plus = pos[signs > 0]
    pos_minus = pos[signs < 0]

    # Compute pairwise distances
    r_bins = np.linspace(1, 100, 50)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    def pair_counts(pos1, pos2, r_bins):
        tree = cKDTree(pos2, boxsize=BOX_SIZE)
        counts = np.zeros(len(r_bins) - 1)
        for p in pos1[:5000]:  # Limit for speed
            dists = tree.query_ball_point(p, r_bins[-1])
            for idx in dists:
                d = np.sqrt(np.sum(periodic_distance(p - pos2[idx], BOX_SIZE)**2))
                bin_idx = np.searchsorted(r_bins, d) - 1
                if 0 <= bin_idx < len(counts):
                    counts[bin_idx] += 1
        return counts

    # Compute g(r) approximation (normalized histogram)
    def compute_gr(pos1, pos2, r_bins):
        counts = pair_counts(pos1, pos2, r_bins)
        # Normalize by shell volumes
        dr = np.diff(r_bins)
        shell_vol = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
        n_pairs = len(pos1[:5000]) * len(pos2)
        expected = n_pairs * shell_vol / BOX_SIZE**3
        gr = counts / expected
        return gr

    print("  Computing g++...")
    g_pp = compute_gr(pos_plus, pos_plus, r_bins)
    print("  Computing g--...")
    g_mm = compute_gr(pos_minus, pos_minus, r_bins)
    print("  Computing g+-...")
    g_pm = compute_gr(pos_plus, pos_minus, r_bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_centers, g_pp, 'b-', linewidth=2, label='g++ (m+ × m+)')
    ax.plot(r_centers, g_mm, 'r-', linewidth=2, label='g-- (m- × m-)')
    ax.plot(r_centers, g_pm, 'g-', linewidth=2, label='g+- (m+ × m-)')
    ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='Random')

    ax.set_xlabel('r [Mpc]', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 3)
    ax.legend(fontsize=11)
    ax.set_title('Janus Validation 2M — Correlation Functions (z≈0.55)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig4_correlation.png")

# ============= FIGURE 5: Final state z=0 =============
def fig5_final_state():
    print("Figure 5: Final state at z=0...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_002500.bin")

    pos_plus = np.column_stack([snap['x'][snap['sign'] > 0],
                                 snap['y'][snap['sign'] > 0],
                                 snap['z'][snap['sign'] > 0]])
    pos_minus = np.column_stack([snap['x'][snap['sign'] < 0],
                                  snap['y'][snap['sign'] < 0],
                                  snap['z'][snap['sign'] < 0]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 3D scatter subsample
    n_plot = 20000
    idx_p = np.random.choice(len(pos_plus), min(n_plot, len(pos_plus)), replace=False)
    idx_m = np.random.choice(len(pos_minus), min(n_plot, len(pos_minus)), replace=False)

    # xy projection
    axes[0].scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 1], c='blue', s=0.1, alpha=0.3, label='m+')
    axes[0].scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 1], c='red', s=0.1, alpha=0.3, label='m-')
    axes[0].set_xlim(0, BOX_SIZE)
    axes[0].set_ylim(0, BOX_SIZE)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('x [Mpc]')
    axes[0].set_ylabel('y [Mpc]')
    axes[0].set_title('xy projection (z=0)')
    axes[0].legend(markerscale=20)

    # COM positions
    com_plus = pos_plus.mean(axis=0)
    com_minus = pos_minus.mean(axis=0)
    delta_com = np.sqrt(np.sum((com_plus - com_minus)**2))

    axes[0].scatter([com_plus[0]], [com_plus[1]], c='blue', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[0].scatter([com_minus[0]], [com_minus[1]], c='red', s=200, marker='*', edgecolors='white', linewidths=2)

    # Histogram of radial distribution
    r_plus = np.sqrt(np.sum((pos_plus - com_plus)**2, axis=1))
    r_minus = np.sqrt(np.sum((pos_minus - com_minus)**2, axis=1))

    axes[1].hist(r_plus, bins=50, alpha=0.5, color='blue', label=f'm+ (σ={r_plus.std():.1f} Mpc)', density=True)
    axes[1].hist(r_minus, bins=50, alpha=0.5, color='red', label=f'm- (σ={r_minus.std():.1f} Mpc)', density=True)
    axes[1].set_xlabel('Distance from COM [Mpc]')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].set_title(f'Radial distribution (ΔCOM={delta_com:.1f} Mpc)')

    plt.suptitle(f'Janus Validation 2M — Final State z=0 (η=0.99, λ=9.6 Mpc)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_final_z0.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig5_final_z0.png")
    print(f"  Final ΔCOM = {delta_com:.1f} Mpc")

# ============= FIGURE 6: Segregation profile =============
def fig6_segregation_profile():
    print("Figure 6: Segregation radial profile...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_001450.bin")
    pos = np.column_stack([snap['x'], snap['y'], snap['z']])
    signs = snap['sign']

    # Grid-based segregation
    ncells = 32
    cell_size = BOX_SIZE / ncells

    seg_grid = np.zeros((ncells, ncells, ncells))
    count_grid = np.zeros((ncells, ncells, ncells))

    for i in range(len(pos)):
        ix = int(pos[i, 0] / cell_size) % ncells
        iy = int(pos[i, 1] / cell_size) % ncells
        iz = int(pos[i, 2] / cell_size) % ncells
        count_grid[ix, iy, iz] += 1
        seg_grid[ix, iy, iz] += signs[i]

    # Segregation = |Σsign| / N in each cell
    with np.errstate(divide='ignore', invalid='ignore'):
        local_seg = np.abs(seg_grid) / np.maximum(count_grid, 1)
    local_seg[count_grid < 10] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Segregation map (projection)
    seg_proj = np.nanmean(local_seg, axis=2)
    im = axes[0].imshow(seg_proj.T, origin='lower', extent=[0, BOX_SIZE, 0, BOX_SIZE],
                        cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
    plt.colorbar(im, ax=axes[0], label='Local segregation |S|')
    axes[0].set_title('Local segregation map (z projection)')
    axes[0].set_xlabel('x [Mpc]')
    axes[0].set_ylabel('y [Mpc]')

    # Segregation vs density
    density_flat = count_grid.flatten()
    seg_flat = local_seg.flatten()
    valid = ~np.isnan(seg_flat)

    axes[1].scatter(density_flat[valid], seg_flat[valid], s=5, alpha=0.3)
    axes[1].set_xlabel('Cell density (count)')
    axes[1].set_ylabel('Local segregation |S|')
    axes[1].set_title('Segregation vs local density')

    # Bin average
    bins = np.percentile(density_flat[valid], np.linspace(0, 100, 20))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    for i in range(len(bins)-1):
        mask = (density_flat[valid] >= bins[i]) & (density_flat[valid] < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(seg_flat[valid][mask].mean())
        else:
            bin_means.append(np.nan)
    axes[1].plot(bin_centers, bin_means, 'r-', linewidth=3, label='Mean')
    axes[1].legend()

    plt.suptitle('Janus Validation 2M — Local Segregation Analysis (z≈0.55)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_segregation_profile.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig6_segregation_profile.png")

# ============= SUMMARY METRICS =============
def compute_validation_metrics(filaments, ts):
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)

    # From filaments
    n_filaments = len([f for f in filaments if f['length'] >= 10])
    length_max = filaments[0]['length'] if filaments else 0
    length_mean = np.mean([f['length'] for f in filaments[:10]]) if filaments else 0
    density_mean = np.mean([f['density'] for f in filaments[:10]]) if filaments else 0
    aspect_mean = np.mean([f['aspect'] for f in filaments[:10]]) if filaments else 0

    # From time series
    seg_max = ts['Seg'].max()
    idx_max = np.argmax(ts['Seg'])
    z_seg_max = ts['z'][idx_max]
    delta_com_final = ts['delta_COM'][-1]

    print(f"\nFilament metrics (z≈0.5):")
    print(f"  n_filaments_real   = {n_filaments} (threshold ≥5)")
    print(f"  length_max_real    = {length_max:.1f} Mpc (threshold ≥15)")
    print(f"  length_mean_real   = {length_mean:.1f} Mpc")
    print(f"  density_mean       = {density_mean:.2f} (threshold ≥0.3)")
    print(f"  aspect_ratio_mean  = {aspect_mean:.1f} (threshold ≥3)")

    print(f"\nDynamical metrics:")
    print(f"  S_max              = {seg_max:.4f} at z={z_seg_max:.2f}")
    print(f"  ΔCOM_final         = {delta_com_final:.1f} Mpc")

    # Validation check
    print("\n" + "-"*60)
    print("VALIDATION CHECK:")
    checks = [
        ("n_filaments >= 5", n_filaments >= 5, n_filaments),
        ("length_max >= 15 Mpc", length_max >= 15, length_max),
        ("density_mean >= 0.3", density_mean >= 0.3, density_mean),
        ("aspect_ratio >= 3", aspect_mean >= 3, aspect_mean),
    ]

    all_pass = True
    for name, passed, value in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {value:.2f}")
        all_pass = all_pass and passed

    print("-"*60)
    if all_pass:
        print("✅ VALIDATION PASSED — Ready for 40M publication run")
    else:
        print("⚠️  Some criteria not met — Review before 40M run")
    print("="*60)

    # Save metrics
    with open(OUTPUT_DIR / "validation_metrics.txt", 'w') as f:
        f.write("JANUS VALIDATION 2M — METRICS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Parameters: η=0.99, λ=9.6 Mpc, z_act=3.0\n")
        f.write(f"Box: 200 Mpc, N=2M, Steps=2500\n\n")
        f.write("Filament metrics (z≈0.5):\n")
        f.write(f"  n_filaments_real   = {n_filaments}\n")
        f.write(f"  length_max_real    = {length_max:.1f} Mpc\n")
        f.write(f"  length_mean_real   = {length_mean:.1f} Mpc\n")
        f.write(f"  density_mean       = {density_mean:.2f}\n")
        f.write(f"  aspect_ratio_mean  = {aspect_mean:.1f}\n\n")
        f.write("Dynamical metrics:\n")
        f.write(f"  S_max              = {seg_max:.4f} at z={z_seg_max:.2f}\n")
        f.write(f"  ΔCOM_final         = {delta_com_final:.1f} Mpc\n\n")
        f.write(f"Validation: {'PASSED' if all_pass else 'REVIEW NEEDED'}\n")

    return all_pass

# ============= MAIN =============
if __name__ == "__main__":
    print("="*60)
    print("JANUS VALIDATION 2M — COMPREHENSIVE ANALYSIS")
    print("="*60)
    print(f"Run directory: {RUN_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Generate all figures
    fig1_density_maps()
    ts = fig2_evolution()
    filaments = fig3_filaments()
    fig4_correlation()
    fig5_final_state()
    fig6_segregation_profile()

    # Compute validation metrics
    passed = compute_validation_metrics(filaments, ts)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
