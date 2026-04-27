#!/usr/bin/env python3
"""
Fast analysis for validation 2M run.
Skips slow correlation function, focuses on key metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from scipy import ndimage
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

def periodic_extent(coords, box_size):
    """Extent with periodic correction."""
    if len(coords) == 0:
        return 0.0
    d_direct = coords.max() - coords.min()
    d_periodic = box_size - d_direct
    return min(d_direct, d_periodic)

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

def detect_filaments(density_grid, threshold=1.5, min_cells=15):
    """Detect filaments as elongated structures."""
    binary = density_grid > threshold
    labeled, n_features = ndimage.label(binary)

    filaments = []
    cell_size = BOX_SIZE / density_grid.shape[0]

    for i in range(1, n_features + 1):
        cells = np.array(np.where(labeled == i)).T
        if len(cells) < min_cells:
            continue

        cell_coords = cells * cell_size
        bbox_x = periodic_extent(cell_coords[:, 0], BOX_SIZE)
        bbox_y = periodic_extent(cell_coords[:, 1], BOX_SIZE)
        bbox_z = periodic_extent(cell_coords[:, 2], BOX_SIZE)

        length = max(bbox_x, bbox_y, bbox_z)
        length = min(length, BOX_SIZE / 2)
        width = np.median([bbox_x, bbox_y, bbox_z])
        aspect = length / max(width, 1.0)
        mean_density = np.mean(density_grid[labeled == i])

        if aspect >= 2.0 and length >= 8.0:
            filaments.append({
                'cells': cells,
                'length': length,
                'width': width,
                'aspect': aspect,
                'density': mean_density,
                'n_cells': len(cells)
            })

    return sorted(filaments, key=lambda f: f['length'], reverse=True)

# ============= FIGURE 4: Final state z=0 =============
def fig4_final_state():
    print("Figure 4: Final state at z=0...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_002500.bin")

    pos_plus = np.column_stack([snap['x'][snap['sign'] > 0],
                                 snap['y'][snap['sign'] > 0],
                                 snap['z'][snap['sign'] > 0]])
    pos_minus = np.column_stack([snap['x'][snap['sign'] < 0],
                                  snap['y'][snap['sign'] < 0],
                                  snap['z'][snap['sign'] < 0]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # xy projection (subsample for speed)
    n_plot = 30000
    idx_p = np.random.choice(len(pos_plus), min(n_plot, len(pos_plus)), replace=False)
    idx_m = np.random.choice(len(pos_minus), min(n_plot, len(pos_minus)), replace=False)

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
    plt.savefig(OUTPUT_DIR / "fig4_final_z0.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> fig4_final_z0.png (ΔCOM={delta_com:.1f} Mpc)")
    return delta_com

# ============= FIGURE 5: Segregation profile =============
def fig5_segregation_profile():
    print("Figure 5: Segregation radial profile...")
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

    plt.suptitle('Janus Validation 2M — Local Segregation (z≈0.55)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_segregation_profile.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> fig5_segregation_profile.png")

# ============= Filament analysis on existing figures =============
def analyze_filaments():
    print("\nFilament re-analysis on z≈0.5...")
    snap = load_snapshot(RUN_DIR / "snapshots/snap_001450.bin")
    pos = np.column_stack([snap['x'], snap['y'], snap['z']])

    # Compute density grid
    ncells = 100
    edges = np.linspace(0, BOX_SIZE, ncells + 1)
    H, _ = np.histogramdd(pos, bins=[edges, edges, edges])
    mean_density = len(pos) / ncells**3
    density = H / mean_density

    print(f"  Density: min={density.min():.2f}, max={density.max():.2f}, mean={density.mean():.2f}")

    # Try multiple thresholds
    for threshold in [1.0, 1.2, 1.5, 2.0]:
        filaments = detect_filaments(density, threshold=threshold, min_cells=10)
        n_fil = len([f for f in filaments if f['length'] >= 10])
        if filaments:
            l_max = filaments[0]['length']
            print(f"  threshold={threshold}: n_fil={n_fil}, L_max={l_max:.1f} Mpc")
        else:
            print(f"  threshold={threshold}: n_fil=0")

    # Use threshold=1.2 for final analysis
    filaments = detect_filaments(density, threshold=1.2, min_cells=10)
    return filaments

# ============= SUMMARY =============
def compute_summary(filaments, ts):
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    # Filament metrics
    n_filaments = len([f for f in filaments if f['length'] >= 10])
    length_max = filaments[0]['length'] if filaments else 0
    density_mean = np.mean([f['density'] for f in filaments[:10]]) if filaments else 0
    aspect_mean = np.mean([f['aspect'] for f in filaments[:10]]) if filaments else 0

    # Time series metrics
    seg_max = ts['Seg'].max()
    idx_max = np.argmax(ts['Seg'])
    z_seg_max = ts['z'][idx_max]
    delta_com_final = ts['delta_COM'][-1]

    print(f"\nFilament metrics (z≈0.5):")
    print(f"  n_filaments (L≥10 Mpc) = {n_filaments}")
    print(f"  L_max = {length_max:.1f} Mpc")
    print(f"  density_mean = {density_mean:.2f}")
    print(f"  aspect_mean = {aspect_mean:.1f}")

    print(f"\nDynamical metrics:")
    print(f"  S_max = {seg_max:.4f} at z={z_seg_max:.2f}")
    print(f"  ΔCOM_final = {delta_com_final:.1f} Mpc")

    # Validation
    print("\n" + "-"*60)
    checks = {
        "n_filaments >= 5": n_filaments >= 5,
        "L_max >= 15 Mpc": length_max >= 15,
        "density_mean >= 0.3": density_mean >= 0.3,
        "aspect >= 3": aspect_mean >= 3,
    }

    all_pass = all(checks.values())
    for name, passed in checks.items():
        print(f"  {'✓' if passed else '✗'} {name}")

    print("-"*60)
    if all_pass:
        print("✅ VALIDATION PASSED")
    else:
        print("⚠️  Some criteria need review")
    print("="*60)

    # Save summary
    with open(OUTPUT_DIR / "validation_summary.txt", 'w') as f:
        f.write("JANUS 2M VALIDATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Parameters: η=0.99, λ=9.6 Mpc, z_act=3.0\n")
        f.write(f"Box: 200 Mpc, N=2M, Steps=2500\n\n")
        f.write(f"n_filaments = {n_filaments}\n")
        f.write(f"L_max = {length_max:.1f} Mpc\n")
        f.write(f"S_max = {seg_max:.4f}\n")
        f.write(f"ΔCOM_final = {delta_com_final:.1f} Mpc\n")
        f.write(f"\nValidation: {'PASSED' if all_pass else 'REVIEW'}\n")

    return all_pass

# ============= MAIN =============
if __name__ == "__main__":
    print("JANUS 2M VALIDATION — FAST ANALYSIS")
    print("="*50)

    # Load time series
    ts = load_time_series()
    print(f"Loaded {len(ts['step'])} timesteps")
    print(f"z: {ts['z'][0]:.2f} → {ts['z'][-1]:.2f}")
    print(f"S: {ts['Seg'][0]:.4f} → {ts['Seg'][-1]:.4f}")

    # Generate remaining figures
    fig4_final_state()
    fig5_segregation_profile()

    # Filament analysis
    filaments = analyze_filaments()

    # Summary
    compute_summary(filaments, ts)

    print(f"\nFigures saved to: {OUTPUT_DIR}")
