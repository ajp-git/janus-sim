#!/usr/bin/env python3
"""
Generate halo segregation profile figure.
Left panel: radial density profiles for top 3 halos
Right panel: 2D ρ-/ρ+ map around halo #1
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from scipy import ndimage
from collections import defaultdict
import sys

# Output path
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/vsl_petit_production")
SNAP_DIR = OUTPUT_DIR / "snapshots"
FIG_DIR = Path("/mnt/T2/janus-sim/output/figures")

def read_snapshot(path):
    """Read JSNP v2 format snapshot."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic: {magic}")
        version = struct.unpack('<I', f.read(4))[0]
        n_particles = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]  # redshift
        box_size = struct.unpack('<d', f.read(8))[0]

        print(f"  Reading {n_particles:,} particles, box={box_size:.1f} Mpc, z={z:.4f}")

        # Read particles: x,y,z (f64), sign (i8), type (u8) = 26 bytes each
        positions = np.zeros((n_particles, 3), dtype=np.float64)
        signs = np.zeros(n_particles, dtype=np.int8)

        for i in range(n_particles):
            data = f.read(26)
            px, py, pz = struct.unpack('<ddd', data[:24])  # particle coords (not z redshift!)
            sign = struct.unpack('<b', data[24:25])[0]
            positions[i] = [px, py, pz]
            signs[i] = sign

            if i % 2000000 == 0 and i > 0:
                print(f"    {i:,} / {n_particles:,} particles read...")

    return {
        'positions': positions,
        'signs': signs,
        'box_size': box_size,
        'n_particles': n_particles,
        'z': z
    }

def fof_clustering(positions, box_size, linking_length):
    """FOF clustering with Union-Find."""
    n = len(positions)
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int32)

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi == pj:
            return
        if rank[pi] < rank[pj]:
            pi, pj = pj, pi
        parent[pj] = pi
        if rank[pi] == rank[pj]:
            rank[pi] += 1

    # Grid-based neighbor search
    cell_size = linking_length
    n_cells = int(np.ceil(box_size / cell_size))

    print(f"  Building grid ({n_cells}^3 cells)...")
    cells = defaultdict(list)
    for i, pos in enumerate(positions):
        cx = int(pos[0] / cell_size) % n_cells
        cy = int(pos[1] / cell_size) % n_cells
        cz = int(pos[2] / cell_size) % n_cells
        cells[(cx, cy, cz)].append(i)

    print(f"  Finding neighbors (ll={linking_length:.2f} Mpc)...")
    ll_sq = linking_length ** 2
    n_links = 0

    for (cx, cy, cz), particles in cells.items():
        # Check 27 neighboring cells
        for dcx in [-1, 0, 1]:
            for dcy in [-1, 0, 1]:
                for dcz in [-1, 0, 1]:
                    ncx = (cx + dcx) % n_cells
                    ncy = (cy + dcy) % n_cells
                    ncz = (cz + dcz) % n_cells

                    for i in particles:
                        for j in cells.get((ncx, ncy, ncz), []):
                            if i >= j:
                                continue

                            # Distance with periodic boundary
                            dx = positions[j] - positions[i]
                            dx = dx - box_size * np.round(dx / box_size)
                            d_sq = np.sum(dx**2)

                            if d_sq < ll_sq:
                                union(i, j)
                                n_links += 1

    print(f"  Found {n_links:,} links")

    # Build groups
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Sort by size
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    return sorted_groups

def compute_halo_properties(positions, signs, box_size, group_indices, n_total_particles):
    """Compute halo center, R_200, and particle counts."""
    pos = positions[group_indices]
    sgn = signs[group_indices]

    # Center of mass (minimum image convention)
    ref = pos[0].copy()
    delta = pos - ref
    delta = delta - box_size * np.round(delta / box_size)
    com = ref + np.mean(delta, axis=0)
    com = com % box_size

    # Distances from COM
    dr = pos - com
    dr = dr - box_size * np.round(dr / box_size)
    r = np.sqrt(np.sum(dr**2, axis=1))

    # R_200 estimate: radius where enclosed density = 200 × mean cosmic density
    r_sorted = np.sort(r)
    n_halo = len(group_indices)

    # Mean cosmic density (particles per Mpc³)
    rho_mean = n_total_particles / (box_size ** 3)

    # Find R_200: iterate outward until ρ_enc < 200 × ρ_mean
    r_200 = r_sorted[-1]  # default to max radius
    for i in range(10, n_halo):
        r_enc = r_sorted[i]
        if r_enc < 0.01:  # skip tiny radii
            continue
        vol = 4/3 * np.pi * r_enc**3
        rho_enc = (i + 1) / vol  # +1 because index is 0-based
        if rho_enc < 200 * rho_mean:
            r_200 = r_sorted[i-1] if i > 0 else r_sorted[i]
            break

    # If R_200 too small, estimate from halo mass assuming NFW-like profile
    if r_200 < 0.5:
        # Approximate: R_200 ≈ (3 * N_halo / (4π × 200 × ρ_mean))^(1/3)
        r_200 = (3 * n_halo / (4 * np.pi * 200 * rho_mean)) ** (1/3)

    # Particle counts
    n_plus = np.sum(sgn > 0)
    n_minus = np.sum(sgn < 0)

    return {
        'center': com,
        'r_200': r_200,
        'n_total': n_halo,
        'n_plus': n_plus,
        'n_minus': n_minus,
        'indices': group_indices,
        'positions': pos,
        'signs': sgn,
        'radii': r
    }

def compute_radial_profiles(halo, positions, signs, box_size, n_bins=20, r_max_factor=5.0):
    """Compute radial density profiles for m+ and m-."""
    com = halo['center']
    r_200 = halo['r_200']
    r_max = r_max_factor * r_200

    # All particles within r_max
    dr = positions - com
    dr = dr - box_size * np.round(dr / box_size)
    r = np.sqrt(np.sum(dr**2, axis=1))

    mask = r < r_max
    r_sel = r[mask]
    signs_sel = signs[mask]

    # Bin edges in units of R_200
    r_bins = np.linspace(0, r_max_factor, n_bins + 1) * r_200

    # Count particles in each bin
    rho_plus = np.zeros(n_bins)
    rho_minus = np.zeros(n_bins)

    for i in range(n_bins):
        r_in, r_out = r_bins[i], r_bins[i+1]
        in_shell = (r_sel >= r_in) & (r_sel < r_out)

        shell_vol = 4/3 * np.pi * (r_out**3 - r_in**3)

        n_plus = np.sum((signs_sel > 0) & in_shell)
        n_minus = np.sum((signs_sel < 0) & in_shell)

        rho_plus[i] = n_plus / shell_vol if shell_vol > 0 else 0
        rho_minus[i] = n_minus / shell_vol if shell_vol > 0 else 0

    r_centers = (r_bins[:-1] + r_bins[1:]) / 2 / r_200  # In units of R_200

    return r_centers, rho_plus, rho_minus

def compute_2d_slice(positions, signs, box_size, center, half_width, n_grid=256, slice_thickness=5.0):
    """Compute 2D density slice around center."""
    # Select particles within slice
    dz = positions[:, 2] - center[2]
    dz = dz - box_size * np.round(dz / box_size)
    in_slice = np.abs(dz) < slice_thickness / 2

    pos_slice = positions[in_slice]
    signs_slice = signs[in_slice]

    # Grid
    grid_plus = np.zeros((n_grid, n_grid))
    grid_minus = np.zeros((n_grid, n_grid))

    cell_size = 2 * half_width / n_grid

    for i in range(len(pos_slice)):
        dx = pos_slice[i, 0] - center[0]
        dy = pos_slice[i, 1] - center[1]
        dx = dx - box_size * np.round(dx / box_size)
        dy = dy - box_size * np.round(dy / box_size)

        if abs(dx) < half_width and abs(dy) < half_width:
            ix = int((dx + half_width) / cell_size)
            iy = int((dy + half_width) / cell_size)
            ix = min(max(ix, 0), n_grid - 1)
            iy = min(max(iy, 0), n_grid - 1)

            if signs_slice[i] > 0:
                grid_plus[iy, ix] += 1
            else:
                grid_minus[iy, ix] += 1

    # Compute ratio (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(grid_plus > 0, grid_minus / grid_plus, np.nan)

    return grid_plus, grid_minus, ratio

def main():
    # Find latest snapshot
    snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
    if not snaps:
        print("No snapshots found!")
        return

    snap_path = snaps[-1]
    step = int(snap_path.stem.split('_')[1])
    print(f"Using snapshot: {snap_path.name} (step {step})")

    # Read snapshot
    print("\nReading snapshot...")
    snap = read_snapshot(snap_path)
    positions = snap['positions']
    signs = snap['signs']
    box_size = snap['box_size']
    z_value = snap['z']

    # Global ratio
    n_plus_global = np.sum(signs > 0)
    n_minus_global = np.sum(signs < 0)
    global_ratio = n_minus_global / n_plus_global
    print(f"\nGlobal ratio m-/m+ = {global_ratio:.4f}")

    # Mean separation
    mean_sep = box_size / (len(positions) ** (1/3))
    linking_length = 0.2 * mean_sep
    print(f"Mean separation: {mean_sep:.3f} Mpc, linking length: {linking_length:.3f} Mpc")

    # FOF clustering on m+ particles only (halos are m+ dominated)
    print("\nFOF clustering on m+ particles...")
    mask_plus = signs > 0
    pos_plus = positions[mask_plus]
    idx_plus = np.where(mask_plus)[0]

    groups = fof_clustering(pos_plus, box_size, linking_length)

    # Map back to full indices
    full_groups = [[idx_plus[i] for i in g] for g in groups[:10]]

    print(f"\nTop 3 halos:")
    halos = []
    for i, g in enumerate(full_groups[:3]):
        halo = compute_halo_properties(positions, signs, box_size, g, len(positions))
        print(f"  Halo #{i+1}: N={halo['n_total']:,}, R_200={halo['r_200']:.2f} Mpc, "
              f"N+={halo['n_plus']}, N-={halo['n_minus']}")
        halos.append(halo)

    # Create figure with space for caption
    print("\nGenerating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    plt.subplots_adjust(bottom=0.1)  # Make room for caption

    # Global mean density for normalization
    total_vol = box_size ** 3
    rho_global_plus = n_plus_global / total_vol
    rho_global_minus = n_minus_global / total_vol

    # Floor value for empty bins (detection limit)
    FLOOR_VALUE = 1e-2

    # Left panel: radial profiles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    halo_names = ['Halo #1', 'Halo #2', 'Halo #3']

    # Store exclusion zones for annotation
    exclusion_bins = []

    for i, halo in enumerate(halos):
        r_centers, rho_plus, rho_minus = compute_radial_profiles(
            halo, positions, signs, box_size, n_bins=25, r_max_factor=5.0
        )

        # Normalize by global density
        rho_plus_norm = rho_plus / rho_global_plus
        rho_minus_norm = rho_minus / rho_global_minus

        # Plot m+ (solid line) with explicit label per halo
        mask_plus_nz = rho_plus_norm > 0
        ax1.plot(r_centers[mask_plus_nz], rho_plus_norm[mask_plus_nz],
                 '-', color=colors[i], linewidth=2.5, label=f'{halo_names[i]} m$^+$')

        # Plot m- (dashed line) with floor for empty bins - no legend here
        rho_minus_plot = rho_minus_norm.copy()
        empty_bins = rho_minus_norm == 0
        rho_minus_plot[empty_bins] = FLOOR_VALUE

        ax1.plot(r_centers, rho_minus_plot,
                 '--', color=colors[i], linewidth=1.5, alpha=0.6)

        # Mark empty bins with downward triangles at floor level
        ax1.scatter(r_centers[empty_bins], rho_minus_plot[empty_bins],
                    marker='v', s=50, color=colors[i], zorder=5, alpha=0.8)

        # Track exclusion zone for halo #1
        if i == 0:
            exclusion_bins = r_centers[empty_bins]

    # Add single legend entry for excluded m- (all halos)
    ax1.scatter([], [], marker='v', s=50, color='gray', label=r'm$^-$ exclu (▼)')

    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, label='Moyenne globale')

    # Vertical lines with labels
    ax1.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=3.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(1.05, 2e3, r'$R_{200}$', fontsize=10, ha='left', va='center',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=2))
    ax1.text(3.05, 2e3, r'$3R_{200}$', fontsize=10, ha='left', va='center',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=2))

    # Add annotation for total exclusion
    if len(exclusion_bins) > 0:
        ax1.annotate('EXCLUSION TOTALE\n(aucune particule m$^-$)',
                     xy=(0.5, FLOOR_VALUE * 2), xytext=(2.5, 0.08),
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=2),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))

    # Detection limit annotation
    ax1.axhline(y=FLOOR_VALUE, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax1.text(4.9, FLOOR_VALUE * 0.6, '< limite détection', fontsize=9, color='red',
             ha='right', va='top', style='italic')

    ax1.set_xlabel(r'$r / R_{200}$', fontsize=13)
    ax1.set_ylabel(r'$\rho / \rho_{\mathrm{global}}$ (échelle log)', fontsize=13)
    ax1.set_yscale('log')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(5e-3, 1e4)
    ax1.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.95)
    ax1.set_title('Profils de densité radiaux', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')

    # Right panel: 2D slice
    halo1 = halos[0]
    r_200 = halo1['r_200']

    # Adjust view to show R_200 and 3×R_200 clearly
    half_width = max(5 * r_200, 15.0)  # At least 5×R_200 or 15 Mpc

    grid_plus, grid_minus, ratio = compute_2d_slice(
        positions, signs, box_size, halo1['center'],
        half_width, n_grid=256, slice_thickness=max(2.0, r_200)
    )

    # Use log ratio for visualization
    # Cells with no m-: set to very negative (blue = m+ dominant)
    # Cells with no m+: set to very positive (red = m- dominant)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log10(ratio)

    # Handle special cases explicitly
    no_minus = (grid_minus == 0) & (grid_plus > 0)  # m+ only → exclusion
    no_plus = (grid_plus == 0) & (grid_minus > 0)   # m- only
    empty = (grid_plus == 0) & (grid_minus == 0)    # no particles

    log_ratio[no_minus] = -3  # Strong blue (exclusion)
    log_ratio[no_plus] = 3    # Strong red
    log_ratio[empty] = 0      # Neutral (gray)

    # Clip remaining values
    log_ratio = np.clip(log_ratio, -3, 3)

    extent = [-half_width, half_width, -half_width, half_width]
    im = ax2.imshow(log_ratio, extent=extent, origin='lower',
                    cmap='RdBu_r', vmin=-3, vmax=3, aspect='equal')

    # Circles at R_200 and 3×R_200 (visible)
    circle1 = plt.Circle((0, 0), r_200, fill=False, color='lime', linewidth=3, linestyle='-')
    circle2 = plt.Circle((0, 0), 3*r_200, fill=False, color='lime', linewidth=2, linestyle='--')
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)

    # Label circles with R_200 values - position based on circle size
    # Inner circle label - below the circle
    ax2.text(0, -r_200 - 0.5, f'R$_{{200}}$ = {r_200:.2f} Mpc',
             fontsize=9, color='lime', fontweight='bold', ha='center', va='top',
             bbox=dict(facecolor='black', edgecolor='lime', alpha=0.9, pad=2))
    # Outer circle label - to the right of the circle
    ax2.text(3*r_200 + 0.5, 0, f'3×R$_{{200}}$ = {3*r_200:.1f} Mpc',
             fontsize=9, color='lime', fontweight='bold', ha='left', va='center',
             bbox=dict(facecolor='black', edgecolor='lime', alpha=0.9, pad=2))

    # Center marker
    ax2.scatter([0], [0], marker='+', s=200, color='white', linewidths=3, zorder=10)

    # Label "Halo #1" with arrow to center
    ax2.annotate('Halo #1', xy=(0, 0), xytext=(half_width*0.5, half_width*0.7),
                 fontsize=12, fontweight='bold', color='white',
                 arrowprops=dict(arrowstyle='->', color='white', lw=2),
                 bbox=dict(facecolor='black', edgecolor='white', alpha=0.9, pad=3))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label(r'$\log_{10}(\rho_- / \rho_+)$', fontsize=12)
    cbar.ax.axhline(y=0, color='black', linewidth=1)  # Mark ratio=1

    # Get redshift from snapshot (use stored value)
    z_snap = z_value

    ax2.set_xlabel('X [Mpc]', fontsize=13)
    ax2.set_ylabel('Y [Mpc]', fontsize=13)
    ax2.set_title(f'Carte 2D de ségrégation (tranche z, épaisseur={max(2.0, r_200):.1f} Mpc)',
                  fontsize=12)

    # Add text annotation with exclusion info
    n_cells_exclusion = np.sum(no_minus)
    n_cells_total = np.sum(grid_plus > 0)
    exclusion_pct = 100 * n_cells_exclusion / n_cells_total if n_cells_total > 0 else 0
    ax2.text(0.02, 0.98, f'Bleu = m$^+$ dominant\nRouge = m$^-$ dominant\n'
                         f'Cellules exclues: {exclusion_pct:.0f}%',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.9))

    plt.tight_layout()

    # Add caption at bottom of figure
    fig.text(0.5, 0.01, f'z = {z_snap:.3f}  |  μ = 19  |  VSL Petit (2014)',
             ha='center', fontsize=11, style='italic',
             bbox=dict(facecolor='lightyellow', edgecolor='gray', alpha=0.9, pad=5))

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / "halo_segregation_profile.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Also show statistics
    print("\n" + "="*60)
    print("SEGREGATION SUMMARY")
    print("="*60)
    for i, halo in enumerate(halos):
        print(f"\nHalo #{i+1}:")
        print(f"  Center: ({halo['center'][0]:.1f}, {halo['center'][1]:.1f}, {halo['center'][2]:.1f}) Mpc")
        print(f"  R_200 = {halo['r_200']:.2f} Mpc")
        print(f"  Total particles: {halo['n_total']:,}")
        print(f"  m+ particles: {halo['n_plus']:,}")
        print(f"  m- particles: {halo['n_minus']:,}")
        if halo['n_plus'] > 0:
            ratio_local = halo['n_minus'] / halo['n_plus']
            print(f"  Local m-/m+ ratio: {ratio_local:.4f} (global: {global_ratio:.4f})")
            depletion = (1 - ratio_local / global_ratio) * 100
            print(f"  m- depletion: {depletion:.1f}%")

if __name__ == "__main__":
    main()
