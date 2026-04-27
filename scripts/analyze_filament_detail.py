#!/usr/bin/env python3
"""
Detailed filament analysis for P2_eta088_lambda8_Z1.
- Position and orientation
- Density profile along filament
- Composition (m+ vs m-)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, fof_halos, detect_interhalos_filaments


def analyze_filament_detail(run_dir, box_size=150.0):
    """Detailed analysis of filament structure."""

    # Load final snapshot
    snap_path = run_dir / 'snapshots' / 'snap_002000.bin'
    if not snap_path.exists():
        snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
        snap_path = snaps[-1] if snaps else None

    if not snap_path:
        print("No snapshot found")
        return

    print(f"Loading {snap_path}...")
    pos, signs, vel = load_snapshot(str(snap_path))
    n = len(pos)
    print(f"  N = {n} particles")

    # Find halos
    halos = fof_halos(pos, signs, box_size, b=0.2, min_particles=100)
    halos_plus = [h for h in halos if h['sign'] == 1]
    halos_minus = [h for h in halos if h['sign'] == -1]

    print(f"  Halos m+: {len(halos_plus)}, m-: {len(halos_minus)}")

    # Compute ΔCOM
    mask_plus = signs > 0
    mask_minus = signs < 0
    com_plus = pos[mask_plus].mean(axis=0)
    com_minus = pos[mask_minus].mean(axis=0)
    delta_com = com_plus - com_minus
    delta_com_mag = np.linalg.norm(delta_com)
    delta_com_unit = delta_com / delta_com_mag

    print(f"\n1. ΔCOM Analysis:")
    print(f"   COM m+: ({com_plus[0]:.1f}, {com_plus[1]:.1f}, {com_plus[2]:.1f}) Mpc")
    print(f"   COM m-: ({com_minus[0]:.1f}, {com_minus[1]:.1f}, {com_minus[2]:.1f}) Mpc")
    print(f"   ΔCOM: {delta_com_mag:.1f} Mpc")
    print(f"   Direction: ({delta_com_unit[0]:.3f}, {delta_com_unit[1]:.3f}, {delta_com_unit[2]:.3f})")

    # Detect filaments with detailed info
    result = detect_interhalos_filaments(pos, signs, box_size, n_cells=64,
                                          halo_mask_radius=10.0, min_filament_length=10.0)

    if result['n_filaments_real'] == 0:
        print("\n  No filaments detected!")
        return

    print(f"\n2. Filament Position:")
    filaments = result.get('filaments', [])
    cell_size = box_size / 64

    for i, fil in enumerate(filaments):
        center = fil['center']
        length = fil['length']
        n_cells = fil['n_cells']

        print(f"\n   Filament {i+1}:")
        print(f"   Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) Mpc")
        print(f"   Length: {length:.1f} Mpc")
        print(f"   N cells: {n_cells}")

        # Compute orientation via PCA
        cells = fil['cells']
        cell_coords = cells * cell_size - box_size / 2
        centered = cell_coords - cell_coords.mean(axis=0)
        if len(centered) > 1:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            principal_axis = eigenvectors[:, -1]  # Largest eigenvalue

            # Angle with ΔCOM direction
            cos_angle = abs(np.dot(principal_axis, delta_com_unit))
            angle_deg = np.degrees(np.arccos(cos_angle))

            print(f"   Principal axis: ({principal_axis[0]:.3f}, {principal_axis[1]:.3f}, {principal_axis[2]:.3f})")
            print(f"   Angle with ΔCOM: {angle_deg:.1f}°")
            if angle_deg < 30:
                print(f"   → ALIGNED with segregation axis!")
            elif angle_deg > 60:
                print(f"   → PERPENDICULAR to segregation axis")
            else:
                print(f"   → Intermediate orientation")

    # 3. Density along filament
    print(f"\n3. Density Along Filament:")

    # Use the filament cells to compute local density
    if filaments:
        fil = filaments[0]  # First filament
        cells = fil['cells']
        cell_coords = cells * cell_size - box_size / 2

        # Project onto principal axis
        centered = cell_coords - cell_coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]

        projections = np.dot(cell_coords - cell_coords.mean(axis=0), principal_axis)
        sort_idx = np.argsort(projections)

        # Sample density along filament
        n_samples = min(10, len(cells))
        sample_indices = np.linspace(0, len(cells)-1, n_samples, dtype=int)

        # Mean density in simulation
        rho_mean = n / (box_size ** 3)

        print(f"   Mean simulation density: {rho_mean:.2f} particles/Mpc³")
        print(f"\n   Position along axis | Local density | ρ/ρ̄")
        print(f"   " + "-" * 50)

        densities = []
        for idx in sample_indices:
            cell_idx = sort_idx[idx]
            cell_center = cell_coords[cell_idx]

            # Count particles within 5 Mpc of this cell
            r = 5.0
            dists = np.linalg.norm(pos - cell_center, axis=1)
            n_local = (dists < r).sum()
            vol_local = (4/3) * np.pi * r**3
            rho_local = n_local / vol_local
            rho_ratio = rho_local / rho_mean
            densities.append(rho_ratio)

            proj = projections[cell_idx]
            print(f"   {proj:+6.1f} Mpc          | {rho_local:.2f}         | {rho_ratio:.2f}")

        mean_rho_ratio = np.mean(densities)
        print(f"\n   Mean filament density: {mean_rho_ratio:.2f} ρ̄")
        if 1.2 < mean_rho_ratio < 2.5:
            print(f"   → Intermediate density (typical filament)")
        elif mean_rho_ratio >= 2.5:
            print(f"   → High density (collapsed structure)")
        else:
            print(f"   → Low density (void-like)")

    # 4. Composition analysis
    print(f"\n4. Filament Composition:")

    if filaments:
        fil = filaments[0]
        cells = fil['cells']
        cell_coords = cells * cell_size - box_size / 2

        # Count m+ and m- particles within filament region
        n_plus_fil = 0
        n_minus_fil = 0

        for cell_center in cell_coords:
            r = cell_size * 1.5  # Slightly larger than cell
            dists = np.linalg.norm(pos - cell_center, axis=1)
            mask_near = dists < r

            n_plus_fil += (signs[mask_near] > 0).sum()
            n_minus_fil += (signs[mask_near] < 0).sum()

        total_fil = n_plus_fil + n_minus_fil
        if total_fil > 0:
            frac_plus = n_plus_fil / total_fil
            frac_minus = n_minus_fil / total_fil

            print(f"   Particles in filament region:")
            print(f"   m+: {n_plus_fil} ({100*frac_plus:.1f}%)")
            print(f"   m-: {n_minus_fil} ({100*frac_minus:.1f}%)")

            if frac_plus > 0.8:
                print(f"   → PURE m+ filament")
            elif frac_minus > 0.8:
                print(f"   → PURE m- filament")
            elif 0.4 < frac_plus < 0.6:
                print(f"   → MIXED composition")
            elif frac_plus > frac_minus:
                print(f"   → m+ DOMINATED")
            else:
                print(f"   → m- DOMINATED")

    # 5. Create visualization
    print(f"\n5. Creating visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: XY projection with filament highlighted
    ax = axes[0]

    # Plot all particles (subsampled)
    idx_plus = np.where(mask_plus)[0]
    idx_minus = np.where(mask_minus)[0]
    sub_plus = np.random.choice(idx_plus, min(25000, len(idx_plus)), replace=False)
    sub_minus = np.random.choice(idx_minus, min(25000, len(idx_minus)), replace=False)
    ax.scatter(pos[sub_plus, 0], pos[sub_plus, 1],
               c='red', s=0.1, alpha=0.3, label='m+')
    ax.scatter(pos[sub_minus, 0], pos[sub_minus, 1],
               c='blue', s=0.1, alpha=0.3, label='m-')

    # Plot halos
    for h in halos_plus:
        ax.scatter(h['com'][0], h['com'][1], marker='*', s=200, c='darkred',
                   edgecolors='white', linewidths=1, zorder=10)

    # Plot filament cells
    if filaments:
        fil = filaments[0]
        cells = fil['cells']
        cell_coords = cells * cell_size - box_size / 2
        ax.scatter(cell_coords[:, 0], cell_coords[:, 1], c='lime', s=50,
                   marker='s', alpha=0.7, edgecolors='black', linewidths=0.5,
                   label='Filament', zorder=5)

    # Plot ΔCOM arrow
    ax.arrow(com_minus[0], com_minus[1], delta_com[0]*0.8, delta_com[1]*0.8,
             head_width=3, head_length=2, fc='black', ec='black', linewidth=2,
             zorder=15)
    ax.text(com_minus[0] + delta_com[0]*0.4, com_minus[1] + delta_com[1]*0.4 + 5,
            f'ΔCOM={delta_com_mag:.1f} Mpc', fontsize=10, ha='center')

    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('XY Projection')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # Panel 2: Density profile along filament
    ax = axes[1]
    if filaments and len(densities) > 0:
        positions = projections[sort_idx[sample_indices]]
        ax.plot(positions, densities, 'ko-', markersize=8, linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', label='ρ̄')
        ax.axhline(y=1.5, color='green', linestyle=':', alpha=0.7, label='1.5 ρ̄')
        ax.axhline(y=2.0, color='orange', linestyle=':', alpha=0.7, label='2.0 ρ̄')
        ax.set_xlabel('Position along filament (Mpc)')
        ax.set_ylabel('ρ / ρ̄')
        ax.set_title('Density Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: Composition pie chart
    ax = axes[2]
    if filaments and total_fil > 0:
        sizes = [frac_plus, frac_minus]
        labels = [f'm+ ({100*frac_plus:.1f}%)', f'm- ({100*frac_minus:.1f}%)']
        colors = ['red', 'blue']
        ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax.set_title('Filament Composition')

    plt.suptitle(f'P2_eta088_lambda8_Z1: Filament Analysis\nLength={filaments[0]["length"]:.1f} Mpc',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = run_dir / 'figures' / 'filament_detail.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    run_dir = Path('/mnt/T2/janus-sim/output/nuit3/P2_eta088_lambda8_Z1')
    analyze_filament_detail(run_dir, box_size=150.0)
