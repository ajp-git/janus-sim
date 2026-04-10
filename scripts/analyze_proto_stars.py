#!/usr/bin/env python3
"""
Analyze proto-star candidates in Janus VSL snapshots with FOF clustering.

Snapshot format JSNP v2:
  Header (32 bytes):
    - magic: 4 bytes "JSNP"
    - version: u32 (4 bytes)
    - n: u64 (8 bytes)
    - z: f64 (8 bytes)
    - box_size: f64 (8 bytes)

  Particles (26 bytes each):
    - x, y, z: f64 (24 bytes)
    - sign: i8 (1 byte)
    - type: u8 (1 byte)  [0=m+, 255=m-]
"""

import numpy as np
from pathlib import Path
import struct
from scipy.spatial import cKDTree
import sys

# Configuration
SNAP_DIR = Path("/mnt/T2/janus-sim/output/vsl_petit_production/snapshots")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/vsl_petit_production")

# Grid sizes
GRID_SIZE_COARSE = 64   # For quick stats (matches simulation)
GRID_SIZE_FINE = 256    # For clustering (1.95 Mpc/cell)

# Thresholds
DENSITY_THRESHOLDS = [100, 300, 500, 1000, 2000]
FOF_DENSITY_THRESHOLD = 300  # Minimum density for FOF candidates

# Cosmology
BOX_SIZE = 500.0  # Mpc
N_PLUS_EXPECTED = 5_110_024
MEAN_SEP = (BOX_SIZE**3 / N_PLUS_EXPECTED)**(1/3)  # ~4.64 Mpc
LINKING_LENGTH = 0.2 * MEAN_SEP  # ~0.93 Mpc

# Mass units (arbitrary, but consistent)
M_PARTICLE = 1.0  # Each particle has unit mass


def read_snapshot(path):
    """Read JSNP v2 snapshot."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]

        print(f"  Version: {version}")
        print(f"  N particles: {n:,}")
        print(f"  Redshift z: {z:.4f}")
        print(f"  Box size: {box_size:.1f} Mpc")

        dt = np.dtype([
            ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
            ('sign', 'i1'), ('type', 'u1')
        ])
        data = np.frombuffer(f.read(n * 26), dtype=dt)

    return {
        'n': n, 'z': z, 'box': box_size,
        'x': data['x'].copy(),
        'y': data['y'].copy(),
        'zpos': data['z'].copy(),
        'sign': data['sign'].copy(),
        'type': data['type'].copy()
    }


def compute_density_field(x, y, z, box_size, grid_size):
    """Compute 3D density field on a grid."""
    half = box_size / 2
    bins = np.linspace(-half, half, grid_size + 1)
    H, edges = np.histogramdd(np.column_stack([x, y, z]), bins=[bins, bins, bins])
    return H, edges


def assign_density_to_particles(x, y, z, density_field, box_size):
    """Assign grid density to each particle."""
    grid_size = density_field.shape[0]
    half = box_size / 2
    cell_size = box_size / grid_size

    ix = np.clip(((x + half) / cell_size).astype(int), 0, grid_size - 1)
    iy = np.clip(((y + half) / cell_size).astype(int), 0, grid_size - 1)
    iz = np.clip(((z + half) / cell_size).astype(int), 0, grid_size - 1)

    return density_field[ix, iy, iz]


def fof_clustering(positions, linking_length):
    """
    Friends-of-Friends clustering using KDTree.
    Returns array of group IDs (-1 for ungrouped).
    """
    n = len(positions)
    if n == 0:
        return np.array([], dtype=int)

    # Build KDTree
    tree = cKDTree(positions)

    # Find all pairs within linking length
    pairs = tree.query_pairs(linking_length)

    # Union-Find for grouping
    parent = np.arange(n)

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i, j in pairs:
        union(i, j)

    # Flatten
    for i in range(n):
        find(i)

    # Renumber groups
    unique_parents = np.unique(parent)
    group_map = {p: idx for idx, p in enumerate(unique_parents)}
    group_ids = np.array([group_map[parent[i]] for i in range(n)])

    return group_ids


def analyze_fof_groups(x, y, z, rho, group_ids, box_size):
    """Analyze FOF groups and return statistics."""
    unique_groups = np.unique(group_ids)
    groups = []

    for gid in unique_groups:
        mask = group_ids == gid
        n_members = mask.sum()

        if n_members < 10:  # Skip tiny groups
            continue

        gx, gy, gz = x[mask], y[mask], z[mask]
        grho = rho[mask]

        # Center of mass
        com_x = gx.mean()
        com_y = gy.mean()
        com_z = gz.mean()

        # Mass
        mass = n_members * M_PARTICLE

        # Radius (half-mass radius approximation)
        r = np.sqrt((gx - com_x)**2 + (gy - com_y)**2 + (gz - com_z)**2)
        r_sorted = np.sort(r)
        r_half = r_sorted[n_members // 2] if n_members > 1 else 0

        # R_200 approximation (radius containing density = 200 × mean)
        # Simple estimate: 1.5 × half-mass radius
        r_200 = 1.5 * r_half

        # Peak density
        rho_max = grho.max()

        groups.append({
            'id': gid,
            'n_members': n_members,
            'mass': mass,
            'com': (com_x, com_y, com_z),
            'r_half': r_half,
            'r_200': r_200,
            'rho_max': rho_max
        })

    # Sort by mass (descending)
    groups.sort(key=lambda g: -g['mass'])

    return groups


def analyze_top_halos(snap, top_groups):
    """Detailed analysis of top halos: segregation and radial profiles."""

    # All particles (both m+ and m-)
    x_all = snap['x']
    y_all = snap['y']
    z_all = snap['zpos']
    types = snap['type']

    m_plus_mask = types == 0
    m_minus_mask = types == 255

    n_plus_total = m_plus_mask.sum()
    n_minus_total = m_minus_mask.sum()
    global_ratio = n_minus_total / n_plus_total

    print(f"\n{'='*80}")
    print(f"DETAILED HALO ANALYSIS — SEGREGATION CHECK")
    print(f"{'='*80}")
    print(f"  Global ratio ρ-/ρ+ = {n_minus_total:,} / {n_plus_total:,} = {global_ratio:.4f}")

    # Radial bins for density profile
    radial_bins = [0, 1, 2, 3, 5, 10]  # Mpc

    for rank, g in enumerate(top_groups):
        com = g['com']
        r_200 = g['r_200']
        search_radius = 3 * r_200

        print(f"\n  {'─'*76}")
        print(f"  HALO #{rank+1}: COM = ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) Mpc, R_200 = {r_200:.2f} Mpc")
        print(f"  {'─'*76}")

        # Distance from halo center for all particles
        dx = x_all - com[0]
        dy = y_all - com[1]
        dz = z_all - com[2]

        # Periodic boundary handling (minimum image)
        box = snap['box']
        dx = np.where(dx > box/2, dx - box, dx)
        dx = np.where(dx < -box/2, dx + box, dx)
        dy = np.where(dy > box/2, dy - box, dy)
        dy = np.where(dy < -box/2, dy + box, dy)
        dz = np.where(dz > box/2, dz - box, dz)
        dz = np.where(dz < -box/2, dz + box, dz)

        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Count within 3×R_200
        in_sphere = r < search_radius
        n_plus_sphere = (in_sphere & m_plus_mask).sum()
        n_minus_sphere = (in_sphere & m_minus_mask).sum()

        print(f"\n  Within 3×R_200 = {search_radius:.2f} Mpc:")
        print(f"    N_m+ = {n_plus_sphere:,}")
        print(f"    N_m- = {n_minus_sphere:,}")
        if n_plus_sphere > 0:
            local_ratio_3r = n_minus_sphere / n_plus_sphere
            print(f"    Local ratio ρ-/ρ+ = {local_ratio_3r:.4f}")
            deviation_3r = (local_ratio_3r - global_ratio) / global_ratio * 100
            print(f"    Deviation from global: {deviation_3r:+.1f}%")

        # Within R_200
        in_r200 = r < r_200
        n_plus_r200 = (in_r200 & m_plus_mask).sum()
        n_minus_r200 = (in_r200 & m_minus_mask).sum()

        print(f"\n  Within R_200 = {r_200:.2f} Mpc:")
        print(f"    N_m+ = {n_plus_r200:,}")
        print(f"    N_m- = {n_minus_r200:,}")
        if n_plus_r200 > 0:
            local_ratio_r200 = n_minus_r200 / n_plus_r200
            print(f"    Local ratio ρ-/ρ+ = {local_ratio_r200:.4f}")
            deviation_r200 = (local_ratio_r200 - global_ratio) / global_ratio * 100
            print(f"    Deviation from global: {deviation_r200:+.1f}%")

            if local_ratio_r200 < global_ratio * 0.8:
                print(f"    ★★★ STRONG SEGREGATION: m- depleted by {-deviation_r200:.0f}%!")
            elif local_ratio_r200 < global_ratio * 0.95:
                print(f"    ★★ MODERATE SEGREGATION: m- depleted")
            elif local_ratio_r200 > global_ratio * 1.05:
                print(f"    ★ m- ENHANCED in halo core")

        # Radial density profile
        print(f"\n  Radial density profile:")
        print(f"    {'r_min':>6} - {'r_max':>6} | {'N_m+':>8} | {'N_m-':>8} | {'ρ-/ρ+':>8} | {'vs global':>10}")
        print(f"    {'-'*60}")

        for i in range(len(radial_bins) - 1):
            r_min, r_max = radial_bins[i], radial_bins[i+1]
            shell = (r >= r_min) & (r < r_max)

            n_plus_shell = (shell & m_plus_mask).sum()
            n_minus_shell = (shell & m_minus_mask).sum()

            if n_plus_shell > 0:
                ratio_shell = n_minus_shell / n_plus_shell
                deviation = (ratio_shell - global_ratio) / global_ratio * 100
                print(f"    {r_min:>6.0f} - {r_max:>6.0f} | {n_plus_shell:>8,} | {n_minus_shell:>8,} | {ratio_shell:>8.4f} | {deviation:>+9.1f}%")
            else:
                print(f"    {r_min:>6.0f} - {r_max:>6.0f} | {n_plus_shell:>8,} | {n_minus_shell:>8,} |      N/A |       N/A")


def main():
    # Find latest snapshot
    snaps = sorted(SNAP_DIR.glob('snap_*.bin'))
    if not snaps:
        print("ERROR: No snapshots found in", SNAP_DIR)
        sys.exit(1)

    latest = snaps[-1]
    step = int(latest.stem.split('_')[1])

    print("=" * 80)
    print(f"PROTO-STAR ANALYSIS WITH FOF CLUSTERING — Step {step}")
    print("=" * 80)
    print(f"\nReading: {latest}")

    snap = read_snapshot(latest)

    # Filter m+ particles
    m_plus = snap['type'] == 0
    n_plus = m_plus.sum()
    print(f"\n  m+ particles: {n_plus:,}")

    x = snap['x'][m_plus]
    y = snap['y'][m_plus]
    z = snap['zpos'][m_plus]

    # === COARSE DENSITY (64³) for quick stats ===
    print(f"\n{'='*80}")
    print(f"DENSITY ANALYSIS (64³ grid)")
    print(f"{'='*80}")

    density_64, _ = compute_density_field(x, y, z, snap['box'], GRID_SIZE_COARSE)
    rho_64 = assign_density_to_particles(x, y, z, density_64, snap['box'])

    print(f"  ρ_max  = {rho_64.max():.1f}")
    print(f"  ρ_mean = {rho_64.mean():.2f}")
    print(f"  ρ_99.9%= {np.percentile(rho_64, 99.9):.1f}")

    peak_idx = np.argmax(rho_64)
    print(f"  Peak: ({x[peak_idx]:.1f}, {y[peak_idx]:.1f}, {z[peak_idx]:.1f}) Mpc")

    # === FINE DENSITY (256³) for clustering ===
    print(f"\n{'='*80}")
    print(f"HIGH-RESOLUTION DENSITY (256³ grid, {BOX_SIZE/GRID_SIZE_FINE:.2f} Mpc/cell)")
    print(f"{'='*80}")

    density_256, _ = compute_density_field(x, y, z, snap['box'], GRID_SIZE_FINE)
    rho_256 = assign_density_to_particles(x, y, z, density_256, snap['box'])

    # Scale factor: 256³ has (256/64)³ = 64× more cells, so ~64× fewer particles per cell
    # We need to adjust threshold accordingly
    scale_factor = (GRID_SIZE_FINE / GRID_SIZE_COARSE)**3
    fof_threshold_256 = FOF_DENSITY_THRESHOLD / scale_factor

    print(f"  ρ_max (256³) = {rho_256.max():.1f}")
    print(f"  Threshold for FOF: ρ > {fof_threshold_256:.1f} (equivalent to {FOF_DENSITY_THRESHOLD} on 64³)")

    # === FOF CLUSTERING ===
    print(f"\n{'='*80}")
    print(f"FOF CLUSTERING")
    print(f"{'='*80}")
    print(f"  Mean particle separation: {MEAN_SEP:.2f} Mpc")
    print(f"  Linking length (0.2×sep): {LINKING_LENGTH:.2f} Mpc")

    # Select high-density candidates
    fof_mask = rho_256 >= fof_threshold_256
    n_candidates = fof_mask.sum()
    print(f"  FOF candidates (ρ > {fof_threshold_256:.1f}): {n_candidates:,}")

    if n_candidates < 10:
        print("\n  Too few candidates for meaningful clustering.")
        print("  Try again at lower redshift when structures are denser.")
        return

    if n_candidates > 500000:
        print(f"  WARNING: {n_candidates:,} candidates is large, subsampling to 500k")
        # Subsample to avoid memory issues
        fof_indices = np.where(fof_mask)[0]
        fof_indices = np.random.choice(fof_indices, 500000, replace=False)
        fof_mask = np.zeros(len(x), dtype=bool)
        fof_mask[fof_indices] = True
        n_candidates = 500000

    x_fof = x[fof_mask]
    y_fof = y[fof_mask]
    z_fof = z[fof_mask]
    rho_fof = rho_64[fof_mask]  # Use 64³ density for group stats

    print(f"\n  Running FOF algorithm...")
    positions = np.column_stack([x_fof, y_fof, z_fof])
    group_ids = fof_clustering(positions, LINKING_LENGTH)

    n_groups = len(np.unique(group_ids))
    print(f"  Found {n_groups:,} FOF groups")

    # Analyze groups
    print(f"\n  Analyzing groups...")
    groups = analyze_fof_groups(x_fof, y_fof, z_fof, rho_fof, group_ids, snap['box'])

    print(f"  Groups with N > 10: {len(groups)}")

    # === TOP 10 HALOS ===
    print(f"\n{'='*80}")
    print(f"TOP 10 MOST MASSIVE HALOS")
    print(f"{'='*80}")
    print(f"{'Rank':>4} | {'N_members':>10} | {'Mass':>10} | {'R_200 (Mpc)':>12} | {'ρ_max':>8} | {'COM (x,y,z) Mpc'}")
    print("-" * 80)

    for i, g in enumerate(groups[:10]):
        com = g['com']
        print(f"{i+1:>4} | {g['n_members']:>10,} | {g['mass']:>10.0f} | {g['r_200']:>12.2f} | {g['rho_max']:>8.0f} | ({com[0]:>6.1f}, {com[1]:>6.1f}, {com[2]:>6.1f})")

    # === SUMMARY ===
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    if groups:
        total_mass_in_halos = sum(g['mass'] for g in groups)
        print(f"  Total groups (N>10): {len(groups)}")
        print(f"  Total mass in halos: {total_mass_in_halos:,.0f} particles")
        print(f"  Fraction in halos: {100*total_mass_in_halos/n_plus:.2f}%")

        if groups[0]['n_members'] >= 1000:
            print(f"\n  ★★ HALO DETECTED — Largest group has {groups[0]['n_members']:,} members")
        elif groups[0]['n_members'] >= 100:
            print(f"\n  ★ PROTO-HALO — Largest group has {groups[0]['n_members']:,} members")

    # Export top halos to CSV
    if groups:
        csv_path = OUTPUT_DIR / f"halos_step{step:06d}.csv"
        print(f"\n  Exporting {len(groups)} halos to: {csv_path}")
        with open(csv_path, 'w') as f:
            f.write("rank,n_members,mass,r_200,rho_max,com_x,com_y,com_z\n")
            for i, g in enumerate(groups):
                com = g['com']
                f.write(f"{i+1},{g['n_members']},{g['mass']:.0f},{g['r_200']:.3f},{g['rho_max']:.1f},{com[0]:.3f},{com[1]:.3f},{com[2]:.3f}\n")

    # === DETAILED HALO ANALYSIS ===
    if groups and len(groups) >= 3:
        analyze_top_halos(snap, groups[:3])

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
