#!/usr/bin/env python3
"""
Filament detection and metrics for Janus simulations.
Version 2: Inter-halo filaments only (excludes halo interiors).
"""

import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree
from collections import defaultdict
import struct


def load_snapshot(path):
    """Load binary snapshot - auto-detect format v1 or v2."""
    import os
    file_size = os.path.getsize(path)

    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]

        expected_v1 = 4 + n * 13
        expected_v2 = 4 + n * 25

        if file_size == expected_v2:
            data = np.frombuffer(f.read(), dtype=np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'),
                ('sign', 'i1')
            ]))
            pos = np.column_stack([data['x'], data['y'], data['z']])
            vel = np.column_stack([data['vx'], data['vy'], data['vz']])
            signs = data['sign']
            return pos, signs, vel
        else:
            data = np.frombuffer(f.read(), dtype=np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('sign', 'i1')
            ]))
            pos = np.column_stack([data['x'], data['y'], data['z']])
            signs = data['sign']
            return pos, signs, None


def fof_halos(pos, signs, box_size, b=0.2, min_particles=50):
    """Friends-of-Friends halo finder."""
    linking_length = b * (box_size / len(pos)**(1/3))

    halos = []
    for sign_val in [1, -1]:
        mask = signs == sign_val
        if mask.sum() < min_particles:
            continue
        pos_sub = pos[mask]

        # Shift to [0, L] for periodic KD-tree
        pos_sub_shifted = pos_sub + box_size / 2
        # Ensure all coordinates are in [0, box_size)
        pos_sub_shifted = np.mod(pos_sub_shifted, box_size)
        tree = cKDTree(pos_sub_shifted, boxsize=box_size)

        # Union-Find
        n = len(pos_sub)
        parent = np.arange(n)

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        pairs = tree.query_pairs(linking_length)
        for i, j in pairs:
            union(i, j)

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        for indices in groups.values():
            if len(indices) >= min_particles:
                halo_pos = pos_sub[indices]
                com = halo_pos.mean(axis=0)
                halos.append({
                    'sign': sign_val,
                    'n_particles': len(indices),
                    'com': com,
                    'positions': halo_pos
                })

    return halos


def periodic_extent(coords_1d, box_size):
    """
    Compute the extent of coordinates with periodic boundary correction.

    For a set of coordinates on a periodic axis, the extent is the minimum
    of the direct span vs the wrap-around span.

    Example: coords at x=5 and x=145 in a 150 Mpc box
      - direct span: 145-5 = 140 Mpc
      - periodic span: 150-140 = 10 Mpc (wraps around)
      - true extent: min(140, 10) = 10 Mpc
    """
    if len(coords_1d) == 0:
        return 0.0
    d_direct = coords_1d.max() - coords_1d.min()
    d_periodic = box_size - d_direct
    return min(d_direct, d_periodic)


def detect_interhalos_filaments(pos, signs, box_size, n_cells=None,
                                 halo_mask_radius=10.0, min_filament_length=10.0):
    """
    Detect ONLY filaments between halos.
    Excludes structures internal to halos.

    Algorithm:
    1. Create density grid for m+ particles
    2. Binarize: rho > 2 * rho_mean
    3. Mask cells within halo_mask_radius of any halo FOF center
    4. Label connected components
    5. Keep only components with length > min_filament_length
    6. Measure length, connectivity, distance to nearest halo

    Returns:
        dict with n_filaments_real, length_mean_real, length_max_real,
        filament_cells (for visualization)

    Note: Uses periodic boundary correction for bbox and length calculations.

    n_cells scaling: If n_cells is None, auto-scale to maintain constant
    particle density per cell (~120 particles/cell at 500k reference):
        n_cells = int(64 * (N / 500000) ** (1/3))
    """
    # Auto-scale n_cells if not specified
    if n_cells is None:
        N = len(pos)
        n_cells = int(64 * (N / 500000) ** (1/3))
        n_cells = max(32, min(n_cells, 256))  # Clamp to reasonable range
    # Find halos
    halos = fof_halos(pos, signs, box_size, b=0.2, min_particles=100)
    halos_plus = [h for h in halos if h['sign'] == 1]

    if len(halos_plus) < 2:
        return {
            'n_filaments_real': 0,
            'length_mean_real': 0.0,
            'length_max_real': 0.0,
            'filament_cells': None,
            'halo_positions': [h['com'] for h in halos_plus],
            'reason': 'less than 2 halos'
        }

    # Create density grid for m+
    cell_size = box_size / n_cells
    mask_plus = signs > 0
    pos_plus = pos[mask_plus]

    # Shift to [0, L] for binning
    pos_shifted = pos_plus + box_size / 2

    H, edges = np.histogramdd(pos_shifted, bins=n_cells,
                               range=[[0, box_size]]*3)

    # Binarize
    rho_mean = mask_plus.sum() / (n_cells**3)
    threshold = 2.0 * rho_mean
    binary_grid = (H > threshold).astype(np.uint8)

    # Create halo mask (cells within halo_mask_radius of any halo)
    halo_mask = np.zeros_like(binary_grid, dtype=bool)

    for halo in halos_plus:
        # Halo center in grid coordinates
        com_shifted = halo['com'] + box_size / 2
        ix = int(com_shifted[0] / cell_size)
        iy = int(com_shifted[1] / cell_size)
        iz = int(com_shifted[2] / cell_size)

        # Mask radius in cells
        mask_cells = int(np.ceil(halo_mask_radius / cell_size))

        # Mark all cells within radius
        for dx in range(-mask_cells, mask_cells + 1):
            for dy in range(-mask_cells, mask_cells + 1):
                for dz in range(-mask_cells, mask_cells + 1):
                    if dx*dx + dy*dy + dz*dz <= mask_cells*mask_cells:
                        # Periodic boundary
                        jx = (ix + dx) % n_cells
                        jy = (iy + dy) % n_cells
                        jz = (iz + dz) % n_cells
                        halo_mask[jx, jy, jz] = True

    # Apply mask: keep only inter-halo regions
    inter_halo_grid = binary_grid.copy()
    inter_halo_grid[halo_mask] = 0

    # Label connected components
    structure = generate_binary_structure(3, 1)  # 6-connectivity
    labeled, n_components = label(inter_halo_grid, structure=structure)

    # Measure each component with STRICT criteria for real filaments
    filaments = []
    rejected_reasons = {'too_small': 0, 'too_large': 0, 'not_elongated': 0,
                        'too_short': 0, 'too_close_halo': 0}

    for comp_id in range(1, n_components + 1):
        cells = np.argwhere(labeled == comp_id)
        n_cells_comp = len(cells)

        # Criterion 1: Too small
        if n_cells_comp < 3:
            rejected_reasons['too_small'] += 1
            continue

        # Criterion 2: Too large = global structure, not a filament
        if n_cells_comp > 500:
            rejected_reasons['too_large'] += 1
            continue

        # Compute bounding box and coordinates
        cell_coords = cells * cell_size - box_size / 2

        # Bounding box dimensions WITH PERIODIC CORRECTION
        # This prevents false long filaments from wrap-around artifacts
        bbox_x = periodic_extent(cell_coords[:, 0], box_size)
        bbox_y = periodic_extent(cell_coords[:, 1], box_size)
        bbox_z = periodic_extent(cell_coords[:, 2], box_size)
        dims = sorted([bbox_x, bbox_y, bbox_z])

        # Criterion 3: Aspect ratio >= 3.0 (elongated structure)
        min_dim = max(dims[0], cell_size)  # Avoid division by zero
        aspect_ratio = dims[2] / min_dim
        if aspect_ratio < 3.0:
            rejected_reasons['not_elongated'] += 1
            continue

        # PCA to find principal axis and skeleton length
        # First unwrap coordinates if they wrap around the periodic boundary
        centered = cell_coords.copy()
        for dim in range(3):
            coords = centered[:, dim]
            # Check if wrapping occurred (span > box_size/2)
            if coords.max() - coords.min() > box_size / 2:
                # Shift negative-side coordinates to positive side
                median = np.median(coords)
                coords[coords < median - box_size/4] += box_size
        centered = centered - centered.mean(axis=0)

        if len(centered) > 1:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Length along principal axis
            projections = np.dot(centered, eigenvectors[:, -1])
            length = projections.max() - projections.min()
        else:
            length = cell_size

        # CRITICAL: Cap length at half box size (anything longer is an artifact)
        length = min(length, box_size / 2)

        # Criterion 4: Minimum length
        if length < min_filament_length:
            rejected_reasons['too_short'] += 1
            continue

        # Distance to nearest halo (check ALL cells, not just center)
        min_dist_to_halo = float('inf')
        for h in halos_plus:
            dists = np.linalg.norm(cell_coords - h['com'], axis=1)
            min_dist_to_halo = min(min_dist_to_halo, dists.min())

        # Criterion 5: Not too close to any halo
        if min_dist_to_halo < 5.0:
            rejected_reasons['too_close_halo'] += 1
            continue

        # Passed all criteria - this is a REAL filament
        cell_center = cell_coords.mean(axis=0)
        filaments.append({
            'length': length,
            'n_cells': n_cells_comp,
            'center': cell_center,
            'min_dist_to_halo': min_dist_to_halo,
            'aspect_ratio': aspect_ratio,
            'bbox': (bbox_x, bbox_y, bbox_z),
            'cells': cells
        })

    # Compile results
    if len(filaments) == 0:
        return {
            'n_filaments_real': 0,
            'length_mean_real': 0.0,
            'length_max_real': 0.0,
            'filament_cells': None,
            'halo_positions': [h['com'].tolist() for h in halos_plus],
            'n_halos_plus': len(halos_plus),
            'reason': 'no filaments after halo masking'
        }

    lengths = [f['length'] for f in filaments]

    # Combine all filament cells for visualization
    all_filament_cells = np.vstack([f['cells'] for f in filaments])

    halos_minus = [h for h in halos if h['sign'] == -1]

    return {
        'n_filaments_real': len(filaments),
        'length_mean_real': float(np.mean(lengths)),
        'length_max_real': float(np.max(lengths)),
        'length_total_real': float(np.sum(lengths)),
        'filament_cells': all_filament_cells,
        'halo_positions': [h['com'].tolist() for h in halos_plus],
        'halos_plus': [h['com'].tolist() for h in halos_plus],
        'halos_minus': [h['com'].tolist() for h in halos_minus],
        'n_halos_plus': len(halos_plus),
        'n_halos_minus': len(halos_minus),
        'filaments': filaments
    }


def compute_pi_prime(eta, lambda_base_mpc, sigma_v, rho_mean,
                     G=4.302e-6):
    """
    Compute Pi' invariant (improved physical invariant).

    Pi' = (G * rho_mean * lambda^2) / sigma_v^2 * eta
        = (lambda / lambda_Debye)^2 * eta

    Pi' << 1 -> filaments possible
    Pi' ~  1 -> transition
    Pi' >> 1 -> dipole dominant

    Args:
        eta: mass ratio parameter
        lambda_base_mpc: Yukawa screening length (Mpc)
        sigma_v: velocity dispersion (Mpc/Gyr)
        rho_mean: mean density (M_sun / Mpc^3)
        G: gravitational constant (Mpc^3 / (M_sun * Gyr^2))

    Returns:
        Pi' value
    """
    if sigma_v <= 0:
        return float('inf')

    # Debye length
    lambda_debye = sigma_v / np.sqrt(4 * np.pi * G * rho_mean)

    pi_prime = eta * (lambda_base_mpc / lambda_debye) ** 2

    return float(pi_prime)


def analyze_velocity_field_interhalo(pos, vel, signs, box_size, halos,
                                      min_dist=5.0, max_dist=30.0,
                                      cell_size=5.0):
    """
    Analyze velocity field in inter-halo regions.

    Args:
        pos: particle positions
        vel: particle velocities
        signs: particle signs
        box_size: simulation box size
        halos: list of halo dicts with 'com' key
        min_dist: minimum distance from halo (Mpc)
        max_dist: maximum distance from halo (Mpc)
        cell_size: grid cell size for averaging (Mpc)

    Returns:
        dict with velocity field data and coherence metrics
    """
    if vel is None:
        return {'status': 'no_velocities'}

    mask_plus = signs > 0
    pos_plus = pos[mask_plus]
    vel_plus = vel[mask_plus]

    halos_plus = [h for h in halos if h['sign'] == 1]
    if len(halos_plus) < 2:
        return {'status': 'insufficient_halos'}

    # Find particles in inter-halo region
    inter_halo_mask = np.zeros(len(pos_plus), dtype=bool)

    for i, p in enumerate(pos_plus):
        # Distance to nearest halo
        dists = [np.linalg.norm(p - h['com']) for h in halos_plus]
        min_d = min(dists)

        if min_dist <= min_d <= max_dist:
            inter_halo_mask[i] = True

    n_inter = inter_halo_mask.sum()
    if n_inter < 100:
        return {
            'status': 'insufficient_particles',
            'n_particles': int(n_inter)
        }

    pos_inter = pos_plus[inter_halo_mask]
    vel_inter = vel_plus[inter_halo_mask]

    # Grid the velocity field
    n_cells = int(box_size / cell_size)
    extent = box_size / 2

    vx_grid = np.zeros((n_cells, n_cells, n_cells))
    vy_grid = np.zeros((n_cells, n_cells, n_cells))
    vz_grid = np.zeros((n_cells, n_cells, n_cells))
    count_grid = np.zeros((n_cells, n_cells, n_cells))

    for i in range(len(pos_inter)):
        ix = int((pos_inter[i, 0] + extent) / cell_size)
        iy = int((pos_inter[i, 1] + extent) / cell_size)
        iz = int((pos_inter[i, 2] + extent) / cell_size)

        if 0 <= ix < n_cells and 0 <= iy < n_cells and 0 <= iz < n_cells:
            vx_grid[ix, iy, iz] += vel_inter[i, 0]
            vy_grid[ix, iy, iz] += vel_inter[i, 1]
            vz_grid[ix, iy, iz] += vel_inter[i, 2]
            count_grid[ix, iy, iz] += 1

    # Average
    mask = count_grid > 5
    vx_grid[mask] /= count_grid[mask]
    vy_grid[mask] /= count_grid[mask]
    vz_grid[mask] /= count_grid[mask]

    # Analyze flow coherence
    # For each filled cell, check if velocity points toward nearest halo
    coherent_count = 0
    total_count = 0

    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                if not mask[ix, iy, iz]:
                    continue

                # Cell center
                cx = (ix + 0.5) * cell_size - extent
                cy = (iy + 0.5) * cell_size - extent
                cz = (iz + 0.5) * cell_size - extent
                cell_pos = np.array([cx, cy, cz])

                # Velocity
                v = np.array([vx_grid[ix, iy, iz],
                             vy_grid[ix, iy, iz],
                             vz_grid[ix, iy, iz]])
                v_mag = np.linalg.norm(v)

                if v_mag < 0.01:
                    continue

                # Direction to nearest halo
                dists = [(np.linalg.norm(cell_pos - h['com']), h['com'])
                         for h in halos_plus]
                _, nearest_com = min(dists)

                to_halo = nearest_com - cell_pos
                to_halo_norm = to_halo / np.linalg.norm(to_halo)

                # Check alignment
                alignment = np.dot(v / v_mag, to_halo_norm)

                total_count += 1
                if alignment > 0.3:  # Pointing toward halo
                    coherent_count += 1

    coherence_fraction = coherent_count / total_count if total_count > 0 else 0

    return {
        'status': 'completed',
        'n_particles_interhalo': int(n_inter),
        'n_cells_filled': int(mask.sum()),
        'coherent_flow_fraction': float(coherence_fraction),
        'coherent_toward_halos': coherence_fraction > 0.5,
        'vx_grid': vx_grid,
        'vy_grid': vy_grid,
        'vz_grid': vz_grid,
        'count_grid': count_grid
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filament_metrics_v2.py <snapshot_path> [box_size]")
        sys.exit(1)

    snap_path = sys.argv[1]
    box_size = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0

    print(f"Loading {snap_path}...")
    pos, signs, vel = load_snapshot(snap_path)
    print(f"  N = {len(pos)} particles")
    print(f"  Velocities: {'available' if vel is not None else 'not available'}")

    print("\nDetecting inter-halo filaments...")
    result = detect_interhalos_filaments(pos, signs, box_size)

    print(f"\nResults:")
    print(f"  n_filaments_real: {result['n_filaments_real']}")
    print(f"  length_mean_real: {result['length_mean_real']:.1f} Mpc")
    print(f"  length_max_real:  {result['length_max_real']:.1f} Mpc")
    print(f"  n_halos_plus:     {result.get('n_halos_plus', 'N/A')}")

    if result['n_filaments_real'] > 0:
        print("\n  FILAMENTS DETECTED!")
    else:
        print(f"\n  No filaments. Reason: {result.get('reason', 'unknown')}")
