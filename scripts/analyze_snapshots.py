#!/usr/bin/env python3
"""
Analyze binary snapshots by sampling - never load full file in memory.
Format: N (u64) + N × (f32, f32, f32, i8) = 8 + N×13 bytes
"""

import struct
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters
SNAPSHOT_DIR = Path("/mnt/T2/janus-sim/output/40M_publication_2026-02-27/snapshots")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/40M_publication_2026-02-27")
BOX_SIZE = 736.8
SAMPLE_SIZE = 100_000
CHUNK_SIZE = 1000
PARTICLE_SIZE = 13  # 3×f32 + 1×i8 = 12 + 1 = 13 bytes

def read_snapshot_header(filepath):
    """Read just the header (N particles)"""
    with open(filepath, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
    return n

def sample_snapshot(filepath, sample_size=100_000, chunk_size=1000):
    """
    Sample particles from snapshot without loading full file.
    Returns: positions (Nx3), signs (N,)
    """
    n_total = read_snapshot_header(filepath)

    # Calculate sampling rate
    sample_every = max(1, n_total // sample_size)

    positions = []
    signs = []

    with open(filepath, 'rb') as f:
        # Skip header
        f.seek(8)

        particle_idx = 0
        sampled = 0

        while particle_idx < n_total and sampled < sample_size:
            # Read chunk
            chunk_to_read = min(chunk_size, n_total - particle_idx)
            data = f.read(chunk_to_read * PARTICLE_SIZE)

            if len(data) < chunk_to_read * PARTICLE_SIZE:
                break

            # Parse chunk
            for i in range(chunk_to_read):
                if (particle_idx + i) % sample_every == 0 and sampled < sample_size:
                    offset = i * PARTICLE_SIZE
                    x, y, z = struct.unpack('<fff', data[offset:offset+12])
                    sign = struct.unpack('<b', data[offset+12:offset+13])[0]
                    positions.append([x, y, z])
                    signs.append(sign)
                    sampled += 1

            particle_idx += chunk_to_read

    return np.array(positions, dtype=np.float32), np.array(signs, dtype=np.int8), n_total


def compute_full_segregation(filepath, chunk_size=100_000):
    """
    Compute exact segregation by streaming through ALL particles.
    Only keeps running sums in memory - O(1) space complexity.
    Returns: (seg_mpc, n_plus, n_minus)
    """
    n_total = read_snapshot_header(filepath)

    # Running sums for COM calculation
    sum_plus = np.zeros(3, dtype=np.float64)
    sum_minus = np.zeros(3, dtype=np.float64)
    n_plus = 0
    n_minus = 0

    with open(filepath, 'rb') as f:
        f.seek(8)  # Skip header

        particles_read = 0
        while particles_read < n_total:
            # Read large chunks for efficiency
            to_read = min(chunk_size, n_total - particles_read)
            data = f.read(to_read * PARTICLE_SIZE)

            if len(data) < to_read * PARTICLE_SIZE:
                break

            # Parse all particles in chunk using numpy for speed
            chunk_data = np.frombuffer(data, dtype=np.uint8).reshape(-1, PARTICLE_SIZE)

            # Extract positions (bytes 0-11) and signs (byte 12)
            pos_bytes = chunk_data[:, :12].tobytes()
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            signs = chunk_data[:, 12].astype(np.int8)

            # Accumulate sums
            mask_pos = signs > 0
            mask_neg = signs < 0

            sum_plus += positions[mask_pos].sum(axis=0).astype(np.float64)
            sum_minus += positions[mask_neg].sum(axis=0).astype(np.float64)
            n_plus += mask_pos.sum()
            n_minus += mask_neg.sum()

            particles_read += to_read

    # Compute COMs
    com_plus = sum_plus / n_plus if n_plus > 0 else np.zeros(3)
    com_minus = sum_minus / n_minus if n_minus > 0 else np.zeros(3)

    # Segregation = distance between COMs
    d_cm = np.linalg.norm(com_plus - com_minus)

    return d_cm, n_plus, n_minus, com_plus, com_minus

def compute_statistics(pos, signs, box_size):
    """Compute all validation statistics"""
    stats = {}

    # Separate populations
    mask_pos = signs > 0
    mask_neg = signs < 0

    pos_plus = pos[mask_pos]
    pos_minus = pos[mask_neg]

    n_plus = len(pos_plus)
    n_minus = len(pos_minus)

    stats['n_plus'] = n_plus
    stats['n_minus'] = n_minus
    stats['n_ratio'] = n_plus / n_minus if n_minus > 0 else 0
    stats['eta_measured'] = n_minus / n_plus if n_plus > 0 else 0

    # Center of mass
    cm_plus = pos_plus.mean(axis=0) if n_plus > 0 else np.zeros(3)
    cm_minus = pos_minus.mean(axis=0) if n_minus > 0 else np.zeros(3)

    stats['cm_plus'] = cm_plus
    stats['cm_minus'] = cm_minus

    # Segregation
    d_cm = np.linalg.norm(cm_plus - cm_minus)
    stats['segregation'] = d_cm / box_size
    stats['d_cm_absolute'] = d_cm

    # Spatial distribution - histogram per axis
    stats['histograms'] = {}
    for ax, name in enumerate(['x', 'y', 'z']):
        hist, edges = np.histogram(pos[:, ax], bins=10, range=(-box_size/2, box_size/2))
        expected = len(pos) / 10
        chi2 = np.sum((hist - expected)**2 / expected)
        stats['histograms'][name] = {
            'counts': hist,
            'chi2': chi2,
            'p_uniform': chi2 < 16.92  # chi2 critical value for 9 dof, p=0.05
        }

    # Density fluctuation (using 10x10x10 grid)
    grid_size = 10
    cell_size = box_size / grid_size
    density_grid = np.zeros((grid_size, grid_size, grid_size))

    for p in pos:
        ix = int((p[0] + box_size/2) / cell_size) % grid_size
        iy = int((p[1] + box_size/2) / cell_size) % grid_size
        iz = int((p[2] + box_size/2) / cell_size) % grid_size
        density_grid[ix, iy, iz] += 1

    mean_density = density_grid.mean()
    std_density = density_grid.std()
    stats['delta_rho_over_rho'] = std_density / mean_density if mean_density > 0 else 0

    return stats

def create_validation_plot(pos, signs, cm_plus, cm_minus, filepath, title=""):
    """Create XY projection plot with CM markers"""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    mask_pos = signs > 0
    mask_neg = signs < 0

    # Plot particles with low alpha for density
    ax.scatter(pos[mask_pos, 0], pos[mask_pos, 1],
               c='blue', s=0.1, alpha=0.3, label=f'+1 (n={mask_pos.sum()})')
    ax.scatter(pos[mask_neg, 0], pos[mask_neg, 1],
               c='red', s=0.1, alpha=0.3, label=f'-1 (n={mask_neg.sum()})')

    # Plot CM markers
    ax.scatter([cm_plus[0]], [cm_plus[1]], c='yellow', s=200, marker='x',
               linewidths=3, label=f'CM+ ({cm_plus[0]:.1f}, {cm_plus[1]:.1f})')
    ax.scatter([cm_minus[0]], [cm_minus[1]], c='lime', s=200, marker='x',
               linewidths=3, label=f'CM- ({cm_minus[0]:.1f}, {cm_minus[1]:.1f})')

    ax.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, facecolor='black')
    plt.close()
    print(f"  Saved: {filepath}")

def main():
    print("=" * 60)
    print("SNAPSHOT VALIDATION ANALYSIS")
    print("=" * 60)
    print()

    # Check if snapshots exist
    snap0 = SNAPSHOT_DIR / "snapshot_00000.bin"
    snap100 = SNAPSHOT_DIR / "snapshot_00100.bin"

    if not snap0.exists():
        print(f"ERROR: {snap0} not found")
        return

    # === SNAPSHOT 0 ===
    print(f"[1] Analyzing {snap0.name}...")

    # Full segregation calculation (streaming through all 40M particles)
    print(f"    Computing EXACT segregation from all particles...")
    seg0_full, n_plus_full, n_minus_full, com_plus_full, com_minus_full = compute_full_segregation(snap0)

    print()
    print("    === EXACT STATISTICS (all 40M particles) ===")
    print(f"    N+ total: {n_plus_full:,}")
    print(f"    N- total: {n_minus_full:,}")
    print(f"    eta = N-/N+ = {n_minus_full/n_plus_full:.6f} (expected: 1.045)")
    print()
    print(f"    CM+ = ({com_plus_full[0]:.4f}, {com_plus_full[1]:.4f}, {com_plus_full[2]:.4f}) Mpc")
    print(f"    CM- = ({com_minus_full[0]:.4f}, {com_minus_full[1]:.4f}, {com_minus_full[2]:.4f}) Mpc")
    print(f"    |CM+ - CM-| = {seg0_full:.6f} Mpc")
    print(f"    S_0/box = {seg0_full/BOX_SIZE:.8f}")
    print()

    # Sample for visualization and uniformity tests
    print(f"    Sampling {SAMPLE_SIZE:,} particles for plots...")
    pos0, signs0, n_total = sample_snapshot(snap0, SAMPLE_SIZE, CHUNK_SIZE)

    stats0 = compute_statistics(pos0, signs0, BOX_SIZE)

    print(f"    delta_rho/rho = {stats0['delta_rho_over_rho']:.4f} (expect ~0.05 for Zeldovich at z=5)")
    print()
    print("    Spatial uniformity (chi2 test, p>0.05 = uniform):")
    for ax, data in stats0['histograms'].items():
        status = "UNIFORM" if data['p_uniform'] else "NON-UNIFORM"
        print(f"      {ax}: chi2={data['chi2']:.2f}, {status}")

    # Store exact segregation for comparison
    stats0['seg_exact'] = seg0_full
    stats0['com_plus_exact'] = com_plus_full
    stats0['com_minus_exact'] = com_minus_full

    # Create plot for step 0 (using exact COM values)
    create_validation_plot(pos0, signs0, com_plus_full, com_minus_full,
                          OUTPUT_DIR / "validation_step0.png",
                          f"Step 0: Seg={seg0_full:.4f} Mpc, N={n_total:,}")

    # === SNAPSHOT 100 (if exists) ===
    print()
    if snap100.exists():
        print(f"[2] Analyzing {snap100.name}...")

        # Full segregation calculation
        print(f"    Computing EXACT segregation from all particles...")
        seg100_full, n_plus_100, n_minus_100, com_plus_100, com_minus_100 = compute_full_segregation(snap100)

        print()
        print("    === EXACT STATISTICS (step 100) ===")
        print(f"    CM+ = ({com_plus_100[0]:.4f}, {com_plus_100[1]:.4f}, {com_plus_100[2]:.4f}) Mpc")
        print(f"    CM- = ({com_minus_100[0]:.4f}, {com_minus_100[1]:.4f}, {com_minus_100[2]:.4f}) Mpc")
        print(f"    |CM+ - CM-| = {seg100_full:.6f} Mpc")
        print(f"    S_100/box = {seg100_full/BOX_SIZE:.8f}")
        print()

        # Compare segregation
        delta_s = seg100_full - seg0_full
        if delta_s > 0:
            print(f"    SEGREGATION GROWING: delta_S = +{delta_s:.6f} Mpc ({100*delta_s/seg0_full:.1f}%)")
        else:
            print(f"    WARNING: Segregation decreasing! delta_S = {delta_s:.6f} Mpc")

        # Sample for visualization
        pos100, signs100, _ = sample_snapshot(snap100, SAMPLE_SIZE, CHUNK_SIZE)

        # Create plot for step 100
        create_validation_plot(pos100, signs100, com_plus_100, com_minus_100,
                              OUTPUT_DIR / "validation_step100.png",
                              f"Step 100: Seg={seg100_full:.4f} Mpc")

        stats0['seg100_exact'] = seg100_full
    else:
        print(f"[2] {snap100.name} not yet available (run in progress)")

    # === SUMMARY ===
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print()
    print(f"  N+ = {n_plus_full:,}, N- = {n_minus_full:,}")
    print(f"  eta = N-/N+ = {n_minus_full/n_plus_full:.6f} (expected: 1.045)")
    print()
    print(f"  Seg_0 (exact):    {stats0['seg_exact']:.6f} Mpc")
    if 'seg100_exact' in stats0:
        print(f"  Seg_100 (exact):  {stats0['seg100_exact']:.6f} Mpc")
        print(f"  delta_Seg:        {stats0['seg100_exact'] - stats0['seg_exact']:.6f} Mpc")
    print()
    print(f"  delta_rho/rho (sampled): {stats0['delta_rho_over_rho']:.4f}")
    print()

    # Validation checks
    checks = []
    eta_measured = n_minus_full / n_plus_full
    checks.append(("eta ~ 1.045", abs(eta_measured - 1.045) < 0.01))
    checks.append(("Seg_0 matches CSV (~0.063 Mpc)", abs(stats0['seg_exact'] - 0.063) < 0.01))
    if 'seg100_exact' in stats0:
        checks.append(("Segregation growing", stats0['seg100_exact'] > stats0['seg_exact']))

    print("  Validation checks:")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED - Review above")
    print()

if __name__ == "__main__":
    main()
