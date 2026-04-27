#!/usr/bin/env python3
"""
Stellar Post-Processing — Apply SF criteria to existing snapshots

Analyzes all snapshots from janus_baryonic_calibrated run and identifies
proto-stars based on local overdensity and velocity divergence criteria.

Algorithm:
1. For each snapshot, compute local SPH density using kNN
2. Identify proto-stars where: overdensity > 5.0 AND div(v) < 0
3. Track stellar mass evolution M_stellar(z)
4. Compute SFR and compare to Madau plot

Optimizations:
- scipy.spatial.KDTree for vectorized kNN
- 10% sampling for N > 1M particles
- Multiprocessing for parallel snapshot processing
"""

import numpy as np
import struct
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree
import time
import sys

# Physical constants
M_PARTICLE_MSUN = 3e11 * (500.0 / 500.0)**3 / 10_000_000  # ~3e4 M_sun per particle


def read_snapshot(path):
    """Read JSNP binary snapshot format"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]

        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    z = 1.0 / a - 1.0
    step = int(Path(path).stem.replace('snap_', ''))

    return {
        'n': n, 'a': a, 't': t, 'z': z, 'step': step,
        'pos': pos, 'vel': vel, 'signs': signs
    }


def cubic_spline_kernel(r, h):
    """
    Cubic spline SPH kernel (3D normalized)
    W(r, h) for smoothing length h
    """
    q = r / h
    sigma = 1.0 / (np.pi * h**3)

    w = np.zeros_like(q)

    # q < 1
    mask1 = q < 1.0
    w[mask1] = sigma * (1.0 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3)

    # 1 <= q < 2
    mask2 = (q >= 1.0) & (q < 2.0)
    w[mask2] = sigma * 0.25 * (2.0 - q[mask2])**3

    return w


def compute_velocity_divergence(pos_i, vel_i, pos_neighbors, vel_neighbors, h):
    """
    Estimate velocity divergence using SPH gradient
    div(v) = sum_j (v_j - v_i) . grad_W(r_ij, h) * m_j / rho_j

    Simplified: use finite difference approximation
    """
    dr = pos_neighbors - pos_i
    dv = vel_neighbors - vel_i
    r = np.linalg.norm(dr, axis=1)

    # Avoid division by zero
    r = np.maximum(r, 1e-10)

    # Radial velocity component (expansion/contraction)
    v_radial = np.sum(dv * dr, axis=1) / r

    # Approximate divergence as mean radial velocity / mean distance
    # Negative = converging flow (collapse)
    div_v = np.mean(v_radial) / np.mean(r)

    return div_v


def process_snapshot(args):
    """
    Process a single snapshot and identify proto-stars

    Returns: dict with step, z, n_proto_stars, positions of proto-stars
    """
    snap_path, n_neighbors, overdensity_threshold, sample_fraction, l_box = args

    try:
        data = read_snapshot(snap_path)
    except Exception as e:
        print(f"Error reading {snap_path}: {e}", file=sys.stderr)
        return None

    step = data['step']
    z = data['z']
    pos = data['pos']
    vel = data['vel']
    signs = data['signs']

    # Extract m+ particles only
    plus_mask = signs > 0
    pos_plus = pos[plus_mask].astype(np.float64)
    vel_plus = vel[plus_mask].astype(np.float64)
    n_plus = len(pos_plus)

    if n_plus == 0:
        return {'step': step, 'z': z, 'n_proto_stars': 0,
                'n_sampled': 0, 'scale_factor': 1.0}

    # Compute mean density
    rho_mean = n_plus / l_box**3

    # Sample if too many particles
    if n_plus > 1_000_000 and sample_fraction < 1.0:
        n_sample = int(n_plus * sample_fraction)
        sample_idx = np.random.choice(n_plus, n_sample, replace=False)
        pos_sample = pos_plus[sample_idx]
        vel_sample = vel_plus[sample_idx]
        scale_factor = 1.0 / sample_fraction
    else:
        pos_sample = pos_plus
        vel_sample = vel_plus
        n_sample = n_plus
        scale_factor = 1.0

    # Build KD-tree for all m+ particles (for neighbor queries)
    # Shift coordinates to [0, L_box] range for periodic KDTree
    pos_plus_shifted = pos_plus + l_box / 2.0
    pos_sample_shifted = pos_sample + l_box / 2.0

    tree = KDTree(pos_plus_shifted, boxsize=l_box)

    # Query k nearest neighbors for sampled particles
    k = min(n_neighbors + 1, n_plus)  # +1 because particle finds itself
    distances, indices = tree.query(pos_sample_shifted, k=k)

    # Remove self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Compute smoothing length as mean distance to kth neighbor
    h_array = np.mean(distances, axis=1)

    # Compute local density using SPH kernel
    rho_local = np.zeros(n_sample)
    for i in range(n_sample):
        h = h_array[i]
        r = distances[i]
        w = cubic_spline_kernel(r, h)
        rho_local[i] = np.sum(w)  # Sum of kernel weights (particle count weighted)

    # Normalize to get overdensity
    # rho_local is in units of "particles per kernel volume"
    # Convert to overdensity relative to mean
    h_mean = np.mean(h_array)
    kernel_volume = (4.0/3.0) * np.pi * h_mean**3
    overdensity = rho_local / (rho_mean * kernel_volume) * n_neighbors

    # Compute velocity divergence for high-density regions
    # Only compute for particles above half the threshold (optimization)
    high_density_mask = overdensity > (overdensity_threshold / 2)
    div_v = np.zeros(n_sample)

    high_density_idx = np.where(high_density_mask)[0]
    for i in high_density_idx:
        neighbor_idx = indices[i]
        pos_neighbors = pos_plus[neighbor_idx]
        vel_neighbors = vel_plus[neighbor_idx]
        div_v[i] = compute_velocity_divergence(
            pos_sample[i], vel_sample[i],
            pos_neighbors, vel_neighbors,
            h_array[i]
        )

    # Apply proto-star criteria
    proto_star_mask = (overdensity > overdensity_threshold) & (div_v < 0)
    n_proto_stars = int(np.sum(proto_star_mask) * scale_factor)

    # Find position of densest region
    if np.any(overdensity > 1):
        max_density_idx = np.argmax(overdensity)
        halo_pos = pos_sample[max_density_idx].tolist()
        max_overdensity = overdensity[max_density_idx]
    else:
        halo_pos = [0.0, 0.0, 0.0]
        max_overdensity = 0.0

    # Proto-star positions (for visualization)
    proto_star_pos = pos_sample[proto_star_mask].tolist() if np.any(proto_star_mask) else []

    return {
        'step': step,
        'z': z,
        't_gyr': data['t'],
        'n_proto_stars': n_proto_stars,
        'n_sampled': n_sample,
        'scale_factor': scale_factor,
        'max_overdensity': max_overdensity,
        'halo_pos': halo_pos,
        'n_converging': int(np.sum(div_v < 0) * scale_factor),
        'proto_star_pos': proto_star_pos[:100]  # Limit for storage
    }


def main():
    parser = argparse.ArgumentParser(description='Stellar post-processing')
    parser.add_argument('--snap-dir', required=True, help='Snapshot directory')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--n-neighbors', type=int, default=32, help='kNN neighbors')
    parser.add_argument('--overdensity-threshold', type=float, default=5.0,
                        help='Overdensity threshold for SF')
    parser.add_argument('--sample-fraction', type=float, default=0.1,
                        help='Fraction of particles to sample')
    parser.add_argument('--n-cores', type=int, default=4, help='Number of cores')
    parser.add_argument('--l-box', type=float, default=500.0, help='Box size [Mpc]')

    args = parser.parse_args()

    print("=" * 70)
    print("STELLAR POST-PROCESSING — Proto-star identification")
    print("=" * 70)
    print(f"Snapshot dir: {args.snap_dir}")
    print(f"Output: {args.output}")
    print(f"k-NN neighbors: {args.n_neighbors}")
    print(f"Overdensity threshold: {args.overdensity_threshold}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Cores: {args.n_cores}")
    print("=" * 70)

    # Find all snapshots
    snap_dir = Path(args.snap_dir)
    snapshots = sorted(snap_dir.glob('snap_*.bin'))
    print(f"\nFound {len(snapshots)} snapshots")

    if len(snapshots) == 0:
        print("ERROR: No snapshots found!")
        return

    # Prepare arguments for parallel processing
    task_args = [
        (str(snap), args.n_neighbors, args.overdensity_threshold,
         args.sample_fraction, args.l_box)
        for snap in snapshots
    ]

    # Process in parallel
    print(f"\nProcessing with {args.n_cores} cores...")
    start_time = time.time()

    results = []
    with Pool(args.n_cores) as pool:
        for i, result in enumerate(pool.imap(process_snapshot, task_args)):
            if result is not None:
                results.append(result)
                if (i + 1) % 50 == 0 or i == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(snapshots) - i - 1) / rate / 60
                    print(f"  [{i+1}/{len(snapshots)}] step={result['step']}, "
                          f"z={result['z']:.2f}, N_proto={result['n_proto_stars']}, "
                          f"ETA={eta:.1f}min")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes")

    # Sort by step
    results = sorted(results, key=lambda x: x['step'])

    # Compute SFR
    dt = 0.001  # Gyr per step (from simulation config)
    m_particle = M_PARTICLE_MSUN

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("step,z,t_gyr,n_proto_stars,m_stellar_msun,sfr_msun_gyr,"
                "max_overdensity,n_converging,halo_x,halo_y,halo_z\n")

        m_stellar_cumulative = 0.0
        prev_n_proto = 0

        for r in results:
            # Cumulative stellar mass (once a proto-star, always a star)
            new_stars = max(0, r['n_proto_stars'] - prev_n_proto)
            m_stellar_cumulative += new_stars * m_particle

            # SFR
            sfr = new_stars * m_particle / dt if new_stars > 0 else 0.0

            f.write(f"{r['step']},{r['z']:.4f},{r['t_gyr']:.4f},"
                    f"{r['n_proto_stars']},{m_stellar_cumulative:.2e},{sfr:.2e},"
                    f"{r['max_overdensity']:.2f},{r['n_converging']},"
                    f"{r['halo_pos'][0]:.2f},{r['halo_pos'][1]:.2f},{r['halo_pos'][2]:.2f}\n")

            prev_n_proto = r['n_proto_stars']

    print(f"\nSaved: {output_path}")

    # Generate figure
    try:
        generate_figure(results, output_path.parent / 'stellar_sfr.png', m_particle, dt)
    except Exception as e:
        print(f"Warning: Could not generate figure: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find key redshifts
    for target_z in [3.0, 2.0, 1.0, 0.5, 0.0]:
        closest = min(results, key=lambda x: abs(x['z'] - target_z))
        if abs(closest['z'] - target_z) < 0.2:
            print(f"  z={target_z:.1f}: N_proto_stars = {closest['n_proto_stars']}, "
                  f"max_overdensity = {closest['max_overdensity']:.1f}")

    # Peak SFR
    if results:
        sfr_values = []
        prev_n = 0
        for r in results:
            new_stars = max(0, r['n_proto_stars'] - prev_n)
            sfr_values.append((r['z'], new_stars * m_particle / dt))
            prev_n = r['n_proto_stars']

        if sfr_values:
            peak_z, peak_sfr = max(sfr_values, key=lambda x: x[1])
            print(f"\n  Peak SFR: {peak_sfr:.2e} M_sun/Gyr at z={peak_z:.2f}")

    print("=" * 70)


def generate_figure(results, output_path, m_particle, dt):
    """Generate stellar evolution figure"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Extract data
    z_vals = [r['z'] for r in results]
    n_proto = [r['n_proto_stars'] for r in results]
    max_od = [r['max_overdensity'] for r in results]

    # Compute cumulative stellar mass and SFR
    m_stellar = []
    sfr = []
    cumulative = 0.0
    prev_n = 0
    for r in results:
        new_stars = max(0, r['n_proto_stars'] - prev_n)
        cumulative += new_stars * m_particle
        m_stellar.append(cumulative)
        sfr.append(new_stars * m_particle / dt if new_stars > 0 else 0.0)
        prev_n = r['n_proto_stars']

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: N_proto_stars(z)
    ax1 = axes[0, 0]
    ax1.semilogy(z_vals, np.maximum(n_proto, 1), 'b-', lw=1.5)
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('N proto-stars')
    ax1.set_title('Proto-star Count Evolution')
    ax1.set_xlim(max(z_vals), 0)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1000, color='r', ls='--', label='Target N=1000')
    ax1.legend()

    # Panel 2: SFR(z) vs Madau
    ax2 = axes[0, 1]
    ax2.semilogy(z_vals, np.maximum(sfr, 1e-5), 'g-', lw=1.5, label='Janus')

    # Madau & Dickinson 2014 fit (approximate)
    z_madau = np.linspace(0, 8, 100)
    sfr_madau = 0.015 * (1 + z_madau)**2.7 / (1 + ((1 + z_madau)/2.9)**5.6)
    sfr_madau *= 1e9  # Convert to M_sun/Gyr (original is per year)
    ax2.semilogy(z_madau, sfr_madau, 'k--', lw=2, alpha=0.7, label='Madau+14')

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('SFR [M$_\\odot$/Gyr]')
    ax2.set_title('Star Formation Rate')
    ax2.set_xlim(max(z_vals), 0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Cumulative stellar mass
    ax3 = axes[1, 0]
    ax3.semilogy(z_vals, np.maximum(m_stellar, 1), 'r-', lw=1.5)
    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('M$_{stellar}$ [M$_\\odot$]')
    ax3.set_title('Cumulative Stellar Mass')
    ax3.set_xlim(max(z_vals), 0)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Max overdensity
    ax4 = axes[1, 1]
    ax4.semilogy(z_vals, np.maximum(max_od, 0.1), 'purple', lw=1.5)
    ax4.axhline(5.0, color='r', ls='--', label='SF threshold')
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Max Overdensity')
    ax4.set_title('Peak Density Evolution')
    ax4.set_xlim(max(z_vals), 0)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {output_path}")


if __name__ == '__main__':
    main()
