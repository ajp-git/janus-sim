#!/usr/bin/env python3
"""
Analyze old runs with new spinodal pipeline while Phase 1 runs.
CPU only - does not interfere with GPU simulations.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, fof_halos, detect_interhalos_filaments

# Runs to analyze
RUNS = [
    {
        'name': 'B3_lambda8',
        'path': '/mnt/T2/janus-sim/output/phase_b/B3_lambda8',
        'box_size': 150.0,
        'target_z': 0.0,
    },
    {
        'name': 'C3_Z1_eta05',
        'path': '/mnt/T2/janus-sim/output/phase_c/C3_Z1_eta05',
        'box_size': 150.0,
        'target_z': 0.0,
    },
    {
        'name': 'P2_z1.0',
        'path': '/mnt/T2/janus-sim/output/nuit3/P2_eta088_lambda8_Z1',
        'box_size': 150.0,
        'target_z': 1.0,
    },
    {
        'name': 'P2_z0.5',
        'path': '/mnt/T2/janus-sim/output/nuit3/P2_eta088_lambda8_Z1',
        'box_size': 150.0,
        'target_z': 0.5,
    },
]

# Detection parameters (more sensitive)
PARAMS = {
    'n_cells': 64,
    'halo_mask_radius': 10.0,
    'min_filament_length': 8.0,
}


def find_snapshot_for_z(run_dir, target_z):
    """Find snapshot closest to target redshift."""
    run_dir = Path(run_dir)
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))

    if not snaps:
        return None

    # For z=0, take the last snapshot
    if target_z == 0.0:
        return snaps[-1]

    # Try to find by naming convention or return appropriate one
    # Snapshots are usually at specific redshifts
    # snap_000000 = z_start, snap_NNNNN = z_end

    # Check if we have snapshot_redshifts info
    config_path = run_dir / 'config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        z_start = config.get('simulation', {}).get('z_start', 5.0)
        z_end = config.get('simulation', {}).get('z_end', 0.0)
        n_steps = config.get('simulation', {}).get('n_steps', 2000)

        # Linear interpolation to find step number
        # z(step) = z_start - (z_start - z_end) * step / n_steps
        target_step = int((z_start - target_z) / (z_start - z_end) * n_steps)

        # Find closest snapshot
        best_snap = None
        best_diff = float('inf')

        for snap in snaps:
            step = int(snap.stem.replace('snap_', ''))
            diff = abs(step - target_step)
            if diff < best_diff:
                best_diff = diff
                best_snap = snap

        return best_snap

    # Fallback: return middle snapshot for z=1.0, 75% for z=0.5
    n = len(snaps)
    if target_z >= 1.0:
        return snaps[n // 3]  # Early
    elif target_z >= 0.5:
        return snaps[2 * n // 3]  # Mid-late
    else:
        return snaps[-1]  # Final


def analyze_run(run_info):
    """Analyze a single run."""
    name = run_info['name']
    run_dir = Path(run_info['path'])
    box_size = run_info['box_size']
    target_z = run_info['target_z']

    print(f"\n{'='*60}")
    print(f"ANALYZING: {name} (target z={target_z})")
    print(f"{'='*60}")

    # Find snapshot
    snap_path = find_snapshot_for_z(run_dir, target_z)
    if not snap_path:
        print(f"  ERROR: No snapshots found in {run_dir}")
        return None

    print(f"  Snapshot: {snap_path.name}")

    # Load data
    try:
        pos, signs, vel = load_snapshot(str(snap_path))
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None

    n = len(pos)
    print(f"  N = {n} particles")
    print(f"  Velocities: {'yes' if vel is not None else 'no'}")

    # Basic metrics
    mask_plus = signs > 0
    mask_minus = signs < 0
    com_plus = pos[mask_plus].mean(axis=0)
    com_minus = pos[mask_minus].mean(axis=0)
    dcom = np.linalg.norm(com_plus - com_minus)
    seg = dcom / box_size

    print(f"  ΔCOM = {dcom:.1f} Mpc, S = {seg:.4f}")

    # Filament detection with sensitive parameters
    print(f"\n  Detecting filaments (threshold=1.1-2.5, mask={PARAMS['halo_mask_radius']} Mpc)...")

    result = detect_interhalos_filaments(
        pos, signs, box_size,
        n_cells=PARAMS['n_cells'],
        halo_mask_radius=PARAMS['halo_mask_radius'],
        min_filament_length=PARAMS['min_filament_length']
    )

    n_filaments = result['n_filaments_real']
    length_max = result['length_max_real']
    length_mean = result.get('length_mean_real', 0.0)
    n_halos_plus = result.get('n_halos_plus', 0)

    print(f"\n  RESULTS:")
    print(f"    n_filaments_real = {n_filaments}")
    print(f"    length_max_real  = {length_max:.1f} Mpc")
    print(f"    length_mean_real = {length_mean:.1f} Mpc")
    print(f"    n_halos_plus     = {n_halos_plus}")

    # Compute density in filament region if filaments exist
    density_mean = 1.0
    if n_filaments > 0 and 'filaments' in result:
        cell_size = box_size / PARAMS['n_cells']
        filaments = result['filaments']
        densities = []

        rho_mean_sim = n / (box_size ** 3)

        for fil in filaments:
            cells = fil['cells']
            cell_coords = cells * cell_size - box_size / 2

            for cc in cell_coords[:10]:  # Sample first 10 cells
                r = 5.0
                dists = np.linalg.norm(pos - cc, axis=1)
                n_local = (dists < r).sum()
                vol = (4/3) * np.pi * r**3
                rho_local = n_local / vol
                densities.append(rho_local / rho_mean_sim)

        if densities:
            density_mean = np.mean(densities)

    print(f"    density_mean     = {density_mean:.2f} ρ̄")

    if n_filaments > 0:
        print(f"\n  *** FILAMENTS DETECTED! ***")
    else:
        reason = result.get('reason', 'unknown')
        print(f"\n  No filaments. Reason: {reason}")

    return {
        'name': name,
        'target_z': target_z,
        'n_particles': n,
        'dcom': dcom,
        'segregation': seg,
        'n_filaments': n_filaments,
        'length_max': length_max,
        'length_mean': length_mean,
        'density_mean': density_mean,
        'n_halos_plus': n_halos_plus,
    }


def main():
    print("\n" + "="*60)
    print("OLD RUNS ANALYSIS (Parallel to Phase 1)")
    print("="*60)
    print(f"Parameters: mask_halos={PARAMS['halo_mask_radius']} Mpc, min_length={PARAMS['min_filament_length']} Mpc")

    results = []

    for run_info in RUNS:
        result = analyze_run(run_info)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Run':<20} {'z':>5} {'n_fil':>6} {'L_max':>8} {'ρ_mean':>8} {'ΔCOM':>8}")
    print("-"*60)

    for r in results:
        print(f"{r['name']:<20} {r['target_z']:>5.1f} {r['n_filaments']:>6} "
              f"{r['length_max']:>8.1f} {r['density_mean']:>8.2f} {r['dcom']:>8.1f}")

    # Check if any had filaments we missed
    with_filaments = [r for r in results if r['n_filaments'] > 0]

    print("\n" + "="*60)
    if with_filaments:
        print(f"FOUND {len(with_filaments)} RUNS WITH FILAMENTS:")
        for r in with_filaments:
            print(f"  - {r['name']}: {r['n_filaments']} filaments, max {r['length_max']:.1f} Mpc")
    else:
        print("NO FILAMENTS DETECTED IN OLD RUNS")
        print("(confirms P2 z=0 finding was special)")

    print("="*60 + "\n")


if __name__ == '__main__':
    main()
