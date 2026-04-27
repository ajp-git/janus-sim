#!/usr/bin/env python3
"""
Fast Phase 1 analysis - only z=0.5 snapshot.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

OUTPUT_DIR = Path('/mnt/T2/janus-sim/output/trichotomie')
PHASE1_DIR = OUTPUT_DIR / 'phase1'


def find_snapshot_z05(run_dir, n_steps=600, z_start=5.0, z_end=0.0):
    """Find snapshot at z~0.5."""
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None
    # z=0.5 is 90% of the way from z=5 to z=0
    target_step = int(0.9 * n_steps)
    best_snap = None
    best_diff = float('inf')
    for snap in snaps:
        step = int(snap.stem.replace('snap_', ''))
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_snap = snap
    return best_snap


def analyze_single_run(run_dir, box_size=150.0):
    """Analyze a single run at z~0.5 only."""
    name = run_dir.name
    parts = name.split('_')
    eta = float(parts[1].replace('eta', ''))
    lam = int(parts[2].replace('lam', ''))
    z_act = float(parts[3].replace('z', ''))

    snap = find_snapshot_z05(run_dir)
    if snap is None:
        return None

    try:
        pos, signs, vel = load_snapshot(str(snap))
    except:
        return None

    n = len(pos)

    # Use STRICT criteria
    result = detect_interhalos_filaments(pos, signs, box_size,
                                          n_cells=64, halo_mask_radius=10.0,
                                          min_filament_length=8.0)

    n_filaments = result['n_filaments_real']
    length_max = result['length_max_real']
    n_halos = result.get('n_halos_plus', 0)

    # Compute score
    s_length = min(length_max / 30.0, 1.0)
    s_count = min(n_filaments / 3.0, 1.0)
    score = 0.30*s_length + 0.25*s_count + 0.45*0.5  # Simplified

    return {
        'run_name': name,
        'eta': eta,
        'lambda': lam,
        'z_act': z_act,
        'n_filaments': n_filaments,
        'length_max': length_max,
        'n_halos': n_halos,
        'score': score,
    }


def main():
    print("="*70)
    print("PHASE 1 FAST ANALYSIS - 27 RUNS (z=0.5 only)")
    print("="*70)

    runs = sorted(PHASE1_DIR.glob('T1_*'))
    print(f"Found {len(runs)} runs\n")

    results = []

    # Sequential for now (FOF is memory-intensive)
    for run_dir in runs:
        print(f"Analyzing {run_dir.name}...", end=' ', flush=True)
        result = analyze_single_run(run_dir)

        if result:
            print(f"n_fil={result['n_filaments']}, L={result['length_max']:.1f}")
            results.append(result)
        else:
            print("FAILED")

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False)

    # Save
    df.to_csv(OUTPUT_DIR / 'phase1_results_strict.csv', index=False)

    print("\n" + "="*70)
    print("TOP 10 RESULTS (STRICT CRITERIA)")
    print("="*70)
    print(df[['run_name', 'eta', 'lambda', 'z_act', 'n_filaments',
              'length_max', 'n_halos', 'score']].head(10).to_string(index=False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Runs with filaments
    with_fil = df[df['n_filaments'] > 0]
    print(f"\nRuns with filaments: {len(with_fil)} / {len(df)}")

    if len(with_fil) > 0:
        print(f"\nTop 3 for Phase 2:")
        for i, row in enumerate(with_fil.head(3).itertuples()):
            print(f"  #{i+1}: {row.run_name}")
            print(f"       η={row.eta}, λ={row._3}, z_act={row.z_act}")
            print(f"       n_filaments={row.n_filaments}, length_max={row.length_max:.1f} Mpc")


if __name__ == '__main__':
    main()
