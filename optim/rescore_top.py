#!/usr/bin/env python3
"""
Rescore only the TOP runs from each tour with periodic-corrected metrics.
Much faster than rescoring all 1353 runs.
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments


@dataclass
class FilamentMetrics:
    n_filaments: int
    length_max: float
    length_mean: float
    density_mean: float
    fraction_mplus: float
    coherent_flow: float
    n_halos: int


def filament_score(m: FilamentMetrics) -> float:
    s_length = min(m.length_max / 30.0, 1.0)
    s_count = min(m.n_filaments / 3.0, 1.0)
    s_density = min(max(m.density_mean - 1.0, 0) / 0.5, 1.0)
    s_purity = min(m.fraction_mplus / 0.8, 1.0)
    s_flow = min(m.coherent_flow / 0.5, 1.0)
    return (0.30 * s_length + 0.25 * s_count + 0.20 * s_density +
            0.15 * s_purity + 0.10 * s_flow)


def find_snapshot_z05(run_dir: Path) -> Path:
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None
    steps = [int(s.stem.replace('snap_', '')) for s in snaps]
    max_step = max(steps)
    target = int(0.9 * max_step)
    return min(snaps, key=lambda s: abs(int(s.stem.replace('snap_', '')) - target))


def analyze_run(run_dir: Path, box_size: float = 150.0) -> dict:
    snap = find_snapshot_z05(run_dir)
    if snap is None:
        return None

    pos, signs, vel = load_snapshot(str(snap))
    n = len(pos)

    result = detect_interhalos_filaments(pos, signs, box_size,
                                          n_cells=64, halo_mask_radius=10.0,
                                          min_filament_length=8.0)

    n_filaments = result['n_filaments_real']
    length_max = result['length_max_real']
    length_mean = result.get('length_mean_real', 0.0)
    n_halos = result.get('n_halos_plus', 0)

    # Calculate density and purity
    density_mean = 1.0
    fraction_mplus = 0.5

    if n_filaments > 0 and 'filaments' in result:
        cell_size = box_size / 64
        densities = []
        total_plus = 0
        total_minus = 0

        for fil in result['filaments'][:3]:  # Only first 3 for speed
            cells = fil['cells']
            cell_coords = cells * cell_size - box_size / 2
            for cc in cell_coords[:5]:  # Only first 5 cells
                r = 5.0
                dists = np.linalg.norm(pos - cc, axis=1)
                mask_near = dists < r
                n_local = mask_near.sum()
                vol = (4/3) * np.pi * r**3
                rho_mean = n / (box_size ** 3)
                densities.append(n_local / vol / rho_mean)
                total_plus += (signs[mask_near] > 0).sum()
                total_minus += (signs[mask_near] < 0).sum()

        if densities:
            density_mean = float(np.mean(densities))
        total = total_plus + total_minus
        if total > 0:
            fraction_mplus = total_plus / total

    metrics = FilamentMetrics(
        n_filaments=n_filaments, length_max=length_max, length_mean=length_mean,
        density_mean=density_mean, fraction_mplus=fraction_mplus,
        coherent_flow=0.0, n_halos=n_halos
    )

    return {
        'n_filaments': n_filaments,
        'length_max': length_max,
        'length_mean': length_mean,
        'score': filament_score(metrics)
    }


def main():
    BASE = Path('/mnt/T2/janus-sim/output/trichotomie_gpu')

    # Top runs to rescore (from previous results)
    top_runs = [
        # Tour 2 top
        BASE / 'tour2' / 'T2_B1_eta0.99_lam9.6_z3.0',
        BASE / 'tour2' / 'T2_B3_eta0.99_lam9.6_z3.0',
        BASE / 'tour2' / 'T2_B1_eta0.99_lam8.4_z2.64',
        BASE / 'tour2' / 'T2_B2_eta0.99_lam9.6_z3.0',
        # Pass 3.1 top
        BASE / 'tour3' / 'P3.1_eta0.998_lam10.8_z3.00',
        BASE / 'tour3' / 'P3.1_eta0.990_lam8.4_z2.64',
        BASE / 'tour3' / 'P3.1_eta0.990_lam10.8_z2.64',
        BASE / 'tour3' / 'P3.1_eta0.990_lam8.4_z3.36',
        # Pass 3.2 top
        BASE / 'tour3' / 'P3.2_eta0.99800_lam10.80_z3.300',
        BASE / 'tour3' / 'P3.2_eta0.99600_lam11.88_z3.000',
        BASE / 'tour3' / 'P3.2_eta0.99600_lam11.88_z3.300',
        # Pass 3.3 (partial)
        BASE / 'tour3' / 'P3.3_eta0.99600_lam10.80_z3.300',
        BASE / 'tour3' / 'P3.3_eta0.99600_lam11.45_z3.498',
    ]

    results = []

    print("=" * 60)
    print("RESCORING TOP RUNS (periodic correction)")
    print("=" * 60)

    for run_dir in top_runs:
        if not run_dir.exists():
            print(f"⚠️  Not found: {run_dir.name}")
            continue

        print(f"\n{run_dir.name}...", end=" ", flush=True)
        result = analyze_run(run_dir)

        if result:
            result['name'] = run_dir.name
            results.append(result)
            print(f"n_fil={result['n_filaments']}, L_max={result['length_max']:.1f}, "
                  f"score={result['score']:.3f}")
        else:
            print("FAILED")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "=" * 60)
    print("CLASSEMENT APRÈS CORRECTION")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"#{i+1:2d} {r['name']}")
        print(f"    n_fil={r['n_filaments']}, L_max={r['length_max']:.1f} Mpc, "
              f"score={r['score']:.3f}")

    # Save
    output = BASE / 'rescored_top.json'
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == '__main__':
    main()
