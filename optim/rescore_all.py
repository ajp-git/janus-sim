#!/usr/bin/env python3
"""
Rescore all trichotomy runs with periodic-corrected filament metrics.
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

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


def find_snapshot_z05(run_dir: Path, n_steps: int = 600) -> Optional[Path]:
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None
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


def analyze_run(run_dir: Path, box_size: float = 150.0) -> Optional[dict]:
    import numpy as np

    snap = find_snapshot_z05(run_dir)
    if snap is None:
        return None

    try:
        pos, signs, vel = load_snapshot(str(snap))
    except:
        return None

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

        for fil in result['filaments']:
            cells = fil['cells']
            cell_coords = cells * cell_size - box_size / 2
            for cc in cell_coords[:10]:
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

    score = filament_score(metrics)

    return {
        'n_filaments': n_filaments,
        'length_max': length_max,
        'length_mean': length_mean,
        'density_mean': density_mean,
        'fraction_mplus': fraction_mplus,
        'n_halos': n_halos,
        'score': score
    }


def main():
    BASE = Path('/mnt/T2/janus-sim/output/trichotomie_gpu')

    # Tours to rescore
    tours = [
        ('tour1', BASE / 'tour1'),
        ('tour2', BASE / 'tour2'),
        ('tour3', BASE / 'tour3'),
    ]

    all_results = []

    for tour_name, tour_dir in tours:
        if not tour_dir.exists():
            print(f"⚠️  {tour_name} not found")
            continue

        print(f"\n{'='*60}")
        print(f"RESCORING {tour_name.upper()}")
        print(f"{'='*60}")

        run_dirs = sorted([d for d in tour_dir.iterdir() if d.is_dir() and 'config' not in d.name])

        for run_dir in run_dirs:
            if not (run_dir / 'snapshots').exists():
                continue

            result = analyze_run(run_dir)
            if result is None:
                continue

            # Extract params from name
            name = run_dir.name
            result['name'] = name
            result['tour'] = tour_name

            all_results.append(result)

            print(f"{name}: n_fil={result['n_filaments']}, "
                  f"L_max={result['length_max']:.1f}, score={result['score']:.3f}")

    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 10 APRÈS CORRECTION PÉRIODIQUE")
    print(f"{'='*60}")
    for i, r in enumerate(all_results[:10]):
        print(f"#{i+1:2d} {r['tour']}/{r['name']}")
        print(f"    n_fil={r['n_filaments']}, L_max={r['length_max']:.1f} Mpc, "
              f"score={r['score']:.3f}")

    # Save results
    output_path = BASE / 'rescored_all.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
