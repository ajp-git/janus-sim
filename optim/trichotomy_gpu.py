#!/usr/bin/env python3
"""
Trichotomie Infinie Filaments — Version GPU
Recherche des paramètres optimaux (η, λ, z_act) maximisant filament_score.

Usage:
  python trichotomy_gpu.py --tour 1 --eta 0.80 0.88 0.95 --lambda 5 8 12 --z-act 1.5 2.0 3.0
  python trichotomy_gpu.py --continue-from output/trichotomie_gpu/tour1/
"""

import argparse
import subprocess
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add parent for filament_metrics_v2
sys.path.insert(0, str(Path(__file__).parent))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

BASE_DIR = Path('/mnt/T2/janus-sim')
OUTPUT_BASE = BASE_DIR / 'output' / 'trichotomie_gpu'


@dataclass
class FilamentMetrics:
    """Filament detection results."""
    n_filaments: int
    length_max: float
    length_mean: float
    density_mean: float
    fraction_mplus: float
    coherent_flow: float
    n_halos: int


def filament_score(m: FilamentMetrics) -> float:
    """
    Score ∈ [0, 1]. Critère unique de sélection.
    Poids figés: 0.30/0.25/0.20/0.15/0.10
    """
    s_length = min(m.length_max / 30.0, 1.0)
    s_count = min(m.n_filaments / 3.0, 1.0)
    s_density = min(max(m.density_mean - 1.0, 0) / 0.5, 1.0)
    s_purity = min(m.fraction_mplus / 0.8, 1.0)
    s_flow = min(m.coherent_flow / 0.5, 1.0)

    return (0.30 * s_length +
            0.25 * s_count +
            0.20 * s_density +
            0.15 * s_purity +
            0.10 * s_flow)


def generate_config(name: str, eta: float, lam: float, z_act: float,
                    n_particles: int, n_steps: int, output_dir: str) -> dict:
    """Generate YAML config for a single run."""
    r_smooth = lam * 0.20  # auto r_smooth

    return {
        'simulation': {
            'n_particles': n_particles,
            'n_steps': n_steps,
            'box_size_mpc': 150.0,
            'z_start': 5.0,
            'z_end': 0.0,
            'seed': 42,
            'theta': 0.7,
        },
        'physics': {
            'eta': eta,
            'lambda_base_mpc': lam,
            'lambda_floor': 0.01,
            'r_smooth_mpc': r_smooth,
            'hubble_friction': True,
            'cross_force_z_start': z_act,
            'cross_force_z_width': 0.5,
        },
        'pm_grid': {
            'n_cells': 128,
            'k_min': 2,
        },
        'output': {
            'dir': output_dir,
            'save_snapshots': True,
            'snapshot_redshifts': [2.0, 1.0, 0.5, 0.0],
            'metrics_every_steps': 50,
            'save_velocities': True,
        }
    }


def run_simulation(config_path: Path, timeout_sec: int = 900) -> bool:
    """Run a single simulation via Docker."""
    rel_config = config_path.relative_to(BASE_DIR)
    cmd = [
        'docker', 'compose', 'run', '--rm', 'dev',
        'cargo', 'run', '--release', '--features', 'cuda,cufft',
        '--bin', 'janus_optim', '--',
        '--config', f'/app/{rel_config}'
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def find_snapshot_z05(run_dir: Path, n_steps: int) -> Optional[Path]:
    """Find snapshot closest to z=0.5."""
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


def analyze_run(run_dir: Path, n_steps: int, box_size: float = 150.0) -> Optional[FilamentMetrics]:
    """Analyze run with STRICT filament criteria at z=0.5."""
    snap = find_snapshot_z05(run_dir, n_steps)
    if snap is None:
        return None

    try:
        pos, signs, vel = load_snapshot(str(snap))
    except:
        return None

    n = len(pos)

    # Strict criteria: n_cells<500, AR>3, L>8, d_halo>5
    result = detect_interhalos_filaments(pos, signs, box_size,
                                          n_cells=64, halo_mask_radius=10.0,
                                          min_filament_length=8.0)

    n_filaments = result['n_filaments_real']
    length_max = result['length_max_real']
    length_mean = result.get('length_mean_real', 0.0)
    n_halos = result.get('n_halos_plus', 0)

    # Compute density and purity from filaments
    density_mean = 1.0
    fraction_mplus = 0.5
    coherent_flow = 0.0

    if n_filaments > 0 and 'filaments' in result:
        import numpy as np
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
                rho_local = n_local / vol
                rho_mean = n / (box_size ** 3)
                densities.append(rho_local / rho_mean)
                total_plus += (signs[mask_near] > 0).sum()
                total_minus += (signs[mask_near] < 0).sum()

        if densities:
            density_mean = float(np.mean(densities))
        total = total_plus + total_minus
        if total > 0:
            fraction_mplus = total_plus / total

    return FilamentMetrics(
        n_filaments=n_filaments,
        length_max=length_max,
        length_mean=length_mean,
        density_mean=density_mean,
        fraction_mplus=fraction_mplus,
        coherent_flow=coherent_flow,
        n_halos=n_halos
    )


def generate_zoom_runs(center: dict, tour_number: int) -> List[Tuple[float, float, float]]:
    """
    Generate zoom runs around a basin center.
    Returns list of (eta, lambda, z_act) tuples.
    """
    zoom = 0.20 * (0.60 ** (tour_number - 2))  # Tour 2: 20%, Tour 3: 12%, etc.

    eta_c = center['eta']
    lam_c = center['lambda']
    z_c = center['z_act']

    runs = []

    # 3x3x3 grid around center with zoom factor
    for eta_mult in [1 - zoom, 1.0, 1 + zoom]:
        for lam_mult in [1 - zoom, 1.0, 1 + zoom]:
            for z_mult in [1 - zoom, 1.0, 1 + zoom]:
                eta = max(0.50, min(0.99, eta_c * eta_mult))
                lam = max(2.0, min(20.0, lam_c * lam_mult))
                z_act = max(1.0, min(4.0, z_c * z_mult))
                runs.append((eta, lam, z_act))

    # Remove duplicates
    unique = []
    for r in runs:
        if r not in unique:
            unique.append(r)

    return unique


def run_tour_n(winner: dict, tour_number: int, n_particles: int, n_steps: int,
               output_dir: Path, prev_best_score: float) -> List[dict]:
    """Run Tour N: zoom around a single winner."""
    configs_dir = output_dir / 'configs'
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Generate zoom runs
    zoom_runs = generate_zoom_runs(winner, tour_number)

    print("=" * 70)
    print(f"TOUR {tour_number} — {len(zoom_runs)} RUNS (GPU)")
    print("=" * 70)
    zoom_pct = 0.20 * (0.60 ** (tour_number - 2)) * 100
    print(f"Center: η={winner['eta']}, λ={winner['lambda']}, z_act={winner['z_act']}")
    print(f"Zoom: ±{zoom_pct:.1f}%, N = {n_particles:,}, steps = {n_steps}")
    print()

    results = []

    for run_idx, (eta, lam, z_act) in enumerate(zoom_runs):
        name = f"T{tour_number}_eta{eta:.2f}_lam{lam:.1f}_z{z_act:.1f}"
        run_dir = output_dir / name

        print(f"[{run_idx+1}/{len(zoom_runs)}] {name}")
        print(f"  η={eta:.3f}, λ={lam:.1f}, z_act={z_act:.2f}")

        # Check if already completed
        snap_dir = run_dir / 'snapshots'
        if snap_dir.exists() and len(list(snap_dir.glob('snap_*.bin'))) >= 4:
            print("  Already exists, analyzing...")
        else:
            # Generate config
            config = generate_config(
                name, eta, lam, z_act,
                n_particles, n_steps,
                f'output/trichotomie_gpu/tour{tour_number}/{name}'
            )
            config_path = configs_dir / f'{name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            print("  Running simulation...")
            success = run_simulation(config_path, timeout_sec=900)

            if not success:
                print("  TIMEOUT or FAILED")
                results.append({
                    'name': name,
                    'eta': eta,
                    'lambda': lam,
                    'z_act': z_act,
                    'n_filaments': 0,
                    'length_max': 0,
                    'score': 0,
                    'status': 'failed'
                })
                continue

        # Analyze
        print("  Analyzing (strict criteria)...")
        metrics = analyze_run(run_dir, n_steps)

        if metrics:
            score = filament_score(metrics)
            print(f"  n_fil={metrics.n_filaments}, L_max={metrics.length_max:.1f} Mpc, "
                  f"ρ={metrics.density_mean:.2f}, score={score:.3f}")

            results.append({
                'name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'n_filaments': metrics.n_filaments,
                'length_max': metrics.length_max,
                'length_mean': metrics.length_mean,
                'density_mean': metrics.density_mean,
                'fraction_mplus': metrics.fraction_mplus,
                'n_halos': metrics.n_halos,
                'score': score,
                'status': 'ok'
            })
        else:
            print("  No data")
            results.append({
                'name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'n_filaments': 0,
                'length_max': 0,
                'score': 0,
                'status': 'no_data'
            })

    return results


def run_tour2_multi_basin(top_basins: List[dict], n_particles: int, n_steps: int,
                           output_dir: Path, prev_best_score: float) -> List[dict]:
    """Run Tour 2 exploring multiple basins in parallel."""
    configs_dir = output_dir / 'configs'
    configs_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []  # (name, eta, lam, z_act, basin_idx)

    # Generate runs for each basin
    for basin_idx, basin in enumerate(top_basins):
        zoom_runs = generate_zoom_runs(basin, tour_number=2)
        for eta, lam, z_act in zoom_runs:
            name = f"T2_B{basin_idx+1}_eta{eta:.2f}_lam{lam:.1f}_z{z_act:.1f}"
            all_runs.append((name, eta, lam, z_act, basin_idx))

    print("=" * 70)
    print(f"TOUR 2 — {len(all_runs)} RUNS (GPU) — {len(top_basins)} BASINS")
    print("=" * 70)
    for i, basin in enumerate(top_basins):
        print(f"Basin {i+1}: η={basin['eta']}, λ={basin['lambda']}, z_act={basin['z_act']} (score={basin['score']:.3f})")
    print(f"Zoom: ±20%, N = {n_particles:,}, steps = {n_steps}")
    print()

    results = []

    for run_idx, (name, eta, lam, z_act, basin_idx) in enumerate(all_runs):
        run_dir = output_dir / name

        print(f"[{run_idx+1}/{len(all_runs)}] {name}")
        print(f"  Basin {basin_idx+1}: η={eta:.3f}, λ={lam:.1f}, z_act={z_act:.2f}")

        # Check if already completed
        snap_dir = run_dir / 'snapshots'
        if snap_dir.exists() and len(list(snap_dir.glob('snap_*.bin'))) >= 4:
            print("  Already exists, analyzing...")
        else:
            # Generate config
            config = generate_config(
                name, eta, lam, z_act,
                n_particles, n_steps,
                f'output/trichotomie_gpu/tour2/{name}'
            )
            config_path = configs_dir / f'{name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            print("  Running simulation...")
            success = run_simulation(config_path, timeout_sec=900)

            if not success:
                print("  TIMEOUT or FAILED")
                results.append({
                    'name': name,
                    'eta': eta,
                    'lambda': lam,
                    'z_act': z_act,
                    'basin': basin_idx + 1,
                    'n_filaments': 0,
                    'length_max': 0,
                    'score': 0,
                    'status': 'failed'
                })
                continue

        # Analyze
        print("  Analyzing (strict criteria)...")
        metrics = analyze_run(run_dir, n_steps)

        if metrics:
            score = filament_score(metrics)
            print(f"  n_fil={metrics.n_filaments}, L_max={metrics.length_max:.1f} Mpc, "
                  f"ρ={metrics.density_mean:.2f}, score={score:.3f}")

            results.append({
                'name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'basin': basin_idx + 1,
                'n_filaments': metrics.n_filaments,
                'length_max': metrics.length_max,
                'length_mean': metrics.length_mean,
                'density_mean': metrics.density_mean,
                'fraction_mplus': metrics.fraction_mplus,
                'n_halos': metrics.n_halos,
                'score': score,
                'status': 'ok'
            })
        else:
            print("  No data")
            results.append({
                'name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'basin': basin_idx + 1,
                'n_filaments': 0,
                'length_max': 0,
                'score': 0,
                'status': 'no_data'
            })

    return results


def run_tour1(etas: List[float], lambdas: List[float], z_acts: List[float],
              n_particles: int, n_steps: int, output_dir: Path) -> List[dict]:
    """Run Tour 1: full grid exploration."""
    configs_dir = output_dir / 'configs'
    configs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_runs = len(etas) * len(lambdas) * len(z_acts)
    run_idx = 0

    print("=" * 70)
    print(f"TOUR 1 — {total_runs} RUNS (GPU)")
    print("=" * 70)
    print(f"η ∈ {etas}")
    print(f"λ ∈ {lambdas}")
    print(f"z_act ∈ {z_acts}")
    print(f"N = {n_particles:,}, steps = {n_steps}")
    print()

    for eta in etas:
        for lam in lambdas:
            for z_act in z_acts:
                run_idx += 1
                name = f"T1_eta{eta:.2f}_lam{lam:.0f}_z{z_act:.1f}"
                run_dir = output_dir / name

                print(f"[{run_idx}/{total_runs}] {name}")
                print(f"  η={eta}, λ={lam}, z_act={z_act}")

                # Check if already completed
                snap_dir = run_dir / 'snapshots'
                if snap_dir.exists() and len(list(snap_dir.glob('snap_*.bin'))) >= 4:
                    print("  Already exists, analyzing...")
                else:
                    # Generate config
                    config = generate_config(
                        name, eta, lam, z_act,
                        n_particles, n_steps,
                        f'output/trichotomie_gpu/tour1/{name}'
                    )
                    config_path = configs_dir / f'{name}.yaml'
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)

                    print("  Running simulation...")
                    success = run_simulation(config_path, timeout_sec=900)

                    if not success:
                        print("  TIMEOUT or FAILED")
                        results.append({
                            'name': name,
                            'eta': eta,
                            'lambda': lam,
                            'z_act': z_act,
                            'n_filaments': 0,
                            'length_max': 0,
                            'score': 0,
                            'status': 'failed'
                        })
                        continue

                # Analyze
                print("  Analyzing (strict criteria)...")
                metrics = analyze_run(run_dir, n_steps)

                if metrics:
                    score = filament_score(metrics)
                    print(f"  n_fil={metrics.n_filaments}, L_max={metrics.length_max:.1f} Mpc, "
                          f"ρ={metrics.density_mean:.2f}, score={score:.3f}")

                    results.append({
                        'name': name,
                        'eta': eta,
                        'lambda': lam,
                        'z_act': z_act,
                        'n_filaments': metrics.n_filaments,
                        'length_max': metrics.length_max,
                        'length_mean': metrics.length_mean,
                        'density_mean': metrics.density_mean,
                        'fraction_mplus': metrics.fraction_mplus,
                        'n_halos': metrics.n_halos,
                        'score': score,
                        'status': 'ok'
                    })
                else:
                    print("  No data")
                    results.append({
                        'name': name,
                        'eta': eta,
                        'lambda': lam,
                        'z_act': z_act,
                        'n_filaments': 0,
                        'length_max': 0,
                        'score': 0,
                        'status': 'no_data'
                    })

    return results


def display_tour_results(results: List[dict], tour_num: int, prev_best_score: float = 0.0):
    """Display tour results dashboard."""
    ok_results = [r for r in results if r['status'] == 'ok']
    ok_results.sort(key=lambda r: r['score'], reverse=True)

    with_filaments = [r for r in ok_results if r['n_filaments'] > 0]

    if not ok_results:
        print("\n" + "=" * 70)
        print(f"TOUR {tour_num} — AUCUN RÉSULTAT")
        print("=" * 70)
        return None

    best = ok_results[0]
    progress = ((best['score'] - prev_best_score) / prev_best_score * 100) if prev_best_score > 0 else 0

    print("\n" + "═" * 55)
    print(f"TOUR {tour_num} TERMINÉ")
    print("═" * 55)
    print(f"Gagnant    : η={best['eta']}, λ={best['lambda']} Mpc, z_act={best['z_act']}")
    print(f"Score      : {best['score']:.3f}" + (f" (progression: +{progress:.1f}%)" if prev_best_score > 0 else ""))
    print(f"Filaments  : n={best['n_filaments']}, L_max={best['length_max']:.1f} Mpc, "
          f"ρ_mean={best.get('density_mean', 1.0):.2f}×ρ̄")
    print(f"N runs     : {len(results)} runs, {len(with_filaments)} avec filaments réels")
    print()
    print("Top 3 :")
    for i, r in enumerate(ok_results[:3]):
        basin_info = f" (B{r['basin']})" if 'basin' in r else ""
        print(f"  #{i+1} η={r['eta']:.2f} λ={r['lambda']:.1f} z={r['z_act']:.1f} → score={r['score']:.3f}{basin_info}")
    print()

    # Decision
    if best['score'] > 0.80:
        decision = "OBJECTIF ATTEINT → validation haute résolution"
    elif prev_best_score > 0 and progress < 5:
        decision = "CONVERGENCE → validation haute résolution"
    elif tour_num >= 10:
        decision = "LIMITE SÉCURITÉ → validation haute résolution"
    else:
        decision = f"CONTINUER → Tour {tour_num + 1}"

    print(f"Décision   : {decision}")
    print("═" * 55)

    return best


def main():
    parser = argparse.ArgumentParser(description='Trichotomie GPU - Filament Optimization')
    parser.add_argument('--tour', type=int, required=True, help='Tour number')
    parser.add_argument('--eta', type=float, nargs='+', help='η values')
    parser.add_argument('--lambda', dest='lam', type=float, nargs='+', help='λ values')
    parser.add_argument('--z-act', type=float, nargs='+', help='z_act values')
    parser.add_argument('--n-particles', type=int, default=500000, help='Number of particles')
    parser.add_argument('--steps', type=int, default=600, help='Number of steps')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    output_dir = (BASE_DIR / args.output) if args.output else OUTPUT_BASE / f'tour{args.tour}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JANUS TRICHOTOMIE GPU — FILAMENT OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Output: {output_dir}")
    print()

    if args.tour == 1:
        if not args.eta or not args.lam or not args.z_act:
            print("ERROR: Tour 1 requires --eta, --lambda, and --z-act")
            sys.exit(1)

        results = run_tour1(
            args.eta, args.lam, args.z_act,
            args.n_particles, args.steps,
            output_dir
        )

        # Save results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        best = display_tour_results(results, 1)

        if best:
            print(f"\nResults saved: {results_path}")

    elif args.tour == 2:
        # Tour 2: Zoom on top basins from Tour 1
        tour1_results = None
        tour1_dir = OUTPUT_BASE / 'tour1'
        tour1_json = tour1_dir / 'results.json'

        if tour1_json.exists():
            with open(tour1_json) as f:
                tour1_results = json.load(f)
        else:
            print(f"ERROR: Tour 1 results not found at {tour1_json}")
            sys.exit(1)

        # Find top basins (score within 1% of best)
        ok_results = [r for r in tour1_results if r['status'] == 'ok' and r['n_filaments'] > 0]
        ok_results.sort(key=lambda r: r['score'], reverse=True)
        best_score = ok_results[0]['score']
        top_basins = [r for r in ok_results if r['score'] >= best_score * 0.99]

        print(f"Found {len(top_basins)} basins with score >= {best_score * 0.99:.3f}")

        results = run_tour2_multi_basin(
            top_basins,
            args.n_particles, args.steps,
            output_dir, prev_best_score=best_score
        )

        # Save results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        best = display_tour_results(results, 2, prev_best_score=best_score)
        if best:
            print(f"\nResults saved: {results_path}")

    elif args.tour >= 3:
        # Tour 3+: Zoom around winner from previous tour
        prev_tour = args.tour - 1
        prev_dir = OUTPUT_BASE / f'tour{prev_tour}'
        prev_json = prev_dir / 'results.json'

        if not prev_json.exists():
            print(f"ERROR: Tour {prev_tour} results not found at {prev_json}")
            sys.exit(1)

        with open(prev_json) as f:
            prev_results = json.load(f)

        # Find winner
        ok_results = [r for r in prev_results if r['status'] == 'ok' and r['n_filaments'] > 0]
        ok_results.sort(key=lambda r: r['score'], reverse=True)
        winner = ok_results[0]
        prev_best_score = winner['score']

        print(f"Tour {prev_tour} winner: η={winner['eta']}, λ={winner['lambda']}, z_act={winner['z_act']}, score={winner['score']:.3f}")

        results = run_tour_n(
            winner,
            tour_number=args.tour,
            n_particles=args.n_particles,
            n_steps=args.steps,
            output_dir=output_dir,
            prev_best_score=prev_best_score
        )

        # Save results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        best = display_tour_results(results, args.tour, prev_best_score=prev_best_score)
        if best:
            print(f"\nResults saved: {results_path}")

    else:
        print(f"Tour {args.tour} not yet implemented")

    print(f"\nFinished: {datetime.now()}")


if __name__ == '__main__':
    main()
