#!/usr/bin/env python3
"""
Tour 3 — Trichotomie infinie avec passes de 27 runs
Chaque passe est une grille 3×3×3, zoom décroissant jusqu'à convergence.
"""

import subprocess
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

BASE_DIR = Path('/mnt/T2/janus-sim')
OUTPUT_BASE = BASE_DIR / 'output' / 'trichotomie_gpu' / 'tour3'


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


def generate_config(name: str, eta: float, lam: float, z_act: float,
                    n_particles: int, n_steps: int, output_dir: str) -> dict:
    r_smooth = lam * 0.20
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
    rel_config = config_path.relative_to(BASE_DIR)
    cmd = [
        'docker', 'compose', 'run', '--rm', 'dev',
        'cargo', 'run', '--release', '--features', 'cuda,cufft',
        '--bin', 'janus_optim', '--',
        '--config', f'/app/{rel_config}'
    ]
    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True,
                                text=True, timeout=timeout_sec)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def find_snapshot_z05(run_dir: Path, n_steps: int) -> Optional[Path]:
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


def analyze_run(run_dir: Path, n_steps: int, box_size: float = 150.0) -> Optional[FilamentMetrics]:
    snap = find_snapshot_z05(run_dir, n_steps)
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
        n_filaments=n_filaments, length_max=length_max, length_mean=length_mean,
        density_mean=density_mean, fraction_mplus=fraction_mplus,
        coherent_flow=coherent_flow, n_halos=n_halos
    )


def clip_eta(eta: float) -> float:
    return max(0.90, min(0.999, eta))


def run_pass(pass_num: float, etas: List[float], lambdas: List[float],
             z_acts: List[float], n_particles: int, n_steps: int,
             output_dir: Path) -> List[dict]:
    """Run a single pass of 27 runs."""
    configs_dir = output_dir / 'configs'
    configs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_runs = len(etas) * len(lambdas) * len(z_acts)
    run_idx = 0

    print("=" * 60)
    print(f"PASSE {pass_num} — {total_runs} RUNS")
    print("=" * 60)
    print(f"η     ∈ {[round(e, 4) for e in etas]}")
    print(f"λ     ∈ {[round(l, 2) for l in lambdas]} Mpc")
    print(f"z_act ∈ {[round(z, 3) for z in z_acts]}")
    print(f"N = {n_particles:,}, steps = {n_steps}")
    print()

    for eta in etas:
        for lam in lambdas:
            for z_act in z_acts:
                run_idx += 1
                name = f"P{pass_num}_eta{eta:.3f}_lam{lam:.1f}_z{z_act:.2f}"
                run_dir = output_dir / name

                print(f"[{run_idx}/{total_runs}] {name}")

                snap_dir = run_dir / 'snapshots'
                if snap_dir.exists() and len(list(snap_dir.glob('snap_*.bin'))) >= 4:
                    print("  Existe, analyse...")
                else:
                    config = generate_config(
                        name, eta, lam, z_act,
                        n_particles, n_steps,
                        str(run_dir.relative_to(BASE_DIR))
                    )
                    config_path = configs_dir / f'{name}.yaml'
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)

                    print("  Simulation...")
                    success = run_simulation(config_path, timeout_sec=900)
                    if not success:
                        print("  ÉCHEC")
                        results.append({
                            'name': name, 'eta': eta, 'lambda': lam, 'z_act': z_act,
                            'n_filaments': 0, 'length_max': 0, 'score': 0, 'status': 'failed'
                        })
                        continue

                print("  Analyse...")
                metrics = analyze_run(run_dir, n_steps)

                if metrics:
                    score = filament_score(metrics)
                    print(f"  n_fil={metrics.n_filaments}, L_max={metrics.length_max:.1f} Mpc, "
                          f"score={score:.3f}")
                    results.append({
                        'name': name, 'eta': eta, 'lambda': lam, 'z_act': z_act,
                        'n_filaments': metrics.n_filaments, 'length_max': metrics.length_max,
                        'length_mean': metrics.length_mean, 'density_mean': metrics.density_mean,
                        'fraction_mplus': metrics.fraction_mplus, 'n_halos': metrics.n_halos,
                        'score': score, 'status': 'ok'
                    })
                else:
                    print("  Pas de données")
                    results.append({
                        'name': name, 'eta': eta, 'lambda': lam, 'z_act': z_act,
                        'n_filaments': 0, 'length_max': 0, 'score': 0, 'status': 'no_data'
                    })

    return results


def display_pass_results(results: List[dict], pass_num: float, prev_score: float = 0.0):
    """Display pass results and return winner."""
    ok_results = [r for r in results if r['status'] == 'ok' and r.get('n_filaments', 0) > 0]
    ok_results.sort(key=lambda r: r['score'], reverse=True)

    if not ok_results:
        print(f"\n{'='*60}")
        print(f"PASSE {pass_num} — AUCUN FILAMENT")
        print(f"{'='*60}")
        return None

    best = ok_results[0]
    progress = ((best['score'] - prev_score) / prev_score * 100) if prev_score > 0 else 0

    print()
    print("═" * 60)
    print(f"PASSE {pass_num} | Gagnant : η={best['eta']:.4f}, λ={best['lambda']:.2f}, z={best['z_act']:.3f}")
    print(f"Score     : {best['score']:.3f}" + (f" (progression: {progress:+.1f}%)" if prev_score > 0 else ""))
    print(f"Filaments : n={best['n_filaments']}, L_max={best['length_max']:.1f} Mpc")

    print("\nTop 5 :")
    for i, r in enumerate(ok_results[:5]):
        print(f"  #{i+1} η={r['eta']:.4f} λ={r['lambda']:.2f} z={r['z_act']:.3f} → score={r['score']:.3f}")

    return best, progress


def check_convergence(best: dict, prev_best: dict, pass_num: float, progress: float) -> Tuple[bool, str]:
    """Check if we should stop."""
    if progress < 5 and pass_num > 3.1:
        return True, f"CONVERGENCE (progression {progress:.1f}% < 5%)"

    if prev_best:
        d_eta = abs(best['eta'] - prev_best['eta'])
        d_lam = abs(best['lambda'] - prev_best['lambda'])
        d_z = abs(best['z_act'] - prev_best['z_act'])

        if d_eta < 0.002 and d_lam < 0.3 and d_z < 0.05:
            return True, f"CONVERGENCE (intervalles minimaux atteints)"

    if pass_num >= 3.8:
        return True, "LIMITE (8 passes max)"

    return False, f"CONTINUER → Passe {pass_num + 0.1:.1f}"


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TOUR 3 — TRICHOTOMIE INFINIE (passes de 27 runs)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Centre initial: η=0.99, λ=9.6, z=3.0 (gagnant Tour 2)")
    print()

    n_particles = 500000
    n_steps = 600

    # Initial center from Tour 2 winner
    eta_c, lam_c, z_c = 0.99, 9.6, 3.0
    prev_score = 0.700
    prev_best = None

    all_results = {}

    # Pass 3.1: Large grid
    pass_num = 3.1
    etas = [clip_eta(0.96), clip_eta(0.99), clip_eta(0.998)]
    lambdas = [8.4, 9.6, 10.8]
    z_acts = [2.64, 3.0, 3.36]

    while True:
        results = run_pass(pass_num, etas, lambdas, z_acts, n_particles, n_steps, OUTPUT_BASE)

        # Save results
        with open(OUTPUT_BASE / f'results_pass{pass_num:.1f}.json', 'w') as f:
            json.dump(results, f, indent=2)
        all_results[pass_num] = results

        result = display_pass_results(results, pass_num, prev_score)
        if result is None:
            print("ÉCHEC — pas de filaments, arrêt")
            break

        best, progress = result

        # Check convergence
        converged, decision = check_convergence(best, prev_best, pass_num, progress)
        print(f"Décision  : {decision}")
        print("═" * 60)

        if converged:
            print(f"\n{'='*70}")
            print("CONVERGENCE ATTEINTE")
            print(f"{'='*70}")
            print(f"Paramètres optimaux : η={best['eta']:.4f}, λ={best['lambda']:.2f} Mpc, z_act={best['z_act']:.3f}")
            print(f"Score final         : {best['score']:.3f}")
            print(f"Filaments           : n={best['n_filaments']}, L_max={best['length_max']:.1f} Mpc")
            break

        # Setup next pass
        prev_score = best['score']
        prev_best = best
        eta_c = best['eta']
        lam_c = best['lambda']
        z_c = best['z_act']

        # Compute zoom factor
        if pass_num == 3.1:
            zoom = 0.10  # ±10% for pass 3.2
        else:
            zoom = 0.06 * (0.6 ** (pass_num - 3.2))  # decreasing

        pass_num = round(pass_num + 0.1, 1)

        # Generate new grid
        etas = [clip_eta(eta_c * (1 - zoom)), clip_eta(eta_c), clip_eta(eta_c * (1 + zoom))]
        lambdas = [lam_c * (1 - zoom), lam_c, lam_c * (1 + zoom)]
        z_acts = [z_c * (1 - zoom), z_c, z_c * (1 + zoom)]

        # Remove duplicates
        etas = sorted(set(etas))
        lambdas = sorted(set(lambdas))
        z_acts = sorted(set(z_acts))

        print(f"\nProchain zoom: ±{zoom*100:.1f}%")

    print(f"\nFinished: {datetime.now()}")

    # Save all results
    with open(OUTPUT_BASE / 'all_results.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)


if __name__ == '__main__':
    main()
