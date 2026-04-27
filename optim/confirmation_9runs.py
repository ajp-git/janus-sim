#!/usr/bin/env python3
"""
9 runs de confirmation autour du gagnant T2 (η=0.99, λ=9.6, z=3.0)
"""
import subprocess
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

BASE_DIR = Path('/mnt/T2/janus-sim')
OUTPUT_BASE = BASE_DIR / 'output' / 'confirmation_9runs'


@dataclass
class FilamentMetrics:
    n_filaments: int
    length_max: float
    length_mean: float
    density_mean: float
    fraction_mplus: float
    n_halos: int


def filament_score(m: FilamentMetrics) -> float:
    s_length = min(m.length_max / 30.0, 1.0)
    s_count = min(m.n_filaments / 3.0, 1.0)
    s_density = min(max(m.density_mean - 1.0, 0) / 0.5, 1.0)
    s_purity = min(m.fraction_mplus / 0.8, 1.0)
    return (0.30 * s_length + 0.25 * s_count + 0.20 * s_density + 0.15 * s_purity + 0.10 * 0.0)


def generate_config(name: str, eta: float, lam: float, z_act: float,
                    n_particles: int, n_steps: int, output_dir: str) -> dict:
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
            'r_smooth_mpc': lam * 0.20,
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


def find_snapshot_z05(run_dir: Path) -> Path:
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None
    steps = [int(s.stem.replace('snap_', '')) for s in snaps]
    max_step = max(steps)
    target = int(0.9 * max_step)
    return min(snaps, key=lambda s: abs(int(s.stem.replace('snap_', '')) - target))


def analyze_run(run_dir: Path, box_size: float = 150.0):
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

    density_mean = 1.0
    fraction_mplus = 0.5

    if n_filaments > 0 and 'filaments' in result:
        cell_size = box_size / 64
        densities = []
        total_plus = 0
        total_minus = 0

        for fil in result['filaments'][:3]:
            cells = fil['cells']
            cell_coords = cells * cell_size - box_size / 2
            for cc in cell_coords[:5]:
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
        density_mean=density_mean, fraction_mplus=fraction_mplus, n_halos=n_halos
    )

    return {
        'n_filaments': n_filaments,
        'length_max': length_max,
        'length_mean': length_mean,
        'density_mean': density_mean,
        'fraction_mplus': fraction_mplus,
        'n_halos': n_halos,
        'score': filament_score(metrics)
    }


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    configs_dir = OUTPUT_BASE / 'configs'
    configs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CONFIRMATION 9 RUNS")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print()

    # Grid
    etas = [0.97, 0.99, 0.995]
    lambdas = [8.4, 9.6, 10.8]
    z_act = 3.0  # Fixed

    n_particles = 500000
    n_steps = 600

    results = []
    run_idx = 0
    total = len(etas) * len(lambdas)

    print(f"η     ∈ {etas}")
    print(f"λ     ∈ {lambdas} Mpc")
    print(f"z_act = {z_act} (fixé)")
    print(f"N = {n_particles:,}, steps = {n_steps}")
    print()

    for eta in etas:
        for lam in lambdas:
            run_idx += 1
            name = f"conf_eta{eta:.3f}_lam{lam:.1f}"
            run_dir = OUTPUT_BASE / name

            print(f"[{run_idx}/{total}] η={eta:.3f} λ={lam:.1f}", flush=True)

            # Check if already done
            snap_dir = run_dir / 'snapshots'
            if snap_dir.exists() and len(list(snap_dir.glob('snap_*.bin'))) >= 4:
                print("  Déjà fait, analyse...", flush=True)
            else:
                # Generate config
                config = generate_config(
                    name, eta, lam, z_act,
                    n_particles, n_steps,
                    str(run_dir.relative_to(BASE_DIR))
                )
                config_path = configs_dir / f'{name}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                print("  Simulation...", flush=True)
                success = run_simulation(config_path, timeout_sec=900)
                if not success:
                    print("  ÉCHEC", flush=True)
                    results.append({
                        'name': name, 'eta': eta, 'lambda': lam, 'z_act': z_act,
                        'score': 0, 'status': 'failed'
                    })
                    continue

            print("  Analyse...", flush=True)
            metrics = analyze_run(run_dir)

            if metrics:
                print(f"  n_fil={metrics['n_filaments']}, L_max={metrics['length_max']:.1f}, "
                      f"score={metrics['score']:.3f}", flush=True)
                results.append({
                    'name': name, 'eta': eta, 'lambda': lam, 'z_act': z_act,
                    **metrics, 'status': 'ok'
                })
            else:
                print("  Pas de données", flush=True)
                results.append({
                    'name': name, 'eta': eta, 'lambda': lam, 'z_act': z_act,
                    'score': 0, 'status': 'no_data'
                })

    # Sort by score
    ok_results = [r for r in results if r['status'] == 'ok']
    ok_results.sort(key=lambda x: x['score'], reverse=True)

    print()
    print("=" * 60)
    print("RÉSULTATS CONFIRMATION")
    print("=" * 60)

    # Create grid display
    print("\nGrille des scores:")
    print("        λ=8.4   λ=9.6   λ=10.8")
    for eta in etas:
        row = f"η={eta:.3f}"
        for lam in lambdas:
            r = next((x for x in results if x['eta'] == eta and x['lambda'] == lam), None)
            if r and r['status'] == 'ok':
                row += f"  {r['score']:.3f}"
            else:
                row += "    -  "
        print(row)

    print("\nClassement:")
    for i, r in enumerate(ok_results[:5]):
        marker = " ← GAGNANT" if i == 0 else ""
        print(f"#{i+1} η={r['eta']:.3f} λ={r['lambda']:.1f} → "
              f"n_fil={r['n_filaments']}, L_max={r['length_max']:.1f}, "
              f"score={r['score']:.3f}{marker}")

    # Check if (0.99, 9.6) is the winner
    winner = ok_results[0] if ok_results else None
    if winner and winner['eta'] == 0.99 and winner['lambda'] == 9.6:
        print("\n✓ CONFIRMÉ: (η=0.99, λ=9.6, z=3.0) est bien l'optimum!")
    else:
        print(f"\n⚠ Nouveau gagnant: η={winner['eta']:.3f}, λ={winner['lambda']:.1f}")

    # Save results
    with open(OUTPUT_BASE / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinished: {datetime.now()}")
    print(f"Saved to {OUTPUT_BASE / 'results.json'}")


if __name__ == '__main__':
    main()
