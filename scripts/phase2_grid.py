#!/usr/bin/env python3
"""
Phase 2: 9 runs (3×3 grid)
- η ∈ [0.92, 0.95, 0.98]
- λ ∈ [5, 8.5, 12]
- z_act = 2.0 (fixed)
- N = 1M, steps = 1200
"""

import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

BASE_DIR = Path('/mnt/T2/janus-sim')
OUTPUT_DIR = BASE_DIR / 'output' / 'trichotomie' / 'phase2_v2'
CONFIG_DIR = OUTPUT_DIR / 'configs'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Phase 2 grid
ETAS = [0.92, 0.95, 0.98]
LAMBDAS = [5.0, 8.5, 12.0]
Z_ACT = 2.0  # Fixed

# Simulation params
N_PARTICLES = 1_000_000
N_STEPS = 1200
BOX_SIZE = 150.0
TIMEOUT_SEC = 1800  # 30 min per run


def generate_configs():
    """Generate 9 Phase 2 configs."""
    configs = []

    for eta in ETAS:
        for lam in LAMBDAS:
            name = f"P2_eta{eta:.2f}_lam{lam:.1f}"

            config = {
                'simulation': {
                    'n_particles': N_PARTICLES,
                    'n_steps': N_STEPS,
                    'box_size_mpc': BOX_SIZE,
                    'seed': 42,
                    'z_start': 5.0,
                    'z_end': 0.0,
                    'theta': 0.7,
                },
                'physics': {
                    'eta': eta,
                    'lambda_base_mpc': lam,
                    'lambda_floor': 0.01,
                    'r_smooth_mpc': 1.6,
                    'hubble_friction': True,
                    'cross_force_z_start': Z_ACT,
                    'cross_force_z_width': 0.5,
                },
                'pm_grid': {
                    'n_cells': 256,
                    'k_min': 2,
                },
                'output': {
                    'dir': f'output/trichotomie/phase2_v2/{name}',
                    'save_snapshots': True,
                    'snapshot_redshifts': [1.0, 0.5, 0.0],
                    'metrics_every_steps': 50,
                }
            }

            config_path = CONFIG_DIR / f'{name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            configs.append({
                'name': name,
                'eta': eta,
                'lambda': lam,
                'config_path': config_path,
            })

    return configs


def run_simulation(config):
    """Run a single simulation."""
    cmd = [
        'docker', 'compose', 'run', '--rm', 'dev',
        'cargo', 'run', '--release', '--features', 'cuda,cufft',
        '--bin', 'janus_optim', '--',
        '--config', f'/app/{config["config_path"].relative_to(BASE_DIR)}'
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {TIMEOUT_SEC}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def find_snapshot_z05(run_dir):
    """Find snapshot closest to z=0.5."""
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None

    # z=0.5 is 90% of the way from z=5 to z=0
    # With 1200 steps: target_step = 0.9 * 1200 = 1080
    target_step = int(0.9 * N_STEPS)

    best_snap = None
    best_diff = float('inf')
    for snap in snaps:
        step = int(snap.stem.replace('snap_', ''))
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_snap = snap
    return best_snap


def analyze_run(run_dir):
    """Analyze with STRICT criteria at z=0.5."""
    snap = find_snapshot_z05(run_dir)
    if snap is None:
        return None

    try:
        pos, signs, vel = load_snapshot(str(snap))
    except:
        return None

    result = detect_interhalos_filaments(pos, signs, BOX_SIZE,
                                          n_cells=64, halo_mask_radius=10.0,
                                          min_filament_length=8.0)

    return {
        'n_filaments': result['n_filaments_real'],
        'length_max': result['length_max_real'],
        'n_halos': result.get('n_halos_plus', 0),
        'filaments': result.get('filaments', []),
    }


def compute_score(metrics):
    """Compute filament score."""
    if metrics is None:
        return 0.0
    s_length = min(metrics['length_max'] / 30.0, 1.0)
    s_count = min(metrics['n_filaments'] / 5.0, 1.0)
    return 0.40 * s_length + 0.35 * s_count + 0.25 * 0.5


def main():
    print("=" * 70)
    print("PHASE 2: 9 RUNS (3×3 GRID)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"η ∈ {ETAS}")
    print(f"λ ∈ {LAMBDAS}")
    print(f"z_act = {Z_ACT} (fixed)")
    print(f"N = {N_PARTICLES:,}, steps = {N_STEPS}")
    print()

    # Generate configs
    configs = generate_configs()
    print(f"Generated {len(configs)} configs\n")

    results = []

    for i, config in enumerate(configs):
        name = config['name']
        print(f"[{i+1}/9] {name}")
        print(f"  η={config['eta']}, λ={config['lambda']}")

        run_dir = OUTPUT_DIR / name
        snap_dir = run_dir / 'snapshots'

        # Check if already exists
        if snap_dir.exists() and len(list(snap_dir.glob('snap_*.bin'))) >= 3:
            print("  Already exists, analyzing...")
        else:
            print("  Running simulation...")
            success = run_simulation(config)
            if not success:
                print("  FAILED")
                results.append({
                    'name': name,
                    'eta': config['eta'],
                    'lambda': config['lambda'],
                    'n_filaments': 0,
                    'length_max': 0,
                    'score': 0,
                    'status': 'failed'
                })
                continue

        print("  Analyzing (strict criteria)...")
        metrics = analyze_run(run_dir)

        if metrics:
            score = compute_score(metrics)
            print(f"  n_fil={metrics['n_filaments']}, L_max={metrics['length_max']:.1f} Mpc, score={score:.3f}")

            # Show filament details
            if metrics['n_filaments'] > 0:
                for j, fil in enumerate(metrics['filaments'][:3]):
                    print(f"    fil#{j+1}: L={fil['length']:.1f}, cells={fil['n_cells']}, AR={fil['aspect_ratio']:.1f}")

            results.append({
                'name': name,
                'eta': config['eta'],
                'lambda': config['lambda'],
                'n_filaments': metrics['n_filaments'],
                'length_max': metrics['length_max'],
                'score': score,
                'status': 'ok'
            })
        else:
            print("  No data")
            results.append({
                'name': name,
                'eta': config['eta'],
                'lambda': config['lambda'],
                'n_filaments': 0,
                'length_max': 0,
                'score': 0,
                'status': 'no_data'
            })

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS")
    print("=" * 70)

    ok_results = [r for r in results if r['status'] == 'ok']
    ok_results.sort(key=lambda r: r['score'], reverse=True)

    print(f"\nSuccessful: {len(ok_results)}/9\n")

    print(f"{'Name':<25} {'η':>6} {'λ':>6} {'n_fil':>6} {'L_max':>8} {'score':>7}")
    print("-" * 65)
    for r in ok_results:
        print(f"{r['name']:<25} {r['eta']:>6.2f} {r['lambda']:>6.1f} {r['n_filaments']:>6} {r['length_max']:>8.1f} {r['score']:>7.3f}")

    if len(ok_results) >= 3:
        print("\n" + "=" * 70)
        print("TOP 3 FOR PHASE 3")
        print("=" * 70)
        for i, r in enumerate(ok_results[:3]):
            print(f"\n#{i+1}: {r['name']}")
            print(f"    η={r['eta']}, λ={r['lambda']}, z_act={Z_ACT}")
            print(f"    n_filaments={r['n_filaments']}, L_max={r['length_max']:.1f} Mpc")
            print(f"    score={r['score']:.3f}")

    print(f"\nFinished: {datetime.now()}")


if __name__ == '__main__':
    main()
