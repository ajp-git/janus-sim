#!/usr/bin/env python3
"""
Phase 2: Zoom on top 3 from Phase 1.
Each winner gets 3 variants (±20% on λ).
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

OUTPUT_DIR = Path('/mnt/T2/janus-sim/output/trichotomie')
PHASE2_DIR = OUTPUT_DIR / 'phase2'
PHASE2_DIR.mkdir(exist_ok=True)

# Top 3 from Phase 1 (strict criteria)
TOP3 = [
    # #1: η=0.95, λ=12, z_act=2.0
    {'eta': 0.95, 'lambda': 12, 'z_act': 2.0, 'lambda_range': [9.6, 12, 14.4]},
    # #2: η=0.88, λ=5, z_act=2.0
    {'eta': 0.88, 'lambda': 5, 'z_act': 2.0, 'lambda_range': [4, 5, 6]},
    # #3: η=0.95, λ=5, z_act=2.0
    {'eta': 0.95, 'lambda': 5, 'z_act': 2.0, 'lambda_range': [4, 5, 6]},
]


def build_config(name, eta, lam, z_act, n_particles=500000, steps=600, box=150.0):
    """Build Rust config file."""
    return f'''// Phase 2: {name}
SimulationConfig {{
    n_particles: {n_particles},
    box_size: {box:.1f},
    eta: {eta:.4f},
    dt: 0.005,
    steps: {steps},
    theta: 0.7,
    softening: 0.1,
    snapshot_interval: 100,
    output_dir: "/app/output/trichotomie/phase2/{name}",

    // Yukawa screening
    yukawa_screening: true,
    lambda_base_mpc: {lam:.1f},

    // Sigmoid cross-force (Z1)
    cross_force_z_start: {z_act:.1f},
    cross_force_z_width: 0.5,
}}
'''


def run_simulation(name, config_content, timeout=600):
    """Run a simulation with the given config."""
    config_path = Path('/tmp') / f'{name}.rs'
    with open(config_path, 'w') as f:
        f.write(config_content)

    run_dir = PHASE2_DIR / name
    run_dir.mkdir(exist_ok=True)

    cmd = [
        'timeout', str(timeout),
        'docker', 'compose', 'run', '--rm',
        '-v', f'{config_path}:/app/config.rs:ro',
        'dev', 'cargo', 'run', '--release', '--features', 'cuda',
        '--bin', 'janus_trichotomy', '--',
        '--config', '/app/config.rs'
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd='/mnt/T2/janus-sim',
            capture_output=True,
            text=True,
            timeout=timeout + 60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def analyze_run(run_dir, box_size=150.0):
    """Analyze run at z~0.5."""
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None

    # z=0.5 is ~90% of the way from z=5 to z=0
    target_step = int(0.9 * 600)
    best_snap = None
    best_diff = float('inf')
    for snap in snaps:
        step = int(snap.stem.replace('snap_', ''))
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_snap = snap

    if best_snap is None:
        return None

    try:
        pos, signs, vel = load_snapshot(str(best_snap))
    except:
        return None

    result = detect_interhalos_filaments(pos, signs, box_size,
                                          n_cells=64, halo_mask_radius=10.0,
                                          min_filament_length=8.0)

    return {
        'n_filaments': result['n_filaments_real'],
        'length_max': result['length_max_real'],
        'n_halos': result.get('n_halos_plus', 0),
    }


def main():
    print("="*70)
    print("PHASE 2: ZOOM ON TOP 3")
    print("="*70)
    print(f"Started: {datetime.now()}")
    print()

    results = []
    run_count = 0

    for i, winner in enumerate(TOP3):
        print(f"\nTop #{i+1}: η={winner['eta']}, λ={winner['lambda']}, z_act={winner['z_act']}")
        print(f"  Zoom λ range: {winner['lambda_range']}")

        for lam in winner['lambda_range']:
            run_count += 1
            name = f"T2_top{i+1}_eta{winner['eta']:.2f}_lam{lam:.1f}_z{winner['z_act']:.1f}"

            print(f"\n[{run_count}/9] {name}")

            # Check if already exists
            run_dir = PHASE2_DIR / name / 'snapshots'
            if run_dir.exists() and len(list(run_dir.glob('snap_*.bin'))) > 5:
                print("  Already exists, analyzing...")
            else:
                print("  Running simulation...")
                config = build_config(name, winner['eta'], lam, winner['z_act'])
                success = run_simulation(name, config, timeout=600)

                if not success:
                    print("  TIMEOUT or FAILED")
                    results.append({
                        'name': name,
                        'eta': winner['eta'],
                        'lambda': lam,
                        'z_act': winner['z_act'],
                        'n_filaments': 0,
                        'length_max': 0,
                        'status': 'failed'
                    })
                    continue

            print("  Analyzing...")
            run_dir = PHASE2_DIR / name
            metrics = analyze_run(run_dir)

            if metrics:
                print(f"  n_fil={metrics['n_filaments']}, L_max={metrics['length_max']:.1f}")
                results.append({
                    'name': name,
                    'eta': winner['eta'],
                    'lambda': lam,
                    'z_act': winner['z_act'],
                    'n_filaments': metrics['n_filaments'],
                    'length_max': metrics['length_max'],
                    'status': 'ok'
                })
            else:
                print("  Analysis failed")
                results.append({
                    'name': name,
                    'eta': winner['eta'],
                    'lambda': lam,
                    'z_act': winner['z_act'],
                    'n_filaments': 0,
                    'length_max': 0,
                    'status': 'no_data'
                })

    print("\n" + "="*70)
    print("PHASE 2 RESULTS")
    print("="*70)

    # Sort by score
    ok_results = [r for r in results if r['status'] == 'ok']
    ok_results.sort(key=lambda r: (r['n_filaments'], r['length_max']), reverse=True)

    print(f"\nSuccessful runs: {len(ok_results)}/9\n")

    for r in ok_results:
        score = 0.30 * min(r['length_max']/30, 1) + 0.25 * min(r['n_filaments']/3, 1) + 0.45*0.5
        print(f"  {r['name']}: n_fil={r['n_filaments']}, L={r['length_max']:.1f} Mpc, score={score:.3f}")

    if len(ok_results) >= 3:
        print("\n" + "="*70)
        print("TOP 3 FOR PHASE 3")
        print("="*70)
        for i, r in enumerate(ok_results[:3]):
            print(f"  #{i+1}: {r['name']}")
            print(f"       η={r['eta']}, λ={r['lambda']}, z_act={r['z_act']}")
            print(f"       n_filaments={r['n_filaments']}, length_max={r['length_max']:.1f} Mpc")


if __name__ == '__main__':
    main()
