#!/usr/bin/env python3
"""Generate 10 grid configs for eta x R_smooth exploration."""

import yaml
from pathlib import Path

# Grid: 5 eta x 2 R_smooth, lambda fixed at 40
ETAS = [0.76, 0.80, 0.84, 0.88, 0.92]
RSMOOTHS = [6.5, 8.0]
LAMBDA_BASE = 40.0

# Fixed parameters
N_PARTICLES = 500000
N_STEPS = 1200
BOX_SIZE = 150.0
Z_START = 5.0
Z_END = 0.0
SEED = 42

def main():
    output_dir = Path(__file__).parent

    run_num = 1
    configs = []

    for eta in ETAS:
        for r_smooth in RSMOOTHS:
            config = {
                'output': {
                    'dir': f'output/grid_10/run_{run_num:02d}',
                    'metrics_every_steps': 10,
                    'save_snapshots': True,
                    'snapshot_redshifts': [3.0, 2.0, 1.0, 0.5, 0.0]
                },
                'physics': {
                    'eta': eta,
                    'hubble_friction': True,
                    'lambda_base_mpc': LAMBDA_BASE,
                    'lambda_floor': 0.01,
                    'r_smooth_mpc': r_smooth
                },
                'pm_grid': {
                    'k_min': 2,
                    'n_cells': 256
                },
                'simulation': {
                    'box_size_mpc': BOX_SIZE,
                    'n_particles': N_PARTICLES,
                    'n_steps': N_STEPS,
                    'seed': SEED,
                    'theta': 0.7,
                    'z_end': Z_END,
                    'z_start': Z_START
                }
            }

            config_path = output_dir / f'config_run_{run_num:02d}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            configs.append({
                'run': run_num,
                'eta': eta,
                'r_smooth': r_smooth,
                'lambda_base': LAMBDA_BASE
            })
            run_num += 1

    # Print summary
    print("Generated 10 grid configs:")
    print(f"{'Run':>4} {'eta':>6} {'R_smooth':>9} {'lambda':>8}")
    print("-" * 30)
    for c in configs:
        print(f"{c['run']:4d} {c['eta']:6.2f} {c['r_smooth']:9.1f} {c['lambda_base']:8.1f}")

    # Save params table
    with open(output_dir / 'grid_params.csv', 'w') as f:
        f.write("run,eta,r_smooth,lambda_base\n")
        for c in configs:
            f.write(f"{c['run']},{c['eta']},{c['r_smooth']},{c['lambda_base']}\n")

    print(f"\nConfigs saved to {output_dir}")

if __name__ == '__main__':
    main()
