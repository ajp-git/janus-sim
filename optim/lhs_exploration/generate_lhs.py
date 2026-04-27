#!/usr/bin/env python3
"""Generate 30 Latin Hypercube Sampling configs for Janus parameter exploration."""

import numpy as np
from scipy.stats.qmc import LatinHypercube
import yaml
from pathlib import Path

# Parameter ranges
ETA_MIN, ETA_MAX = 0.30, 1.00
LAMBDA_MIN, LAMBDA_MAX = 15.0, 50.0  # Respect L/3 = 50 Mpc
RSMOOTH_MIN, RSMOOTH_MAX = 3.0, 10.0

N_SAMPLES = 30
SEED = 42

# Fixed simulation parameters
N_PARTICLES = 200000
N_STEPS = 500
BOX_SIZE = 150.0
Z_START = 5.0
Z_END = 1.5

def main():
    output_dir = Path(__file__).parent

    # Generate LHS samples
    sampler = LatinHypercube(d=3, seed=SEED)
    samples = sampler.random(n=N_SAMPLES)

    # Scale to parameter ranges
    etas = ETA_MIN + samples[:, 0] * (ETA_MAX - ETA_MIN)
    lambdas = LAMBDA_MIN + samples[:, 1] * (LAMBDA_MAX - LAMBDA_MIN)
    rsmooths = RSMOOTH_MIN + samples[:, 2] * (RSMOOTH_MAX - RSMOOTH_MIN)

    # Generate configs
    configs = []
    for i, (eta, lam, rsmooth) in enumerate(zip(etas, lambdas, rsmooths), 1):
        config = {
            'output': {
                'dir': f'output/lhs_exploration/lhs_run_{i:02d}',
                'metrics_every_steps': 10,
                'save_snapshots': True,
                'snapshot_redshifts': [5.0, 3.0, 2.0, 1.5]
            },
            'physics': {
                'eta': round(float(eta), 3),
                'hubble_friction': True,
                'lambda_base_mpc': round(float(lam), 1),
                'lambda_floor': 0.01,
                'r_smooth_mpc': round(float(rsmooth), 1)
            },
            'pm_grid': {
                'k_min': 2,
                'n_cells': 128  # Smaller grid for speed
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

        config_path = output_dir / f'config_run_{i:02d}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        configs.append({
            'run': i,
            'eta': config['physics']['eta'],
            'lambda_base': config['physics']['lambda_base_mpc'],
            'r_smooth': config['physics']['r_smooth_mpc']
        })

    # Save summary
    print("Generated 30 LHS configs:")
    print(f"{'Run':>4} {'eta':>6} {'lambda':>8} {'R_smooth':>9}")
    print("-" * 30)
    for c in configs:
        print(f"{c['run']:4d} {c['eta']:6.3f} {c['lambda_base']:8.1f} {c['r_smooth']:9.1f}")

    # Save parameter table
    with open(output_dir / 'lhs_params.csv', 'w') as f:
        f.write("run,eta,lambda_base,r_smooth\n")
        for c in configs:
            f.write(f"{c['run']},{c['eta']},{c['lambda_base']},{c['r_smooth']}\n")

    print(f"\nConfigs saved to {output_dir}")
    print(f"Parameter table: {output_dir / 'lhs_params.csv'}")

if __name__ == '__main__':
    main()
