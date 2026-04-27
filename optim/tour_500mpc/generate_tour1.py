#!/usr/bin/env python3
"""Generate Tour 1 configurations for 500 Mpc trichotomy."""
import os

# η values from the plan
eta_values = [
    0.70, 0.73, 0.77, 0.80, 0.83, 0.87, 0.90, 0.93, 0.97, 1.00,
    1.03, 1.07, 1.10, 1.13, 1.17, 1.20, 1.23, 1.27, 1.30, 0.99
]

template = """simulation:
  box_size_mpc: 500.0
  n_particles: 500000
  n_steps: 800
  z_start: 5.0
  z_end: 0.0
  seed: 42
  theta: 0.7
  softening_mpc: null

physics:
  eta: {eta:.2f}
  lambda_base_mpc: 25.0
  r_smooth_mpc: 5.0
  lambda_floor: 0.01
  hubble_friction: true
  cross_force_asymmetry: 1.0
  cross_force_z_start: 2.0
  cross_force_z_width: 0.5

pm_grid:
  n_cells: 128
  k_min: 2

output:
  dir: output/tour_500mpc/tour1/E{num:02d}
  snapshot_redshifts:
  - 3.0
  - 2.0
  - 1.0
  - 0.0
  metrics_every_steps: 20
  save_snapshots: true
"""

out_dir = "optim/tour_500mpc/tour1"
os.makedirs(out_dir, exist_ok=True)

for i, eta in enumerate(eta_values, 1):
    config = template.format(eta=eta, num=i)
    fname = f"{out_dir}/E{i:02d}_eta{eta:.2f}.yaml"
    with open(fname, 'w') as f:
        f.write(config)
    print(f"Created {fname}")

print(f"\nTotal: {len(eta_values)} configs")
