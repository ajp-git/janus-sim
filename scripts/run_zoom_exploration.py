#!/usr/bin/env python3
"""
Zoom Exploration — 20 LHS runs to find optimal filament parameters
"""

import subprocess
import json
import numpy as np
import os
import sys
from pathlib import Path
from scipy.stats.qmc import LatinHypercube
from scipy.ndimage import gaussian_filter
import struct

# Parameter ranges for LHS
PARAM_RANGES = {
    'k_min':    (5, 50),      # min k mode for ICs
    'eps':      (0.02, 0.20), # softening in Mpc
    'amp':      (0.1, 1.0),   # IC amplitude
    'box':      (30, 80),     # box size in Mpc
    'n':        (50000, 300000),  # particle count
}

N_RUNS = 20
STEPS = 2000
# Use local path when testing, Docker path when running
import os
if os.path.exists("/app"):
    BASE_OUTPUT = "/app/output/zoom_exploration"
else:
    BASE_OUTPUT = "/mnt/T2/janus-sim/output/zoom_exploration"


def generate_lhs_configs(n_runs=N_RUNS, seed=42):
    """Generate Latin Hypercube samples for parameter space."""
    np.random.seed(seed)
    sampler = LatinHypercube(d=5, seed=seed)
    samples = sampler.random(n=n_runs)

    configs = []
    for i, s in enumerate(samples):
        config = {
            'run_id': i + 1,
            'k_min': int(PARAM_RANGES['k_min'][0] + s[0] * (PARAM_RANGES['k_min'][1] - PARAM_RANGES['k_min'][0])),
            'eps': PARAM_RANGES['eps'][0] + s[1] * (PARAM_RANGES['eps'][1] - PARAM_RANGES['eps'][0]),
            'amp': PARAM_RANGES['amp'][0] + s[2] * (PARAM_RANGES['amp'][1] - PARAM_RANGES['amp'][0]),
            'box': PARAM_RANGES['box'][0] + s[3] * (PARAM_RANGES['box'][1] - PARAM_RANGES['box'][0]),
            'n': int(PARAM_RANGES['n'][0] + s[4] * (PARAM_RANGES['n'][1] - PARAM_RANGES['n'][0])),
            'seed': 42 + i,
        }
        configs.append(config)
    return configs


def load_snapshot(path):
    """Load binary snapshot: header u64 N, then N x 28 bytes (7 x f32)."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    return data[:, :3], data[:, 3:6], data[:, 6]  # pos, vel, mass_sign


def compute_tweb(pos, signs, box, grid_size=32):
    """
    Compute T-web classification using Hessian of density field.
    Returns fractions: (void, sheet, filament, node)
    """
    cell = box / grid_size
    g = grid_size

    # Build density grid for m-
    mask_m = signs < 0
    pm = pos[mask_m]
    if pm.min() < -box * 0.1:
        pm = pm + box / 2

    rho = np.zeros((g, g, g), dtype=np.float64)
    ix = np.clip((pm[:, 0] / box * g).astype(int), 0, g-1)
    iy = np.clip((pm[:, 1] / box * g).astype(int), 0, g-1)
    iz = np.clip((pm[:, 2] / box * g).astype(int), 0, g-1)
    np.add.at(rho, (ix, iy, iz), 1.0)

    # Smooth density field
    rho_smooth = gaussian_filter(rho, sigma=1.0)

    # Compute Hessian at each cell
    n_void, n_sheet, n_filament, n_node = 0, 0, 0, 0

    for iz in range(g):
        for iy in range(g):
            for ix in range(g):
                # Second derivatives (finite difference)
                dxx = (rho_smooth[(ix+1)%g, iy, iz] - 2*rho_smooth[ix, iy, iz] + rho_smooth[(ix-1)%g, iy, iz]) / cell**2
                dyy = (rho_smooth[ix, (iy+1)%g, iz] - 2*rho_smooth[ix, iy, iz] + rho_smooth[ix, (iy-1)%g, iz]) / cell**2
                dzz = (rho_smooth[ix, iy, (iz+1)%g] - 2*rho_smooth[ix, iy, iz] + rho_smooth[ix, iy, (iz-1)%g]) / cell**2

                dxy = (rho_smooth[(ix+1)%g, (iy+1)%g, iz] - rho_smooth[(ix+1)%g, (iy-1)%g, iz] -
                       rho_smooth[(ix-1)%g, (iy+1)%g, iz] + rho_smooth[(ix-1)%g, (iy-1)%g, iz]) / (4*cell**2)
                dxz = (rho_smooth[(ix+1)%g, iy, (iz+1)%g] - rho_smooth[(ix+1)%g, iy, (iz-1)%g] -
                       rho_smooth[(ix-1)%g, iy, (iz+1)%g] + rho_smooth[(ix-1)%g, iy, (iz-1)%g]) / (4*cell**2)
                dyz = (rho_smooth[ix, (iy+1)%g, (iz+1)%g] - rho_smooth[ix, (iy+1)%g, (iz-1)%g] -
                       rho_smooth[ix, (iy-1)%g, (iz+1)%g] + rho_smooth[ix, (iy-1)%g, (iz-1)%g]) / (4*cell**2)

                H = np.array([[dxx, dxy, dxz],
                              [dxy, dyy, dyz],
                              [dxz, dyz, dzz]])

                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.sort(eigvals)[::-1]  # λ1 >= λ2 >= λ3

                # Classification threshold
                th = 0.0
                if eigvals[0] < th:
                    n_void += 1
                elif eigvals[1] < th:
                    n_sheet += 1
                elif eigvals[2] < th:
                    n_filament += 1
                else:
                    n_node += 1

    total = g**3
    return n_void/total, n_sheet/total, n_filament/total, n_node/total


def analyze_run(output_dir, box):
    """Analyze completed run and return metrics."""
    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    # Find final snapshot
    snap_dir = Path(output_dir) / "snapshots"
    snaps = sorted(snap_dir.glob("snap_*.bin"))
    if not snaps:
        return None

    final_snap = snaps[-1]
    pos, vel, signs = load_snapshot(final_snap)

    # Compute T-web
    f_void, f_sheet, f_filament, f_node = compute_tweb(pos, signs, box)

    return {
        'seg_final': summary.get('seg_final', 0),
        'sigma_P': summary.get('sigma_P', 0),
        'L_J': summary.get('L_J', 0),
        'KE_max': summary.get('KE_max', 1),
        'KE_final': summary.get('KE_final', 1),
        'time_s': summary.get('time_s', 0),
        'f_void': f_void,
        'f_sheet': f_sheet,
        'f_filament': f_filament,
        'f_node': f_node,
    }


def compute_score(metrics):
    """Compute composite score for ranking."""
    if metrics is None:
        return -1

    # Reject if KE exploded
    if metrics['KE_max'] > 3.0:
        return -1

    # Score: 50% filament fraction + 30% segregation + 20% (1 - void fraction)
    score = (0.5 * metrics['f_filament'] +
             0.3 * metrics['seg_final'] +
             0.2 * (1 - metrics['f_void']))
    return score


def run_single(config, dry_run=False):
    """Run a single exploration configuration."""
    run_dir = f"{BASE_OUTPUT}/run_{config['run_id']:03d}"

    cmd = [
        "cargo", "run", "--release", "--features", "cuda cufft",
        "--bin", "zoom_exploration", "--",
        "--box-size", str(config['box']),
        "--n", str(config['n']),
        "--k-min", str(config['k_min']),
        "--eps", str(config['eps']),
        "--amp", str(config['amp']),
        "--steps", str(STEPS),
        "--seed", str(config['seed']),
        "--output", run_dir,
    ]

    print(f"\n{'='*60}")
    print(f"RUN {config['run_id']:03d} | box={config['box']:.0f} k_min={config['k_min']} "
          f"ε={config['eps']:.3f} amp={config['amp']:.2f} N={config['n']}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd[:10])}...")
        return None

    try:
        result = subprocess.run(cmd, cwd="/app", capture_output=False, timeout=7200)
        if result.returncode != 0:
            print(f"  [ERROR] Run failed with code {result.returncode}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Run exceeded 2 hours")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    return analyze_run(run_dir, config['box'])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print configs without running')
    parser.add_argument('--start', type=int, default=1, help='Start from run N')
    parser.add_argument('--end', type=int, default=N_RUNS, help='End at run N')
    args = parser.parse_args()

    print("="*60)
    print("  ZOOM EXPLORATION — 20 LHS RUNS")
    print("="*60)

    configs = generate_lhs_configs()

    # Save configs
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    with open(f"{BASE_OUTPUT}/configs.json", 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"\nGenerated {len(configs)} configurations:")
    print(f"{'Run':>4} | {'Box':>5} | {'k_min':>5} | {'ε':>6} | {'amp':>5} | {'N':>7}")
    print("-"*50)
    for c in configs:
        print(f"{c['run_id']:>4} | {c['box']:>5.0f} | {c['k_min']:>5} | {c['eps']:>6.3f} | {c['amp']:>5.2f} | {c['n']:>7}")

    if args.dry_run:
        print("\n[DRY RUN MODE - no simulations will be executed]")
        return

    # Run simulations
    results = []
    for config in configs:
        if config['run_id'] < args.start or config['run_id'] > args.end:
            continue

        metrics = run_single(config)
        if metrics:
            metrics['run_id'] = config['run_id']
            metrics['config'] = config
            metrics['score'] = compute_score(metrics)
            results.append(metrics)

            # Save intermediate results
            with open(f"{BASE_OUTPUT}/results.json", 'w') as f:
                json.dump(results, f, indent=2)

    # Final ranking
    print("\n" + "="*80)
    print("  RESULTS — RANKED BY SCORE")
    print("="*80)

    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"{'Run':>4} | {'Box':>4} | {'k_min':>5} | {'ε':>5} | {'amp':>4} | "
          f"{'Fil%':>5} | {'Seg':>5} | {'Score':>5} | {'Status':>8}")
    print("-"*75)

    for r in results_sorted:
        c = r['config']
        status = "OK" if r['KE_max'] < 3.0 else "KE_FAIL"
        print(f"{r['run_id']:>4} | {c['box']:>4.0f} | {c['k_min']:>5} | {c['eps']:>5.2f} | {c['amp']:>4.1f} | "
              f"{r['f_filament']*100:>5.1f} | {r['seg_final']:>5.3f} | {r['score']:>5.3f} | {status:>8}")

    # Top 3
    print("\n" + "="*60)
    print("  TOP 3 CONFIGURATIONS FOR PHASE 1")
    print("="*60)

    for i, r in enumerate(results_sorted[:3], 1):
        c = r['config']
        print(f"\n{i}. Run {r['run_id']:03d} — Score={r['score']:.3f}")
        print(f"   box={c['box']:.0f} Mpc, k_min={c['k_min']}, ε={c['eps']:.3f}, amp={c['amp']:.2f}, N={c['n']}")
        print(f"   Filaments={r['f_filament']*100:.1f}%, Seg={r['seg_final']:.3f}, σ_P={r['sigma_P']:.3f}")


if __name__ == "__main__":
    main()
