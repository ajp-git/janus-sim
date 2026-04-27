#!/usr/bin/env python3
"""
Trichotomy exploration for Janus filament optimization.
Phase 1: 27 runs (3×3×3 grid)
Phase 2: 9 runs (zoom on top 3)
Phase 3: 3 runs (validation)
"""

import os
import sys
import yaml
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, fof_halos, detect_interhalos_filaments

# Base directories
BASE_DIR = Path('/mnt/T2/janus-sim')
OUTPUT_DIR = BASE_DIR / 'output' / 'trichotomie'
CONFIG_DIR = BASE_DIR / 'optim' / 'trichotomie'


def generate_phase1_configs():
    """Generate 27 configs for Phase 1 grid search."""

    etas = [0.80, 0.88, 0.95]
    lambdas = [5, 8, 12]
    z_acts = [1.5, 2.0, 3.0]

    configs = []

    for eta in etas:
        for lam in lambdas:
            for z_act in z_acts:
                name = f"T1_eta{eta:.2f}_lam{lam}_z{z_act:.1f}"

                config = {
                    'simulation': {
                        'n_particles': 500000,
                        'n_steps': 600,
                        'box_size_mpc': 150.0,
                        'seed': 42,
                        'z_start': 5.0,
                        'z_end': 0.0,
                        'theta': 0.7,
                    },
                    'physics': {
                        'eta': eta,
                        'lambda_base_mpc': float(lam),
                        'lambda_floor': 0.01,
                        'r_smooth_mpc': 1.6,
                        'hubble_friction': True,
                        'cross_force_z_start': z_act,
                        'cross_force_z_width': 0.5,
                    },
                    'pm_grid': {
                        'n_cells': 256,
                        'k_min': 2,
                    },
                    'output': {
                        'dir': f'output/trichotomie/phase1/{name}',
                        'save_snapshots': True,
                        'snapshot_redshifts': [2.0, 1.0, 0.5, 0.0],
                        'metrics_every_steps': 25,
                    }
                }

                config_path = CONFIG_DIR / f'{name}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                configs.append({
                    'name': name,
                    'eta': eta,
                    'lambda': lam,
                    'z_act': z_act,
                    'config_path': str(config_path),
                })

    print(f"Generated {len(configs)} Phase 1 configs")
    return configs


def run_simulation(config_path, timeout_sec=600):
    """Run a single simulation."""
    cmd = [
        'docker', 'compose', 'run', '--rm', 'dev',
        'cargo', 'run', '--release', '--features', 'cuda,cufft',
        '--bin', 'janus_optim', '--',
        '--config', f'/app/{config_path.relative_to(BASE_DIR)}'
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
        print(f"  TIMEOUT after {timeout_sec}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def find_snapshot_for_z(run_dir, target_z, n_steps=600, z_start=5.0, z_end=0.0):
    """Find snapshot closest to target redshift."""
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None

    if target_z == 0.0:
        return snaps[-1]

    # Estimate step for target_z
    target_step = int((z_start - target_z) / (z_start - z_end) * n_steps)

    best_snap = None
    best_diff = float('inf')
    for snap in snaps:
        step = int(snap.stem.replace('snap_', ''))
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_snap = snap
    return best_snap


def analyze_snapshot(snap_path, box_size=150.0):
    """Analyze a single snapshot for filaments."""
    try:
        pos, signs, vel = load_snapshot(str(snap_path))
    except Exception as e:
        return None

    n = len(pos)

    mask_plus = signs > 0
    mask_minus = signs < 0
    com_plus = pos[mask_plus].mean(axis=0)
    com_minus = pos[mask_minus].mean(axis=0)
    dcom = np.linalg.norm(com_plus - com_minus)
    seg = dcom / box_size

    result = detect_interhalos_filaments(
        pos, signs, box_size,
        n_cells=64,
        halo_mask_radius=10.0,
        min_filament_length=8.0
    )

    n_filaments = result['n_filaments_real']
    length_max = result['length_max_real']
    length_mean = result.get('length_mean_real', 0.0)

    halos = fof_halos(pos, signs, box_size, b=0.2, min_particles=50)
    n_halos_plus = sum(1 for h in halos if h['sign'] == 1)
    n_halos_minus = sum(1 for h in halos if h['sign'] == -1)

    density_mean = 1.0
    fraction_mplus = 0.5

    if n_filaments > 0 and 'filaments' in result:
        cell_size = box_size / 64
        filaments = result['filaments']
        total_plus = 0
        total_minus = 0
        densities = []

        for fil in filaments:
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
            density_mean = np.mean(densities)
        total = total_plus + total_minus
        if total > 0:
            fraction_mplus = total_plus / total

    return {
        'n_particles': n,
        'n_filaments': n_filaments,
        'length_max': length_max,
        'length_mean': length_mean,
        'density_mean': density_mean,
        'fraction_mplus': fraction_mplus,
        'coherent_flow': 0.0,
        'n_halos_plus': n_halos_plus,
        'n_halos_minus': n_halos_minus,
        'S_final': seg,
        'DCOM': dcom,
    }


def analyze_run(run_dir, box_size=150.0):
    """Analyze a run at multiple redshifts, return BEST score.

    Key insight: Peak filaments occur at z=0.5, not z=0.
    Score = max(score_z0.5, score_z1.0, score_z0.0)
    """

    run_dir = Path(run_dir)

    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None

    # Analyze at multiple redshifts
    target_redshifts = [0.5, 1.0, 0.0]  # Priority order: z=0.5 peak first
    best_metrics = None
    best_score = -1
    best_z = None

    for target_z in target_redshifts:
        snap_path = find_snapshot_for_z(run_dir, target_z)
        if snap_path is None:
            continue

        metrics = analyze_snapshot(snap_path, box_size)
        if metrics is None:
            continue

        score = compute_filament_score(metrics)

        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_z = target_z

    if best_metrics is None:
        return None

    # Add which redshift was best
    best_metrics['best_z'] = best_z
    best_metrics['best_score'] = best_score

    # Also get final z=0 metrics for S_final
    final_snap = snaps[-1]
    try:
        pos, signs, _ = load_snapshot(str(final_snap))
        mask_plus = signs > 0
        mask_minus = signs < 0
        com_plus = pos[mask_plus].mean(axis=0)
        com_minus = pos[mask_minus].mean(axis=0)
        dcom = np.linalg.norm(com_plus - com_minus)
        best_metrics['S_final'] = dcom / box_size
        best_metrics['DCOM'] = dcom
    except:
        pass

    return best_metrics


def compute_filament_score(metrics):
    """Compute filament score from metrics."""

    if metrics is None:
        return 0.0

    # Length: objective > 30 Mpc
    s_length = min(metrics['length_max'] / 30.0, 1.0)

    # Count: objective > 3 filaments
    s_count = min(metrics['n_filaments'] / 3.0, 1.0)

    # Density: objective > 1.5 rho_bar
    s_density = min(max(metrics['density_mean'] - 1.0, 0) / 0.5, 1.0)

    # Purity m+: objective > 80%
    s_purity = min(metrics['fraction_mplus'] / 0.8, 1.0)

    # Coherent flow: objective > 0.5
    s_flow = min(metrics['coherent_flow'] / 0.5, 1.0)

    score = (0.30 * s_length +
             0.25 * s_count +
             0.20 * s_density +
             0.15 * s_purity +
             0.10 * s_flow)

    return score


def run_phase1():
    """Run Phase 1: 27 runs grid search."""

    print("\n" + "="*70)
    print("PHASE 1: 27 RUNS GRID SEARCH")
    print("="*70 + "\n")

    # Generate configs
    configs = generate_phase1_configs()

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/27] {cfg['name']}")
        print(f"  eta={cfg['eta']}, lambda={cfg['lambda']}, z_act={cfg['z_act']}")

        run_dir = OUTPUT_DIR / 'phase1' / cfg['name']

        # Check if already done
        if (run_dir / 'snapshots').exists() and len(list((run_dir / 'snapshots').glob('*.bin'))) > 0:
            print("  Already done, analyzing...")
        else:
            print("  Running simulation...")
            success = run_simulation(Path(cfg['config_path']), timeout_sec=300)
            if not success:
                print("  FAILED")
                results.append({
                    'run_name': cfg['name'],
                    'eta': cfg['eta'],
                    'lambda': cfg['lambda'],
                    'z_act': cfg['z_act'],
                    'n_filaments': 0,
                    'length_max': 0,
                    'density_mean': 1.0,
                    'filament_score': 0.0,
                    'status': 'FAILED'
                })
                continue

        # Analyze at z=0.5, 1.0, 0.0 - take best score
        print("  Analyzing (z=0.5, 1.0, 0.0)...")
        metrics = analyze_run(run_dir, box_size=150.0)

        if metrics:
            score = compute_filament_score(metrics)
            best_z = metrics.get('best_z', 0.0)
            print(f"  n_filaments={metrics['n_filaments']}, length_max={metrics['length_max']:.1f} Mpc @ z={best_z}")
            print(f"  filament_score = {score:.3f}")

            results.append({
                'run_name': cfg['name'],
                'eta': cfg['eta'],
                'lambda': cfg['lambda'],
                'z_act': cfg['z_act'],
                'best_z': best_z,
                'n_filaments': metrics['n_filaments'],
                'length_max': metrics['length_max'],
                'length_mean': metrics.get('length_mean', 0),
                'density_mean': metrics['density_mean'],
                'fraction_mplus': metrics['fraction_mplus'],
                'n_halos_plus': metrics['n_halos_plus'],
                'n_halos_minus': metrics['n_halos_minus'],
                'S_final': metrics['S_final'],
                'DCOM': metrics['DCOM'],
                'filament_score': score,
                'status': 'OK'
            })
        else:
            results.append({
                'run_name': cfg['name'],
                'eta': cfg['eta'],
                'lambda': cfg['lambda'],
                'z_act': cfg['z_act'],
                'n_filaments': 0,
                'length_max': 0,
                'density_mean': 1.0,
                'filament_score': 0.0,
                'status': 'ANALYSIS_FAILED'
            })

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('filament_score', ascending=False)
    df.to_csv(OUTPUT_DIR / 'phase1_results.csv', index=False)

    print("\n" + "="*70)
    print("PHASE 1 RESULTS")
    print("="*70)
    print(df[['run_name', 'eta', 'lambda', 'z_act', 'n_filaments',
              'length_max', 'best_z', 'filament_score']].head(10).to_string(index=False))

    # Create heatmaps
    create_phase1_heatmaps(df)

    return df


def create_phase1_heatmaps(df):
    """Create heatmaps for Phase 1 results."""

    z_acts = df['z_act'].unique()

    fig, axes = plt.subplots(1, len(z_acts), figsize=(5*len(z_acts), 4))
    if len(z_acts) == 1:
        axes = [axes]

    for ax, z_act in zip(axes, sorted(z_acts)):
        subset = df[df['z_act'] == z_act]

        etas = sorted(subset['eta'].unique())
        lambdas = sorted(subset['lambda'].unique())

        scores = np.zeros((len(etas), len(lambdas)))
        for i, eta in enumerate(etas):
            for j, lam in enumerate(lambdas):
                row = subset[(subset['eta'] == eta) & (subset['lambda'] == lam)]
                if len(row) > 0:
                    scores[i, j] = row['filament_score'].values[0]

        im = ax.imshow(scores, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=max(0.5, scores.max()))

        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([str(l) for l in lambdas])
        ax.set_yticks(range(len(etas)))
        ax.set_yticklabels([f'{e:.2f}' for e in etas])
        ax.set_xlabel('λ (Mpc)')
        ax.set_ylabel('η')
        ax.set_title(f'z_act = {z_act}')

        # Add text annotations
        for i in range(len(etas)):
            for j in range(len(lambdas)):
                ax.text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center',
                       color='white' if scores[i,j] > 0.25 else 'black')

        plt.colorbar(im, ax=ax, label='filament_score')

    plt.suptitle('Phase 1: Filament Score Grid')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase1_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'phase1_heatmaps.png'}")


def generate_phase2_configs(top3):
    """Generate Phase 2 configs by zooming on top 3 from Phase 1."""

    configs = []

    for rank, row in enumerate(top3.itertuples()):
        base_eta = row.eta
        base_lam = row._3  # lambda
        base_z = row.z_act

        # Determine which parameter differs most from baseline P2
        # Baseline P2: eta=0.88, lambda=8, z=2.0
        diff_eta = abs(base_eta - 0.88)
        diff_lam = abs(base_lam - 8)
        diff_z = abs(base_z - 2.0)

        # Zoom on the most different parameter
        if diff_eta >= diff_lam and diff_eta >= diff_z:
            # Zoom on eta
            zoom_param = 'eta'
            values = [base_eta * 0.95, base_eta, base_eta * 1.05]
            fixed = {'lambda': base_lam, 'z_act': base_z}
        elif diff_lam >= diff_z:
            # Zoom on lambda
            zoom_param = 'lambda'
            values = [base_lam * 0.8, base_lam, base_lam * 1.2]
            fixed = {'eta': base_eta, 'z_act': base_z}
        else:
            # Zoom on z_act
            zoom_param = 'z_act'
            values = [base_z * 0.85, base_z, base_z * 1.15]
            fixed = {'eta': base_eta, 'lambda': base_lam}

        for v in values:
            if zoom_param == 'eta':
                eta, lam, z_act = v, fixed['lambda'], fixed['z_act']
            elif zoom_param == 'lambda':
                eta, lam, z_act = fixed['eta'], v, fixed['z_act']
            else:
                eta, lam, z_act = fixed['eta'], fixed['lambda'], v

            name = f"T2_r{rank+1}_eta{eta:.3f}_lam{lam:.1f}_z{z_act:.2f}"

            config = {
                'simulation': {
                    'n_particles': 1000000,
                    'n_steps': 1200,
                    'box_size_mpc': 150.0,
                    'seed': 42,
                    'z_start': 5.0,
                    'z_end': 0.0,
                    'theta': 0.7,
                },
                'physics': {
                    'eta': float(eta),
                    'lambda_base_mpc': float(lam),
                    'lambda_floor': 0.01,
                    'r_smooth_mpc': 1.6,
                    'hubble_friction': True,
                    'cross_force_z_start': float(z_act),
                    'cross_force_z_width': 0.5,
                },
                'pm_grid': {
                    'n_cells': 256,
                    'k_min': 2,
                },
                'output': {
                    'dir': f'output/trichotomie/phase2/{name}',
                    'save_snapshots': True,
                    'snapshot_redshifts': [2.0, 1.5, 1.0, 0.5, 0.2, 0.0],
                    'metrics_every_steps': 25,
                }
            }

            config_path = CONFIG_DIR / f'{name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            configs.append({
                'name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'config_path': str(config_path),
                'parent_rank': rank + 1,
            })

    print(f"Generated {len(configs)} Phase 2 configs")
    return configs


def run_phase2(phase1_df):
    """Run Phase 2: 9 runs zooming on top 3."""

    print("\n" + "="*70)
    print("PHASE 2: 9 RUNS ZOOM")
    print("="*70 + "\n")

    # Get top 3 from Phase 1
    top3 = phase1_df.head(3)
    print("Top 3 from Phase 1:")
    print(top3[['run_name', 'filament_score']].to_string(index=False))
    print()

    # Generate configs
    configs = generate_phase2_configs(top3)

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/9] {cfg['name']}")

        run_dir = OUTPUT_DIR / 'phase2' / cfg['name']

        # Check if already done
        if (run_dir / 'snapshots').exists() and len(list((run_dir / 'snapshots').glob('*.bin'))) > 0:
            print("  Already done, analyzing...")
        else:
            print("  Running simulation...")
            success = run_simulation(Path(cfg['config_path']), timeout_sec=900)
            if not success:
                print("  FAILED")
                results.append({
                    'run_name': cfg['name'],
                    'eta': cfg['eta'],
                    'lambda': cfg['lambda'],
                    'z_act': cfg['z_act'],
                    'filament_score': 0.0,
                    'status': 'FAILED'
                })
                continue

        # Analyze
        print("  Analyzing...")
        metrics = analyze_run(run_dir, box_size=150.0)

        if metrics:
            score = compute_filament_score(metrics)
            print(f"  n_filaments={metrics['n_filaments']}, length_max={metrics['length_max']:.1f} Mpc")
            print(f"  filament_score = {score:.3f}")

            results.append({
                'run_name': cfg['name'],
                'eta': cfg['eta'],
                'lambda': cfg['lambda'],
                'z_act': cfg['z_act'],
                'n_filaments': metrics['n_filaments'],
                'length_max': metrics['length_max'],
                'density_mean': metrics['density_mean'],
                'fraction_mplus': metrics['fraction_mplus'],
                'n_halos_plus': metrics['n_halos_plus'],
                'S_final': metrics['S_final'],
                'DCOM': metrics['DCOM'],
                'filament_score': score,
                'status': 'OK'
            })
        else:
            results.append({
                'run_name': cfg['name'],
                'eta': cfg['eta'],
                'lambda': cfg['lambda'],
                'z_act': cfg['z_act'],
                'filament_score': 0.0,
                'status': 'ANALYSIS_FAILED'
            })

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('filament_score', ascending=False)
    df.to_csv(OUTPUT_DIR / 'phase2_results.csv', index=False)

    print("\n" + "="*70)
    print("PHASE 2 RESULTS")
    print("="*70)
    print(df[['run_name', 'eta', 'lambda', 'z_act', 'n_filaments',
              'length_max', 'filament_score']].to_string(index=False))

    return df


def generate_phase3_configs(top3):
    """Generate Phase 3 configs for validation runs."""

    configs = []

    for rank, row in enumerate(top3.itertuples()):
        eta = row.eta
        lam = row._3  # lambda
        z_act = row.z_act

        name = f"T3_r{rank+1}_eta{eta:.3f}_lam{lam:.1f}_z{z_act:.2f}"

        config = {
            'simulation': {
                'n_particles': 2000000,
                'n_steps': 2000,
                'box_size_mpc': 200.0,
                'seed': 42,
                'z_start': 5.0,
                'z_end': 0.0,
                'theta': 0.7,
            },
            'physics': {
                'eta': float(eta),
                'lambda_base_mpc': float(lam),
                'lambda_floor': 0.01,
                'r_smooth_mpc': 1.6,
                'hubble_friction': True,
                'cross_force_z_start': float(z_act),
                'cross_force_z_width': 0.5,
            },
            'pm_grid': {
                'n_cells': 512,
                'k_min': 2,
            },
            'output': {
                'dir': f'output/trichotomie/phase3/{name}',
                'save_snapshots': True,
                'snapshot_redshifts': [3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0],
                'metrics_every_steps': 25,
            }
        }

        config_path = CONFIG_DIR / f'{name}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        configs.append({
            'name': name,
            'eta': eta,
            'lambda': lam,
            'z_act': z_act,
            'config_path': str(config_path),
        })

    print(f"Generated {len(configs)} Phase 3 configs")
    return configs


def run_phase3(phase2_df):
    """Run Phase 3: 3 validation runs."""

    print("\n" + "="*70)
    print("PHASE 3: 3 VALIDATION RUNS (HIGH RES)")
    print("="*70 + "\n")

    # Get top 3 from Phase 2
    top3 = phase2_df.head(3)
    print("Top 3 from Phase 2:")
    print(top3[['run_name', 'filament_score']].to_string(index=False))
    print()

    # Generate configs
    configs = generate_phase3_configs(top3)

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/3] {cfg['name']}")

        run_dir = OUTPUT_DIR / 'phase3' / cfg['name']

        # Check if already done
        if (run_dir / 'snapshots').exists() and len(list((run_dir / 'snapshots').glob('*.bin'))) > 0:
            print("  Already done, analyzing...")
        else:
            print("  Running simulation (high res, ~30min)...")
            success = run_simulation(Path(cfg['config_path']), timeout_sec=3600)
            if not success:
                print("  FAILED")
                results.append({
                    'run_name': cfg['name'],
                    'eta': cfg['eta'],
                    'lambda': cfg['lambda'],
                    'z_act': cfg['z_act'],
                    'filament_score': 0.0,
                    'status': 'FAILED'
                })
                continue

        # Analyze
        print("  Analyzing...")
        metrics = analyze_run(run_dir, box_size=200.0)

        if metrics:
            score = compute_filament_score(metrics)
            print(f"  n_filaments={metrics['n_filaments']}, length_max={metrics['length_max']:.1f} Mpc")
            print(f"  density_mean={metrics['density_mean']:.2f}, fraction_m+={metrics['fraction_mplus']:.2f}")
            print(f"  filament_score = {score:.3f}")

            results.append({
                'run_name': cfg['name'],
                'eta': cfg['eta'],
                'lambda': cfg['lambda'],
                'z_act': cfg['z_act'],
                'n_filaments': metrics['n_filaments'],
                'length_max': metrics['length_max'],
                'length_mean': metrics.get('length_mean', 0),
                'density_mean': metrics['density_mean'],
                'fraction_mplus': metrics['fraction_mplus'],
                'coherent_flow': metrics['coherent_flow'],
                'n_halos_plus': metrics['n_halos_plus'],
                'n_halos_minus': metrics['n_halos_minus'],
                'S_final': metrics['S_final'],
                'DCOM': metrics['DCOM'],
                'filament_score': score,
                'status': 'OK'
            })
        else:
            results.append({
                'run_name': cfg['name'],
                'eta': cfg['eta'],
                'lambda': cfg['lambda'],
                'z_act': cfg['z_act'],
                'filament_score': 0.0,
                'status': 'ANALYSIS_FAILED'
            })

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('filament_score', ascending=False)
    df.to_csv(OUTPUT_DIR / 'phase3_results.csv', index=False)

    print("\n" + "="*70)
    print("PHASE 3 RESULTS - FINAL")
    print("="*70)
    print(df.to_string(index=False))

    return df


def main():
    """Run the full trichotomy exploration."""

    print("\n" + "="*70)
    print("TRICHOTOMY FILAMENT EXPLORATION")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print(f"Output: {OUTPUT_DIR}")

    # Phase 1
    phase1_df = run_phase1()

    # Check if any filaments found
    if phase1_df['filament_score'].max() == 0:
        print("\n*** NO FILAMENTS IN PHASE 1 ***")
        print("*** Consider expanding grid ***")
        return

    # Phase 2
    phase2_df = run_phase2(phase1_df)

    # Phase 3
    phase3_df = run_phase3(phase2_df)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    winner = phase3_df.iloc[0]
    print(f"\nWINNER: {winner['run_name']}")
    print(f"  eta = {winner['eta']:.3f}")
    print(f"  lambda = {winner['lambda']:.1f} Mpc")
    print(f"  z_act = {winner['z_act']:.2f}")
    print(f"  n_filaments = {winner['n_filaments']}")
    print(f"  length_max = {winner['length_max']:.1f} Mpc")
    print(f"  filament_score = {winner['filament_score']:.3f}")

    if winner['filament_score'] > 0.60:
        print("\n*** SCORE > 0.60: PREPARE PUBLICATION RUN ***")
        print("*** N=5M, BOX=300 Mpc ***")
    elif winner['filament_score'] > 0.50:
        print("\n*** SCORE > 0.50: OBJECTIVE MET ***")
    else:
        print("\n*** SCORE < 0.50: CONTINUE EXPLORATION ***")

    print(f"\nCompleted at: {datetime.now()}")


if __name__ == '__main__':
    main()
