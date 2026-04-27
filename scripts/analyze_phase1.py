#!/usr/bin/env python3
"""
Analyze Phase 1 results and answer the 4 key questions.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'optim'))
from filament_metrics_v2 import load_snapshot, fof_halos, detect_interhalos_filaments

OUTPUT_DIR = Path('/mnt/T2/janus-sim/output/trichotomie')
PHASE1_DIR = OUTPUT_DIR / 'phase1'


def find_snapshot_for_z(run_dir, target_z, n_steps=600, z_start=5.0, z_end=0.0):
    """Find snapshot closest to target redshift."""
    snaps = sorted(run_dir.glob('snapshots/snap_*.bin'))
    if not snaps:
        return None
    if target_z == 0.0:
        return snaps[-1]
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


def analyze_snapshot(snap_path, box_size=150.0, verbose=False):
    """Analyze a single snapshot with strict filament criteria."""
    try:
        pos, signs, vel = load_snapshot(str(snap_path))
    except:
        return None

    n = len(pos)
    mask_plus = signs > 0
    mask_minus = signs < 0
    com_plus = pos[mask_plus].mean(axis=0)
    com_minus = pos[mask_minus].mean(axis=0)
    dcom = np.linalg.norm(com_plus - com_minus)
    seg = dcom / box_size

    # Use STRICT criteria: min_filament_length=8 Mpc
    result = detect_interhalos_filaments(pos, signs, box_size,
                                          n_cells=64, halo_mask_radius=10.0,
                                          min_filament_length=8.0)

    n_filaments = result['n_filaments_real']
    length_max = result['length_max_real']
    n_halos = result.get('n_halos_plus', 0)

    density_mean = 1.0
    fraction_mplus = 0.5

    if n_filaments > 0 and 'filaments' in result:
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
            density_mean = np.mean(densities)
        total = total_plus + total_minus
        if total > 0:
            fraction_mplus = total_plus / total

        if verbose:
            print(f"    Filaments détails:")
            for i, fil in enumerate(result['filaments']):
                print(f"      #{i+1}: L={fil['length']:.1f} Mpc, n_cells={fil['n_cells']}, "
                      f"AR={fil['aspect_ratio']:.1f}, d_halo={fil['min_dist_to_halo']:.1f}")

    return {
        'n_filaments': n_filaments,
        'length_max': length_max,
        'density_mean': density_mean,
        'fraction_mplus': fraction_mplus,
        'seg': seg,
        'dcom': dcom,
        'n_halos': n_halos,
    }


def compute_score(m):
    if m is None:
        return 0.0
    s_length = min(m['length_max'] / 30.0, 1.0)
    s_count = min(m['n_filaments'] / 3.0, 1.0)
    s_density = min(max(m['density_mean'] - 1.0, 0) / 0.5, 1.0)
    s_purity = min(m['fraction_mplus'] / 0.8, 1.0)
    return 0.30*s_length + 0.25*s_count + 0.20*s_density + 0.15*s_purity + 0.10*0


def analyze_run(run_dir):
    """Analyze at z=0.5, 1.0, 0.0 and return best."""
    best_metrics = None
    best_score = -1
    best_z = None

    for target_z in [0.5, 1.0, 0.0]:
        snap = find_snapshot_for_z(run_dir, target_z)
        if snap is None:
            continue
        metrics = analyze_snapshot(snap)
        if metrics is None:
            continue
        score = compute_score(metrics)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_z = target_z

    if best_metrics:
        best_metrics['best_z'] = best_z
        best_metrics['score'] = best_score

    return best_metrics


def main():
    print("="*70)
    print("PHASE 1 ANALYSIS - 27 RUNS")
    print("="*70)

    runs = sorted(PHASE1_DIR.glob('T1_*'))
    print(f"Found {len(runs)} runs\n")

    results = []

    for run_dir in runs:
        name = run_dir.name
        # Parse params from name: T1_eta0.80_lam5_z1.5
        parts = name.split('_')
        eta = float(parts[1].replace('eta', ''))
        lam = int(parts[2].replace('lam', ''))
        z_act = float(parts[3].replace('z', ''))

        print(f"Analyzing {name}...", end=' ', flush=True)
        metrics = analyze_run(run_dir)

        if metrics:
            print(f"n_fil={metrics['n_filaments']}, L={metrics['length_max']:.1f} @ z={metrics['best_z']}, score={metrics['score']:.3f}")
            results.append({
                'run_name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'n_filaments': metrics['n_filaments'],
                'length_max': metrics['length_max'],
                'density_mean': metrics['density_mean'],
                'best_z': metrics['best_z'],
                'score': metrics['score'],
            })
        else:
            print("FAILED")
            results.append({
                'run_name': name,
                'eta': eta,
                'lambda': lam,
                'z_act': z_act,
                'n_filaments': 0,
                'length_max': 0,
                'density_mean': 1.0,
                'best_z': 0,
                'score': 0,
            })

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False)

    # Save
    df.to_csv(OUTPUT_DIR / 'phase1_results.csv', index=False)

    print("\n" + "="*70)
    print("TOP 10 RESULTS")
    print("="*70)
    print(df[['run_name', 'eta', 'lambda', 'z_act', 'n_filaments',
              'length_max', 'best_z', 'score']].head(10).to_string(index=False))

    # Answer the 4 key questions
    print("\n" + "="*70)
    print("ANSWERS TO 4 KEY QUESTIONS")
    print("="*70)

    top5 = df.head(5)

    # Q1: z_act=2.0 dominates top 5?
    print("\n1. z_act=2.0 DOMINATES TOP 5?")
    z_act_counts = top5['z_act'].value_counts()
    print(f"   Top 5 z_act distribution: {dict(z_act_counts)}")
    if 2.0 in z_act_counts.index and z_act_counts[2.0] >= 3:
        print("   → YES: z_act=2.0 dominates!")
    else:
        dominant = z_act_counts.index[0] if len(z_act_counts) > 0 else "N/A"
        print(f"   → NO: z_act={dominant} dominates")

    # Q2: η ordering: 0.95 > 0.88 > 0.80?
    print("\n2. η ORDERING: 0.95 > 0.88 > 0.80?")
    eta_means = df.groupby('eta')['score'].mean().sort_values(ascending=False)
    print(f"   Mean scores by η:")
    for eta, score in eta_means.items():
        print(f"     η={eta:.2f}: {score:.4f}")
    if list(eta_means.index) == [0.95, 0.88, 0.80]:
        print("   → YES: η=0.95 > η=0.88 > η=0.80")
    else:
        print(f"   → ORDER: {' > '.join([f'η={e:.2f}' for e in eta_means.index])}")

    # Q3: λ=8 optimal?
    print("\n3. λ=8 OPTIMAL?")
    lambda_means = df.groupby('lambda')['score'].mean().sort_values(ascending=False)
    print(f"   Mean scores by λ:")
    for lam, score in lambda_means.items():
        print(f"     λ={lam}: {score:.4f}")
    best_lambda = lambda_means.index[0]
    if best_lambda == 8:
        print("   → YES: λ=8 is optimal")
    else:
        print(f"   → NO: λ={best_lambda} is better")

    # Q4: z=0.5 dominates best_z?
    print("\n4. z=0.5 DOMINATES BEST SCORES?")
    best_z_counts = df[df['score'] > 0]['best_z'].value_counts()
    print(f"   Distribution of best_z (for runs with score>0):")
    for z, count in best_z_counts.items():
        print(f"     z={z}: {count} runs")
    if 0.5 in best_z_counts.index and best_z_counts.get(0.5, 0) >= best_z_counts.get(1.0, 0):
        print("   → YES: z=0.5 dominates!")
    else:
        print(f"   → MIXED or z={best_z_counts.index[0] if len(best_z_counts) > 0 else 'N/A'} dominates")

    print("\n" + "="*70)
    print("TOP 3 FOR PHASE 2")
    print("="*70)
    top3 = df.head(3)
    for i, row in enumerate(top3.itertuples()):
        print(f"\n#{i+1}: {row.run_name}")
        print(f"    η={row.eta}, λ={row._3}, z_act={row.z_act}")
        print(f"    n_filaments={row.n_filaments}, length_max={row.length_max:.1f} Mpc")
        print(f"    best_z={row.best_z}, score={row.score:.3f}")


if __name__ == '__main__':
    main()
