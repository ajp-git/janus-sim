#!/usr/bin/env python3
"""
Tour 3 Filament Analysis — 5 runs trichotomie fine à 2M particules
η ∈ {0.95, 0.975, 1.00, 1.025, 1.05}, λ=25, z_act=2.5 fixés

Objectif : confirmer η optimal et vérifier convergence avec résolution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/T2/janus-sim/optim')
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

BASE_DIR = Path("/mnt/T2/janus-sim/output/tour3_2M")
RESULTS_FILE = BASE_DIR / "tour3_results.csv"
BOX_SIZE = 500.0

TARGET_REDSHIFTS = [0.5, 0.0]


def find_snapshot(run_dir, target_z):
    """Find snapshot closest to target redshift."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.exists():
        return None

    snaps = list(snap_dir.glob("snap_*.bin"))
    if not snaps:
        return None

    if target_z == 0.0:
        return sorted(snaps)[-1]
    elif target_z == 0.5:
        # z=0.5 is roughly step 800 out of 1200 for z_start=5
        for snap in sorted(snaps):
            step = int(snap.stem.split('_')[1])
            if 750 <= step <= 900:
                return snap
        idx = int(len(snaps) * 2 / 3)
        return sorted(snaps)[min(idx, len(snaps)-1)]
    return sorted(snaps)[-1]


def analyze_filaments_for_run(run_name, run_dir):
    """Analyze filaments for a single run."""
    results = {'run': run_name}

    for z in TARGET_REDSHIFTS:
        snap_path = find_snapshot(run_dir, z)
        if snap_path is None:
            results[f'n_fil_z{z}'] = 0
            results[f'len_mean_z{z}'] = 0.0
            results[f'len_max_z{z}'] = 0.0
            results[f'n_halos_z{z}'] = 0
            continue

        try:
            pos, signs, vel = load_snapshot(str(snap_path))
            fil_result = detect_interhalos_filaments(
                pos, signs, BOX_SIZE,
                n_cells=64,
                halo_mask_radius=10.0,
                min_filament_length=15.0
            )

            results[f'n_fil_z{z}'] = fil_result['n_filaments_real']
            results[f'len_mean_z{z}'] = fil_result['length_mean_real']
            results[f'len_max_z{z}'] = fil_result['length_max_real']
            results[f'n_halos_z{z}'] = fil_result.get('n_halos_plus', 0)

        except Exception as e:
            print(f"  Error analyzing {run_name} z={z}: {e}")
            results[f'n_fil_z{z}'] = -1
            results[f'len_mean_z{z}'] = 0.0
            results[f'len_max_z{z}'] = 0.0
            results[f'n_halos_z{z}'] = 0

    return results


def compute_filament_score(row):
    """Compute composite filament_score."""
    n_05 = row.get('n_fil_z0.5', 0)
    len_05 = row.get('len_mean_z0.5', 0)
    n_00 = row.get('n_fil_z0.0', 0)
    len_00 = row.get('len_mean_z0.0', 0)

    score_05 = n_05 * np.sqrt(max(len_05, 0)) if n_05 > 0 else 0
    score_00 = n_00 * np.sqrt(max(len_00, 0)) if n_00 > 0 else 0

    return 0.7 * score_05 + 0.3 * score_00


def main():
    print("="*70)
    print("TOUR 3 FILAMENT ANALYSIS — 5 runs trichotomie fine")
    print("2M particules — η ∈ {0.95, 0.975, 1.00, 1.025, 1.05}")
    print("="*70)

    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found")
        return

    df_basic = pd.read_csv(RESULTS_FILE)
    print(f"\nLoaded {len(df_basic)} runs from {RESULTS_FILE}")

    # Analyze filaments
    print("\nAnalyzing filaments at z=0.5 and z=0.0...")
    filament_results = []

    for _, row in df_basic.iterrows():
        run_name = row['run']
        run_dir = BASE_DIR / run_name

        print(f"  {run_name}...", end=" ", flush=True)
        result = analyze_filaments_for_run(run_name, run_dir)
        result['eta'] = row['eta']
        result['S_max'] = row['S_max']
        result['dcom_max'] = row['dcom_max']
        filament_results.append(result)

        n05 = result.get('n_fil_z0.5', 0)
        n00 = result.get('n_fil_z0.0', 0)
        print(f"z=0.5: {n05} filaments, z=0.0: {n00} filaments")

    df = pd.DataFrame(filament_results)
    df['filament_score'] = df.apply(compute_filament_score, axis=1)

    # Save
    out_file = BASE_DIR / "tour3_filament_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")

    # Summary
    print("\n" + "="*70)
    print("RÉSULTATS TOUR 3 — Convergence filament_score")
    print("="*70)
    print(f"\n{'η':<8} {'n_fil(z=0.5)':<14} {'len_mean':<10} {'fil_score':<12} {'S_max':<10}")
    print("-"*60)

    for _, row in df.sort_values('eta').iterrows():
        print(f"{row['eta']:<8.3f} {row['n_fil_z0.5']:<14.0f} "
              f"{row['len_mean_z0.5']:<10.1f} {row['filament_score']:<12.2f} "
              f"{row['S_max']:<10.4f}")

    # Find optimal
    best = df.loc[df['filament_score'].idxmax()]
    print(f"\n{'='*70}")
    print(f"η OPTIMAL = {best['eta']:.3f}")
    print(f"  filament_score = {best['filament_score']:.2f}")
    print(f"  n_filaments(z=0.5) = {best['n_fil_z0.5']:.0f}")
    print(f"  length_mean = {best['len_mean_z0.5']:.1f} Mpc")
    print(f"  S_max = {best['S_max']:.4f}")
    print(f"{'='*70}")

    # Comparison with Tour 2
    print("\nCOMPARAISON TOUR 2 (1M) vs TOUR 3 (2M) pour η=1.00:")
    tour2_file = Path("/mnt/T2/janus-sim/output/tour2_1M/tour2_filament_results.csv")
    if tour2_file.exists():
        df_t2 = pd.read_csv(tour2_file)
        t2_eta100 = df_t2[df_t2['eta'] == 1.0]
        if len(t2_eta100) > 0:
            t2_score = t2_eta100['filament_score'].mean()
            t3_eta100 = df[df['eta'] == 1.0]
            if len(t3_eta100) > 0:
                t3_score = t3_eta100['filament_score'].values[0]
                print(f"  Tour 2 (1M): fil_score = {t2_score:.2f}")
                print(f"  Tour 3 (2M): fil_score = {t3_score:.2f}")
                ratio = t3_score / t2_score if t2_score > 0 else 0
                print(f"  Ratio: {ratio:.2f}×")
                if ratio > 1.2:
                    print("  → Amélioration significative avec 2M particules")
                elif ratio > 0.8:
                    print("  → Convergence atteinte")
                else:
                    print("  → Régression - vérifier détection")

    # Plot
    plot_convergence(df)

    # Recommend Tour 4
    recommend_tour4(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Figures: {BASE_DIR}")
    print("="*70)


def plot_convergence(df):
    """Plot filament_score vs η."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = df.sort_values('eta')
    ax.plot(df_sorted['eta'], df_sorted['filament_score'],
            'o-', markersize=12, linewidth=2, color='blue', label='Tour 3 (2M)')
    ax.fill_between(df_sorted['eta'], 0, df_sorted['filament_score'],
                    alpha=0.2, color='blue')

    # Mark optimal
    best_idx = df_sorted['filament_score'].idxmax()
    best = df_sorted.loc[best_idx]
    ax.scatter([best['eta']], [best['filament_score']],
               c='red', s=200, marker='*', zorder=5,
               label=f'Optimal: η={best["eta"]:.3f}')

    ax.set_xlabel('η', fontsize=14)
    ax.set_ylabel('filament_score', fontsize=14)
    ax.set_title('Tour 3 — Trichotomie fine autour de η=1.00\n'
                 '2M particules, λ=25 Mpc, z_act=2.5', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out = BASE_DIR / 'tour3_convergence.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSauvegardé: {out}")
    plt.close()


def recommend_tour4(df):
    """Recommend Tour 4 parameters."""
    print("\n" + "="*70)
    print("RECOMMANDATIONS TOUR 4")
    print("="*70)

    best = df.loc[df['filament_score'].idxmax()]

    print(f"""
CONCLUSION TOUR 3 :
  η optimal confirmé = {best['eta']:.3f}
  filament_score = {best['filament_score']:.2f}
  n_filaments = {best['n_fil_z0.5']:.0f}

TOUR 4 — Validation haute résolution :
  η = {best['eta']:.3f} (fixé)
  λ = 25 Mpc (fixé)
  z_act = 2.5 (fixé)
  n_particles = 5 000 000

  Objectif : confirmer convergence et produire visualisation publication
""")


if __name__ == '__main__':
    main()
