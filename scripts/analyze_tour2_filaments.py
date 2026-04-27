#!/usr/bin/env python3
"""
Tour 2 Filament Analysis — 36 runs (4η × 3λ × 3z_act) at 1M particles

Objectif : filament_score RÉEL basé sur détection inter-halo
Focus sur snapshot z=0.5 (meilleure visibilité filaments)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/T2/janus-sim/optim')
from filament_metrics_v2 import load_snapshot, detect_interhalos_filaments

BASE_DIR = Path("/mnt/T2/janus-sim/output/tour2_1M")
RESULTS_FILE = BASE_DIR / "tour2_results.csv"
BOX_SIZE = 500.0

# Target redshifts for filament analysis
TARGET_REDSHIFTS = [0.5, 0.0]  # z=0.5 prioritaire


def find_snapshot(run_dir, target_z):
    """Find snapshot closest to target redshift."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.exists():
        return None

    # Pattern: snap_NNNNNN.bin
    snaps = list(snap_dir.glob("snap_*.bin"))
    if not snaps:
        return None

    # For z=0.5, try snap at ~2/3 of simulation
    # For z=0.0, try last snap
    if target_z == 0.0:
        return sorted(snaps)[-1]
    elif target_z == 0.5:
        # z=0.5 is roughly step 667 out of 1000 for z_start=5
        # Try to find snap closest to step 650-700
        for snap in sorted(snaps):
            step = int(snap.stem.split('_')[1])
            if 600 <= step <= 750:
                return snap
        # Fallback: try snap at index 2/3
        idx = int(len(snaps) * 2 / 3)
        return sorted(snaps)[min(idx, len(snaps)-1)]
    return sorted(snaps)[-1]


def analyze_filaments_for_run(run_name, run_dir):
    """Analyze filaments for a single run at multiple redshifts."""
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
                min_filament_length=15.0  # Plus strict pour 500 Mpc
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
    """
    Compute composite filament_score.

    Score = n_filaments * sqrt(length_mean) * weight(z)

    Focus sur z=0.5 car meilleure visibilité historique.
    """
    # z=0.5 contribue 70%, z=0.0 contribue 30%
    n_05 = row.get('n_fil_z0.5', 0)
    len_05 = row.get('len_mean_z0.5', 0)
    n_00 = row.get('n_fil_z0.0', 0)
    len_00 = row.get('len_mean_z0.0', 0)

    # Score components
    score_05 = n_05 * np.sqrt(max(len_05, 0)) if n_05 > 0 else 0
    score_00 = n_00 * np.sqrt(max(len_00, 0)) if n_00 > 0 else 0

    return 0.7 * score_05 + 0.3 * score_00


def main():
    print("="*70)
    print("TOUR 2 FILAMENT ANALYSIS — 36 runs (4η × 3λ × 3z_act)")
    print("1M particules — Objectif : filament_score réel")
    print("="*70)

    # Load basic results
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found")
        return

    df_basic = pd.read_csv(RESULTS_FILE)
    print(f"\nLoaded {len(df_basic)} runs from {RESULTS_FILE}")

    # Analyze filaments for each run
    print("\nAnalyzing filaments at z=0.5 and z=0.0...")
    filament_results = []

    for _, row in df_basic.iterrows():
        run_name = row['run']
        run_dir = BASE_DIR / run_name

        print(f"  {run_name}...", end=" ", flush=True)
        result = analyze_filaments_for_run(run_name, run_dir)
        result['eta'] = row['eta']
        result['lambda'] = row['lambda']
        result['z_act'] = row['z_act']
        result['S_max'] = row['S_max']
        result['dcom_max'] = row['dcom_max']
        filament_results.append(result)

        n05 = result.get('n_fil_z0.5', 0)
        n00 = result.get('n_fil_z0.0', 0)
        print(f"z=0.5: {n05} filaments, z=0.0: {n00} filaments")

    df = pd.DataFrame(filament_results)

    # Compute filament_score
    df['filament_score'] = df.apply(compute_filament_score, axis=1)

    # Save extended results
    out_file = BASE_DIR / "tour2_filament_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")

    # Analysis
    print("\n" + "="*70)
    print("FILAMENT DETECTION SUMMARY")
    print("="*70)

    # Stats by eta
    print(f"\n{'η':<6} {'n_fil(z=0.5)':<14} {'len_mean':<10} {'n_fil(z=0)':<12} {'fil_score':<12}")
    print("-"*60)

    for eta in sorted(df['eta'].unique()):
        sub = df[df['eta'] == eta]
        n05 = sub['n_fil_z0.5'].mean()
        len05 = sub['len_mean_z0.5'].mean()
        n00 = sub['n_fil_z0.0'].mean()
        score = sub['filament_score'].mean()
        print(f"{eta:<6.2f} {n05:<14.1f} {len05:<10.1f} {n00:<12.1f} {score:<12.2f}")

    # Top 10 by filament_score
    print("\n" + "="*70)
    print("TOP 10 PAR FILAMENT_SCORE RÉEL")
    print("="*70)

    top10 = df.nlargest(10, 'filament_score')
    print(f"{'#':<4} {'η':<6} {'λ':<5} {'z_act':<6} {'n_fil(0.5)':<12} "
          f"{'len_mean':<10} {'fil_score':<12} {'S_max':<8}")
    print("-"*70)

    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<4} {row['eta']:<6.2f} {row['lambda']:<5.0f} "
              f"{row['z_act']:<6.1f} {row['n_fil_z0.5']:<12.0f} "
              f"{row['len_mean_z0.5']:<10.1f} {row['filament_score']:<12.2f} "
              f"{row['S_max']:<8.4f}")

    # Correlation S_max vs filament_score
    corr = df['S_max'].corr(df['filament_score'])
    print(f"\nCorrélation S_max ↔ filament_score : {corr:.3f}")
    if abs(corr) < 0.3:
        print("  → S_max et filament_score sont INDÉPENDANTS")
        print("  → Tour 1 (basé sur S_max) ne prédit PAS les filaments")
    elif corr > 0.5:
        print("  → S_max et filament_score sont CORRÉLÉS positivement")
    else:
        print("  → S_max et filament_score sont ANTI-CORRÉLÉS")

    # Plot
    plot_filament_heatmaps(df)
    plot_filament_vs_smax(df)

    # Recommendations
    recommend_tour3(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Figures: {BASE_DIR}")
    print("="*70)


def plot_filament_heatmaps(df):
    """Heatmaps de filament_score par η × λ, un par z_act."""
    z_acts = sorted(df['z_act'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmax = df['filament_score'].max()
    vmin = 0

    for idx, z_act in enumerate(z_acts):
        ax = axes[idx]
        subset = df[df['z_act'] == z_act]

        pivot = subset.pivot_table(
            values='filament_score', index='lambda', columns='eta', aggfunc='mean')

        etas = pivot.columns.values
        lambdas = pivot.index.values
        eta_edges = np.append(etas - 0.05, etas[-1] + 0.05)
        lam_edges = np.append(lambdas - 5, lambdas[-1] + 5)

        mesh = ax.pcolormesh(eta_edges, lam_edges, pivot.values,
                             cmap='plasma', vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, ax=ax, label='filament_score')

        # Mark max
        best = subset.loc[subset['filament_score'].idxmax()]
        ax.scatter(best['eta'], best['lambda'],
                   c='lime', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel('η')
        ax.set_ylabel('λ (Mpc)')
        ax.set_title(f'z_act = {z_act}')

    plt.suptitle('Tour 2 — filament_score RÉEL (FOF + inter-halo)\n'
                 '1M particules, 500 Mpc', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = BASE_DIR / 'tour2_filament_heatmaps.png'
    plt.savefig(out, dpi=150)
    print(f"\nSauvegardé: {out}")
    plt.close()


def plot_filament_vs_smax(df):
    """Scatter plot filament_score vs S_max."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for eta in sorted(df['eta'].unique()):
        sub = df[df['eta'] == eta]
        ax.scatter(sub['S_max'], sub['filament_score'],
                   label=f'η={eta:.2f}', s=80, alpha=0.7)

    ax.set_xlabel('S_max (ségrégation)')
    ax.set_ylabel('filament_score (réel)')
    ax.set_title('Tour 2 — S_max vs filament_score\n'
                 '(vérification indépendance des métriques)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation
    corr = df['S_max'].corr(df['filament_score'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out = BASE_DIR / 'tour2_smax_vs_filament.png'
    plt.savefig(out, dpi=150)
    print(f"Sauvegardé: {out}")
    plt.close()


def recommend_tour3(df):
    """Recommendations for Tour 3."""
    print("\n" + "="*70)
    print("RECOMMANDATIONS TOUR 3")
    print("="*70)

    # Best by filament_score
    best = df.loc[df['filament_score'].idxmax()]

    # Check if any filaments were detected
    total_filaments = df['n_fil_z0.5'].sum() + df['n_fil_z0.0'].sum()

    if total_filaments == 0:
        print("""
AUCUN FILAMENT DÉTECTÉ dans Tour 2.

Actions possibles :
  1. Vérifier que les snapshots z=0.5 existent bien
  2. Baisser min_filament_length à 10 Mpc
  3. Tester z_act ∈ {0.5, 1.0} (activation plus tardive)
  4. Augmenter à 2M particules pour meilleure résolution
        """)
    else:
        print(f"""
FILAMENTS DÉTECTÉS — Paramètres optimaux :
  η*     = {best['eta']:.2f}
  λ*     = {best['lambda']:.0f} Mpc
  z_act* = {best['z_act']:.1f}

  filament_score = {best['filament_score']:.2f}
  n_filaments(z=0.5) = {best['n_fil_z0.5']:.0f}
  length_mean = {best['len_mean_z0.5']:.1f} Mpc

TOUR 3 — Trichotomie à 2M particules :
  Centrer sur η = {best['eta']:.2f} ± 0.05
  Grille fine : 5 valeurs de η autour de l'optimum

  Objectif : convergence du filament_score avec la résolution
        """)


if __name__ == '__main__':
    main()
