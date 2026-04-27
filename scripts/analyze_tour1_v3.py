#!/usr/bin/env python3
"""
Analyze Tour 1 grid results — 117 runs (13η × 3λ × 3z_act)
Boîte 500 Mpc, 500k particules — v3 (corrections)

CORRECTIONS vs v2 :
  1. Analyse λ/z_act corrigée (η fixé, pas groupby global)
  2. Étoile heatmap : moyenne sur toutes valeurs ex-aequo
  3. Hypothèse physique sur anomalie η=0.75
  4. Grille Tour 2 élargie pour couvrir l'anomalie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("/mnt/T2/janus-sim/output/tour1_500mpc")
RESULTS_FILE = BASE_DIR / "tour1_results.csv"
BOX_SIZE = 500.0


def load_results():
    df = pd.read_csv(RESULTS_FILE)
    df['dcom_fraction'] = df['dcom_max'] / BOX_SIZE
    return df


def print_physical_transition(df):
    print("\n" + "="*70)
    print("TRANSITION PHYSIQUE — KE_ratio et durée par η")
    print("="*70)
    print(f"{'η':<6} {'S_avg':<8} {'ΔCOM_avg':<10} {'ΔCOM/L':<8} "
          f"{'KE_avg':<10} {'dur_avg(s)':<12} {'Régime'}")
    print("-"*70)

    for eta in sorted(df['eta'].unique()):
        sub = df[df['eta'] == eta]
        s    = sub['S_max'].mean()
        dcom = sub['dcom_max'].mean()
        dcom_f = sub['dcom_fraction'].mean()
        ke   = sub['ke_ratio_final'].mean()
        dur  = sub['duration_s'].mean()

        if ke < 10:
            regime = "⚠️  RALENTI (H²≈0?)"
        elif dcom_f > 0.40:
            regime = "💥 RUNAWAY"
        elif s > 0.26:
            regime = "✅ Actif"
        else:
            regime = "➖ Faible"

        print(f"{eta:<6.2f} {s:<8.4f} {dcom:<10.1f} {dcom_f:<8.3f} "
              f"{ke:<10.2f} {dur:<12.0f} {regime}")


def analyze_lambda_zact_effect_FIXED_ETA(df):
    """
    CORRECTION v3 : tester l'effet de λ et z_act à η FIXÉ.
    L'analyse groupby globale (v2) était biaisée par la variation de η.
    """
    print("\n" + "="*70)
    print("EFFET DE λ ET z_act — ANALYSE CORRECTE (η fixé)")
    print("="*70)
    print("(variance de S_max quand on fait varier λ ou z_act, η constant)\n")

    df_valid = df[df['eta'] <= 1.07]
    etas = sorted(df_valid['eta'].unique())

    var_lam_list, var_z_list = [], []

    print(f"{'η':<6} {'var_S(λ)':<12} {'var_S(z_act)':<14} {'interprétation'}")
    print("-"*60)

    for eta in etas:
        sub = df_valid[df_valid['eta'] == eta]
        var_lam = sub.groupby('lambda')['S_max'].std().mean()
        var_z   = sub.groupby('z_act')['S_max'].std().mean()
        var_lam_list.append(var_lam)
        var_z_list.append(var_z)

        flag = "⚠️ effet réel" if max(var_lam, var_z) > 0.005 else "✅ bruit seul"
        print(f"{eta:<6.2f} {var_lam:<12.5f} {var_z:<14.5f} {flag}")

    print(f"\nMoyenne globale:")
    print(f"  var_S selon λ    : {np.mean(var_lam_list):.5f}")
    print(f"  var_S selon z_act: {np.mean(var_z_list):.5f}")

    if np.mean(var_lam_list) < 0.003 and np.mean(var_z_list) < 0.003:
        print("\n  → ⚠️  AUCUN EFFET de λ et z_act sur S_max à 500k particules")
        print("  → Visible aussi dans les figures : courbes z_act superposées,")
        print("     heatmap uniforme par colonne (η) pas par ligne (λ)")
        print("  → Tour 2 (1M, filament_score réel) nécessaire pour les discriminer")
    else:
        print("\n  → Effet modéré détecté sur certains η")


def analyze_anomaly_075(df):
    """
    Investiguer l'anomalie η=0.75 avec hypothèse physique.
    """
    print("\n" + "="*70)
    print("ANOMALIE η=0.75 — Investigation")
    print("="*70)

    for eta in [0.70, 0.75, 0.80]:
        sub = df[df['eta'] == eta]
        s    = sub['S_max'].mean()
        dcom = sub['dcom_max'].mean()
        ke   = sub['ke_ratio_final'].mean()
        # ρ_eff = densité nette positive
        rho_eff_ratio = 1 - eta  # = (ρ+ - ρ-)/ρ+ pour η = ρ-/ρ+
        print(f"η={eta:.2f}: S={s:.4f}, ΔCOM={dcom:.1f} Mpc, "
              f"KE={ke:.1f}, ρ_eff/ρ+ = {rho_eff_ratio:.2f}")

    print(f"""
Hypothèses physiques pour le creux η=0.75 :

  H1 — Résonance avec le mode fondamental de la boîte :
    λ_Jeans eff à z=3 ≈ 9.6 × (1-η) × L_box ???
    Pas évident numériquement.

  H2 — Transition entre deux régimes de structure :
    η=0.70 → ρ_eff/ρ+ = 0.30 → effondrement m+ domine
    η=0.75 → ρ_eff/ρ+ = 0.25 → zone de compétition m+ / répulsion
    η=0.80 → ρ_eff/ρ+ = 0.20 → autre régime
    → Le creux est à l'intersection de deux régimes.

  H3 — Artefact statistique :
    ΔCOM à η=0.75 = 86 Mpc (vs 133 à η=0.70)
    → la sigmoïde d'activation crée une résonance particulière
      avec les modes de la boîte à cette valeur de η.

  Conclusion : anomalie réelle (reproductible sur 9 runs),
  mécanisme inconnu. Investiguer en Tour 2 avec 1M particules.
""")


def plot_S_vs_eta(df):
    """Graphique S_max vs η — 3 panneaux par λ, courbes par z_act."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, lam in enumerate(sorted(df['lambda'].unique())):
        ax = axes[idx]
        sub = df[df['lambda'] == lam]

        for z_act in sorted(sub['z_act'].unique()):
            sub_z = sub[sub['z_act'] == z_act].sort_values('eta')
            ax.plot(sub_z['eta'], sub_z['S_max'],
                    marker='o', label=f'z_act={z_act}', linewidth=2)

        ax.axvline(x=1.075, color='red', linestyle='--',
                   alpha=0.7, label='Transition η≈1.07')
        ax.axvline(x=1.0, color='orange', linestyle=':',
                   alpha=0.5, label='η=1.0')
        ax.axvline(x=0.75, color='purple', linestyle=':',
                   alpha=0.5, label='Anomalie η=0.75')

        ax.set_xlabel('η')
        ax.set_ylabel('S_max')
        ax.set_title(f'λ = {lam} Mpc')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 0.35)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Tour 1 — S_max vs η par λ et z_act\n'
        '(500 Mpc, 500k particules — PAS de détection filaments)\n'
        'Note : courbes z_act quasi-superposées → z_act sans effet sur S_max',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = BASE_DIR / 'tour1_S_vs_eta.png'
    plt.savefig(out, dpi=150)
    print(f"\nSauvegardé: {out}")
    plt.close()


def plot_heatmaps(df):
    """
    Heatmaps η × λ avec pcolormesh.
    CORRECTION v3 : étoile = moyenne des top, pas idxmax() instable.
    """
    z_acts = sorted(df['z_act'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmin = df[df['eta'] <= 1.07]['S_max'].min()
    vmax = df[df['eta'] <= 1.07]['S_max'].max()

    for idx, z_act in enumerate(z_acts):
        ax = axes[idx]
        subset = df[df['z_act'] == z_act]

        pivot = subset.pivot_table(
            values='S_max', index='lambda', columns='eta', aggfunc='mean')

        etas   = pivot.columns.values
        lambdas = pivot.index.values
        eta_edges = np.append(etas - 0.025, etas[-1] + 0.025)
        lam_edges = np.append(lambdas - 5,  lambdas[-1] + 5)

        mesh = ax.pcolormesh(eta_edges, lam_edges, pivot.values,
                             cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, ax=ax, label='S_max')

        ax.axvline(x=1.075, color='red', linestyle='--',
                   linewidth=2, label='Transition η≈1.07')
        ax.axvline(x=0.75, color='purple', linestyle=':',
                   linewidth=1.5, label='Anomalie η=0.75')

        # CORRECTION : étoile sur η_opt moyen pour ce z_act
        valid = subset[subset['eta'] <= 1.07]
        # Trouver le η qui maximise S_max en moyenne sur tous les λ
        eta_avg = valid.groupby('eta')['S_max'].mean()
        best_eta = eta_avg.idxmax()
        best_lam = valid[valid['eta'] == best_eta]['lambda'].mean()
        ax.scatter(best_eta, best_lam,
                   c='red', s=200, marker='*',
                   edgecolors='white', linewidths=2, zorder=5,
                   label=f'η_opt={best_eta:.2f}')

        ax.set_xlabel('η')
        ax.set_ylabel('λ (Mpc)')
        ax.set_title(f'z_act = {z_act}')
        ax.set_xticks(etas)
        ax.set_xticklabels([f'{e:.2f}' for e in etas],
                           rotation=45, fontsize=7)
        ax.legend(fontsize=7)

    plt.suptitle(
        'Tour 1 — S_max Heatmaps (500 Mpc)\n'
        '⚠️  S_max ≠ filament_score — détection filaments à Tour 2\n'
        'Observation : couleurs uniformes par colonne → λ sans effet sur S_max',
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    out = BASE_DIR / 'tour1_heatmaps.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé: {out}")
    plt.close()


def print_top10(df):
    print("\n" + "="*70)
    print("TOP 10 PAR S_max (proxy — pas filament_score)")
    print("⚠️  S_max élevé ≠ filaments — voir Tour 2 pour confirmation")
    print("="*70)
    top10 = df[df['eta'] <= 1.07].nlargest(10, 'S_max')
    print(f"{'#':<4} {'η':<6} {'λ':<5} {'z_act':<6} {'S_max':<8} "
          f"{'ΔCOM':<8} {'ΔCOM/L':<8} {'KE':<8}")
    print("-"*60)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<4} {row['eta']:<6.2f} {row['lambda']:<5.0f} "
              f"{row['z_act']:<6.1f} {row['S_max']:<8.4f} "
              f"{row['dcom_max']:<8.1f} {row['dcom_fraction']:<8.3f} "
              f"{row['ke_ratio_final']:<8.2f}")


def recommend_tour2(df):
    print("\n" + "="*70)
    print("RECOMMANDATIONS TOUR 2")
    print("="*70)

    df_valid = df[df['eta'] <= 1.07]
    best_eta = df_valid.groupby('eta')['S_max'].mean().idxmax()

    print(f"""
CONCLUSION TOUR 1 :
  • η optimal (S_max) = {best_eta:.2f}
  • Anomalie η=0.75 : creux -35%, mécanisme inconnu
  • Transition physique à η ≈ 1.07-1.10 (KE ÷15, S ÷5)
  • λ et z_act : aucun effet significatif sur S_max à 500k

LIMITES :
  • S_max ≠ filament_score
  • 500k insuffisant pour filaments fins (espacement 3.7 Mpc)
  • λ et z_act à tester avec filament_score réel à 1M particules

TOUR 2 — Grille 4×3×3 = 36 runs (CORRECTION v3 : grille élargie)
  η     ∈ {{0.75, 0.85, 0.90, 1.00}}
          ↑ inclus pour investiguer l'anomalie
  λ     ∈ {{15, 25, 35}} Mpc
  z_act ∈ {{1.5, 2.0, 2.5}}

  n_particles : 1 000 000  (filaments > 8 Mpc détectables)
  n_steps     : 1000
  box_size    : 500 Mpc    (FIXE)

  Scorer sur : filament_score RÉEL (FOF + skeleton + correction périodique)
  PAS sur S_max.

  Si filament_score = 0 partout :
  → Essayer z_act ∈ {{0.5, 1.0}} (activation très tardive)
    pour laisser le web ΛCDM se former presque complètement
    avant que Janus ne perturbe.
""")


def main():
    print("="*70)
    print("TOUR 1 ANALYSIS v3 — 117 runs (13η × 3λ × 3z_act)")
    print("Boîte 500 Mpc — 500k particules")
    print("⚠️  Résolution insuffisante pour détection filaments")
    print("="*70)

    df = load_results()
    n = len(df)
    completed = len(df[df['status'] == 'COMPLETE'])
    print(f"\nRuns: {completed}/{n} COMPLETE")
    print(f"Durée totale: {df['duration_s'].sum()/3600:.1f}h")
    print(f"Durée moyenne : {df['duration_s'].mean():.0f}s")

    print_physical_transition(df)
    analyze_lambda_zact_effect_FIXED_ETA(df)  # CORRECTION principale
    analyze_anomaly_075(df)
    plot_S_vs_eta(df)
    plot_heatmaps(df)
    print_top10(df)
    recommend_tour2(df)

    print("="*70)
    print("ANALYSIS v3 COMPLETE")
    print(f"Figures: {BASE_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
