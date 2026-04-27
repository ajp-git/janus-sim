#!/usr/bin/env python3
"""Agrège les résultats JSON de plusieurs runs μ-scan et produit un plot de comparaison."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', required=True)
    ap.add_argument('--out-png', default='mu_scan_summary.png')
    ap.add_argument('--out-md', default='mu_scan_summary.md')
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    results = []
    for jp in sorted(json_dir.glob('results_mu_*.json')):
        with open(jp) as f:
            results.append(json.load(f))
    if not results:
        print("Aucun résultat trouvé")
        return

    results.sort(key=lambda r: r['mu'])

    mus = [r['mu'] for r in results]
    scores = [r['score'] for r in results]
    corrs = [r['corr_delta'] for r in results]
    rk_mid = [r['rk']['mid'] for r in results]
    voids = [r['void_fraction'] for r in results]
    var_ratios = [r['segregation']['ratio'] for r in results]
    vrms = [r['v_rms_km_s'] for r in results]
    nplus = [r['n_plus'] for r in results]

    # Figure 6 panneaux
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0,0].semilogx(mus, corrs, 'o-')
    axes[0,0].axhline(-0.07, color='red', ls='--', label='cible run μ=19 antérieur')
    axes[0,0].set_xlabel('μ'); axes[0,0].set_ylabel('corr(δ+, δ-)')
    axes[0,0].set_title('Décorrélation Janus')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].semilogx(mus, rk_mid, 'o-')
    axes[0,1].axhline(0, color='gray', alpha=0.5)
    axes[0,1].set_xlabel('μ'); axes[0,1].set_ylabel('r(k) mid')
    axes[0,1].set_title('Cross-correlation P(k) — échelles 0.05-0.2 h/Mpc')
    axes[0,1].grid(True, alpha=0.3)

    axes[0,2].semilogx(mus, voids, 'o-')
    axes[0,2].set_xlabel('μ'); axes[0,2].set_ylabel('fraction volume')
    axes[0,2].set_title('Void fraction (δ+ < -0.5)')
    axes[0,2].grid(True, alpha=0.3)

    axes[1,0].loglog(mus, var_ratios, 'o-')
    axes[1,0].set_xlabel('μ'); axes[1,0].set_ylabel('var(δ-) / var(δ+)')
    axes[1,0].set_title('Ratio variance — plus grand = plus ségrégué')
    axes[1,0].grid(True, alpha=0.3, which='both')

    axes[1,1].semilogx(mus, [v/299792 for v in vrms], 'o-')
    axes[1,1].axhline(0.1, color='red', ls='--', label='v/c = 0.1 (limite validité)')
    axes[1,1].set_xlabel('μ'); axes[1,1].set_ylabel('v_rms / c')
    axes[1,1].set_title('Stabilité')
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].semilogx(mus, scores, 'o-', color='darkred')
    axes[1,2].set_xlabel('μ'); axes[1,2].set_ylabel('score (plus bas = mieux)')
    axes[1,2].set_title('Score composite Janus')
    axes[1,2].grid(True, alpha=0.3)

    # Marquer le meilleur
    idx_best = np.argmin(scores)
    axes[1,2].plot(mus[idx_best], scores[idx_best], 'r*', markersize=20,
                   label=f'meilleur: μ={mus[idx_best]}')
    axes[1,2].legend()

    plt.suptitle(f'μ-scan summary — {len(results)} runs, z≈{results[0]["z_snap"]:.1f}')
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=100, bbox_inches='tight')
    plt.close()

    # Tableau markdown
    with open(args.out_md, 'w') as f:
        f.write(f"# μ-scan summary — {len(results)} runs\n\n")
        f.write(f"Analyse à z≈{results[0]['z_snap']:.2f}\n\n")
        f.write("| μ | N+ | corr(δ+,δ-) | r(k)mid | void_f | var_ratio | v/c | score |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['mu']:.0f} | {r['n_plus']:,} | {r['corr_delta']:+.4f} | "
                    f"{r['rk']['mid']:+.4f} | {r['void_fraction']:.3f} | "
                    f"{r['segregation']['ratio']:.2f} | {r['v_rms_km_s']/299792:.3f} | "
                    f"{r['score']:.3f} |\n")

        f.write(f"\n**Meilleur μ : {mus[idx_best]:.0f}** (score = {scores[idx_best]:.3f})\n\n")
        f.write(f"**Figure** : {args.out_png}\n")

    print(f"Summary écrit dans {args.out_png} et {args.out_md}")
    print(f"Meilleur μ: {mus[idx_best]}")


if __name__ == '__main__':
    main()
