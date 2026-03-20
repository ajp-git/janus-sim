"""
acceleration_multiprobe.py
==========================
Analyse la corrélation ρ_local(m−) → v_rad(m+) pour plusieurs rayons de sonde
simultanément, en UN SEUL passage sur la grille spatiale.

Conçu pour les snapshots précoces (steps 50–200) où l'expulsion est en cours.

Usage :
    python acceleration_multiprobe.py --snap snapshot_0100.hdf5 --step 100 --z 4.631
    python acceleration_multiprobe.py --snap snap.npy --step 200 --z 4.284 --n_sample 8000

Options :
    --r_probes   Rayons de sonde en Mpc séparés par virgule (défaut: 3,5,10,15,20)
    --r_halo     Rayon d'analyse autour du halo (Mpc, défaut 120)
    --n_sample   N max de m+ à analyser (-1 = toutes, défaut 5000)
    --halo_pos   Position du halo "x,y,z" (défaut: 168,127,73)
    --box        Taille de la boîte en Mpc (défaut: 256)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import defaultdict
import argparse
import os

# ── Paramètres par défaut ────────────────────────────────────────────
DEFAULT_HALO_POS = np.array([168.0, 127.0, 73.0])
DEFAULT_BOX      = 256.0
DEFAULT_R_PROBES = [3.0, 5.0, 10.0, 15.0, 20.0]


# ── Chargement snapshot ──────────────────────────────────────────────
def load_snapshot(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.hdf5', '.h5']:
        import h5py
        with h5py.File(path, 'r') as f:
            try:
                pos  = f['PartType0/Coordinates'][:]
                vel  = f['PartType0/Velocities'][:]
                mass = f['PartType0/Masses'][:]
            except KeyError:
                print("Clés HDF5 non standard. Disponibles :")
                f.visit(print)
                raise
        return pos.astype(np.float64), vel.astype(np.float64), mass.astype(np.float64)
    elif ext == '.npy':
        data = np.load(path)
        # shape attendue : (N, 7) → x y z vx vy vz mass_signed
        return data[:, :3].astype(np.float64), \
               data[:, 3:6].astype(np.float64), \
               data[:, 6].astype(np.float64)
    else:
        # Binaire Rust : float32, 7 valeurs par particule
        data = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
        return data[:, :3].astype(np.float64), \
               data[:, 3:6].astype(np.float64), \
               data[:, 6].astype(np.float64)


# ── Distance avec conditions périodiques ─────────────────────────────
def dist_periodic(pos, center, box):
    d = pos - center
    d -= box * np.round(d / box)
    return np.sqrt((d * d).sum(axis=1)), d


# ── Densité locale multi-probe (un seul passage grille) ─────────────
def compute_multi_density(pos_query, pos_sources, box, r_probes):
    """
    Pour chaque point de pos_query, compte le nombre de pos_sources
    dans une sphère de rayon r_probe (pour chaque r_probe), avec
    conditions périodiques.

    Stratégie : grille au r_probe maximum → un seul parcours,
    seuillage pour chaque r_probe à la volée.

    Retourne : densités shape (N_query, N_probes) en N/Mpc³
    """
    r_max      = max(r_probes)
    r_max_sq   = r_max * r_max
    n_query    = len(pos_query)
    n_probes   = len(r_probes)
    r_probes_sq = [r * r for r in r_probes]

    # Grille spatiale — cell_size = r_max
    n_cells   = max(1, int(box / r_max))
    cell_size = box / n_cells

    # Remplissage grille des sources
    idx_src = (np.floor(pos_sources / cell_size).astype(int)) % n_cells
    grid = defaultdict(list)
    for i in range(len(pos_sources)):
        grid[(idx_src[i, 0], idx_src[i, 1], idx_src[i, 2])].append(i)

    # Index cellule des queries
    idx_qry = (np.floor(pos_query / cell_size).astype(int)) % n_cells

    counts = np.zeros((n_query, n_probes), dtype=np.int32)

    for q in range(n_query):
        qx = idx_qry[q, 0]
        qy = idx_qry[q, 1]
        qz = idx_qry[q, 2]
        pq = pos_query[q]

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = ((qx + dx) % n_cells,
                            (qy + dy) % n_cells,
                            (qz + dz) % n_cells)
                    for src_i in grid[cell]:
                        # Distance périodique
                        dx_ = pos_sources[src_i, 0] - pq[0]
                        dy_ = pos_sources[src_i, 1] - pq[1]
                        dz_ = pos_sources[src_i, 2] - pq[2]
                        dx_ -= box * round(dx_ / box)
                        dy_ -= box * round(dy_ / box)
                        dz_ -= box * round(dz_ / box)
                        d2 = dx_*dx_ + dy_*dy_ + dz_*dz_
                        if d2 < r_max_sq:
                            for k in range(n_probes):
                                if d2 < r_probes_sq[k]:
                                    counts[q, k] += 1

    # Conversion en densité
    densities = np.empty((n_query, n_probes), dtype=np.float64)
    for k, r in enumerate(r_probes):
        vol = (4.0 / 3.0) * np.pi * r * r * r
        densities[:, k] = counts[:, k] / vol

    return densities


# ── Analyse principale ───────────────────────────────────────────────
def analyse(snap_path, step, z, r_probes, r_halo, n_sample,
            halo_pos, box):

    print(f"\nChargement : {snap_path}")
    pos, vel, mass = load_snapshot(snap_path)

    sign   = np.sign(mass)
    mask_m = sign < 0
    mask_p = sign > 0

    # Coordonnées corrigées périodiquement (centrées sur le halo)
    r_all, dr_all = dist_periodic(pos, halo_pos, box)
    pos_corr = halo_pos + dr_all   # sans saut périodique

    # Sélection dans r_halo
    in_m = (r_all < r_halo) & mask_m
    in_p = (r_all < r_halo) & mask_p

    pos_m = pos_corr[in_m]
    pos_p = pos_corr[in_p]
    vel_p = vel[in_p]
    r_p   = r_all[in_p]
    dr_p  = dr_all[in_p]

    N_m = len(pos_m)
    N_p = len(pos_p)
    print(f"Dans r<{r_halo} Mpc : N−={N_m:,}  N+={N_p:,}")

    if N_p == 0:
        print("Aucune m+ dans ce rayon — abandon.")
        return
    if N_m == 0:
        print("Aucune m− dans ce rayon — abandon.")
        return

    # Sous-échantillonnage m+
    rng = np.random.default_rng(42)
    if 0 < n_sample < N_p:
        idx = rng.choice(N_p, n_sample, replace=False)
        print(f"Sous-échantillonnage : {n_sample}/{N_p} m+")
    else:
        idx = np.arange(N_p)

    pos_p_s = pos_m[idx] if False else pos_p[idx]   # coordonnées corrigées
    vel_p_s = vel_p[idx]
    r_p_s   = r_p[idx]
    dr_p_s  = dr_p[idx]

    # Vitesse radiale v · r̂ par rapport au halo
    r_safe  = np.maximum(r_p_s, 0.01)
    r_hat   = dr_p_s / r_safe[:, None]
    v_rad   = (vel_p_s * r_hat).sum(axis=1)

    # Densités locales multi-probe
    n_s = len(idx)
    print(f"Calcul densités locales pour {n_s} m+ et {N_m:,} m− "
          f"| r_probes={r_probes} Mpc ...")
    rho = compute_multi_density(pos_p_s, pos_m, box, r_probes)
    print("Calcul terminé.")

    # Vérification signal : fraction de m+ avec rho=0 par probe
    for k, r in enumerate(r_probes):
        frac_zero = (rho[:, k] == 0).mean()
        print(f"  r_probe={r:4.1f} Mpc : ρ=0 pour {frac_zero*100:.1f}% des m+ "
              f"| ρ_max={rho[:,k].max():.4f} N/Mpc³")

    # Corrélations Pearson et Spearman pour chaque probe
    pearson_r  = []
    pearson_p  = []
    spearman_r = []
    spearman_p = []

    for k in range(len(r_probes)):
        rho_k = rho[:, k]
        # Seulement les m+ avec ρ > 0 pour la corrélation
        valid = rho_k > 0
        if valid.sum() > 10:
            pr, pp = stats.pearsonr(rho_k[valid], v_rad[valid])
            sr, sp = stats.spearmanr(rho_k[valid], v_rad[valid])
        else:
            pr = pp = sr = sp = np.nan
        pearson_r.append(pr)
        pearson_p.append(pp)
        spearman_r.append(sr)
        spearman_p.append(sp)

    # Résultats console
    print(f"\n{'r_probe':>8} {'Pearson r':>10} {'p':>10} {'Spearman r':>10} {'p':>10}")
    print("─" * 55)
    for k, r in enumerate(r_probes):
        pr = f"{pearson_r[k]:+.4f}" if not np.isnan(pearson_r[k]) else "   NaN"
        pp = f"{pearson_p[k]:.2e}"  if not np.isnan(pearson_p[k]) else "   NaN"
        sr = f"{spearman_r[k]:+.4f}" if not np.isnan(spearman_r[k]) else "   NaN"
        sp = f"{spearman_p[k]:.2e}"  if not np.isnan(spearman_p[k]) else "   NaN"
        print(f"{r:>8.1f} {pr:>10} {pp:>10} {sr:>10} {sp:>10}")

    # Meilleur r_probe (Pearson max)
    valid_r = [(k, pearson_r[k]) for k in range(len(r_probes))
               if not np.isnan(pearson_r[k])]
    best_k  = max(valid_r, key=lambda x: x[1])[0] if valid_r else 0
    best_r  = r_probes[best_k]
    print(f"\nMeilleur r_probe : {best_r} Mpc (Pearson r={pearson_r[best_k]:.4f})")

    # ── Figure ───────────────────────────────────────────────────────
    n_p = len(r_probes)
    n_cols = 3
    n_rows = 2 + int(np.ceil(n_p / n_cols))   # 2 rangées résumé + rangées scatter

    fig = plt.figure(figsize=(18, 5 * n_rows))
    z_str = f"z={z:.3f}" if z is not None else ""
    fig.suptitle(
        f"Répulsion Janus — ρ_local(m−) → v_rad(m+) | "
        f"step {step}  {z_str}\n"
        f"N−={N_m:,}  N+ analysées={n_s}  "
        f"r_probes={r_probes} Mpc",
        fontsize=11, fontweight='bold')

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.38)

    # ── Rangée 0 : Pearson r et Spearman r vs r_probe ────────────────
    ax_pr = fig.add_subplot(gs[0, 0])
    colors_pr = ['crimson' if not np.isnan(p) and p < 0.05
                 else 'steelblue' for p in pearson_p]
    ax_pr.bar(r_probes, pearson_r, color=colors_pr, alpha=0.8, width=1.8)
    ax_pr.axhline(0, color='k', lw=0.8)
    ax_pr.set_xlabel('r_probe (Mpc)')
    ax_pr.set_ylabel('Pearson r')
    ax_pr.set_title('Pearson r vs r_probe\n(rouge = p<0.05)', fontsize=9)
    ax_pr.grid(alpha=0.3, axis='y')
    for k, (r, rv, pv) in enumerate(zip(r_probes, pearson_r, pearson_p)):
        if not np.isnan(rv):
            ax_pr.text(r, rv + 0.005 * np.sign(rv + 1e-9),
                       f'{rv:.3f}', ha='center', fontsize=7)

    ax_sr = fig.add_subplot(gs[0, 1])
    colors_sr = ['crimson' if not np.isnan(p) and p < 0.05
                 else 'steelblue' for p in spearman_p]
    ax_sr.bar(r_probes, spearman_r, color=colors_sr, alpha=0.8, width=1.8)
    ax_sr.axhline(0, color='k', lw=0.8)
    ax_sr.set_xlabel('r_probe (Mpc)')
    ax_sr.set_ylabel('Spearman r')
    ax_sr.set_title('Spearman r vs r_probe\n(rouge = p<0.05)', fontsize=9)
    ax_sr.grid(alpha=0.3, axis='y')

    # Résumé texte
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis('off')
    lines = [
        f"RÉSUMÉ — step {step}  {z_str}",
        "─" * 36,
        f"N m−   : {N_m:,}",
        f"N m+ analysées : {n_s:,}",
        f"r_halo : {r_halo} Mpc",
        "",
    ]
    for k, r in enumerate(r_probes):
        pr = pearson_r[k]
        pp = pearson_p[k]
        if np.isnan(pr):
            lines.append(f"r={r:4.1f}: r=NaN (ρ=0 partout)")
        else:
            sig = "✓" if pp < 0.05 else "~" if pp < 0.2 else "✗"
            lines.append(f"r={r:4.1f}: r={pr:+.3f}  p={pp:.1e} {sig}")
    lines += ["", f"Meilleur r_probe : {best_r} Mpc",
              f"r_Pearson max    : {pearson_r[best_k]:.4f}" if not np.isnan(pearson_r[best_k]) else "NaN"]

    ax_txt.text(0.03, 0.97, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ── Rangée 1 : v_rad(r) dense vs vide pour chaque probe ──────────
    r_bins_plot = np.linspace(0, r_halo, 13)
    r_mid_plot  = 0.5 * (r_bins_plot[:-1] + r_bins_plot[1:])

    for k, r in enumerate(r_probes):
        col = k % n_cols
        row = 1 + k // n_cols
        ax = fig.add_subplot(gs[row, col])

        rho_k  = rho[:, k]
        median = np.median(rho_k)
        hi     = rho_k > median
        lo     = ~hi

        vrad_hi, vrad_lo = [], []
        for i in range(len(r_bins_plot) - 1):
            in_sh = (r_p_s >= r_bins_plot[i]) & (r_p_s < r_bins_plot[i+1])
            h = in_sh & hi
            l = in_sh & lo
            vrad_hi.append(v_rad[h].mean() if h.sum() > 2 else np.nan)
            vrad_lo.append(v_rad[l].mean() if l.sum() > 2 else np.nan)

        ax.plot(r_mid_plot, vrad_hi, 'r-o', ms=4, lw=1.6,
                label=f'ρ > med ({median:.3f})')
        ax.plot(r_mid_plot, vrad_lo, 'b-o', ms=4, lw=1.6,
                label=f'ρ ≤ med')
        ax.axhline(0, color='gray', lw=0.8, ls='--')

        pr = pearson_r[k]
        sig_str = f"r={pr:+.3f}" if not np.isnan(pr) else "r=NaN"
        ax.set_title(f'r_probe = {r} Mpc  |  {sig_str}', fontsize=9)
        ax.set_xlabel('r (Mpc)', fontsize=8)
        ax.set_ylabel('v_rad moyen m+', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Scatter meilleur probe (dernier panneau si disponible)
    remaining_slot = (1 + n_p // n_cols, n_p % n_cols)
    if remaining_slot[1] < n_cols and remaining_slot[0] < n_rows:
        ax_sc = fig.add_subplot(gs[remaining_slot[0], remaining_slot[1]])
        rho_best = rho[:, best_k]
        valid    = rho_best > 0
        if valid.sum() > 10:
            sc = ax_sc.scatter(rho_best[valid], v_rad[valid],
                               c=r_p_s[valid], cmap='plasma_r',
                               s=8, alpha=0.4, rasterized=True)
            plt.colorbar(sc, ax=ax_sc, label='r (Mpc)')
            m_lin, b_lin, _, _, _ = stats.linregress(rho_best[valid], v_rad[valid])
            x_lin = np.linspace(rho_best[valid].min(), rho_best[valid].max(), 100)
            ax_sc.plot(x_lin, m_lin*x_lin + b_lin, 'k-', lw=1.8,
                       label=f'r={pearson_r[best_k]:.3f}')
            ax_sc.axhline(0, color='gray', lw=0.8, ls='--')
            ax_sc.legend(fontsize=8)
        else:
            ax_sc.text(0.5, 0.5, 'ρ=0\npour toutes les m+',
                       ha='center', va='center', transform=ax_sc.transAxes,
                       fontsize=11, color='gray')
        ax_sc.set_title(f'Scatter meilleur probe r={best_r} Mpc', fontsize=9)
        ax_sc.set_xlabel('ρ_local m− (N/Mpc³)', fontsize=8)
        ax_sc.set_ylabel('v_rad m+', fontsize=8)
        ax_sc.grid(alpha=0.3)

    out = f"accel_multiprobe_step{step:04d}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure → {out}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap',     required=True)
    parser.add_argument('--step',     type=int,   default=0)
    parser.add_argument('--z',        type=float, default=None)
    parser.add_argument('--r_probes', type=str,   default='3,5,10,15,20')
    parser.add_argument('--r_halo',   type=float, default=120.0)
    parser.add_argument('--n_sample', type=int,   default=5000)
    parser.add_argument('--halo_pos', type=str,   default='168,127,73')
    parser.add_argument('--box',      type=float, default=256.0)
    args = parser.parse_args()

    r_probes = [float(x) for x in args.r_probes.split(',')]
    halo_pos = np.array([float(x) for x in args.halo_pos.split(',')])

    analyse(args.snap, args.step, args.z,
            r_probes, args.r_halo, args.n_sample,
            halo_pos, args.box)
