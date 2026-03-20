"""
acceleration_janus.py
=====================
Test de la prédiction Janus : une particule m+ dans une région dense en m−
doit avoir une vitesse radiale plus élevée (expulsion plus forte).

Mesure : corrélation entre ρ_local_m− (densité m− dans sphère de rayon r_probe
autour de chaque m+) et v_rad (vitesse radiale de la m+ par rapport au centre
du halo).

Si r(ρ_local_m−, v_rad_m+) > 0 → preuve directe que la répulsion Janus
est proportionnelle à la densité locale de l'espèce opposée.

Usage :
    python acceleration_janus.py --snap snapshot.hdf5 --step 1500 --z 1.634
    python acceleration_janus.py --snap snapshot.npy   --step 500  --z 3.390

Options :
    --r_probe   Rayon de la sphère locale (Mpc, défaut 5.0)
    --r_halo    Rayon d'analyse autour du halo (Mpc, défaut 120.0)
    --n_sample  Nombre max de m+ à analyser (défaut 5000, -1 = toutes)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import argparse
import os

# ── Paramètres du halo ───────────────────────────────────────────────
HALO_POS = np.array([168.0, 127.0, 73.0])   # Mpc
BOX_SIZE = 256.0                              # Mpc

# ── Chargement snapshot (même logique que analyse_halo_spatial.py) ───
def load_snapshot(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.hdf5', '.h5']:
        import h5py
        with h5py.File(path, 'r') as f:
            # Affiche les clés disponibles si les clés standard échouent
            try:
                pos  = f['PartType0/Coordinates'][:]
                vel  = f['PartType0/Velocities'][:]
                mass = f['PartType0/Masses'][:]
            except KeyError:
                print("Clés HDF5 non standard. Clés disponibles :")
                f.visit(print)
                raise
        return pos, vel, mass
    elif ext == '.npy':
        data = np.load(path)
        return data[:, :3], data[:, 3:6], data[:, 6]
    else:
        # Binaire Rust : f32, 7 valeurs par particule
        data = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
        return (data[:, :3].astype(np.float64),
                data[:, 3:6].astype(np.float64),
                data[:, 6].astype(np.float64))


# ── Distance périodique ──────────────────────────────────────────────
def dist_periodic(pos, center, box):
    d = pos - center
    d -= box * np.round(d / box)
    return np.sqrt((d**2).sum(axis=1)), d


# ── Densité locale m− autour de chaque m+ ───────────────────────────
def compute_local_density(pos_query, pos_sources, box, r_probe):
    """
    Pour chaque point de pos_query, compte le nombre de points de
    pos_sources dans une sphère de rayon r_probe (conditions périodiques).
    Retourne le tableau de densités N/Mpc³.

    Optimisation : grid spatial pour éviter O(N²) brutal.
    """
    n_query   = len(pos_query)
    vol_probe = (4/3) * np.pi * r_probe**3
    counts    = np.zeros(n_query, dtype=np.int32)

    # Grille spatiale sur pos_sources
    cell_size = r_probe
    n_cells   = max(1, int(box / cell_size))
    cell_size = box / n_cells  # recalibré

    # Index de cellule pour chaque source
    idx_src = np.floor(pos_sources / cell_size).astype(int) % n_cells

    # Table de hachage : cell → liste d'indices
    from collections import defaultdict
    grid = defaultdict(list)
    for i, (cx, cy, cz) in enumerate(idx_src):
        grid[(cx, cy, cz)].append(i)

    # Pour chaque query, regarder les cellules voisines
    r2 = r_probe**2
    idx_qry = np.floor(pos_query / cell_size).astype(int) % n_cells

    for q in range(n_query):
        qx, qy, qz = idx_qry[q]
        pq = pos_query[q]
        # Voisinage cubique ±1 cellule
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    cell = ((qx+dx) % n_cells,
                            (qy+dy) % n_cells,
                            (qz+dz) % n_cells)
                    for src_i in grid[cell]:
                        d = pos_sources[src_i] - pq
                        d -= box * np.round(d / box)
                        if (d**2).sum() < r2:
                            counts[q] += 1

    return counts / vol_probe


# ── Analyse principale ───────────────────────────────────────────────
def analyse(snap_path, step, z, r_probe, r_halo, n_sample):

    print(f"Chargement : {snap_path}")
    pos, vel, mass = load_snapshot(snap_path)

    sign   = np.sign(mass)
    mask_m = sign < 0
    mask_p = sign > 0

    # Coordonnées corrigées périodiquement (centrées sur le halo)
    r_all, dr_all = dist_periodic(pos, HALO_POS, BOX_SIZE)
    pos_corr = HALO_POS + dr_all   # coordonnées sans saut périodique

    # Sélection dans r_halo
    in_halo_m = (r_all < r_halo) & mask_m
    in_halo_p = (r_all < r_halo) & mask_p

    pos_m  = pos_corr[in_halo_m]
    pos_p  = pos_corr[in_halo_p]
    vel_p  = vel[in_halo_p]
    r_p    = r_all[in_halo_p]
    dr_p   = dr_all[in_halo_p]

    N_m = len(pos_m)
    N_p = len(pos_p)
    print(f"Dans r<{r_halo} Mpc : N−={N_m:,} | N+={N_p:,}")

    if N_p == 0:
        print("Aucune particule m+ dans ce rayon. Abandon.")
        return
    if N_m == 0:
        print("Aucune particule m− dans ce rayon. Abandon.")
        return

    # Sous-échantillonnage si n_sample > 0
    if n_sample > 0 and N_p > n_sample:
        idx = np.random.default_rng(42).choice(N_p, n_sample, replace=False)
        print(f"Sous-échantillonnage : {n_sample}/{N_p} m+ analysées")
    else:
        idx = np.arange(N_p)

    pos_p_s  = pos_p[idx]
    vel_p_s  = vel_p[idx]
    r_p_s    = r_p[idx]
    dr_p_s   = dr_p[idx]

    # Vitesse radiale : v · r_hat (>0 = fuite du halo)
    r_safe  = np.maximum(r_p_s, 0.01)
    r_hat   = dr_p_s / r_safe[:, None]
    v_rad_p = (vel_p_s * r_hat).sum(axis=1)

    # Densité locale m− autour de chaque m+ sélectionnée
    print(f"Calcul densité locale m− (r_probe={r_probe} Mpc) pour {len(idx)} m+...")
    rho_local = compute_local_density(pos_p_s, pos_m, BOX_SIZE, r_probe)
    print("Calcul terminé.")

    # ── Statistiques ─────────────────────────────────────────────────
    # Corrélation globale ρ_local_m− vs v_rad_m+
    r_pearson, p_pearson = stats.pearsonr(rho_local, v_rad_p)
    r_spearman, p_spearman = stats.spearmanr(rho_local, v_rad_p)

    print(f"\n=== CORRÉLATION ρ_local_m− → v_rad_m+ ===")
    print(f"Pearson  r = {r_pearson:.4f}  p = {p_pearson:.2e}")
    print(f"Spearman r = {r_spearman:.4f}  p = {p_spearman:.2e}")
    if r_pearson > 0.3 and p_pearson < 0.01:
        print("→ CORRÉLATION POSITIVE SIGNIFICATIVE : répulsion Janus confirmée")
    elif r_pearson > 0 and p_pearson < 0.05:
        print("→ Tendance positive faible")
    else:
        print("→ Corrélation non significative à ce step")

    # Binning : v_rad moyen par quartile de ρ_local
    quartiles = np.percentile(rho_local, [0, 25, 50, 75, 100])
    q_labels  = ['Q1\n(ρ faible)', 'Q2', 'Q3', 'Q4\n(ρ fort)']
    vrad_q    = []
    vrad_q_std = []
    for i in range(4):
        mask_q = (rho_local >= quartiles[i]) & (rho_local < quartiles[i+1])
        if mask_q.sum() > 0:
            vrad_q.append(v_rad_p[mask_q].mean())
            vrad_q_std.append(v_rad_p[mask_q].std() / np.sqrt(mask_q.sum()))
        else:
            vrad_q.append(np.nan)
            vrad_q_std.append(np.nan)

    print(f"\nV_rad moyen par quartile de ρ_local_m− :")
    for i, (lbl, v, e) in enumerate(zip(q_labels, vrad_q, vrad_q_std)):
        print(f"  {lbl.replace(chr(10),' ')}: v_rad = {v:.1f} ± {e:.1f}")

    # ── Figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13))
    title = f"Répulsion Janus : ρ_local(m−) → v_rad(m+)\nstep {step}"
    if z is not None:
        title += f"  z={z:.3f}"
    title += f"  |  r_probe={r_probe} Mpc  |  N+ analysées={len(idx)}"
    fig.suptitle(title, fontsize=12, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Scatter ρ_local vs v_rad (avec régression)
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(rho_local, v_rad_p, c=r_p_s, cmap='plasma_r',
                     s=6, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax1, label='r (Mpc)')
    # Régression linéaire
    if len(rho_local) > 2:
        m_lin, b_lin, _, _, _ = stats.linregress(rho_local, v_rad_p)
        x_lin = np.linspace(rho_local.min(), rho_local.max(), 100)
        ax1.plot(x_lin, m_lin * x_lin + b_lin, 'k-', lw=1.8,
                 label=f'r={r_pearson:.3f}  p={p_pearson:.1e}')
    ax1.axhline(0, color='gray', lw=0.8, ls='--')
    ax1.set_xlabel('ρ_local m− (N/Mpc³)')
    ax1.set_ylabel('v_rad m+ (>0 = fuite)')
    ax1.set_title('Scatter : densité locale m−\nvs vitesse de fuite m+', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. V_rad moyen par quartile de ρ_local (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    colors_q = ['#2196F3', '#64B5F6', '#EF9A9A', '#F44336']
    bars = ax2.bar(range(4), vrad_q, yerr=vrad_q_std,
                   color=colors_q, alpha=0.85, capsize=5, width=0.6)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(q_labels, fontsize=9)
    ax2.set_ylabel('v_rad moyen m+')
    ax2.set_title('V_rad par quartile de ρ_local_m−\n(Q4 = entouré de m−)', fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    # Annotation tendance
    if not np.isnan(vrad_q[0]) and not np.isnan(vrad_q[3]):
        delta = vrad_q[3] - vrad_q[0]
        sign_txt = "↑ plus rapide" if delta > 0 else "↓ moins rapide"
        ax2.text(0.5, 0.95, f'Δv_rad Q4−Q1 = {delta:.1f} {sign_txt}',
                 transform=ax2.transAxes, ha='center', va='top',
                 fontsize=9, color='darkred', fontweight='bold')

    # 3. Distribution de ρ_local (histogramme)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(rho_local[rho_local > 0], bins=40, color='steelblue',
             alpha=0.75, edgecolor='white', lw=0.5)
    for q in quartiles[1:-1]:
        ax3.axvline(q, color='red', ls='--', lw=1)
    ax3.set_xlabel('ρ_local m− (N/Mpc³)')
    ax3.set_ylabel('N particules m+')
    ax3.set_title('Distribution de ρ_local m−\naround each m+', fontsize=9)
    ax3.grid(alpha=0.3)

    # 4. V_rad vs rayon r (coloré par ρ_local)
    ax4 = fig.add_subplot(gs[1, 0])
    # Colormap discret : Q1 bleu, Q4 rouge
    q_color = np.digitize(rho_local, quartiles[1:-1])  # 0,1,2,3
    cmap4   = plt.cm.RdBu_r
    sc4 = ax4.scatter(r_p_s, v_rad_p, c=q_color, cmap=cmap4,
                      s=6, alpha=0.5, vmin=0, vmax=3, rasterized=True)
    cbar4 = plt.colorbar(sc4, ax=ax4, ticks=[0, 1, 2, 3])
    cbar4.set_ticklabels(['Q1\n(faible)', 'Q2', 'Q3', 'Q4\n(fort)'])
    ax4.axhline(0, color='gray', lw=0.8, ls='--')
    ax4.set_xlabel('r (Mpc)')
    ax4.set_ylabel('v_rad m+')
    ax4.set_title('V_rad vs rayon\ncoloré par quartile ρ_local_m−', fontsize=9)
    ax4.grid(alpha=0.3)

    # 5. Profil v_rad moyen vs r, séparé par ρ_local haut/bas
    ax5 = fig.add_subplot(gs[1, 1])
    r_bins_plot = np.linspace(0, r_halo, 16)
    r_mid_plot  = 0.5 * (r_bins_plot[:-1] + r_bins_plot[1:])
    median_rho  = np.median(rho_local)
    mask_hi_rho = rho_local > median_rho
    mask_lo_rho = rho_local <= median_rho

    vrad_hi = []
    vrad_lo = []
    for i in range(len(r_bins_plot) - 1):
        in_shell = (r_p_s >= r_bins_plot[i]) & (r_p_s < r_bins_plot[i+1])
        hi = in_shell & mask_hi_rho
        lo = in_shell & mask_lo_rho
        vrad_hi.append(v_rad_p[hi].mean() if hi.sum() > 2 else np.nan)
        vrad_lo.append(v_rad_p[lo].mean() if lo.sum() > 2 else np.nan)

    ax5.plot(r_mid_plot, vrad_hi, 'r-o', ms=5, lw=1.8,
             label=f'ρ_local > médiane ({median_rho:.3f})')
    ax5.plot(r_mid_plot, vrad_lo, 'b-o', ms=5, lw=1.8,
             label=f'ρ_local ≤ médiane')
    ax5.axhline(0, color='gray', lw=0.8, ls='--')
    ax5.set_xlabel('r (Mpc)')
    ax5.set_ylabel('v_rad moyen m+')
    ax5.set_title('V_rad(r) : m+ dans zone dense vs vide\n(rouge = entouré de m−)', fontsize=9)
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # 6. Résumé statistique — texte
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = (
        f"RÉSUMÉ — step {step}"
        + (f"  z={z:.3f}" if z else "")
        + f"\n{'─'*38}\n"
        f"N m− analysées    : {N_m:,}\n"
        f"N m+ analysées    : {len(idx):,}\n"
        f"r_probe           : {r_probe} Mpc\n"
        f"ρ_local médiane   : {np.median(rho_local):.4f} N/Mpc³\n"
        f"ρ_local max       : {rho_local.max():.4f} N/Mpc³\n\n"
        f"Pearson  r = {r_pearson:+.4f}\n"
        f"         p = {p_pearson:.2e}\n"
        f"Spearman r = {r_spearman:+.4f}\n"
        f"         p = {p_spearman:.2e}\n\n"
        f"v_rad Q1 (ρ faible): {vrad_q[0]:+.1f}\n"
        f"v_rad Q4 (ρ fort)  : {vrad_q[3]:+.1f}\n"
        f"Δv_rad Q4−Q1       : {vrad_q[3]-vrad_q[0]:+.1f}\n\n"
    )
    if r_pearson > 0.3 and p_pearson < 0.01:
        summary += "✓ RÉPULSION JANUS CONFIRMÉE\n  (corrélation positive significative)"
    elif r_pearson > 0:
        summary += "~ Tendance positive (r > 0)"
    else:
        summary += "✗ Corrélation non significative"

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    out = f"acceleration_janus_step{step:04d}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure sauvegardée : {out}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test répulsion Janus : ρ_local(m−) → v_rad(m+)')
    parser.add_argument('--snap',     required=True,  help='Snapshot')
    parser.add_argument('--step',     type=int,   default=0)
    parser.add_argument('--z',        type=float, default=None)
    parser.add_argument('--r_probe',  type=float, default=5.0,
                        help='Rayon sphère locale (Mpc, défaut 5.0)')
    parser.add_argument('--r_halo',   type=float, default=120.0,
                        help='Rayon analyse autour du halo (Mpc, défaut 120.0)')
    parser.add_argument('--n_sample', type=int,   default=5000,
                        help='N max de m+ à analyser (-1 = toutes)')
    args = parser.parse_args()

    np.random.seed(42)
    analyse(args.snap, args.step, args.z,
            args.r_probe, args.r_halo, args.n_sample)
