"""
gradient_janus.py
=================
Test direct de la répulsion Janus par alignement gradient/vitesse.

Pour chaque m+ : calcule le gradient de densité m− locale (∇ρ_m−)
par différences finies centrées, puis mesure :

    cos_θ = (∇ρ_m−) · v_m+ / (|∇ρ_m−| |v_m+|)

Interprétation :
    cos_θ < 0  →  m+ fuit la concentration m− = RÉPULSION JANUS
    cos_θ > 0  →  m+ va vers la concentration m− = attraction (physique standard)
    cos_θ = 0  →  mouvement perpendiculaire = pas de signal

Métrique clé : fraction de m+ avec cos_θ < 0
    > 50%  →  répulsion dominante (test binomial vs H0=0.5)
    = 50%  →  mouvement aléatoire

Usage :
    python gradient_janus.py --snap snapshot_0100.hdf5 --step 100 --z 4.631
    python gradient_janus.py --snap snapshot.npy --step 200 --z 4.284 --r_probe 15

Options :
    --r_probe    Rayon de la sphère locale (Mpc, défaut 10.0)
                 h = r_probe/2 pour les différences finies
    --r_halo     Rayon d'analyse autour du halo (Mpc, défaut 120.0)
    --n_sample   N max de m+ à analyser (-1 = toutes, défaut 5000)
    --halo_pos   Centre du halo "x,y,z" (défaut: 168,127,73)
    --box        Taille de la boîte en Mpc (défaut: 256)
    --min_grad   Norme minimale du gradient pour inclure une m+ (défaut: 1e-9)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import defaultdict
import argparse
import os

# ── Chargement snapshot ──────────────────────────────────────────────
def load_snapshot(path):
    """Format Janus: 8 bytes N (u64), then N × 28 bytes (x,y,z,vx,vy,vz,sign as f32)"""
    import struct
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
        return data[:,:3].astype(np.float64), data[:,3:6].astype(np.float64), data[:,6].astype(np.float64)
    elif ext == '.bin':
        # Format Janus: 8-byte header (u64 N), then N×28 bytes
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            raw = f.read(n * 28)
            data = np.frombuffer(raw, dtype=np.float32).reshape(n, 7)
        return data[:,:3].astype(np.float64), data[:,3:6].astype(np.float64), data[:,6].astype(np.float64)
    else:
        # Legacy format without header
        data = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
        return data[:,:3].astype(np.float64), data[:,3:6].astype(np.float64), data[:,6].astype(np.float64)


# ── Distance périodique ──────────────────────────────────────────────
def dist_periodic(pos, center, box):
    d = pos - center
    d -= box * np.round(d / box)
    return np.sqrt((d * d).sum(axis=1)), d


# ── Densité locale batch ─────────────────────────────────────────────
def compute_density_batch(pos_queries, pos_sources, box, r_probe):
    """
    Densité m− en N/Mpc³ pour un tableau de positions query.
    Grille spatiale pour éviter O(N²).
    """
    r2        = r_probe * r_probe
    n_q       = len(pos_queries)
    n_cells   = max(1, int(box / r_probe))
    cell_size = box / n_cells
    vol       = (4.0 / 3.0) * np.pi * r_probe**3

    idx_src = (np.floor(pos_sources / cell_size).astype(int)) % n_cells
    grid = defaultdict(list)
    for i in range(len(pos_sources)):
        grid[(idx_src[i, 0], idx_src[i, 1], idx_src[i, 2])].append(i)

    idx_qry = (np.floor(pos_queries / cell_size).astype(int)) % n_cells
    counts  = np.zeros(n_q, dtype=np.int32)

    for q in range(n_q):
        qx = idx_qry[q, 0]; qy = idx_qry[q, 1]; qz = idx_qry[q, 2]
        pq = pos_queries[q]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = ((qx+dx) % n_cells,
                            (qy+dy) % n_cells,
                            (qz+dz) % n_cells)
                    for src_i in grid[cell]:
                        ddx = pos_sources[src_i, 0] - pq[0]
                        ddy = pos_sources[src_i, 1] - pq[1]
                        ddz = pos_sources[src_i, 2] - pq[2]
                        ddx -= box * round(ddx / box)
                        ddy -= box * round(ddy / box)
                        ddz -= box * round(ddz / box)
                        if ddx*ddx + ddy*ddy + ddz*ddz < r2:
                            counts[q] += 1

    return counts / vol


# ── Gradient ρ_m− par différences finies centrées ───────────────────
def compute_gradient(pos_p, pos_m, box, r_probe):
    """
    Gradient ∇ρ_m− en chaque point de pos_p.
    h = r_probe/2 pour que les sphères déplacées chevauchent
    la sphère centrale et détectent les sources voisines.

    Retourne :
        grad    : shape (N, 3) — gradient non normalisé
        rho_c   : shape (N,)   — densité au point central
    """
    h  = r_probe / 2.0
    N  = len(pos_p)

    # 7 positions par particule : ±x, ±y, ±z, centre
    offsets = np.array([
        [ h,  0,  0], [-h,  0,  0],
        [ 0,  h,  0], [ 0, -h,  0],
        [ 0,  0,  h], [ 0,  0, -h],
        [ 0,  0,  0],
    ])

    all_queries = np.empty((7 * N, 3))
    for i, off in enumerate(offsets):
        all_queries[i*N:(i+1)*N] = (pos_p + off) % box

    rho_all = compute_density_batch(all_queries, pos_m, box, r_probe)
    rho     = rho_all.reshape(7, N)

    grad       = np.empty((N, 3))
    grad[:, 0] = (rho[0] - rho[1]) / (2.0 * h)
    grad[:, 1] = (rho[2] - rho[3]) / (2.0 * h)
    grad[:, 2] = (rho[4] - rho[5]) / (2.0 * h)
    rho_c      = rho[6]

    return grad, rho_c


# ── Analyse principale ───────────────────────────────────────────────
def analyse(snap_path, step, z, r_probe, r_halo, n_sample,
            halo_pos, box, min_grad):

    print(f"\nChargement : {snap_path}")
    pos, vel, mass = load_snapshot(snap_path)

    sign   = np.sign(mass)
    mask_m = sign < 0
    mask_p = sign > 0

    r_all, dr_all = dist_periodic(pos, halo_pos, box)
    pos_corr = halo_pos + dr_all   # coordonnées sans saut périodique

    in_m = (r_all < r_halo) & mask_m
    in_p = (r_all < r_halo) & mask_p

    pos_m = pos_corr[in_m]
    pos_p = pos_corr[in_p]
    vel_p = vel[in_p]
    r_p   = r_all[in_p]

    N_m = len(pos_m)
    N_p = len(pos_p)
    print(f"Dans r<{r_halo} Mpc : N−={N_m:,}  N+={N_p:,}")

    if N_p == 0 or N_m == 0:
        print("Pas de particules — abandon."); return

    # Sous-échantillonnage m+
    rng = np.random.default_rng(42)
    if 0 < n_sample < N_p:
        idx = rng.choice(N_p, n_sample, replace=False)
        print(f"Sous-échantillonnage : {n_sample}/{N_p} m+")
    else:
        idx = np.arange(N_p)

    pos_ps = pos_p[idx]
    vel_ps = vel_p[idx]
    r_ps   = r_p[idx]
    n_s    = len(idx)

    # Vitesse radiale par rapport au halo (pour référence)
    dr_ps  = pos_ps - halo_pos
    r_safe = np.maximum(r_ps, 0.01)
    r_hat  = dr_ps / r_safe[:, None]
    v_rad  = (vel_ps * r_hat).sum(axis=1)

    # Gradient ∇ρ_m−
    h = r_probe / 2.0
    print(f"Calcul gradient ∇ρ_m− | r_probe={r_probe} Mpc | h={h} Mpc "
          f"| {n_s} m+ × 7 positions ...")
    grad, rho_c = compute_gradient(pos_ps, pos_m, box, r_probe)
    print("Calcul terminé.")

    # Norme du gradient
    grad_norm_val = np.linalg.norm(grad, axis=1)
    valid = grad_norm_val > min_grad
    N_valid = valid.sum()
    frac_valid = N_valid / n_s
    print(f"Gradient non nul : {N_valid}/{n_s} = {frac_valid*100:.1f}% des m+")

    if N_valid < 10:
        print("ATTENTION : moins de 10 m+ avec gradient non nul.")
        print("Suggestions : augmenter r_probe ou utiliser un step plus précoce.")

    # cos_θ = angle entre gradient et vitesse
    # ∇ρ pointe vers plus de m−
    # cos_θ < 0 → m+ s'éloigne de la concentration m− = RÉPULSION
    g_n = grad[valid] / grad_norm_val[valid, None]
    v_n = vel_ps[valid] / (np.linalg.norm(vel_ps[valid], axis=1, keepdims=True) + 1e-30)
    cos_theta = (g_n * v_n).sum(axis=1)

    frac_neg    = (cos_theta < 0).mean()      # fraction répulsion
    mean_cos    = cos_theta.mean()
    median_cos  = np.median(cos_theta)

    # Test binomial : H0 = 0.5, H1 = frac_neg > 0.5
    n_neg = (cos_theta < 0).sum()
    binom_p = stats.binomtest(n_neg, N_valid, p=0.5, alternative='greater').pvalue

    # Test t : H0 = mean_cos = 0
    t_stat, t_p = stats.ttest_1samp(cos_theta, 0.0, alternative='less')

    print(f"\n=== RÉSULTATS GRADIENT JANUS ===")
    print(f"N m+ avec gradient non nul : {N_valid}")
    print(f"Fraction cos_θ < 0 (répulsion) : {frac_neg:.4f}  (attendu > 0.5)")
    print(f"cos_θ moyen   : {mean_cos:.4f}  (attendu < 0)")
    print(f"cos_θ médian  : {median_cos:.4f}")
    print(f"Test binomial p = {binom_p:.2e}  {'✓ SIGNIFICATIF' if binom_p<0.05 else ''}")
    print(f"Test t (mean<0) p = {t_p:.2e}  {'✓ SIGNIFICATIF' if t_p<0.05 else ''}")

    if binom_p < 0.01 and frac_neg > 0.5:
        verdict = "RÉPULSION JANUS CONFIRMÉE"
    elif binom_p < 0.05:
        verdict = "Tendance répulsion (p<0.05)"
    else:
        verdict = "Signal non significatif"
    print(f"\n→ {verdict}")

    # ── Profil cos_θ vs rayon r ──────────────────────────────────────
    r_valid = r_ps[valid]
    r_bins  = np.linspace(0, r_halo, 13)
    r_mid   = 0.5 * (r_bins[:-1] + r_bins[1:])
    cos_mean_r, cos_std_r, frac_neg_r, n_r = [], [], [], []
    for i in range(len(r_bins)-1):
        in_sh = (r_valid >= r_bins[i]) & (r_valid < r_bins[i+1])
        if in_sh.sum() > 2:
            cos_mean_r.append(cos_theta[in_sh].mean())
            cos_std_r.append(cos_theta[in_sh].std() / np.sqrt(in_sh.sum()))
            frac_neg_r.append((cos_theta[in_sh] < 0).mean())
            n_r.append(in_sh.sum())
        else:
            cos_mean_r.append(np.nan)
            cos_std_r.append(np.nan)
            frac_neg_r.append(np.nan)
            n_r.append(0)

    cos_mean_r = np.array(cos_mean_r)
    cos_std_r  = np.array(cos_std_r)
    frac_neg_r = np.array(frac_neg_r)

    # ── Figure 4K ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(38, 28))
    z_str = f"z={z:.3f}" if z is not None else ""
    fig.suptitle(
        f"Test répulsion Janus : ∇ρ_m− · v_m+ — step {step}  {z_str}\n"
        f"r_probe={r_probe} Mpc  |  h={h} Mpc  |  "
        f"N−={N_m:,}  N+ analysées={n_s}  N+ valides={N_valid}",
        fontsize=11, fontweight='bold')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # 1. Histogramme de cos_θ
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(cos_theta, bins=40, color='steelblue', alpha=0.8, edgecolor='white', lw=0.4)
    ax1.axvline(0, color='k', lw=1.2, ls='--', label='cos=0 (perp.)')
    ax1.axvline(mean_cos, color='red', lw=1.5, label=f'moy={mean_cos:.3f}')
    ax1.set_xlabel('cos_θ  (∇ρ · v̂)')
    ax1.set_ylabel('N particules m+')
    ax1.set_title('Distribution de cos_θ\n< 0 = répulsion  > 0 = attraction', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    # Coloration zones
    xlim = ax1.get_xlim()
    ax1.axvspan(xlim[0], 0, alpha=0.07, color='blue', label='répulsion')
    ax1.axvspan(0, xlim[1], alpha=0.07, color='red',  label='attraction')

    # 2. cos_θ moyen vs rayon
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(r_mid, cos_mean_r, yerr=cos_std_r, fmt='ko-', ms=5,
                 lw=1.5, capsize=4, label='⟨cos_θ⟩(r)')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    fill_color = 'blue' if np.nanmean(cos_mean_r) < 0 else 'red'
    ax2.fill_between(r_mid, 0, np.where(np.isnan(cos_mean_r), 0, cos_mean_r),
                     where=~np.isnan(cos_mean_r),
                     alpha=0.25,
                     color=fill_color)
    ax2.set_xlabel('r (Mpc)')
    ax2.set_ylabel('⟨cos_θ⟩')
    ax2.set_title('cos_θ moyen vs rayon\n<0 partout = répulsion isotrope', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Fraction cos_θ < 0 vs rayon (test binomial local)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(r_mid, frac_neg_r, 'bo-', ms=5, lw=1.5, label='frac(cos_θ<0)')
    ax3.axhline(0.5, color='k', lw=1, ls='--', label='50% (aléatoire)')
    ax3.axhline(0.6, color='green', lw=0.8, ls=':', label='60%')
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('r (Mpc)')
    ax3.set_ylabel('Fraction cos_θ < 0')
    ax3.set_title('Fraction répulsion vs rayon\n>50% = répulsion Janus locale', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Scatter cos_θ vs v_rad (pour voir la corrélation)
    ax4 = fig.add_subplot(gs[1, 0])
    sc = ax4.scatter(v_rad[valid], cos_theta,
                     c=r_valid, cmap='plasma_r', s=6, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax4, label='r (Mpc)')
    ax4.axhline(0, color='gray', lw=0.8, ls='--')
    ax4.axvline(0, color='gray', lw=0.8, ls='--')
    ax4.set_xlabel('v_rad m+ (>0=fuite halo)')
    ax4.set_ylabel('cos_θ  (∇ρ·v̂)')
    ax4.set_title('Scatter cos_θ vs v_rad\n(quadrant bas-droit = fuite+répulsion)', fontsize=9)
    ax4.grid(alpha=0.3)

    # 5. Scatter cos_θ vs ρ_centre (densité locale)
    ax5 = fig.add_subplot(gs[1, 1])
    rho_v = rho_c[valid]
    sc2 = ax5.scatter(rho_v, cos_theta,
                      c=r_valid, cmap='viridis_r', s=6, alpha=0.4, rasterized=True)
    plt.colorbar(sc2, ax=ax5, label='r (Mpc)')
    ax5.axhline(0, color='gray', lw=0.8, ls='--')
    # Régression
    if (rho_v > 0).sum() > 10:
        valid2 = rho_v > 0
        m_r, b_r, r_r, _, _ = stats.linregress(rho_v[valid2], cos_theta[valid2])
        x_r = np.linspace(rho_v[valid2].min(), rho_v[valid2].max(), 100)
        ax5.plot(x_r, m_r*x_r+b_r, 'k-', lw=1.5, label=f'r={r_r:.3f}')
        ax5.legend(fontsize=8)
    ax5.set_xlabel('ρ_local m− (N/Mpc³)')
    ax5.set_ylabel('cos_θ')
    ax5.set_title('cos_θ vs densité locale\n(milieu dense → plus grande répulsion ?)', fontsize=9)
    ax5.grid(alpha=0.3)

    # 6. Norme du gradient vs rayon
    ax6 = fig.add_subplot(gs[1, 2])
    grad_valid = grad_norm_val[valid]
    ax6.scatter(r_valid, grad_valid, c=cos_theta, cmap='RdBu',
                s=6, alpha=0.4, vmin=-1, vmax=1, rasterized=True)
    ax6.set_xlabel('r (Mpc)')
    ax6.set_ylabel('|∇ρ_m−|')
    ax6.set_title('Norme gradient vs rayon\n(coloré par cos_θ)', fontsize=9)
    ax6.set_yscale('log')
    ax6.grid(alpha=0.3)

    # 7. Résumé statistique complet
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Tableau cos_θ par rayon
    col_labels = ['r (Mpc)', '⟨cos_θ⟩', 'frac<0', 'N', 'Verdict']
    table_data = []
    for i in range(len(r_mid)):
        if n_r[i] > 2:
            cosi = cos_mean_r[i]
            fi   = frac_neg_r[i]
            v    = '← répulsion' if cosi < -0.05 else ('→ attraction' if cosi > 0.05 else '~ neutre')
            table_data.append([f'{r_mid[i]:.0f}', f'{cosi:.3f}', f'{fi:.3f}', str(n_r[i]), v])

    if table_data:
        tbl = ax7.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='upper left',
            cellLoc='center',
            bbox=[0.0, 0.3, 0.55, 0.65]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)

    # Boîte résumé global
    summary = (
        f"RÉSUMÉ GLOBAL — step {step}  {z_str}\n"
        f"{'─'*40}\n"
        f"N m−           : {N_m:,}\n"
        f"N m+ analysées : {n_s:,}\n"
        f"N m+ valides   : {N_valid}  ({frac_valid*100:.1f}%)\n"
        f"r_probe / h    : {r_probe} / {h} Mpc\n\n"
        f"Fraction cos_θ<0 : {frac_neg:.4f}\n"
        f"cos_θ moyen      : {mean_cos:.4f}\n"
        f"cos_θ médian     : {median_cos:.4f}\n\n"
        f"Test binomial (frac>0.5) : p={binom_p:.2e}\n"
        f"Test t (mean<0)          : p={t_p:.2e}\n\n"
        f"→ {verdict}"
    )
    ax7.text(0.60, 0.98, summary, transform=ax7.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    out_dir = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/analysis_gradient'
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/gradient_janus_step{step:04d}_4K.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f"\nFigure → {out}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap',     required=True)
    parser.add_argument('--step',     type=int,   default=0)
    parser.add_argument('--z',        type=float, default=None)
    parser.add_argument('--r_probe',  type=float, default=10.0)
    parser.add_argument('--r_halo',   type=float, default=120.0)
    parser.add_argument('--n_sample', type=int,   default=8000)
    parser.add_argument('--halo_pos', type=str,   default='168,127,73')
    parser.add_argument('--box',      type=float, default=500.0)
    parser.add_argument('--min_grad', type=float, default=1e-9)
    args = parser.parse_args()

    halo_pos = np.array([float(x) for x in args.halo_pos.split(',')])

    analyse(args.snap, args.step, args.z,
            args.r_probe, args.r_halo, args.n_sample,
            halo_pos, args.box, args.min_grad)
