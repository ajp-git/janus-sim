"""
analyse_halo_spatial.py
=======================
Analyse spatiale approfondie du méga-halo− (168, 127, 73) Mpc
Rayon étendu : 0–120 Mpc autour du centre

Usage :
    python analyse_halo_spatial.py --snap /path/to/snapshot_XXXX.hdf5 --step 3500
    python analyse_halo_spatial.py --snap /path/to/snapshot_XXXX.npy  --step 3500

Le script détecte automatiquement le format (HDF5, NPY, binaire Rust).
Adapte les clés HDF5 si nécessaire (voir KEYS ci-dessous).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import argparse
import os

# ── Paramètres du halo ────────────────────────────────────────────────
HALO_POS  = np.array([168.0, 127.0, 73.0])   # Mpc
BOX_SIZE  = 256.0                              # Mpc
R_CORE    = 60.0                               # rayon de suivi original
R_EXT     = 120.0                              # rayon étendu pour analyse

# Clés HDF5 à adapter selon ton format
HDF5_KEYS = {
    'pos'  : 'PartType0/Coordinates',   # shape (N, 3)
    'vel'  : 'PartType0/Velocities',    # shape (N, 3)
    'mass' : 'PartType0/Masses',        # shape (N,) — signé : +m ou -m
}

# ── Chargement snapshot ────────────────────────────────────────────────
def load_snapshot(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.hdf5', '.h5']:
        import h5py
        with h5py.File(path, 'r') as f:
            pos  = f[HDF5_KEYS['pos']][:]
            vel  = f[HDF5_KEYS['vel']][:]
            mass = f[HDF5_KEYS['mass']][:]
        return pos, vel, mass

    elif ext == '.npy':
        data = np.load(path)
        # Attend shape (N, 7) : x y z vx vy vz mass
        pos  = data[:, :3]
        vel  = data[:, 3:6]
        mass = data[:, 6]
        return pos, vel, mass

    else:
        # Format binaire Rust : f32 x y z vx vy vz mass_sign
        # Adapte si nécessaire
        data = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
        pos  = data[:, :3].astype(np.float64)
        vel  = data[:, 3:6].astype(np.float64)
        mass = data[:, 6].astype(np.float64)   # +1 ou -1
        return pos, vel, mass


# ── Distance avec conditions périodiques ──────────────────────────────
def dist_periodic(pos, center, box):
    d = pos - center
    d = d - box * np.round(d / box)
    return np.sqrt((d**2).sum(axis=1)), d


# ── Analyse principale ────────────────────────────────────────────────
def analyse(snap_path, step, z=None):
    print(f"Chargement : {snap_path}")
    pos, vel, mass = load_snapshot(snap_path)

    sign = np.sign(mass)
    mask_m  = sign < 0    # masses négatives
    mask_p  = sign > 0    # masses positives

    # Distances au centre du méga-halo
    r, dr = dist_periodic(pos, HALO_POS, BOX_SIZE)

    # Positions corrigées périodiquement (centrées sur le halo)
    # Indispensable pour les histogrammes 2D : pos brut peut déborder hors BOX_SIZE
    pos_corr = HALO_POS + dr   # shape (N, 3), centré sur HALO_POS, sans saut périodique

    # Sélection dans le rayon étendu
    in_ext  = r < R_EXT
    in_core = r < R_CORE

    N_total_ext = in_ext.sum()
    N_minus_ext = (in_ext & mask_m).sum()
    N_plus_ext  = (in_ext & mask_p).sum()
    N_minus_core = (in_core & mask_m).sum()
    N_plus_core  = (in_core & mask_p).sum()

    print(f"\n=== HALO @ step={step} {'z='+str(z) if z else ''} ===")
    print(f"Rayon étendu R<{R_EXT} Mpc : {N_total_ext} particules")
    print(f"  N− = {N_minus_ext:,} | N+ = {N_plus_ext:,} | N−/N+ = {N_minus_ext/(N_plus_ext+1):.1f}")
    print(f"Rayon core   R<{R_CORE} Mpc : N−={N_minus_core:,} | N+={N_plus_core:,}")

    # Profil de densité radial (12 bins log de 1 à 120 Mpc)
    r_bins = np.logspace(np.log10(1), np.log10(R_EXT), 20)
    r_mid  = 0.5 * (r_bins[:-1] + r_bins[1:])
    vol    = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)

    dens_m = np.histogram(r[mask_m], bins=r_bins)[0] / vol
    dens_p = np.histogram(r[mask_p], bins=r_bins)[0] / vol

    # Vitesse radiale (infalling < 0 / outflowing > 0)
    # v_rad = v · r_hat
    r_safe = np.maximum(r, 0.01)
    r_hat  = dr / r_safe[:, None]
    v_rad  = (vel * r_hat).sum(axis=1)

    v_rad_m = v_rad[mask_m]
    v_rad_p = v_rad[mask_p]
    r_m     = r[mask_m]
    r_p     = r[mask_p]

    # Dispersion de vitesse par couronnes
    sigma_m = []
    sigma_p = []
    frac_out_p = []   # fraction m+ qui fuient (v_rad > 0)
    for i in range(len(r_bins)-1):
        bm = (r_m >= r_bins[i]) & (r_m < r_bins[i+1])
        bp = (r_p >= r_bins[i]) & (r_p < r_bins[i+1])
        sigma_m.append(v_rad_m[bm].std() if bm.sum() > 2 else np.nan)
        sigma_p.append(v_rad_p[bp].std() if bp.sum() > 2 else np.nan)
        frac_out_p.append((v_rad_p[bp] > 0).mean() if bp.sum() > 2 else np.nan)

    sigma_m  = np.array(sigma_m)
    sigma_p  = np.array(sigma_p)
    frac_out = np.array(frac_out_p)

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    title = f"Méga-halo− (168,127,73) Mpc — step {step}"
    if z: title += f" — z={z:.3f}"
    fig.suptitle(title + f"\nRayon étendu {R_EXT} Mpc", fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Carte densité 2D — projection XY (slice z ± 30 Mpc)
    ax1 = fig.add_subplot(gs[0, 0])
    dz = 30.0
    slice_mask = np.abs(dr[:, 2]) < dz
    ext_xy = [HALO_POS[0]-R_EXT, HALO_POS[0]+R_EXT,
               HALO_POS[1]-R_EXT, HALO_POS[1]+R_EXT]
    for sp, col, lbl in [(mask_m & slice_mask, 'Blues', 'm−'),
                          (mask_p & slice_mask, 'Reds',  'm+')]:
        if sp.sum() > 10:
            h, xe, ye = np.histogram2d(pos_corr[sp, 0], pos_corr[sp, 1], bins=80,
                                        range=[[ext_xy[0], ext_xy[1]],
                                               [ext_xy[2], ext_xy[3]]])
            h = np.ma.masked_where(h == 0, h)
            ax1.pcolormesh(xe, ye, h.T, cmap=col, alpha=0.6,
                           norm=LogNorm(vmin=1))
    circle_c = plt.Circle(HALO_POS[:2], R_CORE, fill=False, color='white',
                           ls='--', lw=1.2, label=f'r={R_CORE}')
    circle_e = plt.Circle(HALO_POS[:2], R_EXT, fill=False, color='yellow',
                           ls=':', lw=1, label=f'r={R_EXT}')
    ax1.add_patch(circle_c); ax1.add_patch(circle_e)
    ax1.set_aspect('equal')
    ax1.set_title(f'Carte XY (|Δz|<{dz} Mpc)\nBleu=m−  Rouge=m+', fontsize=9)
    ax1.set_xlabel('X (Mpc)'); ax1.set_ylabel('Y (Mpc)')
    ax1.legend(fontsize=7, loc='upper right')

    # 2. Carte densité 2D — projection XZ
    ax2 = fig.add_subplot(gs[0, 1])
    dy = 30.0
    slice_mask2 = np.abs(dr[:, 1]) < dy
    for sp, col in [(mask_m & slice_mask2, 'Blues'),
                     (mask_p & slice_mask2, 'Reds')]:
        if sp.sum() > 10:
            h, xe, ze = np.histogram2d(pos_corr[sp, 0], pos_corr[sp, 2], bins=80,
                                        range=[[ext_xy[0], ext_xy[1]],
                                               [HALO_POS[2]-R_EXT, HALO_POS[2]+R_EXT]])
            h = np.ma.masked_where(h == 0, h)
            ax2.pcolormesh(xe, ze, h.T, cmap=col, alpha=0.6, norm=LogNorm(vmin=1))
    ax2.add_patch(plt.Circle((HALO_POS[0], HALO_POS[2]), R_CORE,
                              fill=False, color='white', ls='--', lw=1.2))
    ax2.set_aspect('equal')
    ax2.set_title(f'Carte XZ (|Δy|<{dy} Mpc)\nBleu=m−  Rouge=m+', fontsize=9)
    ax2.set_xlabel('X (Mpc)'); ax2.set_ylabel('Z (Mpc)')

    # Distribution N(r) sur les MÊMES bins log que sigma (r_bins) — pour KE cohérent
    # (utilisé panel 3 ET panel 9)
    r_shell_edges = np.linspace(0, R_EXT, 25)          # bins linéaires pour panel 3 (visuel)
    r_shell_mid   = 0.5*(r_shell_edges[:-1]+r_shell_edges[1:])
    n_m_sh = np.histogram(r[mask_m], bins=r_shell_edges)[0]
    n_p_sh = np.histogram(r[mask_p], bins=r_shell_edges)[0]

    # N(r) sur les bins log (r_bins) — même grille que sigma_m/sigma_p → pour KE panel 9
    n_m_logbins = np.histogram(r[mask_m], bins=r_bins)[0]
    n_p_logbins = np.histogram(r[mask_p], bins=r_bins)[0]
    # 3. Distribution N par couronnes (histogram sphérique)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.fill_between(r_shell_mid, 0, n_m_sh, alpha=0.6, color='steelblue', label='m−')
    ax3.fill_between(r_shell_mid, 0, n_p_sh, alpha=0.6, color='crimson', label='m+')
    for R in [R_CORE]:
        ax3.axvline(R, color='gray', ls='--', lw=1, label=f'r={R}')
    ax3.set_xlabel('r (Mpc)'); ax3.set_ylabel('N particules/shell')
    ax3.set_title('Distribution radiale N(r)', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Profil de densité ρ(r)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.loglog(r_mid, np.maximum(dens_m, 1e-10), 'b-o', ms=4, lw=1.8, label='ρ− (m−)')
    ax4.loglog(r_mid, np.maximum(dens_p, 1e-10), 'r-o', ms=4, lw=1.8, label='ρ+ (m+)')
    # Profil NFW indicatif : ∝ r^-2 (ancré sur le premier bin non-nul)
    nz = np.where(dens_m > 0)[0]
    if len(nz) > 0:
        r_ref   = r_mid[nz[0]]
        rho_ref = dens_m[nz[0]]
        ax4.loglog(r_mid, rho_ref * (r_ref/r_mid)**2, 'k--', lw=1, label='∝ r⁻²')
    ax4.axvline(R_CORE, color='gray', ls=':', lw=1)
    ax4.set_xlabel('r (Mpc)'); ax4.set_ylabel('Densité (N/Mpc³)')
    ax4.set_title('Profil de densité ρ(r)', fontsize=9)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # 5. Vitesse radiale moyenne par couronnes
    ax5 = fig.add_subplot(gs[1, 1])
    vrad_m_mean = []
    vrad_p_mean = []
    for i in range(len(r_bins)-1):
        bm = (r_m >= r_bins[i]) & (r_m < r_bins[i+1])
        bp = (r_p >= r_bins[i]) & (r_p < r_bins[i+1])
        vrad_m_mean.append(v_rad_m[bm].mean() if bm.sum() > 2 else np.nan)
        vrad_p_mean.append(v_rad_p[bp].mean() if bp.sum() > 2 else np.nan)
    ax5.plot(r_mid, vrad_m_mean, 'b-o', ms=4, lw=1.8, label='⟨v_rad⟩ m−')
    ax5.plot(r_mid, vrad_p_mean, 'r-o', ms=4, lw=1.8, label='⟨v_rad⟩ m+')
    ax5.axhline(0, color='k', lw=0.8)
    ax5.axvline(R_CORE, color='gray', ls=':', lw=1)
    ax5.set_xlabel('r (Mpc)'); ax5.set_ylabel('v_rad moyen')
    ax5.set_title('Vitesse radiale ⟨v_r⟩(r)\n>0 = fuite, <0 = infalling', fontsize=9)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # 6. Fraction de m+ en fuite (v_rad > 0) par couronne
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(r_mid, frac_out, 'r-o', ms=4, lw=1.8)
    ax6.axhline(0.5, color='k', ls='--', lw=1, label='50% (isotrope)')
    ax6.axvline(R_CORE, color='gray', ls=':', lw=1, label=f'r={R_CORE}')
    ax6.set_ylim(0, 1)
    ax6.set_xlabel('r (Mpc)'); ax6.set_ylabel('Fraction m+ avec v_r > 0')
    ax6.set_title('Direction de fuite des m+\n(>0.5 = expulsion nette)', fontsize=9)
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    # 7. Dispersion σ_v radiale (m− et m+)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(r_mid, sigma_m, 'b-o', ms=4, lw=1.8, label='σ_v m−')
    ax7.plot(r_mid, sigma_p, 'r-o', ms=4, lw=1.8, label='σ_v m+')
    ax7.axvline(R_CORE, color='gray', ls=':', lw=1)
    ax7.set_xlabel('r (Mpc)'); ax7.set_ylabel('σ_v radiale')
    ax7.set_title('Dispersion de vitesse σ_v(r)', fontsize=9)
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3)

    # 8. Pureté locale P(r) = (N− − N+) / (N− + N+)
    ax8 = fig.add_subplot(gs[2, 1])
    P_r = (dens_m - dens_p) / (dens_m + dens_p + 1e-10)
    ax8.semilogx(r_mid, P_r, 'k-o', ms=4, lw=1.8)
    ax8.axhline(1.0, color='b', ls='--', lw=1, label='P=1 (pur m−)')
    ax8.axhline(0.0, color='gray', ls=':', lw=1, label='P=0 (mixte)')
    ax8.axvline(R_CORE, color='gray', ls=':', lw=1)
    ax8.set_ylim(-0.1, 1.1)
    ax8.set_xlabel('r (Mpc)'); ax8.set_ylabel('Pureté P(r)')
    ax8.set_title('Pureté radiale P = (N−−N+)/(N−+N+)', fontsize=9)
    ax8.legend(fontsize=8); ax8.grid(alpha=0.3)

    # 9. Énergie cinétique radiale : KE(r) = ½ σ²(r) · N(r)
    # sigma_m/sigma_p et n_m_logbins/n_p_logbins sont sur les MÊMES r_bins → cohérent
    ax9 = fig.add_subplot(gs[2, 2])
    ke_m = 0.5 * sigma_m**2 * n_m_logbins
    ke_p = 0.5 * sigma_p**2 * n_p_logbins
    ax9.semilogy(r_mid, np.maximum(ke_m, 1e-3), 'b-o', ms=4, lw=1.8, label='KE m−')
    ax9.semilogy(r_mid, np.maximum(ke_p, 1e-3), 'r-o', ms=4, lw=1.8, label='KE m+')
    ax9.axvline(R_CORE, color='gray', ls=':', lw=1)
    ax9.set_xlabel('r (Mpc)'); ax9.set_ylabel('KE(r) [u.a.]')
    ax9.set_title('Énergie cinétique radiale\n½ σ²(r) · N(r)', fontsize=9)
    ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

    out = f"halo_spatial_step{step:04d}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure sauvegardée : {out}")

    # Rapport texte
    print("\n=== RAPPORT SPATIAL ===")
    print(f"Profil densité m− au centre (r<5 Mpc)  : {dens_m[0]:.2f} N/Mpc³")
    print(f"Profil densité m− à r=60 Mpc           : {dens_m[np.argmin(np.abs(r_mid-60))]:.2f} N/Mpc³")
    print(f"Pureté P à r<5 Mpc    : {P_r[0]:.4f}")
    print(f"Pureté P à r=60 Mpc   : {P_r[np.argmin(np.abs(r_mid-60))]:.4f}")
    print(f"Pureté P à r=120 Mpc  : {P_r[-1]:.4f}")
    if N_plus_ext > 5:
        frac_fuite = (v_rad_p[r_p < R_EXT] > 0).mean()
        print(f"Fraction m+ en fuite (R<{R_EXT}) : {frac_fuite:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap', required=True, help='Chemin vers le snapshot')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--z', type=float, default=None)
    args = parser.parse_args()
    analyse(args.snap, args.step, args.z)
