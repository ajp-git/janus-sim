#!/usr/bin/env python3
"""
analyse_pk.py — Spectres P(k) et corrélation ξ(r) pour simulation Janus
Usage: python3 analyse_pk.py <snap.bin> [out_prefix]

Calcule :
  1. P_rho(k)   — spectre densité totale
  2. P_P(k)     — spectre polarisation (ρ+−ρ−)/(ρ++ρ−)
  3. P_pm(k)    — corrélation croisée ρ+ × ρ−
  4. ξ(r)       — fonction de corrélation à 2 points
"""

import sys, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BOX  = 492.0
RES  = 256       # grille 3D — 256³ suffit pour P(k)
BG   = '#06060f'
C_TEXT = '#aaaacc'

# ── Lecture snapshot ──────────────────────────────────────────────────────────
def read_snapshot(path):
    with open(path, 'rb') as f:
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        data = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)
    sign  = data[:, 3]
    pos_p = data[sign >  0, :3]
    pos_m = data[sign <  0, :3]
    return n, step, pos_p, pos_m

# ── Grille de densité 3D ──────────────────────────────────────────────────────
def density_grid_3d(pos, box, res):
    half = box / 2.0
    xi = np.clip(((pos[:,0]+half)/box*res).astype(int), 0, res-1)
    yi = np.clip(((pos[:,1]+half)/box*res).astype(int), 0, res-1)
    zi = np.clip(((pos[:,2]+half)/box*res).astype(int), 0, res-1)
    grid = np.zeros((res,res,res), dtype=np.float64)
    np.add.at(grid, (xi,yi,zi), 1)
    return grid

# ── Champ de contraste δ = (ρ − ρ̄) / ρ̄ ──────────────────────────────────────
def delta_field(grid):
    mean = grid.mean()
    if mean < 1e-10:
        return grid
    return (grid - mean) / mean

# ── Spectre de puissance P(k) ─────────────────────────────────────────────────
def power_spectrum(delta, box, res):
    """P(k) par FFT 3D + moyenne radiale."""
    dk     = np.fft.rfftn(delta)
    pk_3d  = (np.abs(dk)**2) * (box/res)**3 / res**3

    # Fréquences
    kx = np.fft.fftfreq(res,  d=box/res) * 2*np.pi
    ky = np.fft.fftfreq(res,  d=box/res) * 2*np.pi
    kz = np.fft.rfftfreq(res, d=box/res) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Bins logarithmiques
    k_min_bin = 2*np.pi / box
    k_max_bin = np.pi * res / box
    k_bins = np.logspace(np.log10(k_min_bin), np.log10(k_max_bin), 40)
    k_centers = 0.5*(k_bins[:-1] + k_bins[1:])

    pk_mean = np.zeros(len(k_centers))
    for i, (klo, khi) in enumerate(zip(k_bins[:-1], k_bins[1:])):
        mask = (K_mag >= klo) & (K_mag < khi)
        if mask.sum() > 0:
            pk_mean[i] = pk_3d[mask].mean()

    return k_centers, pk_mean

# ── Corrélation croisée P_{+-}(k) ────────────────────────────────────────────
def cross_spectrum(delta_p, delta_m, box, res):
    dk_p = np.fft.rfftn(delta_p)
    dk_m = np.fft.rfftn(delta_m)
    pk_cross = np.real(dk_p * np.conj(dk_m)) * (box/res)**3 / res**3

    kx = np.fft.fftfreq(res,  d=box/res) * 2*np.pi
    ky = np.fft.fftfreq(res,  d=box/res) * 2*np.pi
    kz = np.fft.rfftfreq(res, d=box/res) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_min_bin = 2*np.pi / box
    k_max_bin = np.pi * res / box
    k_bins = np.logspace(np.log10(k_min_bin), np.log10(k_max_bin), 40)
    k_centers = 0.5*(k_bins[:-1] + k_bins[1:])

    pk_mean = np.zeros(len(k_centers))
    for i, (klo, khi) in enumerate(zip(k_bins[:-1], k_bins[1:])):
        mask = (K_mag >= klo) & (K_mag < khi)
        if mask.sum() > 0:
            pk_mean[i] = pk_cross[mask].mean()

    return k_centers, pk_mean

# ── Fonction de corrélation ξ(r) ─────────────────────────────────────────────
def correlation_function(delta, box, res, n_rbins=30):
    """ξ(r) par transformée de Fourier inverse de P(k)."""
    dk    = np.fft.rfftn(delta)
    pk_3d = (np.abs(dk)**2) * (box/res)**3 / res**3

    # Transformée inverse → ξ(r) dans l'espace réel
    xi_3d = np.fft.irfftn(pk_3d * res**3 / box**3).real

    # Distance de chaque cellule à l'origine
    cell = box / res
    idx  = np.arange(res)
    idx  = np.where(idx <= res//2, idx, idx - res)
    IX, IY, IZ = np.meshgrid(idx, idx, idx[:res//2+1], indexing='ij')
    R = np.sqrt(IX**2 + IY**2 + IZ**2) * cell

    r_max  = box / 2
    r_bins = np.linspace(0, r_max, n_rbins+1)
    r_cen  = 0.5*(r_bins[:-1] + r_bins[1:])
    xi_r   = np.zeros(n_rbins)

    xi_full = np.fft.irfftn(pk_3d * res**3 / box**3, s=(res,res,res)).real
    R_full  = np.zeros((res,res,res))
    ix = np.arange(res); ix = np.where(ix<=res//2, ix, ix-res)
    IXf,IYf,IZf = np.meshgrid(ix,ix,ix,indexing='ij')
    R_full = np.sqrt(IXf**2+IYf**2+IZf**2)*cell

    for i,(rlo,rhi) in enumerate(zip(r_bins[:-1],r_bins[1:])):
        mask = (R_full >= rlo) & (R_full < rhi)
        if mask.sum() > 0:
            xi_r[i] = xi_full[mask].mean()

    return r_cen, xi_r

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(step, z, k, pk_rho, pk_pol, pk_pm, r, xi, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(f'Janus Spectral Analysis — Step {step:06d}  z = {z:.3f}',
                 color='white', fontsize=14, fontweight='bold', fontfamily='monospace')

    kw = dict(facecolor=BG, labelcolor=C_TEXT)

    # ── P(k) ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    mask = pk_rho > 0
    ax.loglog(k[mask], pk_rho[mask], color='#4db8ff', lw=2, label='P_ρ(k)  densité totale')
    mask = np.abs(pk_pol) > 0
    ax.loglog(k[mask], np.abs(pk_pol[mask]), color='#aaffaa', lw=2, label='P_P(k)  polarisation')
    # Référence ΛCDM n=-2
    k_ref = k[k > 0]
    ax.loglog(k_ref, k_ref**-2 * pk_rho[mask][len(mask)//2] / k_ref[len(k_ref)//2]**-2,
              color='#555577', lw=1, ls='--', label='k⁻² référence')
    ax.set_xlabel('k  (Mpc⁻¹)', color=C_TEXT, fontfamily='monospace')
    ax.set_ylabel('P(k)  (Mpc³)', color=C_TEXT, fontfamily='monospace')
    ax.set_title('Spectres de puissance', color=C_TEXT, fontfamily='monospace')
    ax.tick_params(colors=C_TEXT)
    ax.legend(fontsize=8, framealpha=0, labelcolor=C_TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor('#1a1a2e')

    # ── P_{+-}(k) ─────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)
    pos_mask = pk_pm > 0
    neg_mask = pk_pm < 0
    if pos_mask.sum() > 0:
        ax.loglog(k[pos_mask], pk_pm[pos_mask],  color='#ff9944', lw=2, label='P₊₋ > 0')
    if neg_mask.sum() > 0:
        ax.loglog(k[neg_mask], -pk_pm[neg_mask], color='#ff4444', lw=2, ls='--', label='P₊₋ < 0')
    ax.set_xlabel('k  (Mpc⁻¹)', color=C_TEXT, fontfamily='monospace')
    ax.set_ylabel('|P₊₋(k)|', color=C_TEXT, fontfamily='monospace')
    ax.set_title('Corrélation croisée ρ₊ × ρ₋', color=C_TEXT, fontfamily='monospace')
    ax.tick_params(colors=C_TEXT)
    ax.legend(fontsize=8, framealpha=0, labelcolor=C_TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor('#1a1a2e')

    # ── ξ(r) ──────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(BG)
    mask_xi = xi > 0
    if mask_xi.sum() > 1:
        ax.loglog(r[mask_xi], xi[mask_xi], color='#ff5533', lw=2, label='ξ(r)')
        # Référence r^-1.8
        r_ref = r[mask_xi]
        norm  = xi[mask_xi][0] / r_ref[0]**-1.8
        ax.loglog(r_ref, norm * r_ref**-1.8, color='#555577', lw=1,
                  ls='--', label='r⁻¹·⁸ ΛCDM')
    ax.set_xlabel('r  (Mpc)', color=C_TEXT, fontfamily='monospace')
    ax.set_ylabel('ξ(r)', color=C_TEXT, fontfamily='monospace')
    ax.set_title('Fonction de corrélation', color=C_TEXT, fontfamily='monospace')
    ax.tick_params(colors=C_TEXT)
    ax.legend(fontsize=8, framealpha=0, labelcolor=C_TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor('#1a1a2e')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Sauvegardé → {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def analyse(snap_path, out_prefix, z_cosmo=None):
    print(f"Lecture {snap_path}...")
    n, step, pos_p, pos_m = read_snapshot(snap_path)
    print(f"  N+={len(pos_p):,}  N-={len(pos_m):,}")

    print("Calcul grilles 3D...")
    grid_p = density_grid_3d(pos_p, BOX, RES)
    grid_m = density_grid_3d(pos_m, BOX, RES)
    grid_t = grid_p + grid_m  # densité totale

    # Champ de polarisation P = (ρ+−ρ−)/(ρ++ρ−)
    total  = grid_t.copy(); total[total < 1e-10] = 1e-10
    grid_pol = (grid_p - grid_m) / total

    delta_t   = delta_field(grid_t)
    delta_pol = grid_pol - grid_pol.mean()
    delta_p   = delta_field(grid_p)
    delta_m   = delta_field(grid_m)

    print("Calcul P(k)...")
    k, pk_rho = power_spectrum(delta_t,   BOX, RES)
    _, pk_pol = power_spectrum(delta_pol, BOX, RES)
    _, pk_pm  = cross_spectrum(delta_p, delta_m, BOX, RES)

    print("Calcul ξ(r)...")
    r, xi = correlation_function(delta_t, BOX, RES)

    # Pente P(k) sur la partie centrale
    mid = len(k)//2
    if pk_rho[mid] > 0 and pk_rho[mid-3] > 0:
        slope = np.log(pk_rho[mid]/pk_rho[mid-3]) / np.log(k[mid]/k[mid-3])
    else:
        slope = 0

    # ── Métriques R et épaisseur filaments ───────────────────────────────────
    sigma_rho = np.std(delta_t)
    sigma_pol = np.std(delta_pol)
    R = sigma_rho / (sigma_pol + 1e-10)

    # Épaisseur filaments : profil transverse moyen sur les voxels filaments
    # Hessian 3D simplifié sur grille lissée
    grid_smooth = gaussian_filter(grid_t / (grid_t.mean()+1e-10), sigma=2.0)
    gx,gy,gz = np.gradient(grid_smooth)
    gxx,_,_ = np.gradient(gx); _,gyy,_ = np.gradient(gy); _,_,gzz = np.gradient(gz)
    laplacian = gxx + gyy + gzz
    # Voxels filaments : laplacien négatif fort (régions courbées)
    threshold = np.percentile(laplacian, 15)
    fil_mask  = laplacian < threshold
    fil_frac  = fil_mask.mean() * 100

    # Largeur filaments : taille caractéristique des clusters de voxels filaments
    from scipy.ndimage import label
    labeled, n_clusters = label(fil_mask)
    cell_mpc = BOX / RES
    if n_clusters > 10:
        sizes = np.bincount(labeled.ravel())[1:]
        median_size = np.median(sizes)
        width_mpc = (median_size / np.pi) ** (1/3) * cell_mpc * 2
    else:
        width_mpc = 0.0

    print(f"\n=== Résultats step {step} ===")
    print(f"  σ_ρ        : {sigma_rho:.4f}")
    print(f"  σ_P        : {sigma_pol:.4f}")
    print(f"  R = σρ/σP  : {R:.2f}  (filaments physiques si R ≈ 3–5)")
    print(f"  Filaments  : {fil_frac:.1f}% des voxels")
    print(f"  Largeur    : ~{width_mpc:.1f} Mpc  (Janus prédit 1–3 Mpc)")

    # ── Test anisotropie P(k) — détection artefact grille PM ─────────────────
    # Calculer P(k) séparément selon kx, ky, kz
    # Si filaments réels → P(kx) ≈ P(ky) ≈ P(kz)
    # Si artefact PM    → pics aux harmoniques de k_grid sur un axe
    dk_t = np.fft.rfftn(delta_t)
    pk_3d = np.abs(dk_t)**2

    res2 = RES
    k1d  = np.fft.fftfreq(res2, d=BOX/res2) * 2*np.pi
    kz1d = np.fft.rfftfreq(res2, d=BOX/res2) * 2*np.pi

    # Tranches kx=0 (plan YZ), ky=0 (plan XZ), kz=0 (plan XY)
    pk_kx0 = pk_3d[0, :, :]   # kx=0 → structure dans YZ
    pk_ky0 = pk_3d[:, 0, :]   # ky=0 → structure dans XZ
    pk_kz0 = pk_3d[:, :, 0]   # kz=0 → structure dans XY

    # Moyenne radiale 2D dans chaque plan
    def radial_mean_2d(plane, k1, k2):
        K1, K2 = np.meshgrid(k1[:plane.shape[0]], k2[:plane.shape[1]], indexing='ij')
        Kmag = np.sqrt(K1**2 + K2**2)
        k_bins = np.logspace(np.log10(2*np.pi/BOX+1e-10), np.log10(np.pi*RES/BOX), 20)
        pk_r = np.zeros(len(k_bins)-1)
        kc   = 0.5*(k_bins[:-1]+k_bins[1:])
        for i,(lo,hi) in enumerate(zip(k_bins[:-1],k_bins[1:])):
            m = (Kmag>=lo)&(Kmag<hi)
            if m.sum()>0: pk_r[i] = plane[m].mean()
        return kc, pk_r

    kc, pyz = radial_mean_2d(pk_kx0, k1d, kz1d)
    _,  pxz = radial_mean_2d(pk_ky0, k1d, kz1d)
    _,  pxy = radial_mean_2d(pk_kz0, k1d, k1d)

    # Anisotropie = écart-type relatif entre les 3 plans
    stack = np.array([pyz, pxz, pxy])
    mean_pk = stack.mean(axis=0)
    aniso   = np.std(stack, axis=0) / (mean_pk + 1e-10)
    aniso_max = aniso[mean_pk > mean_pk.max()*0.01].mean()

    print(f"  Anisotropie P(k) : {aniso_max:.3f}  "
          f"({'✓ isotrope — filaments physiques' if aniso_max < 0.3 else '⚠ anisotrope — vérifier artefact grille'})")
    print(f"  Pente P_rho(k) ~ k^{slope:.2f}  (ΛCDM attendu: -1 à -3)")
    print(f"  P(k) max à k = {k[np.argmax(pk_rho)]:.4f} Mpc⁻¹  "
          f"(λ = {2*np.pi/k[np.argmax(pk_rho)]:.1f} Mpc)")
    if xi[xi>0].sum() > 0:
        xi_pos = xi[xi>0]; r_pos = r[xi>0]
        if len(r_pos) > 3:
            slope_xi = np.log(xi_pos[0]/xi_pos[3]) / np.log(r_pos[0]/r_pos[3])
            print(f"  Pente ξ(r) ~ r^{slope_xi:.2f}  (ΛCDM attendu: -1.8)")

    out_png = f"{out_prefix}_step{step:06d}.png"
    plot_results(step, z_cosmo or 0.0, k, pk_rho, pk_pol, pk_pm, r, xi, out_png)
    return k, pk_rho, pk_pol, pk_pm, r, xi


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyse_pk.py <snap.bin> [out_prefix] [z]")
        sys.exit(1)
    snap   = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else snap.replace('.bin','')
    z      = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    analyse(snap, prefix, z)
