#!/usr/bin/env python3
"""
render_frame.py v4 — Frame 4K Janus : vue 2.5D + profil Z
Usage: python3 render_frame.py <snap.bin> [out.png] [z] [seg] [ke] [t_gyr]
"""

import sys, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d

BOX_MPC      = 492.0
Z_BINS       = 120       # résolution profil Z
PROFILE_SIGMA = 2.0      # lissage gaussien profil

BG       = '#050508'
C_PLUS   = '#4db8ff'
C_MINUS  = '#ff5533'
C_TEXT   = '#9999bb'
C_TITLE  = '#ffffff'
C_GRID   = '#1a1a2e'

# ── I/O ───────────────────────────────────────────────────────────────────────
def read_snapshot(path):
    with open(path, 'rb') as f:
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        data = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)
    sign = data[:, 3]
    pos_p = data[sign >  0, :3]
    pos_m = data[sign < 0, :3]
    return n, step, pos_p, pos_m

# ── Projection isométrique ────────────────────────────────────────────────────
def iso(pos):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    xi = (x - y) * np.cos(np.radians(30))
    yi = (x + y) * np.sin(np.radians(30)) * 0.45 + z * 0.75
    return xi, yi

# ── Profil de densité 1D en Z ─────────────────────────────────────────────────
def z_profile(pos, box, n_bins, sigma):
    half = box / 2
    counts, edges = np.histogram(pos[:, 2], bins=n_bins, range=(-half, half))
    centers = (edges[:-1] + edges[1:]) / 2
    # Normaliser par le nombre de particules
    density = counts / (counts.sum() + 1e-10) * n_bins
    if sigma > 0:
        density = gaussian_filter1d(density, sigma=sigma)
    return centers, density

# ── Render ────────────────────────────────────────────────────────────────────
def render(snap_path, out_path, z_cosmo=None, seg=None, ke=None, t_gyr=None):

    print(f"Lecture {snap_path}...")
    n, step, pos_p, pos_m = read_snapshot(snap_path)
    print(f"  step={step}  N+={len(pos_p):,}  N-={len(pos_m):,}")

    # Tous les 12M projetés via grille 2D (pas de sous-échantillonnage)
    np.random.seed(42)
    xp, yp = iso(pos_p)
    xm, ym = iso(pos_m)

    zc, rho_p = z_profile(pos_p, BOX_MPC, Z_BINS, PROFILE_SIGMA)
    _,  rho_m = z_profile(pos_m, BOX_MPC, Z_BINS, PROFILE_SIGMA)

    # ── Figure ────────────────────────────────────────────────────────────────
    DPI = 200
    fig = plt.figure(figsize=(3840/DPI, 2160/DPI), dpi=DPI, facecolor=BG)

    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.07, 0.82, 0.11],
                  width_ratios=[0.58, 0.42],
                  hspace=0.04, wspace=0.06,
                  left=0.03, right=0.97,
                  top=0.97, bottom=0.02)

    # Header
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(BG); ax_h.axis('off')
    ax_h.text(0.5, 0.65, 'JANUS COSMOLOGICAL SIMULATION',
              ha='center', va='center', color=C_TITLE,
              fontsize=22, fontweight='bold', fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.5, 0.1,
              'N-Body Gravitational Segregation of Positive and Negative Masses  —  Petit et al.',
              ha='center', va='center', color=C_TEXT,
              fontsize=10, fontfamily='monospace',
              transform=ax_h.transAxes)

    # ── Vue 2.5D (gauche) ─────────────────────────────────────────────────────
    ax_3d = fig.add_subplot(gs[1, 0])
    ax_3d.set_facecolor(BG)

    # Grille RGBA 2D — utilise tous les 12M sans sous-échantillonnage
    ISO_RES = 1200  # résolution grille isométrique
    def iso_grid(xi, yi, res, sigma_px=0.8):
        xmin, xmax = xi.min(), xi.max()
        ymin, ymax = yi.min(), yi.max()
        margin = (xmax - xmin) * 0.05
        xi_n = ((xi - xmin + margin) / (xmax - xmin + 2*margin) * res).astype(int)
        yi_n = ((yi - ymin + margin) / (ymax - ymin + 2*margin) * res).astype(int)
        xi_n = np.clip(xi_n, 0, res-1)
        yi_n = np.clip(yi_n, 0, res-1)
        g = np.zeros((res, res), dtype=np.float64)
        np.add.at(g, (yi_n, xi_n), 1)
        g = gaussian_filter(g, sigma=sigma_px)
        return g, xmin-margin, xmax+margin, ymin-margin, ymax+margin

    gp, x0, x1, y0, y1 = iso_grid(xp, yp, ISO_RES)
    gm, *_              = iso_grid(xm, ym, ISO_RES)

    # Normalisation log douce
    def soft_log(g):
        m = g.mean()
        if m < 1e-10: return g
        g = np.log1p(g / (m * 0.05))
        return g / (g.max() + 1e-10)

    gp_n = soft_log(gp)
    gm_n = soft_log(gm)

    # RGBA additif : bleu pour +, rouge pour -
    rgba = np.zeros((ISO_RES, ISO_RES, 4), dtype=np.float32)
    rgba[:,:,0] = np.clip(gp_n * 0.30 + gm_n * 1.00, 0, 1)
    rgba[:,:,1] = np.clip(gp_n * 0.72 + gm_n * 0.33, 0, 1)
    rgba[:,:,2] = np.clip(gp_n * 1.00 + gm_n * 0.20, 0, 1)
    rgba[:,:,3] = np.clip(gp_n + gm_n, 0, 1) ** 0.5

    ax_3d.imshow(rgba, origin='lower', extent=[x0, x1, y0, y1],
                 interpolation='gaussian', aspect='equal')

    ax_3d.set_aspect('equal')
    ax_3d.axis('off')
    ax_3d.set_facecolor(BG)

    # Légende
    ax_3d.scatter([], [], c=C_PLUS,  s=30, label=f'Masses +   N₊ = {len(pos_p)/1e6:.2f}M', alpha=0.9)
    ax_3d.scatter([], [], c=C_MINUS, s=30, label=f'Masses −   N₋ = {len(pos_m)/1e6:.2f}M', alpha=0.9)
    leg = ax_3d.legend(loc='lower left', fontsize=10, framealpha=0,
                       labelcolor=C_TEXT, handletextpad=0.5, borderpad=0.3)
    ax_3d.set_title('Projection isométrique 2.5D',
                    color=C_TEXT, fontsize=11, pad=8, fontfamily='monospace')

    # ── Profil Z (droite) ─────────────────────────────────────────────────────
    ax_z = fig.add_subplot(gs[1, 1])
    ax_z.set_facecolor(BG)

    # Remplissage conditionnel : ρ+ pour Z<0, ρ- pour Z>0
    ax_z.fill_betweenx(zc, rho_p, where=(zc < 0), alpha=0.25, color=C_PLUS)
    ax_z.fill_betweenx(zc, rho_m, where=(zc > 0), alpha=0.25, color=C_MINUS)
    # Courbes
    ax_z.plot(rho_p, zc, color=C_PLUS,  lw=2.0, label='ρ₊(z)', alpha=0.9)
    ax_z.plot(rho_m, zc, color=C_MINUS, lw=2.0, label='ρ₋(z)', alpha=0.9)

    # Ligne Z=0
    ax_z.axhline(0, color='#444466', lw=0.8, ls='--', alpha=0.6)
    # Ligne densité uniforme
    ax_z.axvline(1.0, color='#334433', lw=0.8, ls=':', alpha=0.6)

    ax_z.set_xlabel('Densité normalisée', color=C_TEXT, fontsize=10, fontfamily='monospace')
    ax_z.set_ylabel('Z  (Mpc)', color=C_TEXT, fontsize=10, fontfamily='monospace')
    ax_z.set_ylim(-BOX_MPC/2, BOX_MPC/2)
    ax_z.set_xlim(-0.05, None)
    ax_z.tick_params(colors=C_TEXT, labelsize=8)
    ax_z.set_facecolor(BG)
    for sp in ax_z.spines.values():
        sp.set_edgecolor(C_GRID)
    ax_z.set_title('Profil de densité  ρ(z)',
                   color=C_TEXT, fontsize=11, pad=8, fontfamily='monospace')
    leg2 = ax_z.legend(loc='upper right', fontsize=10, framealpha=0,
                       labelcolor=C_TEXT)

    # Annotation COM
    com_p = pos_p[:, 2].mean()
    com_m = pos_m[:, 2].mean()
    xlim = ax_z.get_xlim()
    xmax = ax_z.get_xlim()[1] if ax_z.get_xlim()[1] > 0 else 3.0
    ax_z.annotate(f'⟨z⟩₊ = {com_p:.0f} Mpc',
                  xy=(0, com_p), xytext=(0.55, com_p),
                  textcoords=('axes fraction', 'data'),
                  color=C_PLUS, fontsize=8, fontfamily='monospace',
                  va='center',
                  arrowprops=dict(arrowstyle='->', color=C_PLUS, lw=1))
    ax_z.annotate(f'⟨z⟩₋ = {com_m:.0f} Mpc',
                  xy=(0, com_m), xytext=(0.55, com_m),
                  textcoords=('axes fraction', 'data'),
                  color=C_MINUS, fontsize=8, fontfamily='monospace',
                  va='center',
                  arrowprops=dict(arrowstyle='->', color=C_MINUS, lw=1))

    # ── Footer ────────────────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.set_facecolor(BG); ax_f.axis('off')

    parts = []
    if z_cosmo is not None: parts.append(f'z = {z_cosmo:.3f}')
    if seg     is not None: parts.append(f'Seg = {seg:.4f}')
    if ke      is not None: parts.append(f'KE/KE₀ = {ke:.3f}')
    if t_gyr   is not None: parts.append(f't = {t_gyr:.2f} Gyr')
    parts.append(f'Step {step:06d}')
    parts.append(f'N = {n/1e6:.2f}M  │  Box = {BOX_MPC:.0f} Mpc  │  η = 1.045')

    ax_f.text(0.5, 0.5, '    ·    '.join(parts),
              ha='center', va='center', color=C_TEXT,
              fontsize=9, fontfamily='monospace',
              transform=ax_f.transAxes)

    print(f"Sauvegarde {out_path}...")
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"Done → {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 render_frame.py <snap.bin> [out.png] [z] [seg] [ke] [t_gyr]")
        sys.exit(1)
    snap   = sys.argv[1]
    out    = sys.argv[2] if len(sys.argv) > 2 else snap.replace('.bin', '.png')
    z_c    = float(sys.argv[3]) if len(sys.argv) > 3 else None
    seg    = float(sys.argv[4]) if len(sys.argv) > 4 else None
    ke     = float(sys.argv[5]) if len(sys.argv) > 5 else None
    t_gyr  = float(sys.argv[6]) if len(sys.argv) > 6 else None
    render(snap, out, z_c, seg, ke, t_gyr)
