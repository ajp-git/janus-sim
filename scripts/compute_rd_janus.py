#!/usr/bin/env python3
"""
compute_rd_janus.py
===================
Calcule l'horizon acoustique r_d^Janus en fonction des paramètres VSL.

Trouve la combinaison (z_c, α) telle que r_d^Janus = r_d^ΛCDM = 147 Mpc,
ce qui résoudrait la tension CC+BAO du modèle.

Usage :
    python compute_rd_janus.py
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = '/mnt/T2/janus-sim/output/paper_figures'
os.makedirs(OUT, exist_ok=True)

# ── Constantes ────────────────────────────────────────────────────────
ETA    = 1.045
H0_JAN = 76.0    # km/s/Mpc (fit Panthéon+)
H0_CDM = 67.4    # km/s/Mpc (Planck)
OM_R   = 9.24e-5 # Ω_r (photons + neutrinos)
Z_DRAG = 1060.0
Z_MAX  = 1e6


# ══════════════════════════════════════════════════════════════════════
# MODÈLES H(z)
# ══════════════════════════════════════════════════════════════════════

def H_lcdm(z, H0=H0_CDM, Om_m=0.308, Om_r=OM_R):
    """H(z) ΛCDM complet (matière + rayonnement + Λ)."""
    Om_L = 1 - Om_m - Om_r
    return H0 * np.sqrt(Om_m*(1+z)**3 + Om_r*(1+z)**4 + Om_L)


def H_janus(z, eta=ETA, H0=H0_JAN, Om_r=OM_R):
    """
    H_Janus(z) incluant le rayonnement.
    Secteur matière : équations Friedmann bimétriques (Petit 2014)
    Secteur rayonnement : contribution standard (non modifiée par Janus)
    """
    Om_ = 1.0/(1+eta)
    E_  = (1-eta)/(1+eta)
    a   = 1.0/(1+z)
    # Matière Janus (adimensionnel, normalisé à H(0)=√Ω+)
    ad2_m = Om_ + 3*E_*(1.0/a - 1.0)
    H0n   = np.sqrt(Om_)   # H_janus(0) adim
    # Conversion adim → physique + ajout rayonnement
    H2_mat = max(ad2_m, 0.0)/a**2 * (H0/H0n)**2
    H2_rad = Om_r * (H0/H0n)**2 * (1+z)**4 / Om_
    # Note : le facteur Om_ normalise Ω_r à l'échelle Janus
    # Simplification : H²_rad = Om_r × H0² × (1+z)⁴ directement
    H2_rad_phys = Om_r * H0**2 * (1+z)**4
    return float(np.sqrt(H2_mat + H2_rad_phys))


def H_janus_vsl(z, z_c, alpha, eta=ETA, H0=H0_JAN):
    """
    H(z) Janus+VSL :
    - z < z_div : H_Janus standard
    - z >= z_div mais z < z_c : interpolation
    - z >= z_c  : régime VSL H = H(z_c) × ((1+z)/(1+z_c))^alpha
    """
    # z_div analytique
    Om_  = 1.0/(1+eta); E_ = (1-eta)/(1+eta)
    z_div = Om_/(3*abs(E_))  # ≈ 7.41

    if z <= min(z_c, z_div - 0.01):
        return H_janus(z, eta, H0)
    else:
        # Raccordement au point de transition
        z_switch = min(z_c, z_div - 0.01)
        Hc = H_janus(z_switch, eta, H0)
        return Hc * ((1+z)/(1+z_switch))**alpha


# ══════════════════════════════════════════════════════════════════════
# VITESSE DU SON
# ══════════════════════════════════════════════════════════════════════

def c_sound(z, Ob_h2=0.02225, T_cmb=2.7255):
    """
    Vitesse du son dans le plasma primordial (km/s).
    c_s = c / √(3(1+R))  avec R = baryon/photon
    R(z) = 31500 × Ω_b h² × (T/2.7K)^{-4} / (1+z)
    """
    A_b = 31500 * Ob_h2 * (T_cmb/2.7)**(-4)
    R   = A_b / (1+z)
    return 299792.458 / np.sqrt(3*(1+R))


# ══════════════════════════════════════════════════════════════════════
# HORIZON ACOUSTIQUE r_d
# ══════════════════════════════════════════════════════════════════════

def compute_rd(H_func, z_drag=Z_DRAG, z_max=Z_MAX):
    """
    r_d = ∫_{z_drag}^{∞} c_s(z) / H(z) dz   [Mpc]
    """
    def intg(z):
        H = H_func(z)
        if H <= 0: return 0.0
        return c_sound(z) / H
    r, err = quad(intg, z_drag, z_max, limit=1000,
                  epsabs=1e-4, epsrel=1e-4)
    return r, err


def compute_rd_janus_vsl(z_c, alpha, eta=ETA, H0=H0_JAN):
    """r_d^Janus avec paramètres VSL (z_c, alpha)."""
    def H_func(z):
        return H_janus_vsl(z, z_c, alpha, eta, H0)
    r, err = compute_rd(H_func)
    return r


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

print("="*60)
print("CALCUL r_d^Janus — Horizon Acoustique BAO")
print("="*60)

# ── Référence ΛCDM ───────────────────────────────────────────────────
rd_lcdm, _ = compute_rd(lambda z: H_lcdm(z))
print(f"\nr_d ΛCDM    = {rd_lcdm:.2f} Mpc  (Planck = 147.09 Mpc)")
print(f"H_Janus(0)  = {H_janus(0):.2f} km/s/Mpc  (fit Panthéon+ = 76)")
print(f"H_Janus(z_drag) = {H_janus(Z_DRAG):.0f} km/s/Mpc")
print(f"z_div       = {1/(3*abs((1-ETA)/(1+ETA))) * (1/(1+ETA)):.3f}")

# ── Scan z_c × alpha ─────────────────────────────────────────────────
print("\n── Scan r_d^Janus(z_c, alpha) ──────────────────────────")
z_c_arr   = np.array([4.0, 5.0, 6.0, 6.5, 7.0])
alpha_arr = np.linspace(1.4, 2.2, 17)

RD_grid = np.zeros((len(z_c_arr), len(alpha_arr)))
for i, z_c in enumerate(z_c_arr):
    for j, alpha in enumerate(alpha_arr):
        RD_grid[i, j] = compute_rd_janus_vsl(z_c, alpha)
    print(f"  z_c={z_c:.1f} : r_d ∈ [{RD_grid[i].min():.0f}, "
          f"{RD_grid[i].max():.0f}] Mpc")

# ── Trouver alpha* tel que r_d = rd_lcdm pour chaque z_c ─────────────
print("\n── Solutions r_d^Janus = r_d^ΛCDM ──────────────────────")
print(f"{'z_c':>6} {'alpha*':>8} {'r_d':>10}  Interp")
print("─"*45)

solutions = []
for i, z_c in enumerate(z_c_arr):
    # Interpoler pour trouver alpha* via spline
    rd_arr = RD_grid[i]
    # Trouver intervalle où rd croise rd_lcdm
    idx = np.where(np.diff(np.sign(rd_arr - rd_lcdm)))[0]
    if len(idx) > 0:
        j0 = idx[0]
        # Interpolation linéaire
        a0, a1 = alpha_arr[j0], alpha_arr[j0+1]
        r0, r1 = rd_arr[j0], rd_arr[j0+1]
        alpha_star = a0 + (rd_lcdm - r0)/(r1 - r0) * (a1 - a0)
        rd_star = compute_rd_janus_vsl(z_c, alpha_star)
        solutions.append((z_c, alpha_star, rd_star))
        print(f"  {z_c:4.1f}   {alpha_star:7.3f}   {rd_star:8.1f} Mpc  ✓")
    else:
        print(f"  {z_c:4.1f}   {'—':>7}   {'pas de solution':>10}")

# ── Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Horizon acoustique $r_d$ — Modèle Janus vs ΛCDM\n'
             f'η={ETA}, $H_0^{{Janus}}$={H0_JAN} km/s/Mpc',
             fontsize=11)

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_c_arr)))

# Panneau gauche : r_d(alpha) pour chaque z_c
ax = axes[0]
for i, (z_c, c) in enumerate(zip(z_c_arr, colors)):
    ax.semilogy(alpha_arr, RD_grid[i], color=c, lw=2,
                label=f'$z_c$={z_c:.1f}')

ax.axhline(rd_lcdm, color='k', ls='--', lw=2,
           label=f'$r_d^{{\\Lambda CDM}}$ = {rd_lcdm:.0f} Mpc')
ax.axhspan(rd_lcdm*0.9, rd_lcdm*1.1, alpha=0.1, color='k',
           label='±10%')

# Marquer les solutions
for z_c, alpha_star, rd_star in solutions:
    ax.plot(alpha_star, rd_star, 'k*', ms=14, zorder=5)

ax.set_xlabel('Exposant VSL $\\alpha$', fontsize=10)
ax.set_ylabel('$r_d$ (Mpc)', fontsize=10)
ax.set_title('$r_d^{Janus}(\\alpha)$ pour différents $z_c$\n'
             '★ = solution $r_d^{Janus} = r_d^{\\Lambda CDM}$',
             fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')
ax.set_ylim(10, 5000)

# Panneau droit : comparaison H(z)
ax = axes[1]
z_plot = np.logspace(-1, 3.2, 300)
H_janus_arr = np.array([H_janus(z) for z in z_plot])
H_lcdm_arr  = np.array([H_lcdm(z)  for z in z_plot])

ax.loglog(z_plot, H_janus_arr, 'b-', lw=2, label=f'Janus ($H_0$={H0_JAN})')
ax.loglog(z_plot, H_lcdm_arr,  'r--', lw=2, label=f'ΛCDM ($H_0$={H0_CDM})')

# H_VSL pour les solutions trouvées
for z_c, alpha_star, _ in solutions:
    H_vsl_arr = np.array([H_janus_vsl(z, z_c, alpha_star)
                           for z in z_plot])
    ax.loglog(z_plot, H_vsl_arr, '--', lw=1, alpha=0.6,
              label=f'Janus+VSL $z_c$={z_c:.1f}, α={alpha_star:.2f}')

ax.axvline(Z_DRAG, color='gray', ls=':', lw=1)
ax.text(Z_DRAG*1.1, 1e3, '$z_{drag}$', fontsize=8, color='gray')
ax.set_xlabel('Redshift $z$', fontsize=10)
ax.set_ylabel('$H(z)$ (km/s/Mpc)', fontsize=10)
ax.set_title('Comparaison $H(z)$ : Janus vs ΛCDM\n'
             'avec extension VSL au-delà de $z_{div}$', fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
out = os.path.join(OUT, 'figure4_rd_janus.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure → {out}")

# ── Rapport final ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RÉSULTATS CLÉS")
print(f"{'='*60}")
print(f"r_d ΛCDM (Planck)  = {rd_lcdm:.1f} Mpc")
print()
print("Solutions r_d^Janus = r_d^ΛCDM :")
for z_c, alpha_star, rd_star in solutions:
    print(f"  z_c = {z_c:.1f}  →  alpha* = {alpha_star:.3f}  "
          f"(r_d = {rd_star:.1f} Mpc)")
print()
print("Interprétation :")
print("  alpha = exposant VSL dans H_VSL = H(z_c) × ((1+z)/(1+z_c))^alpha")
print("  Petit 1995 prédit alpha ≈ 1.5 pour c ∝ a^(-1/2)")
print("  Si alpha_star ≈ 1.5-1.8 → cohérent avec VSL de Petit")
print("  Si alpha_star >> 2 → régime VSL plus agressif nécessaire")
