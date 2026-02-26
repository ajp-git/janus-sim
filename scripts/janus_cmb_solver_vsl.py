#!/usr/bin/env python3
"""
JANUS VSL CMB SOLVER — VERSION STRICTE
Implémentation conforme à janus_cmb_solver_vsl_strict.md

Test de falsifiabilité du modèle Janus VSL pour le CMB.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# 1. PARAMÈTRES FIXES (AUCUN FIT)
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("JANUS VSL CMB SOLVER — VERSION STRICTE")
print("=" * 70)
print()

# Constantes cosmologiques
H0_km_s_Mpc = 76.0          # km/s/Mpc — contraint par Pantheon+
Omega_b = 0.0493            # fraction baryonique
T0 = 2.725                  # K, CMB aujourd'hui
c0_km_s = 299792.458        # km/s

# Constantes physiques SI
G = 6.67430e-11             # m³/(kg·s²)
kB = 1.380649e-23           # J/K
me = 9.10938e-31            # kg (masse électron)
mp = 1.67262e-27            # kg (masse proton)
h_planck = 6.62607e-34      # J·s
sigma_T = 6.6524e-29        # m² (section efficace Thomson)
eV_to_J = 1.60218e-19       # J/eV
Ei = 13.6 * eV_to_J         # énergie ionisation H en J

# Conversions
H0_SI = H0_km_s_Mpc * 1000 / (3.086e22)  # s⁻¹
c0_SI = c0_km_s * 1000                    # m/s
Mpc_to_m = 3.086e22

# Omega_gamma (rayonnement)
h = H0_km_s_Mpc / 100
Omega_gamma = 2.47e-5 / h**2

# Densité critique
rho_crit = 3 * H0_SI**2 / (8 * np.pi * G)  # kg/m³

# Domaine d'intégration
z_max = 10000
z_min = 0

print("Paramètres fixes:")
print(f"  H0 = {H0_km_s_Mpc} km/s/Mpc")
print(f"  Ωb = {Omega_b}")
print(f"  T0 = {T0} K")
print(f"  c0 = {c0_km_s} km/s")
print(f"  Ωγ = {Omega_gamma:.4e}")
print(f"  z_max = {z_max}")
print()

# ══════════════════════════════════════════════════════════════════
# 2. ÉQUATION DE SAHA — IONISATION
# ══════════════════════════════════════════════════════════════════
print("Étape 2: Calcul x_e(z) via Saha...")

def n_b(z):
    """Densité baryonique en m⁻³"""
    return (rho_crit * Omega_b / mp) * (1 + z)**3

def T(z):
    """Température CMB en K"""
    return T0 * (1 + z)

def saha_S(z):
    """Paramètre S de l'équation de Saha"""
    T_z = T(z)
    nb_z = n_b(z)

    # S = (1/nb) * (2π me kB T / h²)^(3/2) * exp(-Ei / kB T)
    thermal_factor = (2 * np.pi * me * kB * T_z / h_planck**2)**(1.5)
    boltzmann = np.exp(-Ei / (kB * T_z))

    return thermal_factor * boltzmann / nb_z

def x_e_saha(z):
    """Fraction d'ionisation via Saha exacte: x_e²/(1-x_e) = S"""
    S = saha_S(z)

    # Solution analytique: x_e = (-S + sqrt(S² + 4S)) / 2
    # Mais pour éviter instabilités numériques:
    if S > 1e10:
        return 1.0
    elif S < 1e-10:
        return np.sqrt(S)  # approximation x_e << 1

    x_e = (-S + np.sqrt(S**2 + 4*S)) / 2
    return np.clip(x_e, 0.0, 1.0)

# Vérification Saha
print("\n  Vérification x_e(z) via Saha:")
test_z = [800, 1100, 1300, 1400, 1500, 2000]
for z in test_z:
    print(f"    z={z:4d}: x_e = {x_e_saha(z):.4f}")

# ══════════════════════════════════════════════════════════════════
# 3. DÉTERMINATION DE z_drag (x_e = 0.5)
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 3: Détermination z_drag...")

def x_e_minus_half(z):
    return x_e_saha(z) - 0.5

z_drag = brentq(x_e_minus_half, 1000, 2000)
print(f"  z_drag (x_e = 0.5) = {z_drag:.1f}")

# ══════════════════════════════════════════════════════════════════
# 4. ÉVOLUTION DE c(z) — INTÉGRATION dc/dz
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 4: Intégration dc/dz...")
print("  dc/dz = +c(z)/2 * x_e(z) / (1+z)")
print(f"  CI: c(z={z_max}) = c0")

def dc_dz(z, c):
    """Équation différentielle pour c(z)"""
    x_e = x_e_saha(z)
    return c * x_e / (2 * (1 + z))

# Intégration de z_max vers 0 (sens décroissant)
z_eval = np.linspace(z_max, z_min, 10001)

# solve_ivp veut t croissant, donc on utilise u = -z
def dc_du(u, c):
    z = -u
    return -dc_dz(z, c)

sol = solve_ivp(
    dc_du,
    [-z_max, -z_min],
    [c0_SI],
    t_eval=-z_eval,
    method='RK45',
    dense_output=True
)

# Interpolateur pour c(z)
def c_of_z(z):
    """c(z) en m/s, interpolé depuis la solution"""
    return sol.sol(-z)[0]

# Vérification: c devrait ~= c0 * sqrt(1+z)/sqrt(1+z_max) pour z >> z_drag
print("\n  Vérification c(z)/c0:")
for z in [0, 500, 1000, 1500, 2000, 5000, 10000]:
    c_ratio = c_of_z(z) / c0_SI
    expected = np.sqrt(1 + z) / np.sqrt(1 + z_max)
    print(f"    z={z:5d}: c/c0 = {c_ratio:.6f}  (théorique x_e=1: {expected:.6f})")

# ══════════════════════════════════════════════════════════════════
# 5. H(z) — RÉGIME VSL
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 5: H(z) = H0*(1+z) dans régime VSL")

def H_vsl(z):
    """H(z) en s⁻¹ — régime VSL"""
    return H0_SI * (1 + z)

# ══════════════════════════════════════════════════════════════════
# 6. VITESSE DU SON
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 6: Vitesse du son c_s(z)")

def R_baryon(z):
    """Rapport baryon/photon"""
    return (3 * Omega_b / (4 * Omega_gamma)) / (1 + z)

def c_s(z):
    """Vitesse du son en m/s"""
    c_z = c_of_z(z)
    R_z = R_baryon(z)
    return c_z / np.sqrt(3 * (1 + R_z))

# ══════════════════════════════════════════════════════════════════
# 7. RAYON ACOUSTIQUE r_d
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 7: Calcul r_d...")
print(f"  r_d = ∫_{z_drag:.0f}^{z_max} c_s(z)/H(z) dz")

def integrand_rd(z):
    """Intégrande pour r_d en m"""
    return c_s(z) / H_vsl(z)

# Intégration numérique
r_d_m, r_d_err = quad(integrand_rd, z_drag, z_max, limit=1000)
r_d_Mpc = r_d_m / Mpc_to_m

print(f"  r_d = {r_d_Mpc:.2f} Mpc")
print(f"  Cible ΛCDM: 147 Mpc")
print(f"  Écart: {100 * abs(r_d_Mpc - 147) / 147:.1f}%")

# ══════════════════════════════════════════════════════════════════
# 8. DISTANCE DE DIAMÈTRE ANGULAIRE D_M
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 8: Calcul D_M...")
print(f"  D_M = ∫_0^{z_drag:.0f} c(z)/H(z) dz")

def integrand_DM(z):
    """Intégrande pour D_M en m"""
    return c_of_z(z) / H_vsl(z)

D_M_m, D_M_err = quad(integrand_DM, 0, z_drag, limit=1000)
D_M_Mpc = D_M_m / Mpc_to_m

print(f"  D_M = {D_M_Mpc:.2f} Mpc")

# ══════════════════════════════════════════════════════════════════
# 9. PREMIER PIC CMB
# ══════════════════════════════════════════════════════════════════
print("\nÉtape 9: Premier pic CMB")

ell_1 = np.pi * D_M_m / r_d_m

print(f"  ℓ₁ = π × D_M / r_d = {ell_1:.1f}")
print(f"  Cible: 220")
print(f"  Écart: {100 * abs(ell_1 - 220) / 220:.1f}%")

# ══════════════════════════════════════════════════════════════════
# 10. VALIDATION STRICTE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VALIDATION STRICTE")
print("=" * 70)

err_rd = abs(r_d_Mpc - 147) / 147
err_ell = abs(ell_1 - 220) / 220

print(f"\n  |r_d − 147| / 147 = {100*err_rd:.1f}%  {'✓' if err_rd < 0.10 else '✗'} (seuil: 10%)")
print(f"  |ℓ₁ − 220| / 220  = {100*err_ell:.1f}%  {'✓' if err_ell < 0.10 else '✗'} (seuil: 10%)")

if err_rd < 0.10 and err_ell < 0.10:
    print("\n  ➜ MODÈLE VIABLE")
    validation_status = "VIABLE"
else:
    print("\n  ➜ FALSIFICATION DOCUMENTÉE")
    validation_status = "FALSIFIÉ"

# ══════════════════════════════════════════════════════════════════
# 11. BALAYAGE η
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BALAYAGE η ∈ [1.05, 5.0]")
print("=" * 70)
print("\nNote: Dans le régime VSL, η n'affecte que H(z) pour z < z_drag")
print("      Saha et dc/dz ne dépendent pas de η")
print()

eta_grid = np.linspace(1.05, 5.0, 40)
results_eta = []

def H_janus_friedmann(z, eta):
    """H(z) Janus Friedmann pour z < z_drag"""
    Omega_plus = 1.0 / (1.0 + eta)
    # Simplifié: H² = H0² * [Ω₊(1+z)³ + (1-Ω₊)]
    H_sq = H0_SI**2 * (Omega_plus * (1 + z)**3 + (1 - Omega_plus))
    if H_sq > 0:
        return np.sqrt(H_sq)
    else:
        return H0_SI * (1 + z)  # fallback si H²<0

print(f"{'η':>6} {'r_d (Mpc)':>12} {'D_M (Mpc)':>12} {'ℓ₁':>8} {'Status':>10}")
print("-" * 52)

for eta in eta_grid:
    # r_d reste identique (régime VSL z > z_drag)
    rd_eta = r_d_Mpc

    # D_M avec H Janus
    def integrand_DM_eta(z):
        return c_of_z(z) / H_janus_friedmann(z, eta)

    DM_eta_m, _ = quad(integrand_DM_eta, 0, z_drag, limit=500)
    DM_eta_Mpc = DM_eta_m / Mpc_to_m

    ell_eta = np.pi * DM_eta_m / r_d_m

    err_rd_eta = abs(rd_eta - 147) / 147
    err_ell_eta = abs(ell_eta - 220) / 220

    status = "✓" if (err_rd_eta < 0.10 and err_ell_eta < 0.10) else ""

    results_eta.append({
        'eta': eta,
        'r_d_Mpc': rd_eta,
        'D_M_Mpc': DM_eta_Mpc,
        'ell_1': ell_eta,
        'err_rd': err_rd_eta,
        'err_ell': err_ell_eta
    })

    if len(results_eta) % 10 == 0 or status:
        print(f"{eta:6.2f} {rd_eta:12.2f} {DM_eta_Mpc:12.2f} {ell_eta:8.1f} {status:>10}")

# ══════════════════════════════════════════════════════════════════
# 12. GÉNÉRATION DES FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GÉNÉRATION DES FIGURES")
print("=" * 70)

output_dir = '/mnt/T2/janus-sim/output/cmb_vsl'
os.makedirs(output_dir, exist_ok=True)

# Figure principale: 4 panels
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Janus VSL CMB Solver — Test de Falsifiabilité', fontsize=14, fontweight='bold')

# Panel 1: x_e(z)
ax1 = axes[0, 0]
z_plot = np.linspace(800, 2000, 500)
x_e_plot = [x_e_saha(z) for z in z_plot]
ax1.plot(z_plot, x_e_plot, 'b-', lw=2)
ax1.axhline(0.5, color='gray', ls='--', label='x_e = 0.5')
ax1.axvline(z_drag, color='r', ls='--', label=f'z_drag = {z_drag:.0f}')
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('Fraction d\'ionisation x_e')
ax1.set_title('Ionisation Saha')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(800, 2000)
ax1.set_ylim(0, 1.05)

# Panel 2: c(z)/c0
ax2 = axes[0, 1]
z_plot2 = np.linspace(0, 2000, 500)
c_plot = [c_of_z(z) / c0_SI for z in z_plot2]
c_theory = [np.sqrt(1 + z) / np.sqrt(1 + z_max) for z in z_plot2]
ax2.plot(z_plot2, c_plot, 'b-', lw=2, label='c(z) intégré')
ax2.plot(z_plot2, c_theory, 'r--', lw=1, alpha=0.5, label=r'$\sqrt{(1+z)/(1+z_{max})}$ (x_e=1)')
ax2.axvline(z_drag, color='orange', ls=':', label=f'z_drag = {z_drag:.0f}')
ax2.set_xlabel('Redshift z')
ax2.set_ylabel('c(z) / c₀')
ax2.set_title('Évolution de c(z) — VSL dynamique')
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3: Intégrande c_s/H
ax3 = axes[1, 0]
z_plot3 = np.linspace(z_drag - 200, z_drag + 500, 300)
z_plot3 = z_plot3[z_plot3 > 0]
integrand_plot = [integrand_rd(z) / Mpc_to_m for z in z_plot3]  # en Mpc
ax3.plot(z_plot3, integrand_plot, 'g-', lw=2)
ax3.axvline(z_drag, color='r', ls='--', label=f'z_drag = {z_drag:.0f}')
ax3.fill_between(z_plot3, integrand_plot, alpha=0.3, color='green')
ax3.set_xlabel('Redshift z')
ax3.set_ylabel('c_s(z) / H(z)  [Mpc]')
ax3.set_title('Intégrande du rayon acoustique')
ax3.legend()
ax3.grid(alpha=0.3)

# Panel 4: Tableau de résultats
ax4 = axes[1, 1]
ax4.axis('off')

results_text = f"""
RÉSULTATS — η primordial (régime VSL)

z_drag = {z_drag:.1f}  (x_e = 0.5, Saha)

┌─────────────────────────────────────────┐
│  Quantité    Calculé    Cible    Écart  │
├─────────────────────────────────────────┤
│  r_d (Mpc)   {r_d_Mpc:7.2f}     147    {100*err_rd:5.1f}%  │
│  ℓ₁          {ell_1:7.1f}     220    {100*err_ell:5.1f}%  │
└─────────────────────────────────────────┘

Validation (seuil 10%):
  • r_d:  {'PASS' if err_rd < 0.10 else 'FAIL'}
  • ℓ₁:   {'PASS' if err_ell < 0.10 else 'FAIL'}

Verdict: {validation_status}

Paramètres:
  H0 = {H0_km_s_Mpc} km/s/Mpc
  Ωb = {Omega_b}
  H(z) = H0×(1+z)  [régime VSL]
"""

ax4.text(0.1, 0.95, results_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_cmb_results.png', dpi=150, bbox_inches='tight')
print(f"  ✓ {output_dir}/janus_cmb_results.png")

# Figure balayage η
fig2, (ax_rd, ax_ell) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Balayage η — Janus VSL CMB', fontsize=14, fontweight='bold')

etas = [r['eta'] for r in results_eta]
rds = [r['r_d_Mpc'] for r in results_eta]
ells = [r['ell_1'] for r in results_eta]

ax_rd.plot(etas, rds, 'bo-', lw=2, ms=4)
ax_rd.axhline(147, color='r', ls='--', lw=2, label='Cible: 147 Mpc')
ax_rd.axhline(147*0.9, color='orange', ls=':', alpha=0.7, label='±10%')
ax_rd.axhline(147*1.1, color='orange', ls=':', alpha=0.7)
ax_rd.set_xlabel('η')
ax_rd.set_ylabel('r_d (Mpc)')
ax_rd.set_title('Rayon acoustique r_d(η)')
ax_rd.legend()
ax_rd.grid(alpha=0.3)

ax_ell.plot(etas, ells, 'go-', lw=2, ms=4)
ax_ell.axhline(220, color='r', ls='--', lw=2, label='Cible: ℓ₁ = 220')
ax_ell.axhline(220*0.9, color='orange', ls=':', alpha=0.7, label='±10%')
ax_ell.axhline(220*1.1, color='orange', ls=':', alpha=0.7)
ax_ell.set_xlabel('η')
ax_ell.set_ylabel('ℓ₁')
ax_ell.set_title('Premier pic CMB ℓ₁(η)')
ax_ell.legend()
ax_ell.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_eta_scan.png', dpi=150, bbox_inches='tight')
print(f"  ✓ {output_dir}/janus_eta_scan.png")

# ══════════════════════════════════════════════════════════════════
# 13. SAUVEGARDE DES DONNÉES
# ══════════════════════════════════════════════════════════════════
print("\nSauvegarde des données...")

# JSON summary (will be updated after n scan)
summary = {
    'timestamp': datetime.now().isoformat(),
    'parameters': {
        'H0_km_s_Mpc': H0_km_s_Mpc,
        'Omega_b': Omega_b,
        'T0': T0,
        'c0_km_s': c0_km_s,
        'z_max': z_max
    },
    'results': {
        'z_drag': z_drag,
        'r_d_Mpc': r_d_Mpc,
        'D_M_Mpc': D_M_Mpc,
        'ell_1': ell_1,
        'err_rd_percent': 100 * err_rd,
        'err_ell_percent': 100 * err_ell,
        'validation_status': validation_status
    },
    'eta_scan': results_eta,
    'n_scan': []  # Will be populated after n scan
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ {output_dir}/summary.json")

# CSV x_e(z)
import csv
with open(f'{output_dir}/ionization_xe.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['z', 'x_e', 'T_K'])
    for z in np.linspace(500, 3000, 501):
        writer.writerow([z, x_e_saha(z), T(z)])
print(f"  ✓ {output_dir}/ionization_xe.csv")

# CSV c(z)
with open(f'{output_dir}/c_of_z.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['z', 'c_m_s', 'c_over_c0'])
    for z in np.linspace(0, z_max, 1001):
        c_z = c_of_z(z)
        writer.writerow([z, c_z, c_z/c0_SI])
print(f"  ✓ {output_dir}/c_of_z.csv")

# CSV eta scan
with open(f'{output_dir}/eta_scan.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['eta', 'r_d_Mpc', 'D_M_Mpc', 'ell_1', 'err_rd', 'err_ell'])
    for r in results_eta:
        writer.writerow([r['eta'], r['r_d_Mpc'], r['D_M_Mpc'], r['ell_1'], r['err_rd'], r['err_ell']])
print(f"  ✓ {output_dir}/eta_scan.csv")

# ══════════════════════════════════════════════════════════════════
# 14. TEST COMPLÉMENTAIRE — EXPOSANT n DE H(z)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST COMPLÉMENTAIRE: EXPOSANT n DANS H(z) = H0*(1+z)^n")
print("=" * 70)
print("\nAnalyse théorique:")
print("  c_s/H ∝ c(z)^0.5 / (1+z)^n = (1+z)^(0.5 - n)")
print("  n=1.0 → divergence forte")
print("  n=1.5 → logarithmique")
print("  n=2.0 → convergence forte")
print()

n_grid = np.linspace(1.0, 2.0, 21)
results_n = []

def H_power_n(z, n):
    """H(z) = H0 * (1+z)^n"""
    return H0_SI * (1 + z)**n

print(f"{'n':>6} {'r_d (Mpc)':>12} {'D_M (Mpc)':>12} {'ell_1':>8} {'err_rd%':>10} {'err_ell%':>10}")
print("-" * 62)

n_critical_rd = None
best_n = None
best_err_total = float('inf')

for n in n_grid:
    # r_d avec H(z) = H0*(1+z)^n
    def integrand_rd_n(z):
        return c_s(z) / H_power_n(z, n)

    rd_n_m, _ = quad(integrand_rd_n, z_drag, z_max, limit=1000)
    rd_n_Mpc = rd_n_m / Mpc_to_m

    # D_M avec H(z) = H0*(1+z)^n
    def integrand_DM_n(z):
        return c_of_z(z) / H_power_n(z, n)

    DM_n_m, _ = quad(integrand_DM_n, 0, z_drag, limit=500)
    DM_n_Mpc = DM_n_m / Mpc_to_m

    ell_n = np.pi * DM_n_m / rd_n_m

    err_rd_n = abs(rd_n_Mpc - 147) / 147
    err_ell_n = abs(ell_n - 220) / 220
    err_total = err_rd_n + err_ell_n

    results_n.append({
        'n': n,
        'r_d_Mpc': rd_n_Mpc,
        'D_M_Mpc': DM_n_Mpc,
        'ell_1': ell_n,
        'err_rd': err_rd_n,
        'err_ell': err_ell_n
    })

    # Trouver n critique où r_d ≈ 147
    if n_critical_rd is None and rd_n_Mpc < 147:
        n_critical_rd = n

    # Trouver meilleur n
    if err_total < best_err_total:
        best_err_total = err_total
        best_n = n

    status = "✓" if (err_rd_n < 0.10 and err_ell_n < 0.10) else ""
    print(f"{n:6.2f} {rd_n_Mpc:12.2f} {DM_n_Mpc:12.2f} {ell_n:8.1f} {100*err_rd_n:10.1f} {100*err_ell_n:10.1f} {status}")

print()
if n_critical_rd:
    print(f"n critique (r_d = 147 Mpc) ≈ {n_critical_rd:.2f}")
print(f"Meilleur n (erreur totale min) = {best_n:.2f}")

# Figure n scan
fig3, (ax_rd_n, ax_ell_n) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Balayage exposant n — H(z) = H₀(1+z)^n', fontsize=14, fontweight='bold')

ns = [r['n'] for r in results_n]
rds_n = [r['r_d_Mpc'] for r in results_n]
ells_n = [r['ell_1'] for r in results_n]

ax_rd_n.plot(ns, rds_n, 'bo-', lw=2, ms=6)
ax_rd_n.axhline(147, color='r', ls='--', lw=2, label='Cible: 147 Mpc')
ax_rd_n.axhline(147*0.9, color='orange', ls=':', alpha=0.7, label='±10%')
ax_rd_n.axhline(147*1.1, color='orange', ls=':', alpha=0.7)
if n_critical_rd:
    ax_rd_n.axvline(n_critical_rd, color='green', ls=':', lw=2, label=f'n_crit ≈ {n_critical_rd:.2f}')
ax_rd_n.set_xlabel('Exposant n')
ax_rd_n.set_ylabel('r_d (Mpc)')
ax_rd_n.set_title('Rayon acoustique r_d(n)')
ax_rd_n.legend()
ax_rd_n.grid(alpha=0.3)
ax_rd_n.set_yscale('log')

ax_ell_n.plot(ns, ells_n, 'go-', lw=2, ms=6)
ax_ell_n.axhline(220, color='r', ls='--', lw=2, label='Cible: ℓ₁ = 220')
ax_ell_n.axhline(220*0.9, color='orange', ls=':', alpha=0.7, label='±10%')
ax_ell_n.axhline(220*1.1, color='orange', ls=':', alpha=0.7)
ax_ell_n.set_xlabel('Exposant n')
ax_ell_n.set_ylabel('ℓ₁')
ax_ell_n.set_title('Premier pic CMB ℓ₁(n)')
ax_ell_n.legend()
ax_ell_n.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_n_scan.png', dpi=150, bbox_inches='tight')
print(f"\n  ✓ {output_dir}/janus_n_scan.png")

# CSV n scan
with open(f'{output_dir}/n_scan.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n', 'r_d_Mpc', 'D_M_Mpc', 'ell_1', 'err_rd', 'err_ell'])
    for r in results_n:
        writer.writerow([r['n'], r['r_d_Mpc'], r['D_M_Mpc'], r['ell_1'], r['err_rd'], r['err_ell']])
print(f"  ✓ {output_dir}/n_scan.csv")

# Update summary with n scan results
summary['n_scan'] = results_n
summary['n_scan_analysis'] = {
    'n_critical_rd': n_critical_rd,
    'best_n': best_n,
    'best_err_total': best_err_total
}

# Rewrite summary.json with n scan data
with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ {output_dir}/summary.json (updated with n scan)")

# ══════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"""
Modèle: Janus VSL
  - dc/dz = +c(z)/2 × x_e(z)/(1+z)
  - H(z) = H0 × (1+z)^n
  - x_e(z) via Saha exacte

Résultats (n=1):
  z_drag = {z_drag:.1f}
  r_d    = {r_d_Mpc:.2f} Mpc  (cible: 147 Mpc, écart: {100*err_rd:.1f}%)
  ℓ₁     = {ell_1:.1f}       (cible: 220, écart: {100*err_ell:.1f}%)

VERDICT (n=1): {validation_status}

Test exposant n:
  n critique (r_d=147) ≈ {n_critical_rd if n_critical_rd else 'N/A'}
  Meilleur n           = {best_n:.2f}

Fichiers générés:
  {output_dir}/janus_cmb_results.png
  {output_dir}/janus_eta_scan.png
  {output_dir}/janus_n_scan.png
  {output_dir}/summary.json
  {output_dir}/ionization_xe.csv
  {output_dir}/c_of_z.csv
  {output_dir}/eta_scan.csv
  {output_dir}/n_scan.csv
""")
print("=" * 70)
