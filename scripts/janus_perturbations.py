#!/usr/bin/env python3
"""
JANUS — Solveur perturbations couplées CMB
Calcul de ℓ_peak SANS utiliser ℓ₁ = π D_M / r_d
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("JANUS — SOLVEUR PERTURBATIONS COUPLÉES CMB")
print("Calcul direct de ℓ_peak sans formule r_d")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. CONSTANTES ET PARAMÈTRES
# ══════════════════════════════════════════════════════════════════
H0_km_s_Mpc = 76.0
Omega_b = 0.0493
T0 = 2.725
c0_SI = 299792458.0  # m/s
G = 6.67430e-11
kB = 1.380649e-23
me = 9.10938e-31
mp = 1.67262e-27
h_planck = 6.62607e-34
Ei = 13.6 * 1.60218e-19
Mpc_to_m = 3.086e22

H0_SI = H0_km_s_Mpc * 1000 / Mpc_to_m  # s⁻¹
h = H0_km_s_Mpc / 100
Omega_gamma = 2.47e-5 / h**2
rho_crit_0 = 3 * H0_SI**2 / (8 * np.pi * G)  # kg/m³

z_init = 10000
z_BBN = 1e9

print(f"\nParamètres:")
print(f"  H0 = {H0_km_s_Mpc} km/s/Mpc")
print(f"  z_init = {z_init}")

# ══════════════════════════════════════════════════════════════════
# 2. IONISATION SAHA
# ══════════════════════════════════════════════════════════════════

def x_e_saha(z):
    T_z = T0 * (1 + z)
    nb_z = (rho_crit_0 * Omega_b / mp) * (1 + z)**3
    thermal = (2 * np.pi * me * kB * T_z / h_planck**2)**1.5
    S = thermal * np.exp(-Ei / (kB * T_z)) / nb_z
    if S > 1e10: return 1.0
    if S < 1e-10: return max(0, np.sqrt(S))
    return np.clip((-S + np.sqrt(S**2 + 4*S)) / 2, 0, 1)

z_drag = brentq(lambda z: x_e_saha(z) - 0.5, 1000, 2000)
print(f"  z_drag = {z_drag:.1f}")

# ══════════════════════════════════════════════════════════════════
# 3. MODÈLE H(z) — 3 RÉGIMES (meilleur du scan précédent)
# ══════════════════════════════════════════════════════════════════
# Utiliser les valeurs optimales du test sans continuité
n_high = 1.34
n_low = 1.03

def H_of_z(z):
    """H(z) modèle 2 régimes (sans continuité stricte pour test)"""
    if z > z_drag:
        return H0_SI * (1 + z)**n_high
    else:
        return H0_SI * (1 + z)**n_low

def dH_dz(z, eps=1e-3):
    """Dérivée numérique de H"""
    return (H_of_z(z + eps) - H_of_z(z - eps)) / (2 * eps)

print(f"\nModèle H(z):")
print(f"  n_high = {n_high} (z > z_drag)")
print(f"  n_low = {n_low} (z < z_drag)")

# ══════════════════════════════════════════════════════════════════
# 4. ÉVOLUTION c(z) — VSL
# ══════════════════════════════════════════════════════════════════

def dc_du(u, c):
    z = -u
    return -c[0] * x_e_saha(z) / (2 * (1 + z))

sol_c = solve_ivp(dc_du, [-z_init, 0], [c0_SI], method='RK45', dense_output=True)

def c_of_z(z):
    if z > z_init:
        return c0_SI * np.sqrt((1 + z) / (1 + z_init))
    return float(sol_c.sol(-z)[0])

print(f"\nVSL:")
print(f"  c(z_drag)/c0 = {c_of_z(z_drag)/c0_SI:.4f}")
print(f"  c(0)/c0 = {c_of_z(0)/c0_SI:.4f}")

# ══════════════════════════════════════════════════════════════════
# 5. VITESSE DU SON
# ══════════════════════════════════════════════════════════════════

def R_baryon(z):
    return (3 * Omega_b / (4 * Omega_gamma)) / (1 + z)

def c_s(z):
    """Vitesse du son dans le plasma"""
    return c_of_z(z) / np.sqrt(3 * (1 + R_baryon(z)))

# ══════════════════════════════════════════════════════════════════
# 6. DENSITÉS
# ══════════════════════════════════════════════════════════════════

def rho_plus(z):
    """Densité matière positive ∝ (1+z)³"""
    return rho_crit_0 * (1 - Omega_b) * (1 + z)**3  # Approximation

def rho_minus(z, eta):
    """Densité matière négative = η × ρ_plus"""
    return eta * rho_plus(z)

# ══════════════════════════════════════════════════════════════════
# 7. SYSTÈME D'ÉQUATIONS PERTURBATIONS
# ══════════════════════════════════════════════════════════════════
"""
Équations originales (en t):
d²δ+/dt² + 2H dδ+/dt + cs²k²δ+ = 4πG(ρ+δ+ - ρ-δ-)
d²δ-/dt² + 2H dδ-/dt + cs²k²δ- = 4πG(ρ-δ- - ρ+δ+)

Changement: d/dt = -H(1+z) d/dz

Variables: y = [δ+, δ+', δ-, δ-']  où ' = d/dz
"""

def perturbation_system(z, y, k, eta):
    """
    Système différentiel pour les perturbations
    y = [delta_plus, delta_plus_prime, delta_minus, delta_minus_prime]
    """
    delta_p, delta_p_prime, delta_m, delta_m_prime = y

    H = H_of_z(z)
    H_prime = dH_dz(z)
    cs = c_s(z)

    rho_p = rho_plus(z)
    rho_m = rho_minus(z, eta)

    # Coefficient pour la conversion t → z
    # d²δ/dt² = H²(1+z)² d²δ/dz² + H(1+z)[H'(1+z) + H] dδ/dz
    # 2H dδ/dt = -2H²(1+z) dδ/dz

    H2 = H**2
    z1 = 1 + z

    # Termes sources Janus
    source_p = 4 * np.pi * G * (rho_p * delta_p - rho_m * delta_m)
    source_m = 4 * np.pi * G * (rho_m * delta_m - rho_p * delta_p)

    # Coefficient de friction effective
    # De l'équation: d²δ/dz² + A dδ/dz + B δ = S
    # A = [H'(1+z) + H - 2H] / [H(1+z)] = [H'(1+z) - H] / [H(1+z)]
    # B = cs²k² / [H²(1+z)²]
    # S = source / [H²(1+z)²]

    A = (H_prime * z1 - H) / (H * z1)
    B_p = (cs * k)**2 / (H2 * z1**2)
    B_m = B_p  # Même vitesse du son pour les deux

    S_p = source_p / (H2 * z1**2)
    S_m = source_m / (H2 * z1**2)

    # Équations
    d_delta_p = delta_p_prime
    d_delta_p_prime = -A * delta_p_prime - B_p * delta_p + S_p

    d_delta_m = delta_m_prime
    d_delta_m_prime = -A * delta_m_prime - B_m * delta_m + S_m

    return [d_delta_p, d_delta_p_prime, d_delta_m, d_delta_m_prime]

# ══════════════════════════════════════════════════════════════════
# 8. RÉSOLUTION POUR DIFFÉRENTS k
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RÉSOLUTION DES PERTURBATIONS")
print("=" * 70)

# Conditions initiales à z_init
delta_init = 1e-5
delta_prime_init = 0.0
y0 = [delta_init, delta_prime_init, delta_init, delta_prime_init]

# Grille de k (en Mpc⁻¹)
k_min = 0.001  # Mpc⁻¹
k_max = 0.5    # Mpc⁻¹
n_k = 100

k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
k_grid_SI = k_grid / Mpc_to_m  # Conversion en m⁻¹

# Test avec η = 1.045 (valeur Pantheon+)
eta_test = 1.045

print(f"\nη = {eta_test}")
print(f"k ∈ [{k_min}, {k_max}] Mpc⁻¹ ({n_k} points)")
print(f"\nIntégration z: {z_init} → {z_drag:.0f}")

delta_at_drag = []
z_eval = np.linspace(z_init, z_drag, 500)

print(f"\n{'k (Mpc⁻¹)':>12} {'δ+(z_drag)':>15} {'δ-(z_drag)':>15}")
print("-" * 45)

for i, (k_mpc, k_si) in enumerate(zip(k_grid, k_grid_SI)):
    try:
        sol = solve_ivp(
            lambda z, y: perturbation_system(z, y, k_si, eta_test),
            [z_init, z_drag],
            y0,
            method='RK45',
            dense_output=True,
            max_step=100
        )

        # Valeur à z_drag
        y_drag = sol.sol(z_drag)
        delta_p_drag = abs(y_drag[0])
        delta_m_drag = abs(y_drag[2])

        delta_at_drag.append((k_mpc, delta_p_drag, delta_m_drag))

        if i % 20 == 0:
            print(f"{k_mpc:12.4f} {delta_p_drag:15.6e} {delta_m_drag:15.6e}")

    except Exception as e:
        delta_at_drag.append((k_mpc, np.nan, np.nan))

print("-" * 45)

# ══════════════════════════════════════════════════════════════════
# 9. TROUVER k_peak
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DÉTERMINATION DU PIC")
print("=" * 70)

k_vals = np.array([d[0] for d in delta_at_drag])
delta_p_vals = np.array([d[1] for d in delta_at_drag])

# Filtrer les NaN
valid = ~np.isnan(delta_p_vals)
k_valid = k_vals[valid]
delta_valid = delta_p_vals[valid]

if len(delta_valid) > 0:
    # Trouver le maximum
    idx_max = np.argmax(delta_valid)
    k_peak = k_valid[idx_max]
    delta_peak = delta_valid[idx_max]

    print(f"\nk_peak = {k_peak:.4f} Mpc⁻¹")
    print(f"δ+(k_peak) = {delta_peak:.6e}")

    # Calculer D_M pour ℓ_peak = k_peak × D_M
    def integrand_DM(z):
        return c_of_z(z) / H_of_z(z)

    DM_m, _ = quad(integrand_DM, 0, z_drag, limit=1000)
    DM_Mpc = DM_m / Mpc_to_m

    print(f"\nD_M = {DM_Mpc:.1f} Mpc")

    # ℓ_peak = k_peak × D_M
    ell_peak = k_peak * DM_Mpc

    print(f"\nℓ_peak = k_peak × D_M = {k_peak:.4f} × {DM_Mpc:.1f} = {ell_peak:.1f}")

    # Comparaison avec formule standard
    def c_s_func(z):
        return c_of_z(z) / np.sqrt(3 * (1 + R_baryon(z)))

    rd_m, _ = quad(lambda z: c_s_func(z) / H_of_z(z), z_drag, z_init, limit=2000)
    rd_Mpc = rd_m / Mpc_to_m

    ell_standard = np.pi * DM_Mpc / rd_Mpc

    print(f"\nComparaison:")
    print(f"  ℓ_peak (perturbations) = {ell_peak:.1f}")
    print(f"  ℓ₁ = π D_M/r_d        = {ell_standard:.1f}")
    print(f"  r_d = {rd_Mpc:.1f} Mpc")

    F = ell_peak / ell_standard
    print(f"\nFacteur correctif F = ℓ_peak / (π D_M/r_d) = {F:.3f}")

else:
    print("Erreur: aucune solution valide")
    k_peak = np.nan
    ell_peak = np.nan

# ══════════════════════════════════════════════════════════════════
# 10. SCAN η
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SCAN η ∈ [1.0, 1.3]")
print("=" * 70)

eta_grid = np.linspace(1.0, 1.3, 7)
results_eta = []

for eta in eta_grid:
    delta_at_drag_eta = []

    for k_mpc, k_si in zip(k_grid[::5], k_grid_SI[::5]):  # Sous-échantillonner
        try:
            sol = solve_ivp(
                lambda z, y: perturbation_system(z, y, k_si, eta),
                [z_init, z_drag],
                y0,
                method='RK45',
                max_step=100
            )
            y_drag = sol.y[:, -1]
            delta_at_drag_eta.append((k_mpc, abs(y_drag[0])))
        except:
            pass

    if delta_at_drag_eta:
        k_vals_eta = np.array([d[0] for d in delta_at_drag_eta])
        delta_vals_eta = np.array([d[1] for d in delta_at_drag_eta])
        valid_eta = ~np.isnan(delta_vals_eta) & (delta_vals_eta > 0)

        if np.any(valid_eta):
            idx_max_eta = np.argmax(delta_vals_eta[valid_eta])
            k_peak_eta = k_vals_eta[valid_eta][idx_max_eta]
            ell_peak_eta = k_peak_eta * DM_Mpc
            results_eta.append((eta, k_peak_eta, ell_peak_eta))
            print(f"  η = {eta:.2f}: k_peak = {k_peak_eta:.4f} Mpc⁻¹, ℓ_peak = {ell_peak_eta:.1f}")

# ══════════════════════════════════════════════════════════════════
# 11. FIGURES
# ══════════════════════════════════════════════════════════════════

output_dir = '/mnt/T2/janus-sim/output/cmb_vsl'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Perturbations couplées Janus — η = {eta_test}', fontsize=14, fontweight='bold')

# Panel 1: δ+(k)
ax1 = axes[0, 0]
ax1.loglog(k_valid, delta_valid, 'b-', lw=2)
if not np.isnan(k_peak):
    ax1.axvline(k_peak, color='r', ls='--', lw=2, label=f'k_peak = {k_peak:.4f} Mpc⁻¹')
    ax1.plot(k_peak, delta_peak, 'ro', ms=10)
ax1.set_xlabel('k (Mpc⁻¹)')
ax1.set_ylabel('|δ+(k, z_drag)|')
ax1.set_title('Spectre des perturbations à z_drag')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel 2: Comparaison ℓ
ax2 = axes[0, 1]
if results_eta:
    etas = [r[0] for r in results_eta]
    ells = [r[2] for r in results_eta]
    ax2.plot(etas, ells, 'go-', lw=2, ms=8, label='ℓ_peak (perturbations)')
ax2.axhline(220, color='r', ls='--', lw=2, label='Cible ℓ₁ = 220')
ax2.axhline(ell_standard, color='orange', ls=':', lw=2, label=f'π D_M/r_d = {ell_standard:.0f}')
ax2.set_xlabel('η')
ax2.set_ylabel('ℓ_peak')
ax2.set_title('Position du pic vs η')
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3: Évolution temporelle pour k_peak
ax3 = axes[1, 0]
if not np.isnan(k_peak):
    k_peak_si = k_peak / Mpc_to_m
    sol_evol = solve_ivp(
        lambda z, y: perturbation_system(z, y, k_peak_si, eta_test),
        [z_init, z_drag],
        y0,
        method='RK45',
        dense_output=True,
        max_step=50
    )
    z_plot = np.linspace(z_init, z_drag, 500)
    y_plot = sol_evol.sol(z_plot)
    ax3.semilogy(z_plot, np.abs(y_plot[0]), 'b-', lw=2, label='δ+')
    ax3.semilogy(z_plot, np.abs(y_plot[2]), 'r--', lw=2, label='δ-')
    ax3.axvline(z_drag, color='gray', ls=':', label=f'z_drag = {z_drag:.0f}')
    ax3.set_xlabel('z')
    ax3.set_ylabel('|δ|')
    ax3.set_title(f'Évolution δ(z) pour k = k_peak = {k_peak:.4f} Mpc⁻¹')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.invert_xaxis()

# Panel 4: Résumé
ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
RÉSULTATS — PERTURBATIONS COUPLÉES JANUS

Paramètres:
  H0 = {H0_km_s_Mpc} km/s/Mpc
  η = {eta_test}
  n_high = {n_high}, n_low = {n_low}
  z_drag = {z_drag:.1f}

Résultats (méthode perturbations):
  k_peak = {k_peak:.4f} Mpc⁻¹
  D_M = {DM_Mpc:.1f} Mpc

  ℓ_peak = k_peak × D_M = {ell_peak:.1f}

Comparaison (formule standard):
  r_d = {rd_Mpc:.1f} Mpc
  ℓ₁ = π D_M / r_d = {ell_standard:.1f}

Facteur correctif:
  F = ℓ_peak / ℓ_standard = {F:.3f}

Cible: ℓ₁ = 220
Écart: {100 * abs(ell_peak - 220) / 220:.1f}%
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_perturbations.png', dpi=150, bbox_inches='tight')
print(f"\n✓ {output_dir}/janus_perturbations.png")

# ══════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"""
Méthode: Résolution directe des perturbations couplées Janus
         d²δ±/dt² + 2H dδ±/dt + cs²k²δ± = 4πG(ρ±δ± - ρ∓δ∓)

Résultat:
  ℓ_peak = {ell_peak:.1f}  (via k_peak × D_M)
  ℓ_std  = {ell_standard:.1f}  (via π D_M / r_d)

  Cible  = 220
  Écart  = {100 * abs(ell_peak - 220) / 220:.1f}%

Facteur correctif F = {F:.3f}
  (F = 1 signifierait accord parfait avec formule standard)
""")
print("=" * 70)
