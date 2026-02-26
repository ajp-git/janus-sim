#!/usr/bin/env python3
"""
JANUS — Test "Secteur négatif lisse"
Perturbations découplées : δ- n'intervient pas
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
print("JANUS — TEST SECTEUR NÉGATIF LISSE")
print("Perturbations positives seules, fond Janus")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. PARAMÈTRES FIXES (AUCUN AJUSTEMENT)
# ══════════════════════════════════════════════════════════════════
H0_km_s_Mpc = 76.0       # Fixé
eta = 1.045              # Fixé
Omega_plus = 0.489       # Fixé
z_max = 10000            # Fixé

# Constantes physiques SI
c0_SI = 299792458.0      # m/s
G = 6.67430e-11          # m³/(kg·s²)
kB = 1.380649e-23        # J/K
me = 9.10938e-31         # kg
mp = 1.67262e-27         # kg
h_planck = 6.62607e-34   # J·s
Ei = 13.6 * 1.60218e-19  # J
Mpc_to_m = 3.086e22      # m/Mpc

H0_SI = H0_km_s_Mpc * 1000 / Mpc_to_m  # s⁻¹
h = H0_km_s_Mpc / 100

# Paramètres dérivés
Omega_b = 0.0493
Omega_gamma = 2.47e-5 / h**2
rho_crit_0 = 3 * H0_SI**2 / (8 * np.pi * G)

# Exposants H(z) validés (seuls qui donnent r_d ≈ 147)
n_low = 1.03
n_high = 1.34

print(f"\nPARAMÈTRES FIXES:")
print(f"  H0 = {H0_km_s_Mpc} km/s/Mpc")
print(f"  η = {eta}")
print(f"  Ω+ = {Omega_plus}")
print(f"  n_low = {n_low}, n_high = {n_high}")
print(f"  z_max = {z_max}")

# ══════════════════════════════════════════════════════════════════
# 2. RECOMBINAISON — SAHA STANDARD
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RECOMBINAISON — SAHA")
print("=" * 70)

T0 = 2.725  # K

def T_of_z(z):
    return T0 * (1 + z)

def n_b(z):
    """Densité baryonique en m⁻³"""
    return (rho_crit_0 * Omega_b / mp) * (1 + z)**3

def x_e_saha(z):
    """Fraction d'ionisation via Saha"""
    T = T_of_z(z)
    nb = n_b(z)
    thermal = (2 * np.pi * me * kB * T / h_planck**2)**1.5
    S = thermal * np.exp(-Ei / (kB * T)) / nb
    if S > 1e10: return 1.0
    if S < 1e-10: return max(0, np.sqrt(S))
    return np.clip((-S + np.sqrt(S**2 + 4*S)) / 2, 0, 1)

z_drag = brentq(lambda z: x_e_saha(z) - 0.5, 1000, 2000)
print(f"  z_drag (x_e = 0.5) = {z_drag:.1f}")

# ══════════════════════════════════════════════════════════════════
# 3. FOND COSMOLOGIQUE H(z)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FOND H(z) — MODÈLE VALIDÉ")
print("=" * 70)

def H_of_z(z):
    """H(z) avec exposants validés"""
    if z > z_drag:
        return H0_SI * (1 + z)**n_high
    else:
        return H0_SI * (1 + z)**n_low

def dH_dz(z, eps=0.1):
    """Dérivée numérique de H"""
    return (H_of_z(z + eps) - H_of_z(z - eps)) / (2 * eps)

print(f"  H(0) / H0 = {H_of_z(0) / H0_SI:.4f}")
print(f"  H(z_drag) / H0 = {H_of_z(z_drag) / H0_SI:.4f}")
print(f"  H(z_max) / H0 = {H_of_z(z_max) / H0_SI:.4f}")

# ══════════════════════════════════════════════════════════════════
# 4. VITESSE DU SON c_s(z)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VITESSE DU SON")
print("=" * 70)

# c(z) VSL
def dc_du(u, c):
    z = -u
    return -c[0] * x_e_saha(z) / (2 * (1 + z))

sol_c = solve_ivp(dc_du, [-z_max, 0], [c0_SI], method='RK45', dense_output=True)

def c_of_z(z):
    if z > z_max:
        return c0_SI * np.sqrt((1 + z) / (1 + z_max))
    return float(sol_c.sol(-z)[0])

def R_baryon(z):
    """R = 3ρ_b / 4ρ_γ"""
    return (3 * Omega_b / (4 * Omega_gamma)) / (1 + z)

def c_s(z):
    """Vitesse du son"""
    return c_of_z(z) / np.sqrt(3 * (1 + R_baryon(z)))

print(f"  c(z_drag) / c0 = {c_of_z(z_drag) / c0_SI:.4f}")
print(f"  c_s(z_drag) / c0 = {c_s(z_drag) / c0_SI:.4f}")
print(f"  R(z_drag) = {R_baryon(z_drag):.4f}")

# ══════════════════════════════════════════════════════════════════
# 5. RAYON ACOUSTIQUE r_d
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RAYON ACOUSTIQUE")
print("=" * 70)

def integrand_rd(z):
    return c_s(z) / H_of_z(z)

rd_m, _ = quad(integrand_rd, z_drag, z_max, limit=2000)
rd_Mpc = rd_m / Mpc_to_m

print(f"  r_d = {rd_Mpc:.2f} Mpc")

# ══════════════════════════════════════════════════════════════════
# 6. DISTANCE COMOBILE D_M
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DISTANCE COMOBILE")
print("=" * 70)

def integrand_DM(z):
    return c_of_z(z) / H_of_z(z)

DM_m, _ = quad(integrand_DM, 0, z_drag, limit=1000)
DM_Mpc = DM_m / Mpc_to_m

print(f"  D_M = {DM_Mpc:.2f} Mpc")

# ══════════════════════════════════════════════════════════════════
# 7. PREMIER PIC (FORMULE STANDARD)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PREMIER PIC — FORMULE STANDARD")
print("=" * 70)

ell_standard = np.pi * DM_Mpc / rd_Mpc
print(f"  ℓ₁ = π D_M / r_d = {ell_standard:.1f}")

# ══════════════════════════════════════════════════════════════════
# 8. SOLVEUR PERTURBATIONS — SECTEUR POSITIF SEUL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SOLVEUR PERTURBATIONS — SECTEUR POSITIF SEUL")
print("=" * 70)
print("  Équation: δ̈+ + 2Hδ̇+ + cs²k²δ+ = 4πGρ+δ+")
print("  PAS de couplage avec δ-")

# Densité matière positive
def rho_plus(z):
    """ρ+(z) = Ω+ ρcrit (1+z)³"""
    return rho_crit_0 * Omega_plus * (1 + z)**3

def perturbation_positive_only(z, y, k_SI):
    """
    Équation pour δ+ seul (pas de couplage avec secteur négatif)
    y = [δ+, dδ+/dz]

    Équation: d²δ/dt² + 2H dδ/dt + cs²k²δ = 4πG ρ+ δ
    En z: d²δ/dz² + A dδ/dz + B δ = S δ
    """
    delta, delta_prime = y

    H = H_of_z(z)
    H_prime = dH_dz(z)
    cs = c_s(z)
    rho_p = rho_plus(z)

    z1 = 1 + z
    H2 = H**2

    # Coefficients de l'équation en z
    # A = [H'(1+z) - H] / [H(1+z)]
    A = (H_prime * z1 - H) / (H * z1)

    # Terme de pression: cs²k² / [H²(1+z)²]
    pressure_term = (cs * k_SI)**2 / (H2 * z1**2)

    # Terme gravitationnel: 4πG ρ+ / [H²(1+z)²]
    gravity_term = 4 * np.pi * G * rho_p / (H2 * z1**2)

    # δ'' = -A δ' - (pressure - gravity) δ
    d_delta = delta_prime
    d_delta_prime = -A * delta_prime - (pressure_term - gravity_term) * delta

    return [d_delta, d_delta_prime]

# Conditions initiales
z_init = z_max
delta_init = 1e-5
delta_prime_init = 0.0
y0 = [delta_init, delta_prime_init]

# Grille de k
k_min = 1e-3   # Mpc⁻¹
k_max = 0.5    # Mpc⁻¹
n_k = 200

k_grid_Mpc = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
k_grid_SI = k_grid_Mpc / Mpc_to_m

print(f"\n  k ∈ [{k_min}, {k_max}] Mpc⁻¹ ({n_k} points)")
print(f"  Intégration: z = {z_init} → {z_drag:.0f}")

# Résolution pour chaque k
delta_at_drag = []
print(f"\n  {'k (Mpc⁻¹)':>12} {'|δ+(z_drag)|':>15} {'log10|δ|':>12}")
print("  " + "-" * 42)

for i, (k_Mpc, k_SI) in enumerate(zip(k_grid_Mpc, k_grid_SI)):
    try:
        sol = solve_ivp(
            lambda z, y: perturbation_positive_only(z, y, k_SI),
            [z_init, z_drag],
            y0,
            method='RK45',
            max_step=50,
            rtol=1e-8,
            atol=1e-12
        )

        delta_final = sol.y[0, -1]
        delta_at_drag.append((k_Mpc, delta_final))

        if i % 25 == 0:
            print(f"  {k_Mpc:12.4f} {abs(delta_final):15.6e} {np.log10(abs(delta_final)):12.2f}")

    except Exception as e:
        delta_at_drag.append((k_Mpc, np.nan))

print("  " + "-" * 42)

# ══════════════════════════════════════════════════════════════════
# 9. ANALYSE DU SPECTRE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSE DU SPECTRE")
print("=" * 70)

k_vals = np.array([d[0] for d in delta_at_drag])
delta_vals = np.array([d[1] for d in delta_at_drag])

# Spectre de puissance
power_spectrum = np.abs(delta_vals)**2

# Filtrer NaN
valid = ~np.isnan(delta_vals)
k_valid = k_vals[valid]
delta_valid = delta_vals[valid]
power_valid = power_spectrum[valid]

# Détecter les oscillations
# Chercher les maxima locaux
from scipy.signal import find_peaks

# Normaliser pour la détection de pics
if len(power_valid) > 10:
    power_norm = power_valid / np.max(power_valid)
    peaks, properties = find_peaks(power_norm, height=0.1, distance=5)

    print(f"\n  Pics détectés: {len(peaks)}")

    if len(peaks) > 0:
        print(f"\n  {'Pic':>6} {'k (Mpc⁻¹)':>12} {'|δ|²':>15} {'ℓ = k×D_M':>12}")
        print("  " + "-" * 48)

        for i, peak_idx in enumerate(peaks[:5]):  # Max 5 pics
            k_peak = k_valid[peak_idx]
            power_peak = power_valid[peak_idx]
            ell_peak = k_peak * DM_Mpc
            print(f"  {i+1:6d} {k_peak:12.4f} {power_peak:15.6e} {ell_peak:12.1f}")

        # Premier pic
        k_first_peak = k_valid[peaks[0]]
        ell_first_peak = k_first_peak * DM_Mpc

        print(f"\n  PREMIER PIC:")
        print(f"    k_peak = {k_first_peak:.4f} Mpc⁻¹")
        print(f"    ℓ_peak = k × D_M = {ell_first_peak:.1f}")

        oscillations_detected = True
    else:
        print("  Aucun pic détecté — spectre monotone")
        k_first_peak = k_valid[np.argmax(power_valid)]
        ell_first_peak = k_first_peak * DM_Mpc
        oscillations_detected = False
else:
    print("  Données insuffisantes")
    oscillations_detected = False
    k_first_peak = np.nan
    ell_first_peak = np.nan

# ══════════════════════════════════════════════════════════════════
# 10. CRITÈRE DE SUCCÈS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CRITÈRE DE SUCCÈS")
print("=" * 70)

target_ell = 220
tolerance = 0.05  # 5%

if oscillations_detected:
    error_ell = abs(ell_first_peak - target_ell) / target_ell
    success_ell = error_ell < tolerance

    print(f"\n  Oscillations: ✓ DÉTECTÉES")
    print(f"  ℓ_peak = {ell_first_peak:.1f} (cible: {target_ell} ± 5%)")
    print(f"  Erreur: {100*error_ell:.1f}%")
    print(f"  Critère ℓ: {'✓ PASS' if success_ell else '✗ FAIL'}")

    if success_ell:
        print(f"\n  ➜ SUCCÈS — Janus compatible si secteur négatif lisse")
    else:
        print(f"\n  ➜ ÉCHEC PARTIEL — Oscillations présentes mais ℓ incorrect")
else:
    print(f"\n  Oscillations: ✗ NON DÉTECTÉES")
    print(f"  Spectre plat ou monotone")
    print(f"\n  ➜ ÉCHEC — Incompatibilité dynamique")
    success_ell = False

# ══════════════════════════════════════════════════════════════════
# 11. FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GÉNÉRATION DES FIGURES")
print("=" * 70)

output_dir = '/mnt/T2/janus-sim/output/cmb_vsl'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Test Secteur Négatif Lisse — Perturbations δ+ seules',
             fontsize=14, fontweight='bold')

# Panel 1: Spectre δ(k)
ax1 = axes[0, 0]
ax1.loglog(k_valid, np.abs(delta_valid), 'b-', lw=1.5, alpha=0.8)
if oscillations_detected and len(peaks) > 0:
    ax1.loglog(k_valid[peaks], np.abs(delta_valid[peaks]), 'ro', ms=8, label='Pics')
ax1.set_xlabel('k (Mpc⁻¹)')
ax1.set_ylabel('|δ+(k, z_drag)|')
ax1.set_title('Amplitude des perturbations')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel 2: Spectre de puissance
ax2 = axes[0, 1]
ax2.loglog(k_valid, power_valid, 'g-', lw=1.5)
if oscillations_detected and len(peaks) > 0:
    ax2.loglog(k_valid[peaks], power_valid[peaks], 'ro', ms=8, label='Pics')
    ax2.axvline(k_first_peak, color='r', ls='--', alpha=0.5,
                label=f'k₁ = {k_first_peak:.4f}')
ax2.set_xlabel('k (Mpc⁻¹)')
ax2.set_ylabel('|δ+(k)|²')
ax2.set_title('Spectre de puissance')
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3: δ(k) en échelle linéaire pour voir oscillations
ax3 = axes[1, 0]
ax3.semilogx(k_valid, delta_valid, 'b-', lw=1)
ax3.axhline(0, color='gray', ls=':')
ax3.set_xlabel('k (Mpc⁻¹)')
ax3.set_ylabel('δ+(k, z_drag)')
ax3.set_title('Perturbations (échelle linéaire) — oscillations?')
ax3.grid(alpha=0.3)

# Panel 4: Résumé
ax4 = axes[1, 1]
ax4.axis('off')

status = "SUCCÈS" if (oscillations_detected and success_ell) else "ÉCHEC"

summary = f"""
RÉSUMÉ — TEST SECTEUR NÉGATIF LISSE

Hypothèse testée:
  Secteur négatif contribue au fond (η={eta})
  mais PAS aux perturbations acoustiques

Paramètres fixes:
  H0 = {H0_km_s_Mpc} km/s/Mpc
  η = {eta}
  Ω+ = {Omega_plus}
  n_high = {n_high}, n_low = {n_low}

Résultats géométriques:
  z_drag = {z_drag:.1f}
  r_d = {rd_Mpc:.2f} Mpc
  D_M = {DM_Mpc:.2f} Mpc
  ℓ_standard = π D_M/r_d = {ell_standard:.1f}

Résultats perturbations:
  Oscillations: {'OUI' if oscillations_detected else 'NON'}
  ℓ_peak = {f'{ell_first_peak:.1f}' if not np.isnan(ell_first_peak) else 'N/A'}
  Cible: 220 ± 5%

VERDICT: {status}
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round',
                   facecolor='lightgreen' if status == "SUCCÈS" else 'lightyellow',
                   alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/janus_smooth_negative.png', dpi=150, bbox_inches='tight')
print(f"  ✓ {output_dir}/janus_smooth_negative.png")

# ══════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"""
Test: Secteur négatif lisse (pas de δ-)
Équation: δ̈+ + 2Hδ̇+ + cs²k²δ+ = 4πGρ+δ+

Oscillations acoustiques: {'DÉTECTÉES' if oscillations_detected else 'ABSENTES'}
Premier pic ℓ_peak: {ell_first_peak:.1f if not np.isnan(ell_first_peak) else 'N/A'}
Cible: 220

Formule standard ℓ = π D_M/r_d: {ell_standard:.1f}

VERDICT: {status}
""")

if oscillations_detected and success_ell:
    print("INTERPRÉTATION:")
    print("  Janus est compatible avec le CMB si le secteur négatif")
    print("  reste lisse (non perturbé) à l'époque primordiale.")
elif oscillations_detected:
    print("INTERPRÉTATION:")
    print("  Les oscillations sont présentes mais la position du pic")
    print("  ne correspond pas à ℓ = 220.")
else:
    print("INTERPRÉTATION:")
    print("  Incompatibilité dynamique plus profonde.")
    print("  Même sans couplage ±, pas d'oscillations acoustiques.")

print("=" * 70)
