#!/usr/bin/env python3
"""
JANUS MODEL WITH M_B OFFSET

The Pantheon+ data uses SH0ES calibration (H0 = 73.04 km/s/Mpc).
If we use H0 = 70 km/s/Mpc (Janus prediction), we need to fit M_B.

The distance modulus is:
μ = m_B - M_B = 5 log₁₀(d_L/10pc)

where d_L depends on H0. If we use different H0, M_B absorbs the difference.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
C = 2.997924580e8  # m/s
MPC_M = 3.0856775815e22
PC_M = 3.0856775815e16

print("=" * 70)
print("JANUS MODEL WITH FREE M_B OFFSET")
print("=" * 70)

def e_z_lcdm(z, omega_m):
    """LCDM: E(z)² = Ωm(1+z)³ + ΩΛ"""
    return np.sqrt(omega_m * (1 + z)**3 + (1 - omega_m))

def comoving_distance(z, omega_m):
    """χ(z) in units of c/H0"""
    n = 500
    z_arr = np.linspace(0, z, n + 1)[1:]
    dz = z / n
    return np.sum(dz / e_z_lcdm(z_arr, omega_m))

def distance_modulus_unit(z, omega_m):
    """Distance modulus in units where d_L = χ*(1+z)*c/H0"""
    chi = comoving_distance(z, omega_m)
    # d_L / (c/H0) in dimensionless units
    d_L_unit = (1 + z) * chi
    # μ = 5 log₁₀(d_L / 10pc)
    # d_L = d_L_unit * c/H0
    # For H0 = 100h km/s/Mpc, c/H0 = 2997.9 Mpc
    # μ = 5 log₁₀(d_L_unit * 2997.9 Mpc / 10pc)
    # μ = 5 log₁₀(d_L_unit) + 5 log₁₀(2997.9e6) = 5 log₁₀(d_L_unit) + 42.38
    # The constant depends on H0, absorbed in M_B offset
    return 5 * np.log10(d_L_unit) + 43.158  # For h=0.70

def load_pantheon(filename):
    z, mu, mu_err = [], [], []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            fields = line.split()
            if len(fields) < 12:
                continue
            z_val = float(fields[2])
            mu_val = float(fields[10])
            err_val = float(fields[11])
            if z_val > 0.01 and err_val > 0:
                z.append(z_val)
                mu.append(mu_val)
                mu_err.append(err_val)
    return np.array(z), np.array(mu), np.array(mu_err)

print("\nLoading Pantheon+ data...")
z_data, mu_data, mu_err = load_pantheon('data/Pantheon+SH0ES.dat')
n_sne = len(z_data)
print(f"  N = {n_sne} supernovae")

# =============================================================================
# FIT WITH 2 PARAMETERS: Ω_m and ΔM_B
# =============================================================================

def chi2_2param(params, z_data, mu_data, mu_err):
    omega_m, delta_mb = params
    if omega_m < 0.01 or omega_m > 0.99:
        return 1e10
    chi2_sum = 0.0
    for z, mu_obs, sigma in zip(z_data, mu_data, mu_err):
        mu_model = distance_modulus_unit(z, omega_m) + delta_mb
        chi2_sum += ((mu_obs - mu_model) / sigma)**2
    return chi2_sum

print("\n" + "=" * 70)
print("FITTING JANUS (= LCDM) with Ω_m and ΔM_B free")
print("=" * 70)

# Initial guess: Omega_m = 0.3, delta_mb = 0 (our baseline agrees with data)
x0 = [0.3, 0.0]
result = minimize(chi2_2param, x0, args=(z_data, mu_data, mu_err),
                  method='Nelder-Mead')

omega_m_best, delta_mb_best = result.x
chi2_best = result.fun

print(f"\n  Best Ω_m = {omega_m_best:.4f}")
print(f"  Best ΔM_B = {delta_mb_best:.4f} mag")
print(f"  χ² = {chi2_best:.1f}")
print(f"  χ²/dof = {chi2_best/(n_sne-2):.4f} (dof = {n_sne-2})")

# Implied eta
eta_best = (1 - omega_m_best) / omega_m_best
print(f"\n  → η = {eta_best:.4f}")

# H0 implication
# ΔM_B = 5 log₁₀(H0_ours / H0_sh0es)
# H0_sh0es = 73.04 km/s/Mpc (Pantheon+ SH0ES calibration)
h0_ratio = 10**(delta_mb_best / 5)
h0_implied = 70.0 * h0_ratio
print(f"\n  If our H0 = 70 km/s/Mpc:")
print(f"    ΔM_B implies H0 ratio = {h0_ratio:.4f}")
print(f"    Data was calibrated with H0 ≈ {h0_implied:.1f} km/s/Mpc")

# =============================================================================
# RESIDUALS
# =============================================================================

residuals = []
for z, mu_obs in zip(z_data, mu_data):
    mu_model = distance_modulus_unit(z, omega_m_best) + delta_mb_best
    residuals.append(mu_obs - mu_model)
residuals = np.array(residuals)

# Bin residuals
z_bins = np.linspace(0, 2.3, 12)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

binned = []
binned_err = []
for i in range(len(z_bins) - 1):
    mask = (z_data >= z_bins[i]) & (z_data < z_bins[i+1])
    if np.sum(mask) > 5:
        binned.append(np.mean(residuals[mask]))
        binned_err.append(np.std(residuals[mask]) / np.sqrt(np.sum(mask)))
    else:
        binned.append(np.nan)
        binned_err.append(np.nan)

binned = np.array(binned)
binned_err = np.array(binned_err)

valid = ~np.isnan(binned)
coef = np.polyfit(z_centers[valid], binned[valid], 1)

print(f"\n  Residual drift: {coef[0]:.4f}*z + {coef[1]:.4f}")
print(f"  Residual RMS: {np.std(residuals):.4f} mag")

# =============================================================================
# COMPARE: FIX Ω_m = 0.3 (STANDARD VALUE)
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: FIX Ω_m = 0.3, fit only ΔM_B")
print("=" * 70)

def chi2_1param(delta_mb, omega_m, z_data, mu_data, mu_err):
    chi2_sum = 0.0
    for z, mu_obs, sigma in zip(z_data, mu_data, mu_err):
        mu_model = distance_modulus_unit(z, omega_m) + delta_mb
        chi2_sum += ((mu_obs - mu_model) / sigma)**2
    return chi2_sum

from scipy.optimize import minimize_scalar

result_03 = minimize_scalar(lambda dm: chi2_1param(dm, 0.3, z_data, mu_data, mu_err),
                            bounds=(-1, 1), method='bounded')
delta_mb_03 = result_03.x
chi2_03 = result_03.fun

print(f"\n  Ω_m = 0.3 (fixed)")
print(f"  Best ΔM_B = {delta_mb_03:.4f} mag")
print(f"  χ²/dof = {chi2_03/(n_sne-1):.4f}")
print(f"  η = {0.7/0.3:.4f}")

# Residuals for Om=0.3
residuals_03 = []
for z, mu_obs in zip(z_data, mu_data):
    mu_model = distance_modulus_unit(z, 0.3) + delta_mb_03
    residuals_03.append(mu_obs - mu_model)
residuals_03 = np.array(residuals_03)

binned_03 = []
for i in range(len(z_bins) - 1):
    mask = (z_data >= z_bins[i]) & (z_data < z_bins[i+1])
    if np.sum(mask) > 5:
        binned_03.append(np.mean(residuals_03[mask]))
    else:
        binned_03.append(np.nan)
binned_03 = np.array(binned_03)

coef_03 = np.polyfit(z_centers[valid], binned_03[valid], 1)
print(f"  Residual drift: {coef_03[0]:.4f}*z")
print(f"  Residual RMS: {np.std(residuals_03):.4f} mag")

# =============================================================================
# FIX Ω_m at D'Agostini (2018) best fit value
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: Ω_m = 0.287 (D'Agostini 2018)")
print("=" * 70)

result_da = minimize_scalar(lambda dm: chi2_1param(dm, 0.287, z_data, mu_data, mu_err),
                            bounds=(-1, 1), method='bounded')
delta_mb_da = result_da.x
chi2_da = result_da.fun

print(f"\n  Ω_m = 0.287 (D'Agostini 2018)")
print(f"  Best ΔM_B = {delta_mb_da:.4f} mag")
print(f"  χ²/dof = {chi2_da/(n_sne-1):.4f}")
print(f"  η = {0.713/0.287:.4f}")

residuals_da = []
for z, mu_obs in zip(z_data, mu_data):
    mu_model = distance_modulus_unit(z, 0.287) + delta_mb_da
    residuals_da.append(mu_obs - mu_model)
residuals_da = np.array(residuals_da)

binned_da = []
for i in range(len(z_bins) - 1):
    mask = (z_data >= z_bins[i]) & (z_data < z_bins[i+1])
    if np.sum(mask) > 5:
        binned_da.append(np.mean(residuals_da[mask]))
    else:
        binned_da.append(np.nan)
binned_da = np.array(binned_da)

coef_da = np.polyfit(z_centers[valid], binned_da[valid], 1)
print(f"  Residual drift: {coef_da[0]:.4f}*z")
print(f"  Residual RMS: {np.std(residuals_da):.4f} mag")

# =============================================================================
# PLOT
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Chi-squared contours
ax = axes[0, 0]
om_range = np.linspace(0.15, 0.55, 50)
dm_range = np.linspace(-0.3, 0.3, 50)
chi2_grid = np.zeros((len(dm_range), len(om_range)))

for i, dm in enumerate(dm_range):
    for j, om in enumerate(om_range):
        chi2_grid[i, j] = chi2_2param([om, dm], z_data, mu_data, mu_err)

chi2_min = np.min(chi2_grid)
levels = chi2_min + np.array([2.30, 6.17, 11.8])  # 1σ, 2σ, 3σ for 2 params

cs = ax.contour(om_range, dm_range, chi2_grid, levels=levels, colors=['blue', 'green', 'red'])
ax.clabel(cs, fmt={2.30+chi2_min: '1σ', 6.17+chi2_min: '2σ', 11.8+chi2_min: '3σ'})
ax.scatter([omega_m_best], [delta_mb_best], color='black', s=100, marker='*',
           label=f'Best: Ωm={omega_m_best:.3f}, ΔM={delta_mb_best:.3f}')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0.3, color='gray', linestyle=':', alpha=0.5, label='Ωm=0.3')
ax.axvline(0.287, color='orange', linestyle=':', alpha=0.5, label='Ωm=0.287 (D\'A 2018)')

ax.set_xlabel(r'$\Omega_m$', fontsize=12)
ax.set_ylabel(r'$\Delta M_B$ (mag)', fontsize=12)
ax.set_title('χ² Contours (1σ, 2σ, 3σ)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Hubble diagram
ax = axes[0, 1]
z_theory = np.linspace(0.01, 2.3, 200)

mu_best = [distance_modulus_unit(z, omega_m_best) + delta_mb_best for z in z_theory]
mu_03 = [distance_modulus_unit(z, 0.3) + delta_mb_03 for z in z_theory]
mu_da = [distance_modulus_unit(z, 0.287) + delta_mb_da for z in z_theory]

ax.scatter(z_data[::5], mu_data[::5], s=5, alpha=0.3, color='gray', label='Pantheon+')
ax.plot(z_theory, mu_best, 'b-', linewidth=2, label=f'Best fit (Ωm={omega_m_best:.2f})')
ax.plot(z_theory, mu_03, 'g--', linewidth=2, alpha=0.7, label='Ωm=0.30')
ax.plot(z_theory, mu_da, 'r:', linewidth=2, alpha=0.7, label='Ωm=0.287')

ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Distance Modulus μ', fontsize=12)
ax.set_title('Hubble Diagram', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Residuals comparison
ax = axes[1, 0]
ax.errorbar(z_centers, binned, yerr=binned_err, fmt='bo-', markersize=8, capsize=3,
            linewidth=2, label=f'Best fit (drift={coef[0]:.3f}*z)')
ax.errorbar(z_centers + 0.02, binned_03, fmt='gs--', markersize=6, alpha=0.7,
            label=f'Ωm=0.30 (drift={coef_03[0]:.3f}*z)')
ax.errorbar(z_centers + 0.04, binned_da, fmt='r^:', markersize=6, alpha=0.7,
            label=f'Ωm=0.287 (drift={coef_da[0]:.3f}*z)')
ax.axhline(0, color='k', linestyle='-')

ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel(r'$\mu_{obs} - \mu_{model}$', fontsize=12)
ax.set_title('Binned Residuals', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 0.3)

# 4. Summary
ax = axes[1, 1]
ax.axis('off')

summary = f"""
═══════════════════════════════════════════════════════
        JANUS MODEL WITH FREE M_B OFFSET
═══════════════════════════════════════════════════════

DATA: Pantheon+ (N = {n_sne})

FREE FIT (Ω_m, ΔM_B)
────────────────────
  Ω_m  = {omega_m_best:.4f}
  ΔM_B = {delta_mb_best:.4f} mag
  η    = {eta_best:.4f}
  χ²/dof = {chi2_best/(n_sne-2):.4f}
  RMS  = {np.std(residuals):.4f} mag
  Drift = {coef[0]:.4f}*z

FIXED Ω_m = 0.30
────────────────
  ΔM_B = {delta_mb_03:.4f} mag
  χ²/dof = {chi2_03/(n_sne-1):.4f}
  RMS  = {np.std(residuals_03):.4f} mag
  Drift = {coef_03[0]:.4f}*z

FIXED Ω_m = 0.287 (D'Agostini 2018)
───────────────────────────────────
  ΔM_B = {delta_mb_da:.4f} mag
  η    = 2.48
  χ²/dof = {chi2_da/(n_sne-1):.4f}
  RMS  = {np.std(residuals_da):.4f} mag
  Drift = {coef_da[0]:.4f}*z

CONCLUSIONS
───────────
• All fits give χ²/dof ~ 0.5 (errors overestimated)
• Residual drift ~0.2*z persists in ALL cases
• D'Agostini Ω_m=0.287 gives LARGER drift than best fit
• Best fit Ω_m ~ 0.33 minimizes drift
• The drift is NOT from Janus vs ΛCDM difference
  (they are equivalent at z < 2)
"""
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('output/janus_with_mb.png', dpi=150)
print("  Saved output/janus_with_mb.png")

# Save data
import json
results = {
    'model': 'Janus/LCDM with free M_B',
    'n_sne': int(n_sne),
    'free_fit': {
        'omega_m': float(omega_m_best),
        'delta_mb': float(delta_mb_best),
        'eta': float(eta_best),
        'chi2_dof': float(chi2_best/(n_sne-2)),
        'rms': float(np.std(residuals)),
        'drift': float(coef[0])
    },
    'om_030': {
        'omega_m': 0.30,
        'delta_mb': float(delta_mb_03),
        'chi2_dof': float(chi2_03/(n_sne-1)),
        'drift': float(coef_03[0])
    },
    'om_287': {
        'omega_m': 0.287,
        'delta_mb': float(delta_mb_da),
        'chi2_dof': float(chi2_da/(n_sne-1)),
        'drift': float(coef_da[0])
    }
}

with open('output/janus_mb_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved output/janus_mb_results.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
