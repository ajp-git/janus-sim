#!/usr/bin/env python3
"""
Fit Janus exact formula (D'Agostini & Petit 2018 eq.5) to Pantheon+ SNIa data.

Formula:
    mu = 5 * log10(arg) + cst

where:
    arg = z + z^2*(1-q0) / (1 + q0*z + sqrt(1 + 2*q0*z))

Free parameters: q0, cst
Target: q0 = -0.087 (from 2018 paper with 740 SNIa)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import linalg

print("=" * 70)
print("JANUS EXACT FORMULA FIT — D'Agostini & Petit 2018 eq.(5)")
print("=" * 70)

# Load Pantheon+ data
print("\nLoading Pantheon+ data...")
data = np.genfromtxt('data/Pantheon+SH0ES.dat', names=True, dtype=None, encoding='utf-8')

# Extract columns - use zHD and m_b_corr (corrected apparent magnitude)
z_all = data['zHD']
m_b_all = data['m_b_corr']
m_b_err_all = data['m_b_corr_err_DIAG']

# Filter: use only SNe with z > 0.01 to avoid local flow issues
# and z < 2.3 (Pantheon+ range)
mask = (z_all > 0.01) & (z_all < 2.3) & np.isfinite(m_b_all)
z = z_all[mask]
m_b = m_b_all[mask]
m_b_err = m_b_err_all[mask]

n_sne = len(z)
print(f"  Total SNe: {len(z_all)}")
print(f"  After z > 0.01 cut: {n_sne}")
print(f"  Redshift range: z = [{z.min():.4f}, {z.max():.4f}]")

# Load covariance matrix
print("\nLoading covariance matrix...")
cov_file = 'data/Pantheon+SH0ES_STAT+SYS.cov'
with open(cov_file, 'r') as f:
    n_cov = int(f.readline().strip())
    cov_flat = np.loadtxt(f)

cov_full = cov_flat.reshape(n_cov, n_cov)
print(f"  Covariance matrix: {n_cov} x {n_cov}")

# We need to extract the sub-covariance for our filtered SNe
# The data ordering should be preserved
# For simplicity, use diagonal errors for initial fit, then full cov
indices = np.where(mask)[0]
cov = cov_full[np.ix_(indices, indices)]
print(f"  Using {n_sne} x {n_sne} sub-covariance")

# Try to invert; if singular, regularize
try:
    cov_inv = linalg.inv(cov)
    print("  Covariance inverted successfully")
except linalg.LinAlgError:
    print("  Regularizing covariance matrix...")
    cov_reg = cov + 1e-6 * np.eye(n_sne)
    cov_inv = linalg.inv(cov_reg)


def mu_janus_exact(z, q0, cst):
    """
    Exact Janus formula from D'Agostini & Petit 2018 eq.(5)

    mu = 5 * log10(arg) + cst

    where:
        arg = z + z^2*(1-q0) / (1 + q0*z + sqrt(1 + 2*q0*z))
    """
    # Ensure numerical stability for sqrt
    inner = 1.0 + 2.0 * q0 * z
    # If inner < 0, formula not valid (can happen for extreme q0 and high z)
    inner = np.maximum(inner, 1e-10)

    denominator = 1.0 + q0 * z + np.sqrt(inner)
    arg = z + z**2 * (1.0 - q0) / denominator

    # Avoid log of negative/zero
    arg = np.maximum(arg, 1e-10)

    return 5.0 * np.log10(arg) + cst


def chi2_diagonal(params):
    """Chi-squared with diagonal errors only (fast)"""
    q0, cst = params
    mu_model = mu_janus_exact(z, q0, cst)
    residuals = m_b - mu_model
    return np.sum((residuals / m_b_err)**2)


def chi2_full(params):
    """Chi-squared with full covariance matrix"""
    q0, cst = params
    mu_model = mu_janus_exact(z, q0, cst)
    residuals = m_b - mu_model
    return residuals @ cov_inv @ residuals


# Initial guess: q0 ~ -0.1, cst ~ 43 (typical distance modulus offset)
# First fit with diagonal errors for fast convergence
print("\n" + "=" * 70)
print("STEP 1: Diagonal-only fit (fast)")
print("=" * 70)

x0 = [-0.1, 43.0]
result_diag = minimize(chi2_diagonal, x0, method='Nelder-Mead')
q0_diag, cst_diag = result_diag.x
chi2_diag = result_diag.fun
dof = n_sne - 2  # 2 free parameters

print(f"\n  q0  = {q0_diag:.6f}")
print(f"  cst = {cst_diag:.4f}")
print(f"  chi2 = {chi2_diag:.2f}")
print(f"  chi2/dof = {chi2_diag/dof:.4f}  (dof = {dof})")

# Now fit with full covariance
print("\n" + "=" * 70)
print("STEP 2: Full covariance fit")
print("=" * 70)

# Use diagonal result as starting point
x0_full = [q0_diag, cst_diag]
result_full = minimize(chi2_full, x0_full, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
q0_opt, cst_opt = result_full.x
chi2_opt = result_full.fun

print(f"\n  q0  = {q0_opt:.6f}")
print(f"  cst = {cst_opt:.4f}")
print(f"  chi2 = {chi2_opt:.2f}")
print(f"  chi2/dof = {chi2_opt/dof:.4f}  (dof = {dof})")

# Error estimation via Hessian (approximate)
print("\n" + "=" * 70)
print("STEP 3: Error estimation")
print("=" * 70)

# Numerical Hessian at minimum
def hessian_numerical(f, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
    return H

hess = hessian_numerical(chi2_full, [q0_opt, cst_opt])
try:
    cov_params = linalg.inv(hess / 2)  # Factor of 2 for standard chi2 convention
    sigma_q0 = np.sqrt(cov_params[0, 0])
    sigma_cst = np.sqrt(cov_params[1, 1])
    print(f"\n  sigma(q0)  = {sigma_q0:.6f}")
    print(f"  sigma(cst) = {sigma_cst:.4f}")
except:
    sigma_q0 = sigma_cst = np.nan
    print("\n  (Hessian singular, no error estimate)")

# Comparison with 2018 paper
print("\n" + "=" * 70)
print("COMPARISON WITH D'AGOSTINI & PETIT 2018")
print("=" * 70)

q0_2018 = -0.087  # From paper
print(f"\n  Paper value (740 SNIa):     q0 = {q0_2018:.3f}")
print(f"  Our fit (Pantheon+ {n_sne} SNIa): q0 = {q0_opt:.4f} +/- {sigma_q0:.4f}")
print(f"\n  Difference: delta_q0 = {q0_opt - q0_2018:.4f}")

if not np.isnan(sigma_q0):
    n_sigma = abs(q0_opt - q0_2018) / sigma_q0
    print(f"  Tension: {n_sigma:.1f} sigma")

# Derive eta from q0
# From Janus: q0 = (1 - eta) / (1 + eta)  =>  eta = (1 - q0) / (1 + q0)
eta_opt = (1.0 - q0_opt) / (1.0 + q0_opt)
eta_2018 = (1.0 - q0_2018) / (1.0 + q0_2018)

print(f"\n  Derived eta (from q0):")
print(f"    eta_2018     = {eta_2018:.4f}")
print(f"    eta_Pantheon = {eta_opt:.4f}")

# Plot
print("\n" + "=" * 70)
print("Generating plots...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Hubble diagram
ax = axes[0, 0]
ax.errorbar(z, m_b, yerr=m_b_err, fmt='.', ms=2, alpha=0.3, label='Pantheon+ data')
z_model = np.linspace(0.01, z.max(), 200)
mu_model = mu_janus_exact(z_model, q0_opt, cst_opt)
ax.plot(z_model, mu_model, 'r-', lw=2, label=f'Janus exact (q0={q0_opt:.4f})')
mu_2018 = mu_janus_exact(z_model, q0_2018, cst_opt)
ax.plot(z_model, mu_2018, 'g--', lw=2, label=f'2018 paper (q0={q0_2018:.3f})')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Distance modulus m_b', fontsize=12)
ax.set_title('Hubble Diagram — Janus Exact Formula', fontsize=12)
ax.legend()
ax.set_xscale('log')

# 2. Residuals
ax = axes[0, 1]
mu_fit = mu_janus_exact(z, q0_opt, cst_opt)
residuals = m_b - mu_fit
ax.scatter(z, residuals, s=2, alpha=0.3)
ax.axhline(0, color='k', linestyle='-')
ax.axhline(np.std(residuals), color='r', linestyle='--', alpha=0.5, label=f'+/-1 sigma = {np.std(residuals):.3f}')
ax.axhline(-np.std(residuals), color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Residual (m_b - model)', fontsize=12)
ax.set_title('Residuals', fontsize=12)
ax.set_xscale('log')
ax.legend()

# 3. Chi2 scan over q0
ax = axes[1, 0]
q0_range = np.linspace(-0.2, 0.0, 100)
chi2_scan = []
for q in q0_range:
    # For each q0, optimize cst only
    res = minimize(lambda c: chi2_full([q, c[0]]), [cst_opt], method='Nelder-Mead')
    chi2_scan.append(res.fun)
chi2_scan = np.array(chi2_scan)
ax.plot(q0_range, chi2_scan, 'b-', lw=2)
ax.axvline(q0_opt, color='r', linestyle='-', label=f'Best fit: q0 = {q0_opt:.4f}')
ax.axvline(q0_2018, color='g', linestyle='--', label=f'2018: q0 = {q0_2018:.3f}')
ax.axhline(chi2_opt + 1, color='gray', linestyle=':', alpha=0.5, label='1-sigma')
ax.set_xlabel('Deceleration parameter q0', fontsize=12)
ax.set_ylabel('chi2', fontsize=12)
ax.set_title('Chi2 Profile vs q0', fontsize=12)
ax.legend()

# 4. Summary text
ax = axes[1, 1]
ax.axis('off')
summary = f"""
{'='*56}
    JANUS EXACT FIT — D'Agostini & Petit 2018 eq.(5)
{'='*56}

DATA
----
  Dataset:      Pantheon+ SH0ES
  N supernovae: {n_sne}
  Redshift:     {z.min():.4f} < z < {z.max():.4f}
  Covariance:   Full STAT+SYS

RESULTS
-------
  q0  = {q0_opt:.5f} +/- {sigma_q0:.5f}
  cst = {cst_opt:.4f} +/- {sigma_cst:.4f}

  chi2     = {chi2_opt:.2f}
  chi2/dof = {chi2_opt/dof:.4f}

COMPARISON
----------
  D'Agostini & Petit 2018 (740 SNIa):
    q0 = -0.087

  This fit (Pantheon+ {n_sne} SNIa):
    q0 = {q0_opt:.4f}

  Difference: {abs(q0_opt - q0_2018):.4f} ({abs(q0_opt - q0_2018)/sigma_q0:.1f} sigma)

DERIVED PARAMETERS
------------------
  eta = (1-q0)/(1+q0) = {eta_opt:.4f}

  (Paper 2018: eta = {eta_2018:.4f})

{'='*56}
"""
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('output/janus_exact_fit.png', dpi=150)
print("\n  Saved output/janus_exact_fit.png")

# Save results to JSON
import json
results = {
    "formula": "mu = 5*log10(z + z^2*(1-q0)/(1 + q0*z + sqrt(1 + 2*q0*z))) + cst",
    "reference": "D'Agostini & Petit 2018 eq.(5)",
    "n_sne": int(n_sne),
    "z_min": float(z.min()),
    "z_max": float(z.max()),
    "q0": float(q0_opt),
    "q0_err": float(sigma_q0) if not np.isnan(sigma_q0) else None,
    "cst": float(cst_opt),
    "cst_err": float(sigma_cst) if not np.isnan(sigma_cst) else None,
    "chi2": float(chi2_opt),
    "dof": int(dof),
    "chi2_dof": float(chi2_opt / dof),
    "q0_2018": q0_2018,
    "eta_derived": float(eta_opt),
    "eta_2018": eta_2018
}

with open('output/janus_exact_fit.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved output/janus_exact_fit.json")

# Final verdict
print("\n" + "=" * 70)
if chi2_opt / dof < 1.5:
    print("VERDICT: GOOD FIT — chi2/dof < 1.5")
else:
    print("VERDICT: POOR FIT — chi2/dof >= 1.5")
print("=" * 70)
print()
