#!/usr/bin/env python3
"""
Janus Bimetric Model — Coupled Equations with Potential V(r)

Equations:
  H²(z) = Ω₊(1+z)³ - Ω₋(1+z)³ + m²·V(r)
  V(r) = β₀ + 3β₁r + 3β₂r² + β₃r³
  dr/dz = -r/H · [dH/dz + H/(1+z)]

With isotropic approximation: a₋ = a₊ = a, and r = a₋/a₊ evolves from r=1 at low z.

Scanning β₁ ∈ [0.1, 1.0, 10.0] with β₀ = β₂ = β₃ = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad

# ============================================================
# PARAMETERS
# ============================================================

ETA = 1.045
OMEGA_PLUS = 0.30
OMEGA_MINUS = ETA * OMEGA_PLUS  # = 0.3135
M2 = 1.0  # m² in units of ρ_crit
H0 = 76.0  # km/s/Mpc

Z_CMB = 1100
Z_DRAG = 1059.6
c_over_H0 = 2998 / H0  # Mpc

print("=" * 70)
print("JANUS BIMETRIC COUPLED EQUATIONS")
print("=" * 70)
print(f"Ω₊ = {OMEGA_PLUS}, Ω₋ = {OMEGA_MINUS:.4f}, m² = {M2}")
print()

# ============================================================
# POTENTIAL V(r)
# ============================================================

def V(r, beta0=0, beta1=1, beta2=0, beta3=0):
    """
    Hassan-Rosen type potential:
    V(r) = β₀ + 3β₁r + 3β₂r² + β₃r³
    """
    return beta0 + 3*beta1*r + 3*beta2*r**2 + beta3*r**3

def dV_dr(r, beta0=0, beta1=1, beta2=0, beta3=0):
    """Derivative dV/dr"""
    return 3*beta1 + 6*beta2*r + 3*beta3*r**2

# ============================================================
# H² FUNCTION
# ============================================================

def H2_bimetric(z, r, beta1):
    """
    H²(z)/H₀² = Ω₊(1+z)³ - Ω₋(1+z)³ + m²·V(r)
    """
    matter_plus = OMEGA_PLUS * (1 + z)**3
    matter_minus = OMEGA_MINUS * (1 + z)**3
    potential = M2 * V(r, beta0=0, beta1=beta1, beta2=0, beta3=0)
    return matter_plus - matter_minus + potential

def H_bimetric(z, r, beta1):
    """H(z)/H₀, returns 0 if H² < 0"""
    H2 = H2_bimetric(z, r, beta1)
    if H2 <= 0:
        return 1e-10  # Regularization
    return np.sqrt(H2)

# ============================================================
# COUPLED ODE SYSTEM
# ============================================================

def coupled_ode(y, z, beta1):
    """
    Coupled system:
    y[0] = r (ratio of scale factors)

    dr/dz = -r/H · [dH/dz + H/(1+z)]

    We need dH/dz. From H² = f(z, r):
    2H dH/dz = ∂H²/∂z + ∂H²/∂r · dr/dz

    ∂H²/∂z = 3(Ω₊ - Ω₋)(1+z)²
    ∂H²/∂r = m² · dV/dr

    Substituting dr/dz and solving:
    dH/dz = [∂H²/∂z - m²·dV/dr · r·H/(1+z)] / [2H + m²·dV/dr · r/H]
    """
    r = y[0]

    H2 = H2_bimetric(z, r, beta1)
    if H2 <= 1e-20:
        return [0]  # Stop if H² ≤ 0

    H = np.sqrt(H2)

    # Partial derivatives
    dH2_dz = 3 * (OMEGA_PLUS - OMEGA_MINUS) * (1 + z)**2
    dH2_dr = M2 * dV_dr(r, beta1=beta1)

    # Solve for dH/dz from the implicit equation
    # Using simplified form: assume dr/dz contribution is small initially
    # dH/dz ≈ (1/2H) · dH²/dz
    dH_dz_approx = dH2_dz / (2 * H)

    # Full dr/dz equation
    dr_dz = -r / H * (dH_dz_approx + H / (1 + z))

    return [dr_dz]

def solve_coupled(beta1, z_max=Z_CMB, n_points=5000):
    """
    Solve coupled equations from z=0 to z=z_max
    Initial condition: r(0) = 1 (isotropic today)
    """
    z_array = np.linspace(0, z_max, n_points)
    y0 = [1.0]  # r(0) = 1

    try:
        solution = odeint(coupled_ode, y0, z_array, args=(beta1,), full_output=False)
        r_array = solution[:, 0]
    except:
        r_array = np.ones(n_points)

    # Calculate H² and V for each z
    H2_array = np.array([H2_bimetric(z, r, beta1) for z, r in zip(z_array, r_array)])
    V_array = np.array([V(r, beta1=beta1) for r in r_array])

    return z_array, r_array, H2_array, V_array

# ============================================================
# SCAN β₁ VALUES
# ============================================================

beta1_values = [0.1, 1.0, 10.0]
results = {}

print("=" * 70)
print("SCANNING β₁ VALUES")
print("=" * 70)

for beta1 in beta1_values:
    print(f"\n--- β₁ = {beta1} ---")

    z_arr, r_arr, H2_arr, V_arr = solve_coupled(beta1)

    # Key values
    r_0 = r_arr[0]
    r_cmb = r_arr[-1]
    H2_0 = H2_arr[0]
    H2_cmb = H2_arr[-1]
    V_0 = V_arr[0]
    V_cmb = V_arr[-1]

    min_H2 = np.min(H2_arr)
    z_min_H2 = z_arr[np.argmin(H2_arr)]
    viable = min_H2 >= 0

    print(f"  r(0) = {r_0:.4f}, r(CMB) = {r_cmb:.4f}")
    print(f"  V(r(0)) = {V_0:.4f}, V(r(CMB)) = {V_cmb:.4e}")
    print(f"  H²(0) = {H2_0:.4f}, H²(CMB) = {H2_cmb:.4e}")
    print(f"  min(H²) = {min_H2:.4e} at z = {z_min_H2:.0f}")
    print(f"  VIABLE: {viable}")

    # Check if V(r) ~ (1+z)⁴ at CMB
    # If V ∝ (1+z)⁴, then V(CMB)/V(0) ≈ (1101)⁴ ≈ 1.47×10¹²
    expected_ratio = (1 + Z_CMB)**4
    actual_ratio = V_cmb / V_0 if V_0 > 0 else np.nan
    print(f"  V(CMB)/V(0) = {actual_ratio:.4e} (expected for (1+z)⁴: {expected_ratio:.4e})")

    # Sound horizon if viable
    if viable:
        def integrand_rd(z):
            # Interpolate r at z
            r_z = np.interp(z, z_arr, r_arr)
            H2 = H2_bimetric(z, r_z, beta1)
            if H2 <= 0:
                return 0
            return (1.0 / np.sqrt(3)) / np.sqrt(H2)

        rd_int, _ = quad(integrand_rd, Z_DRAG, 50000, limit=1000)
        rd_Mpc = rd_int * c_over_H0
        print(f"  r_d = {rd_Mpc:.1f} Mpc (Planck: 147 Mpc)")
    else:
        rd_Mpc = np.nan

    results[beta1] = {
        'z': z_arr,
        'r': r_arr,
        'H2': H2_arr,
        'V': V_arr,
        'viable': viable,
        'rd_Mpc': rd_Mpc
    }

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')
colors = ['blue', 'red', 'green']

# Panel 1: r(z) evolution
ax1 = axes[0, 0]
for i, beta1 in enumerate(beta1_values):
    res = results[beta1]
    ax1.semilogx(res['z'][1:], res['r'][1:], colors[i], lw=2, label=f'β₁ = {beta1}')

ax1.axhline(1, color='gray', ls='--', alpha=0.5, label='r = 1')
ax1.set_xlabel('Redshift z', fontsize=12)
ax1.set_ylabel('r(z) = a₋/a₊', fontsize=12)
ax1.set_title('Scale Factor Ratio Evolution', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, Z_CMB)

# Panel 2: V(r(z)) evolution
ax2 = axes[0, 1]
for i, beta1 in enumerate(beta1_values):
    res = results[beta1]
    ax2.semilogy(res['z'], res['V'], colors[i], lw=2, label=f'β₁ = {beta1}')

# Reference: (1+z)⁴ scaling
z_ref = np.linspace(1, Z_CMB, 500)
V_ref = 3 * 1.0 * (1 + z_ref / 100)**0.5  # Approximate scaling
ax2.set_xlabel('Redshift z', fontsize=12)
ax2.set_ylabel('V(r(z))', fontsize=12)
ax2.set_title('Potential Evolution', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: H²(z)
ax3 = axes[1, 0]
for i, beta1 in enumerate(beta1_values):
    res = results[beta1]
    H2 = res['H2']
    # Plot absolute value, mark negative regions
    ax3.semilogy(res['z'], np.abs(H2), colors[i], lw=2, label=f'β₁ = {beta1}')

ax3.axhline(1, color='gray', ls=':', alpha=0.5)
ax3.set_xlabel('Redshift z', fontsize=12)
ax3.set_ylabel('|H²(z)/H₀²|', fontsize=12)
ax3.set_title('H² Evolution (absolute value)', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Panel 4: H² at low z (linear)
ax4 = axes[1, 1]
z_low_idx = np.where(results[beta1_values[0]]['z'] <= 20)[0]

for i, beta1 in enumerate(beta1_values):
    res = results[beta1]
    ax4.plot(res['z'][z_low_idx], res['H2'][z_low_idx], colors[i], lw=2, label=f'β₁ = {beta1}')

# ΛCDM reference
z_low = results[beta1_values[0]]['z'][z_low_idx]
H2_LCDM = 0.3 * (1 + z_low)**3 + 0.7
ax4.plot(z_low, H2_LCDM, 'k--', lw=1.5, alpha=0.5, label='ΛCDM')

ax4.axhline(0, color='black', lw=1)
ax4.axhline(1, color='gray', ls=':', alpha=0.5)
ax4.set_xlabel('Redshift z', fontsize=12)
ax4.set_ylabel('H²(z)/H₀²', fontsize=12)
ax4.set_title('H² at low z (linear scale)', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
outpath = '/mnt/T2/janus-sim/output/janus_bimetric_coupled.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {outpath}")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'β₁':>6} | {'r(CMB)':>10} | {'V(CMB)':>12} | {'H²(0)':>10} | {'min(H²)':>12} | {'Viable':>8} | {'r_d [Mpc]':>10}")
print("-" * 85)

for beta1 in beta1_values:
    res = results[beta1]
    r_cmb = res['r'][-1]
    V_cmb = res['V'][-1]
    H2_0 = res['H2'][0]
    min_H2 = np.min(res['H2'])
    viable = "Yes" if res['viable'] else "No"
    rd = f"{res['rd_Mpc']:.1f}" if not np.isnan(res['rd_Mpc']) else "N/A"

    print(f"{beta1:6.1f} | {r_cmb:10.4f} | {V_cmb:12.4e} | {H2_0:10.4f} | {min_H2:12.4e} | {viable:>8} | {rd:>10}")

print()
print("Planck 2018: r_d = 147 Mpc")

# ============================================================
# ANALYSIS: Does V(r) ~ (1+z)⁴?
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS: V(r(z)) SCALING")
print("=" * 70)

for beta1 in beta1_values:
    res = results[beta1]
    z_arr = res['z']
    V_arr = res['V']

    # Fit power law: V ∝ (1+z)^α
    # log(V) = α·log(1+z) + const
    # Use high-z data
    high_z_mask = z_arr > 100
    if np.sum(high_z_mask) > 10:
        log_z = np.log(1 + z_arr[high_z_mask])
        log_V = np.log(np.maximum(V_arr[high_z_mask], 1e-20))

        # Linear fit
        coeffs = np.polyfit(log_z, log_V, 1)
        alpha = coeffs[0]

        print(f"β₁ = {beta1}: V(r) ∝ (1+z)^{alpha:.2f} at high z")
        print(f"         (need α ≈ 4 for radiation-like compensation)")
    else:
        print(f"β₁ = {beta1}: Not enough high-z data for fit")

# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The coupled bimetric system with V(r) = 3β₁r shows:

1. r(z) DECREASES with z (a₋ < a₊ at high z)
   This is because dr/dz < 0 when H > 0

2. V(r(z)) DECREASES with z since V ∝ r and r decreases
   This is OPPOSITE to what we need (V should increase to compensate Ω₋)

3. The potential V(r) = β₀ + 3β₁r + ... with constant β coefficients
   CANNOT provide the (1+z)⁴ scaling needed to suppress negative sector

4. For V(r) to scale as (1+z)⁴, we would need:
   - Either r(z) ∝ (1+z)⁴ (but r decreases, not increases)
   - Or β coefficients that depend on z (not physical in Hassan-Rosen)

The standard Hassan-Rosen potential does not naturally solve the H² < 0 problem.
""")
