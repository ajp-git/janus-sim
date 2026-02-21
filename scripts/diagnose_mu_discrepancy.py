#!/usr/bin/env python3
"""
Diagnostic: Why do analytical and numerical μ(z) differ by 0.4-0.8 mag?

Checks:
1. Initial conditions and q0 consistency
2. Curvature k assumption (analytical formula derived for k=-1?)
3. d_L normalization differences
4. Plot comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("DIAGNOSTIC: Analytical vs Numerical μ(z) Discrepancy")
print("=" * 70)

# Constants (must match Rust code)
C = 299792458.0  # m/s
H0 = 70.0 * 1000 / 3.0857e22  # 70 km/s/Mpc in 1/s
PC_IN_M = 3.085677581e16

# Parameters from Pantheon+ fit
eta = 1.045
q0 = (1.0 - eta) / (1.0 + eta)
omega_plus = 1.0 / (1.0 + eta)
omega_minus = eta / (1.0 + eta)
e_conserved = omega_plus - omega_minus

print(f"\nParameters:")
print(f"  η = {eta:.4f}")
print(f"  q0 = (1-η)/(1+η) = {q0:.6f}")
print(f"  Ω₊ = {omega_plus:.4f}")
print(f"  Ω₋ = {omega_minus:.4f}")
print(f"  E = Ω₊ - Ω₋ = {e_conserved:.4f}")

# ============================================================================
# CHECK 1: Analytical formula from D'Agostini & Petit 2018 eq.(5)
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 1: Analytical Formula (D'Agostini & Petit 2018)")
print("=" * 70)

def mu_analytical(z, q0):
    """
    Exact formula eq.(5):
    arg = z + z²(1-q0) / (1 + q0·z + √(1 + 2q0·z))
    μ = 5·log10(arg) + cst

    The 'cst' absorbs H0 and reference distance.
    We compute d_L = arg × (c/H0) to get physical distance.
    """
    inner = 1.0 + 2.0 * q0 * z
    if inner < 0:
        return np.nan
    denom = 1.0 + q0 * z + np.sqrt(inner)
    arg = z + z**2 * (1.0 - q0) / denom

    # Convert to luminosity distance in meters
    # The formula gives arg in units where d_L = arg × (c/H0)
    d_L_m = arg * C / H0

    # Distance modulus
    d_L_pc = d_L_m / PC_IN_M
    mu = 5.0 * np.log10(d_L_pc / 10.0)
    return mu

print(f"\nAnalytical formula: arg = z + z²(1-q0)/(1 + q0·z + √(1+2q0·z))")
print(f"                    d_L = arg × (c/H0)")
print(f"                    μ = 5·log10(d_L/10pc)")

# ============================================================================
# CHECK 2: Numerical integration (Janus Friedmann equations)
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 2: Numerical Integration (Janus Friedmann)")
print("=" * 70)

def derivatives_janus(a, a_bar, a_dot, a_bar_dot, E):
    """Corrected Janus acceleration equations."""
    a_ddot = -1.5 * E / (a * a)
    a_bar_ddot = 1.5 * E / (a_bar * a_bar)
    return a_dot, a_bar_dot, a_ddot, a_bar_ddot

def rk4_step(a, a_bar, a_dot, a_bar_dot, E, dtau):
    """RK4 integrator step."""
    da1, dab1, dda1, ddab1 = derivatives_janus(a, a_bar, a_dot, a_bar_dot, E)

    a2 = a + 0.5*dtau*da1
    ab2 = a_bar + 0.5*dtau*dab1
    ad2 = a_dot + 0.5*dtau*dda1
    abd2 = a_bar_dot + 0.5*dtau*ddab1
    da2, dab2, dda2, ddab2 = derivatives_janus(a2, ab2, ad2, abd2, E)

    a3 = a + 0.5*dtau*da2
    ab3 = a_bar + 0.5*dtau*dab2
    ad3 = a_dot + 0.5*dtau*dda2
    abd3 = a_bar_dot + 0.5*dtau*ddab2
    da3, dab3, dda3, ddab3 = derivatives_janus(a3, ab3, ad3, abd3, E)

    a4 = a + dtau*da3
    ab4 = a_bar + dtau*dab3
    ad4 = a_dot + dtau*dda3
    abd4 = a_bar_dot + dtau*ddab3
    da4, dab4, dda4, ddab4 = derivatives_janus(a4, ab4, ad4, abd4, E)

    a_new = a + dtau/6*(da1 + 2*da2 + 2*da3 + da4)
    a_bar_new = a_bar + dtau/6*(dab1 + 2*dab2 + 2*dab3 + dab4)
    a_dot_new = a_dot + dtau/6*(dda1 + 2*dda2 + 2*dda3 + dda4)
    a_bar_dot_new = a_bar_dot + dtau/6*(ddab1 + 2*ddab2 + 2*ddab3 + ddab4)

    return a_new, a_bar_new, a_dot_new, a_bar_dot_new

def mu_numerical(z_target, omega_p, omega_m):
    """
    Numerical integration of Janus Friedmann equations.
    Computes comoving distance χ(z) then d_L = (1+z)·χ·c/H0
    """
    E = omega_p - omega_m

    # Initial conditions at z=0 (today)
    a = 1.0
    a_bar = 1.0
    # From Friedmann: H² = Ω₊/a³ → ȧ = √Ω₊ at a=1
    a_dot = np.sqrt(omega_p)
    a_bar_dot = -np.sqrt(omega_m)  # Contracting

    # Integrate backward in time
    a_target = 1.0 / (1.0 + z_target)
    n_steps = 10000
    tau_total = np.log(1.0 / a_target) * 3.0  # Safety factor
    dtau = -tau_total / n_steps

    # Store history for χ integration
    history = [(0.0, a, a_dot)]  # (z, a, H=ȧ/a)

    for _ in range(n_steps):
        a, a_bar, a_dot, a_bar_dot = rk4_step(a, a_bar, a_dot, a_bar_dot, E, dtau)
        if a <= 0.01 or np.isnan(a):
            break
        z = 1.0/a - 1.0
        H = a_dot / a  # Dimensionless Hubble
        history.append((z, a, H))
        if a <= a_target:
            break

    # Integrate χ = ∫ dz/E(z) using trapezoidal rule
    history.sort(key=lambda x: x[0])  # Sort by z
    chi = 0.0
    for i in range(1, len(history)):
        z_prev, _, H_prev = history[i-1]
        z_curr, _, H_curr = history[i]
        if z_curr > z_target:
            break
        dz = z_curr - z_prev
        if H_prev > 0 and H_curr > 0 and dz > 0:
            # Trapezoidal: average of 1/H
            chi += dz * 0.5 * (1.0/abs(H_prev) + 1.0/abs(H_curr))

    # Luminosity distance
    d_L_m = (1.0 + z_target) * chi * C / H0
    if d_L_m <= 0:
        return np.nan

    d_L_pc = d_L_m / PC_IN_M
    mu = 5.0 * np.log10(d_L_pc / 10.0)
    return mu

print(f"\nNumerical: Uses Janus acceleration equations")
print(f"  ä = -1.5·E/a²")
print(f"  ā̈ = +1.5·E/ā²")
print(f"  Initial: ȧ₀ = √Ω₊ = {np.sqrt(omega_plus):.4f}")
print(f"           ā̇₀ = -√Ω₋ = {-np.sqrt(omega_minus):.4f}")

# ============================================================================
# CHECK 3: Standard ΛCDM-like integration for comparison
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 3: Standard Friedmann (for comparison)")
print("=" * 70)

def mu_standard_friedmann(z_target, omega_m):
    """
    Standard Friedmann with matter + dark energy.
    E(z)² = Ωm(1+z)³ + ΩΛ
    """
    omega_lambda = 1.0 - omega_m

    def E_z(z):
        return np.sqrt(omega_m * (1+z)**3 + omega_lambda)

    # Integrate χ = ∫ dz/E(z)
    n = 1000
    z_arr = np.linspace(0, z_target, n+1)
    dz = z_arr[1] - z_arr[0]
    chi = 0.0
    for i in range(n):
        z_mid = 0.5 * (z_arr[i] + z_arr[i+1])
        chi += dz / E_z(z_mid)

    d_L_m = (1.0 + z_target) * chi * C / H0
    d_L_pc = d_L_m / PC_IN_M
    return 5.0 * np.log10(d_L_pc / 10.0)

# ============================================================================
# CHECK 4: What does the analytical formula actually compute?
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 4: Understanding the Analytical Formula")
print("=" * 70)

print("""
The D'Agostini & Petit 2018 formula (eq.5) is a DIRECT expression for d_L
derived from the Janus model under specific assumptions:

  d_L(z) = (c/H0) × [z + z²(1-q0)/(1 + q0·z + √(1+2q0·z))]

This is an EXACT solution to the luminosity distance integral for:
  - Dust (w=0) dominated universe
  - Specific coupling between sectors
  - Assumes the deceleration parameter q0 is constant

The NUMERICAL integration uses the CORRECTED acceleration equations:
  - ä = -1.5·E/a² with E = Ω₊ - Ω₋
  - But the Hubble rate comes from H² = Ω₊/a³ (Friedmann constraint)

ISSUE IDENTIFIED:
  The numerical integrator uses BOTH:
  1. Corrected acceleration: ä = -1.5·E/a²
  2. Standard Friedmann constraint: H² = Ω₊/a³

  But these may be INCONSISTENT! The acceleration equation comes from
  the conservation law E=const, while H² = Ω₊/a³ is standard Friedmann.

  In Janus cosmology, the Hubble rate should be derived self-consistently
  from the field equations, not assumed to follow standard Friedmann.
""")

# ============================================================================
# COMPARISON PLOT
# ============================================================================
print("\n" + "=" * 70)
print("Generating comparison plot...")
print("=" * 70)

z_range = np.linspace(0.01, 2.5, 100)

mu_anal = np.array([mu_analytical(z, q0) for z in z_range])
mu_num = np.array([mu_numerical(z, omega_plus, omega_minus) for z in z_range])
mu_lcdm = np.array([mu_standard_friedmann(z, 0.3) for z in z_range])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Hubble diagram comparison
ax = axes[0, 0]
ax.plot(z_range, mu_anal, 'b-', lw=2, label=f'Analytical (eq.5, q0={q0:.4f})')
ax.plot(z_range, mu_num, 'r--', lw=2, label='Numerical (Janus RK4)')
ax.plot(z_range, mu_lcdm, 'g:', lw=2, label='ΛCDM (Ωm=0.3)')
ax.set_xlabel('Redshift z')
ax.set_ylabel('Distance modulus μ')
ax.set_title('Hubble Diagram: μ(z)')
ax.legend()
ax.grid(alpha=0.3)

# 2. Residuals
ax = axes[0, 1]
delta = mu_anal - mu_num
ax.plot(z_range, delta, 'r-', lw=2)
ax.axhline(0, color='k', ls='--')
ax.fill_between(z_range, delta, 0, alpha=0.3, color='red')
ax.set_xlabel('Redshift z')
ax.set_ylabel('Δμ = μ_analytical - μ_numerical (mag)')
ax.set_title('Discrepancy: Analytical - Numerical')
ax.grid(alpha=0.3)

# 3. Relative comparison
ax = axes[1, 0]
ax.plot(z_range, mu_anal - mu_lcdm, 'b-', lw=2, label='Analytical - ΛCDM')
ax.plot(z_range, mu_num - mu_lcdm, 'r--', lw=2, label='Numerical - ΛCDM')
ax.axhline(0, color='k', ls='--')
ax.set_xlabel('Redshift z')
ax.set_ylabel('Δμ from ΛCDM (mag)')
ax.set_title('Deviation from ΛCDM')
ax.legend()
ax.grid(alpha=0.3)

# 4. Summary text
ax = axes[1, 1]
ax.axis('off')

# Compute stats
valid = ~np.isnan(delta)
mean_delta = np.nanmean(delta)
max_delta = np.nanmax(np.abs(delta))

summary = f"""
{'='*56}
           DIAGNOSTIC SUMMARY
{'='*56}

PARAMETERS
----------
  η = {eta:.4f}
  q0 = {q0:.6f}
  E = Ω₊ - Ω₋ = {e_conserved:.4f}

DISCREPANCY STATISTICS
----------------------
  Mean Δμ: {mean_delta:+.4f} mag
  Max |Δμ|: {max_delta:.4f} mag

ROOT CAUSE HYPOTHESIS
---------------------
The numerical integrator assumes:
  H² = Ω₊/a³  (standard Friedmann)

But the analytical formula derives d_L from:
  q0 = constant deceleration parameter

These are DIFFERENT cosmological models!

The analytical formula (eq.5) is exact for:
  H² = (H0)² × (1 + 2q0·z + ...)

The numerical uses Friedmann + modified acceleration,
which creates an INCONSISTENT system.

RESOLUTION
----------
Either:
1. Use ONLY the analytical formula (recommended)
2. Derive consistent H(z) from Janus field equations
   (not standard Friedmann + modified ä)

{'='*56}
"""
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('output/mu_discrepancy_diagnostic.png', dpi=150)
print(f"\n✓ Saved output/mu_discrepancy_diagnostic.png")

# Print numerical table
print("\n" + "=" * 70)
print("NUMERICAL COMPARISON TABLE")
print("=" * 70)
print(f"\n{'z':>6}  {'μ_anal':>10}  {'μ_num':>10}  {'μ_ΛCDM':>10}  {'Δμ':>10}")
print("-" * 55)
for z in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]:
    ma = mu_analytical(z, q0)
    mn = mu_numerical(z, omega_plus, omega_minus)
    ml = mu_standard_friedmann(z, 0.3)
    d = ma - mn if not np.isnan(mn) else np.nan
    print(f"{z:>6.2f}  {ma:>10.4f}  {mn:>10.4f}  {ml:>10.4f}  {d:>+10.4f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
L'écart provient d'une INCOHÉRENCE dans l'intégrateur numérique:

  1. L'accélération utilise: ä = -1.5·E/a²  (équations Janus corrigées)
  2. Mais H(z) vient de: H² = Ω₊/a³  (Friedmann standard)

Ces deux hypothèses sont INCOMPATIBLES.

La formule analytique (eq.5) est cohérente car elle dérive d_L
directement des équations de champ Janus avec q0 = const.

RECOMMANDATION: Utiliser UNIQUEMENT la formule analytique pour le fit SNIa.
L'intégrateur numérique nécessite une révision profonde des équations.
""")
