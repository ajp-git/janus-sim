#!/usr/bin/env python3
"""
Find effective ΛCDM parameters (Ω_m, Ω_Λ) that best fit Janus H(z).

For each η, integrate the coupled Janus Friedmann equations,
then find (Ω_m, Ω_Λ) that minimizes χ² between H_janus(z) and H_lcdm(z).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def integrate_janus(eta, z_max=5.0, n_points=200):
    """
    Integrate coupled Janus Friedmann equations backward from z=0 to z=z_max.

    Returns arrays: z, H/H0

    Equations (from Petit & D'Agostini 2014):
        ä = -1.5 * E / a²
        ā̈ = +1.5 * E / ā²

    where E = Ω₊ - Ω₋ = (1-η)/(1+η)

    Initial conditions at z=0 (a=ā=1):
        ȧ₀ = √Ω₊
        ā̇₀ = -√Ω₋ (contracting)
    """
    omega_plus = 1.0 / (1.0 + eta)
    omega_minus = eta / (1.0 + eta)
    E = omega_plus - omega_minus

    # State vector: [a, a_bar, a_dot, a_bar_dot]
    def derivatives(state, tau):
        a, a_bar, a_dot, a_bar_dot = state
        if a <= 0.01 or a_bar <= 0.01:
            return [0, 0, 0, 0]
        a_ddot = -1.5 * E / (a * a)
        a_bar_ddot = 1.5 * E / (a_bar * a_bar)
        return [a_dot, a_bar_dot, a_ddot, a_bar_ddot]

    # Initial conditions at z=0
    a0 = 1.0
    a_bar0 = 1.0
    a_dot0 = np.sqrt(omega_plus)
    a_bar_dot0 = -np.sqrt(omega_minus)

    state0 = [a0, a_bar0, a_dot0, a_bar_dot0]

    # Integrate backward (negative tau)
    # Estimate tau range needed to reach z_max
    a_target = 1.0 / (1.0 + z_max)
    tau_total = np.log(1.0 / a_target) * 2.5  # Safety factor

    tau = np.linspace(0, -tau_total, 5000)

    solution = odeint(derivatives, state0, tau)

    a = solution[:, 0]
    a_dot = solution[:, 2]

    # Filter valid points
    valid = (a > 0.01) & (a <= 1.0)
    a = a[valid]
    a_dot = a_dot[valid]

    # Compute z and H/H0
    z = 1.0 / a - 1.0
    H = a_dot / a  # H in units where H(z=0) = √Ω₊

    # Normalize so H(z=0)/H0 = 1
    H0 = np.sqrt(omega_plus)
    H_over_H0 = H / H0

    # Sort by z
    idx = np.argsort(z)
    z = z[idx]
    H_over_H0 = H_over_H0[idx]

    # Interpolate to regular z grid
    z_grid = np.linspace(0, min(z_max, z.max()), n_points)
    H_grid = np.interp(z_grid, z, H_over_H0)

    return z_grid, H_grid


def H_lcdm(z, omega_m, omega_lambda):
    """
    ΛCDM Hubble parameter: H(z)/H0 = sqrt(Ω_m(1+z)³ + Ω_Λ)
    """
    return np.sqrt(omega_m * (1 + z)**3 + omega_lambda)


def chi2_lcdm(params, z_data, H_data):
    """
    χ² between Janus H(z) and ΛCDM H(z).
    """
    omega_m, omega_lambda = params
    if omega_m < 0 or omega_lambda < 0:
        return 1e10
    H_model = H_lcdm(z_data, omega_m, omega_lambda)
    return np.sum((H_data - H_model)**2)


def fit_lcdm_to_janus(eta, z_max_fit=2.0):
    """
    Find best-fit ΛCDM parameters for Janus H(z) with given η.
    """
    # Generate Janus H(z)
    z, H = integrate_janus(eta, z_max=5.0, n_points=500)

    # Fit only up to z_max_fit
    mask = z <= z_max_fit
    z_fit = z[mask]
    H_fit = H[mask]

    # Optimize
    result = minimize(
        chi2_lcdm,
        x0=[0.3, 0.7],  # Initial guess
        args=(z_fit, H_fit),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-6}
    )

    omega_m_best, omega_lambda_best = result.x
    chi2_best = result.fun

    return omega_m_best, omega_lambda_best, chi2_best, z, H


def main():
    print("=" * 70)
    print("Janus → ΛCDM Effective Parameters")
    print("Finding (Ω_m, Ω_Λ) that best match Janus H(z) for various η")
    print("=" * 70)
    print()

    # η values to test
    eta_values = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    results = []

    print(f"{'η':>8}  {'Ω_m':>8}  {'Ω_Λ':>8}  {'Ω_tot':>8}  {'χ²':>12}  {'q₀':>8}")
    print("-" * 70)

    for eta in eta_values:
        omega_m, omega_lambda, chi2, z, H = fit_lcdm_to_janus(eta, z_max_fit=2.0)
        omega_tot = omega_m + omega_lambda

        # Janus deceleration parameter
        q0_janus = (1 - eta) / (1 + eta)

        results.append({
            'eta': eta,
            'omega_m': omega_m,
            'omega_lambda': omega_lambda,
            'omega_tot': omega_tot,
            'chi2': chi2,
            'q0': q0_janus,
            'z': z,
            'H': H
        })

        print(f"{eta:>8.3f}  {omega_m:>8.4f}  {omega_lambda:>8.4f}  "
              f"{omega_tot:>8.4f}  {chi2:>12.6f}  {q0_janus:>8.4f}")

    print()

    # Find η that gives Ω_Λ ≈ 0.7
    print("=" * 70)
    print("Finding η that corresponds to ΛCDM Ω_Λ = 0.7")
    print("=" * 70)
    print()

    # More detailed scan around likely region
    eta_fine = np.linspace(0.85, 1.25, 81)
    omega_lambda_fine = []

    for eta in eta_fine:
        omega_m, omega_lambda, _, _, _ = fit_lcdm_to_janus(eta, z_max_fit=2.0)
        omega_lambda_fine.append(omega_lambda)

    omega_lambda_fine = np.array(omega_lambda_fine)

    # Find η where Ω_Λ ≈ 0.7
    idx_07 = np.argmin(np.abs(omega_lambda_fine - 0.7))
    eta_std = eta_fine[idx_07]
    omega_m_std, omega_lambda_std, chi2_std, _, _ = fit_lcdm_to_janus(eta_std)

    print(f"ΛCDM standard (Ω_Λ = 0.7) corresponds to:")
    print(f"  η = {eta_std:.4f}")
    print(f"  Ω_m = {omega_m_std:.4f}")
    print(f"  Ω_Λ = {omega_lambda_std:.4f}")
    print(f"  χ² = {chi2_std:.6f}")
    print()

    # Check if relation is linear
    eta_arr = np.array([r['eta'] for r in results])
    omega_lambda_arr = np.array([r['omega_lambda'] for r in results])

    # Linear fit: Ω_Λ = a + b*η
    coeffs = np.polyfit(eta_arr, omega_lambda_arr, 1)
    omega_lambda_linear = np.polyval(coeffs, eta_arr)
    r2 = 1 - np.sum((omega_lambda_arr - omega_lambda_linear)**2) / \
             np.sum((omega_lambda_arr - np.mean(omega_lambda_arr))**2)

    print("=" * 70)
    print("Relation Ω_Λ(η)")
    print("=" * 70)
    print()
    print(f"Linear fit: Ω_Λ = {coeffs[1]:.4f} + {coeffs[0]:.4f} × η")
    print(f"R² = {r2:.6f}")
    print()

    if r2 > 0.99:
        print("✓ Excellent linear relation!")
    elif r2 > 0.95:
        print("~ Good linear approximation")
    else:
        print("✗ Non-linear relation")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='#0d1117')

    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Plot 1: Ω_Λ vs η
    ax1 = axes[0, 0]
    ax1.plot(eta_fine, omega_lambda_fine, 'o-', color='#58a6ff',
             markersize=2, label='Fit results')
    ax1.axhline(0.7, color='#f85149', ls='--', label='ΛCDM Ω_Λ=0.7')
    ax1.axvline(eta_std, color='#f85149', ls=':', alpha=0.5)
    ax1.set_xlabel('η', color='#e6edf3')
    ax1.set_ylabel('Ω_Λ (effective)', color='#e6edf3')
    ax1.set_title(f'Effective Ω_Λ vs η\n(ΛCDM Ω_Λ=0.7 → η={eta_std:.3f})',
                  color='#e6edf3')
    ax1.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax1.grid(True, alpha=0.15, color='#8b949e')

    # Plot 2: Ω_m vs η
    ax2 = axes[0, 1]
    omega_m_fine = []
    for eta in eta_fine:
        om, _, _, _, _ = fit_lcdm_to_janus(eta, z_max_fit=2.0)
        omega_m_fine.append(om)
    ax2.plot(eta_fine, omega_m_fine, 'o-', color='#3fb950', markersize=2)
    ax2.axhline(0.3, color='#f85149', ls='--', label='ΛCDM Ω_m=0.3')
    ax2.set_xlabel('η', color='#e6edf3')
    ax2.set_ylabel('Ω_m (effective)', color='#e6edf3')
    ax2.set_title('Effective Ω_m vs η', color='#e6edf3')
    ax2.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax2.grid(True, alpha=0.15, color='#8b949e')

    # Plot 3: H(z) comparison for η=1.045
    ax3 = axes[1, 0]
    eta_ref = 1.045
    om_ref, ol_ref, _, z_ref, H_ref = fit_lcdm_to_janus(eta_ref)
    H_lcdm_ref = H_lcdm(z_ref, om_ref, ol_ref)
    H_lcdm_std = H_lcdm(z_ref, 0.3, 0.7)

    ax3.plot(z_ref, H_ref, '-', color='#58a6ff', lw=2, label=f'Janus η={eta_ref}')
    ax3.plot(z_ref, H_lcdm_ref, '--', color='#3fb950', lw=2,
             label=f'ΛCDM fit (Ω_m={om_ref:.3f}, Ω_Λ={ol_ref:.3f})')
    ax3.plot(z_ref, H_lcdm_std, ':', color='#f85149', lw=2,
             label='ΛCDM std (Ω_m=0.3, Ω_Λ=0.7)')
    ax3.set_xlabel('Redshift z', color='#e6edf3')
    ax3.set_ylabel('H(z)/H₀', color='#e6edf3')
    ax3.set_title(f'H(z) comparison for η={eta_ref}', color='#e6edf3')
    ax3.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax3.grid(True, alpha=0.15, color='#8b949e')
    ax3.set_xlim(0, 2)

    # Plot 4: Multiple η curves
    ax4 = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    for i, r in enumerate(results):
        ax4.plot(r['z'], r['H'], '-', color=colors[i], lw=1.5,
                 label=f"η={r['eta']:.2f}")
    ax4.set_xlabel('Redshift z', color='#e6edf3')
    ax4.set_ylabel('H(z)/H₀', color='#e6edf3')
    ax4.set_title('Janus H(z) for various η', color='#e6edf3')
    ax4.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3',
               loc='upper left')
    ax4.grid(True, alpha=0.15, color='#8b949e')
    ax4.set_xlim(0, 2)

    plt.tight_layout()
    plt.savefig('output/janus_to_lcdm.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print()
    print("✓ Saved output/janus_to_lcdm.png")

    # Save data to CSV
    with open('output/janus_lcdm_params.csv', 'w') as f:
        f.write("eta,omega_m,omega_lambda,omega_tot,chi2,q0_janus\n")
        for r in results:
            f.write(f"{r['eta']:.4f},{r['omega_m']:.6f},{r['omega_lambda']:.6f},"
                    f"{r['omega_tot']:.6f},{r['chi2']:.8f},{r['q0']:.6f}\n")
    print("✓ Saved output/janus_lcdm_params.csv")
    print()


if __name__ == "__main__":
    main()
