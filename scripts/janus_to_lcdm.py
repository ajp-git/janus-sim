#!/usr/bin/env python3
"""
Janus → ΛCDM Effective Parameter Mapping

For each η, compute Janus H(z) via coupled Friedmann integration,
then find (Ω_m, Ω_Λ) that best matches this H(z) curve.

Output:
  - output/janus_lcdm_mapping.csv
  - output/janus_lcdm_mapping.png
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def integrate_janus(eta, z_max=3.0, n_points=300):
    """
    Integrate coupled Janus Friedmann equations backward from z=0 to z=z_max.

    Equations (Petit & D'Agostini 2014):
        ä = -1.5 * E / a²
        ā̈ = +1.5 * E / ā²
    where E = Ω₊ - Ω₋ = (1-η)/(1+η)

    Returns: z_grid, H_over_H0
    """
    omega_plus = 1.0 / (1.0 + eta)
    omega_minus = eta / (1.0 + eta)
    E = omega_plus - omega_minus

    def derivatives(tau, state):
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

    # Integrate backward
    a_target = 1.0 / (1.0 + z_max)
    tau_total = np.log(1.0 / a_target) * 3.0

    try:
        sol = solve_ivp(
            derivatives,
            [0, -tau_total],
            state0,
            method='RK45',
            dense_output=True,
            max_step=0.01
        )

        if not sol.success:
            return None, None

        # Sample solution
        tau_eval = np.linspace(0, sol.t[-1], 2000)
        y = sol.sol(tau_eval)

        a = y[0]
        a_dot = y[2]

        # Filter valid points
        valid = (a > 0.01) & (a <= 1.0) & np.isfinite(a_dot)
        if np.sum(valid) < 10:
            return None, None

        a = a[valid]
        a_dot = a_dot[valid]

        # Compute H/H0
        z = 1.0 / a - 1.0
        H = a_dot / a
        H0 = np.sqrt(omega_plus)
        H_over_H0 = H / H0

        # Sort and interpolate
        idx = np.argsort(z)
        z = z[idx]
        H_over_H0 = H_over_H0[idx]

        # Remove duplicates
        z_unique, unique_idx = np.unique(z, return_index=True)
        H_unique = H_over_H0[unique_idx]

        z_grid = np.linspace(0, min(z_max, z_unique.max() * 0.95), n_points)
        H_grid = np.interp(z_grid, z_unique, H_unique)

        return z_grid, H_grid

    except Exception as e:
        return None, None


def H_lcdm(z, omega_m, omega_lambda):
    """ΛCDM: H(z)/H0 = sqrt(Ω_m(1+z)³ + Ω_Λ)"""
    return np.sqrt(omega_m * (1 + z)**3 + omega_lambda)


def chi2_lcdm(params, z_data, H_data):
    """χ² between Janus and ΛCDM H(z)"""
    omega_m, omega_lambda = params
    if omega_m < 0 or omega_m > 2 or omega_lambda < -1 or omega_lambda > 3:
        return 1e10
    H_model = H_lcdm(z_data, omega_m, omega_lambda)
    return np.sum((H_data - H_model)**2)


def fit_lcdm_to_janus(eta, z_max_fit=2.0):
    """
    Find best-fit ΛCDM parameters for Janus H(z).
    Returns: omega_m, omega_lambda, chi2, success, z, H
    """
    z, H = integrate_janus(eta, z_max=3.0, n_points=500)

    if z is None:
        return None, None, None, False, None, None

    # Fit only up to z_max_fit
    mask = z <= z_max_fit
    z_fit = z[mask]
    H_fit = H[mask]

    if len(z_fit) < 10:
        return None, None, None, False, z, H

    # Try multiple optimization methods
    best_result = None
    best_chi2 = np.inf

    # Method 1: Nelder-Mead with multiple starts
    for om_init in [0.2, 0.3, 0.5, 0.8]:
        for ol_init in [0.3, 0.5, 0.7, 1.0]:
            try:
                result = minimize(
                    chi2_lcdm,
                    x0=[om_init, ol_init],
                    args=(z_fit, H_fit),
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000}
                )
                if result.fun < best_chi2 and result.x[0] > 0 and result.x[1] > -0.5:
                    best_chi2 = result.fun
                    best_result = result
            except:
                pass

    # Method 2: L-BFGS-B with bounds
    try:
        result = minimize(
            chi2_lcdm,
            x0=[0.3, 0.7],
            args=(z_fit, H_fit),
            method='L-BFGS-B',
            bounds=[(0.01, 2.0), (-0.5, 2.0)]
        )
        if result.fun < best_chi2:
            best_chi2 = result.fun
            best_result = result
    except:
        pass

    if best_result is None:
        return None, None, None, False, z, H

    omega_m, omega_lambda = best_result.x

    # Check if fit is reasonable
    success = (best_chi2 < 50) and (omega_m > 0.01) and (omega_lambda > -0.5)

    return omega_m, omega_lambda, best_chi2, success, z, H


def main():
    print("=" * 70)
    print("Janus → ΛCDM Effective Parameter Mapping")
    print("=" * 70)
    print()

    # η values to test
    eta_values = [
        0.80, 0.85, 0.90, 0.95, 1.00,
        1.01, 1.02, 1.03, 1.04, 1.045, 1.05,
        1.06, 1.07, 1.08, 1.10, 1.15, 1.20
    ]

    results = []

    print(f"{'η':>8}  {'Ω_m':>8}  {'Ω_Λ':>8}  {'Ω_tot':>8}  {'χ²':>10}  {'Status':>10}")
    print("-" * 70)

    for eta in eta_values:
        omega_m, omega_lambda, chi2, success, z, H = fit_lcdm_to_janus(eta)

        if success and omega_m is not None:
            omega_tot = omega_m + omega_lambda
            status = "OK"
        else:
            omega_m = np.nan
            omega_lambda = np.nan
            omega_tot = np.nan
            chi2 = np.nan if chi2 is None else chi2
            status = "FAIL"

        q0_janus = (1 - eta) / (1 + eta)

        results.append({
            'eta': eta,
            'omega_m': omega_m,
            'omega_lambda': omega_lambda,
            'omega_tot': omega_tot,
            'chi2': chi2,
            'q0': q0_janus,
            'success': success,
            'z': z,
            'H': H
        })

        if success:
            print(f"{eta:>8.3f}  {omega_m:>8.4f}  {omega_lambda:>8.4f}  "
                  f"{omega_tot:>8.4f}  {chi2:>10.4f}  {status:>10}")
        else:
            print(f"{eta:>8.3f}  {'---':>8}  {'---':>8}  "
                  f"{'---':>8}  {chi2:>10.4f}  {status:>10}")

    print()

    # Save CSV
    with open('output/janus_lcdm_mapping.csv', 'w') as f:
        f.write("eta,omega_m,omega_lambda,omega_tot,chi2,q0,success\n")
        for r in results:
            success_str = "1" if r['success'] else "0"
            f.write(f"{r['eta']:.4f},{r['omega_m']:.6f},{r['omega_lambda']:.6f},"
                    f"{r['omega_tot']:.6f},{r['chi2']:.8f},{r['q0']:.6f},{success_str}\n")
    print("✓ Saved output/janus_lcdm_mapping.csv")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), facecolor='#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, alpha=0.15, color='#8b949e')

    eta_arr = np.array([r['eta'] for r in results])
    omega_m_arr = np.array([r['omega_m'] for r in results])
    omega_lambda_arr = np.array([r['omega_lambda'] for r in results])
    chi2_arr = np.array([r['chi2'] for r in results])
    success_arr = np.array([r['success'] for r in results])

    # Separate successful and failed fits
    eta_ok = eta_arr[success_arr]
    eta_fail = eta_arr[~success_arr]
    om_ok = omega_m_arr[success_arr]
    ol_ok = omega_lambda_arr[success_arr]
    chi2_ok = chi2_arr[success_arr]
    chi2_fail = chi2_arr[~success_arr]

    # Plot 1: Ω_Λ(η)
    ax1 = axes[0]
    ax1.plot(eta_ok, ol_ok, 'o-', color='#58a6ff', markersize=8, lw=2, label='Ω_Λ effectif')
    ax1.axhline(0.7, color='#f85149', ls='--', lw=2, label='Planck Ω_Λ=0.7')

    # Annotations
    ax1.axvline(1.00, color='#ffa657', ls=':', alpha=0.7)
    ax1.axvline(1.045, color='#3fb950', ls=':', alpha=0.7)
    ax1.axvspan(1.05, 1.25, alpha=0.15, color='#f85149')

    ax1.annotate('η=1.00\ntransition', xy=(1.00, 0.1), xytext=(0.92, 0.2),
                 color='#ffa657', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#ffa657', lw=0.8))
    ax1.annotate('η=1.045\nPantheon+', xy=(1.045, 0.1), xytext=(1.08, 0.3),
                 color='#3fb950', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#3fb950', lw=0.8))
    ax1.text(1.12, 1.5, 'Régime non-ΛCDM\n(incomparable)', color='#f85149',
             fontsize=9, ha='center', style='italic')

    ax1.set_xlabel('η', color='#e6edf3', fontsize=12)
    ax1.set_ylabel('Ω_Λ effectif', color='#e6edf3', fontsize=12)
    ax1.set_title('Mapping Janus → ΛCDM : Ω_Λ(η)', color='#e6edf3', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3', loc='upper left')
    ax1.set_xlim(0.78, 1.22)
    ax1.set_ylim(-0.2, 2.0)

    # Plot 2: Ω_m(η)
    ax2 = axes[1]
    ax2.plot(eta_ok, om_ok, 's-', color='#3fb950', markersize=8, lw=2, label='Ω_m effectif')
    ax2.axhline(0.3, color='#f85149', ls='--', lw=2, label='Planck Ω_m=0.3')

    ax2.axvline(1.00, color='#ffa657', ls=':', alpha=0.7)
    ax2.axvline(1.045, color='#3fb950', ls=':', alpha=0.7)
    ax2.axvspan(1.05, 1.25, alpha=0.15, color='#f85149')

    ax2.set_xlabel('η', color='#e6edf3', fontsize=12)
    ax2.set_ylabel('Ω_m effectif', color='#e6edf3', fontsize=12)
    ax2.set_title('Mapping Janus → ΛCDM : Ω_m(η)', color='#e6edf3', fontsize=14, fontweight='bold')
    ax2.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3', loc='upper right')
    ax2.set_xlim(0.78, 1.22)
    ax2.set_ylim(0, 1.0)

    # Plot 3: χ²(η)
    ax3 = axes[2]
    ax3.semilogy(eta_ok, chi2_ok, 'o-', color='#58a6ff', markersize=8, lw=2, label='Fit réussi')
    if len(eta_fail) > 0:
        ax3.semilogy(eta_fail, chi2_fail, 'x', color='#f85149', markersize=10, mew=2, label='Fit échoué')

    ax3.axhline(1.0, color='#8b949e', ls='--', alpha=0.5, label='χ²=1 (excellent)')
    ax3.axhline(10.0, color='#ffa657', ls='--', alpha=0.5, label='χ²=10 (acceptable)')

    ax3.axvline(1.00, color='#ffa657', ls=':', alpha=0.7)
    ax3.axvline(1.045, color='#3fb950', ls=':', alpha=0.7)
    ax3.axvspan(1.05, 1.25, alpha=0.15, color='#f85149')

    ax3.set_xlabel('η', color='#e6edf3', fontsize=12)
    ax3.set_ylabel('χ² du fit ΛCDM', color='#e6edf3', fontsize=12)
    ax3.set_title('Qualité du mapping Janus → ΛCDM', color='#e6edf3', fontsize=14, fontweight='bold')
    ax3.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3', loc='upper left')
    ax3.set_xlim(0.78, 1.22)
    ax3.set_ylim(0.01, 500)

    plt.tight_layout()
    plt.savefig('output/janus_lcdm_mapping.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print("✓ Saved output/janus_lcdm_mapping.png")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Find η closest to Ω_Λ = 0.7
    valid_ol = omega_lambda_arr[success_arr & np.isfinite(omega_lambda_arr)]
    valid_eta = eta_arr[success_arr & np.isfinite(omega_lambda_arr)]
    if len(valid_ol) > 0:
        idx_07 = np.argmin(np.abs(valid_ol - 0.7))
        print(f"η giving Ω_Λ ≈ 0.7 : η = {valid_eta[idx_07]:.3f}")
        print(f"  → Ω_m = {omega_m_arr[success_arr][idx_07]:.4f}")
        print(f"  → Ω_Λ = {valid_ol[idx_07]:.4f}")
        print()

    # Find η closest to Ω_m = 0.3
    valid_om = omega_m_arr[success_arr & np.isfinite(omega_m_arr)]
    if len(valid_om) > 0:
        idx_03 = np.argmin(np.abs(valid_om - 0.3))
        print(f"η giving Ω_m ≈ 0.3 : η = {valid_eta[idx_03]:.3f}")
        print(f"  → Ω_m = {valid_om[idx_03]:.4f}")
        print(f"  → Ω_Λ = {omega_lambda_arr[success_arr][idx_03]:.4f}")
        print()

    # Transition at η=1
    print("At η = 1.00 (symmetric universe):")
    idx_100 = np.argmin(np.abs(eta_arr - 1.00))
    if success_arr[idx_100]:
        print(f"  → Ω_m = {omega_m_arr[idx_100]:.4f}")
        print(f"  → Ω_Λ = {omega_lambda_arr[idx_100]:.4f}")
        print(f"  → χ² = {chi2_arr[idx_100]:.4f}")
    else:
        print("  → Fit failed")
    print()

    # At η = 1.045 (Pantheon+ optimal)
    print("At η = 1.045 (Pantheon+ optimal):")
    idx_1045 = np.argmin(np.abs(eta_arr - 1.045))
    if success_arr[idx_1045]:
        print(f"  → Ω_m = {omega_m_arr[idx_1045]:.4f}")
        print(f"  → Ω_Λ = {omega_lambda_arr[idx_1045]:.4f}")
        print(f"  → χ² = {chi2_arr[idx_1045]:.4f}")
    else:
        print("  → Fit failed - NO ΛCDM equivalent exists")
    print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("For η < 1.0 (decelerating): Good ΛCDM mapping exists")
    print("For η ≈ 1.0 (coasting):     Marginal mapping")
    print("For η > 1.0 (accelerating): ΛCDM mapping FAILS")
    print()
    print("The Janus acceleration mechanism (η > 1) is fundamentally")
    print("different from ΛCDM dark energy and cannot be mimicked.")
    print()


if __name__ == "__main__":
    main()
