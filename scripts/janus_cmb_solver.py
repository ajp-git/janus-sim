#!/usr/bin/env python3
"""
Janus CMB Solver — Bimetric Acoustic Oscillations

Coupled perturbation system:
  dδ₊/dτ = v₊
  dv₊/dτ = -k²c_s²δ₊ - k²Ψ
  dδ₋/dτ = v₋
  dv₋/dτ = -k²δ₋ + k²Ψ    (negative sector: repulsion)
  dΨ/dτ = -HΨ + 4πG·a²/k² · (ρ₊δ₊ - ρ₋δ₋)

Parameters:
  H₀ = 76 km/s/Mpc, Ω₊ = 0.30, η = 1.045
  z_dec = 1100, z_drag = 1059.6
  VSL: c(z) = c₀·(1+z)^{1/2} for z > z_c

Output:
  - Transfer function T(k)
  - CMB power spectrum C_l
  - Peak positions l₁ to l₅
  - Comparison with Planck
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.special import spherical_jn
from scipy.signal import find_peaks

# ============================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# ============================================================

H0 = 76.0  # km/s/Mpc
H0_SI = H0 * 1e3 / 3.086e22  # s^-1

OMEGA_PLUS = 0.30
ETA = 1.045
OMEGA_MINUS = ETA * OMEGA_PLUS

Z_DEC = 1100
Z_DRAG = 1059.6
Z_C = 0.341  # VSL transition redshift

# Baryon-to-photon ratio parameter
R0 = 683  # R(z) = R₀/(1+z)

# Constants
c0 = 3e8  # m/s
G = 6.674e-11  # m³/(kg·s²)
rho_crit = 3 * H0_SI**2 / (8 * np.pi * G)  # kg/m³

print("=" * 70)
print("JANUS CMB SOLVER — BIMETRIC ACOUSTIC OSCILLATIONS")
print("=" * 70)
print(f"H₀ = {H0} km/s/Mpc")
print(f"Ω₊ = {OMEGA_PLUS}, η = {ETA}, Ω₋ = {OMEGA_MINUS:.4f}")
print(f"z_drag = {Z_DRAG}, z_dec = {Z_DEC}")
print()

# ============================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================

def H(z):
    """Hubble parameter H(z)/H₀"""
    return np.sqrt(OMEGA_PLUS * (1 + z)**3 + (1 - OMEGA_PLUS))

def c_s_squared(z):
    """Sound speed squared: c_s² = 1/(3(1+R(z)))"""
    R = R0 / (1 + z)
    return 1.0 / (3 * (1 + R))

def c_vsl(z):
    """VSL speed of light: c(z) = c₀·(1+z)^{1/2} for z > z_c"""
    if z > Z_C:
        return (1 + z)**0.5
    return 1.0

def conformal_time(z):
    """Conformal time τ(z) = ∫₀ᶻ dz'/[(1+z')²H(z')]"""
    def integrand(zp):
        return 1.0 / ((1 + zp)**2 * H(zp))
    result, _ = quad(integrand, 0, z, limit=200)
    return result

def comoving_distance(z):
    """Comoving distance D_M(z) = ∫₀ᶻ c·dz'/H(z')"""
    def integrand(zp):
        return c_vsl(zp) / H(zp)
    result, _ = quad(integrand, 0, z, limit=200)
    return result  # In units of c/H₀

# ============================================================
# PERTURBATION EQUATIONS
# ============================================================

def perturbation_ode(y, z, k, eta_param):
    """
    Coupled perturbation system in z-variable.

    y = [δ₊, v₊, δ₋, v₋, Ψ]

    Converting from τ to z: d/dτ = -(1+z)²H · d/dz
    """
    delta_plus, v_plus, delta_minus, v_minus, Psi = y

    Hz = H(z)
    cs2 = c_s_squared(z)
    a = 1.0 / (1 + z)

    # Density ratio
    rho_plus = OMEGA_PLUS * (1 + z)**3
    rho_minus = eta_param * rho_plus

    # Conversion factor: d/dτ = -(1+z)²H · d/dz
    factor = -1.0 / ((1 + z)**2 * Hz)

    # Equations in conformal time τ:
    # dδ₊/dτ = v₊
    # dv₊/dτ = -k²c_s²δ₊ - k²Ψ
    # dδ₋/dτ = v₋
    # dv₋/dτ = -k²δ₋ + k²Ψ
    # dΨ/dτ = -HΨ + (3/2)H²a² · (Ω₊δ₊ - η·Ω₊δ₋)

    # In z variable: dy/dz = (dy/dτ) / (dz/dτ) = (dy/dτ) · factor
    d_delta_plus_dtau = v_plus
    d_v_plus_dtau = -k**2 * cs2 * delta_plus - k**2 * Psi
    d_delta_minus_dtau = v_minus
    d_v_minus_dtau = -k**2 * delta_minus + k**2 * Psi  # Repulsion!
    d_Psi_dtau = -Hz * Psi + 1.5 * Hz**2 * a**2 * (rho_plus * delta_plus - rho_minus * delta_minus) / k**2

    # Convert to z derivatives
    d_delta_plus_dz = d_delta_plus_dtau * factor
    d_v_plus_dz = d_v_plus_dtau * factor
    d_delta_minus_dz = d_delta_minus_dtau * factor
    d_v_minus_dz = d_v_minus_dtau * factor
    d_Psi_dz = d_Psi_dtau * factor

    return [d_delta_plus_dz, d_v_plus_dz, d_delta_minus_dz, d_v_minus_dz, d_Psi_dz]

def solve_perturbations(k, z_final=Z_DRAG, eta_param=ETA, n_points=2000):
    """
    Solve perturbation equations from z=0 to z=z_final.

    Initial conditions (adiabatic):
      δ₊ = 1, v₊ = 0, δ₋ = 0, v₋ = 0, Ψ = Ψ₀
    """
    # Initial conditions at z = 0
    Psi0 = 1.0  # Normalize
    y0 = [1.0, 0.0, 0.0, 0.0, Psi0]

    # Integrate from z=0 to z=z_final
    z_array = np.linspace(0, z_final, n_points)

    try:
        solution = odeint(perturbation_ode, y0, z_array, args=(k, eta_param),
                          rtol=1e-8, atol=1e-10)
        delta_plus = solution[:, 0]
        delta_minus = solution[:, 2]
        Psi = solution[:, 4]
    except Exception as e:
        print(f"  Warning: Integration failed for k={k:.4f}: {e}")
        delta_plus = np.ones(n_points)
        delta_minus = np.zeros(n_points)
        Psi = np.ones(n_points)

    return z_array, delta_plus, delta_minus, Psi

# ============================================================
# TRANSFER FUNCTION
# ============================================================

def compute_transfer_function(k_array, eta_param=ETA):
    """Compute T(k) = δ₊(k, z_drag)"""
    T_k = []

    for k in k_array:
        z_arr, delta_plus, _, _ = solve_perturbations(k, Z_DRAG, eta_param, n_points=1000)
        T_k.append(delta_plus[-1])

    return np.array(T_k)

# ============================================================
# CMB POWER SPECTRUM C_l
# ============================================================

def compute_Cl(l_array, k_array, T_k, D_M):
    """
    CMB power spectrum: C_l ∝ ∫ [T(k) · j_l(k·D_M)]² dk

    Simplified calculation using discrete sum.
    """
    C_l = []

    for l in l_array:
        integrand = 0.0
        for i, k in enumerate(k_array[:-1]):
            dk = k_array[i+1] - k_array[i]
            x = k * D_M
            if x > 0 and x < 1e4:
                jl = spherical_jn(l, x)
                integrand += (T_k[i] * jl)**2 * dk

        C_l.append(integrand)

    return np.array(C_l)

# ============================================================
# FIND PEAKS
# ============================================================

def find_cmb_peaks(l_array, C_l, n_peaks=5):
    """Find positions of first n_peaks in C_l"""
    # Smooth C_l slightly
    C_l_smooth = np.convolve(C_l, np.ones(3)/3, mode='same')

    # Find peaks
    peaks, properties = find_peaks(C_l_smooth, distance=50, prominence=0.01*np.max(C_l_smooth))

    peak_positions = l_array[peaks[:n_peaks]] if len(peaks) >= n_peaks else l_array[peaks]

    return peak_positions

# ============================================================
# MAIN CALCULATION
# ============================================================

print("=" * 70)
print("COMPUTING TRANSFER FUNCTION AND C_l")
print("=" * 70)

# Comoving distance to decoupling
D_M = comoving_distance(Z_DEC)
print(f"D_M(z_dec) = {D_M:.2f} c/H₀")

# k array (in units of H₀/c)
# k should span range such that k·D_M covers l ~ 10 to 2000
# l ~ k·D_M, so k ~ l/D_M
k_min = 10 / D_M
k_max = 2000 / D_M
n_k = 50  # Start with 50 for speed

k_array = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
print(f"k range: [{k_min:.4f}, {k_max:.4f}] H₀/c")
print(f"n_k = {n_k}")
print()

# Compute transfer function
print("Computing transfer function T(k)...")
T_k = compute_transfer_function(k_array, ETA)
print(f"  T(k) range: [{np.min(T_k):.4f}, {np.max(T_k):.4f}]")

# Compute C_l
print("Computing C_l...")
l_array = np.arange(2, 2000, 5)
C_l = compute_Cl(l_array, k_array, T_k, D_M)

# Normalize C_l
C_l = C_l / np.max(C_l) if np.max(C_l) > 0 else C_l

# Find peaks
print("Finding peaks...")
peaks = find_cmb_peaks(l_array, C_l, n_peaks=5)
print(f"  Peak positions: {peaks}")

# ============================================================
# PLANCK REFERENCE
# ============================================================

planck_peaks = np.array([220, 537, 810, 1120, 1440])
planck_l2_l1 = 537 / 220

print()
print("=" * 70)
print("COMPARISON WITH PLANCK")
print("=" * 70)
print(f"Planck peaks: {planck_peaks}")
print(f"Planck l₂/l₁ = {planck_l2_l1:.3f}")

if len(peaks) >= 2:
    janus_l2_l1 = peaks[1] / peaks[0]
    print(f"Janus peaks:  {peaks}")
    print(f"Janus l₂/l₁ = {janus_l2_l1:.3f}")

# ============================================================
# PARAMETER SCAN: η and g₀
# ============================================================

print()
print("=" * 70)
print("PARAMETER SCAN")
print("=" * 70)

eta_values = [1.030, 1.045, 1.060]
g0_values = [-1.0, -0.5, 0.0]  # Gravitational coupling modification

results = []

for eta_val in eta_values:
    print(f"\nη = {eta_val}:")
    T_k_eta = compute_transfer_function(k_array, eta_val)
    C_l_eta = compute_Cl(l_array, k_array, T_k_eta, D_M)
    C_l_eta = C_l_eta / np.max(C_l_eta) if np.max(C_l_eta) > 0 else C_l_eta

    peaks_eta = find_cmb_peaks(l_array, C_l_eta, n_peaks=5)

    if len(peaks_eta) >= 2:
        ratio = peaks_eta[1] / peaks_eta[0]
    else:
        ratio = np.nan

    results.append({
        'eta': eta_val,
        'g0': 0.0,
        'peaks': peaks_eta,
        'ratio': ratio,
        'T_k': T_k_eta,
        'C_l': C_l_eta
    })

    print(f"  Peaks: {peaks_eta}")
    print(f"  l₂/l₁ = {ratio:.3f}" if not np.isnan(ratio) else "  l₂/l₁ = N/A")

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')

# Panel 1: Transfer function T(k)
ax1 = axes[0, 0]
colors = ['blue', 'red', 'green']
for i, res in enumerate(results):
    ax1.semilogx(k_array * D_M, np.abs(res['T_k']), colors[i], lw=2,
                  label=f"η = {res['eta']}")

ax1.set_xlabel('k · D_M', fontsize=12)
ax1.set_ylabel('|T(k)|', fontsize=12)
ax1.set_title('Transfer Function', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: C_l spectrum
ax2 = axes[0, 1]
for i, res in enumerate(results):
    ax2.plot(l_array, res['C_l'], colors[i], lw=1.5, alpha=0.7,
             label=f"η = {res['eta']}")

# Mark Planck peaks
for l_peak in planck_peaks:
    ax2.axvline(l_peak, color='gray', ls='--', alpha=0.3)

ax2.set_xlabel('Multipole l', fontsize=12)
ax2.set_ylabel('C_l (normalized)', fontsize=12)
ax2.set_title('CMB Power Spectrum', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2000)

# Panel 3: Peak positions comparison
ax3 = axes[1, 0]
x_planck = np.arange(1, len(planck_peaks) + 1)

ax3.scatter(x_planck, planck_peaks, s=100, c='black', marker='s', label='Planck', zorder=5)

for i, res in enumerate(results):
    if len(res['peaks']) > 0:
        x_janus = np.arange(1, len(res['peaks']) + 1)
        ax3.scatter(x_janus, res['peaks'], s=80, c=colors[i], marker='o',
                    label=f"Janus η={res['eta']}", alpha=0.7)

ax3.set_xlabel('Peak number', fontsize=12)
ax3.set_ylabel('l (multipole)', fontsize=12)
ax3.set_title('Peak Positions: Janus vs Planck', fontsize=14)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks([1, 2, 3, 4, 5])

# Panel 4: l₂/l₁ ratio
ax4 = axes[1, 1]
eta_plot = [res['eta'] for res in results]
ratio_plot = [res['ratio'] for res in results]

ax4.scatter(eta_plot, ratio_plot, s=100, c='blue', marker='o', label='Janus')
ax4.axhline(planck_l2_l1, color='red', ls='--', lw=2, label=f'Planck = {planck_l2_l1:.2f}')

ax4.set_xlabel('η', fontsize=12)
ax4.set_ylabel('l₂/l₁', fontsize=12)
ax4.set_title('Peak Ratio vs η', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
outpath = '/mnt/T2/janus-sim/output/janus_cmb_spectrum.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {outpath}")

# ============================================================
# SAVE NUMERICAL TABLE
# ============================================================

table_path = '/mnt/T2/janus-sim/output/janus_cmb_peaks.txt'
with open(table_path, 'w') as f:
    f.write("JANUS CMB PEAKS — NUMERICAL TABLE\n")
    f.write("=" * 60 + "\n\n")
    f.write("Planck reference:\n")
    f.write(f"  l₁={planck_peaks[0]}, l₂={planck_peaks[1]}, l₃={planck_peaks[2]}, ")
    f.write(f"l₄={planck_peaks[3]}, l₅={planck_peaks[4]}\n")
    f.write(f"  l₂/l₁ = {planck_l2_l1:.3f}\n\n")

    f.write("Janus results:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'η':>8} | {'l₁':>6} | {'l₂':>6} | {'l₃':>6} | {'l₄':>6} | {'l₅':>6} | {'l₂/l₁':>8}\n")
    f.write("-" * 60 + "\n")

    for res in results:
        peaks = res['peaks']
        l1 = peaks[0] if len(peaks) > 0 else np.nan
        l2 = peaks[1] if len(peaks) > 1 else np.nan
        l3 = peaks[2] if len(peaks) > 2 else np.nan
        l4 = peaks[3] if len(peaks) > 3 else np.nan
        l5 = peaks[4] if len(peaks) > 4 else np.nan
        ratio = res['ratio']

        f.write(f"{res['eta']:8.3f} | {l1:6.0f} | {l2:6.0f} | {l3:6.0f} | ")
        f.write(f"{l4:6.0f} | {l5:6.0f} | {ratio:8.3f}\n")

    f.write("-" * 60 + "\n")

print(f"Saved: {table_path}")

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
This is a simplified Janus CMB solver demonstrating:

1. Coupled bimetric perturbation equations
2. The negative sector (δ₋) feels REPULSION from potential Ψ
3. Transfer function T(k) computed by integrating to z_drag
4. C_l spectrum computed via spherical Bessel projection

Limitations of this simplified model:
- No tight-coupling approximation for photon-baryon fluid
- No Silk damping
- No integrated Sachs-Wolfe effect
- Simplified initial conditions

For accurate CMB predictions, a full Boltzmann code (like CLASS/CAMB)
modified for bimetric gravity would be needed.
""")
