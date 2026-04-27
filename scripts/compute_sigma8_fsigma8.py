#!/usr/bin/env python3
"""
Compute σ8(z) and fσ8(z) from μ=19 canonical Janus snapshots

σ8 = variance of m+ density field smoothed at R = 8 h⁻¹Mpc
f = d(ln σ8) / d(ln a) = growth rate
fσ8 = f × σ8

Compare to BOSS/eBOSS observations
"""

import numpy as np
import struct
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.interpolate import UnivariateSpline

# Parameters
BASE_DIR = "/mnt/T2/janus-sim/output/scan_mu_evolution/mu19"
BOX_SIZE = 1000.0  # Mpc
N_GRID = 256
H0 = 70.0  # km/s/Mpc
h = H0 / 100.0  # h = 0.7
R_SMOOTH = 8.0 / h  # 8 h⁻¹Mpc = 11.43 Mpc

# Janus cosmology parameters
MU = 19.0
OMEGA_B = 0.05
OMEGA_M = OMEGA_B * (1 + MU)  # = 1.00 for flat universe

print("="*70)
print("σ8(z) and fσ8(z) COMPUTATION — μ=19 CANONICAL JANUS")
print("="*70)
print(f"  Box: {BOX_SIZE} Mpc, Grid: {N_GRID}³")
print(f"  Smoothing scale: R = 8 h⁻¹Mpc = {R_SMOOTH:.2f} Mpc")
print(f"  Ωm = {OMEGA_M:.2f} (flat universe)")
print("="*70)

def read_snapshot(filename):
    """Read binary snapshot: N particles × 25 bytes (x,y,z,vx,vy,vz,sign)"""
    with open(filename, 'rb') as f:
        data = f.read()

    n_particles = len(data) // 25
    positions = []
    signs = []

    for i in range(n_particles):
        offset = i * 25
        x, y, z = struct.unpack('<fff', data[offset:offset+12])
        # Skip velocities (12 bytes)
        sign = struct.unpack('<b', data[offset+24:offset+25])[0]
        positions.append([x, y, z])
        signs.append(sign)

    return np.array(positions), np.array(signs)

def compute_density_field(positions, signs, box_size, n_grid, positive_only=True):
    """Compute density field on grid using CIC assignment"""

    # Filter by sign
    if positive_only:
        mask = signs > 0
        pos = positions[mask]
    else:
        pos = positions

    # Wrap positions to [0, box_size)
    pos = pos % box_size

    # Grid cell size
    cell_size = box_size / n_grid

    # CIC (Cloud-In-Cell) assignment
    density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    # Cell indices
    grid_pos = pos / cell_size
    i = grid_pos.astype(int) % n_grid
    f = grid_pos - grid_pos.astype(int)  # fractional part

    # CIC weights to 8 neighboring cells
    for di in [0, 1]:
        for dj in [0, 1]:
            for dk in [0, 1]:
                wi = (1 - f[:, 0]) if di == 0 else f[:, 0]
                wj = (1 - f[:, 1]) if dj == 0 else f[:, 1]
                wk = (1 - f[:, 2]) if dk == 0 else f[:, 2]
                weight = wi * wj * wk

                ii = (i[:, 0] + di) % n_grid
                jj = (i[:, 1] + dj) % n_grid
                kk = (i[:, 2] + dk) % n_grid

                np.add.at(density, (ii, jj, kk), weight)

    return density

def tophat_filter_fourier(density, box_size, R):
    """Apply top-hat filter in Fourier space

    W(kR) = 3(sin(kR) - kR·cos(kR)) / (kR)³
    """
    n = density.shape[0]

    # FFT
    density_k = np.fft.fftn(density)

    # k-space grid
    kx = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    ky = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    kz = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Top-hat window function
    kR = k * R
    W = np.ones_like(kR)
    mask = kR > 1e-10
    W[mask] = 3 * (np.sin(kR[mask]) - kR[mask] * np.cos(kR[mask])) / (kR[mask]**3)

    # Apply filter
    density_k_filtered = density_k * W

    # IFFT
    density_filtered = np.fft.ifftn(density_k_filtered).real

    return density_filtered

def compute_sigma8(positions, signs, box_size, n_grid, R):
    """Compute σ8 = std of smoothed density contrast δ = (ρ - ρ̄)/ρ̄"""

    # Compute density field (m+ only)
    density = compute_density_field(positions, signs, box_size, n_grid, positive_only=True)

    # Mean density
    rho_mean = density.mean()
    if rho_mean == 0:
        return 0.0

    # Density contrast
    delta = (density - rho_mean) / rho_mean

    # Apply top-hat smoothing
    delta_smoothed = tophat_filter_fourier(delta, box_size, R)

    # σ8 = standard deviation of smoothed field
    sigma8 = np.std(delta_smoothed)

    return sigma8

# Get list of snapshots
snapshot_dir = os.path.join(BASE_DIR, "snapshots")
snapshots = sorted(glob(os.path.join(snapshot_dir, "snap_*.bin")))
print(f"\nFound {len(snapshots)} snapshots")

# Read time series for z and t values
time_series_file = os.path.join(BASE_DIR, "time_series.csv")
time_data = np.genfromtxt(time_series_file, delimiter=',', skip_header=1)
# Columns: step, t_gyr, z, a, H, diff_pois, corr_delta, exc_var_minus

# Create mapping step -> (z, t_Gyr, a)
step_to_z = {}
step_to_t = {}
step_to_a = {}
for row in time_data:
    step = int(row[0])
    t_gyr = row[1]
    z = row[2]
    a = row[3]
    step_to_z[step] = z
    step_to_t[step] = t_gyr
    step_to_a[step] = a

# Process snapshots
results = []
print("\nProcessing snapshots...")

for i, snap_file in enumerate(snapshots):
    # Extract step number from filename
    basename = os.path.basename(snap_file)
    step = int(basename.replace("snap_", "").replace(".bin", ""))

    # Get z, t, a for this step
    if step not in step_to_z:
        # Interpolate if not exact match
        closest_step = min(step_to_z.keys(), key=lambda x: abs(x - step))
        z = step_to_z[closest_step]
        t_gyr = step_to_t[closest_step]
        a = step_to_a[closest_step]
    else:
        z = step_to_z[step]
        t_gyr = step_to_t[step]
        a = step_to_a[step]

    # Read snapshot
    positions, signs = read_snapshot(snap_file)

    # Compute σ8
    sigma8 = compute_sigma8(positions, signs, BOX_SIZE, N_GRID, R_SMOOTH)

    results.append({
        'step': step,
        'z': z,
        't_Gyr': t_gyr,
        'a': a,
        'sigma8': sigma8
    })

    if i % 50 == 0 or i == len(snapshots) - 1:
        print(f"  [{i+1}/{len(snapshots)}] step={step}, z={z:.3f}, σ8={sigma8:.4f}")

# Sort by step
results = sorted(results, key=lambda x: x['step'])

# Convert to arrays
steps = np.array([r['step'] for r in results])
z_arr = np.array([r['z'] for r in results])
t_arr = np.array([r['t_Gyr'] for r in results])
a_arr = np.array([r['a'] for r in results])
sigma8_arr = np.array([r['sigma8'] for r in results])

print("\n" + "="*70)
print("COMPUTING GROWTH RATE f(z)")
print("="*70)

# Compute f = d(ln σ8) / d(ln a)
# Use numerical derivative with smoothing
ln_sigma8 = np.log(sigma8_arr + 1e-10)
ln_a = np.log(a_arr + 1e-10)

# Smooth with spline then differentiate
# Sort by ln_a (increasing)
sort_idx = np.argsort(ln_a)
ln_a_sorted = ln_a[sort_idx]
ln_sigma8_sorted = ln_sigma8[sort_idx]

# Fit spline
spline = UnivariateSpline(ln_a_sorted, ln_sigma8_sorted, s=0.01)
f_sorted = spline.derivative()(ln_a_sorted)

# Restore original order
f_arr = np.zeros_like(f_sorted)
f_arr[sort_idx] = f_sorted

# Compute fσ8
fsigma8_arr = f_arr * sigma8_arr

# ΛCDM prediction for comparison
# f(z) ≈ Ωm(z)^0.55 for ΛCDM
# σ8(z) = σ8(0) × D(z)/D(0) where D is growth factor
SIGMA8_LCDM = 0.81
OMEGA_M_LCDM = 0.31

def lcdm_fsigma8(z):
    """ΛCDM prediction for fσ8(z)"""
    a = 1.0 / (1.0 + z)
    # Simplified growth factor for ΛCDM
    Omega_m_z = OMEGA_M_LCDM * (1+z)**3 / (OMEGA_M_LCDM * (1+z)**3 + (1-OMEGA_M_LCDM))
    f = Omega_m_z ** 0.55
    # Growth factor approximation
    D = a * (OMEGA_M_LCDM / (OMEGA_M_LCDM + (1-OMEGA_M_LCDM)*a**3))**0.23
    D0 = OMEGA_M_LCDM ** 0.23
    sigma8_z = SIGMA8_LCDM * D / D0
    return f * sigma8_z

# BOSS/eBOSS data points
boss_data = [
    (0.38, 0.497, 0.045),  # z, fσ8, error
    (0.51, 0.458, 0.038),
    (0.70, 0.473, 0.041),
]

# Additional RSD measurements
other_data = [
    (0.15, 0.490, 0.045, '6dFGS'),
    (0.32, 0.384, 0.095, 'BOSS lowz'),
    (0.57, 0.441, 0.043, 'BOSS CMASS'),
    (0.85, 0.315, 0.095, 'VIPERS'),
    (1.40, 0.482, 0.116, 'eBOSS QSO'),
]

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"  σ8(z=4.0) = {sigma8_arr[0]:.4f}")
print(f"  σ8(z=0)   = {sigma8_arr[-1]:.4f}")
print(f"  f(z=0)    = {f_arr[-1]:.4f}")
print(f"  fσ8(z=0)  = {fsigma8_arr[-1]:.4f}")
print(f"\n  S8 = σ8√(Ωm/0.3) = {sigma8_arr[-1] * np.sqrt(OMEGA_M/0.3):.4f}")
print(f"  (ΛCDM: σ8=0.81, fσ8(z=0)≈0.46)")
print("="*70)

# Save CSV
output_csv = os.path.join(BASE_DIR, "sigma8_evolution.csv")
with open(output_csv, 'w') as f:
    f.write("step,z,t_Gyr,a,sigma8,f,fsigma8\n")
    for i in range(len(steps)):
        f.write(f"{steps[i]},{z_arr[i]:.4f},{t_arr[i]:.4f},{a_arr[i]:.6f},"
                f"{sigma8_arr[i]:.6f},{f_arr[i]:.6f},{fsigma8_arr[i]:.6f}\n")
print(f"\nSaved: {output_csv}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: σ8(z)
ax1 = axes[0, 0]
ax1.plot(z_arr, sigma8_arr, 'k-', linewidth=2, label='Janus μ=19')
ax1.axhline(y=SIGMA8_LCDM, color='gray', linestyle='--', alpha=0.5, label=f'ΛCDM σ8={SIGMA8_LCDM}')
ax1.set_xlabel('Redshift z', fontsize=12)
ax1.set_ylabel('σ₈(z)', fontsize=12)
ax1.set_title('σ₈ Evolution', fontsize=14)
ax1.set_xlim(0, 4)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# Panel 2: f(z)
ax2 = axes[0, 1]
ax2.plot(z_arr, f_arr, 'k-', linewidth=2, label='Janus μ=19')
# ΛCDM f(z)
z_lcdm = np.linspace(0, 2, 100)
omega_m_z = OMEGA_M_LCDM * (1+z_lcdm)**3 / (OMEGA_M_LCDM * (1+z_lcdm)**3 + (1-OMEGA_M_LCDM))
f_lcdm = omega_m_z ** 0.55
ax2.plot(z_lcdm, f_lcdm, 'b--', linewidth=1.5, alpha=0.7, label='ΛCDM')
ax2.set_xlabel('Redshift z', fontsize=12)
ax2.set_ylabel('f(z) = d(ln σ₈)/d(ln a)', fontsize=12)
ax2.set_title('Growth Rate', fontsize=14)
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 1.5)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

# Panel 3: fσ8(z) - main result
ax3 = axes[1, 0]
ax3.plot(z_arr, fsigma8_arr, 'k-', linewidth=2.5, label='Janus μ=19 (Ωm=1.00)')

# ΛCDM prediction
z_pred = np.linspace(0, 2, 100)
fsigma8_lcdm = [lcdm_fsigma8(z) for z in z_pred]
ax3.plot(z_pred, fsigma8_lcdm, 'b--', linewidth=1.5, alpha=0.7, label='ΛCDM (Ωm=0.31, σ8=0.81)')

# BOSS/eBOSS data points
for z, fs8, err in boss_data:
    ax3.errorbar(z, fs8, yerr=err, fmt='ro', markersize=8, capsize=4,
                 label='BOSS/eBOSS' if z == 0.38 else '')

# Other data points
for z, fs8, err, name in other_data:
    ax3.errorbar(z, fs8, yerr=err, fmt='s', markersize=6, capsize=3,
                 alpha=0.7, label=name)

ax3.set_xlabel('Redshift z', fontsize=12)
ax3.set_ylabel('fσ₈(z)', fontsize=12)
ax3.set_title('fσ₈(z) — Janus vs Observations', fontsize=14)
ax3.set_xlim(0, 1.6)
ax3.set_ylim(0.2, 0.8)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

# Panel 4: S8 comparison
ax4 = axes[1, 1]
# S8 as function of z
S8_janus = sigma8_arr * np.sqrt(OMEGA_M / 0.3)
ax4.plot(z_arr, S8_janus, 'k-', linewidth=2, label='Janus μ=19')

# KiDS/DES constraints
ax4.axhspan(0.76 - 0.02, 0.76 + 0.02, alpha=0.3, color='green', label='KiDS-1000 (S8=0.76±0.02)')
ax4.axhspan(0.77 - 0.02, 0.77 + 0.02, alpha=0.3, color='orange', label='DES-Y3 (S8=0.77±0.02)')
ax4.axhline(y=0.83, color='blue', linestyle='--', alpha=0.5, label='Planck ΛCDM (S8=0.83)')

ax4.set_xlabel('Redshift z', fontsize=12)
ax4.set_ylabel('S₈ = σ₈√(Ωm/0.3)', fontsize=12)
ax4.set_title('S₈ Comparison', fontsize=14)
ax4.set_xlim(0, 2)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.invert_xaxis()

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "figure_fsigma8.png"), dpi=200)
print(f"Saved: {os.path.join(BASE_DIR, 'figure_fsigma8.png')}")

# Final summary
print("\n" + "="*70)
print("FINAL RESULTS — μ=19 CANONICAL JANUS")
print("="*70)
print(f"  σ8(z=0) = {sigma8_arr[-1]:.4f}")
print(f"  f(z=0)  = {f_arr[-1]:.4f}")
print(f"  fσ8(z=0) = {fsigma8_arr[-1]:.4f}")
print(f"  S8 = σ8√(Ωm/0.3) = {sigma8_arr[-1] * np.sqrt(OMEGA_M/0.3):.4f}")
print()
print("  Comparaison BOSS z=0.51: fσ8 = 0.458 ± 0.038")
janus_fs8_z051 = np.interp(0.51, z_arr[::-1], fsigma8_arr[::-1])
print(f"  Janus fσ8(z=0.51) = {janus_fs8_z051:.4f}")
print(f"  Tension: {abs(janus_fs8_z051 - 0.458) / 0.038:.1f}σ")
print("="*70)
