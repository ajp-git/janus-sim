#!/usr/bin/env python3
"""
FAST σ8(z) and fσ8(z) computation from μ=19 snapshots
Uses NGP (Nearest Grid Point) assignment with numpy histogramdd
"""

import numpy as np
import struct
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Parameters
BASE_DIR = "/mnt/T2/janus-sim/output/scan_mu_evolution/mu19"
BOX_SIZE = 1000.0  # Mpc
N_GRID = 256
H0 = 70.0  # km/s/Mpc
h = H0 / 100.0  # h = 0.7
R_SMOOTH = 8.0 / h  # 8 h⁻¹Mpc = 11.43 Mpc

# Janus cosmology
MU = 19.0
OMEGA_B = 0.05
OMEGA_M = OMEGA_B * (1 + MU)  # = 1.00

print("="*70, flush=True)
print("σ8(z) and fσ8(z) — μ=19 CANONICAL JANUS (FAST)", flush=True)
print("="*70, flush=True)
print(f"  Box: {BOX_SIZE} Mpc, Grid: {N_GRID}³", flush=True)
print(f"  R = 8 h⁻¹Mpc = {R_SMOOTH:.2f} Mpc", flush=True)
print("="*70, flush=True)

def read_snapshot_fast(filename):
    """Read binary snapshot with 16-byte header"""
    with open(filename, 'rb') as f:
        # Read 16-byte header: n_particles(u32), box(f32), ?(f32), z(f32)
        header = f.read(16)
        n_particles = struct.unpack('<I', header[:4])[0]

        # Read particle data
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # 25 bytes per particle: x,y,z (3×f32), vx,vy,vz (3×f32), sign (i8)
    data = data.reshape(n_particles, 25)

    # Extract positions (bytes 0-11)
    positions = np.zeros((n_particles, 3), dtype=np.float32)
    for i in range(3):
        positions[:, i] = np.frombuffer(
            data[:, i*4:(i+1)*4].tobytes(), dtype=np.float32
        )

    # Extract signs (byte 24) - stored as i8, values are +1 or -1
    signs = data[:, 24].view(np.int8)

    return positions, signs

def compute_density_ngp(positions, box_size, n_grid):
    """NGP density assignment using histogramdd"""
    # Wrap to [0, box_size)
    pos = positions % box_size

    # Use histogramdd for fast binning
    bins = np.linspace(0, box_size, n_grid + 1)
    density, _ = np.histogramdd(pos, bins=[bins, bins, bins])

    return density.astype(np.float64)

def tophat_filter_fourier(delta, box_size, R):
    """Top-hat filter: W(kR) = 3(sin(kR) - kR·cos(kR))/(kR)³"""
    n = delta.shape[0]

    # FFT
    delta_k = np.fft.rfftn(delta)

    # k-space grid (rfft has reduced last dimension)
    kx = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    ky = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size/n) * 2 * np.pi

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Top-hat window
    kR = k * R
    W = np.ones_like(kR)
    mask = kR > 1e-10
    W[mask] = 3 * (np.sin(kR[mask]) - kR[mask] * np.cos(kR[mask])) / (kR[mask]**3)

    # Apply and IFFT
    delta_k *= W
    return np.fft.irfftn(delta_k, s=(n, n, n))

def compute_sigma8(positions, signs, box_size, n_grid, R):
    """Compute σ8 from m+ particles"""

    # Filter m+ only
    pos_plus = positions[signs > 0]

    if len(pos_plus) == 0:
        return 0.0

    # Density field
    density = compute_density_ngp(pos_plus, box_size, n_grid)

    # Density contrast
    rho_mean = density.mean()
    if rho_mean < 1e-10:
        return 0.0
    delta = (density - rho_mean) / rho_mean

    # Smooth
    delta_smooth = tophat_filter_fourier(delta, box_size, R)

    return np.std(delta_smooth)

# Get snapshots
snapshot_dir = os.path.join(BASE_DIR, "snapshots")
snapshots = sorted(glob(os.path.join(snapshot_dir, "snap_*.bin")))
print(f"\n{len(snapshots)} snapshots found", flush=True)

# Read time series
time_data = np.genfromtxt(os.path.join(BASE_DIR, "time_series.csv"),
                          delimiter=',', skip_header=1)

step_info = {}
for row in time_data:
    step = int(row[0])
    step_info[step] = {'t_gyr': row[1], 'z': row[2], 'a': row[3]}

# Process snapshots
results = []
print("\nProcessing...", flush=True)

for i, snap_file in enumerate(snapshots):
    step = int(os.path.basename(snap_file).split('_')[1].split('.')[0])

    # Get z, t, a
    if step in step_info:
        info = step_info[step]
    else:
        closest = min(step_info.keys(), key=lambda x: abs(x-step))
        info = step_info[closest]

    # Read and compute
    positions, signs = read_snapshot_fast(snap_file)
    sigma8 = compute_sigma8(positions, signs, BOX_SIZE, N_GRID, R_SMOOTH)

    results.append({
        'step': step,
        'z': info['z'],
        't_Gyr': info['t_gyr'],
        'a': info['a'],
        'sigma8': sigma8
    })

    if (i+1) % 40 == 0 or i == 0:
        print(f"  [{i+1:3d}/{len(snapshots)}] z={info['z']:.2f} σ8={sigma8:.4f}", flush=True)

# Sort and convert to arrays
results = sorted(results, key=lambda x: x['step'])
steps = np.array([r['step'] for r in results])
z_arr = np.array([r['z'] for r in results])
t_arr = np.array([r['t_Gyr'] for r in results])
a_arr = np.array([r['a'] for r in results])
sigma8_arr = np.array([r['sigma8'] for r in results])

print("\n" + "="*70, flush=True)
print("COMPUTING GROWTH RATE f(z)", flush=True)
print("="*70, flush=True)

# f = d(ln σ8) / d(ln a)
ln_sigma8 = np.log(sigma8_arr + 1e-10)
ln_a = np.log(a_arr + 1e-10)

# Sort by ln_a
idx = np.argsort(ln_a)
spline = UnivariateSpline(ln_a[idx], ln_sigma8[idx], s=0.01)
f_sorted = spline.derivative()(ln_a[idx])

f_arr = np.zeros_like(f_sorted)
f_arr[idx] = f_sorted

# fσ8
fsigma8_arr = f_arr * sigma8_arr

# Results
print(f"\n  σ8(z=4.0) = {sigma8_arr[0]:.4f}", flush=True)
print(f"  σ8(z=0)   = {sigma8_arr[-1]:.4f}", flush=True)
print(f"  f(z=0)    = {f_arr[-1]:.4f}", flush=True)
print(f"  fσ8(z=0)  = {fsigma8_arr[-1]:.4f}", flush=True)
print(f"\n  S8 = σ8√(Ωm/0.3) = {sigma8_arr[-1] * np.sqrt(OMEGA_M/0.3):.4f}", flush=True)

# Save CSV
csv_file = os.path.join(BASE_DIR, "sigma8_evolution.csv")
with open(csv_file, 'w') as f:
    f.write("step,z,t_Gyr,a,sigma8,f,fsigma8\n")
    for i in range(len(steps)):
        f.write(f"{steps[i]},{z_arr[i]:.4f},{t_arr[i]:.4f},{a_arr[i]:.6f},"
                f"{sigma8_arr[i]:.6f},{f_arr[i]:.6f},{fsigma8_arr[i]:.6f}\n")
print(f"\nSaved: {csv_file}", flush=True)

# ΛCDM reference
SIGMA8_LCDM = 0.81
OMEGA_M_LCDM = 0.31

def lcdm_fsigma8(z):
    a = 1.0 / (1.0 + z)
    Omega_m_z = OMEGA_M_LCDM * (1+z)**3 / (OMEGA_M_LCDM * (1+z)**3 + (1-OMEGA_M_LCDM))
    f = Omega_m_z ** 0.55
    D = a * (OMEGA_M_LCDM / (OMEGA_M_LCDM + (1-OMEGA_M_LCDM)*a**3))**0.23
    D0 = OMEGA_M_LCDM ** 0.23
    return f * SIGMA8_LCDM * D / D0

# Observations
boss_data = [(0.38, 0.497, 0.045), (0.51, 0.458, 0.038), (0.70, 0.473, 0.041)]
other_data = [
    (0.15, 0.490, 0.045, '6dFGS'),
    (0.32, 0.384, 0.095, 'BOSS lowz'),
    (0.57, 0.441, 0.043, 'BOSS CMASS'),
    (0.85, 0.315, 0.095, 'VIPERS'),
    (1.40, 0.482, 0.116, 'eBOSS QSO'),
]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# σ8(z)
ax1 = axes[0, 0]
ax1.plot(z_arr, sigma8_arr, 'k-', lw=2, label='Janus μ=19')
ax1.axhline(y=SIGMA8_LCDM, color='gray', ls='--', alpha=0.5, label=f'ΛCDM σ8={SIGMA8_LCDM}')
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('σ₈(z)')
ax1.set_title('σ₈ Evolution')
ax1.set_xlim(0, 4)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# f(z)
ax2 = axes[0, 1]
ax2.plot(z_arr, f_arr, 'k-', lw=2, label='Janus μ=19')
z_lcdm = np.linspace(0, 2, 100)
omega_m_z = OMEGA_M_LCDM * (1+z_lcdm)**3 / (OMEGA_M_LCDM * (1+z_lcdm)**3 + (1-OMEGA_M_LCDM))
ax2.plot(z_lcdm, omega_m_z**0.55, 'b--', lw=1.5, alpha=0.7, label='ΛCDM')
ax2.set_xlabel('Redshift z')
ax2.set_ylabel('f(z)')
ax2.set_title('Growth Rate')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 1.5)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

# fσ8(z) - main plot
ax3 = axes[1, 0]
ax3.plot(z_arr, fsigma8_arr, 'k-', lw=2.5, label='Janus μ=19 (Ωm=1.00)')

z_pred = np.linspace(0, 2, 100)
ax3.plot(z_pred, [lcdm_fsigma8(z) for z in z_pred], 'b--', lw=1.5, alpha=0.7,
         label='ΛCDM (Ωm=0.31, σ8=0.81)')

# BOSS
for i, (z, fs8, err) in enumerate(boss_data):
    ax3.errorbar(z, fs8, yerr=err, fmt='ro', ms=8, capsize=4,
                 label='BOSS/eBOSS' if i==0 else '')

# Others
markers = ['s', '^', 'v', 'D', 'p']
colors = ['green', 'purple', 'orange', 'cyan', 'magenta']
for i, (z, fs8, err, name) in enumerate(other_data):
    ax3.errorbar(z, fs8, yerr=err, fmt=markers[i], color=colors[i],
                 ms=6, capsize=3, alpha=0.7, label=name)

ax3.set_xlabel('Redshift z')
ax3.set_ylabel('fσ₈(z)')
ax3.set_title('fσ₈(z) — Janus vs Observations')
ax3.set_xlim(0, 1.6)
ax3.set_ylim(0.2, 0.8)
ax3.legend(fontsize=8, loc='upper right', ncol=2)
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

# S8
ax4 = axes[1, 1]
S8_janus = sigma8_arr * np.sqrt(OMEGA_M / 0.3)
ax4.plot(z_arr, S8_janus, 'k-', lw=2, label='Janus μ=19')
ax4.axhspan(0.74, 0.78, alpha=0.3, color='green', label='KiDS-1000')
ax4.axhspan(0.75, 0.79, alpha=0.3, color='orange', label='DES-Y3')
ax4.axhline(y=0.83, color='blue', ls='--', alpha=0.5, label='Planck')
ax4.set_xlabel('Redshift z')
ax4.set_ylabel('S₈ = σ₈√(Ωm/0.3)')
ax4.set_title('S₈ Comparison')
ax4.set_xlim(0, 2)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.invert_xaxis()

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "figure_fsigma8.png"), dpi=200)
print(f"Saved: {os.path.join(BASE_DIR, 'figure_fsigma8.png')}", flush=True)

# Final summary
print("\n" + "="*70)
print("FINAL RESULTS — μ=19 CANONICAL JANUS")
print("="*70)
print(f"  σ8(z=0) = {sigma8_arr[-1]:.4f}")
print(f"  f(z=0)  = {f_arr[-1]:.4f}")
print(f"  fσ8(z=0) = {fsigma8_arr[-1]:.4f}")
print(f"  S8 = {sigma8_arr[-1] * np.sqrt(OMEGA_M/0.3):.4f}")
print()
janus_z051 = np.interp(0.51, z_arr[::-1], fsigma8_arr[::-1])
print(f"  BOSS z=0.51: fσ8 = 0.458 ± 0.038")
print(f"  Janus z=0.51: fσ8 = {janus_z051:.4f}")
print(f"  Tension: {abs(janus_z051 - 0.458)/0.038:.1f}σ")
print("="*70)
