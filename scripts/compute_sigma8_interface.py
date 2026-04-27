#!/usr/bin/env python3
"""
σ8 at interfaces: measure clustering where m+ and m- segregate

Hypothesis: Janus galaxies form at interfaces between m+ and m- voids
σ8_interface could be much higher than σ8_global

Method:
1. Compute δ+ and δ- density fields on 256³ grid
2. Compute local correlation using sliding window
3. Identify "interface" cells where Corr(δ+,δ-) is most negative
4. Measure σ8 in interface regions only
"""

import numpy as np
import struct
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, generic_filter
from scipy.interpolate import UnivariateSpline

sys.stdout.reconfigure(line_buffering=True)

# Parameters
BASE_DIR = "/mnt/T2/janus-sim/output/scan_mu_evolution/mu19"
BOX_SIZE = 1000.0
N_GRID = 256
h = 0.7
R_SMOOTH = 8.0 / h  # 11.43 Mpc

# Interface detection parameters
INTERFACE_WINDOW = 5  # cells for local correlation
INTERFACE_PERCENTILE = 10  # top 10% most anti-correlated cells

print("="*70)
print("σ8 at INTERFACES — μ=19 CANONICAL JANUS")
print("="*70)
print(f"  Grid: {N_GRID}³, R_smooth = {R_SMOOTH:.1f} Mpc")
print(f"  Interface detection: {INTERFACE_WINDOW}³ window")
print(f"  Interface selection: top {INTERFACE_PERCENTILE}% anti-correlated")
print("="*70)

def read_snapshot(filename):
    """Read binary snapshot with 16-byte header"""
    with open(filename, 'rb') as f:
        header = f.read(16)
        n_particles = struct.unpack('<I', header[:4])[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)

    data = data.reshape(n_particles, 25)

    positions = np.zeros((n_particles, 3), dtype=np.float32)
    for i in range(3):
        positions[:, i] = np.frombuffer(
            data[:, i*4:(i+1)*4].tobytes(), dtype=np.float32
        )

    signs = data[:, 24].view(np.int8)
    return positions, signs

def compute_density_fields(positions, signs, box_size, n_grid):
    """Compute both m+ and m- density fields using NGP"""

    pos_plus = positions[signs > 0] % box_size
    pos_minus = positions[signs < 0] % box_size

    bins = np.linspace(0, box_size, n_grid + 1)

    density_plus, _ = np.histogramdd(pos_plus, bins=[bins, bins, bins])
    density_minus, _ = np.histogramdd(pos_minus, bins=[bins, bins, bins])

    return density_plus.astype(np.float64), density_minus.astype(np.float64)

def compute_delta(density):
    """Compute density contrast δ = (ρ - ρ̄)/ρ̄"""
    rho_mean = density.mean()
    if rho_mean < 1e-10:
        return np.zeros_like(density)
    return (density - rho_mean) / rho_mean

def tophat_filter(delta, box_size, R):
    """Top-hat filter in Fourier space"""
    n = delta.shape[0]
    delta_k = np.fft.rfftn(delta)

    kx = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    ky = np.fft.fftfreq(n, d=box_size/n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size/n) * 2 * np.pi

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    kR = k * R
    W = np.ones_like(kR)
    mask = kR > 1e-10
    W[mask] = 3 * (np.sin(kR[mask]) - kR[mask] * np.cos(kR[mask])) / (kR[mask]**3)

    delta_k *= W
    return np.fft.irfftn(delta_k, s=(n, n, n))

def compute_local_correlation(delta_plus, delta_minus, window_size):
    """
    Compute local correlation between δ+ and δ- using sliding window
    Corr = <δ+ δ-> / (σ+ σ-)
    """
    # Local means
    mean_plus = uniform_filter(delta_plus, size=window_size, mode='wrap')
    mean_minus = uniform_filter(delta_minus, size=window_size, mode='wrap')

    # Local variances
    var_plus = uniform_filter(delta_plus**2, size=window_size, mode='wrap') - mean_plus**2
    var_minus = uniform_filter(delta_minus**2, size=window_size, mode='wrap') - mean_minus**2

    # Local covariance
    cov = uniform_filter(delta_plus * delta_minus, size=window_size, mode='wrap') - mean_plus * mean_minus

    # Local correlation
    denom = np.sqrt(np.maximum(var_plus, 1e-20) * np.maximum(var_minus, 1e-20))
    corr = cov / denom

    return corr

def identify_interface_cells(local_corr, percentile):
    """
    Identify interface cells where local correlation is most negative
    Returns boolean mask
    """
    threshold = np.percentile(local_corr, percentile)
    return local_corr < threshold

def compute_sigma8_in_regions(delta_smoothed, mask):
    """Compute σ8 only in selected regions"""
    if mask.sum() == 0:
        return 0.0
    return np.std(delta_smoothed[mask])

# Get snapshots
snapshot_dir = os.path.join(BASE_DIR, "snapshots")
snapshots = sorted(glob(os.path.join(snapshot_dir, "snap_*.bin")))

# Read time series
time_data = np.genfromtxt(os.path.join(BASE_DIR, "time_series.csv"),
                          delimiter=',', skip_header=1)
step_info = {}
for row in time_data:
    step_info[int(row[0])] = {'t_gyr': row[1], 'z': row[2], 'a': row[3]}

# Process selected snapshots (every 10th for speed)
sample_indices = list(range(0, len(snapshots), 10))
if len(snapshots)-1 not in sample_indices:
    sample_indices.append(len(snapshots)-1)

print(f"\nProcessing {len(sample_indices)} snapshots...\n")

results = []

for idx in sample_indices:
    snap_file = snapshots[idx]
    step = int(os.path.basename(snap_file).split('_')[1].split('.')[0])

    if step in step_info:
        info = step_info[step]
    else:
        closest = min(step_info.keys(), key=lambda x: abs(x-step))
        info = step_info[closest]

    # Read snapshot
    positions, signs = read_snapshot(snap_file)

    # Compute density fields
    rho_plus, rho_minus = compute_density_fields(positions, signs, BOX_SIZE, N_GRID)

    # Compute density contrasts
    delta_plus = compute_delta(rho_plus)
    delta_minus = compute_delta(rho_minus)

    # Smooth both fields
    delta_plus_smooth = tophat_filter(delta_plus, BOX_SIZE, R_SMOOTH)
    delta_minus_smooth = tophat_filter(delta_minus, BOX_SIZE, R_SMOOTH)

    # Compute global σ8
    sigma8_global = np.std(delta_plus_smooth)

    # Compute local correlation
    local_corr = compute_local_correlation(delta_plus_smooth, delta_minus_smooth, INTERFACE_WINDOW)

    # Identify interface cells (most negative correlation)
    interface_mask = identify_interface_cells(local_corr, INTERFACE_PERCENTILE)
    n_interface = interface_mask.sum()

    # Compute σ8 at interfaces
    sigma8_interface = compute_sigma8_in_regions(delta_plus_smooth, interface_mask)

    # Also compute σ8 in "bulk" regions (not interface)
    bulk_mask = ~interface_mask
    sigma8_bulk = compute_sigma8_in_regions(delta_plus_smooth, bulk_mask)

    # Mean local correlation at interface
    mean_corr_interface = local_corr[interface_mask].mean()

    # Mean δ+ at interface (density enhancement?)
    mean_delta_interface = delta_plus_smooth[interface_mask].mean()

    results.append({
        'step': step,
        'z': info['z'],
        't_Gyr': info['t_gyr'],
        'a': info['a'],
        'sigma8_global': sigma8_global,
        'sigma8_interface': sigma8_interface,
        'sigma8_bulk': sigma8_bulk,
        'ratio_interface_global': sigma8_interface / max(sigma8_global, 1e-10),
        'n_interface_cells': n_interface,
        'mean_corr_interface': mean_corr_interface,
        'mean_delta_interface': mean_delta_interface
    })

    print(f"  z={info['z']:.2f}: σ8_global={sigma8_global:.4f}, "
          f"σ8_interface={sigma8_interface:.4f} "
          f"(ratio={sigma8_interface/max(sigma8_global,1e-10):.2f}x), "
          f"<δ+>_int={mean_delta_interface:.3f}")

# Convert to arrays
results = sorted(results, key=lambda x: x['step'])
z_arr = np.array([r['z'] for r in results])
sigma8_global = np.array([r['sigma8_global'] for r in results])
sigma8_interface = np.array([r['sigma8_interface'] for r in results])
sigma8_bulk = np.array([r['sigma8_bulk'] for r in results])
ratio = np.array([r['ratio_interface_global'] for r in results])
mean_delta_int = np.array([r['mean_delta_interface'] for r in results])

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\n  At z=0:")
print(f"    σ8_global    = {sigma8_global[-1]:.4f}")
print(f"    σ8_interface = {sigma8_interface[-1]:.4f}")
print(f"    σ8_bulk      = {sigma8_bulk[-1]:.4f}")
print(f"    Ratio (interface/global) = {ratio[-1]:.2f}x")
print(f"    <δ+> at interface = {mean_delta_int[-1]:.3f}")
print()
print(f"  At z=4:")
print(f"    σ8_global    = {sigma8_global[0]:.4f}")
print(f"    σ8_interface = {sigma8_interface[0]:.4f}")
print(f"    Ratio = {ratio[0]:.2f}x")
print("="*70)

# Save CSV
csv_file = os.path.join(BASE_DIR, "sigma8_interface.csv")
with open(csv_file, 'w') as f:
    f.write("step,z,t_Gyr,sigma8_global,sigma8_interface,sigma8_bulk,ratio,mean_delta_interface\n")
    for r in results:
        f.write(f"{r['step']},{r['z']:.4f},{r['t_Gyr']:.4f},"
                f"{r['sigma8_global']:.6f},{r['sigma8_interface']:.6f},"
                f"{r['sigma8_bulk']:.6f},{r['ratio_interface_global']:.4f},"
                f"{r['mean_delta_interface']:.6f}\n")
print(f"\nSaved: {csv_file}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: σ8 comparison
ax1 = axes[0, 0]
ax1.plot(z_arr, sigma8_global, 'b-', lw=2, label='σ8 global')
ax1.plot(z_arr, sigma8_interface, 'r-', lw=2, label='σ8 interface')
ax1.plot(z_arr, sigma8_bulk, 'g--', lw=1.5, alpha=0.7, label='σ8 bulk')
ax1.axhline(y=0.81, color='gray', ls=':', alpha=0.5, label='ΛCDM σ8')
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('σ8')
ax1.set_title('σ8: Global vs Interface vs Bulk')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# Panel 2: Ratio interface/global
ax2 = axes[0, 1]
ax2.plot(z_arr, ratio, 'k-', lw=2)
ax2.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
ax2.set_xlabel('Redshift z')
ax2.set_ylabel('σ8_interface / σ8_global')
ax2.set_title('Interface Enhancement Ratio')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

# Panel 3: Mean δ+ at interface
ax3 = axes[1, 0]
ax3.plot(z_arr, mean_delta_int, 'r-', lw=2)
ax3.axhline(y=0, color='gray', ls='--', alpha=0.5)
ax3.set_xlabel('Redshift z')
ax3.set_ylabel('<δ+> at interface')
ax3.set_title('Mean Overdensity at Interfaces')
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

# Panel 4: Growth rate comparison
ax4 = axes[1, 1]
# Compute growth rates
a_arr = np.array([r['a'] for r in results])
ln_a = np.log(a_arr)
ln_s8_global = np.log(sigma8_global + 1e-10)
ln_s8_interface = np.log(sigma8_interface + 1e-10)

# Smooth and differentiate
idx = np.argsort(ln_a)
from scipy.interpolate import UnivariateSpline
sp_global = UnivariateSpline(ln_a[idx], ln_s8_global[idx], s=0.1)
sp_interface = UnivariateSpline(ln_a[idx], ln_s8_interface[idx], s=0.1)

f_global = sp_global.derivative()(ln_a)
f_interface = sp_interface.derivative()(ln_a)

# fσ8
fs8_global = f_global * sigma8_global
fs8_interface = f_interface * sigma8_interface

ax4.plot(z_arr, fs8_global, 'b-', lw=2, label='fσ8 global')
ax4.plot(z_arr, fs8_interface, 'r-', lw=2, label='fσ8 interface')

# BOSS data
boss_z = [0.38, 0.51, 0.70]
boss_fs8 = [0.497, 0.458, 0.473]
boss_err = [0.045, 0.038, 0.041]
ax4.errorbar(boss_z, boss_fs8, yerr=boss_err, fmt='ko', ms=8, capsize=4, label='BOSS/eBOSS')

ax4.set_xlabel('Redshift z')
ax4.set_ylabel('fσ8')
ax4.set_title('fσ8: Global vs Interface')
ax4.set_xlim(0, 1.5)
ax4.set_ylim(-0.1, 0.7)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.invert_xaxis()

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "figure_sigma8_interface.png"), dpi=200)
print(f"Saved: {os.path.join(BASE_DIR, 'figure_sigma8_interface.png')}")

print("\n" + "="*70)
print("PHYSICAL INTERPRETATION")
print("="*70)
print("""
In Janus cosmology, galaxies may form preferentially at interfaces
between m+ regions and m- voids. If σ8_interface > σ8_global,
this would indicate enhanced clustering at these boundaries.

The ratio σ8_interface/σ8_global measures this enhancement.
""")
print("="*70)
