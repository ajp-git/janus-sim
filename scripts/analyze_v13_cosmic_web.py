#!/usr/bin/env python3
"""
JANUS V13 COSMIC WEB — COMPREHENSIVE ANALYSIS
Analyzes snapshots from janus_v13_cosmic_web simulation.

Steps:
1. Load snapshots
2. Compute density fields
3. Segregation metrics
4. Domain analysis
5. Interface thickness
6. Power spectrum
7. Correlation function
8. Cosmic web classification
9. Visualization
10. Summary report
"""

import numpy as np
import struct
import os
import glob
from scipy import ndimage
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SNAP_DIR = "/mnt/T2/janus-sim/output/janus_v13_cosmic_web/snapshots"
OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_v13_cosmic_web/analysis"
TIME_SERIES = "/mnt/T2/janus-sim/output/janus_v13_cosmic_web/time_series.csv"

L_BOX = 200.0  # Mpc
N_GRID = 128   # Grid resolution for analysis (128 for speed, 256 for quality)
ETA = 1.045

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD SNAPSHOTS
# ============================================================
def load_snapshot(path):
    """Load binary snapshot: n(u64) + n*(x,y,z,vx,vy,vz,sign)(f32)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 7 * 4), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    vel = data[:, 3:6]
    signs = data[:, 6]
    return pos, vel, signs, n

def get_snapshots():
    """Get sorted list of snapshot files with step numbers"""
    files = sorted(glob.glob(f"{SNAP_DIR}/snap_*.bin"))
    snapshots = []
    for f in files:
        step = int(os.path.basename(f).split('_')[1].split('.')[0])
        snapshots.append((step, f))
    return snapshots

def load_time_series():
    """Load time series CSV"""
    data = np.genfromtxt(TIME_SERIES, delimiter=',', names=True, dtype=None, encoding='utf-8')
    return data

print("=" * 70)
print("JANUS V13 COSMIC WEB — COMPREHENSIVE ANALYSIS")
print("=" * 70)
print(f"Output: {OUTPUT_DIR}")
print()

# Get available snapshots
snapshots = get_snapshots()
print(f"STEP 1: Found {len(snapshots)} snapshots")
for step, path in snapshots[:5]:
    print(f"  Step {step}: {os.path.basename(path)}")
if len(snapshots) > 5:
    print(f"  ... and {len(snapshots)-5} more")
print()

# Load time series for redshift mapping
ts = load_time_series()
step_to_z = {int(row['step']): float(row['z']) for row in ts}

# ============================================================
# STEP 2: DENSITY FIELD COMPUTATION
# ============================================================
def compute_density_fields(pos, signs, n_grid=N_GRID):
    """Deposit particles on grid, compute rho_plus, rho_minus, polarization"""
    cell = L_BOX / n_grid

    # Shift positions to [0, L_BOX]
    pos_shifted = pos + L_BOX / 2.0

    # Grid indices
    ix = np.clip((pos_shifted[:, 0] / cell).astype(int), 0, n_grid - 1)
    iy = np.clip((pos_shifted[:, 1] / cell).astype(int), 0, n_grid - 1)
    iz = np.clip((pos_shifted[:, 2] / cell).astype(int), 0, n_grid - 1)

    # Separate by sign
    plus_mask = signs > 0
    minus_mask = signs < 0

    # Deposit
    rho_plus = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    rho_minus = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    np.add.at(rho_plus, (ix[plus_mask], iy[plus_mask], iz[plus_mask]), 1)
    np.add.at(rho_minus, (ix[minus_mask], iy[minus_mask], iz[minus_mask]), 1)

    # Normalize to density
    vol_cell = cell ** 3
    rho_plus /= vol_cell
    rho_minus /= vol_cell

    rho_total = rho_plus + rho_minus

    # Polarization field
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(rho_total > 0, (rho_plus - rho_minus) / rho_total, 0.0)

    return rho_plus, rho_minus, rho_total, P

# ============================================================
# STEP 3: SEGREGATION METRICS
# ============================================================
def compute_segregation_metrics(P):
    """Compute sigma_P and segregation fraction"""
    sigma_P = np.std(P)
    seg_frac = np.mean(np.abs(P) > 0.5)
    return sigma_P, seg_frac

# ============================================================
# STEP 4: DOMAIN ANALYSIS
# ============================================================
def analyze_domains(P, threshold=0.5):
    """Segment domains and compute statistics (optimized)"""
    cell = L_BOX / P.shape[0]

    # Check if there's significant polarization first
    seg_frac = np.mean(np.abs(P) > threshold)
    if seg_frac < 0.01:
        # Too early, no real domains yet
        return 0, 0.0, np.array([])

    # Label connected regions (positive domains)
    pos_domains = P > threshold
    labels_pos, n_pos = label(pos_domains)

    # Label connected regions (negative domains)
    neg_domains = P < -threshold
    labels_neg, n_neg = label(neg_domains)

    n_domains = n_pos + n_neg

    # Skip volume computation if too many domains (noise)
    if n_domains > 10000:
        return n_domains, 0.0, np.array([])

    # Compute domain volumes using bincount (faster than loop)
    volumes = []
    if n_pos > 0:
        sizes_pos = np.bincount(labels_pos.ravel())[1:]  # skip 0 (background)
        volumes.extend(sizes_pos * cell**3)
    if n_neg > 0:
        sizes_neg = np.bincount(labels_neg.ravel())[1:]
        volumes.extend(sizes_neg * cell**3)

    if len(volumes) > 0:
        volumes = np.array(volumes)
        # Estimate diameter from volume (spherical approximation)
        diameters = 2 * (3 * volumes / (4 * np.pi)) ** (1/3)
        median_diameter = np.median(diameters)
    else:
        median_diameter = 0.0
        volumes = np.array([])

    return n_domains, median_diameter, volumes

# ============================================================
# STEP 5: INTERFACE THICKNESS
# ============================================================
def compute_interface_thickness(P, sigma_P):
    """Compute L_J = sigma_P / sqrt(<|grad P|^2>)"""
    cell = L_BOX / P.shape[0]

    # Compute gradient (periodic)
    grad_x = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)) / (2 * cell)
    grad_y = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) / (2 * cell)
    grad_z = (np.roll(P, -1, axis=2) - np.roll(P, 1, axis=2)) / (2 * cell)

    grad_sq = grad_x**2 + grad_y**2 + grad_z**2
    mean_grad_sq = np.mean(grad_sq)

    if mean_grad_sq > 0:
        L_J = sigma_P / np.sqrt(mean_grad_sq)
    else:
        L_J = 0.0

    return L_J

# ============================================================
# STEP 6: POWER SPECTRUM
# ============================================================
def compute_power_spectrum(rho, n_bins=50):
    """Compute isotropic power spectrum P(k)"""
    n_grid = rho.shape[0]
    cell = L_BOX / n_grid

    # Overdensity
    rho_mean = np.mean(rho)
    if rho_mean > 0:
        delta = (rho - rho_mean) / rho_mean
    else:
        delta = rho

    # FFT
    delta_k = np.fft.fftn(delta)
    Pk_3d = np.abs(delta_k) ** 2 / n_grid**6 * L_BOX**3

    # k values
    kx = np.fft.fftfreq(n_grid, d=cell) * 2 * np.pi
    ky = np.fft.fftfreq(n_grid, d=cell) * 2 * np.pi
    kz = np.fft.fftfreq(n_grid, d=cell) * 2 * np.pi
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Bin by k
    k_min = 2 * np.pi / L_BOX
    k_max = np.pi * n_grid / L_BOX
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])

    Pk = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            Pk[i] = np.mean(Pk_3d[mask])
            counts[i] = np.sum(mask)

    # Filter out empty bins
    valid = counts > 0
    return k_centers[valid], Pk[valid]

# ============================================================
# STEP 7: CORRELATION FUNCTION
# ============================================================
def compute_correlation_function(rho, n_bins=40):
    """Compute two-point correlation xi(r)"""
    n_grid = rho.shape[0]
    cell = L_BOX / n_grid

    # Overdensity
    rho_mean = np.mean(rho)
    if rho_mean > 0:
        delta = (rho - rho_mean) / rho_mean
    else:
        return np.array([]), np.array([])

    # FFT method: xi(r) = FT^-1(|delta_k|^2)
    delta_k = np.fft.fftn(delta)
    Pk_3d = np.abs(delta_k) ** 2
    xi_3d = np.real(np.fft.ifftn(Pk_3d))

    # Create r grid
    rx = np.fft.fftfreq(n_grid, d=1.0/L_BOX)
    ry = np.fft.fftfreq(n_grid, d=1.0/L_BOX)
    rz = np.fft.fftfreq(n_grid, d=1.0/L_BOX)
    rx3d, ry3d, rz3d = np.meshgrid(rx, ry, rz, indexing='ij')
    r_mag = np.sqrt(rx3d**2 + ry3d**2 + rz3d**2)

    # Bin by r
    r_max = L_BOX / 2
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    xi_r = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r_mag >= r_bins[i]) & (r_mag < r_bins[i+1])
        if np.any(mask):
            xi_r[i] = np.mean(xi_3d[mask])
            counts[i] = np.sum(mask)

    valid = counts > 0
    return r_centers[valid], xi_r[valid]

# ============================================================
# STEP 8: COSMIC WEB CLASSIFICATION
# ============================================================
def classify_cosmic_web(rho, sigma_smooth=2.0):
    """Classify cells using simplified density-based method (fast)"""
    n_grid = rho.shape[0]

    # Smooth density field
    rho_smooth = ndimage.gaussian_filter(rho, sigma=sigma_smooth)

    # Compute Laplacian (trace of Hessian) - faster than full Hessian
    laplacian = ndimage.laplace(rho_smooth)

    # Simplified classification based on density and Laplacian
    classification = np.zeros((n_grid, n_grid, n_grid), dtype=np.int8)

    rho_mean = np.mean(rho_smooth)
    rho_std = np.std(rho_smooth)

    high_density = rho_smooth > rho_mean + 0.5 * rho_std
    low_density = rho_smooth < rho_mean - 0.5 * rho_std

    # 0=void, 1=sheet, 2=filament, 3=node
    classification[low_density] = 0  # void
    classification[~low_density & ~high_density & (laplacian < 0)] = 1  # sheet
    classification[~low_density & ~high_density & (laplacian >= 0)] = 2  # filament
    classification[high_density] = 3  # node

    # Volume fractions
    void_frac = np.mean(classification == 0)
    sheet_frac = np.mean(classification == 1)
    filament_frac = np.mean(classification == 2)
    node_frac = np.mean(classification == 3)

    return classification, (void_frac, sheet_frac, filament_frac, node_frac)

# ============================================================
# MAIN ANALYSIS LOOP
# ============================================================
print("STEP 2-8: Processing snapshots...")
print()

# Storage for results
results = {
    'step': [],
    'z': [],
    'sigma_P': [],
    'seg_frac': [],
    'n_domains': [],
    'xi_diameter': [],
    'L_J': [],
    'void_frac': [],
    'sheet_frac': [],
    'filament_frac': [],
    'node_frac': [],
}

# Store power spectra and correlation functions for selected redshifts
pk_data = {}
xi_data = {}
selected_steps = []

# Process each snapshot
for i, (step, path) in enumerate(snapshots):
    z = step_to_z.get(step, 5.0 - step * 0.001)  # fallback estimate

    print(f"  [{i+1}/{len(snapshots)}] Step {step}, z={z:.2f}...", end=" ", flush=True)

    # Load
    pos, vel, signs, n = load_snapshot(path)

    # Compute density fields
    rho_plus, rho_minus, rho_total, P = compute_density_fields(pos, signs)

    # Segregation metrics
    sigma_P, seg_frac = compute_segregation_metrics(P)

    # Domain analysis
    n_domains, xi_diameter, volumes = analyze_domains(P)

    # Interface thickness
    L_J = compute_interface_thickness(P, sigma_P)

    # Cosmic web classification
    classification, (void_f, sheet_f, fil_f, node_f) = classify_cosmic_web(rho_total)

    # Store results
    results['step'].append(step)
    results['z'].append(z)
    results['sigma_P'].append(sigma_P)
    results['seg_frac'].append(seg_frac)
    results['n_domains'].append(n_domains)
    results['xi_diameter'].append(xi_diameter)
    results['L_J'].append(L_J)
    results['void_frac'].append(void_f)
    results['sheet_frac'].append(sheet_f)
    results['filament_frac'].append(fil_f)
    results['node_frac'].append(node_f)

    # Compute power spectrum and correlation for selected redshifts
    if z > 3.5 or (1.8 < z < 2.2) or (0.8 < z < 1.2) or z < 0.5:
        k, Pk = compute_power_spectrum(rho_total)
        r, xi_r = compute_correlation_function(rho_total)
        pk_data[step] = (z, k, Pk)
        xi_data[step] = (z, r, xi_r)
        selected_steps.append(step)

    print(f"σ_P={sigma_P:.3f}, Seg={seg_frac:.3f}, ξ={xi_diameter:.1f}Mpc, L_J={L_J:.1f}Mpc")

    # Save density slice for visualization
    if step in [0, snapshots[len(snapshots)//4][0], snapshots[len(snapshots)//2][0], snapshots[-1][0]]:
        np.savez(f"{OUTPUT_DIR}/fields_step_{step:06d}.npz",
                 rho_total=rho_total[:, :, N_GRID//2],
                 P=P[:, :, N_GRID//2],
                 classification=classification[:, :, N_GRID//2],
                 z=z)

print()

# Convert to arrays
for key in results:
    results[key] = np.array(results[key])

# ============================================================
# STEP 9: VISUALIZATION
# ============================================================
print("STEP 9: Generating visualizations...")

# 9.1 Evolution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# σ_P vs z
ax = axes[0, 0]
ax.plot(results['z'], results['sigma_P'], 'b-o', markersize=4)
ax.set_xlabel('Redshift z')
ax.set_ylabel('σ_P')
ax.set_title('Polarization Dispersion')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Seg vs z
ax = axes[0, 1]
ax.plot(results['z'], results['seg_frac'], 'r-o', markersize=4)
ax.set_xlabel('Redshift z')
ax.set_ylabel('Seg (|P|>0.5)')
ax.set_title('Segregation Fraction')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# ξ vs z
ax = axes[0, 2]
ax.plot(results['z'], results['xi_diameter'], 'g-o', markersize=4)
ax.set_xlabel('Redshift z')
ax.set_ylabel('ξ (Mpc)')
ax.set_title('Domain Diameter')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# L_J vs z
ax = axes[1, 0]
ax.plot(results['z'], results['L_J'], 'm-o', markersize=4)
ax.set_xlabel('Redshift z')
ax.set_ylabel('L_J (Mpc)')
ax.set_title('Interface Thickness')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Cosmic web fractions vs z
ax = axes[1, 1]
ax.fill_between(results['z'], 0, results['void_frac'], alpha=0.7, label='Void')
ax.fill_between(results['z'], results['void_frac'],
                results['void_frac'] + results['sheet_frac'], alpha=0.7, label='Sheet')
ax.fill_between(results['z'], results['void_frac'] + results['sheet_frac'],
                results['void_frac'] + results['sheet_frac'] + results['filament_frac'],
                alpha=0.7, label='Filament')
ax.fill_between(results['z'], results['void_frac'] + results['sheet_frac'] + results['filament_frac'],
                1.0, alpha=0.7, label='Node')
ax.set_xlabel('Redshift z')
ax.set_ylabel('Volume Fraction')
ax.set_title('Cosmic Web Classification')
ax.invert_xaxis()
ax.legend(loc='upper right')
ax.set_ylim(0, 1)

# Number of domains vs z
ax = axes[1, 2]
ax.plot(results['z'], results['n_domains'], 'k-o', markersize=4)
ax.set_xlabel('Redshift z')
ax.set_ylabel('N_domains')
ax.set_title('Number of Domains')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/evolution_metrics.png", dpi=150)
plt.close()
print(f"  Saved: evolution_metrics.png")

# 9.2 Power spectrum
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(pk_data)))
for i, step in enumerate(sorted(pk_data.keys())):
    z, k, Pk = pk_data[step]
    ax.loglog(k, Pk, color=colors[i], label=f'z={z:.1f}', linewidth=1.5)
ax.set_xlabel('k (Mpc⁻¹)')
ax.set_ylabel('P(k) (Mpc³)')
ax.set_title('Density Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/power_spectrum.png", dpi=150)
plt.close()
print(f"  Saved: power_spectrum.png")

# 9.3 Correlation function
fig, ax = plt.subplots(figsize=(10, 7))
for i, step in enumerate(sorted(xi_data.keys())):
    z, r, xi_r = xi_data[step]
    valid = xi_r > 0
    if np.any(valid):
        ax.semilogy(r[valid], xi_r[valid], color=colors[i], label=f'z={z:.1f}', linewidth=1.5)
ax.set_xlabel('r (Mpc)')
ax.set_ylabel('ξ(r)')
ax.set_title('Two-Point Correlation Function')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, L_BOX/2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_function.png", dpi=150)
plt.close()
print(f"  Saved: correlation_function.png")

# 9.4 Density and polarization slices
print("  Generating slice visualizations...")
slice_files = sorted(glob.glob(f"{OUTPUT_DIR}/fields_step_*.npz"))
if slice_files:
    n_slices = len(slice_files)
    fig, axes = plt.subplots(2, n_slices, figsize=(5*n_slices, 10))

    for i, f in enumerate(slice_files):
        data = np.load(f)
        z = float(data['z'])
        rho = data['rho_total']
        P = data['P']

        # Density
        ax = axes[0, i] if n_slices > 1 else axes[0]
        im = ax.imshow(np.log10(rho.T + 1), origin='lower', cmap='inferno',
                       extent=[0, L_BOX, 0, L_BOX])
        ax.set_title(f'z={z:.2f} - Density')
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('y (Mpc)')
        plt.colorbar(im, ax=ax, label='log₁₀(ρ+1)')

        # Polarization
        ax = axes[1, i] if n_slices > 1 else axes[1]
        im = ax.imshow(P.T, origin='lower', cmap='RdBu', vmin=-1, vmax=1,
                       extent=[0, L_BOX, 0, L_BOX])
        ax.set_title(f'z={z:.2f} - Polarization')
        ax.set_xlabel('x (Mpc)')
        ax.set_ylabel('y (Mpc)')
        plt.colorbar(im, ax=ax, label='P')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/slices.png", dpi=150)
    plt.close()
    print(f"  Saved: slices.png")

print()

# ============================================================
# STEP 10: SUMMARY REPORT
# ============================================================
print("STEP 10: Generating summary report...")

report = f"""
{'='*70}
JANUS V13 COSMIC WEB — ANALYSIS REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SIMULATION PARAMETERS
---------------------
Box size: L = {L_BOX} Mpc
Grid resolution: {N_GRID}³
η = {ETA}
Snapshots analyzed: {len(snapshots)}
Redshift range: z = {results['z'].max():.2f} → {results['z'].min():.2f}

SEGREGATION EVOLUTION
---------------------
σ_P:
  Initial (z={results['z'][0]:.1f}): {results['sigma_P'][0]:.4f}
  Final (z={results['z'][-1]:.1f}): {results['sigma_P'][-1]:.4f}
  Peak: {results['sigma_P'].max():.4f} at z={results['z'][np.argmax(results['sigma_P'])]:.2f}

Segregation fraction (|P|>0.5):
  Initial: {results['seg_frac'][0]:.4f}
  Final: {results['seg_frac'][-1]:.4f}
  Peak: {results['seg_frac'].max():.4f} at z={results['z'][np.argmax(results['seg_frac'])]:.2f}

DOMAIN STATISTICS
-----------------
Domain diameter ξ:
  Initial: {results['xi_diameter'][0]:.1f} Mpc
  Final: {results['xi_diameter'][-1]:.1f} Mpc
  Maximum: {results['xi_diameter'].max():.1f} Mpc

Number of domains:
  Initial: {results['n_domains'][0]}
  Final: {results['n_domains'][-1]}
  Maximum: {results['n_domains'].max()}

INTERFACE THICKNESS
-------------------
L_J:
  Initial: {results['L_J'][0]:.2f} Mpc
  Final: {results['L_J'][-1]:.2f} Mpc
  Mean: {results['L_J'].mean():.2f} Mpc

COSMIC WEB CLASSIFICATION (Final state)
---------------------------------------
Void fraction: {results['void_frac'][-1]*100:.1f}%
Sheet fraction: {results['sheet_frac'][-1]*100:.1f}%
Filament fraction: {results['filament_frac'][-1]*100:.1f}%
Node fraction: {results['node_frac'][-1]*100:.1f}%

Expected ΛCDM values:
  Void: ~70-80%
  Filaments: ~10-15%

SCIENTIFIC TESTS (from Research Plan)
-------------------------------------
1. Segregation plateau: σ_P ≈ constant?
   → σ_P varies: {results['sigma_P'].min():.3f} - {results['sigma_P'].max():.3f}

2. Domain size: ξ ≈ 30-50 Mpc?
   → ξ_max = {results['xi_diameter'].max():.1f} Mpc

3. Interface thickness: L_J ≈ 5-10 Mpc?
   → L_J_mean = {results['L_J'].mean():.1f} Mpc

OUTPUT FILES
------------
- evolution_metrics.png: Time evolution of all metrics
- power_spectrum.png: P(k) at selected redshifts
- correlation_function.png: ξ(r) at selected redshifts
- slices.png: Density and polarization cross-sections
- fields_step_*.npz: Raw field data for further analysis

{'='*70}
END OF REPORT
{'='*70}
"""

print(report)

# Save report
with open(f"{OUTPUT_DIR}/analysis_report.txt", 'w') as f:
    f.write(report)
print(f"Report saved: {OUTPUT_DIR}/analysis_report.txt")

# Save numerical results
np.savez(f"{OUTPUT_DIR}/results.npz", **results)
print(f"Results saved: {OUTPUT_DIR}/results.npz")
