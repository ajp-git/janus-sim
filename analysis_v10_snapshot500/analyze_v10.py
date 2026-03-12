#!/usr/bin/env python3
"""
V10 Snapshot Analysis - Full Scientific Analysis
Snapshot: step 500, N ≈ 20M particles, L = 200 Mpc
"""

import numpy as np
import struct
import json
import os
from scipy import ndimage
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
SNAPSHOT_PATH = "/mnt/T2/janus-sim/output/janus_v10_highres/snapshots/snap_000500.bin"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500"
L_BOX = 200.0  # Mpc
N_GRID = 256   # Voxel resolution → 200/256 ≈ 0.78 Mpc

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1) LOAD SNAPSHOT
# ============================================================
print("=" * 60)
print("1) LOADING SNAPSHOT")
print("=" * 60)

with open(SNAPSHOT_PATH, 'rb') as f:
    # Header: 8 bytes (u64 step)
    step_bytes = f.read(8)
    step = struct.unpack('<Q', step_bytes)[0]
    print(f"  Step from header: {step}")

    # Rest is particle data: x, y, z, sign as f32
    data = np.frombuffer(f.read(), dtype=np.float32)

n_particles = len(data) // 4
print(f"  N particles: {n_particles:,}")

x = data[0::4].astype(np.float64)
y = data[1::4].astype(np.float64)
z = data[2::4].astype(np.float64)
signs = data[3::4].astype(np.float64)

# Positive and negative masks
pos_mask = signs > 0
neg_mask = signs < 0
n_pos = np.sum(pos_mask)
n_neg = np.sum(neg_mask)
print(f"  N+ = {n_pos:,}, N- = {n_neg:,}")
print(f"  Ratio N-/N+ = {n_neg/n_pos:.4f}")

# ============================================================
# CIC INTERPOLATION
# ============================================================
print("\n  Computing CIC density fields on 256³ grid...")

def cic_deposit(x, y, z, weights, ngrid, box_size):
    """Cloud-in-Cell density deposition"""
    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    cell_size = box_size / ngrid

    # Normalize positions to grid coordinates
    xg = x / cell_size
    yg = y / cell_size
    zg = z / cell_size

    # Integer indices (lower-left corner of cell)
    ix = np.floor(xg).astype(np.int32)
    iy = np.floor(yg).astype(np.int32)
    iz = np.floor(zg).astype(np.int32)

    # Fractional offsets
    dx = xg - ix
    dy = yg - iy
    dz = zg - iz

    # CIC weights for 8 neighboring cells
    for ddx in [0, 1]:
        for ddy in [0, 1]:
            for ddz in [0, 1]:
                wx = (1 - dx) if ddx == 0 else dx
                wy = (1 - dy) if ddy == 0 else dy
                wz = (1 - dz) if ddz == 0 else dz

                w = wx * wy * wz * weights

                iix = (ix + ddx) % ngrid
                iiy = (iy + ddy) % ngrid
                iiz = (iz + ddz) % ngrid

                # Use bincount for fast accumulation
                flat_idx = iix * ngrid * ngrid + iiy * ngrid + iiz
                contrib = np.bincount(flat_idx, weights=w, minlength=ngrid**3)
                density += contrib.reshape((ngrid, ngrid, ngrid))

    return density

# Deposit positive and negative particles
rho_pos = cic_deposit(x[pos_mask], y[pos_mask], z[pos_mask],
                       np.ones(n_pos), N_GRID, L_BOX)
rho_neg = cic_deposit(x[neg_mask], y[neg_mask], z[neg_mask],
                       np.ones(n_neg), N_GRID, L_BOX)

print(f"  ρ+ range: [{rho_pos.min():.1f}, {rho_pos.max():.1f}]")
print(f"  ρ- range: [{rho_neg.min():.1f}, {rho_neg.max():.1f}]")

# ============================================================
# 2) POLARIZATION FIELD
# ============================================================
print("\n" + "=" * 60)
print("2) POLARIZATION FIELD")
print("=" * 60)

rho_total = rho_pos + rho_neg
P = (rho_pos - rho_neg) / (rho_total + 1e-10)

# Density contrast
rho_mean = np.mean(rho_total)
delta = (rho_total - rho_mean) / rho_mean

print(f"  P range: [{P.min():.4f}, {P.max():.4f}]")
print(f"  δ range: [{delta.min():.4f}, {delta.max():.4f}]")

# Save fields
np.save(f"{OUTPUT_DIR}/polarization_field.npy", P.astype(np.float32))
np.save(f"{OUTPUT_DIR}/density_field.npy", rho_total.astype(np.float32))
print(f"  Saved: polarization_field.npy, density_field.npy")

# ============================================================
# 3) CORE JANUS METRICS
# ============================================================
print("\n" + "=" * 60)
print("3) CORE JANUS METRICS")
print("=" * 60)

P_flat = P.flatten()

sigma_P = np.std(P_flat)
mean_P = np.mean(P_flat)
skewness_P = stats.skew(P_flat)
kurtosis_P = stats.kurtosis(P_flat)

# Bimodality coefficient: (skew² + 1) / (kurtosis + 3)
# Values > 0.555 suggest bimodality
bimodality = (skewness_P**2 + 1) / (kurtosis_P + 3)

# Fraction with |P| > 0.5
frac_high_P = np.mean(np.abs(P_flat) > 0.5)

print(f"  σ_P = {sigma_P:.4f}")
print(f"  mean(P) = {mean_P:.4f}")
print(f"  skewness(P) = {skewness_P:.4f}")
print(f"  kurtosis(P) = {kurtosis_P:.4f}")
print(f"  bimodality coeff = {bimodality:.4f}")
print(f"  fraction(|P| > 0.5) = {frac_high_P:.4f}")

polarization_stats = {
    "sigma_P": float(sigma_P),
    "mean_P": float(mean_P),
    "skewness_P": float(skewness_P),
    "kurtosis_P": float(kurtosis_P),
    "bimodality_coefficient": float(bimodality),
    "fraction_high_P": float(frac_high_P),
    "P_min": float(P.min()),
    "P_max": float(P.max())
}

with open(f"{OUTPUT_DIR}/polarization_stats.json", 'w') as f:
    json.dump(polarization_stats, f, indent=2)
print(f"  Saved: polarization_stats.json")

# ============================================================
# 4) DENSITY-POLARIZATION COUPLING
# ============================================================
print("\n" + "=" * 60)
print("4) DENSITY-POLARIZATION COUPLING")
print("=" * 60)

delta_flat = delta.flatten()
absP_flat = np.abs(P_flat)

corr_P_delta = np.corrcoef(P_flat, delta_flat)[0, 1]
corr_absP_delta = np.corrcoef(absP_flat, delta_flat)[0, 1]

print(f"  corr(P, δ) = {corr_P_delta:.4f}")
print(f"  corr(|P|, δ) = {corr_absP_delta:.4f}")

# <|P|> vs density percentile
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
delta_thresholds = np.percentile(delta_flat, percentiles)
mean_absP_by_percentile = []

for i, pctl in enumerate(percentiles):
    if i == 0:
        mask = delta_flat <= delta_thresholds[i]
    else:
        mask = (delta_flat > delta_thresholds[i-1]) & (delta_flat <= delta_thresholds[i])

    if np.sum(mask) > 0:
        mean_absP_by_percentile.append(np.mean(absP_flat[mask]))
    else:
        mean_absP_by_percentile.append(np.nan)

# Save correlation data
with open(f"{OUTPUT_DIR}/polarization_density_correlation.csv", 'w') as f:
    f.write("density_percentile,delta_threshold,mean_absP\n")
    for i, pctl in enumerate(percentiles):
        f.write(f"{pctl},{delta_thresholds[i]:.4f},{mean_absP_by_percentile[i]:.4f}\n")
    f.write(f"\n# corr(P, delta) = {corr_P_delta:.6f}\n")
    f.write(f"# corr(|P|, delta) = {corr_absP_delta:.6f}\n")

print(f"  Saved: polarization_density_correlation.csv")

# ============================================================
# 5) DOMAIN SIZE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5) DOMAIN SIZE ANALYSIS")
print("=" * 60)

# Detect connected regions with |P| > 0.5
high_P_mask = np.abs(P) > 0.5
labeled, n_domains = ndimage.label(high_P_mask)

print(f"  Number of domains (|P| > 0.5): {n_domains}")

# Compute domain volumes
domain_volumes = []
for i in range(1, n_domains + 1):
    vol = np.sum(labeled == i)
    domain_volumes.append(vol)

domain_volumes = np.array(domain_volumes)

# Convert to physical units
cell_size = L_BOX / N_GRID  # Mpc per cell
cell_volume = cell_size**3  # Mpc³ per cell

domain_volumes_mpc3 = domain_volumes * cell_volume

# Equivalent sphere diameter: V = (4/3)πr³ → D = 2 * (3V / 4π)^(1/3)
domain_diameters = 2 * (3 * domain_volumes_mpc3 / (4 * np.pi))**(1/3)

# Filter small domains (< 1 cell = noise)
valid_domains = domain_diameters[domain_volumes > 1]

if len(valid_domains) > 0:
    D_p25 = np.percentile(valid_domains, 25)
    D_p50 = np.percentile(valid_domains, 50)
    D_p75 = np.percentile(valid_domains, 75)
    D_p90 = np.percentile(valid_domains, 90)
    D_p95 = np.percentile(valid_domains, 95)
    D_mean = np.mean(valid_domains)
    D_std = np.std(valid_domains)
    D_max = np.max(valid_domains)
else:
    D_p25 = D_p50 = D_p75 = D_p90 = D_p95 = D_mean = D_std = D_max = 0

print(f"  Valid domains (>1 cell): {len(valid_domains)}")
print(f"  Domain diameter statistics (Mpc):")
print(f"    P25 = {D_p25:.2f}")
print(f"    P50 (median) = {D_p50:.2f}")
print(f"    P75 = {D_p75:.2f}")
print(f"    P90 = {D_p90:.2f}")
print(f"    P95 = {D_p95:.2f}")
print(f"    Mean = {D_mean:.2f}")
print(f"    Std = {D_std:.2f}")
print(f"    Max = {D_max:.2f}")

# Save domain data
np.save(f"{OUTPUT_DIR}/domain_diameters.npy", valid_domains)

domain_stats = {
    "n_domains_total": int(n_domains),
    "n_domains_valid": int(len(valid_domains)),
    "cell_size_mpc": float(cell_size),
    "D_p25_mpc": float(D_p25),
    "D_p50_mpc": float(D_p50),
    "D_p75_mpc": float(D_p75),
    "D_p90_mpc": float(D_p90),
    "D_p95_mpc": float(D_p95),
    "D_mean_mpc": float(D_mean),
    "D_std_mpc": float(D_std),
    "D_max_mpc": float(D_max)
}

with open(f"{OUTPUT_DIR}/domain_stats.json", 'w') as f:
    json.dump(domain_stats, f, indent=2)
print(f"  Saved: domain_diameters.npy, domain_stats.json")

# ============================================================
# 6) POWER SPECTRUM
# ============================================================
print("\n" + "=" * 60)
print("6) POWER SPECTRUM")
print("=" * 60)

def compute_power_spectrum(field, box_size, ngrid):
    """Compute 3D power spectrum P(k)"""
    # FFT
    fft_field = np.fft.fftn(field)
    Pk_3d = np.abs(fft_field)**2 / ngrid**6

    # k-space grid
    kf = 2 * np.pi / box_size  # fundamental frequency
    kx = np.fft.fftfreq(ngrid, d=1.0/ngrid) * kf
    ky = np.fft.fftfreq(ngrid, d=1.0/ngrid) * kf
    kz = np.fft.fftfreq(ngrid, d=1.0/ngrid) * kf

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Bin by |k|
    k_bins = np.linspace(0, kf * ngrid / 2, ngrid // 2 + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    Pk_binned = np.zeros(len(k_centers))
    counts = np.zeros(len(k_centers))

    k_flat = k_mag.flatten()
    Pk_flat = Pk_3d.flatten()

    for i in range(len(k_centers)):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if np.sum(mask) > 0:
            Pk_binned[i] = np.mean(Pk_flat[mask])
            counts[i] = np.sum(mask)

    return k_centers, Pk_binned, counts

# Compute for density field (use delta = (rho - mean) / mean)
k_density, Pk_density, counts_density = compute_power_spectrum(delta, L_BOX, N_GRID)

# Compute for polarization field
k_pol, Pk_pol, counts_pol = compute_power_spectrum(P, L_BOX, N_GRID)

print(f"  k range: [{k_density[1]:.4f}, {k_density[-1]:.4f}] Mpc⁻¹")
print(f"  Pk_density peak at k = {k_density[np.argmax(Pk_density)]:.4f} Mpc⁻¹")
print(f"  Pk_polarization peak at k = {k_pol[np.argmax(Pk_pol)]:.4f} Mpc⁻¹")

# Save power spectra
pk_density_data = {
    "k_mpc_inv": k_density.tolist(),
    "Pk": Pk_density.tolist(),
    "counts": counts_density.tolist()
}
with open(f"{OUTPUT_DIR}/Pk_density.json", 'w') as f:
    json.dump(pk_density_data, f, indent=2)

pk_pol_data = {
    "k_mpc_inv": k_pol.tolist(),
    "Pk": Pk_pol.tolist(),
    "counts": counts_pol.tolist()
}
with open(f"{OUTPUT_DIR}/Pk_polarization.json", 'w') as f:
    json.dump(pk_pol_data, f, indent=2)

print(f"  Saved: Pk_density.json, Pk_polarization.json")

# ============================================================
# 7) VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("7) VISUALIZATION")
print("=" * 60)

# Polarization slice (middle slice)
mid = N_GRID // 2
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(P[:, :, mid].T, origin='lower', cmap='RdBu_r',
               vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')
ax.set_title(f'Polarization Field P (z = {mid * cell_size:.1f} Mpc, step 500)')
cbar = plt.colorbar(im, ax=ax, label='P = (ρ+ - ρ-) / (ρ+ + ρ-)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_slice.png", dpi=150)
plt.close()
print(f"  Saved: polarization_slice.png")

# Density slice
fig, ax = plt.subplots(figsize=(10, 8))
log_rho = np.log10(rho_total[:, :, mid].T + 1)
im = ax.imshow(log_rho, origin='lower', cmap='inferno',
               extent=[0, L_BOX, 0, L_BOX])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')
ax.set_title(f'Density Field log₁₀(ρ+1) (z = {mid * cell_size:.1f} Mpc, step 500)')
cbar = plt.colorbar(im, ax=ax, label='log₁₀(ρ + 1)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/density_slice.png", dpi=150)
plt.close()
print(f"  Saved: density_slice.png")

# Polarization histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(P_flat, bins=100, density=True, alpha=0.7, edgecolor='black')
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.axvline(-0.5, color='r', linestyle='--', alpha=0.5, label='|P| = 0.5 threshold')
ax.axvline(0.5, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Polarization P')
ax.set_ylabel('Probability Density')
ax.set_title(f'Polarization Distribution (σ_P = {sigma_P:.4f}, frac(|P|>0.5) = {frac_high_P:.4f})')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_histogram.png", dpi=150)
plt.close()
print(f"  Saved: polarization_histogram.png")

# Domain size distribution
if len(valid_domains) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_domains, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(D_p50, color='r', linestyle='--', label=f'Median = {D_p50:.2f} Mpc')
    ax.axvline(cell_size, color='g', linestyle=':', label=f'Cell size = {cell_size:.2f} Mpc')
    ax.set_xlabel('Domain Diameter (Mpc)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Domain Size Distribution (N = {len(valid_domains)}, median = {D_p50:.2f} Mpc)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/domain_size_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: domain_size_distribution.png")

# Polarization-density correlation
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(percentiles, mean_absP_by_percentile, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Density Percentile')
ax.set_ylabel('Mean |P|')
ax.set_title(f'Polarization vs Density (corr(|P|, δ) = {corr_absP_delta:.4f})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_density_correlation.png", dpi=150)
plt.close()
print(f"  Saved: polarization_density_correlation.png")

# Power spectrum - density
fig, ax = plt.subplots(figsize=(10, 6))
valid = Pk_density > 0
ax.loglog(k_density[valid], Pk_density[valid], 'b-', linewidth=2, label='P(k) density')
ax.set_xlabel('k (Mpc⁻¹)')
ax.set_ylabel('P(k)')
ax.set_title('Density Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Pk_density.png", dpi=150)
plt.close()
print(f"  Saved: Pk_density.png")

# Power spectrum - polarization
fig, ax = plt.subplots(figsize=(10, 6))
valid = Pk_pol > 0
ax.loglog(k_pol[valid], Pk_pol[valid], 'r-', linewidth=2, label='P(k) polarization')
ax.set_xlabel('k (Mpc⁻¹)')
ax.set_ylabel('P(k)')
ax.set_title('Polarization Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Pk_polarization.png", dpi=150)
plt.close()
print(f"  Saved: Pk_polarization.png")

# ============================================================
# 8) SCIENTIFIC SUMMARY REPORT
# ============================================================
print("\n" + "=" * 60)
print("8) SCIENTIFIC SUMMARY REPORT")
print("=" * 60)

report = f"""
================================================================================
V10 SNAPSHOT ANALYSIS REPORT
================================================================================
Snapshot: step 500
N particles: {n_particles:,}
  N+ = {n_pos:,}
  N- = {n_neg:,}
Box size: {L_BOX} Mpc
Grid resolution: {N_GRID}³ ({cell_size:.2f} Mpc/cell)

--------------------------------------------------------------------------------
POLARIZATION METRICS
--------------------------------------------------------------------------------
σ_P = {sigma_P:.4f}
mean(P) = {mean_P:.4f}
skewness(P) = {skewness_P:.4f}
kurtosis(P) = {kurtosis_P:.4f}
bimodality coefficient = {bimodality:.4f}
fraction(|P| > 0.5) = {frac_high_P:.4f}

--------------------------------------------------------------------------------
DOMAIN SIZE ANALYSIS
--------------------------------------------------------------------------------
Number of domains (|P| > 0.5): {n_domains}
Valid domains (>1 cell): {len(valid_domains)}

Domain diameter (Mpc):
  P25 = {D_p25:.2f}
  P50 (median) = {D_p50:.2f}
  P75 = {D_p75:.2f}
  P90 = {D_p90:.2f}
  P95 = {D_p95:.2f}
  Mean = {D_mean:.2f}
  Std = {D_std:.2f}
  Max = {D_max:.2f}

Minimum resolvable scale (cell size): {cell_size:.2f} Mpc

--------------------------------------------------------------------------------
DENSITY-POLARIZATION COUPLING
--------------------------------------------------------------------------------
corr(P, δ) = {corr_P_delta:.4f}
corr(|P|, δ) = {corr_absP_delta:.4f}

--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------
1. σ_P = {sigma_P:.4f} (target: ~0.715)
   → {"CLOSE TO TARGET" if abs(sigma_P - 0.715) < 0.1 else "BELOW TARGET" if sigma_P < 0.715 else "ABOVE TARGET"}

2. Median domain diameter = {D_p50:.2f} Mpc (target: < 5 Mpc)
   → {"MEETS CRITERION" if D_p50 < 5 else "DOES NOT MEET CRITERION"}

3. Median D / cell size = {D_p50/cell_size:.2f}
   → {"RESOLUTION SUFFICIENT" if D_p50/cell_size > 3 else "RESOLUTION MAY BE LIMITING"}

4. fraction(|P| > 0.5) = {frac_high_P:.4f}
   → {"STRONG SEGREGATION" if frac_high_P > 0.3 else "MODERATE SEGREGATION" if frac_high_P > 0.1 else "WEAK SEGREGATION"}

================================================================================
"""

with open(f"{OUTPUT_DIR}/report.txt", 'w') as f:
    f.write(report)
print(f"  Saved: report.txt")
print(report)

# ============================================================
# 9) FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("9) FINAL SUMMARY")
print("=" * 60)
print(f"  σ_P = {sigma_P:.4f}")
print(f"  median domain diameter = {D_p50:.2f} Mpc")
print(f"  fraction |P| > 0.5 = {frac_high_P:.4f}")
print(f"  corr(|P|, δ) = {corr_absP_delta:.4f}")
print("=" * 60)
