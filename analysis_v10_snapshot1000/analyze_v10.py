#!/usr/bin/env python3
"""
V10 Snapshot 1000 Analysis - Full Scientific Analysis
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
SNAPSHOT_PATH = "/mnt/T2/janus-sim/output/janus_v10_highres/snapshots/snap_001000.bin"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot1000"
L_BOX = 200.0  # Mpc
N_GRID = 256   # Voxel resolution

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1) LOAD SNAPSHOT
# ============================================================
print("=" * 60)
print("1) LOADING SNAPSHOT 1000")
print("=" * 60)

with open(SNAPSHOT_PATH, 'rb') as f:
    step_bytes = f.read(8)
    step = struct.unpack('<Q', step_bytes)[0]
    print(f"  Step from header: {step}")
    data = np.frombuffer(f.read(), dtype=np.float32)

n_particles = len(data) // 4
print(f"  N particles: {n_particles:,}")

x = data[0::4].astype(np.float64)
y = data[1::4].astype(np.float64)
z = data[2::4].astype(np.float64)
signs = data[3::4].astype(np.float64)

pos_mask = signs > 0
neg_mask = signs < 0
n_pos = np.sum(pos_mask)
n_neg = np.sum(neg_mask)
print(f"  N+ = {n_pos:,}, N- = {n_neg:,}")

# ============================================================
# CIC INTERPOLATION
# ============================================================
print("\n  Computing CIC density fields on 256³ grid...")
CELL_SIZE = L_BOX / N_GRID

def cic_deposit(x, y, z, weights, ngrid, box_size):
    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    cell_size = box_size / ngrid
    xg = x / cell_size
    yg = y / cell_size
    zg = z / cell_size
    ix = np.floor(xg).astype(np.int32)
    iy = np.floor(yg).astype(np.int32)
    iz = np.floor(zg).astype(np.int32)
    dx = xg - ix
    dy = yg - iy
    dz = zg - iz

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
                flat_idx = iix * ngrid * ngrid + iiy * ngrid + iiz
                contrib = np.bincount(flat_idx, weights=w, minlength=ngrid**3)
                density += contrib.reshape((ngrid, ngrid, ngrid))
    return density

rho_pos = cic_deposit(x[pos_mask], y[pos_mask], z[pos_mask], np.ones(n_pos), N_GRID, L_BOX)
rho_neg = cic_deposit(x[neg_mask], y[neg_mask], z[neg_mask], np.ones(n_neg), N_GRID, L_BOX)

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
rho_mean = np.mean(rho_total)
delta = (rho_total - rho_mean) / rho_mean

print(f"  P range: [{P.min():.4f}, {P.max():.4f}]")

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
bimodality = (skewness_P**2 + 1) / (kurtosis_P + 3)
frac_high_P = np.mean(np.abs(P_flat) > 0.5)

print(f"  σ_P = {sigma_P:.4f}")
print(f"  bimodality coeff = {bimodality:.4f}")
print(f"  fraction(|P| > 0.5) = {frac_high_P:.4f}")

polarization_stats = {
    "step": 1000,
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

# ============================================================
# 4) GRADIENT & JANUS SCALE
# ============================================================
print("\n" + "=" * 60)
print("4) GRADIENT & JANUS SCALE")
print("=" * 60)

gx, gy, gz = np.gradient(P, CELL_SIZE)
grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
mean_grad = np.mean(grad_mag)

mean_P2 = np.mean(P**2)
mean_grad2 = np.mean(grad_mag**2)
L_J = np.sqrt(mean_P2 / mean_grad2)

print(f"  mean(|∇P|) = {mean_grad:.4f} Mpc⁻¹")
print(f"  L_J = {L_J:.4f} Mpc")

# ============================================================
# 5) DOMAIN SIZE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5) DOMAIN SIZE ANALYSIS")
print("=" * 60)

# Subsample for speed
sub = 2
P_sub = P[::sub, ::sub, ::sub]
cell_sub = CELL_SIZE * sub

pos_sub = P_sub > 0.8
neg_sub = P_sub < -0.8

labeled_pos, n_pos_dom = ndimage.label(pos_sub)
labeled_neg, n_neg_dom = ndimage.label(neg_sub)

print(f"  Positive domains: {n_pos_dom}")
print(f"  Negative domains: {n_neg_dom}")

def get_domain_stats(labeled, n_domains, cell_size):
    if n_domains == 0:
        return {"n": 0}
    sizes = ndimage.sum(np.ones_like(labeled), labeled, range(1, n_domains + 1))
    sizes = np.array(sizes)
    vol_mpc3 = sizes * (cell_size**3)
    diameters = 2 * (3 * vol_mpc3 / (4 * np.pi))**(1/3)
    valid = diameters[sizes > 1]
    if len(valid) == 0:
        return {"n": int(n_domains), "n_valid": 0}
    return {
        "n": int(n_domains),
        "n_valid": int(len(valid)),
        "D_median": float(np.median(valid)),
        "D_mean": float(np.mean(valid)),
        "D_max": float(np.max(valid)),
        "D_p75": float(np.percentile(valid, 75)),
        "D_p90": float(np.percentile(valid, 90))
    }

stats_pos = get_domain_stats(labeled_pos, n_pos_dom, cell_sub)
stats_neg = get_domain_stats(labeled_neg, n_neg_dom, cell_sub)

print(f"  Positive: median D = {stats_pos.get('D_median', 0):.2f} Mpc")
print(f"  Negative: median D = {stats_neg.get('D_median', 0):.2f} Mpc")

domain_stats = {
    "step": 1000,
    "threshold": 0.8,
    "L_J_mpc": float(L_J),
    "cell_size_mpc": float(CELL_SIZE),
    "positive": stats_pos,
    "negative": stats_neg
}

with open(f"{OUTPUT_DIR}/domain_stats.json", 'w') as f:
    json.dump(domain_stats, f, indent=2)

# ============================================================
# 6) DENSITY-POLARIZATION COUPLING
# ============================================================
print("\n" + "=" * 60)
print("6) DENSITY-POLARIZATION COUPLING")
print("=" * 60)

delta_flat = delta.flatten()
corr_P_delta = np.corrcoef(P_flat, delta_flat)[0, 1]
corr_absP_delta = np.corrcoef(np.abs(P_flat), delta_flat)[0, 1]

print(f"  corr(P, δ) = {corr_P_delta:.4f}")
print(f"  corr(|P|, δ) = {corr_absP_delta:.4f}")

# ============================================================
# 7) VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("7) VISUALIZATIONS")
print("=" * 60)

mid = N_GRID // 2

# Polarization slice
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(P[:, :, mid].T, origin='lower', cmap='RdBu_r',
               vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')
ax.set_title(f'Polarization Field P (step 1000, z = {mid * CELL_SIZE:.1f} Mpc)')
plt.colorbar(im, ax=ax, label='P')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_slice.png", dpi=150)
plt.close()
print("  Saved: polarization_slice.png")

# Polarization histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(P_flat, bins=100, density=True, alpha=0.7, edgecolor='black')
ax.axvline(-0.5, color='r', linestyle='--', alpha=0.5)
ax.axvline(0.5, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Polarization P')
ax.set_ylabel('Probability Density')
ax.set_title(f'Step 1000: σ_P = {sigma_P:.4f}, frac(|P|>0.5) = {frac_high_P:.4f}')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_histogram.png", dpi=150)
plt.close()
print("  Saved: polarization_histogram.png")

# Gradient slice
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
im = ax.imshow(P[:, :, mid].T, origin='lower', cmap='RdBu_r',
               vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
ax.set_title('Polarization P')
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(grad_mag[:, :, mid].T, origin='lower', cmap='hot',
               extent=[0, L_BOX, 0, L_BOX])
ax.set_title('|∇P| (Domain Walls)')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gradient_slice.png", dpi=150)
plt.close()
print("  Saved: gradient_slice.png")

# Multi-slice
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
slices = [32, 64, 96, 128, 160, 192]
for ax, z_slice in zip(axes.flat, slices):
    im = ax.imshow(P[:, :, z_slice].T, origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
    ax.set_title(f'z = {z_slice * CELL_SIZE:.1f} Mpc')
fig.colorbar(im, ax=axes, label='P', shrink=0.8)
fig.suptitle('Polarization Slices (Step 1000)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/multislice.png", dpi=150)
plt.close()
print("  Saved: multislice.png")

# ============================================================
# 8) COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("8) SUMMARY - STEP 1000")
print("=" * 60)

print(f"""
  POLARIZATION:
    σ_P = {sigma_P:.4f}
    fraction(|P| > 0.5) = {frac_high_P:.4f}
    bimodality = {bimodality:.4f}

  JANUS SCALE:
    L_J = {L_J:.4f} Mpc

  DOMAINS (|P| > 0.8):
    Positive: n={stats_pos['n']}, D_median={stats_pos.get('D_median', 0):.2f} Mpc
    Negative: n={stats_neg['n']}, D_median={stats_neg.get('D_median', 0):.2f} Mpc

  DENSITY COUPLING:
    corr(|P|, δ) = {corr_absP_delta:.4f}
""")

# Save summary
summary = {
    "step": 1000,
    "sigma_P": float(sigma_P),
    "fraction_high_P": float(frac_high_P),
    "bimodality": float(bimodality),
    "L_J_mpc": float(L_J),
    "mean_grad": float(mean_grad),
    "D_median_pos": stats_pos.get('D_median', 0),
    "D_median_neg": stats_neg.get('D_median', 0),
    "corr_absP_delta": float(corr_absP_delta)
}

with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
