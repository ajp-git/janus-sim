#!/usr/bin/env python3
"""
V10 Snapshot 2000 Analysis - Full Scientific Analysis
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

# Parameters
SNAPSHOT_PATH = "/mnt/T2/janus-sim/output/janus_v10_highres/snapshots/snap_002000.bin"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot2000"
L_BOX = 200.0
N_GRID = 256
CELL_SIZE = L_BOX / N_GRID

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("LOADING SNAPSHOT 2000")
print("=" * 60)

with open(SNAPSHOT_PATH, 'rb') as f:
    f.read(8)
    data = np.frombuffer(f.read(), dtype=np.float32)

n_particles = len(data) // 4
print(f"N particles: {n_particles:,}")

x = data[0::4].astype(np.float64)
y = data[1::4].astype(np.float64)
z = data[2::4].astype(np.float64)
signs = data[3::4].astype(np.float64)

pos_mask = signs > 0
neg_mask = signs < 0
n_pos = np.sum(pos_mask)
n_neg = np.sum(neg_mask)
print(f"N+ = {n_pos:,}, N- = {n_neg:,}")

# CIC
print("\nComputing CIC density fields...")

def cic_deposit(x, y, z, weights, ngrid, box_size):
    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    cell_size = box_size / ngrid
    xg, yg, zg = x / cell_size, y / cell_size, z / cell_size
    ix, iy, iz = np.floor(xg).astype(np.int32), np.floor(yg).astype(np.int32), np.floor(zg).astype(np.int32)
    dx, dy, dz = xg - ix, yg - iy, zg - iz
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

rho_total = rho_pos + rho_neg
P = (rho_pos - rho_neg) / (rho_total + 1e-10)

np.save(f"{OUTPUT_DIR}/polarization_field.npy", P.astype(np.float32))
np.save(f"{OUTPUT_DIR}/density_field.npy", rho_total.astype(np.float32))

# Metrics
print("\n" + "=" * 60)
print("POLARIZATION METRICS")
print("=" * 60)

P_flat = P.flatten()
sigma_P = np.std(P_flat)
frac_high_P = np.mean(np.abs(P_flat) > 0.5)
bimodality = (stats.skew(P_flat)**2 + 1) / (stats.kurtosis(P_flat) + 3)

print(f"σ_P = {sigma_P:.4f}")
print(f"fraction(|P| > 0.5) = {frac_high_P:.4f}")
print(f"bimodality = {bimodality:.4f}")

# Gradient & L_J
gx, gy, gz = np.gradient(P, CELL_SIZE)
grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
L_J = np.sqrt(np.mean(P**2) / np.mean(grad_mag**2))
print(f"L_J = {L_J:.4f} Mpc")

# Domain analysis
print("\n" + "=" * 60)
print("DOMAIN ANALYSIS")
print("=" * 60)

sub = 2
P_sub = P[::sub, ::sub, ::sub]
cell_sub = CELL_SIZE * sub

labeled_pos, n_pos_dom = ndimage.label(P_sub > 0.8)
labeled_neg, n_neg_dom = ndimage.label(P_sub < -0.8)

def get_stats(labeled, n_dom, cell):
    if n_dom == 0:
        return {"n": 0, "D_median": 0}
    sizes = ndimage.sum(np.ones_like(labeled), labeled, range(1, n_dom + 1))
    sizes = np.array(sizes)
    diam = 2 * (3 * sizes * cell**3 / (4 * np.pi))**(1/3)
    valid = diam[sizes > 1]
    return {"n": int(n_dom), "n_valid": int(len(valid)),
            "D_median": float(np.median(valid)) if len(valid) > 0 else 0,
            "D_max": float(np.max(valid)) if len(valid) > 0 else 0}

stats_pos = get_stats(labeled_pos, n_pos_dom, cell_sub)
stats_neg = get_stats(labeled_neg, n_neg_dom, cell_sub)

print(f"Positive: n={stats_pos['n']}, D_median={stats_pos['D_median']:.2f} Mpc")
print(f"Negative: n={stats_neg['n']}, D_median={stats_neg['D_median']:.2f} Mpc")

# Visualizations
print("\n" + "=" * 60)
print("VISUALIZATIONS")
print("=" * 60)

mid = N_GRID // 2

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(P[:, :, mid].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')
ax.set_title(f'Polarization (Step 2000, z={mid*CELL_SIZE:.1f} Mpc)')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_slice.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(P_flat, bins=100, density=True, alpha=0.7, edgecolor='black')
ax.axvline(-0.5, color='r', linestyle='--')
ax.axvline(0.5, color='r', linestyle='--')
ax.set_xlabel('P')
ax.set_ylabel('PDF')
ax.set_title(f'Step 2000: σ_P={sigma_P:.4f}, frac(|P|>0.5)={frac_high_P:.4f}')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/polarization_histogram.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, z_slice in zip(axes.flat, [32, 64, 96, 128, 160, 192]):
    im = ax.imshow(P[:, :, z_slice].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
    ax.set_title(f'z = {z_slice * CELL_SIZE:.1f} Mpc')
fig.colorbar(im, ax=axes, shrink=0.8)
fig.suptitle('Step 2000 Polarization Slices')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/multislice.png", dpi=150)
plt.close()

print("Saved: polarization_slice.png, polarization_histogram.png, multislice.png")

# Summary
summary = {
    "step": 2000,
    "sigma_P": float(sigma_P),
    "fraction_high_P": float(frac_high_P),
    "bimodality": float(bimodality),
    "L_J_mpc": float(L_J),
    "D_median_pos": stats_pos["D_median"],
    "D_median_neg": stats_neg["D_median"],
    "n_domains_pos": stats_pos["n"],
    "n_domains_neg": stats_neg["n"]
}

with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY - STEP 2000")
print("=" * 60)
print(f"""
  σ_P = {sigma_P:.4f}
  L_J = {L_J:.4f} Mpc
  frac(|P|>0.5) = {frac_high_P:.4f}
  D_median = {(stats_pos['D_median'] + stats_neg['D_median'])/2:.2f} Mpc
""")
print("=" * 60)
