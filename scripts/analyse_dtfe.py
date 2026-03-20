#!/usr/bin/env python3
"""
Analyse DTFE (Delaunay Tessellation Field Estimator adapté).
Utilise la densité kNN pour révéler les structures inter-halos.
"""
import sys, time
sys.path.insert(0, '/mnt/T2/janus-sim/scripts')
from filament_utils import load_snapshot, compute_knn_density, render_adaptive_2d

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SNAP   = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/snapshots/snap_004000.bin'
BOX    = 500.0
OUT    = '/mnt/T2/janus-sim/output/filament_analysis/dtfe_snap004000.png'
K_NN   = 16          # voisins pour l'estimation de densité
N_SAMP = 500_000     # sous-échantillon pour kNN (limite mémoire)

import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("Chargement snapshot...")
t0 = time.time()
pos, vel, mass = load_snapshot(SNAP)
mask_m = mass < 0
mask_p = mass > 0
print(f"  N− = {mask_m.sum():,}  N+ = {mask_p.sum():,}  ({time.time()-t0:.1f}s)")

# Sous-échantillonnage stratifié (même proportion m+/m−)
rng = np.random.default_rng(42)
n_m = min(N_SAMP // 2, mask_m.sum())
n_p = min(N_SAMP // 2, mask_p.sum())
idx_m = rng.choice(np.where(mask_m)[0], n_m, replace=False)
idx_p = rng.choice(np.where(mask_p)[0], n_p, replace=False)
idx   = np.concatenate([idx_m, idx_p])
pos_s = pos[idx]
mass_s = mass[idx]

print(f"kNN densité sur {len(idx):,} particules (k={K_NN})...")
t1 = time.time()
rho = compute_knn_density(pos_s, k=K_NN)
print(f"  Done ({time.time()-t1:.1f}s)")

# Rendu adaptatif — 3 projections (XY, XZ, YZ)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'DTFE Adaptive Density — snap_004000 (z=0.225)\n'
             f'N_sample={len(idx):,}  k={K_NN}  BOX={BOX} Mpc', fontsize=12)

W, H = 2048, 2048
proj_names = ['XY (proj Z)', 'XZ (proj Y)', 'YZ (proj X)']
sign_labels = ['m−', 'm+']
sign_masks  = [mass_s < 0, mass_s > 0]
cmaps       = ['Blues', 'Reds']

for row, (lbl, smask, cmap) in enumerate(zip(sign_labels, sign_masks, cmaps)):
    rho_s = rho[smask]
    for col, (name, axis) in enumerate(zip(proj_names, [2, 1, 0])):
        ax = axes[row, col]
        img = render_adaptive_2d(pos_s[smask], rho_s,
                                  axis_proj=axis, W=W, H=H, box=BOX, n_strata=8)
        # Normalisation log
        img_log = np.log1p(img)
        ax.imshow(img_log, origin='lower', cmap=cmap,
                  extent=[-BOX/2, BOX/2, -BOX/2, BOX/2])
        ax.set_title(f'{lbl} — {name}', fontsize=9)
        ax.set_xlabel('Mpc'); ax.set_ylabel('Mpc')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Sauvegardé : {OUT}")

# Stats filaments potentiels
print(f"\nStats densité m− : median={np.median(rho[mass_s<0]):.4f}  "
      f"max={rho[mass_s<0].max():.4f} N/Mpc³")
print(f"Stats densité m+ : median={np.median(rho[mass_s>0]):.4f}  "
      f"max={rho[mass_s>0].max():.4f} N/Mpc³")
