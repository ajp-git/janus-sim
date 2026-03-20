#!/usr/bin/env python3
"""
Détection de filaments par Hessian de densité (remplacement DisPerSE).
Identifie les crêtes 1D dans le champ de densité m−.
"""
import sys, time
sys.path.insert(0, '/mnt/T2/janus-sim/scripts')
from filament_utils import (load_snapshot, make_density_grid,
                             compute_hessian_eigenvalues,
                             filament_score_from_eigs)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SNAP  = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/snapshots/snap_004000.bin'
BOX   = 500.0
G     = 128          # grille 128³ — ~150 MB mémoire
OUT   = '/mnt/T2/janus-sim/output/filament_analysis/hessian_snap004000.png'
# Sigmas de lissage à tester (en Mpc) — multi-échelle
SIGMAS_MPC = [5.0, 10.0, 20.0]

import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("Chargement snapshot...")
pos, vel, mass = load_snapshot(SNAP)
print(f"  N = {len(mass):,}")

print(f"Construction grille densité {G}³...")
t0 = time.time()
grid_m, grid_p = make_density_grid(pos, mass, G=G, box=BOX)
print(f"  Done ({time.time()-t0:.1f}s)")
print(f"  m− : {grid_m.sum():.0f} particules  "
      f"m+ : {grid_p.sum():.0f} particules")

dx_mpc = BOX / G   # taille d'une cellule en Mpc

fig, axes = plt.subplots(len(SIGMAS_MPC), 3, figsize=(18, 6 * len(SIGMAS_MPC)))
fig.suptitle(f'Hessian Filament Detection — snap_004000 (z=0.225)\n'
             f'Grille {G}³  dx={dx_mpc:.1f} Mpc/cellule  BOX={BOX} Mpc', fontsize=12)

for row, sigma_mpc in enumerate(SIGMAS_MPC):
    sigma_px = sigma_mpc / dx_mpc
    print(f"\nSigma = {sigma_mpc} Mpc ({sigma_px:.1f} px)...")

    t1 = time.time()
    eigs_m = compute_hessian_eigenvalues(grid_m, sigma_px=sigma_px)
    score_m, frac_m = filament_score_from_eigs(eigs_m)
    print(f"  m− : {frac_m*100:.1f}% filament  ({time.time()-t1:.1f}s)")

    eigs_p = compute_hessian_eigenvalues(grid_p, sigma_px=sigma_px)
    score_p, frac_p = filament_score_from_eigs(eigs_p)
    print(f"  m+ : {frac_p*100:.1f}% filament")

    # Projection Z (somme sur axe 0 = Z en convention numpy [iz,iy,ix])
    proj_m = score_m.sum(axis=0)   # (G, G) projection XY
    proj_p = score_p.sum(axis=0)
    dens_m = gaussian_filter(grid_m, sigma=sigma_px).sum(axis=0)

    extent = [-BOX/2, BOX/2, -BOX/2, BOX/2]

    ax = axes[row, 0]
    ax.imshow(np.log1p(dens_m), origin='lower', cmap='Blues', extent=extent)
    ax.set_title(f'Densité m− lissée (σ={sigma_mpc} Mpc)', fontsize=9)
    ax.set_xlabel('X (Mpc)'); ax.set_ylabel('Y (Mpc)')

    ax = axes[row, 1]
    ax.imshow(proj_m, origin='lower', cmap='hot', extent=extent)
    ax.set_title(f'Score filament m− ({frac_m*100:.1f}%)', fontsize=9)
    ax.set_xlabel('X (Mpc)')

    ax = axes[row, 2]
    # Superposition : densité m− en bleu, filaments m− en jaune
    rgb = np.zeros((*proj_m.shape, 3))
    dm_n = dens_m / (dens_m.max() + 1e-10)
    sc_n = proj_m / (proj_m.max() + 1e-10)
    rgb[:,:,2] = np.clip(dm_n, 0, 1)       # bleu = densité m−
    rgb[:,:,0] = np.clip(sc_n * 2, 0, 1)   # rouge = filament
    rgb[:,:,1] = np.clip(sc_n, 0, 1)       # vert = filament (→ jaune)
    ax.imshow(rgb, origin='lower', extent=extent)
    ax.set_title(f'Superposition densité+filaments (σ={sigma_mpc} Mpc)', fontsize=9)
    ax.set_xlabel('X (Mpc)')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegardé : {OUT}")
