#!/usr/bin/env python3
"""
Rendu SPH adaptatif sur la simulation zoom Phase 1.
Révèle la sous-structure avec sigma adaptatif à la densité locale.
"""
import sys, time, glob, os
sys.path.insert(0, '/mnt/T2/janus-sim/scripts')
from filament_utils import load_snapshot, compute_knn_density, render_adaptive_2d

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Trouver le dernier snapshot zoom
ZOOM_DIR = '/mnt/T2/janus-sim/output/zoom_phase1_z339/snapshots/'
snaps = sorted(glob.glob(os.path.join(ZOOM_DIR, 'snap_*.bin')))
if not snaps:
    # Essayer d'autres chemins possibles
    for d in ['/mnt/T2/janus-sim/output/zoom_phase1_z339/',
              '/mnt/T2/janus-sim/output/zoom_50Mpc/',
              '/mnt/T2/janus-sim/output/zoom_80Mpc/',
              '/mnt/T2/janus-sim/output/zoom_phase1/']:
        snaps = sorted(glob.glob(os.path.join(d, 'snap_*.bin')))
        if snaps: ZOOM_DIR = d; break
        snaps = sorted(glob.glob(os.path.join(d, 'snapshots', 'snap_*.bin')))
        if snaps: ZOOM_DIR = os.path.join(d, 'snapshots'); break
if not snaps:
    print("ERREUR : aucun snapshot zoom trouvé. Adapter ZOOM_DIR.")
    sys.exit(1)

SNAP_FIRST = snaps[0]
SNAP_LAST  = snaps[-1]
OUT_DIR    = '/mnt/T2/janus-sim/output/filament_analysis/'
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Zoom dir : {ZOOM_DIR}")
print(f"Premier snap : {os.path.basename(SNAP_FIRST)}")
print(f"Dernier snap : {os.path.basename(SNAP_LAST)}")

# Détecter la taille de la boîte zoom depuis les coords
pos0, _, mass0 = load_snapshot(SNAP_FIRST)
BOX_ZOOM = float(np.ceil((pos0.max() - pos0.min()).max() / 10) * 10)
print(f"Boîte zoom détectée : {BOX_ZOOM:.0f} Mpc")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f'Zoom Phase 1 — Rendu adaptatif SPH\n'
             f'BOX={BOX_ZOOM:.0f} Mpc  N={len(mass0):,}', fontsize=12)

K_NN = 32   # plus de voisins pour zoom haute résolution

for col, (snap_path, step_name) in enumerate([(SNAP_FIRST, 'Step 0'),
                                               (SNAP_LAST,  'Step final')]):
    print(f"\nTraitement {step_name} ({os.path.basename(snap_path)})...")
    pos, vel, mass = load_snapshot(snap_path)
    mask_m = mass < 0
    mask_p = mass > 0

    print(f"  N− = {mask_m.sum():,}  N+ = {mask_p.sum():,}")
    print(f"  kNN densité (k={K_NN})...", end=' ', flush=True)
    t0 = time.time()
    rho = compute_knn_density(pos, k=K_NN)
    print(f"{time.time()-t0:.1f}s")

    # Projection XY — les 4 combinaisons
    configs = [
        (mask_m, 'Blues',  'm− adaptatif'),
        (mask_p, 'Reds',   'm+ adaptatif'),
    ]
    for row, (smask, cmap, title) in enumerate(configs):
        ax = axes[row, col * 2]
        if smask.sum() > 0:
            img = render_adaptive_2d(pos[smask], rho[smask],
                                      axis_proj=2, W=1024, H=1024,
                                      box=BOX_ZOOM, n_strata=8)
            ax.imshow(np.log1p(img), origin='lower', cmap=cmap,
                      extent=[-BOX_ZOOM/2, BOX_ZOOM/2]*2)
        ax.set_title(f'{step_name} — {title}', fontsize=8)
        ax.set_xlabel('X (Mpc)'); ax.set_ylabel('Y (Mpc)')

        # Superposition m−/m+
        ax2 = axes[row, col * 2 + 1]
        img_m = render_adaptive_2d(pos[mask_m], rho[mask_m], axis_proj=2,
                                    W=1024, H=1024, box=BOX_ZOOM, n_strata=6)
        img_p = render_adaptive_2d(pos[mask_p], rho[mask_p], axis_proj=2,
                                    W=1024, H=1024, box=BOX_ZOOM, n_strata=6)
        rgb = np.zeros((1024, 1024, 3))
        rgb[:,:,2] = np.clip(np.log1p(img_m) / (np.log1p(img_m).max()+1e-10), 0, 1)
        rgb[:,:,0] = np.clip(np.log1p(img_p) / (np.log1p(img_p).max()+1e-10), 0, 1)
        ax2.imshow(rgb, origin='lower', extent=[-BOX_ZOOM/2, BOX_ZOOM/2]*2)
        ax2.set_title(f'{step_name} — Superposition bleu=m− rouge=m+', fontsize=8)
        ax2.set_xlabel('X (Mpc)')

OUT = os.path.join(OUT_DIR, 'zoom_adaptive_sph.png')
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegardé : {OUT}")
