# Tâche : Détection de filaments Janus — Phase A (analyse gratuite) + Phase B (nouveau run)

## Contexte
Run principal terminé : 15M particules, boîte 500 Mpc, snap_004000.bin (z=0.225).
Les halos m− et m+ sont formés mais aucun filament inter-halos visible avec le
rendu gaussien fixe actuel. L'objectif est de révéler les structures cachées par
trois méthodes d'analyse sans GPU (Phase A), puis de lancer un nouveau run avec
k_min=20 pour produire de vrais filaments (Phase B).

## Environnement
- Python : `/tmp/plotenv/bin/python`
- Packages disponibles : numpy, scipy, matplotlib, sklearn, networkx, astropy
  (installer scikit-image si nécessaire : pip install scikit-image --break-system-packages)
- Snapshot principal : `/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/snap_004000.bin`
- Format .bin : header u64 little-endian (8 bytes) + N × 28 bytes (7 × float32)
  → [x, y, z, vx, vy, vz, mass_sign], coords en [-250, +250] Mpc
- BOX = 500.0 Mpc

---

## PHASE A — Analyse sans GPU (3 scripts, lancer dans l'ordre)

### Prérequis commun : fonctions de base

Créer `/mnt/T2/janus-sim/scripts/filament_utils.py` avec ces fonctions exactes
(vérifiées, ne pas modifier) :

```python
"""filament_utils.py — Utilitaires communs détection filaments Janus"""
import struct
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# BLOC 1 : Chargement snapshot
# Vérifié 3× — ne pas modifier
# ══════════════════════════════════════════════════════════════════════
def load_snapshot(path):
    """
    Charge un snapshot Janus au format binaire.
    Header : u64 little-endian = N particules (8 bytes)
    Data   : N × 28 bytes = N × 7 float32 [x,y,z,vx,vy,vz,mass_sign]
    Coords : [-250, +250] Mpc
    mass_sign : +1.0 (m+) ou -1.0 (m−)
    """
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    pos  = data[:, :3].astype(np.float64)   # (N, 3) Mpc
    vel  = data[:, 3:6].astype(np.float64)  # (N, 3)
    mass = data[:, 6].astype(np.float64)    # (N,)  +1 ou -1
    return pos, vel, mass


# ══════════════════════════════════════════════════════════════════════
# BLOC 2 : Grille de densité 3D
# Vérifié 3× — ne pas modifier
# ══════════════════════════════════════════════════════════════════════
def make_density_grid(pos, mass, G=128, box=500.0):
    """
    Dépose les particules sur une grille G³ (NGP).
    Coords [-box/2, +box/2] → indices [0, G).
    Retourne grid_m (m−) et grid_p (m+), shape (G, G, G), float32.
    Convention d'axe numpy : grid[iz, iy, ix]
    """
    half = box / 2.0
    ix = np.clip(((pos[:, 0] + half) / box * G).astype(np.int32), 0, G - 1)
    iy = np.clip(((pos[:, 1] + half) / box * G).astype(np.int32), 0, G - 1)
    iz = np.clip(((pos[:, 2] + half) / box * G).astype(np.int32), 0, G - 1)

    mask_m = mass < 0
    mask_p = mass > 0

    grid_m = np.zeros((G, G, G), dtype=np.float32)
    grid_p = np.zeros((G, G, G), dtype=np.float32)
    np.add.at(grid_m, (iz[mask_m], iy[mask_m], ix[mask_m]), 1.0)
    np.add.at(grid_p, (iz[mask_p], iy[mask_p], ix[mask_p]), 1.0)
    return grid_m, grid_p


# ══════════════════════════════════════════════════════════════════════
# BLOC 3 : Densité adaptive kNN (DTFE)
# Vérifié 3× — ne pas modifier
# ══════════════════════════════════════════════════════════════════════
def compute_knn_density(pos, k=16):
    """
    Densité adaptive inspirée DTFE : ρ_i = k / ((4/3)π r_k³)
    r_k = distance au k-ème voisin le plus proche (non-soi).
    n_neighbors=k+1 car le 1er voisin est la particule elle-même (dist≈0).
    Pour 15M particules, sous-échantillonner en amont si nécessaire.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=-1)
    nbrs.fit(pos)
    distances, _ = nbrs.kneighbors(pos)
    r_k = np.maximum(distances[:, k], 1e-10)   # évite div/0
    vol_k = (4.0 / 3.0) * np.pi * r_k ** 3
    return k / vol_k   # unités : 1/Mpc³


# ══════════════════════════════════════════════════════════════════════
# BLOC 4 : Hessian de densité + score filament
# Vérifié 3× — ne pas modifier
# ══════════════════════════════════════════════════════════════════════
def compute_hessian_eigenvalues(density_3d, sigma_px):
    """
    Valeurs propres du Hessian de densité sur grille G³.
    Traitement slice par slice pour limiter la mémoire.
    Mémoire : ~150 MB pour G=128, ~1.2 GB pour G=256.

    Retourne eigs : (G, G, G, 3) float32, triées croissantes λ1 ≤ λ2 ≤ λ3.

    Interprétation physique du Hessian de DENSITÉ :
      Filament (crête 1D) : λ1 ≤ λ2 < 0, λ3 ≥ 0
        (compression forte dans 2 directions, étendu dans 1)
      Nœud   (pic 3D)    : λ1 ≤ λ2 ≤ λ3 < 0
      Feuille (crête 2D) : λ1 < 0, λ2 ≥ 0, λ3 ≥ 0
    """
    from scipy.ndimage import gaussian_filter
    G = density_3d.shape[0]
    d = gaussian_filter(density_3d.astype(np.float64), sigma=sigma_px)

    # Six dérivées secondes du tenseur symétrique 3×3
    # Axes numpy : 0=Z, 1=Y, 2=X
    Hxx = np.gradient(np.gradient(d, axis=2), axis=2)
    Hyy = np.gradient(np.gradient(d, axis=1), axis=1)
    Hzz = np.gradient(np.gradient(d, axis=0), axis=0)
    Hxy = np.gradient(np.gradient(d, axis=1), axis=2)
    Hxz = np.gradient(np.gradient(d, axis=0), axis=2)
    Hyz = np.gradient(np.gradient(d, axis=0), axis=1)

    # Valeurs propres slice par slice (mémoire : G×G×3×3×8 bytes par tranche)
    eigs = np.empty((G, G, G, 3), dtype=np.float32)
    for i in range(G):
        H_slice = np.stack([
            np.stack([Hxx[i], Hxy[i], Hxz[i]], axis=-1),
            np.stack([Hxy[i], Hyy[i], Hyz[i]], axis=-1),
            np.stack([Hxz[i], Hyz[i], Hzz[i]], axis=-1),
        ], axis=-2)                               # (G, G, 3, 3)
        eigs[i] = np.linalg.eigvalsh(H_slice)    # triées croissantes

    return eigs


def filament_score_from_eigs(eigs):
    """
    Score filament : min(|λ1|, |λ2|) si λ2 < 0 et λ3 ≥ 0, sinon 0.
    Score élevé = forte compression dans 2 directions = structure filamentaire.
    """
    l1 = eigs[..., 0].astype(np.float64)
    l2 = eigs[..., 1].astype(np.float64)
    l3 = eigs[..., 2].astype(np.float64)
    is_filament = (l2 < 0) & (l3 >= 0)
    score = np.where(is_filament,
                     np.minimum(np.abs(l1), np.abs(l2)),
                     0.0)
    frac = is_filament.mean()
    return score.astype(np.float32), frac


# ══════════════════════════════════════════════════════════════════════
# BLOC 5 : Rendu 2D adaptatif
# Vérifié 3× — ne pas modifier
# ══════════════════════════════════════════════════════════════════════
def render_adaptive_2d(pos, density, axis_proj=2,
                       W=2048, H=2048, box=500.0, n_strata=6):
    """
    Rendu 2D avec sigma adaptatif par strate de densité.
    Zones denses → sigma petit (structure fine préservée).
    Zones vides  → sigma grand (glow diffus visible).

    pos       : (N, 3) en [-box/2, +box/2]
    density   : (N,)   densité locale (kNN)
    axis_proj : axe de projection ignoré (0=X, 1=Y, 2=Z)
    n_strata  : nombre de strates log-uniformes en densité
    """
    from scipy.ndimage import gaussian_filter
    half = box / 2.0
    axes = [i for i in range(3) if i != axis_proj]
    x_coord = pos[:, axes[0]]
    y_coord = pos[:, axes[1]]

    ix = np.clip(((x_coord + half) / box * W).astype(np.int32), 0, W - 1)
    iy = np.clip(((y_coord + half) / box * H).astype(np.int32), 0, H - 1)

    rho_max = np.percentile(density, 99)
    rho_min = np.percentile(density,  1) + 1e-30
    log_edges = np.logspace(np.log10(rho_min),
                            np.log10(rho_max + 1e-30),
                            n_strata + 1)

    result = np.zeros((H, W), dtype=np.float64)
    for i in range(n_strata):
        lo, hi  = log_edges[i], log_edges[i + 1]
        mask    = (density >= lo) & (density < hi)
        if mask.sum() == 0:
            continue
        g = np.zeros((H, W), dtype=np.float32)
        np.add.at(g, (iy[mask], ix[mask]), 1.0)
        rho_mid  = np.sqrt(lo * hi)
        sigma_px = np.clip((rho_max / (rho_mid + 1e-30)) ** 0.33 * 2.0,
                           1.0, 40.0)
        result  += gaussian_filter(g, sigma=float(sigma_px))

    return result
```

---

### Script A1 : DTFE + Rendu adaptatif
**Fichier** : `scripts/analyse_dtfe.py`

Ce script révèle les structures faibles par densité adaptive kNN.
Pour 15M particules, sous-échantillonner à 500K pour le kNN puis
interpoler sur la grille.

```python
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

SNAP   = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/snap_004000.bin'
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
```

---

### Script A2 : Détection de filaments par Hessian
**Fichier** : `scripts/detect_filaments_hessian.py`

```python
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

SNAP  = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/snap_004000.bin'
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
```

---

### Script A3 : Rendu SPH adaptatif sur run zoom
**Fichier** : `scripts/render_zoom_adaptive.py`

Ce script utilise le snapshot de la Phase 1 zoom (1.5M particules, 80 Mpc)
pour un rendu adaptatif de haute qualité.

```python
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
ZOOM_DIR = '/mnt/T2/janus-sim/output/zoom_50Mpc/'  # adapter si nécessaire
snaps = sorted(glob.glob(os.path.join(ZOOM_DIR, 'snap_*.bin')))
if not snaps:
    # Essayer d'autres chemins possibles
    for d in ['/mnt/T2/janus-sim/output/zoom_80Mpc/',
              '/mnt/T2/janus-sim/output/zoom_phase1/']:
        snaps = sorted(glob.glob(os.path.join(d, 'snap_*.bin')))
        if snaps: ZOOM_DIR = d; break
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
```

---

### Ordre d'exécution Phase A

```bash
cd /mnt/T2/janus-sim

# Créer le répertoire de sortie
mkdir -p output/filament_analysis

# A1 : DTFE (sous-échantillon 500K, ~5-10 min)
/tmp/plotenv/bin/python scripts/analyse_dtfe.py

# A2 : Hessian multi-échelle (grille 128³, ~15-30 min)
/tmp/plotenv/bin/python scripts/detect_filaments_hessian.py

# A3 : Rendu adaptatif zoom (quelques minutes)
/tmp/plotenv/bin/python scripts/render_zoom_adaptive.py
```

**Critère de succès Phase A :**
- Si A2 montre `frac_filament > 10%` à sigma=10 Mpc → des filaments EXISTENT
  dans les données mais étaient cachés → le rendu peut les révéler
- Si A2 montre `frac_filament < 5%` à toutes les échelles → les filaments
  n'existent pas dans ce run → Phase B nécessaire

---

## PHASE B — Nouveau run 3M particules, k_min=20

**À lancer uniquement si Phase A ne révèle aucun filament.**

### B1 — Modifier le code Rust

Trouver le fichier source du run principal :
```bash
grep -rl "set_pm_k_min" /mnt/T2/janus-sim/src/bin/ | head -5
```

Dans le fichier trouvé, effectuer ces deux modifications :

**Modification 1 — k_min : 2 → 20**
```rust
// AVANT :
sim.set_pm_k_min(2);
// APRÈS :
sim.set_pm_k_min(20);
```

**Modification 2 — N : 15M → 3M**
Chercher la ligne qui définit N (chercher `15_000_000` ou `n_particles` ou `N_PARTICLES`):
```bash
grep -n "15_000_000\|n_particles\|N_PARTICLES\|15000000" \
  /mnt/T2/janus-sim/src/bin/<fichier_trouvé>.rs | head -10
```
Changer 15_000_000 → 3_000_000.

**Modification 3 — Répertoire de sortie**
Changer le output_dir de `janus_v13_500Mpc_15M` → `janus_v14_500Mpc_3M_kmin20`

### B2 — Compiler et valider

```bash
cd /mnt/T2/janus-sim
docker compose run --rm dev cargo build --release \
  --features "cuda cufft" --bin <nom_binaire>

# Test stabilité : lancer 50 steps seulement
docker compose run --rm dev ./target/release/<nom_binaire> \
  --steps 50 2>&1 | tail -20
# Vérifier : KE/KE₀ < 3.0 au step 50
```

### B3 — Lancer le run complet

```bash
docker compose run --rm -d dev \
  ./target/release/<nom_binaire> 2>&1 | tee \
  /mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/run.log &

# Surveiller
tail -f /mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/run.log
```

**Durée estimée :** ~20-25h (3M particules ≈ 20% du coût du run 15M)

**Métriques de succès à step 500 (z≈3.4) :**
- KE/KE₀ < 3.0 (stable)
- Seg > 0.05 (ségrégation amorcée)
- Filaments visibles dans les projections de densité

---

## Notes générales

- Tous les blocs de code dans `filament_utils.py` sont **vérifiés et testés** —
  ne pas les modifier sans raison explicite
- Si sklearn n'est pas disponible dans le container Docker :
  `pip install scikit-learn --break-system-packages`
- Les scripts A1/A2/A3 peuvent tourner en parallèle si la mémoire le permet
  (chacun utilise ~2-4 GB RAM)
- Sauvegarder les images dans `output/filament_analysis/` et les uploader
  pour interprétation avant de lancer Phase B
