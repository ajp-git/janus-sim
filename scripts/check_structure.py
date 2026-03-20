"""
Vérifie la structure spatiale réelle du snapshot :
- Distribution radiale des particules depuis les halos
- Fraction de particules dans les vides
- Densité dans les filaments potentiels
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os

BOX = 500.0

def load_snapshot(path):
    import struct
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.hdf5', '.h5']:
        import h5py
        with h5py.File(path, 'r') as f:
            pos  = f['PartType0/Coordinates'][:].astype(np.float64)
            mass = f['PartType0/Masses'][:].astype(np.float64)
        return pos, mass
    elif ext == '.npy':
        d = np.load(path)
        return d[:,:3].astype(np.float64), d[:,6].astype(np.float64)
    elif ext == '.bin':
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            d = np.frombuffer(f.read(n*28), dtype=np.float32).reshape(n,7)
        return d[:,:3].astype(np.float64), d[:,6].astype(np.float64)
    else:
        d = np.fromfile(path, dtype=np.float32).reshape(-1,7)
        return d[:,:3].astype(np.float64), d[:,6].astype(np.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--snap', required=True)
parser.add_argument('--step', type=int, default=0)
parser.add_argument('--z',    type=float, default=None)
args = parser.parse_args()

pos, mass = load_snapshot(args.snap)
sign = np.sign(mass)
mask_m = sign < 0
mask_p = sign > 0
N_m, N_p = mask_m.sum(), mask_p.sum()
print(f"N− = {N_m:,}  N+ = {N_p:,}")

# ── 1. Densité sur grille grossière (32³) ─────────────────────────────
G = 32
grid_m = np.zeros((G,G,G), dtype=np.float32)
grid_p = np.zeros((G,G,G), dtype=np.float32)

for mask, grid in [(mask_m, grid_m), (mask_p, grid_p)]:
    p = pos[mask]
    # coordonnées normalisées [0, BOX] → [0, G)
    if p.min() < -BOX*0.1:
        p = p + BOX/2
    ix = np.clip((p[:,0]/BOX*G).astype(int), 0, G-1)
    iy = np.clip((p[:,1]/BOX*G).astype(int), 0, G-1)
    iz = np.clip((p[:,2]/BOX*G).astype(int), 0, G-1)
    np.add.at(grid, (ix,iy,iz), 1)

# Cellules vides vs peuplées
n_cells = G**3
n_empty_m = (grid_m == 0).sum()
n_empty_p = (grid_p == 0).sum()
print(f"\nGrille {G}³ ({BOX/G:.1f} Mpc/cellule) :")
print(f"  m− : {n_empty_m}/{n_cells} cellules vides ({100*n_empty_m/n_cells:.1f}%)")
print(f"  m+ : {n_empty_p}/{n_cells} cellules vides ({100*n_empty_p/n_cells:.1f}%)")

# Distribution des occupations
occ_m = grid_m[grid_m > 0]
occ_p = grid_p[grid_p > 0]
print(f"  m− occupées : médiane={np.median(occ_m):.0f}  max={occ_m.max():.0f}  "
      f"ratio max/médiane={occ_m.max()/np.median(occ_m):.0f}×")
if len(occ_p):
    print(f"  m+ occupées : médiane={np.median(occ_p):.0f}  max={occ_p.max():.0f}")

# ── 2. Projection XY pour voir la structure ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(f"Structure spatiale — step {args.step}"
             + (f"  z={args.z:.3f}" if args.z else ""), fontsize=12)

# Projection m− sur grille fine 2048×2048 (projection complète)
RES = 2048
proj_m = np.zeros((RES, RES), dtype=np.float32)
proj_p = np.zeros((RES, RES), dtype=np.float32)

for mask, proj in [(mask_m, proj_m), (mask_p, proj_p)]:
    p = pos[mask].copy()
    if p.min() < -BOX*0.1:
        p += BOX/2
    # Projection complète (toutes les particules Z)
    ix = np.clip((p[:,0]/BOX*RES).astype(int), 0, RES-1)
    iy = np.clip((p[:,1]/BOX*RES).astype(int), 0, RES-1)
    np.add.at(proj, (iy, ix), 1)

from scipy.ndimage import gaussian_filter
sm_m = gaussian_filter(proj_m, sigma=0.8)
sm_p = gaussian_filter(proj_p, sigma=0.8)

# Log scale pour révéler les structures faibles
axes[0].imshow(np.log1p(sm_m), origin='lower', cmap='Blues',
               extent=[0,BOX,0,BOX])
axes[0].set_title(f'm− log-densité (projection complète)\n{mask_m.sum():,} particules',
                  fontsize=9)
axes[0].set_xlabel('X (Mpc)'); axes[0].set_ylabel('Y (Mpc)')

axes[1].imshow(np.log1p(sm_p), origin='lower', cmap='Reds',
               extent=[0,BOX,0,BOX])
axes[1].set_title(f'm+ log-densité (projection complète)\n{mask_p.sum():,} particules',
                  fontsize=9)
axes[1].set_xlabel('X (Mpc)')

# Superposition
rgb = np.zeros((RES, RES, 3))
rgb[:,:,0] = np.clip(np.log1p(sm_p) / (np.log1p(sm_p).max()+1e-10), 0, 1)
rgb[:,:,2] = np.clip(np.log1p(sm_m) / (np.log1p(sm_m).max()+1e-10), 0, 1)
axes[2].imshow(rgb, origin='lower', extent=[0,BOX,0,BOX])
axes[2].set_title('Superposition bleu=m−  rouge=m+', fontsize=9)
axes[2].set_xlabel('X (Mpc)')

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(script_dir, f"structure_step{args.step:04d}.png")
plt.savefig(outfile, dpi=120, bbox_inches='tight')
print(f"\nFigure → {outfile}")

# ── 3. Stats densité inter-halos ──────────────────────────────────────
# Percentiles de densité pour m−
flat_m = grid_m.flatten()
flat_m_sorted = np.sort(flat_m)
cumul = np.cumsum(flat_m_sorted) / flat_m_sorted.sum()
# Fraction de la masse dans les cellules les plus denses
print(f"\nConcentration de masse m− :")
for frac in [0.5, 0.8, 0.9, 0.95, 0.99]:
    idx = np.searchsorted(cumul, frac)
    n_cells_needed = n_cells - idx
    print(f"  {frac*100:.0f}% de la masse dans {n_cells_needed}/{n_cells} "
          f"cellules ({100*n_cells_needed/n_cells:.1f}%)")
