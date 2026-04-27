# Rendu VSL — Deux types de frames 4K
## À exécuter après le run VSL 2M (step 2000 atteint)

---

## ÉTAPE 0 — Vérifier les snapshots disponibles

```bash
ls -la /mnt/T2/janus-sim/output/janus_vsl_2M/snapshots/ | head -20
ls /mnt/T2/janus-sim/output/janus_vsl_2M/snapshots/ | wc -l
```

---

## ÉTAPE 1 — Précalculer la normalisation globale

Créer `/tmp/compute_normalization.py` :

```python
#!/usr/bin/env python3
"""
Précalcule vmin/vmax globaux sur TOUS les snapshots
pour éviter le flickering dans la vidéo.
Sauvegarder dans normalization.json
"""
import numpy as np
import struct
import json
from pathlib import Path
from tqdm import tqdm

SNAP_DIR = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/snapshots")
OUT_FILE = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/normalization.json")
GRID = 128

def read_snapshot(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]
        data = np.frombuffer(f.read(n * 25), dtype=np.dtype([
            ('x','f4'),('y','f4'),('z','f4'),
            ('vx','f4'),('vy','f4'),('vz','f4'),
            ('sign','i1')
        ]))
    return n, box, step, z, data

def compute_density_grid(pos, box, grid):
    cell = box / grid
    hist, _ = np.histogramdd(
        pos,
        bins=grid,
        range=[[-box/2,box/2]]*3
    )
    return hist / cell**3

snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
print(f"Trouvé {len(snaps)} snapshots")

rho_plus_min  = np.inf
rho_plus_max  = -np.inf
rho_minus_min = np.inf
rho_minus_max = -np.inf

for snap in tqdm(snaps[::5], desc="Scan normalisation"):  # échantillon 1/5
    n, box, step, z, data = read_snapshot(snap)
    
    pos_plus  = np.stack([data['x'][data['sign']>0],
                          data['y'][data['sign']>0],
                          data['z'][data['sign']>0]], axis=1)
    pos_minus = np.stack([data['x'][data['sign']<0],
                          data['y'][data['sign']<0],
                          data['z'][data['sign']<0]], axis=1)
    
    if len(pos_plus) > 0:
        d = compute_density_grid(pos_plus, box, GRID)
        d_log = np.log10(d[d>0])
        if len(d_log):
            rho_plus_min = min(rho_plus_min, np.percentile(d_log, 2))
            rho_plus_max = max(rho_plus_max, np.percentile(d_log, 99.5))
    
    if len(pos_minus) > 0:
        d = compute_density_grid(pos_minus, box, GRID)
        d_log = np.log10(d[d>0])
        if len(d_log):
            rho_minus_min = min(rho_minus_min, np.percentile(d_log, 2))
            rho_minus_max = max(rho_minus_max, np.percentile(d_log, 99.5))

norm = {
    "rho_plus_log_min":  float(rho_plus_min),
    "rho_plus_log_max":  float(rho_plus_max),
    "rho_minus_log_min": float(rho_minus_min),
    "rho_minus_log_max": float(rho_minus_max),
    "purity_min": -1.0,
    "purity_max":  1.0,
    "grid": GRID,
}

with open(OUT_FILE, 'w') as f:
    json.dump(norm, f, indent=2)

print(f"Normalisation sauvegardée dans {OUT_FILE}")
print(json.dumps(norm, indent=2))
```

```bash
/tmp/plotenv/bin/python /tmp/compute_normalization.py
```

---

## ÉTAPE 2 — RENDU A : Cinématique 4K plein écran

Créer `/tmp/render_cinematic.py` :

```python
#!/usr/bin/env python3
"""
RENDU A — Cinématique 4K plein écran
Pour vidéo grand public / YouTube

Layout :
  Gauche (15% largeur) : 3 projections XY, XZ, YZ empilées
  Centre (85% largeur) : scatter 3D plein écran

Fond noir, m+ rouge/orange chaleureux, m- bleu froid
Max de particules affichées (rasterized pour vitesse)
"""
import numpy as np
import struct
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import gc

# ─── Configuration ────────────────────────────────────────
SNAP_DIR  = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/snapshots")
FRAME_DIR = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/frames_cinematic")
NORM_FILE = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/normalization.json")
FRAME_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 150          # 4K = 3840×2160 à 150dpi → figsize (25.6, 14.4)
FIGSIZE   = (25.6, 14.4) # 4K
N_3D      = 500_000      # particules dans le scatter 3D
N_2D      = 300_000      # particules dans les projections 2D
ALPHA_PLUS  = 0.25       # transparence m+
ALPHA_MINUS = 0.06       # transparence m- (plus nombreuses)
S_PLUS  = 0.08           # taille points m+
S_MINUS = 0.03           # taille points m-

# Couleurs
COLOR_PLUS  = '#FF4500'  # orangered — m+
COLOR_MINUS = '#00BFFF'  # deepskyblue — m-
BG_COLOR    = '#000000'  # fond noir

with open(NORM_FILE) as f:
    norm = json.load(f)

def read_snapshot(path):
    with open(path, 'rb') as f:
        n    = struct.unpack('<I', f.read(4))[0]
        box  = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z    = struct.unpack('<f', f.read(4))[0]
        data = np.frombuffer(f.read(n * 25), dtype=np.dtype([
            ('x','f4'),('y','f4'),('z','f4'),
            ('vx','f4'),('vy','f4'),('vz','f4'),
            ('sign','i1')
        ]))
    return n, box, step, z, data

def subsample(arr, n_max, seed=42):
    if len(arr) <= n_max:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), n_max, replace=False)
    return arr[idx]

def render_frame(snap_path, frame_idx):
    n, box, step, z, data = read_snapshot(snap_path)
    half = box / 2.0

    mask_plus  = data['sign'] > 0
    mask_minus = data['sign'] < 0

    pos_plus  = np.stack([data['x'][mask_plus],
                          data['y'][mask_plus],
                          data['z'][mask_plus]], axis=1)
    pos_minus = np.stack([data['x'][mask_minus],
                          data['y'][mask_minus],
                          data['z'][mask_minus]], axis=1)

    # Sous-échantillonnage
    pp3d = subsample(pos_plus,  N_3D // 2)
    pm3d = subsample(pos_minus, N_3D // 2)
    pp2d = subsample(pos_plus,  N_2D // 2)
    pm2d = subsample(pos_minus, N_2D // 2)

    # ─── Figure ───────────────────────────────────────────
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG_COLOR)
    gs  = GridSpec(3, 5, figure=fig,
                   left=0.01, right=0.99,
                   top=0.93, bottom=0.02,
                   wspace=0.02, hspace=0.08)

    # Couleurs axes
    ax_kw = dict(facecolor=BG_COLOR)
    spine_color = '#333333'

    # ─── 3D central (colonnes 1-4, toutes les lignes) ─────
    ax3d = fig.add_subplot(gs[:, 1:], projection='3d')
    ax3d.set_facecolor(BG_COLOR)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('#1a1a1a')
    ax3d.yaxis.pane.set_edgecolor('#1a1a1a')
    ax3d.zaxis.pane.set_edgecolor('#1a1a1a')
    ax3d.grid(False)

    if len(pm3d):
        ax3d.scatter(pm3d[:,0], pm3d[:,1], pm3d[:,2],
                     c=COLOR_MINUS, s=S_MINUS, alpha=ALPHA_MINUS,
                     rasterized=True, depthshade=True, linewidths=0)
    if len(pp3d):
        ax3d.scatter(pp3d[:,0], pp3d[:,1], pp3d[:,2],
                     c=COLOR_PLUS,  s=S_PLUS,  alpha=ALPHA_PLUS,
                     rasterized=True, depthshade=True, linewidths=0)

    ax3d.set_xlim(-half, half)
    ax3d.set_ylim(-half, half)
    ax3d.set_zlim(-half, half)
    ax3d.set_xlabel('X [Mpc]', color='#666666', fontsize=7)
    ax3d.set_ylabel('Y [Mpc]', color='#666666', fontsize=7)
    ax3d.set_zlabel('Z [Mpc]', color='#666666', fontsize=7)
    ax3d.tick_params(colors='#444444', labelsize=6)
    ax3d.view_init(elev=25, azim=45 + step * 0.05)  # rotation lente

    # ─── Projections 2D (colonne 0) ───────────────────────
    projs = [
        ('XY', 0, 1, gs[0, 0]),
        ('XZ', 0, 2, gs[1, 0]),
        ('YZ', 1, 2, gs[2, 0]),
    ]

    for label, ix, iy, gs_pos in projs:
        ax = fig.add_subplot(gs_pos, **ax_kw)
        ax.set_facecolor(BG_COLOR)
        for sp in ax.spines.values():
            sp.set_edgecolor(spine_color)

        if len(pm2d):
            ax.scatter(pm2d[:,ix], pm2d[:,iy],
                       c=COLOR_MINUS, s=0.02,
                       alpha=0.04, rasterized=True, linewidths=0)
        if len(pp2d):
            ax.scatter(pp2d[:,ix], pp2d[:,iy],
                       c=COLOR_PLUS, s=0.05,
                       alpha=0.15, rasterized=True, linewidths=0)

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.tick_params(colors='#444444', labelsize=5)
        ax.set_title(label, color='#666666', fontsize=7, pad=2)

    # ─── Header ───────────────────────────────────────────
    n_plus  = mask_plus.sum()
    n_minus = mask_minus.sum()
    rho_plus_max_val  = float(np.max(np.bincount(
        np.ravel_multi_index(
            np.clip(((pos_plus + half) / box * 64).astype(int),
                    0, 63).T,
            (64,64,64)
        )
    ))) if len(pos_plus) else 0

    title = (f"JANUS VSL  |  c⁻/c⁺ = 10  |  μ = 19  |  "
             f"z = {z:.3f}  |  step {step}/2000  |  "
             f"N⁺ = {n_plus:,}  N⁻ = {n_minus:,}  |  "
             f"Box = {box:.0f} Mpc")
    fig.suptitle(title, color='white', fontsize=9,
                 y=0.97, fontfamily='monospace')

    # ─── Sauvegarde ───────────────────────────────────────
    out = FRAME_DIR / f"frame_{frame_idx:05d}.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight',
                facecolor=BG_COLOR)
    plt.close(fig)
    gc.collect()
    return out

# ─── Lancer le rendu ──────────────────────────────────────
snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
print(f"Rendu cinématique : {len(snaps)} frames")

for i, snap in enumerate(tqdm(snaps, desc="Frames cinématiques")):
    out = render_frame(snap, i)

print(f"\n✓ {len(snaps)} frames dans {FRAME_DIR}")
```

```bash
/tmp/plotenv/bin/python /tmp/render_cinematic.py
```

---

## ÉTAPE 3 — RENDU B : Scientifique 10 panels

Créer `/tmp/render_scientific.py` :

```python
#!/usr/bin/env python3
"""
RENDU B — Scientifique 10 panels 4K
Pour analyse et papier

Layout 5×2 :
  [1] XY m+      [2] XZ m+      [3] Scatter 3D  [4] Temp map   [5] Purity
  [6] XY m-      [7] XZ m-      [8] SPH density  [9] Sound speed [10] Velocity
"""
import numpy as np
import struct
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from pathlib import Path
from tqdm import tqdm
import gc

SNAP_DIR  = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/snapshots")
FRAME_DIR = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/frames_scientific")
NORM_FILE = Path("/mnt/T2/janus-sim/output/janus_vsl_2M/normalization.json")
FRAME_DIR.mkdir(parents=True, exist_ok=True)

DPI     = 150
FIGSIZE = (25.6, 14.4)
GRID    = 256
N_3D    = 200_000
BG      = '#0a0a0a'
K_B_OVER_MP_CODE = 8.7e-9
MU_MOL  = 0.6
GAMMA   = 5.0 / 3.0

with open(NORM_FILE) as f:
    norm = json.load(f)

def read_snapshot(path):
    with open(path, 'rb') as f:
        n    = struct.unpack('<I', f.read(4))[0]
        box  = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z    = struct.unpack('<f', f.read(4))[0]
        data = np.frombuffer(f.read(n * 25), dtype=np.dtype([
            ('x','f4'),('y','f4'),('z','f4'),
            ('vx','f4'),('vy','f4'),('vz','f4'),
            ('sign','i1')
        ]))
    return n, box, step, z, data

def density_map_2d(pos, box, grid, axis_drop=2):
    """Projection 2D log-densité"""
    axes = [i for i in range(3) if i != axis_drop]
    h, _, _ = np.histogram2d(
        pos[:, axes[0]], pos[:, axes[1]],
        bins=grid,
        range=[[-box/2, box/2]] * 2
    )
    h = gaussian_filter(h, sigma=1.0)
    h[h <= 0] = np.nan
    return np.log10(h)

def purity_map(pos_plus, pos_minus, box, grid):
    """Carte de pureté (n+ - n-) / (n+ + n-)"""
    def hist2d(pos):
        h, _, _ = np.histogram2d(
            pos[:, 0], pos[:, 1],
            bins=grid,
            range=[[-box/2, box/2]] * 2
        )
        return h
    hp = hist2d(pos_plus)  if len(pos_plus)  else np.zeros((grid,grid))
    hm = hist2d(pos_minus) if len(pos_minus) else np.zeros((grid,grid))
    total = hp + hm
    with np.errstate(invalid='ignore', divide='ignore'):
        purity = np.where(total > 0, (hp - hm) / total, 0)
    return purity

def sound_speed_map(pos_plus, temp_arr, box, grid):
    """Carte vitesse du son cs = sqrt(γ k_B T / μ m_p)"""
    cell = box / grid
    cs_grid = np.zeros((grid, grid))
    count   = np.zeros((grid, grid))
    ix = np.clip(((pos_plus[:, 0] + box/2) / cell).astype(int), 0, grid-1)
    iy = np.clip(((pos_plus[:, 1] + box/2) / cell).astype(int), 0, grid-1)
    np.add.at(cs_grid, (ix, iy), np.sqrt(GAMMA * K_B_OVER_MP_CODE * temp_arr / MU_MOL))
    np.add.at(count,   (ix, iy), 1)
    with np.errstate(invalid='ignore'):
        cs_grid = np.where(count > 0, cs_grid / count, np.nan)
    return cs_grid * 977.8  # Mpc/Gyr → km/s

def render_frame(snap_path, frame_idx):
    n, box, step, z, data = read_snapshot(snap_path)
    half = box / 2.0

    mask_plus  = data['sign'] > 0
    mask_minus = data['sign'] < 0

    pos_plus  = np.stack([data['x'][mask_plus],
                          data['y'][mask_plus],
                          data['z'][mask_plus]], axis=1)
    pos_minus = np.stack([data['x'][mask_minus],
                          data['y'][mask_minus],
                          data['z'][mask_minus]], axis=1)

    # Température (si disponible dans le snapshot étendu)
    # Sinon T=1e4 K uniforme
    try:
        temp_plus = data['temp'][mask_plus]
    except:
        temp_plus = np.full(mask_plus.sum(), 1e4)

    # Sous-échantillon 3D
    rng = np.random.default_rng(42)
    n3p = min(N_3D // 2, len(pos_plus))
    n3m = min(N_3D // 2, len(pos_minus))
    pp3 = pos_plus[rng.choice(len(pos_plus), n3p, replace=False)] if n3p else pos_plus
    pm3 = pos_minus[rng.choice(len(pos_minus), n3m, replace=False)] if n3m else pos_minus

    # Calcul des maps
    grid = norm['grid']
    dm_plus_xy  = density_map_2d(pos_plus,  box, grid, axis_drop=2)
    dm_plus_xz  = density_map_2d(pos_plus,  box, grid, axis_drop=1)
    dm_minus_xy = density_map_2d(pos_minus, box, grid, axis_drop=2)
    dm_minus_xz = density_map_2d(pos_minus, box, grid, axis_drop=1)
    pmap        = purity_map(pos_plus, pos_minus, box, grid)
    csmap       = sound_speed_map(pos_plus, temp_plus, box, grid)

    # Normalisations fixes
    vmin_p  = norm['rho_plus_log_min']
    vmax_p  = norm['rho_plus_log_max']
    vmin_m  = norm['rho_minus_log_min']
    vmax_m  = norm['rho_minus_log_max']

    # ─── Figure ───────────────────────────────────────────
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    gs  = GridSpec(2, 5, figure=fig,
                   left=0.04, right=0.97,
                   top=0.91, bottom=0.04,
                   wspace=0.12, hspace=0.18)

    ext = [-half, half, -half, half]
    im_kw = dict(origin='lower', extent=ext, aspect='auto')

    # [1] XY m+
    ax = fig.add_subplot(gs[0, 0], facecolor=BG)
    ax.imshow(dm_plus_xy.T, cmap='hot', vmin=vmin_p, vmax=vmax_p, **im_kw)
    ax.set_title('XY  m⁺', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [2] XZ m+
    ax = fig.add_subplot(gs[0, 1], facecolor=BG)
    ax.imshow(dm_plus_xz.T, cmap='hot', vmin=vmin_p, vmax=vmax_p, **im_kw)
    ax.set_title('XZ  m⁺', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [3] Scatter 3D
    ax3 = fig.add_subplot(gs[0, 2], projection='3d', facecolor=BG)
    ax3.set_facecolor(BG)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.grid(False)
    if len(pm3):
        ax3.scatter(pm3[:,0], pm3[:,1], pm3[:,2],
                    c='#00BFFF', s=0.05, alpha=0.05,
                    rasterized=True, linewidths=0)
    if len(pp3):
        ax3.scatter(pp3[:,0], pp3[:,1], pp3[:,2],
                    c='#FF4500', s=0.15, alpha=0.20,
                    rasterized=True, linewidths=0)
    ax3.set_xlim(-half, half)
    ax3.set_ylim(-half, half)
    ax3.set_zlim(-half, half)
    ax3.tick_params(colors='#444', labelsize=5)
    ax3.set_title('3D', color='white', fontsize=8)

    # [4] Température map
    ax = fig.add_subplot(gs[0, 3], facecolor=BG)
    # Température sur grille
    t_grid = np.zeros((grid, grid))
    t_count = np.zeros((grid, grid))
    cell = box / grid
    ix = np.clip(((pos_plus[:,0]+half)/cell).astype(int), 0, grid-1)
    iy = np.clip(((pos_plus[:,1]+half)/cell).astype(int), 0, grid-1)
    np.add.at(t_grid,  (ix, iy), temp_plus)
    np.add.at(t_count, (ix, iy), 1)
    t_grid = np.where(t_count > 0, t_grid / t_count, np.nan)
    im = ax.imshow(t_grid.T, cmap='plasma', origin='lower',
                   extent=ext, aspect='auto',
                   vmin=100, vmax=1e4)
    plt.colorbar(im, ax=ax, label='T [K]', fraction=0.046)
    ax.set_title('Température m⁺', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [5] Purity map
    ax = fig.add_subplot(gs[0, 4], facecolor=BG)
    im = ax.imshow(pmap.T, cmap='RdBu_r', origin='lower',
                   extent=ext, aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Purity', fraction=0.046)
    ax.set_title('Purity (m⁺−m⁻)/(m⁺+m⁻)', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [6] XY m-
    ax = fig.add_subplot(gs[1, 0], facecolor=BG)
    ax.imshow(dm_minus_xy.T, cmap='Blues', vmin=vmin_m, vmax=vmax_m, **im_kw)
    ax.set_title('XY  m⁻', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [7] XZ m-
    ax = fig.add_subplot(gs[1, 1], facecolor=BG)
    ax.imshow(dm_minus_xz.T, cmap='Blues', vmin=vmin_m, vmax=vmax_m, **im_kw)
    ax.set_title('XZ  m⁻', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [8] SPH density m+ (log densité projetée = même que XY m+ mais cmap inferno)
    ax = fig.add_subplot(gs[1, 2], facecolor=BG)
    ax.imshow(dm_plus_xy.T, cmap='inferno', vmin=vmin_p, vmax=vmax_p, **im_kw)
    ax.set_title('Log Densité m⁺', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [9] Sound speed map
    ax = fig.add_subplot(gs[1, 3], facecolor=BG)
    im = ax.imshow(csmap.T, cmap='viridis', origin='lower',
                   extent=ext, aspect='auto')
    plt.colorbar(im, ax=ax, label='cs [km/s]', fraction=0.046)
    ax.set_title('Vitesse du son m⁺', color='white', fontsize=8)
    ax.tick_params(colors='#555', labelsize=6)

    # [10] Distribution vitesses
    ax = fig.add_subplot(gs[1, 4], facecolor=BG)
    v_plus  = np.sqrt(data['vx'][mask_plus]**2 +
                      data['vy'][mask_plus]**2 +
                      data['vz'][mask_plus]**2) * 977.8
    v_minus = np.sqrt(data['vx'][mask_minus]**2 +
                      data['vy'][mask_minus]**2 +
                      data['vz'][mask_minus]**2) * 977.8
    ax.hist(v_plus,  bins=80, color='#FF4500', alpha=0.7,
            density=True, label=f'm⁺ <v>={v_plus.mean():.0f}')
    ax.hist(v_minus, bins=80, color='#00BFFF', alpha=0.5,
            density=True, label=f'm⁻ <v>={v_minus.mean():.0f}')
    ax.set_xlabel('|v| [km/s]', color='white', fontsize=7)
    ax.set_ylabel('PDF', color='white', fontsize=7)
    ax.tick_params(colors='#555', labelsize=6)
    ax.legend(fontsize=6, facecolor='#111', labelcolor='white')
    ax.set_facecolor(BG)
    ax.set_title('Distribution vitesses', color='white', fontsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')

    # ─── Header & Sidebar ─────────────────────────────────
    n_plus  = mask_plus.sum()
    n_minus = mask_minus.sum()
    rho_p_max = float(np.nanmax(10**dm_plus_xy)) if not np.all(np.isnan(dm_plus_xy)) else 0
    rho_m_max = float(np.nanmax(10**dm_minus_xy)) if not np.all(np.isnan(dm_minus_xy)) else 0

    title = (f"Janus VSL  |  c⁻/c⁺=10  |  μ=19  |  λ₀=0  |  "
             f"z={z:.3f}  |  ρ⁺_max={rho_p_max:.0f}  |  "
             f"ρ⁻_max={rho_m_max:.0f}  |  step {step}/2000")
    fig.suptitle(title, color='white', fontsize=9,
                 y=0.97, fontfamily='monospace')

    # Sidebar texte
    sidebar = (f"c⁻/c⁺ = 10\n"
               f"μ = 19\n"
               f"N⁺ = {n_plus:,}\n"
               f"N⁻ = {n_minus:,}\n"
               f"─────────\n"
               f"z = {z:.3f}\n"
               f"ρ⁺_max = {rho_p_max:.0f}\n"
               f"ρ⁻_max = {rho_m_max:.0f}\n"
               f"─────────\n"
               f"T_min = {temp_plus.min():.0f} K\n"
               f"v_rms = {v_plus.std():.0f} km/s")
    fig.text(0.985, 0.5, sidebar, color='#aaaaaa',
             fontsize=6.5, fontfamily='monospace',
             va='center', ha='right',
             transform=fig.transFigure)

    out = FRAME_DIR / f"frame_{frame_idx:05d}.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    gc.collect()
    return out

snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
print(f"Rendu scientifique : {len(snaps)} frames")

for i, snap in enumerate(tqdm(snaps, desc="Frames scientifiques")):
    render_frame(snap, i)

print(f"✓ {len(snaps)} frames dans {FRAME_DIR}")
```

```bash
/tmp/plotenv/bin/python /tmp/render_scientific.py
```

---

## ÉTAPE 4 — Assembler les deux vidéos

```bash
# Vidéo cinématique
ffmpeg -y -framerate 24 \
  -pattern_type glob \
  -i '/mnt/T2/janus-sim/output/janus_vsl_2M/frames_cinematic/frame_*.png' \
  -c:v libx264 -crf 14 -preset slow \
  -pix_fmt yuv420p \
  /mnt/T2/janus-sim/output/janus_vsl_2M/janus_vsl_cinematic_4k.mp4

echo "✓ Vidéo cinématique générée"

# Vidéo scientifique
ffmpeg -y -framerate 24 \
  -pattern_type glob \
  -i '/mnt/T2/janus-sim/output/janus_vsl_2M/frames_scientific/frame_*.png' \
  -c:v libx264 -crf 16 -preset slow \
  -pix_fmt yuv420p \
  /mnt/T2/janus-sim/output/janus_vsl_2M/janus_vsl_scientific_4k.mp4

echo "✓ Vidéo scientifique générée"
```

---

## RÉSUMÉ

| Rendu | Frames dir | Vidéo | Usage |
|-------|-----------|-------|-------|
| A Cinématique | frames_cinematic/ | janus_vsl_cinematic_4k.mp4 | YouTube/public |
| B Scientifique | frames_scientific/ | janus_vsl_scientific_4k.mp4 | Papier/analyse |

**Ordre d'exécution :**
1. Attendre step 2000 du run VSL
2. `compute_normalization.py` (~5 min)
3. Les deux rendus en parallèle si possible
4. ffmpeg pour les deux vidéos
