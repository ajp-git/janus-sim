"""
render_janus_4K.py
==================
Rendu astronomique 4K d'un snapshot Janus.
Deux sorties :
  - render_step{N}_clean.png  : rendu pur, style Hubble Space Telescope
  - render_step{N}_trails.png : avec traînées de vitesse sur les m+

Technique :
  - Gaussian splatting multi-échelle (σ fin + σ large = effet nébuleuse)
  - Depth cuing Z : atténuation des particules hors du plan médian
  - Additive blending logarithmique
  - Tone mapping ACES filmic
  - CPU only, numpy + scipy

Usage :
    python render_janus_4K.py --snap snapshot_1500.npy --step 1500 --z 1.634
    python render_janus_4K.py --snap snapshot_1500.hdf5 --step 1500 --z 1.634
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import os
import time

# ── Résolution et paramètres rendu ──────────────────────────────────
W, H   = 3840, 2160        # 4K
DPI    = 100               # figure matplotlib (W/DPI × H/DPI pouces)

BOX    = 500.0             # Mpc
HALO   = np.array([168.0, 127.0, 73.0])

# Champ de vue 16:9 — étendue X = 520 Mpc, Y = 520/1.778 = 292.5 Mpc
# Centré sur le milieu de la boîte (250, 250)
VIEW_X = (-10.0, 510.0)
VIEW_Y = (103.75, 396.25)

# Couleurs de base (RGB 0–1)
COLOR_M = np.array([0.15, 0.55, 1.00])   # bleu-cyan froid
COLOR_P = np.array([1.00, 0.35, 0.05])   # orange-rouge vif

# Tone mapping ACES (approximation filmic)
def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0, 1)


# ── Chargement ──────────────────────────────────────────────────────
def load_snapshot(path):
    """Format Janus: 8 bytes N (u64), then N × 28 bytes"""
    import struct
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.hdf5', '.h5']:
        import h5py
        with h5py.File(path, 'r') as f:
            try:
                pos  = f['PartType0/Coordinates'][:].astype(np.float64)
                vel  = f['PartType0/Velocities'][:].astype(np.float64)
                mass = f['PartType0/Masses'][:].astype(np.float64)
            except KeyError:
                print("Clés HDF5 non standard :")
                f.visit(print); raise
        return pos, vel, mass
    elif ext == '.npy':
        d = np.load(path)
        return d[:,:3].astype(np.float64), d[:,3:6].astype(np.float64), d[:,6].astype(np.float64)
    elif ext == '.bin':
        # Format Janus: 8-byte header (u64 N), then N×28 bytes
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            raw = f.read(n * 28)
            d = np.frombuffer(raw, dtype=np.float32).reshape(n, 7)
        return d[:,:3].astype(np.float64), d[:,3:6].astype(np.float64), d[:,6].astype(np.float64)
    else:
        d = np.fromfile(path, dtype=np.float32).reshape(-1,7)
        return d[:,:3].astype(np.float64), d[:,3:6].astype(np.float64), d[:,6].astype(np.float64)


# ── Splatting gaussien multi-échelle ────────────────────────────────
def splat_particles(px, py, pz, weights, W, H, view_x, view_y,
                    sigma_fine, sigma_glow, depth_sigma=40.0):
    """
    Projette les particules sur une grille 2D par Gaussian splatting.
    - Passe fine (σ_fine) : structure compacte, halos
    - Passe glow (σ_glow) : nébuleuse diffuse autour des structures denses
    - Depth cuing : atténuation selon distance Z au plan médian (Z=BOX/2)

    Retourne : array shape (H, W), valeurs positives
    """
    # Coordonnées pixel
    ix = ((px - view_x[0]) / (view_x[1] - view_x[0]) * W).astype(np.float32)
    iy = ((py - view_y[0]) / (view_y[1] - view_y[0]) * H).astype(np.float32)

    # Depth cuing : poids selon Z (plan médian = BOX/2 = 128)
    z_mid   = BOX / 2.0
    dz      = np.abs(pz - z_mid)
    depth_w = np.exp(-0.5 * (dz / depth_sigma)**2)   # Gaussienne sur Z
    w_total = weights * depth_w

    # Masque particules dans le champ
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix_v  = ix[valid].astype(np.int32)
    iy_v  = iy[valid].astype(np.int32)
    w_v   = w_total[valid]

    # Accumulation sur grille
    grid = np.zeros((H, W), dtype=np.float32)
    np.add.at(grid, (iy_v, ix_v), w_v)

    # Passe fine
    fine = gaussian_filter(grid, sigma=sigma_fine, truncate=3.0)

    # Passe glow (nébuleuse)
    glow = gaussian_filter(grid, sigma=sigma_glow, truncate=3.0)

    # Combinaison additive : fine dominant + glow ambiant
    return fine * 1.0 + glow * 0.4


# ── Canal RGB depuis densité ─────────────────────────────────────────
def density_to_rgb(density, base_color, exposure=1.0):
    """
    Convertit une carte de densité en canal RGB.
    - Normalisation logarithmique
    - Couleur de base pour les zones denses, blanc pour les zones très denses
    """
    eps   = density[density > 0].min() * 0.1 if (density > 0).any() else 1e-6
    d_log = np.log1p(density / (eps + 1e-10))
    d_norm = d_log / (d_log.max() + 1e-10)
    d_norm = d_norm * exposure

    # Interpolation couleur : noir → base_color → blanc
    r = base_color[0] * d_norm + 1.0 * np.clip(d_norm - 0.7, 0, 0.3) / 0.3
    g = base_color[1] * d_norm + 1.0 * np.clip(d_norm - 0.7, 0, 0.3) / 0.3
    b = base_color[2] * d_norm + 1.0 * np.clip(d_norm - 0.7, 0, 0.3) / 0.3

    return np.stack([
        np.clip(r, 0, 1).astype(np.float32),
        np.clip(g, 0, 1).astype(np.float32),
        np.clip(b, 0, 1).astype(np.float32)
    ], axis=2)


# ── Traînées de vitesse ──────────────────────────────────────────────
def render_velocity_trails(px, py, vx, vy, speeds,
                           W, H, view_x, view_y,
                           trail_length=0.03, n_steps=8):
    """
    Pour chaque m+, trace une traînée dans la direction opposée à la vitesse
    (effet motion blur physique — on voit d'où vient la particule).
    Plus la vitesse est élevée, plus la traînée est longue et brillante.

    trail_length : longueur max en fraction de la boîte
    n_steps      : nombre de points par traînée
    """
    # Normalisation de la vitesse pour longueur de traînée
    v_scale = view_x[1] - view_x[0]   # Mpc par pixel × width
    trail_mpc = trail_length * v_scale  # longueur max en Mpc

    # Direction normalisée de la vitesse (dans le plan XY)
    vmag = np.sqrt(vx**2 + vy**2) + 1e-10
    vx_n = vx / vmag
    vy_n = vy / vmag

    # Longueur proportionnelle à la vitesse (plafonnée)
    v_ref = np.percentile(vmag, 80)
    trail_len = np.clip(vmag / v_ref, 0.1, 1.0) * trail_mpc

    grid = np.zeros((H, W), dtype=np.float32)

    for s in range(1, n_steps + 1):
        t = s / n_steps             # 0→1 : bout de la traînée → particule
        alpha = (1 - t)**1.5        # plus brillant près de la particule

        # Position du point de traînée (derrière la particule)
        trail_x = px - vx_n * trail_len * t
        trail_y = py - vy_n * trail_len * t

        ix = ((trail_x - view_x[0]) / (view_x[1] - view_x[0]) * W)
        iy = ((trail_y - view_y[0]) / (view_y[1] - view_y[0]) * H)

        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix_v  = ix[valid].astype(np.int32)
        iy_v  = iy[valid].astype(np.int32)
        w_v   = alpha * speeds[valid] / (speeds.max() + 1e-10)

        np.add.at(grid, (iy_v, ix_v), w_v)

    return gaussian_filter(grid, sigma=1.2, truncate=3.0)


# ── Rendu final ──────────────────────────────────────────────────────
def render(snap_path, step, z, with_trails=False):

    print(f"\n{'='*55}")
    mode = 'TRAILS' if with_trails else 'CLEAN'
    print(f"Rendu {mode} | step {step} | z={z}")
    t0 = time.time()

    pos, vel, mass = load_snapshot(snap_path)
    sign   = np.sign(mass)
    mask_m = sign < 0
    mask_p = sign > 0

    # Coordonnées périodiquement normalisées
    px_m, py_m, pz_m = pos[mask_m,0], pos[mask_m,1], pos[mask_m,2]
    px_p, py_p, pz_p = pos[mask_p,0], pos[mask_p,1], pos[mask_p,2]
    vx_p, vy_p       = vel[mask_p,0], vel[mask_p,1]
    speed_p = np.sqrt((vel[mask_p]**2).sum(axis=1))

    N_m, N_p = len(px_m), len(px_p)
    print(f"N− = {N_m:,}  N+ = {N_p:,}")

    # ── Splatting m− ────────────────────────────────────────────────
    print("Splatting m−...", end=' ', flush=True)
    # Poids : tous égaux (1 particule = 1 unité de lumière)
    w_m = np.ones(N_m, dtype=np.float32)
    density_m = splat_particles(
        px_m, py_m, pz_m, w_m,
        W, H, VIEW_X, VIEW_Y,
        sigma_fine=1.5,    # structure compacte
        sigma_glow=10.0,   # nébuleuse diffuse
        depth_sigma=90.0   # scaled for 500 Mpc box
    )
    print(f"max={density_m.max():.3f}")

    # ── Splatting m+ ────────────────────────────────────────────────
    print("Splatting m+...", end=' ', flush=True)
    w_p = np.ones(N_p, dtype=np.float32)
    # Booster les m+ pour qu'elles soient visibles malgré leur faible nombre
    boost_p = max(1.0, N_m / (N_p * 5.0))
    density_p = splat_particles(
        px_p, py_p, pz_p, w_p * boost_p,
        W, H, VIEW_X, VIEW_Y,
        sigma_fine=2.0,    # légèrement plus diffus
        sigma_glow=15.0,   # glow plus large
        depth_sigma=90.0   # scaled for 500 Mpc box
    )
    print(f"max={density_p.max():.3f}")

    # ── Canaux RGB ───────────────────────────────────────────────────
    print("Conversion RGB...", end=' ', flush=True)
    # Exposition relative : m− domine, m+ visible
    exp_m = 1.2
    exp_p = 1.8

    rgb_m = density_to_rgb(density_m, COLOR_M, exposure=exp_m)
    rgb_p = density_to_rgb(density_p, COLOR_P, exposure=exp_p)
    print("OK")

    # ── Composition additive ─────────────────────────────────────────
    print("Composition...", end=' ', flush=True)
    rgb = rgb_m + rgb_p   # additive blending

    # ── Traînées de vitesse m+ (version avec trails) ─────────────────
    if with_trails and N_p > 0:
        print("\nTraînées m+...", end=' ', flush=True)
        trail_grid = render_velocity_trails(
            px_p, py_p, vx_p, vy_p, speed_p,
            W, H, VIEW_X, VIEW_Y,
            trail_length=0.025,
            n_steps=10
        )
        trail_max = trail_grid.max() + 1e-10
        trail_norm = trail_grid / trail_max
        # Traînées en blanc-jaune chaud
        trail_color = np.array([1.0, 0.9, 0.4])
        for c in range(3):
            rgb[:,:,c] += trail_color[c] * trail_norm * 0.8
        print(f"max={trail_grid.max():.4f}")

    # ── Tone mapping ACES ────────────────────────────────────────────
    print("Tone mapping...", end=' ', flush=True)
    rgb = aces(rgb * 1.5)   # exposer légèrement avant ACES
    rgb = np.clip(rgb, 0, 1)
    print("OK")

    # ── Légère augmentation saturation ──────────────────────────────
    # Convertir en HSV-like : booster S
    luma = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
    sat_boost = 1.3
    for c in range(3):
        rgb[:,:,c] = np.clip(luma + sat_boost*(rgb[:,:,c]-luma), 0, 1)

    print(f"Composition : min={rgb.min():.3f} max={rgb.max():.3f}")

    # ── Figure matplotlib 4K ─────────────────────────────────────────
    print("Rendu figure 4K...", end=' ', flush=True)
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor='black')
    ax  = fig.add_axes([0, 0, 1, 1])   # plein cadre, zéro marge
    ax.set_facecolor('black')
    ax.imshow(rgb, origin='lower', aspect='auto',
              extent=[VIEW_X[0], VIEW_X[1], VIEW_Y[0], VIEW_Y[1]],
              interpolation='lanczos')
    ax.axis('off')

    # ── Cercle du méga-halo ──────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 360)
    ax.plot(HALO[0] + 60*np.cos(theta),
            HALO[1] + 60*np.sin(theta),
            color='white', lw=0.6, alpha=0.35, ls='--')

    # ── Barre d'échelle 50 Mpc ───────────────────────────────────────
    scale_x0  = VIEW_X[0] + 10
    scale_x1  = scale_x0 + 50
    scale_y   = VIEW_Y[0] + 10
    ax.plot([scale_x0, scale_x1], [scale_y, scale_y],
            color='white', lw=1.5, alpha=0.7)
    ax.text((scale_x0+scale_x1)/2, scale_y+3, '50 Mpc',
            color='white', fontsize=18, ha='center', va='bottom',
            alpha=0.7, fontfamily='monospace')

    # ── Texte principal : z et step ──────────────────────────────────
    z_str = f"z = {z:.3f}" if z is not None else f"step {step}"
    ax.text(VIEW_X[0]+8, VIEW_Y[1]-12, z_str,
            color='white', fontsize=72, alpha=0.92,
            fontfamily='monospace', fontweight='bold',
            va='top')
    ax.text(VIEW_X[0]+8, VIEW_Y[1]-28,
            f"step {step}  |  N\u2212 = {N_m:,}  N+ = {N_p:,}",
            color='white', fontsize=20, alpha=0.60,
            fontfamily='monospace', va='top')

    # ── Légende minimaliste ──────────────────────────────────────────
    legend_x = VIEW_X[1] - 55
    legend_y  = VIEW_Y[0] + 8
    ax.plot([legend_x], [legend_y + 6], 'o',
            color=COLOR_M, ms=8, alpha=0.9)
    ax.text(legend_x + 4, legend_y + 6, 'masse négative (m\u2212)',
            color='white', fontsize=16, va='center', alpha=0.75)
    ax.plot([legend_x], [legend_y], 'o',
            color=COLOR_P, ms=8, alpha=0.9)
    ax.text(legend_x + 4, legend_y, 'masse positive (m+)',
            color='white', fontsize=16, va='center', alpha=0.75)

    if with_trails:
        ax.text(VIEW_X[1]-55, VIEW_Y[1]-12,
                'traînées = direction de fuite m+',
                color='#FFE066', fontsize=14, alpha=0.6,
                ha='right', va='top')

    # ── Signature ────────────────────────────────────────────────────
    ax.text(VIEW_X[1]-4, VIEW_Y[0]+4,
            'Simulation Janus N-corps  |  modèle Petit (2014)',
            color='white', fontsize=12, alpha=0.35,
            ha='right', va='bottom', fontfamily='monospace')

    suffix = 'trails' if with_trails else 'clean'
    out_dir = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/renders'
    os.makedirs(out_dir, exist_ok=True)
    outfile = f"{out_dir}/render_step{step:04d}_{suffix}_4K.png"
    fig.savefig(outfile, dpi=DPI, bbox_inches=None,
                pad_inches=0, facecolor='black',
                pil_kwargs={'compress_level': 1})
    plt.close(fig)
    print(f"OK → {outfile}  ({time.time()-t0:.1f}s)")
    return outfile


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap',  required=True)
    parser.add_argument('--step',  type=int,   default=1500)
    parser.add_argument('--z',     type=float, default=None)
    parser.add_argument('--clean_only',  action='store_true')
    parser.add_argument('--trails_only', action='store_true')
    args = parser.parse_args()

    if not args.trails_only:
        render(args.snap, args.step, args.z, with_trails=False)
    if not args.clean_only:
        render(args.snap, args.step, args.z, with_trails=True)
