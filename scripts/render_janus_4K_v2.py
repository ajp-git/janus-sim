"""
render_janus_4K_v2.py
=====================
Rendu astronomique 4K d'un snapshot Janus — version 2.
Corrections v2 :
  - figimage() pour remplissage 4K garanti (zero artefact matplotlib)
  - Normalisation adaptative par percentile (résiste à 15M particules)
  - Sigma adaptatif selon N (évite la sursaturation)
  - Annotations en coordonnées pixel directes

Deux sorties :
  - render_step{N}_clean.png
  - render_step{N}_trails.png

Usage :
    python render_janus_4K_v2.py --snap snapshot_1500.hdf5 --step 1500 --z 1.634
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os, time

# ── Résolution ───────────────────────────────────────────────────────
W, H   = 3840, 2160
DPI    = 100
BOX    = 500.0
HALO   = np.array([168.0, 127.0, 73.0])

# Champ de vue 16:9 — toute la boîte en X, bande centrale en Y
VIEW_X = (-10.0, 510.0)    # 520 Mpc en X ≈ boîte entière
VIEW_Y = (103.75, 396.25)  # 292.5 Mpc en Y (520/16*9 ≈ 292.5)

# Couleurs de base
COLOR_M = np.array([0.10, 0.50, 1.00])   # bleu cyan
COLOR_P = np.array([1.00, 0.30, 0.05])   # orange rouge vif

# Tone mapping ACES filmic
def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b)) / (x*(c*x+d)+e), 0, 1)


# ── Chargement ───────────────────────────────────────────────────────
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
        d = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
        return d[:,:3].astype(np.float64), d[:,3:6].astype(np.float64), d[:,6].astype(np.float64)


# ── Splatting gaussien multi-échelle ─────────────────────────────────
def splat_particles(px, py, pz, W, H, view_x, view_y,
                    sigma_fine, sigma_glow, depth_sigma=90.0):
    """
    Projette les particules sur grille (H×W) par gaussian splatting.
    Depth cuing Z : atténuation selon distance au plan médian (BOX/2).
    """
    # Coordonnées pixel
    ix = ((px - view_x[0]) / (view_x[1] - view_x[0]) * W).astype(np.float32)
    iy = ((py - view_y[0]) / (view_y[1] - view_y[0]) * H).astype(np.float32)

    # Depth cuing
    dz      = np.abs(pz - BOX / 2.0)
    depth_w = np.exp(-0.5 * (dz / depth_sigma)**2).astype(np.float32)

    # Masque dans le champ
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix_v  = ix[valid].astype(np.int32)
    iy_v  = iy[valid].astype(np.int32)
    w_v   = depth_w[valid]

    # Accumulation
    grid = np.zeros((H, W), dtype=np.float32)
    np.add.at(grid, (iy_v, ix_v), w_v)

    # Sigma adaptatif : évite sursaturation avec grandes N
    # sigma_fine est la valeur de base pour ~1M particules
    fine = gaussian_filter(grid, sigma=sigma_fine, truncate=3.0)
    glow = gaussian_filter(grid, sigma=sigma_glow,  truncate=3.0)

    return fine * 1.0 + glow * 0.35


# ── Normalisation robuste + RGB ───────────────────────────────────────
def density_to_rgb(density, base_color, clip_pct=99.5, exposure=1.0):
    """
    Normalisation au percentile clip_pct → pas de saturation.
    Interpolation couleur : noir → base_color → blanc (zones très denses).
    """
    d = density.copy().astype(np.float64)

    if (d > 0).any():
        vmax = np.percentile(d[d > 0], clip_pct)
        d = np.clip(d, 0, vmax) / (vmax + 1e-12)
    else:
        return np.zeros((density.shape[0], density.shape[1], 3), dtype=np.float32)

    d = d * exposure

    # Couleur de base + blanc aux hautes densités
    bloom = np.clip((d - 0.65) / 0.35, 0, 1)   # bloom vers blanc au-dessus de 65%
    r = base_color[0] * d + bloom
    g = base_color[1] * d + bloom
    b = base_color[2] * d + bloom

    rgb = np.stack([
        np.clip(r, 0, 1).astype(np.float32),
        np.clip(g, 0, 1).astype(np.float32),
        np.clip(b, 0, 1).astype(np.float32),
    ], axis=2)
    return rgb


# ── Traînées de vitesse ───────────────────────────────────────────────
def render_velocity_trails(px, py, vx, vy, speeds,
                           W, H, view_x, view_y,
                           trail_length=0.025, n_steps=10):
    """
    Traînées des m+ dans la direction opposée à leur vitesse.
    """
    v_scale   = view_x[1] - view_x[0]
    trail_mpc = trail_length * v_scale

    vmag  = np.sqrt(vx**2 + vy**2) + 1e-10
    vx_n  = vx / vmag
    vy_n  = vy / vmag

    v_ref     = np.percentile(vmag, 80)
    trail_len = np.clip(vmag / v_ref, 0.1, 1.0) * trail_mpc

    grid = np.zeros((H, W), dtype=np.float32)
    speed_norm = speeds / (speeds.max() + 1e-10)

    for s in range(1, n_steps + 1):
        t     = s / n_steps
        alpha = (1.0 - t)**1.5

        trail_x = px - vx_n * trail_len * t
        trail_y = py - vy_n * trail_len * t

        ix = ((trail_x - view_x[0]) / (view_x[1] - view_x[0]) * W)
        iy = ((trail_y - view_y[0]) / (view_y[1] - view_y[0]) * H)

        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        np.add.at(grid,
                  (iy[valid].astype(np.int32), ix[valid].astype(np.int32)),
                  alpha * speed_norm[valid])

    return gaussian_filter(grid, sigma=1.0, truncate=3.0)


# ── Annotations pixel-space via figimage ─────────────────────────────
def add_text_pixel(canvas_rgb, text, x_px, y_px, fontsize,
                   color=(255,255,255), alpha=0.9, bold=False):
    """
    Écrit du texte sur le canvas numpy en utilisant matplotlib
    dans une figure temporaire de la même taille.
    Retourne le canvas modifié.
    """
    # On utilise une figure dédiée aux annotations
    # (appelée UNE FOIS puis fermée)
    pass   # Voir ci-dessous, on utilise ax.text avec transform=fig.transFigure


# ── Rendu principal ───────────────────────────────────────────────────
def render(snap_path, step, z, with_trails=False):

    mode = 'TRAILS' if with_trails else 'CLEAN'
    print(f"\n{'='*55}")
    print(f"Rendu {mode} | step {step} | z={z} | {W}×{H}")
    t0 = time.time()

    pos, vel, mass = load_snapshot(snap_path)
    sign   = np.sign(mass)
    mask_m = sign < 0
    mask_p = sign > 0

    px_m, py_m, pz_m = pos[mask_m,0], pos[mask_m,1], pos[mask_m,2]
    px_p, py_p, pz_p = pos[mask_p,0], pos[mask_p,1], pos[mask_p,2]
    vx_p, vy_p       = vel[mask_p,0], vel[mask_p,1]
    speed_p = np.sqrt((vel[mask_p]**2).sum(axis=1))

    N_m, N_p = len(px_m), len(px_p)
    print(f"N− = {N_m:,}  N+ = {N_p:,}")

    # Sigma adaptatif : plus de particules → sigma plus large pour lisser
    ref_N    = 1_000_000
    scale    = np.sqrt(ref_N / max(N_m, 1))
    sig_fine = np.clip(1.0 * scale, 0.5, 3.0)
    sig_glow = np.clip(7.0 * scale, 4.0, 20.0)
    print(f"σ_fine={sig_fine:.2f}  σ_glow={sig_glow:.2f}")

    # ── Splatting ────────────────────────────────────────────────────
    print("Splatting m−...", end=' ', flush=True)
    dens_m = splat_particles(px_m, py_m, pz_m,
                             W, H, VIEW_X, VIEW_Y,
                             sig_fine, sig_glow)
    print(f"peak={dens_m.max():.2f}")

    print("Splatting m+...", end=' ', flush=True)
    # Boost m+ pour visibilité quand minoritaires
    # On boost plus si peu de m+ mais on plafonne
    ratio     = N_m / max(N_p, 1)
    boost_log = np.clip(np.log10(ratio + 1) * 2.5, 1.0, 8.0)
    dens_p = splat_particles(px_p, py_p, pz_p,
                             W, H, VIEW_X, VIEW_Y,
                             sig_fine * 1.5, sig_glow * 1.5)
    dens_p *= boost_log
    print(f"peak={dens_p.max():.2f}  boost×{boost_log:.1f}")

    # ── RGB ──────────────────────────────────────────────────────────
    print("RGB + composition...", end=' ', flush=True)
    rgb_m = density_to_rgb(dens_m, COLOR_M, clip_pct=99.5, exposure=1.3)
    rgb_p = density_to_rgb(dens_p, COLOR_P, clip_pct=99.5, exposure=1.5)

    rgb = rgb_m + rgb_p   # additive blending

    # ── Traînées ─────────────────────────────────────────────────────
    if with_trails and N_p > 0:
        print("\nTraînées m+...", end=' ', flush=True)
        trail = render_velocity_trails(
            px_p, py_p, vx_p, vy_p, speed_p,
            W, H, VIEW_X, VIEW_Y,
            trail_length=0.025, n_steps=10)
        t_max = trail.max() + 1e-10
        t_norm = trail / t_max
        # Couleur traînée : jaune chaud
        rgb[:,:,0] += 1.00 * t_norm * 0.7
        rgb[:,:,1] += 0.85 * t_norm * 0.7
        rgb[:,:,2] += 0.30 * t_norm * 0.7
        print(f"peak={trail.max():.4f}")

    # ── Tone mapping + saturation ────────────────────────────────────
    rgb = aces(rgb * 1.4)
    rgb = np.clip(rgb, 0, 1)

    # Boost saturation
    luma = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
    for c in range(3):
        rgb[:,:,c] = np.clip(luma + 1.35*(rgb[:,:,c]-luma), 0, 1)

    print(f"  done | range [{rgb.min():.3f}, {rgb.max():.3f}]")

    # ── Canvas uint8 ─────────────────────────────────────────────────
    # figimage attend (H, W, 3) uint8, origin='upper' (row 0 = haut)
    # Notre rgb est origin='lower' (row 0 = bas) → flip vertical
    canvas = (np.flipud(rgb) * 255).astype(np.uint8)

    # ── Figure + figimage ─────────────────────────────────────────────
    print("Figure 4K...", end=' ', flush=True)
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor='black')
    fig.figimage(canvas, xo=0, yo=0, origin='upper', zorder=0)

    # Annotations via axes full-frame en coordonnées normalisées [0,1]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.patch.set_alpha(0)   # transparent

    # Conversion Mpc → fraction [0,1] du frame
    def mx(mpc): return (mpc - VIEW_X[0]) / (VIEW_X[1] - VIEW_X[0])
    def my(mpc): return (mpc - VIEW_Y[0]) / (VIEW_Y[1] - VIEW_Y[0])

    # Cercle méga-halo r=60 Mpc
    theta = np.linspace(0, 2*np.pi, 361)
    hx = mx(HALO[0] + 60*np.cos(theta))
    hy = my(HALO[1] + 60*np.sin(theta))
    ax.plot(hx, hy, color='white', lw=0.7, alpha=0.30, ls='--', zorder=2)

    # Barre d'échelle 50 Mpc (bas gauche)
    sx0, sx1 = mx(8),  mx(58)
    sy       = my(VIEW_Y[0] + (VIEW_Y[1]-VIEW_Y[0])*0.04)
    ax.plot([sx0, sx1], [sy, sy], color='white', lw=2.0, alpha=0.75, zorder=2)
    ax.text((sx0+sx1)/2, sy + 0.012, '50 Mpc',
            color='white', fontsize=16, ha='center', va='bottom',
            alpha=0.70, fontfamily='monospace', zorder=2)

    # z = valeur (grand, haut gauche)
    z_str = f"z = {z:.3f}" if z is not None else f"step {step}"
    ax.text(0.018, 0.96, z_str,
            color='white', fontsize=68, alpha=0.93,
            fontfamily='monospace', fontweight='bold',
            va='top', transform=ax.transAxes, zorder=2)
    ax.text(0.018, 0.88,
            f"step {step}   N\u2212 = {N_m:,}   N+ = {N_p:,}",
            color='white', fontsize=19, alpha=0.58,
            fontfamily='monospace', va='top', transform=ax.transAxes, zorder=2)

    # Légende (bas droit)
    ax.plot([0.84], [0.07], 'o', color=COLOR_M, ms=9, alpha=0.9,
            transform=ax.transAxes, zorder=2)
    ax.text(0.852, 0.07, 'masse négative (m\u2212)',
            color='white', fontsize=15, va='center', alpha=0.72,
            transform=ax.transAxes, zorder=2)
    ax.plot([0.84], [0.04], 'o', color=COLOR_P, ms=9, alpha=0.9,
            transform=ax.transAxes, zorder=2)
    ax.text(0.852, 0.04, 'masse positive (m+)',
            color='white', fontsize=15, va='center', alpha=0.72,
            transform=ax.transAxes, zorder=2)

    if with_trails:
        ax.text(0.98, 0.96,
                'traînées \u2192 direction de fuite m+',
                color='#FFE066', fontsize=13, alpha=0.65,
                ha='right', va='top', transform=ax.transAxes, zorder=2)

    # Signature
    ax.text(0.994, 0.012,
            'Simulation Janus N-corps  \u2502  Petit (2014)',
            color='white', fontsize=11, alpha=0.30,
            ha='right', va='bottom', fontfamily='monospace',
            transform=ax.transAxes, zorder=2)

    suffix  = 'trails' if with_trails else 'clean'
    out_dir = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/renders'
    os.makedirs(out_dir, exist_ok=True)
    outfile = f"{out_dir}/render_v2_step{step:04d}_{suffix}_4K.png"
    fig.savefig(outfile, dpi=DPI, bbox_inches=None,
                facecolor='black', pil_kwargs={'compress_level': 1})
    plt.close(fig)
    print(f"OK → {outfile}  ({time.time()-t0:.1f}s)")
    return outfile


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap',        required=True)
    parser.add_argument('--step',        type=int,   default=1500)
    parser.add_argument('--z',           type=float, default=None)
    parser.add_argument('--clean_only',  action='store_true')
    parser.add_argument('--trails_only', action='store_true')
    args = parser.parse_args()

    if not args.trails_only:
        render(args.snap, args.step, args.z, with_trails=False)
    if not args.clean_only:
        render(args.snap, args.step, args.z, with_trails=True)
