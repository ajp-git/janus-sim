"""
render_janus_4K_v3.py
=====================
Corrections v3 :
  - Canvas carré 2160×2160 (boîte complète 256 Mpc), letterbox 16:9
  - Sigma basé sur l'espacement physique réel des particules
  - Zéro coupure de la boîte, zéro étirement

Usage :
    python render_janus_4K_v3.py --snap snapshot.hdf5 --step 1500 --z 1.634
    python render_janus_4K_v3.py --snap snapshot.npy  --step 100  --z 4.631
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, time

# ── Dimensions ───────────────────────────────────────────────────────
W, H   = 3840, 2160        # frame 4K 16:9
S      = H                 # canvas carré = 2160 px (= 256 Mpc)
X_OFF  = (W - S) // 2     # 840 px de letterbox de chaque côté
DPI    = 100
BOX    = 500.0             # Mpc
MPX    = S / BOX           # pixels par Mpc = 8.4375
HALO   = np.array([168.0, 127.0, 73.0])   # À VÉRIFIER pour box 500 Mpc

# Couleurs
COLOR_M = np.array([0.10, 0.50, 1.00])   # bleu cyan
COLOR_P = np.array([1.00, 0.30, 0.05])   # orange rouge

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
                print("Clés HDF5 non standard :"); f.visit(print); raise
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


# ── Sigmas physiques fixes (indépendants de N) ───────────────────────
# BOX=500 Mpc, canvas S=2160 px → 1 Mpc = 4.32 px
#   SIG_CORE =  1 Mpc ~  4 px  → noyau compact des halos
#   SIG_HALO =  8 Mpc ~ 35 px  → étendue virialisée du halo
#   SIG_GLOW = 30 Mpc ~130 px  → nébuleuse / glow diffus inter-halos
PX_PER_MPC = S / BOX
SIG_CORE   = max(1.0  * PX_PER_MPC, 2.0)
SIG_HALO   = max(8.0  * PX_PER_MPC, 8.0)
SIG_GLOW   = max(30.0 * PX_PER_MPC, 20.0)


# ── Splatting gaussien 3 passes sur canvas carré ─────────────────────
def splat(px, py, pz, depth_sigma=150.0):
    """
    Projette sur canvas S×S. 3 passes gaussiennes à sigmas physiques fixes.
    Détection automatique du système de coordonnées :
      - Si coords en [0, BOX]    → pas de shift
      - Si coords en [-BOX/2, +BOX/2] → shift +BOX/2
    """
    # Détection coordonnées
    if px.min() < -BOX * 0.1:
        # Centré sur 0 → shift
        ix = ((px + BOX/2) / BOX * S).astype(np.float32)
        iy = ((py + BOX/2) / BOX * S).astype(np.float32)
        dz = np.abs(pz)
    else:
        # [0, BOX] → direct
        ix = (px / BOX * S).astype(np.float32)
        iy = (py / BOX * S).astype(np.float32)
        dz = np.abs(pz - BOX / 2.0)

    # Depth cuing
    depth_w = np.exp(-0.5 * (dz / depth_sigma)**2).astype(np.float32)

    valid = (ix >= 0) & (ix < S) & (iy >= 0) & (iy < S)
    grid  = np.zeros((S, S), dtype=np.float32)
    np.add.at(grid,
              (iy[valid].astype(np.int32), ix[valid].astype(np.int32)),
              depth_w[valid])

    # 3 passes : noyau + halo + glow
    core = gaussian_filter(grid, sigma=SIG_CORE, truncate=4.0)
    halo = gaussian_filter(grid, sigma=SIG_HALO, truncate=4.0)
    glow = gaussian_filter(grid, sigma=SIG_GLOW, truncate=3.0)

    pct_valid = valid.mean() * 100
    print(f"  σ={SIG_CORE:.0f}/{SIG_HALO:.0f}/{SIG_GLOW:.0f}px  "
          f"raw_peak={grid.max():.0f}  valid={pct_valid:.0f}%", end='')
    return core * 1.0 + halo * 0.5 + glow * 0.2


# ── Densité → RGB ─────────────────────────────────────────────────────
def to_rgb(density, color, clip_pct=99.5, exposure=1.0):
    d = density.copy().astype(np.float64)
    if (d > 0).any():
        vmax = np.percentile(d[d > 0], clip_pct)
        d = np.clip(d, 0, vmax) / (vmax + 1e-12)
    else:
        return np.zeros((S, S, 3), dtype=np.float32)
    d *= exposure
    bloom = np.clip((d - 0.60) / 0.40, 0, 1)
    rgb = np.stack([
        np.clip(color[0]*d + bloom, 0, 1),
        np.clip(color[1]*d + bloom, 0, 1),
        np.clip(color[2]*d + bloom, 0, 1),
    ], axis=2).astype(np.float32)
    return rgb


# ── Traînées de vitesse ───────────────────────────────────────────────
def trails(px, py, vx, vy, speeds, trail_mpc=5.0, n_steps=12):
    """Traînées dans la direction opposée à v. trail_mpc en Mpc."""
    vmag  = np.sqrt(vx**2 + vy**2) + 1e-10
    vx_n, vy_n = vx/vmag, vy/vmag
    v_ref     = np.percentile(vmag, 80)
    tlen      = np.clip(vmag / v_ref, 0.1, 1.0) * trail_mpc
    speed_n   = speeds / (speeds.max() + 1e-10)

    # Détection coordonnées
    coords_centered = px.min() < -BOX * 0.1
    shift = BOX/2 if coords_centered else 0.0

    grid = np.zeros((S, S), dtype=np.float32)
    for s in range(1, n_steps+1):
        t     = s / n_steps
        alpha = (1 - t)**1.5
        tx = ((px - vx_n*tlen*t + shift) / BOX * S)
        ty = ((py - vy_n*tlen*t + shift) / BOX * S)
        ok = (tx>=0)&(tx<S)&(ty>=0)&(ty<S)
        np.add.at(grid,
                  (ty[ok].astype(np.int32), tx[ok].astype(np.int32)),
                  alpha * speed_n[ok])
    return gaussian_filter(grid, sigma=1.5, truncate=4.0)


# ── Assemblage 4K letterbox ───────────────────────────────────────────
def assemble_4K(square_rgb):
    """
    Place le canvas carré S×S au centre du frame W×H.
    Bandes noires gauche et droite.
    square_rgb : (S, S, 3) float32 [0,1], origin=lower
    Retourne : (H, W, 3) uint8, origin=upper (pour figimage)
    """
    frame = np.zeros((H, W, 3), dtype=np.float32)
    # flipud : notre canvas est origin=lower, figimage veut origin=upper
    sq = np.flipud(square_rgb)
    frame[:, X_OFF:X_OFF+S, :] = sq
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


# ── Rendu ─────────────────────────────────────────────────────────────
def render(snap_path, step, z, with_trails=False):
    mode = 'TRAILS' if with_trails else 'CLEAN'
    print(f"\n{'='*60}")
    print(f"Rendu {mode} | step {step} | z={z} | {W}×{H} (canvas {S}×{S})")
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
    print(f"N− = {N_m:,}   N+ = {N_p:,}")

    # ── Splatting ────────────────────────────────────────────────────
    print("m−...", end=' ', flush=True)
    d_m = splat(px_m, py_m, pz_m)
    print(f" → peak={d_m.max():.3f}")

    print("m+...", end=' ', flush=True)
    boost = np.clip(np.log10(N_m/(N_p+1)+1)*2.5, 1.0, 8.0)
    d_p   = splat(px_p, py_p, pz_p) * boost
    print(f" → peak={d_p.max():.3f}  boost×{boost:.1f}")

    # ── RGB ──────────────────────────────────────────────────────────
    rgb = to_rgb(d_m, COLOR_M, clip_pct=99.5, exposure=1.3) \
        + to_rgb(d_p, COLOR_P, clip_pct=99.5, exposure=1.6)

    # ── Traînées ─────────────────────────────────────────────────────
    if with_trails and N_p > 0:
        print("Traînées m+...", end=' ', flush=True)
        tr    = trails(px_p, py_p, vx_p, vy_p, speed_p, trail_mpc=4.0)
        tr_n  = tr / (tr.max() + 1e-10)
        rgb[:,:,0] += 1.00 * tr_n * 0.75
        rgb[:,:,1] += 0.85 * tr_n * 0.75
        rgb[:,:,2] += 0.25 * tr_n * 0.75
        print(f" peak={tr.max():.4f}")

    # ── Tone mapping + saturation ────────────────────────────────────
    rgb  = aces(np.clip(rgb, 0, None) * 1.4)
    luma = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
    for c in range(3):
        rgb[:,:,c] = np.clip(luma + 1.35*(rgb[:,:,c]-luma), 0, 1)

    # ── Frame 4K letterbox ───────────────────────────────────────────
    canvas_u8 = assemble_4K(rgb)

    # ── Figure matplotlib + figimage ─────────────────────────────────
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor='black')
    fig.figimage(canvas_u8, xo=0, yo=0, origin='upper', zorder=0)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off'); ax.patch.set_alpha(0)

    # Détection système de coordonnées (même logique que splat)
    coords_centered = px_m.min() < -BOX * 0.1
    if coords_centered:
        def px_frame(mpc_x): return X_OFF + (mpc_x + BOX/2) / BOX * S
        def py_frame(mpc_y): return (mpc_y + BOX/2) / BOX * S
    else:
        def px_frame(mpc_x): return X_OFF + mpc_x / BOX * S
        def py_frame(mpc_y): return mpc_y / BOX * S

    # Cercle méga-halo r=60 Mpc
    theta = np.linspace(0, 2*np.pi, 361)
    ax.plot(px_frame(HALO[0] + 60*np.cos(theta)),
            py_frame(HALO[1] + 60*np.sin(theta)),
            color='white', lw=0.8, alpha=0.28, ls='--', zorder=2)

    # Barre d'échelle 50 Mpc (bas gauche du canvas)
    sx0 = px_frame(8);  sx1 = px_frame(58)
    sy  = py_frame(10)
    ax.plot([sx0, sx1], [sy, sy], 'w-', lw=2.5, alpha=0.7, zorder=2)
    ax.text((sx0+sx1)/2, sy+18, '50 Mpc',
            color='white', fontsize=16, ha='center', va='bottom',
            alpha=0.68, fontfamily='monospace', zorder=2)

    # z (grand, haut gauche)
    z_str = f"z = {z:.3f}" if z is not None else f"step {step}"
    ax.text(X_OFF + 40, H - 55, z_str,
            color='white', fontsize=72, alpha=0.93,
            fontfamily='monospace', fontweight='bold',
            va='top', zorder=2)
    ax.text(X_OFF + 40, H - 145,
            f"step {step}   N\u2212 = {N_m:,}   N+ = {N_p:,}",
            color='white', fontsize=20, alpha=0.55,
            fontfamily='monospace', va='top', zorder=2)

    # Légende
    lx = X_OFF + S - 60
    ax.plot([lx], [120], 'o', color=COLOR_M, ms=10, alpha=0.90, zorder=2)
    ax.text(lx+18, 120, 'masse négative (m\u2212)',
            color='white', fontsize=15, va='center', alpha=0.72, zorder=2)
    ax.plot([lx], [75], 'o', color=COLOR_P, ms=10, alpha=0.90, zorder=2)
    ax.text(lx+18, 75, 'masse positive (m+)',
            color='white', fontsize=15, va='center', alpha=0.72, zorder=2)

    if with_trails:
        ax.text(X_OFF + S - 20, H - 40,
                'traînées \u2192 direction de fuite m+',
                color='#FFE066', fontsize=13, alpha=0.60,
                ha='right', va='top', zorder=2)

    ax.text(X_OFF + S - 10, 18,
            'Simulation Janus N-corps  \u2502  Petit (2014)',
            color='white', fontsize=11, alpha=0.28,
            ha='right', va='bottom', fontfamily='monospace', zorder=2)

    suffix  = 'trails' if with_trails else 'clean'
    out_dir = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/renders'
    os.makedirs(out_dir, exist_ok=True)
    outfile = f"{out_dir}/render_v5_step{step:04d}_{suffix}_4K.png"
    fig.savefig(outfile, dpi=DPI, bbox_inches=None,
                facecolor='black', pil_kwargs={'compress_level': 1})
    plt.close(fig)
    print(f"→ {outfile}  ({time.time()-t0:.1f}s)")
    return outfile


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
