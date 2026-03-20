#!/usr/bin/env python3
"""
render_3d_frame.py
Usage : python render_3d_frame.py --snap SNAP --frame_idx N --total_frames M
        [--out_dir DIR] [--subsample K]
"""
import sys, os, argparse, time
sys.path.insert(0, '/mnt/T2/janus-sim/scripts')
from render_utils_3d import (load_snapshot, camera_trajectory,
                              project_and_splat_3d, assemble_frame_4k)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Annotations ──────────────────────────────────────────────────────
def add_annotations(canvas_u8, azimuth, elevation, step, z_val,
                     N_m, N_p, W=3840, H=2160, X_OFF=840, S=2160):
    """Ajoute les textes sur le canvas numpy via matplotlib."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor='black')
    fig.figimage(canvas_u8, xo=0, yo=0, origin='upper', zorder=0)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off'); ax.patch.set_alpha(0)

    z_str = f"z = {z_val:.3f}"
    ax.text(X_OFF + 40, H - 55, z_str,
            color='white', fontsize=68, alpha=0.93,
            fontfamily='monospace', fontweight='bold', va='top', zorder=2)
    ax.text(X_OFF + 40, H - 145,
            f"step {step}   N\u2212 = {N_m:,}   N+ = {N_p:,}",
            color='white', fontsize=20, alpha=0.55,
            fontfamily='monospace', va='top', zorder=2)

    # Barre d'échelle 100 Mpc
    BOX = 500.0
    px_per_mpc = S / BOX
    sx0 = X_OFF + 40
    sx1 = sx0 + int(100 * px_per_mpc)
    sy  = 40
    ax.plot([sx0, sx1], [sy, sy], 'w-', lw=2.5, alpha=0.65, zorder=2)
    ax.text((sx0+sx1)//2, sy + 18, '100 Mpc',
            color='white', fontsize=16, ha='center', va='bottom',
            alpha=0.60, fontfamily='monospace', zorder=2)

    # Légende
    lx = X_OFF + S - 60
    ax.plot([lx], [100], 'o', color=[0.10,0.50,1.00], ms=10, alpha=0.9, zorder=2)
    ax.text(lx+18, 100, 'masse n\u00e9gative (m\u2212)',
            color='white', fontsize=15, va='center', alpha=0.72, zorder=2)
    ax.plot([lx], [55], 'o', color=[1.00,0.30,0.05], ms=10, alpha=0.9, zorder=2)
    ax.text(lx+18, 55, 'masse positive (m+)',
            color='white', fontsize=15, va='center', alpha=0.72, zorder=2)

    # Infos caméra (discret)
    ax.text(X_OFF + S - 10, H - 30,
            f'az={azimuth:.0f}°  el={elevation:.0f}°',
            color='white', fontsize=10, alpha=0.25,
            ha='right', va='top', fontfamily='monospace', zorder=2)

    ax.text(X_OFF + S - 10, 18,
            'Simulation Janus N-corps  \u2502  Petit (2014)',
            color='white', fontsize=11, alpha=0.25,
            ha='right', va='bottom', fontfamily='monospace', zorder=2)

    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches=None, facecolor='black',
                format='png', pil_kwargs={'compress_level': 1})
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return np.array(Image.open(buf))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap',         required=True)
    parser.add_argument('--frame_idx',    type=int, required=True)
    parser.add_argument('--total_frames', type=int, required=True)
    parser.add_argument('--step',         type=int, default=0)
    parser.add_argument('--z',            type=float, default=0.0)
    parser.add_argument('--out_dir',      default='/mnt/T2/janus-sim/output/frames_3d')
    parser.add_argument('--subsample',    type=int, default=0,
                        help='Sous-échantillon N particules (0=toutes)')
    parser.add_argument('--n_rotations',  type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"frame_{args.frame_idx:06d}.png")

    if os.path.exists(out_path):
        print(f"Existe déjà : {out_path}")
        return

    t0 = time.time()
    print(f"Frame {args.frame_idx}/{args.total_frames} | "
          f"step={args.step} z={args.z:.3f} | {args.snap}")

    # Chargement
    pos, vel, mass = load_snapshot(args.snap)
    N_m = (mass < 0).sum()
    N_p = (mass > 0).sum()

    # Sous-échantillonnage optionnel (pour accélérer)
    if args.subsample > 0 and len(mass) > args.subsample:
        rng = np.random.default_rng(args.frame_idx)
        idx = rng.choice(len(mass), args.subsample, replace=False)
        pos, mass = pos[idx], mass[idx]
        print(f"  Sous-échantillon : {args.subsample}/{N_m+N_p}")

    # Trajectoire caméra
    azimuth, elevation = camera_trajectory(
        args.frame_idx, args.total_frames,
        n_rotations=args.n_rotations)

    # Rendu 3D
    print(f"  az={azimuth:.1f}° el={elevation:.1f}°", end=' ', flush=True)
    layer_m, layer_p = project_and_splat_3d(
        pos, mass, azimuth, elevation,
        S=2160, box=500.0,
        sig_core_mpc=0.8, sig_halo_mpc=4.0, sig_glow_mpc=12.0)

    # Assemblage 4K + annotations
    canvas = assemble_frame_4k(layer_m, layer_p)
    canvas = add_annotations(canvas, azimuth, elevation,
                              args.step, args.z, N_m, N_p)

    # Sauvegarde
    from PIL import Image
    Image.fromarray(canvas).save(out_path, compress_level=1)
    print(f"→ {os.path.basename(out_path)}  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
