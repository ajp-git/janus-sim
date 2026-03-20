# Tâche : Pipeline vidéo 4K — Simulation Janus V14

## Contexte
Run V14 en cours : 3M particules, 500 Mpc, k_min=20, snapshots tous les 5 steps.
~1000 snapshots attendus → vidéo 4K fluide montrant l'évolution z=5→0
avec une caméra qui orbite lentement autour de la structure.

## Architecture générale

```
snapshots/snap_XXXXXX.bin
        ↓
render_3d_frame.py  (par snapshot)
        ↓
frames/frame_XXXXXX.png  (4K PNG)
        ↓
make_video.sh  (ffmpeg)
        ↓
janus_v14_3D_4K.mp4
```

---

## Fichier 1 : `scripts/render_utils_3d.py`

Bibliothèque commune. **Copier exactement — code vérifié.**

```python
"""render_utils_3d.py — Utilitaires rendu 3D orbital Janus"""
import struct
import numpy as np
from scipy.ndimage import gaussian_filter


# ══════════════════════════════════════════════════════════════════════
# BLOC 1 : Chargement snapshot
# ══════════════════════════════════════════════════════════════════════
def load_snapshot(path):
    """
    Format : header u64 LE (8 bytes) + N×28 bytes (7×float32)
    Champs : [x, y, z, vx, vy, vz, mass_sign]
    Coords : [-250, +250] Mpc  |  mass_sign : +1.0 ou -1.0
    """
    with open(path, 'rb') as f:
        n    = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    return (data[:, :3].astype(np.float64),   # pos
            data[:, 3:6].astype(np.float64),  # vel
            data[:, 6].astype(np.float64))    # mass_sign


# ══════════════════════════════════════════════════════════════════════
# BLOC 2 : Matrice de rotation caméra — vérifiée 3×
# ══════════════════════════════════════════════════════════════════════
def camera_rotation_matrix(azimuth_deg, elevation_deg):
    """
    Rotation caméra pour orbite autour de l'origine.
    azimuth  : tour autour de Z (0→360° = un tour complet)
    elevation: inclinaison (0=équateur, 30=légèrement au-dessus)

    Usage : pos_cam = pos @ R.T
    pos_cam[:, 0] = horizontal (écran X)
    pos_cam[:, 1] = vertical   (écran Y)
    pos_cam[:, 2] = profondeur (depth cuing)
    """
    phi   = np.radians(azimuth_deg)
    theta = np.radians(elevation_deg)

    # Rotation azimut autour de Z
    Rz = np.array([
        [ np.cos(phi), -np.sin(phi), 0.0],
        [ np.sin(phi),  np.cos(phi), 0.0],
        [ 0.0,          0.0,         1.0],
    ])

    # Rotation élévation autour de X (après azimut)
    Rx = np.array([
        [1.0, 0.0,              0.0           ],
        [0.0, np.cos(theta), -np.sin(theta)   ],
        [0.0, np.sin(theta),  np.cos(theta)   ],
    ])

    return Rx @ Rz   # shape (3, 3)


# ══════════════════════════════════════════════════════════════════════
# BLOC 3 : Trajectoire caméra — vérifiée 3×
# ══════════════════════════════════════════════════════════════════════
def camera_trajectory(frame_idx, total_frames,
                      n_rotations=2.0,
                      elev_mean=20.0,
                      elev_amp=10.0):
    """
    Orbite caméra fluide sur total_frames images.
    - azimuth  : 0 → n_rotations×360° (tours complets)
    - elevation: elev_mean ± elev_amp (oscillation sinusoïdale)

    Retourne (azimuth_deg, elevation_deg)
    """
    t         = frame_idx / max(total_frames - 1, 1)
    azimuth   = n_rotations * 360.0 * t
    elevation = elev_mean + elev_amp * np.sin(2.0 * np.pi * t)
    return azimuth, elevation


# ══════════════════════════════════════════════════════════════════════
# BLOC 4 : Normalisation par passe
# ══════════════════════════════════════════════════════════════════════
def norm_pass(arr, pct=99.8):
    """Normalise au percentile pct — rend chaque échelle spatiale visible."""
    v = arr[arr > 0]
    if len(v) == 0:
        return arr * 0.0
    vmax = np.percentile(v, pct)
    return np.clip(arr, 0, vmax) / (vmax + 1e-12)


# ══════════════════════════════════════════════════════════════════════
# BLOC 5 : Projection + splatting 3D — vérifiée 3×
# ══════════════════════════════════════════════════════════════════════
def project_and_splat_3d(pos, mass, azimuth_deg, elevation_deg,
                          S=2160, box=500.0,
                          sig_core_mpc=0.8,
                          sig_halo_mpc=4.0,
                          sig_glow_mpc=12.0):
    """
    1. Applique la rotation caméra aux positions
    2. Projection orthographique (écran XY du repère caméra)
    3. Depth cuing : Z_cam → poids lumineux
    4. 3 passes gaussiennes normalisées indépendamment

    Retourne (layer_m, layer_p) : deux canvas (S, S) float32
    """
    PX = S / box   # pixels par Mpc

    # Rotation caméra
    R       = camera_rotation_matrix(azimuth_deg, elevation_deg)
    pos_cam = pos @ R.T   # (N, 3) coords caméra

    # Projection orthographique → pixels
    half = box / 2.0
    ix = np.clip(
        ((pos_cam[:, 0] + half) / box * S).astype(np.int32), 0, S - 1)
    iy = np.clip(
        ((pos_cam[:, 1] + half) / box * S).astype(np.int32), 0, S - 1)

    # Depth cuing : particules en avant (grand Z_cam) = plus brillantes
    z     = pos_cam[:, 2]
    z_n   = (z - z.min()) / (z.max() - z.min() + 1e-10)
    depth = (0.25 + 0.75 * z_n).astype(np.float32)

    mask_m = mass < 0
    mask_p = mass > 0

    sig_c = max(sig_core_mpc * PX,  2.0)
    sig_h = max(sig_halo_mpc * PX,  8.0)
    sig_g = max(sig_glow_mpc * PX, 20.0)

    def make_layer(mask):
        grid = np.zeros((S, S), dtype=np.float32)
        np.add.at(grid, (iy[mask], ix[mask]), depth[mask])
        c = norm_pass(gaussian_filter(grid, sigma=sig_c, truncate=4.0))
        h = norm_pass(gaussian_filter(grid, sigma=sig_h, truncate=4.0))
        g = norm_pass(gaussian_filter(grid, sigma=sig_g, truncate=3.0))
        # Filaments : halo dominant
        return c * 0.7 + h * 0.8 + g * 0.4

    return make_layer(mask_m), make_layer(mask_p)


# ══════════════════════════════════════════════════════════════════════
# BLOC 6 : Tone mapping + assemblage frame 4K — vérifiée 3×
# ══════════════════════════════════════════════════════════════════════
def assemble_frame_4k(layer_m, layer_p,
                       W=3840, H=2160, S=2160, X_OFF=840):
    """
    Colorie m− (bleu-cyan) et m+ (orange-rouge),
    additive blending, ACES tone mapping, boost saturation,
    letterbox dans frame W×H.

    Retourne uint8 (H, W, 3) — prêt pour imageio / ffmpeg.
    """
    COLOR_M = np.array([0.10, 0.50, 1.00])
    COLOR_P = np.array([1.00, 0.30, 0.05])

    def to_rgb(d, color, exposure=1.3):
        if d.max() < 1e-10:
            return np.zeros((*d.shape, 3), np.float32)
        d = d / (d.max() + 1e-10) * exposure
        bloom = np.clip((d - 0.60) / 0.40, 0, 1)
        return np.stack([
            np.clip(color[c] * d + bloom, 0, 1).astype(np.float32)
            for c in range(3)
        ], axis=2)

    def aces(x):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

    rgb  = to_rgb(layer_m, COLOR_M) + to_rgb(layer_p, COLOR_P, exposure=1.6)
    rgb  = aces(np.clip(rgb, 0, None) * 1.4)
    luma = (0.2126 * rgb[:,:,0]
          + 0.7152 * rgb[:,:,1]
          + 0.0722 * rgb[:,:,2])
    for c in range(3):
        rgb[:,:,c] = np.clip(luma + 1.35 * (rgb[:,:,c] - luma), 0, 1)

    # Letterbox : canvas carré centré dans frame 16:9
    frame = np.zeros((H, W, 3), np.float32)
    frame[:, X_OFF:X_OFF + S] = np.flipud(rgb)   # flipud : origin lower→upper
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)
```

---

## Fichier 2 : `scripts/render_3d_frame.py`

Script principal — un snapshot → une frame PNG.

```python
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
```

---

## Fichier 3 : `scripts/render_batch_3d.py`

Orchestre le rendu de tous les snapshots en séquence.

```python
#!/usr/bin/env python3
"""
render_batch_3d.py — Rendu batch de tous les snapshots V14.
Lance render_3d_frame.py pour chaque snapshot dans l'ordre.
Reprend là où il s'est arrêté (skip des frames déjà rendues).

Usage : python render_batch_3d.py [--subsample 1500000] [--n_rotations 2]
"""
import os, glob, subprocess, sys, time, argparse

SNAP_DIR  = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/snapshots'
OUT_DIR   = '/mnt/T2/janus-sim/output/frames_3d'
SCRIPT    = '/mnt/T2/janus-sim/scripts/render_3d_frame.py'
PYTHON    = '/tmp/plotenv/bin/python'
TIME_CSV  = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/time_series.csv'

def parse_time_series(csv_path):
    """Retourne dict step → z depuis le time_series.csv"""
    step_z = {}
    try:
        with open(csv_path) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        step_z[int(parts[0])] = float(parts[1])
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return step_z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsample',   type=int, default=1_500_000,
                        help='N particules à rendre (défaut 1.5M pour performance)')
    parser.add_argument('--n_rotations', type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Liste des snapshots triée
    snaps = sorted(glob.glob(os.path.join(SNAP_DIR, 'snap_*.bin')))
    if not snaps:
        print(f"ERREUR : aucun snapshot dans {SNAP_DIR}")
        sys.exit(1)

    total = len(snaps)
    step_z = parse_time_series(TIME_CSV)
    print(f"Snapshots trouvés : {total}")
    print(f"Sous-échantillon  : {args.subsample:,} particules")
    print(f"Rotations caméra  : {args.n_rotations}×")
    print(f"ETA estimée       : {total * 40 / 3600:.1f}h "
          f"(~40s/frame sur CPU)")
    print()

    t_start = time.time()
    n_done  = 0

    for frame_idx, snap_path in enumerate(snaps):
        # Extraire le step depuis le nom de fichier (snap_001500.bin → 1500)
        snap_name = os.path.basename(snap_path)
        try:
            step = int(snap_name.replace('snap_', '').replace('.bin', ''))
        except ValueError:
            step = frame_idx * 5

        z_val = step_z.get(step, 0.0)

        out_path = os.path.join(OUT_DIR, f"frame_{frame_idx:06d}.png")
        if os.path.exists(out_path):
            continue  # reprise automatique

        cmd = [
            PYTHON, SCRIPT,
            '--snap',         snap_path,
            '--frame_idx',    str(frame_idx),
            '--total_frames', str(total),
            '--step',         str(step),
            '--z',            f"{z_val:.4f}",
            '--out_dir',      OUT_DIR,
            '--subsample',    str(args.subsample),
            '--n_rotations',  str(args.n_rotations),
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        n_done += 1

        if result.returncode != 0:
            print(f"ERREUR frame {frame_idx}:")
            print(result.stderr[-500:])
        else:
            elapsed = time.time() - t_start
            remaining = (total - frame_idx - 1) * (elapsed / max(n_done, 1))
            print(f"[{frame_idx+1:4d}/{total}] step={step:5d} z={z_val:.3f} "
                  f"{dt:.0f}s | ETA {remaining/3600:.1f}h")

    print(f"\nTerminé ! {n_done} frames rendues dans {OUT_DIR}")


if __name__ == '__main__':
    main()
```

---

## Fichier 4 : `scripts/make_video.sh`

Assemblage ffmpeg en MP4 4K publication.

```bash
#!/bin/bash
# make_video.sh — Assemble les frames PNG en vidéo MP4 4K
# Usage : bash make_video.sh [fps] [output_name]

FRAMES_DIR="/mnt/T2/janus-sim/output/frames_3d"
OUTPUT="/mnt/T2/janus-sim/output/janus_v14_3D_4K.mp4"
FPS="${1:-24}"
OUTPUT="${2:-$OUTPUT}"

# Vérifier ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installation ffmpeg..."
    apt-get install -y ffmpeg 2>/dev/null || \
    docker compose run --rm dev apt-get install -y ffmpeg
fi

# Compter les frames
N_FRAMES=$(ls "$FRAMES_DIR"/frame_*.png 2>/dev/null | wc -l)
echo "Frames trouvées : $N_FRAMES"
echo "FPS : $FPS"
echo "Durée : $(echo "$N_FRAMES / $FPS" | bc)s"
echo "Output : $OUTPUT"

# Encodage H264 qualité publication
# -crf 18 : quasi-lossless (0=parfait, 51=pire)
# -preset slow : meilleure compression
# -pix_fmt yuv420p : compatibilité maximale (YouTube, etc.)
ffmpeg -y \
    -framerate "$FPS" \
    -pattern_type glob \
    -i "${FRAMES_DIR}/frame_%06d.png" \
    -c:v libx264 \
    -crf 18 \
    -preset slow \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -vf "scale=3840:2160:flags=lanczos" \
    "$OUTPUT"

echo ""
echo "Vidéo générée : $OUTPUT"
echo "Taille : $(du -h "$OUTPUT" | cut -f1)"

# Version web légère (1080p, pour partage rapide)
OUTPUT_1080="${OUTPUT%.mp4}_1080p.mp4"
ffmpeg -y \
    -i "$OUTPUT" \
    -c:v libx264 \
    -crf 20 \
    -preset fast \
    -pix_fmt yuv420p \
    -vf "scale=1920:1080:flags=lanczos" \
    "$OUTPUT_1080"
echo "Version 1080p : $OUTPUT_1080"
```

---

## Ordre d'exécution

### Étape 1 — Installer Pillow si absent

```bash
/tmp/plotenv/bin/pip install Pillow --break-system-packages -q
```

### Étape 2 — Test sur un seul snapshot

```bash
/tmp/plotenv/bin/python /mnt/T2/janus-sim/scripts/render_3d_frame.py \
    --snap /mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/snapshots/snap_000500.bin \
    --frame_idx 100 \
    --total_frames 1000 \
    --step 500 \
    --z 3.39 \
    --out_dir /tmp/test_render \
    --subsample 500000
```

Vérifier que l'image est belle avant de lancer le batch.

### Étape 3 — Lancer le batch complet (en arrière-plan)

```bash
nohup /tmp/plotenv/bin/python \
    /mnt/T2/janus-sim/scripts/render_batch_3d.py \
    --subsample 1500000 \
    --n_rotations 2 \
    > /mnt/T2/janus-sim/output/render_log.txt 2>&1 &

echo "PID: $!"
tail -f /mnt/T2/janus-sim/output/render_log.txt
```

### Étape 4 — Assembler la vidéo

```bash
bash /mnt/T2/janus-sim/scripts/make_video.sh 24
```

---

## Paramètres à ajuster selon les résultats du test frame unique

| Paramètre | Défaut | Si image trop sombre | Si image surexposée |
|-----------|--------|---------------------|---------------------|
| sig_core_mpc | 0.8 | Augmenter → 1.5 | Diminuer → 0.5 |
| sig_halo_mpc | 4.0 | Augmenter → 8.0 | Diminuer → 2.0 |
| sig_glow_mpc | 12.0 | Augmenter → 20.0 | Diminuer → 8.0 |
| subsample | 1.5M | Augmenter → 3M | Diminuer → 500K |
| n_rotations | 2.0 | — | Réduire → 1.0 |

## Estimation temps et espace

| Ressource | Valeur |
|-----------|--------|
| Temps/frame (1.5M part.) | ~40-60s CPU |
| Frames totales | ~1000 |
| Temps total rendu | ~12-16h |
| Espace frames PNG | ~8 GB (8 MB/frame) |
| Vidéo MP4 4K finale | ~2-4 GB |
| Vidéo 1080p | ~500 MB |

Le batch reprend automatiquement si interrompu (skip des frames existantes).
