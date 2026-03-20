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
