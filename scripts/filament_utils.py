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
