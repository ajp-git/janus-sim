#!/usr/bin/env python3
"""
Mesure le paramètre de spin λ sur tous les halos m− détectables.
Produit la distribution λ(halos_m−) vs λ_ΛCDM attendu.
"""
import struct, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, gaussian_filter

SNAP    = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/snapshots/snap_004895.bin'
BOX     = 500.0
OUT_DIR = '/mnt/T2/janus-sim/output/rotation_analysis'
os.makedirs(OUT_DIR, exist_ok=True)


def load_snapshot(path):
    with open(path, 'rb') as f:
        n    = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n*28), dtype=np.float32).reshape(n,7)
    return (data[:,:3].astype(np.float64),
            data[:,3:6].astype(np.float64),
            data[:,6].astype(np.float64))


def find_halo_centers(pos, mass, G=64, box=500.0,
                      min_particles=5000, smooth_sigma=1.5):
    """
    Identifie les centres de halos m− par FOF simplifié :
    1. Grille de densité G³
    2. Lissage gaussien
    3. Labellisation des régions > seuil
    4. Centre de masse de chaque région
    Retourne liste de (center, N_approx)
    """
    half = box / 2.0
    mask_m = mass < 0
    pos_m  = pos[mask_m]

    ix = np.clip(((pos_m[:,0]+half)/box*G).astype(int), 0, G-1)
    iy = np.clip(((pos_m[:,1]+half)/box*G).astype(int), 0, G-1)
    iz = np.clip(((pos_m[:,2]+half)/box*G).astype(int), 0, G-1)

    grid = np.zeros((G,G,G), dtype=np.float32)
    np.add.at(grid, (iz,iy,ix), 1.0)

    # Lissage + seuil
    smoothed = gaussian_filter(grid, sigma=smooth_sigma)
    threshold = np.percentile(smoothed[smoothed > 0], 85)
    binary    = smoothed > threshold

    # Labellisation des régions connexes
    labeled, n_labels = label(binary)
    print(f"  {n_labels} régions détectées au-dessus du seuil")

    centers = []
    for lbl in range(1, n_labels + 1):
        region = labeled == lbl
        n_voxels = region.sum()

        # Voxels de la région → coordonnées Mpc
        iz_r, iy_r, ix_r = np.where(region)
        x_mpc = (ix_r / G * box) - half
        y_mpc = (iy_r / G * box) - half
        z_mpc = (iz_r / G * box) - half
        center_mpc = np.array([x_mpc.mean(), y_mpc.mean(), z_mpc.mean()])

        # Compter les vraies particules dans une sphère R=40 Mpc
        dr = pos_m - center_mpc
        r  = np.sqrt((dr**2).sum(axis=1))
        N_real = (r < 40).sum()

        if N_real >= min_particles:
            centers.append((center_mpc, N_real))
            print(f"  Halo #{len(centers)} : centre=({center_mpc[0]+250:.0f}, "
                  f"{center_mpc[1]+250:.0f}, {center_mpc[2]+250:.0f}) Mpc  "
                  f"N_real={N_real:,}")

    return centers


def measure_lambda(pos, vel, mass, center, R=40.0, sign=-1):
    """
    Mesure λ pour un halo.
    Retourne (lambda, L_mag, N, sigma_v) ou None.
    """
    dr_all = pos - center
    r_all  = np.sqrt((dr_all**2).sum(axis=1))
    mask   = (mass * sign > 0) & (r_all < R)

    if mask.sum() < 500:
        return None

    pos_h = pos[mask];  vel_h = vel[mask]
    com   = pos_h.mean(axis=0)
    v_com = vel_h.mean(axis=0)
    dr    = pos_h - com
    dv    = vel_h - v_com

    N       = mask.sum()
    R_med   = np.percentile(np.linalg.norm(dr, axis=1), 50)
    V_rms   = np.sqrt(np.mean((dv**2).sum(axis=1)))
    L       = np.linalg.norm(np.cross(dr, dv).sum(axis=0))
    sigma_v = V_rms

    lam = L / (np.sqrt(2) * N * V_rms * R_med + 1e-30)
    return lam, L, N, sigma_v, R_med


# ── Main ──────────────────────────────────────────────────────────────
print("Chargement snapshot...")
pos, vel, mass = load_snapshot(SNAP)
print(f"  N = {len(mass):,}  N− = {(mass<0).sum():,}  N+ = {(mass>0).sum():,}")

print("\nDétection des halos m−...")
centers = find_halo_centers(pos, mass, G=64, min_particles=5000)
print(f"\n{len(centers)} halos m− détectés.")

if not centers:
    print("Aucun halo détecté — ajuster min_particles ou smooth_sigma")
    exit(1)

# Mesure λ pour chaque halo
results = []
print("\nMesure λ par halo :")
for i, (center, N_approx) in enumerate(centers):
    res = measure_lambda(pos, vel, mass, center, R=40.0, sign=-1)
    if res is not None:
        lam, L, N, sv, R_med = res
        results.append({
            'halo_id': i+1, 'center': center+250,  # retour en [0,500]
            'lambda': lam, 'L': L, 'N': N,
            'sigma_v': sv, 'R_med': R_med
        })
        print(f"  Halo {i+1:2d} : λ={lam:.4f}  N={N:,}  "
              f"σ_v={sv:.1f}  R_med={R_med:.1f} Mpc")
    else:
        print(f"  Halo {i+1:2d} : < 500 particules, skip")

if not results:
    print("Aucun résultat valide.")
    exit(1)

lambdas = np.array([r['lambda'] for r in results])
print(f"\n--- Distribution λ ({len(lambdas)} halos m−) ---")
print(f"  Médiane : {np.median(lambdas):.4f}")
print(f"  Moyenne : {np.mean(lambdas):.4f}")
print(f"  Std     : {np.std(lambdas):.4f}")
print(f"  Min/Max : {lambdas.min():.4f} / {lambdas.max():.4f}")
print(f"  ΛCDM attendu : 0.035")
print(f"  Rapport λ_Janus/λ_ΛCDM : {np.median(lambdas)/0.035:.2f}×")

# ── Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Distribution λ — Halos m− Janus V14 (snap z≈1.7)\n'
             f'N_halos={len(lambdas)}  λ_médian={np.median(lambdas):.4f}  '
             f'vs ΛCDM≈0.035', fontsize=11)

# Histogramme λ
ax = axes[0]
bins = np.linspace(0, max(0.25, lambdas.max()*1.1), 15)
ax.hist(lambdas, bins=bins, color='steelblue', alpha=0.8, edgecolor='white')
ax.axvline(0.035, color='red', lw=2, ls='--', label='ΛCDM typique (0.035)')
ax.axvline(0.05,  color='orange', lw=1.5, ls=':', label='ΛCDM max (~0.05)')
ax.axvline(np.median(lambdas), color='blue', lw=2,
           label=f'Médiane Janus ({np.median(lambdas):.3f})')
ax.set_xlabel('Paramètre de spin λ')
ax.set_ylabel('N halos')
ax.set_title('Distribution λ (halos m−)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# λ vs N_particules
ax2 = axes[1]
Ns  = np.array([r['N'] for r in results])
sc  = ax2.scatter(Ns, lambdas, c=lambdas, cmap='viridis',
                   s=80, alpha=0.8, edgecolors='white', lw=0.5)
plt.colorbar(sc, ax=ax2, label='λ')
ax2.axhline(0.035, color='red', lw=1.5, ls='--', label='ΛCDM')
ax2.set_xlabel('N particules m− dans R=40 Mpc')
ax2.set_ylabel('λ')
ax2.set_title('λ vs taille du halo', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'lambda_distribution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure → {out}")
