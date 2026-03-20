#!/usr/bin/env python3
"""
compute_pk_janus.py
===================
Calcule et compare le spectre de puissance P(k) des runs Janus vs ΛCDM.

Analyses produites :
  1. P(k) séparé m− et m+ pour chaque snapshot analysé
  2. P(k) croisé m+/m− (corrélations entre les deux populations)
  3. Comparaison avec P(k) ΛCDM analytique (Bardeen 1986)
  4. Évolution temporelle P(k, z) si plusieurs snapshots fournis

Usage :
    # Snapshot unique
    python compute_pk_janus.py --snap snap_004895.bin --z 0.0

    # Plusieurs snapshots (évolution temporelle)
    python compute_pk_janus.py \
        --snaps snap_000500.bin snap_001500.bin snap_004895.bin \
        --zvals 3.39 1.634 0.0

    # Run complet V14 (sous-sélection automatique)
    python compute_pk_janus.py --snap_dir /mnt/T2/.../snapshots --every 200
"""

import struct, os, glob, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Chemins par défaut ───────────────────────────────────────────────
SNAP_DEFAULT = ('/mnt/T2/janus-sim/output/'
                'janus_v14_500Mpc_3M_kmin20/snapshots/snap_004895.bin')
OUT_DIR      = '/mnt/T2/janus-sim/output/pk_analysis'
TIME_CSV_V14 = ('/mnt/T2/janus-sim/output/'
                'janus_v14_500Mpc_3M_kmin20/time_series.csv')
BOX          = 500.0


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS VÉRIFIÉES — NE PAS MODIFIER
# ══════════════════════════════════════════════════════════════════════

def load_snapshot(path):
    """Format .bin : header u64 LE + N×28 bytes (7×float32)"""
    with open(path, 'rb') as f:
        n    = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    return (data[:, :3].astype(np.float64),
            data[:, 3:6].astype(np.float64),
            data[:, 6].astype(np.float64))


def compute_pk(pos, mass, G=256, box=500.0, sign=None):
    """
    Spectre de puissance P(k) par FFT 3D sur grille NGP.

    pos    : (N, 3) coords en [-box/2, +box/2] Mpc
    mass   : (N,)  mass_sign (+1 ou -1)
    G      : résolution grille (défaut 256)
    sign   : None=toutes, +1=m+, -1=m−

    Retourne : k_centers (Mpc⁻¹), P_k (Mpc³), N_modes
    """
    if sign is not None:
        mask = mass * sign > 0
        pos  = pos[mask]
        mass = mass[mask]

    N_part = len(pos)
    if N_part == 0:
        return np.array([]), np.array([]), np.array([])

    half = box / 2.0
    ix   = np.clip(((pos[:, 0] + half) / box * G).astype(np.int32), 0, G-1)
    iy   = np.clip(((pos[:, 1] + half) / box * G).astype(np.int32), 0, G-1)
    iz   = np.clip(((pos[:, 2] + half) / box * G).astype(np.int32), 0, G-1)

    grid   = np.zeros((G, G, G), dtype=np.float64)
    np.add.at(grid, (iz, iy, ix), 1.0)

    # Champ de contraste δ = (n - n̄) / n̄
    n_mean = N_part / G**3
    delta  = (grid - n_mean) / (n_mean + 1e-30)

    # FFT 3D + spectre
    delta_k = np.fft.fftn(delta)
    Pk_3d   = np.abs(delta_k)**2 * (box / G)**3 / box**3

    # Modules de k
    kf      = 2.0 * np.pi / box
    kmax    = np.pi * G / box
    freq    = np.fft.fftfreq(G, d=1.0/G).astype(np.int32)
    kx_1d   = freq * kf
    kx, ky, kz = np.meshgrid(kx_1d, kx_1d, kx_1d, indexing='ij')
    k_3d    = np.sqrt(kx**2 + ky**2 + kz**2)

    # Shot noise
    shot = box**3 / N_part

    # Bins logarithmiques
    n_bins  = 40
    k_edges = np.logspace(np.log10(kf), np.log10(kmax), n_bins + 1)

    k_centers = np.zeros(n_bins)
    Pk_binned = np.zeros(n_bins)
    N_modes   = np.zeros(n_bins, dtype=int)

    k_flat  = k_3d.flatten()
    Pk_flat = Pk_3d.flatten()

    for i in range(n_bins):
        in_bin = (k_flat >= k_edges[i]) & (k_flat < k_edges[i+1])
        if in_bin.sum() > 0:
            k_centers[i] = k_flat[in_bin].mean()
            Pk_binned[i] = Pk_flat[in_bin].mean() - shot
            N_modes[i]   = in_bin.sum()

    valid = (k_centers > 0) & (N_modes > 0)
    return k_centers[valid], Pk_binned[valid], N_modes[valid]


def compute_pk_cross(pos, mass, G=256, box=500.0):
    """
    Spectre de puissance croisé P_×(k) entre m+ et m−.
    P_× < 0 → anti-corrélation (m+ et m− occupent des régions opposées)
    P_× = 0 → populations décorrélées
    P_× > 0 → corrélation (rare dans Janus)
    """
    half   = box / 2.0
    N_part = len(pos)

    def make_delta(sign):
        msk  = mass * sign > 0
        ps   = pos[msk]
        ix   = np.clip(((ps[:,0]+half)/box*G).astype(int), 0, G-1)
        iy   = np.clip(((ps[:,1]+half)/box*G).astype(int), 0, G-1)
        iz   = np.clip(((ps[:,2]+half)/box*G).astype(int), 0, G-1)
        grid = np.zeros((G,G,G), np.float64)
        np.add.at(grid, (iz, iy, ix), 1.0)
        n_mean = msk.sum() / G**3
        return (grid - n_mean) / (n_mean + 1e-30)

    delta_p = np.fft.fftn(make_delta(+1))
    delta_m = np.fft.fftn(make_delta(-1))

    # Spectre croisé : Re(δ+* × δ−)
    cross_k = np.real(np.conj(delta_p) * delta_m) * (box/G)**3 / box**3

    kf    = 2.0 * np.pi / box
    kmax  = np.pi * G / box
    freq  = np.fft.fftfreq(G, d=1.0/G).astype(int)
    kx_1d = freq * kf
    kx, ky, kz = np.meshgrid(kx_1d, kx_1d, kx_1d, indexing='ij')
    k_3d  = np.sqrt(kx**2 + ky**2 + kz**2)

    n_bins  = 40
    k_edges = np.logspace(np.log10(kf), np.log10(kmax), n_bins + 1)
    k_flat  = k_3d.flatten()
    Pk_flat = cross_k.flatten()

    k_c, Pk_c, Nm = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, int)
    for i in range(n_bins):
        in_bin = (k_flat >= k_edges[i]) & (k_flat < k_edges[i+1])
        if in_bin.sum() > 0:
            k_c[i]  = k_flat[in_bin].mean()
            Pk_c[i] = Pk_flat[in_bin].mean()
            Nm[i]   = in_bin.sum()

    valid = (k_c > 0) & (Nm > 0)
    return k_c[valid], Pk_c[valid], Nm[valid]


def pk_lcdm_approx(k, z=0.0, h=0.678, Omega_m=0.308, Omega_b=0.048,
                   n_s=0.968, sigma8=0.815):
    """
    Spectre de puissance ΛCDM analytique (Bardeen 1986 + Sugiyama 1995).
    Précision ~10% par rapport aux codes Boltzmann (CLASS/CAMB).
    k en Mpc⁻¹, retourne P(k) en Mpc³.
    """
    Gamma = Omega_m * h * np.exp(-Omega_b * (1 + np.sqrt(2*h)/Omega_m))
    q     = k / (Gamma + 1e-30)
    T_k   = (np.log(1 + 2.34*q) / (2.34*q + 1e-30) *
             (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25))
    T_k   = np.where(k < 1e-6, 1.0, T_k)

    k_pivot = 0.05
    Pk_prim = (k / k_pivot)**n_s * T_k**2

    # Normalisation σ8
    R8  = 8.0 / h
    k_W = np.linspace(1e-4, 10.0, 5000)
    W_k = 3*(np.sin(k_W*R8) - k_W*R8*np.cos(k_W*R8)) / (k_W*R8 + 1e-30)**3
    q_W = k_W / Gamma
    T_W = (np.log(1+2.34*q_W)/(2.34*q_W+1e-30) *
           (1+3.89*q_W+(16.1*q_W)**2+(5.46*q_W)**3+(6.71*q_W)**4)**(-0.25))
    Pk_W           = (k_W/k_pivot)**n_s * T_W**2
    sigma2_unnorm  = np.trapezoid(k_W**2 * Pk_W * W_k**2, k_W) / (2*np.pi**2)
    norm           = sigma8**2 / (sigma2_unnorm + 1e-30)

    Dz = 1.0 / (1.0 + z)  # facteur de croissance approx
    return norm * Pk_prim * Dz**2


def parse_time_series(csv_path):
    step_z = {}
    try:
        with open(csv_path) as f:
            next(f)
            for line in f:
                p = line.strip().split(',')
                if len(p) >= 2:
                    try: step_z[int(p[0])] = float(p[1])
                    except: pass
    except FileNotFoundError:
        pass
    return step_z


# ══════════════════════════════════════════════════════════════════════
# SCRIPT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def process_snapshot(snap_path, z_val, G=256, label=None):
    """Calcule tous les spectres pour un snapshot donné."""
    print(f"\n{'─'*60}")
    print(f"Snapshot : {os.path.basename(snap_path)}  z={z_val:.3f}")
    t0 = time.time()

    pos, vel, mass = load_snapshot(snap_path)
    N_m = (mass < 0).sum()
    N_p = (mass > 0).sum()
    print(f"  N− = {N_m:,}   N+ = {N_p:,}")

    # P(k) individuel
    print("  Calcul P(k) m−...", end=' ', flush=True)
    k_m, Pk_m, Nm_m = compute_pk(pos, mass, G=G, box=BOX, sign=-1)
    print(f"  {len(k_m)} bins")

    print("  Calcul P(k) m+...", end=' ', flush=True)
    k_p, Pk_p, Nm_p = compute_pk(pos, mass, G=G, box=BOX, sign=+1)
    print(f"  {len(k_p)} bins")

    print("  Calcul P(k) total...", end=' ', flush=True)
    k_t, Pk_t, Nm_t = compute_pk(pos, mass, G=G, box=BOX, sign=None)
    print(f"  {len(k_t)} bins")

    print("  Calcul P_× croisé...", end=' ', flush=True)
    k_x, Pk_x, Nm_x = compute_pk_cross(pos, mass, G=G, box=BOX)
    print(f"  {len(k_x)} bins  ({time.time()-t0:.0f}s)")

    # Coefficient de corrélation r(k) = P_× / sqrt(P_m × P_p)
    # Interpoler sur k commun
    k_common = k_x
    Pk_m_interp = np.interp(k_common, k_m, Pk_m)
    Pk_p_interp = np.interp(k_common, k_p, Pk_p)
    r_k = Pk_x / np.sqrt(np.abs(Pk_m_interp * Pk_p_interp) + 1e-30)

    return {
        'z': z_val, 'label': label or f'z={z_val:.2f}',
        'N_m': N_m, 'N_p': N_p,
        'k_m': k_m, 'Pk_m': Pk_m,
        'k_p': k_p, 'Pk_p': Pk_p,
        'k_t': k_t, 'Pk_t': Pk_t,
        'k_x': k_x, 'Pk_x': Pk_x,
        'r_k': r_k,
    }


def make_figure(results, out_path):
    """Produit la figure comparative P(k) Janus vs ΛCDM."""

    k_ref = np.logspace(-3, np.log10(np.pi * 256 / BOX), 200)

    n_snaps = len(results)
    n_cols  = min(n_snaps, 3)
    n_rows  = 4  # P(k) m±, P_× croisé, r(k) corrélation, ratio ΛCDM

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6*n_cols, 4.5*n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle('Spectre de puissance P(k) — Simulation Janus vs ΛCDM\n'
                 f'BOX={BOX} Mpc  grille 256³', fontsize=12)

    colors_z = plt.cm.plasma(np.linspace(0.1, 0.9, max(n_snaps, 2)))

    for col, (res, cz) in enumerate(zip(results, colors_z)):
        z = res['z']

        # ── Ligne 0 : P(k) m− et m+ ──────────────────────────────
        ax = axes[0, col]
        if len(res['k_m']) > 0:
            ax.loglog(res['k_m'], np.abs(res['Pk_m']),
                      'b-o', ms=3, lw=1.8, label='m− (négatif)')
        if len(res['k_p']) > 0:
            ax.loglog(res['k_p'], np.abs(res['Pk_p']),
                      'r-s', ms=3, lw=1.8, label='m+ (positif)')
        # ΛCDM référence
        Pk_cdm = pk_lcdm_approx(k_ref, z=z)
        ax.loglog(k_ref, Pk_cdm, 'k--', lw=1.2, alpha=0.5, label='ΛCDM')
        ax.set_xlabel('k (Mpc⁻¹)'); ax.set_ylabel('P(k) (Mpc³)')
        ax.set_title(f'P(k) m± — z={z:.3f}', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # ── Ligne 1 : P_× croisé ─────────────────────────────────
        ax = axes[1, col]
        if len(res['k_x']) > 0:
            # Couleur selon le signe : rouge=positif, bleu=négatif
            Pk_x = res['Pk_x']
            k_x  = res['k_x']
            pos_mask = Pk_x >= 0
            neg_mask = Pk_x < 0
            if pos_mask.any():
                ax.loglog(k_x[pos_mask], Pk_x[pos_mask],
                          'go', ms=4, label='P_× > 0 (corrélé)')
            if neg_mask.any():
                ax.loglog(k_x[neg_mask], -Pk_x[neg_mask],
                          'rs', ms=4, label='|P_×| < 0 (anti-corrélé)')
            ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('k (Mpc⁻¹)'); ax.set_ylabel('|P_×(k)| (Mpc³)')
        ax.set_title(f'Spectre croisé m+/m− — z={z:.3f}\n'
                     'Anti-corrélation → ségrégation', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # ── Ligne 2 : Coefficient de corrélation r(k) ─────────────
        ax = axes[2, col]
        if len(res['k_x']) > 0 and 'r_k' in res:
            k_x = res['k_x']
            r_k = res['r_k']
            valid = k_x < 0.5
            ax.semilogx(k_x[valid], r_k[valid], 'purple', lw=2,
                        label='r(k) = P_×/√(P_m·P_p)')
            ax.axhline(-1, color='r', ls='--', lw=1.5, label='ségrégation parfaite')
            ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('k (Mpc⁻¹)'); ax.set_ylabel('r(k)')
        ax.set_title(f'Coefficient corrélation r(k) — z={z:.3f}', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
        ax.set_ylim(-1.1, 0.1)

        # ── Ligne 3 : Ratio Janus / ΛCDM ─────────────────────────
        ax = axes[3, col]
        if len(res['k_t']) > 0:
            # Couper à k_max = 0.5 Mpc⁻¹
            valid_ratio = res['k_t'] < 0.5
            k_plot = res['k_t'][valid_ratio]
            Pk_plot = res['Pk_t'][valid_ratio]
            Pk_cdm_interp = pk_lcdm_approx(k_plot, z=z)
            ratio = Pk_plot / (Pk_cdm_interp + 1e-30)
            ax.semilogx(k_plot, ratio, color=cz, lw=2,
                        label=f'P_Janus/P_ΛCDM')
            ax.axhline(1.0, color='k', ls='--', lw=1, label='ΛCDM = 1')
            ax.fill_between(k_plot,
                             ratio * 0.9, ratio * 1.1,
                             alpha=0.15, color=cz)

        ax.set_xlabel('k (Mpc⁻¹)')
        ax.set_ylabel('P_Janus(k) / P_ΛCDM(k)')
        ax.set_title(f'Ratio Janus/ΛCDM — z={z:.3f}', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
        ax.set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure → {out_path}")


def make_evolution_figure(results, out_path):
    """Figure d'évolution temporelle P(k, z)."""
    if len(results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Évolution temporelle P(k, z) — Janus vs ΛCDM', fontsize=12)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
    k_ref  = np.logspace(-3, np.log10(np.pi * 256 / BOX), 200)

    for res, c in zip(sorted(results, key=lambda r: r['z'], reverse=True),
                       colors):
        z  = res['z']
        lbl = f"z={z:.2f}"

        if len(res['k_m']) > 0:
            axes[0].loglog(res['k_m'], np.abs(res['Pk_m']),
                           color=c, lw=1.5, label=lbl)
        if len(res['k_x']) > 0:
            Px = res['Pk_x']
            kx = res['k_x']
            axes[1].semilogx(kx, Px, color=c, lw=1.5, label=lbl)
        if len(res['k_t']) > 0:
            valid = res['k_t'] < 0.5
            k_plot = res['k_t'][valid]
            ratio = res['Pk_t'][valid] / (pk_lcdm_approx(k_plot, z=z) + 1e-30)
            axes[2].semilogx(k_plot, ratio, color=c, lw=1.5, label=lbl)

    # ΛCDM z=0
    axes[0].loglog(k_ref, pk_lcdm_approx(k_ref, z=0),
                   'k--', lw=1.5, alpha=0.5, label='ΛCDM z=0')
    axes[0].set_title('P(k) m− — évolution en z', fontsize=10)
    axes[0].set_xlabel('k (Mpc⁻¹)'); axes[0].set_ylabel('P(k) (Mpc³)')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3, which='both')

    axes[1].axhline(0, color='k', lw=1, ls='--')
    axes[1].set_title('Spectre croisé P_× — anti-corrélation = ségrégation',
                       fontsize=10)
    axes[1].set_xlabel('k (Mpc⁻¹)'); axes[1].set_ylabel('P_×(k) (Mpc³)')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, which='both')

    axes[2].axhline(1, color='k', ls='--', lw=1)
    axes[2].set_title('Ratio P_Janus(k) / P_ΛCDM(k)', fontsize=10)
    axes[2].set_xlabel('k (Mpc⁻¹)'); axes[2].set_ylabel('Ratio')
    axes[2].set_ylim(0, 60)
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure évolution → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap',     default=SNAP_DEFAULT,
                        help='Snapshot unique')
    parser.add_argument('--snaps',    nargs='+', default=None,
                        help='Liste de snapshots')
    parser.add_argument('--zvals',    nargs='+', type=float, default=None,
                        help='Redshifts correspondants (même ordre)')
    parser.add_argument('--snap_dir', default=None,
                        help='Répertoire de snapshots (avec --every)')
    parser.add_argument('--every',    type=int, default=200,
                        help='1 snapshot sur N si --snap_dir')
    parser.add_argument('--grid',     type=int, default=256,
                        help='Résolution grille FFT (défaut 256)')
    parser.add_argument('--z',        type=float, default=0.0,
                        help='Redshift pour --snap unique')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Construction de la liste de snapshots ─────────────────────────
    snap_list = []

    if args.snap_dir:
        all_snaps = sorted(glob.glob(
            os.path.join(args.snap_dir, 'snap_*.bin')))
        all_snaps = all_snaps[::args.every]
        step_z    = parse_time_series(TIME_CSV_V14)
        for sp in all_snaps:
            try:
                step = int(os.path.basename(sp)
                           .replace('snap_','').replace('.bin',''))
                z    = step_z.get(step, 0.0)
            except:
                z = 0.0
            snap_list.append((sp, z))

    elif args.snaps:
        zvals = args.zvals or [0.0] * len(args.snaps)
        for sp, z in zip(args.snaps, zvals):
            snap_list.append((sp, z))

    else:
        snap_list.append((args.snap, args.z))

    print(f"Snapshots à analyser : {len(snap_list)}")
    print(f"Grille FFT : {args.grid}³")

    # ── Calcul ────────────────────────────────────────────────────────
    results = []
    for sp, z in snap_list:
        if not os.path.exists(sp):
            print(f"  SKIP (introuvable) : {sp}")
            continue
        res = process_snapshot(sp, z, G=args.grid)
        results.append(res)

    if not results:
        print("Aucun snapshot valide.")
        return

    # ── Figures ───────────────────────────────────────────────────────
    # Figure principale : jusqu'à 3 snapshots côte à côte
    for_main = results[:3]
    out1 = os.path.join(OUT_DIR, 'pk_comparison.png')
    make_figure(for_main, out1)

    # Figure évolution si plusieurs snapshots
    if len(results) > 1:
        out2 = os.path.join(OUT_DIR, 'pk_evolution.png')
        make_evolution_figure(results, out2)

    # ── Rapport texte ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RAPPORT P(k)")
    print(f"{'='*60}")
    for res in results:
        z = res['z']
        print(f"\nz = {z:.3f}  (N− = {res['N_m']:,}  N+ = {res['N_p']:,})")
        if len(res['k_m']) > 0 and len(res['k_p']) > 0:
            # Rapport P(k) Janus/ΛCDM à k=0.1 Mpc⁻¹
            k_ref_val = 0.1
            if len(res['k_t']) > 0:
                idx = np.argmin(np.abs(res['k_t'] - k_ref_val))
                Pk_cdm_ref = pk_lcdm_approx(
                    np.array([res['k_t'][idx]]), z=z)[0]
                ratio = res['Pk_t'][idx] / (Pk_cdm_ref + 1e-30)
                print(f"  P_Janus(k=0.1)/P_ΛCDM(k=0.1) = {ratio:.3f}")
        if len(res['k_x']) > 0:
            # Signe du spectre croisé à grande échelle
            large = res['k_x'] < 0.05
            if large.any():
                Px_mean = res['Pk_x'][large].mean()
                sign_str = "NÉGATIF (anti-corrélé = ségrégé)" \
                    if Px_mean < 0 else "POSITIF (corrélé)"
                print(f"  P_× à grande échelle (k<0.05) : {Px_mean:.1f} → {sign_str}")


if __name__ == '__main__':
    main()
