#!/usr/bin/env python3
"""
analyse_rotation.py
===================
Analyse de la rotation du méga-halo m− Janus V14.
Halo principal : centre ≈ (117, 133, 102) Mpc, R=50 Mpc

4 analyses :
  1. Évolution de L (direction + amplitude) sur tous les snapshots
  2. Profil de vitesse tangentielle v_tan(r) par couronne
  3. Carte 2D de vitesse tangentielle dans le plan de rotation
  4. Paramètre de spin λ (Bullock 2001)

Usage :
    python analyse_rotation.py
    python analyse_rotation.py --center 117 133 102 --R 50 --snap_dir /path/
"""

import struct, os, glob, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── Configuration ────────────────────────────────────────────────────
SNAP_DIR  = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/snapshots'
TIME_CSV  = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/time_series.csv'
OUT_DIR   = '/mnt/T2/janus-sim/output/rotation_analysis'
BOX       = 500.0

# Halo m− principal (coordonnées [-250, +250] Mpc)
HALO_CENTER = np.array([117.0, 133.0, 102.0]) - 250.0  # → [-133, -117, -148]
HALO_R      = 50.0   # Mpc


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS VÉRIFIÉES — NE PAS MODIFIER
# ══════════════════════════════════════════════════════════════════════

def load_snapshot(path):
    """Format : header u64 LE (8b) + N×28b (7×float32)"""
    with open(path, 'rb') as f:
        n    = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    return (data[:, :3].astype(np.float64),
            data[:, 3:6].astype(np.float64),
            data[:, 6].astype(np.float64))


def select_halo(pos, vel, mass, center, R, sign=-1):
    """
    Sélectionne les particules d'un halo, calcule COM et moment angulaire.
    Retourne dict avec dr, dv, L, L_hat, L_mag, com, v_com, N, sigma_v
    ou None si < 10 particules.
    """
    dr_all = pos - center
    r_all  = np.sqrt((dr_all**2).sum(axis=1))
    mask   = (mass * sign > 0) & (r_all < R)

    if mask.sum() < 10:
        return None

    pos_h = pos[mask]
    vel_h = vel[mask]

    com   = pos_h.mean(axis=0)
    v_com = vel_h.mean(axis=0)
    dr    = pos_h - com
    dv    = vel_h - v_com

    L     = np.cross(dr, dv).sum(axis=0)
    L_mag = np.linalg.norm(L)
    L_hat = L / (L_mag + 1e-30)
    sigma_v = np.sqrt(np.mean((dv**2).sum(axis=1)))

    return {
        'dr': dr, 'dv': dv,
        'L': L, 'L_hat': L_hat, 'L_mag': L_mag,
        'com': com, 'v_com': v_com,
        'N': mask.sum(), 'sigma_v': sigma_v
    }


def track_center_shrinking_sphere(pos, mass, center_init,
                                   R_init=50.0, sign=-1,
                                   n_iter=5, shrink=0.7):
    """
    Shrinking sphere : raffine le centre du halo par itérations.
    À chaque itération, calcule le COM dans la sphère,
    puis réduit R de shrink%.
    Converge vers le noyau dense du halo.

    Retourne le centre raffiné (3,) ou center_init si < 20 particules.
    """
    center = center_init.copy()
    R = R_init

    for i in range(n_iter):
        dr   = pos - center
        r    = np.sqrt((dr**2).sum(axis=1))
        mask = (mass * sign > 0) & (r < R)

        if mask.sum() < 20:
            break   # trop peu de particules → arrêter

        # Nouveau centre = COM des particules dans la sphère
        center = pos[mask].mean(axis=0)
        R *= shrink   # réduire la sphère

    return center


def decompose_velocities(dr, dv, L_hat):
    """
    Décompose les vitesses en (v_rad, v_tan, v_ax, r_perp).
    v_tan > 0 = prograde (même sens que L)
    v_tan < 0 = rétrograde
    """
    # Composante axiale de dr
    dr_ax   = np.einsum('ij,j->i', dr, L_hat)[:, None] * L_hat[None, :]
    dr_perp = dr - dr_ax
    r_perp  = np.linalg.norm(dr_perp, axis=1)

    # Vecteur radial dans le plan équatorial
    r_hat_perp = dr_perp / (r_perp[:, None] + 1e-10)

    # Vecteur tangentiel : L_hat × r_hat_perp
    t_hat = np.cross(L_hat[None, :], r_hat_perp)   # (N, 3)

    v_rad = np.einsum('ij,ij->i', dv, r_hat_perp)
    v_tan = np.einsum('ij,ij->i', dv, t_hat)
    v_ax  = np.einsum('ij,j->i',  dv, L_hat)

    return v_rad, v_tan, v_ax, r_perp


def rotation_profile(r_perp, v_tan, v_rad, n_bins=12, r_max=None):
    """Profil radial de v_tan et v_rad par couronne."""
    if r_max is None:
        r_max = np.percentile(r_perp, 95)
    edges  = np.linspace(0, r_max, n_bins + 1)
    r_mid  = 0.5 * (edges[:-1] + edges[1:])
    vt_m   = np.full(n_bins, np.nan)
    vt_s   = np.full(n_bins, np.nan)
    vr_m   = np.full(n_bins, np.nan)
    n_b    = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        m = (r_perp >= edges[i]) & (r_perp < edges[i+1])
        if m.sum() > 2:
            vt_m[i] = v_tan[m].mean()
            vt_s[i] = v_tan[m].std()
            vr_m[i] = v_rad[m].mean()
            n_b[i]  = m.sum()

    return r_mid, vt_m, vt_s, vr_m, n_b


def velocity_map_2d(dr, dv, L_hat, S=256, r_max=50.0):
    """
    Carte 2D de v_tan dans le plan ⊥ L_hat.
    Retourne (v_map S×S, e1, e2) où e1,e2 sont les axes du plan de rotation.
    """
    _, v_tan, _, _ = decompose_velocities(dr, dv, L_hat)

    # Base orthonormée dans le plan ⊥ L_hat
    ref = np.array([0., 0., 1.])
    if abs(np.dot(L_hat, ref)) > 0.9:
        ref = np.array([1., 0., 0.])
    e1 = np.cross(L_hat, ref);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(L_hat, e1);   e2 /= np.linalg.norm(e2)

    x2d = np.einsum('ij,j->i', dr, e1)
    y2d = np.einsum('ij,j->i', dr, e2)

    ix = np.clip(((x2d + r_max) / (2*r_max) * S).astype(int), 0, S-1)
    iy = np.clip(((y2d + r_max) / (2*r_max) * S).astype(int), 0, S-1)

    v_sum = np.zeros((S, S), dtype=np.float64)
    n_map = np.zeros((S, S), dtype=np.int32)
    np.add.at(v_sum, (iy, ix), v_tan)
    np.add.at(n_map, (iy, ix), 1)

    v_map = np.where(n_map > 0, v_sum / (n_map + 1e-10), np.nan)
    return v_map, e1, e2


def spin_parameter(dr, dv):
    """
    Paramètre de spin λ = |L| / (√2 × N × V_rms × R_med)
    Halos ΛCDM typiques : λ ≈ 0.03-0.05
    """
    N   = len(dr)
    R   = np.percentile(np.linalg.norm(dr, axis=1), 50)
    V   = np.sqrt(np.mean((dv**2).sum(axis=1)))
    L   = np.linalg.norm(np.cross(dr, dv).sum(axis=0))
    lam = L / (np.sqrt(2) * N * V * R + 1e-30)
    return lam, L, N, V, R


# ══════════════════════════════════════════════════════════════════════
# ANALYSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--center', type=float, nargs=3,
                        default=[117., 133., 102.])
    parser.add_argument('--R',       type=float, default=50.0)
    parser.add_argument('--snap_dir', default=SNAP_DIR)
    parser.add_argument('--every',   type=int, default=5,
                        help='Analyser 1 snapshot sur N (défaut 5)')
    parser.add_argument('--track', action='store_true',
                        help='Activer le tracking dynamique du centre (shrinking sphere)')
    args = parser.parse_args()

    center_raw = np.array(args.center)
    # Détecter si les coords sont en [0,BOX] ou [-BOX/2,+BOX/2]
    # Les snapshots V14 sont en [-250,+250] d'après les tests
    # Le centre fourni est en [0,500] → convertir
    center = center_raw - BOX/2.0
    R      = args.R

    os.makedirs(OUT_DIR, exist_ok=True)

    # Liste des snapshots
    snaps = sorted(glob.glob(os.path.join(args.snap_dir, 'snap_*.bin')))
    snaps = snaps[::args.every]   # sous-sélection

    if not snaps:
        print(f"ERREUR : aucun snapshot dans {args.snap_dir}")
        return

    step_z = parse_time_series(TIME_CSV)
    print(f"Halo centre : {center_raw} Mpc → coords sim : {center}")
    print(f"Rayon       : {R} Mpc")
    print(f"Snapshots   : {len(snaps)} (1/{args.every})")
    print()

    # ── Collecte des métriques sur tous les snapshots ─────────────────
    records = []   # liste de dicts
    L_hats  = []   # pour l'analyse de direction
    L_ref   = None # référence pour précession (premier step avec N>50000)
    current_center = center.copy()  # pour tracking dynamique

    for snap_path in snaps:
        snap_name = os.path.basename(snap_path)
        try:
            step = int(snap_name.replace('snap_','').replace('.bin',''))
        except:
            continue
        z = step_z.get(step, np.nan)

        t0 = time.time()
        pos, vel, mass = load_snapshot(snap_path)

        # Tracking dynamique du centre si activé
        if args.track:
            current_center = track_center_shrinking_sphere(
                pos, mass, current_center, R_init=R, sign=-1)

        halo = select_halo(pos, vel, mass, current_center, R, sign=-1)

        if halo is None:
            print(f"step {step:5d} z={z:.3f} : < 10 particules, skip")
            continue

        v_rad, v_tan, v_ax, r_perp = decompose_velocities(
            halo['dr'], halo['dv'], halo['L_hat'])

        r_m, vt_m, vt_s, vr_m, n_b = rotation_profile(
            r_perp, v_tan, v_rad, n_bins=10, r_max=R*0.9)

        lam, L_val, N_lam, V_lam, R_lam = spin_parameter(
            halo['dr'], halo['dv'])

        # Définir L_ref au premier step avec N > 50000
        if L_ref is None and halo['N'] > 50000:
            L_ref = halo['L_hat'].copy()
            print(f"  → L_ref défini à step={step} z={z:.3f} (N={halo['N']:,})")

        records.append({
            'step': step, 'z': z,
            'N': halo['N'],
            'L_mag': halo['L_mag'],
            'L_hat': halo['L_hat'].copy(),
            'sigma_v': halo['sigma_v'],
            'lam': lam,
            'vt_mean': np.nanmean(vt_m),
            'vt_outer': np.nanmean(vt_m[7:]),   # couronne externe
            'vr_mean': np.nanmean(vr_m),
            'r_mid': r_m, 'vt_profile': vt_m, 'vt_std': vt_s,
        })
        L_hats.append(halo['L_hat'])
        dt = time.time() - t0
        print(f"step {step:5d} z={z:.3f} | N={halo['N']:,} "
              f"| |L|={halo['L_mag']:.2e} | λ={lam:.4f} "
              f"| v_tan={np.nanmean(vt_m):+.1f} | {dt:.1f}s")

    if not records:
        print("Aucun résultat — vérifier le centre du halo.")
        return

    print(f"\n{len(records)} snapshots analysés.")

    # ── Figure 1 : Évolution temporelle ──────────────────────────────
    steps   = np.array([r['step']   for r in records])
    zvals   = np.array([r['z']      for r in records])
    L_mags  = np.array([r['L_mag']  for r in records])
    sigmas  = np.array([r['sigma_v'] for r in records])
    lams    = np.array([r['lam']    for r in records])
    vt_means = np.array([r['vt_mean'] for r in records])
    Ns      = np.array([r['N']      for r in records])

    # Direction de L : angle par rapport à L_ref (premier step avec N>50000)
    L_hats_arr = np.array(L_hats)
    if L_ref is None:
        L_ref = L_hats_arr[0]  # fallback si aucun step n'a N>50000
        print("  ⚠ L_ref non défini (aucun step avec N>50000), utilisation L[0]")
    cos_angle = np.clip([np.dot(L_ref, lh) for lh in L_hats_arr], -1, 1)
    angle_deg  = np.degrees(np.arccos(cos_angle))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    track_str = " [tracking ON]" if args.track else ""
    fig.suptitle(f'Analyse rotation halo m− Janus V14{track_str}\n'
                 f'Centre={center_raw} Mpc  R={R} Mpc', fontsize=12)

    ax = axes[0, 0]
    ax.semilogy(zvals[::-1], L_mags, 'b-o', ms=4, lw=1.8)
    ax.set_xlabel('z'); ax.set_ylabel('|L| (moment angulaire)')
    ax.set_title('|L| vs redshift', fontsize=9)
    ax.invert_xaxis(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(zvals[::-1], angle_deg, 'r-o', ms=4, lw=1.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('z'); ax.set_ylabel('Angle L vs L_ref [°]')
    ax.set_title('Direction axe rotation vs L_ref\n(L_ref = premier step N>50000)', fontsize=9)
    ax.invert_xaxis(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(zvals[::-1], lams, 'g-o', ms=4, lw=1.8)
    ax.axhline(0.035, color='k', ls='--', lw=1, label='λ=0.035 (ΛCDM typique)')
    ax.axhline(0.05,  color='k', ls=':', lw=1, label='λ=0.05')
    ax.set_xlabel('z'); ax.set_ylabel('Paramètre de spin λ')
    ax.set_title('Spin λ vs redshift\n(ΛCDM : 0.03-0.05)', fontsize=9)
    ax.legend(fontsize=8); ax.invert_xaxis(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(zvals[::-1], vt_means, 'purple', marker='o', ms=4, lw=1.8,
            label='v_tan moyen')
    ax.plot(zvals[::-1], sigmas,   'orange', marker='s', ms=4, lw=1.8,
            label='σ_v (dispersion)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('z'); ax.set_ylabel('km/s (unités sim.)')
    ax.set_title('v_tan moyen vs σ_v\n(v_tan/σ_v > 0.1 = rotation significative)',
                 fontsize=9)
    ax.legend(fontsize=8); ax.invert_xaxis(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(zvals[::-1], Ns, 'k-o', ms=4, lw=1.8)
    ax.set_xlabel('z'); ax.set_ylabel('N particules m−')
    ax.set_title('Croissance du halo', fontsize=9)
    ax.invert_xaxis(); ax.grid(alpha=0.3)

    # Profil moyen de v_tan (dernier snapshot)
    ax = axes[1, 2]
    r_ref   = records[-1]['r_mid']
    vt_ref  = records[-1]['vt_profile']
    vt_s_ref = records[-1]['vt_std']
    valid = ~np.isnan(vt_ref)
    ax.fill_between(r_ref[valid],
                    vt_ref[valid] - vt_s_ref[valid],
                    vt_ref[valid] + vt_s_ref[valid],
                    alpha=0.25, color='blue')
    ax.plot(r_ref[valid], vt_ref[valid], 'b-o', ms=5, lw=1.8,
            label=f'step {records[-1]["step"]} z={records[-1]["z"]:.2f}')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('r_perp (Mpc)'); ax.set_ylabel('v_tan (km/s sim.)')
    ax.set_title('Profil v_tan(r) — dernier snapshot\n'
                 '(>0 = prograde, <0 = rétrograde)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    suffix = '_tracked' if args.track else ''
    out1 = os.path.join(OUT_DIR, f'rotation_evolution{suffix}.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 1 → {out1}")

    # ── Figure 2 : Carte 2D v_tan (dernier snapshot) ─────────────────
    print("Génération carte 2D v_tan (dernier snapshot)...")
    last = records[-1]
    snap_last = sorted(glob.glob(os.path.join(args.snap_dir,
                                              'snap_*.bin')))[-1]
    pos, vel, mass = load_snapshot(snap_last)
    halo_last = select_halo(pos, vel, mass, center, R, sign=-1)

    if halo_last is not None:
        v_map, e1, e2 = velocity_map_2d(
            halo_last['dr'], halo_last['dv'], halo_last['L_hat'],
            S=256, r_max=R)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle(f'Carte vitesse tangentielle — step {last["step"]} '
                      f'z={last["z"]:.3f}\nBleu = prograde, Rouge = rétrograde',
                      fontsize=11)

        valid_vals = v_map[~np.isnan(v_map)]
        if len(valid_vals) > 0:
            v_abs = np.percentile(np.abs(valid_vals), 95)
        else:
            v_abs = 1.0

        norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs)
        ext  = [-R, R, -R, R]

        ax2 = axes2[0]
        im = ax2.imshow(v_map, origin='lower', cmap='RdBu_r',
                        norm=norm, extent=ext, aspect='equal')
        plt.colorbar(im, ax=ax2, label='v_tan (km/s sim.)')
        # Cercles de rayon
        for r_c in [10, 25, 40]:
            th = np.linspace(0, 2*np.pi, 200)
            ax2.plot(r_c*np.cos(th), r_c*np.sin(th),
                     'white', lw=0.8, alpha=0.4, ls='--')
        ax2.set_xlabel(f'e1 ({e1[0]:.2f},{e1[1]:.2f},{e1[2]:.2f}) Mpc')
        ax2.set_ylabel(f'e2 ({e2[0]:.2f},{e2[1]:.2f},{e2[2]:.2f}) Mpc')
        ax2.set_title('Carte v_tan dans plan ⊥ L_hat', fontsize=9)
        ax2.plot(0, 0, '+w', ms=12, mew=2)

        # Profil v_tan(r) avec barres d'erreur
        ax2b = axes2[1]
        r_m, vt_m, vt_s, vr_m, n_b = rotation_profile(
            np.linalg.norm(halo_last['dr'] -
                np.einsum('ij,j->i', halo_last['dr'],
                           halo_last['L_hat'])[:, None] *
                halo_last['L_hat'][None, :], axis=1),
            decompose_velocities(halo_last['dr'], halo_last['dv'],
                                  halo_last['L_hat'])[1],
            decompose_velocities(halo_last['dr'], halo_last['dv'],
                                  halo_last['L_hat'])[0],
            n_bins=14, r_max=R*0.95)

        valid2 = ~np.isnan(vt_m)
        ax2b.fill_between(r_m[valid2],
                          vt_m[valid2]-vt_s[valid2],
                          vt_m[valid2]+vt_s[valid2],
                          alpha=0.2, color='blue')
        ax2b.plot(r_m[valid2], vt_m[valid2], 'b-o', ms=5, lw=2,
                  label='v_tan (prograde)')
        ax2b.plot(r_m[valid2], vr_m[valid2], 'r--o', ms=4, lw=1.5,
                  label='v_rad (expansion)')
        ax2b.axhline(0, color='k', lw=1, ls='--')
        ax2b.set_xlabel('r_perp (Mpc)')
        ax2b.set_ylabel('vitesse (unités sim.)')
        ax2b.set_title('Profil radial v_tan vs v_rad', fontsize=9)
        ax2b.legend(fontsize=9); ax2b.grid(alpha=0.3)

        plt.tight_layout()
        out2 = os.path.join(OUT_DIR, f'velocity_map_2d{suffix}.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure 2 → {out2}")

    # ── Rapport texte ─────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("RAPPORT ROTATION")
    print(f"{'='*50}")
    print(f"N steps analysés : {len(records)}")
    print(f"z range          : {zvals.max():.2f} → {zvals.min():.2f}")
    print(f"|L| moyen        : {L_mags.mean():.2e}")
    print(f"|L| variation    : {L_mags.std()/L_mags.mean()*100:.1f}%")
    print(f"Précession max   : {angle_deg.max():.1f}°")
    print(f"λ moyen          : {lams.mean():.4f}  (ΛCDM: 0.03-0.05)")
    print(f"λ std            : {lams.std():.4f}")
    print(f"v_tan moyen      : {vt_means.mean():+.1f}")
    print(f"σ_v moyen        : {sigmas.mean():.1f}")
    print(f"Ratio v_tan/σ_v  : {abs(vt_means.mean())/sigmas.mean():.3f}")
    print(f"  > 0.1 = rotation significative")
    print(f"  < 0.1 = dispersion dominante (pas de rotation nette)")


if __name__ == '__main__':
    main()
