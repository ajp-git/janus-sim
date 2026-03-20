#!/usr/bin/env python3
"""
paper_figures.py
================
Génère les figures manquantes du preprint Janus :

Figure 1 : S(z) comparatif V13 / V14 / V16
  - σ_P (notre métrique robuste) ET S_COM (métrique du document)
  - Montre que σ_P est plus robuste que S_COM

Figure 2 : P(k) à 10 redshifts sur V14
  - P_m−(k), P_m+(k), P_×(k), r(k)
  - Comparaison ΛCDM analytique

Figure 3 : P(k) évolution temporelle synthétique
  - r(k,z) = force de l'anti-corrélation dans le temps

Usage :
    python paper_figures.py

Durée estimée : ~30-45 min (P(k) G=256 sur 10 snapshots)
"""

import struct, glob, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ── Chemins ──────────────────────────────────────────────────────────
V13_CSV   = '/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/time_series.csv'
V14_CSV   = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/time_series.csv'
V16_CSV   = '/mnt/T2/janus-sim/output/janus_v16_500Mpc_30M_kmin10/time_series.csv'
V14_SNAPS = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/snapshots'
OUT_DIR   = '/mnt/T2/janus-sim/output/paper_figures'
BOX       = 500.0

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def load_snap(path):
    with open(path, 'rb') as f:
        n    = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    pos  = data[:, :3].astype(np.float64)
    mass = data[:, 6].astype(np.float64)
    if pos.min() < -BOX * 0.1:
        pos += BOX / 2.0
    return pos, mass


def load_csv(path):
    """Charge time_series.csv → dict step → {z, seg, sigma_P, L_J}"""
    result = {}
    try:
        with open(path) as f:
            header = f.readline().strip().split(',')
            for line in f:
                p = line.strip().split(',')
                if len(p) < 6: continue
                try:
                    s = int(p[0])
                    result[s] = {
                        'z':       float(p[1]),
                        'KE':      float(p[4]) if p[4] else 0,
                        'seg':     float(p[5]) if p[5] else 0,
                        'sigma_P': float(p[6]) if len(p)>6 and p[6] else 0,
                        'L_J':     float(p[7]) if len(p)>7 and p[7] else 0,
                    }
                except: pass
    except FileNotFoundError:
        print(f"  MISSING: {path}")
    return result


def compute_pk_fast(pos, mass, G=256, box=500.0, sign=None):
    """P(k) rapide sur grille 128³ (3 min par snapshot)."""
    if sign is not None:
        mask = mass * sign > 0
        pos  = pos[mask]
        mass = mass[mask]
    N = len(pos)
    if N < 100: return np.array([]), np.array([])

    half = box / 2.0
    ix   = np.clip(((pos[:,0]+half)/box*G).astype(int), 0, G-1)
    iy   = np.clip(((pos[:,1]+half)/box*G).astype(int), 0, G-1)
    iz   = np.clip(((pos[:,2]+half)/box*G).astype(int), 0, G-1)
    grid = np.zeros((G,G,G), np.float64)
    np.add.at(grid, (iz,iy,ix), 1.0)

    n_mean = N / G**3
    delta  = (grid - n_mean) / (n_mean + 1e-30)
    dk     = np.fft.fftn(delta)
    Pk_3d  = np.abs(dk)**2 * (box/G)**3 / box**3

    kf     = 2*np.pi/box
    kmax   = np.pi*G/box
    freq   = np.fft.fftfreq(G, d=1.0/G).astype(int)
    kx,ky,kz = np.meshgrid(freq*kf, freq*kf, freq*kf, indexing='ij')
    k_3d   = np.sqrt(kx**2+ky**2+kz**2)
    shot   = box**3 / N

    n_bins  = 30
    k_edges = np.logspace(np.log10(kf), np.log10(kmax), n_bins+1)
    k_c, Pk_c = np.zeros(n_bins), np.zeros(n_bins)
    k_flat, Pk_flat = k_3d.flatten(), Pk_3d.flatten()

    for i in range(n_bins):
        m = (k_flat >= k_edges[i]) & (k_flat < k_edges[i+1])
        if m.sum() > 0:
            k_c[i]  = k_flat[m].mean()
            Pk_c[i] = Pk_flat[m].mean() - shot

    valid = k_c > 0
    return k_c[valid], Pk_c[valid]


def compute_pk_cross_fast(pos, mass, G=256, box=500.0):
    """Spectre croisé m+/m− rapide."""
    half = box/2.0
    def delta(sign):
        msk = mass*sign>0; p = pos[msk]; N = msk.sum()
        if N < 100: return None, N
        ix = np.clip(((p[:,0]+half)/box*G).astype(int),0,G-1)
        iy = np.clip(((p[:,1]+half)/box*G).astype(int),0,G-1)
        iz = np.clip(((p[:,2]+half)/box*G).astype(int),0,G-1)
        g  = np.zeros((G,G,G),np.float64)
        np.add.at(g,(iz,iy,ix),1.0)
        nm = N/G**3
        return (g-nm)/(nm+1e-30), N
    dm, Nm = delta(-1); dp, Np = delta(+1)
    if dm is None or dp is None: return np.array([]), np.array([])

    cross = np.real(np.conj(np.fft.fftn(dp)) * np.fft.fftn(dm)) \
            * (box/G)**3 / box**3

    kf    = 2*np.pi/box; kmax = np.pi*G/box
    freq  = np.fft.fftfreq(G,d=1.0/G).astype(int)
    kx,ky,kz = np.meshgrid(freq*kf,freq*kf,freq*kf,indexing='ij')
    k_3d  = np.sqrt(kx**2+ky**2+kz**2)

    n_bins  = 30
    k_edges = np.logspace(np.log10(kf),np.log10(kmax),n_bins+1)
    k_c, Pk_c = np.zeros(n_bins), np.zeros(n_bins)
    k_flat, Pk_flat = k_3d.flatten(), cross.flatten()
    for i in range(n_bins):
        m = (k_flat>=k_edges[i])&(k_flat<k_edges[i+1])
        if m.sum()>0:
            k_c[i]=k_flat[m].mean(); Pk_c[i]=Pk_flat[m].mean()
    valid = k_c>0
    return k_c[valid], Pk_c[valid]


def pk_lcdm(k, z=0.0, h=0.678, Om=0.308, Ob=0.048, ns=0.968, s8=0.815):
    """P(k) ΛCDM analytique (Bardeen 1986)."""
    G  = Om*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
    q  = k/(G+1e-30)
    T  = (np.log(1+2.34*q)/(2.34*q+1e-30) *
          (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-0.25))
    T  = np.where(k<1e-6, 1.0, T)
    kp = 0.05
    Pp = (k/kp)**ns * T**2
    R8 = 8.0/h
    kw = np.linspace(1e-4,10,5000)
    Wk = 3*(np.sin(kw*R8)-kw*R8*np.cos(kw*R8))/(kw*R8+1e-30)**3
    Gw = Om*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
    qw = kw/(Gw+1e-30)
    Tw = (np.log(1+2.34*qw)/(2.34*qw+1e-30)*
          (1+3.89*qw+(16.1*qw)**2+(5.46*qw)**3+(6.71*qw)**4)**(-0.25))
    Pw = (kw/kp)**ns*Tw**2
    s2 = np.trapezoid(kw**2*Pw*Wk**2,kw)/(2*np.pi**2)
    norm = s8**2/(s2+1e-30)
    return norm*Pp*(1/(1+z))**2


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 : S(z) comparatif V13/V14/V16
# ══════════════════════════════════════════════════════════════════════

def make_figure1():
    print("\n── FIGURE 1 : S(z) comparatif ──────────────────────────")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ségrégation Janus — Comparaison V13 / V14 / V16\n'
                 'Métrique σ_P (robuste) vs S_COM (document initial)',
                 fontsize=11)

    runs = [
        (V13_CSV, 'V13 (15M, k_min=3)',  'gray',      '--'),
        (V14_CSV, 'V14 (3M,  k_min=20)', 'steelblue', '-'),
        (V16_CSV, 'V16 (30M, k_min=10)', 'crimson',   '-'),
    ]

    for csv, label, color, ls in runs:
        data = load_csv(csv)
        if not data: continue

        # Trier par z décroissant → z croissant pour affichage
        steps = sorted(data.keys())
        zvals    = np.array([data[s]['z']       for s in steps])
        seg      = np.array([data[s]['seg']      for s in steps])
        sigma_P  = np.array([data[s]['sigma_P']  for s in steps])

        # σ_P
        valid_sP = sigma_P > 0
        if valid_sP.any():
            axes[0].semilogy(zvals[valid_sP][::-1], sigma_P[valid_sP][::-1],
                             color=color, ls=ls, lw=2, label=label)

        # seg (σ_P / σ_ρ ou distance COM selon les runs)
        valid_seg = seg > 0
        if valid_seg.any():
            axes[1].semilogy(zvals[valid_seg][::-1], seg[valid_seg][::-1],
                             color=color, ls=ls, lw=2, label=label)

        print(f"  {label}: {sum(valid_sP)} points σ_P, "
              f"max_seg={seg[valid_seg].max():.3f} @ z={zvals[valid_seg][seg[valid_seg].argmax()]:.2f}")

    for ax, title, ylabel in zip(axes,
        ['σ_P(z) — Écart-type champ polarisation\n(mesure robuste ségrégation)',
         'S(z) — Métrique time_series\n(comparaison directe avec document)'],
        ['σ_P', 'S']):
        ax.set_xlabel('z (redshift)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.invert_xaxis()
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')
        ax.axvline(5, color='k', ls=':', lw=0.8, alpha=0.5)
        ax.text(5.1, ax.get_ylim()[0]*2, 'z_init=5', fontsize=7, alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'figure1_segregation_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 : P(k) à 10 redshifts sur V14
# ══════════════════════════════════════════════════════════════════════

def make_figure2():
    print("\n── FIGURE 2 : P(k) à 10 redshifts ─────────────────────")

    # Sélectionner 10 snapshots répartis uniformément en z
    data_v14 = load_csv(V14_CSV)
    all_snaps = sorted(glob.glob(os.path.join(V14_SNAPS, 'snap_*.bin')))

    if not all_snaps:
        print("  SKIP : aucun snapshot V14 trouvé")
        return

    # 10 snapshots logarithmiquement espacés en z
    target_z = np.array([4.5, 3.5, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.2, 0.0])
    def snap_step(s):
        return int(os.path.basename(s).replace('snap_','').replace('.bin',''))
    snap_z = {snap_step(s): data_v14.get(snap_step(s), {}).get('z', np.nan)
              for s in all_snaps}

    selected = []
    for tz in target_z:
        best = min(snap_z.items(), key=lambda x: abs(x[1]-tz) if not np.isnan(x[1]) else 999)
        snap_path = os.path.join(V14_SNAPS, f'snap_{best[0]:06d}.bin')
        selected.append((snap_path, best[1]))

    # Dédupliquer
    seen = set()
    selected_unique = []
    for sp, z in selected:
        if sp not in seen:
            seen.add(sp); selected_unique.append((sp, z))

    print(f"  {len(selected_unique)} snapshots sélectionnés")

    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(selected_unique)))
    k_ref  = np.logspace(-3, np.log10(np.pi*256/BOX), 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('P(k) Janus V14 — Évolution temporelle z=4.5→0\n'
                 '3M particules, BOX=500 Mpc, k_min=20', fontsize=11)

    r_k_collection = []

    for (sp, z), c in zip(selected_unique, colors):
        if not os.path.exists(sp):
            print(f"  SKIP {os.path.basename(sp)}")
            continue

        t0 = time.time()
        pos, mass = load_snap(sp)

        k_m, Pk_m = compute_pk_fast(pos, mass, G=256, box=BOX, sign=-1)
        k_p, Pk_p = compute_pk_fast(pos, mass, G=256, box=BOX, sign=+1)
        k_x, Pk_x = compute_pk_cross_fast(pos, mass, G=256, box=BOX)

        lbl = f'z={z:.2f}'
        print(f"  {lbl} ({time.time()-t0:.0f}s)", end=' ')

        # Ax 0 : P(k) m−
        if len(k_m): axes[0,0].loglog(k_m, np.abs(Pk_m), color=c, lw=1.5,
                                        label=lbl, alpha=0.9)

        # Ax 1 : P(k) m+ vs m−
        if len(k_m) and len(k_p):
            axes[0,1].loglog(k_m, np.abs(Pk_m), color=c, lw=1.5, ls='-',  alpha=0.9)
            axes[0,1].loglog(k_p, np.abs(Pk_p), color=c, lw=1.0, ls='--', alpha=0.6)

        # Ax 2 : P_× croisé
        if len(k_x):
            neg  = Pk_x < 0
            axes[1,0].loglog(k_x[neg],  -Pk_x[neg], color=c, lw=1.5,
                              label=lbl, alpha=0.9)

        # Ax 3 : r(k) coefficient de corrélation
        if len(k_m) and len(k_p) and len(k_x):
            k_common = k_x
            Pm_interp = np.interp(k_common, k_m, np.abs(Pk_m))
            Pp_interp = np.interp(k_common, k_p, np.abs(Pk_p))
            r_k = Pk_x / (np.sqrt(Pm_interp * Pp_interp) + 1e-30)
            valid = (k_common < 0.4) & np.isfinite(r_k)
            if valid.any():
                axes[1,1].semilogx(k_common[valid], r_k[valid], color=c,
                                    lw=1.5, label=lbl, alpha=0.9)
                r_k_collection.append((z, k_common[valid], r_k[valid]))
        print()

    # ΛCDM référence sur ax 0
    Pk_cdm0 = pk_lcdm(k_ref, z=0)
    axes[0,0].loglog(k_ref, Pk_cdm0, 'k--', lw=2, alpha=0.4, label='ΛCDM z=0')

    # Formatage
    titles = ['P(k) m− (masses négatives)',
              'P(k) m− (plein) vs m+ (pointillé)',
              '|P_×(k)| — Anti-corrélation m+/m−\n(négatif = ségrégation)',
              'r(k) = P_×/√(P₊P₋)\n(-1 = ségrégation parfaite)']
    ylabels= ['P(k) (Mpc³)', 'P(k) (Mpc³)',
              '|P_×(k)| (Mpc³)', 'r(k)']

    for ax, title, yl in zip(axes.flat, titles, ylabels):
        ax.set_xlabel('k (Mpc⁻¹)')
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, ncol=2)
        ax.axvline(0.4, color='gray', ls=':', lw=1, alpha=0.5)

    axes[1,0].set_title('|P_×(k)| — Anti-corrélation\n'
                         'négatif = ségrégation spatiale Janus', fontsize=9)
    axes[1,1].axhline(-1, color='red', ls='--', lw=1.5, label='Ségrégation parfaite')
    axes[1,1].axhline(0,  color='k',   ls='-',  lw=0.8)
    axes[1,1].set_ylim(-1.1, 0.2)
    axes[1,1].legend(fontsize=7)

    # Colorbar z
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(0, 4.5))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.flat[-1], shrink=0.8)
    cb.set_label('Redshift z', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'figure2_pk_evolution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out}")

    return r_k_collection


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 : r(k,z) heatmap
# ══════════════════════════════════════════════════════════════════════

def make_figure3(r_k_collection):
    if not r_k_collection:
        print("\n── FIGURE 3 : skip (pas de données r(k)) ───────────")
        return

    print("\n── FIGURE 3 : r(k,z) heatmap ───────────────────────────")

    # Grille k commune
    k_grid = np.logspace(-2, np.log10(0.4), 25)
    z_vals = np.array([item[0] for item in r_k_collection])
    R_grid = np.full((len(r_k_collection), len(k_grid)), np.nan)

    for i, (z, k, r) in enumerate(r_k_collection):
        R_grid[i] = np.interp(k_grid, k, r, left=np.nan, right=np.nan)

    # Trier par z décroissant
    idx = np.argsort(z_vals)[::-1]
    z_sorted = z_vals[idx]
    R_sorted = R_grid[idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('r(k, z) = P_×(k) / √(P₊P₋) — Force de la ségrégation Janus\n'
                 'V14 (3M, k_min=20) — r = −1 : ségrégation parfaite',
                 fontsize=11)

    im = ax.pcolormesh(k_grid, z_sorted, R_sorted,
                        cmap='RdBu_r', vmin=-1, vmax=0.2,
                        shading='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('r(k, z)', fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel('k (Mpc⁻¹)', fontsize=10)
    ax.set_ylabel('z (redshift)', fontsize=10)
    ax.set_title('Bleu foncé = forte anti-corrélation = ségrégation forte\n'
                 'Blanc = pas de corrélation (ΛCDM attendu = blanc partout)',
                 fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.4, color='white', ls='--', lw=1, alpha=0.5)
    ax.text(0.42, z_sorted[-1]+0.1, 'k_max fiable', color='white',
            fontsize=7, alpha=0.7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'figure3_rk_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    print("="*60)
    print("PAPER FIGURES — Simulation Janus")
    print("="*60)

    make_figure1()
    r_k_data = make_figure2()
    make_figure3(r_k_data)

    print(f"\n{'='*60}")
    print(f"TERMINÉ en {(time.time()-t_start)/60:.0f} min")
    print(f"Figures dans : {OUT_DIR}")
    print("="*60)
