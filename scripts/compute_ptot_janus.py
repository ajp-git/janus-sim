#!/usr/bin/env python3
"""
compute_ptot_janus.py
=====================
Calcule P_tot^Janus(k) = P₊(k) + P₋(k) + 2×P_×(k)
et compare avec P(k) ΛCDM analytique.

P_tot est le spectre de puissance total de la matière Janus —
ce qui serait observé si on pouvait tracer indifféremment m+ et m−.

Usage :
    python compute_ptot_janus.py

Durée : ~10 min (V14 3M particules, G=256)
"""

import struct, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ── Chemins ───────────────────────────────────────────────────────────
V14_SNAPS = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/snapshots'
V14_CSV   = '/mnt/T2/janus-sim/output/janus_v14_500Mpc_3M_kmin20/time_series.csv'
OUT_DIR   = '/mnt/T2/janus-sim/output/paper_figures'
BOX       = 500.0
G         = 256
os.makedirs(OUT_DIR, exist_ok=True)


# ── Utilitaires ───────────────────────────────────────────────────────

def load_snap(path):
    with open(path, 'rb') as f:
        n    = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n*28), dtype=np.float32).reshape(n,7)
    pos  = data[:,:3].astype(np.float64)
    mass = data[:,6].astype(np.float64)
    if pos.min() < -BOX*0.1: pos += BOX/2
    return pos, mass


def make_delta(pos, mass, sign, G=256, box=500.0):
    """Champ de contraste δ pour une population."""
    mask = mass*sign > 0
    p = pos[mask]; N = mask.sum()
    if N == 0: return np.zeros((G,G,G)), 0
    half = box/2
    ix = np.clip(((p[:,0]+half)/box*G).astype(int), 0, G-1)
    iy = np.clip(((p[:,1]+half)/box*G).astype(int), 0, G-1)
    iz = np.clip(((p[:,2]+half)/box*G).astype(int), 0, G-1)
    grid = np.zeros((G,G,G), np.float64)
    np.add.at(grid, (iz,iy,ix), 1.0)
    nm = N/G**3
    return (grid - nm)/(nm + 1e-30), N


def bin_pk(Pk_3d, k_3d, n_bins=35, kf=None, kmax=None, box=500.0):
    """Moyenne sphérique de P(k)."""
    if kf is None: kf = 2*np.pi/box
    if kmax is None: kmax = np.pi*G/box
    k_edges = np.logspace(np.log10(kf), np.log10(kmax), n_bins+1)
    k_c = np.zeros(n_bins); Pk_c = np.zeros(n_bins); Nm = np.zeros(n_bins,int)
    k_flat = k_3d.flatten(); Pk_flat = Pk_3d.flatten()
    for i in range(n_bins):
        m = (k_flat >= k_edges[i]) & (k_flat < k_edges[i+1])
        if m.sum() > 0:
            k_c[i] = k_flat[m].mean()
            Pk_c[i] = Pk_flat[m].mean()
            Nm[i] = m.sum()
    valid = k_c > 0
    return k_c[valid], Pk_c[valid], Nm[valid]


def compute_k3d(G=256, box=500.0):
    kf   = 2*np.pi/box
    freq = np.fft.fftfreq(G, d=1.0/G).astype(int)
    kx,ky,kz = np.meshgrid(freq*kf, freq*kf, freq*kf, indexing='ij')
    return np.sqrt(kx**2+ky**2+kz**2)


def compute_all_spectra(pos, mass, G=256, box=500.0):
    """Calcule P₊, P₋, P_×, P_tot en un seul passage FFT."""
    kf    = 2*np.pi/box
    vol   = box**3

    delta_p, Np = make_delta(pos, mass, +1, G, box)
    delta_m, Nm = make_delta(pos, mass, -1, G, box)
    delta_tot   = (delta_p*Np + delta_m*Nm) / (Np + Nm)  # densité totale pondérée

    dk_p   = np.fft.fftn(delta_p)
    dk_m   = np.fft.fftn(delta_m)
    dk_tot = np.fft.fftn(delta_tot)

    norm  = (box/G)**3 / vol
    Pk_p_3d   = np.abs(dk_p)**2   * norm
    Pk_m_3d   = np.abs(dk_m)**2   * norm
    Pk_x_3d   = np.real(np.conj(dk_p)*dk_m) * norm
    Pk_tot_3d = np.abs(dk_tot)**2 * norm

    k_3d = compute_k3d(G, box)
    shot_p   = vol/Np
    shot_m   = vol/Nm
    shot_tot = vol/(Np+Nm)

    k, Pk_p,   _ = bin_pk(Pk_p_3d   - shot_p,   k_3d, box=box)
    _, Pk_m,   _ = bin_pk(Pk_m_3d   - shot_m,   k_3d, box=box)
    _, Pk_x,   _ = bin_pk(Pk_x_3d,               k_3d, box=box)
    _, Pk_tot, _ = bin_pk(Pk_tot_3d - shot_tot,  k_3d, box=box)

    # P_tot analytique = P+ + P- + 2P×
    Pk_tot_analytic = Pk_p + Pk_m + 2*Pk_x

    return k, Pk_p, Pk_m, Pk_x, Pk_tot, Pk_tot_analytic


def pk_lcdm(k, z=0.0, h=0.678, Om=0.308, Ob=0.048, ns=0.968, s8=0.815):
    """P(k) ΛCDM analytique normalisé σ8."""
    G_  = Om*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
    q   = k/(G_+1e-30)
    T   = (np.log(1+2.34*q)/(2.34*q+1e-30)*
           (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-0.25))
    T   = np.where(k<1e-6,1.0,T)
    kp  = 0.05; Pp = (k/kp)**ns*T**2
    R8  = 8.0/h
    kw  = np.linspace(1e-4,10,5000)
    Wk  = 3*(np.sin(kw*R8)-kw*R8*np.cos(kw*R8))/(kw*R8+1e-30)**3
    Gw  = Om*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
    qw  = kw/(Gw+1e-30)
    Tw  = (np.log(1+2.34*qw)/(2.34*qw+1e-30)*
           (1+3.89*qw+(16.1*qw)**2+(5.46*qw)**3+(6.71*qw)**4)**(-0.25))
    Pw  = (kw/kp)**ns*Tw**2
    s2  = np.trapezoid(kw**2*Pw*Wk**2,kw)/(2*np.pi**2)
    norm= s8**2/(s2+1e-30)
    Dz  = 1.0/(1.0+z)
    return norm*Pp*Dz**2


# ── Sélection des snapshots ───────────────────────────────────────────
def load_csv(path):
    data = {}
    try:
        with open(path) as f:
            next(f)
            for line in f:
                p = line.strip().split(',')
                if len(p) >= 2:
                    try: data[int(p[0])] = float(p[1])
                    except: pass
    except: pass
    return data

step_z = load_csv(V14_CSV)
all_snaps = sorted(glob.glob(os.path.join(V14_SNAPS,'snap_*.bin')))

# 6 snapshots : z=4.5, 3, 2, 1, 0.5, 0
target_z = [4.5, 3.0, 2.0, 1.0, 0.5, 0.0]
selected = []
for tz in target_z:
    best = min(
        [(int(os.path.basename(s).replace('snap_','').replace('.bin','')), s)
         for s in all_snaps],
        key=lambda x: abs(step_z.get(x[0], 99) - tz)
    )
    z_actual = step_z.get(best[0], tz)
    selected.append((best[1], z_actual))

# Dédupliquer
seen = set(); unique = []
for sp,z in selected:
    if sp not in seen: seen.add(sp); unique.append((sp,z))

print(f"Snapshots sélectionnés : {len(unique)}")
for sp,z in unique:
    print(f"  z={z:.3f}  {os.path.basename(sp)}")


# ── Calcul ────────────────────────────────────────────────────────────
k_ref   = np.logspace(-3, np.log10(np.pi*G/BOX), 200)
colors  = plt.cm.plasma(np.linspace(0.05, 0.95, len(unique)))

results = []
for (sp, z), c in zip(unique, colors):
    if not os.path.exists(sp): continue
    print(f"\n  Calcul z={z:.3f}...", end=' ', flush=True)
    pos, mass = load_snap(sp)
    k, Pk_p, Pk_m, Pk_x, Pk_tot_fft, Pk_tot = compute_all_spectra(pos, mass, G, BOX)
    Pk_cdm = pk_lcdm(k, z=z)
    ratio  = Pk_tot / (Pk_cdm + 1e-30)
    r_k    = Pk_x / (np.sqrt(np.abs(Pk_p)*np.abs(Pk_m)) + 1e-30)
    results.append({
        'z':z,'c':c,'k':k,
        'Pk_p':Pk_p,'Pk_m':Pk_m,'Pk_x':Pk_x,
        'Pk_tot':Pk_tot,'Pk_cdm':Pk_cdm,
        'ratio':ratio,'r_k':r_k
    })
    # Stats rapides
    valid = (k > 0.01) & (k < 0.3)
    if valid.any():
        ratio_mean = ratio[valid].mean()
        rx_mean    = r_k[valid].mean()
        print(f"P_tot/P_ΛCDM(0.01-0.3)={ratio_mean:.1f}×  r(k)={rx_mean:.3f}")


# ── Figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('P_tot(k) Janus = P₊ + P₋ + 2P_× — Comparaison ΛCDM analytique\n'
             'V14 (3M particules, k_min=20, BOX=500 Mpc)', fontsize=11)

for res in results:
    z = res['z']; c = res['c']; k = res['k']
    lbl = f'z={z:.2f}'
    kv  = k < 0.5   # domaine fiable

    # 1. P_tot vs ΛCDM
    ax = axes[0,0]
    ax.loglog(k[kv], np.abs(res['Pk_tot'][kv]), color=c, lw=2, label=lbl)

    # 2. Décomposition P₊, P₋
    ax = axes[0,1]
    ax.loglog(k[kv], np.abs(res['Pk_p'][kv]), color=c, lw=1.5, ls='-')
    ax.loglog(k[kv], np.abs(res['Pk_m'][kv]), color=c, lw=1.0, ls='--')

    # 3. 2P_× contribution
    ax = axes[0,2]
    neg = res['Pk_x'][kv] < 0
    pos_ = res['Pk_x'][kv] >= 0
    if neg.any():
        ax.loglog(k[kv][neg], -2*res['Pk_x'][kv][neg],
                  color=c, lw=2, label=lbl)

    # 4. Ratio P_tot/P_ΛCDM
    ax = axes[1,0]
    valid = kv & (res['ratio'] > 0) & (res['ratio'] < 200)
    ax.semilogx(k[valid], res['ratio'][valid], color=c, lw=2, label=lbl)

    # 5. r(k) - skip individual curves, will add heatmap after loop

    # 6. P_tot/P_ΛCDM cumulé
    ax = axes[1,2]
    Pk_cdm_interp = pk_lcdm(k[kv], z=z)
    ax.loglog(k[kv], np.abs(res['Pk_tot'][kv]), color=c, lw=2, label=lbl)
    ax.loglog(k_ref, pk_lcdm(k_ref, z=z), color=c, lw=0.8, ls=':', alpha=0.5)

# ΛCDM référence
for ax in [axes[0,0], axes[0,1], axes[1,2]]:
    ax.loglog(k_ref, pk_lcdm(k_ref,z=0), 'k--', lw=2, alpha=0.4,
              label='ΛCDM z=0')

# Lignes de référence
axes[1,0].axhline(1, color='k', ls='--', lw=1.5, label='ΛCDM = 1')
axes[1,0].axhline(10, color='gray', ls=':', lw=1, alpha=0.7, label='×10')
axes[1,0].axhline(50, color='red', ls=':', lw=1, alpha=0.7, label='×50')
axes[1,0].set_ylim(0, 120)

# Panel 5: r(k,z) heatmap (like figure 3)
ax = axes[1,1]
k_grid = np.logspace(-2, np.log10(0.4), 30)
z_vals = np.array([res['z'] for res in results])
R_grid = np.zeros((len(results), len(k_grid)))
for i, res in enumerate(results):
    R_grid[i] = np.interp(k_grid, res['k'], res['r_k'], left=np.nan, right=np.nan)
im = ax.pcolormesh(k_grid, z_vals, R_grid, cmap='RdBu_r', vmin=-1, vmax=0.5, shading='auto')
ax.set_xscale('log')
ax.invert_yaxis()
cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
cb.set_label('r(k)', fontsize=8)

# Formatage
titles = [
    'P_tot^Janus(k) vs ΛCDM\n= P₊ + P₋ + 2P_× (total matière Janus)',
    'Décomposition\nP₊ (plein) vs P₋ (pointillé)',
    '|2P_×(k)| — Contribution croisée\n(terme d\'interférence négatif)',
    'Ratio P_tot^Janus / P_ΛCDM\n(tension observationnelle ?)',
    'r(k,z) heatmap\n(bleu = anti-corrélation Janus)',
    'P_tot^Janus vs P_ΛCDM par z\n(évolution temporelle)',
]
xlabels = ['k (Mpc⁻¹)'] * 6
ylabels = ['P(k) (Mpc³)', 'P(k) (Mpc³)', '|2P_×(k)| (Mpc³)',
           'Ratio', 'r(k)', 'P(k) (Mpc³)']

for ax, title, xl, yl in zip(axes.flat, titles, xlabels, ylabels):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')
    ax.axvline(0.3, color='gray', ls=':', lw=0.8, alpha=0.5)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 4.5))
sm.set_array([])
cb = fig.colorbar(sm, ax=axes[1,2], shrink=0.8)
cb.set_label('Redshift z', fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'figure5_ptot_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n\nFigure → {out}")

# ── Rapport texte ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RAPPORT P_tot^Janus vs ΛCDM")
print(f"{'='*60}")
print(f"{'z':>6} {'P_tot/P_ΛCDM (k=0.1)':>22} {'r(k) moyen':>12} {'Tension?'}")
print("─"*60)
for res in results:
    k = res['k']
    idx = np.argmin(np.abs(k - 0.1))
    ratio_01 = res['ratio'][idx]
    valid = (k > 0.01) & (k < 0.3)
    r_mean = res['r_k'][valid].mean() if valid.any() else 0
    tension = "⚠️ FORTE" if ratio_01 > 20 else ("OK" if ratio_01 < 5 else "Modérée")
    print(f"  {res['z']:4.2f}   {ratio_01:>20.1f}×   {r_mean:>10.3f}   {tension}")

print()
print("Note : P_tot^Janus >> P_ΛCDM car :")
print("  1. Halos ultra-compacts → variance δ très élevée")
print("  2. Vides quasi-absolus → contraste extrême")
print("  3. Comparaison directe impossible sans biais galaxies b²")
print()
print("Pour comparaison avec DESI/Euclid : P_obs ≈ b² × P_matière")
print("avec b ≈ 1.5-2 (biais galaxies) → ne réduit pas le facteur 50")
print("→ Tension réelle à quantifier avec modélisation baryonique")
