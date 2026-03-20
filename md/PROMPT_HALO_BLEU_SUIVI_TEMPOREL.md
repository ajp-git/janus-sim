# Investigation halo bleu — Suivi temporel + correction unités G
# Step actuel : ~3500-4000, run en cours jusqu'à z=0

## Contexte
- Halo bleu identifié : pos=(168, 127, 73) Mpc, N−=1,060,707, σ_v=654, R_half=14.8 Mpc
- Problème unités viriel : G=1 incompatible avec v en km/s et r en Mpc
- Objectif : suivi temporel du halo + correction unités

---

## ÉTAPE 0 — Identifier les unités internes du code Janus

```bash
# Chercher les constantes G dans le code source Rust
grep -r "G\|gravit\|const G\|G_N\|6.674" /mnt/T2/janus-sim/src/ | head -20
grep -r "unit\|Mpc\|km_s\|kms\|vel_unit" /mnt/T2/janus-sim/src/ | head -20

# Chercher dans les fichiers de config
grep -r "G\|unit\|Mpc\|velocity" /mnt/T2/janus-sim/*.toml 2>/dev/null | head -20
grep -r "G\|unit\|Mpc\|velocity" /mnt/T2/janus-sim/config*.toml 2>/dev/null | head -20
```

Une fois G_sim trouvé, le facteur de conversion viriel est :
```
virial_correct = 2*KE / |PE_spitzer| / facteur_G
```

---

## ÉTAPE 1 — Suivi temporel du halo bleu

Analyser tous les snapshots disponibles pour suivre :
- σ_v(t) du halo bleu → chauffage continu ?
- N−(t) → croissance du halo ?
- R_half(t) → expansion ou contraction ?
- Nombre de particules+ expulsées dans r<60 Mpc

```python
import numpy as np
import struct
import glob
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BOX  = 500.0
HALF = BOX / 2.0

def read_snapshot(path):
    with open(path, 'rb') as f:
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        raw = f.read(n * 28)
        if len(raw) != n * 28:
            raise ValueError(f"Tronqué: {path}")
        data = np.frombuffer(raw, dtype=np.float32).reshape(n, 7)
    return data[:,:3].copy(), data[:,3:6].copy(), data[:,6].copy(), int(step)

os.makedirs('analysis_halo_bleu', exist_ok=True)

# Position connue du halo bleu à step 3500
C_HALO = np.array([168.0, 127.0, 73.0])
R_SEARCH = 60.0  # Mpc

snaps = sorted(glob.glob('snapshots/snap_0*.bin'))
print(f"Snapshots disponibles : {len(snaps)}")

# Lire time_series pour avoir z(step)
ts = {int(float(r['step'])): float(r['z'])
      for r in csv.DictReader(open('time_series.csv'))}

# ============================================================
# SUIVI TEMPOREL
# ============================================================
tracking = []

for snap in snaps:
    pos, vel, sign, step = read_snapshot(snap)
    z_val = ts.get(step, -1)
    if z_val < 0:
        # Interpoler z
        steps_ts = sorted(ts.keys())
        z_val = ts[min(steps_ts, key=lambda s: abs(s-step))]

    # Particules dans r < R_SEARCH du centre connu
    dist = np.sqrt(((pos - C_HALO)**2).sum(axis=1))
    mask_zone = dist < R_SEARCH

    n_plus  = int(((sign > 0) & mask_zone).sum())
    n_minus = int(((sign < 0) & mask_zone).sum())

    if n_minus < 10:
        tracking.append({'step': step, 'z': z_val,
                         'n_plus': n_plus, 'n_minus': n_minus,
                         'sigma_v': 0, 'R_half': 0, 'T_mean': 0})
        continue

    # Propriétés du halo−
    mask_m = (sign < 0) & mask_zone
    pos_m  = pos[mask_m]
    vel_m  = vel[mask_m]
    v_cm   = vel_m.mean(axis=0)
    v_pec  = vel_m - v_cm
    sigma_v = float(np.sqrt((v_pec**2).sum(axis=1).mean()))
    T_mean  = float((v_pec**2).sum(axis=1).mean() / 3)

    dist_m = np.sqrt(((pos_m - C_HALO)**2).sum(axis=1))
    R_half = float(np.median(dist_m))

    tracking.append({
        'step': step, 'z': z_val,
        'n_plus': n_plus, 'n_minus': n_minus,
        'sigma_v': sigma_v, 'R_half': R_half,
        'T_mean': T_mean,
    })

    if step % 500 == 0 or step == snaps[-1]:
        print(f"  step {step} z={z_val:.3f}: N+={n_plus}, N-={n_minus}, "
              f"σ_v={sigma_v:.0f}, R_half={R_half:.1f}")

# Sauvegarder tracking
with open('analysis_halo_bleu/tracking.csv', 'w') as f:
    f.write('step,z,n_plus,n_minus,sigma_v,R_half,T_mean\n')
    for t in tracking:
        f.write(f"{t['step']},{t['z']:.4f},{t['n_plus']},{t['n_minus']},"
                f"{t['sigma_v']:.2f},{t['R_half']:.3f},{t['T_mean']:.2f}\n")

# ============================================================
# PLOTS SUIVI TEMPOREL
# ============================================================
steps_t  = [t['step'] for t in tracking]
z_vals   = [t['z'] for t in tracking]
n_plus_t = [t['n_plus'] for t in tracking]
n_minus_t= [t['n_minus'] for t in tracking]
sigma_t  = [t['sigma_v'] for t in tracking]
R_half_t = [t['R_half'] for t in tracking]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Suivi temporel du méga-halo− (pos=168,127,73 Mpc)', fontsize=12)

# N particules
axes[0,0].plot(z_vals, n_minus_t, 'b-o', ms=4, lw=2, label='N masse−')
axes[0,0].plot(z_vals, n_plus_t,  'r-o', ms=4, lw=1.5, label='N masse+')
axes[0,0].set_xlabel('z'); axes[0,0].set_ylabel('N particules')
axes[0,0].set_title('Croissance du halo (r < 60 Mpc)')
axes[0,0].legend(); axes[0,0].invert_xaxis()

# σ_v
axes[0,1].plot(z_vals, sigma_t, 'purple', marker='o', ms=4, lw=2)
axes[0,1].set_xlabel('z'); axes[0,1].set_ylabel('σ_v')
axes[0,1].set_title('Dispersion de vitesse\n(augmente = chauffage par répulsion)')
axes[0,1].invert_xaxis()

# R_half
axes[1,0].plot(z_vals, R_half_t, 'g-o', ms=4, lw=2)
axes[1,0].set_xlabel('z'); axes[1,0].set_ylabel('R_half [Mpc]')
axes[1,0].set_title('Rayon demi-masse\n(croît = halo s\'étend)')
axes[1,0].invert_xaxis()

# Ratio N-/N+
ratio = [m/max(p,1) for m,p in zip(n_minus_t, n_plus_t)]
axes[1,1].plot(z_vals, ratio, 'k-o', ms=4, lw=2)
axes[1,1].set_xlabel('z'); axes[1,1].set_ylabel('N− / N+')
axes[1,1].set_title('Pureté du halo\n(augmente = expulsion masses+ en cours)')
axes[1,1].invert_xaxis()

plt.tight_layout()
plt.savefig('analysis_halo_bleu/01_temporal_tracking.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ 01_temporal_tracking.png")

# ============================================================
# VITESSE D'EXPULSION DES MASSES+
# dN+/dt dans r < 60 Mpc
# ============================================================
if len(n_plus_t) > 2:
    dn_dt = np.gradient(n_plus_t, steps_t)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_vals, dn_dt, 'r-o', ms=4, lw=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('z'); ax.set_ylabel('dN+/dstep')
    ax.set_title('Taux d\'expulsion des masses+ hors du halo−\n(négatif = expulsion active)')
    ax.invert_xaxis()
    plt.savefig('analysis_halo_bleu/02_expulsion_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("→ 02_expulsion_rate.png")

# ============================================================
# CARTE ÉVOLUTION SPATIALE
# Snapshots clés : step 500, 1000, 1500, 2000, 2500, 3000, 3500
# ============================================================
key_snaps = [s for s in snaps
             if any(int(float(s.split('snap_')[1].split('.')[0])) == k
                    for k in [500,1000,1500,2000,2500,3000,3500])]
key_snaps = key_snaps[:min(7, len(key_snaps))]

if key_snaps:
    n_plots = len(key_snaps)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, snap in enumerate(key_snaps):
        pos, vel, sign, step = read_snapshot(snap)
        z_val = ts.get(step, 0)

        # Zoom 120×120 Mpc autour du halo bleu
        margin = 60
        extent = [C_HALO[0]-margin, C_HALO[0]+margin,
                  C_HALO[1]-margin, C_HALO[1]+margin]

        # Sélection particules dans la tranche z ± 20 Mpc
        mask_slice = np.abs(pos[:,2] - C_HALO[2]) < 20

        pos_sl  = pos[mask_slice]
        sign_sl = sign[mask_slice]

        # Grille 128²
        g = 128
        gp = np.zeros((g,g)); gm = np.zeros((g,g))
        ix = ((pos_sl[:,0]-extent[0])/(extent[1]-extent[0])*g).astype(int)
        iy = ((pos_sl[:,1]-extent[2])/(extent[3]-extent[2])*g).astype(int)
        valid = (ix>=0)&(ix<g)&(iy>=0)&(iy<g)
        np.add.at(gp, (ix[valid&(sign_sl>0)], iy[valid&(sign_sl>0)]), 1.0)
        np.add.at(gm, (ix[valid&(sign_sl<0)], iy[valid&(sign_sl<0)]), 1.0)

        gp = gaussian_filter(gp, sigma=2)
        gm = gaussian_filter(gm, sigma=2)
        total = gp + gm + 1e-10
        P_map = (gp - gm) / total

        im = axes[i].imshow(P_map.T, origin='lower', cmap='RdBu_r',
                            vmin=-1, vmax=1, extent=extent)
        axes[i].set_title(f'step {step} — z={z_val:.2f}', fontsize=9)
        axes[i].axvline(C_HALO[0], color='white', ls='--', lw=0.7, alpha=0.5)
        axes[i].axhline(C_HALO[1], color='white', ls='--', lw=0.7, alpha=0.5)
        axes[i].set_xlabel('X [Mpc]', fontsize=7)
        axes[i].set_ylabel('Y [Mpc]', fontsize=7)

    # Cacher axes vides
    for j in range(len(key_snaps), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Évolution spatiale du méga-halo− (zoom 120×120 Mpc)', fontsize=11)
    plt.tight_layout()
    plt.savefig('analysis_halo_bleu/03_spatial_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("→ 03_spatial_evolution.png")

# ============================================================
# ÉNERGIE DE RÉPULSION ACCUMULÉE
# E_rep = ΔKE = KE(t) - KE(t0) du halo−
# C'est l'énergie injectée par la répulsion Janus
# ============================================================
KE_vals = [0.5 * t['sigma_v']**2 * t['n_minus'] for t in tracking]
if max(KE_vals) > 0:
    KE0 = KE_vals[0] if KE_vals[0] > 0 else KE_vals[1]
    dKE = [k - KE0 for k in KE_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_vals, dKE, 'orange', marker='o', ms=4, lw=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('z'); ax.set_ylabel('ΔKE (énergie injectée)')
    ax.set_title('Énergie cinétique accumulée par le méga-halo−\n'
                 '(augmente = chauffage par répulsion Janus confirmé)')
    ax.invert_xaxis()
    plt.savefig('analysis_halo_bleu/04_energy_injection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("→ 04_energy_injection.png")

# ============================================================
# RAPPORT TEXTE
# ============================================================
print("\n" + "="*50)
print("RÉSUMÉ SUIVI HALO BLEU")
print("="*50)
first = next((t for t in tracking if t['n_minus'] > 1000), None)
last  = tracking[-1]
if first:
    print(f"Première apparition : step {first['step']} z={first['z']:.3f}")
print(f"Step actuel        : step {last['step']} z={last['z']:.3f}")
print(f"N− actuel          : {last['n_minus']:,}")
print(f"N+ dans r<60 Mpc   : {last['n_plus']}")
print(f"σ_v actuel         : {last['sigma_v']:.0f}")
print(f"R_half actuel      : {last['R_half']:.1f} Mpc")

# Tendances
if len(tracking) > 5:
    mid = len(tracking)//2
    sigma_early = np.mean([t['sigma_v'] for t in tracking[:mid] if t['sigma_v']>0])
    sigma_late  = np.mean([t['sigma_v'] for t in tracking[mid:] if t['sigma_v']>0])
    if sigma_early > 0 and sigma_late > 0:
        trend = (sigma_late - sigma_early) / sigma_early * 100
        print(f"\nTendance σ_v : {'+' if trend>0 else ''}{trend:.1f}%")
        if trend > 10:
            print("→ CHAUFFAGE CONTINU CONFIRMÉ par répulsion Janus")
        elif abs(trend) < 5:
            print("→ σ_v stable : halo en équilibre thermique")
        else:
            print("→ Refroidissement : halo se contracte")
```

---

## Sorties attendues

```
analysis_halo_bleu/
├── tracking.csv               # données brutes suivi temporel
├── 01_temporal_tracking.png   # N, σ_v, R_half, pureté vs z
├── 02_expulsion_rate.png      # taux expulsion masses+
├── 03_spatial_evolution.png   # 7 cartes P zoom vs temps
└── 04_energy_injection.png    # énergie cinétique accumulée
```

## Questions clés à répondre

1. **σ_v augmente-t-il dans le temps ?**
   → Si oui : chauffage continu par répulsion Janus confirmé

2. **N+ décroît-il dans le temps ?**
   → Si oui : expulsion active visible directement

3. **Le halo grandit-il (N− croît) ?**
   → Si oui : accrétion de masses− → méga-structure en formation

4. **Les cartes P montrent-elles un halo de plus en plus pur ?**
   → Passage de P mixte → P=-1 pur dans le temps

## Note sur les unités G

Pour corriger le viriel une fois G_sim trouvé dans le code :
```python
# Facteur de conversion typique pour simulations cosmologiques
# Si v en km/s, r en Mpc, m en M_soleil/particule :
G_phys = 4.302e-3  # pc M_sun^-1 (km/s)^2
G_Mpc  = G_phys * 1e-6  # Mpc M_sun^-1 (km/s)^2
# Ajuster selon les unités réelles du binaire
```
