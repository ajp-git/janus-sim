# Prochaine étape — Step 3500 (z ≈ 0.38)
# Deux objectifs : viriel local corrigé + investigation halo bleu

## Contexte
- Run : janus_v13_500Mpc_15M
- Step actuel : 3500 / 5000, z ≈ 0.38
- Snapshot à analyser : snap_003500.bin
- Format : 7 floats (x,y,z,vx,vy,vz,sign), 28 bytes/particule

---

## OBJECTIF 1 — Viriel local corrigé

### Problème identifié
L'analyse 20 (viriel local) donnait 2KE/|PE| entre 300 et 3000.
Cause : formule PE ≈ -G*N²/R est une approximation sphérique
qui sous-estime |PE| d'un facteur ~N.

### Formule correcte
Pour N particules dans un halo :
```
PE_exact = -G * sum_{i<j} m_i * m_j / |r_i - r_j|
```

Pour N=15000 particules c'est O(N²) → trop lent.
Utiliser l'estimateur de Spitzer :
```
PE_spitzer = -0.3 * G * M² / R_half
```
où R_half = rayon médian des particules du halo.
C'est exact à ~10% pour des profils NFW.

### Code

```python
import numpy as np
import struct
import glob
from scipy.ndimage import label as ndlabel
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BOX  = 500.0
HALF = BOX / 2.0
G    = 1.0  # unités simulation

def read_snapshot(path):
    with open(path, 'rb') as f:
        n, step, _ = struct.unpack('<QQQ', f.read(24))
        raw = f.read(n * 28)
        data = np.frombuffer(raw, dtype=np.float32).reshape(n, 7)
    return data[:,:3].copy(), data[:,3:6].copy(), data[:,6].copy(), int(step)

os.makedirs('analysis_step3500', exist_ok=True)

snap = 'snapshots/snap_003500.bin'
pos, vel, sign, step = read_snapshot(snap)
print(f"Step {step}, N={len(pos)}")

# Déposer sur grille 64³ pour trouver les halos
grid = 64
gp = np.zeros((grid,grid,grid))
ix = ((pos[:,0]+HALF)/BOX*grid).astype(np.int32)%grid
iy = ((pos[:,1]+HALF)/BOX*grid).astype(np.int32)%grid
iz = ((pos[:,2]+HALF)/BOX*grid).astype(np.int32)%grid
np.add.at(gp, (ix[sign>0],iy[sign>0],iz[sign>0]), 1.0)

lbl, n_halos = ndlabel(gp > gp.mean()*3)
sizes = np.bincount(lbl.ravel())[1:]
print(f"Halos+ détectés : {n_halos}")

# Sous-échantillon pour vitesse
idx_s = np.random.choice(len(pos), min(500000, len(pos)), replace=False)
pos_s, vel_s, sign_s = pos[idx_s], vel[idx_s], sign[idx_s]

virial_data = []
for hid in (np.argsort(sizes)[-15:]+1):
    c = np.argwhere(lbl==hid).mean(axis=0)/grid*BOX - HALF
    dist = np.sqrt(((pos_s-c)**2).sum(axis=1))
    mask = (dist < 40) & (sign_s > 0)
    if mask.sum() < 50: continue

    p_h = pos_s[mask]
    v_h = vel_s[mask]
    v_cm = v_h.mean(axis=0)
    v_pec = v_h - v_cm

    N = mask.sum()
    M = float(N)  # masse en unités particule

    # KE locale (correcte)
    KE = 0.5 * (v_pec**2).sum(axis=1).mean() * M

    # PE estimateur Spitzer (correct à ~10%)
    dists_h = np.sqrt(((p_h - c)**2).sum(axis=1))
    R_half = float(np.median(dists_h))  # rayon médian
    PE_spitzer = -0.3 * G * M**2 / (R_half + 1e-10)

    # Viriel
    virial = 2 * KE / (abs(PE_spitzer) + 1e-10)

    virial_data.append({
        'halo_id': int(hid),
        'N': int(N),
        'R_half': R_half,
        'KE': float(KE),
        'PE': float(PE_spitzer),
        'virial': float(virial),
        'sigma_v': float(np.sqrt((v_pec**2).sum(axis=1).mean())),
    })
    print(f"  halo {hid}: N={N}, R_half={R_half:.1f} Mpc, "
          f"2KE/|PE|={virial:.3f}, σ_v={virial_data[-1]['sigma_v']:.1f}")

# Plot viriel vs masse
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vir_vals = [d['virial'] for d in virial_data]
n_vals   = [d['N'] for d in virial_data]
r_vals   = [d['R_half'] for d in virial_data]

sc = axes[0].scatter(n_vals, vir_vals, c=r_vals,
                     cmap='RdYlGn_r', s=80, zorder=5)
axes[0].axhline(1, color='green', ls='--', lw=2, label='équilibre viriel')
axes[0].axhline(2, color='orange', ls='--', lw=1, alpha=0.7, label='sur-virialisé')
axes[0].axhline(0.5, color='blue', ls='--', lw=1, alpha=0.7, label='sous-virialisé')
plt.colorbar(sc, ax=axes[0], label='R_half [Mpc]')
axes[0].set_xlabel('N particules')
axes[0].set_ylabel('2KE / |PE_Spitzer|')
axes[0].set_title(f'Viriel local corrigé (Spitzer) — step {step}')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 4)

axes[1].scatter(r_vals, vir_vals, c=n_vals, cmap='plasma', s=80)
axes[1].axhline(1, color='green', ls='--', lw=2)
axes[1].set_xlabel('R_half [Mpc]')
axes[1].set_ylabel('2KE / |PE_Spitzer|')
axes[1].set_title('Viriel vs rayon demi-masse')

plt.tight_layout()
plt.savefig('analysis_step3500/20b_virial_spitzer.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ 20b_virial_spitzer.png")
```

---

## OBJECTIF 2 — Investigation halo bleu (T− anormal)

### Contexte
Dans la carte T+(x) − T−(x) du step 3200, un halo bleu isolé
apparaît à Y ≈ -100 Mpc, X ≈ -50 Mpc.
C'est un halo de masses− dont T− > T+ localement.
Signature possible : halo− sur-virialisé ou en collision.

### Code

```python
# Trouver le halo bleu : halo de masses- le plus chaud

grid_T = 64

# Carte de température masses-
gm = np.zeros((grid_T,grid_T,grid_T))
vm2 = np.zeros((grid_T,grid_T,grid_T))  # sum |v|²
gm_n = np.zeros((grid_T,grid_T,grid_T))

ix_all = ((pos[:,0]+HALF)/BOX*grid_T).astype(np.int32)%grid_T
iy_all = ((pos[:,1]+HALF)/BOX*grid_T).astype(np.int32)%grid_T
iz_all = ((pos[:,2]+HALF)/BOX*grid_T).astype(np.int32)%grid_T

mask_m = sign < 0
np.add.at(gm, (ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]), 1.0)

# Vitesse locale moyenne masses-
vx_m = np.zeros((grid_T,grid_T,grid_T))
vy_m = np.zeros((grid_T,grid_T,grid_T))
vz_m = np.zeros((grid_T,grid_T,grid_T))
np.add.at(vx_m, (ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]), vel[mask_m,0])
np.add.at(vy_m, (ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]), vel[mask_m,1])
np.add.at(vz_m, (ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]), vel[mask_m,2])
cnt = gm + 1e-10
vx_m /= cnt; vy_m /= cnt; vz_m /= cnt

# Vitesse peculiaire → T locale
v_px = vel[mask_m,0] - vx_m[ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]]
v_py = vel[mask_m,1] - vy_m[ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]]
v_pz = vel[mask_m,2] - vz_m[ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]]
v_pec2_m = v_px**2 + v_py**2 + v_pz**2

T_m = np.zeros((grid_T,grid_T,grid_T))
np.add.at(T_m, (ix_all[mask_m],iy_all[mask_m],iz_all[mask_m]), v_pec2_m)
T_m = np.where(gm>0, T_m/(3*gm+1e-10), 0)

# Halo− le plus chaud
lbl_m, _ = ndlabel(gm > gm.mean()*2)
sizes_m = np.bincount(lbl_m.ravel())[1:]

# Température moyenne par halo-
halo_temps = []
for hid in range(1, len(sizes_m)+1):
    mask_h = lbl_m == hid
    if sizes_m[hid-1] < 5: continue
    T_mean = T_m[mask_h].mean()
    c_px = np.argwhere(mask_h).mean(axis=0)/grid_T*BOX - HALF
    halo_temps.append({'hid': hid, 'T': T_mean,
                       'x': c_px[0], 'y': c_px[1], 'z': c_px[2],
                       'N': int(sizes_m[hid-1])})

halo_temps.sort(key=lambda x: x['T'], reverse=True)
print("\nTop 5 halos- les plus chauds :")
for h in halo_temps[:5]:
    print(f"  halo {h['hid']}: T={h['T']:.1f}, pos=({h['x']:.0f},{h['y']:.0f},{h['z']:.0f}) Mpc, N={h['N']}")

# Zoom sur le halo le plus chaud
hot_halo = halo_temps[0]
c_hot = np.array([hot_halo['x'], hot_halo['y'], hot_halo['z']])
print(f"\nHalo bleu le plus chaud : {c_hot} Mpc")

# Particules dans 60 Mpc du halo chaud
dist_hot = np.sqrt(((pos - c_hot)**2).sum(axis=1))
mask_zoom = dist_hot < 60

pos_z  = pos[mask_zoom]
vel_z  = vel[mask_zoom]
sign_z = sign[mask_zoom]

print(f"Particules dans r<60 Mpc : {mask_zoom.sum()}")
print(f"  masses+ : {(sign_z>0).sum()}")
print(f"  masses- : {(sign_z<0).sum()}")

# Profil radial T(r) pour ce halo
r_bins = np.linspace(0, 60, 25)
r_c    = 0.5*(r_bins[:-1]+r_bins[1:])
T_profile_m = []
T_profile_p = []

for sp, T_list in [(1, T_profile_p), (-1, T_profile_m)]:
    mask_sp = sign_z == sp
    if mask_sp.sum() < 10:
        T_list.extend([0]*len(r_c))
        continue
    p_sp = pos_z[mask_sp]
    v_sp = vel_z[mask_sp]
    d_sp = np.sqrt(((p_sp - c_hot)**2).sum(axis=1))
    v_cm = v_sp.mean(axis=0)
    v_pec_sp = v_sp - v_cm
    v2_sp = (v_pec_sp**2).sum(axis=1)
    for i in range(len(r_c)):
        mask_r = (d_sp >= r_bins[i]) & (d_sp < r_bins[i+1])
        T_list.append(v2_sp[mask_r].mean()/3 if mask_r.sum()>3 else np.nan)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Investigation halo bleu — step {step} — pos=({c_hot[0]:.0f},{c_hot[1]:.0f},{c_hot[2]:.0f}) Mpc')

# Profil T(r)
axes[0].plot(r_c, T_profile_m, 'b-o', ms=4, lw=2, label='T-(r)')
axes[0].plot(r_c, T_profile_p, 'r-o', ms=4, lw=2, label='T+(r)')
axes[0].set_xlabel('r [Mpc]'); axes[0].set_ylabel('T(r)')
axes[0].set_title('Profil température radial')
axes[0].legend()

# Distribution des vitesses peculiaires
mask_hot_m = (dist_hot < 30) & (sign < 0)
mask_hot_p = (dist_hot < 30) & (sign > 0)
if mask_hot_m.sum() > 10:
    v_cm_m = vel[mask_hot_m].mean(axis=0)
    v_pec_hot_m = np.sqrt(((vel[mask_hot_m]-v_cm_m)**2).sum(axis=1))
    axes[1].hist(v_pec_hot_m, bins=30, color='blue', alpha=0.6,
                 density=True, label=f'masse- σ={v_pec_hot_m.std():.0f}')
if mask_hot_p.sum() > 10:
    v_cm_p = vel[mask_hot_p].mean(axis=0)
    v_pec_hot_p = np.sqrt(((vel[mask_hot_p]-v_cm_p)**2).sum(axis=1))
    axes[1].hist(v_pec_hot_p, bins=30, color='red', alpha=0.6,
                 density=True, label=f'masse+ σ={v_pec_hot_p.std():.0f}')
axes[1].set_xlabel('|v_pec|'); axes[1].legend()
axes[1].set_title('Distribution vitesses dans le halo bleu')

# Carte de densité zoom XY
extent = [c_hot[0]-80, c_hot[0]+80, c_hot[1]-80, c_hot[1]+80]
mask_slice = (np.abs(pos_z[:,2] - c_hot[2]) < 15)
pos_sl = pos_z[mask_slice]
sign_sl = sign_z[mask_slice]
grid_zoom = 128
g_zoom_p = np.zeros((grid_zoom, grid_zoom))
g_zoom_m = np.zeros((grid_zoom, grid_zoom))
ix_z = ((pos_sl[:,0]-extent[0])/(extent[1]-extent[0])*grid_zoom).astype(int)
iy_z = ((pos_sl[:,1]-extent[2])/(extent[3]-extent[2])*grid_zoom).astype(int)
valid = (ix_z>=0)&(ix_z<grid_zoom)&(iy_z>=0)&(iy_z<grid_zoom)
np.add.at(g_zoom_p, (ix_z[valid&(sign_sl>0)], iy_z[valid&(sign_sl>0)]), 1.0)
np.add.at(g_zoom_m, (ix_z[valid&(sign_sl<0)], iy_z[valid&(sign_sl<0)]), 1.0)

from scipy.ndimage import gaussian_filter
g_zoom_p = gaussian_filter(g_zoom_p, sigma=2)
g_zoom_m = gaussian_filter(g_zoom_m, sigma=2)
total_z = g_zoom_p + g_zoom_m + 1e-10
P_zoom = (g_zoom_p - g_zoom_m) / total_z

im = axes[2].imshow(P_zoom.T, origin='lower', cmap='RdBu_r',
                    vmin=-1, vmax=1, extent=extent)
plt.colorbar(im, ax=axes[2], label='P = (ρ+−ρ−)/(ρ++ρ−)')
axes[2].set_xlabel('X [Mpc]'); axes[2].set_ylabel('Y [Mpc]')
axes[2].set_title('Carte P zoom halo bleu (rouge=+, bleu=-)')
axes[2].axvline(c_hot[0], color='white', ls='--', alpha=0.5)
axes[2].axhline(c_hot[1], color='white', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('analysis_step3500/halo_bleu_investigation.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ halo_bleu_investigation.png")

# Viriel du halo bleu
mask_hb = (dist_hot < 30) & (sign < 0)
if mask_hb.sum() > 20:
    p_hb = pos[mask_hb]
    v_hb = vel[mask_hb]
    v_cm_hb = v_hb.mean(axis=0)
    v_pec_hb = v_hb - v_cm_hb
    N_hb = mask_hb.sum()
    KE_hb = 0.5 * (v_pec_hb**2).sum(axis=1).mean() * N_hb
    d_hb = np.sqrt(((p_hb - c_hot)**2).sum(axis=1))
    R_half_hb = float(np.median(d_hb))
    PE_hb = -0.3 * G * N_hb**2 / (R_half_hb + 1e-10)
    virial_hb = 2 * KE_hb / abs(PE_hb)
    print(f"\nHalo bleu viriel : 2KE/|PE| = {virial_hb:.3f}")
    print(f"  N={N_hb}, R_half={R_half_hb:.1f} Mpc, σ_v={np.sqrt((v_pec_hb**2).sum(axis=1).mean()):.1f}")
    if virial_hb > 2:
        print("  → SUR-VIRIALISÉ : énergie cinétique excédentaire")
        print("  → hypothèse : chauffé par répulsion Janus aux interfaces")
    elif virial_hb < 0.5:
        print("  → SOUS-VIRIALISÉ : encore en collapse")
    else:
        print("  → virialisé normalement")
```

---

## Sorties attendues

```
analysis_step3500/
├── 20b_virial_spitzer.png       # viriel corrigé, valeurs ~1
└── halo_bleu_investigation.png  # profil T(r), carte P zoom
```

## Résultats clés à reporter

Pour le viriel corrigé :
- Valeurs attendues entre 0.5 et 2.0 (pas 300-3000)
- Identifier si certains halos sont vraiment sur-virialisés

Pour le halo bleu :
- Position exacte (X, Y, Z) en Mpc
- σ_v local masses− vs masses+
- Viriel local 2KE/|PE|
- Structure de la carte P au zoom : halo pur− ou mixte ?
- Si sur-virialisé ET pur− → chauffage par répulsion Janus
