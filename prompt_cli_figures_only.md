# Mission : Générer les 4 figures PNG (fond blanc) pour le document Janus

## Contexte
Les CSV sont dans le répertoire courant ou output/.
Objectif : 4 fichiers PNG qualité publication, fond blanc, 200 dpi.

## Lancer le fit Friedmann si chi2_map.csv et hz_data.csv n'existent pas

```bash
docker compose run --rm dev cargo run --release --bin friedmann
```

## Créer et lancer generate_figures.py

```python
import pandas as pd, numpy as np, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
})

def find_csv(name):
    for p in [f'./{name}', f'output/{name}']:
        if os.path.exists(p): return p
    hits = glob.glob(f'**/{name}', recursive=True)
    if hits: return hits[0]
    raise FileNotFoundError(name)

# ── FIGURE 1 : Time series ────────────────────────────────
df = pd.read_csv(find_csv('time_series.csv'))
z_vals = 1.0/df['a'] - 1.0
s_max = df['segregation'].max()
step_max = int(df.loc[df['segregation'].idxmax(), 'step'])
z_at_max = float(z_vals.iloc[df['segregation'].idxmax()])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(df['step'], df['segregation'], color='#1565C0', lw=0.9, alpha=0.9)
ax1.axhline(s_max, color='#C62828', ls='--', lw=1.8,
    label=f'$S_{{max}}={s_max:.3f}$ (step {step_max}, $z\\approx{z_at_max:.1f}$)')
ax1.axvline(step_max, color='#E65100', ls=':', lw=1.5)
step_end = int(df['step'].max())
for (x0,x1,label,c) in [(1,126,'Phase 1\nFriction','#1B5E20'),
                          (126,1453,'Phase 2\nCroissance','#F57F17'),
                          (1453,2473,'Phase 3\nPlateau','#BF360C'),
                          (2473,step_end,'Phase 4\nDéclin','#4A148C')]:
    ax1.axvspan(x0, min(x1,step_end), alpha=0.06, color=c)
    ax1.text((x0+min(x1,step_end))/2, s_max*0.55, label,
             ha='center', fontsize=7.5, color=c, fontweight='bold')
ax1.set_xlabel('Step'); ax1.set_ylabel('Ségrégation $S$')
ax1.set_title('Run 500K — Ségrégation ($\\eta=1.045$, $z:5\\to0$)')
ax1.legend(loc='upper left'); ax1.grid(alpha=0.2); ax1.set_ylim(bottom=0)

ax2.plot(df['step'], df['ke_ratio'], color='#2E7D32', lw=0.9)
ke_max = df['ke_ratio'].max()
ax2.axhline(ke_max, color='#C62828', ls='--', lw=1.8,
    label=f'$KE/KE_0^{{max}}={ke_max:.2f}$')
ax2.set_xlabel('Step'); ax2.set_ylabel('$KE / KE_0$')
ax2.set_title('Énergie cinétique relative')
ax2.legend(loc='upper left'); ax2.grid(alpha=0.2); ax2.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('fig_timeseries.png', dpi=200, bbox_inches='tight', facecolor='white')
print("fig_timeseries.png ✓")

# ── FIGURE 2 : Snapshot au pic ────────────────────────────
# Chercher le frame PNG le plus proche du step_max
all_frames = sorted(glob.glob('output/**/frames/frame_*.png', recursive=True))
if all_frames:
    nums = []
    for f in all_frames:
        try:
            n = int(os.path.basename(f).replace('frame_','').replace('.png',''))
            nums.append((abs(n-step_max), n, f))
        except: pass
    if nums:
        _, best_step, best_frame = sorted(nums)[0]
        from PIL import Image
        img = Image.open(best_frame).convert('RGB')
        data = np.array(img)
        dark = (data[:,:,0]<25)&(data[:,:,1]<25)&(data[:,:,2]<25)
        data[dark] = [255,255,255]
        Image.fromarray(data).save('frame_peak.png', dpi=(200,200))
        print(f"frame_peak.png ✓ (step {best_step}, fond noir→blanc)")
else:
    print("ATTENTION : pas de snapshots trouvés — frame_peak.png non généré")
    print("Relancer le run N-corps pour obtenir les snapshots")

# ── FIGURE 3 : Chi2 map ───────────────────────────────────
df2 = pd.read_csv(find_csv('chi2_map.csv'))
etas = sorted(df2['eta'].unique()); h0s = sorted(df2['H0'].unique())
ETA = np.array(etas); H0 = np.array(h0s)
chi2_snia = df2.pivot(index='H0', columns='eta', values='chi2_snia').values / 1701
chi2_hz   = df2.pivot(index='H0', columns='eta', values='chi2_hz').values / 22

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, cmap, vmin, vmax, title in [
    (axes[0], chi2_snia, 'viridis_r', 0.5,  10, 'Pantheon+ SNIa ($N=1701$)'),
    (axes[1], chi2_hz,   'plasma_r',  1.0, 100, 'CC+BAO $H(z)$ ($N=22$)')]:
    im = ax.pcolormesh(ETA, H0, data,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, shading='auto')
    plt.colorbar(im, ax=ax, label='$\\chi^2/\\nu$')
    if data.min() < 1.0:
        cs = ax.contour(ETA, H0, data, levels=[1.0], colors='white', linewidths=2.5)
        ax.clabel(cs, fmt='$\\chi^2/\\nu=1$', fontsize=9)
    iy, ix = np.unravel_index(data.argmin(), data.shape)
    ax.plot(ETA[ix], H0[iy], 'w*', ms=18, zorder=5)
    ax.annotate(f'$\\eta={ETA[ix]:.3f}$, $H_0={H0[iy]:.0f}$\n$\\chi^2/\\nu={data.min():.3f}$',
        xy=(ETA[ix], H0[iy]), xytext=(ETA[ix]+0.015, H0[iy]-4),
        fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.65))
    ax.set_xlabel('$\\eta$'); ax.set_ylabel('$H_0$ [km/s/Mpc]')
    ax.set_title(f'{title}\n$\\chi^2_{{min}}/\\nu={data.min():.3f}$')
plt.tight_layout()
plt.savefig('fig_chi2map.png', dpi=200, bbox_inches='tight', facecolor='white')
print("fig_chi2map.png ✓")

# ── FIGURE 4 : H(z) ──────────────────────────────────────
df_hz = pd.read_csv(find_csv('hz_data.csv'))
H0_janus=76.0; H0_lcdm=67.9
z_arr = df_hz['z'].values
H_janus = df_hz['H_over_H0'].values * H0_janus
z_lcdm = np.linspace(0,2.5,300)
H_lcdm = H0_lcdm * np.sqrt(0.315*(1+z_lcdm)**3 + 0.685)
hz_obs = [(0.07,69.,19.6),(0.09,69.,12.),(0.17,83.,8.),(0.179,75.,4.),
    (0.199,75.,5.),(0.27,77.,14.),(0.352,83.,14.),(0.38,83.,13.5),
    (0.4,95.,17.),(0.44,82.6,7.8),(0.57,96.8,3.4),(0.593,104.,13.),
    (0.6,87.9,6.1),(0.68,92.,8.),(0.73,97.3,7.),(0.781,105.,12.),
    (0.875,125.,17.),(0.9,117.,23.),(1.037,154.,20.),(1.3,168.,17.),
    (1.363,160.,33.6),(1.75,202.,40.),(2.34,222.,7.),(2.36,226.,8.)]
z_obs=np.array([p[0] for p in hz_obs])
H_obs=np.array([p[1] for p in hz_obs])
H_err=np.array([p[2] for p in hz_obs])

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(z_lcdm, H_lcdm, color='#C62828', ls='--', lw=2.8,
    label=f'$\\Lambda$CDM ($H_0={H0_lcdm}$, $\\Omega_m=0.315$)')
mask = z_arr<=2.5
ax.plot(z_arr[mask], H_janus[mask], color='#1B5E20', lw=2.8,
    label=f'Janus ($\\eta=1.045$, $H_0={H0_janus}$)')
ax.errorbar(z_obs, H_obs, yerr=H_err, fmt='o', color='#1A237E',
    ecolor='#5C6BC0', capsize=3.5, ms=6, lw=1.5,
    label='CC+BAO (données observationnelles)', zorder=5)
ax.set_xlabel('Redshift $z$',fontsize=13); ax.set_ylabel('$H(z)$ [km/s/Mpc]',fontsize=13)
ax.set_title('$H(z)$ : Janus vs $\\Lambda$CDM', fontsize=14, fontweight='bold')
ax.set_xlim(0,2.5); ax.set_ylim(50,280)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('fig_hz.png', dpi=200, bbox_inches='tight', facecolor='white')
print("fig_hz.png ✓")

print("\n=== Fichiers générés ===")
for f in ['fig_timeseries.png','frame_peak.png','fig_chi2map.png','fig_hz.png']:
    exists = "✓" if os.path.exists(f) else "✗ MANQUANT"
    print(f"  {f} {exists}")
```

## Résultat

Dépose les 4 PNG dans le répertoire courant du projet.
Uploade-les ensuite dans Claude.ai web pour la compilation LaTeX.

Si frame_peak.png est manquant, relancer le run N-corps d'abord :
```bash
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- --n 500000 --eta 1.045 --dt 0.01 --steps 3600 \
  --output output/run_figures
```
