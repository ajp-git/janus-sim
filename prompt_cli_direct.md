# Lancer les runs Janus et générer les figures

Aucune donnée existante. Tout relancer dans l'ordre ci-dessous sans vérifier.

## 1. Fit Pantheon+

```bash
docker compose run --rm dev cargo run --release --bin friedmann 2>&1 | tee run_friedmann.log
```

Attendre la fin. Vérifier dans le log que η_best ≈ 1.045 et χ²/ν ≈ 0.9.

## 2. Run N-corps 500K

```bash
docker compose run --rm dev cargo run --release --features cuda \
  --bin nbody_overnight -- \
  --n 500000 --eta 1.045 --dt 0.01 --steps 3600 \
  --output output/run_doc \
  2>&1 | tee run_nbody.log
```

Attendre la fin (~4h). Ne pas interrompre.

## 3. Rendu des frames

```bash
python3 scripts/render_overnight.py \
  output/run_doc/snapshots \
  output/run_doc/frames \
  1
```

## 4. Générer les figures

```bash
python3 generate_figures.py
```

Avec ce contenu pour generate_figures.py :

```python
import pandas as pd, numpy as np, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':10,
})

# Figure 1 — chi2(eta)
df = pd.read_csv(glob.glob('**/chi2_snia.csv',recursive=True)[0])
chi2_dof = (df['chi2'].values if 'chi2' in df.columns else df['chi2_snia'].values) / 1701
eta = df['eta'].values
eta_best = eta[np.argmin(chi2_dof)]
chi2_best = chi2_dof.min()
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(eta, chi2_dof, 'o-', color='#1565C0', lw=2, ms=5, label='Janus SNIa ($H_0=76$ fixé)')
ax.axhline(1.0, color='gray', ls=':', lw=1.5, label='$\\chi^2/\\nu=1$')
ax.axvline(eta_best, color='#C62828', ls='--', lw=2, label=f'$\\eta_{{best}}={eta_best:.3f}$')
ax.plot(eta_best, chi2_best, 'r*', ms=16, zorder=5)
ax.annotate(f'$\\chi^2/\\nu={chi2_best:.3f}$', xy=(eta_best,chi2_best),
    xytext=(eta_best+0.02,chi2_best+0.03), fontsize=11, color='#C62828',
    bbox=dict(boxstyle='round,pad=0.3',fc='#FFF9C4',ec='#F9A825'))
ax.axvline(1.045, color='#2E7D32', ls=':', lw=1.5, label='$\\eta=1.045$ (Petit 2014)')
ax.set_xlabel('$\\eta = |\\rho_-|/\\rho_+$'); ax.set_ylabel('$\\chi^2/\\nu$')
ax.set_title('Fit Pantheon+ SNIa — Modèle Janus ($H_0=76.0$ km/s/Mpc)')
ax.legend(loc='upper right'); ax.grid(alpha=0.25); ax.set_ylim(bottom=0.4)
plt.tight_layout()
plt.savefig('fig_chi2.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f"fig_chi2.png ✓  η_best={eta_best:.3f}  χ²/ν={chi2_best:.3f}")

# Figure 2 — time series
df2 = pd.read_csv(glob.glob('**/time_series.csv',recursive=True)[0])
z_vals = 1.0/df2['a'] - 1.0
s_max = df2['segregation'].max()
step_max = int(df2.loc[df2['segregation'].idxmax(),'step'])
z_max = float(z_vals.iloc[df2['segregation'].idxmax()])
ke_max = df2['ke_ratio'].max()
step_end = int(df2['step'].max())
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13,5))
ax1.plot(df2['step'],df2['segregation'],color='#1565C0',lw=0.9,alpha=0.9)
ax1.axhline(s_max,color='#C62828',ls='--',lw=1.8,
    label=f'$S_{{max}}={s_max:.3f}$ (step {step_max}, $z\\approx{z_max:.1f}$)')
ax1.axvline(step_max,color='#E65100',ls=':',lw=1.5)
for x0,x1,label,c in [(0,126,'Phase 1\nFriction','#1B5E20'),
                       (126,1453,'Phase 2\nCroissance','#F57F17'),
                       (1453,2473,'Phase 3\nPlateau','#BF360C'),
                       (2473,step_end,'Phase 4\nDéclin','#4A148C')]:
    ax1.axvspan(x0,min(x1,step_end),alpha=0.07,color=c)
    ax1.text((x0+min(x1,step_end))/2,s_max*0.5,label,
             ha='center',fontsize=8,color=c,fontweight='bold')
ax1.set_xlabel('Step'); ax1.set_ylabel('Ségrégation $S$')
ax1.set_title('Run 500K — Ségrégation ($\\eta=1.045$, $z:5\\to 0$)')
ax1.legend(loc='upper left'); ax1.grid(alpha=0.2); ax1.set_ylim(bottom=0)
ax2.plot(df2['step'],df2['ke_ratio'],color='#2E7D32',lw=0.9)
ax2.axhline(ke_max,color='#C62828',ls='--',lw=1.8,label=f'$KE/KE_0^{{max}}={ke_max:.2f}$')
ax2.set_xlabel('Step'); ax2.set_ylabel('$KE/KE_0$')
ax2.set_title('Énergie cinétique relative')
ax2.legend(loc='upper left'); ax2.grid(alpha=0.2); ax2.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('fig_timeseries.png',dpi=200,bbox_inches='tight',facecolor='white')
print(f"fig_timeseries.png ✓  S_max={s_max:.3f}  step={step_max}  z={z_max:.1f}")

# Figure 3 — snapshot au pic fond blanc
all_frames = sorted(glob.glob('output/**/frames/frame_*.png',recursive=True))
best = min(all_frames, key=lambda f: abs(
    int(os.path.basename(f).replace('frame_','').replace('.png','')) - step_max))
img = Image.open(best).convert('RGB')
data = np.array(img)
dark = (data[:,:,0]<20)&(data[:,:,1]<20)&(data[:,:,2]<20)
data[dark] = [255,255,255]
Image.fromarray(data).save('frame_peak.png',dpi=(200,200))
print(f"frame_peak.png ✓  (depuis {os.path.basename(best)})")

# Figure 4 — H(z)
df_hz = pd.read_csv(glob.glob('**/hz_data.csv',recursive=True)[0])
H0_janus=76.0; H0_lcdm=67.9
z_arr=df_hz['z'].values; H_janus=df_hz['H_over_H0'].values*H0_janus
z_lcdm=np.linspace(0,2.5,300)
H_lcdm=H0_lcdm*np.sqrt(0.315*(1+z_lcdm)**3+0.685)
hz_obs=[(0.07,69.,19.6),(0.09,69.,12.),(0.17,83.,8.),(0.179,75.,4.),
    (0.199,75.,5.),(0.27,77.,14.),(0.352,83.,14.),(0.38,83.,13.5),
    (0.4,95.,17.),(0.44,82.6,7.8),(0.57,96.8,3.4),(0.593,104.,13.),
    (0.6,87.9,6.1),(0.68,92.,8.),(0.73,97.3,7.),(0.781,105.,12.),
    (0.875,125.,17.),(0.9,117.,23.),(1.037,154.,20.),(1.3,168.,17.),
    (1.363,160.,33.6),(1.75,202.,40.),(2.34,222.,7.),(2.36,226.,8.)]
z_obs=np.array([p[0] for p in hz_obs])
H_obs=np.array([p[1] for p in hz_obs])
H_err=np.array([p[2] for p in hz_obs])
fig,ax=plt.subplots(figsize=(9,6))
ax.plot(z_lcdm,H_lcdm,color='#C62828',ls='--',lw=2.8,
    label=f'$\\Lambda$CDM ($H_0={H0_lcdm}$, $\\Omega_m=0.315$)')
ax.plot(z_arr[z_arr<=2.5],H_janus[z_arr<=2.5],color='#1B5E20',lw=2.8,
    label=f'Janus ($\\eta=1.045$, $H_0={H0_janus}$)')
ax.errorbar(z_obs,H_obs,yerr=H_err,fmt='o',color='#1A237E',
    ecolor='#5C6BC0',capsize=3.5,ms=6,lw=1.5,
    label='CC+BAO (données observationnelles)',zorder=5)
ax.set_xlabel('Redshift $z$',fontsize=13); ax.set_ylabel('$H(z)$ [km/s/Mpc]',fontsize=13)
ax.set_title('$H(z)$ : Janus vs $\\Lambda$CDM',fontsize=14,fontweight='bold')
ax.set_xlim(0,2.5); ax.set_ylim(50,280)
ax.legend(fontsize=11,loc='upper left',framealpha=0.95); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('fig_hz.png',dpi=200,bbox_inches='tight',facecolor='white')
print("fig_hz.png ✓")

print("\n=== Résultat ===")
for f in ['fig_chi2.png','fig_timeseries.png','frame_peak.png','fig_hz.png']:
    print(f"  {'✓' if os.path.exists(f) else '✗ MANQUANT'} {f}")
```

## 5. Vérification

```bash
ls -lh fig_chi2.png fig_timeseries.png frame_peak.png fig_hz.png
```

Les 4 PNG sont prêts à uploader dans Claude.ai web.
