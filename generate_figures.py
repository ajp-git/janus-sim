#!/usr/bin/env python3
"""Generate 4 publication-quality figures for Janus document"""

import pandas as pd
import numpy as np
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
})

def find_file(name, prefer_dir=None):
    """Find file in various locations"""
    if prefer_dir:
        p = os.path.join(prefer_dir, name)
        if os.path.exists(p):
            return p
    for p in [f'./{name}', f'output/{name}', f'output/run_doc/{name}']:
        if os.path.exists(p):
            return p
    hits = glob.glob(f'**/{name}', recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(name)

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 : Chi² scan (η parameter fit)
# ══════════════════════════════════════════════════════════════════
print("Figure 1 : Chi2 scan...")

try:
    chi2_file = find_file('chi2_scan.csv')
    df = pd.read_csv(chi2_file)

    if 'chi2_dof' in df.columns:
        chi2_dof = df['chi2_dof'].values
    elif 'chi2' in df.columns:
        chi2_dof = df['chi2'].values / 1590

    eta = df['eta'].values
    valid = np.isfinite(chi2_dof)
    eta = eta[valid]
    chi2_dof = chi2_dof[valid]

    eta_best = eta[np.argmin(chi2_dof)]
    chi2_best = chi2_dof.min()

    print(f"  eta_best = {eta_best:.3f}, chi2/dof = {chi2_best:.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(eta, chi2_dof, 'o-', color='#1565C0', lw=2, ms=5,
            label='Janus fit SNIa ($H_0=76$ km/s/Mpc)')
    ax.axhline(1.0, color='gray', ls=':', lw=1.5, label='$\\chi^2/\\nu=1$')
    ax.axvline(eta_best, color='#C62828', ls='--', lw=2,
               label=f'$\\eta_{{best}}={eta_best:.3f}$')
    ax.plot(eta_best, chi2_best, 'r*', ms=16, zorder=5)
    ax.annotate(f'$\\chi^2/\\nu={chi2_best:.3f}$',
                xy=(eta_best, chi2_best),
                xytext=(eta_best + 0.02, chi2_best + 0.03),
                fontsize=11, color='#C62828',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFF9C4', ec='#F9A825'))

    ax.set_xlabel('$\\eta = |\\rho_-|/\\rho_+$')
    ax.set_ylabel('$\\chi^2/\\nu$')
    ax.set_title('Fit Pantheon+ SNIa — Janus Cosmology ($H_0=76.0$ km/s/Mpc)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=max(0.4, chi2_best - 0.1))

    plt.tight_layout()
    plt.savefig('fig_chi2.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  fig_chi2.png OK")
except Exception as e:
    print(f"  Error: {e}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 : Time series (segregation and KE)
# ══════════════════════════════════════════════════════════════════
print("Figure 2 : Time series...")

try:
    # Prefer run_doc (fresh data)
    ts_file = find_file('time_series.csv', 'output/run_doc')
    df2 = pd.read_csv(ts_file)
    print(f"  Using: {ts_file}")

    # Check columns - new format has: step,time,ke,ke_ratio,segregation,step_time_s
    if 'segregation' in df2.columns:
        seg_col = 'segregation'
    else:
        seg_col = df2.columns[4] if len(df2.columns) > 4 else None

    valid_mask = df2[seg_col] >= 0
    if valid_mask.sum() < 10:
        raise ValueError("Insufficient valid segregation data")

    s_max = df2.loc[valid_mask, seg_col].max()
    idx_max = df2.loc[valid_mask, seg_col].idxmax()
    step_max = int(df2.loc[idx_max, 'step'])
    ke_max = df2['ke_ratio'].max()
    step_end = int(df2['step'].max())

    # Compute z from time if 'a' not available
    # For z:5->0 over 3600 steps with dtau constant
    # tau goes from -1.32 to 0, a = exp(tau)
    if 'a' in df2.columns:
        z_vals = 1.0 / df2['a'] - 1.0
        z_at_max = float(z_vals.iloc[idx_max])
    else:
        # Estimate: linear in conformal time
        z_start, z_end = 5.0, 0.0
        frac = df2['step'] / step_end
        # a = 1/(1+z), z decreases from 5 to 0
        z_vals = z_start * (1 - frac) + z_end * frac
        z_at_max = float(z_vals.iloc[idx_max])

    print(f"  S_max={s_max:.3f} at step {step_max} (z~{z_at_max:.1f})")
    print(f"  KE_max={ke_max:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot segregation
    ax1.plot(df2.loc[valid_mask, 'step'], df2.loc[valid_mask, seg_col],
             color='#1565C0', lw=0.9, alpha=0.9)
    ax1.axhline(s_max, color='#C62828', ls='--', lw=1.8,
                label=f'$S_{{max}}={s_max:.3f}$ (step {step_max})')
    ax1.axvline(step_max, color='#E65100', ls=':', lw=1.5)

    # Phase labels
    phases = [
        (0, step_end//6, 'Phase 1\nFriction', '#1B5E20'),
        (step_end//6, step_end//2, 'Phase 2\nGrowth', '#F57F17'),
        (step_end//2, step_end*3//4, 'Phase 3\nPlateau', '#BF360C'),
        (step_end*3//4, step_end, 'Phase 4\nDecline', '#4A148C')
    ]
    for (x0, x1, label, c) in phases:
        ax1.axvspan(x0, min(x1, step_end), alpha=0.07, color=c)
        ax1.text((x0 + min(x1, step_end))/2, s_max*0.5, label,
                 ha='center', fontsize=8, color=c, fontweight='bold')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Segregation $S$')
    ax1.set_title(f'Run 500K — Segregation ($\\eta=1.045$, $z:5\\to 0$)')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.2)
    ax1.set_ylim(bottom=0)

    # KE ratio
    ax2.plot(df2['step'], df2['ke_ratio'], color='#2E7D32', lw=0.9)
    ax2.axhline(ke_max, color='#C62828', ls='--', lw=1.8,
                label=f'$KE/KE_0^{{max}}={ke_max:.2f}$')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('$KE / KE_0$')
    ax2.set_title('Kinetic Energy Ratio')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.2)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('fig_timeseries.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  fig_timeseries.png OK")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 : Snapshot at peak (black to white background)
# ══════════════════════════════════════════════════════════════════
print("Figure 3 : Snapshot...")

try:
    # Prefer run_doc frames
    all_frames = sorted(glob.glob('output/run_doc/frames/frame_*.png'))
    if not all_frames:
        all_frames = sorted(glob.glob('output/**/frames/frame_*.png', recursive=True))

    if all_frames:
        from PIL import Image

        # Find frame closest to step_max
        best_frame = all_frames[-1]
        best_diff = float('inf')

        for f in all_frames:
            try:
                n = int(os.path.basename(f).replace('frame_', '').replace('.png', ''))
                if 'step_max' in dir() and abs(n - step_max) < best_diff:
                    best_diff = abs(n - step_max)
                    best_frame = f
            except:
                pass

        img = Image.open(best_frame).convert('RGB')
        data = np.array(img)

        # Convert dark pixels to white
        dark = (data[:,:,0] < 30) & (data[:,:,1] < 30) & (data[:,:,2] < 30)
        data[dark] = [255, 255, 255]

        Image.fromarray(data).save('frame_peak.png', dpi=(200, 200))
        print(f"  frame_peak.png OK (from {os.path.basename(best_frame)})")
    else:
        print("  No frames found")
except Exception as e:
    print(f"  Error: {e}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 : H(z) comparison Janus vs LCDM
# ══════════════════════════════════════════════════════════════════
print("Figure 4 : H(z)...")

H0_janus = 76.0
eta = 1.045
Omega_plus = 1.0 / (1.0 + eta)

H0_lcdm = 67.9
Omega_m = 0.315

z_arr = np.linspace(0, 2.5, 300)
H_janus = H0_janus * np.sqrt(Omega_plus * (1 + z_arr)**3 + (1 - Omega_plus))
H_lcdm = H0_lcdm * np.sqrt(Omega_m * (1 + z_arr)**3 + (1 - Omega_m))

hz_obs = [
    (0.07, 69., 19.6), (0.09, 69., 12.), (0.17, 83., 8.), (0.179, 75., 4.),
    (0.199, 75., 5.), (0.27, 77., 14.), (0.352, 83., 14.), (0.38, 83., 13.5),
    (0.4, 95., 17.), (0.44, 82.6, 7.8), (0.57, 96.8, 3.4), (0.593, 104., 13.),
    (0.6, 87.9, 6.1), (0.68, 92., 8.), (0.73, 97.3, 7.), (0.781, 105., 12.),
    (0.875, 125., 17.), (0.9, 117., 23.), (1.037, 154., 20.), (1.3, 168., 17.),
    (1.363, 160., 33.6), (1.75, 202., 40.), (2.34, 222., 7.), (2.36, 226., 8.)
]
z_obs = np.array([p[0] for p in hz_obs])
H_obs = np.array([p[1] for p in hz_obs])
H_err = np.array([p[2] for p in hz_obs])

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(z_arr, H_lcdm, color='#C62828', ls='--', lw=2.8,
        label=f'$\\Lambda$CDM ($H_0={H0_lcdm}$, $\\Omega_m={Omega_m}$)')
ax.plot(z_arr, H_janus, color='#1B5E20', lw=2.8,
        label=f'Janus ($\\eta={eta}$, $H_0={H0_janus}$)')
ax.errorbar(z_obs, H_obs, yerr=H_err, fmt='o', color='#1A237E',
            ecolor='#5C6BC0', capsize=3.5, ms=6, lw=1.5,
            label='CC+BAO (observations)', zorder=5)

ax.set_xlabel('Redshift $z$', fontsize=13)
ax.set_ylabel('$H(z)$ [km/s/Mpc]', fontsize=13)
ax.set_title('$H(z)$ : Janus vs $\\Lambda$CDM', fontsize=14, fontweight='bold')
ax.set_xlim(0, 2.5)
ax.set_ylim(50, 280)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('fig_hz.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  fig_hz.png OK")

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for f in ['fig_chi2.png', 'fig_timeseries.png', 'frame_peak.png', 'fig_hz.png']:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  {f:20s} OK  ({size:.0f} KB)")
    else:
        print(f"  {f:20s} MISSING")
print("="*50)
