#!/usr/bin/env python3
"""
Synthesis plots for μ scan: Diff/Pois and Corr(δ+,δ-) vs redshift
Canonical Janus: μ=19 (flat universe: Ω_tot = Ω_b × (1+μ) = 1.00)
Parameter exploration: μ = {8, 16, 32, 64}
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Output directory
base_dir = "/mnt/T2/janus-sim/output/scan_mu_evolution"

# Load data for each μ - canonical first, then exploration
mu_values = [19, 8, 16, 32, 64]
# Black for canonical, colors for exploration
colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linewidths = [3, 1.5, 1.5, 1.5, 1.5]
linestyles = ['-', '--', '--', '--', '--']

data = {}

for mu in mu_values:
    fname = f"{base_dir}/mu{mu}/time_series.csv"
    df = pd.read_csv(fname)

    # Handle different column formats
    if 'Diff_Pois' in df.columns:
        # μ=64 format
        data[mu] = {
            'z': df['z'].values,
            't_Gyr': df['t_Gyr'].values,
            'Diff_Pois': df['Diff_Pois'].values,
            'Corr': df['Corr'].values
        }
    else:
        # μ=8,16,19,32 format
        data[mu] = {
            'z': df['z'].values,
            't_Gyr': df['t_gyr'].values,
            'Diff_Pois': df['diff_pois'].values,
            'Corr': df['corr_delta'].values
        }

# Instability timescales (from logs)
t_inst = {8: 17.0, 16: 12.4, 19: 11.0, 32: 8.8, 64: 6.3}

# Figure 1: Diff/Pois vs z
fig1, ax1 = plt.subplots(figsize=(12, 8))

for i, mu in enumerate(mu_values):
    z = data[mu]['z']
    dp = data[mu]['Diff_Pois']
    final_dp = dp[-1]
    if mu == 19:
        label = f'μ=19 CANONICAL (Ω=1.00, D/P={final_dp:.2f})'
    else:
        label = f'μ={mu} (D/P={final_dp:.2f})'
    ax1.plot(z, dp, color=colors[i], linewidth=linewidths[i],
             linestyle=linestyles[i], label=label)

ax1.set_xlabel('Redshift z', fontsize=14)
ax1.set_ylabel('Diff/Pois', fontsize=14)
ax1.set_title('Diffusion / Poisson Ratio vs Redshift\nJanus Model: 20M particles, 1000 Mpc box', fontsize=16)
ax1.legend(fontsize=11, loc='upper left')
ax1.set_xlim(4, 0)
ax1.set_ylim(0.95, 2.7)
ax1.grid(True, alpha=0.3)

# Secondary x-axis for cosmic time
ax1_top = ax1.twiny()
z_ticks = [4, 3, 2, 1, 0.5, 0]
t_ticks = [data[19]['t_Gyr'][np.argmin(np.abs(data[19]['z'] - z))] for z in z_ticks]
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks(z_ticks)
ax1_top.set_xticklabels([f'{t:.1f}' for t in t_ticks])
ax1_top.set_xlabel('Cosmic Time (Gyr)', fontsize=12)

ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
fig1.savefig(f'{base_dir}/figure1_diffpois.png', dpi=200)
print(f"Saved: {base_dir}/figure1_diffpois.png")

# Figure 2: Corr(δ+,δ-) vs z
fig2, ax2 = plt.subplots(figsize=(12, 8))

for i, mu in enumerate(mu_values):
    z = data[mu]['z']
    corr = data[mu]['Corr']
    final_corr = corr[-1]
    if mu == 19:
        label = f'μ=19 CANONICAL (Corr={final_corr:.3f})'
    else:
        label = f'μ={mu} (Corr={final_corr:.3f})'
    ax2.plot(z, corr, color=colors[i], linewidth=linewidths[i],
             linestyle=linestyles[i], label=label)

ax2.set_xlabel('Redshift z', fontsize=14)
ax2.set_ylabel('Corr(δ⁺, δ⁻)', fontsize=14)
ax2.set_title('Correlation of Density Contrasts vs Redshift\nJanus Model: 20M particles, 1000 Mpc box', fontsize=16)
ax2.legend(fontsize=11, loc='lower left')
ax2.set_xlim(4, 0)
ax2.grid(True, alpha=0.3)

ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xticks(z_ticks)
ax2_top.set_xticklabels([f'{t:.1f}' for t in t_ticks])
ax2_top.set_xlabel('Cosmic Time (Gyr)', fontsize=12)

ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
fig2.savefig(f'{base_dir}/figure2_corr.png', dpi=200)
print(f"Saved: {base_dir}/figure2_corr.png")

# Print summary table
print("\n" + "="*75)
print("SUMMARY: μ Scan Evolution (20M particles, 1000 Mpc, z=4→0)")
print("="*75)
print(f"{'μ':>6} | {'Ω_tot':>6} | {'t_inst (Gyr)':>12} | {'Diff/Pois|z=0':>14} | {'Corr|z=0':>10}")
print("-"*75)
for mu in mu_values:
    omega = 0.05 * (1 + mu)
    dp_final = data[mu]['Diff_Pois'][-1]
    corr_final = data[mu]['Corr'][-1]
    canonical = " ★" if mu == 19 else ""
    print(f"{mu:>6} | {omega:>6.2f} | {t_inst[mu]:>12.1f} | {dp_final:>14.4f} | {corr_final:>10.4f}{canonical}")
print("="*75)
print("\n★ μ=19 is the canonical Janus cosmology (flat universe, Ω_tot = 1.00)")
print("\nPhysical interpretation:")
print("- Higher μ → faster instability → stronger segregation")
print("- Diff/Pois > 1 indicates clustering beyond Poisson expectation")
print("- Negative Corr(δ⁺,δ⁻) indicates anti-correlation (segregation)")
