#!/usr/bin/env python3
"""
Plot H(z)/H₀ for the Janus cosmological model.

From friedmann.rs lines 155-158:
  At z=0: (ȧ/a)² = Ω₊/a³
  → H(z)/H₀ = √(Ω₊/a³) = √(Ω₊) × (1+z)^(3/2)

For η = 1.045:
  Ω₊ = 1/(1+η) = 0.4890
  Ω₋ = η/(1+η) = 0.5110

Note: This is the MATTER-DOMINATED epoch. The Janus model
has NO dark energy term - acceleration comes from the
coupled equations (negative sector interaction).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Observational H(z) data — Cosmic Chronometers + BAO
# (z, H(z), error) in km/s/Mpc
H0_obs = 70.0  # Normalization constant

hz_data = [
    (0.07,  69.0,  19.6),
    (0.09,  69.0,  12.0),
    (0.12,  68.6,  26.2),
    (0.17,  83.0,  8.0),
    (0.20,  72.9,  29.6),
    (0.27,  77.0,  14.0),
    (0.28,  88.8,  36.6),
    (0.35,  82.7,  8.4),
    (0.40,  95.0,  17.0),
    (0.48, 101.0,  27.0),
    (0.57, 100.3,  3.7),
    (0.59, 104.0,  13.0),
    (0.60,  87.9,  6.1),
    (0.73,  97.3,  7.0),
    (0.78, 105.0,  12.0),
    (0.88,  90.0,  40.0),
    (0.90, 117.0,  23.0),
    (1.30, 168.0,  17.0),
    (1.43, 177.0,  18.0),
    (1.53, 140.0,  14.0),
    (1.75, 202.0,  40.0),
    (2.34, 222.0,  7.0),
]

# Normalize by H0
z_obs = np.array([d[0] for d in hz_data])
H_obs = np.array([d[1] / H0_obs for d in hz_data])
H_err = np.array([d[2] / H0_obs for d in hz_data])

# Janus parameters
eta = 1.045
omega_plus = 1.0 / (1.0 + eta)  # = 0.4890
omega_minus = eta / (1.0 + eta)  # = 0.5110

print(f"Janus parameters:")
print(f"  η = {eta}")
print(f"  Ω₊ = {omega_plus:.4f}")
print(f"  Ω₋ = {omega_minus:.4f}")
print(f"  E = Ω₊ - Ω₋ = {omega_plus - omega_minus:.4f}")

# Redshift range
z = np.linspace(0, 5, 500)
a = 1.0 / (1.0 + z)

# Janus Friedmann: H²/H₀² = Ω₊/a³ (matter-dominated, dust)
# From friedmann.rs line 156: (ȧ/a)² = Ω₊/a³
# Normalize so H(z=0)/H₀ = 1.0
H_janus_raw = np.sqrt(omega_plus / a**3)
H_janus = H_janus_raw / H_janus_raw[0]  # z=0 is first element

# ΛCDM comparison (Ω_m = 0.3, Ω_Λ = 0.7)
omega_m_lcdm = 0.3
omega_lambda = 0.7
H_lcdm_raw = np.sqrt(omega_m_lcdm / a**3 + omega_lambda)
H_lcdm = H_lcdm_raw / H_lcdm_raw[0]  # Already ~1.0, but normalize for consistency

# EdS (Einstein-de Sitter, Ω_m = 1)
H_eds_raw = 1.0 / a**1.5
H_eds = H_eds_raw / H_eds_raw[0]  # Already 1.0, but normalize for consistency

# Create figure with dark theme
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
ax.set_facecolor('#161b22')

# Plot curves
ax.plot(z, H_janus, color='#58a6ff', lw=2.5,
        label=f'Janus η={eta} (Ω₊={omega_plus:.3f})')
ax.plot(z, H_lcdm, color='#f85149', lw=2, ls='--',
        label='ΛCDM (Ω_m=0.3, Ω_Λ=0.7)')
ax.plot(z, H_eds, color='#8b949e', lw=1.5, ls=':',
        label='EdS (Ω_m=1)')

# Observational data points with error bars
ax.errorbar(z_obs, H_obs, yerr=H_err, fmt='o', color='white',
            markersize=4, capsize=2, elinewidth=1, capthick=1,
            label='Observations (CC+BAO)', zorder=10)

# Labels and styling
ax.set_xlabel('Redshift z', color='#e6edf3', fontsize=12)
ax.set_ylabel('H(z) / H₀', color='#e6edf3', fontsize=12)
ax.set_title('Hubble Parameter — Janus vs ΛCDM',
             color='#e6edf3', fontsize=14, fontweight='bold')

ax.tick_params(colors='#8b949e')
ax.spines['bottom'].set_color('#30363d')
ax.spines['top'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['right'].set_color('#30363d')

ax.grid(True, alpha=0.15, color='#8b949e')
ax.legend(loc='upper left', facecolor='#21262d',
          edgecolor='#30363d', labelcolor='#e6edf3')

ax.set_xlim(0, 5)
ax.set_ylim(0, 12)

# Add annotation for H₀
ax.annotate(f'H₀ = H(z=0)', xy=(0.05, H_janus[0]),
            xytext=(0.5, 1.5),
            color='#58a6ff', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=0.8))

# Save
plt.tight_layout()
plt.savefig('output/hz_janus.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
print(f"\nSaved: output/hz_janus.png")

# Also show key values
print(f"\nKey values:")
print(f"  H(z=0)/H₀  = {H_janus[0]:.4f} (Janus), {H_lcdm[0]:.4f} (ΛCDM)")
print(f"  H(z=1)/H₀  = {np.interp(1, z, H_janus):.4f} (Janus), {np.interp(1, z, H_lcdm):.4f} (ΛCDM)")
print(f"  H(z=5)/H₀  = {H_janus[-1]:.4f} (Janus), {H_lcdm[-1]:.4f} (ΛCDM)")
