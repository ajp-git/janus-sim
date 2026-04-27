#!/usr/bin/env python3
"""Spectre angulaire sur snap v7 step 1500."""
import struct, numpy as np
from numpy.fft import fft2, fftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Charger snapshot v3
def load(p):
    with open(p, 'rb') as f:
        f.read(8)  # magic
        f.read(4)  # version
        f.read(4)  # header_size
        n_total = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t_gyr = struct.unpack('<d', f.read(8))[0]
        l_box = struct.unpack('<d', f.read(8))[0]
        f.seek(408)  # skip to particles

        particle_dtype = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('eps', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(), dtype=particle_dtype)

    z = 1.0 / a - 1.0
    pos = particles['pos']
    signs = np.where(particles['sign'] == 1, 1, -1).astype(np.int32)
    return pos, signs, z, l_box


def angular_spectrum(image, n_angles=72):
    fft = np.abs(fftshift(fft2(image - image.mean())))
    n = image.shape[0]
    cy, cx = n//2, n//2
    angles = np.linspace(0, 2*np.pi, n_angles)
    signal = np.zeros(n_angles)
    for i, theta in enumerate(angles):
        for r in range(5, n//2 - 2):
            y = int(cy + r * np.sin(theta))
            x = int(cx + r * np.cos(theta))
            if 0 <= x < n and 0 <= y < n:
                signal[i] += fft[y, x]
    return angles, signal / signal.mean()


snap_path = '/mnt/T2/janus-sim/output/janus_adaptive_v7/snapshots/snap_01500.bin'
pos, signs, z, L = load(snap_path)
half = L / 2

print(f"Snapshot z={z:.3f}, L={L} Mpc, N={len(pos):,}")

# Slab |z| < 5% de L (pareil que validate_ics.py)
mask_z = np.abs(pos[:, 2]) < L * 0.05
pos_slab = pos[mask_z]
signs_slab = signs[mask_z]
print(f"Slab |z|<{L*0.05:.0f} Mpc: {len(pos_slab):,} particles")

# Histogrammes
nbin = 128
bins = np.linspace(-half, half, nbin+1)
mp = pos_slab[signs_slab > 0]
mm = pos_slab[signs_slab < 0]
h_plus, _, _ = np.histogram2d(mp[:, 0], mp[:, 1], bins=bins)
h_minus, _, _ = np.histogram2d(mm[:, 0], mm[:, 1], bins=bins)
h_tot = h_plus + h_minus

print(f"\nm+ in slab: {len(mp):,}")
print(f"m- in slab: {len(mm):,}")

# Spectres angulaires
print("\n=== SPECTRE ANGULAIRE ===")
for name, h in [('m+', h_plus), ('m-', h_minus), ('total', h_tot)]:
    angles, spec = angular_spectrum(h)
    max_axes = [spec[np.argmin(np.abs(angles - np.radians(d)))] for d in [0, 90, 180, 270]]
    max_axis = max(max_axes)
    max_overall = spec.max()
    max_overall_deg = np.degrees(angles[spec.argmax()])
    print(f"{name:6s}: max axes = {max_axis:.3f}  max overall = {max_overall:.3f} @ {max_overall_deg:.0f}°")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, h) in zip(axes, [('m+', h_plus), ('m-', h_minus), ('total', h_tot)]):
    angles, spec = angular_spectrum(h)
    ax.plot(np.degrees(angles), spec)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    for d in [0, 90, 180, 270, 360]:
        ax.axvline(d, color='orange', ls=':', alpha=0.5)
    ax.set_xlabel("angle [°]")
    ax.set_ylabel("spectre FFT normalisé")
    ax.set_title(f"{name} — z={z:.2f}")
    ax.set_ylim(0.5, 1.5)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/T2/janus-sim/output/janus_adaptive_v7/spectre_angulaire_snap_01500.png', dpi=90)
plt.close()
print(f"\nImage : /mnt/T2/janus-sim/output/janus_adaptive_v7/spectre_angulaire_snap_01500.png")
