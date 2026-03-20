#!/usr/bin/env python3
"""Recalculate L_J(z) with fixed 2 Mpc/cell resolution for both runs"""

import numpy as np
import struct
import glob
import os

def load_snapshot(path):
    """Load snapshot: N (u64), then N × (x,y,z,vx,vy,vz,sign) as f32"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 7 * 4), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    signs = np.sign(data[:, 6]).astype(int)
    return pos, signs, n

def compute_metrics(pos, signs, L, cell_size):
    """Compute σ_P, L_J, and segregation with given cell size"""
    g = int(L / cell_size)
    cell = L / g
    n = len(pos)

    # Bin particles
    rp = np.zeros((g, g, g), dtype=np.float64)
    rm = np.zeros((g, g, g), dtype=np.float64)

    pos_shifted = pos + L/2
    ix = np.clip((pos_shifted[:, 0] / cell).astype(int), 0, g-1)
    iy = np.clip((pos_shifted[:, 1] / cell).astype(int), 0, g-1)
    iz = np.clip((pos_shifted[:, 2] / cell).astype(int), 0, g-1)

    np.add.at(rp, (ix[signs > 0], iy[signs > 0], iz[signs > 0]), 1)
    np.add.at(rm, (ix[signs < 0], iy[signs < 0], iz[signs < 0]), 1)

    # Parity field
    total = rp + rm
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(total > 0, (rp - rm) / total, 0)

    # σ_P
    sigma_P = np.std(P)

    # Gradient (vectorized)
    gx = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)) / (2*cell)
    gy = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) / (2*cell)
    gz = (np.roll(P, -1, axis=2) - np.roll(P, 1, axis=2)) / (2*cell)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    mean_grad = grad_mag.mean()

    L_J = sigma_P / mean_grad if mean_grad > 0 else 0

    # Segregation (COM separation)
    pos_plus = pos[signs > 0]
    pos_minus = pos[signs < 0]

    # Periodic COM
    def periodic_com(positions, L):
        theta = 2 * np.pi * (positions + L/2) / L
        cos_mean = np.cos(theta).mean(axis=0)
        sin_mean = np.sin(theta).mean(axis=0)
        com = L * np.arctan2(sin_mean, cos_mean) / (2 * np.pi)
        return com

    com_p = periodic_com(pos_plus, L)
    com_m = periodic_com(pos_minus, L)

    delta = com_p - com_m
    delta = delta - L * np.round(delta / L)  # minimum image
    seg = np.sqrt(np.sum(delta**2)) / L

    return sigma_P, L_J, seg, g

# Parameters
CELL_SIZE = 2.0  # Mpc - fixed physical resolution

print("="*70)
print(f"L_J(z) RECALCULATION — Fixed {CELL_SIZE} Mpc/cell resolution")
print("="*70)

results = {}

for label, snap_dir, L in [
    ("Run1_1M_L200", "/mnt/T2/janus-sim/output/janus_v13_1M/snapshots", 200),
    ("Run2_5M_L500", "/mnt/T2/janus-sim/output/janus_v13_500Mpc/snapshots", 500)
]:
    print(f"\n{label} (L={L} Mpc, grid={(int(L/CELL_SIZE))}³):")

    snaps = sorted(glob.glob(f"{snap_dir}/snap_*.bin"))
    results[label] = []

    for snap_path in snaps:
        step = int(os.path.basename(snap_path).split('_')[1].split('.')[0])

        # z from step (assuming dt=0.01, 5000 steps, z=5→0)
        # a = 1/(1+z), z_init=5, z_final=0
        # Linear interpolation in tau space approximated as linear in step
        z = 5.0 * (1 - step/5000)

        pos, signs, n = load_snapshot(snap_path)
        sigma_P, L_J, seg, g = compute_metrics(pos, signs, L, CELL_SIZE)

        results[label].append((step, z, sigma_P, L_J, seg))
        print(f"  step {step:5d} z={z:.2f}: σ_P={sigma_P:.4f} L_J={L_J:.2f} Mpc seg={seg:.4f}")

# Save results
print("\n" + "="*70)
print("SUMMARY at z≈0:")
print("="*70)

for label in results:
    data = results[label][-1]  # Last snapshot
    print(f"{label}: σ_P={data[2]:.4f}, L_J={data[3]:.2f} Mpc, seg={data[4]:.4f}")

# Save CSV
csv_path = "/mnt/T2/janus-sim/output/LJ_comparison_2Mpc.csv"
with open(csv_path, 'w') as f:
    f.write("run,step,z,sigma_P,L_J,seg\n")
    for label in results:
        for step, z, sigma_P, L_J, seg in results[label]:
            f.write(f"{label},{step},{z:.4f},{sigma_P:.4f},{L_J:.2f},{seg:.4f}\n")
print(f"\nSaved: {csv_path}")

# Quick comparison plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for label, color in [("Run1_1M_L200", "blue"), ("Run2_5M_L500", "red")]:
    data = np.array(results[label])
    z = data[:, 1]
    sigma_P = data[:, 2]
    L_J = data[:, 3]
    seg = data[:, 4]

    axes[0].plot(z, sigma_P, f'{color[0]}.-', label=label, markersize=4)
    axes[1].plot(z, L_J, f'{color[0]}.-', label=label, markersize=4)
    axes[2].plot(z, seg, f'{color[0]}.-', label=label, markersize=4)

axes[0].set_xlabel('z')
axes[0].set_ylabel('σ_P')
axes[0].set_title(f'Parity variance (cell={CELL_SIZE} Mpc)')
axes[0].legend()
axes[0].invert_xaxis()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('z')
axes[1].set_ylabel('L_J [Mpc]')
axes[1].set_title('Jeans length')
axes[1].axhline(4, color='green', ls='--', label='4 Mpc (filaments)')
axes[1].legend()
axes[1].invert_xaxis()
axes[1].grid(True, alpha=0.3)

axes[2].set_xlabel('z')
axes[2].set_ylabel('Segregation S')
axes[2].set_title('Global segregation')
axes[2].legend()
axes[2].invert_xaxis()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/T2/janus-sim/output/LJ_comparison_2Mpc.png', dpi=150)
print(f"Saved: output/LJ_comparison_2Mpc.png")
