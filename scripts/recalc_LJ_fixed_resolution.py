#!/usr/bin/env python3
"""Recalculate L_J with fixed physical resolution (not fixed grid)"""

import numpy as np
import struct

def load_snapshot(path):
    """Load snapshot: N (u64), then N × (x,y,z,vx,vy,vz,sign) as f32"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 7 * 4), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    signs = np.sign(data[:, 6]).astype(int)
    return pos, signs, n

def compute_LJ(pos, signs, L, cell_size):
    """Compute L_J = σ_P / |∇P| with given cell size"""
    g = int(L / cell_size)
    cell = L / g

    # Bin particles
    rp = np.zeros((g, g, g))
    rm = np.zeros((g, g, g))

    # Shift to [0, L] range
    pos_shifted = pos + L/2

    for i in range(len(pos)):
        ix = min(int(pos_shifted[i, 0] / cell), g-1)
        iy = min(int(pos_shifted[i, 1] / cell), g-1)
        iz = min(int(pos_shifted[i, 2] / cell), g-1)
        if signs[i] > 0:
            rp[ix, iy, iz] += 1
        else:
            rm[ix, iy, iz] += 1

    # Parity field P = (ρ+ - ρ-) / (ρ+ + ρ-)
    total = rp + rm
    P = np.where(total > 0, (rp - rm) / total, 0)

    # σ_P
    P_mean = P.mean()
    sigma_P = np.sqrt(((P - P_mean)**2).mean())

    # Gradient magnitude (periodic)
    grad_mag = 0
    for iz in range(g):
        for iy in range(g):
            for ix in range(g):
                gx = (P[(ix+1)%g, iy, iz] - P[(ix-1)%g, iy, iz]) / (2*cell)
                gy = (P[ix, (iy+1)%g, iz] - P[ix, (iy-1)%g, iz]) / (2*cell)
                gz = (P[ix, iy, (iz+1)%g] - P[ix, iy, (iz-1)%g]) / (2*cell)
                grad_mag += np.sqrt(gx**2 + gy**2 + gz**2)

    mean_grad = grad_mag / (g**3)
    L_J = sigma_P / mean_grad if mean_grad > 0 else 0

    return sigma_P, L_J, g

# Test with different cell sizes
print("="*70)
print("L_J CONVERGENCE TEST — Fixed physical resolution")
print("="*70)

# Run 1: L=200, 1M
snap1 = "/mnt/T2/janus-sim/output/janus_v13_1M/snapshots/snap_005000.bin"
# Run 2: L=500, 5M
snap2 = "/mnt/T2/janus-sim/output/janus_v13_500Mpc/snapshots/snap_005000.bin"

for label, snap_path, L in [("Run1 (1M, L=200)", snap1, 200),
                             ("Run2 (5M, L=500)", snap2, 500)]:
    print(f"\n{label}:")
    pos, signs, n = load_snapshot(snap_path)
    print(f"  Loaded {n:,} particles")

    for cell_size in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        if cell_size > L/10:  # need at least 10 cells
            continue
        try:
            sigma_P, L_J, g = compute_LJ(pos, signs, L, cell_size)
            print(f"  cell={cell_size:.1f} Mpc (grid {g}³): σ_P={sigma_P:.4f}, L_J={L_J:.2f} Mpc")
        except Exception as e:
            print(f"  cell={cell_size:.1f} Mpc: Error - {e}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("  If L_J converges to same value for both runs → intrinsic scale")
print("  If L_J differs → resolution or physics difference")
print("="*70)
