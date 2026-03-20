#!/usr/bin/env python3
"""Calculate L_J only on populated cells"""

import numpy as np
import struct

def load_snapshot(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 7 * 4), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    signs = np.sign(data[:, 6]).astype(int)
    return pos, signs, n

def compute_LJ_masked(pos, signs, L, cell_size, min_particles=1):
    """
    Compute L_J = σ_P / |∇P| only on populated cells.
    Gradient computed only between pairs of populated neighbors.
    """
    g = int(L / cell_size)
    cell = L / g

    # Bin particles
    rp = np.zeros((g, g, g), dtype=np.float64)
    rm = np.zeros((g, g, g), dtype=np.float64)

    pos_shifted = pos + L/2
    ix = np.clip((pos_shifted[:, 0] / cell).astype(int), 0, g-1)
    iy = np.clip((pos_shifted[:, 1] / cell).astype(int), 0, g-1)
    iz = np.clip((pos_shifted[:, 2] / cell).astype(int), 0, g-1)

    np.add.at(rp, (ix[signs > 0], iy[signs > 0], iz[signs > 0]), 1)
    np.add.at(rm, (ix[signs < 0], iy[signs < 0], iz[signs < 0]), 1)

    total = rp + rm
    mask = total >= min_particles

    # Parity field (only defined where populated)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(total > 0, (rp - rm) / total, np.nan)

    # σ_P on populated cells only
    P_valid = P[mask]
    sigma_P = np.std(P_valid)

    # Gradient magnitude - only between populated neighbors
    # For each populated cell, compute gradient only if neighbor is also populated
    grad_sum = 0.0
    grad_count = 0

    for iz in range(g):
        for iy in range(g):
            for ix in range(g):
                if not mask[ix, iy, iz]:
                    continue

                # Check each direction - only if both endpoints populated
                # X direction
                ix_p = (ix + 1) % g
                ix_m = (ix - 1) % g
                if mask[ix_p, iy, iz] and mask[ix_m, iy, iz]:
                    gx = (P[ix_p, iy, iz] - P[ix_m, iy, iz]) / (2 * cell)
                elif mask[ix_p, iy, iz]:
                    gx = (P[ix_p, iy, iz] - P[ix, iy, iz]) / cell
                elif mask[ix_m, iy, iz]:
                    gx = (P[ix, iy, iz] - P[ix_m, iy, iz]) / cell
                else:
                    gx = 0  # isolated in x

                # Y direction
                iy_p = (iy + 1) % g
                iy_m = (iy - 1) % g
                if mask[ix, iy_p, iz] and mask[ix, iy_m, iz]:
                    gy = (P[ix, iy_p, iz] - P[ix, iy_m, iz]) / (2 * cell)
                elif mask[ix, iy_p, iz]:
                    gy = (P[ix, iy_p, iz] - P[ix, iy, iz]) / cell
                elif mask[ix, iy_m, iz]:
                    gy = (P[ix, iy, iz] - P[ix, iy_m, iz]) / cell
                else:
                    gy = 0

                # Z direction
                iz_p = (iz + 1) % g
                iz_m = (iz - 1) % g
                if mask[ix, iy, iz_p] and mask[ix, iy, iz_m]:
                    gz = (P[ix, iy, iz_p] - P[ix, iy, iz_m]) / (2 * cell)
                elif mask[ix, iy, iz_p]:
                    gz = (P[ix, iy, iz_p] - P[ix, iy, iz]) / cell
                elif mask[ix, iy, iz_m]:
                    gz = (P[ix, iy, iz] - P[ix, iy, iz_m]) / cell
                else:
                    gz = 0

                grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
                if grad_mag > 0:  # only count if gradient is defined
                    grad_sum += grad_mag
                    grad_count += 1

    mean_grad = grad_sum / grad_count if grad_count > 0 else 0
    L_J = sigma_P / mean_grad if mean_grad > 0 else float('inf')

    n_populated = mask.sum()
    pct_populated = 100 * n_populated / (g**3)

    return sigma_P, L_J, mean_grad, n_populated, pct_populated

print("="*70)
print("L_J CALCULATION ON POPULATED CELLS ONLY")
print("="*70)

# Load both snapshots
snap1 = "/mnt/T2/janus-sim/output/janus_v13_1M/snapshots/snap_005000.bin"
snap2 = "/mnt/T2/janus-sim/output/janus_v13_500Mpc/snapshots/snap_005000.bin"

print("\nLoading snapshots...")
pos1, signs1, n1 = load_snapshot(snap1)
pos2, signs2, n2 = load_snapshot(snap2)
print(f"  Run 1: {n1:,} particles (L=200 Mpc)")
print(f"  Run 2: {n2:,} particles (L=500 Mpc)")

# Test at 2 Mpc resolution
cell_size = 2.0

print(f"\n{'='*70}")
print(f"Resolution: {cell_size} Mpc/cell")
print(f"{'='*70}")

print(f"\n{'Run':<20} {'σ_P':<8} {'|∇P|':<10} {'L_J':<10} {'Populated%':<12}")
print("-"*60)

# Run 1
sp1, lj1, grad1, n_pop1, pct1 = compute_LJ_masked(pos1, signs1, 200, cell_size)
print(f"{'Run 1 (1M, L=200)':<20} {sp1:<8.4f} {grad1:<10.4f} {lj1:<10.2f} {pct1:<12.1f}")

# Run 2
sp2, lj2, grad2, n_pop2, pct2 = compute_LJ_masked(pos2, signs2, 500, cell_size)
print(f"{'Run 2 (5M, L=500)':<20} {sp2:<8.4f} {grad2:<10.4f} {lj2:<10.2f} {pct2:<12.1f}")

print(f"\n{'='*70}")
print("COMPARISON:")
print(f"{'='*70}")

print(f"""
  Run 1: L_J = {lj1:.2f} Mpc (σ_P={sp1:.3f}, |∇P|={grad1:.4f})
  Run 2: L_J = {lj2:.2f} Mpc (σ_P={sp2:.3f}, |∇P|={grad2:.4f})

  Ratio L_J(Run2)/L_J(Run1) = {lj2/lj1:.2f}
""")

if abs(lj1 - lj2) < 1.0:
    print("  ✅ L_J CONVERGES! Both runs give L_J ≈ {:.1f} Mpc".format((lj1+lj2)/2))
    print("     This confirms L_J is an intrinsic physical scale of Janus.")
elif lj2 < lj1 * 1.5:
    print(f"  ⚠️ L_J similar but not identical: {lj1:.1f} vs {lj2:.1f} Mpc")
    print("     May need finer resolution or more particles to fully converge.")
else:
    print(f"  ❌ L_J still differs: {lj1:.1f} vs {lj2:.1f} Mpc")
    print("     The gradient calculation may be affected by sparse sampling.")

# Test at multiple resolutions
print(f"\n{'='*70}")
print("RESOLUTION SCAN (Run 2 only):")
print(f"{'='*70}")
print(f"\n{'Cell [Mpc]':<12} {'Grid':<8} {'σ_P':<8} {'|∇P|':<10} {'L_J [Mpc]':<12} {'Pop%':<8}")
print("-"*60)

for cs in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    g = int(500 / cs)
    sp, lj, grad, n_pop, pct = compute_LJ_masked(pos2, signs2, 500, cs)
    print(f"{cs:<12.1f} {g}³{'':<4} {sp:<8.4f} {grad:<10.4f} {lj:<12.2f} {pct:<8.1f}")

print(f"\nExpected L_J ≈ 3-4 Mpc (cosmic filament scale)")
