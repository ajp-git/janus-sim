#!/usr/bin/env python3
"""Test décisif: σ_P sur cellules non-vides uniquement"""

import numpy as np
import struct

def load_snapshot(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 7 * 4), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    signs = np.sign(data[:, 6]).astype(int)
    return pos, signs, n

def compute_sigma_P(pos, signs, L, cell_size, min_particles=0):
    """
    Compute σ_P with optional filtering of empty cells.
    min_particles=0: all cells (original method)
    min_particles>0: only cells with >= min_particles
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

    # Parity field
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(total > 0, (rp - rm) / total, 0)

    # Mask: only cells with >= min_particles
    if min_particles > 0:
        mask = total >= min_particles
        P_masked = P[mask]
        n_cells = mask.sum()
    else:
        P_masked = P.flatten()
        n_cells = P.size

    sigma_P = np.std(P_masked)
    mean_P = np.mean(P_masked)

    # Also compute stats
    n_empty = (total == 0).sum()
    n_sparse = ((total > 0) & (total < min_particles)).sum() if min_particles > 0 else 0
    mean_occupancy = total[total > 0].mean() if (total > 0).any() else 0

    return sigma_P, mean_P, n_cells, n_empty, n_sparse, mean_occupancy

print("="*70)
print("TEST DÉCISIF: σ_P sur cellules non-vides")
print("="*70)

# Run 2: L=500 Mpc, 5M particles
snap2 = "/mnt/T2/janus-sim/output/janus_v13_500Mpc/snapshots/snap_005000.bin"
L = 500

print(f"\nLoading {snap2}...")
pos, signs, n = load_snapshot(snap2)
print(f"  {n:,} particles loaded")

# Test with different cell sizes and thresholds
cell_sizes = [2.0, 3.0, 4.0, 5.0, 8.0]

print(f"\n{'Cell':<6} {'Grid':<8} {'σ_P all':<10} {'σ_P(≥1)':<10} {'σ_P(≥2)':<10} {'σ_P(≥5)':<10} {'Empty%':<8} {'<N/cell>':<8}")
print("-"*80)

for cell_size in cell_sizes:
    g = int(L / cell_size)

    # All cells
    sp_all, _, n_all, n_empty, _, mean_occ = compute_sigma_P(pos, signs, L, cell_size, min_particles=0)

    # >= 1 particle
    sp_1, _, n_1, _, _, _ = compute_sigma_P(pos, signs, L, cell_size, min_particles=1)

    # >= 2 particles
    sp_2, _, n_2, _, _, _ = compute_sigma_P(pos, signs, L, cell_size, min_particles=2)

    # >= 5 particles
    sp_5, _, n_5, _, _, _ = compute_sigma_P(pos, signs, L, cell_size, min_particles=5)

    empty_pct = 100 * n_empty / (g**3)

    print(f"{cell_size:<6.1f} {g}³{'':<4} {sp_all:<10.4f} {sp_1:<10.4f} {sp_2:<10.4f} {sp_5:<10.4f} {empty_pct:<8.1f} {mean_occ:<8.2f}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)

# The key test at 2 Mpc resolution
sp_all, _, _, n_empty, _, mean_occ = compute_sigma_P(pos, signs, L, 2.0, min_particles=0)
sp_1, _, n_1, _, _, _ = compute_sigma_P(pos, signs, L, 2.0, min_particles=1)
sp_2, _, n_2, _, _, _ = compute_sigma_P(pos, signs, L, 2.0, min_particles=2)

print(f"""
At 2 Mpc resolution (grid 250³):
  - σ_P (all cells):     {sp_all:.4f}
  - σ_P (≥1 particle):   {sp_1:.4f}
  - σ_P (≥2 particles):  {sp_2:.4f}

VERDICT:
""")

if sp_1 > 0.30:
    print(f"  ✅ σ_P increases to {sp_1:.2f} when filtering empty cells")
    print("  → The low σ_P=0.15 was an ARTIFACT of empty cells")
    print("  → Run 2 actually has strong segregation, just sparse sampling")
    print("  → N=15M run will likely show σ_P ~ 0.35")
elif sp_1 < 0.20:
    print(f"  ❌ σ_P stays low at {sp_1:.2f} even when filtering empty cells")
    print("  → The low σ_P is REAL PHYSICS, not an artifact")
    print("  → Large box L=500 Mpc has genuinely diffuse segregation")
    print("  → N=15M run will likely confirm σ_P ~ 0.15-0.20")
else:
    print(f"  ⚠️ σ_P moderately increases to {sp_1:.2f}")
    print("  → Partial artifact, partial physics")
    print("  → N=15M run needed to determine final value")

# Compare with Run 1
snap1 = "/mnt/T2/janus-sim/output/janus_v13_1M/snapshots/snap_005000.bin"
print(f"\n--- Comparison with Run 1 (L=200, 1M) ---")
pos1, signs1, n1 = load_snapshot(snap1)
sp1_all, _, _, _, _, mean_occ1 = compute_sigma_P(pos1, signs1, 200, 2.0, min_particles=0)
sp1_1, _, n1_1, _, _, _ = compute_sigma_P(pos1, signs1, 200, 2.0, min_particles=1)

print(f"  Run 1: σ_P(all)={sp1_all:.4f}, σ_P(≥1)={sp1_1:.4f}, <N/cell>={mean_occ1:.2f}")
print(f"  Run 2: σ_P(all)={sp_all:.4f}, σ_P(≥1)={sp_1:.4f}, <N/cell>={mean_occ:.2f}")

if abs(sp1_1 - sp_1) < 0.05:
    print(f"\n  ✅ When filtering, both runs have similar σ_P!")
    print("  → The difference was purely due to sampling density")
else:
    print(f"\n  Difference persists: {abs(sp1_1 - sp_1):.2f}")
    print("  → Some physical difference remains")
