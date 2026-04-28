#!/usr/bin/env python3
"""
Two diagnostic tests for IC isotropy on a snapshot:
  (1) Raw scatter (no adaptive rendering, 3 projections × 2 species)
  (2) Directional power spectra P_+(k_x), P_+(k_y), P_+(k_z)
"""
import sys
import struct
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SNAP = sys.argv[1] if len(sys.argv) > 1 else \
       "/mnt/T2/janus-sim/output/janus_jpp_production/snapshots/snap_002780.bin"
OUT_DIR = "/mnt/T2/janus-sim/output/anisotropy_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_GRID = 256

def read_snapshot_v3(path):
    with open(path, 'rb') as f:
        header = f.read(408)
        n = struct.unpack('<Q', header[16:24])[0]
        a = struct.unpack('<d', header[24:32])[0]
        l_box = struct.unpack('<d', header[40:48])[0]
        dt = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(n * 36), dtype=dt)
    z = 1.0/a - 1.0 if a > 0 else 0.0
    return n, a, z, l_box, particles

def cic_density(positions, n_grid, box_size):
    cell = box_size / n_grid
    pos = (positions + box_size/2.0)
    pos = pos - box_size * np.floor(pos / box_size)
    coords = pos / cell
    i0 = np.floor(coords).astype(np.int64) % n_grid
    d = coords - np.floor(coords)
    i1 = (i0 + 1) % n_grid
    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    for dx in (0, 1):
        wx = d[:, 0] if dx else (1.0 - d[:, 0])
        ix = i1[:, 0] if dx else i0[:, 0]
        for dy in (0, 1):
            wy = d[:, 1] if dy else (1.0 - d[:, 1])
            iy = i1[:, 1] if dy else i0[:, 1]
            for dz in (0, 1):
                wz = d[:, 2] if dz else (1.0 - d[:, 2])
                iz = i1[:, 2] if dz else i0[:, 2]
                w = wx*wy*wz
                np.add.at(rho, (ix, iy, iz), w)
    return rho

# ─── Read ───
print(f"Reading {SNAP}...", flush=True)
n, a, z, l_box, particles = read_snapshot_v3(SNAP)
print(f"  N={n}, a={a:.4f}, z={z:.3f}, L={l_box} Mpc")
pos = particles['pos'].astype(np.float64)
sign = particles['sign']
is_plus = (sign == 1)
is_minus = (sign == 255)
n_plus = is_plus.sum()
n_minus = is_minus.sum()
print(f"  N+={n_plus}, N-={n_minus}", flush=True)

# ─── Test (1): Raw scatter, 3 projections × 2 species ───
print("\n=== TEST 1: raw scatter projections (no adaptive) ===", flush=True)
rng = np.random.default_rng(42)
n_sub = 50000
idx_p = rng.choice(np.where(is_plus)[0], min(n_sub, n_plus), replace=False)
idx_m = rng.choice(np.where(is_minus)[0], min(n_sub, n_minus), replace=False)
pos_p = pos[idx_p]
pos_m = pos[idx_m]

projections = [
    ('xy', 0, 1),
    ('xz', 0, 2),
    ('yz', 1, 2),
]
species = [
    ('plus',  pos_p, 'tab:blue'),
    ('minus', pos_m, 'tab:red'),
]
for proj_name, ax_a, ax_b in projections:
    for sp_name, ps, color in species:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ps[:, ax_a], ps[:, ax_b], s=0.5, alpha=0.3, color=color)
        ax.set_xlim(-l_box/2, l_box/2)
        ax.set_ylim(-l_box/2, l_box/2)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{proj_name[0]} (Mpc)')
        ax.set_ylabel(f'{proj_name[1]} (Mpc)')
        ax.set_title(f'm{sp_name}, {proj_name} projection, z={z:.2f}, N_sub={len(ps)}')
        ax.grid(True, alpha=0.2)
        path = f"{OUT_DIR}/scatter_{sp_name}_{proj_name}_z{z:.2f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  → {path}")

# ─── Test (2): Directional power spectra ───
print("\n=== TEST 2: directional power spectra ===", flush=True)
print(f"  CIC on {N_GRID}^3...", flush=True)
t0 = time.time()
rho_plus = cic_density(pos[is_plus], N_GRID, l_box)
print(f"  CIC m+: {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
rho_minus = cic_density(pos[is_minus], N_GRID, l_box)
print(f"  CIC m-: {time.time()-t0:.1f}s", flush=True)

def directional_power(rho, n_grid, box_size, axis):
    """Returns (k_axis_centers, P(k_axis))."""
    mean = rho.mean()
    delta = rho/mean - 1.0
    fft = np.fft.fftn(delta)
    P = (fft * np.conj(fft)).real / (n_grid**3)**2

    kf = 2.0 * np.pi / box_size
    k_axis = np.fft.fftfreq(n_grid, d=1.0/n_grid) * kf
    # k_axis is the 1D coords along the axis we want.
    # We sum |δ(k)|² over the OTHER two axes for each value of k_axis_index.
    other_axes = tuple(i for i in range(3) if i != axis)
    P_along = P.sum(axis=other_axes)  # 1D array of length n_grid
    n_modes_per_slab = n_grid * n_grid
    P_along /= n_modes_per_slab

    # Take only positive k (folded for real signal symmetry)
    half = n_grid // 2
    k_pos = np.abs(k_axis[:half+1])
    P_pos = np.zeros(half+1)
    for i in range(half+1):
        # k_axis[i] is positive. Add P_along[i].
        # Symmetric mode at k_axis[-i] = -k_axis[i] for i > 0
        if i == 0:
            P_pos[i] = P_along[i]
        elif i == half and n_grid % 2 == 0:
            P_pos[i] = P_along[i]  # nyquist
        else:
            P_pos[i] = (P_along[i] + P_along[n_grid - i]) / 2.0
    return k_pos, P_pos

print("  Computing P_+(k_x), P_+(k_y), P_+(k_z)...", flush=True)
t0 = time.time()
kx_p, Pkx_p = directional_power(rho_plus, N_GRID, l_box, axis=0)
ky_p, Pky_p = directional_power(rho_plus, N_GRID, l_box, axis=1)
kz_p, Pkz_p = directional_power(rho_plus, N_GRID, l_box, axis=2)
print(f"    m+ done: {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
kx_m, Pkx_m = directional_power(rho_minus, N_GRID, l_box, axis=0)
ky_m, Pky_m = directional_power(rho_minus, N_GRID, l_box, axis=1)
kz_m, Pkz_m = directional_power(rho_minus, N_GRID, l_box, axis=2)
print(f"    m- done: {time.time()-t0:.1f}s", flush=True)

# Plot directional power spectra (m+)
for sp_name, kx, Pkx, ky, Pky, kz, Pkz in [
    ('plus',  kx_p, Pkx_p, ky_p, Pky_p, kz_p, Pkz_p),
    ('minus', kx_m, Pkx_m, ky_m, Pky_m, kz_m, Pkz_m),
]:
    fig, ax = plt.subplots(figsize=(10, 7))
    # Skip k=0 (DC mode)
    ax.loglog(kx[1:], Pkx[1:], 'o-', label='P(k_x)', alpha=0.7, color='tab:red')
    ax.loglog(ky[1:], Pky[1:], 's-', label='P(k_y)', alpha=0.7, color='tab:green')
    ax.loglog(kz[1:], Pkz[1:], '^-', label='P(k_z)', alpha=0.7, color='tab:blue')
    ax.set_xlabel('k (1/Mpc)')
    ax.set_ylabel(f'P_{sp_name}(k_axis)')
    ax.set_title(f'Directional power spectrum, m{sp_name}, z={z:.2f}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    path = f"{OUT_DIR}/directional_pk_{sp_name}_z{z:.2f}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")

# Compute relative deviations
def axis_anisotropy_score(Pkx, Pky, Pkz):
    """Returns max relative deviation from mean across axes (averaged over k>0)."""
    # exclude k=0
    P_mean = (Pkx[1:] + Pky[1:] + Pkz[1:]) / 3
    P_max = np.maximum(np.maximum(Pkx[1:], Pky[1:]), Pkz[1:])
    P_min = np.minimum(np.minimum(Pkx[1:], Pky[1:]), Pkz[1:])
    rel_spread = (P_max - P_min) / np.where(P_mean > 0, P_mean, 1)
    return rel_spread.mean(), rel_spread.max()

print("\n=== ANISOTROPY SCORE (rel spread across axes) ===")
mp_mean, mp_max = axis_anisotropy_score(Pkx_p, Pky_p, Pkz_p)
mm_mean, mm_max = axis_anisotropy_score(Pkx_m, Pky_m, Pkz_m)
print(f"  m+ : mean = {mp_mean*100:.1f}%, max = {mp_max*100:.1f}%")
print(f"  m- : mean = {mm_mean*100:.1f}%, max = {mm_max*100:.1f}%")

# Verdict
print("\n=== VERDICT ===")
threshold_iso = 0.10  # 10%
threshold_aniso = 0.30  # 30%
score = max(mp_mean, mm_mean)
if score < threshold_iso:
    print(f"  ✅ ISOTROPE (mean spread {score*100:.1f}% < 10%) — l'anisotropie visuelle est un artefact de rendu")
elif score > threshold_aniso:
    print(f"  ❌ ANISOTROPE (mean spread {score*100:.1f}% > 30%) — vraie anisotropie d'IC")
else:
    print(f"  ⚠ MARGINAL (mean spread {score*100:.1f}%) — investigation supplémentaire requise")
