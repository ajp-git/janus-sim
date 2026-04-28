#!/usr/bin/env python3
"""
Validation IC complète (post-fix IFFT 3D) :
  - Visual scatter (déjà fait par anisotropy_test.py, ce script ajoute les autres tests)
  - Ratio P[L/8] / neighbors (test pic spurious < 1.5)
  - σ_8(m+) et σ_8(m-) à z=10 (target [0.05, 0.15] / [0.05, 0.20])

Lit le snapshot V3 binary, calcule en CIC 256³ + top-hat FFT.
"""
import sys
import struct
import numpy as np

SNAP = sys.argv[1] if len(sys.argv) > 1 else \
       "/mnt/T2/janus-sim/output/janus_jpp_production/snapshots/snap_000000.bin"
N_GRID = 256
HUBBLE_LITTLE_H = 0.699
R8_MPC = 8.0 / HUBBLE_LITTLE_H

def read_v3(path):
    with open(path, 'rb') as f:
        h = f.read(408)
        n = struct.unpack('<Q', h[16:24])[0]
        a = struct.unpack('<d', h[24:32])[0]
        l_box = struct.unpack('<d', h[40:48])[0]
        dt = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(n * 36), dtype=dt)
    return n, a, 1.0/a-1.0, l_box, particles

def cic(positions, n_grid, box_size):
    cell = box_size / n_grid
    pos = (positions + box_size/2.0)
    pos = pos - box_size * np.floor(pos / box_size)
    coords = pos / cell
    i0 = np.floor(coords).astype(np.int64) % n_grid
    d = coords - np.floor(coords)
    i1 = (i0 + 1) % n_grid
    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    for dx in (0,1):
        wx = d[:,0] if dx else (1.0 - d[:,0])
        ix = i1[:,0] if dx else i0[:,0]
        for dy in (0,1):
            wy = d[:,1] if dy else (1.0 - d[:,1])
            iy = i1[:,1] if dy else i0[:,1]
            for dz in (0,1):
                wz = d[:,2] if dz else (1.0 - d[:,2])
                iz = i1[:,2] if dz else i0[:,2]
                w = wx*wy*wz
                np.add.at(rho, (ix, iy, iz), w)
    return rho

def sigma_R(rho, n_grid, box_size, R):
    n_total = n_grid**3
    mean = rho.mean()
    if mean == 0: return 0.0
    delta = rho/mean - 1.0
    fft = np.fft.fftn(delta)
    kf = 2*np.pi/box_size
    kx = np.fft.fftfreq(n_grid, d=1.0/n_grid) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    x = K*R
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(x > 1e-6,
                     3*(np.sin(x) - x*np.cos(x))/np.where(x!=0, x**3, 1.0),
                     1.0)
    P = (fft * np.conj(fft)).real * (W**2)
    P[0,0,0] = 0.0
    return np.sqrt(max(P.sum() / (n_total**2), 0.0))

def directional_pk(rho, n_grid, box_size, axis):
    mean = rho.mean()
    if mean == 0: return None, None
    delta = rho/mean - 1.0
    fft = np.fft.fftn(delta)
    P = (fft * np.conj(fft)).real / (n_grid**3)**2
    other = tuple(i for i in range(3) if i != axis)
    P_along = P.sum(axis=other) / (n_grid * n_grid)
    half = n_grid // 2
    kf = 2*np.pi/box_size
    k_pos = np.abs(np.fft.fftfreq(n_grid, d=1.0/n_grid)[:half+1]) * kf
    P_pos = np.zeros(half+1)
    for i in range(half+1):
        if i == 0:
            P_pos[i] = P_along[i]
        elif i == half and n_grid % 2 == 0:
            P_pos[i] = P_along[i]
        else:
            P_pos[i] = (P_along[i] + P_along[n_grid - i]) / 2.0
    return k_pos, P_pos

def peak_ratio_at(k_target, k_arr, p_arr, neighbors=2):
    """Returns ratio P(k_target) / mean(P at +/- N_neighbors bins around)"""
    idx = np.argmin(np.abs(k_arr - k_target))
    if idx < neighbors or idx > len(k_arr) - neighbors - 1:
        return float('nan')
    near = np.concatenate([p_arr[idx-neighbors:idx], p_arr[idx+1:idx+neighbors+1]])
    if near.mean() == 0:
        return float('nan')
    return p_arr[idx] / near.mean()

def main():
    print(f"=== Validation IC complète : {SNAP} ===")
    n, a, z, l_box, particles = read_v3(SNAP)
    print(f"  N={n}, a={a:.4f}, z={z:.3f}, L={l_box} Mpc")

    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)
    n_plus = is_plus.sum()
    n_minus = is_minus.sum()
    print(f"  N+={n_plus}, N-={n_minus}")

    # CIC
    print("  Computing CIC densities...", flush=True)
    rho_p = cic(pos[is_plus], N_GRID, l_box)
    rho_m = cic(pos[is_minus], N_GRID, l_box)

    # σ_8 with top-hat FFT
    print("  Computing σ_8 (top-hat R_8 = {:.2f} Mpc)...".format(R8_MPC), flush=True)
    sig8_p = sigma_R(rho_p, N_GRID, l_box, R8_MPC)
    sig8_m = sigma_R(rho_m, N_GRID, l_box, R8_MPC)

    # Directional P(k) at L/8 = 62.5 Mpc → k = 2π/62.5 = 0.1005 1/Mpc
    print("  Computing directional power spectra at k=L/8...", flush=True)
    k_L8 = 2 * np.pi / (l_box / 8.0)  # = 0.1005
    print(f"    k_target = 2π/(L/8) = {k_L8:.4f} 1/Mpc")
    ratios_plus = []
    ratios_minus = []
    for axis, label in [(0, 'x'), (1, 'y'), (2, 'z')]:
        k_arr, p_arr = directional_pk(rho_p, N_GRID, l_box, axis)
        ratio = peak_ratio_at(k_L8, k_arr, p_arr)
        ratios_plus.append(ratio)
        print(f"    P_+(k_{label}): ratio[L/8] = {ratio:.4f}")
        k_arr, p_arr = directional_pk(rho_m, N_GRID, l_box, axis)
        ratio = peak_ratio_at(k_L8, k_arr, p_arr)
        ratios_minus.append(ratio)
        print(f"    P_-(k_{label}): ratio[L/8] = {ratio:.4f}")

    print()
    print("=== VERDICT ===")
    crit = []
    crit.append(("sigma_8(m+) z=10", sig8_p, "[0.05, 0.15]", 0.05 <= sig8_p <= 0.15))
    crit.append(("sigma_8(m-) z=10", sig8_m, "[0.05, 0.20]", 0.05 <= sig8_m <= 0.20))
    crit.append(("max ratio L/8 m+", max(ratios_plus), "< 1.5", max(ratios_plus) < 1.5))
    crit.append(("max ratio L/8 m-", max(ratios_minus), "< 1.5", max(ratios_minus) < 1.5))
    all_pass = True
    for name, val, target, ok in crit:
        sym = "✅" if ok else "❌"
        print(f"  {sym} {name}: {val:.4f} (target {target})")
        if not ok: all_pass = False
    print()
    if all_pass:
        print("✅ TOUS LES CRITÈRES PASSENT — IC SAINE")
        sys.exit(0)
    else:
        print("❌ AU MOINS UN CRITÈRE ÉCHOUE — STOP")
        sys.exit(1)

if __name__ == "__main__":
    main()
