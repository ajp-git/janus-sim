#!/usr/bin/env python3
"""
σ_R diagnostic per AJP spec : multi-scale + analytical Poisson subtraction.
For R_h ∈ [4, 8, 16, 24, 32] Mpc/h (h=0.699):
  σ_R_raw  : top-hat smoothed RMS overdensity
  σ_R_floor: sqrt(V_box / (N · V_R))    (analytical Poisson)
  σ_R_corr : sqrt(max(0, σ²_raw - σ²_floor))
"""
import sys
import struct
import numpy as np

SNAP = sys.argv[1] if len(sys.argv) > 1 else \
       "/mnt/T2/janus-sim/output/janus_jpp_production/snapshots/snap_000000.bin"
N_GRID = 256
HUBBLE_H = 0.699

R_VALUES_HMPC = [4.0, 8.0, 16.0, 24.0, 32.0]   # Mpc/h
R_VALUES_MPC = [r / HUBBLE_H for r in R_VALUES_HMPC]

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
                np.add.at(rho, (ix, iy, iz), wx*wy*wz)
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

def main():
    print(f"=== σ_R diagnostic per AJP spec : {SNAP} ===")
    n, a, z, l_box, particles = read_v3(SNAP)
    print(f"  N={n}, a={a:.4f}, z={z:.3f}, L={l_box} Mpc, h={HUBBLE_H}")
    print()

    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)
    n_plus = int(is_plus.sum())
    n_minus = int(is_minus.sum())
    n_total = n_plus + n_minus
    print(f"  N_+ = {n_plus}, N_- = {n_minus}")

    print("  Computing CIC...", flush=True)
    rho_p = cic(pos[is_plus], N_GRID, l_box)
    rho_m = cic(pos[is_minus], N_GRID, l_box)
    rho_total = rho_p + rho_m

    v_box = l_box ** 3

    print()
    print(f"{'R(h^-1Mpc)':>12} {'R(Mpc)':>10} {'sp_+raw':>10} {'sp_+floor':>10} {'sp_+corr':>10} {'sp_-raw':>10} {'sp_-floor':>10} {'sp_-corr':>10} {'sp_tot_raw':>11} {'sp_tot_floor':>12} {'sp_tot_corr':>12}")
    rows = []
    for R_h, R in zip(R_VALUES_HMPC, R_VALUES_MPC):
        v_R = (4.0/3.0) * np.pi * R**3
        sp_p_raw = sigma_R(rho_p, N_GRID, l_box, R)
        sp_m_raw = sigma_R(rho_m, N_GRID, l_box, R)
        sp_tot_raw = sigma_R(rho_total, N_GRID, l_box, R)
        sp_p_floor = np.sqrt(v_box / (n_plus * v_R))
        sp_m_floor = np.sqrt(v_box / (n_minus * v_R))
        sp_tot_floor = np.sqrt(v_box / (n_total * v_R))
        sp_p_corr = np.sqrt(max(0, sp_p_raw**2 - sp_p_floor**2))
        sp_m_corr = np.sqrt(max(0, sp_m_raw**2 - sp_m_floor**2))
        sp_tot_corr = np.sqrt(max(0, sp_tot_raw**2 - sp_tot_floor**2))
        rows.append((R_h, R, sp_p_raw, sp_p_floor, sp_p_corr,
                     sp_m_raw, sp_m_floor, sp_m_corr,
                     sp_tot_raw, sp_tot_floor, sp_tot_corr))
        print(f"{R_h:>12.1f} {R:>10.2f} {sp_p_raw:>10.4f} {sp_p_floor:>10.4f} {sp_p_corr:>10.4f} {sp_m_raw:>10.4f} {sp_m_floor:>10.4f} {sp_m_corr:>10.4f} {sp_tot_raw:>11.4f} {sp_tot_floor:>12.4f} {sp_tot_corr:>12.4f}")

    print()
    print("=== VERDICT (AJP criteria) ===")
    # σ_R(R=8)_corr m+ and m- : in [0.02, 0.20] → IC normalisation OK
    # σ_R(R=32)_corr m- : in [0.005, 0.05] → IC échelle large OK
    R8 = next((r for r in rows if r[0] == 8.0), None)
    R32 = next((r for r in rows if r[0] == 32.0), None)
    if R8:
        _, _, _, _, sp_p_corr_8, _, _, sp_m_corr_8, _, _, sp_tot_corr_8 = R8
        print(f"  σ_R(R=8 Mpc/h)_corr m+   = {sp_p_corr_8:.4f}  [target 0.02-0.20] : ", end="")
        ok_p = 0.02 <= sp_p_corr_8 <= 0.20
        print("✅" if ok_p else "❌")
        print(f"  σ_R(R=8 Mpc/h)_corr m-   = {sp_m_corr_8:.4f}  [target 0.02-0.20] : ", end="")
        ok_m = 0.02 <= sp_m_corr_8 <= 0.20
        print("✅" if ok_m else "❌")
        print(f"  σ_R(R=8 Mpc/h)_corr tot  = {sp_tot_corr_8:.4f}  [LCDM target ~0.07]")
    if R32:
        _, _, _, _, _, _, _, sp_m_corr_32, _, _, _ = R32
        print(f"  σ_R(R=32 Mpc/h)_corr m-  = {sp_m_corr_32:.4f}  [target 0.005-0.05] : ", end="")
        ok_m_32 = 0.005 <= sp_m_corr_32 <= 0.05
        print("✅" if ok_m_32 else "❌")

    print()
    if R8 and R32:
        all_pass = ok_p and ok_m and ok_m_32
        if all_pass:
            print("✅ IC VALIDÉE — procéder au mini-run")
            sys.exit(0)
        else:
            print("❌ IC NON VALIDÉE — STOP")
            sys.exit(1)

if __name__ == "__main__":
    main()
