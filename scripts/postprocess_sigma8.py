#!/usr/bin/env python3
"""
Post-processing σ_8 calculator (Janus production).

Reads snapshot binaries (V3 format), computes proper σ_8 with:
  - CIC density on 256³ grid (cell_size = 500/256 = 1.95 Mpc < R_8 = 11.44)
  - FFT 3D of δ_+, δ_-, δ_total
  - Top-hat window W_TH(kR_8) = 3·(sin x − x cos x)/x³
  - σ²_R8 = Σ_k |δ_k|² · W²_TH(kR_8)         (using Parseval, sum over modes)
  - Shot-noise correction: σ²_corrected = σ²_measured − V_box/(N_part · V_R8)

Usage:
    python postprocess_sigma8.py [--once] [--snapdir DIR] [--out CSV]

Default: watches snapshot dir for new files, processes each as it appears,
appends to CSV. With --once: processes all existing snapshots and exits.
"""
import argparse
import os
import struct
import sys
import time
import numpy as np

DEFAULT_SNAPDIR = "/mnt/T2/janus-sim/output/janus_jpp_production/snapshots"
DEFAULT_OUT = "/mnt/T2/janus-sim/output/sigma8_postprocess.csv"

N_GRID = 256          # CIC grid (cell ≈ 1.95 Mpc for L=500)
HUBBLE_LITTLE_H = 0.699
R8_MPC = 8.0 / HUBBLE_LITTLE_H  # ≈ 11.44 Mpc

# ────────────────────────────────────────────────────────────────────────
# Snapshot reader (V3 format — same as render_daemon_adaptive_v2.py)
# ────────────────────────────────────────────────────────────────────────

def read_snapshot_v3(path):
    with open(path, 'rb') as f:
        header = f.read(408)
        n = struct.unpack('<Q', header[16:24])[0]
        a = struct.unpack('<d', header[24:32])[0]
        t_gyr = struct.unpack('<d', header[32:40])[0]
        l_box = struct.unpack('<d', header[40:48])[0]

        dt = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('epsilon', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(n * 36), dtype=dt)

    z = 1.0 / a - 1.0 if a > 0 else 0.0
    return n, a, z, t_gyr, l_box, particles

# ────────────────────────────────────────────────────────────────────────
# CIC density assignment
# ────────────────────────────────────────────────────────────────────────

def cic_density(positions, n_grid, box_size):
    """Cloud-in-Cell density assignment on n_grid³.
    positions: (N,3) in [-L/2, L/2]
    Returns (n_grid, n_grid, n_grid) f64 array of weighted counts.
    """
    cell = box_size / n_grid
    pos = (positions + box_size / 2.0)
    # wrap
    pos = pos - box_size * np.floor(pos / box_size)
    coords = pos / cell
    i0 = np.floor(coords).astype(np.int64) % n_grid
    d = coords - np.floor(coords)
    i1 = (i0 + 1) % n_grid

    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    # 8-corner weighted deposit
    for dx in (0, 1):
        wx = d[:, 0] if dx else (1.0 - d[:, 0])
        ix = i1[:, 0] if dx else i0[:, 0]
        for dy in (0, 1):
            wy = d[:, 1] if dy else (1.0 - d[:, 1])
            iy = i1[:, 1] if dy else i0[:, 1]
            for dz in (0, 1):
                wz = d[:, 2] if dz else (1.0 - d[:, 2])
                iz = i1[:, 2] if dz else i0[:, 2]
                w = wx * wy * wz
                np.add.at(rho, (ix, iy, iz), w)
    return rho

# ────────────────────────────────────────────────────────────────────────
# σ_R8 with top-hat window (Parseval form)
# ────────────────────────────────────────────────────────────────────────

def sigma_R8_top_hat(rho, n_grid, box_size, r_smooth):
    """σ² = Σ_k |δ_k|² · W_TH(kR)² / N_modes_normalization
    Using FFT with default unnormalized convention (forward sum, no 1/N).
    Parseval: Σ_x δ(x)² = (1/N³) Σ_k |δ_k|², so σ² = (1/N³) Σ_k |δ_k|².
    With window: σ²_R = (1/N³) Σ_k |δ_k|² W_TH(kR)²
    """
    n_total = n_grid ** 3
    mean = rho.mean()
    if mean == 0:
        return 0.0
    delta = rho / mean - 1.0

    fft = np.fft.fftn(delta)        # complex (n_grid, n_grid, n_grid)
    # k grid (in 1/Mpc)
    kf = 2.0 * np.pi / box_size
    kx = np.fft.fftfreq(n_grid, d=1.0/n_grid) * kf
    ky = kx
    kz = kx
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Top-hat window (avoid 0/0 at k=0)
    x = K * r_smooth
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(x > 1e-6,
                     3.0 * (np.sin(x) - x * np.cos(x)) / np.where(x != 0, x**3, 1.0),
                     1.0)

    # Power spectrum sum with window²
    P = (fft * np.conj(fft)).real * (W ** 2)
    # Exclude k=0 mode (DC)
    P[0, 0, 0] = 0.0
    # σ² = (1/N³) Σ_k |δ_k|² W²(kR)  [Parseval]
    sigma_sq = P.sum() / (n_total ** 2)
    return np.sqrt(max(sigma_sq, 0.0))

# ────────────────────────────────────────────────────────────────────────
# Main per-snapshot processing
# ────────────────────────────────────────────────────────────────────────

def process_snapshot(path, n_grid=N_GRID):
    """Returns dict of metrics for this snapshot."""
    n, a, z, t_gyr, l_box, particles = read_snapshot_v3(path)

    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)  # m- is encoded as 255 in V3

    n_plus = int(is_plus.sum())
    n_minus = int(is_minus.sum())

    # CIC densities
    rho_plus = cic_density(pos[is_plus], n_grid, l_box)
    rho_minus = cic_density(pos[is_minus], n_grid, l_box)
    rho_total = rho_plus + rho_minus  # combined (assuming equal mass per particle, Petit option A)

    # σ_R8 with top-hat
    sig_plus = sigma_R8_top_hat(rho_plus, n_grid, l_box, R8_MPC)
    sig_minus = sigma_R8_top_hat(rho_minus, n_grid, l_box, R8_MPC)
    sig_total = sigma_R8_top_hat(rho_total, n_grid, l_box, R8_MPC)

    # Shot noise: σ²_shot = V_box / (N · V_R8)
    v_box = l_box ** 3
    v_r8 = (4.0/3.0) * np.pi * R8_MPC**3
    s2_shot_plus = v_box / (n_plus * v_r8) if n_plus > 0 else 0.0
    s2_shot_minus = v_box / (n_minus * v_r8) if n_minus > 0 else 0.0
    s2_shot_total = v_box / ((n_plus + n_minus) * v_r8)

    sig_plus_corr = np.sqrt(max(sig_plus**2 - s2_shot_plus, 0.0))
    sig_minus_corr = np.sqrt(max(sig_minus**2 - s2_shot_minus, 0.0))
    sig_total_corr = np.sqrt(max(sig_total**2 - s2_shot_total, 0.0))

    k_nyq = np.pi * n_grid / l_box

    return {
        'snapshot': os.path.basename(path),
        'a': a, 'z': z, 't_Gyr': t_gyr,
        'n_plus': n_plus, 'n_minus': n_minus,
        'sigma8_plus_raw': sig_plus,
        'sigma8_minus_raw': sig_minus,
        'sigma8_total_raw': sig_total,
        'sigma8_plus_corrected': sig_plus_corr,
        'sigma8_minus_corrected': sig_minus_corr,
        'sigma8_total_corrected': sig_total_corr,
        's2_shot_plus': s2_shot_plus,
        's2_shot_minus': s2_shot_minus,
        's2_shot_total': s2_shot_total,
        'k_nyquist': k_nyq,
    }

# ────────────────────────────────────────────────────────────────────────
# Main loop / watcher
# ────────────────────────────────────────────────────────────────────────

CSV_HEADER = ("snapshot,a,z,t_Gyr,n_plus,n_minus,"
              "sigma8_plus_raw,sigma8_minus_raw,sigma8_total_raw,"
              "sigma8_plus_corrected,sigma8_minus_corrected,sigma8_total_corrected,"
              "s2_shot_plus,s2_shot_minus,s2_shot_total,k_nyquist\n")

def append_row(out_path, row):
    """Append a row to CSV (creates with header if absent)."""
    new_file = not os.path.exists(out_path)
    with open(out_path, 'a') as f:
        if new_file:
            f.write(CSV_HEADER)
        f.write(f"{row['snapshot']},{row['a']:.6f},{row['z']:.4f},{row['t_Gyr']:.4f},"
                f"{row['n_plus']},{row['n_minus']},"
                f"{row['sigma8_plus_raw']:.6e},{row['sigma8_minus_raw']:.6e},{row['sigma8_total_raw']:.6e},"
                f"{row['sigma8_plus_corrected']:.6e},{row['sigma8_minus_corrected']:.6e},{row['sigma8_total_corrected']:.6e},"
                f"{row['s2_shot_plus']:.6e},{row['s2_shot_minus']:.6e},{row['s2_shot_total']:.6e},"
                f"{row['k_nyquist']:.4f}\n")

def already_processed(out_path):
    """Returns set of snapshot filenames already in CSV."""
    if not os.path.exists(out_path):
        return set()
    seen = set()
    with open(out_path) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.split(',', 1)
            if parts:
                seen.add(parts[0])
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapdir', default=DEFAULT_SNAPDIR)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--once', action='store_true', help='Process all existing snaps and exit')
    ap.add_argument('--n-grid', type=int, default=N_GRID)
    ap.add_argument('--poll', type=float, default=30.0, help='Poll interval (sec)')
    args = ap.parse_args()

    print(f"=== σ_8 post-processor ===")
    print(f"  snapdir : {args.snapdir}")
    print(f"  out     : {args.out}")
    print(f"  n_grid  : {args.n_grid} → cell_size = {500/args.n_grid:.3f} Mpc")
    print(f"  R_8     : {R8_MPC:.3f} Mpc")
    print(f"  k_nyq   : {np.pi * args.n_grid / 500:.3f} 1/Mpc")
    print()

    seen = already_processed(args.out)
    print(f"  {len(seen)} snapshots already processed")

    while True:
        # List all snapshot files, sorted by step number
        files = sorted([f for f in os.listdir(args.snapdir) if f.endswith('.bin')])
        new_files = [f for f in files if f not in seen]

        if new_files:
            print(f"  [{time.strftime('%H:%M:%S')}] {len(new_files)} new snapshot(s) to process")
            for fname in new_files:
                path = os.path.join(args.snapdir, fname)
                # Wait for file to be fully written (size stable)
                size_a = os.path.getsize(path)
                time.sleep(2)
                size_b = os.path.getsize(path)
                if size_a != size_b:
                    print(f"    {fname}: still being written, skip this round")
                    continue
                try:
                    t0 = time.time()
                    row = process_snapshot(path, args.n_grid)
                    dt = time.time() - t0
                    append_row(args.out, row)
                    seen.add(fname)
                    print(f"    {fname}: z={row['z']:.3f}  σ8+={row['sigma8_plus_corrected']:.4f}  "
                          f"σ8-={row['sigma8_minus_corrected']:.4f}  "
                          f"σ8tot={row['sigma8_total_corrected']:.4f}  ({dt:.1f}s)")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"    {fname}: ERROR {e}")
                    sys.stdout.flush()
        elif args.once:
            print("  All existing snapshots processed; exiting (--once)")
            break

        if args.once:
            break

        time.sleep(args.poll)

if __name__ == '__main__':
    main()
