#!/usr/bin/env python3
"""
σ_R post-processing (Janus production) — multi-scale + cross-power.

For each snapshot:
  - CIC density of δ_+, δ_-, δ_total on N_GRID³
  - σ_R via top-hat FFT smoothing at multiple R: 8, 16, 24, 32 Mpc/h
  - Empirical Poisson floor with N_REALIZATIONS realizations (dispersion reported)
  - Cross-power P_×(k) = ⟨δ_+(k)·δ_-*(k)⟩ binned in |k|
  - Auto-power P_+(k), P_-(k) for cross-correlation r(k) = P_×/√(P_+·P_-)

Two CSVs produced:
  - sigma_postprocess_sigma_R.csv   : multi-scale σ_R per snapshot
  - sigma_postprocess_cross_pk.csv  : cross-power per (snapshot, k_bin)
"""
import argparse
import os
import struct
import sys
import time
import numpy as np

DEFAULT_SNAPDIR = "/mnt/T2/janus-sim/output/janus_jpp_production/snapshots"
DEFAULT_OUT_SIGMA = "/mnt/T2/janus-sim/output/sigma_postprocess_sigma_R.csv"
DEFAULT_OUT_CROSS = "/mnt/T2/janus-sim/output/sigma_postprocess_cross_pk.csv"

N_GRID = 256
HUBBLE_LITTLE_H = 0.699
R_VALUES_MPC = [8.0/HUBBLE_LITTLE_H, 16.0/HUBBLE_LITTLE_H,
                24.0/HUBBLE_LITTLE_H, 32.0/HUBBLE_LITTLE_H]
R_LABELS = ['R8', 'R16', 'R24', 'R32']
N_REALIZATIONS_FLOOR = 8  # bootstrap reps for empirical Poisson floor
N_K_BINS = 30             # bins for cross-power spectrum

# ────────────────────────────────────────────────────────────────────────
# Snapshot reader
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

def cic_density(positions, n_grid, box_size):
    cell = box_size / n_grid
    pos = (positions + box_size / 2.0)
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
                w = wx * wy * wz
                np.add.at(rho, (ix, iy, iz), w)
    return rho

# ────────────────────────────────────────────────────────────────────────
# k-grid & smoothing windows (cached per (n_grid, l_box))
# ────────────────────────────────────────────────────────────────────────

_k_cache = {}

def get_k_grid(n_grid, l_box):
    key = (n_grid, l_box)
    if key not in _k_cache:
        kf = 2.0 * np.pi / l_box
        kx = np.fft.fftfreq(n_grid, d=1.0/n_grid) * kf
        KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        _k_cache[key] = K
    return _k_cache[key]

def top_hat_window(K, R):
    x = K * R
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(x > 1e-6,
                     3.0 * (np.sin(x) - x * np.cos(x)) / np.where(x != 0, x**3, 1.0),
                     1.0)
    return W

# ────────────────────────────────────────────────────────────────────────
# σ_R from density grid (multi-R)
# ────────────────────────────────────────────────────────────────────────

def compute_delta_fft(rho, n_grid):
    """Return δ_k array (FFT of overdensity field)."""
    mean = rho.mean()
    if mean == 0:
        return np.zeros_like(rho, dtype=np.complex128)
    delta = rho / mean - 1.0
    return np.fft.fftn(delta)

def sigma_from_fft(delta_k, K, R, n_total):
    """Compute σ_R from precomputed δ_k and K grid."""
    W = top_hat_window(K, R)
    P = (delta_k * np.conj(delta_k)).real * (W ** 2)
    P[0, 0, 0] = 0.0
    sigma_sq = P.sum() / (n_total ** 2)
    return float(np.sqrt(max(sigma_sq, 0.0)))

def sigmas_multi_R(rho, n_grid, l_box, R_list):
    """Compute σ at all R in R_list. Returns list of sigmas."""
    delta_k = compute_delta_fft(rho, n_grid)
    K = get_k_grid(n_grid, l_box)
    n_total = n_grid ** 3
    return [sigma_from_fft(delta_k, K, R, n_total) for R in R_list]

# ────────────────────────────────────────────────────────────────────────
# Empirical Poisson floor (multi-R, N_REALIZATIONS_FLOOR samples)
# ────────────────────────────────────────────────────────────────────────

def poisson_floor_multi_R(n_particles, n_grid, l_box, R_list,
                          n_realizations=N_REALIZATIONS_FLOOR, seed=42):
    """For random uniform sample, compute σ_R at all R in R_list.
    Returns: list of (mean, std) per R."""
    rng = np.random.default_rng(seed)
    sigmas_per_R = [[] for _ in R_list]
    for k in range(n_realizations):
        pos = rng.uniform(-l_box/2, l_box/2, (n_particles, 3))
        rho = cic_density(pos, n_grid, l_box)
        sigs = sigmas_multi_R(rho, n_grid, l_box, R_list)
        for i, s in enumerate(sigs):
            sigmas_per_R[i].append(s)
    return [(float(np.mean(s)), float(np.std(s))) for s in sigmas_per_R]

# ────────────────────────────────────────────────────────────────────────
# Cross-power spectrum
# ────────────────────────────────────────────────────────────────────────

def compute_cross_power(rho_plus, rho_minus, n_grid, l_box, n_bins=N_K_BINS):
    """Return (k_centers, P_x, P_+, P_-, n_modes) per logarithmic |k| bin.
    P_× = ⟨δ_+(k)·δ_-*(k)⟩ averaged over modes in bin."""
    delta_p_k = compute_delta_fft(rho_plus, n_grid)
    delta_m_k = compute_delta_fft(rho_minus, n_grid)
    K = get_k_grid(n_grid, l_box)

    # Cross power & auto powers (real part for cross since FFT of real signals)
    cross = (delta_p_k * np.conj(delta_m_k)).real
    auto_p = (delta_p_k * np.conj(delta_p_k)).real
    auto_m = (delta_m_k * np.conj(delta_m_k)).real

    # Normalize: P(k) = |δ_k|² · V_box / N_total² is the standard convention
    n_total = n_grid ** 3
    norm = (l_box ** 3) / (n_total ** 2)
    cross *= norm
    auto_p *= norm
    auto_m *= norm

    # k bins (log-spaced from k_f to k_nyq)
    kf = 2.0 * np.pi / l_box
    k_nyq = np.pi * n_grid / l_box
    bins = np.logspace(np.log10(kf), np.log10(k_nyq), n_bins + 1)
    bin_idx = np.searchsorted(bins, K.flatten()) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins) & (K.flatten() > 0)

    K_flat = K.flatten()
    cross_flat = cross.flatten()
    auto_p_flat = auto_p.flatten()
    auto_m_flat = auto_m.flatten()

    k_centers = np.zeros(n_bins)
    P_x = np.zeros(n_bins)
    P_p = np.zeros(n_bins)
    P_m = np.zeros(n_bins)
    n_modes = np.zeros(n_bins, dtype=np.int64)

    np.add.at(k_centers, bin_idx[valid], K_flat[valid])
    np.add.at(P_x, bin_idx[valid], cross_flat[valid])
    np.add.at(P_p, bin_idx[valid], auto_p_flat[valid])
    np.add.at(P_m, bin_idx[valid], auto_m_flat[valid])
    np.add.at(n_modes, bin_idx[valid], 1)

    nz = n_modes > 0
    k_centers[nz] /= n_modes[nz]
    P_x[nz] /= n_modes[nz]
    P_p[nz] /= n_modes[nz]
    P_m[nz] /= n_modes[nz]

    return k_centers, P_x, P_p, P_m, n_modes

# ────────────────────────────────────────────────────────────────────────
# Per-snapshot processing
# ────────────────────────────────────────────────────────────────────────

def process_snapshot(path, n_grid, floor_cache):
    """floor_cache: dict {n_part: list_of_(mean,std) per R}"""
    n, a, z, t_gyr, l_box, particles = read_snapshot_v3(path)
    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)
    n_plus = int(is_plus.sum())
    n_minus = int(is_minus.sum())
    n_tot = n_plus + n_minus

    # CIC
    rho_plus = cic_density(pos[is_plus], n_grid, l_box)
    rho_minus = cic_density(pos[is_minus], n_grid, l_box)
    rho_total = rho_plus + rho_minus

    # σ_R raw at multiple R
    sigs_plus_raw = sigmas_multi_R(rho_plus, n_grid, l_box, R_VALUES_MPC)
    sigs_minus_raw = sigmas_multi_R(rho_minus, n_grid, l_box, R_VALUES_MPC)
    sigs_total_raw = sigmas_multi_R(rho_total, n_grid, l_box, R_VALUES_MPC)

    # Empirical Poisson floors (cached per N_part)
    def get_floors(n_part):
        if n_part not in floor_cache:
            print(f"    [floor] Computing empirical floor for N={n_part} "
                  f"(R={R_VALUES_MPC}, {N_REALIZATIONS_FLOOR} reals)...", flush=True)
            t0 = time.time()
            floors = poisson_floor_multi_R(n_part, n_grid, l_box, R_VALUES_MPC)
            for i, R in enumerate(R_VALUES_MPC):
                m, s = floors[i]
                print(f"    [floor] N={n_part} {R_LABELS[i]} (R={R:.2f}): "
                      f"σ_emp = {m:.5e} ± {s:.5e}", flush=True)
            print(f"    [floor] computed in {time.time()-t0:.1f}s", flush=True)
            floor_cache[n_part] = floors
        return floor_cache[n_part]

    floors_plus = get_floors(n_plus)
    floors_minus = get_floors(n_minus)
    floors_total = get_floors(n_tot)

    # Corrected (using empirical floor): σ²_corr = σ²_raw − σ²_emp
    sigs_plus_corr = [np.sqrt(max(r**2 - f[0]**2, 0.0))
                      for r, f in zip(sigs_plus_raw, floors_plus)]
    sigs_minus_corr = [np.sqrt(max(r**2 - f[0]**2, 0.0))
                       for r, f in zip(sigs_minus_raw, floors_minus)]
    sigs_total_corr = [np.sqrt(max(r**2 - f[0]**2, 0.0))
                       for r, f in zip(sigs_total_raw, floors_total)]

    # Excess in σ²: tells us if signal is above/below noise (sign matters!)
    excess_plus = [r**2 - f[0]**2 for r, f in zip(sigs_plus_raw, floors_plus)]
    excess_minus = [r**2 - f[0]**2 for r, f in zip(sigs_minus_raw, floors_minus)]

    # Cross-power
    k_c, P_x, P_p, P_m, n_modes = compute_cross_power(rho_plus, rho_minus, n_grid, l_box)

    return {
        'snapshot': os.path.basename(path),
        'a': a, 'z': z, 't_Gyr': t_gyr,
        'n_plus': n_plus, 'n_minus': n_minus,
        'sigs_plus_raw': sigs_plus_raw,
        'sigs_minus_raw': sigs_minus_raw,
        'sigs_total_raw': sigs_total_raw,
        'floors_plus': floors_plus,
        'floors_minus': floors_minus,
        'floors_total': floors_total,
        'sigs_plus_corr': sigs_plus_corr,
        'sigs_minus_corr': sigs_minus_corr,
        'sigs_total_corr': sigs_total_corr,
        'excess_plus_sq': excess_plus,
        'excess_minus_sq': excess_minus,
        'cross_power': (k_c, P_x, P_p, P_m, n_modes),
    }

# ────────────────────────────────────────────────────────────────────────
# CSV output
# ────────────────────────────────────────────────────────────────────────

def make_sigma_header():
    cols = ['snapshot', 'a', 'z', 't_Gyr', 'n_plus', 'n_minus']
    for sp in ['plus', 'minus', 'total']:
        for label in R_LABELS:
            cols.extend([f'sigma_{label}_{sp}_raw',
                         f'sigma_{label}_{sp}_emp_mean',
                         f'sigma_{label}_{sp}_emp_std',
                         f'sigma_{label}_{sp}_corrected'])
    for label in R_LABELS:
        cols.extend([f'excess_{label}_plus_sq',
                     f'excess_{label}_minus_sq'])
    return ','.join(cols) + '\n'

def append_sigma_row(out_path, row):
    new_file = not os.path.exists(out_path)
    with open(out_path, 'a') as f:
        if new_file:
            f.write(make_sigma_header())
        vals = [row['snapshot'], f"{row['a']:.6f}", f"{row['z']:.4f}",
                f"{row['t_Gyr']:.4f}", str(row['n_plus']), str(row['n_minus'])]
        for sp_key, raw_key, floor_key, corr_key in [
            ('plus',  'sigs_plus_raw',  'floors_plus',  'sigs_plus_corr'),
            ('minus', 'sigs_minus_raw', 'floors_minus', 'sigs_minus_corr'),
            ('total', 'sigs_total_raw', 'floors_total', 'sigs_total_corr')]:
            for i, label in enumerate(R_LABELS):
                vals.append(f"{row[raw_key][i]:.6e}")
                vals.append(f"{row[floor_key][i][0]:.6e}")
                vals.append(f"{row[floor_key][i][1]:.6e}")
                vals.append(f"{row[corr_key][i]:.6e}")
        for i, _ in enumerate(R_LABELS):
            vals.append(f"{row['excess_plus_sq'][i]:.6e}")
            vals.append(f"{row['excess_minus_sq'][i]:.6e}")
        f.write(','.join(vals) + '\n')

CROSS_HEADER = "snapshot,a,z,k_center,P_cross,P_plus,P_minus,n_modes,r_correlation\n"

def append_cross_rows(out_path, row):
    new_file = not os.path.exists(out_path)
    k_c, P_x, P_p, P_m, n_modes = row['cross_power']
    with open(out_path, 'a') as f:
        if new_file:
            f.write(CROSS_HEADER)
        for i in range(len(k_c)):
            if n_modes[i] == 0:
                continue
            r_xx = (P_x[i] / np.sqrt(P_p[i] * P_m[i])
                    if P_p[i] > 0 and P_m[i] > 0 else 0.0)
            f.write(f"{row['snapshot']},{row['a']:.6f},{row['z']:.4f},"
                    f"{k_c[i]:.6e},{P_x[i]:.6e},{P_p[i]:.6e},{P_m[i]:.6e},"
                    f"{int(n_modes[i])},{r_xx:.6f}\n")

def already_processed(out_path):
    if not os.path.exists(out_path):
        return set()
    seen = set()
    with open(out_path) as f:
        next(f, None)
        for line in f:
            parts = line.split(',', 1)
            if parts:
                seen.add(parts[0])
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapdir', default=DEFAULT_SNAPDIR)
    ap.add_argument('--out-sigma', default=DEFAULT_OUT_SIGMA)
    ap.add_argument('--out-cross', default=DEFAULT_OUT_CROSS)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--n-grid', type=int, default=N_GRID)
    ap.add_argument('--poll', type=float, default=30.0)
    args = ap.parse_args()

    print(f"=== σ_R post-processor (multi-scale + cross-power) ===")
    print(f"  snapdir   : {args.snapdir}")
    print(f"  out_sigma : {args.out_sigma}")
    print(f"  out_cross : {args.out_cross}")
    print(f"  n_grid    : {args.n_grid} → cell_size = {500/args.n_grid:.3f} Mpc")
    print(f"  R values  : {R_LABELS} = {[f'{r:.2f}' for r in R_VALUES_MPC]} Mpc")
    print(f"  k_nyq     : {np.pi * args.n_grid / 500:.3f} 1/Mpc")
    print(f"  Floor reals: {N_REALIZATIONS_FLOOR}")
    print(f"  k bins   : {N_K_BINS}")
    print()

    seen = already_processed(args.out_sigma)
    print(f"  {len(seen)} snapshots already processed")
    floor_cache = {}

    while True:
        files = sorted([f for f in os.listdir(args.snapdir) if f.endswith('.bin')])
        new_files = [f for f in files if f not in seen]

        if new_files:
            print(f"  [{time.strftime('%H:%M:%S')}] {len(new_files)} new snapshot(s)", flush=True)
            for fname in new_files:
                path = os.path.join(args.snapdir, fname)
                size_a = os.path.getsize(path)
                time.sleep(2)
                if os.path.getsize(path) != size_a:
                    print(f"    {fname}: still being written, skip", flush=True)
                    continue
                try:
                    t0 = time.time()
                    row = process_snapshot(path, args.n_grid, floor_cache)
                    dt = time.time() - t0
                    append_sigma_row(args.out_sigma, row)
                    append_cross_rows(args.out_cross, row)
                    seen.add(fname)
                    # Compact summary line
                    s8p = row['sigs_plus_raw'][0]
                    e8p = row['floors_plus'][0][0]
                    s8m = row['sigs_minus_raw'][0]
                    s32p = row['sigs_plus_raw'][3]
                    e32p = row['floors_plus'][3][0]
                    print(f"    {fname}: z={row['z']:.3f}  "
                          f"σ_R8(+)raw={s8p:.4f}/emp={e8p:.4f}  "
                          f"σ_R32(+)raw={s32p:.4f}/emp={e32p:.4f}  "
                          f"σ_R8(-)raw={s8m:.4f}  ({dt:.1f}s)", flush=True)
                except Exception as e:
                    print(f"    {fname}: ERROR {e}", flush=True)
                    import traceback
                    traceback.print_exc()
        elif args.once:
            print("  All processed; exiting (--once)")
            break
        if args.once:
            break
        time.sleep(args.poll)

if __name__ == '__main__':
    main()
