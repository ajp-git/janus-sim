#!/usr/bin/env python3
"""
λ_Debye post-processing (Janus production).

For each snapshot in z ∈ [3, 8] (a ∈ [1/9, 1/4]):
  - CIC δ_+(x), δ_-(x) on 256³ grid (zero-mean)
  - FFT 3D → δ_+(k), δ_-(k)
  - Cross-power P_×(k) = ⟨δ_+(k)·δ_-*(k)⟩ binned in |k|
  - Inverse FFT to ξ_+-(r), binned in |r|
  - Fit ξ_+-(r) ≈ -A·exp(-r/λ_D)/r over r ∈ [r_min, r_max] (default 5-50 Mpc)
  - Theoretical: λ²_D_theory = c̄² / (4πG · (|ρ_+| + |ρ_-|))

Outputs:
  - sigma_postprocess_lambda_debye.csv
  - lambda_D vs z plot
  - log(λ_D) vs log(a) plot

Skip z>8 (IC transition) and z<3 (box too small for λ_D > L/3).
"""
import argparse
import os
import struct
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DEFAULT_SNAPDIR = "/mnt/T2/janus-sim/output/janus_jpp_production/snapshots"
DEFAULT_OUT_CSV = "/mnt/T2/janus-sim/output/sigma_postprocess_lambda_debye.csv"
DEFAULT_OUT_PLOT_VS_Z = "/mnt/T2/janus-sim/output/lambda_debye_vs_z.png"
DEFAULT_OUT_PLOT_POWER = "/mnt/T2/janus-sim/output/lambda_debye_power_law.png"
DEFAULT_OUT_PLOT_RKZ = "/mnt/T2/janus-sim/output/cross_correlation_r_k_z.png"
DEFAULT_OUT_PLOT_XIR_MULTIZ = "/mnt/T2/janus-sim/output/xi_pm_r_multi_z.png"
DEFAULT_OUT_BOOTSTRAP = "/mnt/T2/janus-sim/output/lambda_debye_bootstrap_z5.txt"
DEFAULT_CROSS_PK_CSV = "/mnt/T2/janus-sim/output/sigma_postprocess_cross_pk.csv"
DEFAULT_XIR_CACHE = "/mnt/T2/janus-sim/output/lambda_debye_xir_cache.npz"

# Physical constants (code units)
G_CODE = 4.499e-15            # Mpc³/(M_sun·Gyr²)
C_LIGHT_MPC_GYR = 306.6       # 2.998e5 km/s in Mpc/Gyr (1 Mpc/Gyr = 977.8 km/s)
M_PER_PARTICLE = 1.7053e12    # M_sun (from production CSV [MASS] line)
ETA = 1.045                   # Janus VSL exponent
DELTA_VSL = (ETA - 1.0) / ETA # δ for c̄²(z) = (1+z)^δ

# Analysis parameters
N_GRID = 256
Z_MIN_FIT = 3.0
Z_MAX_FIT = 8.0
R_MIN_FIT = 5.0   # Mpc
R_MAX_FIT = 50.0  # Mpc
N_R_BINS = 40

# ────────────────────────────────────────────────────────────────────────
# Snapshot reader (V3 format)
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
# CIC density
# ────────────────────────────────────────────────────────────────────────

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
# ξ_+-(r) via FFT cross-power
# ────────────────────────────────────────────────────────────────────────

def cross_power_binned(rho_plus, rho_minus, n_grid, l_box, n_k_bins=30):
    """Compute P_×(k) in 1D k bins (log-spaced).
    Returns (k_centers, P_x, n_modes_in_bin)."""
    mean_p = rho_plus.mean()
    mean_m = rho_minus.mean()
    delta_p = rho_plus / mean_p - 1.0 if mean_p > 0 else np.zeros_like(rho_plus)
    delta_m = rho_minus / mean_m - 1.0 if mean_m > 0 else np.zeros_like(rho_minus)
    fft_p = np.fft.fftn(delta_p)
    fft_m = np.fft.fftn(delta_m)
    cross = (fft_p * np.conj(fft_m)).real

    # Normalize: P(k) = |δ_k|² · V_box / N_total² is the standard convention
    n_total = n_grid ** 3
    norm = (l_box ** 3) / (n_total ** 2)
    cross *= norm

    # k grid
    kf = 2.0 * np.pi / l_box
    kx = np.fft.fftfreq(n_grid, d=1.0/n_grid) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_min = kf
    k_max = kf * n_grid / 2.0
    bins = np.logspace(np.log10(k_min), np.log10(k_max), n_k_bins + 1)
    K_flat = K.flatten()
    cross_flat = cross.flatten()
    idx = np.searchsorted(bins, K_flat) - 1
    valid = (idx >= 0) & (idx < n_k_bins) & (K_flat > 0)

    k_centers = np.zeros(n_k_bins)
    P_sum = np.zeros(n_k_bins)
    n_modes = np.zeros(n_k_bins, dtype=np.int64)
    np.add.at(k_centers, idx[valid], K_flat[valid])
    np.add.at(P_sum, idx[valid], cross_flat[valid])
    np.add.at(n_modes, idx[valid], 1)
    nz = n_modes > 0
    k_centers[nz] /= n_modes[nz]
    P_sum[nz] /= n_modes[nz]
    return k_centers, P_sum, n_modes

def fit_gaussian_kspace(k, P_x, k_min=0.02, k_max=0.10):
    """Fit P_×(k) ≈ -A · exp(-k²·λ²/2)  →  log(-P_×) = log(A) - k²·λ²/2
    NOTE: This is NOT the FT of Yukawa real-space ξ. Kept for reference.
    Returns dict with lambda_D, A, chi2, or None if fit fails."""
    mask = (k >= k_min) & (k <= k_max) & (P_x < 0)
    if mask.sum() < 4:
        return None
    k_fit = k[mask]
    P_fit = P_x[mask]
    y = np.log(-P_fit)
    if not np.all(np.isfinite(y)):
        return None
    x = k_fit ** 2
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = coeffs
    if slope >= 0:
        return None
    lambda_D = float(np.sqrt(-2.0 * slope))
    A = float(np.exp(intercept))
    y_pred = slope * x + intercept
    chi2 = ((y - y_pred) ** 2).sum()
    chi2_red = chi2 / (len(k_fit) - 2)
    return {
        'lambda_D': lambda_D, 'A': A,
        'chi2_red': float(chi2_red), 'n_pts': int(mask.sum()),
        'k_min_used': float(k_fit.min()), 'k_max_used': float(k_fit.max()),
    }

def fit_lorentzian_kspace(k, P_x, k_min=0.02, k_max=0.20):
    """Fit P_×(k) = -A·4π/(k² + 1/λ²) — the EXACT FT of Yukawa ξ(r)=-A·exp(-r/λ)/r.
    Linearization: 1/(-P_×) = (k² + 1/λ²)/(4π·A)
                  → 1/(-P_×) = a + b·k² with a = 1/(4πA·λ²), b = 1/(4πA)
    So a/b = 1/λ² → λ = √(b/a),  4π·A = 1/b.
    Returns dict with lambda_D, A, chi2, or None."""
    mask = (k >= k_min) & (k <= k_max) & (P_x < 0)
    if mask.sum() < 4:
        return None
    k_fit = k[mask]
    P_fit = P_x[mask]
    y = 1.0 / (-P_fit)   # = (k² + 1/λ²)/(4πA)
    if not np.all(np.isfinite(y)):
        return None
    x = k_fit ** 2
    # Linear fit y = a + b·x
    coeffs = np.polyfit(x, y, 1)
    b, a = coeffs   # numpy returns highest-degree first: y = b·x + a
    if b <= 0 or a <= 0:
        return None  # not Yukawa-like
    lambda_D = float(np.sqrt(a / b))   # since a/b = 1/λ²... wait
    # Actually: y = a_intercept + b_slope · x
    # We have y = (1/4πA)·(1/λ² + k²) → intercept = (1/4πA)·(1/λ²), slope = (1/4πA)
    # So slope = 1/(4πA), intercept = slope·(1/λ²) → 1/λ² = intercept/slope
    # numpy polyfit(deg=1) returns [slope, intercept] (highest degree first).
    slope, intercept = b, a
    if slope <= 0 or intercept <= 0:
        return None
    inv_lam2 = intercept / slope
    if inv_lam2 <= 0:
        return None
    lambda_D = float(1.0 / np.sqrt(inv_lam2))
    A = float(1.0 / (4.0 * np.pi * slope))
    y_pred = slope * x + intercept
    chi2 = ((y - y_pred) ** 2).sum() / np.mean(y)**2  # relative chi2
    chi2_red = chi2 / max(len(k_fit) - 2, 1)
    return {
        'lambda_D': lambda_D, 'A': A,
        'chi2_red': float(chi2_red), 'n_pts': int(mask.sum()),
        'k_min_used': float(k_fit.min()), 'k_max_used': float(k_fit.max()),
    }

def cross_correlation_real_space(rho_plus, rho_minus, n_grid, l_box):
    """Compute ξ_+-(r) by inverse FFT of P_×(k).
    Returns (r_centers, xi_r, n_modes_in_bin)."""
    mean_p = rho_plus.mean()
    mean_m = rho_minus.mean()
    delta_p = rho_plus / mean_p - 1.0 if mean_p > 0 else np.zeros_like(rho_plus)
    delta_m = rho_minus / mean_m - 1.0 if mean_m > 0 else np.zeros_like(rho_minus)

    fft_p = np.fft.fftn(delta_p)
    fft_m = np.fft.fftn(delta_m)

    # Cross-power in 3D k space
    P_x_grid = (fft_p * np.conj(fft_m)).real

    # Inverse FFT to get ξ(r) at every grid point
    # ξ(r) = (1/V) Σ_k P(k) exp(-i k·r). Using IFFT convention:
    xi_grid = np.fft.ifftn(P_x_grid).real

    # Now we have ξ at each cell; cell coords correspond to r_x, r_y, r_z
    # in [-L/2, L/2] with origin at corner. For radial binning, we need
    # the "shifted" version where the origin is at the center:
    xi_grid = np.fft.fftshift(xi_grid)

    # Build r grid
    cell = l_box / n_grid
    coords = (np.arange(n_grid) - n_grid // 2) * cell
    RX, RY, RZ = np.meshgrid(coords, coords, coords, indexing='ij')
    R = np.sqrt(RX**2 + RY**2 + RZ**2)

    # Bin by |r|
    # Linear bins from r=0 to r=L/2
    r_max = l_box / 2.0
    bins = np.linspace(0, r_max, N_R_BINS + 1)
    idx = np.searchsorted(bins, R.flatten()) - 1
    valid = (idx >= 0) & (idx < N_R_BINS)

    r_centers = np.zeros(N_R_BINS)
    xi_sum = np.zeros(N_R_BINS)
    n_modes = np.zeros(N_R_BINS, dtype=np.int64)

    R_flat = R.flatten()
    xi_flat = xi_grid.flatten()
    np.add.at(r_centers, idx[valid], R_flat[valid])
    np.add.at(xi_sum, idx[valid], xi_flat[valid])
    np.add.at(n_modes, idx[valid], 1)

    nz = n_modes > 0
    r_centers[nz] /= n_modes[nz]
    xi_sum[nz] /= n_modes[nz]

    return r_centers, xi_sum, n_modes

# ────────────────────────────────────────────────────────────────────────
# Fit ξ_+-(r) = -A·exp(-r/λ)/r  →  log(-ξ·r) = log(A) - r/λ
# ────────────────────────────────────────────────────────────────────────

def fit_yukawa(r, xi, r_min=R_MIN_FIT, r_max=R_MAX_FIT):
    """Fit ξ(r) = -A·exp(-r/λ)/r over r ∈ [r_min, r_max].
    Returns (lambda_D, A, chi2_red, n_pts, r_min_used, r_max_used).
    Returns None if fit fails or insufficient negative points."""
    mask = (r >= r_min) & (r <= r_max) & (xi < 0)
    if mask.sum() < 4:
        return None
    r_fit = r[mask]
    xi_fit = xi[mask]
    # Linearize: log(-ξ·r) = log(A) - r/λ
    y = np.log(-xi_fit * r_fit)
    if not np.all(np.isfinite(y)):
        return None
    # Linear regression in (r_fit, y)
    coeffs, cov = np.polyfit(r_fit, y, 1, cov=True)
    slope, intercept = coeffs
    if slope >= 0:
        # Not exponential decay
        return None
    lambda_D = -1.0 / slope
    A = np.exp(intercept)
    # χ² of fit
    y_pred = slope * r_fit + intercept
    residuals = y - y_pred
    chi2 = (residuals ** 2).sum()
    chi2_red = chi2 / (len(r_fit) - 2)
    return {
        'lambda_D': lambda_D,
        'A': A,
        'chi2_red': chi2_red,
        'n_pts': int(mask.sum()),
        'r_min_used': float(r_fit.min()),
        'r_max_used': float(r_fit.max()),
        'slope': slope, 'intercept': intercept,
    }

# ────────────────────────────────────────────────────────────────────────
# Theoretical λ_D
# ────────────────────────────────────────────────────────────────────────

def lambda_D_theory(z, n_plus, n_minus, l_box, a):
    """Returns (λ_sum, λ_reduced, c̄², ρ_+, ρ_-)  in (Mpc, Mpc, (Mpc/Gyr)², M⊙/Mpc³, M⊙/Mpc³).

    Two conventions:
      - λ²_sum     = c̄² / (4πG · (ρ_+ + ρ_-))             [sum of magnitudes]
      - λ²_reduced = c̄² / (4πG · (ρ_+·ρ_-)/(ρ_+ + ρ_-))    [reduced density, 2-body analog]

    Densities in proper units. For µ=19, λ_reduced = λ_sum · √(N_tot²/(N_+·N_-)) ≈ √20·λ_sum.
    """
    c_bar_sq = (C_LIGHT_MPC_GYR ** 2) * ((1.0 + z) ** DELTA_VSL)
    v_proper = (a ** 3) * (l_box ** 3)
    rho_plus = n_plus * M_PER_PARTICLE / v_proper
    rho_minus = n_minus * M_PER_PARTICLE / v_proper  # magnitude
    rho_sum = rho_plus + rho_minus
    rho_reduced = (rho_plus * rho_minus) / rho_sum if rho_sum > 0 else 0.0
    lam_sum = float(np.sqrt(c_bar_sq / (4 * np.pi * G_CODE * rho_sum)))
    lam_red = float(np.sqrt(c_bar_sq / (4 * np.pi * G_CODE * rho_reduced))) if rho_reduced > 0 else float('nan')
    return lam_sum, lam_red, float(c_bar_sq), float(rho_plus), float(rho_minus)

# ────────────────────────────────────────────────────────────────────────
# Per-snapshot processing
# ────────────────────────────────────────────────────────────────────────

def process_snapshot(path, n_grid):
    n, a, z, t_gyr, l_box, particles = read_snapshot_v3(path)
    if z > Z_MAX_FIT or z < Z_MIN_FIT:
        return {'snapshot': os.path.basename(path), 'a': a, 'z': z, 't_Gyr': t_gyr,
                'skipped': True, 'reason': f'z={z:.2f} outside [{Z_MIN_FIT}, {Z_MAX_FIT}]'}

    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)
    n_plus = int(is_plus.sum())
    n_minus = int(is_minus.sum())

    rho_plus = cic_density(pos[is_plus], n_grid, l_box)
    rho_minus = cic_density(pos[is_minus], n_grid, l_box)

    # Method 1: ξ_+-(r) real-space fit
    r_c, xi, n_modes_r = cross_correlation_real_space(rho_plus, rho_minus, n_grid, l_box)
    fit_r = fit_yukawa(r_c, xi)

    # Method 2 (a): P_×(k) k-space Lorentzian fit (correct FT of Yukawa)
    k_c, P_x, n_modes_k = cross_power_binned(rho_plus, rho_minus, n_grid, l_box)
    fit_k_lorentz = fit_lorentzian_kspace(k_c, P_x)
    # Method 2 (b): Gaussian fit kept for diagnostic
    fit_k_gauss = fit_gaussian_kspace(k_c, P_x)

    # Theory: both conventions
    lam_sum, lam_red, c_bar_sq, rho_p, rho_m = lambda_D_theory(z, n_plus, n_minus, l_box, a)

    result = {
        'snapshot': os.path.basename(path),
        'a': a, 'z': z, 't_Gyr': t_gyr,
        'n_plus': n_plus, 'n_minus': n_minus,
        'l_box': l_box,
        'lambda_D_theory_sum':     lam_sum,
        'lambda_D_theory_reduced': lam_red,
        'c_bar_sq': c_bar_sq,
        'rho_plus_proper':  rho_p,
        'rho_minus_proper': rho_m,
        'skipped': False,
        # Cache raw ξ(r) and P_×(k) data for the multi-z plot
        '_xir_data': (r_c.tolist(), xi.tolist()),
        '_pk_data':   (k_c.tolist(), P_x.tolist()),
    }
    # Real-space ξ_+-(r) fit
    if fit_r is None:
        result.update({
            'lambda_D_from_xir': float('nan'),
            'A_xir': float('nan'),
            'chi2_xir': float('nan'),
            'n_pts_xir': 0,
            'r_min_used': float('nan'),
            'r_max_used': float('nan'),
            'fit_status_xir': 'FAILED',
        })
    else:
        result.update({
            'lambda_D_from_xir': fit_r['lambda_D'],
            'A_xir': fit_r['A'],
            'chi2_xir': fit_r['chi2_red'],
            'n_pts_xir': fit_r['n_pts'],
            'r_min_used': fit_r['r_min_used'],
            'r_max_used': fit_r['r_max_used'],
            'fit_status_xir': 'OK',
        })
    # k-space Lorentzian fit (correct FT of Yukawa)
    if fit_k_lorentz is None:
        result.update({
            'lambda_D_from_Pk_lorentz': float('nan'),
            'A_Pk_lorentz': float('nan'),
            'chi2_Pk_lorentz': float('nan'),
            'n_pts_Pk_lorentz': 0,
            'k_min_lorentz': float('nan'),
            'k_max_lorentz': float('nan'),
            'fit_status_Pk_lorentz': 'FAILED',
        })
    else:
        result.update({
            'lambda_D_from_Pk_lorentz': fit_k_lorentz['lambda_D'],
            'A_Pk_lorentz': fit_k_lorentz['A'],
            'chi2_Pk_lorentz': fit_k_lorentz['chi2_red'],
            'n_pts_Pk_lorentz': fit_k_lorentz['n_pts'],
            'k_min_lorentz': fit_k_lorentz['k_min_used'],
            'k_max_lorentz': fit_k_lorentz['k_max_used'],
            'fit_status_Pk_lorentz': 'OK',
        })

    # k-space Gaussian fit (kept for cross-check, not the right form)
    if fit_k_gauss is None:
        result.update({
            'lambda_D_from_Pk_gauss': float('nan'),
            'fit_status_Pk_gauss': 'FAILED',
        })
    else:
        result.update({
            'lambda_D_from_Pk_gauss': fit_k_gauss['lambda_D'],
            'fit_status_Pk_gauss': 'OK',
        })

    # Combined best estimate: prefer (xir + Pk_lorentz) since both expect Yukawa
    measurements = []
    if fit_r:
        measurements.append(fit_r['lambda_D'])
    if fit_k_lorentz:
        measurements.append(fit_k_lorentz['lambda_D'])

    if len(measurements) == 2:
        rm, km = measurements
        if 0.5 < rm/km < 2.0:
            result['lambda_D_measured'] = float(np.sqrt(rm * km))
            result['method_used'] = 'geo_mean(xir, Pk_lor)'
        else:
            # Disagreement: prefer xir which is fit on the natural form
            result['lambda_D_measured'] = rm
            result['method_used'] = f'xir (Pk_lor disagrees: {rm:.1f} vs {km:.1f})'
    elif len(measurements) == 1:
        result['lambda_D_measured'] = measurements[0]
        result['method_used'] = 'xir' if fit_r else 'Pk_lor'
    else:
        result['lambda_D_measured'] = float('nan')
        result['method_used'] = 'NONE'

    if np.isfinite(result['lambda_D_measured']):
        result['ratio_to_theory_sum'] = result['lambda_D_measured'] / lam_sum if lam_sum > 0 else float('nan')
        result['ratio_to_theory_reduced'] = (result['lambda_D_measured'] / lam_red
                                              if (lam_red > 0 and np.isfinite(lam_red)) else float('nan'))
    else:
        result['ratio_to_theory_sum'] = float('nan')
        result['ratio_to_theory_reduced'] = float('nan')

    return result

# ────────────────────────────────────────────────────────────────────────
# CSV
# ────────────────────────────────────────────────────────────────────────

CSV_HEADER = ("snapshot,a,z,t_Gyr,n_plus,n_minus,l_box,"
              "lambda_D_measured,method_used,"
              "lambda_D_theory_sum,lambda_D_theory_reduced,"
              "ratio_to_theory_sum,ratio_to_theory_reduced,"
              "lambda_D_from_xir,A_xir,chi2_xir,n_pts_xir,r_min_used,r_max_used,fit_status_xir,"
              "lambda_D_from_Pk_lorentz,A_Pk_lorentz,chi2_Pk_lorentz,n_pts_Pk_lorentz,"
              "k_min_lorentz,k_max_lorentz,fit_status_Pk_lorentz,"
              "lambda_D_from_Pk_gauss,fit_status_Pk_gauss,"
              "c_bar_sq,rho_plus_proper,rho_minus_proper\n")

def append_row(out_path, row):
    new_file = not os.path.exists(out_path)
    with open(out_path, 'a') as f:
        if new_file:
            f.write(CSV_HEADER)
        if row.get('skipped'):
            return
        def fnum(x, fmt='{:.4f}'):
            return fmt.format(x) if (isinstance(x, (int, float)) and np.isfinite(x)) else 'nan'
        f.write(
            f"{row['snapshot']},{row['a']:.6f},{row['z']:.4f},{row['t_Gyr']:.4f},"
            f"{row['n_plus']},{row['n_minus']},{row['l_box']:.2f},"
            f"{fnum(row['lambda_D_measured'])},"
            f"\"{row['method_used']}\","
            f"{row['lambda_D_theory_sum']:.4f},{fnum(row['lambda_D_theory_reduced'])},"
            f"{fnum(row.get('ratio_to_theory_sum'))},{fnum(row.get('ratio_to_theory_reduced'))},"
            f"{fnum(row['lambda_D_from_xir'])},{fnum(row['A_xir'],'{:.4e}')},"
            f"{fnum(row['chi2_xir'],'{:.4e}')},{row['n_pts_xir']},"
            f"{fnum(row['r_min_used'],'{:.2f}')},{fnum(row['r_max_used'],'{:.2f}')},"
            f"\"{row['fit_status_xir']}\","
            f"{fnum(row['lambda_D_from_Pk_lorentz'])},{fnum(row['A_Pk_lorentz'],'{:.4e}')},"
            f"{fnum(row['chi2_Pk_lorentz'],'{:.4e}')},{row['n_pts_Pk_lorentz']},"
            f"{fnum(row['k_min_lorentz'],'{:.4f}')},{fnum(row['k_max_lorentz'],'{:.4f}')},"
            f"\"{row['fit_status_Pk_lorentz']}\","
            f"{fnum(row['lambda_D_from_Pk_gauss'])},"
            f"\"{row['fit_status_Pk_gauss']}\","
            f"{row['c_bar_sq']:.4e},{row['rho_plus_proper']:.4e},{row['rho_minus_proper']:.4e}\n"
        )

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

# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────

def make_rkz_map(cross_pk_csv, plot_path):
    """Read cross-power CSV (snap, z, k_center, P_x, P_+, P_-, n_modes, r_correlation),
    build 2D map r(k, z) with contours at r=0, -0.3, -0.5."""
    if not os.path.exists(cross_pk_csv):
        return False
    rows = []
    with open(cross_pk_csv) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            try:
                z = float(parts[2])
                k = float(parts[3])
                r = float(parts[8])
                rows.append((z, k, r))
            except (ValueError, IndexError):
                continue
    if len(rows) < 50:
        return False

    rows = np.array(rows)
    zs = rows[:, 0]
    ks = rows[:, 1]
    rs = rows[:, 2]

    # Build 2D grid for imshow / contour
    z_unique = np.unique(zs)
    k_unique = np.unique(ks)
    if len(z_unique) < 3 or len(k_unique) < 3:
        return False

    # Pivot: r(z, k) matrix
    Z_to_idx = {z: i for i, z in enumerate(sorted(z_unique))}
    K_to_idx = {k: i for i, k in enumerate(sorted(k_unique))}
    R_grid = np.full((len(z_unique), len(k_unique)), np.nan)
    for z, k, r in rows:
        if z in Z_to_idx and k in K_to_idx:
            R_grid[Z_to_idx[z], K_to_idx[k]] = r

    fig, ax = plt.subplots(figsize=(12, 7))
    z_sorted = np.array(sorted(z_unique))
    k_sorted = np.array(sorted(k_unique))
    KK, ZZ = np.meshgrid(k_sorted, z_sorted)
    pcm = ax.pcolormesh(KK, ZZ, R_grid, cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                        shading='nearest')
    cs = ax.contour(KK, ZZ, R_grid, levels=[-0.5, -0.3, 0.0],
                    colors=['darkred', 'red', 'black'], linewidths=[2, 1.5, 1])
    ax.clabel(cs, inline=True, fontsize=9, fmt={-0.5: 'r=-0.5', -0.3: 'r=-0.3', 0.0: 'r=0'})
    ax.set_xscale('log')
    ax.set_xlabel('k (1/Mpc)')
    ax.set_ylabel('redshift z')
    ax.set_title('Cross-correlation r(k, z) = P_×/√(P_+·P_-)')
    ax.invert_yaxis()
    fig.colorbar(pcm, ax=ax, label='r(k)')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True

def update_xir_cache(cache_path, snap_name, z, r_arr, xi_arr):
    """Append/replace one snapshot's ξ(r) data in the npz cache."""
    data = {}
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as f:
                for key in f.files:
                    data[key] = f[key]
        except Exception:
            data = {}
    # Use snapshot name as key suffix
    key = snap_name.replace('.bin', '')
    data[f'r_{key}'] = np.array(r_arr)
    data[f'xi_{key}'] = np.array(xi_arr)
    data[f'z_{key}'] = np.array([z])
    np.savez_compressed(cache_path, **data)

def make_xir_multi_z_plot(cache_path, plot_path, target_zs=(8, 6, 4, 3, 2)):
    """Plot ξ_+-(r) at target z values, log-log."""
    if not os.path.exists(cache_path):
        return False
    try:
        with np.load(cache_path, allow_pickle=True) as f:
            data = {key: f[key] for key in f.files}
    except Exception:
        return False
    # Find unique snapshots and their z
    snaps = set()
    for k in data:
        if k.startswith('r_'):
            snaps.add(k[2:])
    if not snaps:
        return False
    snap_z = []
    for s in snaps:
        zk = f'z_{s}'
        if zk in data:
            snap_z.append((s, float(data[zk][0])))
    if len(snap_z) < 2:
        return False
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.plasma
    found = []
    for target in target_zs:
        # Find closest available z
        best = min(snap_z, key=lambda sz: abs(sz[1] - target))
        if abs(best[1] - target) > 1.0:
            continue
        s, z = best
        r = data[f'r_{s}']
        xi = data[f'xi_{s}']
        # Plot |ξ| in log-log, sign indicator via color and marker style
        positive = xi > 0
        negative = xi < 0
        col = cmap(target / max(target_zs))
        # Plot |ξ| where ξ < 0 (the Yukawa form expects xi < 0)
        ax.plot(r[negative], -xi[negative], 'o-', color=col,
                label=f'z = {z:.2f} (N_neg = {negative.sum()})')
        if positive.any():
            ax.plot(r[positive], xi[positive], 'x', color=col, alpha=0.4)
        found.append(target)
    if not found:
        plt.close(fig)
        return False
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r (Mpc)')
    ax.set_ylabel('|ξ_+-(r)|')
    ax.set_title('ξ_+-(r) at multiple z (filled = anti-corr ξ<0, x = positive ξ>0)')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True

def bootstrap_lambda_D(snap_path, n_grid, n_realizations=50, frac_keep=0.7, seed=42):
    """Bootstrap λ_D estimate: subsample particles and re-fit.
    Returns (lambda_array, mean, std, ci_low, ci_high)."""
    n, a, z, t_gyr, l_box, particles = read_snapshot_v3(snap_path)
    pos = particles['pos'].astype(np.float64)
    sign = particles['sign']
    is_plus = (sign == 1)
    is_minus = (sign == 255)
    pos_plus = pos[is_plus]
    pos_minus = pos[is_minus]
    rng = np.random.default_rng(seed)
    lambdas = []
    print(f"  [bootstrap] z={z:.3f}, {n_realizations} realizations of frac={frac_keep}")
    for k in range(n_realizations):
        # Resample with replacement (or just subsample)
        n_p_sub = int(len(pos_plus) * frac_keep)
        n_m_sub = int(len(pos_minus) * frac_keep)
        idx_p = rng.choice(len(pos_plus), n_p_sub, replace=True)
        idx_m = rng.choice(len(pos_minus), n_m_sub, replace=True)
        rho_p = cic_density(pos_plus[idx_p], n_grid, l_box)
        rho_m = cic_density(pos_minus[idx_m], n_grid, l_box)
        r_c, xi, _ = cross_correlation_real_space(rho_p, rho_m, n_grid, l_box)
        fit = fit_yukawa(r_c, xi)
        if fit is not None:
            lambdas.append(fit['lambda_D'])
        if (k+1) % 10 == 0:
            print(f"    realization {k+1}/{n_realizations}: {len(lambdas)} valid fits so far",
                  flush=True)
    lambdas = np.array(lambdas)
    if len(lambdas) < 5:
        return lambdas, float('nan'), float('nan'), float('nan'), float('nan')
    mean = float(np.mean(lambdas))
    std = float(np.std(lambdas))
    ci_low, ci_high = np.percentile(lambdas, [16, 84]).tolist()
    return lambdas, mean, std, ci_low, ci_high

def make_plots(csv_path, plot_z_path, plot_power_path):
    if not os.path.exists(csv_path):
        return
    # Parse using header
    rows = []
    with open(csv_path) as f:
        header = next(f).strip().split(',')
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(',')]
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            try:
                z = float(d['z'])
                a = float(d['a'])
                lam_meas = float(d['lambda_D_measured']) if d['lambda_D_measured'] != 'nan' else float('nan')
                lam_th_sum = float(d['lambda_D_theory_sum'])
                lam_th_red = float(d['lambda_D_theory_reduced']) if d['lambda_D_theory_reduced'] != 'nan' else float('nan')
                lam_xir = float(d['lambda_D_from_xir']) if d['lambda_D_from_xir'] != 'nan' else float('nan')
                lam_pk_lor = float(d['lambda_D_from_Pk_lorentz']) if d['lambda_D_from_Pk_lorentz'] != 'nan' else float('nan')
                if not np.isfinite(lam_meas):
                    continue
                rows.append({'a': a, 'z': z,
                             'lambda_D_measured': lam_meas,
                             'lambda_D_theory_sum': lam_th_sum,
                             'lambda_D_theory_reduced': lam_th_red,
                             'lambda_D_from_xir': lam_xir,
                             'lambda_D_from_Pk_lorentz': lam_pk_lor})
            except (ValueError, IndexError, KeyError):
                continue

    if not rows:
        print("[plot] No valid rows yet")
        return

    z_vals = np.array([r['z'] for r in rows])
    a_vals = np.array([r['a'] for r in rows])
    lam_meas = np.array([r['lambda_D_measured'] for r in rows])
    lam_th_sum = np.array([r['lambda_D_theory_sum'] for r in rows])
    lam_th_red = np.array([r['lambda_D_theory_reduced'] for r in rows])
    lam_xir = np.array([r['lambda_D_from_xir'] for r in rows])
    lam_pk_lor = np.array([r['lambda_D_from_Pk_lorentz'] for r in rows])

    # Plot 1: λ_D vs z
    fig, ax = plt.subplots(figsize=(10, 6))
    sort_z = np.argsort(z_vals)
    ax.plot(z_vals[sort_z], lam_meas[sort_z], 'o-', label='λ_D combined', color='tab:blue')
    if np.isfinite(lam_xir).any():
        ax.plot(z_vals[sort_z], lam_xir[sort_z], 'v:', label='λ_D from ξ(r) Yukawa', color='tab:green', alpha=0.7)
    if np.isfinite(lam_pk_lor).any():
        ax.plot(z_vals[sort_z], lam_pk_lor[sort_z], '^:', label='λ_D from P_×(k) Lorentz', color='tab:purple', alpha=0.7)
    ax.plot(z_vals[sort_z], lam_th_sum[sort_z], 's--', label='λ_D theory (sum ρ)', color='tab:orange')
    if np.isfinite(lam_th_red).any():
        ax.plot(z_vals[sort_z], lam_th_red[sort_z], 'D:', label='λ_D theory (reduced ρ)', color='tab:red', alpha=0.7)
    ax.set_xlabel('redshift z')
    ax.set_ylabel('λ_D (Mpc)')
    ax.set_title(f'Janus Debye length vs z (N={len(rows)} snapshots)')
    ax.invert_xaxis()  # z descending = time ascending
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_z_path, dpi=150)
    plt.close(fig)

    # Plot 2: log(λ_D) vs log(a) power law
    fig, ax = plt.subplots(figsize=(10, 6))
    sort_a = np.argsort(a_vals)
    log_a = np.log10(a_vals[sort_a])
    log_lam_meas = np.log10(lam_meas[sort_a])
    log_lam_th_sum = np.log10(lam_th_sum[sort_a])
    ax.plot(log_a, log_lam_meas, 'o-', label='measured', color='tab:blue')
    ax.plot(log_a, log_lam_th_sum, 's--', label='theory (sum ρ)', color='tab:orange')
    if np.isfinite(lam_th_red).any():
        log_lam_th_red = np.log10(lam_th_red[sort_a])
        ax.plot(log_a, log_lam_th_red, 'D:', label='theory (reduced ρ)', color='tab:red')
    # Power law fit on measured
    if len(rows) > 3:
        slope, intercept = np.polyfit(log_a, log_lam_meas, 1)
        ax.plot(log_a, slope*log_a + intercept, 'k:',
                label=f'fit: λ ∝ a^{slope:.2f}')
    ax.set_xlabel('log10(a)')
    ax.set_ylabel('log10(λ_D / Mpc)')
    ax.set_title('λ_D power-law scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_power_path, dpi=150)
    plt.close(fig)

    print(f"[plot] {len(rows)} snapshots → {plot_z_path}, {plot_power_path}")

# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapdir', default=DEFAULT_SNAPDIR)
    ap.add_argument('--out', default=DEFAULT_OUT_CSV)
    ap.add_argument('--plot-z', default=DEFAULT_OUT_PLOT_VS_Z)
    ap.add_argument('--plot-power', default=DEFAULT_OUT_PLOT_POWER)
    ap.add_argument('--plot-rkz', default=DEFAULT_OUT_PLOT_RKZ)
    ap.add_argument('--plot-xir-multiz', default=DEFAULT_OUT_PLOT_XIR_MULTIZ)
    ap.add_argument('--cross-pk-csv', default=DEFAULT_CROSS_PK_CSV)
    ap.add_argument('--xir-cache', default=DEFAULT_XIR_CACHE)
    ap.add_argument('--bootstrap-snapshot', default=None,
                    help='Path to snapshot for bootstrap analysis (z=5 typical)')
    ap.add_argument('--bootstrap-n', type=int, default=50)
    ap.add_argument('--bootstrap-out', default=DEFAULT_OUT_BOOTSTRAP)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--n-grid', type=int, default=N_GRID)
    ap.add_argument('--poll', type=float, default=60.0)
    args = ap.parse_args()

    print(f"=== λ_Debye post-processor ===")
    print(f"  snapdir   : {args.snapdir}")
    print(f"  out       : {args.out}")
    print(f"  plots     : {args.plot_z}, {args.plot_power}, {args.plot_rkz}, {args.plot_xir_multiz}")
    print(f"  n_grid    : {args.n_grid}")
    print(f"  z range   : [{Z_MIN_FIT}, {Z_MAX_FIT}]")
    print(f"  r fit     : [{R_MIN_FIT}, {R_MAX_FIT}] Mpc, {N_R_BINS} bins")
    print()

    # Bootstrap mode (one-shot)
    if args.bootstrap_snapshot:
        print(f"=== BOOTSTRAP MODE ===")
        print(f"  snapshot: {args.bootstrap_snapshot}")
        print(f"  N realiz: {args.bootstrap_n}")
        lambdas, mean, std, ci_low, ci_high = bootstrap_lambda_D(
            args.bootstrap_snapshot, args.n_grid, args.bootstrap_n)
        with open(args.bootstrap_out, 'w') as f:
            f.write(f"# Bootstrap λ_D analysis\n")
            f.write(f"# snapshot: {args.bootstrap_snapshot}\n")
            f.write(f"# n_realizations: {args.bootstrap_n}\n")
            f.write(f"# n_valid_fits: {len(lambdas)}\n")
            f.write(f"# mean(λ_D): {mean:.4f} Mpc\n")
            f.write(f"# std(λ_D):  {std:.4f} Mpc\n")
            f.write(f"# CI 16-84: [{ci_low:.4f}, {ci_high:.4f}] Mpc\n")
            f.write(f"#\n# All values:\n")
            for v in lambdas:
                f.write(f"{v:.4f}\n")
        print(f"  → {args.bootstrap_out}")
        print(f"  λ_D = {mean:.2f} ± {std:.2f} Mpc  (CI: [{ci_low:.2f}, {ci_high:.2f}])")
        return

    seen = already_processed(args.out)
    print(f"  {len(seen)} snapshots already in CSV")

    while True:
        files = sorted([f for f in os.listdir(args.snapdir) if f.endswith('.bin')])
        new_files = [f for f in files if f not in seen]
        any_new_in_band = False

        if new_files:
            for fname in new_files:
                path = os.path.join(args.snapdir, fname)
                size_a = os.path.getsize(path)
                time.sleep(2)
                if os.path.getsize(path) != size_a:
                    print(f"    {fname}: still being written, skip", flush=True)
                    continue
                try:
                    t0 = time.time()
                    row = process_snapshot(path, args.n_grid)
                    dt = time.time() - t0
                    if row.get('skipped'):
                        # Mark as seen so we don't redo, but don't write CSV row
                        seen.add(fname)
                        if row['z'] >= 5:  # log only z>=5 skips (high z, near IC)
                            print(f"    {fname}: SKIP {row['reason']} ({dt:.1f}s)", flush=True)
                        continue
                    append_row(args.out, row)
                    # Cache ξ(r) for multi-z plot (only if at target z)
                    if not row.get('skipped'):
                        r_arr, xi_arr = row['_xir_data'] if '_xir_data' in row else row.get('_xi_r_data', ([], []))
                        if len(r_arr) > 0:
                            update_xir_cache(args.xir_cache, fname, row['z'],
                                             r_arr, xi_arr)
                    seen.add(fname)
                    any_new_in_band = True
                    if np.isfinite(row['lambda_D_measured']):
                        print(f"    {fname}: z={row['z']:.3f}  "
                              f"λ_meas={row['lambda_D_measured']:.2f}  "
                              f"λ_xir={row.get('lambda_D_from_xir', float('nan')):.2f}  "
                              f"λ_PkLor={row.get('lambda_D_from_Pk_lorentz', float('nan')):.2f}  "
                              f"λ_th_sum={row['lambda_D_theory_sum']:.2f}  "
                              f"λ_th_red={row['lambda_D_theory_reduced']:.2f}  "
                              f"r/sum={row['ratio_to_theory_sum']:.3f}  "
                              f"r/red={row['ratio_to_theory_reduced']:.3f}  ({dt:.1f}s)",
                              flush=True)
                    else:
                        print(f"    {fname}: z={row['z']:.3f}  FIT_FAILED  "
                              f"λ_th_sum={row['lambda_D_theory_sum']:.2f}  ({dt:.1f}s)",
                              flush=True)
                except Exception as e:
                    print(f"    {fname}: ERROR {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        # Refresh plots periodically
        if any_new_in_band:
            try:
                make_plots(args.out, args.plot_z, args.plot_power)
            except Exception as e:
                print(f"    [plot] ERROR {e}", flush=True)
            try:
                if make_rkz_map(args.cross_pk_csv, args.plot_rkz):
                    print(f"    [plot] r(k,z) map → {args.plot_rkz}", flush=True)
            except Exception as e:
                print(f"    [plot rkz] ERROR {e}", flush=True)
            try:
                if make_xir_multi_z_plot(args.xir_cache, args.plot_xir_multiz):
                    print(f"    [plot] ξ_+-(r) multi-z → {args.plot_xir_multiz}", flush=True)
            except Exception as e:
                print(f"    [plot xir multiz] ERROR {e}", flush=True)

        if args.once:
            break
        time.sleep(args.poll)

if __name__ == '__main__':
    main()
