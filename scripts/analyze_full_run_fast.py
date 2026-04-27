#!/usr/bin/env python3
"""
analyze_full_run_fast.py - Optimized analysis of Janus 20M production run
Uses NGP (Nearest Grid Point) instead of CIC for speed
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
SNAP_DIR = "/mnt/T2/janus-sim/output/janus_20m_production/snapshots"
OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_20m_production"
COSMO_FILE = "/mnt/T2/janus-sim/output/janus_20m_production/janus_cosmology.csv"

BOX_SIZE = 1000.0  # Mpc
N_CELLS = 32       # Grid for segregation metrics
N_FFT = 128        # Grid for power spectrum
MU = 64
T_INST = 6.33      # Instability timescale in Gyr

# k-band definitions (h/Mpc)
K_LARGE_MAX = 0.05
K_INTER_MIN = 0.05
K_INTER_MAX = 0.2
K_SMALL_MIN = 0.2

def read_snapshot_fast(path):
    """Read snapshot using numpy for speed"""
    with open(path, 'rb') as f:
        header = f.read(16)
        n = struct.unpack('<I', header[0:4])[0]
        box = struct.unpack('<f', header[4:8])[0]
        step = struct.unpack('<I', header[8:12])[0]
        z = struct.unpack('<f', header[12:16])[0]

        # Read all particle data at once
        data = np.frombuffer(f.read(n * 25), dtype=np.uint8).reshape(n, 25)

        # Extract positions (bytes 0-11: 3 floats)
        pos = np.frombuffer(data[:, 0:12].tobytes(), dtype=np.float32).reshape(n, 3)

        # Extract signs (byte 24)
        signs = data[:, 24].astype(np.int8)
        # Convert unsigned to signed
        signs = np.where(signs > 127, signs.astype(np.int16) - 256, signs).astype(np.int8)

    return n, box, step, z, pos, signs

def compute_density_ngp(pos, signs, sign_val, box_size, n_cells):
    """Compute density field using NGP (Nearest Grid Point) - vectorized"""
    mask = signs == sign_val
    pos_sel = pos[mask]

    if len(pos_sel) == 0:
        return np.zeros((n_cells, n_cells, n_cells), dtype=np.float64)

    # Shift to [0, box_size]
    pos_shifted = pos_sel + box_size / 2.0

    # Wrap periodic
    pos_shifted = pos_shifted % box_size

    # Cell indices
    cell_size = box_size / n_cells
    indices = (pos_shifted / cell_size).astype(np.int32)
    indices = np.clip(indices, 0, n_cells - 1)

    # Count using bincount
    flat_idx = indices[:, 0] * n_cells * n_cells + indices[:, 1] * n_cells + indices[:, 2]
    counts = np.bincount(flat_idx, minlength=n_cells**3)
    density = counts.reshape(n_cells, n_cells, n_cells).astype(np.float64)

    return density

def compute_segregation_metrics(pos, signs, box_size, n_cells):
    """Compute Diff/Pois, Corr(δ+,δ-), ExcVar-"""
    rho_plus = compute_density_ngp(pos, signs, +1, box_size, n_cells)
    rho_minus = compute_density_ngp(pos, signs, -1, box_size, n_cells)

    n_plus = rho_plus.sum()
    n_minus = rho_minus.sum()

    if n_plus == 0 or n_minus == 0:
        return 1.0, 0.0, 1.0

    mean_plus = n_plus / (n_cells ** 3)
    mean_minus = n_minus / (n_cells ** 3)

    # Delta fields
    delta_plus = (rho_plus - mean_plus) / mean_plus if mean_plus > 0 else np.zeros_like(rho_plus)
    delta_minus = (rho_minus - mean_minus) / mean_minus if mean_minus > 0 else np.zeros_like(rho_minus)

    # Diff/Pois
    diff = rho_plus - rho_minus
    var_diff = np.var(diff)
    poisson_var = mean_plus + mean_minus
    diff_pois = var_diff / poisson_var if poisson_var > 0 else 1.0

    # Correlation
    dp_flat = delta_plus.flatten()
    dm_flat = delta_minus.flatten()
    if np.std(dp_flat) > 0 and np.std(dm_flat) > 0:
        corr = np.corrcoef(dp_flat, dm_flat)[0, 1]
    else:
        corr = 0.0
    if np.isnan(corr):
        corr = 0.0

    # ExcVar-
    var_minus = np.var(rho_minus)
    exc_var_minus = var_minus / mean_minus if mean_minus > 0 else 1.0

    return diff_pois, corr, exc_var_minus

def compute_pk_ngp(pos, signs, sign_val, box_size, n_grid):
    """Compute power spectrum using NGP - fast vectorized version"""
    mask = signs == sign_val
    pos_sel = pos[mask]
    n_part = len(pos_sel)

    if n_part == 0:
        return np.zeros(n_grid // 2), np.zeros(n_grid // 2)

    # NGP assignment
    density = compute_density_ngp(pos, signs, sign_val, box_size, n_grid)

    # Overdensity
    mean_rho = n_part / (n_grid ** 3)
    if mean_rho > 0:
        delta = (density - mean_rho) / mean_rho
    else:
        delta = np.zeros_like(density)

    # FFT
    delta_k = np.fft.rfftn(delta)
    pk_3d = np.abs(delta_k) ** 2 / n_grid ** 3

    # k-values
    cell_size = box_size / n_grid
    kx = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    kz = np.fft.rfftfreq(n_grid, d=cell_size) * 2 * np.pi

    kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kxx**2 + kyy**2 + kzz**2)

    # Bin
    k_max = kx.max()
    k_bins = np.linspace(0, k_max, n_grid // 2 + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    pk_binned = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if mask.sum() > 0:
            pk_binned[i] = pk_3d[mask].mean()

    return k_centers, pk_binned

def compute_rk_bands(pos, signs, box_size, n_grid):
    """Compute r(k) = P+/P- in 3 k-bands"""
    k, pk_plus = compute_pk_ngp(pos, signs, +1, box_size, n_grid)
    _, pk_minus = compute_pk_ngp(pos, signs, -1, box_size, n_grid)

    with np.errstate(divide='ignore', invalid='ignore'):
        rk = np.where(pk_minus > 0, pk_plus / pk_minus, 1.0)

    large_mask = k < K_LARGE_MAX
    inter_mask = (k >= K_INTER_MIN) & (k < K_INTER_MAX)
    small_mask = k >= K_SMALL_MIN

    r_large = np.nanmean(rk[large_mask]) if large_mask.sum() > 0 else 1.0
    r_inter = np.nanmean(rk[inter_mask]) if inter_mask.sum() > 0 else 1.0
    r_small = np.nanmean(rk[small_mask]) if small_mask.sum() > 0 else 1.0

    r_large = np.clip(r_large if np.isfinite(r_large) else 1.0, 0.01, 100)
    r_inter = np.clip(r_inter if np.isfinite(r_inter) else 1.0, 0.01, 100)
    r_small = np.clip(r_small if np.isfinite(r_small) else 1.0, 0.01, 100)

    return r_large, r_inter, r_small

def analyze_snapshot(snap_path):
    """Analyze a single snapshot"""
    try:
        n, box, step, z, pos, signs = read_snapshot_fast(snap_path)
        diff_pois, corr, exc_var = compute_segregation_metrics(pos, signs, box, N_CELLS)
        r_large, r_inter, r_small = compute_rk_bands(pos, signs, box, N_FFT)

        return {
            'step': step,
            'z': z,
            'diff_pois': diff_pois,
            'corr': corr,
            'exc_var': exc_var,
            'r_large': r_large,
            'r_inter': r_inter,
            'r_small': r_small
        }
    except Exception as e:
        print(f"Error: {snap_path}: {e}", flush=True)
        return None

def load_cosmology_table():
    """Load z -> t_Gyr mapping"""
    data = np.loadtxt(COSMO_FILE, delimiter=',', skiprows=1)
    z_vals = data[:, 5]
    t_vals = data[:, 1]
    return t_vals, z_vals

def z_to_t(z, t_vals, z_vals):
    """Interpolate z to t_Gyr"""
    return np.interp(z, z_vals[::-1], t_vals[::-1])

def main():
    print("=" * 60, flush=True)
    print("  JANUS 20M FULL ANALYSIS (FAST) — 401 snapshots", flush=True)
    print("=" * 60, flush=True)

    snap_files = sorted(Path(SNAP_DIR).glob("snap_*.bin"))
    print(f"Found {len(snap_files)} snapshots", flush=True)

    t_table, z_table = load_cosmology_table()
    print(f"Cosmology: z ∈ [{z_table.min():.2f}, {z_table.max():.2f}]", flush=True)

    results = []
    n_total = len(snap_files)

    for i, snap_path in enumerate(snap_files):
        if i % 10 == 0:
            print(f"[{i+1:3d}/{n_total}] Processing {snap_path.name}...", flush=True)

        result = analyze_snapshot(str(snap_path))
        if result is not None:
            result['t_gyr'] = z_to_t(result['z'], t_table, z_table)
            results.append(result)

    print(f"\nProcessed {len(results)} snapshots", flush=True)

    results.sort(key=lambda x: x['step'])

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "time_series_full.csv")
    with open(csv_path, 'w') as f:
        f.write("step,z,t_Gyr,Diff_Pois,Corr,ExcVar,r_large,r_inter,r_small\n")
        for r in results:
            f.write(f"{r['step']},{r['z']:.4f},{r['t_gyr']:.4f},"
                    f"{r['diff_pois']:.6f},{r['corr']:.6f},{r['exc_var']:.6f},"
                    f"{r['r_large']:.6f},{r['r_inter']:.6f},{r['r_small']:.6f}\n")
    print(f"Saved: {csv_path}", flush=True)

    # Extract arrays
    steps = np.array([r['step'] for r in results])
    z_arr = np.array([r['z'] for r in results])
    t_arr = np.array([r['t_gyr'] for r in results])
    diff_pois = np.array([r['diff_pois'] for r in results])
    corr = np.array([r['corr'] for r in results])
    exc_var = np.array([r['exc_var'] for r in results])
    r_large = np.array([r['r_large'] for r in results])
    r_inter = np.array([r['r_inter'] for r in results])
    r_small = np.array([r['r_small'] for r in results])

    # Theory: e^((t-t0)/t_inst)
    t_start = t_arr[0]
    theory_growth = np.exp((t_arr - t_start) / T_INST)

    # Normalize theory to match initial Diff/Pois
    theory_diff_pois = diff_pois[0] * theory_growth

    print("\nGenerating plots...", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a15')

    for ax in axes.flat:
        ax.set_facecolor('#0a0a15')
        ax.tick_params(colors='#ccccdd')
        ax.xaxis.label.set_color('#ccccdd')
        ax.yaxis.label.set_color('#ccccdd')
        ax.title.set_color('#ccccdd')
        for spine in ax.spines.values():
            spine.set_color('#444466')

    # Plot 1: Diff/Pois(z)
    ax1 = axes[0, 0]
    ax1.plot(z_arr, diff_pois, '-', color='#4db8ff', linewidth=1.5, label='Diff/Pois (simulation)')
    ax1.plot(z_arr, theory_diff_pois, '--', color='#ff9933', linewidth=2, alpha=0.8,
             label=f'Theory: $e^{{(t-t_0)/t_{{inst}}}}$, $t_{{inst}}$={T_INST:.1f} Gyr')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Diff/Pois')
    ax1.set_title('Segregation: Diff/Pois vs z')
    ax1.set_xlim(z_arr.max(), z_arr.min())
    ax1.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccdd')
    ax1.grid(True, alpha=0.3, color='#444466')

    ax1_t = ax1.twiny()
    ax1_t.set_xlim(t_arr[0], t_arr[-1])
    ax1_t.set_xlabel('Cosmic time t [Gyr]', color='#ff9933')
    ax1_t.tick_params(colors='#ff9933')

    # Plot 2: r(k)(z)
    ax2 = axes[0, 1]
    ax2.plot(z_arr, r_large, '-', color='#ff5533', linewidth=1.5, label=f'Large (k<{K_LARGE_MAX})')
    ax2.plot(z_arr, r_inter, '-', color='#33ff55', linewidth=1.5, label=f'Inter ({K_INTER_MIN}<k<{K_INTER_MAX})')
    ax2.plot(z_arr, r_small, '-', color='#5533ff', linewidth=1.5, label=f'Small (k>{K_SMALL_MIN})')
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='r=1 (no bias)')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('r(k) = P₊/P₋')
    ax2.set_title('Power ratio r(k) vs z')
    ax2.set_xlim(z_arr.max(), z_arr.min())
    ax2.set_yscale('log')
    ax2.set_ylim(0.1, 10)
    ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccdd')
    ax2.grid(True, alpha=0.3, color='#444466')

    ax2_t = ax2.twiny()
    ax2_t.set_xlim(t_arr[0], t_arr[-1])
    ax2_t.set_xlabel('Cosmic time t [Gyr]', color='#ff9933')
    ax2_t.tick_params(colors='#ff9933')

    # Plot 3: ExcVar-(z)
    ax3 = axes[1, 0]
    ax3.plot(z_arr, exc_var, '-', color='#ff5533', linewidth=1.5, label='ExcVar⁻')
    ax3.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Poisson (=1)')
    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('ExcVar⁻ = Var(N⁻)/⟨N⁻⟩')
    ax3.set_title('Excess Variance (m⁻) vs z')
    ax3.set_xlim(z_arr.max(), z_arr.min())
    ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccdd')
    ax3.grid(True, alpha=0.3, color='#444466')

    ax3_t = ax3.twiny()
    ax3_t.set_xlim(t_arr[0], t_arr[-1])
    ax3_t.set_xlabel('Cosmic time t [Gyr]', color='#ff9933')
    ax3_t.tick_params(colors='#ff9933')

    # Plot 4: Corr(δ+,δ-)(z)
    ax4 = axes[1, 1]
    ax4.plot(z_arr, corr, '-', color='#aa55ff', linewidth=1.5, label='Corr(δ⁺,δ⁻)')
    ax4.axhline(y=0.0, color='white', linestyle='--', alpha=0.5, label='Uncorrelated')
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Corr(δ⁺, δ⁻)')
    ax4.set_title('Density Correlation vs z')
    ax4.set_xlim(z_arr.max(), z_arr.min())
    ax4.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccdd')
    ax4.grid(True, alpha=0.3, color='#444466')

    ax4_t = ax4.twiny()
    ax4_t.set_xlim(t_arr[0], t_arr[-1])
    ax4_t.set_xlabel('Cosmic time t [Gyr]', color='#ff9933')
    ax4_t.tick_params(colors='#ff9933')

    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "evolution_metrics.png")
    plt.savefig(plot_path, dpi=150, facecolor='#0a0a15', edgecolor='none')
    print(f"Saved: {plot_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  z: {z_arr[0]:.2f} → {z_arr[-1]:.2f}", flush=True)
    print(f"  t: {t_arr[0]:.2f} → {t_arr[-1]:.2f} Gyr (Δt = {t_arr[-1]-t_arr[0]:.1f} Gyr)", flush=True)
    print(f"  Δt/t_inst = {(t_arr[-1]-t_arr[0])/T_INST:.2f}", flush=True)
    print(flush=True)
    print(f"  Diff/Pois: {diff_pois[0]:.3f} → {diff_pois[-1]:.3f} (×{diff_pois[-1]/diff_pois[0]:.2f})", flush=True)
    print(f"  Corr:      {corr[0]:.4f} → {corr[-1]:.4f}", flush=True)
    print(f"  ExcVar⁻:   {exc_var[0]:.3f} → {exc_var[-1]:.3f}", flush=True)
    print(flush=True)
    print(f"  r_large:   {r_large[0]:.3f} → {r_large[-1]:.3f}", flush=True)
    print(f"  r_inter:   {r_inter[0]:.3f} → {r_inter[-1]:.3f}", flush=True)
    print(f"  r_small:   {r_small[0]:.3f} → {r_small[-1]:.3f}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
