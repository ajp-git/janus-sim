#!/usr/bin/env python3
"""
analyze_full_run.py - Complete analysis of Janus 20M production run
Processes all 401 snapshots and computes segregation metrics + r(k)

Output: time_series_full.csv + 4 evolution plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
import struct
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# k-band definitions (h/Mpc, assuming h=0.7)
K_LARGE_MAX = 0.05    # Large scale: k < 0.05
K_INTER_MIN = 0.05    # Intermediate: 0.05 < k < 0.2
K_INTER_MAX = 0.2
K_SMALL_MIN = 0.2     # Small scale: k > 0.2

def read_snapshot_header(path):
    """Read only header from snapshot"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]
    return n, box, step, z

def read_snapshot(path):
    """Read full snapshot data"""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        box = struct.unpack('<f', f.read(4))[0]
        step = struct.unpack('<I', f.read(4))[0]
        z = struct.unpack('<f', f.read(4))[0]

        pos = np.zeros((n, 3), dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            data = f.read(25)
            pos[i] = struct.unpack('<fff', data[0:12])
            vel[i] = struct.unpack('<fff', data[12:24])
            signs[i] = struct.unpack('<b', data[24:25])[0]

    return n, box, step, z, pos, signs

def compute_density_field(pos, signs, sign_val, box_size, n_cells):
    """Compute density field for particles of given sign"""
    mask = signs == sign_val
    pos_sel = pos[mask]

    # Shift to [0, box_size]
    pos_shifted = pos_sel + box_size / 2.0

    # Compute cell indices
    cell_size = box_size / n_cells
    ix = np.clip((pos_shifted[:, 0] / cell_size).astype(int), 0, n_cells - 1)
    iy = np.clip((pos_shifted[:, 1] / cell_size).astype(int), 0, n_cells - 1)
    iz = np.clip((pos_shifted[:, 2] / cell_size).astype(int), 0, n_cells - 1)

    # Count particles per cell
    density = np.zeros((n_cells, n_cells, n_cells), dtype=np.float64)
    np.add.at(density, (ix, iy, iz), 1)

    return density

def compute_segregation_metrics(pos, signs, box_size, n_cells):
    """Compute Diff/Pois, Corr(δ+,δ-), ExcVar-"""
    # Density fields
    rho_plus = compute_density_field(pos, signs, +1, box_size, n_cells)
    rho_minus = compute_density_field(pos, signs, -1, box_size, n_cells)

    n_plus = rho_plus.sum()
    n_minus = rho_minus.sum()

    if n_plus == 0 or n_minus == 0:
        return 1.0, 0.0, 1.0

    # Mean densities
    mean_plus = n_plus / (n_cells ** 3)
    mean_minus = n_minus / (n_cells ** 3)

    # Delta fields (overdensity)
    delta_plus = (rho_plus - mean_plus) / mean_plus if mean_plus > 0 else np.zeros_like(rho_plus)
    delta_minus = (rho_minus - mean_minus) / mean_minus if mean_minus > 0 else np.zeros_like(rho_minus)

    # Diff/Pois = Var(N+ - N-) / (N+ + N-)
    diff = rho_plus - rho_minus
    var_diff = np.var(diff)
    poisson_var = mean_plus + mean_minus
    diff_pois = var_diff / poisson_var if poisson_var > 0 else 1.0

    # Correlation
    corr = np.corrcoef(delta_plus.flatten(), delta_minus.flatten())[0, 1]
    if np.isnan(corr):
        corr = 0.0

    # ExcVar- = Var(N-) / <N->
    var_minus = np.var(rho_minus)
    exc_var_minus = var_minus / mean_minus if mean_minus > 0 else 1.0

    return diff_pois, corr, exc_var_minus

def compute_power_spectrum(pos, signs, sign_val, box_size, n_grid):
    """Compute 3D power spectrum for given mass sign"""
    mask = signs == sign_val
    pos_sel = pos[mask]
    n_part = len(pos_sel)

    if n_part == 0:
        return np.zeros(n_grid // 2), np.zeros(n_grid // 2)

    # Shift to [0, box_size]
    pos_shifted = pos_sel + box_size / 2.0

    # CIC mass assignment
    cell_size = box_size / n_grid
    density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    for p in range(n_part):
        x, y, z = pos_shifted[p] / cell_size
        ix, iy, iz = int(x), int(y), int(z)
        dx, dy, dz = x - ix, y - iy, z - iz

        # CIC weights
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    wx = 1 - dx if di == 0 else dx
                    wy = 1 - dy if dj == 0 else dy
                    wz = 1 - dz if dk == 0 else dz
                    ii = (ix + di) % n_grid
                    jj = (iy + dj) % n_grid
                    kk = (iz + dk) % n_grid
                    density[ii, jj, kk] += wx * wy * wz

    # Overdensity
    mean_rho = n_part / (n_grid ** 3)
    delta = (density - mean_rho) / mean_rho

    # FFT
    delta_k = np.fft.fftn(delta)
    pk_3d = np.abs(delta_k) ** 2 / n_grid ** 3

    # Spherical average
    kx = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    kz = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kxx**2 + kyy**2 + kzz**2)

    # Bin edges
    k_bins = np.linspace(0, kx.max(), n_grid // 2 + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    pk_binned = np.zeros(len(k_centers))
    counts = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if mask.sum() > 0:
            pk_binned[i] = pk_3d[mask].mean()
            counts[i] = mask.sum()

    return k_centers, pk_binned

def compute_rk_bands(pos, signs, box_size, n_grid):
    """Compute r(k) = P+/P- in 3 k-bands"""
    k, pk_plus = compute_power_spectrum(pos, signs, +1, box_size, n_grid)
    _, pk_minus = compute_power_spectrum(pos, signs, -1, box_size, n_grid)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rk = np.where(pk_minus > 0, pk_plus / pk_minus, 1.0)

    # Band averages
    large_mask = k < K_LARGE_MAX
    inter_mask = (k >= K_INTER_MIN) & (k < K_INTER_MAX)
    small_mask = k >= K_SMALL_MIN

    r_large = np.mean(rk[large_mask]) if large_mask.sum() > 0 else 1.0
    r_inter = np.mean(rk[inter_mask]) if inter_mask.sum() > 0 else 1.0
    r_small = np.mean(rk[small_mask]) if small_mask.sum() > 0 else 1.0

    # Clamp extreme values
    r_large = np.clip(r_large, 0.01, 100)
    r_inter = np.clip(r_inter, 0.01, 100)
    r_small = np.clip(r_small, 0.01, 100)

    return r_large, r_inter, r_small

def analyze_snapshot(snap_path):
    """Analyze a single snapshot and return metrics"""
    try:
        n, box, step, z, pos, signs = read_snapshot(snap_path)

        # Segregation metrics
        diff_pois, corr, exc_var = compute_segregation_metrics(pos, signs, box, N_CELLS)

        # r(k) bands
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
        print(f"Error processing {snap_path}: {e}")
        return None

def load_cosmology_table():
    """Load μ -> t_Gyr mapping from cosmology file"""
    data = np.loadtxt(COSMO_FILE, delimiter=',', skiprows=1)
    # Columns: mu, t_gyr, a_plus, h_plus_gyr, h_plus_km_s_mpc, z
    mu_vals = data[:, 0]
    t_vals = data[:, 1]
    z_vals = data[:, 5]
    return mu_vals, t_vals, z_vals

def z_to_t(z, mu_vals, t_vals, z_vals):
    """Interpolate z to t_Gyr"""
    # Find closest z in table
    idx = np.argmin(np.abs(z_vals - z))
    return t_vals[idx]

def main():
    print("=" * 60)
    print("  JANUS 20M FULL ANALYSIS — 401 snapshots")
    print("=" * 60)

    # Find all snapshots
    snap_files = sorted(Path(SNAP_DIR).glob("snap_*.bin"))
    print(f"Found {len(snap_files)} snapshots")

    # Load cosmology table
    mu_vals, t_vals, z_vals = load_cosmology_table()
    print(f"Loaded cosmology table: z ∈ [{z_vals.min():.2f}, {z_vals.max():.2f}]")

    # Process all snapshots
    results = []
    n_total = len(snap_files)

    for i, snap_path in enumerate(snap_files):
        if i % 20 == 0:
            print(f"Processing {i+1}/{n_total} ({100*i/n_total:.1f}%)...")

        result = analyze_snapshot(str(snap_path))
        if result is not None:
            # Add cosmic time
            result['t_gyr'] = z_to_t(result['z'], mu_vals, t_vals, z_vals)
            results.append(result)

    print(f"Processed {len(results)} snapshots successfully")

    # Sort by step
    results.sort(key=lambda x: x['step'])

    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "time_series_full.csv")
    with open(csv_path, 'w') as f:
        f.write("step,z,t_Gyr,Diff_Pois,Corr,ExcVar,r_large,r_inter,r_small\n")
        for r in results:
            f.write(f"{r['step']},{r['z']:.4f},{r['t_gyr']:.4f},"
                    f"{r['diff_pois']:.6f},{r['corr']:.6f},{r['exc_var']:.6f},"
                    f"{r['r_large']:.6f},{r['r_inter']:.6f},{r['r_small']:.6f}\n")
    print(f"Saved: {csv_path}")

    # Extract arrays for plotting
    steps = np.array([r['step'] for r in results])
    z_arr = np.array([r['z'] for r in results])
    t_arr = np.array([r['t_gyr'] for r in results])
    diff_pois = np.array([r['diff_pois'] for r in results])
    corr = np.array([r['corr'] for r in results])
    exc_var = np.array([r['exc_var'] for r in results])
    r_large = np.array([r['r_large'] for r in results])
    r_inter = np.array([r['r_inter'] for r in results])
    r_small = np.array([r['r_small'] for r in results])

    # Theoretical prediction: e^(t/t_inst)
    t_start = t_arr[0]
    theory_growth = np.exp((t_arr - t_start) / T_INST)

    # Create plots
    print("\nGenerating plots...")

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
    ax1.plot(z_arr, diff_pois, 'o-', color='#4db8ff', markersize=2, linewidth=1, label='Diff/Pois')
    ax1.plot(z_arr, theory_growth, '--', color='#ff9933', linewidth=2, alpha=0.7,
             label=f'$e^{{(t-t_0)/t_{{inst}}}}$, $t_{{inst}}$={T_INST:.1f} Gyr')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Diff/Pois')
    ax1.set_title('Segregation: Diff/Pois vs z')
    ax1.set_xlim(z_arr.max(), z_arr.min())  # Decreasing z
    ax1.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccdd')
    ax1.grid(True, alpha=0.3, color='#444466')

    # Secondary x-axis for t_Gyr
    ax1_t = ax1.twiny()
    ax1_t.set_xlim(t_arr[0], t_arr[-1])
    ax1_t.set_xlabel('Cosmic time t [Gyr]', color='#ff9933')
    ax1_t.tick_params(colors='#ff9933')

    # Plot 2: r(k)(z) for 3 bands
    ax2 = axes[0, 1]
    ax2.plot(z_arr, r_large, 'o-', color='#ff5533', markersize=2, linewidth=1, label=f'Large (k<{K_LARGE_MAX})')
    ax2.plot(z_arr, r_inter, 's-', color='#33ff55', markersize=2, linewidth=1, label=f'Inter ({K_INTER_MIN}<k<{K_INTER_MAX})')
    ax2.plot(z_arr, r_small, '^-', color='#5533ff', markersize=2, linewidth=1, label=f'Small (k>{K_SMALL_MIN})')
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='r=1 (no bias)')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('r(k) = P₊/P₋')
    ax2.set_title('Power ratio r(k) vs z')
    ax2.set_xlim(z_arr.max(), z_arr.min())
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccdd')
    ax2.grid(True, alpha=0.3, color='#444466')

    ax2_t = ax2.twiny()
    ax2_t.set_xlim(t_arr[0], t_arr[-1])
    ax2_t.set_xlabel('Cosmic time t [Gyr]', color='#ff9933')
    ax2_t.tick_params(colors='#ff9933')

    # Plot 3: ExcVar-(z)
    ax3 = axes[1, 0]
    ax3.plot(z_arr, exc_var, 'o-', color='#ff5533', markersize=2, linewidth=1, label='ExcVar⁻')
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
    ax4.plot(z_arr, corr, 'o-', color='#aa55ff', markersize=2, linewidth=1, label='Corr(δ⁺,δ⁻)')
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
    print(f"Saved: {plot_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  z range: {z_arr.max():.2f} → {z_arr.min():.2f}")
    print(f"  t range: {t_arr.min():.2f} → {t_arr.max():.2f} Gyr")
    print(f"  Δt = {t_arr.max() - t_arr.min():.2f} Gyr = {(t_arr.max() - t_arr.min())/T_INST:.2f} t_inst")
    print()
    print(f"  Diff/Pois: {diff_pois[0]:.3f} → {diff_pois[-1]:.3f} (growth: {diff_pois[-1]/diff_pois[0]:.2f}×)")
    print(f"  Corr:      {corr[0]:.4f} → {corr[-1]:.4f}")
    print(f"  ExcVar⁻:   {exc_var[0]:.3f} → {exc_var[-1]:.3f}")
    print()
    print(f"  r_large:   {r_large[0]:.3f} → {r_large[-1]:.3f}")
    print(f"  r_inter:   {r_inter[0]:.3f} → {r_inter[-1]:.3f}")
    print(f"  r_small:   {r_small[0]:.3f} → {r_small[-1]:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
