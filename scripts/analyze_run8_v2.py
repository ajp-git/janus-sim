#!/usr/bin/env python3
"""
Generate 4 analysis figures for Run 8 - CORRECTED VERSION
Coordinates are centered: [-75, 75] not [0, 150]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct

# Constants
RUN_DIR = Path("/mnt/T2/janus-sim/output/grid_10/run_08")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/grid_10/run_08/figures")
BOX_SIZE = 150.0  # Mpc
BOX_HALF = BOX_SIZE / 2  # Coordinates are [-75, 75]

def load_snapshot(path):
    """Load binary snapshot: header (u32 n) + n*(3*f32 + i8)."""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        pos = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            x, y, z = struct.unpack('<fff', f.read(12))
            sign = struct.unpack('<b', f.read(1))[0]
            pos[i] = [x, y, z]
            signs[i] = sign
    return pos, signs

def figure1_density_map(pos, signs, output_path):
    """Projected density map with 20 Mpc slice, colored by mass sign."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Select 20 Mpc slice in z (centered coordinates)
    slice_width = 20.0
    mask = np.abs(pos[:, 2]) < slice_width / 2

    pos_slice = pos[mask]
    signs_slice = signs[mask]

    pos_plus = pos_slice[signs_slice > 0]
    pos_minus = pos_slice[signs_slice < 0]

    print(f"  Slice: {len(pos_slice)} particles ({len(pos_plus)} m+, {len(pos_minus)} m-)")

    # Plot 1: All particles scatter
    ax = axes[0]
    ax.scatter(pos_minus[:, 0], pos_minus[:, 1], s=0.1, c='blue', alpha=0.3, label='m-')
    ax.scatter(pos_plus[:, 0], pos_plus[:, 1], s=0.1, c='red', alpha=0.3, label='m+')
    ax.set_xlim(-BOX_HALF, BOX_HALF)
    ax.set_ylim(-BOX_HALF, BOX_HALF)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title(f'All particles (|z| < {slice_width/2:.0f} Mpc)')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Plot 2: 2D histogram m+
    ax = axes[1]
    h_plus, xedges, yedges = np.histogram2d(pos_plus[:, 0], pos_plus[:, 1],
                                             bins=128, range=[[-BOX_HALF, BOX_HALF], [-BOX_HALF, BOX_HALF]])
    im = ax.imshow(h_plus.T, origin='lower', extent=[-BOX_HALF, BOX_HALF, -BOX_HALF, BOX_HALF],
                   cmap='Reds', vmin=0)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('m+ density')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='count')

    # Plot 3: 2D histogram m-
    ax = axes[2]
    h_minus, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 1],
                                    bins=128, range=[[-BOX_HALF, BOX_HALF], [-BOX_HALF, BOX_HALF]])
    im = ax.imshow(h_minus.T, origin='lower', extent=[-BOX_HALF, BOX_HALF, -BOX_HALF, BOX_HALF],
                   cmap='Blues', vmin=0)
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')
    ax.set_title('m- density')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='count')

    plt.suptitle(f'Run 8: Density projection at z=0 (η=0.88, R=8.0)\nm+ x_mean={pos[signs>0,0].mean():.1f}, m- x_mean={pos[signs<0,0].mean():.1f} Mpc', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def figure2_segregation_curve(output_path):
    """S(z) curve from time_series.csv."""
    import pandas as pd

    ts_path = RUN_DIR / "time_series.csv"
    df = pd.read_csv(ts_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['z'], df['segregation'], 'b-', linewidth=2)
    ax.axhline(y=df['segregation'].max(), color='r', linestyle='--',
               label=f'S_max = {df["segregation"].max():.3f}')
    ax.axhline(y=df['segregation'].iloc[-1], color='g', linestyle='--',
               label=f'S_final = {df["segregation"].iloc[-1]:.3f}')

    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Segregation S', fontsize=12)
    ax.set_title('Run 8: Segregation evolution (η=0.88, R=8.0, λ=40)', fontsize=14)
    ax.set_xlim(df['z'].max(), 0)
    ax.set_ylim(0, df['segregation'].max() * 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(1/(1+df['z'].max()), 1)
    ax2.set_xlabel('Scale factor a', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def compute_power_spectrum(pos, signs, n_grid=256):
    """Compute power spectrum P(k) using FFT - CORRECTED."""
    # Shift coordinates to [0, L] for FFT
    pos_shifted = pos + BOX_HALF

    grid_plus = np.zeros((n_grid, n_grid, n_grid))
    grid_minus = np.zeros((n_grid, n_grid, n_grid))

    cell_size = BOX_SIZE / n_grid

    for i in range(len(pos)):
        ix = int(pos_shifted[i, 0] / cell_size) % n_grid
        iy = int(pos_shifted[i, 1] / cell_size) % n_grid
        iz = int(pos_shifted[i, 2] / cell_size) % n_grid

        if signs[i] > 0:
            grid_plus[ix, iy, iz] += 1
        else:
            grid_minus[ix, iy, iz] += 1

    # Convert to overdensity
    mean_plus = grid_plus.mean()
    mean_minus = grid_minus.mean()

    delta_plus = (grid_plus - mean_plus) / mean_plus if mean_plus > 0 else grid_plus
    delta_minus = (grid_minus - mean_minus) / mean_minus if mean_minus > 0 else grid_minus

    # FFT
    fft_plus = np.fft.fftn(delta_plus)
    fft_minus = np.fft.fftn(delta_minus)

    # Power spectra (auto)
    pk_plus = np.abs(fft_plus)**2
    pk_minus = np.abs(fft_minus)**2
    pk_total = np.abs(fft_plus + fft_minus)**2 / 4  # Total matter

    # Cross power spectrum - CORRECT FORMULA
    # P_cross = Re(δ_+ * δ_-*)
    pk_cross = np.real(fft_plus * np.conj(fft_minus))

    # k-space grid
    kx = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    kz = np.fft.fftfreq(n_grid, d=cell_size) * 2 * np.pi
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k3d = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Radial binning - extend to k=10 h/Mpc
    k_max = min(kx.max(), 10.0)
    k_bins = np.linspace(0.01, k_max, 50)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    pk_plus_binned = np.zeros(len(k_centers))
    pk_minus_binned = np.zeros(len(k_centers))
    pk_total_binned = np.zeros(len(k_centers))
    pk_cross_binned = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask = (k3d >= k_bins[i]) & (k3d < k_bins[i+1])
        if mask.sum() > 0:
            pk_plus_binned[i] = pk_plus[mask].mean()
            pk_minus_binned[i] = pk_minus[mask].mean()
            pk_total_binned[i] = pk_total[mask].mean()
            pk_cross_binned[i] = pk_cross[mask].mean()

    # Normalize
    vol = BOX_SIZE**3
    norm = vol / (n_grid**3)**2
    pk_plus_binned *= norm
    pk_minus_binned *= norm
    pk_total_binned *= norm
    pk_cross_binned *= norm

    return k_centers, pk_plus_binned, pk_minus_binned, pk_total_binned, pk_cross_binned

def figure3_power_spectrum(pos, signs, output_path):
    """P(k) at z=0 vs ΛCDM approximation."""
    print("  Computing power spectra (n_grid=256)...")
    k, pk_plus, pk_minus, pk_total, pk_cross = compute_power_spectrum(pos, signs, n_grid=256)

    # Simple ΛCDM approximation
    k_eq = 0.01
    n_s = 0.96
    valid_idx = np.where((k > 0.03) & (pk_total > 0))[0]
    if len(valid_idx) > 0:
        ref_idx = valid_idx[min(5, len(valid_idx)-1)]
        A_s = pk_total[ref_idx] / (k[ref_idx]**n_s / (1 + (k[ref_idx]/k_eq)**2)**2)
    else:
        A_s = 1e4
    pk_lcdm = A_s * k**n_s / (1 + (k/k_eq)**2)**2

    fig, ax = plt.subplots(figsize=(10, 7))

    valid = (k > 0.03) & (pk_total > 0) & (pk_plus > 0) & (pk_minus > 0)

    ax.loglog(k[valid], pk_total[valid], 'k-', linewidth=2, label='P_total(k)')
    ax.loglog(k[valid], pk_plus[valid], 'r--', linewidth=1.5, label='P_+(k)')
    ax.loglog(k[valid], pk_minus[valid], 'b--', linewidth=1.5, label='P_-(k)')
    ax.loglog(k[valid], pk_lcdm[valid], 'g:', linewidth=2, label='ΛCDM approx')

    ax.set_xlabel('k (h/Mpc)', fontsize=12)
    ax.set_ylabel('P(k) (Mpc/h)³', fontsize=12)
    ax.set_title('Run 8: Power spectrum at z=0 (η=0.88, R=8.0)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.03, 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return k, pk_plus, pk_minus, pk_cross

def figure4_cross_correlation(k, pk_plus, pk_minus, pk_cross, output_path):
    """r(k) = P_cross(k) / sqrt(P_+(k) * P_-(k))"""

    valid = (pk_plus > 0) & (pk_minus > 0) & (k > 0.03)
    r_k = np.zeros_like(k)
    r_k[valid] = pk_cross[valid] / np.sqrt(pk_plus[valid] * pk_minus[valid])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(k[valid], r_k[valid], 'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='r=1 (correlation)')
    ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='r=-1 (anti-correlation)')

    ax.set_xlabel('k (h/Mpc)', fontsize=12)
    ax.set_ylabel('r(k) = P_cross / √(P_+ · P_-)', fontsize=12)
    ax.set_title('Run 8: Cross-correlation m+ / m- at z=0', fontsize=14)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0.03, 10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    mean_r = r_k[valid].mean()
    ax.text(0.95, 0.95, f'<r(k)> = {mean_r:.3f}', transform=ax.transAxes,
            fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Interpretation
    if mean_r > 0.5:
        interp = "m+ et m- tracent la même structure"
    elif mean_r < -0.5:
        interp = "m+ et m- anti-corrélés (ségrégation spectrale)"
    else:
        interp = "faible corrélation"
    ax.text(0.05, 0.05, interp, transform=ax.transAxes,
            fontsize=10, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANALYSIS v2: Run 8 (η=0.88, R=8.0, λ=40)")
    print("=" * 60)

    snap_path = RUN_DIR / "snapshots" / "snap_001200.bin"
    print(f"\nLoading snapshot: {snap_path}")
    pos, signs = load_snapshot(snap_path)
    print(f"  N = {len(signs)} (N+ = {(signs > 0).sum()}, N- = {(signs < 0).sum()})")
    print(f"  m+ x_mean = {pos[signs>0, 0].mean():.2f} Mpc")
    print(f"  m- x_mean = {pos[signs<0, 0].mean():.2f} Mpc")
    print(f"  Δx = {pos[signs>0, 0].mean() - pos[signs<0, 0].mean():.2f} Mpc")

    print("\n[1/4] Density map...")
    figure1_density_map(pos, signs, OUTPUT_DIR / "fig1_density_map_v2.png")

    print("\n[2/4] S(z) curve...")
    figure2_segregation_curve(OUTPUT_DIR / "fig2_segregation_v2.png")

    print("\n[3/4] Power spectrum...")
    k, pk_plus, pk_minus, pk_cross = figure3_power_spectrum(pos, signs, OUTPUT_DIR / "fig3_power_spectrum_v2.png")

    print("\n[4/4] Cross-correlation (k up to 10 h/Mpc)...")
    figure4_cross_correlation(k, pk_plus, pk_minus, pk_cross, OUTPUT_DIR / "fig4_cross_correlation_v2.png")

    print("\n" + "=" * 60)
    print("ALL 4 FIGURES GENERATED (v2)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
