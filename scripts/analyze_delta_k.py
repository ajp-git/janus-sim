#!/usr/bin/env python3
"""
Analyze growth of density perturbations δ(k) at different scales.
Computes δ_k(t)/δ_k(0) for m+ and m- separately.
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def read_snapshot(path):
    """Read snapshot binary file"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        scale_factor = struct.unpack('<d', f.read(8))[0]
        segregation = struct.unpack('<d', f.read(8))[0]

        positions = np.zeros((n, 3), dtype=np.float32)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            x, y, z = struct.unpack('<fff', f.read(12))
            sign = struct.unpack('<b', f.read(1))[0]
            positions[i] = [x, y, z]
            signs[i] = sign

    return positions, signs, step, scale_factor


def compute_delta_k(positions, box_size, n_grid=64):
    """
    Compute density field and FFT to get δ_k.
    Returns: k values and |δ_k|² power spectrum
    """
    # Shift positions to [0, box_size]
    pos_shifted = positions + box_size / 2

    # Grid the density using NGP
    cell_size = box_size / n_grid
    density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    ix = np.clip((pos_shifted[:, 0] / cell_size).astype(int), 0, n_grid - 1)
    iy = np.clip((pos_shifted[:, 1] / cell_size).astype(int), 0, n_grid - 1)
    iz = np.clip((pos_shifted[:, 2] / cell_size).astype(int), 0, n_grid - 1)

    np.add.at(density, (ix, iy, iz), 1)

    # Compute overdensity δ = (ρ - ρ̄) / ρ̄
    mean_density = np.mean(density)
    if mean_density > 0:
        delta = (density - mean_density) / mean_density
    else:
        delta = np.zeros_like(density)

    # FFT
    delta_k = np.fft.fftn(delta)

    # Compute k values
    k_nyquist = np.pi * n_grid / box_size
    kx = np.fft.fftfreq(n_grid, d=box_size/n_grid) * 2 * np.pi
    ky = np.fft.fftfreq(n_grid, d=box_size/n_grid) * 2 * np.pi
    kz = np.fft.fftfreq(n_grid, d=box_size/n_grid) * 2 * np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    return K, delta_k, k_nyquist


def bin_power_spectrum(K, delta_k, k_bins):
    """
    Bin the power spectrum |δ_k|² into k bins.
    Returns mean |δ_k| in each bin.
    """
    power = np.abs(delta_k)**2

    result = []
    for k_low, k_high in zip(k_bins[:-1], k_bins[1:]):
        mask = (K >= k_low) & (K < k_high)
        if np.sum(mask) > 0:
            # Use sqrt of mean power = rms amplitude
            result.append(np.sqrt(np.mean(power[mask])))
        else:
            result.append(0.0)

    return np.array(result)


def analyze_run(snapshot_dir, box_size=400.0, n_grid=64):
    """
    Analyze all snapshots in a run.
    Returns time series of δ_k for different k scales.
    """
    snap_dir = Path(snapshot_dir)
    snapshots = sorted(snap_dir.glob("snap_*.bin"))

    if len(snapshots) == 0:
        print(f"No snapshots found in {snapshot_dir}")
        return None

    print(f"Found {len(snapshots)} snapshots")

    # Define k bins for 3 scales
    # k = 2π/λ, so λ=50 Mpc → k≈0.126, λ=10 Mpc → k≈0.628, λ=2 Mpc → k≈3.14
    k_small = 2 * np.pi / 50.0   # ~0.126 Mpc⁻¹ (large scales)
    k_medium = 2 * np.pi / 10.0  # ~0.628 Mpc⁻¹ (intermediate)
    k_large = 2 * np.pi / 2.0    # ~3.14 Mpc⁻¹ (small scales)

    # Bin edges (centered on target k)
    k_bins = np.array([
        0.05, 0.2,      # k_small bin (λ ~ 30-120 Mpc)
        0.4, 0.9,       # k_medium bin (λ ~ 7-16 Mpc)
        2.0, 5.0,       # k_large bin (λ ~ 1.3-3 Mpc)
    ]).reshape(3, 2)

    k_centers = np.array([k_small, k_medium, k_large])
    k_labels = ['k=0.13 (λ~50 Mpc)', 'k=0.63 (λ~10 Mpc)', 'k=3.1 (λ~2 Mpc)']

    # Storage
    steps = []
    delta_k_plus = {0: [], 1: [], 2: []}   # 3 k bins
    delta_k_minus = {0: [], 1: [], 2: []}

    # Sample snapshots (every 10th to speed up)
    sample_indices = list(range(0, len(snapshots), max(1, len(snapshots) // 30)))
    if len(snapshots) - 1 not in sample_indices:
        sample_indices.append(len(snapshots) - 1)

    print(f"Analyzing {len(sample_indices)} snapshots...")

    for idx in sample_indices:
        snap_path = snapshots[idx]
        pos, signs, step, a = read_snapshot(str(snap_path))

        mask_plus = signs > 0
        mask_minus = signs < 0

        # Compute δ_k for m+
        K, delta_k_p, _ = compute_delta_k(pos[mask_plus], box_size, n_grid)

        # Compute δ_k for m-
        K, delta_k_m, _ = compute_delta_k(pos[mask_minus], box_size, n_grid)

        # Bin into 3 scales
        for i in range(3):
            k_low, k_high = k_bins[i]
            mask_k = (K >= k_low) & (K < k_high)

            if np.sum(mask_k) > 0:
                delta_k_plus[i].append(np.sqrt(np.mean(np.abs(delta_k_p[mask_k])**2)))
                delta_k_minus[i].append(np.sqrt(np.mean(np.abs(delta_k_m[mask_k])**2)))
            else:
                delta_k_plus[i].append(np.nan)
                delta_k_minus[i].append(np.nan)

        steps.append(step)

        if len(steps) % 10 == 0:
            print(f"  Processed step {step}")

    return {
        'steps': np.array(steps),
        'k_labels': k_labels,
        'k_centers': k_centers,
        'delta_k_plus': {k: np.array(v) for k, v in delta_k_plus.items()},
        'delta_k_minus': {k: np.array(v) for k, v in delta_k_minus.items()},
    }


def plot_growth(results, output_path):
    """Plot δ_k(t)/δ_k(0) for all scales"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    steps = results['steps']
    k_labels = results['k_labels']
    colors = ['blue', 'green', 'red']

    # m+ panel
    ax = axes[0]
    ax.set_title('m+ (positive mass)', fontsize=14)

    for i in range(3):
        delta_k = results['delta_k_plus'][i]
        if delta_k[0] > 0:
            growth = delta_k / delta_k[0]
            ax.plot(steps, growth, color=colors[i], linewidth=2, label=k_labels[i])

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('δ_k(t) / δ_k(0)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # m- panel
    ax = axes[1]
    ax.set_title('m- (negative mass)', fontsize=14)

    for i in range(3):
        delta_k = results['delta_k_minus'][i]
        if delta_k[0] > 0:
            growth = delta_k / delta_k[0]
            ax.plot(steps, growth, color=colors[i], linewidth=2, label=k_labels[i])

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('δ_k(t) / δ_k(0)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('Growth of density perturbations δ_k(t)/δ_k(0)\nJanus α=1, Zel\'dovich ICs',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Also plot ratio m+/m-
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Ratio δ_k(m+) / δ_k(m-) — Should be ~1 if α is k-independent', fontsize=14)

    for i in range(3):
        delta_plus = results['delta_k_plus'][i]
        delta_minus = results['delta_k_minus'][i]
        ratio = delta_plus / delta_minus
        ax.plot(steps, ratio, color=colors[i], linewidth=2, label=k_labels[i])

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('δ_k(m+) / δ_k(m-)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0.5, 1.5)

    ratio_path = output_path.replace('.png', '_ratio.png')
    plt.savefig(ratio_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {ratio_path}")
    plt.close()


def print_summary(results):
    """Print summary statistics"""
    steps = results['steps']
    k_labels = results['k_labels']

    print("\n" + "="*60)
    print("SUMMARY: δ_k growth factors")
    print("="*60)

    print("\n--- m+ ---")
    for i in range(3):
        delta_k = results['delta_k_plus'][i]
        if delta_k[0] > 0 and len(delta_k) > 1:
            growth = delta_k[-1] / delta_k[0]
            print(f"  {k_labels[i]}: δ_k(final)/δ_k(0) = {growth:.3f}")

    print("\n--- m- ---")
    for i in range(3):
        delta_k = results['delta_k_minus'][i]
        if delta_k[0] > 0 and len(delta_k) > 1:
            growth = delta_k[-1] / delta_k[0]
            print(f"  {k_labels[i]}: δ_k(final)/δ_k(0) = {growth:.3f}")

    print("\n--- Ratio m+/m- at final step ---")
    for i in range(3):
        delta_plus = results['delta_k_plus'][i]
        delta_minus = results['delta_k_minus'][i]
        if delta_minus[-1] > 0:
            ratio = delta_plus[-1] / delta_minus[-1]
            print(f"  {k_labels[i]}: {ratio:.3f}")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("  - Parallel curves → α is k-independent (pure Janus)")
    print("  - Diverging curves → α depends on k (modified gravity)")
    print("  - Ratio ≈ 1 for all k → m+ and m- trace same structures")
    print("="*60)


if __name__ == '__main__':
    snap_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/T2/janus-sim/output/jour4_corrected_1771892736/snapshots"

    results = analyze_run(snap_dir)

    if results is not None:
        print_summary(results)
        plot_growth(results, "/tmp/delta_k_growth.png")
