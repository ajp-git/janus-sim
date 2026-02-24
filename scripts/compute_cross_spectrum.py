#!/usr/bin/env python3
"""
Compute cross power spectrum P_+-(k) from simulation snapshots.
Measures the correlation between m+ and m- density fields at each scale k.

P_+-(k) / sqrt(P_++(k) * P_--(k)) = r(k)
- r(k) = +1 : perfectly correlated (m+ and m- cluster together)
- r(k) = -1 : perfectly anti-correlated (segregation)
- r(k) = 0  : uncorrelated

For Janus with alpha=1, lambda_- = 0, so initially anti-correlated fields
should evolve toward r(k) = +1 (both driven by lambda_+ mode).
"""

import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt

def read_snapshot(path):
    """Read binary snapshot file"""
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

    return positions, signs, step, scale_factor, segregation


def compute_density_grid(positions, box_size, n_grid=64):
    """Compute density field on grid using NGP"""
    cell_size = box_size / n_grid
    pos_shifted = positions + box_size / 2

    density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    ix = np.clip((pos_shifted[:, 0] / cell_size).astype(int), 0, n_grid - 1)
    iy = np.clip((pos_shifted[:, 1] / cell_size).astype(int), 0, n_grid - 1)
    iz = np.clip((pos_shifted[:, 2] / cell_size).astype(int), 0, n_grid - 1)
    np.add.at(density, (ix, iy, iz), 1)

    mean_density = np.mean(density)
    if mean_density > 0:
        delta = (density - mean_density) / mean_density
    else:
        delta = np.zeros_like(density)
    return delta


def compute_cross_spectrum(delta_plus, delta_minus, box_size):
    """
    Compute cross power spectrum P_+-(k).
    Returns k bins and normalized cross-correlation r(k).
    """
    n_grid = delta_plus.shape[0]

    # FFT
    fft_plus = np.fft.fftn(delta_plus)
    fft_minus = np.fft.fftn(delta_minus)

    # Power spectra
    P_pp = np.abs(fft_plus)**2
    P_mm = np.abs(fft_minus)**2
    P_pm = np.real(fft_plus * np.conj(fft_minus))  # Cross-spectrum (real part)

    # k values in physical units
    dk = 2 * np.pi / box_size
    kx = np.fft.fftfreq(n_grid, d=1.0/n_grid) * dk
    ky = np.fft.fftfreq(n_grid, d=1.0/n_grid) * dk
    kz = np.fft.fftfreq(n_grid, d=1.0/n_grid) * dk
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Bin into k shells
    k_max = dk * n_grid / 2
    n_bins = n_grid // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2

    P_pp_binned = []
    P_mm_binned = []
    P_pm_binned = []

    for i in range(len(k_edges) - 1):
        mask = (K >= k_edges[i]) & (K < k_edges[i+1])
        if np.sum(mask) > 0:
            P_pp_binned.append(np.mean(P_pp[mask]))
            P_mm_binned.append(np.mean(P_mm[mask]))
            P_pm_binned.append(np.mean(P_pm[mask]))
        else:
            P_pp_binned.append(np.nan)
            P_mm_binned.append(np.nan)
            P_pm_binned.append(np.nan)

    P_pp_binned = np.array(P_pp_binned)
    P_mm_binned = np.array(P_mm_binned)
    P_pm_binned = np.array(P_pm_binned)

    # Normalized cross-correlation: r(k) = P_+-(k) / sqrt(P_++(k) * P_--(k))
    denom = np.sqrt(P_pp_binned * P_mm_binned)
    r_k = np.where(denom > 0, P_pm_binned / denom, np.nan)

    return k_centers, r_k, P_pp_binned, P_mm_binned, P_pm_binned


def analyze_run(snap_dir, box_size=400.0, n_grid=64):
    """Analyze multiple snapshots from a run"""
    snap_dir = Path(snap_dir)
    snapshots = sorted(snap_dir.glob("snap_*.bin"))

    if len(snapshots) == 0:
        print(f"No snapshots found in {snap_dir}")
        return None

    print(f"Found {len(snapshots)} snapshots")

    # Sample snapshots (every ~1000 steps)
    sample_steps = [0, 1000, 2000, 3000, 5000, 8000, 10000, 12000]
    results = []

    for target_step in sample_steps:
        # Find closest snapshot
        closest = None
        min_diff = float('inf')
        for snap in snapshots:
            step = int(snap.stem.split('_')[1])
            if abs(step - target_step) < min_diff:
                min_diff = abs(step - target_step)
                closest = snap

        if closest is None or min_diff > 200:
            continue

        actual_step = int(closest.stem.split('_')[1])
        print(f"Processing step {actual_step}...")

        pos, signs, step, a, seg = read_snapshot(str(closest))

        mask_plus = signs > 0
        mask_minus = signs < 0

        delta_plus = compute_density_grid(pos[mask_plus], box_size, n_grid)
        delta_minus = compute_density_grid(pos[mask_minus], box_size, n_grid)

        k, r_k, P_pp, P_mm, P_pm = compute_cross_spectrum(delta_plus, delta_minus, box_size)

        results.append({
            'step': actual_step,
            'a': a,
            'seg': seg,
            'k': k,
            'r_k': r_k,
            'P_pp': P_pp,
            'P_mm': P_mm,
            'P_pm': P_pm,
        })

    return results


def plot_results(results, output_path):
    """Plot cross-correlation evolution"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: r(k) at different times
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, r in enumerate(results):
        valid = ~np.isnan(r['r_k'])
        ax.plot(r['k'][valid], r['r_k'][valid],
                color=colors[i], linewidth=1.5,
                label=f"step {r['step']}")

    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='r=+1 (same)')
    ax.axhline(y=-1, color='blue', linestyle='--', alpha=0.5, label='r=-1 (antisym)')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('k [Mpc⁻¹]', fontsize=12)
    ax.set_ylabel('r(k) = P₊₋(k) / √(P₊₊P₋₋)', fontsize=12)
    ax.set_title('Cross-correlation r(k) evolution', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel 2: Mean r(k) vs step
    ax = axes[0, 1]
    steps = [r['step'] for r in results]
    r_mean = [np.nanmean(r['r_k'][2:20]) for r in results]  # Skip k=0 and high k noise

    ax.plot(steps, r_mean, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('⟨r(k)⟩', fontsize=12)
    ax.set_title('Mean cross-correlation vs time', fontsize=14)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Start: anti-correlated\n(Zel\'dovich ICs)', xy=(steps[0], r_mean[0]),
                xytext=(steps[0]+1500, r_mean[0]-0.3),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')

    ax.annotate('End: correlated\n(λ₋=0 → same structures)', xy=(steps[-1], r_mean[-1]),
                xytext=(steps[-1]-3000, r_mean[-1]+0.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # Panel 3: Power spectra at step 0
    ax = axes[1, 0]
    r0 = results[0]
    valid = ~np.isnan(r0['P_pp'])
    ax.loglog(r0['k'][valid], r0['P_pp'][valid], 'b-', linewidth=2, label='P₊₊(k) [m+]')
    ax.loglog(r0['k'][valid], r0['P_mm'][valid], 'r-', linewidth=2, label='P₋₋(k) [m-]')
    ax.loglog(r0['k'][valid], np.abs(r0['P_pm'][valid]), 'g--', linewidth=2, label='|P₊₋(k)| [cross]')
    ax.set_xlabel('k [Mpc⁻¹]', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title(f'Power spectra at step {r0["step"]}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: Power spectra at final step
    ax = axes[1, 1]
    rf = results[-1]
    valid = ~np.isnan(rf['P_pp'])
    ax.loglog(rf['k'][valid], rf['P_pp'][valid], 'b-', linewidth=2, label='P₊₊(k) [m+]')
    ax.loglog(rf['k'][valid], rf['P_mm'][valid], 'r-', linewidth=2, label='P₋₋(k) [m-]')
    ax.loglog(rf['k'][valid], np.abs(rf['P_pm'][valid]), 'g--', linewidth=2, label='|P₊₋(k)| [cross]')
    ax.set_xlabel('k [Mpc⁻¹]', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title(f'Power spectra at step {rf["step"]}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Cross Power Spectrum P₊₋(k) Analysis\nJanus α=1, λ₋=0 verification',
                 fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(results):
    """Print summary table"""
    print("\n" + "="*70)
    print("CROSS POWER SPECTRUM SUMMARY — P₊₋(k)")
    print("="*70)

    print("\n| Step  |  a(t)  |   S(Mpc)  |  ⟨r(k)⟩  | Interpretation |")
    print("|-------|--------|-----------|----------|----------------|")

    for r in results:
        r_mean = np.nanmean(r['r_k'][2:20])
        if r_mean < -0.7:
            interp = "Anti-correlated"
        elif r_mean > 0.7:
            interp = "Correlated"
        else:
            interp = "Mixed"

        print(f"| {r['step']:5d} | {r['a']:.4f} | {r['seg']:9.4f} | {r_mean:+8.4f} | {interp:14s} |")

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("  - ⟨r(k)⟩ ≈ -1 : m+ and m- are spatially anti-correlated (segregated)")
    print("  - ⟨r(k)⟩ ≈ +1 : m+ and m- trace the same structures (co-located)")
    print("  - Transition from -1 to +1 confirms λ₋=0: antisymmetric mode frozen")
    print("="*70)


if __name__ == '__main__':
    snap_dir = "/mnt/T2/janus-sim/output/jour4_corrected_1771892736/snapshots"

    results = analyze_run(snap_dir)

    if results is not None:
        print_summary(results)
        output_path = "/tmp/cross_spectrum_analysis.png"
        plot_results(results, output_path)

        # Also save to run directory
        run_output = "/mnt/T2/janus-sim/output/jour4_corrected_1771892736/cross_spectrum_analysis.png"
        import shutil
        shutil.copy(output_path, run_output)
        print(f"Also saved to: {run_output}")
