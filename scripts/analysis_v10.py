#!/usr/bin/env python3
"""
JANUS V10 Analysis — High Resolution Domain Size Distribution

Goal: Measure intrinsic Janus segregation scale D_Janus
Method: 256³ grid analysis with multiple smoothing scales

Critical test:
- If median_D ≈ D_min → resolution still insufficient
- If median_D >> D_min → intrinsic scale detected!
"""

import numpy as np
import struct
import glob
import os
import json
from scipy import ndimage
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configuration
BASE_DIR = "/mnt/T2/janus-sim/output/janus_v10_highres"
OUTPUT_DIR = f"{BASE_DIR}/analysis_v10"
GRID_SIZE = 256  # High resolution!
SMOOTHING_SCALES = [0, 1, 2]  # voxels

def load_snapshot(path):
    """Load binary snapshot: (step, N×(x,y,z,sign))"""
    with open(path, 'rb') as f:
        step = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)

    n = len(data) // 4
    x = data[0::4]
    y = data[1::4]
    z = data[2::4]
    signs = data[3::4]

    return step, x, y, z, signs

def compute_polarization_field(x, y, z, signs, grid_size, box_size):
    """Compute polarization field P = (ρ+ - ρ-) / (ρ+ + ρ-)"""
    cell = box_size / grid_size
    half = box_size / 2

    # Grid coordinates
    ix = np.clip(((x + half) / cell).astype(int), 0, grid_size - 1)
    iy = np.clip(((y + half) / cell).astype(int), 0, grid_size - 1)
    iz = np.clip(((z + half) / cell).astype(int), 0, grid_size - 1)

    # Count particles
    rho_plus = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    rho_minus = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    for i in range(len(x)):
        if signs[i] > 0:
            rho_plus[ix[i], iy[i], iz[i]] += 1
        else:
            rho_minus[ix[i], iy[i], iz[i]] += 1

    return rho_plus, rho_minus

def analyze_domains(rho_plus, rho_minus, sigma, voxel_size):
    """
    Analyze polarization domains with specified smoothing.
    Returns domain statistics and diameter distribution.
    """
    # Apply Gaussian smoothing
    if sigma > 0:
        rho_plus_s = ndimage.gaussian_filter(rho_plus, sigma)
        rho_minus_s = ndimage.gaussian_filter(rho_minus, sigma)
    else:
        rho_plus_s = rho_plus.copy()
        rho_minus_s = rho_minus.copy()

    rho_total = rho_plus_s + rho_minus_s

    # Polarization field
    valid_mask = rho_total > 0
    P = np.zeros_like(rho_total)
    P[valid_mask] = (rho_plus_s[valid_mask] - rho_minus_s[valid_mask]) / rho_total[valid_mask]

    # σ_P = std of polarization
    sigma_P = np.std(P[valid_mask]) if np.any(valid_mask) else 0.0

    # Domain analysis - connected components of strongly polarized regions
    strong_plus = P > 0.5
    strong_minus = P < -0.5

    # Label connected components (with periodic boundary handling)
    def label_periodic(mask):
        padded = np.pad(mask, 1, mode='wrap')
        labeled, n_labels = ndimage.label(padded)
        return labeled[1:-1, 1:-1, 1:-1], n_labels

    labeled_plus, n_plus = label_periodic(strong_plus)
    labeled_minus, n_minus = label_periodic(strong_minus)

    # Measure domain sizes (equivalent spherical diameters)
    def get_diameters(labeled, n_labels, voxel_size):
        diams = []
        for i in range(1, n_labels + 1):
            volume = np.sum(labeled == i)
            if volume > 0:
                r = (3 * volume / (4 * np.pi)) ** (1/3)
                diams.append(2 * r * voxel_size)
        return np.array(diams) if diams else np.array([0.0])

    diams_plus = get_diameters(labeled_plus, n_plus, voxel_size)
    diams_minus = get_diameters(labeled_minus, n_minus, voxel_size)
    all_diams = np.concatenate([diams_plus, diams_minus])
    all_diams = all_diams[all_diams > 0]

    if len(all_diams) > 0:
        stats = {
            'sigma_P': float(sigma_P),
            'median': float(np.median(all_diams)),
            'mean': float(np.mean(all_diams)),
            'std': float(np.std(all_diams)),
            'P25': float(np.percentile(all_diams, 25)),
            'P75': float(np.percentile(all_diams, 75)),
            'P10': float(np.percentile(all_diams, 10)),
            'P90': float(np.percentile(all_diams, 90)),
            'min': float(np.min(all_diams)),
            'max': float(np.max(all_diams)),
            'n_domains': len(all_diams),
            'diameters': all_diams.tolist()
        }
    else:
        stats = {
            'sigma_P': float(sigma_P),
            'median': 0.0, 'mean': 0.0, 'std': 0.0,
            'P25': 0.0, 'P75': 0.0, 'P10': 0.0, 'P90': 0.0,
            'min': 0.0, 'max': 0.0, 'n_domains': 0,
            'diameters': []
        }

    return stats

def plot_distribution(diameters, voxel_size, d_min, output_path, title):
    """Plot domain size distribution histogram and CDF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, max(diameters) * 1.1, 50)
    ax.hist(diameters, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(d_min, color='red', linestyle='--', linewidth=2, label=f'D_min = {d_min:.2f} Mpc')
    ax.axvline(np.median(diameters), color='green', linestyle='-', linewidth=2,
               label=f'Median = {np.median(diameters):.2f} Mpc')
    ax.set_xlabel('Domain Diameter (Mpc)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'{title}\nHistogram', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # CDF
    ax = axes[1]
    sorted_d = np.sort(diameters)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, 'b-', linewidth=2)
    ax.axvline(d_min, color='red', linestyle='--', linewidth=2, label=f'D_min = {d_min:.2f} Mpc')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(np.median(diameters), color='green', linestyle='-', linewidth=2)
    ax.set_xlabel('Domain Diameter (Mpc)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(f'{title}\nCumulative Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    print("═" * 70)
    print("  JANUS V10 — HIGH RESOLUTION DOMAIN SIZE ANALYSIS")
    print("═" * 70)
    print()

    # Load parameters
    params_path = f"{BASE_DIR}/params.json"
    if not os.path.exists(params_path):
        print(f"ERROR: {params_path} not found. Run simulation first.")
        return

    with open(params_path) as f:
        params = json.load(f)

    L_box = params['L_box_mpc']
    voxel_size = L_box / GRID_SIZE

    # Theoretical minimum domain diameter (1 voxel volume)
    d_min = voxel_size * 2 * (3 / (4 * np.pi)) ** (1/3)

    print(f"Configuration:")
    print(f"  L_box = {L_box} Mpc")
    print(f"  Grid = {GRID_SIZE}³")
    print(f"  Voxel size = {voxel_size:.4f} Mpc")
    print(f"  D_min (1 voxel) = {d_min:.4f} Mpc")
    print(f"  Smoothing scales: {SMOOTHING_SCALES} voxels")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find snapshots
    snapshots = sorted(glob.glob(f"{BASE_DIR}/snapshots/snap_*.bin"))
    print(f"Found {len(snapshots)} snapshots")
    print()

    if len(snapshots) == 0:
        print("ERROR: No snapshots found. Run simulation first.")
        return

    # Analyze each snapshot
    results = {}

    for snap_path in snapshots:
        snap_name = os.path.basename(snap_path)
        step = int(snap_name.split('_')[1].split('.')[0])
        print(f"Analyzing {snap_name} (step {step})...")

        # Load snapshot
        _, x, y, z, signs = load_snapshot(snap_path)
        print(f"  Loaded {len(x)} particles")

        # Compute density fields
        rho_plus, rho_minus = compute_polarization_field(x, y, z, signs, GRID_SIZE, L_box)

        # Analyze at each smoothing scale
        snap_results = {}

        for sigma in SMOOTHING_SCALES:
            print(f"  Smoothing σ = {sigma} voxels ({sigma * voxel_size:.2f} Mpc)...")
            stats = analyze_domains(rho_plus, rho_minus, sigma, voxel_size)
            snap_results[f'sigma_{sigma}'] = stats

            print(f"    σ_P = {stats['sigma_P']:.4f}")
            print(f"    Median D = {stats['median']:.2f} Mpc ({stats['n_domains']} domains)")
            print(f"    P25-P75 = [{stats['P25']:.2f}, {stats['P75']:.2f}] Mpc")

            # Plot distribution for σ=0 (raw resolution)
            if sigma == 0 and len(stats['diameters']) > 10:
                plot_distribution(
                    np.array(stats['diameters']),
                    voxel_size, d_min,
                    f"{OUTPUT_DIR}/domain_dist_step{step:06d}.png",
                    f"Step {step} (σ=0)"
                )

        results[f'step_{step}'] = snap_results
        print()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY AND INTERPRETATION
    # ═══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print()

    # Use final snapshot for main results
    final_step = max(int(k.split('_')[1]) for k in results.keys())
    final_results = results[f'step_{final_step}']

    print(f"Final snapshot (step {final_step}):")
    print()
    print(f"{'Smoothing':<15} {'σ_P':<10} {'Median D':<12} {'P25-P75':<20} {'N domains':<10}")
    print("-" * 70)

    summary = {
        'analysis_date': datetime.now().isoformat(),
        'L_box_mpc': L_box,
        'grid_size': GRID_SIZE,
        'voxel_size_mpc': voxel_size,
        'd_min_mpc': d_min,
        'smoothing_scales': SMOOTHING_SCALES,
        'final_step': final_step,
        'scales': {}
    }

    for sigma in SMOOTHING_SCALES:
        s = final_results[f'sigma_{sigma}']
        scale_mpc = sigma * voxel_size
        print(f"σ={sigma} ({scale_mpc:.2f} Mpc)  {s['sigma_P']:.4f}    {s['median']:.2f} Mpc      "
              f"[{s['P25']:.2f}, {s['P75']:.2f}]       {s['n_domains']}")

        summary['scales'][f'sigma_{sigma}'] = {
            'smoothing_voxels': sigma,
            'smoothing_mpc': scale_mpc,
            'sigma_P': s['sigma_P'],
            'median_diameter_mpc': s['median'],
            'mean_diameter_mpc': s['mean'],
            'P25_mpc': s['P25'],
            'P75_mpc': s['P75'],
            'n_domains': s['n_domains']
        }

    print()

    # Critical test
    median_d_raw = final_results['sigma_0']['median']
    ratio = median_d_raw / d_min

    print("═" * 70)
    print("  CRITICAL TEST: Is D_median >> D_min?")
    print("═" * 70)
    print()
    print(f"  D_min (resolution limit) = {d_min:.4f} Mpc")
    print(f"  D_median (measured, σ=0) = {median_d_raw:.4f} Mpc")
    print(f"  Ratio = {ratio:.2f}")
    print()

    if ratio < 1.5:
        conclusion = "RESOLUTION_LIMITED"
        interpretation = (
            "D_median ≈ D_min → Still at resolution limit!\n"
            "The intrinsic Janus scale is BELOW our resolution.\n"
            "Need higher resolution (512³ or smaller box)."
        )
    elif ratio < 3.0:
        conclusion = "MARGINAL"
        interpretation = (
            f"D_median is {ratio:.1f}× D_min → Marginally resolved.\n"
            f"Estimated D_Janus ≈ {median_d_raw:.1f} Mpc (uncertain).\n"
            "Higher resolution recommended for confirmation."
        )
    else:
        conclusion = "RESOLVED"
        interpretation = (
            f"D_median is {ratio:.1f}× D_min → Scale is RESOLVED!\n"
            f"D_Janus = {median_d_raw:.1f} ± {(final_results['sigma_0']['P75'] - final_results['sigma_0']['P25'])/2:.1f} Mpc\n"
            "This is the intrinsic Janus segregation scale."
        )

    print(f"CONCLUSION: {conclusion}")
    print()
    for line in interpretation.split('\n'):
        print(f"  {line}")
    print()

    summary['ratio_median_to_dmin'] = ratio
    summary['conclusion'] = conclusion
    summary['D_Janus_estimate_mpc'] = median_d_raw if ratio > 1.5 else None
    summary['all_results'] = results

    # Save summary
    with open(f"{OUTPUT_DIR}/summary_v10.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate report
    report = f"""JANUS V10 — High Resolution Domain Size Analysis
═══════════════════════════════════════════════════════════════════════

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

CONFIGURATION
─────────────
L_box = {L_box} Mpc
Grid = {GRID_SIZE}³
Voxel size = {voxel_size:.4f} Mpc
D_min (1 voxel) = {d_min:.4f} Mpc
N particles = {params.get('N', 'unknown')}

RESULTS (Final step {final_step})
─────────────────────────────────
σ_P = {final_results['sigma_0']['sigma_P']:.4f}

Domain size distribution (σ=0):
  Median = {median_d_raw:.2f} Mpc
  Mean = {final_results['sigma_0']['mean']:.2f} Mpc
  P25-P75 = [{final_results['sigma_0']['P25']:.2f}, {final_results['sigma_0']['P75']:.2f}] Mpc
  Min-Max = [{final_results['sigma_0']['min']:.2f}, {final_results['sigma_0']['max']:.2f}] Mpc
  N domains = {final_results['sigma_0']['n_domains']}

CRITICAL TEST
─────────────
D_min = {d_min:.4f} Mpc
D_median = {median_d_raw:.4f} Mpc
Ratio = {ratio:.2f}

CONCLUSION: {conclusion}
{interpretation}

═══════════════════════════════════════════════════════════════════════
"""

    with open(f"{OUTPUT_DIR}/report_v10.txt", 'w') as f:
        f.write(report)

    print(f"Results saved to: {OUTPUT_DIR}/")
    print()
    print("═" * 70)
    print("  V10 ANALYSIS COMPLETE")
    print("═" * 70)

if __name__ == "__main__":
    main()
