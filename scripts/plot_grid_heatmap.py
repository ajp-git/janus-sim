#!/usr/bin/env python3
"""Generate heatmap visualization of grid exploration results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Load results
    results_path = Path("/mnt/T2/janus-sim/output/grid_10/results.csv")
    df = pd.read_csv(results_path)

    # Remove duplicate rows (if any)
    df = df.drop_duplicates(subset=['run'])

    print("=" * 70)
    print("GRID EXPLORATION RESULTS (η × R_smooth)")
    print("=" * 70)
    print(f"{'Run':>4} {'η':>6} {'R':>5} {'S_final':>10} {'S_peak':>10} {'z_peak':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{int(row['run']):4d} {row['eta']:6.2f} {row['r_smooth']:5.1f} "
              f"{row['seg_final']:10.4f} {row['seg_peak']:10.4f} {row['z_peak']:8.4f}")

    # Reshape for heatmap
    etas = sorted(df['eta'].unique())
    rsmooths = sorted(df['r_smooth'].unique())

    # Create 2D grids
    seg_final_grid = np.zeros((len(etas), len(rsmooths)))
    seg_peak_grid = np.zeros((len(etas), len(rsmooths)))

    for _, row in df.iterrows():
        i = etas.index(row['eta'])
        j = rsmooths.index(row['r_smooth'])
        seg_final_grid[i, j] = row['seg_final']
        seg_peak_grid[i, j] = row['seg_peak']

    # Create figure with 2 heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap 1: S_final
    ax1 = axes[0]
    im1 = ax1.imshow(seg_final_grid, cmap='hot', aspect='auto', origin='lower',
                     extent=[rsmooths[0]-0.25, rsmooths[-1]+0.25, etas[0]-0.02, etas[-1]+0.02])
    ax1.set_xlabel('R_smooth (Mpc)', fontsize=12)
    ax1.set_ylabel('η (density ratio)', fontsize=12)
    ax1.set_title('S_final (z=0)', fontsize=14)
    ax1.set_xticks(rsmooths)
    ax1.set_yticks(etas)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Segregation')

    # Add text annotations
    for i, eta in enumerate(etas):
        for j, r in enumerate(rsmooths):
            val = seg_final_grid[i, j]
            color = 'white' if val > 0.15 else 'black'
            ax1.text(r, eta, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    # Heatmap 2: S_peak
    ax2 = axes[1]
    im2 = ax2.imshow(seg_peak_grid, cmap='hot', aspect='auto', origin='lower',
                     extent=[rsmooths[0]-0.25, rsmooths[-1]+0.25, etas[0]-0.02, etas[-1]+0.02])
    ax2.set_xlabel('R_smooth (Mpc)', fontsize=12)
    ax2.set_ylabel('η (density ratio)', fontsize=12)
    ax2.set_title('S_peak (max during evolution)', fontsize=14)
    ax2.set_xticks(rsmooths)
    ax2.set_yticks(etas)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Segregation')

    # Add text annotations
    for i, eta in enumerate(etas):
        for j, r in enumerate(rsmooths):
            val = seg_peak_grid[i, j]
            color = 'white' if val > 0.25 else 'black'
            ax2.text(r, eta, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path("/mnt/T2/janus-sim/output/grid_10/grid_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_final = df.loc[df['seg_final'].idxmax()]
    best_peak = df.loc[df['seg_peak'].idxmax()]

    print(f"\nBest S_final: Run {int(best_final['run'])}")
    print(f"  η = {best_final['eta']:.2f}, R = {best_final['r_smooth']:.1f}")
    print(f"  S_final = {best_final['seg_final']:.4f}")

    print(f"\nBest S_peak: Run {int(best_peak['run'])}")
    print(f"  η = {best_peak['eta']:.2f}, R = {best_peak['r_smooth']:.1f}")
    print(f"  S_peak = {best_peak['seg_peak']:.4f} at z = {best_peak['z_peak']:.2f}")

    # Check stability (S_final / S_peak)
    df['stability'] = df['seg_final'] / df['seg_peak']
    print("\nStability (S_final / S_peak):")
    for _, row in df.iterrows():
        flag = "✓" if row['stability'] > 0.9 else "⚠" if row['stability'] > 0.5 else "✗"
        print(f"  Run {int(row['run']):2d}: {row['stability']:.2f} {flag}")

    plt.close()

if __name__ == '__main__':
    main()
