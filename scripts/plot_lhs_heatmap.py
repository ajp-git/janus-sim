#!/usr/bin/env python3
"""Generate heatmap visualization of LHS exploration results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Load results
    results_path = Path("/mnt/T2/janus-sim/output/lhs_exploration/results.csv")
    df = pd.read_csv(results_path)

    # Sort by segregation
    df_sorted = df.sort_values('seg_final', ascending=False)

    print("=" * 60)
    print("LHS EXPLORATION RESULTS - TOP 10")
    print("=" * 60)
    print(f"{'Rank':<5} {'Run':<5} {'eta':<7} {'lambda':<8} {'R_sm':<6} {'S_final':<10}")
    print("-" * 60)
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"{i:<5} {int(row['run']):<5} {row['eta']:<7.3f} {row['lambda_base']:<8.1f} {row['r_smooth']:<6.1f} {row['seg_final']:<10.4f}")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: eta vs lambda, color = segregation, size = R_smooth
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        df['eta'], df['lambda_base'],
        c=df['seg_final'], cmap='hot',
        s=(df['r_smooth'] - 2) * 30 + 50,  # Scale size
        alpha=0.8, edgecolors='black', linewidth=0.5
    )
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Segregation S')
    ax1.set_xlabel('eta (density ratio)')
    ax1.set_ylabel('lambda_base (Mpc)')
    ax1.set_title('LHS Exploration: eta vs lambda\n(size = R_smooth)')
    ax1.grid(True, alpha=0.3)

    # Mark top 5
    top5 = df_sorted.head(5)
    for _, row in top5.iterrows():
        ax1.annotate(f"#{int(row['run'])}", (row['eta'], row['lambda_base']),
                    fontsize=8, ha='center', va='bottom')

    # Plot 2: eta vs R_smooth, color = segregation
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        df['eta'], df['r_smooth'],
        c=df['seg_final'], cmap='hot',
        s=(df['lambda_base'] - 10) * 3 + 50,  # Scale size by lambda
        alpha=0.8, edgecolors='black', linewidth=0.5
    )
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Segregation S')
    ax2.set_xlabel('eta (density ratio)')
    ax2.set_ylabel('R_smooth (Mpc)')
    ax2.set_title('LHS Exploration: eta vs R_smooth\n(size = lambda_base)')
    ax2.grid(True, alpha=0.3)

    # Mark top 5
    for _, row in top5.iterrows():
        ax2.annotate(f"#{int(row['run'])}", (row['eta'], row['r_smooth']),
                    fontsize=8, ha='center', va='bottom')

    plt.tight_layout()

    # Save
    output_path = Path("/mnt/T2/janus-sim/output/lhs_exploration/lhs_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("BASIN IDENTIFICATION")
    print("=" * 60)

    # Find optimal region
    best = df_sorted.iloc[0]
    print(f"\nBest run: #{int(best['run'])}")
    print(f"  eta = {best['eta']:.3f}")
    print(f"  lambda = {best['lambda_base']:.1f} Mpc")
    print(f"  R_smooth = {best['r_smooth']:.1f} Mpc")
    print(f"  S_final = {best['seg_final']:.4f}")

    # Analyze correlation
    print("\nCorrelation with S_final:")
    print(f"  eta: {df['eta'].corr(df['seg_final']):.3f}")
    print(f"  lambda: {df['lambda_base'].corr(df['seg_final']):.3f}")
    print(f"  R_smooth: {df['r_smooth'].corr(df['seg_final']):.3f}")

    # Suggest basin for trichotomy
    top3 = df_sorted.head(3)
    eta_center = top3['eta'].mean()
    eta_range = top3['eta'].std() * 2
    lambda_center = top3['lambda_base'].mean()
    lambda_range = top3['lambda_base'].std() * 2
    r_center = top3['r_smooth'].mean()

    print("\nSuggested trichotomy basin:")
    print(f"  eta: [{eta_center - eta_range:.2f}, {eta_center + eta_range:.2f}]")
    print(f"  lambda: [{lambda_center - lambda_range:.1f}, {lambda_center + lambda_range:.1f}] Mpc")
    print(f"  R_smooth: ~{r_center:.1f} Mpc (fix or explore)")

    plt.close()

if __name__ == '__main__':
    main()
