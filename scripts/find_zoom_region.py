#!/usr/bin/env python3
"""
Find optimal zoom region center for particle splitting.

Scans a grid of positions to find the interface where:
- N+ ≈ N- (balanced populations)
- N+ + N- is high (dense region, not void)

Score = ratio × (N+ + N-) where ratio = min(N+, N-) / max(N+, N-)
"""

import sys
import struct
import numpy as np
import argparse
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def load_snapshot(path):
    """Load binary snapshot: header u64 N, then N × 28 bytes (7 × f32)."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 28), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3].astype(np.float64)
    signs = np.sign(data[:, 6]).astype(np.int8)
    return pos, signs, n


def count_in_sphere(pos, signs, center, radius):
    """Count N+ and N- particles within sphere of given radius from center."""
    dx = pos[:, 0] - center[0]
    dy = pos[:, 1] - center[1]
    dz = pos[:, 2] - center[2]

    # No periodic wrapping needed for scan within box
    r2 = dx*dx + dy*dy + dz*dz
    mask = r2 <= radius*radius

    signs_in = signs[mask]
    n_plus = np.sum(signs_in > 0)
    n_minus = np.sum(signs_in < 0)

    return n_plus, n_minus


def scan_positions(pos, signs, box_size, radius, grid_size=20):
    """
    Scan a grid of positions to find optimal interface center.

    Returns list of (score, ratio, n_plus, n_minus, center) sorted by score.
    """
    # Grid from -box/2 + radius to +box/2 - radius (stay inside box)
    margin = radius + 5  # Extra margin
    half = box_size / 2

    grid_min = -half + margin
    grid_max = half - margin

    step = (grid_max - grid_min) / (grid_size - 1)

    print(f"Scanning {grid_size}³ = {grid_size**3} positions...")
    print(f"Grid range: [{grid_min:.1f}, {grid_max:.1f}] with step {step:.1f} Mpc")
    print(f"Sphere radius: {radius} Mpc")
    print()

    results = []

    for iz in range(grid_size):
        cz = grid_min + iz * step
        for iy in range(grid_size):
            cy = grid_min + iy * step
            for ix in range(grid_size):
                cx = grid_min + ix * step
                center = (cx, cy, cz)

                n_plus, n_minus = count_in_sphere(pos, signs, center, radius)

                if n_plus == 0 or n_minus == 0:
                    continue  # Skip pure regions

                total = n_plus + n_minus
                ratio = min(n_plus, n_minus) / max(n_plus, n_minus)
                score = ratio * total

                results.append({
                    'score': score,
                    'ratio': ratio,
                    'n_plus': n_plus,
                    'n_minus': n_minus,
                    'total': total,
                    'center': center,
                })

        # Progress
        if (iz + 1) % 5 == 0:
            print(f"  Progress: {iz+1}/{grid_size} slices ({len(results)} valid regions found)")

    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    return results


def main():
    parser = argparse.ArgumentParser(description='Find optimal zoom region center')
    parser.add_argument('snapshot', help='Path to snapshot file')
    parser.add_argument('--box', type=float, default=500.0, help='Box size in Mpc')
    parser.add_argument('--radius', type=float, default=40.0, help='Sphere radius for counting')
    parser.add_argument('--grid', type=int, default=20, help='Grid size (N³ positions)')
    parser.add_argument('--top', type=int, default=10, help='Number of top results to show')
    args = parser.parse_args()

    print("=" * 70)
    print("  FIND ZOOM REGION — Interface Scanner")
    print("=" * 70)
    print(f"Snapshot: {args.snapshot}")
    print(f"Box: {args.box} Mpc, Radius: {args.radius} Mpc, Grid: {args.grid}³")
    print()

    # Load snapshot
    print("Loading snapshot...")
    pos, signs, n = load_snapshot(args.snapshot)
    n_plus_total = np.sum(signs > 0)
    n_minus_total = np.sum(signs < 0)
    print(f"  Loaded {n:,} particles (N+={n_plus_total:,}, N-={n_minus_total:,})")
    print()

    # Scan positions
    results = scan_positions(pos, signs, args.box, args.radius, args.grid)

    print()
    print("=" * 70)
    print(f"  TOP {args.top} INTERFACE REGIONS (by score = ratio × total)")
    print("=" * 70)
    print()
    print(f"{'Rank':<5} {'Score':>10} {'Ratio':>8} {'N+':>10} {'N-':>10} {'Total':>10} {'Center (x, y, z)'}")
    print("-" * 85)

    for i, r in enumerate(results[:args.top], 1):
        cx, cy, cz = r['center']
        print(f"{i:<5} {r['score']:>10.0f} {r['ratio']:>8.3f} {r['n_plus']:>10,} {r['n_minus']:>10,} {r['total']:>10,} ({cx:>7.1f}, {cy:>7.1f}, {cz:>7.1f})")

    print()
    print("=" * 70)
    print("  RECOMMENDED CENTER")
    print("=" * 70)

    if results:
        best = results[0]
        cx, cy, cz = best['center']
        print(f"\nBest interface at: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
        print(f"  N+ = {best['n_plus']:,}")
        print(f"  N- = {best['n_minus']:,}")
        print(f"  Ratio = {best['ratio']:.3f}")
        print(f"  Score = {best['score']:.0f}")
        print()
        print("To run zoom simulation with this center:")
        print(f"  --center-x {cx:.1f} --center-y {cy:.1f} --center-z {cz:.1f}")
    else:
        print("\nNo valid interface regions found!")

    print()


if __name__ == "__main__":
    main()
