#!/usr/bin/env python3
"""
Validate ICs for grid contamination via angular spectrum analysis.
Phase 10 fix validation script.
"""
import struct
import sys
import numpy as np
from numpy.fft import fft2, fftshift
import argparse

def load_snapshot_v3(path):
    """Load v3 format snapshot."""
    with open(path, 'rb') as f:
        magic = f.read(8)
        version = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<I', f.read(4))[0]
        n_total = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t_gyr = struct.unpack('<d', f.read(8))[0]
        l_box = struct.unpack('<d', f.read(8))[0]
        f.seek(408)  # skip to particles

        particle_dtype = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('eps', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(), dtype=particle_dtype)

    z = 1.0 / a - 1.0
    pos = particles['pos']
    signs = np.where(particles['sign'] == 1, 1, -1).astype(np.int32)
    return pos, signs, z, l_box, n_total

def angular_spectrum(image, n_angles=72):
    """Compute angular spectrum from 2D FFT."""
    fft = np.abs(fftshift(fft2(image - image.mean())))
    n = image.shape[0]
    cy, cx = n // 2, n // 2
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    signal = np.zeros(n_angles)

    for i, theta in enumerate(angles):
        for r in range(5, n // 2 - 2):
            y = int(cy + r * np.sin(theta))
            x = int(cx + r * np.cos(theta))
            if 0 <= x < n and 0 <= y < n:
                signal[i] += fft[y, x]

    return angles, signal / signal.mean()

def main():
    parser = argparse.ArgumentParser(description='Validate ICs for grid contamination')
    parser.add_argument('--snap', required=True, help='Path to snapshot file')
    parser.add_argument('--nbin', type=int, default=64, help='Number of histogram bins')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PHASE 10 — IC VALIDATION (Angular Spectrum Analysis)")
    print(f"{'='*60}\n")

    pos, signs, z, L, n_total = load_snapshot_v3(args.snap)
    half = L / 2

    print(f"Snapshot: {args.snap}")
    print(f"  z = {z:.3f}, L = {L:.0f} Mpc, N = {n_total:,}")

    # Slab |z| < 5% of L
    mask_z = np.abs(pos[:, 2]) < L * 0.05
    pos_slab = pos[mask_z]
    signs_slab = signs[mask_z]
    print(f"  Slab |z| < {L*0.05:.1f} Mpc: {len(pos_slab):,} particles")

    # Separate populations
    mp = pos_slab[signs_slab > 0]
    mm = pos_slab[signs_slab < 0]
    print(f"  m+ in slab: {len(mp):,}")
    print(f"  m- in slab: {len(mm):,}")

    # 2D histograms
    nbin = args.nbin
    bins = np.linspace(-half, half, nbin + 1)
    h_plus, _, _ = np.histogram2d(mp[:, 0], mp[:, 1], bins=bins)
    h_minus, _, _ = np.histogram2d(mm[:, 0], mm[:, 1], bins=bins)
    h_total = h_plus + h_minus

    # Angular spectrum analysis
    print(f"\n{'='*60}")
    print("ANGULAR SPECTRUM RESULTS")
    print(f"{'='*60}")
    print(f"{'Population':<10} {'max_axes':>12} {'max_overall':>12} {'Status':>12}")
    print("-" * 50)

    results = {}
    for name, h in [('m+', h_plus), ('m-', h_minus), ('total', h_total)]:
        angles, spec = angular_spectrum(h)

        # Check axes (0, 90, 180, 270 degrees)
        axis_angles = [0, 90, 180, 270]
        axis_values = [spec[np.argmin(np.abs(angles - np.radians(d)))] for d in axis_angles]
        max_axes = max(axis_values)
        max_overall = spec.max()

        # Criteria: max_axes < 1.15 = PASS
        status = "PASS" if max_axes < 1.15 else ("WARN" if max_axes < 1.25 else "FAIL")

        print(f"{name:<10} {max_axes:>12.3f} {max_overall:>12.3f} {status:>12}")
        results[name] = {'max_axes': max_axes, 'max_overall': max_overall, 'status': status}

    # Summary
    print(f"\n{'='*60}")
    all_pass = all(r['max_axes'] < 1.15 for r in results.values())
    if all_pass:
        print("ALL POPULATIONS PASS — ICs are clean")
        print("   Ready for production!")
    else:
        print("SOME POPULATIONS FAIL — Grid contamination detected")
        print("   Further investigation needed")
    print(f"{'='*60}\n")

    # Also compute delta_rms for normalization check
    print("delta_rms estimation:")
    delta_plus = (h_plus - h_plus.mean()) / h_plus.mean()
    delta_minus = (h_minus - h_minus.mean()) / h_minus.mean()
    delta_total = (h_total - h_total.mean()) / h_total.mean()

    print(f"  delta_rms(m+) = {delta_plus.std():.4f}")
    print(f"  delta_rms(m-) = {delta_minus.std():.4f}")
    print(f"  delta_rms(total) = {delta_total.std():.4f}")

    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
