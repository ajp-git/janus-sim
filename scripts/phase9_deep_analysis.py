#!/usr/bin/env python3
"""
Phase 9 — Deep Analysis of Janus Simulation
Analyze existing snapshots to determine if Janus physics works at μ=19

Tasks:
1. r(k) vs k at 128³ and 256³
2. r(k) evolution over time at 128³
3. Variance and δ_max metrics
4. Quadrant segregation fractions
5. Radial profile around m- peak
"""

import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def load_snapshot_v3(path):
    """Load v3 snapshot format (408 byte header)."""
    with open(path, 'rb') as f:
        magic = struct.unpack('<Q', f.read(8))[0]
        version = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<I', f.read(4))[0]
        n_total = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t_gyr = struct.unpack('<d', f.read(8))[0]
        l_box = struct.unpack('<d', f.read(8))[0]
        h0 = struct.unpack('<d', f.read(8))[0]
        mu = struct.unpack('<d', f.read(8))[0]
        omega_b = struct.unpack('<d', f.read(8))[0]
        m_plus = struct.unpack('<d', f.read(8))[0]
        m_minus = struct.unpack('<d', f.read(8))[0]
        eps_plus = struct.unpack('<d', f.read(8))[0]
        eps_minus = struct.unpack('<d', f.read(8))[0]
        n_split_max = struct.unpack('<I', f.read(4))[0]
        seed_ic = struct.unpack('<I', f.read(4))[0]
        z_init = struct.unpack('<d', f.read(8))[0]
        n_stars = struct.unpack('<Q', f.read(8))[0]
        z_start_run = struct.unpack('<d', f.read(8))[0]
        sfr = struct.unpack('<d', f.read(8))[0]
        rho_max = struct.unpack('<d', f.read(8))[0]
        run_label = f.read(256).rstrip(b'\x00').decode('utf-8', errors='ignore')

        z = 1.0 / a - 1.0

        particle_dtype = np.dtype([
            ('pos', '<f4', 3), ('vel', '<f4', 3),
            ('mass', '<f4'), ('eps', '<f4'),
            ('sign', 'u1'), ('split_level', 'u1'),
            ('is_star', 'u1'), ('flags', 'u1'),
        ])
        particles = np.frombuffer(f.read(), dtype=particle_dtype)

    return {
        'pos': particles['pos'],
        'vel': particles['vel'],
        'mass': particles['mass'],
        'sign': particles['sign'],
        'n_total': len(particles),
        'z': z,
        'a': a,
        'l_box': l_box,
        't_gyr': t_gyr,
        'mu': mu,
        'h0': h0,
    }


def build_density_grids(snap, grid_size):
    """Build density grids for m+ and m- at given resolution."""
    L = snap['l_box']
    pos = snap['pos']
    sign = snap['sign']
    mass = snap['mass']

    mask_p = sign == 1
    mask_m = sign == 255

    bins = np.linspace(0, L, grid_size + 1)
    rho_p, _ = np.histogramdd(pos[mask_p], bins=[bins]*3, weights=mass[mask_p])
    rho_m, _ = np.histogramdd(pos[mask_m], bins=[bins]*3, weights=mass[mask_m])

    mean_p = rho_p.mean()
    mean_m = rho_m.mean()

    delta_p = (rho_p - mean_p) / mean_p if mean_p > 0 else np.zeros_like(rho_p)
    delta_m = (rho_m - mean_m) / mean_m if mean_m > 0 else np.zeros_like(rho_m)

    return delta_p, delta_m, L


def compute_rk_at_k(delta_p, delta_m, L, k_target, dk=0.01):
    """Compute r(k) at a specific k value with bandwidth dk."""
    from numpy.fft import fftn, fftfreq

    n = delta_p.shape[0]
    fft_p = fftn(delta_p)
    fft_m = fftn(delta_m)

    power_p = np.abs(fft_p)**2
    power_m = np.abs(fft_m)**2
    power_pm = np.real(fft_p * np.conj(fft_m))

    kx = fftfreq(n, d=L/n) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Band around k_target
    k_lo = k_target * (1 - dk)
    k_hi = k_target * (1 + dk)
    mask = (K >= k_lo) & (K < k_hi)

    if mask.sum() == 0:
        # Widen the band
        mask = (K >= k_target * 0.8) & (K < k_target * 1.2)

    if mask.sum() == 0:
        return float('nan')

    p_p = power_p[mask].mean()
    p_m = power_m[mask].mean()
    p_pm = power_pm[mask].mean()
    denom = np.sqrt(p_p * p_m)

    return float(p_pm / denom) if denom > 0 else float('nan')


def task1_rk_vs_k(snap_path, output_dir):
    """Task 1: r(k) vs k at 128³ and 256³."""
    print("\n=== TASK 1: r(k) vs k ===")

    snap = load_snapshot_v3(snap_path)
    k_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    results = {'k': k_values, '128': [], '256': []}

    for grid_size in [128, 256]:
        print(f"  Computing at {grid_size}³...")
        delta_p, delta_m, L = build_density_grids(snap, grid_size)

        for k in k_values:
            rk = compute_rk_at_k(delta_p, delta_m, L, k)
            results[str(grid_size)].append(rk)
            print(f"    k={k:.2f}: r(k)={rk:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(k_values, results['128'], 'o-', label='128³', linewidth=2, markersize=8)
    ax.semilogx(k_values, results['256'], 's--', label='256³', linewidth=2, markersize=8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(1, color='red', linestyle=':', alpha=0.5, label='ΛCDM limit')
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('r(k) = P_+-(k) / sqrt(P_+(k) P_-(k))', fontsize=12)
    ax.set_title(f'Task 1: Cross-correlation r(k) vs k at z={snap["z"]:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase9_rk_vs_k.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'phase9_rk_vs_k.png'}")

    return results


def task2_rk_evolution(snap_dir, output_dir):
    """Task 2: r(k) evolution over time at 128³."""
    print("\n=== TASK 2: r(k) evolution ===")

    snapshots = [
        ('00000', 10.0),
        ('00400', 4.87),
        ('01000', 2.81),
        ('02720', 1.14),
    ]

    k_values = [0.05, 0.1, 0.5, 1.0]
    results = {k: [] for k in k_values}
    z_vals = []

    for snap_id, z_approx in snapshots:
        snap_path = snap_dir / f'snap_{snap_id}.bin'
        if not snap_path.exists():
            print(f"  WARNING: {snap_path} not found, skipping")
            continue

        snap = load_snapshot_v3(snap_path)
        z_vals.append(snap['z'])
        print(f"  z={snap['z']:.2f} (snap_{snap_id})...")

        delta_p, delta_m, L = build_density_grids(snap, 128)

        for k in k_values:
            rk = compute_rk_at_k(delta_p, delta_m, L, k)
            results[k].append(rk)
            print(f"    k={k:.2f}: r(k)={rk:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for i, k in enumerate(k_values):
        ax.plot(z_vals, results[k], f'{markers[i]}-', label=f'k={k}', linewidth=2, markersize=8)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(1, color='red', linestyle=':', alpha=0.5, label='ΛCDM limit')
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('r(k)', fontsize=12)
    ax.set_title('Task 2: Cross-correlation r(k) evolution with time (128³)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # z decreases with time
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase9_rk_evolution.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'phase9_rk_evolution.png'}")

    return {'z': z_vals, 'k_values': k_values, 'rk': results}


def task3_variance_metrics(snap_path):
    """Task 3: Variance and δ_max metrics at 128³."""
    print("\n=== TASK 3: Variance metrics ===")

    snap = load_snapshot_v3(snap_path)
    delta_p, delta_m, L = build_density_grids(snap, 128)

    var_p = float(delta_p.var())
    var_m = float(delta_m.var())
    var_ratio = var_m / var_p if var_p > 0 else float('nan')

    delta_max_p = float(delta_p.max())
    delta_max_m = float(delta_m.max())

    delta_99_p = float(np.percentile(delta_p, 99))
    delta_99_m = float(np.percentile(delta_m, 99))

    results = {
        'var_plus': var_p,
        'var_minus': var_m,
        'var_ratio': var_ratio,
        'delta_max_plus': delta_max_p,
        'delta_max_minus': delta_max_m,
        'delta_99_plus': delta_99_p,
        'delta_99_minus': delta_99_m,
    }

    print(f"  var(δ+)           = {var_p:.4f}")
    print(f"  var(δ-)           = {var_m:.4f}")
    print(f"  var(δ-)/var(δ+)   = {var_ratio:.4f}")
    print(f"  δ_max(m+)         = {delta_max_p:.2f}")
    print(f"  δ_max(m-)         = {delta_max_m:.2f}")
    print(f"  δ_99(m+)          = {delta_99_p:.2f}")
    print(f"  δ_99(m-)          = {delta_99_m:.2f}")

    return results


def task4_quadrant_fractions(snap_path):
    """Task 4: Quadrant segregation fractions at 128³."""
    print("\n=== TASK 4: Quadrant fractions ===")

    snap = load_snapshot_v3(snap_path)
    delta_p, delta_m, L = build_density_grids(snap, 128)

    n_cells = delta_p.size

    # Four quadrants based on signs of δ+ and δ-
    pp = ((delta_p > 0) & (delta_m > 0)).sum() / n_cells  # co-localization
    pm = ((delta_p > 0) & (delta_m < 0)).sum() / n_cells  # m+ in m- void
    mp = ((delta_p < 0) & (delta_m > 0)).sum() / n_cells  # m- in m+ void
    mm = ((delta_p < 0) & (delta_m < 0)).sum() / n_cells  # double under-density

    results = {
        'pp': float(pp),  # δ+ > 0 AND δ- > 0
        'pm': float(pm),  # δ+ > 0 AND δ- < 0
        'mp': float(mp),  # δ+ < 0 AND δ- > 0
        'mm': float(mm),  # δ+ < 0 AND δ- < 0
    }

    # Janus signature: segregation = pm + mp, ΛCDM: co-localization = pp
    segregation = pm + mp
    co_localization = pp + mm

    print(f"  (δ+ > 0, δ- > 0) = {pp:.4f}  [co-localization]")
    print(f"  (δ+ > 0, δ- < 0) = {pm:.4f}  [m+ in m- void]")
    print(f"  (δ+ < 0, δ- > 0) = {mp:.4f}  [m- in m+ void]")
    print(f"  (δ+ < 0, δ- < 0) = {mm:.4f}  [double under-density]")
    print(f"  Segregation (pm+mp) = {segregation:.4f}")
    print(f"  Co-local (pp+mm)    = {co_localization:.4f}")

    results['segregation'] = float(segregation)
    results['co_localization'] = float(co_localization)

    return results


def task5_radial_profile(snap_path, output_dir):
    """Task 5: Radial profile around m- peak at 128³."""
    print("\n=== TASK 5: Radial profile ===")

    snap = load_snapshot_v3(snap_path)
    delta_p, delta_m, L = build_density_grids(snap, 128)

    n = 128
    cell_size = L / n

    # Find peak m- cell
    peak_idx = np.unravel_index(delta_m.argmax(), delta_m.shape)
    peak_pos = np.array(peak_idx) * cell_size + cell_size / 2
    print(f"  m- peak at cell {peak_idx}, position {peak_pos} Mpc")
    print(f"  δ_max(m-) = {delta_m[peak_idx]:.2f}")

    # Compute radial distances for all cells
    ix, iy, iz = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    cell_centers = np.stack([
        ix * cell_size + cell_size / 2,
        iy * cell_size + cell_size / 2,
        iz * cell_size + cell_size / 2,
    ], axis=-1)

    # Distance with periodic boundary
    diff = cell_centers - peak_pos
    diff = np.where(diff > L/2, diff - L, diff)
    diff = np.where(diff < -L/2, diff + L, diff)
    r = np.sqrt((diff**2).sum(axis=-1))

    # Radial bins
    r_bins = [0, 5, 10, 20, 50, 100, 150, 200]
    r_centers = [(r_bins[i] + r_bins[i+1])/2 for i in range(len(r_bins)-1)]

    profile_p = []
    profile_m = []

    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if mask.sum() > 0:
            profile_p.append(float(delta_p[mask].mean()))
            profile_m.append(float(delta_m[mask].mean()))
        else:
            profile_p.append(float('nan'))
            profile_m.append(float('nan'))
        print(f"  r={r_bins[i]}-{r_bins[i+1]} Mpc: δ+={profile_p[-1]:+.3f}, δ-={profile_m[-1]:+.3f}")

    results = {
        'r_bins': r_bins,
        'r_centers': r_centers,
        'delta_plus': profile_p,
        'delta_minus': profile_m,
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_centers, profile_m, 'b-o', label='δ- (m-)', linewidth=2, markersize=8)
    ax.plot(r_centers, profile_p, 'r-s', label='δ+ (m+)', linewidth=2, markersize=8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Distance from m- peak [Mpc]', fontsize=12)
    ax.set_ylabel('Mean δ', fontsize=12)
    ax.set_title(f'Task 5: Radial profile around m- peak at z={snap["z"]:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase9_radial_profile.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'phase9_radial_profile.png'}")

    return results


def write_report(output_dir, task1, task2, task3, task4, task5, snap_z):
    """Write the final Phase 9 report."""

    report = f"""# Phase 9 — Deep Analysis Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Snapshot analyzed**: z = {snap_z:.2f}
**Grid resolution**: 128³ (primary), 256³ (Task 1 comparison)

---

## Task 1 — r(k) vs k at z={snap_z:.2f}

| k [h/Mpc] | r(k) 128³ | r(k) 256³ |
|-----------|-----------|-----------|
"""
    for i, k in enumerate(task1['k']):
        report += f"| {k:.2f} | {task1['128'][i]:+.4f} | {task1['256'][i]:+.4f} |\n"

    # Check if r(k) decreases with k
    rk_128 = task1['128']
    rk_decreases = all(rk_128[i] >= rk_128[i+1] - 0.05 for i in range(len(rk_128)-1) if not np.isnan(rk_128[i]) and not np.isnan(rk_128[i+1]))

    report += f"""
**Criterion 1**: r(k) decreases with k?
- At 128³: {rk_128[0]:+.4f} (k=0.01) → {rk_128[-1]:+.4f} (k=2.0)
- **Verdict**: {'✅ YES' if rk_decreases else '❌ NO'}

---

## Task 2 — r(k) evolution over time (128³)

| z | r(k=0.05) | r(k=0.1) | r(k=0.5) | r(k=1.0) |
|---|-----------|----------|----------|----------|
"""
    for i, z in enumerate(task2['z']):
        row = f"| {z:.2f} |"
        for k in task2['k_values']:
            row += f" {task2['rk'][k][i]:+.4f} |"
        report += row + "\n"

    # Check if r(k) decreases with time at small scales
    rk_k1 = task2['rk'][1.0]
    rk_evolves = len(rk_k1) >= 2 and (rk_k1[0] - rk_k1[-1] > 0.05)

    report += f"""
**Criterion 2**: r(k) decreases with time at small scales?
- r(k=1.0): {rk_k1[0]:+.4f} (z=10) → {rk_k1[-1]:+.4f} (z={task2['z'][-1]:.2f})
- Change: {rk_k1[0] - rk_k1[-1]:+.4f}
- **Verdict**: {'✅ YES' if rk_evolves else '❌ NO'}

---

## Task 3 — Variance metrics (128³)

| Metric | m+ | m- |
|--------|----|----|
| var(δ) | {task3['var_plus']:.4f} | {task3['var_minus']:.4f} |
| δ_max | {task3['delta_max_plus']:.2f} | {task3['delta_max_minus']:.2f} |
| δ_99 | {task3['delta_99_plus']:.2f} | {task3['delta_99_minus']:.2f} |

var(δ-)/var(δ+) = {task3['var_ratio']:.4f}

**Criterion 3**: var(δ-)/var(δ+) > 1?
- **Verdict**: {'✅ YES' if task3['var_ratio'] > 1 else '❌ NO'} (ratio = {task3['var_ratio']:.4f})

---

## Task 4 — Quadrant segregation (128³)

| δ+ \\ δ- | δ- > 0 | δ- < 0 |
|---------|--------|--------|
| δ+ > 0 | {task4['pp']:.4f} | {task4['pm']:.4f} |
| δ+ < 0 | {task4['mp']:.4f} | {task4['mm']:.4f} |

- **Segregation** (pm + mp) = {task4['segregation']:.4f}
- **Co-localization** (pp + mm) = {task4['co_localization']:.4f}

**Criterion 4**: Segregation > Co-localization?
- **Verdict**: {'✅ YES' if task4['segregation'] > task4['co_localization'] else '❌ NO'}

---

## Task 5 — Radial profile around m- peak (128³)

| r [Mpc] | δ+ | δ- |
|---------|----|----|
"""
    for i, r in enumerate(task5['r_centers']):
        report += f"| {task5['r_bins'][i]}-{task5['r_bins'][i+1]} | {task5['delta_plus'][i]:+.4f} | {task5['delta_minus'][i]:+.4f} |\n"

    # Check if m+ is depleted near m- peak
    delta_p_near = task5['delta_plus'][0] if len(task5['delta_plus']) > 0 else 0
    delta_p_far = task5['delta_plus'][-1] if len(task5['delta_plus']) > 0 else 0
    m_plus_depleted = delta_p_near < delta_p_far and delta_p_near < 0

    report += f"""
**Criterion 5**: m+ depleted (δ+ < 0) near m- peak?
- δ+ at r<5 Mpc: {delta_p_near:+.4f}
- δ+ at r>150 Mpc: {delta_p_far:+.4f}
- **Verdict**: {'✅ YES' if m_plus_depleted else '❌ NO'}

---

## VERDICT PHASE 9

| Criterion | Description | Result |
|-----------|-------------|--------|
| 1 | r(k) decreases with k | {'✅' if rk_decreases else '❌'} |
| 2 | r(k) decreases with time | {'✅' if rk_evolves else '❌'} |
| 3 | var(δ-)/var(δ+) > 1 | {'✅' if task3['var_ratio'] > 1 else '❌'} |
| 4 | Segregation > Co-localization | {'✅' if task4['segregation'] > task4['co_localization'] else '❌'} |
| 5 | m+ depleted near m- peak | {'✅' if m_plus_depleted else '❌'} |

"""
    score = sum([rk_decreases, rk_evolves, task3['var_ratio'] > 1,
                 task4['segregation'] > task4['co_localization'], m_plus_depleted])

    report += f"""**Score: {score}/5 criteria satisfied**

### Interpretation

"""
    if score >= 5:
        report += "- 5/5 → Janus physics appears to work correctly at μ=19\n"
    elif score >= 3:
        report += "- 3-4/5 → Signal weak but present, results ambiguous\n"
    else:
        report += "- 0-2/5 → Code does NOT simulate Janus physics, deeper bug likely\n"

    report += """
---

**Note**: This report was generated autonomously. Final GO/NO-GO decision awaits AJP validation.
"""

    with open(output_dir / 'phase9_deep_analysis.md', 'w') as f:
        f.write(report)

    print(f"\nReport saved: {output_dir / 'phase9_deep_analysis.md'}")

    return score


def main():
    parser = argparse.ArgumentParser(description='Phase 9 Deep Analysis')
    parser.add_argument('--snap-dir', required=True, help='Directory with snapshots')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main snapshot for analysis
    snap_path = snap_dir / 'snap_02720.bin'
    if not snap_path.exists():
        print(f"ERROR: {snap_path} not found")
        return

    snap = load_snapshot_v3(snap_path)
    snap_z = snap['z']

    print(f"Phase 9 Deep Analysis")
    print(f"=" * 50)
    print(f"Snapshot: {snap_path}")
    print(f"z = {snap_z:.4f}, L = {snap['l_box']:.0f} Mpc")
    print(f"N = {snap['n_total']:,}, μ = {snap['mu']}")

    # Run all tasks
    task1 = task1_rk_vs_k(snap_path, output_dir)
    task2 = task2_rk_evolution(snap_dir, output_dir)
    task3 = task3_variance_metrics(snap_path)
    task4 = task4_quadrant_fractions(snap_path)
    task5 = task5_radial_profile(snap_path, output_dir)

    # Write report
    score = write_report(output_dir, task1, task2, task3, task4, task5, snap_z)

    print(f"\n{'=' * 50}")
    print(f"PHASE 9 COMPLETE — Score: {score}/5")
    print(f"{'=' * 50}")

    return score


if __name__ == '__main__':
    main()
