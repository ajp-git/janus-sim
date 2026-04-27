#!/usr/bin/env python3
"""
Full snapshot analysis: P(k), r(k), FOF halos, density profile
For Janus 10M production run
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from scipy.spatial import cKDTree
import sys

# Output directory
OUT_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/analysis")
OUT_DIR.mkdir(exist_ok=True)

def read_snapshot(path):
    """Read binary snapshot: header (N,a,t) + pos(f32) + vel(f32) + signs(i8)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        a = struct.unpack('<d', f.read(8))[0]
        t = struct.unpack('<d', f.read(8))[0]
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3).astype(np.float64)
        vel = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3).astype(np.float64)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    z = 1.0/a - 1.0 if a > 0 else 0

    # Split by sign
    mask_plus = signs > 0
    mask_minus = signs < 0

    return {
        'pos_plus': pos[mask_plus],
        'pos_minus': pos[mask_minus],
        'vel_plus': vel[mask_plus],
        'vel_minus': vel[mask_minus],
        'n_plus': np.sum(mask_plus),
        'n_minus': np.sum(mask_minus),
        'a': a, 'z': z, 't': t
    }

def compute_power_spectrum(pos, L_box, n_grid=256):
    """Compute power spectrum from particle positions using CIC"""
    # CIC assignment
    grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    # Normalize positions to grid units
    pos_grid = pos * n_grid / L_box

    for i in range(len(pos)):
        x, y, z = pos_grid[i]

        # Nearest grid point
        ix = int(x) % n_grid
        iy = int(y) % n_grid
        iz = int(z) % n_grid

        # CIC weights
        dx = x - int(x)
        dy = y - int(y)
        dz = z - int(z)

        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    w = ((1-dx) if di==0 else dx) * \
                        ((1-dy) if dj==0 else dy) * \
                        ((1-dz) if dk==0 else dz)
                    ii = (ix + di) % n_grid
                    jj = (iy + dj) % n_grid
                    kk = (iz + dk) % n_grid
                    grid[ii, jj, kk] += w

    # Overdensity
    mean_density = len(pos) / n_grid**3
    delta = grid / mean_density - 1.0

    # FFT
    delta_k = np.fft.fftn(delta)
    Pk_3d = np.abs(delta_k)**2 / n_grid**3

    # Bin by |k|
    k_fundamental = 2 * np.pi / L_box
    kx = np.fft.fftfreq(n_grid, d=1.0/n_grid) * k_fundamental
    ky = np.fft.fftfreq(n_grid, d=1.0/n_grid) * k_fundamental
    kz = np.fft.fftfreq(n_grid, d=1.0/n_grid) * k_fundamental

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Binning
    k_bins = np.logspace(np.log10(k_fundamental), np.log10(k_fundamental * n_grid / 2), 40)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    Pk_binned = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.sum(mask) > 0:
            Pk_binned[i] = np.mean(Pk_3d[mask])

    return k_centers, Pk_binned, delta_k

def compute_cross_spectrum(delta_k1, delta_k2, k_mag, k_bins, n_grid):
    """Compute cross power spectrum"""
    Pk_cross_3d = np.real(delta_k1 * np.conj(delta_k2)) / n_grid**3

    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    Pk_cross = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.sum(mask) > 0:
            Pk_cross[i] = np.mean(Pk_cross_3d[mask])

    return Pk_cross

def find_density_peaks(pos, L_box, n_grid=64, n_peaks=5):
    """Find density peaks using 3D histogram - faster than FOF for 10M particles"""
    half = L_box / 2
    bins = np.linspace(-half, half, n_grid + 1)
    cell_size = L_box / n_grid

    # 3D histogram
    H, edges = np.histogramdd(pos, bins=[bins, bins, bins])

    # Find peaks (cells with density > 2× mean)
    mean_density = len(pos) / n_grid**3
    threshold = 2.0 * mean_density

    # Find all cells above threshold
    peak_mask = H > threshold
    n_peaks_found = np.sum(peak_mask)

    if n_peaks_found == 0:
        return []

    # Get coordinates of peak cells
    peak_indices = np.argwhere(peak_mask)
    peak_counts = H[peak_mask]

    # Sort by count (descending)
    sort_idx = np.argsort(peak_counts)[::-1]
    peak_indices = peak_indices[sort_idx][:n_peaks]
    peak_counts = peak_counts[sort_idx][:n_peaks]

    halos = []
    for i, (idx, count) in enumerate(zip(peak_indices, peak_counts)):
        # Cell center position
        x0 = edges[0][idx[0]] + cell_size / 2
        y0 = edges[1][idx[1]] + cell_size / 2
        z0 = edges[2][idx[2]] + cell_size / 2

        # Estimate radius from local overdensity
        overdensity = count / mean_density
        r_est = cell_size * overdensity**(1/3)

        halos.append({
            'n_particles': int(count),
            'com': np.array([x0, y0, z0]),
            'r_max': r_est,
            'overdensity': overdensity
        })

    return halos


def fof_halos(pos, L_box, linking_length=0.2):
    """
    Simple FOF halo finder
    linking_length in units of mean inter-particle separation
    """
    n = len(pos)
    mean_sep = (L_box**3 / n)**(1/3)
    b = linking_length * mean_sep

    print(f"  FOF: N={n}, mean_sep={mean_sep:.4f} Mpc, b={b:.4f} Mpc")

    # Shift positions to [0, L_box] for KD-tree
    pos_shifted = pos + L_box / 2.0
    pos_shifted = pos_shifted % L_box  # Wrap any out-of-range values

    # Use KD-tree for efficiency
    tree = cKDTree(pos_shifted, boxsize=L_box)

    # Find all pairs within linking length
    pairs = tree.query_pairs(b, output_type='ndarray')

    print(f"  FOF: Found {len(pairs)} pairs")

    # Union-find structure
    parent = np.arange(n)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Connect pairs
    for p1, p2 in pairs:
        union(p1, p2)

    # Resolve all parents
    for i in range(n):
        find(i)

    # Count halos
    unique, counts = np.unique(parent, return_counts=True)

    # Filter halos with > 32 particles
    halo_mask = counts > 32
    halo_ids = unique[halo_mask]
    halo_sizes = counts[halo_mask]

    # Sort by size
    sort_idx = np.argsort(halo_sizes)[::-1]
    halo_ids = halo_ids[sort_idx]
    halo_sizes = halo_sizes[sort_idx]

    print(f"  FOF: Found {len(halo_ids)} halos with N>32")

    # Get properties of top halos
    halos = []
    for i, (hid, size) in enumerate(zip(halo_ids[:10], halo_sizes[:10])):
        members = pos_shifted[parent == hid]
        com = np.mean(members, axis=0)
        # Shift COM back to original centered coordinates
        com_orig = com - L_box / 2.0
        # R_200 approximation: radius containing all members
        dists = np.sqrt(np.sum((members - com)**2, axis=1))
        r_max = np.max(dists)
        halos.append({
            'n_particles': size,
            'com': com_orig,
            'r_max': r_max
        })

    return halos, len(halo_ids)

def radial_profile(pos, center, L_box, n_bins=30, r_max=50.0):
    """Compute radial density profile around center"""
    # Periodic distance
    delta = pos - center
    delta = delta - L_box * np.round(delta / L_box)
    r = np.sqrt(np.sum(delta**2, axis=1))

    # Bin
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        counts[i] = np.sum(mask)

    # Volume of shells
    V_shells = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)

    # Density (particles per Mpc³)
    rho = counts / V_shells

    return r_centers, rho

def main():
    if len(sys.argv) < 2:
        snap_path = Path("/mnt/T2/janus-sim/output/run_final_10m/snapshots/snap_01300.bin")
    else:
        snap_path = Path(sys.argv[1])

    L_BOX = 300.0  # Mpc
    N_GRID = 256

    print("=" * 60)
    print(f"JANUS Full Snapshot Analysis")
    print(f"Snapshot: {snap_path}")
    print("=" * 60)

    # Read data
    print("\n[1/5] Reading snapshot...")
    data = read_snapshot(snap_path)
    z_val = data['z']
    print(f"  N+ = {data['n_plus']:,}, N- = {data['n_minus']:,}")
    print(f"  z = {z_val:.4f}, a = {data['a']:.4f}")

    # Power spectra
    print("\n[2/5] Computing power spectra...")
    k_plus, Pk_plus, delta_k_plus = compute_power_spectrum(data['pos_plus'], L_BOX, N_GRID)
    k_minus, Pk_minus, delta_k_minus = compute_power_spectrum(data['pos_minus'], L_BOX, N_GRID)

    # Cross spectrum
    print("\n[3/5] Computing cross-correlation r(k)...")
    k_fundamental = 2 * np.pi / L_BOX
    k_bins = np.logspace(np.log10(k_fundamental), np.log10(k_fundamental * N_GRID / 2), 40)

    kx = np.fft.fftfreq(N_GRID, d=1.0/N_GRID) * k_fundamental
    ky = np.fft.fftfreq(N_GRID, d=1.0/N_GRID) * k_fundamental
    kz = np.fft.fftfreq(N_GRID, d=1.0/N_GRID) * k_fundamental
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    Pk_cross = compute_cross_spectrum(delta_k_plus, delta_k_minus, k_mag, k_bins, N_GRID)

    # r(k) = P_cross / sqrt(P+ * P-)
    r_k = Pk_cross / np.sqrt(Pk_plus * Pk_minus + 1e-30)

    print(f"  r(k) range: [{r_k.min():.4f}, {r_k.max():.4f}]")
    print(f"  r(k) < 0 on large scales: {r_k[k_plus < 0.1].mean():.4f}")

    # Find density peak for m+ (faster than FOF for 10M particles)
    print("\n[4/5] Finding density peaks...")
    halos_plus = find_density_peaks(data['pos_plus'], L_BOX, n_grid=64)
    n_halos = len(halos_plus)
    if halos_plus:
        print(f"  Found {n_halos} overdense regions")

    if halos_plus:
        print(f"\n  Top 5 halos:")
        for i, h in enumerate(halos_plus[:5]):
            print(f"    #{i+1}: N={h['n_particles']:,}, COM=({h['com'][0]:.1f}, {h['com'][1]:.1f}, {h['com'][2]:.1f}), R_max={h['r_max']:.2f} Mpc")

    # Radial profile around largest halo
    print("\n[5/5] Radial density profile...")
    if halos_plus:
        center = halos_plus[0]['com']
        r_centers, rho_plus = radial_profile(data['pos_plus'], center, L_BOX)
        _, rho_minus = radial_profile(data['pos_minus'], center, L_BOX)
        print(f"  Profile computed around ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

    # ===== FIGURES =====
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    # Figure 1: P(k) spectra
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.loglog(k_plus, Pk_plus, 'b-', lw=2, label='P+(k) [m+ matter]')
    ax1.loglog(k_minus, Pk_minus, 'r-', lw=2, label='P-(k) [m- antimatter]')
    ax1.set_xlabel('k [h/Mpc]', fontsize=14)
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    ax1.set_title(f'Power Spectra - Janus 10M z={z_val:.2f}', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.01, 5])
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "pk_spectrum.png", dpi=150)
    print(f"  Saved: {OUT_DIR}/pk_spectrum.png")

    # Figure 2: r(k) cross-correlation
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.semilogx(k_plus, r_k, 'k-', lw=2)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.fill_between(k_plus, r_k, 0, where=(r_k < 0), color='red', alpha=0.3, label='Anti-correlation')
    ax2.fill_between(k_plus, r_k, 0, where=(r_k > 0), color='blue', alpha=0.3, label='Correlation')
    ax2.set_xlabel('k [h/Mpc]', fontsize=14)
    ax2.set_ylabel('r(k) = P_cross / √(P+ × P-)', fontsize=14)
    ax2.set_title(f'Cross-correlation coefficient - Janus 10M z={z_val:.2f}', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.01, 5])
    ax2.set_ylim([-1, 1])
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "rk_cross.png", dpi=150)
    print(f"  Saved: {OUT_DIR}/rk_cross.png")

    # Figure 3: Radial profile
    if halos_plus:
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        ax3.semilogy(r_centers, rho_plus, 'b-', lw=2, label='ρ+(r) [m+ matter]')
        ax3.semilogy(r_centers, rho_minus, 'r--', lw=2, label='ρ-(r) [m- antimatter]')
        ax3.set_xlabel('r [Mpc]', fontsize=14)
        ax3.set_ylabel('ρ(r) [particles/Mpc³]', fontsize=14)
        ax3.set_title(f'Radial density profile around main halo\nN={halos_plus[0]["n_particles"]:,} particles', fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(OUT_DIR / "radial_profile.png", dpi=150)
        print(f"  Saved: {OUT_DIR}/radial_profile.png")

    # Figure 4: Summary figure
    fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

    # P(k)
    ax = axes[0, 0]
    ax.loglog(k_plus, Pk_plus, 'b-', lw=2, label='P+(k)')
    ax.loglog(k_minus, Pk_minus, 'r-', lw=2, label='P-(k)')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k)')
    ax.set_title('Power Spectra')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # r(k)
    ax = axes[0, 1]
    ax.semilogx(k_plus, r_k, 'k-', lw=2)
    ax.axhline(0, color='gray', ls='--')
    ax.fill_between(k_plus, r_k, 0, where=(r_k < 0), color='red', alpha=0.3)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('r(k)')
    ax.set_title('Cross-correlation')
    ax.set_ylim([-1, 1])
    ax.grid(True, alpha=0.3)

    # Radial profile
    if halos_plus:
        ax = axes[1, 0]
        ax.semilogy(r_centers, rho_plus, 'b-', lw=2, label='m+')
        ax.semilogy(r_centers, rho_minus, 'r--', lw=2, label='m-')
        ax.set_xlabel('r [Mpc]')
        ax.set_ylabel('ρ(r)')
        ax.set_title(f'Density profile (main halo N={halos_plus[0]["n_particles"]:,})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Halo summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""JANUS 10M Analysis Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Snapshot: {snap_path.name}
Box: {L_BOX} Mpc, Grid: {N_GRID}³

Particles:
  N+ = {data['n_plus']:,}
  N- = {data['n_minus']:,}

Cross-correlation r(k):
  Large scales (k<0.1): {r_k[k_plus < 0.1].mean():.4f}
  All scales mean: {r_k.mean():.4f}

FOF Halos (m+):
  Total halos (N>32): {n_halos}
"""
    if halos_plus:
        summary += f"""  Largest halo: N={halos_plus[0]['n_particles']:,}
  R_max = {halos_plus[0]['r_max']:.2f} Mpc
  COM = ({halos_plus[0]['com'][0]:.1f}, {halos_plus[0]['com'][1]:.1f}, {halos_plus[0]['com'][2]:.1f})
"""

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig4.suptitle('JANUS 10M Production Run - Full Analysis', fontsize=16, y=0.98)
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(OUT_DIR / "summary_analysis.png", dpi=150)
    print(f"  Saved: {OUT_DIR}/summary_analysis.png")

    plt.close('all')

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nKey results:")
    print(f"  r(k) < 0 on large scales: {'YES ✓' if r_k[k_plus < 0.1].mean() < 0 else 'NO ✗'}")
    print(f"  Mean r(k) at k<0.1: {r_k[k_plus < 0.1].mean():.4f}")
    print(f"  FOF halos found: {n_halos}")
    if halos_plus:
        print(f"  Largest halo: N={halos_plus[0]['n_particles']:,}, R={halos_plus[0]['r_max']:.1f} Mpc")
    print(f"\nFigures saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
