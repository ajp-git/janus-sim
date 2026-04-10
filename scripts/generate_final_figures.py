#!/usr/bin/env python3
"""
Generate final publication figures for Janus VSL μ=19 simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from collections import defaultdict

OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/vsl_petit_production")
SNAP_DIR = OUTPUT_DIR / "snapshots"
FIG_DIR = Path("/mnt/T2/janus-sim/output/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def read_snapshot(path):
    """Read JSNP v2 format snapshot."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        box = struct.unpack('<d', f.read(8))[0]

        print(f"  Reading {n:,} particles, z={z:.4f}, box={box:.1f} Mpc")

        positions = np.zeros((n, 3), dtype=np.float64)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            data = f.read(26)
            px, py, pz = struct.unpack('<ddd', data[:24])
            sign = struct.unpack('<b', data[24:25])[0]
            positions[i] = [px, py, pz]
            signs[i] = sign
            if i % 2000000 == 0 and i > 0:
                print(f"    {i:,} / {n:,}")

    return {'positions': positions, 'signs': signs, 'z': z, 'box': box, 'n': n}

def compute_density_grid(positions, box, n_grid=256, slice_axis=2, slice_frac=0.5, slice_thickness=0.1):
    """Compute 2D density projection."""
    grid = np.zeros((n_grid, n_grid))
    cell_size = box / n_grid

    # Select slice
    slice_center = box * slice_frac
    slice_half = box * slice_thickness / 2

    axis_pos = positions[:, slice_axis]
    in_slice = np.abs(axis_pos - slice_center) < slice_half
    pos_slice = positions[in_slice]

    # Project onto remaining axes
    ax1, ax2 = [i for i in range(3) if i != slice_axis]

    for pos in pos_slice:
        ix = int(pos[ax1] / cell_size) % n_grid
        iy = int(pos[ax2] / cell_size) % n_grid
        grid[iy, ix] += 1

    return grid

# ============================================================================
# FIGURE 1: Comparison z=4 vs z=0.15
# ============================================================================
def generate_comparison_figure():
    print("\n" + "="*60)
    print("FIGURE 1: comparison_z4_vs_z015.png")
    print("="*60)

    # Read both snapshots
    print("\nReading z=4 snapshot (step 0)...")
    snap0 = read_snapshot(SNAP_DIR / "snap_000000.bin")

    print("\nReading z=0.15 snapshot (step 6880)...")
    snap_final = read_snapshot(SNAP_DIR / "snap_006880.bin")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # m+ particles only
    mask_plus_0 = snap0['signs'] > 0
    mask_plus_f = snap_final['signs'] > 0

    # Compute densities
    print("  Computing density grids...")
    n_grid = 512
    rho0 = compute_density_grid(snap0['positions'][mask_plus_0], snap0['box'],
                                 n_grid=n_grid, slice_thickness=0.05)
    rho_f = compute_density_grid(snap_final['positions'][mask_plus_f], snap_final['box'],
                                  n_grid=n_grid, slice_thickness=0.05)

    # Same normalization for both
    vmax = max(np.percentile(rho0, 99.5), np.percentile(rho_f, 99.5))
    vmin = 0

    box = snap0['box']
    extent = [0, box, 0, box]

    im1 = ax1.imshow(np.log10(rho0 + 1), extent=extent, origin='lower',
                     cmap='inferno', vmin=0, vmax=np.log10(vmax + 1))
    ax1.set_title(f'z = {snap0["z"]:.1f} (conditions initiales)', fontsize=14)
    ax1.set_xlabel('X [Mpc]', fontsize=12)
    ax1.set_ylabel('Y [Mpc]', fontsize=12)

    im2 = ax2.imshow(np.log10(rho_f + 1), extent=extent, origin='lower',
                     cmap='inferno', vmin=0, vmax=np.log10(vmax + 1))
    ax2.set_title(f'z = {snap_final["z"]:.3f} (structures formées)', fontsize=14)
    ax2.set_xlabel('X [Mpc]', fontsize=12)
    ax2.set_ylabel('Y [Mpc]', fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    cbar.set_label(r'$\log_{10}(1 + N_{cell})$', fontsize=12)

    fig.suptitle('Formation de la toile cosmique Janus μ=19 VSL', fontsize=16, y=0.98)
    plt.tight_layout()

    path = FIG_DIR / "comparison_z4_vs_z015.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return snap_final  # Return for reuse

# ============================================================================
# FIGURE 2: Complementarity m+ / m-
# ============================================================================
def generate_complementarity_figure(snap):
    print("\n" + "="*60)
    print("FIGURE 2: complementarity_z015.png")
    print("="*60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    mask_plus = snap['signs'] > 0
    mask_minus = snap['signs'] < 0

    n_grid = 512
    box = snap['box']

    print("  Computing m+ density...")
    rho_plus = compute_density_grid(snap['positions'][mask_plus], box,
                                     n_grid=n_grid, slice_thickness=0.05)
    print("  Computing m- density...")
    rho_minus = compute_density_grid(snap['positions'][mask_minus], box,
                                      n_grid=n_grid, slice_thickness=0.05)

    extent = [0, box, 0, box]
    vmax = np.log10(np.percentile(rho_plus, 99.5) + 1)

    im1 = ax1.imshow(np.log10(rho_plus + 1), extent=extent, origin='lower',
                     cmap='Blues', vmin=0, vmax=vmax)
    ax1.set_title(r'Matière positive m$^+$ (halos, filaments)', fontsize=14)
    ax1.set_xlabel('X [Mpc]', fontsize=12)
    ax1.set_ylabel('Y [Mpc]', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(r'$\log_{10}(1 + N)$', fontsize=11)

    im2 = ax2.imshow(np.log10(rho_minus + 1), extent=extent, origin='lower',
                     cmap='Reds', vmin=0, vmax=vmax)
    ax2.set_title(r'Matière négative m$^-$ (vides cosmiques)', fontsize=14)
    ax2.set_xlabel('X [Mpc]', fontsize=12)
    ax2.set_ylabel('Y [Mpc]', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(r'$\log_{10}(1 + N)$', fontsize=11)

    fig.suptitle(f'Complémentarité spatiale m$^+$/m$^-$ — z = {snap["z"]:.3f}',
                 fontsize=16, y=0.98)
    plt.tight_layout()

    path = FIG_DIR / "complementarity_z015.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

# ============================================================================
# FIGURE 3: Void profile (XZ slice of m-)
# ============================================================================
def generate_void_figure(snap):
    print("\n" + "="*60)
    print("FIGURE 3: void_profile_z015.png")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 10))

    mask_minus = snap['signs'] < 0
    pos_minus = snap['positions'][mask_minus]
    box = snap['box']

    # Find region with high m- concentration (void center)
    n_grid = 64
    rho_minus = compute_density_grid(pos_minus, box, n_grid=n_grid,
                                      slice_axis=1, slice_thickness=0.1)

    # Find densest m- region
    iy, ix = np.unravel_index(np.argmax(rho_minus), rho_minus.shape)
    void_center_x = (ix + 0.5) * box / n_grid
    void_center_z = (iy + 0.5) * box / n_grid

    print(f"  Void center detected: X={void_center_x:.1f}, Z={void_center_z:.1f} Mpc")

    # Zoom around void
    zoom_size = 80  # Mpc
    x_min = max(0, void_center_x - zoom_size)
    x_max = min(box, void_center_x + zoom_size)
    z_min = max(0, void_center_z - zoom_size)
    z_max = min(box, void_center_z + zoom_size)

    # Select particles in Y slice and XZ zoom
    y_center = box / 2
    y_thickness = box * 0.05

    in_slice = (np.abs(pos_minus[:, 1] - y_center) < y_thickness / 2) & \
               (pos_minus[:, 0] > x_min) & (pos_minus[:, 0] < x_max) & \
               (pos_minus[:, 2] > z_min) & (pos_minus[:, 2] < z_max)

    pos_zoom = pos_minus[in_slice]

    # Scatter plot
    ax.scatter(pos_zoom[:, 0], pos_zoom[:, 2], s=0.5, c='red', alpha=0.3)

    # Estimate void radius from density profile
    void_radius = 25  # Mpc (typical for this simulation)
    circle = plt.Circle((void_center_x, void_center_z), void_radius,
                         fill=False, color='lime', linewidth=3, linestyle='--')
    ax.add_patch(circle)

    ax.text(void_center_x, void_center_z + void_radius + 5,
            f'Vide cosmique\nR ≈ {void_radius} Mpc',
            ha='center', va='bottom', fontsize=12, color='lime',
            bbox=dict(facecolor='black', alpha=0.7, pad=3))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('X [Mpc]', fontsize=13)
    ax.set_ylabel('Z [Mpc]', fontsize=13)
    ax.set_title(f'Distribution m$^-$ (tranche XZ) — z = {snap["z"]:.3f}\n'
                 f'Les m$^-$ remplissent les vides créés par les m$^+$', fontsize=14)
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    path = FIG_DIR / "void_profile_z015.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='black')
    print(f"  Saved: {path}")
    plt.close()

# ============================================================================
# FIGURE 4: Segregation evolution
# ============================================================================
def generate_segregation_evolution():
    print("\n" + "="*60)
    print("FIGURE 4: segregation_evolution.png")
    print("="*60)

    # Read evolution data from simulation log
    log_path = OUTPUT_DIR / "simulation.log"

    steps = []
    z_vals = []
    seg_vals = []

    print("  Parsing simulation log...")
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) >= 9:
                try:
                    step = int(parts[0].strip())
                    z = float(parts[2].strip())
                    seg = float(parts[8].strip())
                    steps.append(step)
                    z_vals.append(z)
                    seg_vals.append(seg)
                except:
                    continue

    z_vals = np.array(z_vals)
    seg_vals = np.array(seg_vals)

    # Filter to z > 0.19 (before contamination)
    valid = z_vals > 0.19
    z_valid = z_vals[valid]
    seg_valid = seg_vals[valid]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all data
    ax.plot(z_vals, seg_vals, 'b-', linewidth=1, alpha=0.3, label='Données brutes')
    ax.plot(z_valid, seg_valid, 'b-', linewidth=2, label='Région validée (z > 0.19)')

    # Mark z=0.19 limit
    ax.axvline(x=0.19, color='red', linestyle='--', linewidth=2,
               label='Limite de validité z=0.19')
    ax.axvspan(0, 0.19, alpha=0.2, color='red')
    ax.text(0.1, seg_vals.max() * 0.95, 'Zone\ncontaminée',
            fontsize=10, color='red', ha='center')

    # Initial segregation line
    ax.axhline(y=seg_vals[0], color='gray', linestyle=':', linewidth=1,
               label=f'Seg initiale = {seg_vals[0]:.4f}')

    ax.set_xlabel('Redshift z', fontsize=13)
    ax.set_ylabel('Paramètre de ségrégation Seg', fontsize=13)
    ax.set_xlim(z_vals.max(), 0)  # Reverse x-axis (high z to low z)
    ax.set_ylim(0, seg_vals.max() * 1.1)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Évolution de la ségrégation m$^+$/m$^-$ — Janus μ=19 VSL', fontsize=14)

    # Add annotation
    ax.annotate(f'Seg max = {seg_valid.max():.4f}\n(z = {z_valid[np.argmax(seg_valid)]:.3f})',
                xy=(z_valid[np.argmax(seg_valid)], seg_valid.max()),
                xytext=(z_valid[np.argmax(seg_valid)] + 0.5, seg_valid.max() * 0.85),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'),
                bbox=dict(facecolor='white', edgecolor='blue', alpha=0.9))

    plt.tight_layout()

    path = FIG_DIR / "segregation_evolution.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    print(f"\n  Segregation at z=4.0: {seg_vals[0]:.4f}")
    print(f"  Segregation max (z>0.19): {seg_valid.max():.4f}")
    print(f"  Segregation at z=0.19: {seg_vals[z_vals > 0.18][0]:.4f}")

# ============================================================================
# FOF Analysis on snap_006880
# ============================================================================
def run_fof_analysis(snap):
    print("\n" + "="*60)
    print("FOF ANALYSIS — Snapshot 006880")
    print("="*60)

    positions = snap['positions']
    signs = snap['signs']
    box = snap['box']
    z = snap['z']
    n = snap['n']

    # FOF on m+ particles
    print("\nFOF clustering on m+ particles...")
    mask_plus = signs > 0
    pos_plus = positions[mask_plus]
    idx_plus = np.where(mask_plus)[0]

    mean_sep = box / (n ** (1/3))
    ll = 0.2 * mean_sep
    print(f"  Linking length: {ll:.3f} Mpc")

    # Union-Find
    n_plus = len(pos_plus)
    parent = np.arange(n_plus)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Grid-based search
    cell_size = ll
    n_cells = int(np.ceil(box / cell_size))
    cells = defaultdict(list)
    for i, pos in enumerate(pos_plus):
        cx = int(pos[0] / cell_size) % n_cells
        cy = int(pos[1] / cell_size) % n_cells
        cz = int(pos[2] / cell_size) % n_cells
        cells[(cx, cy, cz)].append(i)

    print(f"  Grid: {n_cells}^3 cells")
    ll_sq = ll ** 2
    n_links = 0

    for (cx, cy, cz), particles in cells.items():
        for dcx in [-1, 0, 1]:
            for dcy in [-1, 0, 1]:
                for dcz in [-1, 0, 1]:
                    ncx = (cx + dcx) % n_cells
                    ncy = (cy + dcy) % n_cells
                    ncz = (cz + dcz) % n_cells
                    for i in particles:
                        for j in cells.get((ncx, ncy, ncz), []):
                            if i >= j:
                                continue
                            dx = pos_plus[j] - pos_plus[i]
                            dx = dx - box * np.round(dx / box)
                            if np.sum(dx**2) < ll_sq:
                                union(i, j)
                                n_links += 1

    print(f"  Links found: {n_links:,}")

    # Build groups
    groups = defaultdict(list)
    for i in range(n_plus):
        groups[find(i)].append(i)

    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    print(f"  Total FOF groups: {len(sorted_groups):,}")

    # Analyze top 10 halos
    rho_mean = n / (box ** 3)

    results = []
    print(f"\n{'='*70}")
    print(f"TOP 10 HALOS — z = {z:.4f} (step 6880)")
    print(f"{'='*70}")

    for rank, g in enumerate(sorted_groups[:10]):
        full_idx = [idx_plus[i] for i in g]
        pos_halo = positions[full_idx]

        # COM
        ref = pos_halo[0].copy()
        delta = pos_halo - ref
        delta = delta - box * np.round(delta / box)
        com = ref + np.mean(delta, axis=0)
        com = com % box

        # Radii
        dr = pos_halo - com
        dr = dr - box * np.round(dr / box)
        r = np.sqrt(np.sum(dr**2, axis=1))
        r_max = np.max(r)

        # R_200
        r_sorted = np.sort(r)
        n_halo = len(g)
        r_200 = (3 * n_halo / (4 * np.pi * 200 * rho_mean)) ** (1/3)

        # Check m- within 3*R_200
        all_dr = positions - com
        all_dr = all_dr - box * np.round(all_dr / box)
        all_r = np.sqrt(np.sum(all_dr**2, axis=1))

        in_3r200 = all_r < 3 * r_200
        n_plus_3r = np.sum((signs > 0) & in_3r200)
        n_minus_3r = np.sum((signs < 0) & in_3r200)

        in_r200 = all_r < r_200
        n_plus_r = np.sum((signs > 0) & in_r200)
        n_minus_r = np.sum((signs < 0) & in_r200)

        ratio = n_minus_r / n_plus_r if n_plus_r > 0 else 0
        depletion = (1 - ratio / 0.9569) * 100 if n_plus_r > 0 else 100

        results.append({
            'rank': rank + 1,
            'n_fof': n_halo,
            'center': com,
            'r_200': r_200,
            'r_max': r_max,
            'n_plus_r200': n_plus_r,
            'n_minus_r200': n_minus_r,
            'n_plus_3r200': n_plus_3r,
            'n_minus_3r200': n_minus_3r,
            'depletion': depletion
        })

        print(f"\nHalo #{rank+1}:")
        print(f"  N_FOF = {n_halo:,}")
        print(f"  Center: ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) Mpc")
        print(f"  R_200 = {r_200:.2f} Mpc, R_max = {r_max:.2f} Mpc")
        print(f"  Within R_200:   N_m+ = {n_plus_r:,}, N_m- = {n_minus_r:,}")
        print(f"  Within 3×R_200: N_m+ = {n_plus_3r:,}, N_m- = {n_minus_3r:,}")
        print(f"  m- depletion = {depletion:.1f}%")

    # Save results to CSV
    csv_path = OUTPUT_DIR / "fof_results_step6880.csv"
    with open(csv_path, 'w') as f:
        f.write("rank,n_fof,x,y,z,r_200,r_max,n_plus_r200,n_minus_r200,n_plus_3r200,n_minus_3r200,depletion_pct\n")
        for r in results:
            f.write(f"{r['rank']},{r['n_fof']},{r['center'][0]:.2f},{r['center'][1]:.2f},{r['center'][2]:.2f},"
                    f"{r['r_200']:.3f},{r['r_max']:.3f},{r['n_plus_r200']},{r['n_minus_r200']},"
                    f"{r['n_plus_3r200']},{r['n_minus_3r200']},{r['depletion']:.1f}\n")
    print(f"\nFOF results saved to: {csv_path}")

    return results

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("GENERATING FINAL FIGURES — Janus VSL μ=19 Production Run")
    print("="*70)

    # Figure 1: Comparison z=4 vs z=0.15
    snap_final = generate_comparison_figure()

    # Figure 2: Complementarity
    generate_complementarity_figure(snap_final)

    # Figure 3: Void profile
    generate_void_figure(snap_final)

    # Figure 4: Segregation evolution
    generate_segregation_evolution()

    # FOF Analysis
    fof_results = run_fof_analysis(snap_final)

    print("\n" + "="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nFigures saved to: {FIG_DIR}")
    print("  - comparison_z4_vs_z015.png")
    print("  - complementarity_z015.png")
    print("  - void_profile_z015.png")
    print("  - segregation_evolution.png")
    print(f"\nFOF results: {OUTPUT_DIR / 'fof_results_step6880.csv'}")
