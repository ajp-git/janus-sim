#!/usr/bin/env python3
"""
Analyse complète du snapshot 15M Run
- Carte de polarisation 6 panneaux
- Carte P rouge/bleu
- Métriques P=±1 dans les halos
- Évolution temporelle
- L_J à résolution adaptative
"""

import numpy as np
import struct
import os
import sys
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Paramètres
BOX = 500.0
HALF = BOX / 2

def read_snapshot(path):
    """Load snapshot: N (u64), then N × (x,y,z,vx,vy,vz,sign) as f32"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * 7 * 4), dtype=np.float32).reshape(n, 7)
    pos = data[:, :3]
    signs = np.sign(data[:, 6]).astype(np.int8)
    return pos, signs, n

def deposit_grid(pos, signs, grid_size):
    """Deposit particles onto a grid, return ρ+ and ρ- grids"""
    cell = BOX / grid_size
    grid_p = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    grid_m = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    ix = np.clip(((pos[:, 0] + HALF) / BOX * grid_size).astype(int), 0, grid_size-1)
    iy = np.clip(((pos[:, 1] + HALF) / BOX * grid_size).astype(int), 0, grid_size-1)
    iz = np.clip(((pos[:, 2] + HALF) / BOX * grid_size).astype(int), 0, grid_size-1)

    mask_p = signs > 0
    mask_m = signs < 0
    np.add.at(grid_p, (ix[mask_p], iy[mask_p], iz[mask_p]), 1)
    np.add.at(grid_m, (ix[mask_m], iy[mask_m], iz[mask_m]), 1)

    return grid_p, grid_m

def compute_P(grid_p, grid_m):
    """Compute parity field P = (ρ+ - ρ-) / (ρ+ + ρ-)"""
    total = grid_p + grid_m
    mask = total > 0
    P = np.where(mask, (grid_p - grid_m) / total, 0)
    return P, mask

# ============================================================================
# 1. Carte de polarisation 6 panneaux
# ============================================================================
def render_6panels(pos, signs, output_path, grid_res=768, smooth_sigma=1.4):
    """Render 6-panel density map (XY, XZ, YZ for ρ+ and ρ-)"""
    print("  Rendering 6-panel polarization map...")

    # Deposit on high-res grid
    grid_p, grid_m = deposit_grid(pos, signs, grid_res)

    # Smooth
    grid_p = gaussian_filter(grid_p, sigma=smooth_sigma)
    grid_m = gaussian_filter(grid_m, sigma=smooth_sigma)

    # Project onto 2D
    proj_xy_p = grid_p.sum(axis=2)
    proj_xy_m = grid_m.sum(axis=2)
    proj_xz_p = grid_p.sum(axis=1)
    proj_xz_m = grid_m.sum(axis=1)
    proj_yz_p = grid_p.sum(axis=0)
    proj_yz_m = grid_m.sum(axis=0)

    # Normalize
    vmax = max(proj_xy_p.max(), proj_xy_m.max(),
               proj_xz_p.max(), proj_xz_m.max(),
               proj_yz_p.max(), proj_yz_m.max())
    vmin = 0.1

    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), facecolor='#0a0a0a')
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    cmap_p = plt.cm.Reds
    cmap_m = plt.cm.Blues

    panels = [
        (proj_xy_p, 'XY ρ+', cmap_p, 0, 0),
        (proj_xz_p, 'XZ ρ+', cmap_p, 0, 1),
        (proj_yz_p, 'YZ ρ+', cmap_p, 0, 2),
        (proj_xy_m, 'XY ρ−', cmap_m, 1, 0),
        (proj_xz_m, 'XZ ρ−', cmap_m, 1, 1),
        (proj_yz_m, 'YZ ρ−', cmap_m, 1, 2),
    ]

    for data, title, cmap, row, col in panels:
        ax = axes[row, col]
        im = ax.imshow(data.T, origin='lower', cmap=cmap,
                       norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                       extent=[-HALF, HALF, -HALF, HALF])
        ax.set_title(title, color='white', fontsize=14)
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.savefig(output_path, dpi=200, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")

# ============================================================================
# 2. Carte de polarisation P (rouge/bleu)
# ============================================================================
def render_P_map(pos, signs, output_path, grid_res=512, slice_thick=5.0):
    """Render P map for central XY slice"""
    print("  Rendering P polarization map...")

    # Select particles in central slice
    mask_slice = np.abs(pos[:, 2]) < slice_thick
    pos_slice = pos[mask_slice]
    signs_slice = signs[mask_slice]

    grid_p, grid_m = deposit_grid(pos_slice, signs_slice, grid_res)
    # Only use XY projection (z=0 slice)
    P, mask = compute_P(grid_p[:,:,grid_res//2], grid_m[:,:,grid_res//2])

    # Actually, for a thin slice, deposit directly onto 2D
    grid_p_2d = np.zeros((grid_res, grid_res), dtype=np.float32)
    grid_m_2d = np.zeros((grid_res, grid_res), dtype=np.float32)

    ix = np.clip(((pos_slice[:, 0] + HALF) / BOX * grid_res).astype(int), 0, grid_res-1)
    iy = np.clip(((pos_slice[:, 1] + HALF) / BOX * grid_res).astype(int), 0, grid_res-1)

    mask_p = signs_slice > 0
    mask_m = signs_slice < 0
    np.add.at(grid_p_2d, (ix[mask_p], iy[mask_p]), 1)
    np.add.at(grid_m_2d, (ix[mask_m], iy[mask_m]), 1)

    total = grid_p_2d + grid_m_2d
    mask = total > 0
    P = np.where(mask, (grid_p_2d - grid_m_2d) / total, 0)

    # Smooth slightly
    P = gaussian_filter(P, sigma=1.0)

    # Custom colormap: blue (-1) -> white (0) -> red (+1)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue, white, red
    cmap = LinearSegmentedColormap.from_list('PolarityMap', colors, N=256)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0a0a0a')
    im = ax.imshow(P.T, origin='lower', cmap=cmap, vmin=-1, vmax=1,
                   extent=[-HALF, HALF, -HALF, HALF])
    ax.set_title(f'Parity P = (ρ+ - ρ−)/(ρ+ + ρ−)  |  XY slice z=0±{slice_thick} Mpc',
                 color='white', fontsize=14)
    ax.set_xlabel('X [Mpc]', color='white')
    ax.set_ylabel('Y [Mpc]', color='white')
    ax.set_facecolor('#0a0a0a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    cbar = plt.colorbar(im, ax=ax, label='P')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")

# ============================================================================
# 3. Métriques P=±1 dans les halos
# ============================================================================
def compute_halo_metrics(pos, signs, grid_size=256):
    """Compute P=±1 metrics in populated cells"""
    print("  Computing halo purity metrics...")

    grid_p, grid_m = deposit_grid(pos, signs, grid_size)
    P, mask = compute_P(grid_p, grid_m)
    total = grid_p + grid_m

    n_populated = mask.sum()
    n_total = grid_size**3

    sigma_P_global = np.std(P)
    sigma_P_populated = np.std(P[mask])

    P_pop = P[mask]
    pct_pure_plus = (P_pop > 0.95).mean() * 100
    pct_pure_minus = (P_pop < -0.95).mean() * 100
    pct_mixed = (np.abs(P_pop) < 0.5).mean() * 100

    mean_occupancy = total[mask].mean()

    metrics = {
        'grid_size': grid_size,
        'n_populated': n_populated,
        'pct_populated': 100 * n_populated / n_total,
        'sigma_P_global': sigma_P_global,
        'sigma_P_populated': sigma_P_populated,
        'pct_pure_plus': pct_pure_plus,
        'pct_pure_minus': pct_pure_minus,
        'pct_pure_total': pct_pure_plus + pct_pure_minus,
        'pct_mixed': pct_mixed,
        'mean_occupancy': mean_occupancy,
    }

    return metrics

# ============================================================================
# 4. Évolution temporelle P=±1
# ============================================================================
def temporal_evolution(snap_dir, output_path):
    """Compute P metrics evolution over all snapshots"""
    print("  Computing temporal evolution...")

    snaps = sorted(glob.glob(f"{snap_dir}/snap_*.bin"))

    results = []
    for snap_path in snaps:
        step = int(os.path.basename(snap_path).split('_')[1].split('.')[0])
        z = 5.0 * (1 - step / 5000)  # approximate

        pos, signs, n = read_snapshot(snap_path)
        metrics = compute_halo_metrics(pos, signs, grid_size=128)  # faster

        results.append({
            'step': step,
            'z': z,
            **metrics
        })
        print(f"    Step {step} (z={z:.2f}): P>0.95={metrics['pct_pure_plus']:.1f}%, P<-0.95={metrics['pct_pure_minus']:.1f}%")

    # Plot
    steps = [r['step'] for r in results]
    z_vals = [r['z'] for r in results]
    pct_plus = [r['pct_pure_plus'] for r in results]
    pct_minus = [r['pct_pure_minus'] for r in results]
    sigma_pop = [r['sigma_P_populated'] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(z_vals, pct_plus, 'r.-', label='P > 0.95 (pure +)')
    ax1.plot(z_vals, pct_minus, 'b.-', label='P < -0.95 (pure −)')
    ax1.plot(z_vals, np.array(pct_plus) + np.array(pct_minus), 'k.-', label='Total pure')
    ax1.set_ylabel('% of populated cells')
    ax1.set_title('Halo purity evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2 = axes[1]
    ax2.plot(z_vals, sigma_pop, 'g.-', label='σ_P (populated)')
    ax2.set_xlabel('z')
    ax2.set_ylabel('σ_P')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")

    return results

# ============================================================================
# 5. L_J à résolution adaptative
# ============================================================================
def LJ_convergence(pos, signs, output_path):
    """Compute L_J at different resolutions"""
    print("  Computing L_J convergence...")

    cell_sizes = [2, 4, 8, 16, 32]
    results = []

    for cell in cell_sizes:
        grid_size = int(BOX / cell)
        if grid_size < 10:
            continue

        grid_p, grid_m = deposit_grid(pos, signs, grid_size)
        P, mask = compute_P(grid_p, grid_m)

        # Smooth P
        P_smooth = gaussian_filter(P, sigma=1.0)

        # Gradient
        grad = np.gradient(P_smooth, cell)  # cell is voxel size in Mpc
        grad_mag = np.sqrt(sum(g**2 for g in grad))

        sigma_P = np.std(P_smooth[mask]) if mask.any() else 0
        mean_grad = np.mean(grad_mag[mask]) if mask.any() else 0
        L_J = sigma_P / (mean_grad + 1e-10)

        pct_pop = 100 * mask.sum() / (grid_size**3)

        results.append({
            'cell': cell,
            'grid': grid_size,
            'sigma_P': sigma_P,
            'mean_grad': mean_grad,
            'L_J': L_J,
            'pct_populated': pct_pop,
        })
        print(f"    cell={cell} Mpc (grid {grid_size}³): L_J={L_J:.2f} Mpc, σ_P={sigma_P:.4f}, pop={pct_pop:.1f}%")

    # Plot
    cells = [r['cell'] for r in results]
    LJs = [r['L_J'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cells, LJs, 'ko-', markersize=10)
    ax.axhline(5, color='green', ls='--', label='L_J ~ 5 Mpc (filament scale)')
    ax.set_xlabel('Cell size [Mpc]')
    ax.set_ylabel('L_J [Mpc]')
    ax.set_title('Jeans length convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")

    return results

# ============================================================================
# Main
# ============================================================================
def main(snap_path):
    print(f"\n{'='*60}")
    print(f"ANALYSE SNAPSHOT 15M")
    print(f"{'='*60}")
    print(f"Input: {snap_path}")

    # Output directory
    snap_name = os.path.basename(snap_path).replace('.bin', '')
    out_dir = f"/tmp/analysis_{snap_name}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}/")

    # Load snapshot
    print("\nLoading snapshot...")
    pos, signs, n = read_snapshot(snap_path)
    print(f"  {n:,} particles loaded")
    print(f"  N+ = {(signs > 0).sum():,}, N- = {(signs < 0).sum():,}")

    # 1. 6-panel map
    print("\n[1/5] 6-panel polarization map")
    render_6panels(pos, signs, f"{out_dir}/polarization_6panels.png")

    # 2. P map
    print("\n[2/5] P polarization map")
    render_P_map(pos, signs, f"{out_dir}/polarization_map_XY.png")

    # 3. Halo metrics
    print("\n[3/5] Halo purity metrics")
    metrics = compute_halo_metrics(pos, signs, grid_size=256)

    metrics_txt = f"{out_dir}/metrics_{snap_name}.txt"
    with open(metrics_txt, 'w') as f:
        f.write(f"Snapshot: {snap_path}\n")
        f.write(f"N particles: {n:,}\n")
        f.write(f"\nGrid analysis ({metrics['grid_size']}³):\n")
        f.write(f"  Populated cells: {metrics['n_populated']:,} / {metrics['grid_size']**3:,} ({metrics['pct_populated']:.1f}%)\n")
        f.write(f"  Mean occupancy: {metrics['mean_occupancy']:.1f} particles/cell\n")
        f.write(f"\nParity metrics:\n")
        f.write(f"  σ_P (global):    {metrics['sigma_P_global']:.4f}\n")
        f.write(f"  σ_P (populated): {metrics['sigma_P_populated']:.4f}\n")
        f.write(f"\nHalo purity (populated cells):\n")
        f.write(f"  P > +0.95 (pure +): {metrics['pct_pure_plus']:.1f}%\n")
        f.write(f"  P < -0.95 (pure −): {metrics['pct_pure_minus']:.1f}%\n")
        f.write(f"  Total pure:         {metrics['pct_pure_total']:.1f}%\n")
        f.write(f"  |P| < 0.5 (mixed):  {metrics['pct_mixed']:.1f}%\n")
    print(f"    Saved: {metrics_txt}")

    print(f"\n  Results:")
    print(f"    σ_P (populated) = {metrics['sigma_P_populated']:.4f}")
    print(f"    Pure + halos:     {metrics['pct_pure_plus']:.1f}%")
    print(f"    Pure − halos:     {metrics['pct_pure_minus']:.1f}%")
    print(f"    Mixed:            {metrics['pct_mixed']:.1f}%")

    # 4. Temporal evolution
    print("\n[4/5] Temporal evolution")
    snap_dir = os.path.dirname(snap_path)
    temporal_evolution(snap_dir, f"{out_dir}/temporal_evolution.png")

    # 5. L_J convergence
    print("\n[5/5] L_J convergence")
    LJ_convergence(pos, signs, f"{out_dir}/LJ_convergence.png")

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {out_dir}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        snap_path = "/mnt/T2/janus-sim/output/janus_v13_500Mpc_15M/snapshots/snap_003200.bin"
    else:
        snap_path = sys.argv[1]

    main(snap_path)
