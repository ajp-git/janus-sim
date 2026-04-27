#!/usr/bin/env python3
"""
Render 10-panel 4K frames for Janus baryonic physics simulation.

Layout (5 cols x 2 rows):
Row 1: [XY m+] [XZ m+] [3D scatter] [Temperature] [Purity]
Row 2: [XY m-] [XZ m-] [SPH density] [Sound speed] [Velocity dist]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
import struct
import glob
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical constants
GAMMA = 5.0/3.0  # Adiabatic index
MU_MOL = 0.6  # Mean molecular weight (ionized hydrogen)
MPC_GYR_TO_KMS = 978.5  # 1 Mpc/Gyr = 978.5 km/s

def read_snapshot_v2(filepath):
    """Read binary snapshot v2 format with interleaved data."""
    with open(filepath, 'rb') as f:
        # Header
        header = struct.unpack('i', f.read(4))[0]

        if header == -2:
            # V2 format with velocities
            n = struct.unpack('I', f.read(4))[0]

            # Each particle: 3 pos + 3 vel + sign + temp = 8 doubles = 64 bytes
            data = np.frombuffer(f.read(n * 64), dtype=np.float64).reshape(n, 8)

            pos = data[:, 0:3].copy()
            vel = data[:, 3:6].copy()
            signs = data[:, 6].astype(np.int32)
            temp = data[:, 7].copy()
        else:
            raise ValueError(f"Unknown format header: {header}")

    # Extract box_size from position range
    box_size = (pos.max() - pos.min()) * 1.05

    # Extract step from filename
    step = int(os.path.basename(filepath).split('_')[1].split('.')[0])

    return {
        'n': n,
        'box_size': box_size,
        'step': step,
        'pos': pos,
        'vel': vel,
        'signs': signs,
        'temp': temp
    }

def compute_density_2d(pos, signs, box_size, grid_size=512, sign_filter=None):
    """Compute 2D density histogram for specified sign."""
    if sign_filter is not None:
        mask = signs == sign_filter
        pos_filtered = pos[mask]
    else:
        pos_filtered = pos

    half_box = box_size / 2

    # XY projection
    H_xy, xedges, yedges = np.histogram2d(
        pos_filtered[:, 0], pos_filtered[:, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    # XZ projection
    H_xz, _, _ = np.histogram2d(
        pos_filtered[:, 0], pos_filtered[:, 2],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    return H_xy.T, H_xz.T, xedges, yedges

def compute_temperature_map(pos, temp, signs, box_size, grid_size=256):
    """Compute temperature map for m+ particles."""
    mask = signs > 0
    pos_plus = pos[mask]
    temp_plus = temp[mask]

    half_box = box_size / 2

    # Binned temperature (mass-weighted average)
    H_counts, xedges, yedges = np.histogram2d(
        pos_plus[:, 0], pos_plus[:, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    H_temp, _, _ = np.histogram2d(
        pos_plus[:, 0], pos_plus[:, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]],
        weights=temp_plus
    )

    # Average temperature per cell
    with np.errstate(divide='ignore', invalid='ignore'):
        T_map = H_temp / H_counts
        T_map = np.nan_to_num(T_map, nan=0)

    return T_map.T

def compute_purity_map(pos, signs, box_size, grid_size=256):
    """Compute purity map: (n+ - n-)/(n+ + n-)."""
    half_box = box_size / 2

    mask_plus = signs > 0
    mask_minus = signs < 0

    H_plus, xedges, yedges = np.histogram2d(
        pos[mask_plus, 0], pos[mask_plus, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    H_minus, _, _ = np.histogram2d(
        pos[mask_minus, 0], pos[mask_minus, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        purity = (H_plus - H_minus) / (H_plus + H_minus + 1e-10)
        purity = np.nan_to_num(purity, nan=0)

    return purity.T

def compute_sph_density_map(pos, signs, box_size, grid_size=256):
    """Compute SPH density map (approximation using histogram + smoothing)."""
    from scipy.ndimage import gaussian_filter

    mask = signs > 0
    pos_plus = pos[mask]

    half_box = box_size / 2

    H, xedges, yedges = np.histogram2d(
        pos_plus[:, 0], pos_plus[:, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    # Approximate SPH density with smoothing
    H_smooth = gaussian_filter(H + 1e-10, sigma=2)

    return H_smooth.T

def compute_sound_speed_map(pos, temp, signs, box_size, grid_size=256):
    """Compute sound speed map: cs = sqrt(gamma * k_B * T / (mu * m_p))."""
    mask = signs > 0
    pos_plus = pos[mask]
    temp_plus = temp[mask]

    half_box = box_size / 2

    # cs in km/s: sqrt(gamma * k_B * T / (mu * m_p))
    # k_B/m_p = 8.314e3 m^2/s^2/K for hydrogen
    cs_plus = np.sqrt(GAMMA * 8314.0 * temp_plus / MU_MOL) * 1e-3  # km/s

    H_counts, xedges, yedges = np.histogram2d(
        pos_plus[:, 0], pos_plus[:, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]]
    )

    H_cs, _, _ = np.histogram2d(
        pos_plus[:, 0], pos_plus[:, 1],
        bins=grid_size, range=[[-half_box, half_box], [-half_box, half_box]],
        weights=cs_plus
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        cs_map = H_cs / H_counts
        cs_map = np.nan_to_num(cs_map, nan=0)

    return cs_map.T

def render_frame(snapshot_path, output_dir, box_size_override=100.0):
    """Render a single 10-panel frame."""
    data = read_snapshot_v2(snapshot_path)

    step = data['step']
    box_size = box_size_override  # Use fixed box size
    n = data['n']
    pos = data['pos']
    vel = data['vel']
    signs = data['signs']
    temp = data['temp']

    # Time estimate from step (dt = 0.001 Gyr)
    time_gyr = step * 0.001

    # Compute derived quantities
    n_plus = np.sum(signs > 0)
    n_minus = np.sum(signs < 0)

    # Redshift estimate (very approximate)
    z_approx = max(0, 10 - time_gyr / 1.38)

    # Densities
    H_xy_plus, H_xz_plus, xedges, yedges = compute_density_2d(pos, signs, box_size, 512, 1)
    H_xy_minus, H_xz_minus, _, _ = compute_density_2d(pos, signs, box_size, 512, -1)

    rho_plus_max = H_xy_plus.max()
    rho_minus_max = H_xy_minus.max()

    # Temperature and derived maps
    T_map = compute_temperature_map(pos, temp, signs, box_size, 256)
    purity_map = compute_purity_map(pos, signs, box_size, 256)
    sph_map = compute_sph_density_map(pos, signs, box_size, 256)
    cs_map = compute_sound_speed_map(pos, temp, signs, box_size, 256)

    # Velocity distributions
    v_mag = np.linalg.norm(vel, axis=1) * MPC_GYR_TO_KMS
    v_plus_kms = v_mag[signs > 0]
    v_minus_kms = v_mag[signs < 0]

    T_mean = np.mean(temp[signs > 0])
    T_min = np.min(temp[signs > 0])
    v_rms = np.sqrt(np.mean(v_mag**2))

    # Correlation
    corr = np.corrcoef(H_xy_plus.flatten(), H_xy_minus.flatten())[0, 1]

    # Create figure (4K: 3840x2160)
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)

    # Grid layout
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.3],
                         left=0.03, right=0.97, top=0.92, bottom=0.05,
                         wspace=0.15, hspace=0.15)

    half_box = box_size / 2
    extent = [-half_box, half_box, -half_box, half_box]

    # Panel 1: XY m+ density
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(H_xy_plus + 1, extent=extent, origin='lower',
                     cmap='hot', norm=LogNorm(vmin=1, vmax=max(200, rho_plus_max)))
    ax1.set_title('XY m+ density', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X [Mpc]')
    ax1.set_ylabel('Y [Mpc]')
    plt.colorbar(im1, ax=ax1, label='counts')

    # Panel 2: XZ m+ density
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(H_xz_plus + 1, extent=extent, origin='lower',
                     cmap='hot', norm=LogNorm(vmin=1, vmax=max(200, rho_plus_max)))
    ax2.set_title('XZ m+ density', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X [Mpc]')
    ax2.set_ylabel('Z [Mpc]')
    plt.colorbar(im2, ax=ax2, label='counts')

    # Panel 3: 3D scatter
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    # Subsample for performance
    n_sample = min(100000, n)
    idx = np.random.choice(n, n_sample, replace=False)
    pos_sample = pos[idx]
    signs_sample = signs[idx]

    mask_plus_sample = signs_sample > 0
    mask_minus_sample = signs_sample < 0

    ax3.scatter(pos_sample[mask_plus_sample, 0],
                pos_sample[mask_plus_sample, 1],
                pos_sample[mask_plus_sample, 2],
                c='red', s=0.1, alpha=0.3, label='m+')
    ax3.scatter(pos_sample[mask_minus_sample, 0],
                pos_sample[mask_minus_sample, 1],
                pos_sample[mask_minus_sample, 2],
                c='blue', s=0.1, alpha=0.3, label='m-')
    ax3.set_xlim(-half_box, half_box)
    ax3.set_ylim(-half_box, half_box)
    ax3.set_zlim(-half_box, half_box)
    ax3.set_title('3D scatter (100k)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Panel 4: Temperature map
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(T_map, extent=extent, origin='lower',
                     cmap='plasma', vmin=100, vmax=1e4)
    ax4.set_title('Temperature [K]', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X [Mpc]')
    ax4.set_ylabel('Y [Mpc]')
    plt.colorbar(im4, ax=ax4, label='T [K]')

    # Panel 5: Purity map
    ax5 = fig.add_subplot(gs[0, 4])
    im5 = ax5.imshow(purity_map, extent=extent, origin='lower',
                     cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_title('Purity (n+-n-)/(n++n-)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('X [Mpc]')
    ax5.set_ylabel('Y [Mpc]')
    plt.colorbar(im5, ax=ax5, label='Purity')

    # Panel 6: XY m- density
    ax6 = fig.add_subplot(gs[1, 0])
    im6 = ax6.imshow(H_xy_minus + 1, extent=extent, origin='lower',
                     cmap='cool', norm=LogNorm(vmin=1, vmax=max(200, rho_minus_max)))
    ax6.set_title('XY m- density', fontsize=14, fontweight='bold')
    ax6.set_xlabel('X [Mpc]')
    ax6.set_ylabel('Y [Mpc]')
    plt.colorbar(im6, ax=ax6, label='counts')

    # Panel 7: XZ m- density
    ax7 = fig.add_subplot(gs[1, 1])
    im7 = ax7.imshow(H_xz_minus + 1, extent=extent, origin='lower',
                     cmap='cool', norm=LogNorm(vmin=1, vmax=max(200, rho_minus_max)))
    ax7.set_title('XZ m- density', fontsize=14, fontweight='bold')
    ax7.set_xlabel('X [Mpc]')
    ax7.set_ylabel('Z [Mpc]')
    plt.colorbar(im7, ax=ax7, label='counts')

    # Panel 8: SPH density
    ax8 = fig.add_subplot(gs[1, 2])
    im8 = ax8.imshow(sph_map, extent=extent, origin='lower',
                     cmap='inferno', norm=LogNorm(vmin=1, vmax=max(100, sph_map.max())))
    ax8.set_title('SPH density (m+)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('X [Mpc]')
    ax8.set_ylabel('Y [Mpc]')
    plt.colorbar(im8, ax=ax8, label=r'$\rho_{SPH}$')

    # Panel 9: Sound speed
    ax9 = fig.add_subplot(gs[1, 3])
    cs_valid = cs_map[cs_map > 0]
    cs_vmax = np.percentile(cs_valid, 99) if len(cs_valid) > 0 else 100
    im9 = ax9.imshow(cs_map, extent=extent, origin='lower',
                     cmap='viridis', vmin=0, vmax=cs_vmax)
    ax9.set_title('Sound speed [km/s]', fontsize=14, fontweight='bold')
    ax9.set_xlabel('X [Mpc]')
    ax9.set_ylabel('Y [Mpc]')
    plt.colorbar(im9, ax=ax9, label=r'$c_s$ [km/s]')

    # Panel 10: Velocity distribution
    ax10 = fig.add_subplot(gs[1, 4])
    v_max = min(50, max(v_plus_kms.max(), v_minus_kms.max()) * 1.1)
    bins = np.linspace(0, v_max, 50)
    ax10.hist(v_plus_kms, bins=bins, alpha=0.7, color='red',
              label=f'm+ (N={len(v_plus_kms):,})', density=True)
    ax10.hist(v_minus_kms, bins=bins, alpha=0.7, color='blue',
              label=f'm- (N={len(v_minus_kms):,})', density=True)
    ax10.axvline(v_rms, color='black', linestyle='--', label=f'v_rms={v_rms:.1f}')
    ax10.set_xlabel('|v| [km/s]')
    ax10.set_ylabel('PDF')
    ax10.set_title('Velocity distribution', fontsize=14, fontweight='bold')
    ax10.legend(loc='upper right', fontsize=8)
    ax10.set_xlim(0, v_max)

    # Sidebar
    ax_side = fig.add_subplot(gs[:, 5])
    ax_side.axis('off')

    sidebar_text = f"""Parameters
-----------
mu = 19
N+ = {n_plus:,}
N- = {n_minus:,}
Box = {box_size:.0f} Mpc

State
-----------
z ~ {z_approx:.2f}
t = {time_gyr:.3f} Gyr

Density
-----------
rho+_max = {rho_plus_max:.1f}
rho-_max = {rho_minus_max:.1f}

Temperature
-----------
T_mean = {T_mean:.0f} K
T_min = {T_min:.0f} K

Kinematics
-----------
v_rms = {v_rms:.1f} km/s

Structure
-----------
Corr(d+,d-) = {corr:.3f}
N_stars = 0
"""
    ax_side.text(0.1, 0.95, sidebar_text, transform=ax_side.transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Header
    header = (f"Janus Baryonic | mu=19 | T_init=100K | Box={box_size:.0f}Mpc | "
              f"z={z_approx:.3f} | rho+_max={rho_plus_max:.1f} | "
              f"T_min={T_min:.0f} K | step {step}/4000")
    fig.suptitle(header, fontsize=18, fontweight='bold', y=0.98)

    # Save
    output_path = os.path.join(output_dir, f"frame_{step:06d}.png")
    plt.savefig(output_path, dpi=100, facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"[{step:5d}] rho+={rho_plus_max:6.1f} T_min={T_min:6.0f}K v_rms={v_rms:5.1f}km/s")
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/T2/janus-sim/output/janus_baryonic_cold_hires')
    parser.add_argument('--output', default=None)
    parser.add_argument('--box', type=float, default=100.0)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output or os.path.join(input_dir, 'frames_10panel')
    os.makedirs(output_dir, exist_ok=True)

    # Find all snapshots
    snap_dir = os.path.join(input_dir, 'snapshots')
    snapshots = sorted(glob.glob(os.path.join(snap_dir, 'snap_*.bin')))

    print(f"Found {len(snapshots)} snapshots")
    print(f"Output: {output_dir}")
    print(f"Box size: {args.box} Mpc")
    print()

    # Render all
    for snap in snapshots:
        render_frame(snap, output_dir, args.box)

    print(f"\nDone! Rendered {len(snapshots)} frames to {output_dir}")

if __name__ == '__main__':
    main()
