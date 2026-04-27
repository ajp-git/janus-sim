#!/usr/bin/env python3
"""
Render daemon for VSL Bimetric simulation frames.
Watches snapshot directory and renders frames as they appear.
"""

import os
import sys
import struct
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

# 4K resolution
FIG_WIDTH = 3840
FIG_HEIGHT = 2160
DPI = 100

def read_snapshot(path):
    """Read JSNP v2 snapshot format."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'JSNP':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        z = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]

        positions = np.zeros((n, 3), dtype=np.float64)
        signs = np.zeros(n, dtype=np.int8)

        for i in range(n):
            px, py, pz = struct.unpack('<ddd', f.read(24))
            sign = struct.unpack('<b', f.read(1))[0]
            ptype = struct.unpack('<B', f.read(1))[0]
            positions[i] = [px, py, pz]
            signs[i] = sign

        return positions, signs, z, box_size, n

def render_frame(snap_path, frame_path, step):
    """Render a single frame from snapshot."""
    try:
        pos, signs, z, box_size, n = read_snapshot(snap_path)

        # Separate populations
        mask_plus = signs > 0
        mask_minus = signs < 0

        pos_plus = pos[mask_plus]
        pos_minus = pos[mask_minus]

        # Create figure with 3 panels
        fig = plt.figure(figsize=(FIG_WIDTH/DPI, FIG_HEIGHT/DPI), dpi=DPI, facecolor='black')

        # XY projection (main panel - left)
        ax1 = fig.add_axes([0.02, 0.1, 0.45, 0.85])
        ax1.set_facecolor('black')

        # Subsample for performance (max 50k per population)
        n_sample = min(50000, len(pos_plus))
        if len(pos_plus) > n_sample:
            idx_p = np.random.choice(len(pos_plus), n_sample, replace=False)
            idx_m = np.random.choice(len(pos_minus), min(n_sample, len(pos_minus)), replace=False)
        else:
            idx_p = np.arange(len(pos_plus))
            idx_m = np.arange(len(pos_minus))

        ax1.scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 1], s=0.1, c='cyan', alpha=0.3, rasterized=True)
        ax1.scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 1], s=0.1, c='red', alpha=0.3, rasterized=True)

        ax1.set_xlim(0, box_size)
        ax1.set_ylim(0, box_size)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X [Mpc]', color='white', fontsize=14)
        ax1.set_ylabel('Y [Mpc]', color='white', fontsize=14)
        ax1.tick_params(colors='white')
        ax1.set_title(f'XY Projection (z={z:.3f})', color='white', fontsize=16)

        # XZ projection (top right)
        ax2 = fig.add_axes([0.52, 0.55, 0.22, 0.4])
        ax2.set_facecolor('black')
        ax2.scatter(pos_plus[idx_p, 0], pos_plus[idx_p, 2], s=0.05, c='cyan', alpha=0.2, rasterized=True)
        ax2.scatter(pos_minus[idx_m, 0], pos_minus[idx_m, 2], s=0.05, c='red', alpha=0.2, rasterized=True)
        ax2.set_xlim(0, box_size)
        ax2.set_ylim(0, box_size)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X [Mpc]', color='white', fontsize=10)
        ax2.set_ylabel('Z [Mpc]', color='white', fontsize=10)
        ax2.tick_params(colors='white', labelsize=8)
        ax2.set_title('XZ', color='white', fontsize=12)

        # YZ projection (bottom right)
        ax3 = fig.add_axes([0.52, 0.1, 0.22, 0.4])
        ax3.set_facecolor('black')
        ax3.scatter(pos_plus[idx_p, 1], pos_plus[idx_p, 2], s=0.05, c='cyan', alpha=0.2, rasterized=True)
        ax3.scatter(pos_minus[idx_m, 1], pos_minus[idx_m, 2], s=0.05, c='red', alpha=0.2, rasterized=True)
        ax3.set_xlim(0, box_size)
        ax3.set_ylim(0, box_size)
        ax3.set_aspect('equal')
        ax3.set_xlabel('Y [Mpc]', color='white', fontsize=10)
        ax3.set_ylabel('Z [Mpc]', color='white', fontsize=10)
        ax3.tick_params(colors='white', labelsize=8)
        ax3.set_title('YZ', color='white', fontsize=12)

        # Info panel (far right)
        ax_info = fig.add_axes([0.78, 0.1, 0.2, 0.85])
        ax_info.set_facecolor('black')
        ax_info.axis('off')

        info_text = f"""VSL Bimetric Simulation

Step: {step}
Redshift: z = {z:.4f}
Scale: a = {1/(1+z):.4f}

N_total: {n:,}
N_m+: {np.sum(mask_plus):,}
N_m-: {np.sum(mask_minus):,}

Box: {box_size:.0f} Mpc

Model:
  SPH: m+ and m-
  VSL: c_ratio(z)
  eta = 1.045
  mu = 19
"""
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=12, color='white', verticalalignment='top',
                    fontfamily='monospace')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='m+ (positive)', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='m- (negative)', linestyle='None'),
        ]
        ax_info.legend(handles=legend_elements, loc='lower left', frameon=False,
                      labelcolor='white', fontsize=10)

        plt.savefig(frame_path, dpi=DPI, facecolor='black', edgecolor='none',
                   bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        return True
    except Exception as e:
        print(f"Error rendering {snap_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Render daemon for bimetric simulation')
    parser.add_argument('--phase', type=int, default=0, help='Phase number (0, 1, or 2)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    phase = args.phase
    base_dir = f"/mnt/T2/janus-sim/output/vsl_bimetric_phase{phase}"
    snap_dir = f"{base_dir}/snapshots"
    frame_dir = f"{base_dir}/frames"

    os.makedirs(frame_dir, exist_ok=True)

    print(f"Render daemon started for Phase {phase}")
    print(f"Watching: {snap_dir}")
    print(f"Output: {frame_dir}")
    print(f"Workers: {args.workers}")

    rendered = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            # Check for result file (simulation complete)
            result_file = f"{base_dir}/phase{phase}_result.txt"
            sim_complete = os.path.exists(result_file)

            # Find new snapshots
            if os.path.exists(snap_dir):
                snaps = sorted(Path(snap_dir).glob("snap_*.bin"))

                for snap_path in snaps:
                    snap_name = snap_path.stem
                    step = int(snap_name.split('_')[1])
                    frame_path = f"{frame_dir}/frame_{step:06d}.png"

                    if snap_name not in rendered and not os.path.exists(frame_path):
                        # Submit render job
                        future = executor.submit(render_frame, str(snap_path), frame_path, step)
                        rendered.add(snap_name)
                        print(f"Rendering step {step}...")

            if sim_complete:
                # Wait for all pending renders
                time.sleep(5)
                print(f"\nSimulation complete. Checking remaining frames...")

                # Final pass
                if os.path.exists(snap_dir):
                    snaps = sorted(Path(snap_dir).glob("snap_*.bin"))
                    pending = []
                    for snap_path in snaps:
                        snap_name = snap_path.stem
                        step = int(snap_name.split('_')[1])
                        frame_path = f"{frame_dir}/frame_{step:06d}.png"
                        if not os.path.exists(frame_path):
                            pending.append((str(snap_path), frame_path, step))

                    if pending:
                        print(f"Rendering {len(pending)} remaining frames...")
                        for snap_path, frame_path, step in pending:
                            render_frame(snap_path, frame_path, step)

                n_frames = len(list(Path(frame_dir).glob("frame_*.png")))
                print(f"\nRender complete: {n_frames} frames in {frame_dir}")
                break

            time.sleep(2)

if __name__ == "__main__":
    main()
