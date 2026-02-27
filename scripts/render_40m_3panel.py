#!/usr/bin/env python3
"""
Render 40M snapshots as 3-panel visualization:
  ┌─────────────────────────┬───────────┐
  │                         │  Masses+  │
  │   Vue isométrique 2.5D  │  (bleu)   │
  │   azimuth=30°, elev=20° ├───────────┤
  │                         │  Masses−  │
  │                         │  (rouge)  │
  └─────────────────────────┴───────────┘

Format 40M: N (u64) + N × (f32, f32, f32, i8) = 8 + N×13 bytes
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Parameters
SNAPSHOT_DIR = Path("/mnt/T2/janus-sim/output/40M_hubble_2026-02-27/snapshots")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/40M_hubble_2026-02-27/frames")
BOX_SIZE = 736.8
SAMPLE_SIZE = 200_000
PARTICLE_SIZE = 13


def read_snapshot_sampled(filepath, sample_size=200_000):
    """Read snapshot with uniform sampling for visualization"""
    with open(filepath, 'rb') as f:
        n_total = struct.unpack('<Q', f.read(8))[0]
        sample_every = max(1, n_total // sample_size)

        positions = []
        signs = []
        chunk_size = 100_000
        particle_idx = 0
        sampled = 0

        while particle_idx < n_total and sampled < sample_size:
            to_read = min(chunk_size, n_total - particle_idx)
            data = f.read(to_read * PARTICLE_SIZE)
            if len(data) < to_read * PARTICLE_SIZE:
                break

            chunk_data = np.frombuffer(data, dtype=np.uint8).reshape(-1, PARTICLE_SIZE)
            pos_bytes = chunk_data[:, :12].tobytes()
            pos = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            sgn = chunk_data[:, 12].astype(np.int8)

            for i in range(to_read):
                if (particle_idx + i) % sample_every == 0 and sampled < sample_size:
                    positions.append(pos[i])
                    signs.append(sgn[i])
                    sampled += 1

            particle_idx += to_read

    return np.array(positions), np.array(signs), n_total


def render_3panel(positions, signs, step, z, H, seg, n_total, output_path):
    """Render 3-panel: main isometric + 2 side panels"""

    mask_pos = signs > 0
    mask_neg = signs < 0

    pos_plus = positions[mask_pos]
    pos_minus = positions[mask_neg]
    n_plus = mask_pos.sum()
    n_minus = mask_neg.sum()

    # Create figure 1920x1080
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='black')

    # Layout
    ax_main = fig.add_axes([0.02, 0.08, 0.62, 0.84], projection='3d', facecolor='black')
    ax_plus = fig.add_axes([0.66, 0.52, 0.32, 0.40], facecolor='black')
    ax_minus = fig.add_axes([0.66, 0.08, 0.32, 0.40], facecolor='black')

    # Colors
    color_plus = '#4488ff'
    color_minus = '#ff4444'

    # === MAIN PANEL: Isometric 3D view ===
    # Subsample for 3D (too slow otherwise)
    max_3d = 50000
    if len(pos_plus) > max_3d // 2:
        idx = np.random.choice(len(pos_plus), max_3d // 2, replace=False)
        pp_3d = pos_plus[idx]
    else:
        pp_3d = pos_plus

    if len(pos_minus) > max_3d // 2:
        idx = np.random.choice(len(pos_minus), max_3d // 2, replace=False)
        pm_3d = pos_minus[idx]
    else:
        pm_3d = pos_minus

    ax_main.scatter(pm_3d[:, 0], pm_3d[:, 1], pm_3d[:, 2],
                    c=color_minus, s=0.3, alpha=0.4, marker='.', rasterized=True)
    ax_main.scatter(pp_3d[:, 0], pp_3d[:, 1], pp_3d[:, 2],
                    c=color_plus, s=0.3, alpha=0.4, marker='.', rasterized=True)

    # Set view angle
    ax_main.view_init(elev=20, azim=30)

    # Set limits
    half = BOX_SIZE / 2
    ax_main.set_xlim(-half, half)
    ax_main.set_ylim(-half, half)
    ax_main.set_zlim(-half, half)

    # Style
    ax_main.set_xlabel('X [Mpc]', color='white', fontsize=10)
    ax_main.set_ylabel('Y [Mpc]', color='white', fontsize=10)
    ax_main.set_zlabel('Z [Mpc]', color='white', fontsize=10)
    ax_main.tick_params(colors='white', labelsize=8)
    ax_main.xaxis.pane.fill = False
    ax_main.yaxis.pane.fill = False
    ax_main.zaxis.pane.fill = False
    ax_main.xaxis.pane.set_edgecolor('gray')
    ax_main.yaxis.pane.set_edgecolor('gray')
    ax_main.zaxis.pane.set_edgecolor('gray')
    ax_main.grid(True, alpha=0.3, color='gray')

    # === RIGHT TOP: Masses+ (XY projection) ===
    ax_plus.scatter(pos_plus[:, 0], pos_plus[:, 1],
                    c=color_plus, s=0.05, alpha=0.4, marker='.', rasterized=True)
    ax_plus.set_xlim(-half, half)
    ax_plus.set_ylim(-half, half)
    ax_plus.set_aspect('equal')
    ax_plus.set_facecolor('black')
    ax_plus.axis('off')
    ax_plus.set_title(f'Masses+ ({n_plus:,} sampled)', color=color_plus, fontsize=12, pad=5)

    # === RIGHT BOTTOM: Masses- (XY projection) ===
    ax_minus.scatter(pos_minus[:, 0], pos_minus[:, 1],
                     c=color_minus, s=0.05, alpha=0.4, marker='.', rasterized=True)
    ax_minus.set_xlim(-half, half)
    ax_minus.set_ylim(-half, half)
    ax_minus.set_aspect('equal')
    ax_minus.set_facecolor('black')
    ax_minus.axis('off')
    ax_minus.set_title(f'Masses− ({n_minus:,} sampled)', color=color_minus, fontsize=12, pad=5)

    # === Title ===
    title = f"Janus Cosmological Model — 40M particles | η = 1.045"
    fig.text(0.5, 0.97, title, ha='center', va='top', fontsize=22,
             color='white', fontweight='bold')

    # === Stats bar ===
    stats = f"Step {step:04d} | z = {z:.3f} | H = {H:.4f} H₀ | Seg = {seg:.4f} Mpc"
    fig.text(0.5, 0.02, stats, ha='center', va='bottom', fontsize=16,
             color='white', family='monospace')

    # Legend
    fig.text(0.02, 0.97, "● m+ (attracts m+, repels m−)", color=color_plus, fontsize=11, va='top')
    fig.text(0.02, 0.94, "● m− (attracts m−, repels m+)", color=color_minus, fontsize=11, va='top')

    plt.savefig(output_path, facecolor='black', dpi=100)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = sorted(SNAPSHOT_DIR.glob("snapshot_*.bin"))
    print(f"Found {len(snapshots)} snapshots")

    if not snapshots:
        print("No snapshots found!")
        return

    # Read CSV for metadata
    csv_path = SNAPSHOT_DIR.parent / "time_series.csv"
    metadata = {}
    if csv_path.exists():
        with open(csv_path) as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    step = int(parts[0])
                    z = float(parts[2])
                    H = float(parts[4])
                    seg = float(parts[5])
                    metadata[step] = {'z': z, 'H': H, 'seg': seg}

    for i, snap_path in enumerate(snapshots):
        step = int(snap_path.stem.split('_')[1])
        output_path = OUTPUT_DIR / f"frame_{step:05d}.png"

        if output_path.exists():
            print(f"[{i+1}/{len(snapshots)}] Skip {snap_path.name} (exists)")
            continue

        print(f"[{i+1}/{len(snapshots)}] Rendering {snap_path.name}...")

        meta = metadata.get(step, metadata.get(1, {'z': 5.0, 'H': 2.4, 'seg': 0.063}))

        positions, signs, n_total = read_snapshot_sampled(snap_path, SAMPLE_SIZE)
        render_3panel(positions, signs, step, meta['z'], meta['H'], meta['seg'], n_total, output_path)

        print(f"    Saved: {output_path}")

    print(f"\nDone! Frames: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
