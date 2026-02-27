#!/usr/bin/env python3
"""
Render frames from 40M publication snapshots.
Format: N (u64) + N × (f32, f32, f32, i8) = 8 + N×13 bytes
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Parameters
SNAPSHOT_DIR = Path("/mnt/T2/janus-sim/output/40M_hubble_2026-02-27/snapshots")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/40M_hubble_2026-02-27/frames")
BOX_SIZE = 736.8
SAMPLE_SIZE = 200_000  # For visualization
PARTICLE_SIZE = 13

def read_snapshot_sampled(filepath, sample_size=200_000):
    """Read snapshot with uniform sampling for visualization"""
    with open(filepath, 'rb') as f:
        n_total = struct.unpack('<Q', f.read(8))[0]

        # Calculate sampling
        sample_every = max(1, n_total // sample_size)

        positions = []
        signs = []

        # Read in chunks
        chunk_size = 100_000
        particle_idx = 0
        sampled = 0

        while particle_idx < n_total and sampled < sample_size:
            to_read = min(chunk_size, n_total - particle_idx)
            data = f.read(to_read * PARTICLE_SIZE)

            if len(data) < to_read * PARTICLE_SIZE:
                break

            # Parse with numpy
            chunk_data = np.frombuffer(data, dtype=np.uint8).reshape(-1, PARTICLE_SIZE)
            pos_bytes = chunk_data[:, :12].tobytes()
            pos = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            sgn = chunk_data[:, 12].astype(np.int8)

            # Sample
            for i in range(to_read):
                if (particle_idx + i) % sample_every == 0 and sampled < sample_size:
                    positions.append(pos[i])
                    signs.append(sgn[i])
                    sampled += 1

            particle_idx += to_read

    return np.array(positions), np.array(signs), n_total


def render_frame(positions, signs, step, seg, z, output_path):
    """Render a single frame - 2D XY projection"""

    mask_pos = signs > 0
    mask_neg = signs < 0

    fig, ax = plt.subplots(figsize=(16, 16), dpi=120, facecolor='black')
    ax.set_facecolor('black')

    # Plot particles
    ax.scatter(positions[mask_pos, 0], positions[mask_pos, 1],
               c='#4444ff', s=0.05, alpha=0.4, marker='.', rasterized=True)
    ax.scatter(positions[mask_neg, 0], positions[mask_neg, 1],
               c='#ff4444', s=0.05, alpha=0.4, marker='.', rasterized=True)

    # Axes
    ax.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    title = f"Janus Cosmological Model — 40M particles"
    ax.text(0.5, 0.98, title, transform=ax.transAxes, ha='center', va='top',
            fontsize=24, color='white', fontweight='bold')

    # Stats
    stats = f"Step {step:04d} | z = {z:.2f} | Seg = {seg:.4f} Mpc"
    ax.text(0.5, 0.02, stats, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=18, color='white', family='monospace')

    # Legend
    ax.text(0.02, 0.98, "● Positive mass (+1)", transform=ax.transAxes,
            ha='left', va='top', fontsize=14, color='#6666ff')
    ax.text(0.02, 0.94, "● Negative mass (-1)", transform=ax.transAxes,
            ha='left', va='top', fontsize=14, color='#ff6666')

    plt.savefig(output_path, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all snapshots
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
                    # CSV: step,time,z,a,H,seg,ke_ratio,step_time_ms
                    seg = float(parts[5])
                    metadata[step] = {'z': z, 'seg': seg}

    # Render each snapshot
    for i, snap_path in enumerate(snapshots):
        # Extract step number from filename
        step = int(snap_path.stem.split('_')[1])

        output_path = OUTPUT_DIR / f"frame_{step:05d}.png"

        if output_path.exists():
            print(f"[{i+1}/{len(snapshots)}] Skip {snap_path.name} (already rendered)")
            continue

        print(f"[{i+1}/{len(snapshots)}] Rendering {snap_path.name}...")

        # Get metadata
        meta = metadata.get(step, {'z': 5.0, 'seg': 0.0})
        if step == 0:
            meta = metadata.get(1, {'z': 5.0, 'seg': 0.063})

        # Read and render
        positions, signs, n_total = read_snapshot_sampled(snap_path, SAMPLE_SIZE)
        render_frame(positions, signs, step, meta['seg'], meta['z'], output_path)

        print(f"    Saved: {output_path}")

    print(f"\nDone! Frames saved to {OUTPUT_DIR}")
    print(f"\nTo create video:")
    print(f"ffmpeg -framerate 30 -i {OUTPUT_DIR}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 janus_40m.mp4")


if __name__ == "__main__":
    main()
