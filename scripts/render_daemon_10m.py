#!/usr/bin/env python3
"""
JANUS 10M Render Daemon
=======================
Monitors simulation progress and renders frames at key redshifts.
After z=0, generates zoom series centered on dominant halo.
"""

import time
import json
from pathlib import Path
import subprocess
import sys

# Configuration
SNAP_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/snapshots")
FRAME_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/frames_10panel")
CSV_PATH = Path("/mnt/T2/janus-sim/output/run_final_10m/time_series.csv")
RENDERER = "/mnt/T2/janus-sim/scripts/render_10panel_4k.py"
PYTHON = "/tmp/plotenv/bin/python"
BOX_SIZE = 300.0

# Key redshifts to render
KEY_REDSHIFTS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0]

def get_latest_csv_row():
    """Get latest step info from CSV"""
    if not CSV_PATH.exists():
        return None
    try:
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        header = lines[0].strip().split(',')
        last = lines[-1].strip().split(',')
        return dict(zip(header, last))
    except:
        return None

def find_snapshot_for_z(target_z, tolerance=0.05):
    """Find snapshot closest to target redshift"""
    if not CSV_PATH.exists():
        return None

    best_snap = None
    best_diff = float('inf')

    try:
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            step = int(parts[0])
            z = float(parts[3])
            diff = abs(z - target_z)

            if diff < best_diff and diff < tolerance:
                snap_path = SNAP_DIR / f"snap_{step:05d}.bin"
                if snap_path.exists():
                    best_snap = snap_path
                    best_diff = diff
    except:
        pass

    return best_snap

def render_frame(snap_path, out_path):
    """Render a single frame"""
    cmd = [PYTHON, RENDERER, str(snap_path), str(out_path), "--box", str(BOX_SIZE)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def render_zoom_series(snap_path, out_dir):
    """Render zoom series for final snapshot"""
    cmd = [PYTHON, RENDERER, str(snap_path), str(out_dir / "zoom.png"), "--box", str(BOX_SIZE), "--zoom"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JANUS 10M Render Daemon")
    print("=" * 60)
    print(f"Monitoring: {SNAP_DIR}")
    print(f"Output: {FRAME_DIR}")
    print(f"Key redshifts: {KEY_REDSHIFTS}")
    print()

    rendered_z = set()
    final_rendered = False

    while True:
        info = get_latest_csv_row()
        if info is None:
            print("Waiting for simulation to start...")
            time.sleep(30)
            continue

        current_step = int(info['step'])
        current_z = float(info['z'])

        print(f"[Step {current_step}] z = {current_z:.4f}")

        # Check for key redshifts to render
        for target_z in KEY_REDSHIFTS:
            if target_z in rendered_z:
                continue

            # Only render if we've passed this redshift
            if current_z <= target_z + 0.05:
                snap_path = find_snapshot_for_z(target_z)
                if snap_path:
                    out_path = FRAME_DIR / f"frame_z{target_z:.1f}.png"
                    if not out_path.exists():
                        print(f"  → Rendering z={target_z:.1f} from {snap_path.name}...")
                        if render_frame(snap_path, out_path):
                            rendered_z.add(target_z)
                            print(f"    ✓ Saved: {out_path}")
                        else:
                            print(f"    ✗ Failed to render")

        # Check if simulation complete (z ≈ 0)
        if current_z < 0.01 and not final_rendered:
            print("\n" + "=" * 60)
            print("SIMULATION COMPLETE - Generating final outputs...")
            print("=" * 60)

            # Find final snapshot
            final_snap = find_snapshot_for_z(0.0, tolerance=0.1)
            if final_snap:
                # Render final frame
                out_path = FRAME_DIR / "frame_final_z0.png"
                print(f"Rendering final frame...")
                render_frame(final_snap, out_path)

                # Generate zoom series
                print(f"\nGenerating zoom series centered on dominant halo...")
                render_zoom_series(final_snap, FRAME_DIR)

                final_rendered = True
                print("\n✓ All frames rendered!")
                print(f"  Frames: {FRAME_DIR}")
                break

        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
