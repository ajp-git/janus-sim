#!/usr/bin/env python3
"""
Wait for simulation to reach z=0, then generate zoom series for validation.
"""
import time
import subprocess
from pathlib import Path

CSV_PATH = Path("/mnt/T2/janus-sim/output/run_final_10m/time_series.csv")
SNAP_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/snapshots")
FRAME_DIR = Path("/mnt/T2/janus-sim/output/run_final_10m/frames_10panel")
RENDERER = "/mnt/T2/janus-sim/scripts/render_10panel_4k.py"
PYTHON = "/tmp/plotenv/bin/python"

def get_current_z():
    if not CSV_PATH.exists():
        return None
    try:
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        last = lines[-1].strip().split(',')
        return float(last[3]), int(last[0])
    except:
        return None

def main():
    print("=" * 60)
    print("Waiting for simulation to reach z=0...")
    print("=" * 60)

    while True:
        result = get_current_z()
        if result is None:
            time.sleep(60)
            continue

        z, step = result
        print(f"[Step {step}] z = {z:.4f}")

        if z < 0.01:
            print("\n" + "=" * 60)
            print("z ≈ 0 REACHED - Generating zoom series...")
            print("=" * 60)

            # Find final snapshot
            snaps = sorted(SNAP_DIR.glob("snap_*.bin"))
            if snaps:
                final_snap = snaps[-1]
                print(f"Final snapshot: {final_snap}")

                # Generate zoom series
                cmd = [PYTHON, RENDERER, str(final_snap),
                       str(FRAME_DIR / "zoom.png"), "--box", "300", "--zoom"]
                subprocess.run(cmd)

                print("\n" + "=" * 60)
                print("ZOOM SERIES GENERATED - Ready for validation")
                print(f"Output: {FRAME_DIR}")
                print("=" * 60)
            break

        time.sleep(120)

if __name__ == "__main__":
    main()
