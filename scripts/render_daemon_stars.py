#!/usr/bin/env python3
"""
render_daemon_stars.py - Auto-render daemon for VSL simulation with stars

Monitors snapshots directory and renders frames as they appear.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Configuration
SNAP_DIR = "/mnt/T2/janus-sim/output/janus_vsl_1000mpc_10M_stars/snapshots"
FRAME_DIR = "/mnt/T2/janus-sim/output/janus_vsl_1000mpc_10M_stars/frames"
RENDER_SCRIPT = "/mnt/T2/janus-sim/scripts/render_stars.py"
POLL_INTERVAL = 10  # seconds

# Track rendered files
rendered = set()


def render_snapshot(snap_path):
    """Render a single snapshot"""
    snap_name = Path(snap_path).stem
    frame_path = Path(FRAME_DIR) / f"{snap_name}.png"

    if frame_path.exists():
        return False

    print(f"Rendering {snap_name}...")
    try:
        result = subprocess.run(
            [sys.executable, RENDER_SCRIPT, str(snap_path), str(frame_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print(f"  ✓ {frame_path.name}")
            return True
        else:
            print(f"  ✗ Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout rendering {snap_name}")
        return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def scan_and_render():
    """Scan for new snapshots and render them"""
    snap_dir = Path(SNAP_DIR)
    if not snap_dir.exists():
        return 0

    snapshots = sorted(snap_dir.glob("snap_*.bin"))
    rendered_count = 0

    for snap in snapshots:
        if str(snap) not in rendered:
            if render_snapshot(snap):
                rendered.add(str(snap))
                rendered_count += 1

    return rendered_count


def main():
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Render Daemon - VSL 1000 Mpc 10M with Stars                 ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Snapshots: {SNAP_DIR}")
    print(f"║  Frames:    {FRAME_DIR}")
    print(f"║  Poll:      every {POLL_INTERVAL}s")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    # Create frame directory
    Path(FRAME_DIR).mkdir(parents=True, exist_ok=True)

    # Load already rendered
    for frame in Path(FRAME_DIR).glob("*.png"):
        snap_name = frame.stem
        snap_path = Path(SNAP_DIR) / f"{snap_name}.bin"
        rendered.add(str(snap_path))

    print(f"Already rendered: {len(rendered)} frames")
    print(f"Watching for new snapshots...\n")

    total_rendered = 0
    while True:
        try:
            count = scan_and_render()
            total_rendered += count

            # Status every 10 polls
            n_snaps = len(list(Path(SNAP_DIR).glob("snap_*.bin"))) if Path(SNAP_DIR).exists() else 0
            n_frames = len(list(Path(FRAME_DIR).glob("*.png")))

            if count > 0:
                print(f"[{time.strftime('%H:%M:%S')}] Rendered {count} new frames. Total: {n_frames}/{n_snaps}")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print(f"\nStopped. Total rendered: {total_rendered}")
            break


if __name__ == "__main__":
    main()
