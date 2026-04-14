#!/usr/bin/env python3
"""
Daemon to render new Zoom-L1 frames as snapshots arrive.
Runs both 9-panel and 3D renderers.
"""

import time
import os
from pathlib import Path
import subprocess
import sys

OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
FRAME_DIR_9P = f"{OUTPUT_DIR}/frames"
FRAME_DIR_3D = f"{OUTPUT_DIR}/frames_3d"

SCRIPT_9P = "/mnt/T2/janus-sim/scripts/render_zoom_L1_9panel.py"
SCRIPT_3D = "/mnt/T2/janus-sim/scripts/render_zoom_L1_3d.py"


def main():
    os.makedirs(FRAME_DIR_9P, exist_ok=True)
    os.makedirs(FRAME_DIR_3D, exist_ok=True)

    rendered_9p = set()
    rendered_3d = set()

    # Pre-populate with existing frames
    for f in Path(FRAME_DIR_9P).glob('frame_*.png'):
        step = int(f.stem.split('_')[1])
        rendered_9p.add(step)

    for f in Path(FRAME_DIR_3D).glob('frame_*.png'):
        step = int(f.stem.split('_')[1])
        rendered_3d.add(step)

    print(f"Daemon started. Already rendered: {len(rendered_9p)} 9-panel, {len(rendered_3d)} 3D")
    print("Checking every 5 minutes for new snapshots...")
    sys.stdout.flush()

    while True:
        snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))

        for snap in snaps:
            step = int(snap.stem.split('_')[1])

            # Render 9-panel if needed
            if step not in rendered_9p:
                out_path = f"{FRAME_DIR_9P}/frame_{step:05d}.png"
                if not os.path.exists(out_path):
                    try:
                        subprocess.run([
                            '/tmp/plotenv/bin/python', SCRIPT_9P,
                            '--step', str(step)
                        ], env={**os.environ, 'MPLBACKEND': 'Agg'},
                        capture_output=True, timeout=120)
                        print(f"9-panel: step {step}")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"9-panel error {step}: {e}")
                rendered_9p.add(step)

            # Render 3D if needed
            if step not in rendered_3d:
                out_path = f"{FRAME_DIR_3D}/frame_{step:05d}.png"
                if not os.path.exists(out_path):
                    try:
                        subprocess.run([
                            '/tmp/plotenv/bin/python', SCRIPT_3D,
                            '--step', str(step)
                        ], env={**os.environ, 'MPLBACKEND': 'Agg'},
                        capture_output=True, timeout=120)
                        print(f"3D: step {step}")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"3D error {step}: {e}")
                rendered_3d.add(step)

        time.sleep(300)  # 5 minutes


if __name__ == '__main__':
    main()
