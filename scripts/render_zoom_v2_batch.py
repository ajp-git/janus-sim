#!/usr/bin/env python3
"""
Batch render all Zoom-L1 frames (black + white) and create videos.
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool
import os

OUTPUT_DIR = "/mnt/T2/janus-sim/output/janus_zoom_L1_baryonic"
SNAP_DIR = f"{OUTPUT_DIR}/snapshots"
FRAME_DIR_BLACK = f"{OUTPUT_DIR}/frames_v2_black"
FRAME_DIR_WHITE = f"{OUTPUT_DIR}/frames_v2_white"
RENDERER = "/mnt/T2/janus-sim/scripts/zoom_l1_renderer_v2.py"

os.makedirs(FRAME_DIR_BLACK, exist_ok=True)
os.makedirs(FRAME_DIR_WHITE, exist_ok=True)

def render_frame(args):
    step, white_bg = args
    suffix = 'white' if white_bg else 'black'
    frame_dir = FRAME_DIR_WHITE if white_bg else FRAME_DIR_BLACK
    out_path = f"{frame_dir}/frame_{step:05d}.png"

    if os.path.exists(out_path):
        return f"[SKIP] {suffix} {step}"

    try:
        cmd = ['/tmp/plotenv/bin/python', RENDERER, '--step', str(step)]
        if white_bg:
            cmd.append('--white-bg')
        result = subprocess.run(
            cmd, env={**os.environ, 'MPLBACKEND': 'Agg'},
            capture_output=True, timeout=180
        )

        # Move to correct directory
        src = f"{OUTPUT_DIR}/frame_{step:05d}_{suffix}.png"
        if os.path.exists(src):
            os.rename(src, out_path)
            return f"[OK] {suffix} {step}"
        else:
            return f"[FAIL] {suffix} {step} - no output"
    except Exception as e:
        return f"[ERR] {suffix} {step}: {e}"

def main():
    # Get all snapshots
    snaps = sorted(Path(SNAP_DIR).glob('snap_*.bin'))
    steps = [int(s.stem.split('_')[1]) for s in snaps]

    print(f"Found {len(steps)} snapshots")
    print(f"Rendering to {FRAME_DIR_BLACK} and {FRAME_DIR_WHITE}")
    sys.stdout.flush()

    # Build task list: all steps x both modes
    tasks = []
    for step in steps:
        tasks.append((step, False))  # black
        tasks.append((step, True))   # white

    print(f"Total tasks: {len(tasks)}")
    sys.stdout.flush()

    # Render with 4 parallel processes
    with Pool(4) as pool:
        for i, result in enumerate(pool.imap_unordered(render_frame, tasks)):
            if i % 50 == 0:
                print(f"[{i}/{len(tasks)}] {result}")
                sys.stdout.flush()

    print("\n=== Rendering complete, creating videos ===")
    sys.stdout.flush()

    # Create black video
    print("Creating black background video...")
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '30',
        '-pattern_type', 'glob',
        '-i', f'{FRAME_DIR_BLACK}/frame_*.png',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        f'{OUTPUT_DIR}/janus_zoom_L1_v2_black_4K.mp4'
    ])

    # Create white video
    print("Creating white background video...")
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '30',
        '-pattern_type', 'glob',
        '-i', f'{FRAME_DIR_WHITE}/frame_*.png',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        f'{OUTPUT_DIR}/janus_zoom_L1_v2_white_4K.mp4'
    ])

    print("\n=== DONE ===")
    print(f"Black video: {OUTPUT_DIR}/janus_zoom_L1_v2_black_4K.mp4")
    print(f"White video: {OUTPUT_DIR}/janus_zoom_L1_v2_white_4K.mp4")

if __name__ == '__main__':
    main()
