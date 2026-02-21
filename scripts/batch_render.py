#!/usr/bin/env python3
"""
Batch render Janus N-body snapshots to 3-panel frames
"""
import os
import sys
import glob
import struct
import subprocess
from multiprocessing import Pool, cpu_count

def get_snapshot_metadata(path):
    """Read metadata from snapshot header"""
    with open(path, 'rb') as f:
        n_particles = struct.unpack('<Q', f.read(8))[0]
        n_positive = struct.unpack('<Q', f.read(8))[0]
        eta = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        sim_time = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
    return {
        'n_particles': n_particles,
        'eta': eta,
        'step': step,
        'time': sim_time,
        'seg': seg,
        'ke_ratio': ke_ratio
    }

def render_single(args):
    """Render a single frame"""
    snap_path, output_path, script_path = args

    try:
        meta = get_snapshot_metadata(snap_path)
        cmd = [
            sys.executable, script_path,
            snap_path, output_path,
            str(meta['eta']),
            str(meta['n_particles']),
            str(meta['step']),
            str(meta['time']),
            str(meta['seg']),
            str(meta['ke_ratio'])
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Error rendering {snap_path}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception rendering {snap_path}: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: batch_render.py <snapshot_dir> <output_dir> [step_interval]")
        sys.exit(1)

    snap_dir = sys.argv[1]
    output_dir = sys.argv[2]
    step_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    os.makedirs(output_dir, exist_ok=True)

    script_path = os.path.join(os.path.dirname(__file__), 'render_overnight.py')

    # Get all snapshots
    snapshots = sorted(glob.glob(os.path.join(snap_dir, 'snap_*.bin')))
    print(f"Found {len(snapshots)} snapshots")

    # Select every Nth snapshot
    selected = snapshots[::step_interval]
    print(f"Rendering {len(selected)} frames (every {step_interval}th)")

    # Prepare render tasks
    tasks = []
    for i, snap_path in enumerate(selected):
        frame_name = f"frame_{i:05d}.png"
        output_path = os.path.join(output_dir, frame_name)
        tasks.append((snap_path, output_path, script_path))

    # Parallel rendering
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers")

    with Pool(n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap(render_single, tasks)):
            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                print(f"Progress: {i+1}/{len(tasks)} ({100*(i+1)/len(tasks):.1f}%)")
            results.append(result)

    success = sum(results)
    print(f"\nCompleted: {success}/{len(tasks)} frames rendered successfully")

    if success == len(tasks):
        print(f"\nTo create video, run:")
        print(f"ffmpeg -framerate 30 -i {output_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4")

if __name__ == "__main__":
    main()
