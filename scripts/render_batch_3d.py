#!/usr/bin/env python3
"""
render_batch_3d.py — Batch 3D rendering for Janus simulation
Usage: python render_batch_3d.py --run_dir DIR [--test N] [--workers W] [--subsample K]
"""
import os
import sys
import glob
import argparse
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_snapshots(run_dir):
    """Find all snapshots and extract step/redshift from time_series.csv."""
    snap_dir = os.path.join(run_dir, 'snapshots')
    pattern = os.path.join(snap_dir, 'snap_*.bin')
    snaps = sorted(glob.glob(pattern))

    if not snaps:
        print(f"No snapshots found in {snap_dir}")
        return []

    # Load time series for z values
    ts_path = os.path.join(run_dir, 'time_series.csv')
    step_to_z = {}
    if os.path.exists(ts_path):
        with open(ts_path, 'r') as f:
            header = f.readline().strip().split(',')
            step_idx = header.index('step') if 'step' in header else 0
            z_idx = header.index('z') if 'z' in header else None
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > max(step_idx, z_idx or 0):
                    step = int(parts[step_idx])
                    z = float(parts[z_idx]) if z_idx else 0.0
                    step_to_z[step] = z

    result = []
    for snap in snaps:
        fname = os.path.basename(snap)
        # Extract step from snap_NNNNNN.bin
        step = int(fname.replace('snap_', '').replace('.bin', ''))
        z = step_to_z.get(step, 0.0)
        result.append({'path': snap, 'step': step, 'z': z})

    return result


def render_frame(args):
    """Render a single frame (worker function)."""
    snap, frame_idx, total_frames, step, z, out_dir, subsample, n_rotations = args

    cmd = [
        sys.executable, '/mnt/T2/janus-sim/scripts/render_3d_frame.py',
        '--snap', snap,
        '--frame_idx', str(frame_idx),
        '--total_frames', str(total_frames),
        '--step', str(step),
        '--z', str(z),
        '--out_dir', out_dir,
        '--subsample', str(subsample),
        '--n_rotations', str(n_rotations),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return frame_idx, result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='Run directory with snapshots/')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: run_dir/frames_3d)')
    parser.add_argument('--test', type=int, default=0,
                        help='Test mode: render N frames uniformly sampled')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--subsample', type=int, default=0,
                        help='Subsample particles (0=all)')
    parser.add_argument('--n_rotations', type=float, default=2.0,
                        help='Number of camera rotations')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Start step (inclusive)')
    parser.add_argument('--end_step', type=int, default=999999,
                        help='End step (inclusive)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, 'frames_3d')
    os.makedirs(out_dir, exist_ok=True)

    # Get all snapshots
    snapshots = get_snapshots(args.run_dir)
    if not snapshots:
        return

    # Filter by step range
    snapshots = [s for s in snapshots if args.start_step <= s['step'] <= args.end_step]
    if not snapshots:
        print(f"No snapshots in range [{args.start_step}, {args.end_step}]")
        return

    print(f"Found {len(snapshots)} snapshots in {args.run_dir}")
    print(f"Steps: {snapshots[0]['step']} to {snapshots[-1]['step']}")
    print(f"Redshift: z={snapshots[0]['z']:.3f} to z={snapshots[-1]['z']:.3f}")

    # Test mode: sample N frames uniformly
    if args.test > 0:
        import numpy as np
        indices = np.linspace(0, len(snapshots)-1, args.test, dtype=int)
        snapshots = [snapshots[i] for i in indices]
        print(f"Test mode: rendering {len(snapshots)} frames")

    total_frames = len(snapshots)

    # Build task list
    tasks = []
    for frame_idx, snap_info in enumerate(snapshots):
        tasks.append((
            snap_info['path'],
            frame_idx,
            total_frames,
            snap_info['step'],
            snap_info['z'],
            out_dir,
            args.subsample,
            args.n_rotations,
        ))

    # Run in parallel
    print(f"\nRendering {total_frames} frames with {args.workers} workers...")
    print(f"Output: {out_dir}\n")

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_frame, t): t[1] for t in tasks}

        for future in as_completed(futures):
            frame_idx = futures[future]
            try:
                idx, code, stdout, stderr = future.result()
                if code == 0:
                    completed += 1
                    # Print progress inline
                    print(stdout.strip())
                else:
                    failed += 1
                    print(f"FAILED frame {idx}: {stderr}")
            except Exception as e:
                failed += 1
                print(f"ERROR frame {frame_idx}: {e}")

    print(f"\n{'='*60}")
    print(f"Done: {completed} rendered, {failed} failed")
    print(f"Frames: {out_dir}")

    # Save metadata
    meta = {
        'run_dir': args.run_dir,
        'total_frames': total_frames,
        'test_mode': args.test > 0,
        'subsample': args.subsample,
        'n_rotations': args.n_rotations,
        'snapshots': [{'step': s['step'], 'z': s['z']} for s in snapshots],
    }
    with open(os.path.join(out_dir, 'render_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
