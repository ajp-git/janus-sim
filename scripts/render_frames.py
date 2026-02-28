#!/usr/bin/env python3
"""
Render Janus simulation binary data to PNG frames.

Input:  render_data/step_XXXXXX.bin (60M × 13 bytes: x,y,z f32 + sign i8)
Output: frames/frame_XXXXXX.png

Format: 2048×2048 total, 3 panels (XY | XZ | YZ)
        Black background, blue(+) / red(-) density
"""

import numpy as np
import struct
import sys
import os
from pathlib import Path
import time

# Try to import matplotlib with Agg backend (no display)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def read_render_bin(filepath):
    """Read binary render data file."""
    with open(filepath, 'rb') as f:
        # Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + redshift(f64) + n(u32)
        step = struct.unpack('<I', f.read(4))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
        redshift = struct.unpack('<d', f.read(8))[0]
        n = struct.unpack('<I', f.read(4))[0]

        # Positions: N × 3 × f32
        pos_data = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)

        # Signs: N × i8
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    return {
        'step': step,
        'box_size': box_size,
        'seg': seg,
        'ke_ratio': ke_ratio,
        'redshift': redshift,
        'n': n,
        'pos': pos_data,
        'signs': signs
    }

def create_density_map(pos, signs, box_size, grid_size=1024):
    """Create density histograms for positive and negative particles."""
    half_box = box_size / 2.0

    # Separate positive and negative
    pos_plus = pos[signs > 0]
    pos_minus = pos[signs < 0]

    # Bins for histogram
    bins = np.linspace(-half_box, half_box, grid_size + 1)

    # XY projection
    h_plus_xy, _, _ = np.histogram2d(pos_plus[:, 0], pos_plus[:, 1], bins=[bins, bins])
    h_minus_xy, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 1], bins=[bins, bins])

    # XZ projection
    h_plus_xz, _, _ = np.histogram2d(pos_plus[:, 0], pos_plus[:, 2], bins=[bins, bins])
    h_minus_xz, _, _ = np.histogram2d(pos_minus[:, 0], pos_minus[:, 2], bins=[bins, bins])

    # YZ projection
    h_plus_yz, _, _ = np.histogram2d(pos_plus[:, 1], pos_plus[:, 2], bins=[bins, bins])
    h_minus_yz, _, _ = np.histogram2d(pos_minus[:, 1], pos_minus[:, 2], bins=[bins, bins])

    return {
        'xy': (h_plus_xy, h_minus_xy),
        'xz': (h_plus_xz, h_minus_xz),
        'yz': (h_plus_yz, h_minus_yz)
    }

def render_frame(data, output_path, grid_size=1024):
    """Render a single frame with 3 projections."""
    t0 = time.time()

    # Create density maps
    density = create_density_map(data['pos'], data['signs'], data['box_size'], grid_size)

    # Create figure: 3 panels side by side (2048x768 total)
    fig, axes = plt.subplots(1, 3, figsize=(20.48, 7.68), facecolor='black')
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08, wspace=0.08)

    projections = [('xy', 'X', 'Y'), ('xz', 'X', 'Z'), ('yz', 'Y', 'Z')]

    for ax, (proj, xlabel, ylabel) in zip(axes, projections):
        h_plus, h_minus = density[proj]

        # Use log scale for better visibility
        h_plus_log = np.log10(h_plus + 1)
        h_minus_log = np.log10(h_minus + 1)

        # Compute difference: positive values = more +, negative = more -
        # This shows segregation as blue vs red regions
        h_diff = h_plus_log - h_minus_log
        h_total = h_plus_log + h_minus_log

        # Normalize difference by total density
        h_ratio = np.divide(h_diff, h_total + 1e-10, where=(h_total > 0.1))
        h_ratio = np.clip(h_ratio, -1, 1)

        # Create RGB: blue where ratio > 0 (more +), red where ratio < 0 (more -)
        # Brightness from total density
        brightness = np.clip(h_total / (h_total.max() + 1e-10) * 2, 0, 1)

        rgb = np.zeros((grid_size, grid_size, 3))
        # Red channel: where ratio < 0 (more negative particles)
        rgb[:, :, 0] = brightness.T * np.clip(-h_ratio.T, 0, 1)
        # Blue channel: where ratio > 0 (more positive particles)
        rgb[:, :, 2] = brightness.T * np.clip(h_ratio.T, 0, 1)
        # Add some white/gray for mixed regions
        rgb[:, :, 1] = brightness.T * (1 - np.abs(h_ratio.T)) * 0.3

        # Boost overall brightness
        rgb = np.clip(rgb * 1.5, 0, 1)

        ax.imshow(rgb, origin='lower', extent=[-data['box_size']/2, data['box_size']/2,
                                                -data['box_size']/2, data['box_size']/2])
        ax.set_xlabel(f'{xlabel} (Mpc)', color='white', fontsize=10)
        ax.set_ylabel(f'{ylabel} (Mpc)', color='white', fontsize=10)
        ax.set_title(f'{xlabel}-{ylabel}', color='white', fontsize=12)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_facecolor('black')

    # Main title
    title = f"Janus 60M — Step {data['step']:06d} | z={data['redshift']:.2f} | Seg={data['seg']:.4f}"
    fig.suptitle(title, color='white', fontsize=14, y=0.96)

    # Save
    plt.savefig(output_path, dpi=100, facecolor='black', edgecolor='none')
    plt.close(fig)

    elapsed = time.time() - t0
    return elapsed

def main():
    if len(sys.argv) < 2:
        print("Usage: python render_frames.py <render_data_dir> [output_dir]")
        print("   or: python render_frames.py <single_file.bin> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_file():
        # Single file mode
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent.parent / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Rendering {input_path}...")
        data = read_render_bin(input_path)
        print(f"  Loaded: {data['n']:,} particles, z={data['redshift']:.2f}, Seg={data['seg']:.4f}")

        output_file = output_dir / f"frame_{data['step']:06d}.png"
        elapsed = render_frame(data, output_file)
        print(f"  Saved: {output_file} ({elapsed:.1f}s)")

    elif input_path.is_dir():
        # Directory mode - process all .bin files
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)

        bin_files = sorted(input_path.glob('step_*.bin'))
        print(f"Found {len(bin_files)} files to process")

        for i, bin_file in enumerate(bin_files):
            print(f"[{i+1}/{len(bin_files)}] Rendering {bin_file.name}...")
            data = read_render_bin(bin_file)
            output_file = output_dir / f"frame_{data['step']:06d}.png"

            if output_file.exists():
                print(f"  Skipping (already exists)")
                continue

            elapsed = render_frame(data, output_file)
            print(f"  z={data['redshift']:.2f}, Seg={data['seg']:.4f} -> {output_file.name} ({elapsed:.1f}s)")
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

if __name__ == '__main__':
    main()
