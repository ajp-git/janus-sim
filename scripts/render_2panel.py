#!/usr/bin/env python3
"""
Render 2-panel 2.5D visualization from Janus binary render_data files.

Usage:
    python render_2panel.py /path/to/render_data/step_000100.bin /path/to/output.png

Binary format:
    - step (u32)
    - box_size (f64)
    - seg (f64)
    - ke_ratio (f64)
    - redshift (f64)
    - n (u32)
    - positions: n × 3 × f32
    - signs: n × i8
"""

import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_render_data(path):
    """Load binary render_data file."""
    with open(path, 'rb') as f:
        step = struct.unpack('<I', f.read(4))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]
        redshift = struct.unpack('<d', f.read(8))[0]
        n = struct.unpack('<I', f.read(4))[0]

        positions = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    return {
        'step': step,
        'box_size': box_size,
        'seg': seg,
        'ke_ratio': ke_ratio,
        'z': redshift,
        'n': n,
        'pos': positions,
        'signs': signs
    }

def create_density_grid(pos, signs, box_size, grid_size=512):
    """Create 2D density grids for positive and negative particles."""
    half_box = box_size / 2

    # XY projection
    xy_pos = np.zeros((grid_size, grid_size), dtype=np.float32)
    xy_neg = np.zeros((grid_size, grid_size), dtype=np.float32)

    # XZ projection
    xz_pos = np.zeros((grid_size, grid_size), dtype=np.float32)
    xz_neg = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Convert positions to grid indices
    scale = grid_size / box_size

    for i in range(len(pos)):
        x, y, z = pos[i]
        ix = int((x + half_box) * scale) % grid_size
        iy = int((y + half_box) * scale) % grid_size
        iz = int((z + half_box) * scale) % grid_size

        if signs[i] > 0:
            xy_pos[iy, ix] += 1
            xz_pos[iz, ix] += 1
        else:
            xy_neg[iy, ix] += 1
            xz_neg[iz, ix] += 1

    return xy_pos, xy_neg, xz_pos, xz_neg

def create_ratio_image(pos_grid, neg_grid, smooth_sigma=1.5):
    """Create RGB image based on ratio of + to - particles."""
    from scipy.ndimage import gaussian_filter

    # Smooth both grids
    pos_smooth = gaussian_filter(pos_grid.astype(np.float64), sigma=smooth_sigma)
    neg_smooth = gaussian_filter(neg_grid.astype(np.float64), sigma=smooth_sigma)

    total = pos_smooth + neg_smooth + 1e-10
    ratio = (pos_smooth - neg_smooth) / total  # [-1, 1]

    # Create RGB: blue=+, red=-, green=mixed
    rgb = np.zeros((*pos_grid.shape, 3), dtype=np.float32)

    # Blue channel: more positive
    rgb[:, :, 2] = np.clip(ratio, 0, 1)
    # Red channel: more negative
    rgb[:, :, 0] = np.clip(-ratio, 0, 1)
    # Green channel: density (brightness)
    brightness = np.log1p(total) / np.log1p(total.max() + 1e-10)
    rgb[:, :, 0] *= brightness
    rgb[:, :, 1] = brightness * (1 - np.abs(ratio)) * 0.3  # slight green for mixed
    rgb[:, :, 2] *= brightness

    return rgb

def render_2panel(data, output_path, grid_size=1024):
    """Render 2-panel visualization: XY and XZ projections."""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "-q"])
        from scipy.ndimage import gaussian_filter

    print(f"Creating density grids ({grid_size}x{grid_size})...")
    xy_pos, xy_neg, xz_pos, xz_neg = create_density_grid(
        data['pos'], data['signs'], data['box_size'], grid_size
    )

    print("Creating ratio images...")
    xy_rgb = create_ratio_image(xy_pos, xy_neg)
    xz_rgb = create_ratio_image(xz_pos, xz_neg)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')

    # XY projection
    axes[0].imshow(xy_rgb, origin='lower', extent=[-data['box_size']/2, data['box_size']/2,
                                                    -data['box_size']/2, data['box_size']/2])
    axes[0].set_xlabel('X (Mpc)', fontsize=12, color='white')
    axes[0].set_ylabel('Y (Mpc)', fontsize=12, color='white')
    axes[0].set_title('XY Projection', fontsize=14, color='white')
    axes[0].tick_params(colors='white')
    for spine in axes[0].spines.values():
        spine.set_color('white')

    # XZ projection
    axes[1].imshow(xz_rgb, origin='lower', extent=[-data['box_size']/2, data['box_size']/2,
                                                    -data['box_size']/2, data['box_size']/2])
    axes[1].set_xlabel('X (Mpc)', fontsize=12, color='white')
    axes[1].set_ylabel('Z (Mpc)', fontsize=12, color='white')
    axes[1].set_title('XZ Projection', fontsize=14, color='white')
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_color('white')

    # Title with metrics
    title = f"Janus 60M — Step {data['step']:,} — z = {data['z']:.2f}\n"
    title += f"KE/KE₀ = {data['ke_ratio']:.4f} — Seg = {data['seg']:.4f}"
    fig.suptitle(title, fontsize=16, color='white', y=0.98)

    # Legend
    fig.text(0.5, 0.02, 'Blue = positive mass | Red = negative mass | Brightness = density',
             ha='center', fontsize=11, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python render_2panel.py <input.bin> [output.png]")
        print("       python render_2panel.py <render_data_dir> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]

    import os
    if os.path.isfile(input_path):
        # Single file
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.bin', '.png')
        print(f"Loading {input_path}...")
        data = load_render_data(input_path)
        print(f"  Step: {data['step']}, z={data['z']:.2f}, N={data['n']:,}")
        render_2panel(data, output_path)
    else:
        # Directory - process all
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('render_data', 'frames')
        os.makedirs(output_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(input_path) if f.endswith('.bin')])
        print(f"Processing {len(files)} files...")

        for f in files:
            data = load_render_data(os.path.join(input_path, f))
            out_file = os.path.join(output_dir, f.replace('.bin', '.png'))
            render_2panel(data, out_file)
