#!/usr/bin/env python3
"""
PM-5 Snapshot Renderer — 4K 3-panel density visualization

Layout:
┌─────────────────────┬──────────────┐
│                     │  Masses+     │
│  Combined           │  (blues)     │
│  XY density         ├──────────────┤
│                     │  Masses-     │
│                     │  (reds)      │
└─────────────────────┴──────────────┘

Usage:
    python render_frames.py <snapshot.bin>                    # single file
    python render_frames.py <input_dir> <output_dir>          # batch mode
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from pathlib import Path

# Visual config
GRID_SIZE = 512  # Smaller grid for better density contrast with 1M particles
FIG_WIDTH = 3840
FIG_HEIGHT = 2160
DPI = 100
BG_COLOR = '#0a0a0a'
BOX_SIZE = 500.0

def read_light_snapshot(path):
    """Read PM-5 light snapshot (f32 positions, i8 signs)."""
    with open(path, 'rb') as f:
        n_particles = struct.unpack('<Q', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        scale_factor = struct.unpack('<d', f.read(8))[0]
        segregation = struct.unpack('<d', f.read(8))[0]

        # Read interleaved data: x(f32), y(f32), z(f32), sign(i8) per particle
        raw = np.frombuffer(f.read(n_particles * 13), dtype=np.uint8)
        raw = raw.reshape(n_particles, 13)

        x = np.frombuffer(raw[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(raw[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(raw[:, 8:12].tobytes(), dtype=np.float32)
        signs = np.frombuffer(raw[:, 12:13].tobytes(), dtype=np.int8)

    return {
        'n': n_particles,
        'step': step,
        'a': scale_factor,
        'seg': segregation,
        'x': x, 'y': y, 'z': z,
        'signs': signs
    }

def compute_density_grid(x, y, grid_size=GRID_SIZE, box_size=BOX_SIZE):
    """Compute 2D density grid by particle accumulation."""
    # Bin particles into grid
    ix = np.clip((x / box_size * grid_size).astype(int), 0, grid_size - 1)
    iy = np.clip((y / box_size * grid_size).astype(int), 0, grid_size - 1)

    # Accumulate
    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(density, (iy, ix), 1)

    return density

def render_frame(data, output_path):
    """Render 4K 3-panel visualization."""
    x, y = data['x'], data['y']
    signs = data['signs']

    pos_mask = signs > 0
    neg_mask = signs < 0

    # Compute density grids
    density_pos = compute_density_grid(x[pos_mask], y[pos_mask])
    density_neg = compute_density_grid(x[neg_mask], y[neg_mask])
    density_all = density_pos + density_neg

    # Log transform: log(1 + density)
    log_pos = np.log1p(density_pos)
    log_neg = np.log1p(density_neg)
    log_all = np.log1p(density_all)

    # Normalize for display
    def normalize(arr):
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            return (arr - vmin) / (vmax - vmin)
        return arr * 0

    norm_pos = normalize(log_pos)
    norm_neg = normalize(log_neg)
    norm_all_pos = normalize(np.log1p(density_pos))
    norm_all_neg = normalize(np.log1p(density_neg))

    # Create figure
    fig = plt.figure(figsize=(FIG_WIDTH/DPI, FIG_HEIGHT/DPI), dpi=DPI, facecolor=BG_COLOR)

    # Layout: left panel 2/3 width, right panels stacked
    # Using gridspec for precise control
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 0.05],
                          height_ratios=[1, 1],
                          left=0.02, right=0.98, bottom=0.08, top=0.92,
                          wspace=0.05, hspace=0.05)

    # Left panel: Combined density with alpha blending
    ax_combined = fig.add_subplot(gs[:, 0])
    ax_combined.set_facecolor(BG_COLOR)

    # Create RGB image with alpha blending
    # Cyan for positive, Red for negative
    rgb_combined = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)

    # Blues colormap for positive (cyan tint)
    rgb_combined[:, :, 0] = norm_all_pos * 0.2  # R
    rgb_combined[:, :, 1] = norm_all_pos * 0.8  # G (cyan)
    rgb_combined[:, :, 2] = norm_all_pos * 1.0  # B

    # Add reds for negative
    rgb_combined[:, :, 0] += norm_all_neg * 1.0  # R
    rgb_combined[:, :, 1] += norm_all_neg * 0.2  # G
    rgb_combined[:, :, 2] += norm_all_neg * 0.2  # B

    # Clip to [0, 1]
    rgb_combined = np.clip(rgb_combined, 0, 1)

    ax_combined.imshow(rgb_combined, origin='lower', extent=[0, BOX_SIZE, 0, BOX_SIZE])
    ax_combined.set_xlabel('X (Mpc)', color='white', fontsize=12)
    ax_combined.set_ylabel('Y (Mpc)', color='white', fontsize=12)
    ax_combined.set_title('Combined XY Density', color='white', fontsize=14)
    ax_combined.tick_params(colors='white')
    for spine in ax_combined.spines.values():
        spine.set_color('#333333')

    # Custom colormaps with black background
    from matplotlib.colors import LinearSegmentedColormap

    # Blues with black background
    blues_data = {'red':   [(0, 0.04, 0.04), (1, 0.2, 0.2)],
                  'green': [(0, 0.04, 0.04), (1, 0.8, 0.8)],
                  'blue':  [(0, 0.04, 0.04), (1, 1.0, 1.0)]}
    cmap_cyan = LinearSegmentedColormap('BlackCyan', blues_data)

    # Reds with black background
    reds_data = {'red':   [(0, 0.04, 0.04), (1, 1.0, 1.0)],
                 'green': [(0, 0.04, 0.04), (1, 0.2, 0.2)],
                 'blue':  [(0, 0.04, 0.04), (1, 0.2, 0.2)]}
    cmap_red = LinearSegmentedColormap('BlackRed', reds_data)

    # Top right: Masses+ (cyan colormap)
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_pos.set_facecolor(BG_COLOR)
    vmax_pos = np.percentile(log_pos[log_pos > 0], 99) if np.any(log_pos > 0) else 1
    im_pos = ax_pos.imshow(log_pos, origin='lower', cmap=cmap_cyan,
                           extent=[0, BOX_SIZE, 0, BOX_SIZE],
                           vmin=0, vmax=vmax_pos)
    ax_pos.set_title(f'Masses+ ({np.sum(pos_mask):,})', color='cyan', fontsize=12)
    ax_pos.tick_params(colors='white', labelsize=8)
    for spine in ax_pos.spines.values():
        spine.set_color('#333333')

    # Bottom right: Masses- (red colormap)
    ax_neg = fig.add_subplot(gs[1, 1])
    ax_neg.set_facecolor(BG_COLOR)
    vmax_neg = np.percentile(log_neg[log_neg > 0], 99) if np.any(log_neg > 0) else 1
    im_neg = ax_neg.imshow(log_neg, origin='lower', cmap=cmap_red,
                           extent=[0, BOX_SIZE, 0, BOX_SIZE],
                           vmin=0, vmax=vmax_neg)
    ax_neg.set_title(f'Masses- ({np.sum(neg_mask):,})', color='red', fontsize=12)
    ax_neg.set_xlabel('X (Mpc)', color='white', fontsize=10)
    ax_neg.tick_params(colors='white', labelsize=8)
    for spine in ax_neg.spines.values():
        spine.set_color('#333333')

    # Info bar at bottom
    info_text = (f"Step {data['step']:05d}  |  "
                 f"a = {data['a']:.4f}  |  "
                 f"Seg = {data['seg']:.6f}  |  "
                 f"N = {data['n']:,}")
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom',
             color='white', fontsize=14, fontfamily='monospace')

    # Title
    fig.suptitle('Janus PM-5: 150M Particles — XY Projection',
                 color='white', fontsize=18, y=0.97)

    plt.savefig(output_path, dpi=DPI, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python render_frames.py <snapshot.bin>")
        print("       python render_frames.py <input_dir> <output_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_file():
        # Single file mode
        output_path = input_path.with_suffix('.png')
        if len(sys.argv) >= 3:
            output_path = Path(sys.argv[2])

        print(f"Rendering {input_path.name}...")
        data = read_light_snapshot(input_path)
        render_frame(data, output_path)
        print(f"  Saved: {output_path}")

    elif input_path.is_dir():
        # Batch mode
        output_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find light snapshots (< 100 MB, numbered)
        snapshots = sorted(input_path.glob("snapshot_[0-9]*.bin"))
        light_snaps = [s for s in snapshots if s.stat().st_size < 100_000_000]

        print(f"Found {len(light_snaps)} light snapshots")

        for i, snap in enumerate(light_snaps):
            step_num = snap.stem.split('_')[1]
            out_file = output_dir / f"frame_{step_num}.png"

            print(f"  [{i+1}/{len(light_snaps)}] {snap.name} -> {out_file.name}")
            try:
                data = read_light_snapshot(snap)
                render_frame(data, out_file)
            except Exception as e:
                print(f"    Error: {e}")

        print(f"Done. Frames saved to {output_dir}")

if __name__ == "__main__":
    main()
