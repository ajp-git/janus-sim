#!/usr/bin/env python3
"""
Render 40M snapshots in 4K with TRUE isometric 2.5D projection.
All 40M particles via density grid, with 3D axes.

Layout:
  ┌─────────────────────────┬───────────┐
  │                         │  Masses+  │
  │   Vue isométrique 2.5D  │  (bleu)   │
  │   azimuth=30°, elev=20° ├───────────┤
  │   ALL 40M particles     │  Masses−  │
  │   + axes X, Y, Z        │  (rouge)  │
  └─────────────────────────┴───────────┘
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

# Parameters
SNAPSHOT_DIR = Path("/mnt/T2/janus-sim/output/40M_v3_2026-02-27/snapshots")
OUTPUT_DIR = Path("/mnt/T2/janus-sim/output/40M_v3_2026-02-27/frames")
BOX_SIZE = 736.8
PARTICLE_SIZE = 13
GRID_SIZE = 1024  # Resolution for density grid

# Isometric projection angles
AZIMUTH = 30  # degrees
ELEVATION = 60  # degrees (higher = more top-down, shows thickness)


def rotation_matrix(azim_deg, elev_deg):
    """
    Create rotation matrix for isometric projection.
    azim: rotation around Z axis (yaw)
    elev: rotation around X axis (pitch) - looking down from above
    """
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)

    # Rotation around Z (azimuth)
    Rz = np.array([
        [np.cos(azim), -np.sin(azim), 0],
        [np.sin(azim),  np.cos(azim), 0],
        [0,             0,            1]
    ])

    # Rotation around X (elevation - tilt down)
    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(elev), -np.sin(elev)],
        [0, np.sin(elev),  np.cos(elev)]
    ])

    # Combined: first azimuth, then elevation
    return Rx @ Rz


def project_isometric(positions, azim=30, elev=20):
    """
    Apply isometric rotation and return 2D projected coordinates.
    Returns (x', y') after rotation - we drop the depth coordinate.
    """
    R = rotation_matrix(azim, elev)

    # Apply rotation: positions is Nx3, R is 3x3
    # rotated = positions @ R.T
    rotated = positions @ R.T

    # Return X and Y of rotated coordinates (drop Z = depth)
    return rotated[:, 0], rotated[:, 1]


def stream_to_isometric_density(filepath, grid_size=1024, box_size=736.8, azim=30, elev=20):
    """
    Stream through ALL particles, apply isometric rotation, accumulate into density grid.
    Returns separate grids for + and - particles.
    """
    R = rotation_matrix(azim, elev)

    # Compute projected box extent
    # The box corners after rotation determine the extent
    corners = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ]) * (box_size / 2)

    corners_rot = corners @ R.T
    x_min, x_max = corners_rot[:, 0].min(), corners_rot[:, 0].max()
    y_min, y_max = corners_rot[:, 1].min(), corners_rot[:, 1].max()

    # Use tight extent (minimal padding)
    x_extent = max(abs(x_min), abs(x_max)) * 1.02
    y_extent = max(abs(y_min), abs(y_max)) * 1.02
    extent = max(x_extent, y_extent)  # For compatibility
    cell_size = (2 * extent) / grid_size

    # Density grids
    density_plus = np.zeros((grid_size, grid_size), dtype=np.float32)
    density_minus = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Also keep XY projections for side panels
    half = box_size / 2
    cell_xy = box_size / grid_size
    density_plus_xy = np.zeros((grid_size, grid_size), dtype=np.float32)
    density_minus_xy = np.zeros((grid_size, grid_size), dtype=np.float32)

    n_plus = 0
    n_minus = 0

    with open(filepath, 'rb') as f:
        n_total = struct.unpack('<Q', f.read(8))[0]

        chunk_size = 500_000
        particles_read = 0

        while particles_read < n_total:
            to_read = min(chunk_size, n_total - particles_read)
            data = f.read(to_read * PARTICLE_SIZE)

            if len(data) < to_read * PARTICLE_SIZE:
                break

            # Parse chunk
            chunk_data = np.frombuffer(data, dtype=np.uint8).reshape(-1, PARTICLE_SIZE)
            pos_bytes = chunk_data[:, :12].tobytes()
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            signs = chunk_data[:, 12].astype(np.int8)

            # Apply isometric rotation
            rotated = positions @ R.T

            # Convert to grid indices (isometric view)
            ix = np.clip(((rotated[:, 0] + extent) / cell_size).astype(np.int32), 0, grid_size - 1)
            iy = np.clip(((rotated[:, 1] + extent) / cell_size).astype(np.int32), 0, grid_size - 1)

            # XY projection indices
            ix_xy = np.clip(((positions[:, 0] + half) / cell_xy).astype(np.int32), 0, grid_size - 1)
            iy_xy = np.clip(((positions[:, 1] + half) / cell_xy).astype(np.int32), 0, grid_size - 1)

            mask_pos = signs > 0
            mask_neg = signs < 0

            # Accumulate isometric
            np.add.at(density_plus, (ix[mask_pos], iy[mask_pos]), 1)
            np.add.at(density_minus, (ix[mask_neg], iy[mask_neg]), 1)

            # Accumulate XY
            np.add.at(density_plus_xy, (ix_xy[mask_pos], iy_xy[mask_pos]), 1)
            np.add.at(density_minus_xy, (ix_xy[mask_neg], iy_xy[mask_neg]), 1)

            n_plus += mask_pos.sum()
            n_minus += mask_neg.sum()

            particles_read += to_read

    return {
        'plus_iso': density_plus,
        'minus_iso': density_minus,
        'plus_xy': density_plus_xy,
        'minus_xy': density_minus_xy,
        'n_plus': n_plus,
        'n_minus': n_minus,
        'n_total': n_total,
        'extent': extent,
        'x_extent': x_extent,
        'y_extent': y_extent
    }


def process_density(g, sigma=0.5):
    """Apply log scaling and normalization to density grid."""
    # Less smoothing (sigma=0.5) to show more individual particles
    g = gaussian_filter(g.astype(np.float32), sigma=sigma)
    # Stronger log scaling to boost faint particles
    g = np.log1p(g * 50)
    p99 = np.percentile(g, 99.5)
    p10 = np.percentile(g, 10)
    # Normalize from p10 to show more particles
    if p99 > p10:
        g = np.clip((g - p10) / (p99 - p10), 0, 1)
    return g


def draw_3d_axes(ax, extent, box_size, azim=30, elev=20):
    """Draw 3D coordinate axes with isometric projection and scale ticks in Mpc."""
    R = rotation_matrix(azim, elev)

    half_box = box_size / 2

    # Draw cube edges (wireframe) - bottom face
    corners_bottom = np.array([
        [-half_box, -half_box, -half_box],
        [half_box, -half_box, -half_box],
        [half_box, half_box, -half_box],
        [-half_box, half_box, -half_box],
        [-half_box, -half_box, -half_box],  # close the loop
    ])
    corners_bottom_rot = corners_bottom @ R.T
    ax.plot(corners_bottom_rot[:, 0], corners_bottom_rot[:, 1],
            color='white', linewidth=0.5, alpha=0.3)

    # Vertical edges (only 3 visible in isometric)
    for corner in [[-half_box, -half_box], [half_box, -half_box], [-half_box, half_box]]:
        bottom = np.array([corner[0], corner[1], -half_box])
        top = np.array([corner[0], corner[1], half_box])
        pts = np.array([bottom, top]) @ R.T
        ax.plot(pts[:, 0], pts[:, 1], color='white', linewidth=0.5, alpha=0.3)

    # Draw axes from corner with tick marks
    origin_3d = np.array([-half_box, -half_box, -half_box])
    origin_rot = origin_3d @ R.T

    # Tick values in Mpc
    tick_values = [-300, -150, 0, 150, 300]
    tick_len = extent * 0.02  # tick mark length

    axes_config = [
        ('X', np.array([1, 0, 0]), np.array([0, -1, 0]), '#00cc00'),
        ('Y', np.array([0, 1, 0]), np.array([-1, 0, 0]), '#4488ff'),
        ('Z', np.array([0, 0, 1]), np.array([-1, 0, 0]), '#ff4444'),
    ]

    for axis_name, direction, tick_dir, color in axes_config:
        # Axis line from -half_box to +half_box
        start_3d = origin_3d.copy()
        end_3d = origin_3d + direction * box_size

        start_rot = start_3d @ R.T
        end_rot = end_3d @ R.T

        ax.plot([start_rot[0], end_rot[0]], [start_rot[1], end_rot[1]],
                color='white', linewidth=2, alpha=0.9)

        # Draw ticks with values
        for val in tick_values:
            tick_pos_3d = origin_3d + direction * (val + half_box)
            tick_pos_rot = tick_pos_3d @ R.T

            # Tick mark
            tick_end_3d = tick_pos_3d + tick_dir * 20
            tick_end_rot = tick_end_3d @ R.T
            ax.plot([tick_pos_rot[0], tick_end_rot[0]], [tick_pos_rot[1], tick_end_rot[1]],
                    color='white', linewidth=1.5, alpha=0.8)

            # Tick label (only for some values to avoid clutter)
            if val in [-300, 0, 300]:
                label_pos_3d = tick_pos_3d + tick_dir * 80
                label_pos_rot = label_pos_3d @ R.T
                ax.text(label_pos_rot[0], label_pos_rot[1], f'{val}', color='white',
                        fontsize=12, fontweight='bold', ha='center', va='center',
                        clip_on=False)


def render_4k(grids, step, z, H, seg, output_path):
    """Render 4K frame with isometric 3-panel layout."""

    n_total = grids['n_total']
    n_plus = grids['n_plus']
    n_minus = grids['n_minus']
    extent = grids['extent']
    x_extent = grids['x_extent']
    y_extent = grids['y_extent']

    # Process density grids
    plus_iso = process_density(grids['plus_iso'])
    minus_iso = process_density(grids['minus_iso'])
    plus_xy = process_density(grids['plus_xy'])
    minus_xy = process_density(grids['minus_xy'])

    # Calculate optimal figure size based on cube aspect ratio
    cube_aspect = x_extent / y_extent  # width/height of projected cube

    # Target: main panel fills ~70% width, side panels ~25%, margins ~5%
    # Height: 2160 px fixed
    fig_height = 21.6  # inches at 100 dpi = 2160 px

    # Main panel dimensions (in figure fraction)
    main_height_frac = 0.92
    main_width_frac = main_height_frac * cube_aspect * (fig_height / 1.0)  # Will adjust fig width

    # Side panels: 25% of main panel width
    side_width_frac = 0.18
    side_height_frac = 0.38
    gap = 0.01  # small gap between main and side panels

    # Total width needed (in figure fraction assuming fig_width = X)
    # main_panel + gap + side_panels + margins
    # Solve: main_width_frac * (fig_height/fig_width) + gap + side_width_frac + margins = ~0.98

    # Calculate optimal figure width
    # main takes 70%, side takes 25%, margins 5%
    target_main_frac = 0.70
    fig_width = (main_height_frac * cube_aspect * fig_height) / target_main_frac
    fig_width = min(fig_width, 38.4)  # cap at 4K width
    fig_width = max(fig_width, 28.0)  # minimum reasonable width

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')

    # Recalculate fractions with actual fig size
    fig_aspect = fig_width / fig_height
    main_width = main_height_frac * cube_aspect / fig_aspect
    main_width = min(main_width, 0.72)  # leave room for side panels

    # Center main panel vertically
    main_bottom = (1.0 - main_height_frac) / 2

    # Layout - add margins for axis labels (left for Y, bottom for Z)
    left_margin = 0.07  # Increased for Y [Mpc] label
    bottom_margin = 0.07
    main_panel_width = main_width - 0.02
    main_panel_right = left_margin + main_panel_width

    # Side panels with gap after main panel
    side_gap = 0.03  # gap between main and side panels
    side_left = main_panel_right + side_gap
    side_width = 0.97 - side_left  # fill to near edge

    ax_main = fig.add_axes([left_margin, bottom_margin, main_panel_width, main_height_frac - 0.02], facecolor='black')
    ax_plus = fig.add_axes([side_left, 0.53, side_width, 0.37], facecolor='black')
    ax_minus = fig.add_axes([side_left, 0.09, side_width, 0.37], facecolor='black')

    # Colors
    color_plus = np.array([0.2, 0.5, 1.0])   # Blue
    color_minus = np.array([1.0, 0.25, 0.25]) # Red

    # === MAIN PANEL: Isometric density ===
    grid_size = plus_iso.shape[0]
    rgb = np.zeros((grid_size, grid_size, 3))
    rgb[:, :, 0] = minus_iso * 0.8
    rgb[:, :, 2] = plus_iso * 0.8
    rgb[:, :, 1] = np.minimum(plus_iso, minus_iso) * 0.2
    rgb = np.clip(rgb, 0, 1)
    rgb = np.power(rgb, 1.2)

    ax_main.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower', aspect='equal',
                   interpolation='bilinear', extent=[-x_extent, x_extent, -y_extent, y_extent])

    # Draw 3D axes with scale
    draw_3d_axes(ax_main, extent, BOX_SIZE, AZIMUTH, ELEVATION)

    # Single Mpc label — bottom right of main panel, in black zone
    ax_main.text(0.97, 0.03, "X, Y, Z [Mpc]",
        transform=ax_main.transAxes,
        color='white', fontsize=13, alpha=0.7,
        ha='right', va='bottom',
        fontfamily='monospace')

    ax_main.set_xlim(-x_extent, x_extent)
    ax_main.set_ylim(-y_extent, y_extent)
    ax_main.set_xlabel('', fontsize=1)
    ax_main.set_ylabel('', fontsize=1)
    ax_main.tick_params(colors='white', labelsize=10)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # === RIGHT TOP: Masses+ (XY projection) ===
    rgb_plus = np.zeros((grid_size, grid_size, 3))
    rgb_plus[:, :, 0] = plus_xy * color_plus[0]
    rgb_plus[:, :, 1] = plus_xy * color_plus[1]
    rgb_plus[:, :, 2] = plus_xy * color_plus[2]
    rgb_plus = np.power(np.clip(rgb_plus, 0, 1), 1.2)

    ax_plus.imshow(np.transpose(rgb_plus, (1, 0, 2)), origin='lower', aspect='equal',
                   interpolation='bilinear', extent=[-BOX_SIZE/2, BOX_SIZE/2, -BOX_SIZE/2, BOX_SIZE/2])
    ax_plus.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_plus.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_plus.axis('off')
    ax_plus.set_title(f'Masses+ (N = {n_plus:,})', color='#2266dd', fontsize=16, pad=10, fontweight='bold')

    # === RIGHT BOTTOM: Masses- (XY projection) ===
    rgb_minus = np.zeros((grid_size, grid_size, 3))
    rgb_minus[:, :, 0] = minus_xy * color_minus[0]
    rgb_minus[:, :, 1] = minus_xy * color_minus[1]
    rgb_minus[:, :, 2] = minus_xy * color_minus[2]
    rgb_minus = np.power(np.clip(rgb_minus, 0, 1), 1.2)

    ax_minus.imshow(np.transpose(rgb_minus, (1, 0, 2)), origin='lower', aspect='equal',
                    interpolation='bilinear', extent=[-BOX_SIZE/2, BOX_SIZE/2, -BOX_SIZE/2, BOX_SIZE/2])
    ax_minus.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_minus.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax_minus.axis('off')
    ax_minus.set_title(f'Masses− (N = {n_minus:,})', color='#dd2222', fontsize=16, pad=10, fontweight='bold')

    # === Title (centered at top) ===
    title = f"Janus Cosmological Model — {n_total/1e6:.0f}M particles | η = 1.045"
    fig.text(0.5, 0.99, title, ha='center', va='top', fontsize=22,
             color='black', fontweight='bold')

    # === Stats bar (centered) ===
    stats = f"Step {step:04d}  |  z = {z:.3f}  |  H = {H:.4f} H₀  |  Seg = {seg:.4f} Mpc"
    fig.text(0.5, 0.012, stats, ha='center', va='bottom', fontsize=18,
             color='black', family='monospace')

    # === Legends inside panels ===
    ax_plus.text(0.05, 0.05, "● m+ (attracts m+, repels m−)",
        transform=ax_plus.transAxes, color='#6699ff',
        fontsize=14, fontweight='bold', va='bottom')
    ax_minus.text(0.05, 0.05, "● m− (attracts m−, repels m+)",
        transform=ax_minus.transAxes, color='#ff6666',
        fontsize=14, fontweight='bold', va='bottom')

    plt.savefig(output_path, facecolor='white', dpi=100)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = sorted(SNAPSHOT_DIR.glob("snapshot_*.bin"))
    print(f"Found {len(snapshots)} snapshots")

    if not snapshots:
        print("No snapshots found!")
        return

    # Read CSV for metadata
    csv_path = SNAPSHOT_DIR.parent / "time_series.csv"
    metadata = {}
    if csv_path.exists():
        with open(csv_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    step = int(parts[0])
                    z = float(parts[2])
                    H = float(parts[4])
                    seg = float(parts[5])
                    metadata[step] = {'z': z, 'H': H, 'seg': seg}

    for i, snap_path in enumerate(snapshots):
        step = int(snap_path.stem.split('_')[1])
        output_path = OUTPUT_DIR / f"frame_{step:05d}_4k.png"

        # Force re-render for new projection
        if output_path.exists():
            output_path.unlink()

        print(f"[{i+1}/{len(snapshots)}] Processing {snap_path.name} (isometric projection)...")

        meta = metadata.get(step, metadata.get(1, {'z': 5.0, 'H': 2.4, 'seg': 0.087}))

        grids = stream_to_isometric_density(snap_path, GRID_SIZE, BOX_SIZE, AZIMUTH, ELEVATION)
        print(f"    N+ = {grids['n_plus']:,}, N- = {grids['n_minus']:,}")

        render_4k(grids, step, meta['z'], meta['H'], meta['seg'], output_path)
        print(f"    Saved: {output_path}")

    print(f"\nDone! 4K frames: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
