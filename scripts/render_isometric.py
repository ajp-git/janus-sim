#!/usr/bin/env python3
"""
Janus 85M - Isometric + Orthographic Frame Renderer

Layout:
  +---------------------------+--------+
  |                           |   XY   |
  |    2.5D Isometric         +--------+
  |    (large, left)          |   XZ   |
  |                           +--------+
  |                           |   YZ   |
  +---------------------------+--------+

- White overall background
- Black background for each render panel
- Blue (m+) and Red (m-) particles

Usage: python3 render_isometric.py <output_dir> [--once]
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import struct
import sys
import time
from pathlib import Path
import math

# Layout dimensions - 8K resolution
ISO_SIZE = 5600          # Isometric panel (left)
ORTHO_SIZE = 1867        # Orthographic panels (right, 3 stacked = 5601)
MARGIN = 40              # Margin around panels
TITLE_HEIGHT = 100       # Title bar height

IMG_WIDTH = MARGIN + ISO_SIZE + MARGIN + ORTHO_SIZE + MARGIN   # 7587
IMG_HEIGHT = TITLE_HEIGHT + MARGIN + ISO_SIZE + MARGIN          # 5780

# Colors (RGB, 0-1 range for accumulation)
COLOR_POS = np.array([0.27, 0.53, 1.0], dtype=np.float32)   # Blue
COLOR_NEG = np.array([1.0, 0.27, 0.13], dtype=np.float32)   # Red

# Isometric projection matrix (rotate around Y, then tilt around X)
# 30° azimuth gives a slightly more frontal view
ISO_ANGLE_Y = math.radians(30)      # Azimuth (was 45°)
ISO_ANGLE_X = math.radians(30)      # Elevation

cos_y, sin_y = math.cos(ISO_ANGLE_Y), math.sin(ISO_ANGLE_Y)
cos_x, sin_x = math.cos(ISO_ANGLE_X), math.sin(ISO_ANGLE_X)

# Combined rotation matrix (Y then X)
ISO_MATRIX = np.array([
    [cos_y, 0, sin_y],
    [sin_x * sin_y, cos_x, -sin_x * cos_y],
    [-cos_x * sin_y, sin_x, cos_x * cos_y]
], dtype=np.float32)


def compute_alpha(n_particles, panel_size):
    """Compute alpha based on particle count and panel size"""
    pixels = panel_size * panel_size
    particles_per_pixel = n_particles / pixels
    # Target ~0.3 average brightness
    return min(0.12, 0.3 / max(particles_per_pixel, 1))


def render_isometric(buf, pos, color, box_size, panel_x, panel_y, panel_size, alpha):
    """Render isometric 2.5D projection"""
    # Apply isometric rotation
    pos_rot = pos @ ISO_MATRIX.T

    # Project to 2D (drop Z after rotation)
    x = pos_rot[:, 0]
    y = pos_rot[:, 1]

    # Scale to fit panel (account for rotation expanding bounds)
    scale_factor = 0.65  # Account for rotation diagonal
    center = panel_size / 2
    scale = (panel_size * scale_factor) / box_size

    ix = (x * scale + center).clip(0, panel_size - 1).astype(np.int32) + panel_x
    iy = (-y * scale + center).clip(0, panel_size - 1).astype(np.int32) + panel_y

    np.add.at(buf, (iy, ix, 0), color[0] * alpha)
    np.add.at(buf, (iy, ix, 1), color[1] * alpha)
    np.add.at(buf, (iy, ix, 2), color[2] * alpha)


def render_ortho_xy(buf, pos, color, box_size, panel_x, panel_y, panel_size, alpha):
    """Render XY orthographic projection"""
    x = pos[:, 0]
    y = pos[:, 1]

    box_min = -box_size / 2.0
    scale = (panel_size - 1) / box_size

    ix = ((x - box_min) * scale).clip(0, panel_size - 1).astype(np.int32) + panel_x
    iy = ((y - box_min) * scale).clip(0, panel_size - 1).astype(np.int32) + panel_y
    iy_flip = panel_y + (panel_size - 1) - (iy - panel_y)

    np.add.at(buf, (iy_flip, ix, 0), color[0] * alpha)
    np.add.at(buf, (iy_flip, ix, 1), color[1] * alpha)
    np.add.at(buf, (iy_flip, ix, 2), color[2] * alpha)


def render_ortho_xz(buf, pos, color, box_size, panel_x, panel_y, panel_size, alpha):
    """Render XZ orthographic projection"""
    x = pos[:, 0]
    z = pos[:, 2]

    box_min = -box_size / 2.0
    scale = (panel_size - 1) / box_size

    ix = ((x - box_min) * scale).clip(0, panel_size - 1).astype(np.int32) + panel_x
    iz = ((z - box_min) * scale).clip(0, panel_size - 1).astype(np.int32) + panel_y
    iz_flip = panel_y + (panel_size - 1) - (iz - panel_y)

    np.add.at(buf, (iz_flip, ix, 0), color[0] * alpha)
    np.add.at(buf, (iz_flip, ix, 1), color[1] * alpha)
    np.add.at(buf, (iz_flip, ix, 2), color[2] * alpha)


def render_ortho_yz(buf, pos, color, box_size, panel_x, panel_y, panel_size, alpha):
    """Render YZ orthographic projection"""
    y = pos[:, 1]
    z = pos[:, 2]

    box_min = -box_size / 2.0
    scale = (panel_size - 1) / box_size

    iy = ((y - box_min) * scale).clip(0, panel_size - 1).astype(np.int32) + panel_x
    iz = ((z - box_min) * scale).clip(0, panel_size - 1).astype(np.int32) + panel_y
    iz_flip = panel_y + (panel_size - 1) - (iz - panel_y)

    np.add.at(buf, (iz_flip, iy, 0), color[0] * alpha)
    np.add.at(buf, (iz_flip, iy, 1), color[1] * alpha)
    np.add.at(buf, (iz_flip, iy, 2), color[2] * alpha)


def render_frame(bin_path: Path, frames_dir: Path) -> bool:
    """Render a single frame from binary data"""
    try:
        with open(bin_path, 'rb') as f:
            # Header: step(u32) + box_size(f64) + seg(f64) + ke_ratio(f64) + redshift(f64) + n(u32)
            step = struct.unpack('<I', f.read(4))[0]
            box_size = struct.unpack('<d', f.read(8))[0]
            seg = struct.unpack('<d', f.read(8))[0]
            ke_ratio = struct.unpack('<d', f.read(8))[0]
            redshift = struct.unpack('<d', f.read(8))[0]
            n = struct.unpack('<I', f.read(4))[0]

            # pos: N x 3 x f32
            pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)

            # signs: N x i8
            signs = np.frombuffer(f.read(n), dtype=np.int8)

        # Check if frame already exists
        out_path = frames_dir / f"frame_{step:06d}.png"
        if out_path.exists():
            print(f"[skip] frame_{step:06d}.png already exists")
            return True

        print(f"[render] Processing step {step:06d}: z={redshift:.2f}, N={n/1e6:.1f}M, S={seg:.3e}, KE/KE0={ke_ratio:.4f}")

        # Compute alpha values
        alpha_iso = compute_alpha(n, ISO_SIZE)
        alpha_ortho = compute_alpha(n, ORTHO_SIZE)

        # Create RGB buffer (white background)
        buf = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

        # Panel positions
        iso_x = MARGIN
        iso_y = TITLE_HEIGHT + MARGIN

        ortho_x = MARGIN + ISO_SIZE + MARGIN
        ortho_spacing = ISO_SIZE // 3  # Each ortho panel takes 1/3 of iso height
        xy_y = TITLE_HEIGHT + MARGIN
        xz_y = TITLE_HEIGHT + MARGIN + ortho_spacing
        yz_y = TITLE_HEIGHT + MARGIN + 2 * ortho_spacing

        # Draw black backgrounds for panels
        buf[iso_y:iso_y+ISO_SIZE, iso_x:iso_x+ISO_SIZE] = 0.0
        buf[xy_y:xy_y+ORTHO_SIZE, ortho_x:ortho_x+ORTHO_SIZE] = 0.0
        buf[xz_y:xz_y+ORTHO_SIZE, ortho_x:ortho_x+ORTHO_SIZE] = 0.0
        buf[yz_y:yz_y+ORTHO_SIZE, ortho_x:ortho_x+ORTHO_SIZE] = 0.0

        # Separate positive and negative particles
        pos_mask = signs > 0
        pos_plus = pos[pos_mask]
        pos_minus = pos[~pos_mask]

        n_plus = len(pos_plus)
        n_minus = len(pos_minus)

        # Render isometric view (both populations)
        if len(pos_plus) > 0:
            render_isometric(buf, pos_plus, COLOR_POS, box_size, iso_x, iso_y, ISO_SIZE, alpha_iso)
        if len(pos_minus) > 0:
            render_isometric(buf, pos_minus, COLOR_NEG, box_size, iso_x, iso_y, ISO_SIZE, alpha_iso)

        # Render orthographic views
        if len(pos_plus) > 0:
            render_ortho_xy(buf, pos_plus, COLOR_POS, box_size, ortho_x, xy_y, ORTHO_SIZE, alpha_ortho)
            render_ortho_xz(buf, pos_plus, COLOR_POS, box_size, ortho_x, xz_y, ORTHO_SIZE, alpha_ortho)
            render_ortho_yz(buf, pos_plus, COLOR_POS, box_size, ortho_x, yz_y, ORTHO_SIZE, alpha_ortho)
        if len(pos_minus) > 0:
            render_ortho_xy(buf, pos_minus, COLOR_NEG, box_size, ortho_x, xy_y, ORTHO_SIZE, alpha_ortho)
            render_ortho_xz(buf, pos_minus, COLOR_NEG, box_size, ortho_x, xz_y, ORTHO_SIZE, alpha_ortho)
            render_ortho_yz(buf, pos_minus, COLOR_NEG, box_size, ortho_x, yz_y, ORTHO_SIZE, alpha_ortho)

        # Convert to uint8
        img_data = (buf * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='RGB')
        draw = ImageDraw.Draw(img)

        # Load font (scaled for 8K)
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        except:
            font_large = ImageFont.load_default()
            font_small = font_large
            font_label = font_large

        # Title bar (black background)
        draw.rectangle([0, 0, IMG_WIDTH, TITLE_HEIGHT], fill=(30, 30, 30))

        # Main title
        title = f"Janus 85M | Step {step:06d} | z = {redshift:.3f} | S = {seg:.2e} | KE/KE0 = {ke_ratio:.4f}"
        draw.text((MARGIN, 15), title, fill=(255, 255, 255), font=font_large)

        # Particle counts
        counts = f"N+ = {n_plus/1e6:.1f}M (blue) | N- = {n_minus/1e6:.1f}M (red)"
        draw.text((IMG_WIDTH - 1100, 25), counts, fill=(200, 200, 200), font=font_small)

        # Panel labels
        draw.text((iso_x + 10, iso_y + 10), "Isometric 2.5D", fill=(255, 255, 255), font=font_label)
        draw.text((ortho_x + 10, xy_y + 5), "XY", fill=(255, 255, 255), font=font_label)
        draw.text((ortho_x + 10, xz_y + 5), "XZ", fill=(255, 255, 255), font=font_label)
        draw.text((ortho_x + 10, yz_y + 5), "YZ", fill=(255, 255, 255), font=font_label)

        # Panel borders (subtle gray)
        border_color = (80, 80, 80)
        draw.rectangle([iso_x-1, iso_y-1, iso_x+ISO_SIZE, iso_y+ISO_SIZE], outline=border_color)
        draw.rectangle([ortho_x-1, xy_y-1, ortho_x+ORTHO_SIZE, xy_y+ORTHO_SIZE], outline=border_color)
        draw.rectangle([ortho_x-1, xz_y-1, ortho_x+ORTHO_SIZE, xz_y+ORTHO_SIZE], outline=border_color)
        draw.rectangle([ortho_x-1, yz_y-1, ortho_x+ORTHO_SIZE, yz_y+ORTHO_SIZE], outline=border_color)

        # Save PNG
        img.save(out_path, optimize=True)
        print(f"[render] frame_{step:06d}.png saved ({IMG_WIDTH}x{IMG_HEIGHT})")
        return True

    except Exception as e:
        print(f"[render] Error processing {bin_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 render_isometric.py <output_dir> [--once]")
        print("  e.g.: python3 render_isometric.py /app/output/85M_2026-02-26")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    once_mode = '--once' in sys.argv

    render_data_dir = output_dir / "render_data"
    frames_dir = output_dir / "frames"

    print("=" * 70)
    print("Janus 85M - Isometric + Orthographic Frame Renderer")
    print("=" * 70)
    print(f"Watching: {render_data_dir}")
    print(f"Output:   {frames_dir}")
    print(f"Mode:     {'once' if once_mode else 'continuous'}")
    print(f"Layout:   {IMG_WIDTH}x{IMG_HEIGHT} (iso={ISO_SIZE}, ortho={ORTHO_SIZE}x3)")
    print()

    render_data_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    while True:
        # Find all .bin files
        bin_files = sorted(render_data_dir.glob("step_*.bin"))

        for bin_path in bin_files:
            t0 = time.time()
            if render_frame(bin_path, frames_dir):
                print(f"[render] Completed in {time.time() - t0:.1f}s")

        if once_mode:
            if not bin_files:
                print("No files to process")
            break

        time.sleep(2)


if __name__ == "__main__":
    main()
