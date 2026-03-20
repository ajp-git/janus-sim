#!/usr/bin/env python3
"""
Janus 85M - Python frame renderer
Watches for .bin files and renders PNG frames

Usage: python3 render_frames.py <output_dir> [--once]
  output_dir: e.g., /app/output/85M_2026-02-25
  --once: Process existing files and exit (don't watch)

Reads from: <output_dir>/render_data/step_XXXXXX.bin
Writes to:  <output_dir>/frames/frame_XXXXXX.png
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import struct
import sys
import time
from pathlib import Path

# Image dimensions
VIEW_SIZE = 1280
IMG_WIDTH = 3840
IMG_HEIGHT = 1280

# Colors (RGB, 0-1 range)
COLOR_POS = np.array([0.27, 0.53, 1.0], dtype=np.float32)   # #4488FF (blue)
COLOR_NEG = np.array([1.0, 0.27, 0.13], dtype=np.float32)   # #FF4422 (red)

def compute_alpha(n_particles):
    """Compute alpha based on particle count for balanced brightness"""
    particles_per_pixel = n_particles / (VIEW_SIZE * VIEW_SIZE)
    # Target ~0.4 average brightness
    return min(0.15, 0.4 / max(particles_per_pixel, 1))


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

            # pos: N×3×f32
            pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)

            # signs: N×i8
            signs = np.frombuffer(f.read(n), dtype=np.int8)

        # Check if frame already exists
        out_path = frames_dir / f"frame_{step:06d}.png"
        if out_path.exists():
            print(f"[skip] frame_{step:06d}.png already exists")
            return True

        # Compute alpha based on particle count
        alpha = compute_alpha(n)
        print(f"[render] Processing step {step:06d}: z={redshift:.2f}, N={n}, S={seg:.3e}, KE/KE0={ke_ratio:.4f}")

        # Create RGBA buffer (black background with full alpha)
        buf = np.zeros((IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.float32)
        buf[:, :, 3] = 1.0  # Opaque black background

        # Coordinate mapping
        box_min = -box_size / 2.0
        scale = (VIEW_SIZE - 1) / box_size

        # Separate positive and negative particles
        pos_mask = signs > 0
        pos_plus = pos[pos_mask]
        pos_minus = pos[~pos_mask]

        # Render positive particles (blue)
        if len(pos_plus) > 0:
            render_particles(buf, pos_plus, COLOR_POS, box_min, scale, 0, alpha)           # XY
            render_particles_xz(buf, pos_plus, COLOR_POS, box_min, scale, VIEW_SIZE, alpha)     # XZ
            render_particles_yz(buf, pos_plus, COLOR_POS, box_min, scale, 2*VIEW_SIZE, alpha)   # YZ

        # Render negative particles (red)
        if len(pos_minus) > 0:
            render_particles(buf, pos_minus, COLOR_NEG, box_min, scale, 0, alpha)          # XY
            render_particles_xz(buf, pos_minus, COLOR_NEG, box_min, scale, VIEW_SIZE, alpha)    # XZ
            render_particles_yz(buf, pos_minus, COLOR_NEG, box_min, scale, 2*VIEW_SIZE, alpha)  # YZ

        # Convert to uint8
        img_data = (buf * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='RGBA')

        # Draw title
        draw = ImageDraw.Draw(img)
        title = f"Step {step:06d}  |  z = {redshift:.2f}  |  S = {seg:.3e}  |  KE/KE0 = {ke_ratio:.4f}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except:
            font = ImageFont.load_default()

        # Black background for title
        draw.rectangle([0, 0, 850, 45], fill=(0, 0, 0, 220))
        draw.text((20, 8), title, fill=(255, 255, 255, 255), font=font)

        # Save PNG
        img.save(out_path)
        print(f"[render] frame_{step:06d}.png saved")
        return True

    except Exception as e:
        print(f"[render] Error processing {bin_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def render_particles(buf, pos, color, box_min, scale, x_offset, alpha):
    """Render XY projection"""
    x = pos[:, 0]
    y = pos[:, 1]

    ix = ((x - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32) + x_offset
    iy = ((y - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32)
    iy_flip = VIEW_SIZE - 1 - iy

    np.add.at(buf, (iy_flip, ix, 0), color[0] * alpha)
    np.add.at(buf, (iy_flip, ix, 1), color[1] * alpha)
    np.add.at(buf, (iy_flip, ix, 2), color[2] * alpha)


def render_particles_xz(buf, pos, color, box_min, scale, x_offset, alpha):
    """Render XZ projection"""
    x = pos[:, 0]
    z = pos[:, 2]

    ix = ((x - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32) + x_offset
    iz = ((z - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32)
    iz_flip = VIEW_SIZE - 1 - iz

    np.add.at(buf, (iz_flip, ix, 0), color[0] * alpha)
    np.add.at(buf, (iz_flip, ix, 1), color[1] * alpha)
    np.add.at(buf, (iz_flip, ix, 2), color[2] * alpha)


def render_particles_yz(buf, pos, color, box_min, scale, x_offset, alpha):
    """Render YZ projection"""
    y = pos[:, 1]
    z = pos[:, 2]

    iy = ((y - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32) + x_offset
    iz = ((z - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32)
    iz_flip = VIEW_SIZE - 1 - iz

    np.add.at(buf, (iz_flip, iy, 0), color[0] * alpha)
    np.add.at(buf, (iz_flip, iy, 1), color[1] * alpha)
    np.add.at(buf, (iz_flip, iy, 2), color[2] * alpha)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 render_frames.py <output_dir> [--once]")
        print("  e.g.: python3 render_frames.py /app/output/85M_2026-02-25")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    once_mode = '--once' in sys.argv

    render_data_dir = output_dir / "render_data"
    frames_dir = output_dir / "frames"

    print("=" * 60)
    print("Janus 85M Python Frame Renderer")
    print("=" * 60)
    print(f"Watching: {render_data_dir}")
    print(f"Output:   {frames_dir}")
    print(f"Mode:     {'once' if once_mode else 'continuous'}")
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

        time.sleep(1)


if __name__ == "__main__":
    main()
