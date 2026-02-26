#!/usr/bin/env python3
"""
Visualize Janus snapshot with 3 projections (XY, XZ, YZ)
Uses numpy accumulator for 85M+ particles (fast raster rendering)

New snapshot format (janus_85m.rs):
  Header: 128 bytes text "step=X time=X.XXX eta=X n=XXXXXXXX\n" + padding
  pos:    N × 3 × f32
  vel:    N × 3 × f32
  signs:  N × i8
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from pathlib import Path

# Image dimensions
VIEW_SIZE = 1280
IMG_WIDTH = 3840
IMG_HEIGHT = 1280

# Colors (RGB, 0-1 range)
COLOR_POS = np.array([0.27, 0.53, 1.0], dtype=np.float32)   # #4488FF (blue)
COLOR_NEG = np.array([1.0, 0.27, 0.13], dtype=np.float32)   # #FF4422 (red)
ALPHA = 0.15


def read_snapshot(path):
    """Read new-format snapshot binary file"""
    with open(path, 'rb') as f:
        # Header: 128 bytes text
        header = f.read(128).decode('utf-8', errors='ignore').strip()

        # Parse header: "step=X time=X.XXX eta=X n=XXXXXXXX"
        parts = {}
        for part in header.split():
            if '=' in part:
                k, v = part.split('=', 1)
                parts[k] = v

        n = int(parts.get('n', 0))
        step = int(parts.get('step', 0))
        eta = float(parts.get('eta', 1.045))
        time = float(parts.get('time', 0))

        print(f"  Header: step={step}, time={time:.3f}, eta={eta}, n={n:,}")

        # pos: N × 3 × f32
        pos = np.frombuffer(f.read(n * 3 * 4), dtype=np.float32).reshape(n, 3)

        # vel: N × 3 × f32 (skip for visualization)
        f.seek(n * 3 * 4, 1)

        # signs: N × i8
        signs = np.frombuffer(f.read(n), dtype=np.int8)

    return pos, signs, step, time, eta


def render_particles(buf, pos, color, box_min, scale, x_offset):
    """Render XY projection using numpy accumulator"""
    x = pos[:, 0]
    y = pos[:, 1]

    ix = ((x - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32) + x_offset
    iy = ((y - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32)
    iy_flip = VIEW_SIZE - 1 - iy

    np.add.at(buf, (iy_flip, ix, 0), color[0] * ALPHA)
    np.add.at(buf, (iy_flip, ix, 1), color[1] * ALPHA)
    np.add.at(buf, (iy_flip, ix, 2), color[2] * ALPHA)


def render_particles_xz(buf, pos, color, box_min, scale, x_offset):
    """Render XZ projection"""
    x = pos[:, 0]
    z = pos[:, 2]

    ix = ((x - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32) + x_offset
    iz = ((z - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32)
    iz_flip = VIEW_SIZE - 1 - iz

    np.add.at(buf, (iz_flip, ix, 0), color[0] * ALPHA)
    np.add.at(buf, (iz_flip, ix, 1), color[1] * ALPHA)
    np.add.at(buf, (iz_flip, ix, 2), color[2] * ALPHA)


def render_particles_yz(buf, pos, color, box_min, scale, x_offset):
    """Render YZ projection"""
    y = pos[:, 1]
    z = pos[:, 2]

    iy = ((y - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32) + x_offset
    iz = ((z - box_min) * scale).clip(0, VIEW_SIZE - 1).astype(np.int32)
    iz_flip = VIEW_SIZE - 1 - iz

    np.add.at(buf, (iz_flip, iy, 0), color[0] * ALPHA)
    np.add.at(buf, (iz_flip, iy, 1), color[1] * ALPHA)
    np.add.at(buf, (iz_flip, iy, 2), color[2] * ALPHA)


def visualize_snapshot(snap_path, output_path, box_size=None):
    """Create 3-panel projection visualization using fast raster rendering"""
    print(f"Reading {snap_path}...")
    pos, signs, step, time, eta = read_snapshot(snap_path)
    n = len(signs)

    # Auto-detect box size from positions
    if box_size is None:
        box_size = (pos.max() - pos.min()) * 1.05

    print(f"  N={n:,}, box={box_size:.2f}")

    # Calculate segregation
    pos_plus = pos[signs > 0]
    pos_minus = pos[signs <= 0]

    com_plus = pos_plus.mean(axis=0) if len(pos_plus) > 0 else np.zeros(3)
    com_minus = pos_minus.mean(axis=0) if len(pos_minus) > 0 else np.zeros(3)
    seg = np.linalg.norm(com_plus - com_minus) / box_size

    print(f"  N+={len(pos_plus):,}, N-={len(pos_minus):,}, S={seg:.6f}")

    # Create RGBA buffer (black background)
    buf = np.zeros((IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.float32)
    buf[:, :, 3] = 1.0  # Opaque black background

    # Coordinate mapping
    box_min = -box_size / 2.0
    scale = (VIEW_SIZE - 1) / box_size

    # Render positive particles (blue)
    if len(pos_plus) > 0:
        render_particles(buf, pos_plus, COLOR_POS, box_min, scale, 0)
        render_particles_xz(buf, pos_plus, COLOR_POS, box_min, scale, VIEW_SIZE)
        render_particles_yz(buf, pos_plus, COLOR_POS, box_min, scale, 2*VIEW_SIZE)

    # Render negative particles (red)
    if len(pos_minus) > 0:
        render_particles(buf, pos_minus, COLOR_NEG, box_min, scale, 0)
        render_particles_xz(buf, pos_minus, COLOR_NEG, box_min, scale, VIEW_SIZE)
        render_particles_yz(buf, pos_minus, COLOR_NEG, box_min, scale, 2*VIEW_SIZE)

    # Convert to uint8
    img_data = (buf * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='RGBA')

    # Draw title
    draw = ImageDraw.Draw(img)
    title = f"Step {step:06d}  |  t = {time:.3f}  |  S = {seg:.3e}  |  N = {n:,}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        font = ImageFont.load_default()

    # Black background for title
    draw.rectangle([0, 0, 850, 45], fill=(0, 0, 0, 220))
    draw.text((20, 8), title, fill=(255, 255, 255, 255), font=font)

    # Panel labels
    labels = [("XY", 640), ("XZ", 1920), ("YZ", 3200)]
    for label, x_center in labels:
        draw.text((x_center - 20, IMG_HEIGHT - 40), label, fill=(255, 255, 255, 200), font=font)

    # Save PNG
    img.save(output_path)
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python visualize_snapshot.py <snapshot.bin> <output.png> [box_size]")
        print("  box_size: optional, auto-detected if not specified")
        sys.exit(1)

    snap_path = sys.argv[1]
    out_path = sys.argv[2]
    box_size = float(sys.argv[3]) if len(sys.argv) > 3 else None

    visualize_snapshot(snap_path, out_path, box_size)
