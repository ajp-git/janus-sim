#!/usr/bin/env python3
"""
3D Visualization of ACTUAL Janus Particles - Step 1000
2M particles from 20M total
"""
import numpy as np
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil

SNAPSHOT_PATH = "/mnt/T2/janus-sim/output/janus_v10_highres/snapshots/snap_001000.bin"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot1000"
L_BOX = 200.0
N_RENDER = 2_000_000  # 2M particles to render

print("=" * 60)
print("LOADING 20M PARTICLES FROM SNAPSHOT")
print("=" * 60)

with open(SNAPSHOT_PATH, 'rb') as f:
    f.read(8)  # Skip header
    data = np.frombuffer(f.read(), dtype=np.float32)

n_total = len(data) // 4
print(f"Total particles: {n_total:,}")

x_all = data[0::4]
y_all = data[1::4]
z_all = data[2::4]
signs_all = data[3::4]

# Separate positive and negative
pos_mask = signs_all > 0
neg_mask = signs_all < 0

x_pos_all = x_all[pos_mask]
y_pos_all = y_all[pos_mask]
z_pos_all = z_all[pos_mask]

x_neg_all = x_all[neg_mask]
y_neg_all = y_all[neg_mask]
z_neg_all = z_all[neg_mask]

n_pos_total = len(x_pos_all)
n_neg_total = len(x_neg_all)
print(f"N+ = {n_pos_total:,}, N- = {n_neg_total:,}")

# Subsample proportionally to get 2M total
frac = N_RENDER / n_total
n_pos_render = int(n_pos_total * frac)
n_neg_render = int(n_neg_total * frac)

print(f"\nSubsampling to {N_RENDER:,} particles:")
print(f"  N+ render = {n_pos_render:,}")
print(f"  N- render = {n_neg_render:,}")

np.random.seed(42)
idx_pos = np.random.choice(n_pos_total, n_pos_render, replace=False)
idx_neg = np.random.choice(n_neg_total, n_neg_render, replace=False)

x_pos = x_pos_all[idx_pos]
y_pos = y_pos_all[idx_pos]
z_pos = z_pos_all[idx_pos]

x_neg = x_neg_all[idx_neg]
y_neg = y_neg_all[idx_neg]
z_neg = z_neg_all[idx_neg]

# Free memory
del data, x_all, y_all, z_all, signs_all
del x_pos_all, y_pos_all, z_pos_all
del x_neg_all, y_neg_all, z_neg_all

print(f"\nTotal points for rendering: {len(x_pos) + len(x_neg):,}")

# Colors
COLOR_POS = '#ff7f0e'  # Orange
COLOR_NEG = '#00d4ff'  # Cyan

# ============================================================
# Static 4K image
# ============================================================
print("\n" + "=" * 60)
print("Creating static 4K image (2M particles)...")
print("=" * 60)

fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
ax = fig.add_subplot(111, projection='3d')

# Use smaller marker size for 2M particles
ax.scatter(x_pos, y_pos, z_pos, c=COLOR_POS, alpha=0.15, s=0.1, label=f'+ ({n_pos_render:,})')
ax.scatter(x_neg, y_neg, z_neg, c=COLOR_NEG, alpha=0.15, s=0.1, label=f'- ({n_neg_render:,})')

ax.set_xlabel('x (Mpc)', fontsize=12)
ax.set_ylabel('y (Mpc)', fontsize=12)
ax.set_zlabel('z (Mpc)', fontsize=12)
ax.set_title(f'Janus Particles (Step 1000, {N_RENDER/1e6:.0f}M of 20M)', fontsize=16)
ax.legend(loc='upper right', fontsize=10, markerscale=10)

ax.set_xlim(0, L_BOX)
ax.set_ylim(0, L_BOX)
ax.set_zlim(0, L_BOX)

ax.view_init(elev=25, azim=45)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(colors='white')
ax.title.set_color('white')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/janus_particles_4k.png", dpi=200, facecolor='black')
plt.close()
print("Saved: janus_particles_4k.png")

# ============================================================
# 4K Rotation Animation
# ============================================================
print("\n" + "=" * 60)
print("Creating 4K rotation animation (360 frames, 2M particles)...")
print("=" * 60)

frames_dir = f"{OUTPUT_DIR}/frames_particles"
os.makedirs(frames_dir, exist_ok=True)

n_frames = 360

for i in range(n_frames):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_pos, y_pos, z_pos, c=COLOR_POS, alpha=0.15, s=0.1)
    ax.scatter(x_neg, y_neg, z_neg, c=COLOR_NEG, alpha=0.15, s=0.1)

    ax.set_xlabel('x (Mpc)', fontsize=12)
    ax.set_ylabel('y (Mpc)', fontsize=12)
    ax.set_zlabel('z (Mpc)', fontsize=12)
    ax.set_xlim(0, L_BOX)
    ax.set_ylim(0, L_BOX)
    ax.set_zlim(0, L_BOX)

    # Cinematic camera path
    elev = 25 + 5 * np.sin(i * 2 * np.pi / n_frames)
    azim = i * 360 / n_frames
    ax.view_init(elev=elev, azim=azim)

    # Dark theme
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig(f"{frames_dir}/frame_{i:04d}.png", dpi=200, facecolor='black')
    plt.close()

    if (i + 1) % 30 == 0:
        print(f"  Frame {i+1}/{n_frames}")

# Compile 4K video
print("\nCompiling 4K video...")
import subprocess

video_path = f"{OUTPUT_DIR}/janus_particles_rotation_4k.mp4"
cmd = [
    'ffmpeg', '-y',
    '-framerate', '30',
    '-i', f'{frames_dir}/frame_%04d.png',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-crf', '18',
    '-preset', 'slow',
    '-vf', 'scale=3840:2160',
    video_path
]
result = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(video_path):
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Saved: janus_particles_rotation_4k.mp4 ({file_size:.1f} MB)")
    shutil.rmtree(frames_dir)
    print("Cleaned up frame directory")
else:
    print(f"Video creation failed: {result.stderr}")

print("\n" + "=" * 60)
print("DONE - 2M PARTICLES RENDERED")
print("=" * 60)
