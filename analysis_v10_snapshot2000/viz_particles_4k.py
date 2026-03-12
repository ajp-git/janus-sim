#!/usr/bin/env python3
"""
3D Visualization of ACTUAL Janus Particles - Step 2000
2M particles from 20M total - FIXED coordinates
"""
import numpy as np
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil

SNAPSHOT_PATH = "/mnt/T2/janus-sim/output/janus_v10_highres/snapshots/snap_002000.bin"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot2000"
L_BOX = 200.0
N_RENDER = 2_000_000

print("=" * 60)
print("LOADING 20M PARTICLES FROM SNAPSHOT 2000")
print("=" * 60)

with open(SNAPSHOT_PATH, 'rb') as f:
    f.read(8)
    data = np.frombuffer(f.read(), dtype=np.float32)

n_total = len(data) // 4
print(f"Total particles: {n_total:,}")

x_all = data[0::4]
y_all = data[1::4]
z_all = data[2::4]
signs_all = data[3::4]

# FIX: Shift coordinates from [-100,100] to [0,200]
x_all = x_all + L_BOX / 2
y_all = y_all + L_BOX / 2
z_all = z_all + L_BOX / 2

print(f"Coordinates shifted to [0, {L_BOX}]")
print(f"x: [{x_all.min():.1f}, {x_all.max():.1f}]")
print(f"y: [{y_all.min():.1f}, {y_all.max():.1f}]")
print(f"z: [{z_all.min():.1f}, {z_all.max():.1f}]")

pos_mask = signs_all > 0
neg_mask = signs_all < 0

x_pos_all, y_pos_all, z_pos_all = x_all[pos_mask], y_all[pos_mask], z_all[pos_mask]
x_neg_all, y_neg_all, z_neg_all = x_all[neg_mask], y_all[neg_mask], z_all[neg_mask]

n_pos_total = len(x_pos_all)
n_neg_total = len(x_neg_all)

# Subsample
frac = N_RENDER / n_total
n_pos_render = int(n_pos_total * frac)
n_neg_render = int(n_neg_total * frac)

print(f"\nSubsampling to {N_RENDER:,} particles:")
print(f"  N+ = {n_pos_render:,}, N- = {n_neg_render:,}")

np.random.seed(42)
idx_pos = np.random.choice(n_pos_total, n_pos_render, replace=False)
idx_neg = np.random.choice(n_neg_total, n_neg_render, replace=False)

x_pos, y_pos, z_pos = x_pos_all[idx_pos], y_pos_all[idx_pos], z_pos_all[idx_pos]
x_neg, y_neg, z_neg = x_neg_all[idx_neg], y_neg_all[idx_neg], z_neg_all[idx_neg]

del data, x_all, y_all, z_all, signs_all
del x_pos_all, y_pos_all, z_pos_all, x_neg_all, y_neg_all, z_neg_all

COLOR_POS = '#ff7f0e'
COLOR_NEG = '#00d4ff'

# ============================================================
# Static 4K image
# ============================================================
print("\nCreating static 4K image...")
fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_pos, y_pos, z_pos, c=COLOR_POS, alpha=0.15, s=0.1, label=f'+ ({n_pos_render:,})')
ax.scatter(x_neg, y_neg, z_neg, c=COLOR_NEG, alpha=0.15, s=0.1, label=f'- ({n_neg_render:,})')

ax.set_xlabel('x (Mpc)', fontsize=12)
ax.set_ylabel('y (Mpc)', fontsize=12)
ax.set_zlabel('z (Mpc)', fontsize=12)
ax.set_title(f'Janus Particles (Step 2000, {N_RENDER/1e6:.0f}M of 20M)', fontsize=16)
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
print("\nCreating 4K rotation animation (360 frames)...")
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

    elev = 25 + 5 * np.sin(i * 2 * np.pi / n_frames)
    azim = i * 360 / n_frames
    ax.view_init(elev=elev, azim=azim)

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

    if (i + 1) % 60 == 0:
        print(f"  Frame {i+1}/{n_frames}")

print("\nCompiling 4K video...")
import subprocess

video_path = f"{OUTPUT_DIR}/janus_particles_rotation_4k.mp4"
cmd = [
    'ffmpeg', '-y', '-framerate', '30',
    '-i', f'{frames_dir}/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-crf', '18', '-preset', 'slow',
    '-vf', 'scale=3840:2160',
    video_path
]
subprocess.run(cmd, capture_output=True)

if os.path.exists(video_path):
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Saved: janus_particles_rotation_4k.mp4 ({file_size:.1f} MB)")
    shutil.rmtree(frames_dir)

print("\nDONE!")
