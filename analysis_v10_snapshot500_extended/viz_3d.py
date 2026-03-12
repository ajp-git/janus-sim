#!/usr/bin/env python3
"""
3D Visualization of Janus Polarization Domains
High-quality 4K rendering with smooth rotation animation
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import os
import shutil

INPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500"
OUTPUT_DIR = "/mnt/T2/janus-sim/analysis_v10_snapshot500_extended"
L_BOX = 200.0

print("Loading polarization field...")
P = np.load(f"{INPUT_DIR}/polarization_field.npy").astype(np.float64)

# Subsample for visualization (higher resolution than before)
sub = 5
P_sub = P[::sub, ::sub, ::sub]
P_smooth = gaussian_filter(P_sub, sigma=0.5)

ng = P_smooth.shape[0]
cell = L_BOX / ng

print(f"Subsampled grid: {ng}³")
print(f"Cell size: {cell:.2f} Mpc")

# Create coordinate arrays
x = np.arange(ng) * cell + cell/2
y = np.arange(ng) * cell + cell/2
z = np.arange(ng) * cell + cell/2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Get positions of high polarization regions
pos_mask = P_smooth > 0.7
neg_mask = P_smooth < -0.7

x_pos, y_pos, z_pos = X[pos_mask].copy(), Y[pos_mask].copy(), Z[pos_mask].copy()
x_neg, y_neg, z_neg = X[neg_mask].copy(), Y[neg_mask].copy(), Z[neg_mask].copy()

print(f"Positive voxels: {len(x_pos)}")
print(f"Negative voxels: {len(x_neg)}")

# Add jitter to remove grid artifact
noise = cell * 0.35
np.random.seed(42)  # Reproducibility

x_pos += np.random.uniform(-noise, noise, len(x_pos))
y_pos += np.random.uniform(-noise, noise, len(y_pos))
z_pos += np.random.uniform(-noise, noise, len(z_pos))

x_neg += np.random.uniform(-noise, noise, len(x_neg))
y_neg += np.random.uniform(-noise, noise, len(y_neg))
z_neg += np.random.uniform(-noise, noise, len(z_neg))

# Show ALL points (no subsampling)
print(f"Points for rendering: {len(x_pos)} positive, {len(x_neg)} negative (ALL)")

# Colors
COLOR_POS = '#ff7f0e'  # Orange
COLOR_NEG = '#00d4ff'  # Cyan

# ============================================================
# Static 4K image
# ============================================================
print("\nCreating static 4K image...")
fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_pos, y_pos, z_pos, c=COLOR_POS, alpha=0.4, s=2, label='P > 0.7 (+)')
ax.scatter(x_neg, y_neg, z_neg, c=COLOR_NEG, alpha=0.4, s=2, label='P < -0.7 (-)')

ax.set_xlabel('x (Mpc)', fontsize=12)
ax.set_ylabel('y (Mpc)', fontsize=12)
ax.set_zlabel('z (Mpc)', fontsize=12)
ax.set_title('Janus Polarization Domains (V10 step 500)', fontsize=16)
ax.legend(loc='upper right', fontsize=10)

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
plt.savefig(f"{OUTPUT_DIR}/janus_3d_scatter_4k.png", dpi=200, facecolor='black')
plt.close()
print("Saved: janus_3d_scatter_4k.png")

# ============================================================
# Multi-slice visualization
# ============================================================
print("\nCreating multi-slice visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

slices = [32, 64, 96, 128, 160, 192]  # z positions in original 256³ grid

for idx, (ax, z_slice) in enumerate(zip(axes.flat, slices)):
    P_slice = P[:, :, z_slice]
    im = ax.imshow(P_slice.T, origin='lower', cmap='RdBu_r',
                   vmin=-1, vmax=1, extent=[0, L_BOX, 0, L_BOX])
    ax.set_title(f'z = {z_slice * L_BOX / 256:.1f} Mpc')
    ax.set_xlabel('x (Mpc)')
    ax.set_ylabel('y (Mpc)')

fig.colorbar(im, ax=axes, label='Polarization P', shrink=0.8)
fig.suptitle('Polarization Field Slices (V10 step 500)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/janus_multislice.png", dpi=150)
plt.close()
print("Saved: janus_multislice.png")

# ============================================================
# 4K Rotation Animation
# ============================================================
print("\nCreating 4K rotation animation (360 frames)...")
frames_dir = f"{OUTPUT_DIR}/frames"
os.makedirs(frames_dir, exist_ok=True)

n_frames = 360

for i in range(n_frames):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_pos, y_pos, z_pos, c=COLOR_POS, alpha=0.4, s=2)
    ax.scatter(x_neg, y_neg, z_neg, c=COLOR_NEG, alpha=0.4, s=2)

    ax.set_xlabel('x (Mpc)', fontsize=12)
    ax.set_ylabel('y (Mpc)', fontsize=12)
    ax.set_zlabel('z (Mpc)', fontsize=12)
    ax.set_xlim(0, L_BOX)
    ax.set_ylim(0, L_BOX)
    ax.set_zlim(0, L_BOX)

    # Cinematic camera path with vertical oscillation
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

    # Remove panes for cleaner look
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

# Compile 4K video
print("\nCompiling 4K video...")
import subprocess

video_path = f"{OUTPUT_DIR}/janus_polarization_rotation_4k.mp4"
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
    print(f"Saved: janus_polarization_rotation_4k.mp4 ({file_size:.1f} MB)")
    # Clean up frames directory
    shutil.rmtree(frames_dir)
    print("Cleaned up frame directory")
else:
    print(f"Video creation failed: {result.stderr}")

print("\nDone!")
