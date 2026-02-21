#!/usr/bin/env python3
"""
Janus N-body visualization — Overnight run frame renderer
Renders a single frame from binary snapshot with specified visual parameters.
"""
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_snapshot(path):
    """Read binary snapshot file"""
    with open(path, 'rb') as f:
        # Read header
        n_particles = struct.unpack('<Q', f.read(8))[0]
        n_positive = struct.unpack('<Q', f.read(8))[0]
        eta = struct.unpack('<d', f.read(8))[0]
        box_size = struct.unpack('<d', f.read(8))[0]
        step = struct.unpack('<Q', f.read(8))[0]
        sim_time = struct.unpack('<d', f.read(8))[0]
        seg = struct.unpack('<d', f.read(8))[0]
        ke_ratio = struct.unpack('<d', f.read(8))[0]

        # Read positions
        positions = np.frombuffer(f.read(), dtype=np.float64).reshape(-1, 3)

    return {
        'n_particles': n_particles,
        'n_positive': n_positive,
        'n_negative': n_particles - n_positive,
        'eta': eta,
        'box_size': box_size,
        'step': step,
        'time': sim_time,
        'seg': seg,
        'ke_ratio': ke_ratio,
        'positions': positions
    }

def render_frame(snapshot_path, output_path, eta, n_particles, step, sim_time, seg, ke_ratio):
    """Render a single 4K frame"""

    # Read snapshot
    data = read_snapshot(snapshot_path)
    positions = data['positions']
    n_positive = data['n_positive']
    box_size = data['box_size']

    # Split by sign
    pos_plus = positions[:n_positive]
    pos_minus = positions[n_positive:]

    # Subsample for visualization (max 100K total)
    sample_size = 50000
    if len(pos_plus) > sample_size:
        idx = np.random.choice(len(pos_plus), sample_size, replace=False)
        pos_plus = pos_plus[idx]
    if len(pos_minus) > sample_size:
        idx = np.random.choice(len(pos_minus), sample_size, replace=False)
        pos_minus = pos_minus[idx]

    # Create figure with 4K resolution
    fig = plt.figure(figsize=(3840/100, 2160/100), dpi=100, facecolor='black')
    # Center the 3D axes in the figure
    ax = fig.add_axes([0.15, 0.1, 0.7, 0.8], projection='3d', facecolor='black')

    # Visual parameters as specified
    point_size = 1.5
    alpha = 0.6
    color_plus = '#4488FF'   # Blue for positive masses
    color_minus = '#FF4444'  # Red for negative masses

    # Plot particles
    ax.scatter(pos_plus[:, 0], pos_plus[:, 1], pos_plus[:, 2],
               c=color_plus, alpha=alpha, s=point_size, marker='.')
    ax.scatter(pos_minus[:, 0], pos_minus[:, 1], pos_minus[:, 2],
               c=color_minus, alpha=alpha, s=point_size, marker='.')

    # Set axis limits
    half_box = box_size / 2
    ax.set_xlim(-half_box, half_box)
    ax.set_ylim(-half_box, half_box)
    ax.set_zlim(-half_box, half_box)

    # Center the view
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    ax.set_axis_off()

    # Remove grid and panes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Title
    title = f"Janus Cosmological Model — {n_particles/1e3:.0f}K particles | η={eta:.3f}"
    fig.text(0.5, 0.95, title, ha='center', va='top', fontsize=28,
             color='white', fontweight='bold')

    # Stats overlay
    stats = f"Step {step:05d} | Time: {sim_time:.3f} | Seg: {seg:.4f} | KE/KE₀: {ke_ratio:.2f}"
    fig.text(0.5, 0.05, stats, ha='center', va='bottom', fontsize=20,
             color='white', family='monospace')

    # Legend
    fig.text(0.02, 0.95, "● Positive masses", ha='left', va='top',
             fontsize=16, color=color_plus)
    fig.text(0.02, 0.91, "● Negative masses", ha='left', va='top',
             fontsize=16, color=color_minus)

    plt.savefig(output_path, facecolor='black', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("Usage: render_overnight.py <snapshot> <output> <eta> <n_particles> <step> <time> <seg> <ke_ratio>")
        sys.exit(1)

    snapshot_path = sys.argv[1]
    output_path = sys.argv[2]
    eta = float(sys.argv[3])
    n_particles = int(sys.argv[4])
    step = int(sys.argv[5])
    sim_time = float(sys.argv[6])
    seg = float(sys.argv[7])
    ke_ratio = float(sys.argv[8])

    render_frame(snapshot_path, output_path, eta, n_particles, step, sim_time, seg, ke_ratio)
