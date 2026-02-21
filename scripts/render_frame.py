#!/usr/bin/env python3
"""
Janus N-body visualization — PNG 4K frame generator
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_initial_conditions(n_particles, eta, box_size, seed=42):
    """Generate random initial particle positions (matching Rust seed)"""
    np.random.seed(seed)

    n_positive = int(n_particles / (1.0 + eta))
    n_negative = n_particles - n_positive

    # Random uniform positions in [-box/2, box/2]
    pos_plus = (np.random.random((n_positive, 3)) - 0.5) * box_size
    pos_minus = (np.random.random((n_negative, 3)) - 0.5) * box_size

    return pos_plus, pos_minus

def render_frame(pos_plus, pos_minus, step, total_steps, sim_time, seg, ke_ratio,
                 n_particles, output_path, sample_size=100000):
    """Render a single frame as 4K PNG"""

    # Subsample for visualization
    if len(pos_plus) > sample_size // 2:
        idx_plus = np.random.choice(len(pos_plus), sample_size // 2, replace=False)
        pos_plus_sample = pos_plus[idx_plus]
    else:
        pos_plus_sample = pos_plus

    if len(pos_minus) > sample_size // 2:
        idx_minus = np.random.choice(len(pos_minus), sample_size // 2, replace=False)
        pos_minus_sample = pos_minus[idx_minus]
    else:
        pos_minus_sample = pos_minus

    # Create figure with 4K resolution
    fig = plt.figure(figsize=(3840/100, 2160/100), dpi=100, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Plot particles
    ax.scatter(pos_plus_sample[:, 0], pos_plus_sample[:, 1], pos_plus_sample[:, 2],
               c='blue', alpha=0.3, s=0.3, marker='.')
    ax.scatter(pos_minus_sample[:, 0], pos_minus_sample[:, 1], pos_minus_sample[:, 2],
               c='red', alpha=0.3, s=0.3, marker='.')

    # Style
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(-250, 250)
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
    title = f"Janus Cosmological Model — {n_particles/1e6:.0f}M particles"
    fig.text(0.5, 0.95, title, ha='center', va='top', fontsize=28,
             color='white', fontweight='bold')

    # Stats overlay
    stats = f"Step {step:03d}/{total_steps} | Time: {sim_time:.3f} | Seg: {seg:.4f} | KE/KE₀: {ke_ratio:.2f}"
    fig.text(0.5, 0.05, stats, ha='center', va='bottom', fontsize=20,
             color='white', family='monospace')

    # Legend
    fig.text(0.02, 0.95, "● Positive masses", ha='left', va='top', fontsize=16, color='#6666ff')
    fig.text(0.02, 0.91, "● Negative masses", ha='left', va='top', fontsize=16, color='#ff6666')

    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # Parameters matching 7M simulation
    n_particles = 7_000_000
    eta = 1.045
    box_size = 412.13  # From 7M test

    # Generate initial conditions
    print("Generating initial conditions...")
    pos_plus, pos_minus = generate_initial_conditions(n_particles, eta, box_size)

    # Render frame 0
    output_path = "/app/output/phase1c/frames/frame_000.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Rendering 4K frame...")
    render_frame(
        pos_plus, pos_minus,
        step=0, total_steps=300,
        sim_time=0.000,
        seg=0.5206,
        ke_ratio=1.00,
        n_particles=n_particles,
        output_path=output_path
    )

    print("Done!")
