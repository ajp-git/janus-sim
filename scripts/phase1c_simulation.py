#!/usr/bin/env python3
"""
PHASE 1C — Janus N-body Simulation with Real-time Visualization

10M particles, η = 1.045, 300 steps
Generates 4K PNG frames during simulation

Uses Particle-Mesh (PM) method for O(N + N_grid log N_grid) complexity.
"""

import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta

# Check for required packages
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.fft import fftn, ifftn
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install with: pip install numpy scipy matplotlib")
    sys.exit(1)

print("=" * 70)
print("PHASE 1C — Janus N-body Simulation (10M particles)")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# PARAMETERS
# =============================================================================
N_TOTAL = 10_000_000
ETA = 1.045
N_STEPS = 300
DT = 0.003
BOX_SIZE = 100.0
GRID_SIZE = 256  # PM grid resolution
SOFTENING = BOX_SIZE / GRID_SIZE * 2  # Softening ~ 2 grid cells

# Visualization subsampling
N_VIS_POS = 100_000  # Positive particles to visualize
N_VIS_NEG = 100_000  # Negative particles to visualize

# Output directories
SNAPSHOT_DIR = "output/phase1c/snapshots"
FRAME_DIR = "output/phase1c/frames"

# Particle counts from η
N_POSITIVE = int(N_TOTAL / (1.0 + ETA))
N_NEGATIVE = N_TOTAL - N_POSITIVE

print(f"\nParameters:")
print(f"  N total     = {N_TOTAL:,}")
print(f"  N positive  = {N_POSITIVE:,}")
print(f"  N negative  = {N_NEGATIVE:,}")
print(f"  η = N₋/N₊   = {N_NEGATIVE/N_POSITIVE:.4f}")
print(f"  Steps       = {N_STEPS}")
print(f"  dt          = {DT}")
print(f"  Box size    = {BOX_SIZE}")
print(f"  Grid        = {GRID_SIZE}³")
print(f"  Softening   = {SOFTENING:.4f}")

# =============================================================================
# INITIALIZATION
# =============================================================================
print("\nInitializing particles...")
np.random.seed(42)

# Positions: uniform random in box
pos = (np.random.rand(N_TOTAL, 3).astype(np.float32) - 0.5) * BOX_SIZE

# Velocities: small random (thermal)
vel = (np.random.rand(N_TOTAL, 3).astype(np.float32) - 0.5) * 0.1

# Mass signs: first N_POSITIVE are positive, rest negative
signs = np.ones(N_TOTAL, dtype=np.float32)
signs[N_POSITIVE:] = -1.0

# Masses: unit mass
masses = np.ones(N_TOTAL, dtype=np.float32)

# Indices for subsampling visualization
vis_idx_pos = np.random.choice(N_POSITIVE, min(N_VIS_POS, N_POSITIVE), replace=False)
vis_idx_neg = np.random.choice(N_NEGATIVE, min(N_VIS_NEG, N_NEGATIVE), replace=False) + N_POSITIVE

print(f"  Visualization subsample: {len(vis_idx_pos):,} pos + {len(vis_idx_neg):,} neg")

# =============================================================================
# PARTICLE-MESH FORCE CALCULATION
# =============================================================================
def compute_forces_pm(pos, signs, masses, box_size, grid_size, softening):
    """
    Compute gravitational forces using Particle-Mesh method.

    Janus rules:
      - Same sign masses attract
      - Opposite sign masses repel

    We compute two density fields (positive and negative) and derive
    the potential for each, then compute forces accounting for Janus rules.
    """
    N = len(pos)
    cell_size = box_size / grid_size

    # Wrap positions to [0, box_size)
    pos_wrapped = (pos + box_size/2) % box_size

    # Grid indices
    idx = (pos_wrapped / cell_size).astype(np.int32) % grid_size

    # Compute density fields for positive and negative masses
    rho_pos = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    rho_neg = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # CIC (Cloud-In-Cell) assignment for smoother density
    for i in range(N):
        ix, iy, iz = idx[i]
        m = masses[i]
        if signs[i] > 0:
            rho_pos[ix, iy, iz] += m
        else:
            rho_neg[ix, iy, iz] += m

    # Normalize to density
    cell_vol = cell_size ** 3
    rho_pos /= cell_vol
    rho_neg /= cell_vol

    # Green's function in k-space (with softening)
    kx = np.fft.fftfreq(grid_size, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(grid_size, d=cell_size) * 2 * np.pi
    kz = np.fft.fftfreq(grid_size, d=cell_size) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2 + (2*np.pi/box_size * softening)**2
    K2[0, 0, 0] = 1.0  # Avoid division by zero

    # Poisson equation: ∇²φ = 4πGρ → φ̂ = -4πGρ̂/k²
    # In code units G=1
    green = -4 * np.pi / K2
    green[0, 0, 0] = 0.0  # Zero mode

    # Solve for potentials
    rho_pos_k = fftn(rho_pos)
    rho_neg_k = fftn(rho_neg)

    phi_pos = np.real(ifftn(green * rho_pos_k))  # Potential from positive masses
    phi_neg = np.real(ifftn(green * rho_neg_k))  # Potential from negative masses

    # Compute acceleration field components via finite differences
    # For positive particles: attracted by pos, repelled by neg
    # For negative particles: attracted by neg, repelled by pos

    # Gradient of potentials (central difference)
    def gradient(phi):
        gx = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * cell_size)
        gy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * cell_size)
        gz = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * cell_size)
        return gx, gy, gz

    grad_phi_pos = gradient(phi_pos)
    grad_phi_neg = gradient(phi_neg)

    # Interpolate forces to particles (NGP - Nearest Grid Point)
    forces = np.zeros_like(pos)

    for i in range(N):
        ix, iy, iz = idx[i]

        if signs[i] > 0:
            # Positive particle: attracted by positive field, repelled by negative
            forces[i, 0] = -grad_phi_pos[0][ix, iy, iz] + grad_phi_neg[0][ix, iy, iz]
            forces[i, 1] = -grad_phi_pos[1][ix, iy, iz] + grad_phi_neg[1][ix, iy, iz]
            forces[i, 2] = -grad_phi_pos[2][ix, iy, iz] + grad_phi_neg[2][ix, iy, iz]
        else:
            # Negative particle: attracted by negative field, repelled by positive
            forces[i, 0] = -grad_phi_neg[0][ix, iy, iz] + grad_phi_pos[0][ix, iy, iz]
            forces[i, 1] = -grad_phi_neg[1][ix, iy, iz] + grad_phi_pos[1][ix, iy, iz]
            forces[i, 2] = -grad_phi_neg[2][ix, iy, iz] + grad_phi_pos[2][ix, iy, iz]

    return forces

# =============================================================================
# DIAGNOSTICS
# =============================================================================
def compute_kinetic_energy(vel, masses):
    """Total kinetic energy."""
    return 0.5 * np.sum(masses * np.sum(vel**2, axis=1))

def compute_segregation(pos, signs):
    """Distance between centers of mass of positive and negative particles."""
    pos_com = np.mean(pos[signs > 0], axis=0)
    neg_com = np.mean(pos[signs < 0], axis=0)
    return np.linalg.norm(pos_com - neg_com)

# =============================================================================
# VISUALIZATION
# =============================================================================
def generate_frame(pos, signs, step, total_steps, sim_time, segregation, ke_ratio,
                   vis_idx_pos, vis_idx_neg, output_path):
    """Generate 4K PNG frame."""
    fig, ax = plt.subplots(figsize=(3840/100, 2160/100), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Extract visualization subset
    pos_vis_pos = pos[vis_idx_pos]
    pos_vis_neg = pos[vis_idx_neg]

    # Project to 2D (x-y plane)
    ax.scatter(pos_vis_pos[:, 0], pos_vis_pos[:, 1],
               c='#4488ff', s=0.5, alpha=0.3, marker='.', linewidths=0)
    ax.scatter(pos_vis_neg[:, 0], pos_vis_neg[:, 1],
               c='#ff4444', s=0.5, alpha=0.3, marker='.', linewidths=0)

    # Styling
    ax.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Overlay text
    overlay = f"Step {step:03d}/{total_steps} | Time: {sim_time:.3f} | Segregation: {segregation:.3f} | KE ratio: {ke_ratio:.2f}"
    ax.text(0.5, 0.02, overlay, transform=ax.transAxes, fontsize=24,
            color='white', ha='center', va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Title
    ax.text(0.5, 0.98, "JANUS COSMOLOGICAL MODEL — Phase 1c (10M particles)",
            transform=ax.transAxes, fontsize=32, color='white',
            ha='center', va='top', fontweight='bold')

    # Legend
    ax.text(0.02, 0.98, "● Positive masses", transform=ax.transAxes,
            fontsize=18, color='#4488ff', va='top')
    ax.text(0.02, 0.94, "● Negative masses", transform=ax.transAxes,
            fontsize=18, color='#ff4444', va='top')

    plt.savefig(output_path, facecolor='black', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# =============================================================================
# SAVE SNAPSHOT
# =============================================================================
def save_snapshot(pos, vel, signs, step, sim_time, segregation, ke, output_path):
    """Save snapshot as compressed NPZ."""
    np.savez_compressed(output_path,
                        pos=pos.astype(np.float32),
                        vel=vel.astype(np.float32),
                        signs=signs.astype(np.int8),
                        step=step,
                        time=sim_time,
                        segregation=segregation,
                        kinetic_energy=ke)

# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================
print("\nComputing initial diagnostics...")
ke_initial = compute_kinetic_energy(vel, masses)
seg_initial = compute_segregation(pos, signs)
print(f"  Initial KE: {ke_initial:.4e}")
print(f"  Initial segregation: {seg_initial:.4f}")

print(f"\n{'='*70}")
print("Starting simulation...")
print(f"{'='*70}\n")

print(f"{'Step':>6} {'Time':>10} {'Segregation':>12} {'KE':>14} {'KE/KE0':>10} {'Frame':>8} {'ETA':>12}")
print("-" * 80)

sim_time = 0.0
start_time = time.time()
step_times = []

for step in range(N_STEPS + 1):
    step_start = time.time()

    # Diagnostics
    ke = compute_kinetic_energy(vel, masses)
    ke_ratio = ke / ke_initial if ke_initial > 0 else 1.0
    segregation = compute_segregation(pos, signs)

    # Save snapshot
    snapshot_path = f"{SNAPSHOT_DIR}/snapshot_{step:03d}.npz"
    save_snapshot(pos, vel, signs, step, sim_time, segregation, ke, snapshot_path)

    # Generate frame
    frame_path = f"{FRAME_DIR}/frame_{step:03d}.png"
    generate_frame(pos, signs, step, N_STEPS, sim_time, segregation, ke_ratio,
                   vis_idx_pos, vis_idx_neg, frame_path)

    # ETA calculation
    step_time = time.time() - step_start
    step_times.append(step_time)
    if len(step_times) > 10:
        step_times = step_times[-10:]  # Keep last 10 for average
    avg_step_time = np.mean(step_times)
    remaining_steps = N_STEPS - step
    eta_seconds = avg_step_time * remaining_steps
    eta_str = str(timedelta(seconds=int(eta_seconds)))

    print(f"{step:>6} {sim_time:>10.4f} {segregation:>12.4f} {ke:>14.4e} {ke_ratio:>10.2f} {'OK':>8} {eta_str:>12}")
    sys.stdout.flush()

    if step >= N_STEPS:
        break

    # === LEAPFROG INTEGRATION ===

    # Half-step velocity update
    forces = compute_forces_pm(pos, signs, masses, BOX_SIZE, GRID_SIZE, SOFTENING)
    vel = vel + 0.5 * DT * forces

    # Full-step position update
    pos = pos + DT * vel

    # Periodic boundary conditions
    pos = ((pos + BOX_SIZE/2) % BOX_SIZE) - BOX_SIZE/2

    # Recompute forces at new positions
    forces = compute_forces_pm(pos, signs, masses, BOX_SIZE, GRID_SIZE, SOFTENING)

    # Half-step velocity update
    vel = vel + 0.5 * DT * forces

    sim_time += DT

# =============================================================================
# FINAL SUMMARY
# =============================================================================
total_time = time.time() - start_time
print(f"\n{'='*70}")
print("SIMULATION COMPLETE")
print(f"{'='*70}")
print(f"\n  Total runtime: {timedelta(seconds=int(total_time))}")
print(f"  Frames generated: {N_STEPS + 1}")
print(f"  Initial segregation: {seg_initial:.4f}")
print(f"  Final segregation: {segregation:.4f}")
print(f"  Segregation change: {(segregation - seg_initial) / seg_initial * 100:+.2f}%")
print(f"  Final KE ratio: {ke_ratio:.2f}")

print(f"\nOutput files:")
print(f"  Snapshots: {SNAPSHOT_DIR}/snapshot_*.npz")
print(f"  Frames: {FRAME_DIR}/frame_*.png")

print(f"\nTo create video:")
print(f"  ffmpeg -framerate 24 -i {FRAME_DIR}/frame_%03d.png \\")
print(f"         -c:v libx264 -crf 18 -pix_fmt yuv420p \\")
print(f"         output/phase1c/janus_phase1c.mp4")

# Save summary JSON
import json
summary = {
    "phase": "1c",
    "n_particles": N_TOTAL,
    "n_positive": N_POSITIVE,
    "n_negative": N_NEGATIVE,
    "eta": ETA,
    "n_steps": N_STEPS,
    "dt": DT,
    "box_size": BOX_SIZE,
    "grid_size": GRID_SIZE,
    "initial_segregation": float(seg_initial),
    "final_segregation": float(segregation),
    "initial_ke": float(ke_initial),
    "final_ke": float(ke),
    "ke_ratio": float(ke_ratio),
    "runtime_seconds": total_time,
    "completed": datetime.now().isoformat()
}
with open("output/phase1c/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary: output/phase1c/summary.json")

print(f"\n{'='*70}")
print("Done!")
print(f"{'='*70}\n")
