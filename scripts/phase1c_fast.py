#!/usr/bin/env python3
"""
PHASE 1C — Janus N-body Simulation (FAST VECTORIZED VERSION)

10M particles, η = 1.045, 300 steps
Uses numpy vectorization for O(N) PM force calculation.
"""

import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.fft import rfftn, irfftn
except ImportError as e:
    print(f"Missing: {e}. Install: pip install numpy scipy matplotlib")
    sys.exit(1)

print("=" * 70)
print("PHASE 1C — Janus N-body (10M particles, VECTORIZED)")
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
GRID_SIZE = 128  # Smaller grid for speed (was 256)
SOFTENING = BOX_SIZE / GRID_SIZE * 2

N_VIS_POS = 100_000
N_VIS_NEG = 100_000

SNAPSHOT_DIR = "output/phase1c/snapshots"
FRAME_DIR = "output/phase1c/frames"

N_POSITIVE = int(N_TOTAL / (1.0 + ETA))
N_NEGATIVE = N_TOTAL - N_POSITIVE

print(f"\nParameters:")
print(f"  N = {N_TOTAL:,} ({N_POSITIVE:,}+ / {N_NEGATIVE:,}-)")
print(f"  η = {N_NEGATIVE/N_POSITIVE:.4f}")
print(f"  Steps = {N_STEPS}, dt = {DT}")
print(f"  Grid = {GRID_SIZE}³")

# =============================================================================
# INITIALIZATION
# =============================================================================
print("\nInitializing...")
np.random.seed(42)

pos = (np.random.rand(N_TOTAL, 3).astype(np.float32) - 0.5) * BOX_SIZE
vel = (np.random.rand(N_TOTAL, 3).astype(np.float32) - 0.5) * 0.1

# Signs: +1 for positive, -1 for negative
signs = np.ones(N_TOTAL, dtype=np.float32)
signs[N_POSITIVE:] = -1.0

# Visualization indices
vis_pos = np.random.choice(N_POSITIVE, min(N_VIS_POS, N_POSITIVE), replace=False)
vis_neg = np.random.choice(N_NEGATIVE, min(N_VIS_NEG, N_NEGATIVE), replace=False) + N_POSITIVE

# =============================================================================
# VECTORIZED PM FORCE CALCULATION
# =============================================================================
def compute_forces_pm_fast(pos, signs, box_size, grid_size, softening):
    """Fully vectorized PM force calculation."""
    N = len(pos)
    cell_size = box_size / grid_size

    # Wrap to [0, box_size)
    pos_wrap = (pos + box_size/2) % box_size

    # Grid indices (NGP - Nearest Grid Point)
    idx = (pos_wrap / cell_size).astype(np.int32) % grid_size
    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]

    # Flat index for fast bincount
    flat_idx = ix * grid_size * grid_size + iy * grid_size + iz

    # Positive and negative masks
    pos_mask = signs > 0
    neg_mask = ~pos_mask

    # Density fields using bincount (vectorized!)
    rho_pos_flat = np.bincount(flat_idx[pos_mask], minlength=grid_size**3).astype(np.float32)
    rho_neg_flat = np.bincount(flat_idx[neg_mask], minlength=grid_size**3).astype(np.float32)

    rho_pos = rho_pos_flat.reshape(grid_size, grid_size, grid_size)
    rho_neg = rho_neg_flat.reshape(grid_size, grid_size, grid_size)

    # Normalize
    cell_vol = cell_size ** 3
    rho_pos /= cell_vol
    rho_neg /= cell_vol

    # Green's function (real FFT for speed)
    kx = np.fft.fftfreq(grid_size, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(grid_size, d=cell_size) * 2 * np.pi
    kz = np.fft.rfftfreq(grid_size, d=cell_size) * 2 * np.pi  # rfft for last axis

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2 + (2*np.pi/box_size * softening)**2
    K2[0, 0, 0] = 1.0

    green = -4 * np.pi / K2
    green[0, 0, 0] = 0.0

    # Potentials via FFT
    phi_pos = irfftn(green * rfftn(rho_pos), s=(grid_size, grid_size, grid_size))
    phi_neg = irfftn(green * rfftn(rho_neg), s=(grid_size, grid_size, grid_size))

    # Gradients (vectorized finite differences)
    def grad_field(phi):
        gx = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * cell_size)
        gy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * cell_size)
        gz = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * cell_size)
        return np.stack([gx, gy, gz], axis=-1)

    grad_pos = grad_field(phi_pos)
    grad_neg = grad_field(phi_neg)

    # Interpolate forces (vectorized NGP)
    forces = np.zeros_like(pos)

    # For positive particles: attracted by pos, repelled by neg
    forces[pos_mask] = -grad_pos[ix[pos_mask], iy[pos_mask], iz[pos_mask]] \
                       + grad_neg[ix[pos_mask], iy[pos_mask], iz[pos_mask]]

    # For negative particles: attracted by neg, repelled by pos
    forces[neg_mask] = -grad_neg[ix[neg_mask], iy[neg_mask], iz[neg_mask]] \
                       + grad_pos[ix[neg_mask], iy[neg_mask], iz[neg_mask]]

    return forces.astype(np.float32)

# =============================================================================
# DIAGNOSTICS
# =============================================================================
def kinetic_energy(vel):
    return 0.5 * np.sum(vel**2)

def segregation(pos, signs):
    com_pos = np.mean(pos[signs > 0], axis=0)
    com_neg = np.mean(pos[signs < 0], axis=0)
    return np.linalg.norm(com_pos - com_neg)

# =============================================================================
# FRAME GENERATION
# =============================================================================
def make_frame(pos, signs, step, t, seg, ke_ratio, vis_pos, vis_neg, path):
    fig, ax = plt.subplots(figsize=(38.4, 21.6), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Subsample for visualization
    p_pos = pos[vis_pos]
    p_neg = pos[vis_neg]

    # Plot with higher visibility
    ax.scatter(p_pos[:, 0], p_pos[:, 1], c='#4488ff', s=1, alpha=0.5, marker='.', linewidths=0)
    ax.scatter(p_neg[:, 0], p_neg[:, 1], c='#ff4444', s=1, alpha=0.5, marker='.', linewidths=0)

    ax.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Overlay
    txt = f"Step {step:03d}/{N_STEPS} | Time: {t:.3f} | Segregation: {seg:.3f} | KE ratio: {ke_ratio:.2f}"
    ax.text(0.5, 0.02, txt, transform=ax.transAxes, fontsize=28, color='white',
            ha='center', va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax.text(0.5, 0.98, "JANUS COSMOLOGICAL MODEL — Phase 1c (10M particles)",
            transform=ax.transAxes, fontsize=36, color='white', ha='center', va='top', fontweight='bold')

    ax.text(0.02, 0.96, "● Positive masses", transform=ax.transAxes, fontsize=20, color='#4488ff', va='top')
    ax.text(0.02, 0.92, "● Negative masses", transform=ax.transAxes, fontsize=20, color='#ff4444', va='top')

    plt.savefig(path, facecolor='black', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)

# =============================================================================
# MAIN LOOP
# =============================================================================
print("\nInitial state:")
ke0 = kinetic_energy(vel)
seg0 = segregation(pos, signs)
print(f"  KE = {ke0:.4e}, Segregation = {seg0:.4f}")

print(f"\n{'='*70}")
print("Running simulation...")
print(f"{'='*70}\n")
print(f"{'Step':>6} {'Time':>8} {'Seg':>10} {'KE':>12} {'KE/KE0':>8} {'dt(s)':>8} {'ETA':>10}")
print("-" * 72)

t = 0.0
start = time.time()
times = []

for step in range(N_STEPS + 1):
    t0 = time.time()

    # Diagnostics
    ke = kinetic_energy(vel)
    ke_ratio = ke / ke0 if ke0 > 0 else 1.0
    seg = segregation(pos, signs)

    # Save frame
    frame_path = f"{FRAME_DIR}/frame_{step:03d}.png"
    make_frame(pos, signs, step, t, seg, ke_ratio, vis_pos, vis_neg, frame_path)

    # Save snapshot (every 10 steps)
    if step % 10 == 0:
        np.savez_compressed(f"{SNAPSHOT_DIR}/snapshot_{step:03d}.npz",
                           pos=pos, vel=vel, signs=signs.astype(np.int8),
                           step=step, time=t, seg=seg, ke=ke)

    dt_step = time.time() - t0
    times.append(dt_step)
    if len(times) > 5:
        times = times[-5:]
    avg_dt = np.mean(times)
    eta_sec = avg_dt * (N_STEPS - step)
    eta_str = str(timedelta(seconds=int(eta_sec)))

    print(f"{step:>6} {t:>8.3f} {seg:>10.4f} {ke:>12.4e} {ke_ratio:>8.2f} {dt_step:>8.1f} {eta_str:>10}")
    sys.stdout.flush()

    if step >= N_STEPS:
        break

    # Leapfrog
    forces = compute_forces_pm_fast(pos, signs, BOX_SIZE, GRID_SIZE, SOFTENING)
    vel += 0.5 * DT * forces
    pos += DT * vel
    pos = ((pos + BOX_SIZE/2) % BOX_SIZE) - BOX_SIZE/2
    forces = compute_forces_pm_fast(pos, signs, BOX_SIZE, GRID_SIZE, SOFTENING)
    vel += 0.5 * DT * forces
    t += DT

total = time.time() - start
print(f"\n{'='*70}")
print(f"COMPLETE in {timedelta(seconds=int(total))}")
print(f"Initial seg: {seg0:.4f} → Final: {seg:.4f} ({(seg-seg0)/seg0*100:+.1f}%)")
print(f"Final KE ratio: {ke_ratio:.2f}")
print(f"\nVideo: ffmpeg -framerate 24 -i {FRAME_DIR}/frame_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p output/phase1c/janus.mp4")
print(f"{'='*70}\n")

import json
json.dump({
    "n": N_TOTAL, "eta": ETA, "steps": N_STEPS, "dt": DT,
    "seg_init": float(seg0), "seg_final": float(seg),
    "ke_ratio": float(ke_ratio), "runtime_s": total
}, open("output/phase1c/summary.json", "w"), indent=2)
