#!/usr/bin/env python3
"""
PHASE 1C — Janus N-body (CUDA GPU VERSION)

10M particles on RTX 3060 using CuPy for GPU-accelerated PM method.
"""

import os
import sys
import time
from datetime import datetime, timedelta

# GPU backend selection
try:
    import cupy as cp
    from cupyx.scipy.fft import rfftn, irfftn
    GPU = True
    print("GPU: CuPy + CUDA enabled")
except ImportError:
    import numpy as cp
    from scipy.fft import rfftn, irfftn
    GPU = False
    print("WARNING: CuPy not available, using CPU (slow)")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print(f"PHASE 1C — Janus N-body ({'CUDA GPU' if GPU else 'CPU'})")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if GPU:
    print(f"GPU: {cp.cuda.Device(0).name.decode()}")
    print(f"VRAM: {cp.cuda.Device(0).mem_info[1] / 1e9:.1f} GB")

# =============================================================================
# PARAMETERS
# =============================================================================
N_TOTAL = 10_000_000
ETA = 1.045
N_STEPS = 300
DT = 0.003
BOX_SIZE = 100.0
GRID_SIZE = 128
SOFTENING = BOX_SIZE / GRID_SIZE * 2

N_VIS = 100_000  # Visualization subsample per sign

SNAPSHOT_DIR = "output/phase1c/snapshots"
FRAME_DIR = "output/phase1c/frames"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

N_POSITIVE = int(N_TOTAL / (1.0 + ETA))
N_NEGATIVE = N_TOTAL - N_POSITIVE

print(f"\nN = {N_TOTAL:,} ({N_POSITIVE:,}+ / {N_NEGATIVE:,}-)")
print(f"η = {N_NEGATIVE/N_POSITIVE:.4f}, Steps = {N_STEPS}")

# =============================================================================
# INITIALIZATION (on GPU)
# =============================================================================
print("\nInitializing on GPU...")
cp.random.seed(42)

pos = (cp.random.rand(N_TOTAL, 3, dtype=cp.float32) - 0.5) * BOX_SIZE
vel = (cp.random.rand(N_TOTAL, 3, dtype=cp.float32) - 0.5) * 0.1

signs = cp.ones(N_TOTAL, dtype=cp.float32)
signs[N_POSITIVE:] = -1.0

# Precompute masks
pos_mask = signs > 0
neg_mask = ~pos_mask

# Visualization indices (on CPU for matplotlib)
np.random.seed(42)
vis_pos_idx = np.random.choice(N_POSITIVE, min(N_VIS, N_POSITIVE), replace=False)
vis_neg_idx = np.random.choice(N_NEGATIVE, min(N_VIS, N_NEGATIVE), replace=False) + N_POSITIVE

# Precompute Green's function (stays on GPU)
cell_size = BOX_SIZE / GRID_SIZE
kx = cp.fft.fftfreq(GRID_SIZE, d=cell_size) * 2 * cp.pi
ky = cp.fft.fftfreq(GRID_SIZE, d=cell_size) * 2 * cp.pi
kz = cp.fft.rfftfreq(GRID_SIZE, d=cell_size) * 2 * cp.pi
KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2 + (2*cp.pi/BOX_SIZE * SOFTENING)**2
K2[0, 0, 0] = 1.0
GREEN = -4 * cp.pi / K2
GREEN[0, 0, 0] = 0.0

print(f"  Green's function precomputed: {GREEN.shape}")

# =============================================================================
# PM FORCE (FULLY ON GPU)
# =============================================================================
def compute_forces_gpu(pos, signs, pos_mask, neg_mask):
    """PM force calculation entirely on GPU."""
    cell_size = BOX_SIZE / GRID_SIZE

    # Wrap positions
    pos_wrap = (pos + BOX_SIZE/2) % BOX_SIZE

    # Grid indices
    idx = (pos_wrap / cell_size).astype(cp.int32) % GRID_SIZE
    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
    flat_idx = ix * GRID_SIZE * GRID_SIZE + iy * GRID_SIZE + iz

    # Density via bincount (GPU)
    rho_pos_flat = cp.bincount(flat_idx[pos_mask], minlength=GRID_SIZE**3).astype(cp.float32)
    rho_neg_flat = cp.bincount(flat_idx[neg_mask], minlength=GRID_SIZE**3).astype(cp.float32)

    rho_pos = rho_pos_flat.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE) / (cell_size**3)
    rho_neg = rho_neg_flat.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE) / (cell_size**3)

    # Potentials via FFT (GPU)
    phi_pos = irfftn(GREEN * rfftn(rho_pos), s=(GRID_SIZE,)*3)
    phi_neg = irfftn(GREEN * rfftn(rho_neg), s=(GRID_SIZE,)*3)

    # Gradients
    def grad(phi):
        gx = (cp.roll(phi, -1, 0) - cp.roll(phi, 1, 0)) / (2*cell_size)
        gy = (cp.roll(phi, -1, 1) - cp.roll(phi, 1, 1)) / (2*cell_size)
        gz = (cp.roll(phi, -1, 2) - cp.roll(phi, 1, 2)) / (2*cell_size)
        return cp.stack([gx, gy, gz], axis=-1)

    grad_pos = grad(phi_pos)
    grad_neg = grad(phi_neg)

    # Forces (Janus rules)
    forces = cp.zeros_like(pos)
    forces[pos_mask] = -grad_pos[ix[pos_mask], iy[pos_mask], iz[pos_mask]] \
                       + grad_neg[ix[pos_mask], iy[pos_mask], iz[pos_mask]]
    forces[neg_mask] = -grad_neg[ix[neg_mask], iy[neg_mask], iz[neg_mask]] \
                       + grad_pos[ix[neg_mask], iy[neg_mask], iz[neg_mask]]

    return forces

# =============================================================================
# DIAGNOSTICS
# =============================================================================
def kinetic_energy(vel):
    return float(0.5 * cp.sum(vel**2))

def segregation(pos, pos_mask, neg_mask):
    com_pos = cp.mean(pos[pos_mask], axis=0)
    com_neg = cp.mean(pos[neg_mask], axis=0)
    return float(cp.linalg.norm(com_pos - com_neg))

# =============================================================================
# FRAME GENERATION (CPU + matplotlib)
# =============================================================================
def make_frame(pos_cpu, step, t, seg, ke_ratio, path):
    fig, ax = plt.subplots(figsize=(38.4, 21.6), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    p_pos = pos_cpu[vis_pos_idx]
    p_neg = pos_cpu[vis_neg_idx]

    ax.scatter(p_pos[:, 0], p_pos[:, 1], c='#4488ff', s=1, alpha=0.5, marker='.', lw=0)
    ax.scatter(p_neg[:, 0], p_neg[:, 1], c='#ff4444', s=1, alpha=0.5, marker='.', lw=0)

    ax.set_xlim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_ylim(-BOX_SIZE/2, BOX_SIZE/2)
    ax.set_aspect('equal')
    ax.axis('off')

    txt = f"Step {step:03d}/{N_STEPS} | Time: {t:.3f} | Segregation: {seg:.3f} | KE ratio: {ke_ratio:.2f}"
    ax.text(0.5, 0.02, txt, transform=ax.transAxes, fontsize=28, color='white',
            ha='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    ax.text(0.5, 0.98, "JANUS COSMOLOGICAL MODEL — Phase 1c (10M particles, CUDA)",
            transform=ax.transAxes, fontsize=36, color='white', ha='center', va='top', fontweight='bold')
    ax.text(0.02, 0.96, "● Positive", transform=ax.transAxes, fontsize=20, color='#4488ff')
    ax.text(0.02, 0.92, "● Negative", transform=ax.transAxes, fontsize=20, color='#ff4444')

    plt.savefig(path, facecolor='black', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()

# =============================================================================
# MAIN LOOP
# =============================================================================
print("\nInitial state:")
ke0 = kinetic_energy(vel)
seg0 = segregation(pos, pos_mask, neg_mask)
print(f"  KE = {ke0:.4e}, Seg = {seg0:.4f}")

print(f"\n{'='*70}")
print(f"{'Step':>6} {'Time':>8} {'Seg':>10} {'KE':>12} {'KE/KE0':>8} {'dt':>6} {'ETA':>10}")
print("-" * 68)

t = 0.0
start = time.time()
step_times = []

for step in range(N_STEPS + 1):
    t0 = time.time()

    # Sync GPU before diagnostics
    if GPU:
        cp.cuda.Stream.null.synchronize()

    ke = kinetic_energy(vel)
    ke_ratio = ke / ke0 if ke0 > 0 else 1.0
    seg = segregation(pos, pos_mask, neg_mask)

    # Transfer to CPU for visualization
    pos_cpu = cp.asnumpy(pos) if GPU else pos

    # Frame
    make_frame(pos_cpu, step, t, seg, ke_ratio, f"{FRAME_DIR}/frame_{step:03d}.png")

    # Snapshot every 10 steps
    if step % 10 == 0:
        np.savez_compressed(f"{SNAPSHOT_DIR}/snap_{step:03d}.npz",
                           pos=pos_cpu, step=step, time=t, seg=seg, ke=ke)

    dt_step = time.time() - t0
    step_times.append(dt_step)
    if len(step_times) > 5:
        step_times = step_times[-5:]
    eta = np.mean(step_times) * (N_STEPS - step)

    print(f"{step:>6} {t:>8.3f} {seg:>10.4f} {ke:>12.4e} {ke_ratio:>8.2f} {dt_step:>6.1f} {str(timedelta(seconds=int(eta))):>10}")
    sys.stdout.flush()

    if step >= N_STEPS:
        break

    # === LEAPFROG (GPU) ===
    forces = compute_forces_gpu(pos, signs, pos_mask, neg_mask)
    vel = vel + 0.5 * DT * forces
    pos = pos + DT * vel
    pos = ((pos + BOX_SIZE/2) % BOX_SIZE) - BOX_SIZE/2
    forces = compute_forces_gpu(pos, signs, pos_mask, neg_mask)
    vel = vel + 0.5 * DT * forces
    t += DT

total = time.time() - start
print(f"\n{'='*70}")
print(f"DONE in {timedelta(seconds=int(total))}")
print(f"Segregation: {seg0:.4f} → {seg:.4f} ({(seg-seg0)/seg0*100:+.1f}%)")
print(f"\nffmpeg -framerate 24 -i {FRAME_DIR}/frame_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p output/phase1c/janus.mp4")

import json
json.dump({"n": N_TOTAL, "eta": ETA, "steps": N_STEPS, "seg0": seg0, "seg_final": seg,
           "ke_ratio": ke_ratio, "runtime": total, "gpu": GPU}, open("output/phase1c/summary.json", "w"), indent=2)
