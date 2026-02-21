#!/usr/bin/env python3
"""
PHASE 1C — Janus N-body (GPU with CuPy, no scipy)
10M particles, η = 1.045, 300 steps
"""

import os, sys, time
from datetime import datetime, timedelta

try:
    import cupy as cp
    GPU = True
    dev = cp.cuda.Device(0)
    print(f"GPU: CUDA device {dev.id}, {dev.mem_info[1]/1e9:.1f} GB VRAM")
except ImportError:
    import numpy as cp
    GPU = False
    print("CPU mode (CuPy not available)")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print(f"PHASE 1C — Janus N-body ({'CUDA' if GPU else 'CPU'})")
print("=" * 70)

# Parameters
N = 10_000_000
ETA = 1.045
STEPS = 300
DT = 0.003
BOX = 100.0
GRID = 128
SOFT = BOX / GRID * 2

N_POS = int(N / (1 + ETA))
N_NEG = N - N_POS

os.makedirs("output/phase1c/frames", exist_ok=True)
os.makedirs("output/phase1c/snapshots", exist_ok=True)

print(f"N={N:,} ({N_POS:,}+/{N_NEG:,}-), Steps={STEPS}")

# Initialize on GPU
cp.random.seed(42)
pos = (cp.random.rand(N, 3, dtype=cp.float32) - 0.5) * BOX
vel = (cp.random.rand(N, 3, dtype=cp.float32) - 0.5) * 0.1
signs = cp.ones(N, dtype=cp.float32)
signs[N_POS:] = -1.0

pos_mask = signs > 0
neg_mask = ~pos_mask

# Visualization indices
np.random.seed(42)
vis_p = np.random.choice(N_POS, min(100000, N_POS), replace=False)
vis_n = np.random.choice(N_NEG, min(100000, N_NEG), replace=False) + N_POS

# Green's function (using cupy.fft)
cell = BOX / GRID
kx = cp.fft.fftfreq(GRID, d=cell) * 2 * cp.pi
ky = cp.fft.fftfreq(GRID, d=cell) * 2 * cp.pi
kz = cp.fft.rfftfreq(GRID, d=cell) * 2 * cp.pi
KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2 + (2*cp.pi/BOX * SOFT)**2
K2[0,0,0] = 1.0
GREEN = -4 * cp.pi / K2
GREEN[0,0,0] = 0.0

def forces_pm(pos, signs, pos_mask, neg_mask):
    pw = (pos + BOX/2) % BOX
    idx = (pw / cell).astype(cp.int32) % GRID
    ix, iy, iz = idx[:,0], idx[:,1], idx[:,2]
    flat = ix * GRID**2 + iy * GRID + iz

    rho_p = cp.bincount(flat[pos_mask], minlength=GRID**3).reshape(GRID,GRID,GRID).astype(cp.float32) / cell**3
    rho_n = cp.bincount(flat[neg_mask], minlength=GRID**3).reshape(GRID,GRID,GRID).astype(cp.float32) / cell**3

    phi_p = cp.fft.irfftn(GREEN * cp.fft.rfftn(rho_p), s=(GRID,)*3)
    phi_n = cp.fft.irfftn(GREEN * cp.fft.rfftn(rho_n), s=(GRID,)*3)

    def grad(phi):
        gx = (cp.roll(phi,-1,0) - cp.roll(phi,1,0)) / (2*cell)
        gy = (cp.roll(phi,-1,1) - cp.roll(phi,1,1)) / (2*cell)
        gz = (cp.roll(phi,-1,2) - cp.roll(phi,1,2)) / (2*cell)
        return cp.stack([gx,gy,gz], axis=-1)

    gp, gn = grad(phi_p), grad(phi_n)
    f = cp.zeros_like(pos)
    f[pos_mask] = -gp[ix[pos_mask],iy[pos_mask],iz[pos_mask]] + gn[ix[pos_mask],iy[pos_mask],iz[pos_mask]]
    f[neg_mask] = -gn[ix[neg_mask],iy[neg_mask],iz[neg_mask]] + gp[ix[neg_mask],iy[neg_mask],iz[neg_mask]]
    return f

def ke(vel): return float(0.5 * cp.sum(vel**2))
def seg(pos, pm, nm): return float(cp.linalg.norm(cp.mean(pos[pm],0) - cp.mean(pos[nm],0)))

def frame(pos_cpu, step, t, s, kr, path):
    fig, ax = plt.subplots(figsize=(38.4, 21.6), dpi=100)
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    ax.scatter(pos_cpu[vis_p,0], pos_cpu[vis_p,1], c='#4488ff', s=1, alpha=0.5, marker='.', lw=0)
    ax.scatter(pos_cpu[vis_n,0], pos_cpu[vis_n,1], c='#ff4444', s=1, alpha=0.5, marker='.', lw=0)
    ax.set_xlim(-BOX/2, BOX/2); ax.set_ylim(-BOX/2, BOX/2); ax.set_aspect('equal'); ax.axis('off')
    ax.text(0.5, 0.02, f"Step {step:03d}/{STEPS} | Time: {t:.3f} | Seg: {s:.3f} | KE: {kr:.2f}",
            transform=ax.transAxes, fontsize=28, color='white', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    ax.text(0.5, 0.98, "JANUS — Phase 1c (10M, CUDA)", transform=ax.transAxes, fontsize=36,
            color='white', ha='center', va='top', fontweight='bold')
    plt.savefig(path, facecolor='black', bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Main loop
ke0 = ke(vel)
seg0 = seg(pos, pos_mask, neg_mask)
print(f"Initial: KE={ke0:.4e}, Seg={seg0:.4f}")
print(f"\n{'Step':>5} {'Time':>7} {'Seg':>8} {'KE':>10} {'KE/KE0':>7} {'dt':>5} {'ETA':>10}")
print("-"*60)

t = 0.0
start = time.time()
times = []

for step in range(STEPS + 1):
    t0 = time.time()
    if GPU: cp.cuda.Stream.null.synchronize()

    k = ke(vel)
    s = seg(pos, pos_mask, neg_mask)
    kr = k/ke0 if ke0 > 0 else 1.0

    pos_cpu = cp.asnumpy(pos) if GPU else pos
    frame(pos_cpu, step, t, s, kr, f"output/phase1c/frames/frame_{step:03d}.png")

    if step % 10 == 0:
        np.savez_compressed(f"output/phase1c/snapshots/snap_{step:03d}.npz",
                           pos=pos_cpu, step=step, time=t, seg=s, ke=k)

    dt_s = time.time() - t0
    times.append(dt_s)
    if len(times) > 5: times = times[-5:]
    eta = np.mean(times) * (STEPS - step)

    print(f"{step:>5} {t:>7.3f} {s:>8.4f} {k:>10.4e} {kr:>7.2f} {dt_s:>5.1f} {str(timedelta(seconds=int(eta))):>10}")
    sys.stdout.flush()

    if step >= STEPS: break

    # Leapfrog
    f = forces_pm(pos, signs, pos_mask, neg_mask)
    vel += 0.5 * DT * f
    pos += DT * vel
    pos = ((pos + BOX/2) % BOX) - BOX/2
    f = forces_pm(pos, signs, pos_mask, neg_mask)
    vel += 0.5 * DT * f
    t += DT

total = time.time() - start
print(f"\nDone in {timedelta(seconds=int(total))}")
print(f"Seg: {seg0:.4f} -> {s:.4f} ({(s-seg0)/seg0*100:+.1f}%)")
print(f"\nffmpeg -framerate 24 -i output/phase1c/frames/frame_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p output/phase1c/janus.mp4")
