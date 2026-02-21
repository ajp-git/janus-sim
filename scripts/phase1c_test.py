#!/usr/bin/env python3
"""
Phase 1c TEST — 100K particles, 50 steps
Verify stability before launching 10M
"""

import numpy as np
import time

try:
    import cupy as cp
    GPU = True
    print("GPU: CUDA enabled")
except ImportError:
    import numpy as cp
    GPU = False
    print("CPU mode")

# TEST PARAMETERS (reduced)
N = 100_000
ETA = 1.045
STEPS = 50
DT = 0.0005      # REDUCED from 0.003
BOX = 100.0
GRID = 64        # Smaller grid for 100K
SOFT = (BOX / GRID) * 6  # INCREASED 3x (was *2)

N_POS = int(N / (1 + ETA))
N_NEG = N - N_POS

print(f"\n{'='*60}")
print(f"STABILITY TEST — {N:,} particles, {STEPS} steps")
print(f"{'='*60}")
print(f"dt = {DT} (was 0.003)")
print(f"softening = {SOFT:.2f} (3x increase)")
print(f"grid = {GRID}³")

# Initialize
cp.random.seed(42)
pos = (cp.random.rand(N, 3, dtype=cp.float32) - 0.5) * BOX
vel = (cp.random.rand(N, 3, dtype=cp.float32) - 0.5) * 0.1
signs = cp.ones(N, dtype=cp.float32)
signs[N_POS:] = -1.0

pos_mask = signs > 0
neg_mask = ~pos_mask

# Green's function
cell = BOX / GRID
kx = cp.fft.fftfreq(GRID, d=cell) * 2 * cp.pi
ky = cp.fft.fftfreq(GRID, d=cell) * 2 * cp.pi
kz = cp.fft.rfftfreq(GRID, d=cell) * 2 * cp.pi
KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2 + (2*cp.pi/BOX * SOFT)**2
K2[0,0,0] = 1.0
GREEN = -4 * cp.pi / K2
GREEN[0,0,0] = 0.0

def forces_pm(pos, pos_mask, neg_mask):
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

# Main loop
ke0 = ke(vel)
seg0 = seg(pos, pos_mask, neg_mask)
print(f"\nInitial: KE={ke0:.4e}, Seg={seg0:.4f}")

print(f"\n{'Step':>5} {'KE/KE0':>10} {'Seg':>10}")
print("-"*30)

t = 0.0
start = time.time()

for step in range(STEPS + 1):
    if GPU: cp.cuda.Stream.null.synchronize()

    k = ke(vel)
    s = seg(pos, pos_mask, neg_mask)
    kr = k/ke0 if ke0 > 0 else 1.0

    if step % 10 == 0 or step == STEPS:
        print(f"{step:>5} {kr:>10.2f} {s:>10.4f}")

    if step >= STEPS:
        break

    # Leapfrog
    f = forces_pm(pos, pos_mask, neg_mask)
    vel += 0.5 * DT * f
    pos += DT * vel
    pos = ((pos + BOX/2) % BOX) - BOX/2
    f = forces_pm(pos, pos_mask, neg_mask)
    vel += 0.5 * DT * f
    t += DT

elapsed = time.time() - start
final_kr = kr

print(f"\n{'='*60}")
print(f"RESULT: KE/KE0 at step 50 = {final_kr:.2f}")
print(f"Time: {elapsed:.1f}s")
print(f"{'='*60}")

if final_kr < 2.0:
    print("✓ STABLE — Ready for 10M particles")
elif final_kr < 10.0:
    print("⚠ MARGINAL — Consider further reducing dt")
elif final_kr < 200.0:
    print("⚠ ACCEPTABLE — Proceed with caution")
else:
    print("✗ UNSTABLE — Do NOT proceed to 10M")
    print("  Suggestions: reduce dt further or increase softening")
