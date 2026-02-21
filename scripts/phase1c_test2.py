#!/usr/bin/env python3
"""
Phase 1c TEST v2 — Softening réduit pour observer la ségrégation
"""

import numpy as np
import time

try:
    import cupy as cp
    GPU = True
except ImportError:
    import numpy as cp
    GPU = False

# PARAMETERS
N = 100_000
ETA = 1.045
STEPS = 50
DT = 0.0005
BOX = 100.0
GRID = 64
SOFT = (BOX / GRID) * 2  # BACK TO ORIGINAL (was *6)

N_POS = int(N / (1 + ETA))
N_NEG = N - N_POS

print(f"{'='*60}")
print(f"STABILITY TEST v2 — softening = {SOFT:.2f} (original)")
print(f"N={N:,}, dt={DT}, grid={GRID}³")
print(f"{'='*60}")

cp.random.seed(42)
pos = (cp.random.rand(N, 3, dtype=cp.float32) - 0.5) * BOX
vel = (cp.random.rand(N, 3, dtype=cp.float32) - 0.5) * 0.1
signs = cp.ones(N, dtype=cp.float32)
signs[N_POS:] = -1.0

pos_mask = signs > 0
neg_mask = ~pos_mask

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

ke0 = ke(vel)
seg0 = seg(pos, pos_mask, neg_mask)

print(f"\n{'Step':>5} {'KE/KE0':>10} {'Seg':>10} {'ΔSeg':>10}")
print("-"*40)

t = 0.0
for step in range(STEPS + 1):
    if GPU: cp.cuda.Stream.null.synchronize()

    k = ke(vel)
    s = seg(pos, pos_mask, neg_mask)
    kr = k/ke0
    ds = s - seg0

    if step % 10 == 0 or step == STEPS:
        print(f"{step:>5} {kr:>10.2f} {s:>10.4f} {ds:>+10.4f}")

    if step >= STEPS:
        break

    f = forces_pm(pos, pos_mask, neg_mask)
    vel += 0.5 * DT * f
    pos += DT * vel
    pos = ((pos + BOX/2) % BOX) - BOX/2
    f = forces_pm(pos, pos_mask, neg_mask)
    vel += 0.5 * DT * f
    t += DT

print(f"\n{'='*60}")
print(f"Seg₀ = {seg0:.4f} → Seg₅₀ = {s:.4f} (Δ = {ds:+.4f})")
print(f"KE/KE0 = {kr:.2f}")

if kr < 2.0 and abs(ds) > 0.01:
    print("✓ STABLE + SEGREGATION VISIBLE → Ready for 10M")
elif kr < 10.0 and abs(ds) > 0.001:
    print("⚠ MARGINAL — Proceed with caution")
elif kr > 200:
    print("✗ UNSTABLE — Increase softening or reduce dt")
else:
    print("⚠ NO SEGREGATION — Softening may be wrong or need more steps")
