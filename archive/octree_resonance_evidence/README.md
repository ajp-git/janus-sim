# Octree resonance L/16 — visual evidence

## Context

Run µ=19 BMAX squared (commit 4d6f797, branche `fix/bmax-mac-squared`),
killed at step 1730 (z=4.10) on 2026-04-29 06:25 after visual confirmation of
octree resonance at L/16 (level-4 subdivision) in frame_01700.

## Files

- `frame_01700_z4.117_grid_visible.png` — adaptive 2.5D rendering, shows
  vertical AND horizontal bands at ~30 Mpc spacing on m+, m-, total density,
  segregation map, at zoom 200 Mpc and 50 Mpc, both raw scatter and projected
  density.
- `snap_001700_pub_dual.png` — publication-quality 2-panel rendering
  (m+ blue / m- red), CIC 256³ + Gaussian smoothing + slab projection ±25 Mpc.
- `snap_001700_pub_segregation.png` — δ_+ − δ_- map.
- `snap_001700_pub_density.png` — total log-density.

## Quantitative confirmation

P(k) 3D radial bin (snap_001680, z=4.13), |k|=16 corresponds to λ=L/16=31.25 Mpc:

| k | λ (Mpc) | ratio_4nbr m- | Verdict |
|---|---|---|---|
| 4 | 125.0 | 24.81 | Cosmological large-scale |
| 8 | 62.5 | 2.10 | L/8 (level-3 octree) present |
| 12 | 41.7 | 1.46 | Noise |
| **16** | **31.2** | **5.31** | **L/16 (level-4) DOMINANT** |
| 32 | 15.6 | 1.76 | L/32 (level-5) emerging |

## Why pure-axis directional P(k) failed to detect the resonance

Earlier diagnostics measured ratio[k] on cardinal axes (x,y,z) and face
diagonals (xy,xz,yz,xyz) showing max ratio[L/16] = 1.42. Insufficient to
trigger the AJP threshold (>2).

The 3D radial bin |k|=16 includes ~3338 modes (vs ~129 modes on a single
axis). The octree level-4 subdivision creates a 16³=4096 cell grid; the
energy is distributed across 3D diagonal modes (k_x=k_y=16, etc.) which
the axis-only test misses but the 3D radial captures.

**Methodological lesson for the preprint**: Always use 3D radial P(|k|)
when looking for grid-aligned resonances; pure-axis tests are insufficient
when the resonance source is a 3D subdivision structure.

## Conclusion

BMAX MAC (Salmon-Warren / Springel 2005) eliminates the L/8 amplification
that affected the pre-fix run (ratio[L/8] reduced from 1.93 to 1.064) but
**displaces** the resonance to L/16 (level-4) instead of suppressing it.
The recursive nature of the octree subdivision is intrinsic; only PM
(isotropic by Fourier construction) eliminates it definitively.

→ Pivot to TreePM (port from existing src/treepm + nbody_gpu_twopass.rs).
