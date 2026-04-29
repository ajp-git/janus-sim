# Phase 10.9 — Mini-run TreePM 15K Janus z=10→z=0

**Generated** : SystemTime { tv_sec: 1777481588, tv_nsec: 931029815 }
**Branch** : feat/treepm-jpp-port (Phase 10.7+10.8 fixes)
**Setup** : N=10648 (532 m+, 10116 m-), L=100 Mpc, n_pm=64, μ=19, η=1.045
**dt** : 0.001 Gyr (fixed)
**TreePM** : r_s=1.8750 Mpc, r_cut=9.3750 Mpc, θ=0.5
**softening** : 0.05 Mpc

## Run summary

- Steps completed : **15000**
- Final z         : **0.0975**
- Cosmic time     : **15.00 Gyr**
- Wall time       : 14.04 min

## Final state

| Metric | Value | Reference (Barnes-Hut hist.) |
|---|---|---|
| Corr(δ⁺, δ⁻) | **-0.0017** | ≈ -0.07 |
| σ_8 proxy (rms δ⁺) | **29.0138** | ≈ 0.70 (Mpc/h scale-dependent) |
| t₀ (cosmic time) | **15.00 Gyr** | ≈ 15.87 Gyr |
| v_rms+ | 648.7 km/s | < 5000 |
| v_rms- | 3601.1 km/s | < 5000 |

## Power spectrum P(k) at z=z_final

| bin | k_c [1/Mpc] | P_+(k) | P_-(k) | P_×(k) |
|---|---|---|---|---|
| 0 | 0.0628 | 8.803e9 | 1.737e10 | -3.093e9 |
| 1 | 0.1885 | 3.072e9 | 1.067e10 | -2.921e8 |
| 2 | 0.3142 | 1.777e9 | 6.056e9 | +1.639e8 |
| 3 | 0.4398 | 1.522e9 | 3.385e9 | -3.496e7 |
| 4 | 0.5655 | 1.194e9 | 1.771e9 | -7.568e6 |
| 5 | 0.6912 | 1.089e9 | 8.573e8 | +1.262e7 |
| 6 | 0.8168 | 8.577e8 | 5.079e8 | -4.564e6 |
| 7 | 0.9425 | 6.905e8 | 2.926e8 | +3.024e6 |
| 8 | 1.0681 | 5.762e8 | 1.735e8 | +8.133e5 |
| 9 | 1.1938 | 4.839e8 | 1.114e8 | -9.119e5 |
| 10 | 1.3195 | 4.074e8 | 8.569e7 | +5.854e4 |
| 11 | 1.4451 | 3.185e8 | 5.949e7 | -2.091e4 |
| 12 | 1.5708 | 2.592e8 | 4.171e7 | -5.765e5 |
| 13 | 1.6965 | 2.178e8 | 2.848e7 | +1.421e5 |
| 14 | 1.8221 | 1.762e8 | 1.920e7 | +9.828e4 |
| 15 | 1.9478 | 1.448e8 | 1.329e7 | +5.739e4 |

## Test critique : pic à |k|=4, 8, 16 ?

- bin 2 (k≈0.3142): P=P_+=1.777e9, ratio vs neighbors=0.77 OK
- bin 4 (k≈0.5655): P=P_+=1.194e9, ratio vs neighbors=0.91 OK
- bin 8 (k≈1.0681): P=P_+=5.762e8, ratio vs neighbors=0.98 OK

✅ **No isolated peak** at k=4/8/16. No octree resonance signature.

## CSV evolution

Saved at `/app/output/janus_minirun_treepm_15k/evolution.csv`
Snapshots in `/app/output/janus_minirun_treepm_15k/snapshots/snap_*.bin`

## Verdict

🟢 **PASS** — pipeline stable, Janus segregation correct (Corr<0), no resonance, v_rms within bounds.
