# Known Fixes and Non-Viable Approaches

This document tracks approaches that were tried and found to be non-viable for
production use, along with the reasons and recommended alternatives.

---

## TreePM CPU: Non-Viable for Production

**Date**: 2026-02-27
**Status**: NOT RECOMMENDED for production

### Performance Comparison

| Method | Time/step @ 100K | Time/step @ 1M (est.) |
|--------|------------------|----------------------|
| TreePM (full PM+Tree) | 5.4s | ~60s |
| TreePM PM-only | 1.6s | ~16s |
| **GPU Barnes-Hut** | **5 ms** | **~50 ms** |

**GPU Barnes-Hut is 1000x faster than CPU TreePM.**

### Detailed Analysis

#### TreePM Architecture
TreePM combines:
1. **PM (Particle-Mesh)**: FFT-based long-range forces, O(G³ log G + N)
2. **Tree (Barnes-Hut)**: Short-range corrections within r_cut, O(N log N)

Benchmark results (100K particles, box=100, r_cut=6.25):
```
PM (FFT + mass assignment): 0.75s
Tree (build + force): 4.69s
Total: 5.44s/step
```

**The CPU Tree is the bottleneck, not the PM.**

#### Why PM-Only Doesn't Work
PM-only mode (skipping Tree short-range) was tested:
- Performance: 1.6s/step @ 10K (still too slow)
- Physics: **Segregation doesn't increase** because short-range Janus
  repulsion/attraction is not computed

PM handles only long-range forces. The crucial Janus physics (same-sign
attraction, opposite-sign repulsion) operates at short range.

#### cuFFT GPU Acceleration
Attempted to accelerate PM with cuFFT (GPU FFT):
- cudarc 0.12: No cuFFT bindings available
- cudarc 0.19: Has cuFFT but breaks existing GPU code (API change)
- Result: Not implemented

**However, this wouldn't help** because PM is already fast (0.75s).
The bottleneck is the CPU Tree (4.69s).

### Recommended Alternative: GPU Barnes-Hut

The existing `GpuNBodySimulation` and `GpuNBodyTwoPass` implementations use
GPU-accelerated Barnes-Hut (Karras 2012) with:

- **θ-BH opening angle**: Configurable (default 0.7, testing 0.5)
- **Morton sorting**: Spatial locality for cache efficiency
- **Warp-coherent traversal**: Optimized for GPU execution

Performance:
- 100K particles: 5 ms/step
- 1M particles: ~50 ms/step
- 8M particles: ~400 ms/step

**This is the production-ready solution.**

### When TreePM Might Be Useful

TreePM could be useful for:
1. **Code validation**: Comparing TreePM vs GPU-BH results
2. **Very large boxes**: Where r_cut << box_size and most pairs are long-range
3. **Future GPU TreePM**: If GPU Tree is implemented

For Janus N-body simulations at 1M-100M particles, **use GPU Barnes-Hut**.

---

## Virialization: PE_binding Method

**Date**: 2026-02-21
**Status**: REQUIRED for Janus systems

### The Problem

Standard virialization uses:
```
2 * KE + PE_total = 0
→ KE_target = -PE_total / 2
```

For Janus systems with η ≈ 1:
- **Repulsive +/- pairs dominate** → PE_total > 0
- KE_target = -PE_total/2 < 0 → **impossible**

### The Solution: PE_binding

Use only **same-sign pairs** (which are always attractive):
```rust
// PE_binding = Σ_{i<j, sign_i = sign_j} -G*m_i*m_j/r_ij
let ke_target = pe_binding.abs() / 2.0;
let alpha = (ke_target / ke_initial).sqrt();
```

Typical results:
- α ≈ 4.5-4.8 for η=1.045
- Virial error: 0.0000%

### TreePM-Coherent Virialization

For TreePM, PE_binding should only include pairs within r_cut:
```rust
fn compute_pe_binding_treepm(particles: &[Particle], r_cut: f64, ...) {
    for i, j where sign_i == sign_j AND r_ij < r_cut {
        pe_binding -= G * m_i * m_j / r_soft;
    }
}
```

This ensures virialization matches what TreePM actually computes.

---

## Grid Artifacts in Rendered Frames

**Date**: 2026-02 (ongoing investigation)
**Status**: Testing θ=0.5 with Zel'dovich ICs

### Context

Grid-like artifacts were observed in rendered frames under specific conditions:
1. GPU Barnes-Hut with θ=0.5
2. Multi-mode Zel'dovich initial conditions (100 modes)
3. Hubble friction enabled
4. High particle counts (1M+)

### FIX-011: GPU BH θ=0.5 Grid Artifact (2026-02-27)

**Status**: CONFIRMED - GPU BH θ=0.5 NOT production-ready

Anti-regression test:
- GPU BH θ=0.5
- 100-mode Zel'dovich ICs (built into GpuNBodyTwoPass)
- Hubble friction (CosmoInterpolator, z_init=5)
- 1M particles, 1000 steps
- Critical frames: step 0, 500, 1000

**Result**: ✗ GRID ARTIFACT DETECTED at step 1000

Evidence: `output/grid_artifact_test/frames/frame_step_001000.png`
- XZ projection: horizontal and vertical lines visible
- YZ projection: horizontal and vertical lines visible
- Pattern: regular grid-like structure, NOT natural cosmic web

**Previously tested**:
- θ=0.5: grid artifact ✗
- θ=0.3: grid artifact ✗ (tested earlier)
- θ=0.7: may work but accuracy concerns

**Root cause**: Barnes-Hut opening angle too aggressive when combined with:
1. Multi-mode Zel'dovich ICs (coherent perturbations)
2. Hubble friction (velocity damping amplifies positional errors)
3. Long integration (errors accumulate over 1000+ steps)

**CONCLUSION**:
- GPU Barnes-Hut with θ ≤ 0.5: NOT production-ready
- TreePM CPU: NOT viable (1000x too slow)
- **TreePM + cuFFT GPU: ONLY viable solution** → must implement

### Additional Issue: Segregation Too Low

Test showed Seg=0.0011 at step 1000, but reference 2M run achieved Seg=0.694.

**Root cause identified**: `GpuNBodyTwoPass` uses **cold start (v=0)**.
- Hubble friction = -H × v × dtau/dt
- With v=0, friction term is zero → no damping
- No kinetic energy to convert to segregation

**Solution**: Either:
1. Use `GpuNBodySimulation` which has virialization
2. Add Zel'dovich velocities to `GpuNBodyTwoPass`
3. Implement proper virialization for TwoPass

---

## Historical Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| rsqrt() precision | rsqrt() is float intrinsic | Use 1.0/sqrt() for f64 |
| COM periodic drift | Simple average ignores wrap | Minimum image convention |
| Zero segregation (PM) | PM smooths short-range | Use direct/tree methods |
| High Seg₀ in ICs | Sequential sign assignment | Random sign per particle |
| α = 2581 virialization | PE_total wrong for Janus | Use PE_binding method |
| α = 47 for TreePM | Full PE vs r_cut-limited | Use r_cut-limited PE |

---

## References

- GPU Barnes-Hut: Karras 2012 "Maximizing Parallelism in the Construction of BVHs"
- TreePM: Bagla 2002 "TreePM: A Code for Cosmological N-Body Simulations"
- Janus model: Petit & D'Agostini 2014, EPJC 2024
