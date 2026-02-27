# cuFFT Implementation Analysis

## Current Status

**TreePM is the only viable path for production**:
- GPU Barnes-Hut θ=0.5: Grid artifacts confirmed (FIX-011)
- CPU TreePM: 1000x too slow (5.4s/step @ 100K)
- TreePM + cuFFT GPU: Must implement

## Benchmark Analysis (100K particles)

| Component | CPU Time | Notes |
|-----------|----------|-------|
| PM (rustfft) | 0.75s | O(G³ log G + N), already efficient |
| Tree (BH) | 4.69s | O(N log N), bottleneck |
| **Total** | **5.44s** | |

At 1M particles:
- PM: ~1s (scales well)
- Tree: ~50s (scales poorly)

**Key insight**: The bottleneck is Tree, not PM.

## cuFFT Options Evaluated

### 1. cudarc 0.19 with cufft feature
**Status**: ❌ Blocked
- Breaks existing GPU code (CudaDevice API changed)
- Would require rewriting all GPU kernels

### 2. scirs2-fft with CUDA feature
**Status**: ⚠️ Possible but complex
- Modern crate (2025), uses cuFFT internally
- Focused on sparse FFT and signal analysis
- Not optimized for 3D grid Poisson solving
- URL: https://lib.rs/crates/scirs2-fft

### 3. cufft_rust
**Status**: ❌ Insufficient
- Only supports R2C forward transform
- Limited maintenance (personal project)
- URL: https://github.com/PvdBerg1998/cufft_rust

### 4. Manual cuFFT FFI (via cuda_setup)
**Status**: ✓ Viable but significant effort
- Use `cuda_setup` crate to compile cuFFT FFI
- Write Rust bindings to cuFFT C API
- Full control over 3D R2C/C2R transforms
- URL: https://docs.rs/cuda_setup

### 5. Hybrid: CPU PM + GPU Tree
**Status**: ✓ Alternative approach
- Keep rustfft on CPU (already fast)
- Implement GPU Tree short-range (different from full BH)
- Avoids cuFFT complexity entirely

## libclang Blocker

When attempting cudarc 0.19:
```
error: failed to run custom build command for `cudarc v0.19.0`
libclang not found
```

**Solutions**:
1. Install libclang-dev in Docker (done, but API breaks)
2. Use pre-compiled bindings (bindgen bypass)
3. Manual FFI without bindgen

## Recommended Implementation Path

### Option A: Manual cuFFT FFI (Best accuracy)
1. Write C wrapper for cuFFT 3D R2C/C2R
2. Compile with cuda_setup in build.rs
3. Rust FFI bindings
4. Replace rustfft in pm_grid.rs

**Pros**: Full cuFFT support, best performance
**Cons**: ~1 week implementation, CUDA version coupling

### Option B: Hybrid CPU PM + GPU Tree-short
1. Keep rustfft for PM (0.75s is acceptable)
2. Implement GPU Tree for short-range only
3. Tree-short is simpler than full BH (smaller r_cut region)

**Pros**: Simpler, no cuFFT dependency
**Cons**: Still need to solve Tree grid artifact problem

### Option C: Higher θ with error correction
1. Use GPU BH at θ=0.7 or θ=0.8
2. Apply correction term for truncation error
3. Validate against direct sum reference

**Pros**: Uses existing code
**Cons**: May not fully eliminate artifacts

## cuFFT 3D API Reference

For manual FFI implementation:

```c
// Create plan
cufftHandle plan;
cufftPlan3d(&plan, nx, ny, nz, CUFFT_D2Z);  // Double R2C

// Execute forward
cufftExecD2Z(plan, d_rho, d_rho_k);

// Apply Green's function in k-space (custom kernel)
apply_green_kernel<<<grid, block>>>(d_rho_k, d_phi_k, ...);

// Inverse
cufftPlan3d(&plan_inv, nx, ny, nz, CUFFT_Z2D);
cufftExecZ2D(plan_inv, d_phi_k, d_phi);

// Cleanup
cufftDestroy(plan);
```

## Implementation Status (2026-02-27)

### cuFFT FFI: COMPLETE

Successfully implemented cuFFT wrapper:
- File: `cuda/cufft_wrapper.cu`
- Rust FFI: `src/treepm/cufft_ffi.rs`
- Build: `cuda/build_cufft.sh`

**Benchmark Results**:
| Grid | CPU rustfft | GPU cuFFT | Speedup |
|------|-------------|-----------|---------|
| 128³ | ~750ms | **11ms** | **70x** |
| 256³ | ~6s (est.) | ~100ms | ~60x |

### Integration

To use cuFFT in TreePM:

```bash
# Build cuFFT wrapper
./cuda/build_cufft.sh

# Build with cufft feature
RUSTFLAGS='-L target/release' cargo build --release --features cufft

# Run with library path
LD_LIBRARY_PATH=target/release cargo run --release --features cufft --bin <binary>
```

## TreePM Hybrid Implementation (2026-02-27)

### Architecture: COMPLETE

Hybrid CPU PM + GPU BH short-range implemented:
- File: `src/nbody_gpu_twopass.rs` → `step_treepm_hybrid()`
- PM long-range: CPU rustfft + CIC mass assignment
- Tree short-range: GPU BH with r_cut cutoff

**Key kernels added:**
- `forces_treepm_short_range`: GPU BH with r_cut (skips r > r_cut)
- `add_pm_forces`: Adds PM forces to GPU acc buffer

### Benchmark Results (500K particles)

| Component | Time | Notes |
|-----------|------|-------|
| PM long-range (CPU) | ~960ms | FFT 750ms + CIC + interp |
| BH short-range (GPU) | ~105ms | With r_cut cutoff |
| **Total** | **~1065ms** | @ 500K |

**Bottleneck analysis:**
- FFT: ~750ms (CPU rustfft) → 11ms with cuFFT (70x)
- CIC mass assign: ~100ms CPU → ~5ms with GPU kernel
- Force interpolation: ~100ms CPU → ~5ms with GPU kernel
- BH short-range: 105ms (already optimized)

**Estimated performance with full GPU:**
| Component | CPU | Full GPU |
|-----------|-----|----------|
| FFT | 750ms | 11ms |
| CIC assign | 100ms | 5ms |
| Force interp | 100ms | 5ms |
| BH short-range | 105ms | 105ms |
| **Total @ 500K** | **~1065ms** | **~126ms** |
| **Estimated @ 1M** | ~2130ms | ~250ms |

### Grid Artifact Elimination

**Verified by construction:**
- Long-range (r > r_cut): PM FFT is exact on grid scale
- Short-range (r < r_cut): GPU BH only processes nearby pairs
- BH θ=0.5 artifacts were from long-range approximation
- With TreePM, BH never computes long-range → no artifacts

### Next Steps

1. ~~Implement cuFFT FFI~~ (DONE)
2. ~~Implement TreePM hybrid step~~ (DONE)
3. **GPU CIC mass assignment** - CUDA kernel for particle → grid
4. **GPU force interpolation** - CUDA kernel for grid → particle
5. Replace rustfft with cuFFT in hybrid step
6. Target: <200ms/step @ 1M

## References

- cuFFT Documentation: https://docs.nvidia.com/cuda/cufft/
- cuda_setup: https://docs.rs/cuda_setup
- scirs2-fft: https://lib.rs/crates/scirs2-fft
- Rust CUDA updates: https://rust-gpu.github.io/blog/2025/08/11/rust-cuda-update/
