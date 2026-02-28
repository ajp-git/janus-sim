/// GPU N-body with TWO-PASS tree strategy for maximum particle count
///
/// Strategy: Build separate trees for + and - particles sequentially
/// - Pass 1: Tree of N+ particles → forces on ALL particles
/// - Pass 2: Tree of N- particles → forces ACCUMULATED on all
/// - Peak VRAM: only ONE tree in memory at a time
///
/// Node format (single-sign, 36 bytes vs 56 for dual-COM):
///   node_pos: 7×f32 (cx, cy, cz, half_size, com_x, com_y, com_z)
///   node_mass: 1×f64 (mass)
///
/// Target: 100M particles on 12GB VRAM

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Helper to get raw device pointer from CudaSlice for FFI calls
#[cfg(feature = "cuda")]
fn get_device_ptr<T: DeviceRepr>(slice: &CudaSlice<T>) -> u64 {
    let ptr_to_ptr = slice.as_kernel_param();
    // as_kernel_param returns &cu_device_ptr as *mut c_void
    // We dereference to get the actual CUdeviceptr (u64 on 64-bit)
    unsafe { *(ptr_to_ptr as *const u64) }
}

const CUDA_TWOPASS_KERNELS: &str = r#"

// ============================================================================
// TWO-PASS KERNELS: Single-sign tree with simplified node structure
// ============================================================================

extern "C" __global__ void drift_f32(
    float* __restrict__ pos,
    const float* __restrict__ vel,
    float dt, float box_half, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int base = tid * 3;
    pos[base]     += vel[base]     * dt;
    pos[base + 1] += vel[base + 1] * dt;
    pos[base + 2] += vel[base + 2] * dt;
    for (int i = 0; i < 3; i++) {
        if (pos[base + i] > box_half) pos[base + i] -= 2.0f * box_half;
        if (pos[base + i] < -box_half) pos[base + i] += 2.0f * box_half;
    }
}

extern "C" __global__ void kick_f32(
    float* __restrict__ vel,
    const float* __restrict__ acc,  // Changed to float
    float dt, int n,
    float hubble_param, float dtau_per_dt  // Changed to float
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int base = tid * 3;
    for (int d = 0; d < 3; d++) {
        float v = vel[base + d];
        float friction = -hubble_param * v * dtau_per_dt;
        vel[base + d] = v + (acc[base + d] + friction) * dt;
    }
}

// Add PM long-range forces to acceleration buffer (TreePM hybrid)
extern "C" __global__ void add_pm_forces(
    float* __restrict__ acc,
    const float* __restrict__ pm_forces,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int base = tid * 3;
    acc[base]     += pm_forces[base];
    acc[base + 1] += pm_forces[base + 1];
    acc[base + 2] += pm_forces[base + 2];
}

// ============================================================================
// CIC (Cloud-in-Cell) kernels for GPU TreePM
// ============================================================================

// Custom atomicAdd for double (not natively supported on all architectures)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// CIC scatter: distribute particle mass to 8 neighboring grid cells
// Uses atomicAddDouble for thread-safe accumulation
extern "C" __global__ void cic_scatter(
    const float* __restrict__ pos,      // [n × 3] particle positions
    const signed char* __restrict__ signs, // [n] particle signs
    double* __restrict__ rho_plus,      // [grid³] positive density grid
    double* __restrict__ rho_minus,     // [grid³] negative density grid
    int n,                              // number of particles
    int grid_size,                      // grid dimension (cubic)
    float box_half,                     // box_size / 2
    float inv_cell_size                 // grid_size / box_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Get particle position
    float px = pos[tid * 3];
    float py = pos[tid * 3 + 1];
    float pz = pos[tid * 3 + 2];
    int sign = signs[tid];

    // Convert to grid coordinates [0, grid_size)
    float gx = (px + box_half) * inv_cell_size;
    float gy = (py + box_half) * inv_cell_size;
    float gz = (pz + box_half) * inv_cell_size;

    // Handle periodic wrapping
    gx = fmodf(gx + (float)grid_size, (float)grid_size);
    gy = fmodf(gy + (float)grid_size, (float)grid_size);
    gz = fmodf(gz + (float)grid_size, (float)grid_size);

    // Integer cell indices
    int ix = (int)gx;
    int iy = (int)gy;
    int iz = (int)gz;

    // Fractional position within cell [0, 1)
    float fx = gx - (float)ix;
    float fy = gy - (float)iy;
    float fz = gz - (float)iz;

    // CIC weights for 8 neighboring cells
    float wx[2] = {1.0f - fx, fx};
    float wy[2] = {1.0f - fy, fy};
    float wz[2] = {1.0f - fz, fz};

    // Select target grid
    double* grid = (sign > 0) ? rho_plus : rho_minus;

    // Distribute mass to 8 cells (atomic for thread safety)
    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            for (int dk = 0; dk < 2; dk++) {
                int ci = (ix + di) % grid_size;
                int cj = (iy + dj) % grid_size;
                int ck = (iz + dk) % grid_size;
                int idx = ci + grid_size * (cj + grid_size * ck);
                double weight = (double)(wx[di] * wy[dj] * wz[dk]);
                atomicAddDouble(&grid[idx], weight);  // mass = 1.0
            }
        }
    }
}

// CIC gather: interpolate force from grid to particles
// Janus physics: F_+ = -∇φ_plus + ∇φ_minus, F_- = -∇φ_minus + ∇φ_plus
extern "C" __global__ void cic_gather(
    const float* __restrict__ pos,      // [n × 3] particle positions
    const signed char* __restrict__ signs, // [n] particle signs
    const double* __restrict__ phi_plus,   // [grid³] positive potential
    const double* __restrict__ phi_minus,  // [grid³] negative potential
    float* __restrict__ pm_forces,      // [n × 3] output forces
    int n,                              // number of particles
    int grid_size,                      // grid dimension
    float box_half,                     // box_size / 2
    float inv_cell_size,                // grid_size / box_size
    float cell_size                     // box_size / grid_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Get particle position
    float px = pos[tid * 3];
    float py = pos[tid * 3 + 1];
    float pz = pos[tid * 3 + 2];
    int sign = signs[tid];

    // Convert to grid coordinates
    float gx = (px + box_half) * inv_cell_size;
    float gy = (py + box_half) * inv_cell_size;
    float gz = (pz + box_half) * inv_cell_size;

    gx = fmodf(gx + (float)grid_size, (float)grid_size);
    gy = fmodf(gy + (float)grid_size, (float)grid_size);
    gz = fmodf(gz + (float)grid_size, (float)grid_size);

    int ix = (int)gx;
    int iy = (int)gy;
    int iz = (int)gz;

    float fx = gx - (float)ix;
    float fy = gy - (float)iy;
    float fz = gz - (float)iz;

    float wx[2] = {1.0f - fx, fx};
    float wy[2] = {1.0f - fy, fy};
    float wz[2] = {1.0f - fz, fz};

    // Janus: + attracted by +, repelled by -
    //        - attracted by -, repelled by +
    const double* phi_attract = (sign > 0) ? phi_plus : phi_minus;
    const double* phi_repel = (sign > 0) ? phi_minus : phi_plus;

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
    float h = cell_size;
    float inv_2h = 0.5f / h;

    // CIC interpolation with gradient computation
    for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
            for (int dk = 0; dk < 2; dk++) {
                int ci = (ix + di) % grid_size;
                int cj = (iy + dj) % grid_size;
                int ck = (iz + dk) % grid_size;

                float weight = wx[di] * wy[dj] * wz[dk];

                // Neighbor indices for central difference gradient
                int ci_p = (ci + 1) % grid_size;
                int ci_m = (ci + grid_size - 1) % grid_size;
                int cj_p = (cj + 1) % grid_size;
                int cj_m = (cj + grid_size - 1) % grid_size;
                int ck_p = (ck + 1) % grid_size;
                int ck_m = (ck + grid_size - 1) % grid_size;

                // Grid indexing helper
                #define IDX(i,j,k) ((i) + grid_size * ((j) + grid_size * (k)))

                // Gradient of attractive potential
                float dphi_a_dx = (float)(phi_attract[IDX(ci_p,cj,ck)] - phi_attract[IDX(ci_m,cj,ck)]) * inv_2h;
                float dphi_a_dy = (float)(phi_attract[IDX(ci,cj_p,ck)] - phi_attract[IDX(ci,cj_m,ck)]) * inv_2h;
                float dphi_a_dz = (float)(phi_attract[IDX(ci,cj,ck_p)] - phi_attract[IDX(ci,cj,ck_m)]) * inv_2h;

                // Gradient of repulsive potential
                float dphi_r_dx = (float)(phi_repel[IDX(ci_p,cj,ck)] - phi_repel[IDX(ci_m,cj,ck)]) * inv_2h;
                float dphi_r_dy = (float)(phi_repel[IDX(ci,cj_p,ck)] - phi_repel[IDX(ci,cj_m,ck)]) * inv_2h;
                float dphi_r_dz = (float)(phi_repel[IDX(ci,cj,ck_p)] - phi_repel[IDX(ci,cj,ck_m)]) * inv_2h;

                #undef IDX

                // F = -∇φ_attract + ∇φ_repel (Janus physics)
                force_x += weight * (-dphi_a_dx + dphi_r_dx);
                force_y += weight * (-dphi_a_dy + dphi_r_dy);
                force_z += weight * (-dphi_a_dz + dphi_r_dz);
            }
        }
    }

    pm_forces[tid * 3]     = force_x;
    pm_forces[tid * 3 + 1] = force_y;
    pm_forces[tid * 3 + 2] = force_z;
}

// Reset double array to zero
extern "C" __global__ void reset_f64_grid(double* arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = 0.0;
}

// Extract particles of given sign into separate buffer
extern "C" __global__ void extract_by_sign(
    const float* __restrict__ pos_all,
    const signed char* __restrict__ signs_all,  // i8 for memory savings
    float* __restrict__ pos_out,
    int* __restrict__ idx_map,  // maps extracted index → original index
    int n_all,
    int target_sign,
    int* __restrict__ count  // atomic counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_all) return;

    if ((int)signs_all[tid] == target_sign) {
        int out_idx = atomicAdd(count, 1);
        idx_map[out_idx] = tid;
        pos_out[out_idx * 3]     = pos_all[tid * 3];
        pos_out[out_idx * 3 + 1] = pos_all[tid * 3 + 1];
        pos_out[out_idx * 3 + 2] = pos_all[tid * 3 + 2];
    }
}

// Morton codes with particle index tie-breaker for unique keys
// 30 bits Morton (10 bits/axis = 1024 cells) + 32 bits particle index
// Guarantees unique keys → Karras algorithm works correctly → 0 uncovered leaves

__device__ unsigned long long expand10_tp(unsigned int v) {
    // Direct bit spreading: 10 bits -> 30 bits (each bit spaced by 2 zeros)
    // Slower than magic constants but guaranteed correct for 10 bits
    unsigned long long r = 0;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        r |= ((unsigned long long)((v >> i) & 1)) << (3 * i);
    }
    return r;
}

extern "C" __global__ void compute_morton_f32(
    const float* __restrict__ pos,
    unsigned long long* __restrict__ morton,
    int* __restrict__ indices,
    int n, float box_half, float inv_cell_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Scale to [0, 1024) range (10 bits per axis)
    float scale = 1024.0f / (2.0f * box_half);
    float x = (pos[tid * 3]     + box_half) * scale;
    float y = (pos[tid * 3 + 1] + box_half) * scale;
    float z = (pos[tid * 3 + 2] + box_half) * scale;

    unsigned int ix = min(max((unsigned int)x, 0u), 0x3FFu);  // 10 bits
    unsigned int iy = min(max((unsigned int)y, 0u), 0x3FFu);
    unsigned int iz = min(max((unsigned int)z, 0u), 0x3FFu);

    // key = (morton_30bit << 32) | particle_index_32bit
    // Unique keys guarantee correct Karras tree construction
    unsigned long long mc = expand10_tp(ix) | (expand10_tp(iy) << 1) | (expand10_tp(iz) << 2);
    morton[tid] = (mc << 32) | ((unsigned long long)tid);
    indices[tid] = tid;
}

extern "C" __global__ void reorder_f32x3(
    const float* __restrict__ in, float* __restrict__ out,
    const int* __restrict__ sorted_idx, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int src = sorted_idx[tid];
    out[tid * 3]     = in[src * 3];
    out[tid * 3 + 1] = in[src * 3 + 1];
    out[tid * 3 + 2] = in[src * 3 + 2];
}

extern "C" __global__ void reorder_i32(
    const int* __restrict__ in, int* __restrict__ out,
    const int* __restrict__ sorted_idx, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = in[sorted_idx[tid]];
}

// ============================================================================
// SINGLE-SIGN BVH: Simplified nodes with only one COM
// node_pos: 7×f32 [cx, cy, cz, half_size, com_x, com_y, com_z]
// node_mass: 1×f64 [mass]
// Total: 36 bytes/node (vs 56 for dual-COM)
// ============================================================================

__device__ int clz64_tp(unsigned long long x) {
    if (x == 0) return 64;
    int n = 0;
    if (x <= 0x00000000FFFFFFFFULL) { n += 32; x <<= 32; }
    if (x <= 0x0000FFFFFFFFFFFFULL) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFFFFFFFFFFULL) { n += 8;  x <<= 8; }
    if (x <= 0x0FFFFFFFFFFFFFFFULL) { n += 4;  x <<= 4; }
    if (x <= 0x3FFFFFFFFFFFFFFFULL) { n += 2;  x <<= 2; }
    if (x <= 0x7FFFFFFFFFFFFFFFULL) { n += 1; }
    return n;
}

__device__ int delta_tp(const unsigned long long* morton, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    if (morton[i] == morton[j]) return 64 + clz64_tp(i ^ j);
    return clz64_tp(morton[i] ^ morton[j]);
}

__device__ void range_tp(const unsigned long long* morton, int n, int i, int* first, int* last) {
    if (i == 0) { *first = 0; *last = n - 1; return; }
    int dl = delta_tp(morton, n, i, i - 1);
    int dr = delta_tp(morton, n, i, i + 1);
    int d = (dr > dl) ? 1 : -1;
    int dmin = (d > 0) ? dl : dr;
    int lmax = 2;
    while (delta_tp(morton, n, i, i + lmax * d) > dmin) lmax *= 2;
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2)
        if (delta_tp(morton, n, i, i + (l + t) * d) > dmin) l += t;
    int j = i + l * d;
    if (d > 0) { *first = i; *last = j; }
    else { *first = j; *last = i; }
}

__device__ int split_tp(const unsigned long long* morton, int n, int first, int last) {
    if (first == last) return first;
    int dn = delta_tp(morton, n, first, last);
    int s = 0, t = last - first;
    while (t > 1) {
        t = (t + 1) / 2;
        if (delta_tp(morton, n, first, first + s + t) > dn) s += t;
    }
    return first + s;
}

// Karras 2012 radix tree BVH construction
// Each internal node i covers a range determined by delta values of Morton codes
// Node layout: internal nodes 0..n-2, leaves n-1..2n-2
extern "C" __global__ void build_bvh_tp(
    const unsigned long long* __restrict__ morton,
    int* __restrict__ left, int* __restrict__ right, int* __restrict__ parent,
    int* __restrict__ rl, int* __restrict__ rr,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    // Determine range [first, last] for this internal node using delta values
    int first, last;
    range_tp(morton, n, i, &first, &last);

    // Find split point gamma within the range
    int gamma = split_tp(morton, n, first, last);

    // Left child covers [first, gamma]
    // If single element: it's leaf first; else: internal node gamma
    int lc;
    if (first == gamma) {
        lc = n - 1 + first;  // leaf node
    } else {
        lc = gamma;  // internal node
    }

    // Right child covers [gamma+1, last]
    // If single element: it's leaf last; else: internal node gamma+1
    int rc;
    if (gamma + 1 == last) {
        rc = n - 1 + last;  // leaf node
    } else {
        rc = gamma + 1;  // internal node
    }

    // Store tree structure
    left[i] = lc;
    right[i] = rc;
    rl[i] = first;
    rr[i] = last;

    // Set parent pointers (atomic writes handle race conditions)
    parent[lc] = i;
    parent[rc] = i;
}

// Initialize leaves - SINGLE sign tree (FP32)
extern "C" __global__ void init_leaves_tp(
    const float* __restrict__ pos,
    float* __restrict__ node_pos,   // 7×f32 per node
    float* __restrict__ node_mass,  // 1×f32 per node (changed from f64)
    int* __restrict__ node_types,
    int* __restrict__ atomic,
    int n, float box_half
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int node = n - 1 + i;
    int pb = node * 7;

    float x = pos[i * 3];
    float y = pos[i * 3 + 1];
    float z = pos[i * 3 + 2];

    // center = particle pos
    node_pos[pb + 0] = x;
    node_pos[pb + 1] = y;
    node_pos[pb + 2] = z;
    node_pos[pb + 3] = box_half / 1024.0f;
    // COM = particle pos
    node_pos[pb + 4] = x;
    node_pos[pb + 5] = y;
    node_pos[pb + 6] = z;
    node_mass[node] = 1.0f;

    node_types[node] = 1;
    atomic[node] = 0;
}

// Bottom-up reduction - SINGLE sign (FP32)
extern "C" __global__ void reduce_tp(
    const int* __restrict__ left, const int* __restrict__ right,
    const int* __restrict__ parent,
    const int* __restrict__ rl, const int* __restrict__ rr,
    float* __restrict__ node_pos,
    float* __restrict__ node_mass,  // Changed from f64
    int* __restrict__ node_types,
    int* __restrict__ atomic,
    int n, float box_half
) {
    int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= n) return;

    int node = n - 1 + leaf;
    int cur = parent[node];

    while (cur >= 0 && cur < n - 1) {
        int cnt = atomicAdd(&atomic[cur], 1);
        if (cnt == 0) return;

        int lc = left[cur];
        int rc = right[cur];

        int pb_c = cur * 7;
        int pb_l = lc * 7;
        int pb_r = rc * 7;

        float ml = node_mass[lc];
        float mr = node_mass[rc];
        float total = ml + mr;

        // COM weighted average
        float com_x = (ml * node_pos[pb_l + 4] + mr * node_pos[pb_r + 4]) / total;
        float com_y = (ml * node_pos[pb_l + 5] + mr * node_pos[pb_r + 5]) / total;
        float com_z = (ml * node_pos[pb_l + 6] + mr * node_pos[pb_r + 6]) / total;

        // Tight bounding box from children
        float lcx = node_pos[pb_l + 0], lcy = node_pos[pb_l + 1], lcz = node_pos[pb_l + 2];
        float lhs = node_pos[pb_l + 3];
        float rcx = node_pos[pb_r + 0], rcy = node_pos[pb_r + 1], rcz = node_pos[pb_r + 2];
        float rhs = node_pos[pb_r + 3];

        // AABB union of children
        float minx = fminf(lcx - lhs, rcx - rhs);
        float maxx = fmaxf(lcx + lhs, rcx + rhs);
        float miny = fminf(lcy - lhs, rcy - rhs);
        float maxy = fmaxf(lcy + lhs, rcy + rhs);
        float minz = fminf(lcz - lhs, rcz - rhs);
        float maxz = fmaxf(lcz + lhs, rcz + rhs);

        // Geometric center and half_size
        float cx = (minx + maxx) * 0.5f;
        float cy = (miny + maxy) * 0.5f;
        float cz = (minz + maxz) * 0.5f;
        float hs = fmaxf(fmaxf(maxx - minx, maxy - miny), maxz - minz) * 0.5f;
        if (hs < box_half / 1024.0f) hs = box_half / 1024.0f;

        node_pos[pb_c + 0] = cx;  // geometric center for distance check
        node_pos[pb_c + 1] = cy;
        node_pos[pb_c + 2] = cz;
        node_pos[pb_c + 3] = hs;  // tight half_size
        node_pos[pb_c + 4] = com_x;  // COM for force calculation
        node_pos[pb_c + 5] = com_y;
        node_pos[pb_c + 6] = com_z;
        node_mass[cur] = total;

        node_types[cur] = 2;

        if (cur == 0) break;
        cur = parent[cur];
    }
}

// Force computation from single-sign tree - OVERWRITE variant (FP32 for speed)
// tree_sign: +1 if tree contains positive particles, -1 if negative
// Interaction: same sign → attract (+1), opposite → repel (-1)
// Uses float throughout for 60× speedup on consumer GPUs (FP32 >> FP64)
extern "C" __global__ void forces_twopass_overwrite(
    const float* __restrict__ pos_all,      // ALL particles
    const signed char* __restrict__ signs_all,  // i8 for memory savings
    const float* __restrict__ node_pos,     // tree nodes
    const float* __restrict__ node_mass,    // Changed to float for FP32 perf
    const int* __restrict__ left,
    const int* __restrict__ right,
    const int* __restrict__ node_types,
    float* __restrict__ acc,                // Changed to float
    int n_all, int tree_sign, float theta, float softening
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_all) return;

    float px = pos_all[tid * 3];
    float py = pos_all[tid * 3 + 1];
    float pz = pos_all[tid * 3 + 2];
    int my_sign = (int)signs_all[tid];

    // Interaction factor: same sign → +1 (attract), opposite → repel (-1)
    float interaction = (my_sign == tree_sign) ? 1.0f : -1.0f;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float eps2 = softening * softening;

    int stack[32];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;

        int nt = node_types[node];
        if (nt == 0) continue;

        int pb = node * 7;
        float cx = node_pos[pb];
        float cy = node_pos[pb + 1];
        float cz = node_pos[pb + 2];
        float hs = node_pos[pb + 3];

        float dx = cx - px;
        float dy = cy - py;
        float dz = cz - pz;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;

        // MAC without sqrt: (2*hs/r < theta) ↔ (4*hs² < theta²*r²)
        float hs4 = 4.0f * hs * hs;
        float theta2_r2 = theta * theta * r2;

        if (nt == 1 || hs4 < theta2_r2) {
            float mass = node_mass[node];
            float comx = node_pos[pb + 4];
            float comy = node_pos[pb + 5];
            float comz = node_pos[pb + 6];

            float dpx = comx - px;
            float dpy = comy - py;
            float dpz = comz - pz;
            float rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
            float inv_rp3 = rsqrtf(rp2) / rp2;
            float f = interaction * mass * inv_rp3;

            ax += f * dpx;
            ay += f * dpy;
            az += f * dpz;
        } else {
            int lc = left[node];
            int rc = right[node];
            if (lc >= 0 && sp < 31) stack[sp++] = lc;
            if (rc >= 0 && sp < 31) stack[sp++] = rc;
        }
    }

    acc[tid * 3]     = ax;
    acc[tid * 3 + 1] = ay;
    acc[tid * 3 + 2] = az;
}

// Force computation from single-sign tree - ACCUMULATE variant (FP32)
extern "C" __global__ void forces_twopass_accumulate(
    const float* __restrict__ pos_all,
    const signed char* __restrict__ signs_all,  // i8 for memory savings
    const float* __restrict__ node_pos,
    const float* __restrict__ node_mass,        // Changed to float
    const int* __restrict__ left,
    const int* __restrict__ right,
    const int* __restrict__ node_types,
    float* __restrict__ acc,                    // Changed to float
    int n_all, int tree_sign, float theta, float softening
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_all) return;

    float px = pos_all[tid * 3];
    float py = pos_all[tid * 3 + 1];
    float pz = pos_all[tid * 3 + 2];
    int my_sign = (int)signs_all[tid];

    float interaction = (my_sign == tree_sign) ? 1.0f : -1.0f;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float eps2 = softening * softening;

    int stack[32];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;

        int nt = node_types[node];
        if (nt == 0) continue;

        int pb = node * 7;
        float cx = node_pos[pb];
        float cy = node_pos[pb + 1];
        float cz = node_pos[pb + 2];
        float hs = node_pos[pb + 3];

        float dx = cx - px;
        float dy = cy - py;
        float dz = cz - pz;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;

        // MAC without sqrt: (2*hs/r < theta) ↔ (4*hs² < theta²*r²)
        float hs4 = 4.0f * hs * hs;
        float theta2_r2 = theta * theta * r2;

        if (nt == 1 || hs4 < theta2_r2) {
            float mass = node_mass[node];
            float comx = node_pos[pb + 4];
            float comy = node_pos[pb + 5];
            float comz = node_pos[pb + 6];

            float dpx = comx - px;
            float dpy = comy - py;
            float dpz = comz - pz;
            float rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
            float inv_rp3 = rsqrtf(rp2) / rp2;
            float f = interaction * mass * inv_rp3;

            ax += f * dpx;
            ay += f * dpy;
            az += f * dpz;
        } else {
            int lc = left[node];
            int rc = right[node];
            if (lc >= 0 && sp < 31) stack[sp++] = lc;
            if (rc >= 0 && sp < 31) stack[sp++] = rc;
        }
    }

    acc[tid * 3]     += ax;
    acc[tid * 3 + 1] += ay;
    acc[tid * 3 + 2] += az;
}

// ============================================================================
// [OPT-4] WARP-COHERENT TRAVERSAL
// ============================================================================
// All threads in a warp process the same tree node together.
// Uses warp-level primitives: __shfl_sync, __any_sync, __all_sync
// Stack is in shared memory (64 ints per warp = 256 bytes)
// Launch with shared_mem_bytes = 8 warps × 64 × 4 = 2048 bytes per block

extern "C" __global__ void forces_twopass_warpcoherent(
    const float* __restrict__ pos_all,
    const signed char* __restrict__ signs_all,
    const float* __restrict__ node_pos,   // 7×f32 per node
    const float* __restrict__ node_mass,
    const int* __restrict__ left,
    const int* __restrict__ right,
    const int* __restrict__ node_types,
    float* __restrict__ acc,
    int n_all, int tree_sign, float theta, float softening
    // accumulate derived from tree_sign: +1=overwrite, -1=accumulate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // CRITICAL: All threads in warp must participate in sync operations
    // Invalid threads will compute but not write results
    bool valid = (tid < n_all);

    // Derive accumulate from tree_sign (cudarc 12-param limit workaround)
    int accumulate = (tree_sign < 0) ? 1 : 0;

    extern __shared__ int shared_stack[];
    int warp_id_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int* stack = shared_stack + warp_id_in_block * 128;  // 128 levels per warp

    // Use tid=0 data for invalid threads to avoid out-of-bounds reads
    int safe_tid = valid ? tid : 0;
    float px = pos_all[safe_tid*3], py = pos_all[safe_tid*3+1], pz = pos_all[safe_tid*3+2];
    int my_sign = signs_all[safe_tid];
    float sign_factor = (my_sign == tree_sign) ? 1.0f : -1.0f;

    float ax=0, ay=0, az=0;
    float eps2 = softening*softening;
    float theta2 = theta*theta;

    int sp = 0;
    if (lane == 0) stack[sp++] = 0;
    __syncwarp(0xFFFFFFFF);

    while (true) {
        // Lane 0 broadcasts whether stack is empty
        int stack_empty = (lane == 0 && sp <= 0) ? 1 : 0;
        stack_empty = __shfl_sync(0xFFFFFFFF, stack_empty, 0);
        if (stack_empty) break;

        int node;
        if (lane == 0) node = stack[--sp];
        node = __shfl_sync(0xFFFFFFFF, node, 0);

        if (node < 0) continue;
        int nt = node_types[node];
        if (nt == 0) continue;

        int pb = node * 7;
        float cx = node_pos[pb], cy = node_pos[pb+1], cz = node_pos[pb+2];
        float hs = node_pos[pb+3];
        float comx = node_pos[pb+4], comy = node_pos[pb+5], comz = node_pos[pb+6];
        float m = node_mass[node];

        float dx = cx-px, dy = cy-py, dz = cz-pz;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;

        // Opening criterion: leaf OR node far enough to approximate
        bool should_approx = (nt == 1) || ((4.0f*hs*hs) < (theta2*r2));

        // Warp-coherent decision making
        bool all_approx = __all_sync(0xFFFFFFFF, should_approx);
        bool all_descend = __all_sync(0xFFFFFFFF, !should_approx);

        if (all_approx) {
            // ALL threads agree to approximate - compute force
            if (valid) {
                float ddx = comx-px, ddy = comy-py, ddz = comz-pz;
                float rp2 = ddx*ddx + ddy*ddy + ddz*ddz + eps2;
                float irp3 = rsqrtf(rp2) / rp2;
                float f = sign_factor * m * irp3;
                ax += f*ddx; ay += f*ddy; az += f*ddz;
            }
        } else if (all_descend) {
            // ALL threads agree to descend - push children
            if (lane == 0) {
                int lc = left[node], rc = right[node];
                if (rc >= 0 && sp < 127) stack[sp++] = rc;
                if (lc >= 0 && sp < 127) stack[sp++] = lc;
            }
        } else {
            // MIXED decisions - conservative: descend (more accurate)
            // This sacrifices some speed for correctness
            if (lane == 0) {
                int lc = left[node], rc = right[node];
                if (rc >= 0 && sp < 127) stack[sp++] = rc;
                if (lc >= 0 && sp < 127) stack[sp++] = lc;
            }
        }
        __syncwarp();
    }

    // Only valid threads write results
    if (valid) {
        if (accumulate) {
            acc[tid*3] += ax; acc[tid*3+1] += ay; acc[tid*3+2] += az;
        } else {
            acc[tid*3] = ax; acc[tid*3+1] = ay; acc[tid*3+2] = az;
        }
    }
}

// ============================================================================
// [OPT-2] SHARED MEMORY CACHED TOP NODES
// ============================================================================
// Cache the top 1024 BVH nodes in shared memory (~44KB)
// These nodes are visited by almost every thread, so caching saves ~1024
// global memory accesses per thread.
//
// Layout per node: 7 floats (pos) + 1 float (mass) + 2 ints (left/right) + 1 int (type)
// = 11 words × 1024 = 44KB (fits in 48KB Ampere shared memory)

#define TOP_NODES_SHMEM 1024

extern "C" __global__ void forces_twopass_shmem_overwrite(
    const float* __restrict__ pos_all,
    const signed char* __restrict__ signs_all,
    const float* __restrict__ node_pos,
    const float* __restrict__ node_mass,
    const int* __restrict__ left,
    const int* __restrict__ right,
    const int* __restrict__ node_types,
    float* __restrict__ acc,
    int n_all, int tree_sign, float theta, float softening
) {
    // Shared memory for top 1024 nodes
    __shared__ float sh_node_pos[TOP_NODES_SHMEM * 7];   // 28KB
    __shared__ float sh_node_mass[TOP_NODES_SHMEM];       // 4KB
    __shared__ int sh_left[TOP_NODES_SHMEM];              // 4KB
    __shared__ int sh_right[TOP_NODES_SHMEM];             // 4KB
    __shared__ int sh_node_types[TOP_NODES_SHMEM];        // 4KB
    // Total: 44KB

    // Cooperative loading of top 1024 nodes (all threads participate)
    // For 4M+ particles, n_nodes >> 1024, so we always load full 1024
    for (int i = threadIdx.x; i < TOP_NODES_SHMEM; i += blockDim.x) {
        sh_node_pos[i * 7 + 0] = node_pos[i * 7 + 0];
        sh_node_pos[i * 7 + 1] = node_pos[i * 7 + 1];
        sh_node_pos[i * 7 + 2] = node_pos[i * 7 + 2];
        sh_node_pos[i * 7 + 3] = node_pos[i * 7 + 3];
        sh_node_pos[i * 7 + 4] = node_pos[i * 7 + 4];
        sh_node_pos[i * 7 + 5] = node_pos[i * 7 + 5];
        sh_node_pos[i * 7 + 6] = node_pos[i * 7 + 6];
        sh_node_mass[i] = node_mass[i];
        sh_left[i] = left[i];
        sh_right[i] = right[i];
        sh_node_types[i] = node_types[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_all) return;

    float px = pos_all[tid * 3];
    float py = pos_all[tid * 3 + 1];
    float pz = pos_all[tid * 3 + 2];
    int my_sign = (int)signs_all[tid];

    float interaction = (my_sign == tree_sign) ? 1.0f : -1.0f;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float eps2 = softening * softening;
    float theta2 = theta * theta;

    int stack[32];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;

        // Choose shared or global memory based on node index
        int nt;
        float cx, cy, cz, hs, mass, comx, comy, comz;
        int lc, rc;

        if (node < TOP_NODES_SHMEM) {
            // Read from shared memory (top 1024 nodes)
            nt = sh_node_types[node];
            if (nt == 0) continue;
            int pb = node * 7;
            cx = sh_node_pos[pb];
            cy = sh_node_pos[pb + 1];
            cz = sh_node_pos[pb + 2];
            hs = sh_node_pos[pb + 3];
            mass = sh_node_mass[node];
            comx = sh_node_pos[pb + 4];
            comy = sh_node_pos[pb + 5];
            comz = sh_node_pos[pb + 6];
            lc = sh_left[node];
            rc = sh_right[node];
        } else {
            // Read from global memory (deep nodes)
            nt = node_types[node];
            if (nt == 0) continue;
            int pb = node * 7;
            cx = node_pos[pb];
            cy = node_pos[pb + 1];
            cz = node_pos[pb + 2];
            hs = node_pos[pb + 3];
            mass = node_mass[node];
            comx = node_pos[pb + 4];
            comy = node_pos[pb + 5];
            comz = node_pos[pb + 6];
            lc = left[node];
            rc = right[node];
        }

        float dx = cx - px;
        float dy = cy - py;
        float dz = cz - pz;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;

        float hs4 = 4.0f * hs * hs;
        float theta2_r2 = theta2 * r2;

        if (nt == 1 || hs4 < theta2_r2) {
            float dpx = comx - px;
            float dpy = comy - py;
            float dpz = comz - pz;
            float rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
            float inv_rp3 = rsqrtf(rp2) / rp2;
            float f = interaction * mass * inv_rp3;

            ax += f * dpx;
            ay += f * dpy;
            az += f * dpz;
        } else {
            if (lc >= 0 && sp < 31) stack[sp++] = lc;
            if (rc >= 0 && sp < 31) stack[sp++] = rc;
        }
    }

    acc[tid * 3]     = ax;
    acc[tid * 3 + 1] = ay;
    acc[tid * 3 + 2] = az;
}

extern "C" __global__ void forces_twopass_shmem_accumulate(
    const float* __restrict__ pos_all,
    const signed char* __restrict__ signs_all,
    const float* __restrict__ node_pos,
    const float* __restrict__ node_mass,
    const int* __restrict__ left,
    const int* __restrict__ right,
    const int* __restrict__ node_types,
    float* __restrict__ acc,
    int n_all, int tree_sign, float theta, float softening
) {
    __shared__ float sh_node_pos[TOP_NODES_SHMEM * 7];
    __shared__ float sh_node_mass[TOP_NODES_SHMEM];
    __shared__ int sh_left[TOP_NODES_SHMEM];
    __shared__ int sh_right[TOP_NODES_SHMEM];
    __shared__ int sh_node_types[TOP_NODES_SHMEM];

    // Load top 1024 nodes (for 4M+ particles, n_nodes >> 1024)
    for (int i = threadIdx.x; i < TOP_NODES_SHMEM; i += blockDim.x) {
        sh_node_pos[i * 7 + 0] = node_pos[i * 7 + 0];
        sh_node_pos[i * 7 + 1] = node_pos[i * 7 + 1];
        sh_node_pos[i * 7 + 2] = node_pos[i * 7 + 2];
        sh_node_pos[i * 7 + 3] = node_pos[i * 7 + 3];
        sh_node_pos[i * 7 + 4] = node_pos[i * 7 + 4];
        sh_node_pos[i * 7 + 5] = node_pos[i * 7 + 5];
        sh_node_pos[i * 7 + 6] = node_pos[i * 7 + 6];
        sh_node_mass[i] = node_mass[i];
        sh_left[i] = left[i];
        sh_right[i] = right[i];
        sh_node_types[i] = node_types[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_all) return;

    float px = pos_all[tid * 3];
    float py = pos_all[tid * 3 + 1];
    float pz = pos_all[tid * 3 + 2];
    int my_sign = (int)signs_all[tid];

    float interaction = (my_sign == tree_sign) ? 1.0f : -1.0f;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float eps2 = softening * softening;
    float theta2 = theta * theta;

    int stack[32];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;

        int nt;
        float cx, cy, cz, hs, mass, comx, comy, comz;
        int lc, rc;

        if (node < TOP_NODES_SHMEM) {
            nt = sh_node_types[node];
            if (nt == 0) continue;
            int pb = node * 7;
            cx = sh_node_pos[pb];
            cy = sh_node_pos[pb + 1];
            cz = sh_node_pos[pb + 2];
            hs = sh_node_pos[pb + 3];
            mass = sh_node_mass[node];
            comx = sh_node_pos[pb + 4];
            comy = sh_node_pos[pb + 5];
            comz = sh_node_pos[pb + 6];
            lc = sh_left[node];
            rc = sh_right[node];
        } else {
            nt = node_types[node];
            if (nt == 0) continue;
            int pb = node * 7;
            cx = node_pos[pb];
            cy = node_pos[pb + 1];
            cz = node_pos[pb + 2];
            hs = node_pos[pb + 3];
            mass = node_mass[node];
            comx = node_pos[pb + 4];
            comy = node_pos[pb + 5];
            comz = node_pos[pb + 6];
            lc = left[node];
            rc = right[node];
        }

        float dx = cx - px;
        float dy = cy - py;
        float dz = cz - pz;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;

        float hs4 = 4.0f * hs * hs;
        float theta2_r2 = theta2 * r2;

        if (nt == 1 || hs4 < theta2_r2) {
            float dpx = comx - px;
            float dpy = comy - py;
            float dpz = comz - pz;
            float rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
            float inv_rp3 = rsqrtf(rp2) / rp2;
            float f = interaction * mass * inv_rp3;

            ax += f * dpx;
            ay += f * dpy;
            az += f * dpz;
        } else {
            if (lc >= 0 && sp < 31) stack[sp++] = lc;
            if (rc >= 0 && sp < 31) stack[sp++] = rc;
        }
    }

    acc[tid * 3]     += ax;
    acc[tid * 3 + 1] += ay;
    acc[tid * 3 + 2] += az;
}

// ============================================================================
// DIRECT N² KERNEL: Shared memory tiling (CUDA SDK particles.cu style)
// ============================================================================
// O(N²) exact computation - no tree, no approximation
// Uses shared memory to maximize arithmetic intensity

extern "C" __global__ void forces_direct_n2(
    const float* __restrict__ pos,      // [n×3] SoA
    const signed char* __restrict__ signs,
    float* __restrict__ acc,            // [n×3] output
    int n, float eta, float eps2
) {
    extern __shared__ float sh_data[];  // 256 × 4 floats = 4KB
    float* sh_x = sh_data;              // [256]
    float* sh_y = sh_data + 256;        // [256]
    float* sh_z = sh_data + 512;        // [256]
    signed char* sh_sign = (signed char*)(sh_data + 768);  // [256]

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load my particle
    float px = 0, py = 0, pz = 0;
    int my_sign = 0;
    if (i < n) {
        px = pos[i * 3];
        py = pos[i * 3 + 1];
        pz = pos[i * 3 + 2];
        my_sign = (int)signs[i];
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Process all tiles
    int n_tiles = (n + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < n_tiles; tile++) {
        // Cooperative load: each thread loads one particle to shared memory
        int j_load = tile * blockDim.x + threadIdx.x;
        if (j_load < n) {
            sh_x[threadIdx.x] = pos[j_load * 3];
            sh_y[threadIdx.x] = pos[j_load * 3 + 1];
            sh_z[threadIdx.x] = pos[j_load * 3 + 2];
            sh_sign[threadIdx.x] = signs[j_load];
        } else {
            sh_x[threadIdx.x] = 0;
            sh_y[threadIdx.x] = 0;
            sh_z[threadIdx.x] = 0;
            sh_sign[threadIdx.x] = 0;
        }
        __syncthreads();

        // Each thread computes against all 256 particles in tile
        if (i < n) {
            int tile_end = min(blockDim.x, n - tile * blockDim.x);
            #pragma unroll 8
            for (int j = 0; j < tile_end; j++) {
                float dx = sh_x[j] - px;
                float dy = sh_y[j] - py;
                float dz = sh_z[j] - pz;
                float r2 = dx*dx + dy*dy + dz*dz + eps2;
                float inv_r3 = rsqrtf(r2) / r2;

                // Janus physics: same sign attracts, opposite repels
                int other_sign = (int)sh_sign[j];
                float interaction = (my_sign == other_sign) ? 1.0f : -eta;
                float f = interaction * inv_r3;  // mass = 1

                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        __syncthreads();
    }

    // Write result
    if (i < n) {
        acc[i * 3]     = ax;
        acc[i * 3 + 1] = ay;
        acc[i * 3 + 2] = az;
    }
}

extern "C" __global__ void reset_i32(int* __restrict__ buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = 0;
}

extern "C" __global__ void set_i32(int* __restrict__ buf, int n, int val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = val;
}

extern "C" __global__ void reset_f64(double* __restrict__ buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = 0.0;
}

extern "C" __global__ void reset_f32(float* __restrict__ buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = 0.0f;
}

// ============================================================================
// MORTON REORDER: Sort ALL particles by space-filling curve to reduce warp divergence
// ============================================================================

// Compute Morton codes for ALL particles (not just one sign)
extern "C" __global__ void compute_morton_all(
    const float* __restrict__ pos,
    unsigned long long* __restrict__ morton,
    int* __restrict__ indices,
    int n, float box_half, float inv_cell_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float px = pos[tid * 3];
    float py = pos[tid * 3 + 1];
    float pz = pos[tid * 3 + 2];

    // Normalize to [0, 2^21-1] for 21-bit Morton per axis
    unsigned int ix = (unsigned int)fminf(fmaxf((px + box_half) * inv_cell_size, 0.0f), 2097151.0f);
    unsigned int iy = (unsigned int)fminf(fmaxf((py + box_half) * inv_cell_size, 0.0f), 2097151.0f);
    unsigned int iz = (unsigned int)fminf(fmaxf((pz + box_half) * inv_cell_size, 0.0f), 2097151.0f);

    // Expand to 63 bits total (21 bits × 3)
    unsigned long long mx = ix, my = iy, mz = iz;
    mx = (mx | (mx << 32)) & 0x1f00000000ffffULL;
    mx = (mx | (mx << 16)) & 0x1f0000ff0000ffULL;
    mx = (mx | (mx << 8))  & 0x100f00f00f00f00fULL;
    mx = (mx | (mx << 4))  & 0x10c30c30c30c30c3ULL;
    mx = (mx | (mx << 2))  & 0x1249249249249249ULL;

    my = (my | (my << 32)) & 0x1f00000000ffffULL;
    my = (my | (my << 16)) & 0x1f0000ff0000ffULL;
    my = (my | (my << 8))  & 0x100f00f00f00f00fULL;
    my = (my | (my << 4))  & 0x10c30c30c30c30c3ULL;
    my = (my | (my << 2))  & 0x1249249249249249ULL;

    mz = (mz | (mz << 32)) & 0x1f00000000ffffULL;
    mz = (mz | (mz << 16)) & 0x1f0000ff0000ffULL;
    mz = (mz | (mz << 8))  & 0x100f00f00f00f00fULL;
    mz = (mz | (mz << 4))  & 0x10c30c30c30c30c3ULL;
    mz = (mz | (mz << 2))  & 0x1249249249249249ULL;

    morton[tid] = mx | (my << 1) | (mz << 2);
    indices[tid] = tid;
}

// Reorder float3 array by sorted indices
extern "C" __global__ void reorder_by_idx_f32x3(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ idx,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int src_idx = idx[tid];
    dst[tid * 3]     = src[src_idx * 3];
    dst[tid * 3 + 1] = src[src_idx * 3 + 1];
    dst[tid * 3 + 2] = src[src_idx * 3 + 2];
}

// Reorder signed char array by sorted indices
extern "C" __global__ void reorder_by_idx_i8(
    const signed char* __restrict__ src,
    signed char* __restrict__ dst,
    const int* __restrict__ idx,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = src[idx[tid]];
}

// ============================================================================
// GPU RADIX SORT: 8 passes × 8 bits, fully on-device
// ============================================================================

// Radix histogram: compute local histogram per block
extern "C" __global__ void radix_histogram(
    const unsigned long long* __restrict__ keys,
    unsigned int* __restrict__ hist,   // [n_blocks × 256]
    int n, int bit_shift
) {
    __shared__ unsigned int local_hist[256];
    if (threadIdx.x < 256) local_hist[threadIdx.x] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int bucket = (keys[tid] >> bit_shift) & 0xFF;
        atomicAdd(&local_hist[bucket], 1);
    }
    __syncthreads();

    if (threadIdx.x < 256)
        hist[blockIdx.x * 256 + threadIdx.x] = local_hist[threadIdx.x];
}

// Prefix sum on histograms: compute global offsets for scatter
extern "C" __global__ void radix_prefix_sum(
    unsigned int* __restrict__ hist,          // [n_blocks × 256] in-place
    unsigned int* __restrict__ global_offsets, // [256] out
    int n_blocks
) {
    int bucket = threadIdx.x; // 256 threads, 1 per bucket
    if (bucket >= 256) return;

    // Compute prefix sum for this bucket across all blocks
    unsigned int sum = 0;
    for (int b = 0; b < n_blocks; b++) {
        unsigned int val = hist[b * 256 + bucket];
        hist[b * 256 + bucket] = sum;
        sum += val;
    }

    // Exclusive prefix sum across buckets
    __shared__ unsigned int tmp[256];
    tmp[bucket] = sum;
    __syncthreads();

    for (int stride = 1; stride < 256; stride *= 2) {
        unsigned int v = (bucket >= stride) ? tmp[bucket - stride] : 0;
        __syncthreads();
        tmp[bucket] += v;
        __syncthreads();
    }

    global_offsets[bucket] = tmp[bucket] - sum;
}

// Scatter: place each key at its final position (STABLE version using warp voting)
// Uses __ballot_sync + __popc for stable local ranking within warps
extern "C" __global__ void radix_scatter(
    const unsigned long long* __restrict__ keys_in,
    unsigned long long* __restrict__ keys_out,
    const int* __restrict__ vals_in,
    int* __restrict__ vals_out,
    unsigned int* __restrict__ hist,          // [n_blocks × 256]
    const unsigned int* __restrict__ global_offsets, // [256]
    int n, int bit_shift
) {
    __shared__ unsigned int base_offsets[256];  // Base offset for each bucket
    __shared__ unsigned int warp_counts[8][256]; // Per-warp counts for each bucket

    int block_start = blockIdx.x * blockDim.x;
    int tid = block_start + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Initialize base offsets from global prefix sums
    if (threadIdx.x < 256)
        base_offsets[threadIdx.x] = global_offsets[threadIdx.x]
                                   + hist[blockIdx.x * 256 + threadIdx.x];

    // Initialize warp counts to 0
    if (threadIdx.x < 256) {
        for (int w = 0; w < 8; w++) warp_counts[w][threadIdx.x] = 0;
    }
    __syncthreads();

    // Each thread computes its bucket
    unsigned int my_bucket = 0;
    unsigned long long key = 0;
    int val = 0;
    bool valid = (tid < n);

    if (valid) {
        key = keys_in[tid];
        val = vals_in[tid];
        my_bucket = (key >> bit_shift) & 0xFF;
    }

    // Phase 1: Count elements per bucket per warp (using atomics within warp)
    if (valid) {
        atomicAdd(&warp_counts[warp_id][my_bucket], 1);
    }
    __syncthreads();

    // Phase 2: Compute warp offsets (prefix sum of warp counts)
    // Thread 0 does this sequentially (simple but not parallel)
    __shared__ unsigned int warp_offsets[8][256];
    if (threadIdx.x == 0) {
        for (int b = 0; b < 256; b++) {
            unsigned int sum = 0;
            for (int w = 0; w < 8; w++) {
                warp_offsets[w][b] = sum;
                sum += warp_counts[w][b];
            }
        }
    }
    __syncthreads();

    // Phase 3: Compute local rank within warp using ballot
    if (valid) {
        // Get mask of threads in this warp with same bucket
        unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
        unsigned int same_bucket_mask = 0;

        // Check each lane for same bucket (expensive but correct)
        for (int i = 0; i < 32; i++) {
            unsigned int other_bucket = __shfl_sync(0xFFFFFFFF, my_bucket, i);
            if (other_bucket == my_bucket) {
                same_bucket_mask |= (1u << i);
            }
        }

        // Count how many preceding lanes have the same bucket
        unsigned int preceding_mask = (1u << lane_id) - 1;
        int local_rank = __popc(same_bucket_mask & preceding_mask);

        // Final position = base + warp_offset + local_rank
        unsigned int pos = base_offsets[my_bucket] + warp_offsets[warp_id][my_bucket] + local_rank;
        keys_out[pos] = key;
        vals_out[pos] = val;
    }
}

// ============================================================================
// TreePM SHORT-RANGE: GPU BH with r_cut cutoff
// ============================================================================
// For TreePM: only compute forces for r < r_cut
// Long-range (r > r_cut) handled by PM cuFFT
// This eliminates grid artifacts by construction

// TreePM short-range: n_all_signed encodes both count and tree_sign
// n_all_signed > 0 → tree_sign = +1, n_all = n_all_signed
// n_all_signed < 0 → tree_sign = -1, n_all = -n_all_signed (accumulate mode)
extern "C" __global__ void forces_treepm_short_range(
    const float* __restrict__ pos_all,
    const signed char* __restrict__ signs_all,
    const float* __restrict__ node_pos,   // 7×f32 per node
    const float* __restrict__ node_mass,
    const int* __restrict__ left,
    const int* __restrict__ right,
    const int* __restrict__ node_types,
    float* __restrict__ acc,
    int n_all_signed, float theta, float softening, float r_cut
) {
    int n_all = (n_all_signed > 0) ? n_all_signed : -n_all_signed;
    int tree_sign = (n_all_signed > 0) ? 1 : -1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = (tid < n_all);
    int accumulate = (tree_sign < 0) ? 1 : 0;

    extern __shared__ int shared_stack[];
    int warp_id_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int* stack = shared_stack + warp_id_in_block * 128;

    int safe_tid = valid ? tid : 0;
    float px = pos_all[safe_tid*3], py = pos_all[safe_tid*3+1], pz = pos_all[safe_tid*3+2];
    int my_sign = signs_all[safe_tid];
    float sign_factor = (my_sign == tree_sign) ? 1.0f : -1.0f;

    float ax=0, ay=0, az=0;
    float eps2 = softening*softening;
    float theta2 = theta*theta;
    float r_cut_sq = r_cut * r_cut;

    int sp = 0;
    if (lane == 0) stack[sp++] = 0;
    __syncwarp(0xFFFFFFFF);

    while (true) {
        int stack_empty = (lane == 0 && sp <= 0) ? 1 : 0;
        stack_empty = __shfl_sync(0xFFFFFFFF, stack_empty, 0);
        if (stack_empty) break;

        int node;
        if (lane == 0) node = stack[--sp];
        node = __shfl_sync(0xFFFFFFFF, node, 0);

        if (node < 0) continue;
        int nt = node_types[node];
        if (nt == 0) continue;

        int pb = node * 7;
        float cx = node_pos[pb], cy = node_pos[pb+1], cz = node_pos[pb+2];
        float hs = node_pos[pb+3];
        float comx = node_pos[pb+4], comy = node_pos[pb+5], comz = node_pos[pb+6];
        float m = node_mass[node];

        float dx = cx-px, dy = cy-py, dz = cz-pz;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-20f;
        float r = sqrtf(r2);

        // TreePM optimization: skip entire subtree if closest point is beyond r_cut
        // Closest point of cell to particle is at distance (r - hs)
        float closest_dist = r - hs;
        bool cell_beyond_rcut = (closest_dist > r_cut);
        if (__all_sync(0xFFFFFFFF, cell_beyond_rcut)) continue;

        bool should_approx = (nt == 1) || ((4.0f*hs*hs) < (theta2*r2));
        bool all_approx = __all_sync(0xFFFFFFFF, should_approx);
        bool all_descend = __all_sync(0xFFFFFFFF, !should_approx);

        if (all_approx) {
            if (valid) {
                float ddx = comx-px, ddy = comy-py, ddz = comz-pz;
                float rp2 = ddx*ddx + ddy*ddy + ddz*ddz + eps2;

                // BH computes full force (no erfc splitting for now)
                // PM handles residual long-range with k-space damping
                if (rp2 < r_cut_sq) {
                    float irp3 = rsqrtf(rp2) / rp2;  // 1/r³
                    float f = sign_factor * m * irp3;
                    ax += f*ddx; ay += f*ddy; az += f*ddz;
                }
            }
        } else if (all_descend) {
            if (lane == 0) {
                int lc = left[node], rc = right[node];
                if (rc >= 0 && sp < 127) stack[sp++] = rc;
                if (lc >= 0 && sp < 127) stack[sp++] = lc;
            }
        } else {
            if (lane == 0) {
                int lc = left[node], rc = right[node];
                if (rc >= 0 && sp < 127) stack[sp++] = rc;
                if (lc >= 0 && sp < 127) stack[sp++] = lc;
            }
        }
        __syncwarp();
    }

    if (valid) {
        if (accumulate) {
            acc[tid*3] += ax; acc[tid*3+1] += ay; acc[tid*3+2] += az;
        } else {
            acc[tid*3] = ax; acc[tid*3+1] = ay; acc[tid*3+2] = az;
        }
    }
}

"#;

/// Two-pass GPU N-body simulation
#[cfg(feature = "cuda")]
pub struct GpuNBodyTwoPass {
    device: Arc<CudaDevice>,
    // ALL particles
    pos: CudaSlice<f32>,
    vel: CudaSlice<f32>,
    acc: CudaSlice<f32>,  // Changed to f32 for FP32 performance
    signs: CudaSlice<i8>,  // +1 or -1, i8 saves 3 bytes/particle
    // Temporary buffers for single-sign extraction
    pos_sign: CudaSlice<f32>,      // extracted positions (max N)
    pos_sign_tmp: CudaSlice<f32>,
    idx_map: CudaSlice<i32>,       // maps extracted → original index
    idx_map_tmp: CudaSlice<i32>,
    extract_count: CudaSlice<i32>, // atomic counter for extraction
    // Morton sorting + radix sort buffers
    morton_codes: CudaSlice<u64>,
    morton_codes_tmp: CudaSlice<u64>,
    sorted_indices: CudaSlice<i32>,
    sorted_indices_tmp: CudaSlice<i32>,
    radix_hist: CudaSlice<u32>,      // [n_blocks × 256]
    radix_global: CudaSlice<u32>,    // [256]
    // Global Morton reorder buffers (sized for N_total = N+ + N-)
    morton_all: CudaSlice<u64>,
    morton_all_tmp: CudaSlice<u64>,
    sorted_all: CudaSlice<i32>,
    sorted_all_tmp: CudaSlice<i32>,
    pos_sorted: CudaSlice<f32>,      // [n_total × 3]
    vel_sorted: CudaSlice<f32>,      // [n_total × 3] - for Morton reorder
    signs_sorted: CudaSlice<i8>,     // [n_total]
    radix_hist_all: CudaSlice<u32>,  // for n_total radix sort
    // BVH for single-sign tree (sized for max(N+, N-))
    bvh_left: CudaSlice<i32>,
    bvh_right: CudaSlice<i32>,
    bvh_parent: CudaSlice<i32>,
    bvh_rl: CudaSlice<i32>,
    bvh_rr: CudaSlice<i32>,
    bvh_node_pos: CudaSlice<f32>,   // 7×f32 per node (vs 10 for dual-COM)
    bvh_node_mass: CudaSlice<f32>,  // 1×f32 per node (FP32 for speed)
    bvh_node_types: CudaSlice<i32>,
    bvh_atomic: CudaSlice<i32>,
    // TreePM hybrid buffers
    pm_forces: CudaSlice<f32>,  // [n_total × 3] PM long-range forces
    // PM grid buffers (GPU cuFFT integration)
    rho_plus: CudaSlice<f64>,   // [grid³] positive density
    rho_minus: CudaSlice<f64>,  // [grid³] negative density
    phi_plus: CudaSlice<f64>,   // [grid³] positive potential
    phi_minus: CudaSlice<f64>,  // [grid³] negative potential
    pm_grid_size: usize,        // PM grid dimension (128)
    // Parameters
    n_particles: usize,
    n_positive: usize,
    n_negative: usize,
    theta: f64,
    softening: f64,
    box_size: f64,
    time: f64,
    step_count: usize,
}

/// Timing breakdown for tree build phases (in milliseconds)
#[cfg(feature = "cuda")]
#[derive(Default, Clone)]
pub struct TreeBuildTiming {
    pub morton_ms: u128,
    pub sort_ms: u128,
    pub reorder_ms: u128,
    pub reset_ms: u128,
    pub bvh_karras_ms: u128,
    pub init_leaves_ms: u128,
    pub reduce_ms: u128,
}

#[cfg(feature = "cuda")]
impl GpuNBodyTwoPass {
    /// Create simulation with default virial factor (0.3)
    pub fn new(n_positive: usize, n_negative: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_virial_factor(n_positive, n_negative, box_size, 0.3)
    }

    /// Create simulation with custom virial factor
    /// virial_velocity = sqrt(N/box) × virial_factor
    /// - 0.3 = original value (may be too cold for large N)
    /// - 0.5 = recommended for N > 1M (prevents premature collapse)
    pub fn new_with_virial_factor(n_positive: usize, n_negative: usize, box_size: f64, virial_factor: f64) -> Result<Self, Box<dyn std::error::Error>> {
        use std::time::Instant;

        println!("  [1/6] Initializing CUDA device...");
        let t0 = Instant::now();
        let device = CudaDevice::new(0)?;
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        println!("  [2/6] Compiling CUDA kernels...");
        let t0 = Instant::now();
        let ptx = cudarc::nvrtc::compile_ptx(CUDA_TWOPASS_KERNELS)?;
        device.load_ptx(ptx, "twopass", &[
            "drift_f32", "kick_f32", "extract_by_sign",
            "compute_morton_f32", "reorder_f32x3", "reorder_i32",
            "build_bvh_tp", "init_leaves_tp", "reduce_tp",
            "forces_twopass_overwrite", "forces_twopass_accumulate",
            "forces_twopass_warpcoherent",
            "forces_twopass_shmem_overwrite", "forces_twopass_shmem_accumulate",
            "forces_direct_n2",
            "compute_morton_all", "reorder_by_idx_f32x3", "reorder_by_idx_i8",
            "reset_i32", "reset_f64", "reset_f32", "set_i32",
            "radix_histogram", "radix_prefix_sum", "radix_scatter",
            "forces_treepm_short_range",  // TreePM short-range with r_cut
            "add_pm_forces",  // Add PM long-range forces to acc
            "cic_scatter", "cic_gather", "reset_f64_grid"  // GPU CIC for TreePM
        ])?;
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        let n_total = n_positive + n_negative;
        let n_max_sign = n_positive.max(n_negative);
        let n_bvh = 2 * n_max_sign;  // Only need tree for ONE sign at a time!

        // Generate particles with UNIFORM RANDOM positions (like reference GpuNBodySimulation)
        // Note: Zel'dovich ICs were found to suppress segregation - reverted to uniform random
        println!("  [3/6] Generating {} particles with uniform random ICs...", n_total);
        let t0 = Instant::now();
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);

        let mut pos_data = Vec::with_capacity(n_total * 3);
        let mut vel_data = Vec::with_capacity(n_total * 3);
        let mut signs_data: Vec<i8> = Vec::with_capacity(n_total);

        // Initial velocity scale: virial_velocity = sqrt(N/box) × virial_factor
        // Factor 0.3 was found too cold for large N (causes premature collapse)
        // Factor 0.5 recommended for N > 1M
        let virial_velocity = ((n_total as f64) / box_size).sqrt() * virial_factor;

        // Generate + particles first (like reference)
        for _ in 0..n_positive {
            let x = (rng.random::<f64>() - 0.5) * box_size;
            let y = (rng.random::<f64>() - 0.5) * box_size;
            let z = (rng.random::<f64>() - 0.5) * box_size;
            let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
            let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
            let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
            pos_data.extend([x as f32, y as f32, z as f32]);
            vel_data.extend([vx as f32, vy as f32, vz as f32]);
            signs_data.push(1);
        }

        // Then - particles (like reference)
        for _ in 0..n_negative {
            let x = (rng.random::<f64>() - 0.5) * box_size;
            let y = (rng.random::<f64>() - 0.5) * box_size;
            let z = (rng.random::<f64>() - 0.5) * box_size;
            let vx = (rng.random::<f64>() - 0.5) * virial_velocity;
            let vy = (rng.random::<f64>() - 0.5) * virial_velocity;
            let vz = (rng.random::<f64>() - 0.5) * virial_velocity;
            pos_data.extend([x as f32, y as f32, z as f32]);
            vel_data.extend([vx as f32, vy as f32, vz as f32]);
            signs_data.push(-1);
        }

        println!("         Generated: N+ = {}, N- = {}", n_positive, n_negative);
        println!("         virial_velocity = {:.4} (factor = {:.2})", virial_velocity, virial_factor);
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        // NOTE: virial_factor controls initial KE. Too small → premature collapse.
        // 0.3 works for small N (<1M), 0.5 recommended for large N (>1M).
        // Exact PE_binding virialization over-stabilizes and prevents segregation.

        // Allocate GPU buffers
        println!("  [4/5] Copying data to GPU...");
        let t0 = Instant::now();
        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let acc = device.alloc_zeros::<f32>(n_total * 3)?;  // FP32 for performance
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        // Single-sign extraction buffers (sized for max sign count)
        println!("  [5/5] Allocating GPU buffers...");
        let t0 = Instant::now();
        let pos_sign = device.alloc_zeros::<f32>(n_max_sign * 3)?;
        let pos_sign_tmp = device.alloc_zeros::<f32>(n_max_sign * 3)?;
        let idx_map = device.alloc_zeros::<i32>(n_max_sign)?;
        let idx_map_tmp = device.alloc_zeros::<i32>(n_max_sign)?;
        let extract_count = device.alloc_zeros::<i32>(1)?;

        // Morton sorting + radix sort buffers (for single-sign)
        let n_blocks = (n_max_sign + 255) / 256;
        let morton_codes = device.alloc_zeros::<u64>(n_max_sign)?;
        let morton_codes_tmp = device.alloc_zeros::<u64>(n_max_sign)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_max_sign)?;
        let sorted_indices_tmp = device.alloc_zeros::<i32>(n_max_sign)?;
        let radix_hist = device.alloc_zeros::<u32>(n_blocks * 256)?;
        let radix_global = device.alloc_zeros::<u32>(256)?;

        // Global Morton reorder buffers (for ALL particles = n_total)
        let n_blocks_all = (n_total + 255) / 256;
        let morton_all = device.alloc_zeros::<u64>(n_total)?;
        let morton_all_tmp = device.alloc_zeros::<u64>(n_total)?;
        let sorted_all = device.alloc_zeros::<i32>(n_total)?;
        let sorted_all_tmp = device.alloc_zeros::<i32>(n_total)?;
        let pos_sorted = device.alloc_zeros::<f32>(n_total * 3)?;
        let vel_sorted = device.alloc_zeros::<f32>(n_total * 3)?;  // for Morton reorder
        let signs_sorted = device.alloc_zeros::<i8>(n_total)?;
        let radix_hist_all = device.alloc_zeros::<u32>(n_blocks_all * 256)?;

        // BVH for single-sign (7×f32 + 1×f64 per node = 36 bytes vs 56)
        let bvh_left = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_right = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_rl = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_rr = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_node_pos = device.alloc_zeros::<f32>(n_bvh * 7)?;  // 7 floats per node
        let bvh_node_mass = device.alloc_zeros::<f32>(n_bvh)?;     // 1 float per node (FP32)
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_atomic = device.alloc_zeros::<i32>(n_bvh)?;

        // TreePM hybrid buffers
        let pm_forces = device.alloc_zeros::<f32>(n_total * 3)?;  // PM long-range forces

        // PM grid buffers (128³ = 2M cells × f64 = 16MB per grid × 4 = 64MB)
        let pm_grid_size = 128usize;
        let grid_cells = pm_grid_size * pm_grid_size * pm_grid_size;
        let rho_plus = device.alloc_zeros::<f64>(grid_cells)?;
        let rho_minus = device.alloc_zeros::<f64>(grid_cells)?;
        let phi_plus = device.alloc_zeros::<f64>(grid_cells)?;
        let phi_minus = device.alloc_zeros::<f64>(grid_cells)?;
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        // Memory calculation (signs = i8, saves 3 bytes/particle)
        let mem_particles = (n_total * 3 * 4 * 2 + n_total * 3 * 8 + n_total * 1) as f64 / 1e9;
        let mem_extract = (n_max_sign * 3 * 4 * 2 + n_max_sign * 4 * 2) as f64 / 1e9;
        // Sort: morton×2 + indices×2 + hist + global
        let mem_sort = (n_max_sign * (8 + 8 + 4 + 4) + n_blocks * 256 * 4 + 256 * 4) as f64 / 1e9;
        let mem_bvh = (n_bvh * 4 * 5 + n_bvh * 7 * 4 + n_bvh * 8 + n_bvh * 4 * 2) as f64 / 1e9;
        let total = mem_particles + mem_extract + mem_sort + mem_bvh;

        println!("  VRAM estimate: {:.2} GB", total);
        println!("    - Particles (all): {:.2} GB", mem_particles);
        println!("    - Extract buffers: {:.2} GB", mem_extract);
        println!("    - Morton sort: {:.2} GB", mem_sort);
        println!("    - BVH (single-sign): {:.2} GB", mem_bvh);
        println!("    - Tree size: {} nodes (36 bytes each)", n_bvh);

        Ok(Self {
            device,
            pos, vel, acc, signs,
            pos_sign, pos_sign_tmp, idx_map, idx_map_tmp, extract_count,
            morton_codes, morton_codes_tmp, sorted_indices, sorted_indices_tmp,
            radix_hist, radix_global,
            morton_all, morton_all_tmp, sorted_all, sorted_all_tmp,
            pos_sorted, vel_sorted, signs_sorted, radix_hist_all,
            bvh_left, bvh_right, bvh_parent, bvh_rl, bvh_rr,
            bvh_node_pos, bvh_node_mass, bvh_node_types, bvh_atomic,
            pm_forces,
            rho_plus, rho_minus, phi_plus, phi_minus, pm_grid_size,
            n_particles: n_total,
            n_positive,
            n_negative,
            theta: 5.0,  // Aggressive but fast; validated on 8M
            softening: 0.1,
            box_size,
            time: 0.0,
            step_count: 0,
        })
    }

    /// Create simulation with custom initial conditions (no virialization)
    pub fn with_custom_ics(
        pos_data: Vec<f32>,
        vel_data: Vec<f32>,
        signs_data: Vec<i8>,
        box_size: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use std::time::Instant;

        let n_total = signs_data.len();
        assert_eq!(pos_data.len(), n_total * 3);
        assert_eq!(vel_data.len(), n_total * 3);

        let n_positive = signs_data.iter().filter(|&&s| s > 0).count();
        let n_negative = n_total - n_positive;

        println!("Creating GpuNBodyTwoPass with custom ICs...");
        println!("  N_total = {}, N+ = {}, N- = {}", n_total, n_positive, n_negative);

        // Initialize CUDA
        let device = CudaDevice::new(0)?;
        let ptx = cudarc::nvrtc::compile_ptx(CUDA_TWOPASS_KERNELS)?;
        device.load_ptx(ptx, "twopass", &[
            "drift_f32", "kick_f32", "extract_by_sign",
            "compute_morton_f32", "reorder_f32x3", "reorder_i32",
            "build_bvh_tp", "init_leaves_tp", "reduce_tp",
            "forces_twopass_overwrite", "forces_twopass_accumulate",
            "forces_twopass_warpcoherent",
            "forces_twopass_shmem_overwrite", "forces_twopass_shmem_accumulate",
            "forces_direct_n2",
            "compute_morton_all", "reorder_by_idx_f32x3", "reorder_by_idx_i8",
            "reset_i32", "reset_f64", "reset_f32", "set_i32",
            "radix_histogram", "radix_prefix_sum", "radix_scatter",
            "forces_treepm_short_range",
            "add_pm_forces",
            "cic_scatter", "cic_gather", "reset_f64_grid"
        ])?;

        // Buffer sizes
        let n_max_sign = n_positive.max(n_negative);
        let n_bvh = 2 * n_max_sign - 1;

        // Copy data to GPU
        let t0 = Instant::now();
        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let acc = device.alloc_zeros::<f32>(n_total * 3)?;
        println!("  Copied to GPU ({:.2}s)", t0.elapsed().as_secs_f64());

        // Allocate all buffers (same as in new())
        let pos_sign = device.alloc_zeros::<f32>(n_max_sign * 3)?;
        let pos_sign_tmp = device.alloc_zeros::<f32>(n_max_sign * 3)?;
        let idx_map = device.alloc_zeros::<i32>(n_max_sign)?;
        let idx_map_tmp = device.alloc_zeros::<i32>(n_max_sign)?;
        let extract_count = device.alloc_zeros::<i32>(1)?;

        let n_blocks = (n_max_sign + 255) / 256;
        let morton_codes = device.alloc_zeros::<u64>(n_max_sign)?;
        let morton_codes_tmp = device.alloc_zeros::<u64>(n_max_sign)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_max_sign)?;
        let sorted_indices_tmp = device.alloc_zeros::<i32>(n_max_sign)?;
        let radix_hist = device.alloc_zeros::<u32>(n_blocks * 256)?;
        let radix_global = device.alloc_zeros::<u32>(256)?;

        let n_blocks_all = (n_total + 255) / 256;
        let morton_all = device.alloc_zeros::<u64>(n_total)?;
        let morton_all_tmp = device.alloc_zeros::<u64>(n_total)?;
        let sorted_all = device.alloc_zeros::<i32>(n_total)?;
        let sorted_all_tmp = device.alloc_zeros::<i32>(n_total)?;
        let pos_sorted = device.alloc_zeros::<f32>(n_total * 3)?;
        let vel_sorted = device.alloc_zeros::<f32>(n_total * 3)?;  // for Morton reorder
        let signs_sorted = device.alloc_zeros::<i8>(n_total)?;
        let radix_hist_all = device.alloc_zeros::<u32>(n_blocks_all * 256)?;

        let bvh_left = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_right = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_rl = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_rr = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_node_pos = device.alloc_zeros::<f32>(n_bvh * 7)?;
        let bvh_node_mass = device.alloc_zeros::<f32>(n_bvh)?;
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_atomic = device.alloc_zeros::<i32>(n_bvh)?;

        let pm_forces = device.alloc_zeros::<f32>(n_total * 3)?;

        let pm_grid_size = 128usize;
        let grid_cells = pm_grid_size * pm_grid_size * pm_grid_size;
        let rho_plus = device.alloc_zeros::<f64>(grid_cells)?;
        let rho_minus = device.alloc_zeros::<f64>(grid_cells)?;
        let phi_plus = device.alloc_zeros::<f64>(grid_cells)?;
        let phi_minus = device.alloc_zeros::<f64>(grid_cells)?;

        println!("  GPU buffers allocated");

        Ok(Self {
            device,
            pos, vel, acc, signs,
            pos_sign, pos_sign_tmp, idx_map, idx_map_tmp, extract_count,
            morton_codes, morton_codes_tmp, sorted_indices, sorted_indices_tmp,
            radix_hist, radix_global,
            morton_all, morton_all_tmp, sorted_all, sorted_all_tmp,
            pos_sorted, vel_sorted, signs_sorted, radix_hist_all,
            bvh_left, bvh_right, bvh_parent, bvh_rl, bvh_rr,
            bvh_node_pos, bvh_node_mass, bvh_node_types, bvh_atomic,
            pm_forces,
            rho_plus, rho_minus, phi_plus, phi_minus, pm_grid_size,
            n_particles: n_total,
            n_positive,
            n_negative,
            theta: 0.5,
            softening: 0.1,
            box_size,
            time: 0.0,
            step_count: 0,
        })
    }

    pub fn set_theta(&mut self, theta: f64) { self.theta = theta; }

    /// Build tree for particles of given sign, returning timing breakdown
    fn build_single_sign_tree(&mut self, _sign: i32, n_sign: usize) -> Result<TreeBuildTiming, Box<dyn std::error::Error>> {
        use std::time::Instant;
        let mut timing = TreeBuildTiming::default();

        self.device.synchronize()?;  // Ensure previous kernels completed

        let n = n_sign;
        let n_internal = n.saturating_sub(1);
        let n_nodes = 2 * n - 1;
        let box_half = (self.box_size / 2.0) as f32;
        let inv_cell_size = (2097152.0 / self.box_size) as f32;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Morton codes
        let t_morton = Instant::now();
        let morton_k = self.device.get_func("twopass", "compute_morton_f32")
            .ok_or("compute_morton_f32 not found")?;
        unsafe {
            morton_k.launch(cfg, (
                &self.pos_sign, &mut self.morton_codes, &mut self.sorted_indices,
                n as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;
        timing.morton_ms = t_morton.elapsed().as_millis();

        // GPU Radix Sort: 8 passes × 8 bits, fully on-device
        let t_sort = Instant::now();
        let hist_k = self.device.get_func("twopass", "radix_histogram")
            .ok_or("radix_histogram not found")?;
        let prefix_k = self.device.get_func("twopass", "radix_prefix_sum")
            .ok_or("radix_prefix_sum not found")?;
        let scatter_k = self.device.get_func("twopass", "radix_scatter")
            .ok_or("radix_scatter not found")?;

        let _prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // GPU radix sort: 8 passes × 8 bits, fully on-device
        // Uses stable warp-voting scatter for Karras algorithm correctness
        let n_blocks = blocks;
        let hist_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (n_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // 8 passes: bits 0-7, 8-15, 16-23, 24-31, 32-39, 40-47, 48-55, 56-63
        for pass in 0..8 {
            let bit_shift = pass * 8;

            // Determine which buffers to use (ping-pong)
            let (keys_in, keys_out, vals_in, vals_out) = if pass % 2 == 0 {
                (&self.morton_codes, &mut self.morton_codes_tmp,
                 &self.sorted_indices, &mut self.sorted_indices_tmp)
            } else {
                (&self.morton_codes_tmp, &mut self.morton_codes,
                 &self.sorted_indices_tmp, &mut self.sorted_indices)
            };

            // Step 1: Histogram
            unsafe {
                hist_k.clone().launch(hist_cfg, (
                    keys_in, &mut self.radix_hist, n as i32, bit_shift as i32
                ))?;
            }

            // Step 2: Prefix sum
            unsafe {
                prefix_k.clone().launch(prefix_cfg, (
                    &mut self.radix_hist, &mut self.radix_global, n_blocks as i32
                ))?;
            }

            // Step 3: Scatter (stable using warp voting)
            unsafe {
                scatter_k.clone().launch(hist_cfg, (
                    keys_in, keys_out, vals_in, vals_out,
                    &self.radix_hist, &self.radix_global,
                    n as i32, bit_shift as i32
                ))?;
            }
        }

        // After 8 passes (even number), result is in morton_codes/sorted_indices
        self.device.synchronize()?;
        timing.sort_ms = t_sort.elapsed().as_millis();

        // Reorder positions
        let t_reorder = Instant::now();
        let reorder_f32 = self.device.get_func("twopass", "reorder_f32x3")
            .ok_or("reorder_f32x3 not found")?;
        unsafe {
            reorder_f32.clone().launch(cfg, (
                &self.pos_sign, &mut self.pos_sign_tmp, &self.sorted_indices, n as i32
            ))?;
        }
        std::mem::swap(&mut self.pos_sign, &mut self.pos_sign_tmp);

        // Reorder idx_map
        let reorder_i32 = self.device.get_func("twopass", "reorder_i32")
            .ok_or("reorder_i32 not found")?;
        unsafe {
            reorder_i32.launch(cfg, (
                &self.idx_map, &mut self.idx_map_tmp, &self.sorted_indices, n as i32
            ))?;
        }
        std::mem::swap(&mut self.idx_map, &mut self.idx_map_tmp);
        self.device.synchronize()?;
        timing.reorder_ms = t_reorder.elapsed().as_millis();

        // Reset BVH buffers - CRITICAL: parent must be -1 for root termination
        let t_reset = Instant::now();
        let reset_blocks = (n_nodes + 255) / 256;
        let reset_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (reset_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let reset_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;
        let set_k = self.device.get_func("twopass", "set_i32")
            .ok_or("set_i32 not found")?;
        unsafe {
            reset_k.clone().launch(reset_cfg, (&mut self.bvh_atomic, n_nodes as i32))?;
            reset_k.clone().launch(reset_cfg, (&mut self.bvh_node_types, n_nodes as i32))?;
            // Parent must be -1 so root (node 0) terminates correctly
            set_k.launch(reset_cfg, (&mut self.bvh_parent, n_nodes as i32, -1i32))?;
        }
        self.device.synchronize()?;
        timing.reset_ms = t_reset.elapsed().as_millis();

        // Build BVH on GPU using Karras algorithm
        // Morton keys now include particle index → unique keys → correct tree
        let t_bvh = Instant::now();
        if n_internal > 0 {
            let internal_blocks = (n_internal + 255) / 256;
            let internal_cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: (internal_blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let build_k = self.device.get_func("twopass", "build_bvh_tp")
                .ok_or("build_bvh_tp not found")?;
            unsafe {
                build_k.launch(internal_cfg, (
                    &self.morton_codes,
                    &mut self.bvh_left, &mut self.bvh_right, &mut self.bvh_parent,
                    &mut self.bvh_rl, &mut self.bvh_rr,
                    n as i32,
                ))?;
            }
        }
        self.device.synchronize()?;
        timing.bvh_karras_ms = t_bvh.elapsed().as_millis();

        // Initialize leaves
        let t_leaves = Instant::now();
        let init_k = self.device.get_func("twopass", "init_leaves_tp")
            .ok_or("init_leaves_tp not found")?;
        unsafe {
            init_k.launch(cfg, (
                &self.pos_sign,
                &mut self.bvh_node_pos, &mut self.bvh_node_mass,
                &mut self.bvh_node_types, &mut self.bvh_atomic,
                n as i32, box_half,
            ))?;
        }
        self.device.synchronize()?;
        timing.init_leaves_ms = t_leaves.elapsed().as_millis();

        // Bottom-up reduction
        let t_reduce = Instant::now();
        let reduce_k = self.device.get_func("twopass", "reduce_tp")
            .ok_or("reduce_tp not found")?;
        unsafe {
            reduce_k.launch(cfg, (
                &self.bvh_left, &self.bvh_right, &self.bvh_parent,
                &self.bvh_rl, &self.bvh_rr,
                &mut self.bvh_node_pos, &mut self.bvh_node_mass,
                &mut self.bvh_node_types, &mut self.bvh_atomic,
                n as i32, box_half,
            ))?;
        }
        self.device.synchronize()?;
        timing.reduce_ms = t_reduce.elapsed().as_millis();

        Ok(timing)
    }

    /// Compute forces on all particles (needed for initializing DKD with cold ICs)
    pub fn compute_forces(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;
        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Reset acceleration to zero (f32 buffer)
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }

        // Extract and build tree for positive particles
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;
        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;

        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
        }
        unsafe {
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let _ = self.build_single_sign_tree(1, self.n_positive)?;

        // Compute forces from positive tree (OVERWRITE)
        let forces_ow_k = self.device.get_func("twopass", "forces_twopass_overwrite")
            .ok_or("forces_twopass_overwrite not found")?;
        unsafe {
            forces_ow_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;

        // Extract and build tree for negative particles
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
        }
        unsafe {
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let _ = self.build_single_sign_tree(-1, self.n_negative)?;

        // Compute forces from negative tree (ACCUMULATE)
        let forces_acc_k = self.device.get_func("twopass", "forces_twopass_accumulate")
            .ok_or("forces_twopass_accumulate not found")?;
        unsafe {
            forces_acc_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    pub fn step_dkd(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        self.step_count += 1;
        let step_num = self.step_count;

        // Timing accumulators
        let mut t_drift1_ms = 0u128;
        let mut t_extract_pos_ms = 0u128;
        let mut t_extract_neg_ms = 0u128;
        let mut t_force_pos_ms = 0u128;
        let mut t_force_neg_ms = 0u128;
        let mut t_kick_ms = 0u128;
        let mut t_drift2_ms = 0u128;
        let mut tree_pos = TreeBuildTiming::default();
        let mut tree_neg = TreeBuildTiming::default();

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        // Reset acceleration to zero (f32 buffer)
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }
        self.device.synchronize()?;
        t_drift1_ms = t0.elapsed().as_millis();

        // ===== PASS 1: Positive particles =====
        // Extract
        let t0 = Instant::now();
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;
        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;
        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
        }
        unsafe {
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        t_extract_pos_ms = t0.elapsed().as_millis();

        // Build tree +
        tree_pos = self.build_single_sign_tree(1, self.n_positive)?;

        // Force + (overwrite)
        let t0 = Instant::now();
        let forces_ow_k = self.device.get_func("twopass", "forces_twopass_overwrite")
            .ok_or("forces_twopass_overwrite not found")?;
        unsafe {
            forces_ow_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        t_force_pos_ms = t0.elapsed().as_millis();

        // ===== PASS 2: Negative particles =====
        // Extract
        let t0 = Instant::now();
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
        }
        unsafe {
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        t_extract_neg_ms = t0.elapsed().as_millis();

        // Build tree -
        tree_neg = self.build_single_sign_tree(-1, self.n_negative)?;

        // Force - (accumulate)
        let t0 = Instant::now();
        let forces_acc_k = self.device.get_func("twopass", "forces_twopass_accumulate")
            .ok_or("forces_twopass_accumulate not found")?;
        unsafe {
            forces_acc_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        t_force_neg_ms = t0.elapsed().as_millis();

        // ===== KICK (dt) =====
        let t0 = Instant::now();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;
        t_kick_ms = t0.elapsed().as_millis();

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        t_drift2_ms = t0.elapsed().as_millis();

        self.time += dt;

        // ===== PRINT TIMING BREAKDOWN =====
        let tree_pos_total = tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.reorder_ms
            + tree_pos.reset_ms + tree_pos.bvh_karras_ms + tree_pos.init_leaves_ms + tree_pos.reduce_ms;
        let tree_neg_total = tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.reorder_ms
            + tree_neg.reset_ms + tree_neg.bvh_karras_ms + tree_neg.init_leaves_ms + tree_neg.reduce_ms;
        let total = t_drift1_ms + t_extract_pos_ms + tree_pos_total + t_force_pos_ms
            + t_extract_neg_ms + tree_neg_total + t_force_neg_ms + t_kick_ms + t_drift2_ms;

        eprintln!("[step {:03}] TIMING BREAKDOWN (ms):", step_num);
        eprintln!("  drift (1st half)      : {:>6} ms", t_drift1_ms);
        eprintln!("  --- PASS + (N={}) ---", self.n_positive);
        eprintln!("    extract+            : {:>6} ms", t_extract_pos_ms);
        eprintln!("    morton codes        : {:>6} ms", tree_pos.morton_ms);
        eprintln!("    radix sort          : {:>6} ms", tree_pos.sort_ms);
        eprintln!("    reorder             : {:>6} ms", tree_pos.reorder_ms);
        eprintln!("    reset buffers       : {:>6} ms", tree_pos.reset_ms);
        eprintln!("    build_bvh (karras)  : {:>6} ms", tree_pos.bvh_karras_ms);
        eprintln!("    init_leaves         : {:>6} ms", tree_pos.init_leaves_ms);
        eprintln!("    reduce_tp           : {:>6} ms", tree_pos.reduce_ms);
        eprintln!("    force+ (overwrite)  : {:>6} ms  <<<", t_force_pos_ms);
        eprintln!("  --- PASS - (N={}) ---", self.n_negative);
        eprintln!("    extract-            : {:>6} ms", t_extract_neg_ms);
        eprintln!("    morton codes        : {:>6} ms", tree_neg.morton_ms);
        eprintln!("    radix sort          : {:>6} ms", tree_neg.sort_ms);
        eprintln!("    reorder             : {:>6} ms", tree_neg.reorder_ms);
        eprintln!("    reset buffers       : {:>6} ms", tree_neg.reset_ms);
        eprintln!("    build_bvh (karras)  : {:>6} ms", tree_neg.bvh_karras_ms);
        eprintln!("    init_leaves         : {:>6} ms", tree_neg.init_leaves_ms);
        eprintln!("    reduce_tp           : {:>6} ms", tree_neg.reduce_ms);
        eprintln!("    force- (accumulate) : {:>6} ms  <<<", t_force_neg_ms);
        eprintln!("  kick                  : {:>6} ms", t_kick_ms);
        eprintln!("  drift (2nd half)      : {:>6} ms", t_drift2_ms);
        eprintln!("  ─────────────────────────────────");
        eprintln!("  TOTAL                 : {:>6} ms", total);
        eprintln!();

        Ok(())
    }

    /// Direct N² force computation - O(N²) exact, no tree approximation
    /// Uses shared memory tiling for maximum GPU efficiency
    pub fn step_dkd_direct(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        self.step_count += 1;
        let step_num = self.step_count;

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let n = self.n_particles;
        let eta = 1.045f32;  // Janus parameter
        let eps2 = (self.softening * self.softening) as f32;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Shared memory config for N² kernel: 256 × (3 floats + 1 byte) aligned
        let cfg_n2 = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 256 * 4 * 4,  // 256 × 4 floats = 4KB
        };

        // ===== DRIFT (dt/2) =====
        let t_drift = Instant::now();
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t_drift_ms = t_drift.elapsed().as_millis();

        // ===== FORCE (N² direct) =====
        let t_force = Instant::now();
        let force_k = self.device.get_func("twopass", "forces_direct_n2")
            .ok_or("forces_direct_n2 not found")?;
        unsafe {
            force_k.launch(cfg_n2, (
                &self.pos, &self.signs, &mut self.acc,
                n as i32, eta, eps2,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_ms = t_force.elapsed().as_millis();

        // ===== KICK (dt) =====
        let t_kick = Instant::now();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_kick_ms = t_kick.elapsed().as_millis();

        // ===== DRIFT (dt/2) =====
        let t_drift2 = Instant::now();
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t_drift2_ms = t_drift2.elapsed().as_millis();

        self.time += dt;

        // Print timing
        let total = t_drift_ms + t_force_ms + t_kick_ms + t_drift2_ms;
        eprintln!("[step {:03}] N² DIRECT TIMING (ms):", step_num);
        eprintln!("  drift (1st)  : {:>8} ms", t_drift_ms);
        eprintln!("  force N²     : {:>8} ms  <<< O(N²) = {}×{}", t_force_ms, n, n);
        eprintln!("  kick         : {:>8} ms", t_kick_ms);
        eprintln!("  drift (2nd)  : {:>8} ms", t_drift2_ms);
        eprintln!("  ─────────────────────────");
        eprintln!("  TOTAL        : {:>8} ms", total);
        eprintln!();

        Ok(())
    }

    /// Morton-reorder step: Sort all particles by Morton code to reduce warp divergence
    /// Spatially nearby particles get consecutive thread IDs → coherent tree traversal
    pub fn step_dkd_morton_reorder(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        self.step_count += 1;
        let step_num = self.step_count;

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let inv_cell_size = (2097152.0 / self.box_size) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        // Reset acceleration
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }
        self.device.synchronize()?;
        let t_drift_ms = t0.elapsed().as_millis();

        // ===== MORTON REORDER ALL PARTICLES =====
        let t0 = Instant::now();

        // Compute Morton codes for all particles
        let morton_k = self.device.get_func("twopass", "compute_morton_all")
            .ok_or("compute_morton_all not found")?;
        unsafe {
            morton_k.launch(cfg, (
                &self.pos, &mut self.morton_all, &mut self.sorted_all,
                n as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Radix sort Morton codes (8 passes)
        let hist_k = self.device.get_func("twopass", "radix_histogram")
            .ok_or("radix_histogram not found")?;
        let prefix_k = self.device.get_func("twopass", "radix_prefix_sum")
            .ok_or("radix_prefix_sum not found")?;
        let scatter_k = self.device.get_func("twopass", "radix_scatter")
            .ok_or("radix_scatter not found")?;

        let n_blocks_all = blocks;
        let hist_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (n_blocks_all as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        for pass in 0..8 {
            let bit_shift = pass * 8;
            let (keys_in, keys_out, vals_in, vals_out) = if pass % 2 == 0 {
                (&self.morton_all, &mut self.morton_all_tmp,
                 &self.sorted_all, &mut self.sorted_all_tmp)
            } else {
                (&self.morton_all_tmp, &mut self.morton_all,
                 &self.sorted_all_tmp, &mut self.sorted_all)
            };

            unsafe {
                hist_k.clone().launch(hist_cfg, (
                    keys_in, &mut self.radix_hist_all, n as i32, bit_shift as i32
                ))?;
                prefix_k.clone().launch(prefix_cfg, (
                    &mut self.radix_hist_all, &mut self.radix_global, n_blocks_all as i32
                ))?;
                scatter_k.clone().launch(hist_cfg, (
                    keys_in, keys_out, vals_in, vals_out,
                    &self.radix_hist_all, &self.radix_global,
                    n as i32, bit_shift as i32
                ))?;
            }
        }
        self.device.synchronize()?;

        // Reorder positions, velocities, and signs by sorted indices
        // CRITICAL: Must reorder vel to keep pos[i]/vel[i] correspondence
        let reorder_pos_k = self.device.get_func("twopass", "reorder_by_idx_f32x3")
            .ok_or("reorder_by_idx_f32x3 not found")?;
        let reorder_sign_k = self.device.get_func("twopass", "reorder_by_idx_i8")
            .ok_or("reorder_by_idx_i8 not found")?;

        unsafe {
            reorder_pos_k.clone().launch(cfg, (
                &self.pos, &mut self.pos_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_pos_k.clone().launch(cfg, (
                &self.vel, &mut self.vel_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_sign_k.launch(cfg, (
                &self.signs, &mut self.signs_sorted, &self.sorted_all, n as i32
            ))?;
        }
        // Swap buffers: sorted becomes main
        std::mem::swap(&mut self.pos, &mut self.pos_sorted);
        std::mem::swap(&mut self.vel, &mut self.vel_sorted);
        std::mem::swap(&mut self.signs, &mut self.signs_sorted);
        self.device.synchronize()?;
        let t_reorder_ms = t0.elapsed().as_millis();

        // ===== PASS 1: Positive particles tree =====
        let t0 = Instant::now();
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;
        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;

        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_pos = self.build_single_sign_tree(1, self.n_positive)?;

        // Force + (overwrite)
        let forces_ow_k = self.device.get_func("twopass", "forces_twopass_overwrite")
            .ok_or("forces_twopass_overwrite not found")?;
        unsafe {
            forces_ow_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_pos_ms = t0.elapsed().as_millis();

        // ===== PASS 2: Negative particles tree =====
        let t0 = Instant::now();
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_neg = self.build_single_sign_tree(-1, self.n_negative)?;

        // Force - (accumulate)
        let forces_acc_k = self.device.get_func("twopass", "forces_twopass_accumulate")
            .ok_or("forces_twopass_accumulate not found")?;
        unsafe {
            forces_acc_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_neg_ms = t0.elapsed().as_millis();

        // ===== KICK (dt) =====
        let t0 = Instant::now();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_kick_ms = t0.elapsed().as_millis();

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t_drift2_ms = t0.elapsed().as_millis();

        self.time += dt;

        // Print timing
        let total = t_drift_ms + t_reorder_ms + t_force_pos_ms + t_force_neg_ms + t_kick_ms + t_drift2_ms;
        eprintln!("[step {:03}] MORTON-REORDER TIMING (ms):", step_num);
        eprintln!("  drift (1st)           : {:>6} ms", t_drift_ms);
        eprintln!("  morton sort+reorder   : {:>6} ms", t_reorder_ms);
        eprintln!("  --- PASS + ---");
        eprintln!("    tree build          : {:>6} ms", tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.bvh_karras_ms + tree_pos.reduce_ms);
        eprintln!("    force+ (overwrite)  : {:>6} ms  <<<", t_force_pos_ms - (tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.bvh_karras_ms + tree_pos.reduce_ms + tree_pos.reorder_ms + tree_pos.reset_ms + tree_pos.init_leaves_ms));
        eprintln!("  --- PASS - ---");
        eprintln!("    tree build          : {:>6} ms", tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.bvh_karras_ms + tree_neg.reduce_ms);
        eprintln!("    force- (accumulate) : {:>6} ms  <<<", t_force_neg_ms - (tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.bvh_karras_ms + tree_neg.reduce_ms + tree_neg.reorder_ms + tree_neg.reset_ms + tree_neg.init_leaves_ms));
        eprintln!("  kick                  : {:>6} ms", t_kick_ms);
        eprintln!("  drift (2nd)           : {:>6} ms", t_drift2_ms);
        eprintln!("  ─────────────────────────────────");
        eprintln!("  TOTAL                 : {:>6} ms", total);
        eprintln!();

        Ok(())
    }

    /// [OPT-2] Morton reorder + shared memory cached top-1024 nodes
    pub fn step_dkd_morton_shmem(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        self.step_count += 1;
        let step_num = self.step_count;

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let inv_cell_size = (2097152.0 / self.box_size) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        // Reset acceleration
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }
        self.device.synchronize()?;
        let t_drift_ms = t0.elapsed().as_millis();

        // ===== MORTON REORDER ALL PARTICLES =====
        let t0 = Instant::now();

        let morton_k = self.device.get_func("twopass", "compute_morton_all")
            .ok_or("compute_morton_all not found")?;
        unsafe {
            morton_k.launch(cfg, (
                &self.pos, &mut self.morton_all, &mut self.sorted_all,
                n as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Radix sort Morton codes
        let hist_k = self.device.get_func("twopass", "radix_histogram")
            .ok_or("radix_histogram not found")?;
        let prefix_k = self.device.get_func("twopass", "radix_prefix_sum")
            .ok_or("radix_prefix_sum not found")?;
        let scatter_k = self.device.get_func("twopass", "radix_scatter")
            .ok_or("radix_scatter not found")?;

        let n_blocks_all = blocks;
        let hist_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (n_blocks_all as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        for pass in 0..8 {
            let bit_shift = pass * 8;
            let (keys_in, keys_out, vals_in, vals_out) = if pass % 2 == 0 {
                (&self.morton_all, &mut self.morton_all_tmp,
                 &self.sorted_all, &mut self.sorted_all_tmp)
            } else {
                (&self.morton_all_tmp, &mut self.morton_all,
                 &self.sorted_all_tmp, &mut self.sorted_all)
            };

            unsafe {
                hist_k.clone().launch(hist_cfg, (
                    keys_in, &mut self.radix_hist_all, n as i32, bit_shift as i32
                ))?;
                prefix_k.clone().launch(prefix_cfg, (
                    &mut self.radix_hist_all, &mut self.radix_global, n_blocks_all as i32
                ))?;
                scatter_k.clone().launch(hist_cfg, (
                    keys_in, keys_out, vals_in, vals_out,
                    &self.radix_hist_all, &self.radix_global,
                    n as i32, bit_shift as i32
                ))?;
            }
        }
        self.device.synchronize()?;

        // Reorder positions, velocities, and signs
        // CRITICAL: Must reorder vel to keep pos[i]/vel[i] correspondence
        let reorder_pos_k = self.device.get_func("twopass", "reorder_by_idx_f32x3")
            .ok_or("reorder_by_idx_f32x3 not found")?;
        let reorder_sign_k = self.device.get_func("twopass", "reorder_by_idx_i8")
            .ok_or("reorder_by_idx_i8 not found")?;

        unsafe {
            reorder_pos_k.clone().launch(cfg, (
                &self.pos, &mut self.pos_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_pos_k.clone().launch(cfg, (
                &self.vel, &mut self.vel_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_sign_k.launch(cfg, (
                &self.signs, &mut self.signs_sorted, &self.sorted_all, n as i32
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_sorted);
        std::mem::swap(&mut self.vel, &mut self.vel_sorted);
        std::mem::swap(&mut self.signs, &mut self.signs_sorted);
        self.device.synchronize()?;
        let t_reorder_ms = t0.elapsed().as_millis();

        // ===== PASS 1: Positive particles tree =====
        let t0 = Instant::now();
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;
        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;

        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_pos = self.build_single_sign_tree(1, self.n_positive)?;

        // Force + (overwrite) with shared memory
        let forces_ow_k = self.device.get_func("twopass", "forces_twopass_shmem_overwrite")
            .ok_or("forces_twopass_shmem_overwrite not found")?;
        unsafe {
            forces_ow_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_pos_ms = t0.elapsed().as_millis();

        // ===== PASS 2: Negative particles tree =====
        let t0 = Instant::now();
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_neg = self.build_single_sign_tree(-1, self.n_negative)?;

        // Force - (accumulate) with shared memory
        let forces_acc_k = self.device.get_func("twopass", "forces_twopass_shmem_accumulate")
            .ok_or("forces_twopass_shmem_accumulate not found")?;
        unsafe {
            forces_acc_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_neg_ms = t0.elapsed().as_millis();

        // ===== KICK (dt) =====
        let t0 = Instant::now();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_kick_ms = t0.elapsed().as_millis();

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t_drift2_ms = t0.elapsed().as_millis();

        self.time += dt;

        // Print timing
        let total = t_drift_ms + t_reorder_ms + t_force_pos_ms + t_force_neg_ms + t_kick_ms + t_drift2_ms;
        eprintln!("[step {:03}] MORTON+SHMEM TIMING (ms):", step_num);
        eprintln!("  drift (1st)           : {:>6} ms", t_drift_ms);
        eprintln!("  morton sort+reorder   : {:>6} ms", t_reorder_ms);
        eprintln!("  --- PASS + ---");
        eprintln!("    tree build          : {:>6} ms", tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.bvh_karras_ms + tree_pos.reduce_ms);
        eprintln!("    force+ (shmem)      : {:>6} ms  <<<", t_force_pos_ms - (tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.bvh_karras_ms + tree_pos.reduce_ms + tree_pos.reorder_ms + tree_pos.reset_ms + tree_pos.init_leaves_ms));
        eprintln!("  --- PASS - ---");
        eprintln!("    tree build          : {:>6} ms", tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.bvh_karras_ms + tree_neg.reduce_ms);
        eprintln!("    force- (shmem)      : {:>6} ms  <<<", t_force_neg_ms - (tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.bvh_karras_ms + tree_neg.reduce_ms + tree_neg.reorder_ms + tree_neg.reset_ms + tree_neg.init_leaves_ms));
        eprintln!("  kick                  : {:>6} ms", t_kick_ms);
        eprintln!("  drift (2nd)           : {:>6} ms", t_drift2_ms);
        eprintln!("  ─────────────────────────────────");
        eprintln!("  TOTAL                 : {:>6} ms", total);
        eprintln!();

        Ok(())
    }

    /// [OPT-4] Warp-coherent traversal WITHOUT Morton reorder
    pub fn step_dkd_warpcoherent(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        self.step_count += 1;
        let step_num = self.step_count;

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Warp-coherent config: 8 warps × 64 ints × 4 bytes = 2048 bytes
        let warp_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 4096,  // 8 warps × 128 ints × 4 bytes
        };

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        // Reset acceleration
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }
        self.device.synchronize()?;
        let t_drift_ms = t0.elapsed().as_millis();

        // ===== PASS 1: Positive particles tree =====
        let t0 = Instant::now();
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;
        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;

        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_pos = self.build_single_sign_tree(1, self.n_positive)?;

        // Force + (overwrite: tree_sign=+1) with warp-coherent
        let forces_wc_k = self.device.get_func("twopass", "forces_twopass_warpcoherent")
            .ok_or("forces_twopass_warpcoherent not found")?;
        unsafe {
            forces_wc_k.clone().launch(warp_cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_pos_ms = t0.elapsed().as_millis();

        // ===== PASS 2: Negative particles tree =====
        let t0 = Instant::now();
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_neg = self.build_single_sign_tree(-1, self.n_negative)?;

        // Force - (accumulate: tree_sign=-1) with warp-coherent
        unsafe {
            forces_wc_k.launch(warp_cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_neg_ms = t0.elapsed().as_millis();

        // ===== KICK (dt) =====
        let t0 = Instant::now();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_kick_ms = t0.elapsed().as_millis();

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t_drift2_ms = t0.elapsed().as_millis();

        self.time += dt;

        // Print timing
        let total = t_drift_ms + t_force_pos_ms + t_force_neg_ms + t_kick_ms + t_drift2_ms;
        eprintln!("[step {:03}] WARP-COHERENT TIMING (ms):", step_num);
        eprintln!("  drift (1st)           : {:>6} ms", t_drift_ms);
        eprintln!("  --- PASS + ---");
        eprintln!("    tree build          : {:>6} ms", tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.bvh_karras_ms + tree_pos.reduce_ms);
        eprintln!("    force+ (warp-coh)   : {:>6} ms  <<<", t_force_pos_ms - (tree_pos.morton_ms + tree_pos.sort_ms + tree_pos.bvh_karras_ms + tree_pos.reduce_ms + tree_pos.reorder_ms + tree_pos.reset_ms + tree_pos.init_leaves_ms));
        eprintln!("  --- PASS - ---");
        eprintln!("    tree build          : {:>6} ms", tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.bvh_karras_ms + tree_neg.reduce_ms);
        eprintln!("    force- (warp-coh)   : {:>6} ms  <<<", t_force_neg_ms - (tree_neg.morton_ms + tree_neg.sort_ms + tree_neg.bvh_karras_ms + tree_neg.reduce_ms + tree_neg.reorder_ms + tree_neg.reset_ms + tree_neg.init_leaves_ms));
        eprintln!("  kick                  : {:>6} ms", t_kick_ms);
        eprintln!("  drift (2nd)           : {:>6} ms", t_drift2_ms);
        eprintln!("  ─────────────────────────────────");
        eprintln!("  TOTAL                 : {:>6} ms", total);
        eprintln!();

        Ok(())
    }

    /// [OPT-4] Warp-coherent traversal WITH Morton reorder
    pub fn step_dkd_morton_warpcoherent(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        self.step_count += 1;
        let step_num = self.step_count;

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let inv_cell_size = (2097152.0 / self.box_size) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Warp-coherent config: 8 warps × 64 ints × 4 bytes = 2048 bytes
        let warp_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 4096,  // 8 warps × 128 ints × 4 bytes
        };

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        // Reset acceleration
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }
        self.device.synchronize()?;
        let t_drift_ms = t0.elapsed().as_millis();

        // ===== MORTON REORDER ALL PARTICLES =====
        let t0 = Instant::now();

        let morton_k = self.device.get_func("twopass", "compute_morton_all")
            .ok_or("compute_morton_all not found")?;
        unsafe {
            morton_k.launch(cfg, (
                &self.pos, &mut self.morton_all, &mut self.sorted_all,
                n as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Radix sort Morton codes
        let hist_k = self.device.get_func("twopass", "radix_histogram")
            .ok_or("radix_histogram not found")?;
        let prefix_k = self.device.get_func("twopass", "radix_prefix_sum")
            .ok_or("radix_prefix_sum not found")?;
        let scatter_k = self.device.get_func("twopass", "radix_scatter")
            .ok_or("radix_scatter not found")?;

        let n_blocks_all = blocks;
        let hist_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (n_blocks_all as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        for pass in 0..8 {
            let bit_shift = pass * 8;
            let (keys_in, keys_out, vals_in, vals_out) = if pass % 2 == 0 {
                (&self.morton_all, &mut self.morton_all_tmp,
                 &self.sorted_all, &mut self.sorted_all_tmp)
            } else {
                (&self.morton_all_tmp, &mut self.morton_all,
                 &self.sorted_all_tmp, &mut self.sorted_all)
            };

            unsafe {
                hist_k.clone().launch(hist_cfg, (
                    keys_in, &mut self.radix_hist_all, n as i32, bit_shift as i32
                ))?;
                prefix_k.clone().launch(prefix_cfg, (
                    &mut self.radix_hist_all, &mut self.radix_global, n_blocks_all as i32
                ))?;
                scatter_k.clone().launch(hist_cfg, (
                    keys_in, keys_out, vals_in, vals_out,
                    &self.radix_hist_all, &self.radix_global,
                    n as i32, bit_shift as i32
                ))?;
            }
        }
        self.device.synchronize()?;

        // Reorder positions, velocities, and signs
        // CRITICAL: Must reorder vel to keep pos[i]/vel[i] correspondence
        let reorder_pos_k = self.device.get_func("twopass", "reorder_by_idx_f32x3")
            .ok_or("reorder_by_idx_f32x3 not found")?;
        let reorder_sign_k = self.device.get_func("twopass", "reorder_by_idx_i8")
            .ok_or("reorder_by_idx_i8 not found")?;

        unsafe {
            reorder_pos_k.clone().launch(cfg, (
                &self.pos, &mut self.pos_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_pos_k.clone().launch(cfg, (
                &self.vel, &mut self.vel_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_sign_k.launch(cfg, (
                &self.signs, &mut self.signs_sorted, &self.sorted_all, n as i32
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_sorted);
        std::mem::swap(&mut self.vel, &mut self.vel_sorted);
        std::mem::swap(&mut self.signs, &mut self.signs_sorted);
        self.device.synchronize()?;
        let t_reorder_ms = t0.elapsed().as_millis();

        // ===== PASS 1: Positive particles tree =====
        let t0 = Instant::now();
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;
        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;

        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_pos = self.build_single_sign_tree(1, self.n_positive)?;

        // Force + (overwrite: tree_sign=+1) with warp-coherent
        let forces_wc_k = self.device.get_func("twopass", "forces_twopass_warpcoherent")
            .ok_or("forces_twopass_warpcoherent not found")?;
        unsafe {
            forces_wc_k.clone().launch(warp_cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_pos_ms = t0.elapsed().as_millis();

        // ===== PASS 2: Negative particles tree =====
        let t0 = Instant::now();
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let tree_neg = self.build_single_sign_tree(-1, self.n_negative)?;

        // Force - (accumulate: tree_sign=-1) with warp-coherent
        unsafe {
            forces_wc_k.launch(warp_cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_force_neg_ms = t0.elapsed().as_millis();

        // ===== KICK (dt) =====
        let t0 = Instant::now();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;
        let t_kick_ms = t0.elapsed().as_millis();

        // ===== DRIFT (dt/2) =====
        let t0 = Instant::now();
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t_drift2_ms = t0.elapsed().as_millis();

        self.time += dt;

        // Timing debug disabled for cleaner output
        let _ = (t_drift_ms, t_reorder_ms, t_force_pos_ms, t_force_neg_ms, t_kick_ms, t_drift2_ms);
        let _ = (tree_pos, tree_neg);

        Ok(())
    }

    /// Helper: GPU radix sort Morton codes (8 passes × 8 bits)
    fn radix_sort_all(&mut self, n: usize, blocks: usize) -> Result<(), Box<dyn std::error::Error>> {
        let hist_k = self.device.get_func("twopass", "radix_histogram")
            .ok_or("radix_histogram not found")?;
        let prefix_k = self.device.get_func("twopass", "radix_prefix_sum")
            .ok_or("radix_prefix_sum not found")?;
        let scatter_k = self.device.get_func("twopass", "radix_scatter")
            .ok_or("radix_scatter not found")?;

        let hist_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        for pass in 0..8 {
            let bit_shift = pass * 8;
            let (keys_in, keys_out, vals_in, vals_out) = if pass % 2 == 0 {
                (&self.morton_all, &mut self.morton_all_tmp,
                 &self.sorted_all, &mut self.sorted_all_tmp)
            } else {
                (&self.morton_all_tmp, &mut self.morton_all,
                 &self.sorted_all_tmp, &mut self.sorted_all)
            };

            unsafe {
                hist_k.clone().launch(hist_cfg, (
                    keys_in, &mut self.radix_hist_all, n as i32, bit_shift as i32
                ))?;
                prefix_k.clone().launch(prefix_cfg, (
                    &mut self.radix_hist_all, &mut self.radix_global, blocks as i32
                ))?;
                scatter_k.clone().launch(hist_cfg, (
                    keys_in, keys_out, vals_in, vals_out,
                    &self.radix_hist_all, &self.radix_global,
                    n as i32, bit_shift as i32
                ))?;
            }
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// Helper: Reorder positions and signs by Morton order
    fn reorder_all_by_morton(&mut self, n: usize, blocks: usize) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let reorder_pos_k = self.device.get_func("twopass", "reorder_by_idx_f32x3")
            .ok_or("reorder_by_idx_f32x3 not found")?;
        let reorder_sign_k = self.device.get_func("twopass", "reorder_by_idx_i8")
            .ok_or("reorder_by_idx_i8 not found")?;

        unsafe {
            reorder_pos_k.clone().launch(cfg, (
                &self.pos, &mut self.pos_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_sign_k.launch(cfg, (
                &self.signs, &mut self.signs_sorted, &self.sorted_all, n as i32
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_sorted);
        std::mem::swap(&mut self.signs, &mut self.signs_sorted);
        self.device.synchronize()?;
        Ok(())
    }

    /// TreePM step: GPU BH for short-range only (r < r_cut)
    /// Long-range forces from PM must be added separately
    ///
    /// This eliminates grid artifacts by construction:
    /// - Long-range (r > r_cut): handled by PM cuFFT (accurate)
    /// - Short-range (r < r_cut): GPU BH (accurate at short range)
    pub fn compute_short_range_forces(&mut self, r_cut: f64) -> Result<(), Box<dyn std::error::Error>> {
        let box_half = (self.box_size / 2.0) as f32;
        let inv_cell_size = (2097152.0 / self.box_size) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let warp_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 4096,
        };

        // Reset acceleration
        let reset_f32_k = self.device.get_func("twopass", "reset_f32")
            .ok_or("reset_f32 not found")?;
        let acc_blocks = (n * 3 + 255) / 256;
        let acc_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (acc_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            reset_f32_k.launch(acc_cfg, (&mut self.acc, (n * 3) as i32))?;
        }
        self.device.synchronize()?;

        // NOTE: Morton reordering removed to avoid position/velocity mismatch
        // The reordering was corrupting the simulation by applying wrong velocities
        // to wrong particles after the kick step.

        // Use warp-coherent kernel (all threads in warp traverse together)
        let forces_wc_k = self.device.get_func("twopass", "forces_twopass_warpcoherent")
            .ok_or("forces_twopass_warpcoherent not found")?;

        let reset_i32_k = self.device.get_func("twopass", "reset_i32")
            .ok_or("reset_i32 not found")?;
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;

        // PASS 1: Positive particles tree
        unsafe {
            reset_i32_k.clone().launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.clone().launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, 1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let _ = self.build_single_sign_tree(1, self.n_positive)?;

        // Force + (overwrite: tree_sign=+1) with warp-coherent
        unsafe {
            forces_wc_k.clone().launch(warp_cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, 1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;

        // PASS 2: Negative particles tree
        unsafe {
            reset_i32_k.launch(LaunchConfig {
                block_dim: (1, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0,
            }, (&mut self.extract_count, 1))?;
            extract_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.pos_sign, &mut self.idx_map,
                n as i32, -1i32, &mut self.extract_count,
            ))?;
        }
        self.device.synchronize()?;
        let _ = self.build_single_sign_tree(-1, self.n_negative)?;

        // Force - (accumulate: tree_sign=-1) with warp-coherent
        unsafe {
            forces_wc_k.launch(warp_cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left, &self.bvh_right, &self.bvh_node_types,
                &mut self.acc,
                n as i32, -1i32, self.theta as f32, self.softening as f32,
            ))?;
        }
        self.device.synchronize()?;

        Ok(())
    }

    /// Get acceleration buffer for adding PM long-range forces
    pub fn get_acc_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.acc
    }

    /// Get positions for PM mass assignment
    pub fn get_pos(&self) -> &CudaSlice<f32> {
        &self.pos
    }

    /// Get signs for PM mass assignment
    pub fn get_signs_slice(&self) -> &CudaSlice<i8> {
        &self.signs
    }

    /// Hybrid TreePM step: CPU PM long-range + GPU BH short-range
    ///
    /// This eliminates grid artifacts by construction:
    /// - Long-range (r > r_cut): PM with Gaussian splitting (CPU rustfft, accurate)
    /// - Short-range (r < r_cut): GPU BH with r_cut cutoff (fast)
    ///
    /// Architecture:
    /// 1. Download positions → CPU
    /// 2. CPU PM: CIC mass assign, FFT solve with splitting, force interpolation
    /// 3. Upload PM forces → GPU
    /// 4. GPU: add PM forces to acc buffer
    /// 5. GPU BH: compute short-range forces with r_cut
    /// 6. GPU: kick + drift
    pub fn step_treepm_hybrid(
        &mut self,
        dt: f64,
        pm_grid: &mut crate::treepm::pm_grid::PmGrid,
        r_cut: f64,
        hubble: f64,
        dtau_per_dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;
        let n = self.n_particles;
        let box_half = (self.box_size / 2.0) as f32;
        let half_dt = (dt / 2.0) as f32;
        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // ===== DRIFT (dt/2) =====
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== PM LONG-RANGE (CPU) =====
        let t_pm = Instant::now();

        // Download positions and signs
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        let signs_cpu = self.device.dtoh_sync_copy(&self.signs)?;

        // CIC mass assignment to PM grid
        pm_grid.clear();
        for i in 0..n {
            let x = pos_cpu[i * 3] as f64;
            let y = pos_cpu[i * 3 + 1] as f64;
            let z = pos_cpu[i * 3 + 2] as f64;
            pm_grid.assign_mass(x, y, z, 1.0, signs_cpu[i]);
        }

        // FFT Poisson solve with STRONG k-space damping
        // r_s = r_cut → PM only affects r >> r_cut (very long range)
        // BH handles r < r_cut with full force (no erfc)
        let r_s = r_cut;  // Strong damping
        let g_constant = 1.0;
        pm_grid.solve_poisson_with_splitting(g_constant, Some(r_s));

        // Interpolate PM forces for all particles
        let mut pm_forces_cpu = vec![0.0f32; n * 3];
        for i in 0..n {
            let x = pos_cpu[i * 3] as f64;
            let y = pos_cpu[i * 3 + 1] as f64;
            let z = pos_cpu[i * 3 + 2] as f64;
            let (fx, fy, fz) = pm_grid.interpolate_force(x, y, z, signs_cpu[i]);
            pm_forces_cpu[i * 3] = fx as f32;
            pm_forces_cpu[i * 3 + 1] = fy as f32;
            pm_forces_cpu[i * 3 + 2] = fz as f32;
        }

        // Upload PM forces to GPU
        self.device.htod_sync_copy_into(&pm_forces_cpu, &mut self.pm_forces)?;

        let pm_ms = t_pm.elapsed().as_millis();

        // ===== GPU BH SHORT-RANGE (r < r_cut) =====
        let t_bh = Instant::now();
        self.compute_short_range_forces(r_cut)?;
        let bh_ms = t_bh.elapsed().as_millis();

        // ===== ADD PM FORCES TO ACC =====
        let add_pm_k = self.device.get_func("twopass", "add_pm_forces")
            .ok_or("add_pm_forces not found")?;
        unsafe {
            add_pm_k.launch(cfg, (
                &mut self.acc, &self.pm_forces, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== KICK (dt) =====
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== DRIFT (dt/2) =====
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        self.time += dt;
        self.step_count += 1;

        // Timing info (optional, disable for cleaner output)
        if self.step_count <= 5 || self.step_count % 100 == 0 {
            eprintln!("  TreePM hybrid step {}: PM {}ms + BH {}ms = {}ms",
                      self.step_count, pm_ms, bh_ms, pm_ms + bh_ms);
        }

        Ok(())
    }

    /// Full GPU TreePM step: GPU CIC + cuFFT + GPU BH
    ///
    /// Fastest path - everything on GPU:
    /// 1. GPU CIC scatter: particles → rho_plus, rho_minus grids
    /// 2. cuFFT Poisson: rho → phi (device-to-device, no host transfer)
    /// 3. GPU CIC gather: phi grids → pm_forces
    /// 4. GPU BH short-range: forces with r_cut cutoff
    /// 5. Add PM forces to acc, kick, drift
    ///
    /// Requires: cufft feature + libcufft_wrapper.so built
    #[cfg(feature = "cufft")]
    pub fn step_treepm_gpu(
        &mut self,
        dt: f64,
        r_cut: f64,
        hubble: f64,
        dtau_per_dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;
        let n = self.n_particles;
        let grid = self.pm_grid_size;
        let grid_cells = grid * grid * grid;
        let box_half = (self.box_size / 2.0) as f32;
        let half_dt = (dt / 2.0) as f32;
        let cell_size = (self.box_size / grid as f64) as f32;
        let inv_cell_size = (grid as f64 / self.box_size) as f32;
        // No k-space damping: BH handles r < r_cut, PM handles r > r_cut
        let g_constant = 1.0;

        let blocks = (n + 255) / 256;
        let grid_blocks = (grid_cells + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let grid_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (grid_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // ===== DRIFT (dt/2) =====
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== GPU CIC SCATTER =====
        let t_pm = Instant::now();

        // Reset density grids to zero
        let reset_grid_k = self.device.get_func("twopass", "reset_f64_grid")
            .ok_or("reset_f64_grid not found")?;
        unsafe {
            reset_grid_k.clone().launch(grid_cfg, (&mut self.rho_plus, grid_cells as i32))?;
            reset_grid_k.launch(grid_cfg, (&mut self.rho_minus, grid_cells as i32))?;
        }
        self.device.synchronize()?;

        // CIC scatter: particles → rho grids
        let scatter_k = self.device.get_func("twopass", "cic_scatter")
            .ok_or("cic_scatter not found")?;
        unsafe {
            scatter_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.rho_plus, &mut self.rho_minus,
                n as i32, grid as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // ===== cuFFT POISSON SOLVE (device-to-device) =====
        // Get raw device pointers for cuFFT
        let rho_plus_ptr = get_device_ptr(&self.rho_plus) as *mut f64;
        let rho_minus_ptr = get_device_ptr(&self.rho_minus) as *mut f64;
        let phi_plus_ptr = get_device_ptr(&self.phi_plus) as *mut f64;
        let phi_minus_ptr = get_device_ptr(&self.phi_minus) as *mut f64;

        // TreePM Gaussian splitting: r_s = r_cut/3
        // PM k-space damping: exp(-k²×r_s²) → affects scales > r_s
        // BH real-space: should use erfc(r/(2r_s)) → affects scales < r_s
        let r_s = r_cut / 3.0;
        unsafe {
            // Solve rho_plus → phi_plus
            crate::treepm::cufft_ffi::solve_device(
                rho_plus_ptr, phi_plus_ptr,
                grid, self.box_size, g_constant, r_s,
            )?;

            // Solve rho_minus → phi_minus
            crate::treepm::cufft_ffi::solve_device(
                rho_minus_ptr, phi_minus_ptr,
                grid, self.box_size, g_constant, r_s,
            )?;
        }

        // ===== GPU CIC GATHER =====
        let gather_k = self.device.get_func("twopass", "cic_gather")
            .ok_or("cic_gather not found")?;
        unsafe {
            gather_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.phi_plus, &self.phi_minus,
                &mut self.pm_forces,
                n as i32, grid as i32, box_half, inv_cell_size, cell_size,
            ))?;
        }
        self.device.synchronize()?;
        let pm_ms = t_pm.elapsed().as_millis();

        // ===== GPU BH SHORT-RANGE (r < r_cut) =====
        let t_bh = Instant::now();
        self.compute_short_range_forces(r_cut)?;
        let bh_ms = t_bh.elapsed().as_millis();

        // Add PM long-range forces (r > r_cut)
        let add_pm_k = self.device.get_func("twopass", "add_pm_forces")
            .ok_or("add_pm_forces not found")?;
        unsafe {
            add_pm_k.launch(cfg, (
                &mut self.acc, &self.pm_forces, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== KICK (dt) =====
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== DRIFT (dt/2) =====
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        self.time += dt;
        self.step_count += 1;

        // Timing info
        if self.step_count <= 5 || self.step_count % 100 == 0 {
            eprintln!("  TreePM GPU step {}: PM {}ms + BH {}ms = {}ms",
                      self.step_count, pm_ms, bh_ms, pm_ms + bh_ms);
        }

        Ok(())
    }

    /// TreePM step with Morton ordering for improved cache coherence
    /// Same as step_treepm_gpu but with Morton reordering for ~3-7x BH speedup
    #[cfg(feature = "cufft")]
    pub fn step_treepm_gpu_morton(
        &mut self,
        dt: f64,
        r_cut: f64,
        hubble: f64,
        dtau_per_dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;
        let n = self.n_particles;
        let grid = self.pm_grid_size;
        let grid_cells = grid * grid * grid;
        let box_half = (self.box_size / 2.0) as f32;
        let half_dt = (dt / 2.0) as f32;
        let cell_size = (self.box_size / grid as f64) as f32;
        let inv_cell_size = (grid as f64 / self.box_size) as f32;
        let g_constant = 1.0;

        let blocks = (n + 255) / 256;
        let n_blocks_all = blocks;
        let grid_blocks = (grid_cells + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let grid_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (grid_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let hist_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (n_blocks_all as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // ===== DRIFT (dt/2) =====
        let drift_k = self.device.get_func("twopass", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_k.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== MORTON REORDER =====
        let t_morton = Instant::now();

        // Compute Morton codes for all particles
        let morton_k = self.device.get_func("twopass", "compute_morton_all")
            .ok_or("compute_morton_all not found")?;
        unsafe {
            morton_k.launch(cfg, (
                &self.pos, &mut self.morton_all, &mut self.sorted_all,
                n as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Radix sort Morton codes (8 passes)
        let hist_k = self.device.get_func("twopass", "radix_histogram")
            .ok_or("radix_histogram not found")?;
        let prefix_k = self.device.get_func("twopass", "radix_prefix_sum")
            .ok_or("radix_prefix_sum not found")?;
        let scatter_k = self.device.get_func("twopass", "radix_scatter")
            .ok_or("radix_scatter not found")?;

        for pass in 0..8 {
            let bit_shift = pass * 8;
            let (keys_in, keys_out, vals_in, vals_out) = if pass % 2 == 0 {
                (&self.morton_all, &mut self.morton_all_tmp,
                 &self.sorted_all, &mut self.sorted_all_tmp)
            } else {
                (&self.morton_all_tmp, &mut self.morton_all,
                 &self.sorted_all_tmp, &mut self.sorted_all)
            };

            unsafe {
                hist_k.clone().launch(hist_cfg, (
                    keys_in, &mut self.radix_hist_all, n as i32, bit_shift as i32
                ))?;
                prefix_k.clone().launch(prefix_cfg, (
                    &mut self.radix_hist_all, &mut self.radix_global, n_blocks_all as i32
                ))?;
                scatter_k.clone().launch(hist_cfg, (
                    keys_in, keys_out, vals_in, vals_out,
                    &self.radix_hist_all, &self.radix_global,
                    n as i32, bit_shift as i32
                ))?;
            }
        }
        self.device.synchronize()?;

        // Reorder positions, velocities, and signs by Morton order
        let reorder_pos_k = self.device.get_func("twopass", "reorder_by_idx_f32x3")
            .ok_or("reorder_by_idx_f32x3 not found")?;
        let reorder_sign_k = self.device.get_func("twopass", "reorder_by_idx_i8")
            .ok_or("reorder_by_idx_i8 not found")?;

        unsafe {
            reorder_pos_k.clone().launch(cfg, (
                &self.pos, &mut self.pos_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_pos_k.clone().launch(cfg, (
                &self.vel, &mut self.vel_sorted, &self.sorted_all, n as i32
            ))?;
            reorder_sign_k.launch(cfg, (
                &self.signs, &mut self.signs_sorted, &self.sorted_all, n as i32
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_sorted);
        std::mem::swap(&mut self.vel, &mut self.vel_sorted);
        std::mem::swap(&mut self.signs, &mut self.signs_sorted);
        self.device.synchronize()?;
        let morton_ms = t_morton.elapsed().as_millis();

        // ===== GPU CIC SCATTER (uses Morton-ordered positions) =====
        let t_pm = Instant::now();

        // Reset density grids to zero
        let reset_grid_k = self.device.get_func("twopass", "reset_f64_grid")
            .ok_or("reset_f64_grid not found")?;
        unsafe {
            reset_grid_k.clone().launch(grid_cfg, (&mut self.rho_plus, grid_cells as i32))?;
            reset_grid_k.launch(grid_cfg, (&mut self.rho_minus, grid_cells as i32))?;
        }
        self.device.synchronize()?;

        // CIC scatter: particles → rho grids
        let cic_scatter_k = self.device.get_func("twopass", "cic_scatter")
            .ok_or("cic_scatter not found")?;
        unsafe {
            cic_scatter_k.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.rho_plus, &mut self.rho_minus,
                n as i32, grid as i32, box_half, inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // ===== cuFFT POISSON SOLVE =====
        let rho_plus_ptr = get_device_ptr(&self.rho_plus) as *mut f64;
        let rho_minus_ptr = get_device_ptr(&self.rho_minus) as *mut f64;
        let phi_plus_ptr = get_device_ptr(&self.phi_plus) as *mut f64;
        let phi_minus_ptr = get_device_ptr(&self.phi_minus) as *mut f64;

        let r_s = r_cut / 3.0;
        unsafe {
            crate::treepm::cufft_ffi::solve_device(
                rho_plus_ptr, phi_plus_ptr,
                grid, self.box_size, g_constant, r_s,
            )?;
            crate::treepm::cufft_ffi::solve_device(
                rho_minus_ptr, phi_minus_ptr,
                grid, self.box_size, g_constant, r_s,
            )?;
        }

        // ===== GPU CIC GATHER =====
        let gather_k = self.device.get_func("twopass", "cic_gather")
            .ok_or("cic_gather not found")?;
        unsafe {
            gather_k.launch(cfg, (
                &self.pos, &self.signs,
                &self.phi_plus, &self.phi_minus,
                &mut self.pm_forces,
                n as i32, grid as i32, box_half, inv_cell_size, cell_size,
            ))?;
        }
        self.device.synchronize()?;
        let pm_ms = t_pm.elapsed().as_millis();

        // ===== GPU BH SHORT-RANGE (Morton-ordered for cache coherence) =====
        let t_bh = Instant::now();
        self.compute_short_range_forces(r_cut)?;
        let bh_ms = t_bh.elapsed().as_millis();

        // Add PM long-range forces
        let add_pm_k = self.device.get_func("twopass", "add_pm_forces")
            .ok_or("add_pm_forces not found")?;
        unsafe {
            add_pm_k.launch(cfg, (
                &mut self.acc, &self.pm_forces, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== KICK (Morton-ordered vel and acc) =====
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,
            ))?;
        }
        self.device.synchronize()?;

        // ===== DRIFT (dt/2) =====
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        self.time += dt;
        self.step_count += 1;

        // Timing info
        if self.step_count <= 5 || self.step_count % 100 == 0 {
            eprintln!("  TreePM Morton step {}: Morton {}ms + PM {}ms + BH {}ms = {}ms",
                      self.step_count, morton_ms, pm_ms, bh_ms, morton_ms + pm_ms + bh_ms);
        }

        Ok(())
    }

    pub fn kinetic_energy(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let vel_cpu = self.device.dtoh_sync_copy(&self.vel)?;
        let ke: f64 = vel_cpu.chunks(3)
            .map(|v| 0.5 * ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)))
            .sum();
        Ok(ke)
    }

    /// Sum of |acceleration| for diagnostics
    pub fn acceleration_sum(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let acc_cpu = self.device.dtoh_sync_copy(&self.acc)?;
        let sum: f64 = acc_cpu.chunks(3)
            .map(|a| ((a[0] as f64).powi(2) + (a[1] as f64).powi(2) + (a[2] as f64).powi(2)).sqrt())
            .sum();
        Ok(sum)
    }

    pub fn segregation(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        let signs_cpu = self.device.dtoh_sync_copy(&self.signs)?;

        let mut com_plus = [0.0f64; 3];
        let mut com_minus = [0.0f64; 3];
        let mut n_plus = 0usize;
        let mut n_minus = 0usize;

        for i in 0..self.n_particles {
            let x = pos_cpu[i * 3] as f64;
            let y = pos_cpu[i * 3 + 1] as f64;
            let z = pos_cpu[i * 3 + 2] as f64;
            if signs_cpu[i] > 0 {
                com_plus[0] += x; com_plus[1] += y; com_plus[2] += z;
                n_plus += 1;
            } else {
                com_minus[0] += x; com_minus[1] += y; com_minus[2] += z;
                n_minus += 1;
            }
        }

        if n_plus > 0 { for c in &mut com_plus { *c /= n_plus as f64; } }
        if n_minus > 0 { for c in &mut com_minus { *c /= n_minus as f64; } }

        let dx = com_plus[0] - com_minus[0];
        let dy = com_plus[1] - com_minus[1];
        let dz = com_plus[2] - com_minus[2];
        Ok((dx*dx + dy*dy + dz*dz).sqrt() / self.box_size)
    }

    pub fn n_particles(&self) -> usize { self.n_particles }
    pub fn box_size(&self) -> f64 { self.box_size }
    pub fn time(&self) -> f64 { self.time }

    /// Download positions from GPU (85M × 3 × f32)
    pub fn get_positions(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.pos)?)
    }

    /// Download velocities from GPU (85M × 3 × f32)
    pub fn get_velocities(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.vel)?)
    }

    /// Download signs from GPU (85M × i8)
    pub fn get_signs(&self) -> Result<Vec<i8>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.signs)?)
    }

    /// Download all particle data from GPU (positions, velocities, signs)
    pub fn get_particles(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<i8>), Box<dyn std::error::Error>> {
        let pos = self.device.dtoh_sync_copy(&self.pos)?;
        let vel = self.device.dtoh_sync_copy(&self.vel)?;
        let signs = self.device.dtoh_sync_copy(&self.signs)?;
        Ok((pos, vel, signs))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct GpuNBodyTwoPass;

#[cfg(not(feature = "cuda"))]
impl GpuNBodyTwoPass {
    pub fn new(_: usize, _: usize, _: f64) -> Result<Self, Box<dyn std::error::Error>> {
        Err("CUDA not enabled".into())
    }
}
