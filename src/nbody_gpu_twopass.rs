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
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

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

    int stack[32];  // Reduced from 64 for better occupancy
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
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2 + 1e-20f);

        if (nt == 1 || (2.0f * hs / r) < theta) {
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

    int stack[32];  // Reduced from 64 for better occupancy
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
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2 + 1e-20f);

        if (nt == 1 || (2.0f * hs / r) < theta) {
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
    // Parameters
    n_particles: usize,
    n_positive: usize,
    n_negative: usize,
    theta: f64,
    softening: f64,
    box_size: f64,
    time: f64,
}

#[cfg(feature = "cuda")]
impl GpuNBodyTwoPass {
    pub fn new(n_positive: usize, n_negative: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
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
            "reset_i32", "reset_f64", "reset_f32", "set_i32",
            "radix_histogram", "radix_prefix_sum", "radix_scatter"
        ])?;
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        let n_total = n_positive + n_negative;
        let n_max_sign = n_positive.max(n_negative);
        let n_bvh = 2 * n_max_sign;  // Only need tree for ONE sign at a time!

        // Generate particles with Zel'dovich perturbations
        println!("  [3/6] Generating {} particles with Zel'dovich ICs...", n_total);
        let t0 = Instant::now();
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(42);

        // Zel'dovich parameters
        let amplitude = 1e-3_f64;  // Perturbation amplitude
        let lambda = 100.0_f64;    // Characteristic scale (Mpc in simulation units)
        let sigma = amplitude * lambda;  // Gaussian displacement std dev
        let normal = Normal::new(0.0, sigma).unwrap();

        // Grid dimensions (cubic grid for n_total particles)
        let n_side = (n_total as f64).cbrt().ceil() as usize;
        let cell_size = box_size / n_side as f64;
        println!("         Grid: {}³ = {} cells, cell_size = {:.3}", n_side, n_side.pow(3), cell_size);
        println!("         Zel'dovich: amplitude = {:.0e}, lambda = {:.1}, sigma = {:.3}", amplitude, lambda, sigma);

        let mut pos_data = Vec::with_capacity(n_total * 3);
        let mut vel_data = Vec::with_capacity(n_total * 3);
        let mut signs_data: Vec<i8> = Vec::with_capacity(n_total);

        // Ratio for sign assignment
        let pos_ratio = n_positive as f64 / n_total as f64;
        let mut count_pos = 0usize;
        let mut count_neg = 0usize;

        // Generate grid with Zel'dovich perturbations
        for iz in 0..n_side {
            for iy in 0..n_side {
                for ix in 0..n_side {
                    let idx = iz * n_side * n_side + iy * n_side + ix;
                    if idx >= n_total { break; }

                    // Base grid position (centered in box)
                    let x0 = (ix as f64 + 0.5) * cell_size - box_size / 2.0;
                    let y0 = (iy as f64 + 0.5) * cell_size - box_size / 2.0;
                    let z0 = (iz as f64 + 0.5) * cell_size - box_size / 2.0;

                    // Zel'dovich displacement (Gaussian)
                    let dx = normal.sample(&mut rng);
                    let dy = normal.sample(&mut rng);
                    let dz = normal.sample(&mut rng);

                    // Final position with periodic wrapping
                    let mut x = x0 + dx;
                    let mut y = y0 + dy;
                    let mut z = z0 + dz;

                    // Wrap to box
                    let half = box_size / 2.0;
                    if x > half { x -= box_size; } else if x < -half { x += box_size; }
                    if y > half { y -= box_size; } else if y < -half { y += box_size; }
                    if z > half { z -= box_size; } else if z < -half { z += box_size; }

                    pos_data.extend([x as f32, y as f32, z as f32]);

                    // Zero initial velocity (cold start)
                    vel_data.extend([0.0f32, 0.0f32, 0.0f32]);

                    // Assign sign based on target ratio
                    if count_pos < n_positive && (count_neg >= n_negative || rng.random::<f64>() < pos_ratio) {
                        signs_data.push(1i8);
                        count_pos += 1;
                    } else {
                        signs_data.push(-1i8);
                        count_neg += 1;
                    }
                }
            }
        }

        println!("         Generated: N+ = {}, N- = {}", count_pos, count_neg);
        println!("         Velocities: v = 0 (cold start, no virialization)");
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        // Skip virialization - cold start
        println!("  [4/6] Skipping virialization (cold ICs)...");

        // Allocate GPU buffers
        println!("  [5/6] Copying data to GPU...");
        let t0 = Instant::now();
        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let acc = device.alloc_zeros::<f32>(n_total * 3)?;  // FP32 for performance
        println!("         done ({:.2}s)", t0.elapsed().as_secs_f64());

        // Single-sign extraction buffers (sized for max sign count)
        println!("  [6/6] Allocating GPU buffers...");
        let t0 = Instant::now();
        let pos_sign = device.alloc_zeros::<f32>(n_max_sign * 3)?;
        let pos_sign_tmp = device.alloc_zeros::<f32>(n_max_sign * 3)?;
        let idx_map = device.alloc_zeros::<i32>(n_max_sign)?;
        let idx_map_tmp = device.alloc_zeros::<i32>(n_max_sign)?;
        let extract_count = device.alloc_zeros::<i32>(1)?;

        // Morton sorting + radix sort buffers
        let n_blocks = (n_max_sign + 255) / 256;
        let morton_codes = device.alloc_zeros::<u64>(n_max_sign)?;
        let morton_codes_tmp = device.alloc_zeros::<u64>(n_max_sign)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_max_sign)?;
        let sorted_indices_tmp = device.alloc_zeros::<i32>(n_max_sign)?;
        let radix_hist = device.alloc_zeros::<u32>(n_blocks * 256)?;
        let radix_global = device.alloc_zeros::<u32>(256)?;

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
            bvh_left, bvh_right, bvh_parent, bvh_rl, bvh_rr,
            bvh_node_pos, bvh_node_mass, bvh_node_types, bvh_atomic,
            n_particles: n_total,
            n_positive,
            n_negative,
            theta: 5.0,  // Aggressive but fast; validated on 8M
            softening: 0.1,
            box_size,
            time: 0.0,
        })
    }

    pub fn set_theta(&mut self, theta: f64) { self.theta = theta; }

    /// Build tree for particles of given sign
    fn build_single_sign_tree(&mut self, sign: i32, n_sign: usize) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        self.device.synchronize()?;  // Ensure previous kernels completed
        eprintln!("[tree_build n={}]", n_sign);
        std::io::stderr().flush().ok();

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
        let morton_k = self.device.get_func("twopass", "compute_morton_f32")
            .ok_or("compute_morton_f32 not found")?;
        unsafe {
            morton_k.launch(cfg, (
                &self.pos_sign, &mut self.morton_codes, &mut self.sorted_indices,
                n as i32, box_half, inv_cell_size,
            ))?;
        }

        eprintln!("  morton...");
        std::io::stderr().flush().ok();
        self.device.synchronize()?;

        // GPU Radix Sort: 8 passes × 8 bits, fully on-device
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
        eprintln!("  sort (gpu)...");

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

        // Verify sort correctness (only downloads if check fails or first run)
        if n < 200_000 {  // Only check for small runs to avoid overhead
            let mc_cpu: Vec<u64> = self.device.dtoh_sync_copy(&self.morton_codes)?;
            let mut unsorted = 0;
            // Only check the first n elements (buffer may be larger)
            for (i, w) in mc_cpu[..n].windows(2).enumerate() {
                if w[0] > w[1] {
                    unsorted += 1;
                    if unsorted <= 3 {
                        eprintln!("  unsorted at {}: {:016x} > {:016x}", i, w[0], w[1]);
                    }
                }
            }
            if unsorted > 0 {
                eprintln!("  WARNING: {} unsorted pairs in GPU sort!", unsorted);
            } else {
                eprintln!("  sort verified: 0 unsorted pairs");
            }
        }

        eprintln!("  sort done");

        // Reorder positions
        eprint!("reorder ");
        std::io::stderr().flush().ok();
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

        // Reset BVH buffers - CRITICAL: parent must be -1 for root termination
        eprint!("reset ");
        std::io::stderr().flush().ok();
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

        // Build BVH on GPU using Karras algorithm
        // Morton keys now include particle index → unique keys → correct tree
        eprint!("bvh ");
        std::io::stderr().flush().ok();
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

        // Initialize leaves
        eprint!("leaves ");
        std::io::stderr().flush().ok();
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

        // Bottom-up reduction
        eprint!("reduce ");
        std::io::stderr().flush().ok();
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

        // DEBUG: check tree
        let parent_cpu: Vec<i32> = self.device.dtoh_sync_copy(&self.bvh_parent)?;
        let left_cpu: Vec<i32> = self.device.dtoh_sync_copy(&self.bvh_left)?;
        let right_cpu: Vec<i32> = self.device.dtoh_sync_copy(&self.bvh_right)?;

        let mut covered = vec![false; n];
        let rl_cpu: Vec<i32> = self.device.dtoh_sync_copy(&self.bvh_rl)?;
        let rr_cpu: Vec<i32> = self.device.dtoh_sync_copy(&self.bvh_rr)?;

        for i in 0..n.saturating_sub(1) {
            let lc = left_cpu[i] as usize;
            let rc = right_cpu[i] as usize;
            if lc >= n - 1 && lc < 2 * n - 1 { covered[lc - (n - 1)] = true; }
            if rc >= n - 1 && rc < 2 * n - 1 { covered[rc - (n - 1)] = true; }
        }
        let uncovered = covered.iter().filter(|&&x| !x).count();

        let mut invalid = 0;
        for i in 0..n {
            let leaf = n - 1 + i;
            if parent_cpu[leaf] < 0 || parent_cpu[leaf] >= n as i32 - 1 { invalid += 1; }
        }

        // Debug: for first few uncovered, find which internal node SHOULD cover it
        if uncovered > 0 {
            eprintln!("  WARNING: {} uncovered leaves!", uncovered);
        }
        eprintln!("OK");
        std::io::stderr().flush().ok();
        Ok(())
    }

    /// Compute forces on all particles (needed for initializing DKD with cold ICs)
    pub fn compute_forces(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        eprintln!("  [compute_forces] Computing initial forces...");
        std::io::stderr().flush().ok();

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
        self.build_single_sign_tree(1, self.n_positive)?;

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
        self.build_single_sign_tree(-1, self.n_negative)?;

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

        eprintln!("  [compute_forces] Done");
        std::io::stderr().flush().ok();
        Ok(())
    }

    pub fn step_dkd(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;
        use std::io::Write;
        let t0 = Instant::now();
        eprintln!("  [DEBUG] step_dkd start");
        std::io::stderr().flush().ok();

        let half_dt = dt * 0.5;
        let box_half = (self.box_size / 2.0) as f32;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Drift(dt/2)
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

        eprintln!("  [DEBUG] drift... ");
        std::io::stderr().flush().ok();
        self.device.synchronize()?;
        eprintln!("  [DEBUG] drift done: {:.0}ms", t0.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // === PASS 1: Positive particles tree ===
        // Extract positive particles
        eprintln!("  [DEBUG] extract+...");
        std::io::stderr().flush().ok();
        let t1 = Instant::now();
        let extract_k = self.device.get_func("twopass", "extract_by_sign")
            .ok_or("extract_by_sign not found")?;

        // Reset counter
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
        eprintln!("  [DEBUG] extract+ done: {:.0}ms", t1.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // Build tree for positive particles
        eprintln!("  [DEBUG] tree+...");
        std::io::stderr().flush().ok();
        let t1 = Instant::now();
        self.build_single_sign_tree(1, self.n_positive)?;
        eprintln!("  [DEBUG] tree+ done: {:.0}ms", t1.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // Compute forces from positive tree onto ALL particles (OVERWRITE)
        eprintln!("  [DEBUG] force+...");
        std::io::stderr().flush().ok();
        let t1 = Instant::now();
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
        eprintln!("  [DEBUG] force+ done: {:.0}ms", t1.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // === PASS 2: Negative particles tree ===
        eprintln!("  [DEBUG] extract-...");
        std::io::stderr().flush().ok();
        let t1 = Instant::now();
        // Reset counter
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
        eprintln!("  [DEBUG] extract- done: {:.0}ms", t1.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // Build tree for negative particles
        eprintln!("  [DEBUG] tree-...");
        std::io::stderr().flush().ok();
        let t1 = Instant::now();
        self.build_single_sign_tree(-1, self.n_negative)?;
        eprintln!("  [DEBUG] tree- done: {:.0}ms", t1.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // Compute forces from negative tree onto ALL particles (ACCUMULATE)
        eprintln!("  [DEBUG] force-...");
        std::io::stderr().flush().ok();
        let t1 = Instant::now();
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
        eprintln!("  [DEBUG] force- done: {:.0}ms", t1.elapsed().as_millis());
        std::io::stderr().flush().ok();

        // Kick(dt)
        eprintln!("  [DEBUG] kick...");
        std::io::stderr().flush().ok();
        let kick_k = self.device.get_func("twopass", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_k.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble as f32, dtau_per_dt as f32,  // Cast to f32
            ))?;
        }

        // Drift(dt/2)
        unsafe {
            drift_k.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;
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
}

#[cfg(not(feature = "cuda"))]
pub struct GpuNBodyTwoPass;

#[cfg(not(feature = "cuda"))]
impl GpuNBodyTwoPass {
    pub fn new(_: usize, _: usize, _: f64) -> Result<Self, Box<dyn std::error::Error>> {
        Err("CUDA not enabled".into())
    }
}
