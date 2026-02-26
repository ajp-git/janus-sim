/// GPU N-body simulation with MIXED PRECISION optimization
///
/// Optimizations:
/// 1. pos/vel stored as f32 (saves ~40% VRAM)
/// 2. Accumulation in f64 for accuracy
/// 3. GPU-only BVH tree (no CPU transfer!)
/// 4. Mixed precision tree: f32 positions, f64 masses (42% tree savings)
///
/// Memory for 30M particles:
/// - Old f64 version: ~11.3 GB (OOM)
/// - New mixed version: ~7.5 GB (fits in 12GB!)

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Mixed precision CUDA kernels
/// Tree format: 10×f32 for positions, 2×f64 for masses
const CUDA_MIXED_KERNELS: &str = r#"

// ============================================================================
// MIXED PRECISION KERNELS
// pos/vel: f32, acc: f64, tree: f32 positions + f64 masses
// ============================================================================

// Drift with f32 pos/vel
extern "C" __global__ void drift_f32(
    float* __restrict__ pos,
    const float* __restrict__ vel,
    float dt,
    float box_half,
    int n
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

// Kick with f32 vel, f64 acc
extern "C" __global__ void kick_f32(
    float* __restrict__ vel,
    const double* __restrict__ acc,
    float dt,
    int n,
    double hubble_param,
    double dtau_per_dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 3;

    for (int d = 0; d < 3; d++) {
        double v = (double)vel[base + d];
        double friction = -hubble_param * v * dtau_per_dt;
        double new_v = v + (acc[base + d] + friction) * (double)dt;
        vel[base + d] = (float)new_v;
    }
}

// Morton code from f32 positions
extern "C" __global__ void compute_morton_f32(
    const float* __restrict__ pos,
    unsigned long long* __restrict__ morton_codes,
    int* __restrict__ indices,
    int n,
    float box_half,
    float inv_cell_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = (pos[tid * 3]     + box_half) * inv_cell_size;
    float y = (pos[tid * 3 + 1] + box_half) * inv_cell_size;
    float z = (pos[tid * 3 + 2] + box_half) * inv_cell_size;

    unsigned int ix = min(max((unsigned int)x, 0u), 0x1fffffu);
    unsigned int iy = min(max((unsigned int)y, 0u), 0x1fffffu);
    unsigned int iz = min(max((unsigned int)z, 0u), 0x1fffffu);

    unsigned long long expand_x = ix & 0x1fffff;
    expand_x = (expand_x | (expand_x << 32)) & 0x1f00000000ffffULL;
    expand_x = (expand_x | (expand_x << 16)) & 0x1f0000ff0000ffULL;
    expand_x = (expand_x | (expand_x << 8))  & 0x100f00f00f00f00fULL;
    expand_x = (expand_x | (expand_x << 4))  & 0x10c30c30c30c30c3ULL;
    expand_x = (expand_x | (expand_x << 2))  & 0x1249249249249249ULL;

    unsigned long long expand_y = iy & 0x1fffff;
    expand_y = (expand_y | (expand_y << 32)) & 0x1f00000000ffffULL;
    expand_y = (expand_y | (expand_y << 16)) & 0x1f0000ff0000ffULL;
    expand_y = (expand_y | (expand_y << 8))  & 0x100f00f00f00f00fULL;
    expand_y = (expand_y | (expand_y << 4))  & 0x10c30c30c30c30c3ULL;
    expand_y = (expand_y | (expand_y << 2))  & 0x1249249249249249ULL;

    unsigned long long expand_z = iz & 0x1fffff;
    expand_z = (expand_z | (expand_z << 32)) & 0x1f00000000ffffULL;
    expand_z = (expand_z | (expand_z << 16)) & 0x1f0000ff0000ffULL;
    expand_z = (expand_z | (expand_z << 8))  & 0x100f00f00f00f00fULL;
    expand_z = (expand_z | (expand_z << 4))  & 0x10c30c30c30c30c3ULL;
    expand_z = (expand_z | (expand_z << 2))  & 0x1249249249249249ULL;

    morton_codes[tid] = expand_x | (expand_y << 1) | (expand_z << 2);
    indices[tid] = tid;
}

// Reorder f32 data
extern "C" __global__ void reorder_f32x3(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int* __restrict__ sorted_indices,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int src = sorted_indices[tid];
    out[tid * 3]     = in[src * 3];
    out[tid * 3 + 1] = in[src * 3 + 1];
    out[tid * 3 + 2] = in[src * 3 + 2];
}

extern "C" __global__ void reorder_i32(
    const int* __restrict__ in,
    int* __restrict__ out,
    const int* __restrict__ sorted_indices,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = in[sorted_indices[tid]];
}

// ============================================================================
// GPU BVH with MIXED PRECISION tree
// node_pos: 10×f32 per node (cx,cy,cz,half_size, com+xyz, com-xyz)
// node_mass: 2×f64 per node (mass+, mass-)
// ============================================================================

__device__ int clz64_m(unsigned long long x) {
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

__device__ int delta_m(const unsigned long long* morton, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    if (morton[i] == morton[j]) return 64 + clz64_m(i ^ j);
    return clz64_m(morton[i] ^ morton[j]);
}

__device__ void determine_range_m(const unsigned long long* morton, int n, int i, int* first, int* last) {
    if (i == 0) { *first = 0; *last = n - 1; return; }

    int d_left = delta_m(morton, n, i, i - 1);
    int d_right = delta_m(morton, n, i, i + 1);
    int d = (d_right > d_left) ? 1 : -1;
    int d_min = (d > 0) ? d_left : d_right;

    int lmax = 2;
    while (delta_m(morton, n, i, i + lmax * d) > d_min) lmax *= 2;

    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (delta_m(morton, n, i, i + (l + t) * d) > d_min) l += t;
    }
    int j = i + l * d;

    if (d > 0) { *first = i; *last = j; }
    else { *first = j; *last = i; }
}

__device__ int find_split_m(const unsigned long long* morton, int n, int first, int last) {
    if (first == last) return first;
    int d_node = delta_m(morton, n, first, last);
    int s = 0, t = last - first;
    while (t > 1) {
        t = (t + 1) / 2;
        if (delta_m(morton, n, first, first + s + t) > d_node) s += t;
    }
    return first + s;
}

extern "C" __global__ void build_bvh_internal_m(
    const unsigned long long* __restrict__ morton,
    int* __restrict__ left_child,
    int* __restrict__ right_child,
    int* __restrict__ parent,
    int* __restrict__ range_left,
    int* __restrict__ range_right,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    int first, last;
    determine_range_m(morton, n, i, &first, &last);
    int split = find_split_m(morton, n, first, last);

    int left = (split == first) ? (n - 1 + split) : split;
    int right = (split + 1 == last) ? (n - 1 + split + 1) : (split + 1);

    left_child[i] = left;
    right_child[i] = right;
    range_left[i] = first;
    range_right[i] = last;

    parent[left] = i;
    parent[right] = i;
}

// Initialize leaves with MIXED precision tree
// node_pos: 10 floats per node [cx,cy,cz,half_size, com+x,y,z, com-x,y,z]
// node_mass: 2 doubles per node [mass+, mass-]
extern "C" __global__ void init_leaves_mixed(
    const float* __restrict__ pos,
    const int* __restrict__ signs,
    float* __restrict__ node_pos,      // 10×f32 per node
    double* __restrict__ node_mass,    // 2×f64 per node
    int* __restrict__ node_types,
    int* __restrict__ atomic_counter,
    int n,
    float box_half
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int node_idx = n - 1 + i;
    int pos_base = node_idx * 10;
    int mass_base = node_idx * 2;

    float x = pos[i * 3];
    float y = pos[i * 3 + 1];
    float z = pos[i * 3 + 2];
    int sign = signs[i];

    // Center and half_size
    node_pos[pos_base + 0] = x;
    node_pos[pos_base + 1] = y;
    node_pos[pos_base + 2] = z;
    node_pos[pos_base + 3] = box_half / 1024.0f;

    if (sign > 0) {
        // COM+ = particle position
        node_pos[pos_base + 4] = x;
        node_pos[pos_base + 5] = y;
        node_pos[pos_base + 6] = z;
        node_mass[mass_base + 0] = 1.0;  // mass+
        // COM- = 0
        node_pos[pos_base + 7] = 0.0f;
        node_pos[pos_base + 8] = 0.0f;
        node_pos[pos_base + 9] = 0.0f;
        node_mass[mass_base + 1] = 0.0;  // mass-
    } else {
        node_pos[pos_base + 4] = 0.0f;
        node_pos[pos_base + 5] = 0.0f;
        node_pos[pos_base + 6] = 0.0f;
        node_mass[mass_base + 0] = 0.0;
        node_pos[pos_base + 7] = x;
        node_pos[pos_base + 8] = y;
        node_pos[pos_base + 9] = z;
        node_mass[mass_base + 1] = 1.0;
    }

    node_types[node_idx] = 1;
    atomic_counter[node_idx] = 0;
}

// Bottom-up COM reduction with MIXED precision
extern "C" __global__ void reduce_com_mixed(
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ parent,
    const int* __restrict__ range_left,
    const int* __restrict__ range_right,
    float* __restrict__ node_pos,
    double* __restrict__ node_mass,
    int* __restrict__ node_types,
    int* __restrict__ atomic_counter,
    int n,
    float box_half
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= n) return;

    int node_idx = n - 1 + leaf_idx;
    int current = parent[node_idx];

    while (current >= 0 && current < n - 1) {
        int count = atomicAdd(&atomic_counter[current], 1);
        if (count == 0) return;

        int left = left_child[current];
        int right = right_child[current];

        int pos_c = current * 10;
        int pos_l = left * 10;
        int pos_r = right * 10;
        int mass_c = current * 2;
        int mass_l = left * 2;
        int mass_r = right * 2;

        double l_mp = node_mass[mass_l + 0];
        double l_mm = node_mass[mass_l + 1];
        double r_mp = node_mass[mass_r + 0];
        double r_mm = node_mass[mass_r + 1];

        // COM+ (compute in double, store in float)
        double total_mp = l_mp + r_mp;
        float com_plus_x = 0.0f, com_plus_y = 0.0f, com_plus_z = 0.0f;
        if (total_mp > 0.0) {
            com_plus_x = (float)((l_mp * node_pos[pos_l + 4] + r_mp * node_pos[pos_r + 4]) / total_mp);
            com_plus_y = (float)((l_mp * node_pos[pos_l + 5] + r_mp * node_pos[pos_r + 5]) / total_mp);
            com_plus_z = (float)((l_mp * node_pos[pos_l + 6] + r_mp * node_pos[pos_r + 6]) / total_mp);
        }

        // COM-
        double total_mm = l_mm + r_mm;
        float com_minus_x = 0.0f, com_minus_y = 0.0f, com_minus_z = 0.0f;
        if (total_mm > 0.0) {
            com_minus_x = (float)((l_mm * node_pos[pos_l + 7] + r_mm * node_pos[pos_r + 7]) / total_mm);
            com_minus_y = (float)((l_mm * node_pos[pos_l + 8] + r_mm * node_pos[pos_r + 8]) / total_mm);
            com_minus_z = (float)((l_mm * node_pos[pos_l + 9] + r_mm * node_pos[pos_r + 9]) / total_mm);
        }

        // Bounding box
        int first = range_left[current];
        int last = range_right[current];
        int range_size = last - first + 1;
        float frac = (float)range_size / (float)n;
        float half_size = box_half * cbrtf(frac);
        if (half_size < box_half / 1024.0f) half_size = box_half / 1024.0f;

        // Geometric center
        double total_mass = total_mp + total_mm + 1e-20;
        float cx = (float)((total_mp * com_plus_x + total_mm * com_minus_x) / total_mass);
        float cy = (float)((total_mp * com_plus_y + total_mm * com_minus_y) / total_mass);
        float cz = (float)((total_mp * com_plus_z + total_mm * com_minus_z) / total_mass);

        // Store
        node_pos[pos_c + 0] = cx;
        node_pos[pos_c + 1] = cy;
        node_pos[pos_c + 2] = cz;
        node_pos[pos_c + 3] = half_size;
        node_pos[pos_c + 4] = com_plus_x;
        node_pos[pos_c + 5] = com_plus_y;
        node_pos[pos_c + 6] = com_plus_z;
        node_pos[pos_c + 7] = com_minus_x;
        node_pos[pos_c + 8] = com_minus_y;
        node_pos[pos_c + 9] = com_minus_z;
        node_mass[mass_c + 0] = total_mp;
        node_mass[mass_c + 1] = total_mm;

        node_types[current] = 2;

        if (current == 0) break;
        current = parent[current];
    }
}

// Force computation with MIXED precision tree
extern "C" __global__ void compute_forces_mixed(
    const float* __restrict__ pos,
    const int* __restrict__ signs,
    const float* __restrict__ node_pos,
    const double* __restrict__ node_mass,
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ node_types,
    double* __restrict__ acc,
    int n_particles,
    int n_internal,
    double theta,
    double softening
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    // Read particle position (f32 → f64 for computation)
    double px = (double)pos[tid * 3];
    double py = (double)pos[tid * 3 + 1];
    double pz = (double)pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    double eps2 = softening * softening;

    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0) continue;

        int node_type = node_types[node_idx];
        if (node_type == 0) continue;

        int pos_base = node_idx * 10;
        int mass_base = node_idx * 2;

        double cx = (double)node_pos[pos_base + 0];
        double cy = (double)node_pos[pos_base + 1];
        double cz = (double)node_pos[pos_base + 2];
        double half_size = (double)node_pos[pos_base + 3];

        double dx = cx - px;
        double dy = cy - py;
        double dz = cz - pz;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2 + 1e-20);

        double s_over_r = (2.0 * half_size) / r;

        if (node_type == 1 || s_over_r < theta) {
            double mass_plus = node_mass[mass_base + 0];
            double mass_minus = node_mass[mass_base + 1];

            if (mass_plus > 0.0) {
                double cpx = (double)node_pos[pos_base + 4];
                double cpy = (double)node_pos[pos_base + 5];
                double cpz = (double)node_pos[pos_base + 6];
                double dpx = cpx - px;
                double dpy = cpy - py;
                double dpz = cpz - pz;
                double rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
                double inv_rp3 = 1.0 / (rp2 * sqrt(rp2));
                double interaction = (my_sign > 0) ? 1.0 : -1.0;
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            if (mass_minus > 0.0) {
                double cmx = (double)node_pos[pos_base + 7];
                double cmy = (double)node_pos[pos_base + 8];
                double cmz = (double)node_pos[pos_base + 9];
                double dmx = cmx - px;
                double dmy = cmy - py;
                double dmz = cmz - pz;
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double inv_rm3 = 1.0 / (rm2 * sqrt(rm2));
                double interaction = (my_sign < 0) ? 1.0 : -1.0;
                double f = interaction * mass_minus * inv_rm3;
                ax += f * dmx;
                ay += f * dmy;
                az += f * dmz;
            }
        } else {
            int left = left_child[node_idx];
            int right = right_child[node_idx];
            if (left >= 0 && stack_ptr < 63) stack[stack_ptr++] = left;
            if (right >= 0 && stack_ptr < 63) stack[stack_ptr++] = right;
        }
    }

    acc[tid * 3] = ax;
    acc[tid * 3 + 1] = ay;
    acc[tid * 3 + 2] = az;
}

extern "C" __global__ void reset_atomic_m(int* __restrict__ counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) counter[tid] = 0;
}

extern "C" __global__ void bitonic_sort_m(
    unsigned long long* __restrict__ keys,
    int* __restrict__ values,
    int j, int k, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int ixj = i ^ j;
    if (ixj > i && ixj < n) {
        bool ascending = ((i & k) == 0);
        if ((keys[i] > keys[ixj]) == ascending) {
            unsigned long long tmp_k = keys[i];
            keys[i] = keys[ixj];
            keys[ixj] = tmp_k;
            int tmp_v = values[i];
            values[i] = values[ixj];
            values[ixj] = tmp_v;
        }
    }
}

"#;

/// Mixed precision GPU N-body simulation
#[cfg(feature = "cuda")]
pub struct GpuNBodyMixed {
    device: Arc<CudaDevice>,
    // Particles: f32 for pos/vel
    pos: CudaSlice<f32>,
    vel: CudaSlice<f32>,
    pos_tmp: CudaSlice<f32>,
    vel_tmp: CudaSlice<f32>,
    // f64 for acceleration (precision critical)
    acc: CudaSlice<f64>,
    signs: CudaSlice<i32>,
    signs_tmp: CudaSlice<i32>,
    // Morton sorting
    morton_codes: CudaSlice<u64>,
    sorted_indices: CudaSlice<i32>,
    // BVH structure
    bvh_left_child: CudaSlice<i32>,
    bvh_right_child: CudaSlice<i32>,
    bvh_parent: CudaSlice<i32>,
    bvh_range_left: CudaSlice<i32>,
    bvh_range_right: CudaSlice<i32>,
    // MIXED PRECISION TREE: f32 for positions, f64 for masses
    bvh_node_pos: CudaSlice<f32>,   // 10×f32 per node (56→40 bytes)
    bvh_node_mass: CudaSlice<f64>,  // 2×f64 per node (16 bytes)
    bvh_node_types: CudaSlice<i32>,
    bvh_atomic_counter: CudaSlice<i32>,
    // Parameters
    n_particles: usize,
    theta: f64,
    softening: f64,
    box_size: f64,
    time: f64,
}

#[cfg(feature = "cuda")]
impl GpuNBodyMixed {
    pub fn new(n_positive: usize, n_negative: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_MIXED_KERNELS)?;
        device.load_ptx(ptx, "mixed", &[
            "drift_f32", "kick_f32", "compute_morton_f32",
            "reorder_f32x3", "reorder_i32",
            "build_bvh_internal_m", "init_leaves_mixed", "reduce_com_mixed",
            "compute_forces_mixed", "reset_atomic_m", "bitonic_sort_m"
        ])?;

        let n_total = n_positive + n_negative;
        let n_bvh = 2 * n_total - 1;

        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);

        let v_init = 1.0f32;
        let mut pos_data: Vec<f32> = Vec::with_capacity(n_total * 3);
        let mut vel_data: Vec<f32> = Vec::with_capacity(n_total * 3);
        let mut signs_data: Vec<i32> = Vec::with_capacity(n_total);

        for _ in 0..n_positive {
            pos_data.extend_from_slice(&[
                (rng.random::<f32>() - 0.5) * box_size as f32,
                (rng.random::<f32>() - 0.5) * box_size as f32,
                (rng.random::<f32>() - 0.5) * box_size as f32,
            ]);
            vel_data.extend_from_slice(&[
                (rng.random::<f32>() - 0.5) * v_init,
                (rng.random::<f32>() - 0.5) * v_init,
                (rng.random::<f32>() - 0.5) * v_init,
            ]);
            signs_data.push(1);
        }

        for _ in 0..n_negative {
            pos_data.extend_from_slice(&[
                (rng.random::<f32>() - 0.5) * box_size as f32,
                (rng.random::<f32>() - 0.5) * box_size as f32,
                (rng.random::<f32>() - 0.5) * box_size as f32,
            ]);
            vel_data.extend_from_slice(&[
                (rng.random::<f32>() - 0.5) * v_init,
                (rng.random::<f32>() - 0.5) * v_init,
                (rng.random::<f32>() - 0.5) * v_init,
            ]);
            signs_data.push(-1);
        }

        // Analytical virialization
        let mass = 1.0f64;
        let g_code = 1.0f64;
        let mean_sep_plus = 0.554 * box_size / (n_positive as f64).cbrt();
        let mean_sep_minus = 0.554 * box_size / (n_negative as f64).cbrt();
        let pe_plus = -g_code * mass * mass * (n_positive * n_positive.saturating_sub(1) / 2) as f64 / mean_sep_plus;
        let pe_minus = -g_code * mass * mass * (n_negative * n_negative.saturating_sub(1) / 2) as f64 / mean_sep_minus;
        let pe_binding = pe_plus + pe_minus;

        let ke_current: f64 = vel_data.chunks(3)
            .map(|v| 0.5 * mass * ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)))
            .sum();

        let ke_target = pe_binding.abs() / 2.0;
        let alpha = if ke_current > 1e-20 { (ke_target / ke_current).sqrt() } else { 1.0 };

        for v in vel_data.iter_mut() {
            *v *= alpha as f32;
        }

        println!("Mixed precision virialization:");
        println!("  PE_binding = {:.4e}", pe_binding);
        println!("  alpha      = {:.6}", alpha);

        // Allocate GPU buffers
        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;

        let pos_tmp = device.alloc_zeros::<f32>(n_total * 3)?;
        let vel_tmp = device.alloc_zeros::<f32>(n_total * 3)?;
        let signs_tmp = device.alloc_zeros::<i32>(n_total)?;
        let acc = device.alloc_zeros::<f64>(n_total * 3)?;

        let morton_codes = device.alloc_zeros::<u64>(n_total)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_total)?;

        // BVH structure
        let bvh_left_child = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_right_child = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_range_left = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_range_right = device.alloc_zeros::<i32>(n_bvh)?;
        // MIXED: f32 for positions (10 per node), f64 for masses (2 per node)
        let bvh_node_pos = device.alloc_zeros::<f32>(n_bvh * 10)?;
        let bvh_node_mass = device.alloc_zeros::<f64>(n_bvh * 2)?;
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh)?;
        let bvh_atomic_counter = device.alloc_zeros::<i32>(n_bvh)?;

        // Memory calculation
        let mem_particles = (n_total * 3 * 4 * 4 + n_total * 3 * 8 + n_total * 4 * 3) as f64 / 1e9;
        let mem_sort = (n_total * (8 + 4)) as f64 / 1e9;
        let mem_bvh_struct = (n_bvh * 4 * 5) as f64 / 1e9;
        let mem_bvh_data = (n_bvh * 10 * 4 + n_bvh * 2 * 8) as f64 / 1e9;  // Mixed!
        let total = mem_particles + mem_sort + mem_bvh_struct + mem_bvh_data;
        println!("  VRAM estimate: {:.2} GB", total);
        println!("    - Particles: {:.2} GB", mem_particles);
        println!("    - Sorting: {:.2} GB", mem_sort);
        println!("    - BVH struct: {:.2} GB", mem_bvh_struct);
        println!("    - BVH data (mixed): {:.2} GB", mem_bvh_data);

        Ok(Self {
            device,
            pos, vel, pos_tmp, vel_tmp, acc, signs, signs_tmp,
            morton_codes, sorted_indices,
            bvh_left_child, bvh_right_child, bvh_parent,
            bvh_range_left, bvh_range_right,
            bvh_node_pos, bvh_node_mass,
            bvh_node_types, bvh_atomic_counter,
            n_particles: n_total,
            theta: 1.5,
            softening: 0.1,
            box_size,
            time: 0.0,
        })
    }

    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
    }

    pub fn build_tree_gpu(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;
        let n_internal = n - 1;
        let n_total_nodes = 2 * n - 1;
        let box_half = (self.box_size / 2.0) as f32;
        let inv_cell_size = (2097152.0 / self.box_size) as f32;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Morton codes
        let morton_kernel = self.device.get_func("mixed", "compute_morton_f32")
            .ok_or("compute_morton_f32 not found")?;
        unsafe {
            morton_kernel.launch(cfg, (
                &self.pos, &mut self.morton_codes, &mut self.sorted_indices,
                n as i32, box_half, inv_cell_size,
            ))?;
        }

        // GPU Bitonic sort
        let sort_kernel = self.device.get_func("mixed", "bitonic_sort_m")
            .ok_or("bitonic_sort_m not found")?;

        let mut size = 1;
        while size < n { size *= 2; }

        let sort_blocks = (size + 255) / 256;
        let sort_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (sort_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut k = 2;
        while k <= size {
            let mut j = k / 2;
            while j > 0 {
                unsafe {
                    sort_kernel.clone().launch(sort_cfg, (
                        &mut self.morton_codes, &mut self.sorted_indices,
                        j as i32, k as i32, n as i32,
                    ))?;
                }
                j /= 2;
            }
            k *= 2;
        }

        // Reorder particles
        let reorder_f32 = self.device.get_func("mixed", "reorder_f32x3")
            .ok_or("reorder_f32x3 not found")?;
        unsafe {
            reorder_f32.clone().launch(cfg, (&self.pos, &mut self.pos_tmp, &self.sorted_indices, n as i32))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_tmp);

        unsafe {
            reorder_f32.launch(cfg, (&self.vel, &mut self.vel_tmp, &self.sorted_indices, n as i32))?;
        }
        std::mem::swap(&mut self.vel, &mut self.vel_tmp);

        let reorder_i32 = self.device.get_func("mixed", "reorder_i32")
            .ok_or("reorder_i32 not found")?;
        unsafe {
            reorder_i32.launch(cfg, (&self.signs, &mut self.signs_tmp, &self.sorted_indices, n as i32))?;
        }
        std::mem::swap(&mut self.signs, &mut self.signs_tmp);

        // Reset atomic counters
        let reset_blocks = (n_total_nodes + 255) / 256;
        let reset_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (reset_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let reset_kernel = self.device.get_func("mixed", "reset_atomic_m")
            .ok_or("reset_atomic_m not found")?;
        unsafe {
            reset_kernel.launch(reset_cfg, (&mut self.bvh_atomic_counter, n_total_nodes as i32))?;
        }

        // Build internal nodes
        let internal_blocks = (n_internal + 255) / 256;
        let internal_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (internal_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let build_kernel = self.device.get_func("mixed", "build_bvh_internal_m")
            .ok_or("build_bvh_internal_m not found")?;
        unsafe {
            build_kernel.launch(internal_cfg, (
                &self.morton_codes,
                &mut self.bvh_left_child, &mut self.bvh_right_child, &mut self.bvh_parent,
                &mut self.bvh_range_left, &mut self.bvh_range_right,
                n as i32,
            ))?;
        }

        // Initialize leaves (mixed precision)
        let init_kernel = self.device.get_func("mixed", "init_leaves_mixed")
            .ok_or("init_leaves_mixed not found")?;
        unsafe {
            init_kernel.launch(cfg, (
                &self.pos, &self.signs,
                &mut self.bvh_node_pos, &mut self.bvh_node_mass,
                &mut self.bvh_node_types, &mut self.bvh_atomic_counter,
                n as i32, box_half,
            ))?;
        }

        // Bottom-up reduction
        let reduce_kernel = self.device.get_func("mixed", "reduce_com_mixed")
            .ok_or("reduce_com_mixed not found")?;
        unsafe {
            reduce_kernel.launch(cfg, (
                &self.bvh_left_child, &self.bvh_right_child, &self.bvh_parent,
                &self.bvh_range_left, &self.bvh_range_right,
                &mut self.bvh_node_pos, &mut self.bvh_node_mass,
                &mut self.bvh_node_types, &mut self.bvh_atomic_counter,
                n as i32, box_half,
            ))?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    pub fn step_dkd(&mut self, dt: f64, hubble: f64, dtau_per_dt: f64) -> Result<(), Box<dyn std::error::Error>> {
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
        let drift_kernel = self.device.get_func("mixed", "drift_f32")
            .ok_or("drift_f32 not found")?;
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt as f32, box_half, n as i32,
            ))?;
        }

        // Build tree on GPU
        self.build_tree_gpu()?;

        // Compute forces
        let forces_kernel = self.device.get_func("mixed", "compute_forces_mixed")
            .ok_or("compute_forces_mixed not found")?;
        unsafe {
            forces_kernel.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_pos, &self.bvh_node_mass,
                &self.bvh_left_child, &self.bvh_right_child, &self.bvh_node_types,
                &mut self.acc,
                n as i32, (n - 1) as i32,
                self.theta, self.softening,
            ))?;
        }

        // Kick(dt)
        let kick_kernel = self.device.get_func("mixed", "kick_f32")
            .ok_or("kick_f32 not found")?;
        unsafe {
            kick_kernel.launch(cfg, (
                &mut self.vel, &self.acc,
                dt as f32, n as i32,
                hubble, dtau_per_dt,
            ))?;
        }

        // Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
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

    pub fn segregation(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        let signs_cpu = self.device.dtoh_sync_copy(&self.signs)?;

        let n = self.n_particles;
        let mut com_plus = [0.0f64; 3];
        let mut com_minus = [0.0f64; 3];
        let mut n_plus = 0usize;
        let mut n_minus = 0usize;

        for i in 0..n {
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

    pub fn positions_f64(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let pos_f32 = self.device.dtoh_sync_copy(&self.pos)?;
        Ok(pos_f32.iter().map(|&x| x as f64).collect())
    }

    pub fn velocities_f64(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let vel_f32 = self.device.dtoh_sync_copy(&self.vel)?;
        Ok(vel_f32.iter().map(|&x| x as f64).collect())
    }

    pub fn signs(&self) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.signs)?)
    }

    pub fn n_particles(&self) -> usize { self.n_particles }
    pub fn box_size(&self) -> f64 { self.box_size }
    pub fn time(&self) -> f64 { self.time }
}

#[cfg(not(feature = "cuda"))]
pub struct GpuNBodyMixed;

#[cfg(not(feature = "cuda"))]
impl GpuNBodyMixed {
    pub fn new(_: usize, _: usize, _: f64) -> Result<Self, Box<dyn std::error::Error>> {
        Err("CUDA not enabled".into())
    }
}
