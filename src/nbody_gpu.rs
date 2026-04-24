/// GPU-accelerated Barnes-Hut N-body simulation using CUDA
///
/// Based on Bedorf et al. 2012: "A sparse octree gravitational N-body code
/// that runs entirely on the GPU processor"
///
/// Key optimizations:
/// - Linear tree array representation (no pointers)
/// - Separate positive/negative mass COM for Janus interactions
/// - Parallel tree traversal on GPU

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};

use crate::nbody::{Vec3, BoundingBox};
use std::sync::Arc;

/// Simulation parameters passed to GPU
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SimParams {
    pub n_particles: i32,
    pub n_nodes: i32,
    pub theta: f32,
    pub softening: f32,
    pub box_half: f32,
    pub dt: f32,
    pub half_dt: f32,
    pub _pad: f32,
}

/// Linear tree node for GPU (Bedorf-style) - f64 precision
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct LinearTreeNode {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub half_size: f64,
    pub com_plus_x: f64,
    pub com_plus_y: f64,
    pub com_plus_z: f64,
    pub mass_plus: f64,
    pub com_minus_x: f64,
    pub com_minus_y: f64,
    pub com_minus_z: f64,
    pub mass_minus: f64,
    pub children: [u32; 8],
    pub node_type: u32,
    pub particle_idx: u32,
    pub _pad: [u32; 2],
}

/// Particle data in GPU-friendly format (f64 for precision)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct GpuParticle {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub mass: f64,
    pub sign: i32,
}

/// CUDA kernels using double precision (f64) for accuracy
const CUDA_KERNEL_SRC: &str = r#"
// Asymmetric softening ratio for m- particles (numerical artifact)
// m- uses SOFTENING_MINUS_RATIO × softening to simulate diffuse gas nature
#define SOFTENING_MINUS_RATIO 5.0

// Minimum image convention for periodic boundaries
__device__ inline double minimum_image(double d, double box_size, double box_half) {
    if (d >  box_half) d -= box_size;
    if (d < -box_half) d += box_size;
    return d;
}

extern "C" __global__ void compute_forces_simple(
    const double* __restrict__ pos,      // x,y,z interleaved
    const int* __restrict__ signs,
    const double* __restrict__ node_data,
    const int* __restrict__ node_children,
    const int* __restrict__ node_types,
    double* __restrict__ acc,            // ax,ay,az interleaved
    int n_particles,
    double theta,
    double softening,
    double c_ratio_sq,  // (c_minus/c_plus)^2, default=1.0, VSL=100.0
    double repulsion_scale,  // 0.0=no cross-species, 1.0=full Janus (gradual ramp)
    double box_size  // Box size for periodic boundary conditions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double box_half = 0.5 * box_size;
    double px = pos[tid * 3];
    double py = pos[tid * 3 + 1];
    double pz = pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    // Asymmetric softening: m- uses larger value to simulate diffuse gas nature
    double eps = (my_sign > 0) ? softening : (softening * SOFTENING_MINUS_RATIO);
    double eps2 = eps * eps;

    // Stack for tree traversal
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0) continue;  // node_type == 0 check below handles invalid nodes

        int node_type = node_types[node_idx];
        if (node_type == 0) continue;

        int base = node_idx * 12;
        double cx = node_data[base + 0];
        double cy = node_data[base + 1];
        double cz = node_data[base + 2];
        double half_size = node_data[base + 3];

        double dx = minimum_image(cx - px, box_size, box_half);
        double dy = minimum_image(cy - py, box_size, box_half);
        double dz = minimum_image(cz - pz, box_size, box_half);
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2 + 1e-20);

        double s_over_r = (2.0 * half_size) / r;

        if (node_type == 1 || s_over_r < theta) {
            double com_plus_x = node_data[base + 4];
            double com_plus_y = node_data[base + 5];
            double com_plus_z = node_data[base + 6];
            double mass_plus = node_data[base + 7];

            double com_minus_x = node_data[base + 8];
            double com_minus_y = node_data[base + 9];
            double com_minus_z = node_data[base + 10];
            double mass_minus = node_data[base + 11];

            if (mass_plus > 0.0) {
                double dpx = minimum_image(com_plus_x - px, box_size, box_half);
                double dpy = minimum_image(com_plus_y - py, box_size, box_half);
                double dpz = minimum_image(com_plus_z - pz, box_size, box_half);
                double rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
                double inv_rp3 = 1.0 / (rp2 * sqrt(rp2));
                // VSL Option B: m- feels AMPLIFIED repulsion from m+ (factor c_ratio_sq)
                // repulsion_scale: 0.0 = no cross-species, 1.0 = full Janus
                double interaction;
                if (my_sign > 0) {
                    interaction = 1.0;  // m+ ← m+ attraction (always)
                } else {
                    interaction = -c_ratio_sq * repulsion_scale;  // m- ← m+ repulsion (scaled)
                }
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            if (mass_minus > 0.0) {
                double dmx = minimum_image(com_minus_x - px, box_size, box_half);
                double dmy = minimum_image(com_minus_y - py, box_size, box_half);
                double dmz = minimum_image(com_minus_z - pz, box_size, box_half);
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double inv_rm3 = 1.0 / (rm2 * sqrt(rm2));
                // VSL Option B: m+ feels STANDARD repulsion from m- (factor 1.0)
                // repulsion_scale: 0.0 = no cross-species, 1.0 = full Janus
                double interaction;
                if (my_sign < 0) {
                    interaction = 1.0;  // m- ← m- attraction (always)
                } else {
                    interaction = -1.0 * repulsion_scale;  // m+ ← m- repulsion (scaled)
                }
                double f = interaction * mass_minus * inv_rm3;
                ax += f * dmx;
                ay += f * dmy;
                az += f * dmz;
            }
        } else {
            int child_base = node_idx * 8;
            for (int i = 0; i < 8; i++) {
                int child = node_children[child_base + i];
                if (child > 0 && stack_ptr < 32) {
                    stack[stack_ptr++] = child - 1;
                }
            }
        }
    }

    acc[tid * 3] = ax;
    acc[tid * 3 + 1] = ay;
    acc[tid * 3 + 2] = az;
}

extern "C" __global__ void leapfrog_kick_drift(
    double* __restrict__ pos,
    double* __restrict__ vel,
    const double* __restrict__ acc,
    double half_dt,
    double dt,
    double box_half,
    int n,
    int do_drift,
    double scale_factor,    // a(t) - facteur d'echelle cosmologique
    double hubble_param,    // H(t) = adot/a - parametre de Hubble
    double dtau_per_dt      // facteur de conversion dtau_cosmo / dt_nbody
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 3;

    // Kick avec Hubble friction (coordonnees physiques)
    // Tout en unites dt N-corps, H scale par dtau_per_dt
    for (int d = 0; d < 3; d++) {
        // 1. Friction de Hubble avec conversion d'unites
        double friction = -hubble_param * vel[base + d] * dtau_per_dt;

        // 2. Mise a jour de la vitesse (Kick) - tout en half_dt
        vel[base + d] += (acc[base + d] + friction) * half_dt;
    }

    // Drift (only if do_drift)
    if (do_drift) {
        pos[base]     += vel[base]     * dt;
        pos[base + 1] += vel[base + 1] * dt;
        pos[base + 2] += vel[base + 2] * dt;

        // Periodic BC
        for (int i = 0; i < 3; i++) {
            if (pos[base + i] > box_half) pos[base + i] -= 2.0 * box_half;
            if (pos[base + i] < -box_half) pos[base + i] += 2.0 * box_half;
        }
    }
}

// DKD optimization: drift-only kernel for DKD integrator
extern "C" __global__ void drift_only(
    double* __restrict__ pos,
    const double* __restrict__ vel,
    double dt,
    double box_half,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 3;

    // Drift positions
    pos[base]     += vel[base]     * dt;
    pos[base + 1] += vel[base + 1] * dt;
    pos[base + 2] += vel[base + 2] * dt;

    // Periodic boundary conditions
    for (int i = 0; i < 3; i++) {
        if (pos[base + i] > box_half) pos[base + i] -= 2.0 * box_half;
        if (pos[base + i] < -box_half) pos[base + i] += 2.0 * box_half;
    }
}

// DKD optimization: kick-only kernel with Hubble friction
extern "C" __global__ void kick_only(
    double* __restrict__ vel,
    const double* __restrict__ acc,
    double dt,
    int n,
    double hubble_param,
    double dtau_per_dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 3;

    // Kick with Hubble friction
    for (int d = 0; d < 3; d++) {
        double friction = -hubble_param * vel[base + d] * dtau_per_dt;
        vel[base + d] += (acc[base + d] + friction) * dt;
    }
}

// Morton code computation for spatial sorting
// Maps 3D position to 1D Morton code (Z-order curve) for cache locality
// Expand 10-bit integer to 30 bits with 2 zeros between each bit
__device__ unsigned long long expand_bits_10(unsigned int v) {
    unsigned long long x = v & 0x3ff;  // 10 bits
    x = (x | (x << 16)) & 0x30000ffULL;
    x = (x | (x << 8))  & 0x300f00fULL;
    x = (x | (x << 4))  & 0x30c30c3ULL;
    x = (x | (x << 2))  & 0x9249249ULL;
    return x;
}

extern "C" __global__ void compute_morton_codes(
    const double* __restrict__ pos,
    unsigned long long* __restrict__ morton_codes,
    int* __restrict__ indices,
    int n,
    double box_half,
    double inv_cell_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Read position and normalize to [0, 1024) range (10 bits per axis)
    double scale = 1024.0 / (2.0 * box_half);
    double x = (pos[tid * 3]     + box_half) * scale;
    double y = (pos[tid * 3 + 1] + box_half) * scale;
    double z = (pos[tid * 3 + 2] + box_half) * scale;

    // Clamp to valid range (10 bits = 0..1023)
    unsigned int ix = min(max((unsigned int)x, 0u), 0x3ffu);
    unsigned int iy = min(max((unsigned int)y, 0u), 0x3ffu);
    unsigned int iz = min(max((unsigned int)z, 0u), 0x3ffu);

    // Morton code = (30-bit spatial << 32) | particle_index
    // Unique keys guarantee correct Karras tree construction
    unsigned long long mc = expand_bits_10(ix) | (expand_bits_10(iy) << 1) | (expand_bits_10(iz) << 2);
    morton_codes[tid] = (mc << 32) | ((unsigned long long)tid);
    indices[tid] = tid;
}

// Reorder positions based on sorted indices
extern "C" __global__ void reorder_positions(
    const double* __restrict__ pos_in,
    double* __restrict__ pos_out,
    const int* __restrict__ sorted_indices,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int src = sorted_indices[tid];
    pos_out[tid * 3]     = pos_in[src * 3];
    pos_out[tid * 3 + 1] = pos_in[src * 3 + 1];
    pos_out[tid * 3 + 2] = pos_in[src * 3 + 2];
}

// Reorder velocities based on sorted indices
extern "C" __global__ void reorder_velocities(
    const double* __restrict__ vel_in,
    double* __restrict__ vel_out,
    const int* __restrict__ sorted_indices,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int src = sorted_indices[tid];
    vel_out[tid * 3]     = vel_in[src * 3];
    vel_out[tid * 3 + 1] = vel_in[src * 3 + 1];
    vel_out[tid * 3 + 2] = vel_in[src * 3 + 2];
}

// Reorder signs based on sorted indices
extern "C" __global__ void reorder_signs(
    const int* __restrict__ signs_in,
    int* __restrict__ signs_out,
    const int* __restrict__ sorted_indices,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    signs_out[tid] = signs_in[sorted_indices[tid]];
}

// Reorder masses based on sorted indices (for adaptive splitting)
extern "C" __global__ void reorder_masses(
    const double* __restrict__ masses_in,
    double* __restrict__ masses_out,
    const int* __restrict__ sorted_indices,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    masses_out[tid] = masses_in[sorted_indices[tid]];
}

// ============================================================================
// GPU BVH Construction (Karras 2012)
// ============================================================================

// Count leading zeros in 64-bit integer
__device__ int clz64(unsigned long long x) {
    if (x == 0) return 64;
    int n = 0;
    if ((x & 0xFFFFFFFF00000000ULL) == 0) { n += 32; x <<= 32; }
    if ((x & 0xFFFF000000000000ULL) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF00000000000000ULL) == 0) { n += 8;  x <<= 8; }
    if ((x & 0xF000000000000000ULL) == 0) { n += 4;  x <<= 4; }
    if ((x & 0xC000000000000000ULL) == 0) { n += 2;  x <<= 2; }
    if ((x & 0x8000000000000000ULL) == 0) { n += 1; }
    return n;
}

// Compute longest common prefix length between two Morton codes
__device__ int delta(const unsigned long long* morton, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    unsigned long long ki = morton[i];
    unsigned long long kj = morton[j];
    if (ki == kj) {
        // If codes are equal, use index to break tie
        return 64 + clz64((unsigned long long)(i ^ j));
    }
    return clz64(ki ^ kj);
}

// Karras 2012: Determine range of keys covered by internal node i
__device__ void determine_range(
    const unsigned long long* morton,
    int n,
    int i,
    int* first,
    int* last
) {
    // Determine direction of range (+1 or -1)
    int d_left = delta(morton, i, i - 1, n);
    int d_right = delta(morton, i, i + 1, n);
    int d = (d_right > d_left) ? 1 : -1;
    int d_min = (d > 0) ? d_left : d_right;

    // Find upper bound for range length
    int l_max = 2;
    while (delta(morton, i, i + l_max * d, n) > d_min) {
        l_max *= 2;
    }

    // Binary search for actual range length
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta(morton, i, i + (l + t) * d, n) > d_min) {
            l += t;
        }
    }

    int j = i + l * d;
    *first = min(i, j);
    *last = max(i, j);
}

// Karras 2012: Find split position within range
__device__ int find_split(
    const unsigned long long* morton,
    int n,
    int first,
    int last
) {
    unsigned long long first_code = morton[first];
    unsigned long long last_code = morton[last];

    if (first_code == last_code) {
        return (first + last) / 2;
    }

    int common_prefix = clz64(first_code ^ last_code);

    // Binary search for split position
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) / 2;
        int new_split = split + step;

        if (new_split < last) {
            unsigned long long split_code = morton[new_split];
            int split_prefix = clz64(first_code ^ split_code);
            if (split_prefix > common_prefix) {
                split = new_split;
            }
        }
    } while (step > 1);

    return split;
}

// Build internal nodes of BVH (Karras 2012)
// n = number of leaves (particles)
// Internal nodes: 0 to n-2
// Leaf nodes: n-1 to 2n-2 (store particle indices)
extern "C" __global__ void build_bvh_internal(
    const unsigned long long* __restrict__ morton,
    int* __restrict__ left_child,   // left child index (-1 for leaf)
    int* __restrict__ right_child,  // right child index (-1 for leaf)
    int* __restrict__ parent,       // parent index
    int* __restrict__ range_left,   // leftmost leaf in subtree
    int* __restrict__ range_right,  // rightmost leaf in subtree
    int n  // number of leaves
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;  // n-1 internal nodes

    int first, last;
    determine_range(morton, n, i, &first, &last);
    int split = find_split(morton, n, first, last);

    // Children indices
    // Left child: if split == first, it's a leaf; else internal node
    int left = (split == first) ? (n - 1 + split) : split;
    // Right child: if split+1 == last, it's a leaf; else internal node
    int right = (split + 1 == last) ? (n - 1 + split + 1) : (split + 1);

    left_child[i] = left;
    right_child[i] = right;
    range_left[i] = first;
    range_right[i] = last;

    // Set parent pointers
    if (left < n - 1) parent[left] = i;
    else parent[left] = i;  // leaf parent
    if (right < n - 1) parent[right] = i;
    else parent[right] = i;  // leaf parent
}

// Initialize leaf nodes with particle data
// masses parameter added for adaptive splitting support
extern "C" __global__ void init_leaves(
    const double* __restrict__ pos,
    const int* __restrict__ signs,
    const double* __restrict__ masses,  // Per-particle masses (1.0 for uniform)
    double* __restrict__ node_data,  // 12 floats per node: cx,cy,cz,half_size,com+,com-
    int* __restrict__ node_types,
    int* __restrict__ atomic_counter,  // for bottom-up traversal
    int n,  // number of leaves
    double box_half
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int node_idx = n - 1 + i;  // Leaf nodes start at n-1
    int base = node_idx * 12;

    double x = pos[i * 3];
    double y = pos[i * 3 + 1];
    double z = pos[i * 3 + 2];
    int sign = signs[i];
    double mass = masses[i];  // Use per-particle mass

    // For leaves, center = particle position, half_size = small
    node_data[base + 0] = x;
    node_data[base + 1] = y;
    node_data[base + 2] = z;
    node_data[base + 3] = box_half / 1024.0;  // Small for leaf

    // COM for this particle (using actual mass)
    if (sign > 0) {
        node_data[base + 4] = x;  // com_plus
        node_data[base + 5] = y;
        node_data[base + 6] = z;
        node_data[base + 7] = mass;  // mass_plus (per-particle)
        node_data[base + 8] = 0.0;  // com_minus
        node_data[base + 9] = 0.0;
        node_data[base + 10] = 0.0;
        node_data[base + 11] = 0.0;  // mass_minus
    } else {
        node_data[base + 4] = 0.0;
        node_data[base + 5] = 0.0;
        node_data[base + 6] = 0.0;
        node_data[base + 7] = 0.0;
        node_data[base + 8] = x;
        node_data[base + 9] = y;
        node_data[base + 10] = z;
        node_data[base + 11] = mass;  // mass_minus (per-particle)
    }

    node_types[node_idx] = 1;  // Leaf
    atomic_counter[node_idx] = 0;
}

// Bottom-up COM reduction for internal nodes
extern "C" __global__ void reduce_com(
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ parent,
    const int* __restrict__ range_left,
    const int* __restrict__ range_right,
    double* __restrict__ node_data,
    int* __restrict__ node_types,
    int* __restrict__ atomic_counter,
    int n,  // number of leaves
    double box_half,
    int* __restrict__ diag_counters  // [0]=nodes_processed, [1]=both_mplus, [2]=boundary_cross_mplus, [3]=both_mminus, [4]=boundary_cross_mminus
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= n) return;

    int node_idx = n - 1 + leaf_idx;  // Start from this leaf

    // Walk up the tree
    int current = parent[node_idx];

    while (current >= 0 && current < n - 1) {
        // Atomically increment counter for this internal node
        int count = atomicAdd(&atomic_counter[current], 1);

        // If we're the second thread to arrive, process this node
        if (count == 0) {
            // First thread - exit and let second thread handle it
            return;
        }

        // We're the second thread - both children are ready
        int left = left_child[current];
        int right = right_child[current];

        int base = current * 12;
        int left_base = left * 12;
        int right_base = right * 12;

        // Diagnostic: count nodes processed
        if (diag_counters != NULL) atomicAdd(&diag_counters[0], 1);

        // Read children data
        double l_mp = node_data[left_base + 7];
        double l_mm = node_data[left_base + 11];
        double r_mp = node_data[right_base + 7];
        double r_mm = node_data[right_base + 11];

        // Box size for periodic unfolding
        double box_size = 2.0 * box_half;

        // Compute combined COM for positive masses with periodic unfolding
        double total_mp = l_mp + r_mp;
        double com_plus_x = 0.0, com_plus_y = 0.0, com_plus_z = 0.0;
        if (total_mp > 0.0) {
            if (l_mp > 0.0 && r_mp > 0.0) {
                // Both children have m+ : unfold right relative to left
                if (diag_counters != NULL) atomicAdd(&diag_counters[1], 1);  // both_mplus

                double left_x = node_data[left_base + 4];
                double left_y = node_data[left_base + 5];
                double left_z = node_data[left_base + 6];
                double right_x = node_data[right_base + 4];
                double right_y = node_data[right_base + 5];
                double right_z = node_data[right_base + 6];

                // Check if boundary crossing occurs (before minimum_image)
                double raw_dx = right_x - left_x;
                double raw_dy = right_y - left_y;
                double raw_dz = right_z - left_z;
                if (diag_counters != NULL && (fabs(raw_dx) > box_half || fabs(raw_dy) > box_half || fabs(raw_dz) > box_half)) {
                    atomicAdd(&diag_counters[2], 1);  // boundary_cross_mplus
                }

                double dx = minimum_image(right_x - left_x, box_size, box_half);
                double dy = minimum_image(right_y - left_y, box_size, box_half);
                double dz = minimum_image(right_z - left_z, box_size, box_half);

                double w_right = r_mp / total_mp;
                com_plus_x = left_x + w_right * dx;
                com_plus_y = left_y + w_right * dy;
                com_plus_z = left_z + w_right * dz;

                // Wrap back into [-box_half, +box_half]
                if (com_plus_x >  box_half) com_plus_x -= box_size;
                if (com_plus_x < -box_half) com_plus_x += box_size;
                if (com_plus_y >  box_half) com_plus_y -= box_size;
                if (com_plus_y < -box_half) com_plus_y += box_size;
                if (com_plus_z >  box_half) com_plus_z -= box_size;
                if (com_plus_z < -box_half) com_plus_z += box_size;
            } else if (l_mp > 0.0) {
                // Only left child has m+
                com_plus_x = node_data[left_base + 4];
                com_plus_y = node_data[left_base + 5];
                com_plus_z = node_data[left_base + 6];
            } else {
                // Only right child has m+
                com_plus_x = node_data[right_base + 4];
                com_plus_y = node_data[right_base + 5];
                com_plus_z = node_data[right_base + 6];
            }
        }

        // Compute combined COM for negative masses with periodic unfolding
        double total_mm = l_mm + r_mm;
        double com_minus_x = 0.0, com_minus_y = 0.0, com_minus_z = 0.0;
        if (total_mm > 0.0) {
            if (l_mm > 0.0 && r_mm > 0.0) {
                // Both children have m- : unfold right relative to left
                if (diag_counters != NULL) atomicAdd(&diag_counters[3], 1);  // both_mminus

                double left_x = node_data[left_base + 8];
                double left_y = node_data[left_base + 9];
                double left_z = node_data[left_base + 10];
                double right_x = node_data[right_base + 8];
                double right_y = node_data[right_base + 9];
                double right_z = node_data[right_base + 10];

                // Check if boundary crossing occurs (before minimum_image)
                double raw_dx = right_x - left_x;
                double raw_dy = right_y - left_y;
                double raw_dz = right_z - left_z;
                if (diag_counters != NULL && (fabs(raw_dx) > box_half || fabs(raw_dy) > box_half || fabs(raw_dz) > box_half)) {
                    atomicAdd(&diag_counters[4], 1);  // boundary_cross_mminus
                }

                double dx = minimum_image(right_x - left_x, box_size, box_half);
                double dy = minimum_image(right_y - left_y, box_size, box_half);
                double dz = minimum_image(right_z - left_z, box_size, box_half);

                double w_right = r_mm / total_mm;
                com_minus_x = left_x + w_right * dx;
                com_minus_y = left_y + w_right * dy;
                com_minus_z = left_z + w_right * dz;

                // Wrap back into [-box_half, +box_half]
                if (com_minus_x >  box_half) com_minus_x -= box_size;
                if (com_minus_x < -box_half) com_minus_x += box_size;
                if (com_minus_y >  box_half) com_minus_y -= box_size;
                if (com_minus_y < -box_half) com_minus_y += box_size;
                if (com_minus_z >  box_half) com_minus_z -= box_size;
                if (com_minus_z < -box_half) com_minus_z += box_size;
            } else if (l_mm > 0.0) {
                // Only left child has m-
                com_minus_x = node_data[left_base + 8];
                com_minus_y = node_data[left_base + 9];
                com_minus_z = node_data[left_base + 10];
            } else {
                // Only right child has m-
                com_minus_x = node_data[right_base + 8];
                com_minus_y = node_data[right_base + 9];
                com_minus_z = node_data[right_base + 10];
            }
        }

        // Compute bounding box half-size from range
        // For Morton BVH: half_size proportional to range^(1/3)
        // Root (range=n): half_size = box_half
        // Leaf (range=1): half_size = box_half / n^(1/3)
        int first = range_left[current];
        int last = range_right[current];
        int range_size = last - first + 1;
        // Use cbrt to get cube root (spatial extent scales as volume^(1/3))
        double frac = (double)range_size / (double)n;
        double half_size = box_half * cbrt(frac);
        if (half_size < box_half / 1024.0) half_size = box_half / 1024.0;

        // Geometric center (mass-weighted average of COMs with periodic unfolding)
        double cx = 0.0, cy = 0.0, cz = 0.0;
        if (total_mp > 0.0 && total_mm > 0.0) {
            // Both populations exist: unfold com_minus relative to com_plus
            double dx = minimum_image(com_minus_x - com_plus_x, box_size, box_half);
            double dy = minimum_image(com_minus_y - com_plus_y, box_size, box_half);
            double dz = minimum_image(com_minus_z - com_plus_z, box_size, box_half);

            double w_minus = total_mm / (total_mp + total_mm);
            cx = com_plus_x + w_minus * dx;
            cy = com_plus_y + w_minus * dy;
            cz = com_plus_z + w_minus * dz;

            // Wrap back
            if (cx >  box_half) cx -= box_size;
            if (cx < -box_half) cx += box_size;
            if (cy >  box_half) cy -= box_size;
            if (cy < -box_half) cy += box_size;
            if (cz >  box_half) cz -= box_size;
            if (cz < -box_half) cz += box_size;
        } else if (total_mp > 0.0) {
            cx = com_plus_x;
            cy = com_plus_y;
            cz = com_plus_z;
        } else if (total_mm > 0.0) {
            cx = com_minus_x;
            cy = com_minus_y;
            cz = com_minus_z;
        }

        // Store node data
        node_data[base + 0] = cx;
        node_data[base + 1] = cy;
        node_data[base + 2] = cz;
        node_data[base + 3] = half_size;
        node_data[base + 4] = com_plus_x;
        node_data[base + 5] = com_plus_y;
        node_data[base + 6] = com_plus_z;
        node_data[base + 7] = total_mp;
        node_data[base + 8] = com_minus_x;
        node_data[base + 9] = com_minus_y;
        node_data[base + 10] = com_minus_z;
        node_data[base + 11] = total_mm;

        node_types[current] = 2;  // Internal node

        // Move up to parent
        if (current == 0) break;  // Reached root
        current = parent[current];
    }
}

// Compute forces using GPU-built BVH
extern "C" __global__ void compute_forces_bvh(
    const double* __restrict__ pos,
    const int* __restrict__ signs,
    const double* __restrict__ node_data,
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ node_types,
    double* __restrict__ acc,
    int n_particles,
    double theta,
    double softening,
    double c_ratio_sq,  // VSL: (c_minus/c_plus)^2, default=1.0, VSL=100.0
    double box_size  // Box size for periodic boundary conditions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double box_half = 0.5 * box_size;
    double px = pos[tid * 3];
    double py = pos[tid * 3 + 1];
    double pz = pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    // Asymmetric softening: m- uses larger value to simulate diffuse gas nature
    double eps = (my_sign > 0) ? softening : (softening * SOFTENING_MINUS_RATIO);
    double eps2 = eps * eps;

    // Stack-based traversal starting from root
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Root is node 0

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0) continue;

        int node_type = node_types[node_idx];
        if (node_type == 0) continue;

        int base = node_idx * 12;
        double cx = node_data[base + 0];
        double cy = node_data[base + 1];
        double cz = node_data[base + 2];
        double half_size = node_data[base + 3];

        double dx = minimum_image(cx - px, box_size, box_half);
        double dy = minimum_image(cy - py, box_size, box_half);
        double dz = minimum_image(cz - pz, box_size, box_half);
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2 + 1e-20);

        double s_over_r = (2.0 * half_size) / r;

        // Use monopole if leaf or passes opening criterion
        if (node_type == 1 || s_over_r < theta) {
            double mass_plus = node_data[base + 7];
            double mass_minus = node_data[base + 11];

            // Interaction with m+ mass in this node
            if (mass_plus > 0.0) {
                double com_plus_x = node_data[base + 4];
                double com_plus_y = node_data[base + 5];
                double com_plus_z = node_data[base + 6];
                double dpx = minimum_image(com_plus_x - px, box_size, box_half);
                double dpy = minimum_image(com_plus_y - py, box_size, box_half);
                double dpz = minimum_image(com_plus_z - pz, box_size, box_half);
                double rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
                double inv_rp3 = 1.0 / (rp2 * sqrt(rp2));
                // VSL Option B: m- feels AMPLIFIED repulsion from m+ (factor c_ratio_sq)
                double interaction = (my_sign > 0) ? 1.0 : -c_ratio_sq;
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            // Interaction with m- mass in this node
            if (mass_minus > 0.0) {
                double com_minus_x = node_data[base + 8];
                double com_minus_y = node_data[base + 9];
                double com_minus_z = node_data[base + 10];
                double dmx = minimum_image(com_minus_x - px, box_size, box_half);
                double dmy = minimum_image(com_minus_y - py, box_size, box_half);
                double dmz = minimum_image(com_minus_z - pz, box_size, box_half);
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double inv_rm3 = 1.0 / (rm2 * sqrt(rm2));
                // VSL Option B: m+ feels STANDARD repulsion from m- (factor 1.0)
                double interaction = (my_sign < 0) ? 1.0 : -1.0;
                double f = interaction * mass_minus * inv_rm3;
                ax += f * dmx;
                ay += f * dmy;
                az += f * dmz;
            }
        } else {
            // Descend into children
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

// Compute forces with configurable cross-sign interaction
// cross_factor: 0.0 = no cross interaction (attraction only)
//              -1.0 = Janus (repulsion between opposite signs)
extern "C" __global__ void compute_forces_bvh_cross(
    const double* __restrict__ pos,
    const int* __restrict__ signs,
    const double* __restrict__ node_data,
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ node_types,
    double* __restrict__ acc,
    int n_particles,
    double theta,
    double softening,
    double cross_factor,
    double box_size  // Box size for periodic boundary conditions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double box_half = 0.5 * box_size;
    double px = pos[tid * 3];
    double py = pos[tid * 3 + 1];
    double pz = pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    // Asymmetric softening: m- uses larger value
    double eps = (my_sign > 0) ? softening : (softening * SOFTENING_MINUS_RATIO);
    double eps2 = eps * eps;

    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0) continue;

        int node_type = node_types[node_idx];
        if (node_type == 0) continue;

        int base = node_idx * 12;
        double cx = node_data[base + 0];
        double cy = node_data[base + 1];
        double cz = node_data[base + 2];
        double half_size = node_data[base + 3];

        double dx = minimum_image(cx - px, box_size, box_half);
        double dy = minimum_image(cy - py, box_size, box_half);
        double dz = minimum_image(cz - pz, box_size, box_half);
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2 + 1e-20);

        double s_over_r = (2.0 * half_size) / r;

        if (node_type == 1 || s_over_r < theta) {
            double mass_plus = node_data[base + 7];
            double mass_minus = node_data[base + 11];

            if (mass_plus > 0.0) {
                double com_plus_x = node_data[base + 4];
                double com_plus_y = node_data[base + 5];
                double com_plus_z = node_data[base + 6];
                double dpx = minimum_image(com_plus_x - px, box_size, box_half);
                double dpy = minimum_image(com_plus_y - py, box_size, box_half);
                double dpz = minimum_image(com_plus_z - pz, box_size, box_half);
                double rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
                double inv_rp3 = 1.0 / (rp2 * sqrt(rp2));
                // Same sign: attraction (1.0), Opposite sign: cross_factor
                double interaction = (my_sign > 0) ? 1.0 : cross_factor;
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            if (mass_minus > 0.0) {
                double com_minus_x = node_data[base + 8];
                double com_minus_y = node_data[base + 9];
                double com_minus_z = node_data[base + 10];
                double dmx = minimum_image(com_minus_x - px, box_size, box_half);
                double dmy = minimum_image(com_minus_y - py, box_size, box_half);
                double dmz = minimum_image(com_minus_z - pz, box_size, box_half);
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double inv_rm3 = 1.0 / (rm2 * sqrt(rm2));
                // Same sign: attraction (1.0), Opposite sign: cross_factor
                double interaction = (my_sign < 0) ? 1.0 : cross_factor;
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

// Yukawa-screened Janus interaction kernel
// α(r) = 1 - ε×exp(-r/r_c) for opposite-sign interactions
// At r → 0: α → 1-ε (partial attraction)
// At r → ∞: α → 1 (standard Janus repulsion)
// Note: yukawa_packed = r_c + epsilon (epsilon in fractional part, r_c in integer part)
extern "C" __global__ void compute_forces_bvh_yukawa(
    const double* __restrict__ pos,
    const int* __restrict__ signs,
    const double* __restrict__ node_data,
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ node_types,
    double* __restrict__ acc,
    int n_particles,
    double theta,
    double softening,
    double yukawa_packed,   // r_c (integer part) + epsilon (fractional part)
    double box_size   // Box size for periodic boundary conditions
) {
    // Unpack Yukawa parameters
    double r_c = floor(yukawa_packed);
    double epsilon = yukawa_packed - r_c;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    double box_half = 0.5 * box_size;
    double px = pos[tid * 3];
    double py = pos[tid * 3 + 1];
    double pz = pos[tid * 3 + 2];
    int my_sign = signs[tid];

    double ax = 0.0, ay = 0.0, az = 0.0;
    // Asymmetric softening: m- uses larger value
    double soft = (my_sign > 0) ? softening : (softening * SOFTENING_MINUS_RATIO);
    double eps2 = soft * soft;

    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        if (node_idx < 0) continue;

        int node_type = node_types[node_idx];
        if (node_type == 0) continue;

        int base = node_idx * 12;
        double cx = node_data[base + 0];
        double cy = node_data[base + 1];
        double cz = node_data[base + 2];
        double half_size = node_data[base + 3];

        double dx = minimum_image(cx - px, box_size, box_half);
        double dy = minimum_image(cy - py, box_size, box_half);
        double dz = minimum_image(cz - pz, box_size, box_half);
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2 + 1e-20);

        double s_over_r = (2.0 * half_size) / r;

        if (node_type == 1 || s_over_r < theta) {
            double mass_plus = node_data[base + 7];
            double mass_minus = node_data[base + 11];

            if (mass_plus > 0.0) {
                double com_plus_x = node_data[base + 4];
                double com_plus_y = node_data[base + 5];
                double com_plus_z = node_data[base + 6];
                double dpx = minimum_image(com_plus_x - px, box_size, box_half);
                double dpy = minimum_image(com_plus_y - py, box_size, box_half);
                double dpz = minimum_image(com_plus_z - pz, box_size, box_half);
                double rp2 = dpx*dpx + dpy*dpy + dpz*dpz + eps2;
                double rp = sqrt(rp2);
                double inv_rp3 = 1.0 / (rp2 * rp);

                double interaction;
                if (my_sign > 0) {
                    // Same sign (+/+): attraction
                    interaction = 1.0;
                } else {
                    // Opposite sign (-/+): Yukawa-screened repulsion
                    // α(r) = 1 - ε×exp(-r/r_c)
                    // interaction = -α(r)
                    double alpha_r = 1.0 - epsilon * exp(-rp / r_c);
                    interaction = -alpha_r;
                }
                double f = interaction * mass_plus * inv_rp3;
                ax += f * dpx;
                ay += f * dpy;
                az += f * dpz;
            }

            if (mass_minus > 0.0) {
                double com_minus_x = node_data[base + 8];
                double com_minus_y = node_data[base + 9];
                double com_minus_z = node_data[base + 10];
                double dmx = minimum_image(com_minus_x - px, box_size, box_half);
                double dmy = minimum_image(com_minus_y - py, box_size, box_half);
                double dmz = minimum_image(com_minus_z - pz, box_size, box_half);
                double rm2 = dmx*dmx + dmy*dmy + dmz*dmz + eps2;
                double rm = sqrt(rm2);
                double inv_rm3 = 1.0 / (rm2 * rm);

                double interaction;
                if (my_sign < 0) {
                    // Same sign (-/-): attraction
                    interaction = 1.0;
                } else {
                    // Opposite sign (+/-): Yukawa-screened repulsion
                    double alpha_r = 1.0 - epsilon * exp(-rm / r_c);
                    interaction = -alpha_r;
                }
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

// Simple GPU bitonic sort for Morton codes (for small N)
// For larger N, we use CPU sort (hybrid approach)
extern "C" __global__ void bitonic_sort_step(
    unsigned long long* __restrict__ keys,
    int* __restrict__ values,
    int j,
    int k,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ij = i ^ j;

    if (ij > i && i < n && ij < n) {
        if ((i & k) == 0) {
            // Ascending
            if (keys[i] > keys[ij]) {
                unsigned long long tmp_k = keys[i];
                keys[i] = keys[ij];
                keys[ij] = tmp_k;
                int tmp_v = values[i];
                values[i] = values[ij];
                values[ij] = tmp_v;
            }
        } else {
            // Descending
            if (keys[i] < keys[ij]) {
                unsigned long long tmp_k = keys[i];
                keys[i] = keys[ij];
                keys[ij] = tmp_k;
                int tmp_v = values[i];
                values[i] = values[ij];
                values[ij] = tmp_v;
            }
        }
    }
}

// Fast GPU reset of atomic counters for incremental tree updates
// Only resets atomic_counter (not entire tree structure)
extern "C" __global__ void reset_atomic_counters(
    int* __restrict__ atomic_counter,
    int n_total_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total_nodes) return;
    atomic_counter[tid] = 0;
}

// Fast GPU memset for i32 arrays
extern "C" __global__ void memset_i32(
    int* __restrict__ arr,
    int value,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    arr[tid] = value;
}

// Fast GPU memset for f64 arrays
extern "C" __global__ void memset_f64(
    double* __restrict__ arr,
    double value,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    arr[tid] = value;
}
"#;

/// Linear octree for GPU Barnes-Hut
pub struct LinearOctree {
    pub nodes: Vec<LinearTreeNode>,
    pub node_data: Vec<f64>,
    pub node_children: Vec<i32>,
    pub node_types: Vec<i32>,
    pub bounds: BoundingBox,
}

impl LinearOctree {
    pub fn build(particles: &[GpuParticle], box_size: f64) -> Self {
        let half = box_size / 2.0;
        let bounds = BoundingBox::new(
            Vec3::new(-half, -half, -half),
            Vec3::new(half, half, half),
        );

        let max_nodes = particles.len() * 2 + 1000;
        let mut nodes = Vec::with_capacity(max_nodes);

        nodes.push(LinearTreeNode {
            center_x: 0.0,
            center_y: 0.0,
            center_z: 0.0,
            half_size: half,
            ..Default::default()
        });

        for (idx, p) in particles.iter().enumerate() {
            Self::insert(&mut nodes, 0, idx, p, 0.0, 0.0, 0.0, half, particles);
        }

        Self::compute_com(&mut nodes, 0, particles);

        let n = nodes.len();
        let mut node_data = vec![0.0f64; n * 12];
        let mut node_children = vec![0i32; n * 8];
        let mut node_types = vec![0i32; n];

        for (i, node) in nodes.iter().enumerate() {
            let base = i * 12;
            node_data[base + 0] = node.center_x;
            node_data[base + 1] = node.center_y;
            node_data[base + 2] = node.center_z;
            node_data[base + 3] = node.half_size;
            node_data[base + 4] = node.com_plus_x;
            node_data[base + 5] = node.com_plus_y;
            node_data[base + 6] = node.com_plus_z;
            node_data[base + 7] = node.mass_plus;
            node_data[base + 8] = node.com_minus_x;
            node_data[base + 9] = node.com_minus_y;
            node_data[base + 10] = node.com_minus_z;
            node_data[base + 11] = node.mass_minus;

            let child_base = i * 8;
            for j in 0..8 {
                node_children[child_base + j] = node.children[j] as i32;
            }
            node_types[i] = node.node_type as i32;
        }

        Self { nodes, node_data, node_children, node_types, bounds }
    }

    fn insert(nodes: &mut Vec<LinearTreeNode>, node_idx: usize, particle_idx: usize,
              particle: &GpuParticle, cx: f64, cy: f64, cz: f64, half_size: f64,
              particles: &[GpuParticle]) {

        while nodes.len() <= node_idx {
            nodes.push(LinearTreeNode::default());
        }

        let node_type = nodes[node_idx].node_type;

        match node_type {
            0 => {
                nodes[node_idx].node_type = 1;
                nodes[node_idx].particle_idx = particle_idx as u32;
                nodes[node_idx].center_x = cx;
                nodes[node_idx].center_y = cy;
                nodes[node_idx].center_z = cz;
                nodes[node_idx].half_size = half_size;
            }
            1 => {
                // Converting leaf to internal node - must re-insert existing particle
                let existing_idx = nodes[node_idx].particle_idx as usize;
                let existing_p = &particles[existing_idx];

                nodes[node_idx].node_type = 2;
                nodes[node_idx].particle_idx = 0;

                let base_child = nodes.len();
                for i in 0..8 {
                    nodes.push(LinearTreeNode {
                        center_x: cx + if i & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 },
                        center_y: cy + if i & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 },
                        center_z: cz + if i & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 },
                        half_size: half_size / 2.0,
                        ..Default::default()
                    });
                    nodes[node_idx].children[i] = (base_child + i + 1) as u32;
                }

                // Re-insert the existing particle into appropriate child
                let oct_exist = Self::get_octant(existing_p.x, existing_p.y, existing_p.z, cx, cy, cz);
                let child_exist = (nodes[node_idx].children[oct_exist] - 1) as usize;
                let cx_e = cx + if oct_exist & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let cy_e = cy + if oct_exist & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let cz_e = cz + if oct_exist & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                Self::insert(nodes, child_exist, existing_idx, existing_p, cx_e, cy_e, cz_e, half_size / 2.0, particles);

                // Insert the new particle into appropriate child
                let octant = Self::get_octant(particle.x, particle.y, particle.z, cx, cy, cz);
                let child_idx = (nodes[node_idx].children[octant] - 1) as usize;
                let new_cx = cx + if octant & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let new_cy = cy + if octant & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                let new_cz = cz + if octant & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                Self::insert(nodes, child_idx, particle_idx, particle, new_cx, new_cy, new_cz, half_size / 2.0, particles);
            }
            2 => {
                let octant = Self::get_octant(particle.x, particle.y, particle.z, cx, cy, cz);
                let child = nodes[node_idx].children[octant];

                if child == 0 {
                    let new_idx = nodes.len();
                    let new_cx = cx + if octant & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cy = cy + if octant & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cz = cz + if octant & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };

                    nodes.push(LinearTreeNode {
                        center_x: new_cx,
                        center_y: new_cy,
                        center_z: new_cz,
                        half_size: half_size / 2.0,
                        ..Default::default()
                    });
                    nodes[node_idx].children[octant] = (new_idx + 1) as u32;
                    Self::insert(nodes, new_idx, particle_idx, particle, new_cx, new_cy, new_cz, half_size / 2.0, particles);
                } else {
                    let child_idx = (child - 1) as usize;
                    let new_cx = cx + if octant & 1 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cy = cy + if octant & 2 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    let new_cz = cz + if octant & 4 != 0 { half_size / 2.0 } else { -half_size / 2.0 };
                    Self::insert(nodes, child_idx, particle_idx, particle, new_cx, new_cy, new_cz, half_size / 2.0, particles);
                }
            }
            _ => {}
        }
    }

    fn get_octant(x: f64, y: f64, z: f64, cx: f64, cy: f64, cz: f64) -> usize {
        let mut oct = 0;
        if x > cx { oct |= 1; }
        if y > cy { oct |= 2; }
        if z > cz { oct |= 4; }
        oct
    }

    fn compute_com(nodes: &mut Vec<LinearTreeNode>, node_idx: usize, particles: &[GpuParticle])
        -> (f64, f64, f64, f64, f64, f64, f64, f64)
    {
        if node_idx >= nodes.len() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let node_type = nodes[node_idx].node_type;

        match node_type {
            0 => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1 => {
                let p_idx = nodes[node_idx].particle_idx as usize;
                if p_idx < particles.len() {
                    let p = &particles[p_idx];
                    if p.sign > 0 {
                        nodes[node_idx].com_plus_x = p.x;
                        nodes[node_idx].com_plus_y = p.y;
                        nodes[node_idx].com_plus_z = p.z;
                        nodes[node_idx].mass_plus = p.mass;
                        (p.x * p.mass, p.y * p.mass, p.z * p.mass, p.mass, 0.0, 0.0, 0.0, 0.0)
                    } else {
                        nodes[node_idx].com_minus_x = p.x;
                        nodes[node_idx].com_minus_y = p.y;
                        nodes[node_idx].com_minus_z = p.z;
                        nodes[node_idx].mass_minus = p.mass;
                        (0.0, 0.0, 0.0, 0.0, p.x * p.mass, p.y * p.mass, p.z * p.mass, p.mass)
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                }
            }
            2 => {
                let mut sum_px = 0.0f64;
                let mut sum_py = 0.0f64;
                let mut sum_pz = 0.0f64;
                let mut sum_mp = 0.0f64;
                let mut sum_mx = 0.0f64;
                let mut sum_my = 0.0f64;
                let mut sum_mz = 0.0f64;
                let mut sum_mm = 0.0f64;

                for i in 0..8 {
                    let child = nodes[node_idx].children[i];
                    if child > 0 {
                        let (cpx, cpy, cpz, cmp, cmx, cmy, cmz, cmm) =
                            Self::compute_com(nodes, (child - 1) as usize, particles);
                        sum_px += cpx;
                        sum_py += cpy;
                        sum_pz += cpz;
                        sum_mp += cmp;
                        sum_mx += cmx;
                        sum_my += cmy;
                        sum_mz += cmz;
                        sum_mm += cmm;
                    }
                }

                if sum_mp > 0.0 {
                    nodes[node_idx].com_plus_x = sum_px / sum_mp;
                    nodes[node_idx].com_plus_y = sum_py / sum_mp;
                    nodes[node_idx].com_plus_z = sum_pz / sum_mp;
                    nodes[node_idx].mass_plus = sum_mp;
                }
                if sum_mm > 0.0 {
                    nodes[node_idx].com_minus_x = sum_mx / sum_mm;
                    nodes[node_idx].com_minus_y = sum_my / sum_mm;
                    nodes[node_idx].com_minus_z = sum_mz / sum_mm;
                    nodes[node_idx].mass_minus = sum_mm;
                }

                (sum_px, sum_py, sum_pz, sum_mp, sum_mx, sum_my, sum_mz, sum_mm)
            }
            _ => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }
    }

    /// Compute gravitational potential at position (x,y,z) from same-sign particles only.
    /// Uses Barnes-Hut approximation with opening angle theta.
    /// Returns potential (negative for attractive interactions).
    pub fn compute_potential_same_sign(
        &self,
        x: f64, y: f64, z: f64,
        particle_sign: i32,
        particle_idx: usize,
        softening: f64,
        theta: f64,
    ) -> f64 {
        self.potential_tree_walk(0, x, y, z, particle_sign, particle_idx, softening, theta)
    }

    fn potential_tree_walk(
        &self,
        node_idx: usize,
        x: f64, y: f64, z: f64,
        particle_sign: i32,
        exclude_idx: usize,
        softening: f64,
        theta: f64,
    ) -> f64 {
        if node_idx >= self.nodes.len() {
            return 0.0;
        }

        let node = &self.nodes[node_idx];

        match node.node_type {
            0 => 0.0, // Empty node
            1 => {
                // Leaf node - single particle
                let p_idx = node.particle_idx as usize;
                if p_idx == exclude_idx {
                    return 0.0; // Skip self
                }

                // Get position and sign from node's COM
                let (com_x, com_y, com_z, mass) = if particle_sign > 0 {
                    // Looking for positive mass contribution
                    if node.mass_plus > 0.0 {
                        (node.com_plus_x, node.com_plus_y, node.com_plus_z, node.mass_plus)
                    } else {
                        return 0.0; // This leaf is a negative particle
                    }
                } else {
                    // Looking for negative mass contribution
                    if node.mass_minus > 0.0 {
                        (node.com_minus_x, node.com_minus_y, node.com_minus_z, node.mass_minus)
                    } else {
                        return 0.0; // This leaf is a positive particle
                    }
                };

                let dx = com_x - x;
                let dy = com_y - y;
                let dz = com_z - z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = (r2 + softening * softening).sqrt();

                -mass / r  // Plummer potential (negative = attractive)
            }
            2 => {
                // Internal node
                // Get same-sign mass and COM
                let (com_x, com_y, com_z, mass) = if particle_sign > 0 {
                    (node.com_plus_x, node.com_plus_y, node.com_plus_z, node.mass_plus)
                } else {
                    (node.com_minus_x, node.com_minus_y, node.com_minus_z, node.mass_minus)
                };

                if mass <= 0.0 {
                    return 0.0; // No same-sign particles in this branch
                }

                let dx = com_x - x;
                let dy = com_y - y;
                let dz = com_z - z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();

                // Barnes-Hut criterion: s/d < theta
                let s = 2.0 * node.half_size;
                if s / r < theta {
                    // Use monopole approximation
                    let r_soft = (r2 + softening * softening).sqrt();
                    -mass / r_soft
                } else {
                    // Descend into children
                    let mut potential = 0.0;
                    for i in 0..8 {
                        let child = node.children[i];
                        if child > 0 {
                            potential += self.potential_tree_walk(
                                (child - 1) as usize,
                                x, y, z,
                                particle_sign,
                                exclude_idx,
                                softening,
                                theta,
                            );
                        }
                    }
                    potential
                }
            }
            _ => 0.0,
        }
    }
}

/// Double-buffered BVH tree for async pipelining
#[cfg(feature = "cuda")]
pub struct BvhBuffer {
    left_child: CudaSlice<i32>,
    right_child: CudaSlice<i32>,
    parent: CudaSlice<i32>,
    range_left: CudaSlice<i32>,
    range_right: CudaSlice<i32>,
    node_data: CudaSlice<f64>,
    node_types: CudaSlice<i32>,
    atomic_counter: CudaSlice<i32>,
}

/// GPU Barnes-Hut simulation (f64 precision)
#[cfg(feature = "cuda")]
pub struct GpuNBodySimulation {
    device: Arc<CudaDevice>,
    pos: CudaSlice<f64>,        // Interleaved x,y,z
    vel: CudaSlice<f64>,        // Interleaved vx,vy,vz
    signs: CudaSlice<i32>,
    masses: CudaSlice<f64>,     // Per-particle masses (for adaptive splitting)
    acc: CudaSlice<f64>,        // Interleaved ax,ay,az
    node_data: CudaSlice<f64>,
    node_children: CudaSlice<i32>,
    node_types: CudaSlice<i32>,
    // Morton sorting buffers
    morton_codes: CudaSlice<u64>,
    sorted_indices: CudaSlice<i32>,
    pos_tmp: CudaSlice<f64>,
    vel_tmp: CudaSlice<f64>,
    signs_tmp: CudaSlice<i32>,
    masses_tmp: CudaSlice<f64>,
    // GPU BVH buffers (Karras 2012) - primary (backwards compat)
    bvh_left_child: CudaSlice<i32>,
    bvh_right_child: CudaSlice<i32>,
    bvh_parent: CudaSlice<i32>,
    bvh_range_left: CudaSlice<i32>,
    bvh_range_right: CudaSlice<i32>,
    bvh_node_data: CudaSlice<f64>,
    bvh_node_types: CudaSlice<i32>,
    bvh_atomic_counter: CudaSlice<i32>,
    // Double-buffered BVH for async pipeline (opt7)
    bvh_buffers: Option<[BvhBuffer; 2]>,
    current_bvh: usize,  // 0 or 1
    n_particles: usize,
    n_nodes: usize,
    theta: f64,
    softening: f64,
    softening_minus: f64,  // Asymmetric softening for m- (default = softening)
    box_size: f64,
    time: f64,
    particles_cpu: Vec<GpuParticle>,
    /// VSL c_ratio squared: (c_minus/c_plus)^2, default=1.0, VSL=100.0
    pub c_ratio_sq: f64,
    /// Relaxation mode: if true, skip inter-species forces (m+ ↔ m- = 0)
    pub relax_mode: bool,
    /// Repulsion scale: 0.0 = no cross-species, 1.0 = full Janus physics
    /// Used for gradual ramp-up during initialization
    pub repulsion_scale: f64,
}

/// Kernel names used in PTX
const KERNEL_NAMES: &[&str] = &[
    "compute_forces_simple", "leapfrog_kick_drift", "drift_only", "kick_only",
    "compute_morton_codes", "reorder_positions", "reorder_velocities", "reorder_signs", "reorder_masses",
    "build_bvh_internal", "init_leaves", "reduce_com", "compute_forces_bvh",
    "compute_forces_bvh_cross", "compute_forces_bvh_yukawa", "bitonic_sort_step", "reset_atomic_counters", "memset_i32", "memset_f64"
];

#[cfg(feature = "cuda")]
impl GpuNBodySimulation {
    /// Compile CUDA kernels ONCE and return initialized device
    /// Call this at startup, then reuse the device for all simulations
    pub fn compile_kernels() -> Result<Arc<CudaDevice>, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;
        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        device.load_ptx(ptx, "nbody", KERNEL_NAMES)?;
        Ok(device)
    }

    pub fn new(n_positive: usize, n_negative: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        device.load_ptx(ptx, "nbody", KERNEL_NAMES)?;

        let n_total = n_positive + n_negative;

        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let mut particles_cpu = Vec::with_capacity(n_total);

        let virial_velocity = ((n_total as f64) / box_size).sqrt() * 0.3;

        // f64 precision throughout
        for _ in 0..n_positive {
            particles_cpu.push(GpuParticle {
                x: (rng.random::<f64>() - 0.5) * box_size,
                y: (rng.random::<f64>() - 0.5) * box_size,
                z: (rng.random::<f64>() - 0.5) * box_size,
                vx: (rng.random::<f64>() - 0.5) * virial_velocity,
                vy: (rng.random::<f64>() - 0.5) * virial_velocity,
                vz: (rng.random::<f64>() - 0.5) * virial_velocity,
                mass: 1.0,
                sign: 1,
            });
        }

        for _ in 0..n_negative {
            particles_cpu.push(GpuParticle {
                x: (rng.random::<f64>() - 0.5) * box_size,
                y: (rng.random::<f64>() - 0.5) * box_size,
                z: (rng.random::<f64>() - 0.5) * box_size,
                vx: (rng.random::<f64>() - 0.5) * virial_velocity,
                vy: (rng.random::<f64>() - 0.5) * virial_velocity,
                vz: (rng.random::<f64>() - 0.5) * virial_velocity,
                mass: 1.0,
                sign: -1,
            });
        }

        // Interleaved position/velocity arrays (f64)
        let mut pos_data = Vec::with_capacity(n_total * 3);
        let mut vel_data = Vec::with_capacity(n_total * 3);
        let mut signs_data = Vec::with_capacity(n_total);

        for p in &particles_cpu {
            pos_data.push(p.x);
            pos_data.push(p.y);
            pos_data.push(p.z);
            vel_data.push(p.vx);
            vel_data.push(p.vy);
            vel_data.push(p.vz);
            signs_data.push(p.sign);
        }

        // Cosmological mass calculation: mass_per_particle = G × M_total / N
        // This makes forces N-independent (correct for cosmological simulations)
        let g_cosmo = 4.499e-15;  // Mpc³/(M_sun·Gyr²) — gravitational constant
        let rho_crit = 2.775e11;  // M_sun/Mpc³ — critical density
        let omega_m = 0.3;        // matter fraction
        let m_total = omega_m * rho_crit * box_size.powi(3);
        let mass_per_particle = g_cosmo * m_total / n_total as f64;
        let masses_data: Vec<f64> = vec![mass_per_particle; n_total];

        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let masses = device.htod_sync_copy(&masses_data)?;
        let acc = device.alloc_zeros::<f64>(n_total * 3)?;

        let tree = LinearOctree::build(&particles_cpu, box_size);
        let n_nodes = tree.nodes.len();
        let node_data = device.htod_sync_copy(&tree.node_data)?;
        let node_children = device.htod_sync_copy(&tree.node_children)?;
        let node_types = device.htod_sync_copy(&tree.node_types)?;

        // Morton sorting buffers
        let morton_codes = device.alloc_zeros::<u64>(n_total)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_total)?;
        let pos_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let vel_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let signs_tmp = device.alloc_zeros::<i32>(n_total)?;
        let masses_tmp = device.alloc_zeros::<f64>(n_total)?;

        // GPU BVH buffers (Karras 2012)
        // Total nodes: 2*n - 1 (n-1 internal + n leaves)
        let n_bvh_nodes = 2 * n_total - 1;
        let bvh_left_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_right_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_left = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_right = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_node_data = device.alloc_zeros::<f64>(n_bvh_nodes * 12)?;
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_atomic_counter = device.alloc_zeros::<i32>(n_bvh_nodes)?;

        let mean_sep = box_size / (n_total as f64).powf(1.0/3.0);
        let softening = 0.5 * mean_sep;

        Ok(Self {
            device,
            pos, vel, signs, masses, acc,
            node_data, node_children, node_types,
            morton_codes, sorted_indices, pos_tmp, vel_tmp, signs_tmp, masses_tmp,
            bvh_left_child, bvh_right_child, bvh_parent,
            bvh_range_left, bvh_range_right,
            bvh_node_data, bvh_node_types, bvh_atomic_counter,
            bvh_buffers: None,  // Initialized lazily for async mode
            current_bvh: 0,
            n_particles: n_total,
            n_nodes,
            theta: 2.0,  // Optimized for performance (was 0.7)
            softening,
            softening_minus: softening,  // Default: same as softening
            box_size,
            time: 0.0,
            particles_cpu,
            c_ratio_sq: 1.0,  // VSL: (c_minus/c_plus)^2, default=1.0
            relax_mode: false,
            repulsion_scale: 1.0,
        })
    }

    /// Create simulation optimized for BVH-only mode (no CPU LinearOctree)
    /// Saves ~3-4 GB VRAM by not allocating the legacy tree structures
    pub fn new_bvh_only(n_positive: usize, n_negative: usize, box_size: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        device.load_ptx(ptx, "nbody", &[
            "compute_forces_simple", "leapfrog_kick_drift", "drift_only", "kick_only",
            "compute_morton_codes", "reorder_positions", "reorder_velocities", "reorder_signs", "reorder_masses",
            "build_bvh_internal", "init_leaves", "reduce_com", "compute_forces_bvh",
            "compute_forces_bvh_cross", "compute_forces_bvh_yukawa", "bitonic_sort_step", "reset_atomic_counters", "memset_i32", "memset_f64"
        ])?;

        let n_total = n_positive + n_negative;

        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);

        // Initial velocity scale (will be adjusted by analytical virialization)
        let v_init = 1.0;

        // Generate particles with initial random velocities
        let mut pos_data = Vec::with_capacity(n_total * 3);
        let mut vel_data = Vec::with_capacity(n_total * 3);
        let mut signs_data = Vec::with_capacity(n_total);
        let mut particles_cpu = Vec::with_capacity(n_total);

        for _ in 0..n_positive {
            let x = (rng.random::<f64>() - 0.5) * box_size;
            let y = (rng.random::<f64>() - 0.5) * box_size;
            let z = (rng.random::<f64>() - 0.5) * box_size;
            let vx = (rng.random::<f64>() - 0.5) * v_init;
            let vy = (rng.random::<f64>() - 0.5) * v_init;
            let vz = (rng.random::<f64>() - 0.5) * v_init;
            pos_data.extend_from_slice(&[x, y, z]);
            vel_data.extend_from_slice(&[vx, vy, vz]);
            signs_data.push(1);
            particles_cpu.push(GpuParticle { x, y, z, vx, vy, vz, mass: 1.0, sign: 1 });
        }

        for _ in 0..n_negative {
            let x = (rng.random::<f64>() - 0.5) * box_size;
            let y = (rng.random::<f64>() - 0.5) * box_size;
            let z = (rng.random::<f64>() - 0.5) * box_size;
            let vx = (rng.random::<f64>() - 0.5) * v_init;
            let vy = (rng.random::<f64>() - 0.5) * v_init;
            let vz = (rng.random::<f64>() - 0.5) * v_init;
            pos_data.extend_from_slice(&[x, y, z]);
            vel_data.extend_from_slice(&[vx, vy, vz]);
            signs_data.push(-1);
            particles_cpu.push(GpuParticle { x, y, z, vx, vy, vz, mass: 1.0, sign: -1 });
        }

        // Analytical virialization (mean-field approximation) - O(1) instead of O(N²)
        // PE_binding ≈ -G * m² * N_same * (N_same - 1) / (2 * mean_separation)
        // where mean_separation ≈ 0.554 * L / N^(1/3) for uniform distribution
        let mass = 1.0;
        let g_code = 1.0;  // G in code units
        let mean_sep_plus = 0.554 * box_size / (n_positive as f64).cbrt();
        let mean_sep_minus = 0.554 * box_size / (n_negative as f64).cbrt();

        let pe_plus = -g_code * mass * mass * (n_positive * (n_positive.saturating_sub(1)) / 2) as f64 / mean_sep_plus;
        let pe_minus = -g_code * mass * mass * (n_negative * (n_negative.saturating_sub(1)) / 2) as f64 / mean_sep_minus;
        let pe_binding = pe_plus + pe_minus;

        // Current KE = 0.5 * m * sum(v²)
        let ke_current: f64 = vel_data.chunks(3)
            .map(|v| 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
            .sum();

        // Virial: 2*KE + PE_binding = 0 → KE_target = |PE_binding| / 2
        let ke_target = pe_binding.abs() / 2.0;
        let alpha = if ke_current > 1e-20 { (ke_target / ke_current).sqrt() } else { 1.0 };

        // Scale velocities
        for v in vel_data.iter_mut() {
            *v *= alpha;
        }
        for p in particles_cpu.iter_mut() {
            p.vx *= alpha;
            p.vy *= alpha;
            p.vz *= alpha;
        }

        println!("Analytical virialization (mean-field):");
        println!("  PE_binding = {:.4e}", pe_binding);
        println!("  KE_initial = {:.4e}", ke_current);
        println!("  KE_target  = {:.4e}", ke_target);
        println!("  alpha      = {:.6}", alpha);

        // Cosmological mass calculation: mass_per_particle = G × M_total / N
        // This makes forces N-independent (correct for cosmological simulations)
        let g_cosmo = 4.499e-15;  // Mpc³/(M_sun·Gyr²) — gravitational constant
        let rho_crit = 2.775e11;  // M_sun/Mpc³ — critical density
        let omega_m = 0.3;        // matter fraction
        let m_total = omega_m * rho_crit * box_size.powi(3);
        let mass_per_particle = g_cosmo * m_total / n_total as f64;
        let masses_data: Vec<f64> = vec![mass_per_particle; n_total];

        let pos = device.htod_sync_copy(&pos_data)?;
        let vel = device.htod_sync_copy(&vel_data)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let masses = device.htod_sync_copy(&masses_data)?;
        let acc = device.alloc_zeros::<f64>(n_total * 3)?;

        // SKIP LinearOctree - use minimal placeholder buffers instead
        // This saves ~3-4 GB for large N
        let node_data = device.alloc_zeros::<f64>(1)?;  // Minimal placeholder
        let node_children = device.alloc_zeros::<i32>(1)?;
        let node_types = device.alloc_zeros::<i32>(1)?;
        let n_nodes = 1;

        // Morton sorting buffers
        let morton_codes = device.alloc_zeros::<u64>(n_total)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_total)?;
        let pos_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let vel_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let signs_tmp = device.alloc_zeros::<i32>(n_total)?;
        let masses_tmp = device.alloc_zeros::<f64>(n_total)?;

        // GPU BVH buffers (Karras 2012)
        let n_bvh_nodes = 2 * n_total - 1;
        let bvh_left_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_right_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_left = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_right = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_node_data = device.alloc_zeros::<f64>(n_bvh_nodes * 12)?;
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_atomic_counter = device.alloc_zeros::<i32>(n_bvh_nodes)?;

        let mean_sep = box_size / (n_total as f64).powf(1.0/3.0);
        let softening = 0.5 * mean_sep;

        Ok(Self {
            device,
            pos, vel, signs, masses, acc,
            node_data, node_children, node_types,
            morton_codes, sorted_indices, pos_tmp, vel_tmp, signs_tmp, masses_tmp,
            bvh_left_child, bvh_right_child, bvh_parent,
            bvh_range_left, bvh_range_right,
            bvh_node_data, bvh_node_types, bvh_atomic_counter,
            bvh_buffers: None,
            current_bvh: 0,
            n_particles: n_total,
            n_nodes,
            theta: 2.0,
            softening,
            softening_minus: softening,  // Default: same as softening
            box_size,
            time: 0.0,
            particles_cpu,
            c_ratio_sq: 1.0,  // VSL: (c_minus/c_plus)^2, default=1.0
            relax_mode: false,
            repulsion_scale: 1.0,
        })
    }

    /// Create simulation with specific initial state (for comparison tests)
    /// OPTIMIZED: Skips CPU LinearOctree build (saves ~4GB RAM for 10M particles)
    pub fn new_with_state(
        n_positive: usize, n_negative: usize, box_size: f64,
        positions: Vec<f64>, velocities: Vec<f64>, signs_data: Vec<i32>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        device.load_ptx(ptx, "nbody", &[
            "compute_forces_simple", "leapfrog_kick_drift", "drift_only", "kick_only",
            "compute_morton_codes", "reorder_positions", "reorder_velocities", "reorder_signs", "reorder_masses",
            "build_bvh_internal", "init_leaves", "reduce_com", "compute_forces_bvh",
            "compute_forces_bvh_cross", "compute_forces_bvh_yukawa", "bitonic_sort_step", "reset_atomic_counters", "memset_i32", "memset_f64"
        ])?;

        let n_total = n_positive + n_negative;

        // SKIP particles_cpu - we use GPU BVH only, no need to keep CPU copy
        let particles_cpu = Vec::new();

        // Cosmological mass calculation: mass_per_particle = G × M_total / N
        // This makes forces N-independent (correct for cosmological simulations)
        let g_cosmo = 4.499e-15;  // Mpc³/(M_sun·Gyr²) — gravitational constant
        let rho_crit = 2.775e11;  // M_sun/Mpc³ — critical density
        let omega_m = 0.3;        // matter fraction
        let m_total = omega_m * rho_crit * box_size.powi(3);
        let mass_per_particle = g_cosmo * m_total / n_total as f64;
        let masses_data: Vec<f64> = vec![mass_per_particle; n_total];

        let pos = device.htod_sync_copy(&positions)?;
        let vel = device.htod_sync_copy(&velocities)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let masses = device.htod_sync_copy(&masses_data)?;
        let acc = device.alloc_zeros::<f64>(n_total * 3)?;

        // SKIP CPU LinearOctree - we use GPU BVH only (saves ~3-4 GB RAM for 10M particles)
        // Allocate minimal dummy data for legacy node_data/node_children/node_types fields
        let n_nodes = 1;  // Minimal placeholder
        let node_data = device.alloc_zeros::<f64>(12)?;  // 1 node × 12 f64
        let node_children = device.alloc_zeros::<i32>(8)?;  // 1 node × 8 children
        let node_types = device.alloc_zeros::<i32>(1)?;  // 1 node

        // Morton sorting buffers
        let morton_codes = device.alloc_zeros::<u64>(n_total)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_total)?;
        let pos_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let vel_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let signs_tmp = device.alloc_zeros::<i32>(n_total)?;
        let masses_tmp = device.alloc_zeros::<f64>(n_total)?;

        // GPU BVH buffers (Karras 2012)
        let n_bvh_nodes = 2 * n_total - 1;
        let bvh_left_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_right_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_left = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_right = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_node_data = device.alloc_zeros::<f64>(n_bvh_nodes * 12)?;
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_atomic_counter = device.alloc_zeros::<i32>(n_bvh_nodes)?;

        let mean_sep = box_size / (n_total as f64).powf(1.0/3.0);
        let softening = 0.5 * mean_sep;

        Ok(Self {
            device,
            pos, vel, signs, masses, acc,
            node_data, node_children, node_types,
            morton_codes, sorted_indices, pos_tmp, vel_tmp, signs_tmp, masses_tmp,
            bvh_left_child, bvh_right_child, bvh_parent,
            bvh_range_left, bvh_range_right,
            bvh_node_data, bvh_node_types, bvh_atomic_counter,
            bvh_buffers: None,
            current_bvh: 0,
            n_particles: n_total,
            n_nodes,
            theta: 2.0,  // Optimized for performance (was 0.7)
            softening,
            softening_minus: softening,  // Default: same as softening
            box_size,
            time: 0.0,
            particles_cpu,
            c_ratio_sq: 1.0,  // VSL: (c_minus/c_plus)^2, default=1.0
            relax_mode: false,
            repulsion_scale: 1.0,
        })
    }

    /// Create simulation with specific initial state AND per-particle masses
    /// This is the primary constructor for adaptive splitting simulations
    /// OPTIMIZED: Skips CPU LinearOctree build (saves ~4GB RAM for 10M particles)
    pub fn new_with_state_and_masses(
        n_positive: usize, n_negative: usize, box_size: f64,
        positions: Vec<f64>, velocities: Vec<f64>, signs_data: Vec<i32>,
        masses_data: Vec<f64>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Self::compile_kernels()?;
        Self::new_with_state_and_masses_with_device(
            device, n_positive, n_negative, box_size,
            positions, velocities, signs_data, masses_data
        )
    }

    /// Create simulation with pre-compiled device (FAST - no PTX recompilation)
    /// Use compile_kernels() once at startup, then pass the device here for each split
    pub fn new_with_state_and_masses_with_device(
        device: Arc<CudaDevice>,
        n_positive: usize, n_negative: usize, box_size: f64,
        positions: Vec<f64>, velocities: Vec<f64>, signs_data: Vec<i32>,
        masses_data: Vec<f64>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let n_total = n_positive + n_negative;

        // SKIP particles_cpu - we use GPU BVH only, no need to keep CPU copy
        let particles_cpu = Vec::new();

        let pos = device.htod_sync_copy(&positions)?;
        let vel = device.htod_sync_copy(&velocities)?;
        let signs = device.htod_sync_copy(&signs_data)?;
        let masses = device.htod_sync_copy(&masses_data)?;
        let acc = device.alloc_zeros::<f64>(n_total * 3)?;

        // SKIP CPU LinearOctree - we use GPU BVH only (saves ~3-4 GB RAM for 10M particles)
        // Allocate minimal dummy data for legacy node_data/node_children/node_types fields
        let n_nodes = 1;  // Minimal placeholder
        let node_data = device.alloc_zeros::<f64>(12)?;  // 1 node × 12 f64
        let node_children = device.alloc_zeros::<i32>(8)?;  // 1 node × 8 children
        let node_types = device.alloc_zeros::<i32>(1)?;  // 1 node

        // Morton sorting buffers
        let morton_codes = device.alloc_zeros::<u64>(n_total)?;
        let sorted_indices = device.alloc_zeros::<i32>(n_total)?;
        let pos_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let vel_tmp = device.alloc_zeros::<f64>(n_total * 3)?;
        let signs_tmp = device.alloc_zeros::<i32>(n_total)?;
        let masses_tmp = device.alloc_zeros::<f64>(n_total)?;

        // GPU BVH buffers (Karras 2012)
        let n_bvh_nodes = 2 * n_total - 1;
        let bvh_left_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_right_child = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_parent = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_left = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_range_right = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_node_data = device.alloc_zeros::<f64>(n_bvh_nodes * 12)?;
        let bvh_node_types = device.alloc_zeros::<i32>(n_bvh_nodes)?;
        let bvh_atomic_counter = device.alloc_zeros::<i32>(n_bvh_nodes)?;

        let mean_sep = box_size / (n_total as f64).powf(1.0/3.0);
        let softening = 0.5 * mean_sep;

        Ok(Self {
            device,
            pos, vel, signs, masses, acc,
            node_data, node_children, node_types,
            morton_codes, sorted_indices, pos_tmp, vel_tmp, signs_tmp, masses_tmp,
            bvh_left_child, bvh_right_child, bvh_parent,
            bvh_range_left, bvh_range_right,
            bvh_node_data, bvh_node_types, bvh_atomic_counter,
            bvh_buffers: None,
            current_bvh: 0,
            n_particles: n_total,
            n_nodes,
            theta: 2.0,  // Optimized for performance (was 0.7)
            softening,
            softening_minus: softening,  // Default: same as softening
            box_size,
            time: 0.0,
            particles_cpu,
            c_ratio_sq: 1.0,  // VSL: (c_minus/c_plus)^2, default=1.0
            relax_mode: false,
            repulsion_scale: 1.0,
        })
    }

    /// Get current positions (for comparison tests)
    pub fn positions(&self) -> Vec<f64> {
        self.device.dtoh_sync_copy(&self.pos).unwrap_or_default()
    }

    /// Get current velocities (for comparison tests)
    pub fn velocities(&self) -> Vec<f64> {
        self.device.dtoh_sync_copy(&self.vel).unwrap_or_default()
    }

    /// Get particle signs (for comparison tests)
    pub fn signs(&self) -> Vec<i32> {
        self.device.dtoh_sync_copy(&self.signs).unwrap_or_default()
    }

    /// Set Barnes-Hut opening criterion theta (default 0.7)
    /// Higher theta = more approximate but faster
    /// theta=0.5: accurate, theta=1.0: fast
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
    }

    /// Set softening length (default: 0.5 * mean_sep)
    pub fn set_softening(&mut self, softening: f64) {
        self.softening = softening;
    }

    /// Scale all particle masses by a factor (for Janus physics correction)
    /// Call after construction: sim.set_mass_factor(omega_b * (1.0 + mu) / 0.3)
    pub fn set_mass_factor(&mut self, factor: f64) {
        // Download masses from GPU
        let mut masses_host = self.device.dtoh_sync_copy(&self.masses).unwrap();

        // Scale all masses
        for m in masses_host.iter_mut() {
            *m *= factor;
        }

        // Upload back to GPU
        self.masses = self.device.htod_sync_copy(&masses_host).unwrap();

        println!("  [MASS] Scaled all masses by factor {:.4}", factor);
    }

    /// Get current softening length
    pub fn get_softening(&self) -> f64 {
        self.softening
    }

    /// Set asymmetric softening for m- particles
    /// Numerical artifact to simulate diffuse gas nature of m- without SPH
    pub fn set_softening_minus(&mut self, softening_minus: f64) {
        self.softening_minus = softening_minus;
    }

    /// Set VSL c_ratio (c_minus/c_plus)
    /// Default c_ratio=1.0 (standard Janus)
    /// VSL: c_ratio=10 → (c⁻/c⁺)²=100
    /// Effect: m+ feels 100× stronger repulsion from m-
    ///         m- feels 100× weaker repulsion from m+
    pub fn set_c_ratio(&mut self, c_ratio: f64) {
        self.c_ratio_sq = c_ratio * c_ratio;
    }

    /// Get current theta value
    pub fn get_theta(&self) -> f64 {
        self.theta
    }

    /// Get current c_ratio squared
    pub fn get_c_ratio_sq(&self) -> f64 {
        self.c_ratio_sq
    }

    /// Set relaxation mode (intra-species only, no m+ ↔ m- forces)
    /// Also sets repulsion_scale: 0.0 for relax=true, 1.0 for relax=false
    pub fn set_relax_mode(&mut self, relax: bool) {
        self.relax_mode = relax;
        self.repulsion_scale = if relax { 0.0 } else { 1.0 };
    }

    /// Get current relaxation mode
    pub fn get_relax_mode(&self) -> bool {
        self.relax_mode
    }

    /// Set repulsion scale for gradual ramp-up (0.0 = no repulsion, 1.0 = full Janus)
    /// Use this to smoothly transition from relaxation to production
    pub fn set_repulsion_scale(&mut self, scale: f64) {
        self.repulsion_scale = scale.clamp(0.0, 1.0);
    }

    /// Get current repulsion scale
    pub fn get_repulsion_scale(&self) -> f64 {
        self.repulsion_scale
    }

    /// Step sans expansion cosmologique (a=1, H=0)
    pub fn step(&mut self, dt: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.step_with_expansion(dt, 1.0, 0.0, 0.0)
    }

    /// Step avec expansion cosmologique
    /// scale_factor: a(t) facteur d'echelle
    /// hubble: H(t) = adot/a parametre de Hubble
    /// dtau_per_dt: facteur de conversion temps cosmo/temps N-corps (constant)
    pub fn step_with_expansion(&mut self, dt: f64, scale_factor: f64, hubble: f64, dtau_per_dt: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let half_dt = dt * 0.5;
        // dtau_per_dt passe directement (constant = 0.013205)
        let box_half = self.box_size / 2.0;

        // Download positions for tree rebuild
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;

        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
        }

        // Rebuild tree
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        let blocks = (self.n_particles + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernel functions
        let compute_forces = self.device.get_func("nbody", "compute_forces_simple")
            .ok_or("Failed to get compute_forces_simple kernel")?;
        let leapfrog = self.device.get_func("nbody", "leapfrog_kick_drift")
            .ok_or("Failed to get leapfrog_kick_drift kernel")?;

        // Compute forces (12 args with VSL c_ratio_sq + repulsion_scale + box_size)
        unsafe {
            compute_forces.clone().launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                self.n_particles as i32,
                self.theta, self.softening, self.c_ratio_sq, self.repulsion_scale,
                self.box_size,  // Periodic boundary conditions
            ))?;
        }

        // Kick + Drift avec parametres cosmologiques
        unsafe {
            leapfrog.clone().launch(cfg, (
                &mut self.pos, &mut self.vel, &self.acc,
                half_dt, dt, box_half,
                self.n_particles as i32, 1i32, // do_drift = 1
                scale_factor, hubble, dtau_per_dt,
            ))?;
        }

        // Download new positions for second force calc
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
        }

        // Rebuild tree
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        // Compute forces again
        unsafe {
            compute_forces.clone().launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                self.n_particles as i32, self.n_nodes as i32,
                self.theta, self.softening, self.c_ratio_sq, self.repulsion_scale,
            ))?;
        }

        // Kick only (no drift) avec parametres cosmologiques
        unsafe {
            leapfrog.launch(cfg, (
                &mut self.pos, &mut self.vel, &self.acc,
                half_dt, 0.0f64, box_half,
                self.n_particles as i32, 0i32, // do_drift = 0
                scale_factor, hubble, dtau_per_dt,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// DKD integrator: Drift-Kick-Drift (1 force calculation per step)
    /// More efficient than KDK for expensive force calculations.
    /// Structure: Drift(dt/2) → Force → Kick(dt) → Drift(dt/2)
    pub fn step_with_expansion_dkd(&mut self, dt: f64, scale_factor: f64, hubble: f64, dtau_per_dt: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;

        let blocks = (self.n_particles + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernel functions
        let compute_forces = self.device.get_func("nbody", "compute_forces_simple")
            .ok_or("Failed to get compute_forces_simple kernel")?;
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only kernel")?;

        // Step 1: Drift(dt/2) - move particles by half timestep
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                self.n_particles as i32,
            ))?;
        }

        // Step 2: Download positions for tree rebuild (only once per step!)
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
        }

        // Step 3: Rebuild tree (only once per step!)
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        // Step 4: Compute forces (only once per step!)
        unsafe {
            compute_forces.launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                self.n_particles as i32,
                self.theta, self.softening, self.c_ratio_sq, self.repulsion_scale,
                self.box_size,  // Periodic boundary conditions
            ))?;
        }

        // Step 5: Kick(dt) - full timestep velocity update with Hubble friction
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, self.n_particles as i32,
                hubble, dtau_per_dt,
            ))?;
        }

        // Step 6: Drift(dt/2) - move particles by half timestep again
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                self.n_particles as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    pub fn kinetic_energy(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let vel = self.device.dtoh_sync_copy(&self.vel)?;

        let ke: f64 = (0..self.n_particles)
            .map(|i| {
                let vx = vel[i * 3];
                let vy = vel[i * 3 + 1];
                let vz = vel[i * 3 + 2];
                0.5 * (vx * vx + vy * vy + vz * vz)
            })
            .sum();

        Ok(ke)
    }

    /// Morton sort particles for improved cache locality
    /// Uses Z-order curve to map 3D positions to 1D Morton codes
    pub fn morton_sort_particles(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let blocks = (self.n_particles + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let box_half = self.box_size / 2.0;
        // Scale factor to map positions to 21-bit integer range (2^21 = 2097152)
        let inv_cell_size = 2097152.0 / self.box_size;

        // Step 1: Compute Morton codes on GPU
        let morton_kernel = self.device.get_func("nbody", "compute_morton_codes")
            .ok_or("Failed to get compute_morton_codes kernel")?;

        unsafe {
            morton_kernel.launch(cfg, (
                &self.pos,
                &mut self.morton_codes,
                &mut self.sorted_indices,
                self.n_particles as i32,
                box_half,
                inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Step 2: Download Morton codes and indices to CPU for sorting
        let morton_cpu = self.device.dtoh_sync_copy(&self.morton_codes)?;
        let mut indices_cpu: Vec<i32> = (0..self.n_particles as i32).collect();

        // Step 3: Sort indices by Morton code using parallel sort
        // Create pairs and sort
        use rayon::prelude::*;
        let mut pairs: Vec<(u64, i32)> = morton_cpu.iter()
            .zip(indices_cpu.iter())
            .map(|(&m, &i)| (m, i))
            .collect();

        pairs.par_sort_unstable_by_key(|&(m, _)| m);

        // Extract sorted indices
        indices_cpu = pairs.iter().map(|&(_, i)| i).collect();

        // Step 4: Upload sorted indices to GPU
        self.sorted_indices = self.device.htod_sync_copy(&indices_cpu)?;

        // Step 5: Reorder positions
        let reorder_pos = self.device.get_func("nbody", "reorder_positions")
            .ok_or("Failed to get reorder_positions kernel")?;
        unsafe {
            reorder_pos.launch(cfg, (
                &self.pos,
                &mut self.pos_tmp,
                &self.sorted_indices,
                self.n_particles as i32,
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_tmp);

        // Step 6: Reorder velocities
        let reorder_vel = self.device.get_func("nbody", "reorder_velocities")
            .ok_or("Failed to get reorder_velocities kernel")?;
        unsafe {
            reorder_vel.launch(cfg, (
                &self.vel,
                &mut self.vel_tmp,
                &self.sorted_indices,
                self.n_particles as i32,
            ))?;
        }
        std::mem::swap(&mut self.vel, &mut self.vel_tmp);

        // Step 7: Reorder signs
        let reorder_signs = self.device.get_func("nbody", "reorder_signs")
            .ok_or("Failed to get reorder_signs kernel")?;
        unsafe {
            reorder_signs.launch(cfg, (
                &self.signs,
                &mut self.signs_tmp,
                &self.sorted_indices,
                self.n_particles as i32,
            ))?;
        }
        std::mem::swap(&mut self.signs, &mut self.signs_tmp);

        self.device.synchronize()?;

        // Update CPU particle array to match GPU ordering
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        let vel_cpu = self.device.dtoh_sync_copy(&self.vel)?;
        let signs_cpu = self.device.dtoh_sync_copy(&self.signs)?;

        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
            p.vx = vel_cpu[i * 3];
            p.vy = vel_cpu[i * 3 + 1];
            p.vz = vel_cpu[i * 3 + 2];
            p.sign = signs_cpu[i];
        }

        Ok(())
    }

    /// Build BVH entirely on GPU using Karras 2012 algorithm
    /// Pipeline: Morton codes → Sort → Build BVH → COM reduction
    pub fn build_gpu_tree(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;
        let n_internal = n - 1;
        let n_total_nodes = 2 * n - 1;
        let box_half = self.box_size / 2.0;
        let inv_cell_size = 2097152.0 / self.box_size;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Compute Morton codes
        let morton_kernel = self.device.get_func("nbody", "compute_morton_codes")
            .ok_or("Failed to get compute_morton_codes kernel")?;
        unsafe {
            morton_kernel.launch(cfg, (
                &self.pos,
                &mut self.morton_codes,
                &mut self.sorted_indices,
                n as i32,
                box_half,
                inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Step 2: Sort Morton codes (CPU parallel sort for now)
        // TODO: Replace with CUB RadixSort for full GPU pipeline
        let morton_cpu = self.device.dtoh_sync_copy(&self.morton_codes)?;
        let mut indices_cpu: Vec<i32> = (0..n as i32).collect();

        use rayon::prelude::*;
        let mut pairs: Vec<(u64, i32)> = morton_cpu.iter()
            .zip(indices_cpu.iter())
            .map(|(&m, &i)| (m, i))
            .collect();
        pairs.par_sort_unstable_by_key(|&(m, _)| m);
        indices_cpu = pairs.iter().map(|&(_, i)| i).collect();

        self.sorted_indices = self.device.htod_sync_copy(&indices_cpu)?;
        // Update morton_codes to sorted order
        let sorted_morton: Vec<u64> = pairs.iter().map(|&(m, _)| m).collect();
        self.morton_codes = self.device.htod_sync_copy(&sorted_morton)?;

        // Step 3: Reorder particles based on sorted indices
        let reorder_pos = self.device.get_func("nbody", "reorder_positions")
            .ok_or("Failed to get reorder_positions kernel")?;
        unsafe {
            reorder_pos.launch(cfg, (
                &self.pos,
                &mut self.pos_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_tmp);

        let reorder_signs = self.device.get_func("nbody", "reorder_signs")
            .ok_or("Failed to get reorder_signs kernel")?;
        unsafe {
            reorder_signs.launch(cfg, (
                &self.signs,
                &mut self.signs_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.signs, &mut self.signs_tmp);

        let reorder_vel = self.device.get_func("nbody", "reorder_velocities")
            .ok_or("Failed to get reorder_velocities kernel")?;
        unsafe {
            reorder_vel.launch(cfg, (
                &self.vel,
                &mut self.vel_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.vel, &mut self.vel_tmp);

        // Reorder masses (for adaptive splitting support)
        let reorder_masses = self.device.get_func("nbody", "reorder_masses")
            .ok_or("Failed to get reorder_masses kernel")?;
        unsafe {
            reorder_masses.launch(cfg, (
                &self.masses,
                &mut self.masses_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.masses, &mut self.masses_tmp);

        self.device.synchronize()?;

        // Step 4: Reset ONLY atomic_counter (other buffers are fully overwritten)
        // Karras BVH: writes left_child, right_child, parent, range for all internal nodes
        // init_leaves + reduce_com: writes node_data and node_types for all nodes
        let reset_blocks = (n_total_nodes + 255) / 256;
        let reset_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (reset_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let reset_kernel = self.device.get_func("nbody", "reset_atomic_counters")
            .ok_or("Failed to get reset_atomic_counters kernel")?;
        unsafe {
            reset_kernel.launch(reset_cfg, (
                &mut self.bvh_atomic_counter,
                n_total_nodes as i32,
            ))?;
        }
        self.device.synchronize()?;

        // Step 5: Build internal nodes (Karras algorithm)
        let internal_blocks = (n_internal + 255) / 256;
        let internal_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (internal_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let build_kernel = self.device.get_func("nbody", "build_bvh_internal")
            .ok_or("Failed to get build_bvh_internal kernel")?;
        unsafe {
            build_kernel.launch(internal_cfg, (
                &self.morton_codes,
                &mut self.bvh_left_child,
                &mut self.bvh_right_child,
                &mut self.bvh_parent,
                &mut self.bvh_range_left,
                &mut self.bvh_range_right,
                n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // Step 6: Initialize leaves with particle data (including per-particle masses)
        let init_leaves = self.device.get_func("nbody", "init_leaves")
            .ok_or("Failed to get init_leaves kernel")?;
        unsafe {
            init_leaves.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.masses,  // Per-particle masses for adaptive splitting
                &mut self.bvh_node_data,
                &mut self.bvh_node_types,
                &mut self.bvh_atomic_counter,
                n as i32,
                box_half,
            ))?;
        }
        self.device.synchronize()?;

        // Step 7: Bottom-up COM reduction with diagnostics
        let reduce_kernel = self.device.get_func("nbody", "reduce_com")
            .ok_or("Failed to get reduce_com kernel")?;

        // Allocate diagnostic counters: [nodes_processed, both_mplus, boundary_cross_mplus, both_mminus, boundary_cross_mminus]
        let mut diag_counters = self.device.alloc_zeros::<i32>(5)?;

        unsafe {
            reduce_kernel.launch(cfg, (
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_parent,
                &self.bvh_range_left,
                &self.bvh_range_right,
                &mut self.bvh_node_data,
                &mut self.bvh_node_types,
                &mut self.bvh_atomic_counter,
                n as i32,
                box_half,
                &mut diag_counters,
            ))?;
        }
        self.device.synchronize()?;

        // Read back and print diagnostics (only on first few calls)
        static DIAG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count < 3 {
            let diag = self.device.dtoh_sync_copy(&diag_counters)?;
            println!("[DIAG reduce_com #{}] nodes={}, both_m+={}, boundary_m+={}, both_m-={}, boundary_m-={}",
                count, diag[0], diag[1], diag[2], diag[3], diag[4]);
        }

        Ok(())
    }

    /// Profiled version of build_gpu_tree for timing analysis
    pub fn build_gpu_tree_profiled(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        let n = self.n_particles;
        let n_internal = n - 1;
        let n_total_nodes = 2 * n - 1;
        let box_half = self.box_size / 2.0;
        let inv_cell_size = 2097152.0 / self.box_size;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Compute Morton codes
        let t1 = Instant::now();
        let morton_kernel = self.device.get_func("nbody", "compute_morton_codes")
            .ok_or("Failed to get compute_morton_codes kernel")?;
        unsafe {
            morton_kernel.launch(cfg, (
                &self.pos,
                &mut self.morton_codes,
                &mut self.sorted_indices,
                n as i32,
                box_half,
                inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;
        let t1_elapsed = t1.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Sort Morton codes (CPU)
        let t2 = Instant::now();
        let morton_cpu = self.device.dtoh_sync_copy(&self.morton_codes)?;
        let t2a = t2.elapsed().as_secs_f64() * 1000.0;

        let t2b_start = Instant::now();
        let mut indices_cpu: Vec<i32> = (0..n as i32).collect();
        use rayon::prelude::*;
        let mut pairs: Vec<(u64, i32)> = morton_cpu.iter()
            .zip(indices_cpu.iter())
            .map(|(&m, &i)| (m, i))
            .collect();
        pairs.par_sort_unstable_by_key(|&(m, _)| m);
        indices_cpu = pairs.iter().map(|&(_, i)| i).collect();
        let sorted_morton: Vec<u64> = pairs.iter().map(|&(m, _)| m).collect();
        let t2b = t2b_start.elapsed().as_secs_f64() * 1000.0;

        let t2c_start = Instant::now();
        self.sorted_indices = self.device.htod_sync_copy(&indices_cpu)?;
        self.morton_codes = self.device.htod_sync_copy(&sorted_morton)?;
        let t2c = t2c_start.elapsed().as_secs_f64() * 1000.0;
        let t2_elapsed = t2.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Reorder particles
        let t3 = Instant::now();
        let reorder_pos = self.device.get_func("nbody", "reorder_positions")
            .ok_or("Failed to get reorder_positions kernel")?;
        unsafe {
            reorder_pos.launch(cfg, (
                &self.pos,
                &mut self.pos_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_tmp);

        let reorder_signs = self.device.get_func("nbody", "reorder_signs")
            .ok_or("Failed to get reorder_signs kernel")?;
        unsafe {
            reorder_signs.launch(cfg, (
                &self.signs,
                &mut self.signs_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.signs, &mut self.signs_tmp);

        let reorder_vel = self.device.get_func("nbody", "reorder_velocities")
            .ok_or("Failed to get reorder_velocities kernel")?;
        unsafe {
            reorder_vel.launch(cfg, (
                &self.vel,
                &mut self.vel_tmp,
                &self.sorted_indices,
                n as i32,
            ))?;
        }
        std::mem::swap(&mut self.vel, &mut self.vel_tmp);
        self.device.synchronize()?;
        let t3_elapsed = t3.elapsed().as_secs_f64() * 1000.0;

        // Step 4: Reset buffers
        let t4 = Instant::now();
        let zeros_i32: Vec<i32> = vec![0; n_total_nodes];
        let neg_ones_i32: Vec<i32> = vec![-1; n_total_nodes];
        let zeros_f64: Vec<f64> = vec![0.0; n_total_nodes * 12];
        self.device.htod_sync_copy_into(&zeros_i32, &mut self.bvh_atomic_counter)?;
        self.device.htod_sync_copy_into(&zeros_i32, &mut self.bvh_node_types)?;
        self.device.htod_sync_copy_into(&neg_ones_i32, &mut self.bvh_left_child)?;
        self.device.htod_sync_copy_into(&neg_ones_i32, &mut self.bvh_right_child)?;
        self.device.htod_sync_copy_into(&neg_ones_i32, &mut self.bvh_parent)?;
        self.device.htod_sync_copy_into(&zeros_i32, &mut self.bvh_range_left)?;
        self.device.htod_sync_copy_into(&zeros_i32, &mut self.bvh_range_right)?;
        self.device.htod_sync_copy_into(&zeros_f64, &mut self.bvh_node_data)?;
        let t4_elapsed = t4.elapsed().as_secs_f64() * 1000.0;

        // Step 5: Build Karras BVH
        let t5 = Instant::now();
        let internal_blocks = (n_internal + 255) / 256;
        let internal_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (internal_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let build_kernel = self.device.get_func("nbody", "build_bvh_internal")
            .ok_or("Failed to get build_bvh_internal kernel")?;
        unsafe {
            build_kernel.launch(internal_cfg, (
                &self.morton_codes,
                &mut self.bvh_left_child,
                &mut self.bvh_right_child,
                &mut self.bvh_parent,
                &mut self.bvh_range_left,
                &mut self.bvh_range_right,
                n as i32,
            ))?;
        }
        self.device.synchronize()?;
        let t5_elapsed = t5.elapsed().as_secs_f64() * 1000.0;

        // Step 6: Init leaves (with per-particle masses)
        let t6 = Instant::now();
        let init_leaves = self.device.get_func("nbody", "init_leaves")
            .ok_or("Failed to get init_leaves kernel")?;
        unsafe {
            init_leaves.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.masses,
                &mut self.bvh_node_data,
                &mut self.bvh_node_types,
                &mut self.bvh_atomic_counter,
                n as i32,
                box_half,
            ))?;
        }
        self.device.synchronize()?;
        let t6_elapsed = t6.elapsed().as_secs_f64() * 1000.0;

        // Step 7: COM reduction
        let t7 = Instant::now();
        let reduce_kernel = self.device.get_func("nbody", "reduce_com")
            .ok_or("Failed to get reduce_com kernel")?;
        unsafe {
            let mut diag_counters = self.device.alloc_zeros::<i32>(5)?;
            reduce_kernel.launch(cfg, (
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_parent,
                &self.bvh_range_left,
                &self.bvh_range_right,
                &mut self.bvh_node_data,
                &mut self.bvh_node_types,
                &mut self.bvh_atomic_counter,
                n as i32,
                box_half,
                &mut diag_counters,
            ))?;
        }
        self.device.synchronize()?;
        let t7_elapsed = t7.elapsed().as_secs_f64() * 1000.0;

        let total = t1_elapsed + t2_elapsed + t3_elapsed + t4_elapsed + t5_elapsed + t6_elapsed + t7_elapsed;

        println!("  1. Morton codes GPU:    {:6.1} ms ({:4.1}%)", t1_elapsed, t1_elapsed/total*100.0);
        println!("  2. Sort CPU (total):    {:6.1} ms ({:4.1}%)", t2_elapsed, t2_elapsed/total*100.0);
        println!("     - D2H transfer:      {:6.1} ms", t2a);
        println!("     - Rayon sort:        {:6.1} ms", t2b);
        println!("     - H2D transfer:      {:6.1} ms", t2c);
        println!("  3. Reorder GPU:         {:6.1} ms ({:4.1}%)", t3_elapsed, t3_elapsed/total*100.0);
        println!("  4. Reset buffers:       {:6.1} ms ({:4.1}%)", t4_elapsed, t4_elapsed/total*100.0);
        println!("  5. Karras BVH GPU:      {:6.1} ms ({:4.1}%)", t5_elapsed, t5_elapsed/total*100.0);
        println!("  6. Init leaves GPU:     {:6.1} ms ({:4.1}%)", t6_elapsed, t6_elapsed/total*100.0);
        println!("  7. COM reduction GPU:   {:6.1} ms ({:4.1}%)", t7_elapsed, t7_elapsed/total*100.0);
        println!("  ─────────────────────────────────");
        println!("  TOTAL:                  {:6.1} ms", total);

        Ok(())
    }

    /// Incremental COM update (opt5)
    /// Only updates leaf positions and recomputes COMs, keeping tree structure
    /// ~70ms vs ~280ms for full rebuild @ 2M particles
    pub fn update_com_only(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;
        let n_total_nodes = 2 * n - 1;
        let box_half = self.box_size / 2.0;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Fast GPU reset of atomic counters only
        let reset_blocks = (n_total_nodes + 255) / 256;
        let reset_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (reset_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let reset_kernel = self.device.get_func("nbody", "reset_atomic_counters")
            .ok_or("Failed to get reset_atomic_counters kernel")?;
        unsafe {
            reset_kernel.launch(reset_cfg, (
                &mut self.bvh_atomic_counter,
                n_total_nodes as i32,
            ))?;
        }

        // Step 2: Re-initialize leaves with current particle positions and masses
        let init_leaves = self.device.get_func("nbody", "init_leaves")
            .ok_or("Failed to get init_leaves kernel")?;
        unsafe {
            init_leaves.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.masses,
                &mut self.bvh_node_data,
                &mut self.bvh_node_types,
                &mut self.bvh_atomic_counter,
                n as i32,
                box_half,
            ))?;
        }
        self.device.synchronize()?;

        // Step 3: Bottom-up COM reduction
        let reduce_kernel = self.device.get_func("nbody", "reduce_com")
            .ok_or("Failed to get reduce_com kernel")?;
        let mut diag_counters = self.device.alloc_zeros::<i32>(5)?;
        unsafe {
            reduce_kernel.launch(cfg, (
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_parent,
                &self.bvh_range_left,
                &self.bvh_range_right,
                &mut self.bvh_node_data,
                &mut self.bvh_node_types,
                &mut self.bvh_atomic_counter,
                n as i32,
                box_half,
                &mut diag_counters,
            ))?;
        }
        self.device.synchronize()?;

        Ok(())
    }

    /// DKD integrator with GPU-built BVH tree
    /// Structure: Drift(dt/2) → GPU Tree Build → Force → Kick(dt) → Drift(dt/2)
    pub fn step_with_expansion_dkd_gpu(&mut self, dt: f64, _scale_factor: f64, hubble: f64, dtau_per_dt: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernels
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only kernel")?;
        let force_kernel = self.device.get_func("nbody", "compute_forces_bvh")
            .ok_or("Failed to get compute_forces_bvh kernel")?;

        // Step 1: Drift(dt/2)
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        // Step 2: Build GPU tree
        self.build_gpu_tree()?;

        // Step 3: Compute forces using GPU-built BVH (with VSL c_ratio_sq)
        unsafe {
            force_kernel.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.bvh_node_data,
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_node_types,
                &mut self.acc,
                n as i32,
                self.theta,
                self.softening,
                self.c_ratio_sq,  // VSL: (c_minus/c_plus)^2
                self.box_size,    // Periodic boundary conditions
            ))?;
        }

        // Step 4: Kick(dt)
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, n as i32,
                hubble, dtau_per_dt,
            ))?;
        }

        // Step 5: Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// DKD integrator with configurable cross-sign interaction
    /// cross_factor: 0.0 = attraction only, -1.0 = Janus repulsion
    pub fn step_with_cross_factor(&mut self, dt: f64, cross_factor: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernels
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only kernel")?;
        let force_kernel = self.device.get_func("nbody", "compute_forces_bvh_cross")
            .ok_or("Failed to get compute_forces_bvh_cross kernel")?;

        // Step 1: Drift(dt/2)
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        // Step 2: Build GPU tree
        self.build_gpu_tree()?;

        // Step 3: Compute forces with configurable cross interaction
        unsafe {
            force_kernel.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.bvh_node_data,
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_node_types,
                &mut self.acc,
                n as i32,
                self.theta,
                self.softening,
                cross_factor,  // Cross-sign interaction factor
                self.box_size, // Periodic boundary conditions
            ))?;
        }

        // Step 4: Kick(dt) - no Hubble damping for this test
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, n as i32,
                0.0_f64, 0.0_f64,  // H=0, dtau_per_dt=0
            ))?;
        }

        // Step 5: Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// DKD integrator with Yukawa-screened Janus interaction
    /// α(r) = 1 - ε×exp(-r/r_c) for opposite-sign interactions
    /// At r → 0: α → 1-ε (partial attraction for nearby opposite-sign)
    /// At r → ∞: α → 1 (standard Janus repulsion at large scales)
    pub fn step_with_yukawa(&mut self, dt: f64, epsilon: f64, r_c: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernels
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only kernel")?;
        let force_kernel = self.device.get_func("nbody", "compute_forces_bvh_yukawa")
            .ok_or("Failed to get compute_forces_bvh_yukawa kernel")?;

        // Step 1: Drift(dt/2)
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        // Step 2: Build GPU tree
        self.build_gpu_tree()?;

        // Step 3: Compute forces with Yukawa-screened interaction
        // Pack epsilon and r_c: yukawa_packed = r_c (integer) + epsilon (fractional)
        let yukawa_packed = r_c.floor() + epsilon;
        unsafe {
            force_kernel.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.bvh_node_data,
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_node_types,
                &mut self.acc,
                n as i32,
                self.theta,
                self.softening,
                yukawa_packed,
                self.box_size, // Periodic boundary conditions
            ))?;
        }

        // Step 4: Kick(dt) - no Hubble damping for this test
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, n as i32,
                0.0_f64, 0.0_f64,  // H=0, dtau_per_dt=0
            ))?;
        }

        // Step 5: Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// DKD integrator with incremental tree updates (opt5)
    /// Full rebuild every `rebuild_interval` steps, COM-only updates between
    /// Expected: ~100ms avg vs ~280ms full rebuild @ 2M particles
    pub fn step_with_expansion_dkd_gpu_incremental(
        &mut self,
        dt: f64,
        _scale_factor: f64,
        hubble: f64,
        dtau_per_dt: f64,
        step_num: usize,
        rebuild_interval: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernels
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only kernel")?;
        let force_kernel = self.device.get_func("nbody", "compute_forces_bvh")
            .ok_or("Failed to get compute_forces_bvh kernel")?;

        // Step 1: Drift(dt/2)
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        // Step 2: Build or update tree
        if step_num % rebuild_interval == 0 {
            // Full rebuild (sort + Karras + COM)
            self.build_gpu_tree()?;
        } else {
            // Incremental update (COM only, reuse tree structure)
            self.update_com_only()?;
        }

        // Step 3: Compute forces using GPU-built BVH (with VSL c_ratio_sq)
        unsafe {
            force_kernel.launch(cfg, (
                &self.pos,
                &self.signs,
                &self.bvh_node_data,
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_node_types,
                &mut self.acc,
                n as i32,
                self.theta,
                self.softening,
                self.c_ratio_sq,  // VSL: (c_minus/c_plus)^2
                self.box_size,    // Periodic boundary conditions
            ))?;
        }

        // Step 4: Kick(dt)
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, n as i32,
                hubble, dtau_per_dt,
            ))?;
        }

        // Step 5: Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                n as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// Initialize double-buffered BVH for async pipelining (opt7)
    pub fn init_async_buffers(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.bvh_buffers.is_some() {
            return Ok(());  // Already initialized
        }

        let n_bvh_nodes = 2 * self.n_particles - 1;

        // Allocate two BVH buffers
        let buffer_a = BvhBuffer {
            left_child: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            right_child: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            parent: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            range_left: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            range_right: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            node_data: self.device.alloc_zeros::<f64>(n_bvh_nodes * 12)?,
            node_types: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            atomic_counter: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
        };

        let buffer_b = BvhBuffer {
            left_child: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            right_child: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            parent: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            range_left: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            range_right: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            node_data: self.device.alloc_zeros::<f64>(n_bvh_nodes * 12)?,
            node_types: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
            atomic_counter: self.device.alloc_zeros::<i32>(n_bvh_nodes)?,
        };

        self.bvh_buffers = Some([buffer_a, buffer_b]);
        self.current_bvh = 0;

        Ok(())
    }

    /// Build GPU tree on specified buffer index (for async pipelining)
    fn build_gpu_tree_on_buffer(&mut self, buffer_idx: usize) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;
        let n_internal = n - 1;
        let n_total_nodes = 2 * n - 1;
        let box_half = self.box_size / 2.0;
        let inv_cell_size = 2097152.0 / self.box_size;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Morton codes
        let morton_kernel = self.device.get_func("nbody", "compute_morton_codes")
            .ok_or("Failed to get compute_morton_codes kernel")?;
        unsafe {
            morton_kernel.launch(cfg, (
                &self.pos,
                &mut self.morton_codes,
                &mut self.sorted_indices,
                n as i32,
                box_half,
                inv_cell_size,
            ))?;
        }
        self.device.synchronize()?;

        // Step 2: Sort (CPU)
        let morton_cpu = self.device.dtoh_sync_copy(&self.morton_codes)?;
        use rayon::prelude::*;
        let mut pairs: Vec<(u64, i32)> = morton_cpu.iter()
            .enumerate()
            .map(|(i, &m)| (m, i as i32))
            .collect();
        pairs.par_sort_unstable_by_key(|&(m, _)| m);
        let indices_cpu: Vec<i32> = pairs.iter().map(|&(_, i)| i).collect();
        self.sorted_indices = self.device.htod_sync_copy(&indices_cpu)?;
        let sorted_morton: Vec<u64> = pairs.iter().map(|&(m, _)| m).collect();
        self.morton_codes = self.device.htod_sync_copy(&sorted_morton)?;

        // Step 3: Reorder
        let reorder_pos = self.device.get_func("nbody", "reorder_positions")
            .ok_or("Failed to get reorder_positions")?;
        unsafe {
            reorder_pos.launch(cfg, (
                &self.pos, &mut self.pos_tmp, &self.sorted_indices, n as i32,
            ))?;
        }
        std::mem::swap(&mut self.pos, &mut self.pos_tmp);

        let reorder_signs = self.device.get_func("nbody", "reorder_signs")
            .ok_or("Failed to get reorder_signs")?;
        unsafe {
            reorder_signs.launch(cfg, (
                &self.signs, &mut self.signs_tmp, &self.sorted_indices, n as i32,
            ))?;
        }
        std::mem::swap(&mut self.signs, &mut self.signs_tmp);

        let reorder_vel = self.device.get_func("nbody", "reorder_velocities")
            .ok_or("Failed to get reorder_velocities")?;
        unsafe {
            reorder_vel.launch(cfg, (
                &self.vel, &mut self.vel_tmp, &self.sorted_indices, n as i32,
            ))?;
        }
        std::mem::swap(&mut self.vel, &mut self.vel_tmp);
        self.device.synchronize()?;

        // Get buffer reference
        let buffers = self.bvh_buffers.as_mut().ok_or("Async buffers not initialized")?;
        let buf = &mut buffers[buffer_idx];

        // Step 4: Reset buffers
        let zeros_i32: Vec<i32> = vec![0; n_total_nodes];
        let neg_ones_i32: Vec<i32> = vec![-1; n_total_nodes];
        let zeros_f64: Vec<f64> = vec![0.0; n_total_nodes * 12];

        self.device.htod_sync_copy_into(&zeros_i32, &mut buf.atomic_counter)?;
        self.device.htod_sync_copy_into(&zeros_i32, &mut buf.node_types)?;
        self.device.htod_sync_copy_into(&neg_ones_i32, &mut buf.left_child)?;
        self.device.htod_sync_copy_into(&neg_ones_i32, &mut buf.right_child)?;
        self.device.htod_sync_copy_into(&neg_ones_i32, &mut buf.parent)?;
        self.device.htod_sync_copy_into(&zeros_i32, &mut buf.range_left)?;
        self.device.htod_sync_copy_into(&zeros_i32, &mut buf.range_right)?;
        self.device.htod_sync_copy_into(&zeros_f64, &mut buf.node_data)?;

        // Step 5: Build internal nodes
        let internal_blocks = (n_internal + 255) / 256;
        let internal_cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (internal_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let build_kernel = self.device.get_func("nbody", "build_bvh_internal")
            .ok_or("Failed to get build_bvh_internal")?;
        unsafe {
            build_kernel.launch(internal_cfg, (
                &self.morton_codes,
                &mut buf.left_child,
                &mut buf.right_child,
                &mut buf.parent,
                &mut buf.range_left,
                &mut buf.range_right,
                n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // Step 6: Init leaves (with per-particle masses)
        let init_leaves = self.device.get_func("nbody", "init_leaves")
            .ok_or("Failed to get init_leaves")?;
        unsafe {
            init_leaves.launch(cfg, (
                &self.pos, &self.signs, &self.masses,
                &mut buf.node_data, &mut buf.node_types, &mut buf.atomic_counter,
                n as i32, box_half,
            ))?;
        }
        self.device.synchronize()?;

        // Step 7: COM reduction
        let reduce_kernel = self.device.get_func("nbody", "reduce_com")
            .ok_or("Failed to get reduce_com")?;
        let mut diag_counters = self.device.alloc_zeros::<i32>(5)?;
        unsafe {
            reduce_kernel.launch(cfg, (
                &buf.left_child, &buf.right_child, &buf.parent,
                &buf.range_left, &buf.range_right,
                &mut buf.node_data, &mut buf.node_types, &mut buf.atomic_counter,
                n as i32, box_half,
                &mut diag_counters,
            ))?;
        }
        self.device.synchronize()?;

        Ok(())
    }

    /// DKD integrator with async pipeline (opt7)
    /// Overlaps tree build for step t+1 with force computation for step t
    /// Uses fork_default_stream for concurrent execution
    pub fn step_with_expansion_dkd_gpu_async(
        &mut self,
        dt: f64,
        _scale_factor: f64,
        hubble: f64,
        dtau_per_dt: f64,
        step_num: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize double buffers on first call
        if self.bvh_buffers.is_none() {
            self.init_async_buffers()?;
            // Build initial tree on buffer 0
            self.build_gpu_tree_on_buffer(0)?;
            self.current_bvh = 0;
        }

        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernel handles
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only")?;
        let force_kernel = self.device.get_func("nbody", "compute_forces_bvh")
            .ok_or("Failed to get compute_forces_bvh")?;

        // Step 1: Drift(dt/2)
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }
        self.device.synchronize()?;

        // Current tree buffer for force computation
        let current_idx = self.current_bvh;
        let next_idx = 1 - current_idx;

        // Fork a stream for tree building
        let tree_stream = self.device.fork_default_stream()?;

        // Start building next tree on forked stream (async)
        // Note: This is a simplified version - full async would require
        // launching kernels on the tree_stream. For now, we do sequential
        // tree build but prepare for full async later.

        // Compute forces using current tree (on default stream) with VSL
        {
            let buffers = self.bvh_buffers.as_ref().ok_or("Buffers not initialized")?;
            let buf = &buffers[current_idx];

            unsafe {
                force_kernel.launch(cfg, (
                    &self.pos,
                    &self.signs,
                    &buf.node_data,
                    &buf.left_child,
                    &buf.right_child,
                    &buf.node_types,
                    &mut self.acc,
                    n as i32,
                    self.theta,
                    self.softening,
                    self.c_ratio_sq,  // VSL: (c_minus/c_plus)^2
                    self.box_size,    // Periodic boundary conditions
                ))?;
            }
        }
        self.device.synchronize()?;

        // Wait for tree stream before building (ensures force is done)
        self.device.wait_for(&tree_stream)?;

        // Build tree for next step on buffer next_idx
        self.build_gpu_tree_on_buffer(next_idx)?;

        // Swap buffers for next iteration
        self.current_bvh = next_idx;

        // Step 4: Kick(dt)
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, n as i32,
                hubble, dtau_per_dt,
            ))?;
        }

        // Step 5: Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half, n as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// DKD integrator with Morton sorting for cache locality
    /// Structure: Sort → Drift(dt/2) → Force → Kick(dt) → Drift(dt/2)
    pub fn step_with_expansion_dkd_morton(&mut self, dt: f64, scale_factor: f64, hubble: f64, dtau_per_dt: f64)
        -> Result<(), Box<dyn std::error::Error>>
    {
        // Morton sort particles for better cache locality (every step)
        self.morton_sort_particles()?;

        // Then do standard DKD step
        let half_dt = dt * 0.5;
        let box_half = self.box_size / 2.0;

        let blocks = (self.n_particles + 255) / 256;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernel functions
        let compute_forces = self.device.get_func("nbody", "compute_forces_simple")
            .ok_or("Failed to get compute_forces_simple kernel")?;
        let drift_kernel = self.device.get_func("nbody", "drift_only")
            .ok_or("Failed to get drift_only kernel")?;
        let kick_kernel = self.device.get_func("nbody", "kick_only")
            .ok_or("Failed to get kick_only kernel")?;

        // Step 1: Drift(dt/2)
        unsafe {
            drift_kernel.clone().launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                self.n_particles as i32,
            ))?;
        }

        // Step 2: Download positions for tree rebuild
        let pos_cpu = self.device.dtoh_sync_copy(&self.pos)?;
        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_cpu[i * 3];
            p.y = pos_cpu[i * 3 + 1];
            p.z = pos_cpu[i * 3 + 2];
        }

        // Step 3: Rebuild tree
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        // Step 4: Compute forces
        unsafe {
            compute_forces.launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                self.n_particles as i32,
                self.theta, self.softening, self.c_ratio_sq, self.repulsion_scale,
                self.box_size,  // Periodic boundary conditions
            ))?;
        }

        // Step 5: Kick(dt)
        unsafe {
            kick_kernel.clone().launch(cfg, (
                &mut self.vel, &self.acc,
                dt, self.n_particles as i32,
                hubble, dtau_per_dt,
            ))?;
        }

        // Step 6: Drift(dt/2)
        unsafe {
            drift_kernel.launch(cfg, (
                &mut self.pos, &self.vel,
                half_dt, box_half,
                self.n_particles as i32,
            ))?;
        }

        self.device.synchronize()?;
        self.time += dt;

        Ok(())
    }

    /// Compute binding potential energy (same-sign pairs only, always negative)
    /// Uses Barnes-Hut tree for O(N log N) complexity.
    /// WARNING: For large N (>1M), this builds a LinearOctree which may OOM.
    /// Use potential_energy_binding_sampled() for large N instead.
    pub fn potential_energy_binding(&self) -> Result<f64, Box<dyn std::error::Error>> {
        use rayon::prelude::*;

        // Build tree from CPU particle data
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);

        // Compute potential for each particle using tree traversal
        // Use theta = 0.5 for better accuracy in PE calculation
        let theta_pe = 0.5;

        let pe_bind: f64 = (0..self.n_particles)
            .into_par_iter()
            .map(|i| {
                let p = &self.particles_cpu[i];
                tree.compute_potential_same_sign(
                    p.x, p.y, p.z,
                    p.sign,
                    i,
                    self.softening,
                    theta_pe,
                )
            })
            .sum();

        // Divide by 2 to avoid double counting (each pair counted twice)
        Ok(pe_bind / 2.0)
    }

    /// Compute binding PE using sampling (for large N where full calculation would OOM)
    /// Samples n_sample particles from each sign, computes PE directly, then extrapolates.
    /// Accuracy improves with larger n_sample (recommend 5000-10000 for α estimation).
    pub fn potential_energy_binding_sampled(&self, n_sample: usize) -> Result<f64, Box<dyn std::error::Error>> {
        use rayon::prelude::*;

        // Separate particles by sign
        let positive: Vec<usize> = (0..self.n_particles)
            .filter(|&i| self.particles_cpu[i].sign == 1)
            .collect();
        let negative: Vec<usize> = (0..self.n_particles)
            .filter(|&i| self.particles_cpu[i].sign == -1)
            .collect();

        let n_pos = positive.len();
        let n_neg = negative.len();

        // Sample from each population using stride sampling (uniform distribution assumed)
        let sample_pos: Vec<usize> = if n_pos <= n_sample {
            positive.clone()
        } else {
            // Use stride sampling for speed (approximate uniform sampling)
            let stride = n_pos / n_sample;
            (0..n_sample).map(|i| positive[i * stride]).collect()
        };
        let sample_neg: Vec<usize> = if n_neg <= n_sample {
            negative.clone()
        } else {
            let stride = n_neg / n_sample;
            (0..n_sample).map(|i| negative[i * stride]).collect()
        };

        let ns_pos = sample_pos.len();
        let ns_neg = sample_neg.len();

        println!("  PE sampling: {} of {} positive, {} of {} negative",
            ns_pos, n_pos, ns_neg, n_neg);

        // Compute PE for positive sample (direct N² with parallelization)
        let pe_pos_sample: f64 = sample_pos.par_iter()
            .enumerate()
            .map(|(idx_i, &i)| {
                let pi = &self.particles_cpu[i];
                let mut pe_i = 0.0;
                for (idx_j, &j) in sample_pos.iter().enumerate() {
                    if idx_j <= idx_i { continue; } // Only count each pair once
                    let pj = &self.particles_cpu[j];
                    let dx = pi.x - pj.x;
                    let dy = pi.y - pj.y;
                    let dz = pi.z - pj.z;
                    let r2 = dx*dx + dy*dy + dz*dz + self.softening * self.softening;
                    pe_i -= 1.0 / r2.sqrt(); // -G*m*m/r with G=m=1
                }
                pe_i
            })
            .sum();

        // Compute PE for negative sample
        let pe_neg_sample: f64 = sample_neg.par_iter()
            .enumerate()
            .map(|(idx_i, &i)| {
                let pi = &self.particles_cpu[i];
                let mut pe_i = 0.0;
                for (idx_j, &j) in sample_neg.iter().enumerate() {
                    if idx_j <= idx_i { continue; }
                    let pj = &self.particles_cpu[j];
                    let dx = pi.x - pj.x;
                    let dy = pi.y - pj.y;
                    let dz = pi.z - pj.z;
                    let r2 = dx*dx + dy*dy + dz*dz + self.softening * self.softening;
                    pe_i -= 1.0 / r2.sqrt();
                }
                pe_i
            })
            .sum();

        // Extrapolate to full system
        // PE_sample has ns*(ns-1)/2 pairs, PE_full has N*(N-1)/2 pairs
        // Assuming uniform distribution, PE_full ≈ PE_sample * [N*(N-1)] / [ns*(ns-1)]
        let pairs_pos_sample = (ns_pos * ns_pos.saturating_sub(1)) as f64 / 2.0;
        let pairs_pos_full = (n_pos * n_pos.saturating_sub(1)) as f64 / 2.0;
        let pairs_neg_sample = (ns_neg * ns_neg.saturating_sub(1)) as f64 / 2.0;
        let pairs_neg_full = (n_neg * n_neg.saturating_sub(1)) as f64 / 2.0;

        let pe_pos_full = if pairs_pos_sample > 0.0 {
            pe_pos_sample * pairs_pos_full / pairs_pos_sample
        } else { 0.0 };
        let pe_neg_full = if pairs_neg_sample > 0.0 {
            pe_neg_sample * pairs_neg_full / pairs_neg_sample
        } else { 0.0 };

        let pe_total = pe_pos_full + pe_neg_full;

        println!("  PE_+ sample = {:.4e}, extrapolated = {:.4e}", pe_pos_sample, pe_pos_full);
        println!("  PE_- sample = {:.4e}, extrapolated = {:.4e}", pe_neg_sample, pe_neg_full);
        println!("  PE_binding  = {:.4e}", pe_total);

        Ok(pe_total)
    }

    /// Virialize velocities to satisfy 2KE + PE_binding = 0
    /// Call this once at t=0 after initialization.
    pub fn virialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let ke = self.kinetic_energy()?;
        let pe_bind = self.potential_energy_binding()?;

        println!("Virialization (CPU tree, Janus mode):");
        println!("  KE initial    = {:.4e}", ke);
        println!("  PE binding    = {:.4e}", pe_bind);

        if ke < 1e-20 {
            println!("  WARNING: KE too small, skipping virialization");
            return Ok(());
        }

        if pe_bind >= 0.0 {
            println!("  WARNING: PE_binding >= 0, no bound system to virialize");
            return Ok(());
        }

        // Virial condition: 2KE + PE_bind = 0 → KE_target = |PE_bind|/2
        let ke_target = pe_bind.abs() / 2.0;
        let alpha = (ke_target / ke).sqrt();

        println!("  KE target     = {:.4e}", ke_target);
        println!("  Alpha scale   = {:.6}", alpha);

        // Scale velocities on GPU
        let mut vel = self.device.dtoh_sync_copy(&self.vel)?;
        for v in vel.iter_mut() {
            *v *= alpha;
        }
        self.vel = self.device.htod_sync_copy(&vel)?;

        // Also update CPU copy
        for p in self.particles_cpu.iter_mut() {
            p.vx *= alpha;
            p.vy *= alpha;
            p.vz *= alpha;
        }

        // Verify
        let ke_after = self.kinetic_energy()?;
        let virial_ratio = 2.0 * ke_after + pe_bind;
        let virial_error = virial_ratio.abs() / pe_bind.abs();

        println!("  KE after      = {:.4e}", ke_after);
        println!("  2KE + PE_bind = {:.4e}", virial_ratio);
        println!("  Virial error  = {:.4}%", virial_error * 100.0);

        Ok(())
    }

    /// Virialize using sampled PE calculation (for large N where full calculation would OOM)
    /// n_sample: number of particles to sample from each sign (recommend 5000-10000)
    pub fn virialize_sampled(&mut self, n_sample: usize) -> Result<(), Box<dyn std::error::Error>> {
        let ke = self.kinetic_energy()?;

        println!("Virialization (sampled, Janus mode):");
        println!("  KE initial    = {:.4e}", ke);

        if ke < 1e-20 {
            println!("  WARNING: KE too small, skipping virialization");
            return Ok(());
        }

        let pe_bind = self.potential_energy_binding_sampled(n_sample)?;

        if pe_bind >= 0.0 {
            println!("  WARNING: PE_binding >= 0, no bound system to virialize");
            return Ok(());
        }

        // Virial condition: 2KE + PE_bind = 0 → KE_target = |PE_bind|/2
        let ke_target = pe_bind.abs() / 2.0;
        let alpha = (ke_target / ke).sqrt();

        println!("  KE target     = {:.4e}", ke_target);
        println!("  Alpha (α)     = {:.6}", alpha);

        // Scale velocities on GPU
        let mut vel = self.device.dtoh_sync_copy(&self.vel)?;
        for v in vel.iter_mut() {
            *v *= alpha;
        }
        self.vel = self.device.htod_sync_copy(&vel)?;

        // Also update CPU copy
        for p in self.particles_cpu.iter_mut() {
            p.vx *= alpha;
            p.vy *= alpha;
            p.vz *= alpha;
        }

        // Verify
        let ke_after = self.kinetic_energy()?;
        println!("  KE after      = {:.4e}", ke_after);
        println!("  Expected α    ≈ 4-5 for proper Janus virialization");

        Ok(())
    }

    /// Compute segregation distance using minimum image convention
    /// and normalized by box_size for comparability across scales.
    ///
    /// Uses a SINGLE reference point for both populations to avoid bias.
    /// Computes relative positions using minimum image convention.
    pub fn segregation_distance(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let pos = self.device.dtoh_sync_copy(&self.pos)?;
        let signs = self.device.dtoh_sync_copy(&self.signs)?;
        let l = self.box_size;

        if self.n_particles == 0 {
            return Ok(0.0);
        }

        // Use first particle as COMMON reference for both populations
        // This is critical: using different references biases the result
        let ref_pos = (pos[0], pos[1], pos[2]);

        let mut sum_plus = (0.0f64, 0.0f64, 0.0f64);
        let mut n_plus = 0usize;
        let mut sum_minus = (0.0f64, 0.0f64, 0.0f64);
        let mut n_minus = 0usize;

        for i in 0..self.n_particles {
            let x = pos[i * 3];
            let y = pos[i * 3 + 1];
            let z = pos[i * 3 + 2];

            // Unwrap relative to common reference
            let ux = ref_pos.0 + minimum_image_gpu(x - ref_pos.0, l);
            let uy = ref_pos.1 + minimum_image_gpu(y - ref_pos.1, l);
            let uz = ref_pos.2 + minimum_image_gpu(z - ref_pos.2, l);

            if signs[i] > 0 {
                sum_plus.0 += ux;
                sum_plus.1 += uy;
                sum_plus.2 += uz;
                n_plus += 1;
            } else {
                sum_minus.0 += ux;
                sum_minus.1 += uy;
                sum_minus.2 += uz;
                n_minus += 1;
            }
        }

        // Compute COMs
        let com_plus = if n_plus > 0 {
            let n = n_plus as f64;
            (sum_plus.0 / n, sum_plus.1 / n, sum_plus.2 / n)
        } else {
            (0.0, 0.0, 0.0)
        };

        let com_minus = if n_minus > 0 {
            let n = n_minus as f64;
            (sum_minus.0 / n, sum_minus.1 / n, sum_minus.2 / n)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Minimum image distance between COMs
        let dx = minimum_image_gpu(com_plus.0 - com_minus.0, l);
        let dy = minimum_image_gpu(com_plus.1 - com_minus.1, l);
        let dz = minimum_image_gpu(com_plus.2 - com_minus.2, l);

        // Normalize by box_size for comparability
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        Ok(distance / l)
    }

    pub fn get_positions(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let pos = self.device.dtoh_sync_copy(&self.pos)?;
        Ok(pos)
    }

    pub fn get_velocities(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let vel = self.device.dtoh_sync_copy(&self.vel)?;
        Ok(vel)
    }

    /// Set velocities (for external pressure/force updates)
    pub fn set_velocities(&mut self, vel: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        self.device.htod_sync_copy_into(vel, &mut self.vel)?;
        Ok(())
    }

    pub fn get_signs(&self) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        let signs = self.device.dtoh_sync_copy(&self.signs)?;
        Ok(signs)
    }

    /// Debug method: compare forces between GPU tree and CPU tree
    pub fn compare_forces_debug(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;

        let blocks = (n + 255) / 256;
        let cfg = cudarc::driver::LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Build GPU tree (this reorders particles by Morton code)
        self.build_gpu_tree()?;

        // Step 2: Compute forces with GPU BVH tree (with VSL c_ratio_sq)
        let gpu_force_kernel = self.device.get_func("nbody", "compute_forces_bvh")
            .ok_or("Failed to get compute_forces_bvh kernel")?;
        unsafe {
            gpu_force_kernel.launch(cfg, (
                &self.pos, &self.signs,
                &self.bvh_node_data,
                &self.bvh_left_child,
                &self.bvh_right_child,
                &self.bvh_node_types,
                &mut self.acc,
                n as i32,
                self.theta,
                self.softening,
                self.c_ratio_sq,  // VSL: (c_minus/c_plus)^2
                self.box_size,    // Periodic boundary conditions
            ))?;
        }
        self.device.synchronize()?;
        let acc_gpu = self.device.dtoh_sync_copy(&self.acc)?;

        // Step 3: Get current positions (after Morton reordering)
        let pos_sorted = self.device.dtoh_sync_copy(&self.pos)?;
        let signs_sorted = self.device.dtoh_sync_copy(&self.signs)?;

        // Step 4: Update particles_cpu with sorted positions
        for (i, p) in self.particles_cpu.iter_mut().enumerate() {
            p.x = pos_sorted[i * 3];
            p.y = pos_sorted[i * 3 + 1];
            p.z = pos_sorted[i * 3 + 2];
            p.sign = signs_sorted[i];
        }

        // Step 5: Build CPU tree from sorted positions
        let tree = LinearOctree::build(&self.particles_cpu, self.box_size);
        self.n_nodes = tree.nodes.len();
        self.node_data = self.device.htod_sync_copy(&tree.node_data)?;
        self.node_children = self.device.htod_sync_copy(&tree.node_children)?;
        self.node_types = self.device.htod_sync_copy(&tree.node_types)?;

        // Step 6: Compute forces with CPU tree
        let cpu_force_kernel = self.device.get_func("nbody", "compute_forces_simple")
            .ok_or("Failed to get compute_forces_simple kernel")?;
        unsafe {
            cpu_force_kernel.launch(cfg, (
                &self.pos, &self.signs,
                &self.node_data, &self.node_children, &self.node_types,
                &mut self.acc,
                n as i32,
                self.theta, self.softening, self.c_ratio_sq, self.repulsion_scale,
                self.box_size,  // Periodic boundary conditions
            ))?;
        }
        self.device.synchronize()?;
        let acc_cpu = self.device.dtoh_sync_copy(&self.acc)?;

        // Compare forces
        let mut max_diff = 0.0f64;
        let mut max_diff_idx = 0;
        let mut sum_diff = 0.0;
        let mut sum_mag_cpu = 0.0;
        let mut sum_mag_gpu = 0.0;

        for i in 0..n {
            let ax_cpu = acc_cpu[i * 3];
            let ay_cpu = acc_cpu[i * 3 + 1];
            let az_cpu = acc_cpu[i * 3 + 2];
            let ax_gpu = acc_gpu[i * 3];
            let ay_gpu = acc_gpu[i * 3 + 1];
            let az_gpu = acc_gpu[i * 3 + 2];

            let mag_cpu = (ax_cpu*ax_cpu + ay_cpu*ay_cpu + az_cpu*az_cpu).sqrt();
            let mag_gpu = (ax_gpu*ax_gpu + ay_gpu*ay_gpu + az_gpu*az_gpu).sqrt();

            let dx = ax_cpu - ax_gpu;
            let dy = ay_cpu - ay_gpu;
            let dz = az_cpu - az_gpu;
            let diff = (dx*dx + dy*dy + dz*dz).sqrt();

            sum_diff += diff;
            sum_mag_cpu += mag_cpu;
            sum_mag_gpu += mag_gpu;

            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }

        let avg_diff = sum_diff / n as f64;
        let avg_mag_cpu = sum_mag_cpu / n as f64;
        let avg_mag_gpu = sum_mag_gpu / n as f64;

        println!("\n══════ Force Comparison Results ══════");
        println!("  Particles:          {}", n);
        println!("  Avg |a| CPU tree:   {:.6e}", avg_mag_cpu);
        println!("  Avg |a| GPU tree:   {:.6e}", avg_mag_gpu);
        println!("  Avg |diff|:         {:.6e}", avg_diff);
        println!("  Relative error:     {:.2}%", avg_diff / avg_mag_cpu * 100.0);
        println!("  Max |diff|:         {:.6e} (particle {})", max_diff, max_diff_idx);

        // Show details for worst particle
        let i = max_diff_idx;
        println!("\n  Worst particle {}:", i);
        println!("    CPU: ({:.6e}, {:.6e}, {:.6e})",
                 acc_cpu[i*3], acc_cpu[i*3+1], acc_cpu[i*3+2]);
        println!("    GPU: ({:.6e}, {:.6e}, {:.6e})",
                 acc_gpu[i*3], acc_gpu[i*3+1], acc_gpu[i*3+2]);
        println!("    Position: ({:.3}, {:.3}, {:.3})",
                 pos_sorted[i*3], pos_sorted[i*3+1], pos_sorted[i*3+2]);

        // Sample a few random particles
        println!("\n  Sample comparisons:");
        for &idx in &[0, n/4, n/2, 3*n/4, n-1] {
            let mag_cpu = (acc_cpu[idx*3].powi(2) + acc_cpu[idx*3+1].powi(2) + acc_cpu[idx*3+2].powi(2)).sqrt();
            let mag_gpu = (acc_gpu[idx*3].powi(2) + acc_gpu[idx*3+1].powi(2) + acc_gpu[idx*3+2].powi(2)).sqrt();
            let dx = acc_cpu[idx*3] - acc_gpu[idx*3];
            let dy = acc_cpu[idx*3+1] - acc_gpu[idx*3+1];
            let dz = acc_cpu[idx*3+2] - acc_gpu[idx*3+2];
            let diff = (dx*dx + dy*dy + dz*dz).sqrt();
            let rel_err = if mag_cpu > 1e-10 { diff / mag_cpu * 100.0 } else { 0.0 };
            println!("    [{}]: |a_cpu|={:.4e}, |a_gpu|={:.4e}, rel_err={:.1}%",
                     idx, mag_cpu, mag_gpu, rel_err);
        }

        println!("══════════════════════════════════════\n");

        Ok(())
    }

    /// Debug: dump BVH structure
    pub fn debug_bvh_structure(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_particles;
        let n_internal = n - 1;
        let n_total = 2 * n - 1;

        println!("\n══════ BVH Structure Debug ══════");
        println!("N particles: {}", n);
        println!("N internal nodes: {} (0..{})", n_internal, n_internal - 1);
        println!("N leaves: {} ({}..{})", n, n_internal, n_total - 1);

        // Build GPU tree
        self.build_gpu_tree()?;

        // Download BVH data
        let left_child = self.device.dtoh_sync_copy(&self.bvh_left_child)?;
        let right_child = self.device.dtoh_sync_copy(&self.bvh_right_child)?;
        let parent = self.device.dtoh_sync_copy(&self.bvh_parent)?;
        let range_left = self.device.dtoh_sync_copy(&self.bvh_range_left)?;
        let range_right = self.device.dtoh_sync_copy(&self.bvh_range_right)?;
        let node_types = self.device.dtoh_sync_copy(&self.bvh_node_types)?;
        let node_data = self.device.dtoh_sync_copy(&self.bvh_node_data)?;

        // Find root (node with range [0, n-1])
        let mut root = -1i32;
        for i in 0..n_internal {
            if range_left[i] == 0 && range_right[i] == n as i32 - 1 {
                root = i as i32;
                break;
            }
        }
        println!("\nRoot node: {} (expected to have range [0, {}])", root, n - 1);

        // Check node 0's range
        println!("Node 0: range [{}, {}], left={}, right={}",
                 range_left[0], range_right[0], left_child[0], right_child[0]);

        // Print first few internal nodes
        println!("\nFirst 10 internal nodes:");
        for i in 0..std::cmp::min(10, n_internal) {
            let base = i * 12;
            let cx = node_data[base];
            let cy = node_data[base + 1];
            let cz = node_data[base + 2];
            let half_size = node_data[base + 3];
            let mass_plus = node_data[base + 7];
            let mass_minus = node_data[base + 11];
            println!("  [{}] type={} range=[{},{}] L={} R={} parent={} half_size={:.3} m+={:.0} m-={:.0}",
                     i, node_types[i],
                     range_left[i], range_right[i],
                     left_child[i], right_child[i], parent[i],
                     half_size, mass_plus, mass_minus);
        }

        // Print first few leaves
        println!("\nFirst 10 leaves:");
        for i in 0..std::cmp::min(10, n) {
            let node_idx = n_internal + i;
            let base = node_idx * 12;
            let x = node_data[base];
            let y = node_data[base + 1];
            let z = node_data[base + 2];
            let half_size = node_data[base + 3];
            let mass_plus = node_data[base + 7];
            let mass_minus = node_data[base + 11];
            println!("  [{}] type={} parent={} pos=({:.2},{:.2},{:.2}) m+={:.0} m-={:.0}",
                     node_idx, node_types[node_idx], parent[node_idx],
                     x, y, z, mass_plus, mass_minus);
        }

        // Verify tree connectivity
        println!("\nTree connectivity check:");
        let mut orphan_internal = 0;
        let mut orphan_leaves = 0;
        let mut invalid_children = 0;

        for i in 0..n_internal {
            if parent[i] == -1 && i != root as usize {
                orphan_internal += 1;
            }
            let l = left_child[i];
            let r = right_child[i];
            if l < 0 || l >= n_total as i32 {
                invalid_children += 1;
            }
            if r < 0 || r >= n_total as i32 {
                invalid_children += 1;
            }
        }

        for i in n_internal..n_total {
            if parent[i] == -1 {
                orphan_leaves += 1;
            }
        }

        println!("  Orphan internal nodes (no parent except root): {}", orphan_internal);
        println!("  Orphan leaves: {}", orphan_leaves);
        println!("  Invalid child indices: {}", invalid_children);

        // Check total mass
        let root_idx = if root >= 0 { root as usize } else { 0 };
        let root_base = root_idx * 12;
        let root_mp = node_data[root_base + 7];
        let root_mm = node_data[root_base + 11];
        println!("\nRoot total mass: m+={:.0} m-={:.0} (expected: {}+ {}−)",
                 root_mp, root_mm,
                 (n as f64 / (1.0 + 1.045)).round(),
                 (n as f64 * 1.045 / (1.0 + 1.045)).round());

        println!("══════════════════════════════════════\n");

        Ok(())
    }
}

/// Minimum image convention for periodic boundaries
fn minimum_image_gpu(dx: f64, box_size: f64) -> f64 {
    let mut d = dx;
    while d > box_size / 2.0 { d -= box_size; }
    while d < -box_size / 2.0 { d += box_size; }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_octree_build() {
        let particles = vec![
            GpuParticle { x: 10.0, y: 10.0, z: 10.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0, sign: 1 },
            GpuParticle { x: -10.0, y: -10.0, z: -10.0, vx: 0.0, vy: 0.0, vz: 0.0, mass: 1.0, sign: -1 },
        ];

        let tree = LinearOctree::build(&particles, 100.0);
        assert!(tree.nodes.len() >= 2);
    }

    #[test]
    fn test_octant_calculation() {
        assert_eq!(LinearOctree::get_octant(1.0, 1.0, 1.0, 0.0, 0.0, 0.0), 7);
        assert_eq!(LinearOctree::get_octant(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0), 0);
    }
}
