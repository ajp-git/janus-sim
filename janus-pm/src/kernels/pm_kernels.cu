// PM-5 CUDA Kernels for Janus Particle-Mesh simulation
// All particle data lives on GPU - zero CPU transfers in main loop
//
// Compile: nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC -o libpm_kernels.so pm_kernels.cu

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>

// Constants
#define PI 3.14159265358979323846f

// ============================================================================
// CIC Deposit Kernel
// Particles → density grids (ρ+ and ρ-)
// Uses atomic adds for thread safety
// ============================================================================

__global__ void cic_deposit_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int8_t* __restrict__ signs,
    float* __restrict__ rho_plus,
    float* __restrict__ rho_minus,
    int n_particles,
    int nx, int ny, int nz,
    float box_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    // Get particle position and sign
    double px = pos_x[idx];
    double py = pos_y[idx];
    double pz = pos_z[idx];
    int8_t sign = signs[idx];

    // Grid spacing
    float dx = box_size / (float)nx;
    float inv_dx = 1.0f / dx;

    // Normalize to grid coordinates [0, nx)
    float gx = (float)fmod(px + box_size, (double)box_size) * inv_dx;
    float gy = (float)fmod(py + box_size, (double)box_size) * inv_dx;
    float gz = (float)fmod(pz + box_size, (double)box_size) * inv_dx;

    // Cell indices (lower-left corner)
    int ix = (int)floorf(gx);
    int iy = (int)floorf(gy);
    int iz = (int)floorf(gz);

    // Fractional position within cell
    float fx = gx - (float)ix;
    float fy = gy - (float)iy;
    float fz = gz - (float)iz;

    // CIC weights
    float wx0 = 1.0f - fx, wx1 = fx;
    float wy0 = 1.0f - fy, wy1 = fy;
    float wz0 = 1.0f - fz, wz1 = fz;

    // Select target grid
    float* rho = (sign > 0) ? rho_plus : rho_minus;

    // Deposit to 8 neighboring cells with periodic boundary
    #pragma unroll
    for (int dz = 0; dz < 2; dz++) {
        int jz = (iz + dz) % nz;
        float wz = (dz == 0) ? wz0 : wz1;

        #pragma unroll
        for (int dy = 0; dy < 2; dy++) {
            int jy = (iy + dy) % ny;
            float wy = (dy == 0) ? wy0 : wy1;

            #pragma unroll
            for (int dx_i = 0; dx_i < 2; dx_i++) {
                int jx = (ix + dx_i) % nx;
                float wx = (dx_i == 0) ? wx0 : wx1;

                float weight = wx * wy * wz;
                int cell_idx = jx + nx * (jy + ny * jz);
                atomicAdd(&rho[cell_idx], weight);
            }
        }
    }
}

// ============================================================================
// Green's Function + Gradient Kernel (k-space)
// Applies G(k) = -4π/(k² + k_s²) and computes gradient ∇ = ik
// Input: ρ̂(k) complex, Output: ĝx, ĝy, ĝz complex
// ============================================================================

__global__ void green_gradient_kernel(
    const cufftComplex* __restrict__ rho_k,
    cufftComplex* __restrict__ gx_k,
    cufftComplex* __restrict__ gy_k,
    cufftComplex* __restrict__ gz_k,
    int nx, int ny, int nz,
    float dx,
    float k_softening
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_total = nx * ny * nz;
    if (idx >= n_total) return;

    // Convert linear index to 3D
    int iz = idx / (nx * ny);
    int iy = (idx / nx) % ny;
    int ix = idx % nx;

    // Wave numbers (centered FFT convention)
    float kx = (ix < nx/2) ? (float)ix : (float)(ix - nx);
    float ky = (iy < ny/2) ? (float)iy : (float)(iy - ny);
    float kz = (iz < nz/2) ? (float)iz : (float)(iz - nz);

    // Scale to physical k
    float dk = 2.0f * PI / (nx * dx);
    kx *= dk;
    ky *= dk;
    kz *= dk;

    float k2 = kx*kx + ky*ky + kz*kz;
    float k_s2 = k_softening * k_softening;

    // DC mode: zero potential and gradient
    if (k2 < 1e-10f) {
        gx_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
        gy_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
        gz_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // Green's function with softening: G(k) = -4π / (k² + k_s²)
    float G = -4.0f * PI / (k2 + k_s2);

    // Get ρ̂(k)
    cufftComplex rho = rho_k[idx];

    // φ̂(k) = G(k) × ρ̂(k)
    // ĝ(k) = -∇φ̂(k) = -ik × φ̂(k) = -ik × G × ρ̂
    // For acceleration: g = -∇φ, so ĝ = ik × φ̂ (note sign)
    // Multiply by i: (a + bi) × i = -b + ai

    float phi_re = G * rho.x;
    float phi_im = G * rho.y;

    // ĝx = i × kx × φ̂ = kx × (-φ̂.im + i×φ̂.re)
    gx_k[idx] = make_cuFloatComplex(-kx * phi_im, kx * phi_re);
    gy_k[idx] = make_cuFloatComplex(-ky * phi_im, ky * phi_re);
    gz_k[idx] = make_cuFloatComplex(-kz * phi_im, kz * phi_re);
}

// ============================================================================
// Force Interpolation Kernel
// Interpolates grid accelerations to particles using CIC
// Computes F = g+ - g- for positive mass, F = g- - g+ for negative
// ============================================================================

__global__ void force_interpolation_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int8_t* __restrict__ signs,
    const float* __restrict__ gx_plus,
    const float* __restrict__ gy_plus,
    const float* __restrict__ gz_plus,
    const float* __restrict__ gx_minus,
    const float* __restrict__ gy_minus,
    const float* __restrict__ gz_minus,
    float* __restrict__ fx,
    float* __restrict__ fy,
    float* __restrict__ fz,
    int n_particles,
    int nx, int ny, int nz,
    float box_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    // Get particle position
    double px = pos_x[idx];
    double py = pos_y[idx];
    double pz = pos_z[idx];
    int8_t sign = signs[idx];

    // Grid spacing
    float dx = box_size / (float)nx;
    float inv_dx = 1.0f / dx;

    // Normalize to grid coordinates
    float gx = (float)fmod(px + box_size, (double)box_size) * inv_dx;
    float gy = (float)fmod(py + box_size, (double)box_size) * inv_dx;
    float gz = (float)fmod(pz + box_size, (double)box_size) * inv_dx;

    // Cell indices
    int ix = (int)floorf(gx);
    int iy = (int)floorf(gy);
    int iz = (int)floorf(gz);

    // Fractional position
    float fracx = gx - (float)ix;
    float fracy = gy - (float)iy;
    float fracz = gz - (float)iz;

    // CIC weights
    float wx0 = 1.0f - fracx, wx1 = fracx;
    float wy0 = 1.0f - fracy, wy1 = fracy;
    float wz0 = 1.0f - fracz, wz1 = fracz;

    // Interpolate both fields
    float ax_plus = 0.0f, ay_plus = 0.0f, az_plus = 0.0f;
    float ax_minus = 0.0f, ay_minus = 0.0f, az_minus = 0.0f;

    #pragma unroll
    for (int dz = 0; dz < 2; dz++) {
        int jz = (iz + dz) % nz;
        float wz = (dz == 0) ? wz0 : wz1;

        #pragma unroll
        for (int dy = 0; dy < 2; dy++) {
            int jy = (iy + dy) % ny;
            float wy = (dy == 0) ? wy0 : wy1;

            #pragma unroll
            for (int dx_i = 0; dx_i < 2; dx_i++) {
                int jx = (ix + dx_i) % nx;
                float wx = (dx_i == 0) ? wx0 : wx1;

                float weight = wx * wy * wz;
                int cell_idx = jx + nx * (jy + ny * jz);

                ax_plus += weight * gx_plus[cell_idx];
                ay_plus += weight * gy_plus[cell_idx];
                az_plus += weight * gz_plus[cell_idx];

                ax_minus += weight * gx_minus[cell_idx];
                ay_minus += weight * gy_minus[cell_idx];
                az_minus += weight * gz_minus[cell_idx];
            }
        }
    }

    // Janus force: F+ = g+ - g-, F- = g- - g+
    if (sign > 0) {
        fx[idx] = ax_plus - ax_minus;
        fy[idx] = ay_plus - ay_minus;
        fz[idx] = az_plus - az_minus;
    } else {
        fx[idx] = ax_minus - ax_plus;
        fy[idx] = ay_minus - ay_plus;
        fz[idx] = az_minus - az_plus;
    }
}

// ============================================================================
// Kick Kernel (velocity update)
// v += (F - H×v) × dt
// ============================================================================

__global__ void kick_kernel(
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    int n_particles,
    float dt,
    float hubble_friction  // H × dtau_per_dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    float vx = vel_x[idx];
    float vy = vel_y[idx];
    float vz = vel_z[idx];

    // Kick with Hubble friction
    vel_x[idx] = vx + (fx[idx] - hubble_friction * vx) * dt;
    vel_y[idx] = vy + (fy[idx] - hubble_friction * vy) * dt;
    vel_z[idx] = vz + (fz[idx] - hubble_friction * vz) * dt;
}

// ============================================================================
// Drift Kernel (position update with periodic boundary)
// x += v × dt, then wrap to [0, box_size)
// ============================================================================

__global__ void drift_kernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ vel_z,
    int n_particles,
    float dt,
    double box_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    double px = pos_x[idx] + (double)vel_x[idx] * (double)dt;
    double py = pos_y[idx] + (double)vel_y[idx] * (double)dt;
    double pz = pos_z[idx] + (double)vel_z[idx] * (double)dt;

    // Periodic boundary wrap
    pos_x[idx] = fmod(px + box_size, box_size);
    pos_y[idx] = fmod(py + box_size, box_size);
    pos_z[idx] = fmod(pz + box_size, box_size);
}

// ============================================================================
// Utility Kernels
// ============================================================================

// Zero out a float array
__global__ void zero_float_kernel(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = 0.0f;
}

// Scale velocities by factor (for virialization)
__global__ void scale_velocities_kernel(
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    int n_particles,
    float factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    vel_x[idx] *= factor;
    vel_y[idx] *= factor;
    vel_z[idx] *= factor;
}

// Compute kinetic energy (partial sum per block)
__global__ void kinetic_energy_kernel(
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ vel_z,
    float* __restrict__ partial_sums,
    int n_particles
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < n_particles) {
        float vx = vel_x[idx];
        float vy = vel_y[idx];
        float vz = vel_z[idx];
        sum = 0.5f * (vx*vx + vy*vy + vz*vz);
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Compute segregation metric (partial sums for pos/neg COM)
__global__ void segregation_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int8_t* __restrict__ signs,
    double* __restrict__ sum_pos,  // [6]: x,y,z for positive, count
    double* __restrict__ sum_neg,  // [6]: x,y,z for negative, count
    int n_particles,
    double box_size
) {
    extern __shared__ double sdata_d[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Local accumulators: pos_x, pos_y, pos_z, neg_x, neg_y, neg_z, n_pos, n_neg
    double local[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    if (idx < n_particles) {
        double x = pos_x[idx];
        double y = pos_y[idx];
        double z = pos_z[idx];

        if (signs[idx] > 0) {
            local[0] = x;
            local[1] = y;
            local[2] = z;
            local[6] = 1.0;
        } else {
            local[3] = x;
            local[4] = y;
            local[5] = z;
            local[7] = 1.0;
        }
    }

    // Store in shared memory (8 values per thread)
    for (int i = 0; i < 8; i++) {
        sdata_d[tid * 8 + i] = local[i];
    }
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int i = 0; i < 8; i++) {
                sdata_d[tid * 8 + i] += sdata_d[(tid + s) * 8 + i];
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        int bidx = blockIdx.x;
        sum_pos[bidx * 4 + 0] = sdata_d[0];  // pos_x
        sum_pos[bidx * 4 + 1] = sdata_d[1];  // pos_y
        sum_pos[bidx * 4 + 2] = sdata_d[2];  // pos_z
        sum_pos[bidx * 4 + 3] = sdata_d[6];  // n_pos

        sum_neg[bidx * 4 + 0] = sdata_d[3];  // neg_x
        sum_neg[bidx * 4 + 1] = sdata_d[4];  // neg_y
        sum_neg[bidx * 4 + 2] = sdata_d[5];  // neg_z
        sum_neg[bidx * 4 + 3] = sdata_d[7];  // n_neg
    }
}

// ============================================================================
// Real ↔ Complex Conversion Kernels
// ============================================================================

__global__ void real_to_complex_kernel(
    const float* __restrict__ real_in,
    cufftComplex* __restrict__ complex_out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    complex_out[idx] = make_cuFloatComplex(real_in[idx], 0.0f);
}

__global__ void complex_to_real_kernel(
    const cufftComplex* __restrict__ complex_in,
    float* __restrict__ real_out,
    int n,
    float norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    real_out[idx] = complex_in[idx].x * norm;
}

// ============================================================================
// C Interface for Rust FFI
// ============================================================================

extern "C" {

void launch_real_to_complex(
    const float* real_in,
    cufftComplex* complex_out,
    int n,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    real_to_complex_kernel<<<n_blocks, block_size, 0, stream>>>(real_in, complex_out, n);
}

void launch_complex_to_real(
    const cufftComplex* complex_in,
    float* real_out,
    int n,
    float norm,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    complex_to_real_kernel<<<n_blocks, block_size, 0, stream>>>(complex_in, real_out, n, norm);
}

void launch_cic_deposit(
    const double* pos_x, const double* pos_y, const double* pos_z,
    const int8_t* signs,
    float* rho_plus, float* rho_minus,
    int n_particles, int nx, int ny, int nz, float box_size,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n_particles + block_size - 1) / block_size;

    cic_deposit_kernel<<<n_blocks, block_size, 0, stream>>>(
        pos_x, pos_y, pos_z, signs,
        rho_plus, rho_minus,
        n_particles, nx, ny, nz, box_size
    );
}

void launch_green_gradient(
    const cufftComplex* rho_k,
    cufftComplex* gx_k, cufftComplex* gy_k, cufftComplex* gz_k,
    int nx, int ny, int nz, float dx, float k_softening,
    cudaStream_t stream
) {
    int n_total = nx * ny * nz;
    int block_size = 256;
    int n_blocks = (n_total + block_size - 1) / block_size;

    green_gradient_kernel<<<n_blocks, block_size, 0, stream>>>(
        rho_k, gx_k, gy_k, gz_k,
        nx, ny, nz, dx, k_softening
    );
}

void launch_force_interpolation(
    const double* pos_x, const double* pos_y, const double* pos_z,
    const int8_t* signs,
    const float* gx_plus, const float* gy_plus, const float* gz_plus,
    const float* gx_minus, const float* gy_minus, const float* gz_minus,
    float* fx, float* fy, float* fz,
    int n_particles, int nx, int ny, int nz, float box_size,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n_particles + block_size - 1) / block_size;

    force_interpolation_kernel<<<n_blocks, block_size, 0, stream>>>(
        pos_x, pos_y, pos_z, signs,
        gx_plus, gy_plus, gz_plus,
        gx_minus, gy_minus, gz_minus,
        fx, fy, fz,
        n_particles, nx, ny, nz, box_size
    );
}

void launch_kick(
    float* vel_x, float* vel_y, float* vel_z,
    const float* fx, const float* fy, const float* fz,
    int n_particles, float dt, float hubble_friction,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n_particles + block_size - 1) / block_size;

    kick_kernel<<<n_blocks, block_size, 0, stream>>>(
        vel_x, vel_y, vel_z, fx, fy, fz,
        n_particles, dt, hubble_friction
    );
}

void launch_drift(
    double* pos_x, double* pos_y, double* pos_z,
    const float* vel_x, const float* vel_y, const float* vel_z,
    int n_particles, float dt, double box_size,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n_particles + block_size - 1) / block_size;

    drift_kernel<<<n_blocks, block_size, 0, stream>>>(
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
        n_particles, dt, box_size
    );
}

void launch_zero_float(float* arr, int n, cudaStream_t stream) {
    int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    zero_float_kernel<<<n_blocks, block_size, 0, stream>>>(arr, n);
}

void launch_scale_velocities(
    float* vel_x, float* vel_y, float* vel_z,
    int n_particles, float factor,
    cudaStream_t stream
) {
    int block_size = 256;
    int n_blocks = (n_particles + block_size - 1) / block_size;

    scale_velocities_kernel<<<n_blocks, block_size, 0, stream>>>(
        vel_x, vel_y, vel_z, n_particles, factor
    );
}

void launch_kinetic_energy(
    const float* vel_x, const float* vel_y, const float* vel_z,
    float* partial_sums,
    int n_particles, int n_blocks,
    cudaStream_t stream
) {
    int block_size = 256;
    int shared_size = block_size * sizeof(float);

    kinetic_energy_kernel<<<n_blocks, block_size, shared_size, stream>>>(
        vel_x, vel_y, vel_z, partial_sums, n_particles
    );
}

void launch_segregation(
    const double* pos_x, const double* pos_y, const double* pos_z,
    const int8_t* signs,
    double* sum_pos, double* sum_neg,
    int n_particles, int n_blocks, double box_size,
    cudaStream_t stream
) {
    int block_size = 256;
    int shared_size = block_size * 8 * sizeof(double);

    segregation_kernel<<<n_blocks, block_size, shared_size, stream>>>(
        pos_x, pos_y, pos_z, signs,
        sum_pos, sum_neg,
        n_particles, box_size
    );
}

} // extern "C"
