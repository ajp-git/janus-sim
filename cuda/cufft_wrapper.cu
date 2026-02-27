/**
 * cuFFT Wrapper for Janus TreePM
 *
 * Provides 3D R2C/C2R FFT for Poisson solver.
 * Double precision (f64) for physics accuracy.
 */

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
        return -1; \
    } \
}

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return -1; \
    } \
}

// Global plan handles (reuse for efficiency)
static cufftHandle plan_r2c = 0;
static cufftHandle plan_c2r = 0;
static int plan_nx = 0, plan_ny = 0, plan_nz = 0;

extern "C" {

/**
 * Initialize cuFFT plans for 3D grid
 *
 * @param nx, ny, nz: Grid dimensions
 * @return 0 on success, -1 on error
 */
int cufft_init_3d(int nx, int ny, int nz) {
    // Destroy existing plans if dimensions changed
    if (plan_r2c != 0 && (nx != plan_nx || ny != plan_ny || nz != plan_nz)) {
        cufftDestroy(plan_r2c);
        cufftDestroy(plan_c2r);
        plan_r2c = 0;
        plan_c2r = 0;
    }

    if (plan_r2c == 0) {
        // Create R2C plan (real to complex)
        CUFFT_CHECK(cufftPlan3d(&plan_r2c, nx, ny, nz, CUFFT_D2Z));

        // Create C2R plan (complex to real)
        CUFFT_CHECK(cufftPlan3d(&plan_c2r, nx, ny, nz, CUFFT_Z2D));

        plan_nx = nx;
        plan_ny = ny;
        plan_nz = nz;
    }

    return 0;
}

/**
 * Execute forward FFT (R2C)
 *
 * @param d_input: Device pointer to real input (nx * ny * nz doubles)
 * @param d_output: Device pointer to complex output (nx * ny * (nz/2+1) complex doubles)
 * @return 0 on success, -1 on error
 */
int cufft_exec_r2c(double* d_input, cufftDoubleComplex* d_output) {
    if (plan_r2c == 0) {
        fprintf(stderr, "cuFFT not initialized. Call cufft_init_3d first.\n");
        return -1;
    }

    CUFFT_CHECK(cufftExecD2Z(plan_r2c, d_input, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * Execute inverse FFT (C2R)
 *
 * @param d_input: Device pointer to complex input
 * @param d_output: Device pointer to real output
 * @return 0 on success, -1 on error
 */
int cufft_exec_c2r(cufftDoubleComplex* d_input, double* d_output) {
    if (plan_c2r == 0) {
        fprintf(stderr, "cuFFT not initialized. Call cufft_init_3d first.\n");
        return -1;
    }

    CUFFT_CHECK(cufftExecZ2D(plan_c2r, d_input, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * Cleanup cuFFT plans
 */
void cufft_cleanup() {
    if (plan_r2c != 0) {
        cufftDestroy(plan_r2c);
        plan_r2c = 0;
    }
    if (plan_c2r != 0) {
        cufftDestroy(plan_c2r);
        plan_c2r = 0;
    }
    plan_nx = plan_ny = plan_nz = 0;
}

/**
 * Apply Green's function in k-space for Poisson equation
 *
 * phi_k = -4*pi*G / k² * rho_k * exp(-k²*r_s²)
 *
 * @param d_rho_k: Input density in k-space (modified in place to phi_k)
 * @param nx, ny, nz: Grid dimensions
 * @param dk: k-space spacing (2*pi / box_size)
 * @param g_constant: Gravitational constant
 * @param r_s: Gaussian splitting scale (0 for no splitting)
 */
__global__ void apply_green_kernel(
    cufftDoubleComplex* d_data,
    int nx, int ny, int nz,
    double dk, double g_constant, double r_s
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nz_complex = nz / 2 + 1;
    int total = nx * ny * nz_complex;

    if (idx >= total) return;

    // Convert linear index to 3D (kx, ky, kz)
    int kz = idx % nz_complex;
    int ky = (idx / nz_complex) % ny;
    int kx = idx / (nz_complex * ny);

    // Compute k values with correct Nyquist handling
    double kx_val = (kx <= nx/2) ? kx * dk : (kx - nx) * dk;
    double ky_val = (ky <= ny/2) ? ky * dk : (ky - ny) * dk;
    double kz_val = kz * dk;  // R2C: kz only goes to nz/2

    double k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

    // Skip DC component
    if (k2 < 1e-20) {
        d_data[idx].x = 0.0;
        d_data[idx].y = 0.0;
        return;
    }

    // Green's function: -4*pi*G / k²
    double green = -4.0 * 3.14159265358979323846 * g_constant / k2;

    // Gaussian splitting for TreePM
    if (r_s > 0.0) {
        double r_s_sq = r_s * r_s;
        green *= exp(-k2 * r_s_sq);
    }

    // Apply to both real and imaginary parts
    d_data[idx].x *= green;
    d_data[idx].y *= green;
}

/**
 * Launch Green's function kernel
 */
int cufft_apply_green(
    cufftDoubleComplex* d_data,
    int nx, int ny, int nz,
    double box_size, double g_constant, double r_s
) {
    int nz_complex = nz / 2 + 1;
    int total = nx * ny * nz_complex;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    double dk = 2.0 * 3.14159265358979323846 / box_size;

    apply_green_kernel<<<blocks, threads>>>(d_data, nx, ny, nz, dk, g_constant, r_s);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * Allocate GPU memory for FFT
 *
 * @param size_bytes: Size in bytes
 * @return Device pointer, or NULL on error
 */
void* cufft_alloc(size_t size_bytes) {
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

/**
 * Free GPU memory
 */
void cufft_free(void* ptr) {
    if (ptr != NULL) {
        cudaFree(ptr);
    }
}

/**
 * Copy host to device
 */
int cufft_copy_h2d(void* d_dst, const void* h_src, size_t size_bytes) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size_bytes, cudaMemcpyHostToDevice));
    return 0;
}

/**
 * Copy device to host
 */
int cufft_copy_d2h(void* h_dst, const void* d_src, size_t size_bytes) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size_bytes, cudaMemcpyDeviceToHost));
    return 0;
}

/**
 * Normalize after inverse FFT
 * cuFFT doesn't normalize, so we need to divide by N
 */
__global__ void normalize_kernel(double* data, int n, double factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

int cufft_normalize(double* d_data, int nx, int ny, int nz) {
    int n = nx * ny * nz;
    double factor = 1.0 / (double)n;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    normalize_kernel<<<blocks, threads>>>(d_data, n, factor);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * Device-to-device copy
 */
int cufft_copy_d2d(void* d_dst, const void* d_src, size_t size_bytes) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size_bytes, cudaMemcpyDeviceToDevice));
    return 0;
}

// Internal k-space buffer for solve_device
static cufftDoubleComplex* internal_kspace = NULL;
static int internal_kspace_size = 0;

/**
 * Solve Poisson equation directly on device pointers
 *
 * Uses internal k-space buffer, operates on external rho/phi buffers.
 * This avoids host<->device transfers.
 *
 * @param d_rho: Device pointer to input density (read only)
 * @param d_phi: Device pointer to output potential (written)
 * @param nx, ny, nz: Grid dimensions
 * @param box_size: Physical box size
 * @param g_constant: Gravitational constant
 * @param r_s: Gaussian splitting scale (0 for full, >0 for TreePM long-range)
 * @return 0 on success
 */
int cufft_solve_device(
    double* d_rho,
    double* d_phi,
    int nx, int ny, int nz,
    double box_size, double g_constant, double r_s
) {
    // Ensure plans exist
    if (plan_r2c == 0) {
        int ret = cufft_init_3d(nx, ny, nz);
        if (ret != 0) return ret;
    }

    // Allocate/resize internal k-space buffer
    int n_complex = nx * ny * (nz / 2 + 1);
    if (internal_kspace == NULL || internal_kspace_size != n_complex) {
        if (internal_kspace != NULL) {
            cudaFree(internal_kspace);
        }
        CUDA_CHECK(cudaMalloc(&internal_kspace, n_complex * sizeof(cufftDoubleComplex)));
        internal_kspace_size = n_complex;
    }

    // Forward FFT: rho -> k-space
    // Note: cuFFT can operate in-place, but d_rho is const (don't modify input)
    // So we copy rho to phi, then transform phi
    int n_real = nx * ny * nz;
    CUDA_CHECK(cudaMemcpy(d_phi, d_rho, n_real * sizeof(double), cudaMemcpyDeviceToDevice));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, d_phi, internal_kspace));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply Green's function
    double dk = 2.0 * 3.14159265358979323846 / box_size;
    int threads = 256;
    int blocks = (n_complex + threads - 1) / threads;
    apply_green_kernel<<<blocks, threads>>>(internal_kspace, nx, ny, nz, dk, g_constant, r_s);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Inverse FFT: k-space -> phi
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, internal_kspace, d_phi));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Normalize
    double factor = 1.0 / (double)n_real;
    blocks = (n_real + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(d_phi, n_real, factor);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * Cleanup internal buffers
 */
void cufft_cleanup_internal() {
    if (internal_kspace != NULL) {
        cudaFree(internal_kspace);
        internal_kspace = NULL;
        internal_kspace_size = 0;
    }
}

} // extern "C"
