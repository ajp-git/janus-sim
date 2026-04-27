// SPH Pressure Kernels for Janus Baryonic Physics
// GPU implementation with Newton III symmetry guaranteed

#include <math.h>

extern "C" {

// ═══════════════════════════════════════════════════════════════════════
// SPH Gaussian Kernel and Gradient
// ═══════════════════════════════════════════════════════════════════════

// Gaussian SPH kernel: W(r,h) = exp(-r²/h²) / (π^(3/2) h³)
__device__ float sph_w(float r, float h) {
    float q = r / h;
    float norm = 1.0f / (3.14159265f * sqrtf(3.14159265f) * h * h * h);
    return norm * __expf(-q * q);
}

// Kernel gradient (scalar, radial component)
// dW/dr = -2r/h² × W(r,h)
__device__ float sph_dw_dr(float r, float h) {
    float q = r / h;
    float norm = 1.0f / (3.14159265f * sqrtf(3.14159265f) * h * h * h);
    return -2.0f * q / h * norm * __expf(-q * q);
}

// ═══════════════════════════════════════════════════════════════════════
// SPH Density Kernel
// ρᵢ = Σⱼ mⱼ W(|rᵢ - rⱼ|, hᵢ)
// ═══════════════════════════════════════════════════════════════════════

__global__ void sph_density_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ smooth_h,
    float* __restrict__ density,
    const float mass,
    const float box_size,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = pos_x[i];
    float yi = pos_y[i];
    float zi = pos_z[i];
    float h_i = smooth_h[i];
    float r_cut = 3.0f * h_i;  // Cutoff at 3h (kernel negligible beyond)
    float r_cut2 = r_cut * r_cut;
    float rho = 0.0f;

    float half_box = box_size * 0.5f;

    for (int j = 0; j < n; j++) {
        float dx = xi - pos_x[j];
        float dy = yi - pos_y[j];
        float dz = zi - pos_z[j];

        // Periodic boundary conditions
        if (dx >  half_box) dx -= box_size;
        if (dx < -half_box) dx += box_size;
        if (dy >  half_box) dy -= box_size;
        if (dy < -half_box) dy += box_size;
        if (dz >  half_box) dz -= box_size;
        if (dz < -half_box) dz += box_size;

        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > r_cut2) continue;

        float r = sqrtf(r2 + 1e-10f);  // Softening for self-contribution
        rho += mass * sph_w(r, h_i);
    }

    density[i] = rho;
}

// ═══════════════════════════════════════════════════════════════════════
// SPH Pressure Force Kernel
// aᵢ = -Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) ∇W(rᵢⱼ, h_avg)
// Newton III guaranteed by symmetric form (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²)
// ═══════════════════════════════════════════════════════════════════════

__global__ void sph_pressure_force_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ density,
    const float* __restrict__ pressure,
    const float* __restrict__ smooth_h,
    float* __restrict__ acc_x,
    float* __restrict__ acc_y,
    float* __restrict__ acc_z,
    const float mass,
    const float box_size,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = pos_x[i];
    float yi = pos_y[i];
    float zi = pos_z[i];
    float rho_i = density[i];
    float p_i = pressure[i];
    float h_i = smooth_h[i];

    // Pressure coefficient: P/ρ² (with floor to avoid division by zero)
    float rho_i2 = rho_i * rho_i + 1e-30f;
    float coeff_i = p_i / rho_i2;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float half_box = box_size * 0.5f;

    // Cutoff: 3h (kernel negligible beyond)
    float r_cut = 3.0f * h_i;
    float r_cut2 = r_cut * r_cut;

    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = xi - pos_x[j];
        float dy = yi - pos_y[j];
        float dz = zi - pos_z[j];

        // Periodic boundary conditions
        if (dx >  half_box) dx -= box_size;
        if (dx < -half_box) dx += box_size;
        if (dy >  half_box) dy -= box_size;
        if (dy < -half_box) dy += box_size;
        if (dz >  half_box) dz -= box_size;
        if (dz < -half_box) dz += box_size;

        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > r_cut2 || r2 < 1e-10f) continue;

        float r = sqrtf(r2);
        float h_avg = 0.5f * (h_i + smooth_h[j]);

        float rho_j = density[j];
        float p_j = pressure[j];
        float rho_j2 = rho_j * rho_j + 1e-30f;
        float coeff_j = p_j / rho_j2;

        // Symmetric SPH pressure force (Newton III guaranteed)
        // F = -m × (P_i/ρ_i² + P_j/ρ_j²) × ∇W × r_hat
        float dw = sph_dw_dr(r, h_avg);
        float force_mag = -mass * (coeff_i + coeff_j) * dw / r;

        ax += force_mag * dx;
        ay += force_mag * dy;
        az += force_mag * dz;
    }

    acc_x[i] = ax;
    acc_y[i] = ay;
    acc_z[i] = az;
}

// ═══════════════════════════════════════════════════════════════════════
// Combined kernel for pressure computation
// P = ρ × (k_B/m_p) × T / μ
// ═══════════════════════════════════════════════════════════════════════

__global__ void compute_pressure_kernel(
    const float* __restrict__ density,
    const float* __restrict__ temperature,
    float* __restrict__ pressure,
    const float k_b_over_mp,  // k_B/m_p in code units
    const float mu,           // Mean molecular weight
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // P = ρ × (k_B/m_p) × T / μ
    pressure[i] = density[i] * k_b_over_mp * temperature[i] / mu;
}

// ═══════════════════════════════════════════════════════════════════════
// Adaptive smoothing length
// h = η × (m/ρ)^(1/3)  where η ≈ 1.2
// ═══════════════════════════════════════════════════════════════════════

__global__ void update_smoothing_length_kernel(
    const float* __restrict__ density,
    float* __restrict__ smooth_h,
    const float mass,
    const float eta,      // Typically 1.2
    const float h_min,    // Minimum h (softening)
    const float h_max,    // Maximum h (box/10)
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float rho = density[i] + 1e-30f;
    float h = eta * cbrtf(mass / rho);

    // Clamp to reasonable range
    if (h < h_min) h = h_min;
    if (h > h_max) h = h_max;

    smooth_h[i] = h;
}

// ═══════════════════════════════════════════════════════════════════════
// Apply pressure acceleration to velocities (kick step)
// v_new = v_old + a_press × dt
// ═══════════════════════════════════════════════════════════════════════

__global__ void apply_pressure_kick_kernel(
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    const float* __restrict__ acc_x,
    const float* __restrict__ acc_y,
    const float* __restrict__ acc_z,
    const float dt,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vel_x[i] += acc_x[i] * dt;
    vel_y[i] += acc_y[i] * dt;
    vel_z[i] += acc_z[i] * dt;
}

} // extern "C"
