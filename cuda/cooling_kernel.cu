// Baryonic Cooling, Star Formation, and SN Feedback Kernels
// Native CUDA implementation for Janus MCJ simulation
// Replaces Grackle CPU library for GPU performance

#include <math.h>

extern "C" {

// ═══════════════════════════════════════════════════════════════════════
// Physical Constants (CGS converted to code units)
// ═══════════════════════════════════════════════════════════════════════

// Code units: [length] = Mpc, [time] = Gyr, [mass] = M_sun
// Energy per unit mass: [u] = (km/s)^2

#define K_B_CGS        1.380649e-16    // Boltzmann constant [erg/K]
#define M_P_CGS        1.6726e-24      // Proton mass [g]
#define K_B_OVER_MP    8.254e9         // k_B/m_p in (km/s)^2 / K
#define MU_IONIZED     0.6             // Mean molecular weight (ionized)
#define T_FLOOR        100.0           // Minimum temperature [K]
#define T_CMB_Z0       2.725           // CMB temperature today [K]

// Cooling rate normalization factors (CGS)
#define LAMBDA_H_NORM  7.5e-19         // Collisional ionization H
#define LAMBDA_HE_NORM 9.1e-27         // Collisional excitation He
#define LAMBDA_FF_NORM 1.42e-27        // Free-free (Bremsstrahlung)
#define GAMMA_UV_NORM  1.0e-24         // UV photoheating normalization

// Star formation parameters
#define T_SF_THRESHOLD 10000.0         // Max T for SF [K]
#define N_SF_THRESHOLD 30.0            // Min n_H for SF [cm^-3]
#define EPSILON_STAR   0.02            // SF efficiency per t_ff

// SN feedback parameters
#define E_SN_51        1.0             // SN energy [10^51 erg]
#define EPSILON_SN     0.003           // Coupling efficiency (0.3%)
#define DELAY_SN_GYR   0.01            // SN delay [Gyr]

// ═══════════════════════════════════════════════════════════════════════
// Rahmati+ 2013 Self-Shielding
// Reduces UV photoheating in dense gas
// ═══════════════════════════════════════════════════════════════════════

__device__ double self_shielding_rahmati(double nH, double Gamma_uv) {
    // Rahmati+ 2013 Eq. 14
    // n_0 = 0.01 cm^-3 (transition density)
    // alpha = -2.0 (power law index)
    double n0 = 0.01;
    double x = nH / n0;
    double shield = 0.98 * pow(1.0 + pow(x, 1.64), -2.28)
                  + 0.02 * pow(1.0 + x, -0.84);
    return Gamma_uv * shield;
}

// ═══════════════════════════════════════════════════════════════════════
// S&D93 Tabulated Cooling Function (Primordial CIE)
// Sutherland & Dopita 1993, ApJS 88, 253 — Table 6 (zero metallicity)
// ═══════════════════════════════════════════════════════════════════════

// Table: (log T, log Λ/nH² [erg cm³/s])
#define N_COOLING_TABLE 24

__constant__ double SD93_LOGT[N_COOLING_TABLE] = {
    4.00, 4.20, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90,
    5.00, 5.20, 5.40, 5.60, 5.80, 6.00, 6.20, 6.40,
    6.60, 6.80, 7.00, 7.40, 7.80, 8.20, 8.60, 9.00
};

__constant__ double SD93_LOGL[N_COOLING_TABLE] = {
    -24.46, -23.26, -22.26, -21.89, -21.80, -21.83, -21.88, -22.03,
    -22.24, -22.48, -22.56, -22.54, -22.60, -22.75, -22.76, -22.64,
    -22.54, -22.49, -22.47, -22.35, -22.12, -21.80, -21.44, -21.09
};

__device__ double cooling_lambda_sd93(double T) {
    // Interpolate S&D93 Table 6 for primordial CIE cooling
    if (T < 1e4) return 0.0;
    if (T > 1e9) T = 1e9;

    double logT = log10(T);

    // Clamp to table bounds
    if (logT <= SD93_LOGT[0]) return pow(10.0, SD93_LOGL[0]);
    if (logT >= SD93_LOGT[N_COOLING_TABLE-1]) return pow(10.0, SD93_LOGL[N_COOLING_TABLE-1]);

    // Find bracketing indices
    int i = 0;
    for (i = 0; i < N_COOLING_TABLE - 1; i++) {
        if (logT >= SD93_LOGT[i] && logT < SD93_LOGT[i+1]) break;
    }

    // Linear interpolation in log-log space
    double frac = (logT - SD93_LOGT[i]) / (SD93_LOGT[i+1] - SD93_LOGT[i]);
    double logL = SD93_LOGL[i] + frac * (SD93_LOGL[i+1] - SD93_LOGL[i]);

    return pow(10.0, logL);
}

// ═══════════════════════════════════════════════════════════════════════
// Cooling Rate (Lambda - Gamma) in erg/s/cm^3 per nH^2
// Uses S&D93 tabulated cooling + Haardt-Madau UV heating
// ═══════════════════════════════════════════════════════════════════════

__device__ double cooling_rate(double T, double nH, double z) {
    // Protect against invalid T
    if (T < T_FLOOR) T = T_FLOOR;
    if (T > 1e9) T = 1e9;

    // === Cooling: S&D93 tabulated (primordial CIE) ===
    double Lambda_total = cooling_lambda_sd93(T);

    // === Heating processes ===

    // UV background photoheating (Haardt & Madau 2012 fit)
    // Gamma_UV ∝ (1+z)^2 / (1 + ((1+z)/3)^5) for z < 6
    double zp1 = 1.0 + z;
    double Gamma_uv = GAMMA_UV_NORM * zp1 * zp1
                    / (1.0 + pow(zp1 / 3.0, 5.0));

    // Self-shielding reduces heating in dense gas
    double Gamma_eff = self_shielding_rahmati(nH, Gamma_uv);

    // CMB floor: gas cannot cool below T_CMB(z)
    double T_cmb = T_CMB_Z0 * zp1;
    if (T < T_cmb) {
        // Compton heating from CMB
        Gamma_eff += 5.65e-36 * pow(T_cmb, 4.0) * (T_cmb - T);
    }

    // Net cooling rate: (Lambda * nH^2 - Gamma * nH) / nH^2
    // Return per nH^2 for convenience
    return Lambda_total - Gamma_eff / nH;
}

// ═══════════════════════════════════════════════════════════════════════
// Apply Cooling Kernel
// Updates internal energy (temperature proxy) for m+ particles
// m- particles are IGNORED (collisionless)
// ═══════════════════════════════════════════════════════════════════════

__global__ void apply_cooling_kernel(
    double* __restrict__ internal_energy,  // [N] internal energy u = (3/2)(k_B/m_p)T/mu
    const double* __restrict__ sph_density, // [N] SPH density in code units
    const int* __restrict__ signs,          // [N] particle signs (+1 or -1)
    const double dt_gyr,                    // Time step [Gyr]
    const double z,                         // Current redshift
    const double rho_to_nH,                 // Density conversion factor
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip m- particles (collisionless, no baryonic physics)
    if (signs[i] < 0) return;

    // Convert internal energy to temperature
    // u = (3/2) * (k_B/m_p) * T / mu  =>  T = (2/3) * mu * u / (k_B/m_p)
    double u = internal_energy[i];
    double T = (2.0 / 3.0) * MU_IONIZED * u / K_B_OVER_MP;

    // Protect T
    if (T < T_FLOOR) T = T_FLOOR;
    if (T > 1e9) T = 1e9;

    // Convert density to nH [cm^-3]
    double nH = sph_density[i] * rho_to_nH;

    // Skip very low-density gas (IGM in UV equilibrium)
    // Below nH_min, gas is assumed to be in thermal equilibrium with UV background
    #define NH_MIN_COOLING 0.01  // cm^-3
    if (nH < NH_MIN_COOLING) {
        // No thermal evolution for IGM - stays at photoheating equilibrium
        return;
    }

    // Get cooling rate Lambda [erg cm^3/s] (per nH^2)
    double Lambda_net = cooling_rate(T, nH, z);

    // Volumetric cooling rate: n^2 * Lambda [erg/s/cm^3]
    // Energy density: e = (3/2) * n * k_B * T [erg/cm^3]
    // Cooling time: t_cool = e / (n^2 * Lambda) = (3/2) * k_B * T / (n * Lambda)
    // dT/dt = -T / t_cool = -(2/3) * n * Lambda * T / (k_B * T) = -(2/3) * n * Lambda / k_B

    double k_B = 1.381e-16;   // Boltzmann [erg/K]

    // dT/dt [K/s] = -(2/3) * Lambda * nH / k_B
    // (Lambda is per nH^2, times nH^2 gives volumetric, divide by nH to get per particle)
    double dT_dt = -(2.0/3.0) * Lambda_net * nH / k_B;

    // Convert to K/Gyr and apply with sub-cycling
    // Thermal timescale can be << dynamical timestep, so we limit dT per step
    double dt_s = dt_gyr * 3.156e16;  // Gyr to seconds
    double dT = dT_dt * dt_s;

    // Limit temperature change to prevent thermal runaway
    // Max change: 50% of current T per step (implicit sub-cycling)
    double dT_max = 0.5 * T;
    if (dT > dT_max) dT = dT_max;
    if (dT < -dT_max) dT = -dT_max;

    // Apply temperature change
    T += dT;

    // Temperature floor
    if (T < T_FLOOR) T = T_FLOOR;

    // Convert back to internal energy
    // u = (3/2) * (k_B/m_p) * T / mu
    u = (3.0 / 2.0) * K_B_OVER_MP * T / MU_IONIZED;

    // Check for NaN
    if (isnan(u) || isinf(u)) {
        double u_floor = (3.0 / 2.0) * K_B_OVER_MP * T_FLOOR / MU_IONIZED;
        u = u_floor;
    }

    internal_energy[i] = u;
}

// ═══════════════════════════════════════════════════════════════════════
// Star Formation Kernel
// Probabilistic SF for cold, dense m+ gas
// ═══════════════════════════════════════════════════════════════════════

__global__ void apply_sf_kernel(
    const double* __restrict__ internal_energy,
    const double* __restrict__ sph_density,
    const int* __restrict__ signs,
    int* __restrict__ star_flag,           // Output: 1 if particle forms star
    const double* __restrict__ random_vals, // Pre-generated random [0,1]
    const double dt_gyr,
    const double rho_to_nH,
    const double G_code,                   // G in code units
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    star_flag[i] = 0;  // Default: no star formation

    // Only m+ can form stars
    if (signs[i] < 0) return;

    // Get temperature
    double u = internal_energy[i];
    double T = (2.0 / 3.0) * MU_IONIZED * u / K_B_OVER_MP;

    // Get density
    double nH = sph_density[i] * rho_to_nH;

    // SF criteria: T < T_threshold AND n > n_threshold
    if (T > T_SF_THRESHOLD || nH < N_SF_THRESHOLD) return;

    // Free-fall time: t_ff = sqrt(3π / (32 G ρ))
    double rho_cgs = nH * M_P_CGS / 0.76;  // Total density from nH
    double t_ff_s = sqrt(3.0 * 3.14159265 / (32.0 * 6.674e-8 * rho_cgs));
    double t_ff_gyr = t_ff_s / 3.156e16;

    // SF probability: P = ε* × dt / t_ff
    double prob = EPSILON_STAR * dt_gyr / t_ff_gyr;
    if (prob > 1.0) prob = 1.0;

    // Stochastic SF
    if (random_vals[i] < prob) {
        star_flag[i] = 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SN Feedback Kernel
// Kinetic energy injection into surrounding gas
// ═══════════════════════════════════════════════════════════════════════

__global__ void apply_feedback_kernel(
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    double* __restrict__ internal_energy,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const int* __restrict__ signs,
    const int* __restrict__ sn_flag,       // 1 if SN event at this particle
    const double* __restrict__ random_theta,
    const double* __restrict__ random_phi,
    const double v_sn,                     // SN velocity kick [km/s in code]
    const double box_size,
    const double h_sn,                     // Feedback radius
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Only m+ receives feedback
    if (signs[i] < 0) return;

    // Find nearby SN events and accumulate kicks
    double dv_x = 0.0, dv_y = 0.0, dv_z = 0.0;
    double du_thermal = 0.0;

    double half_box = box_size * 0.5;
    double h_sn2 = h_sn * h_sn;

    for (int j = 0; j < N; j++) {
        if (sn_flag[j] == 0) continue;  // No SN at j

        double dx = pos_x[i] - pos_x[j];
        double dy = pos_y[i] - pos_y[j];
        double dz = pos_z[i] - pos_z[j];

        // Periodic BC
        if (dx >  half_box) dx -= box_size;
        if (dx < -half_box) dx += box_size;
        if (dy >  half_box) dy -= box_size;
        if (dy < -half_box) dy += box_size;
        if (dz >  half_box) dz -= box_size;
        if (dz < -half_box) dz += box_size;

        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > h_sn2 || r2 < 1e-10) continue;

        double r = sqrt(r2);

        // Radial kick (outward from SN)
        double v_kick = v_sn * (1.0 - r / h_sn);  // Linear falloff
        if (v_kick < 0.0) v_kick = 0.0;

        // Add random isotropic component
        double theta = random_theta[j] * 3.14159265;
        double phi = random_phi[j] * 2.0 * 3.14159265;

        double dir_x = sin(theta) * cos(phi);
        double dir_y = sin(theta) * sin(phi);
        double dir_z = cos(theta);

        dv_x += v_kick * (dx/r * 0.7 + dir_x * 0.3);
        dv_y += v_kick * (dy/r * 0.7 + dir_y * 0.3);
        dv_z += v_kick * (dz/r * 0.7 + dir_z * 0.3);

        // Thermal energy injection (10% of kinetic)
        du_thermal += 0.1 * v_kick * v_kick;
    }

    // Apply velocity kick
    vel_x[i] += dv_x;
    vel_y[i] += dv_y;
    vel_z[i] += dv_z;

    // Apply thermal heating
    internal_energy[i] += du_thermal;
}

// ═══════════════════════════════════════════════════════════════════════
// Temperature computation kernel (for diagnostics)
// ═══════════════════════════════════════════════════════════════════════

__global__ void compute_temperature_kernel(
    const double* __restrict__ internal_energy,
    double* __restrict__ temperature,
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double u = internal_energy[i];
    temperature[i] = (2.0 / 3.0) * MU_IONIZED * u / K_B_OVER_MP;
}

// ═══════════════════════════════════════════════════════════════════════
// Initialize internal energy from temperature
// ═══════════════════════════════════════════════════════════════════════

__global__ void init_internal_energy_kernel(
    double* __restrict__ internal_energy,
    const int* __restrict__ signs,
    const double T_init_plus,    // Initial T for m+ [K]
    const double T_init_minus,   // Initial T for m- [K] (if used)
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double T = (signs[i] > 0) ? T_init_plus : T_init_minus;

    // u = (3/2) * (k_B/m_p) * T / mu
    internal_energy[i] = (3.0 / 2.0) * K_B_OVER_MP * T / MU_IONIZED;
}

} // extern "C"
